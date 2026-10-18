const std = @import("std");
const mlx = @import("mlx_host").mlx;
const api = @import("sushi_exl3");
const format = api.format;
const quantize = @import("quantize.zig");
const metal_search = @import("metal_search.zig");
const shard = @import("shard.zig");
const source = @import("source.zig");
const st = @import("safetensors.zig");
const stages_mod = @import("stages.zig");
const pool_mod = @import("pool.zig");
const gpu_factor = @import("gpu_factor.zig");

pub const Geometry = struct { experts: usize, hidden: usize, inter: usize };

pub const Policy = struct {
    rate: format.Rate,
    dec: format.Decode,
    seed: u64 = 0x5eed_e3c0_de00_0001,
    mode: quantize.OutScales = .auto,
    min_rows: u64 = 64,
    batch: usize = 16,
    factor_precision: gpu_factor.Precision = .f64,
    provenance: []const [2][]const u8 = &.{},
};

pub const HessianSource = struct {
    ctx: *anyopaque,
    gate_up: *const fn (ctx: *anyopaque, expert: usize, out: []f32) anyerror!u64,
    down: *const fn (ctx: *anyopaque, expert: usize, out: []f32) anyerror!u64,
};

pub const LayerStats = struct {
    gate_up_identity: usize = 0,
    down_identity: usize = 0,
    experts: usize = 0,
    stages: stages_mod.Stages = .{},
    factor_retries: usize = 0,
    factor_fallbacks: usize = 0,
    factor_gate_up_ns: u64 = 0,
    factor_down_ns: u64 = 0,
};

fn identityHessian(h: []f32, dim: usize) void {
    @memset(h, 0);
    for (0..dim) |i| h[i * dim + i] = 1;
}

const Banks = struct {
    trellis: [3][]u16,
    suh: [3][]u16,
    svh: [3][]u16,

    fn init(allocator: std.mem.Allocator, geom: Geometry, n: u32) !Banks {
        var self: Banks = undefined;
        var made: usize = 0;
        errdefer self.release(allocator, made);
        while (made < 3) : (made += 1) {
            const in = if (made == 2) geom.inter else geom.hidden;
            const out = if (made == 2) geom.hidden else geom.inter;
            self.trellis[made] = try allocator.alloc(u16, geom.experts * (in / 16) * (out / 16) * n);
            errdefer allocator.free(self.trellis[made]);
            self.suh[made] = try allocator.alloc(u16, geom.experts * in);
            errdefer allocator.free(self.suh[made]);
            self.svh[made] = try allocator.alloc(u16, geom.experts * out);
        }
        return self;
    }

    fn release(self: *Banks, allocator: std.mem.Allocator, count: usize) void {
        for (0..count) |p| {
            allocator.free(self.trellis[p]);
            allocator.free(self.suh[p]);
            allocator.free(self.svh[p]);
        }
    }

    fn deinit(self: *Banks, allocator: std.mem.Allocator) void {
        self.release(allocator, 3);
    }
};

/// One bank's worth of tensors in flight: gate and up of a group of experts (they share each expert's
/// input Hessian), or the group's down projections.
const Unit = struct {
    first: usize,
    count: usize,
    down: bool,
};

const Slot = struct {
    allocator: std.mem.Allocator,
    weights: []f32,
    hessians: []f32,
    names: [][160]u8,
    signs: [][]f32,
    jobs: []metal_search.TensorJob,
    unit: Unit = undefined,
    prepared: ?metal_search.Prepared = null,
    err: ?anyerror = null,
    thread: ?std.Thread = null,

    fn init(allocator: std.mem.Allocator, geom: Geometry, group: usize) !Slot {
        const most = 2 * group;
        const weights = try allocator.alloc(f32, most * geom.hidden * geom.inter);
        errdefer allocator.free(weights);
        const hessians = try allocator.alloc(f32, group * @max(geom.hidden * geom.hidden, geom.inter * geom.inter));
        errdefer allocator.free(hessians);
        const names = try allocator.alloc([160]u8, most);
        errdefer allocator.free(names);
        const signs = try allocator.alloc([]f32, most);
        errdefer allocator.free(signs);
        for (signs) |*sg| sg.* = &.{};
        const jobs = try allocator.alloc(metal_search.TensorJob, most);
        return .{ .allocator = allocator, .weights = weights, .hessians = hessians, .names = names, .signs = signs, .jobs = jobs };
    }

    fn release(self: *Slot) void {
        if (self.prepared) |*p| p.deinit();
        self.prepared = null;
        for (self.signs) |*sg| {
            self.allocator.free(sg.*);
            sg.* = &.{};
        }
    }

    fn deinit(self: *Slot) void {
        self.release();
        self.allocator.free(self.weights);
        self.allocator.free(self.hessians);
        self.allocator.free(self.names);
        self.allocator.free(self.signs);
        self.allocator.free(self.jobs);
    }

    fn tensors(self: *const Slot) usize {
        return if (self.unit.down) self.unit.count else 2 * self.unit.count;
    }
};

const roles = [3]source.ProjRole{ .gate, .up, .down };

const Layer = struct {
    allocator: std.mem.Allocator,
    src: *const source.SafetensorsSource,
    layer: u32,
    mtp: bool,
    geom: Geometry,
    policy: Policy,
    prefix: []const u8,
    stats: *LayerStats,
    pool: *metal_search.Pool,

    fn dims(self: *const Layer, down: bool) [2]usize {
        return if (down) .{ self.geom.inter, self.geom.hidden } else .{ self.geom.hidden, self.geom.inter };
    }

    /// The tensor at `t` of a slot: gate/up units interleave gate(e), up(e) so each pair shares one Hessian.
    fn role(slot: *const Slot, t: usize) source.ProjRole {
        return if (slot.unit.down) .down else roles[t % 2];
    }

    fn expertOf(slot: *const Slot, t: usize) usize {
        return slot.unit.first + if (slot.unit.down) t else t / 2;
    }

    fn hessians(self: *const Layer, slot: *Slot, hs: ?HessianSource) !void {
        for (0..slot.unit.count) |b| try self.hessianOne(slot, b, hs);
    }

    fn hessianOne(self: *const Layer, slot: *Slot, b: usize, hs: ?HessianSource) !void {
        const in = self.dims(slot.unit.down)[0];
        const h = slot.hessians[b * in * in ..][0 .. in * in];
        var lap = stages_mod.Lap.start(&self.stats.stages);
        var rows: u64 = 0;
        if (hs) |src| rows = try (if (slot.unit.down) src.down else src.gate_up)(src.ctx, slot.unit.first + b, h);
        lap.mark(.hessian);
        if (hs == null or rows < self.policy.min_rows) {
            identityHessian(h, in);
            if (slot.unit.down) self.stats.down_identity += 1 else self.stats.gate_up_identity += 1;
        }
    }

    const Reader = struct {
        layer: *const Layer,
        slot: *Slot,
        failed: std.atomic.Value(bool) = .init(false),

        fn read(self: *Reader, t: usize) void {
            self.readOne(t) catch {
                self.failed.store(true, .monotonic);
            };
        }

        fn readOne(self: *Reader, t: usize) !void {
            const ly = self.layer;
            const slot = self.slot;
            const d = ly.dims(slot.unit.down);
            const in = d[0];
            const out = d[1];
            const r = role(slot, t);
            const e = expertOf(slot, t);
            const weight = slot.weights[t * in * out ..][0 .. in * out];
            try ly.src.readExpertPublic(ly.allocator, .qwen4_exp, ly.layer, ly.mtp, r, e, in, out, weight);
            const name = try std.fmt.bufPrint(&slot.names[t], "{s}.layers.{d}.mlp.switch_mlp.{s}.e{d}", .{ ly.prefix, ly.layer, r.suffix(), e });
            slot.signs[t] = try quantize.signs(ly.allocator, ly.policy.seed, name, in + out);
            const local = if (slot.unit.down) t else t / 2;
            slot.jobs[t] = .{ .w = weight, .h = slot.hessians[local * in * in ..][0 .. in * in], .sign_in = slot.signs[t][0..in], .sign_out = slot.signs[t][in..] };
        }
    };

    /// The slot's host work, on a thread of its own once the previous bank's is done: read the weights, then
    /// regularize, rotate and factor.
    fn prep(self: *const Layer, slot: *Slot, after: ?*pool_mod.Event, done: *pool_mod.Event) void {
        if (after) |ev| ev.wait();
        self.prepOrFail(slot) catch |e| {
            slot.err = e;
        };
        done.signal();
    }

    fn prepOrFail(self: *const Layer, slot: *Slot) !void {
        const d = self.dims(slot.unit.down);
        var lap = stages_mod.Lap.start(&self.stats.stages);
        var reader = Reader{ .layer = self, .slot = slot };
        self.pool.run(slot.tensors(), &reader, Reader.read);
        if (reader.failed.load(.monotonic)) return error.SourceReadFailed;
        lap.mark(.read);
        slot.prepared = try metal_search.prepareBatch(self.allocator, self.pool, slot.jobs[0..slot.tensors()], d[0], d[1], self.policy.mode, self.policy.factor_precision);
        lap.mark(.prep);
    }
};

/// The bank two ahead: its Hessians, one expert per LDLQ step of the current bank, then its host prep.
const Ahead = struct {
    ly: *const Layer,
    slot: *Slot,
    hs: ?HessianSource,
    after: *pool_mod.Event,
    done: *pool_mod.Event,
    next: usize = 0,

    fn step(raw: *anyopaque) anyerror!bool {
        const self: *Ahead = @ptrCast(@alignCast(raw));
        if (self.next < self.slot.unit.count) {
            try self.ly.hessianOne(self.slot, self.next, self.hs);
            self.next += 1;
        }
        if (self.next < self.slot.unit.count) return true;
        if (self.slot.thread == null) self.slot.thread = try std.Thread.spawn(.{}, Layer.prep, .{ self.ly, self.slot, self.after, self.done });
        return false;
    }

    fn side(self: *Ahead) metal_search.Side {
        return .{ .ctx = self, .step = step };
    }
};

pub fn convertQwenLayer(
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,
    io: std.Io,
    dir: std.Io.Dir,
    src: *const source.SafetensorsSource,
    layer: u32,
    mtp: bool,
    geom: Geometry,
    policy: Policy,
    hessians: ?HessianSource,
) !LayerStats {
    if (geom.experts == 0 or geom.hidden % 128 != 0 or geom.inter % 128 != 0) return error.GeometryUnsupported;
    var banks = try Banks.init(allocator, geom, policy.rate.n);
    defer banks.deinit(allocator);

    const group = @max(1, @min(policy.batch, geom.experts));
    var units = std.ArrayList(Unit).empty;
    defer units.deinit(allocator);
    var first: usize = 0;
    while (first < geom.experts) : (first += group) {
        const count = @min(group, geom.experts - first);
        try units.append(allocator, .{ .first = first, .count = count, .down = false });
        try units.append(allocator, .{ .first = first, .count = count, .down = true });
    }

    var stats = LayerStats{ .experts = geom.experts };
    const prep_pool = try metal_search.Pool.init(allocator, workerThreads() / 2);
    defer prep_pool.deinit();
    const feed_pool = try metal_search.Pool.init(allocator, workerThreads());
    defer feed_pool.deinit();
    const ly = Layer{
        .allocator = allocator,
        .src = src,
        .layer = layer,
        .mtp = mtp,
        .geom = geom,
        .policy = policy,
        .prefix = if (mtp) shard.mtp_prefix else shard.trunk_prefix,
        .stats = &stats,
        .pool = prep_pool,
    };
    const n_units = units.items.len;
    const prepared = try allocator.alloc(pool_mod.Event, n_units);
    defer allocator.free(prepared);
    @memset(prepared, .{});
    // Host work runs two banks ahead: a light down bank alone is too short to hide the next gate/up bank's factors.
    var slots: [3]Slot = undefined;
    var made: usize = 0;
    defer for (slots[0..made]) |*sl| sl.deinit();
    while (made < slots.len) : (made += 1) slots[made] = try Slot.init(allocator, geom, group);
    defer for (slots[0..made]) |*sl| if (sl.thread) |th| {
        th.join();
        sl.thread = null;
    };
    const results = try allocator.alloc(quantize.Quantized, 2 * group);
    defer allocator.free(results);

    for (0..@min(slots.len - 1, n_units)) |k| {
        slots[k].unit = units.items[k];
        try ly.hessians(&slots[k], hessians);
        slots[k].thread = try std.Thread.spawn(.{}, Layer.prep, .{ &ly, &slots[k], if (k > 0) &prepared[k - 1] else null, &prepared[k] });
    }
    for (units.items, 0..) |unit, u| {
        const cur = &slots[u % slots.len];
        const v = u + slots.len - 1;
        var ahead: ?Ahead = null;
        if (v < n_units) {
            const slot = &slots[v % slots.len];
            slot.unit = units.items[v];
            ahead = .{ .ly = &ly, .slot = slot, .hs = hessians, .after = &prepared[v - 1], .done = &prepared[v] };
        }
        var wait_lap = stages_mod.Lap.start(&stats.stages);
        cur.thread.?.join();
        cur.thread = null;
        wait_lap.mark(.wait);
        if (cur.err) |e| return e;

        const d = ly.dims(unit.down);
        const n_t = cur.tensors();
        try metal_search.finishBatch(allocator, s, feed_pool, &cur.prepared.?, policy.rate, policy.dec, &stats.stages, results[0..n_t], if (ahead) |*a| a.side() else null);
        if (ahead) |*a| while (try Ahead.step(a)) {};
        defer for (results[0..n_t]) |*q| q.deinit(allocator);
        const factored = cur.prepared.?.factor_stats;
        stats.factor_retries += factored.retries;
        stats.factor_fallbacks += factored.fallbacks;
        if (unit.down) stats.factor_down_ns += factored.ns else stats.factor_gate_up_ns += factored.ns;
        var pack_lap = stages_mod.Lap.start(&stats.stages);
        const in = d[0];
        const out = d[1];
        const tiles = (in / 16) * (out / 16) * policy.rate.n;
        for (results[0..n_t], 0..) |q, t| {
            const p: usize = switch (Layer.role(cur, t)) {
                .gate => 0,
                .up => 1,
                .down => 2,
            };
            const e = Layer.expertOf(cur, t);
            @memcpy(banks.trellis[p][e * tiles ..][0..tiles], q.packed_out);
            @memcpy(banks.suh[p][e * in ..][0..in], q.suh);
            @memcpy(banks.svh[p][e * out ..][0..out], q.svh);
        }
        cur.release();
        pack_lap.mark(.pack);
    }
    var quant_buf: [160]u8 = undefined;
    _ = try shard.writeLayer(allocator, io, dir, ly.prefix, layer, .{
        .gate = bank(geom, banks, 0),
        .up = bank(geom, banks, 1),
        .down = bank(geom, banks, 2),
        .rate = policy.rate,
        .dec = policy.dec,
        .provenance = policy.provenance,
    }, &quant_buf);
    return stats;
}

fn workerThreads() usize {
    const cpus = std.Thread.getCpuCount() catch 1;
    return @max(1, @min(cpus, 16) - 1);
}

fn bank(geom: Geometry, banks: Banks, p: usize) shard.Bank {
    const in = if (p == 2) geom.inter else geom.hidden;
    const out = if (p == 2) geom.hidden else geom.inter;
    return .{ .experts = geom.experts, .in_features = in, .out_features = out, .trellis = banks.trellis[p], .suh = banks.suh[p], .svh = banks.svh[p] };
}

pub const tiny = struct {
    pub const experts = 3;
    pub const hidden = 128;
    pub const inter = 128;

    pub fn writeSource(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, prng: *std.Random.DefaultPrng, layer_prefix: []const u8) !void {
        return writeSourceDims(allocator, io, dir, prng, layer_prefix, experts, hidden, inter);
    }

    pub fn writeSourceDims(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, prng: *std.Random.DefaultPrng, layer_prefix: []const u8, n_experts: usize, n_hidden: usize, n_inter: usize) !void {
        const r = prng.random();
        const gu = try allocator.alloc(u16, n_experts * 2 * n_inter * n_hidden);
        defer allocator.free(gu);
        const dn = try allocator.alloc(u16, n_experts * n_hidden * n_inter);
        defer allocator.free(dn);
        for (gu) |*v| v.* = @truncate(@as(u32, @bitCast(r.floatNorm(f32) * 0.05)) >> 16);
        for (dn) |*v| v.* = @truncate(@as(u32, @bitCast(r.floatNorm(f32) * 0.05)) >> 16);
        const gu_bytes = std.mem.sliceAsBytes(gu);
        const dn_bytes = std.mem.sliceAsBytes(dn);
        const header = try std.fmt.allocPrint(
            allocator,
            "{{\"{s}.0.mlp.experts.gate_up_proj\":{{\"dtype\":\"BF16\",\"shape\":[{d},{d},{d}],\"data_offsets\":[0,{d}]}},\"{s}.0.mlp.experts.down_proj\":{{\"dtype\":\"BF16\",\"shape\":[{d},{d},{d}],\"data_offsets\":[{d},{d}]}}}}",
            .{ layer_prefix, n_experts, 2 * n_inter, n_hidden, gu_bytes.len, layer_prefix, n_experts, n_hidden, n_inter, gu_bytes.len, gu_bytes.len + dn_bytes.len },
        );
        defer allocator.free(header);
        var file = std.ArrayList(u8).empty;
        defer file.deinit(allocator);
        var padded = std.ArrayList(u8).empty;
        defer padded.deinit(allocator);
        try padded.appendSlice(allocator, header);
        while (padded.items.len % 8 != 0) try padded.append(allocator, ' ');
        var len: [8]u8 = undefined;
        std.mem.writeInt(u64, &len, padded.items.len, .little);
        try file.appendSlice(allocator, &len);
        try file.appendSlice(allocator, padded.items);
        try file.appendSlice(allocator, gu_bytes);
        try file.appendSlice(allocator, dn_bytes);
        try dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = file.items });
    }
};

const FixedHessians = struct {
    rows: u64,
    scale: f32,

    fn fill(self: *FixedHessians, out: []f32) u64 {
        const dim = std.math.sqrt(out.len);
        @memset(out, 0);
        for (0..dim) |i| out[i * dim + i] = self.scale;
        return self.rows;
    }

    fn gateUp(ctx: *anyopaque, expert: usize, out: []f32) anyerror!u64 {
        const self: *FixedHessians = @ptrCast(@alignCast(ctx));
        return if (expert == 1) 0 else self.fill(out);
    }

    fn down(ctx: *anyopaque, _: usize, out: []f32) anyerror!u64 {
        const self: *FixedHessians = @ptrCast(@alignCast(ctx));
        return self.fill(out);
    }

    fn source(self: *FixedHessians) HessianSource {
        return .{ .ctx = self, .gate_up = gateUp, .down = down };
    }
};

test "convert: a tiny layer converts to a shard that decodes near its source, and a starved expert falls back to identity" {
    const t = std.testing;
    const io = t.io;
    var prng = std.Random.DefaultPrng.init(77);
    var src_dir = t.tmpDir(.{});
    defer src_dir.cleanup();
    try tiny.writeSource(t.allocator, io, src_dir.dir, &prng, "model.language_model.layers");
    var src = try source.SafetensorsSource.open(t.allocator, io, src_dir.dir);
    defer src.deinit();

    var out_dir = t.tmpDir(.{});
    defer out_dir.cleanup();
    const rate = format.Rate{ .n = 48 };
    const dec = format.Decode{ .codebook = .mcg, .window = .w12 };
    var fixed = FixedHessians{ .rows = 500, .scale = 40 };
    const stats = try convertQwenLayer(
        t.allocator,
        mlx.gpuStream(),
        io,
        out_dir.dir,
        &src,
        0,
        false,
        .{ .experts = tiny.experts, .hidden = tiny.hidden, .inter = tiny.inter },
        .{ .rate = rate, .dec = dec },
        fixed.source(),
    );
    try t.expectEqual(@as(usize, 1), stats.gate_up_identity);
    try t.expectEqual(@as(usize, 0), stats.down_identity);
    for ([_]stages_mod.Stage{ .read, .hessian, .prep, .scale, .feedback, .search, .pack }) |stage| try t.expect(stats.stages.get(stage) > 0);

    const raw = try out_dir.dir.readFileAlloc(io, "layer-0.safetensors", t.allocator, .limited(1 << 24));
    defer t.allocator.free(raw);
    const aligned = try t.allocator.alignedAlloc(u8, .of(u16), raw.len);
    defer t.allocator.free(aligned);
    @memcpy(aligned, raw);
    var file = try st.parse(t.allocator, aligned);
    defer file.deinit(t.allocator);

    const weight = try t.allocator.alloc(f32, tiny.hidden * tiny.inter);
    defer t.allocator.free(weight);
    const decoded = try t.allocator.alloc(u16, tiny.hidden * tiny.inter);
    defer t.allocator.free(decoded);
    for (roles, 0..) |role, p| {
        const in: usize = if (p == 2) tiny.inter else tiny.hidden;
        const out: usize = if (p == 2) tiny.hidden else tiny.inter;
        var key: [160]u8 = undefined;
        const trellis = st.slice(u16, file.tensors.get(try std.fmt.bufPrint(&key, "language_model.model.layers.0.mlp.switch_mlp.{s}.trellis", .{role.suffix()})).?);
        const suh = st.slice(u16, file.tensors.get(try std.fmt.bufPrint(&key, "language_model.model.layers.0.mlp.switch_mlp.{s}.suh", .{role.suffix()})).?);
        const svh = st.slice(u16, file.tensors.get(try std.fmt.bufPrint(&key, "language_model.model.layers.0.mlp.switch_mlp.{s}.svh", .{role.suffix()})).?);
        const tiles = (in / 16) * (out / 16) * rate.n;
        for (0..tiny.experts) |e| {
            try src.readExpertPublic(t.allocator, .qwen4_exp, 0, false, role, e, in, out, weight);
            try format.reconstructPublic(t.allocator, trellis[e * tiles ..][0..tiles], suh[e * in ..][0..in], svh[e * out ..][0..out], in, out, rate, dec, decoded);
            var num: f64 = 0;
            var den: f64 = 0;
            for (weight, decoded) |w, bits| {
                const d = @as(f64, format.f16BitsToF32(bits)) - w;
                num += d * d;
                den += @as(f64, w) * w;
            }
            try t.expect(@sqrt(num / den) < 0.45);
        }
    }
}

test "convert: the MTP head's layer reads mtp.* and writes the mtp shard with the policy's provenance" {
    const t = std.testing;
    const io = t.io;
    var prng = std.Random.DefaultPrng.init(78);
    var src_dir = t.tmpDir(.{});
    defer src_dir.cleanup();
    try tiny.writeSource(t.allocator, io, src_dir.dir, &prng, "mtp.layers");
    var src = try source.SafetensorsSource.open(t.allocator, io, src_dir.dir);
    defer src.deinit();
    var out_dir = t.tmpDir(.{});
    defer out_dir.cleanup();
    const prov = [_][2][]const u8{.{ "g_search", "on" }};
    const stats = try convertQwenLayer(
        t.allocator,
        mlx.gpuStream(),
        io,
        out_dir.dir,
        &src,
        0,
        true,
        .{ .experts = tiny.experts, .hidden = tiny.hidden, .inter = tiny.inter },
        .{ .rate = .{ .n = 32 }, .dec = .{ .codebook = .mcg, .window = .w12 }, .provenance = &prov },
        null,
    );
    try t.expectEqual(@as(usize, tiny.experts), stats.gate_up_identity);
    try t.expectEqual(@as(usize, tiny.experts), stats.down_identity);
    const raw = try out_dir.dir.readFileAlloc(io, "mtp-0.safetensors", t.allocator, .limited(1 << 24));
    defer t.allocator.free(raw);
    const aligned = try t.allocator.alignedAlloc(u8, .of(u16), raw.len);
    defer t.allocator.free(aligned);
    @memcpy(aligned, raw);
    var file = try st.parse(t.allocator, aligned);
    defer file.deinit(t.allocator);
    try t.expectEqualStrings("on", file.metadata.get("g_search").?);
    try t.expect(file.tensors.get("language_model.mtp.layers.0.mlp.switch_mlp.up_proj.trellis") != null);
}

test "convert: expert groups of any size write the same shard bytes" {
    const t = std.testing;
    const io = t.io;
    var prng = std.Random.DefaultPrng.init(79);
    var src_dir = t.tmpDir(.{});
    defer src_dir.cleanup();
    try tiny.writeSource(t.allocator, io, src_dir.dir, &prng, "model.language_model.layers");
    var src = try source.SafetensorsSource.open(t.allocator, io, src_dir.dir);
    defer src.deinit();

    const rate = format.Rate{ .n = 48 };
    const dec = format.Decode{ .codebook = .mcg, .window = .w12 };
    var shards: [3][]u8 = undefined;
    var made: usize = 0;
    defer for (shards[0..made]) |raw| t.allocator.free(raw);
    for ([_]usize{ 1, 2, 16 }) |batch| {
        var out_dir = t.tmpDir(.{});
        defer out_dir.cleanup();
        var fixed = FixedHessians{ .rows = 500, .scale = 40 };
        _ = try convertQwenLayer(
            t.allocator,
            mlx.gpuStream(),
            io,
            out_dir.dir,
            &src,
            0,
            false,
            .{ .experts = tiny.experts, .hidden = tiny.hidden, .inter = tiny.inter },
            .{ .rate = rate, .dec = dec, .batch = batch },
            fixed.source(),
        );
        shards[made] = try out_dir.dir.readFileAlloc(io, "layer-0.safetensors", t.allocator, .limited(1 << 24));
        made += 1;
    }
    try t.expectEqualSlices(u8, shards[0], shards[1]);
    try t.expectEqualSlices(u8, shards[0], shards[2]);
}
