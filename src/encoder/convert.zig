const std = @import("std");
const mlx = @import("mlx_host").mlx;
const api = @import("sushi_exl3");
const format = api.format;
const quantize = @import("quantize.zig");
const encode = @import("encode.zig");
const metal_search = @import("metal_search.zig");
const shard = @import("shard.zig");
const source = @import("source.zig");
const st = @import("safetensors.zig");
const stages_mod = @import("stages.zig");
const pool_mod = @import("pool.zig");
const gpu_factor = @import("gpu_factor.zig");
const fisher_mod = @import("fisher.zig");
const target_mod = @import("target.zig");

pub const Geometry = struct { experts: usize, hidden: usize, inter: usize };

/// One rate group of a layer: the experts (source ids, ascending) whose gate/up and down share these rates.
pub const Class = struct {
    gate_up: format.Rate,
    down: format.Rate,
    ids: []const u32,
};

pub const Policy = struct {
    rate: format.Rate,
    dec: format.Decode,
    seed: u64 = 0x5eed_e3c0_de00_0001,
    mode: quantize.OutScales = .auto,
    min_rows: u64 = 64,
    batch: usize = 16,
    factor_precision: gpu_factor.Precision = .f64,
    pin_pass: u32 = encode.default_pin_pass,
    provenance: []const [2][]const u8 = &.{},
    /// The layer's rate groups in stored order; null writes one bank of every expert at `rate`.
    classes: ?[]const Class = null,
    /// The layer's output Fisher; each tensor's output scales then come from it (`fisher.scales`).
    fisher: ?*const fisher_mod.LayerFisher = null,
    /// Round the compensated target (`target.zig`) instead of W; the Hessian source must give the cross terms.
    target: bool = false,
    /// The checkpoint the experts are read from.
    arch: source.Arch = .qwen4_exp,
    /// The bank names and shard file; null keeps the Qwen layout. A named layer is always written as rate groups.
    naming: ?shard.Naming = null,
};

pub const HessianSource = struct {
    ctx: *anyopaque,
    gate_up: *const fn (ctx: *anyopaque, expert: usize, out: []f32) anyerror!u64,
    down: *const fn (ctx: *anyopaque, expert: usize, out: []f32) anyerror!u64,
    /// For the compensated target: X~'X~ into `h` and X~'X W into `cw` (`target.zig`); gate/up's `cw` is
    /// [hidden, 2 inter] (gate then up per row), down's is [inter, hidden] over activations through the expert's
    /// quantized gate/up.
    gate_up_target: ?*const fn (ctx: *anyopaque, expert: usize, h: []f32, cw: []f32) anyerror!u64 = null,
    down_target: ?*const fn (ctx: *anyopaque, expert: usize, gate_q: []const f32, up_q: []const f32, h: []f32, cw: []f32) anyerror!u64 = null,
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

    fn init(allocator: std.mem.Allocator, geom: Geometry, cls: Class) !Banks {
        var self: Banks = undefined;
        var made: usize = 0;
        errdefer self.release(allocator, made);
        const experts = cls.ids.len;
        while (made < 3) : (made += 1) {
            const in = if (made == 2) geom.inter else geom.hidden;
            const out = if (made == 2) geom.hidden else geom.inter;
            const n = if (made == 2) cls.down.n else cls.gate_up.n;
            self.trellis[made] = try allocator.alloc(u16, experts * (in / 16) * (out / 16) * n);
            errdefer allocator.free(self.trellis[made]);
            self.suh[made] = try allocator.alloc(u16, experts * in);
            errdefer allocator.free(self.suh[made]);
            self.svh[made] = try allocator.alloc(u16, experts * out);
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
    /// Source expert ids, a run of one group's.
    ids: []const u32,
    /// Where `ids` starts inside its group.
    at: usize,
    group: usize,
    rate: format.Rate,
    down: bool,
};

const Slot = struct {
    allocator: std.mem.Allocator,
    weights: []f32,
    hessians: []f32,
    names: [][160]u8,
    signs: [][]f32,
    scales: []f32,
    jobs: []metal_search.TensorJob,
    /// Target mode only: each local expert's X~'X W and routed rows.
    cw: []f32 = &.{},
    rows: []u64 = &.{},
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
        const scales = try allocator.alloc(f32, most * @max(geom.hidden, geom.inter));
        errdefer allocator.free(scales);
        const jobs = try allocator.alloc(metal_search.TensorJob, most);
        return .{ .allocator = allocator, .weights = weights, .hessians = hessians, .names = names, .signs = signs, .scales = scales, .jobs = jobs };
    }

    fn initTarget(self: *Slot, geom: Geometry, group: usize) !void {
        self.cw = try self.allocator.alloc(f32, group * 2 * geom.hidden * geom.inter);
        self.rows = try self.allocator.alloc(u64, group);
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
        self.allocator.free(self.scales);
        self.allocator.free(self.jobs);
        self.allocator.free(self.cw);
        self.allocator.free(self.rows);
    }

    fn tensors(self: *const Slot) usize {
        return if (self.unit.down) self.unit.ids.len else 2 * self.unit.ids.len;
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
    /// Target mode: each expert's quantized gate and up as public F16 bits ([experts][2][hidden * inter]), filled as
    /// the gate/up banks finish; the down Hessians run through them.
    decoded: []u16 = &.{},
    decoded_ready: []std.atomic.Value(bool) = &.{},
    /// Target mode: the threads decoding finished gate/up banks into `decoded`, joined before a down Hessian needs one.
    decoding: ?*std.ArrayList(std.Thread) = null,
    /// Target mode: down Hessians computed early, while the GPU would wait on the gate/up banks' host work.
    early: ?*EarlyDown = null,
    /// Target mode: the workers the target solves spread over.
    target_pool: ?*metal_search.Pool = null,

    fn dims(self: *const Layer, down: bool) [2]usize {
        return if (down) .{ self.geom.inter, self.geom.hidden } else .{ self.geom.hidden, self.geom.inter };
    }

    /// The tensor at `t` of a slot: gate/up units interleave gate(e), up(e) so each pair shares one Hessian.
    fn role(slot: *const Slot, t: usize) source.ProjRole {
        return if (slot.unit.down) .down else roles[t % 2];
    }

    /// The tensor's position in its slot's run of experts.
    fn localOf(slot: *const Slot, t: usize) usize {
        return if (slot.unit.down) t else t / 2;
    }

    fn expertOf(slot: *const Slot, t: usize) usize {
        return slot.unit.ids[localOf(slot, t)];
    }

    fn hessians(self: *const Layer, slot: *Slot, hs: ?HessianSource) !void {
        for (0..slot.unit.ids.len) |b| try self.hessianOne(slot, b, hs);
    }

    fn hessianOne(self: *const Layer, slot: *Slot, b: usize, hs: ?HessianSource) !void {
        const in = self.dims(slot.unit.down)[0];
        const h = slot.hessians[b * in * in ..][0 .. in * in];
        var lap = stages_mod.Lap.start(&self.stats.stages);
        var rows: u64 = 0;
        if (self.policy.target) {
            rows = try self.targetTerms(slot, b, hs orelse return error.TargetNeedsCapture, h);
        } else if (hs) |src| rows = try (if (slot.unit.down) src.down else src.gate_up)(src.ctx, slot.unit.ids[b], h);
        lap.mark(.hessian);
        if (hs == null or rows < self.policy.min_rows) {
            identityHessian(h, in);
            if (slot.unit.down) self.stats.down_identity += 1 else self.stats.gate_up_identity += 1;
        }
    }

    fn targetTerms(self: *const Layer, slot: *Slot, b: usize, src: HessianSource, h: []f32) !u64 {
        const hid = self.geom.hidden;
        const i = self.geom.inter;
        const e = slot.unit.ids[b];
        const cw = slot.cw[b * 2 * hid * i ..][0 .. 2 * hid * i];
        const rows = if (!slot.unit.down)
            try (src.gate_up_target orelse return error.TargetNeedsCapture)(src.ctx, e, h, cw)
        else blk: {
            if (self.early) |ed| if (ed.take(e, h, cw[0 .. i * hid])) |r| break :blk r;
            break :blk try self.downTerms(e, h, cw[0 .. i * hid], src);
        };
        slot.rows[b] = rows;
        return rows;
    }

    /// The down projection's target terms of expert `e`, through its rounded gate/up.
    fn downTerms(self: *const Layer, e: usize, h: []f32, cw: []f32, src: HessianSource) !u64 {
        const hid = self.geom.hidden;
        const i = self.geom.inter;
        if (!self.decoded_ready[e].load(.acquire)) {
            for (self.decoding.?.items) |th| th.join();
            self.decoding.?.clearRetainingCapacity();
            if (!self.decoded_ready[e].load(.acquire)) return error.TargetGateUpNotRounded;
        }
        const q = try self.allocator.alloc(f32, 2 * hid * i);
        defer self.allocator.free(q);
        for (q, self.decoded[e * 2 * hid * i ..][0 .. 2 * hid * i]) |*o, bits| o.* = format.f16BitsToF32(bits);
        return (src.down_target orelse return error.TargetNeedsCapture)(src.ctx, e, q[0 .. hid * i], q[hid * i ..], h, cw);
    }

    /// Replaces a slot's weights with their compensated targets: each expert's factor, then its solves in column
    /// chunks across the pool; gate and up share one factor.
    const Compensator = struct {
        layer: *const Layer,
        slot: *Slot,
        factors: []?target_mod.Factor,
        alphas: []f64,
        /// Gate/up side by side per expert ([hidden, 2 inter]); unused for down.
        joined: [][]f32,
        failed: std.atomic.Value(bool) = .init(false),

        const chunks = 8;

        fn width(self: *const Compensator) usize {
            const ly = self.layer;
            return if (self.slot.unit.down) ly.geom.hidden else 2 * ly.geom.inter;
        }

        fn weightOf(self: *const Compensator, b: usize) []f32 {
            const ly = self.layer;
            const hid = ly.geom.hidden;
            const i = ly.geom.inter;
            return if (self.slot.unit.down) self.slot.weights[b * i * hid ..][0 .. i * hid] else self.joined[b];
        }

        fn factor(self: *Compensator, b: usize) void {
            self.factorOne(b) catch {
                self.failed.store(true, .monotonic);
            };
        }

        fn factorOne(self: *Compensator, b: usize) !void {
            const ly = self.layer;
            const slot = self.slot;
            const rows = slot.rows[b];
            const in = ly.dims(slot.unit.down)[0];
            self.alphas[b] = if (rows < ly.policy.min_rows) 0 else target_mod.alphaForRows(rows, in);
            if (self.alphas[b] == 0) return;
            self.factors[b] = try target_mod.Factor.init(ly.allocator, slot.hessians[b * in * in ..][0 .. in * in], in);
            if (slot.unit.down) return;
            const hid = ly.geom.hidden;
            const i = ly.geom.inter;
            const both = try ly.allocator.alloc(f32, hid * 2 * i);
            const g = slot.weights[2 * b * hid * i ..][0 .. hid * i];
            const u = slot.weights[(2 * b + 1) * hid * i ..][0 .. hid * i];
            for (0..hid) |r| {
                @memcpy(both[r * 2 * i ..][0..i], g[r * i ..][0..i]);
                @memcpy(both[r * 2 * i + i ..][0..i], u[r * i ..][0..i]);
            }
            self.joined[b] = both;
        }

        fn solve(self: *Compensator, job: usize) void {
            const b = job / chunks;
            const f = self.factors[b] orelse return;
            const k = self.width();
            const c0 = k * (job % chunks) / chunks;
            const c1 = k * (job % chunks + 1) / chunks;
            const cw = self.slot.cw[b * 2 * self.layer.geom.hidden * self.layer.geom.inter ..][0 .. f.n * k];
            f.apply(self.layer.allocator, self.weightOf(b), cw, k, c0, c1, self.alphas[b]) catch {
                self.failed.store(true, .monotonic);
            };
        }

        fn finish(self: *Compensator) void {
            const ly = self.layer;
            const hid = ly.geom.hidden;
            const i = ly.geom.inter;
            for (self.factors, 0..) |*f, b| {
                if (f.*) |*fac| fac.deinit(ly.allocator);
                f.* = null;
                if (self.slot.unit.down or self.joined[b].len == 0) continue;
                const both = self.joined[b];
                const g = self.slot.weights[2 * b * hid * i ..][0 .. hid * i];
                const u = self.slot.weights[(2 * b + 1) * hid * i ..][0 .. hid * i];
                for (0..hid) |r| {
                    @memcpy(g[r * i ..][0..i], both[r * 2 * i ..][0..i]);
                    @memcpy(u[r * i ..][0..i], both[r * 2 * i + i ..][0..i]);
                }
                ly.allocator.free(both);
                self.joined[b] = &.{};
            }
        }

        fn run(ly: *const Layer, slot: *Slot) !void {
            const n = slot.unit.ids.len;
            const factors = try ly.allocator.alloc(?target_mod.Factor, n);
            defer ly.allocator.free(factors);
            @memset(factors, null);
            const alphas = try ly.allocator.alloc(f64, n);
            defer ly.allocator.free(alphas);
            const joined = try ly.allocator.alloc([]f32, n);
            defer ly.allocator.free(joined);
            @memset(joined, &.{});
            var c = Compensator{ .layer = ly, .slot = slot, .factors = factors, .alphas = alphas, .joined = joined };
            defer c.finish();
            const pool = ly.target_pool orelse ly.pool;
            pool.run(n, &c, factor);
            if (c.failed.load(.monotonic)) return error.TargetSolveFailed;
            pool.run(n * chunks, &c, solve);
            if (c.failed.load(.monotonic)) return error.TargetSolveFailed;
        }
    };

    const Scaler = struct {
        layer: *const Layer,
        slot: *Slot,
        f: *const fisher_mod.LayerFisher,

        fn run(self: *Scaler, t: usize) void {
            const d = self.layer.dims(self.slot.unit.down);
            const sc = self.slot.scales[t * d[1] ..][0..d[1]];
            fisher_mod.scales(self.slot.weights[t * d[0] * d[1] ..][0 .. d[0] * d[1]], d[0], d[1], self.f.row(role(self.slot, t), expertOf(self.slot, t)), sc);
            self.slot.jobs[t].out_scales = sc;
        }
    };

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
            try ly.src.readExpertPublic(ly.allocator, ly.policy.arch, ly.layer, ly.mtp, r, e, in, out, weight);
            const name = if (ly.policy.naming) |nm|
                try std.fmt.bufPrint(&slot.names[t], "{s}.mlp.switch_mlp.{s}.e{d}", .{ nm.nest, r.suffix(), e })
            else
                try std.fmt.bufPrint(&slot.names[t], "{s}.layers.{d}.mlp.switch_mlp.{s}.e{d}", .{ ly.prefix, ly.layer, r.suffix(), e });
            slot.signs[t] = try quantize.signs(ly.allocator, ly.policy.seed, name, in + out);
            const local = localOf(slot, t);
            slot.jobs[t] = .{ .w = weight, .h = slot.hessians[local * in * in ..][0 .. in * in], .sign_in = slot.signs[t][0..in], .sign_out = slot.signs[t][in..] };
        }
    };

    /// The slot's host work, on a thread of its own, in two stages that each follow the bank before's same stage:
    /// read, target and output scales, then regularize, rotate and factor. In target mode the first stage has a
    /// pool of its own, so a bank's target solves overlap the bank before's factors.
    fn prep(self: *const Layer, slot: *Slot, chain: *const Chain, v: usize) void {
        if (v > 0) (if (self.target_pool != null) &chain.targeted[v - 1] else &chain.prepared[v - 1]).wait();
        self.prepTarget(slot) catch |e| {
            slot.err = e;
        };
        chain.targeted[v].signal();
        if (v > 0) chain.prepared[v - 1].wait();
        if (slot.err == null) self.prepFactor(slot) catch |e| {
            slot.err = e;
        };
        chain.prepared[v].signal();
    }

    fn prepTarget(self: *const Layer, slot: *Slot) !void {
        const pool = self.target_pool orelse self.pool;
        var lap = stages_mod.Lap.start(&self.stats.stages);
        var reader = Reader{ .layer = self, .slot = slot };
        pool.run(slot.tensors(), &reader, Reader.read);
        if (reader.failed.load(.monotonic)) return error.SourceReadFailed;
        lap.mark(.read);
        if (self.policy.target) {
            try Compensator.run(self, slot);
            lap.mark(.target);
        }
        if (self.policy.fisher) |f| {
            var scaler = Scaler{ .layer = self, .slot = slot, .f = f };
            pool.run(slot.tensors(), &scaler, Scaler.run);
            lap.mark(.prep);
        }
    }

    fn prepFactor(self: *const Layer, slot: *Slot) !void {
        const d = self.dims(slot.unit.down);
        var lap = stages_mod.Lap.start(&self.stats.stages);
        slot.prepared = try metal_search.prepareBatch(self.allocator, self.pool, slot.jobs[0..slot.tensors()], d[0], d[1], self.policy.mode, self.policy.factor_precision);
        lap.mark(.prep);
    }
};

/// Per bank, when each stage of its host work is done.
const Chain = struct {
    targeted: []pool_mod.Event,
    prepared: []pool_mod.Event,
};

/// Target mode: a finished gate/up bank's public reconstructions, for the down Hessians, decoded from the layer's
/// banks on a thread of its own.
fn decodeUnit(ly: *const Layer, banks: *const Banks, unit: Unit) void {
    const hid = ly.geom.hidden;
    const i = ly.geom.inter;
    const tiles = (hid / 16) * (i / 16) * unit.rate.n;
    for (unit.ids, 0..) |e, local| {
        const at = unit.at + local;
        for (0..2) |side| {
            format.reconstructPublic(ly.allocator, banks.trellis[side][at * tiles ..][0..tiles], banks.suh[side][at * hid ..][0..hid], banks.svh[side][at * i ..][0..i], hid, i, unit.rate, ly.policy.dec, ly.decoded[(e * 2 + side) * hid * i ..][0 .. hid * i]) catch return;
        }
        ly.decoded_ready[e].store(true, .release);
    }
}

/// Down Hessians and target terms computed ahead of their banks, in down-bank order, each as soon as its expert's
/// gate/up is decoded. The main thread fills them while it waits on a gate/up bank's host work.
const EarlyDown = struct {
    allocator: std.mem.Allocator,
    order: []const u32,
    next: usize = 0,
    h: []f32,
    cw: []f32,
    rows: []u64,
    ready: []bool,
    in: usize,
    out: usize,

    fn init(allocator: std.mem.Allocator, order: []const u32, geom: Geometry) !EarlyDown {
        const i = geom.inter;
        const hid = geom.hidden;
        const h = try allocator.alloc(f32, geom.experts * i * i);
        errdefer allocator.free(h);
        const cw = try allocator.alloc(f32, geom.experts * i * hid);
        errdefer allocator.free(cw);
        const rows = try allocator.alloc(u64, geom.experts);
        errdefer allocator.free(rows);
        const ready = try allocator.alloc(bool, geom.experts);
        @memset(ready, false);
        return .{ .allocator = allocator, .order = order, .h = h, .cw = cw, .rows = rows, .ready = ready, .in = i, .out = hid };
    }

    fn deinit(self: *EarlyDown) void {
        self.allocator.free(self.h);
        self.allocator.free(self.cw);
        self.allocator.free(self.rows);
        self.allocator.free(self.ready);
    }

    /// Computes the next expert's terms when its gate/up is decoded; false when it is not yet, or none are left.
    fn one(self: *EarlyDown, ly: *const Layer, src: HessianSource) !bool {
        if (self.next >= self.order.len) return false;
        const e = self.order[self.next];
        if (!ly.decoded_ready[e].load(.acquire)) return false;
        const n = self.in;
        var lap = stages_mod.Lap.start(&ly.stats.stages);
        self.rows[e] = try ly.downTerms(e, self.h[e * n * n ..][0 .. n * n], self.cw[e * n * self.out ..][0 .. n * self.out], src);
        lap.mark(.hessian);
        self.ready[e] = true;
        self.next += 1;
        return true;
    }

    fn take(self: *EarlyDown, e: usize, h: []f32, cw: []f32) ?u64 {
        if (!self.ready[e]) return null;
        const n = self.in;
        @memcpy(h, self.h[e * n * n ..][0 .. n * n]);
        @memcpy(cw, self.cw[e * n * self.out ..][0 .. n * self.out]);
        return self.rows[e];
    }
};

/// The bank two ahead: its Hessians, one expert per LDLQ step of the current bank, then its host prep.
const Ahead = struct {
    ly: *const Layer,
    slot: *Slot,
    hs: ?HessianSource,
    chain: *const Chain,
    v: usize,
    next: usize = 0,

    fn step(raw: *anyopaque) anyerror!bool {
        const self: *Ahead = @ptrCast(@alignCast(raw));
        if (self.next < self.slot.unit.ids.len) {
            try self.ly.hessianOne(self.slot, self.next, self.hs);
            self.next += 1;
        }
        if (self.next < self.slot.unit.ids.len) return true;
        if (self.slot.thread == null) self.slot.thread = try std.Thread.spawn(.{}, Layer.prep, .{ self.ly, self.slot, self.chain, self.v });
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
    const all_ids = try allocator.alloc(u32, geom.experts);
    defer allocator.free(all_ids);
    for (all_ids, 0..) |*v, e| v.* = @intCast(e);
    const uniform = [_]Class{.{ .gate_up = policy.rate, .down = policy.rate, .ids = all_ids }};
    const classes = policy.classes orelse &uniform;
    const seen = try allocator.alloc(bool, geom.experts);
    defer allocator.free(seen);
    @memset(seen, false);
    var covered: usize = 0;
    for (classes) |c| for (c.ids) |e| {
        if (e >= geom.experts or seen[e]) return error.RateGroupsInvalid;
        seen[e] = true;
        covered += 1;
    };
    if (covered != geom.experts) return error.RateGroupsInvalid;

    const all_banks = try allocator.alloc(Banks, classes.len);
    defer allocator.free(all_banks);
    var banks_made: usize = 0;
    defer for (all_banks[0..banks_made]) |*b| b.deinit(allocator);
    while (banks_made < classes.len) : (banks_made += 1) all_banks[banks_made] = try Banks.init(allocator, geom, classes[banks_made]);

    const group = @max(1, @min(policy.batch, geom.experts));
    var units = std.ArrayList(Unit).empty;
    defer units.deinit(allocator);
    // The target's down Hessians run through the experts' rounded gate/up, so in target mode every gate/up bank
    // comes first; the down banks' Hessians, computed two banks ahead, then always find theirs rounded.
    const passes: []const ?bool = if (policy.target) &.{ false, true } else &.{null};
    for (passes) |pass| for (classes, 0..) |c, g| {
        var first: usize = 0;
        while (first < c.ids.len) : (first += group) {
            const ids = c.ids[first..@min(first + group, c.ids.len)];
            if (pass != true) try units.append(allocator, .{ .ids = ids, .at = first, .group = g, .rate = c.gate_up, .down = false });
            if (pass != false) try units.append(allocator, .{ .ids = ids, .at = first, .group = g, .rate = c.down, .down = true });
        }
    };
    if (policy.target and units.items.len < 6) return error.TargetNeedsThreeBanks;
    const decoded = try allocator.alloc(u16, if (policy.target) geom.experts * 2 * geom.hidden * geom.inter else 0);
    defer allocator.free(decoded);
    const decoded_ready = try allocator.alloc(std.atomic.Value(bool), if (policy.target) geom.experts else 0);
    defer allocator.free(decoded_ready);
    @memset(decoded_ready, .init(false));
    var decoding = std.ArrayList(std.Thread).empty;
    defer {
        for (decoding.items) |th| th.join();
        decoding.deinit(allocator);
    }

    var stats = LayerStats{ .experts = geom.experts };
    const down_order = try allocator.alloc(u32, if (policy.target) geom.experts else 0);
    defer allocator.free(down_order);
    var early: ?EarlyDown = null;
    defer if (early) |*ed| ed.deinit();
    if (policy.target and hessians != null) {
        var at: usize = 0;
        for (units.items) |unit| if (unit.down) for (unit.ids) |e| {
            down_order[at] = e;
            at += 1;
        };
        early = try EarlyDown.init(allocator, down_order[0..at], geom);
    }
    const prep_pool = try metal_search.Pool.init(allocator, workerThreads() / 2);
    defer prep_pool.deinit();
    const feed_pool = try metal_search.Pool.init(allocator, workerThreads());
    defer feed_pool.deinit();
    const target_pool = if (policy.target) try metal_search.Pool.init(allocator, workerThreads()) else null;
    defer if (target_pool) |tp| tp.deinit();
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
        .decoded = decoded,
        .decoded_ready = decoded_ready,
        .decoding = &decoding,
        .target_pool = target_pool,
        .early = if (early) |*ed| ed else null,
    };
    const n_units = units.items.len;
    const events = try allocator.alloc(pool_mod.Event, 2 * n_units);
    defer allocator.free(events);
    @memset(events, .{});
    const chain = Chain{ .targeted = events[0..n_units], .prepared = events[n_units..] };
    // Host work runs two banks ahead: a light down bank alone is too short to hide the next gate/up bank's factors.
    var slots: [3]Slot = undefined;
    var made: usize = 0;
    defer for (slots[0..made]) |*sl| sl.deinit();
    while (made < slots.len) : (made += 1) {
        slots[made] = try Slot.init(allocator, geom, group);
        if (policy.target) slots[made].initTarget(geom, group) catch |e| {
            made += 1;
            return e;
        };
    }
    defer for (slots[0..made]) |*sl| if (sl.thread) |th| {
        th.join();
        sl.thread = null;
    };
    const results = try allocator.alloc(quantize.Quantized, 2 * group);
    defer allocator.free(results);

    for (0..@min(slots.len - 1, n_units)) |k| {
        slots[k].unit = units.items[k];
        try ly.hessians(&slots[k], hessians);
        slots[k].thread = try std.Thread.spawn(.{}, Layer.prep, .{ &ly, &slots[k], &chain, k });
    }
    for (units.items, 0..) |unit, u| {
        const cur = &slots[u % slots.len];
        const v = u + slots.len - 1;
        var ahead: ?Ahead = null;
        if (v < n_units) {
            const slot = &slots[v % slots.len];
            slot.unit = units.items[v];
            ahead = .{ .ly = &ly, .slot = slot, .hs = hessians, .chain = &chain, .v = v };
        }
        if (early) |*ed| while (!chain.prepared[u].isSet()) {
            if (!try ed.one(&ly, hessians.?)) break;
        };
        var wait_lap = stages_mod.Lap.start(&stats.stages);
        cur.thread.?.join();
        cur.thread = null;
        wait_lap.mark(.wait);
        if (cur.err) |e| return e;

        const d = ly.dims(unit.down);
        const n_t = cur.tensors();
        try metal_search.finishBatch(allocator, s, feed_pool, &cur.prepared.?, unit.rate, policy.dec, policy.pin_pass, &stats.stages, results[0..n_t], if (ahead) |*a| a.side() else null);
        if (ahead) |*a| while (try Ahead.step(a)) {};
        defer for (results[0..n_t]) |*q| q.deinit(allocator);
        const factored = cur.prepared.?.factor_stats;
        stats.factor_retries += factored.retries;
        stats.factor_fallbacks += factored.fallbacks;
        if (unit.down) stats.factor_down_ns += factored.ns else stats.factor_gate_up_ns += factored.ns;
        var pack_lap = stages_mod.Lap.start(&stats.stages);
        const in = d[0];
        const out = d[1];
        const tiles = (in / 16) * (out / 16) * unit.rate.n;
        const banks = &all_banks[unit.group];
        for (results[0..n_t], 0..) |q, t| {
            const p: usize = switch (Layer.role(cur, t)) {
                .gate => 0,
                .up => 1,
                .down => 2,
            };
            const e = unit.at + Layer.localOf(cur, t);
            @memcpy(banks.trellis[p][e * tiles ..][0..tiles], q.packed_out);
            @memcpy(banks.suh[p][e * in ..][0..in], q.suh);
            @memcpy(banks.svh[p][e * out ..][0..out], q.svh);
        }
        if (policy.target and !unit.down) try decoding.append(allocator, try std.Thread.spawn(.{}, decodeUnit, .{ &ly, banks, unit }));
        cur.release();
        pack_lap.mark(.pack);
    }
    if (policy.classes == null and policy.naming == null) {
        var quant_buf: [160]u8 = undefined;
        _ = try shard.writeLayer(allocator, io, dir, ly.prefix, layer, .{
            .gate = bank(geom, all_banks[0], 0, geom.experts),
            .up = bank(geom, all_banks[0], 1, geom.experts),
            .down = bank(geom, all_banks[0], 2, geom.experts),
            .rate = policy.rate,
            .dec = policy.dec,
            .provenance = policy.provenance,
        }, &quant_buf);
        return stats;
    }
    const groups = try allocator.alloc(shard.Group, classes.len);
    defer allocator.free(groups);
    for (groups, classes, all_banks) |*g, c, b| g.* = .{
        .banks = .{ bank(geom, b, 0, c.ids.len), bank(geom, b, 1, c.ids.len), bank(geom, b, 2, c.ids.len) },
        .rates = .{ c.gate_up, c.gate_up, c.down },
    };
    if (policy.naming) |nm|
        try shard.writeGroupsNamed(allocator, io, dir, nm, groups, policy.dec, policy.provenance)
    else
        try shard.writeGroups(allocator, io, dir, ly.prefix, layer, groups, policy.dec, policy.provenance);
    return stats;
}

fn workerThreads() usize {
    const cpus = pool_mod.cpuCount();
    return @max(1, @min(cpus, 16) - 1);
}

fn bank(geom: Geometry, banks: Banks, p: usize, experts: usize) shard.Bank {
    const in = if (p == 2) geom.inter else geom.hidden;
    const out = if (p == 2) geom.hidden else geom.inter;
    return .{ .experts = experts, .in_features = in, .out_features = out, .trellis = banks.trellis[p], .suh = banks.suh[p], .svh = banks.svh[p] };
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
    try mlx.requireGpu();
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
    try mlx.requireGpu();
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
    try mlx.requireGpu();
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

test "convert: a layer under the default policy is the pin-64 layer, and not the full-pass one" {
    const t = std.testing;
    const io = t.io;
    var prng = std.Random.DefaultPrng.init(79);
    var src_dir = t.tmpDir(.{});
    defer src_dir.cleanup();
    try tiny.writeSource(t.allocator, io, src_dir.dir, &prng, "model.language_model.layers");
    var src = try source.SafetensorsSource.open(t.allocator, io, src_dir.dir);
    defer src.deinit();

    const base = Policy{ .rate = .{ .n = 48 }, .dec = .{ .codebook = .mcg, .window = .w12 } };
    var shards: [3][]u8 = undefined;
    var made: usize = 0;
    defer for (shards[0..made]) |raw| t.allocator.free(raw);
    var explicit_64 = base;
    explicit_64.pin_pass = 64;
    var full = base;
    full.pin_pass = encode.full_pin_pass;
    for ([_]Policy{ base, explicit_64, full }) |policy| {
        var out_dir = t.tmpDir(.{});
        defer out_dir.cleanup();
        var fixed = FixedHessians{ .rows = 500, .scale = 40 };
        _ = try convertQwenLayer(t.allocator, mlx.gpuStream(), io, out_dir.dir, &src, 0, false, .{ .experts = tiny.experts, .hidden = tiny.hidden, .inter = tiny.inter }, policy, fixed.source());
        shards[made] = try out_dir.dir.readFileAlloc(io, "layer-0.safetensors", t.allocator, .limited(1 << 24));
        made += 1;
    }
    try t.expectEqualSlices(u8, shards[0], shards[1]);
    try t.expect(!std.mem.eql(u8, shards[0], shards[2]));
}

const ReadShard = struct { bytes: []align(2) u8, file: st.File };

fn readShard(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, name: []const u8) !ReadShard {
    const raw = try dir.readFileAlloc(io, name, allocator, .limited(1 << 24));
    defer allocator.free(raw);
    const aligned = try allocator.alignedAlloc(u8, .of(u16), raw.len);
    errdefer allocator.free(aligned);
    @memcpy(aligned, raw);
    return .{ .bytes = aligned, .file = try st.parse(allocator, aligned) };
}

test "convert: a rate-grouped layer writes each expert at its group's rates with the bytes a uniform layer at that rate gives it" {
    const t = std.testing;
    const io = t.io;
    var prng = std.Random.DefaultPrng.init(80);
    var src_dir = t.tmpDir(.{});
    defer src_dir.cleanup();
    try tiny.writeSource(t.allocator, io, src_dir.dir, &prng, "model.language_model.layers");
    var src = try source.SafetensorsSource.open(t.allocator, io, src_dir.dir);
    defer src.deinit();
    const geom = Geometry{ .experts = tiny.experts, .hidden = tiny.hidden, .inter = tiny.inter };
    const dec = format.Decode{ .codebook = .mcg, .window = .w12 };
    const wide = format.Rate{ .n = 48 };
    const narrow = format.Rate{ .n = 32 };

    var uniform: [2]ReadShard = undefined;
    var made: usize = 0;
    defer for (uniform[0..made]) |*u| {
        u.file.deinit(t.allocator);
        t.allocator.free(u.bytes);
    };
    for ([_]format.Rate{ wide, narrow }) |rate| {
        var out_dir = t.tmpDir(.{});
        defer out_dir.cleanup();
        var fixed = FixedHessians{ .rows = 500, .scale = 40 };
        _ = try convertQwenLayer(t.allocator, mlx.gpuStream(), io, out_dir.dir, &src, 0, false, geom, .{ .rate = rate, .dec = dec, .batch = 2 }, fixed.source());
        uniform[made] = try readShard(t.allocator, io, out_dir.dir, "layer-0.safetensors");
        made += 1;
    }

    const classes = [_]Class{
        .{ .gate_up = wide, .down = narrow, .ids = &.{ 0, 2 } },
        .{ .gate_up = narrow, .down = wide, .ids = &.{1} },
    };
    var out_dir = t.tmpDir(.{});
    defer out_dir.cleanup();
    var fixed = FixedHessians{ .rows = 500, .scale = 40 };
    _ = try convertQwenLayer(t.allocator, mlx.gpuStream(), io, out_dir.dir, &src, 0, false, geom, .{ .rate = wide, .dec = dec, .batch = 2, .classes = &classes }, fixed.source());
    var grouped = try readShard(t.allocator, io, out_dir.dir, "layer-0.safetensors");
    defer {
        grouped.file.deinit(t.allocator);
        t.allocator.free(grouped.bytes);
    }
    try t.expect(grouped.file.tensors.get("language_model.model.layers.0.mlp.switch_mlp.gate_proj.trellis") == null);
    for (classes, 0..) |c, g| for (roles) |role| {
        const rate = if (role == .down) c.down else c.gate_up;
        const from = &uniform[if (rate.n == wide.n) 0 else 1];
        for ([_][]const u8{ "trellis", "suh", "svh" }) |part| {
            var key: [160]u8 = undefined;
            const got = st.slice(u16, grouped.file.tensors.get(try std.fmt.bufPrint(&key, "language_model.model.layers.0.mlp.switch_mlp.{s}.g{d}.{s}", .{ role.suffix(), g, part })).?);
            const all = st.slice(u16, from.file.tensors.get(try std.fmt.bufPrint(&key, "language_model.model.layers.0.mlp.switch_mlp.{s}.{s}", .{ role.suffix(), part })).?);
            const per = all.len / tiny.experts;
            try t.expectEqual(c.ids.len * per, got.len);
            for (c.ids, 0..) |e, local| try t.expectEqualSlices(u16, all[e * per ..][0..per], got[local * per ..][0..per]);
        }
    };
}

test "convert: an output Fisher moves each expert's error out of the channels it weights" {
    const t = std.testing;
    const io = t.io;
    var prng = std.Random.DefaultPrng.init(81);
    var src_dir = t.tmpDir(.{});
    defer src_dir.cleanup();
    try tiny.writeSource(t.allocator, io, src_dir.dir, &prng, "model.language_model.layers");
    var src = try source.SafetensorsSource.open(t.allocator, io, src_dir.dir);
    defer src.deinit();
    const geom = Geometry{ .experts = tiny.experts, .hidden = tiny.hidden, .inter = tiny.inter };
    const rate = format.Rate{ .n = 32 };
    const dec = format.Decode{ .codebook = .mcg, .window = .w12 };

    var lf = fisher_mod.LayerFisher{
        .allocator = t.allocator,
        .experts = tiny.experts,
        .hidden = tiny.hidden,
        .inter = tiny.inter,
        .gate = try t.allocator.alloc(f32, tiny.experts * tiny.inter),
        .up = try t.allocator.alloc(f32, tiny.experts * tiny.inter),
        .down = try t.allocator.alloc(f32, tiny.experts * tiny.hidden),
    };
    defer lf.deinit();
    for ([_][]f32{ lf.gate, lf.up, lf.down }) |side| for (side, 0..) |*v, i| {
        v.* = if (i % 128 < 64) 100 else 1;
    };

    var ratio: [2]f64 = undefined;
    for ([_]?*const fisher_mod.LayerFisher{ null, &lf }, 0..) |f, arm| {
        var out_dir = t.tmpDir(.{});
        defer out_dir.cleanup();
        var fixed = FixedHessians{ .rows = 500, .scale = 40 };
        _ = try convertQwenLayer(t.allocator, mlx.gpuStream(), io, out_dir.dir, &src, 0, false, geom, .{ .rate = rate, .dec = dec, .fisher = f }, fixed.source());
        var shard_file = try readShard(t.allocator, io, out_dir.dir, "layer-0.safetensors");
        defer {
            shard_file.file.deinit(t.allocator);
            t.allocator.free(shard_file.bytes);
        }
        var hi: f64 = 0;
        var lo: f64 = 0;
        const weight = try t.allocator.alloc(f32, tiny.hidden * tiny.inter);
        defer t.allocator.free(weight);
        const decoded = try t.allocator.alloc(u16, tiny.hidden * tiny.inter);
        defer t.allocator.free(decoded);
        for (roles) |role| {
            var key: [160]u8 = undefined;
            const tr = st.slice(u16, shard_file.file.tensors.get(try std.fmt.bufPrint(&key, "language_model.model.layers.0.mlp.switch_mlp.{s}.trellis", .{role.suffix()})).?);
            const suh = st.slice(u16, shard_file.file.tensors.get(try std.fmt.bufPrint(&key, "language_model.model.layers.0.mlp.switch_mlp.{s}.suh", .{role.suffix()})).?);
            const svh = st.slice(u16, shard_file.file.tensors.get(try std.fmt.bufPrint(&key, "language_model.model.layers.0.mlp.switch_mlp.{s}.svh", .{role.suffix()})).?);
            const tiles = (tiny.hidden / 16) * (tiny.inter / 16) * rate.n;
            for (0..tiny.experts) |e| {
                try src.readExpertPublic(t.allocator, .qwen4_exp, 0, false, role, e, tiny.hidden, tiny.inter, weight);
                try format.reconstructPublic(t.allocator, tr[e * tiles ..][0..tiles], suh[e * 128 ..][0..128], svh[e * 128 ..][0..128], 128, 128, rate, dec, decoded);
                for (weight, decoded, 0..) |w, bits, i| {
                    const d = @as(f64, format.f16BitsToF32(bits)) - w;
                    if (i % 128 < 64) hi += d * d else lo += d * d;
                }
            }
        }
        ratio[arm] = hi / lo;
    }
    try t.expect(ratio[0] > 0.7 and ratio[0] < 1.4);
    try t.expect(ratio[1] < 0.5 * ratio[0]);
}

const TargetFake = struct {
    src: *const source.SafetensorsSource,
    /// X~'X W as (1 + drift) (X~'X~ + ridge) W: zero drift is a BF16 path that agrees with the quantized one.
    drift: f32,
    seen_gate: [tiny.experts][]f32 = @splat(&.{}),

    const scale: f32 = 40;
    const rows: u64 = 20000;

    fn fillH(h: []f32, n: usize) void {
        @memset(h, 0);
        for (0..n) |i| h[i * n + i] = scale;
    }

    fn gateUp(_: *anyopaque, _: usize, h: []f32) anyerror!u64 {
        fillH(h, tiny.hidden);
        return rows;
    }

    fn down(_: *anyopaque, _: usize, h: []f32) anyerror!u64 {
        fillH(h, tiny.inter);
        return rows;
    }

    fn agreeing(self: *TargetFake, w: []const f32, cw: []f32) void {
        const k: f32 = (scale + @as(f32, @floatCast(target_mod.damp)) * scale) * (1 + self.drift);
        for (cw, w) |*o, v| o.* = k * v;
    }

    fn gateUpTarget(ctx: *anyopaque, e: usize, h: []f32, cw: []f32) anyerror!u64 {
        const self: *TargetFake = @ptrCast(@alignCast(ctx));
        fillH(h, tiny.hidden);
        var g: [tiny.hidden * tiny.inter]f32 = undefined;
        var u: [tiny.hidden * tiny.inter]f32 = undefined;
        try self.src.readExpertPublic(std.testing.allocator, .qwen4_exp, 0, false, .gate, e, tiny.hidden, tiny.inter, &g);
        try self.src.readExpertPublic(std.testing.allocator, .qwen4_exp, 0, false, .up, e, tiny.hidden, tiny.inter, &u);
        var both: [tiny.hidden * 2 * tiny.inter]f32 = undefined;
        for (0..tiny.hidden) |r| {
            @memcpy(both[r * 2 * tiny.inter ..][0..tiny.inter], g[r * tiny.inter ..][0..tiny.inter]);
            @memcpy(both[r * 2 * tiny.inter + tiny.inter ..][0..tiny.inter], u[r * tiny.inter ..][0..tiny.inter]);
        }
        self.agreeing(&both, cw);
        return rows;
    }

    fn downTarget(ctx: *anyopaque, e: usize, gate_q: []const f32, _: []const f32, h: []f32, cw: []f32) anyerror!u64 {
        const self: *TargetFake = @ptrCast(@alignCast(ctx));
        self.seen_gate[e] = try std.testing.allocator.dupe(f32, gate_q);
        fillH(h, tiny.inter);
        var d: [tiny.inter * tiny.hidden]f32 = undefined;
        try self.src.readExpertPublic(std.testing.allocator, .qwen4_exp, 0, false, .down, e, tiny.inter, tiny.hidden, &d);
        self.agreeing(&d, cw);
        return rows;
    }

    fn hessianSource(self: *TargetFake) HessianSource {
        return .{ .ctx = self, .gate_up = gateUp, .down = down, .gate_up_target = gateUpTarget, .down_target = downTarget };
    }

    fn deinit(self: *TargetFake) void {
        for (self.seen_gate) |g| std.testing.allocator.free(g);
    }
};

test "convert: the compensated target is W when the BF16 path agrees, moves the bytes when it drifts, and down sees the rounded gate" {
    const t = std.testing;
    const io = t.io;
    var prng = std.Random.DefaultPrng.init(82);
    var src_dir = t.tmpDir(.{});
    defer src_dir.cleanup();
    try tiny.writeSource(t.allocator, io, src_dir.dir, &prng, "model.language_model.layers");
    var src = try source.SafetensorsSource.open(t.allocator, io, src_dir.dir);
    defer src.deinit();
    const geom = Geometry{ .experts = tiny.experts, .hidden = tiny.hidden, .inter = tiny.inter };
    const policy = Policy{ .rate = .{ .n = 32 }, .dec = .{ .codebook = .mcg, .window = .w12 }, .batch = 1 };

    var shards: [3]ReadShard = undefined;
    var made: usize = 0;
    defer for (shards[0..made]) |*sh| {
        sh.file.deinit(t.allocator);
        t.allocator.free(sh.bytes);
    };
    var fakes = [2]TargetFake{ .{ .src = &src, .drift = 0 }, .{ .src = &src, .drift = 0.05 } };
    defer for (&fakes) |*f| f.deinit();
    for (0..3) |arm| {
        var out_dir = t.tmpDir(.{});
        defer out_dir.cleanup();
        var p = policy;
        p.target = arm > 0;
        var plain = TargetFake{ .src = &src, .drift = 0 };
        const hs = if (arm == 0) plain.hessianSource() else fakes[arm - 1].hessianSource();
        _ = try convertQwenLayer(t.allocator, mlx.gpuStream(), io, out_dir.dir, &src, 0, false, geom, p, hs);
        shards[made] = try readShard(t.allocator, io, out_dir.dir, "layer-0.safetensors");
        made += 1;
    }
    try t.expectEqualSlices(u8, shards[0].bytes, shards[1].bytes);
    try t.expect(!std.mem.eql(u8, shards[0].bytes, shards[2].bytes));

    const sh = &shards[2].file;
    const tr = st.slice(u16, sh.tensors.get("language_model.model.layers.0.mlp.switch_mlp.gate_proj.trellis").?);
    const suh = st.slice(u16, sh.tensors.get("language_model.model.layers.0.mlp.switch_mlp.gate_proj.suh").?);
    const svh = st.slice(u16, sh.tensors.get("language_model.model.layers.0.mlp.switch_mlp.gate_proj.svh").?);
    const tiles = (tiny.hidden / 16) * (tiny.inter / 16) * 32;
    var bits: [tiny.hidden * tiny.inter]u16 = undefined;
    for (0..tiny.experts) |e| {
        try format.reconstructPublic(t.allocator, tr[e * tiles ..][0..tiles], suh[e * 128 ..][0..128], svh[e * 128 ..][0..128], 128, 128, policy.rate, policy.dec, &bits);
        for (bits, fakes[1].seen_gate[e]) |b, v| try t.expectEqual(format.f16BitsToF32(b), v);
    }
}
