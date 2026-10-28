//! deepseek_v41 routed experts over the shared expert cache: the checkpoint's own e2m1 experts are read
//! from their shards into slabs and gathered as MLX `mxfp4`, one layer at a time.
const std = @import("std");
const mlx = @import("mlx.zig");
const stream = @import("expert_stream.zig");
const cfg_mod = @import("dsv41_config.zig");
const model = @import("dsv41_model.zig");
const sc_mod = @import("dsv41_scope.zig");
const io_util = @import("io_util.zig");
const log = @import("log.zig");

const A = sc_mod.A;
const Scope = sc_mod.Scope;
const Config = cfg_mod.Config;
const exl3 = @import("sushi_exl3");
const expert_usage = @import("expert_usage.zig");

pub fn geometry(cfg: *const Config) stream.Geometry {
    return .{
        .layers = @intCast(cfg.layers),
        .experts = @intCast(cfg.experts),
        .hidden = @intCast(cfg.dim),
        .intermediate = @intCast(cfg.moe_inter),
        .first_moe_layer = 0,
    };
}

/// Bytes of one routed expert as the checkpoint stores it.
pub fn expertBytes(cfg: *const Config) !u64 {
    return stream.mxfp4ExpertBytes(@intCast(cfg.dim), @intCast(cfg.moe_inter));
}

/// The caller owns the engine, which must be built with `layout = .mxfp4_w123` on the model's GPU stream.
/// One inference thread uses a stream at a time.
pub const Stream = struct {
    engine: *stream.Engine,
    cfg: *const Config,
    act: mlx.mlx_dtype,
    /// The slabs of the layer in flight, between `lease` and `release`.
    held: ?stream.Prepared = null,

    pub fn hook(self: *Stream) model.Experts.Hook {
        return .{ .ctx = self, .apply = applyOpaque, .lease = leaseOpaque, .release = releaseOpaque };
    }

    fn applyOpaque(ctx: *anyopaque, sc: *Scope, layer: usize, x: A, idx: A, wts: A, _: A, shared: ?A) anyerror!A {
        const self: *Stream = @ptrCast(@alignCast(ctx));
        const y = try self.apply(sc, layer, x, idx, wts);
        return if (shared) |sy| sc.add(y, try sc.astype(sy, .float32)) else y;
    }

    fn leaseOpaque(ctx: *anyopaque, sc: *Scope, layer: usize, idx: A) anyerror!model.Experts.Lease {
        const self: *Stream = @ptrCast(@alignCast(ctx));
        return self.lease(sc, layer, idx);
    }

    fn releaseOpaque(ctx: *anyopaque) void {
        const self: *Stream = @ptrCast(@alignCast(ctx));
        self.release();
    }

    /// Router ids reach the slabs as slot ids; the banks are views of the slabs until `release`.
    pub fn lease(self: *Stream, sc: *Scope, layer: usize, idx: A) !model.Experts.Lease {
        if (!mlx.streamIsGpu(sc.s)) return error.Dsv41StreamRequiresGpu;
        std.debug.assert(self.held == null);
        const gpa = self.engine.allocator;
        const ids = try sc_mod.hostI32(try sc.astype(idx, .int32));
        const host = try gpa.alloc(u16, ids.len);
        defer gpa.free(host);
        for (host, ids) |*h, id| {
            if (id < 0 or id >= self.cfg.experts) return error.ExpertOutOfRange;
            h.* = @intCast(id);
        }
        var fresh = try self.engine.prepareHost(@intCast(layer), host);
        errdefer {
            _ = mlx.mlx_synchronize(sc.s);
            fresh.deinit();
        }
        const prepared: *const stream.Prepared = &fresh;
        const slots = try gpa.alloc(i32, host.len);
        defer gpa.free(slots);
        for (slots, prepared.remapped) |*s, r| s.* = r;
        const local = try sc.ints(slots, mlx.getShape(idx));
        const C = stream.quant.Component;
        const bank = struct {
            fn of(p: *const stream.Prepared, w: C, s: C) model.Bank {
                return .{ .w = p.quantOperand(w), .s = p.quantOperand(s) };
            }
        }.of;
        const banks = model.BankSet{
            .gate = bank(prepared, .gate_w, .gate_s),
            .up = bank(prepared, .up_w, .up_s),
            .down = bank(prepared, .down_w, .down_s),
        };
        self.held = fresh;
        return .{ .banks = banks, .local = local };
    }

    pub fn release(self: *Stream) void {
        if (self.held) |*p| p.deinit();
        self.held = null;
    }

    /// The routed sum over the leased slabs: scores and the reduction stay the resident path's.
    pub fn apply(self: *Stream, sc: *Scope, layer: usize, x: A, idx: A, wts: A) !A {
        const lease_ = try self.lease(sc, layer, idx);
        defer self.release();
        errdefer _ = mlx.mlx_synchronize(sc.s);
        const out = try model.routedSum(sc, self.cfg, self.act, x, lease_.banks, lease_.local, wts);
        // Every slab reader completes before another layer can refill the union.
        try mlx.check(mlx.mlx_array_eval(out));
        return out;
    }
};

/// The routed experts of an STQ pack (`expert_quant`) over the shared expert cache, built with `layout = .exl3_k4`:
/// each layer's rate groups are slot pools, and the picks run on the grouped trellis chain with the clamped SwiGLU.
pub const StqStream = struct {
    engine: *stream.Engine,
    cfg: *const Config,
    settles: u64 = 0,

    pub fn hook(self: *StqStream) model.Experts.Hook {
        return .{ .ctx = self, .apply = applyOpaque, .lease = leaseOpaque, .release = releaseOpaque, .host_ns = hostNsOpaque, .prefetch = prefetchOpaque, .spec = .{
            .can = canQueue,
            .queue = queueOpaque,
            .settle = settleOpaque,
        } };
    }

    fn canQueue(ctx: *anyopaque, layer: usize) bool {
        const self: *StqStream = @ptrCast(@alignCast(ctx));
        if (stream.pick_tolerance > 0 or layer >= self.engine.layers.len) return false;
        const state = &self.engine.layers[layer];
        return state.grouped != null and state.active and !state.borrowed and state.resident_map.ctx == null;
    }

    /// The chain over the slots the layer's GPU map holds for the picks, which the host checks in `settle`.
    fn queueOpaque(ctx: *anyopaque, sc: *Scope, layer: usize, x: A, idx: A, wts: A, shared: A) anyerror!model.Experts.Queued {
        const self: *StqStream = @ptrCast(@alignCast(ctx));
        const spec = try self.engine.specGrouped(@intCast(layer)) orelse return error.Dsv41StqLayerNotGrouped;
        const idx32 = try sc.astype(idx, .int32);
        // Submitted before the chain, so the host's read of the picks waits for the router only, not for the chain.
        const ids = mlx.mlx_vector_array_new_data(&[_]A{idx32}, 1);
        defer _ = mlx.mlx_vector_array_free(ids);
        try mlx.check(mlx.mlx_async_eval(ids));
        const local = try sc.take(spec.slots, idx32, 0);
        const out = try model.stqRouted(sc, self.cfg.expert_dec.?, @intFromFloat(self.cfg.swiglu_limit), spec.bank, x, local, wts, shared);
        return .{ .out = out, .idx = idx32, .map = spec.host };
    }

    /// Reads the picks and resolves them in the cache: null when each sat in the slot `map` gave the queued chain, else
    /// the chain over the slots they now hold. The cache is touched here only, so a discarded successor never moves it.
    fn settleOpaque(ctx: *anyopaque, sc: *Scope, layer: usize, x: A, idx: A, wts: A, shared: A, map: []const i32) anyerror!?A {
        const self: *StqStream = @ptrCast(@alignCast(ctx));
        const gpa = self.engine.allocator;
        var clock = stream.Clock.init();
        const ids = try sc_mod.hostI32(idx);
        if (self.engine.prefetcher) |pf| {
            self.settles +%= 1;
            if (self.settles % 4096 == 0) log.info("[expert-stream] prefetch: {d} jobs, {d} experts read in {d:.1} ms, the host waited {d:.1} ms for them\n", .{ pf.jobs_run.load(.monotonic), pf.experts_read.load(.monotonic), @as(f64, @floatFromInt(pf.job_ns.load(.monotonic))) / 1e6, @as(f64, @floatFromInt(pf.host_wait_ns.load(.monotonic))) / 1e6 });
        }
        self.engine.noteRoute(@intCast(layer), false, .{ .ids_wait_ns = clock.lap() });
        const host = try gpa.alloc(u16, ids.len);
        defer gpa.free(host);
        for (host, ids) |*h, id| {
            if (id < 0 or id >= self.cfg.experts) return error.ExpertOutOfRange;
            h.* = @intCast(id);
        }
        var prepared = try self.engine.prepareHost(@intCast(layer), host);
        defer prepared.deinit();
        errdefer _ = mlx.mlx_synchronize(sc.s);
        if (stream.specMatches(map, host, prepared.remapped, prepared.workspace)) return null;
        const rb = prepared.ragged orelse return error.Dsv41StqLayerNotGrouped;
        const slots = try gpa.alloc(i32, host.len);
        defer gpa.free(slots);
        for (slots, prepared.remapped) |*o, r| o.* = r;
        const local = try sc.ints(slots, mlx.getShape(idx));
        _ = clock.lap();
        const out = try model.stqRouted(sc, self.cfg.expert_dec.?, @intFromFloat(self.cfg.swiglu_limit), rb, x, local, wts, shared);
        self.engine.noteExpertCompute(@intCast(layer), clock.lap());
        return out;
    }

    fn prefetchOpaque(ctx: *anyopaque, layer: usize, cand: []const u16, rows: usize, n: usize, m: usize) void {
        const self: *StqStream = @ptrCast(@alignCast(ctx));
        self.engine.prefetch(@intCast(layer), cand, rows, n, m);
    }

    fn hostNsOpaque(ctx: *anyopaque) u64 {
        const self: *StqStream = @ptrCast(@alignCast(ctx));
        return self.engine.fill_ns_total + self.engine.copy_ns_total;
    }

    fn applyOpaque(ctx: *anyopaque, sc: *Scope, layer: usize, x: A, idx: A, wts: A, logits: A, shared: ?A) anyerror!A {
        const self: *StqStream = @ptrCast(@alignCast(ctx));
        return self.applyScored(sc, layer, x, idx, wts, logits, shared orelse return error.Dsv41StqNeedsSharedExpert);
    }

    /// The FP4 kernels lease banks; trellis experts never reach them.
    fn leaseOpaque(_: *anyopaque, _: *Scope, _: usize, _: A) anyerror!model.Experts.Lease {
        return error.Dsv41StqExpertsHaveNoFp4Banks;
    }

    fn releaseOpaque(_: *anyopaque) void {}

    /// The routed sum plus the shared expert's `shared`, f32 `[L, dim]`.
    pub fn apply(self: *StqStream, sc: *Scope, layer: usize, x: A, idx: A, wts: A, shared: A) !A {
        return self.applyScored(sc, layer, x, idx, wts, .{ .ctx = null }, shared);
    }

    /// `apply` with the router's `logits`, which a decode-width route's lossy pick (`stream.pick_tolerance`) reads.
    pub fn applyScored(self: *StqStream, sc: *Scope, layer: usize, x: A, idx: A, wts: A, logits: A, shared: A) !A {
        if (!mlx.streamIsGpu(sc.s)) return error.Dsv41StreamRequiresGpu;
        // A resident layer maps the picks to slots on the GPU: no host read, so the forward is never stalled on one.
        if (self.engine.residentRoute(@intCast(layer))) |rr| {
            const local = try sc.take(rr.map, try sc.astype(idx, .int32), 0);
            return model.stqRouted(sc, self.cfg.expert_dec.?, @intFromFloat(self.cfg.swiglu_limit), rr.bank, x, local, wts, shared);
        }
        const gpa = self.engine.allocator;
        var clock = stream.Clock.init();
        const wide = @as(usize, @intCast(mlx.getShape(x)[0])) > exl3.kernels.DECODE_ROWS_MAX;
        const tolerance = stream.pick_tolerance;
        // The pick compares sqrt-softplus scores, the weights' own scale: log of the score is half log softplus.
        const scores: ?A = if (tolerance > 0 and !wide and logits.ctx != null)
            try sc.op1(mlx.mlx_log, try sc.op1(mlx.mlx_sqrt, try sc.op2(mlx.mlx_logaddexp, logits, try sc.f(0))))
        else
            null;
        const idx32 = try sc.astype(idx, .int32);
        // The wait for the picks is the GPU's attention of this layer; the breakdown line reports it as idswait.
        const read_t0 = model.readStart();
        try sc_mod.evalAll(if (scores) |sv| &.{ idx32, sv } else &.{idx32});
        const ids = try sc_mod.hostI32(idx32);
        model.readDone(.picks_sync, read_t0);
        self.engine.noteRoute(@intCast(layer), false, .{ .ids_wait_ns = clock.lap() });
        const host = try gpa.alloc(u16, ids.len);
        defer gpa.free(host);
        for (host, ids) |*h, id| {
            if (id < 0 or id >= self.cfg.experts) return error.ExpertOutOfRange;
            h.* = @intCast(id);
        }
        if (scores) |sv| _ = try self.engine.swapGroupedMisses(@intCast(layer), host, try sc_mod.hostF32(sv), tolerance, @intCast(mlx.getShape(idx)[1]));
        var prepared = try if (wide) self.engine.prepareHostWide(@intCast(layer), host) else self.engine.prepareHost(@intCast(layer), host);
        defer prepared.deinit();
        errdefer _ = mlx.mlx_synchronize(sc.s);
        const rb = prepared.ragged orelse return error.Dsv41StqLayerNotGrouped;
        const slots = try gpa.alloc(i32, host.len);
        defer gpa.free(slots);
        for (slots, prepared.remapped) |*o, r| o.* = r;
        const local = try sc.ints(slots, mlx.getShape(idx));
        _ = clock.lap();
        // The caller evaluates the sum before a later layer can refill the union: the next layer's picks read it.
        const out = try model.stqRouted(sc, self.cfg.expert_dec.?, @intFromFloat(self.cfg.swiglu_limit), rb, x, local, wts, shared);
        self.engine.noteExpertCompute(@intCast(layer), clock.lap());
        return out;
    }
};

const testing = std.testing;

test "dsv41 stream: slab-gathered experts give the resident experts' logits, with a cache that holds every expert and one that does not (DSV41_TINY)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const dir = try model.fixtureDir();
    var fx = try model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    const s = mlx.gpuStream();
    var sc = Scope.init(testing.allocator, s);
    defer sc.deinit();
    const ids = try model.tokens(&sc, &fx);
    defer testing.allocator.free(ids);

    const opts = model.Opts{ .act = .float32, .qat = true };
    const resident = try model.tinyModel(dir, opts, s);
    defer resident.deinit();
    const want_logits = try model.extendSpan(resident, ids, .all, 64);
    defer _ = mlx.mlx_array_free(want_logits);
    const want = try model.hostRows(&sc, want_logits);
    defer testing.allocator.free(want);

    const per_expert = try expertBytes(&resident.cfg);
    for ([_]u64{ 8, 3 }) |slots| {
        var engine = try stream.Engine.initWithOptions(testing.allocator, dir, geometry(&resident.cfg), slots * resident.cfg.layers * per_expert, s, .{
            .layout = .mxfp4_w123,
            .io_workers = 2,
            .bounce_size = 1 << 20,
        });
        defer engine.deinit();
        const streamed = try model.tinyModel(dir, .{ .act = .float32, .qat = true, .experts = false }, s);
        defer streamed.deinit();
        var st = Stream{ .engine = &engine, .cfg = &streamed.cfg, .act = .float32 };
        streamed.experts = .{ .hook = st.hook() };
        const logits = try model.extendSpan(streamed, ids, .all, 64);
        defer _ = mlx.mlx_array_free(logits);
        const got = try model.hostRows(&sc, logits);
        defer testing.allocator.free(got);
        try testing.expectEqual(want.len, got.len);
        for (want, got) |a, b| try testing.expectEqual(a, b);
    }
}

test "dsv41 stream: decode steps through the fused kernels over leased slabs give the resident banks' logits bit for bit (DSV41_TINY)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const dir = try model.fixtureDir();
    var fx = try model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    const s = mlx.gpuStream();
    var sc = Scope.init(testing.allocator, s);
    defer sc.deinit();
    const ids = try model.tokens(&sc, &fx);
    defer testing.allocator.free(ids);
    const resident = try model.tinyModel(dir, .{ .act = .bfloat16 }, s);
    defer resident.deinit();
    const per_expert = try expertBytes(&resident.cfg);
    var engine = try stream.Engine.initWithOptions(testing.allocator, dir, geometry(&resident.cfg), 3 * resident.cfg.layers * per_expert, s, .{
        .layout = .mxfp4_w123,
        .io_workers = 2,
        .bounce_size = 1 << 20,
    });
    defer engine.deinit();
    const streamed = try model.tinyModel(dir, .{ .act = .bfloat16, .experts = false }, s);
    defer streamed.deinit();
    var st = Stream{ .engine = &engine, .cfg = &streamed.cfg, .act = .bfloat16 };
    streamed.experts = .{ .hook = st.hook() };
    var at: usize = 0;
    for ([_]usize{ 30, 1, 1, 1, 1, 3, 1 }) |n| {
        const take = @min(n, ids.len - at);
        const want = try model.extend(resident, ids[at..][0..take], .last);
        defer _ = mlx.mlx_array_free(want);
        const got = try model.extend(streamed, ids[at..][0..take], .last);
        defer _ = mlx.mlx_array_free(got);
        at += take;
        const w = try model.hostRows(&sc, want);
        defer testing.allocator.free(w);
        const g = try model.hostRows(&sc, got);
        defer testing.allocator.free(g);
        for (w, g) |a, b| try testing.expectEqual(a, b);
    }
}

test "dsv41 shell: the transformer serves the streamed model's logits through forward (DSV41_TINY)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const io = std.Io.Threaded.global_single_threaded.io();
    const dir = try model.fixtureDir();
    const model_mod = @import("model.zig");
    const Transformer = @import("transformer.zig").Transformer;
    var fx = try model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    const s = mlx.gpuStream();
    var sc = Scope.init(testing.allocator, s);
    defer sc.deinit();
    const ids = try model.tokens(&sc, &fx);
    defer testing.allocator.free(ids);

    var config = try model_mod.parseConfig(io, testing.allocator, dir);
    defer config.deinit(testing.allocator);
    try testing.expect(config.isDsv41() and config.expertStreamingRequired());
    try testing.expectEqual(@import("expert_quant.zig").Layout.mxfp4_w123, config.expert_layout);
    config.expert_streaming = true;
    config.expert_cache_bytes = 3 * config.num_hidden_layers * try expertBytes(&config.dsv41.?);
    config.expert_source_dir = try testing.allocator.dupe(u8, dir);

    const resident = try model.tinyModel(dir, .{}, s);
    defer resident.deinit();

    var weights = try model_mod.loadWeightsForConfig(io, testing.allocator, dir, &config, false);
    defer weights.deinit();
    var xfm = try Transformer.init(io, testing.allocator, config, &weights);
    defer xfm.deinit();
    try testing.expect(xfm.ownsModuleDecodeState());

    var at: usize = 0;
    for ([_]usize{ 25, 1, 1, 7, 1 }) |n| {
        const take = @min(n, ids.len - at);
        const chunk = try sc.ints(@as([]const i32, @ptrCast(ids[at .. at + take])), &.{ 1, @intCast(take) });
        const logits = try xfm.forward(chunk);
        defer _ = mlx.mlx_array_free(logits);
        const want = try model.extend(resident, ids[at .. at + take], .last);
        defer _ = mlx.mlx_array_free(want);
        at += take;
        const got = try model.hostRows(&sc, logits);
        defer testing.allocator.free(got);
        const exp = try model.hostRows(&sc, want);
        defer testing.allocator.free(exp);
        try testing.expectEqual(exp.len, got.len);
        for (exp, got) |a, b| try testing.expectEqual(a, b);
    }
    try xfm.resetCache();
    try testing.expectEqual(@as(usize, 0), (try model.ensureState(xfm.dsv41.?)).n);
}

/// `DSV41_STQ_TINY`: the tiny STQ pack (PAIR1 experts in rate groups, SPEC1 trunk, affine embedding), with the FP4
/// checkpoint it was made from under `source/`.
fn stqDir() ![]const u8 {
    const raw = std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest;
    return std.mem.span(raw);
}

/// Test seam: the pools of the next `StqServed` land trellis tiles in the pack's order instead of tile-major.
var stq_tile_major: bool = true;

const StqServed = struct {
    engine: stream.Engine,
    hook: StqStream,
    m: *model.Model,

    fn init(self: *StqServed, dir: []const u8, act: mlx.mlx_dtype, qat: bool, dspark: bool, budget_experts: u64, s: mlx.mlx_stream) !void {
        const per_expert: u64 = 256 * 128 * 3 * 64 / 128;
        return self.initBytes(dir, act, qat, dspark, budget_experts * 8 * per_expert, s);
    }

    fn initBytes(self: *StqServed, dir: []const u8, act: mlx.mlx_dtype, qat: bool, dspark: bool, budget_bytes: u64, s: mlx.mlx_stream) !void {
        return self.initBytesDrop(dir, act, qat, dspark, budget_bytes, s, null, false);
    }

    fn initDrop(self: *StqServed, dir: []const u8, act: mlx.mlx_dtype, qat: bool, dspark: bool, budget_experts: u64, s: mlx.mlx_stream, drop: ?*const expert_usage.Drop, bill: bool) !void {
        const per_expert: u64 = 256 * 128 * 3 * 64 / 128;
        return self.initBytesDrop(dir, act, qat, dspark, budget_experts * 8 * per_expert, s, drop, bill);
    }

    /// `drop` masks the routers; with `bill` the engine also counts those experts out of its pools and bill (without
    /// it, the engine is the full-pool reference of the same mask).
    fn initBytesDrop(self: *StqServed, dir: []const u8, act: mlx.mlx_dtype, qat: bool, dspark: bool, budget_bytes: u64, s: mlx.mlx_stream, drop: ?*const expert_usage.Drop, bill: bool) !void {
        const io = std.Io.Threaded.global_single_threaded.io();
        const model_mod = @import("model.zig");
        var config = try model_mod.parseConfig(io, testing.allocator, dir);
        defer config.deinit(testing.allocator);
        try testing.expectEqual(@import("expert_quant.zig").Layout.exl3_k4, config.expert_layout);
        try testing.expectEqual(@as(u32, 256), config.hidden_size);
        try testing.expectEqual(@as(u32, 128), config.moe_intermediate_size);
        try testing.expectEqual(@as(u32, 8), config.num_hidden_layers);
        var geom = config.expertGeometry();
        if (drop) |d| if (bill) {
            geom.drop_n = d.n;
            geom.drop = d;
        };
        self.engine = try stream.Engine.initWithOptions(testing.allocator, dir, geom, budget_bytes, s, .{
            .layout = .exl3_k4,
            .io_workers = 2,
            .bounce_size = 1 << 20,
            .tile_major = stq_tile_major,
        });
        errdefer self.engine.deinit();
        self.m = try model.tinyModel(dir, .{ .act = act, .qat = qat, .experts = false, .dspark = dspark, .drop = drop }, s);
        self.hook = .{ .engine = &self.engine, .cfg = &self.m.cfg };
        self.m.experts = .{ .hook = self.hook.hook() };
    }

    fn deinit(self: *StqServed) void {
        self.m.deinit();
        self.engine.deinit();
    }
};

fn sourceOf(buf: []u8, dir: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s}/source", .{dir});
}

/// The host copy of an evaluated array's bytes as `T`.
fn hostAs(comptime T: type, sc: *Scope, a: A) ![]const T {
    const c = try sc.contiguous(a);
    try mlx.check(mlx.mlx_array_eval(c));
    const p: [*]const T = @ptrCast(@alignCast(mlx.mlx_array_data_uint8(c) orelse return error.Unreadable));
    return p[0..mlx.mlx_array_size(c)];
}

/// Bank `g` of a trellis projection, decoded: f32 `[out, in]`.
fn denseOf(sc: *Scope, p: exl3.Proj, g: usize, dec: exl3.format.Decode) ![]f32 {
    const sh = mlx.getShape(p.trellis);
    const in: usize = @intCast(sh[1] * 16);
    const out: usize = @intCast(sh[2] * 16);
    const n: u32 = @intCast(sh[3]);
    const per = (in / 16) * (out / 16) * n;
    const words = (try hostAs(u16, sc, p.trellis))[g * per ..][0..per];
    const su = (try hostAs(u16, sc, p.suh))[g * in ..][0..in];
    const sv = (try hostAs(u16, sc, p.svh))[g * out ..][0..out];
    const wt = try testing.allocator.alloc(f32, in * out);
    defer testing.allocator.free(wt);
    try exl3.linear.decodeDense(testing.allocator, words, su, sv, in, out, n, dec, wt);
    const w = try testing.allocator.alloc(f32, in * out);
    for (0..in) |r| for (0..out) |c| {
        w[c * in + r] = wt[r * out + c];
    };
    return w;
}

/// `banks` as one f32 array `[G, out, in]` (or `[out, in]` for one bank and `flat`), owned by `m`.
fn ownDense(sc: *Scope, m: *model.Model, p: exl3.Proj, dec: exl3.format.Decode, flat: bool) !A {
    const sh = mlx.getShape(p.trellis);
    const G: usize = @intCast(sh[0]);
    const per: usize = @intCast(sh[1] * 16 * sh[2] * 16);
    const all = try testing.allocator.alloc(f32, G * per);
    defer testing.allocator.free(all);
    for (0..G) |g| {
        const w = try denseOf(sc, p, g, dec);
        defer testing.allocator.free(w);
        @memcpy(all[g * per ..][0..per], w);
    }
    const out: c_int = sh[2] * 16;
    const in: c_int = sh[1] * 16;
    const a = if (flat and G == 1)
        mlx.mlx_array_new_data(all.ptr, &[_]c_int{ out, in }, 2, .float32)
    else
        mlx.mlx_array_new_data(all.ptr, &[_]c_int{ @intCast(G), out, in }, 3, .float32);
    try m.owned.append(m.gpa, a);
    return a;
}

/// Every trellis bank of `m` replaced by its decoded twin through MLX's dense ops, and the routed rate groups of
/// `dir` decoded into resident dense banks in router order.
fn denseTwin(sc: *Scope, m: *model.Model, dir: []const u8) !void {
    const dec = m.cfg.trunk_dec.?;
    const lin = struct {
        fn swap(sc_: *Scope, m_: *model.Model, l: *model.Lin, d: exl3.format.Decode) !void {
            const q = l.stq orelse return;
            l.* = .{ .w = try ownDense(sc_, m_, q.p, d, true) };
        }
    }.swap;
    for (m.layers[0..m.cfg.layers]) |*ly| {
        ly.in_shared = null;
        for ([_]*model.Lin{ &ly.wq_a, &ly.wq_b, &ly.wkv, &ly.wo_a, &ly.wo_b, &ly.shared[0], &ly.shared[1], &ly.shared[2] }) |l| try lin(sc, m, l, dec);
        if (ly.comp) |*cw| {
            cw.shared = null;
            try lin(sc, m, &cw.wkv, dec);
            if (cw.wgate) |*g| try lin(sc, m, g, dec);
        }
        if (ly.idx) |*iw| {
            try lin(sc, m, &iw.wq_b, dec);
            try lin(sc, m, &iw.weights_proj, dec);
            if (iw.wk) |*k| try lin(sc, m, k, dec);
        }
    }
    for (m.engrams) |*e| try lin(sc, m, &e.wkv, dec);
    try lin(sc, m, &m.head, dec);
    const io = std.Io.Threaded.global_single_threaded.io();
    var w = try @import("dsv41_source.zig").loadResident(io, testing.allocator, dir, .{ .experts = true });
    defer w.deinit();
    const banks = try testing.allocator.alloc(model.BankSet, m.cfg.layers);
    m.experts = .{ .resident = banks };
    const xdec = m.cfg.expert_dec.?;
    for (banks, 0..) |*bs, li| {
        inline for (.{ "gate", "up", "down" }) |pn| {
            var parts: [32]A = undefined;
            var count: usize = 0;
            while (true) : (count += 1) {
                var nb: [3][160]u8 = undefined;
                const t = w.get(try std.fmt.bufPrint(&nb[0], "model.layers.{d}.mlp.switch_mlp." ++ pn ++ "_proj.g{d}.trellis", .{ li, count })) orelse break;
                const p: exl3.Proj = .{
                    .trellis = t,
                    .suh = w.get(try std.fmt.bufPrint(&nb[1], "model.layers.{d}.mlp.switch_mlp." ++ pn ++ "_proj.g{d}.suh", .{ li, count })).?,
                    .svh = w.get(try std.fmt.bufPrint(&nb[2], "model.layers.{d}.mlp.switch_mlp." ++ pn ++ "_proj.g{d}.svh", .{ li, count })).?,
                };
                parts[count] = try ownDense(sc, m, p, xdec, false);
            }
            @field(bs, pn) = .{ .w = try sc.concat(parts[0..count], 0), .s = sc_mod.none };
            try mlx.check(mlx.mlx_array_eval(@field(bs, pn).w));
        }
    }
    m.cfg.expert_dec = null;
}

/// Diagnostic: `SUSHI_DSV41_STQ_ROUTER_SCALE` multiplies every backbone router, 1e-3 or so makes the scores near-ties.
fn scaleRouters(sc: *Scope, m: *model.Model) !void {
    const raw = std.c.getenv("SUSHI_DSV41_STQ_ROUTER_SCALE") orelse return;
    const factor = try std.fmt.parseFloat(f32, std.mem.span(raw));
    for (m.layers[0..m.cfg.layers]) |*ly| ly.gate_w = try sc.mul(ly.gate_w, try sc.f(factor));
}

/// A router's pick of one row, with the scores it ranked (`sqrt(softplus(logit)) + bias`).
const RouteRow = struct {
    picks: [16]u32 = undefined,
    k: usize = 0,
    sel: [64]f64 = undefined,
    n: usize = 0,

    fn isPick(self: *const RouteRow, e: usize) bool {
        for (self.picks[0..self.k]) |p| if (p == e) return true;
        return false;
    }

    /// Every pick scores at least every other expert, up to the tie tolerance.
    fn isTopK(self: *const RouteRow) bool {
        var low: f64 = std.math.inf(f64);
        var high: f64 = -std.math.inf(f64);
        for (self.sel[0..self.n], 0..) |v, e| {
            if (self.isPick(e)) low = @min(low, v) else high = @max(high, v);
        }
        return low >= high - topk_tie;
    }

    /// The gap between the k-th and the (k+1)-th best scores: a flip between two forwards needs it within the noise.
    fn margin(self: *const RouteRow) f64 {
        var sorted = self.sel;
        std.mem.sort(f64, sorted[0..self.n], {}, std.sort.desc(f64));
        return sorted[self.k - 1] - sorted[self.k];
    }
};

/// Two forwards of the same weights may pick other experts only where their router scores moved by at most this much.
const route_tie: f64 = 1e-2;
/// A router's own top-k may order scores that tie this closely either way.
const topk_tie: f64 = 1e-3;

fn routeRow(sc: *Scope, m: *const model.Model, li: usize, t: *const model.Trace, row: usize) !RouteRow {
    var nb: [48]u8 = undefined;
    const idx_a = t.get(try std.fmt.bufPrint(&nb, "route_idx_{d}", .{li})) orelse return error.TraceMissing;
    const idx = try model.hostRows(sc, idx_a);
    defer testing.allocator.free(idx);
    const logits_a = t.get(try std.fmt.bufPrint(&nb, "route_logits_{d}", .{li})) orelse return error.TraceMissing;
    const logits = try model.hostRows(sc, logits_a);
    defer testing.allocator.free(logits);
    const bias = try model.hostRows(sc, m.layers[li].gate_bias);
    defer testing.allocator.free(bias);
    const rows: usize = @intCast(mlx.getShape(idx_a)[0]);
    var out: RouteRow = .{};
    out.k = idx.len / rows;
    out.n = logits.len / rows;
    for (0..out.k) |j| out.picks[j] = @intFromFloat(idx[row * out.k + j]);
    for (0..out.n) |e| {
        const x: f64 = logits[row * out.n + e];
        out.sel[e] = @sqrt(@max(x, 0) + std.math.log1p(@exp(-@abs(x)))) + bias[e];
    }
    return out;
}

/// Whether the routers of two forwards picked other experts for row `ra` of `a` and row `rb` of `b` at layer `li`. Each
/// side's picks must be its own top-k, and a difference needs a near-tie on one side: that is the invariant, the bytes
/// of a pick are not.
fn routeDiffers(sc: *Scope, m: *const model.Model, li: usize, a: *const model.Trace, ra: usize, b: *const model.Trace, rb: usize) !bool {
    const x = try routeRow(sc, m, li, a, ra);
    const y = try routeRow(sc, m, li, b, rb);
    var same = x.k == y.k;
    for (x.picks[0..x.k]) |p| same = same and y.isPick(p);
    if (std.c.getenv("SUSHI_DSV41_STQ_REPORT") != null and (!same or !x.isTopK() or !y.isTopK())) {
        std.debug.print("route layer {d}: same {} topk {} {} margins {d:.6} {d:.6} picks {any} {any}\n", .{ li, same, x.isTopK(), y.isTopK(), x.margin(), y.margin(), x.picks[0..x.k], y.picks[0..y.k] });
    }
    try testing.expect(x.isTopK() and y.isTopK());
    if (same) return false;
    // Each side ranks its own picks first, so a swap needs the two forwards' scores to move by the gap they swapped
    // across: the bar is on that movement, which rounding alone keeps small.
    var dev: f64 = 0;
    for (x.sel[0..x.n], y.sel[0..y.n]) |u, v| dev = @max(dev, @abs(u - v));
    if (std.c.getenv("SUSHI_DSV41_STQ_REPORT") != null) std.debug.print("route layer {d}: scores moved {d:.6} between the forwards\n", .{ li, dev });
    try testing.expect(dev <= route_tie);
    return true;
}

fn cosine(a: []const f32, b: []const f32) f64 {
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (a, b) |x, y| {
        dot += @as(f64, x) * y;
        na += @as(f64, x) * x;
        nb += @as(f64, y) * y;
    }
    return dot / @sqrt(na * nb);
}

/// The first layer whose stream, in row `k` of the verify trace, is farther from the serial row than the bar.
fn firstDivergence(sc: *Scope, m: *const model.Model, verify: *const model.Trace, k: usize, serial: *const model.Trace) !?usize {
    var nm: [32]u8 = undefined;
    for (0..m.cfg.layers) |li| {
        const key = try std.fmt.bufPrint(&nm, "stream_{d}", .{li});
        const a = try traceOf(sc, verify, key);
        defer testing.allocator.free(a);
        const b = try traceOf(sc, serial, key);
        defer testing.allocator.free(b);
        if (cosine(a[k * b.len ..][0..b.len], b) <= 0.9999) return li;
    }
    return null;
}

/// One forward's selection of compressed rows for one row (the candidate blocks or the index top-k): its picks and the
/// scores it ranked them by (-inf where the row sees nothing).
const SelRow = struct {
    picks: []u32,
    scores: []f64,

    fn deinit(self: *SelRow) void {
        testing.allocator.free(self.picks);
        testing.allocator.free(self.scores);
    }

    fn isPick(self: *const SelRow, e: usize) bool {
        for (self.picks) |p| if (p == e) return true;
        return false;
    }

    fn sameSet(self: *const SelRow, other: *const SelRow) bool {
        if (self.picks.len != other.picks.len) return false;
        for (self.picks) |p| if (!other.isPick(p)) return false;
        return true;
    }

    /// Every pick scores at least every other seen row, up to `tol`.
    fn isTopK(self: *const SelRow, tol: f64) bool {
        var low: f64 = std.math.inf(f64);
        var high: f64 = -std.math.inf(f64);
        for (self.scores, 0..) |v, e| {
            if (self.isPick(e)) low = @min(low, v) else high = @max(high, v);
        }
        return low >= high - tol;
    }

    /// The gap between the last pick and the best row left out.
    fn margin(self: *const SelRow) !f64 {
        const sorted = try testing.allocator.dupe(f64, self.scores);
        defer testing.allocator.free(sorted);
        std.mem.sort(f64, sorted, {}, std.sort.desc(f64));
        const k = self.picks.len;
        if (k == 0 or k >= sorted.len or sorted[k] == -std.math.inf(f64)) return std.math.inf(f64);
        return sorted[k - 1] - sorted[k];
    }
};

/// Row `row` of the selection `what` ("index" or "cand") at layer `li` of a trace; null where the layer selects none.
fn selRow(sc: *Scope, t: *const model.Trace, what: []const u8, li: usize, row: usize) !?SelRow {
    var nb: [48]u8 = undefined;
    const idx_a = t.get(try std.fmt.bufPrint(&nb, "{s}_idx_{d}", .{ what, li })) orelse return null;
    const sc_a = t.get(try std.fmt.bufPrint(&nb, "{s}_scores_{d}", .{ what, li })) orelse return error.TraceMissing;
    const idx = try model.hostRows(sc, idx_a);
    defer testing.allocator.free(idx);
    const scores = try model.hostRows(sc, sc_a);
    defer testing.allocator.free(scores);
    const k: usize = @intCast(mlx.getShape(idx_a)[1]);
    const n: usize = @intCast(mlx.getShape(sc_a)[1]);
    var picks: std.ArrayList(u32) = .empty;
    errdefer picks.deinit(testing.allocator);
    for (idx[row * k ..][0..k]) |v| if (v >= 0 and scores[row * n + @as(usize, @intFromFloat(v))] != -std.math.inf(f32)) try picks.append(testing.allocator, @intFromFloat(v));
    // Scores stay indexed by row id: a pick names its row, and -inf rows rank below every seen one.
    const out = try testing.allocator.alloc(f64, n);
    for (out, scores[row * n ..][0..n]) |*o, v| o.* = v;
    return .{ .picks = try picks.toOwnedSlice(testing.allocator), .scores = out };
}

/// Two forwards may select other compressed rows only where the scores tie to this much of the last pick's score;
/// the GPU twin's own f32 matmul noise is ~1e-2 of a stream by layer 4, and a layer-5 pick pair sat at 1.01e-2.
const select_tie: f64 = 2e-2;

/// Whether layer `li` selected other compressed rows for row `ra` of `a` and row `rb` of `b` (the candidate blocks or
/// the index top-k). A difference must be a near-tie: each side its own top-k, the last pick tied on one side.
fn selectionDiffers(sc: *Scope, li: usize, a: *const model.Trace, ra: usize, b: *const model.Trace, rb: usize) !bool {
    for ([_][]const u8{ "cand", "index" }) |what| {
        var x = (try selRow(sc, a, what, li, ra)) orelse continue;
        defer x.deinit();
        var y = (try selRow(sc, b, what, li, rb)) orelse return error.TraceMissing;
        defer y.deinit();
        if (x.sameSet(&y)) continue;
        try expectTie(what, li, &x, &y);
        return true;
    }
    return false;
}

fn expectTie(what: []const u8, li: usize, a: *const SelRow, b: *const SelRow) !void {
    const ma = try a.margin();
    const mb = try b.margin();
    var scale: f64 = 1;
    for (a.scores) |v| if (std.math.isFinite(v)) {
        scale = @max(scale, @abs(v));
    };
    const tol = select_tie * scale;
    if (std.c.getenv("SUSHI_DSV41_STQ_REPORT") != null) std.debug.print("{s} selection layer {d}: margins {d:.6} {d:.6} scale {d:.3} picks {any} {any}\n", .{ what, li, ma, mb, scale, a.picks, b.picks });
    try testing.expect(a.isTopK(tol) and b.isTopK(tol));
    try testing.expect(ma <= tol or mb <= tol);
}

/// A kept verify row (`k` of `verify`) that left its serial row: the first selection of compressed rows, at or before
/// the first layer whose stream left the bar, whose picks differ between the two must exist, and each side must have
/// picked its own top-k with the last pick tied on one side. A divergence any other way is the arm's fault.
fn expectSelectionTie(sc: *Scope, m: *const model.Model, verify: *const model.Trace, k: usize, serial: *const model.Trace) !void {
    const div = (try firstDivergence(sc, m, verify, k, serial)) orelse return error.TestUnexpectedResult;
    for (0..div + 1) |li| for ([_][]const u8{ "cand", "index" }) |what| {
        var a = (try selRow(sc, verify, what, li, k)) orelse continue;
        defer a.deinit();
        var b = (try selRow(sc, serial, what, li, 0)) orelse return error.TraceMissing;
        defer b.deinit();
        if (a.sameSet(&b)) continue;
        try expectTie(what, li, &a, &b);
        return;
    };
    return error.TestUnexpectedResult;
}

/// CPU time of the calling thread: the host's share of a step, whatever the GPU is doing.
fn threadCpuNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.THREAD_CPUTIME_ID, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

fn traceOf(sc: *Scope, tr: *const model.Trace, name: []const u8) ![]f32 {
    return model.hostRows(sc, tr.get(name) orelse return error.TraceMissing);
}

test "dsv41 stq: the served STQ pack (trunk banks, affine embedding, streamed rate-grouped experts) follows its dense twin (DSV41_STQ_TINY)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const dir = try stqDir();
    var fx = try model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    const s = mlx.gpuStream();
    var sc = Scope.init(testing.allocator, s);
    defer sc.deinit();
    const ids = try model.tokens(&sc, &fx);
    defer testing.allocator.free(ids);
    var twin_src: StqServed = undefined;
    try twin_src.init(dir, .float32, false, false, 8, s);
    defer twin_src.deinit();
    try scaleRouters(&sc, twin_src.m);
    const twin = twin_src.m;
    try denseTwin(&sc, twin, dir);
    twin.fused = false;
    var tw_tr: model.Trace = .{ .gpa = testing.allocator };
    defer tw_tr.deinit();
    twin.trace = &tw_tr;
    const want_l = try model.extendSpan(twin, ids, .all, 64);
    twin.trace = null;
    defer _ = mlx.mlx_array_free(want_l);
    const want = try model.hostRows(&sc, want_l);
    defer testing.allocator.free(want);
    // A cache that holds every expert, and one that streams.
    for ([_]u64{ 8, 3 }) |slots| {
        var served: StqServed = undefined;
        try served.init(dir, .float32, false, false, slots, s);
        defer served.deinit();
        try scaleRouters(&sc, served.m);
        try testing.expect(served.m.layers[0].wq_b.stq != null and served.m.layers[0].wo_a.stq != null and served.m.head.stq != null);
        try testing.expect(served.m.embed.scales.ctx != null);
        served.m.fused = false;
        var tr: model.Trace = .{ .gpa = testing.allocator };
        defer tr.deinit();
        served.m.trace = &tr;
        const got_l = try model.extendSpan(served.m, ids, .all, 64);
        served.m.trace = null;
        defer _ = mlx.mlx_array_free(got_l);
        const got = try model.hostRows(&sc, got_l);
        defer testing.allocator.free(got);
        // QAT off: a round trip turns a store's rounding into a jump. The first layers agree to the kernels' f16
        // stores; past them a routing near-tie may flip an expert, so the logits hold their direction.
        // Past the first flipped layer the two forwards hold other activations, so only the layers up to it compare.
        var first_flip: usize = served.m.cfg.layers;
        for (0..served.m.cfg.layers) |li| {
            var flipped = false;
            for (0..ids.len) |row| flipped = (try selectionDiffers(&sc, li, &tr, row, &tw_tr, row)) or (try routeDiffers(&sc, served.m, li, &tr, row, &tw_tr, row)) or flipped;
            if (flipped) {
                first_flip = li;
                break;
            }
        }
        for ([_][]const u8{ "attn_in_0", "sa_q_0", "sa_kv_0", "attn_0", "ffn_0", "stream_0", "attn_1", "ffn_1", "stream_1" }) |name| {
            if (name[name.len - 1] - '0' >= first_flip) continue;
            const a = try traceOf(&sc, &tr, name);
            defer testing.allocator.free(a);
            const b = try traceOf(&sc, &tw_tr, name);
            defer testing.allocator.free(b);
            const c = cosine(a, b);
            if (std.c.getenv("SUSHI_DSV41_STQ_REPORT") != null) std.debug.print("dsv41 stq twin slots {d} {s}: cosine {d:.6}\n", .{ slots, name, c });
            try testing.expect(c > 0.9999);
        }
        for (got) |v| try testing.expect(std.math.isFinite(v));
        const c = cosine(got, want);
        if (std.c.getenv("SUSHI_DSV41_STQ_REPORT") != null) std.debug.print("dsv41 stq twin slots {d}: logits cosine {d:.6}\n", .{ slots, c });
        // A router tie broken the other way moves the logits; the routes above then carry the check.
        if (first_flip == served.m.cfg.layers) try testing.expect(c > 0.99);
    }
    // Decode chunks of one row and a few after a prefill, the decode chain's widths.
    var served: StqServed = undefined;
    try served.init(dir, .float32, false, false, 3, s);
    defer served.deinit();
    try scaleRouters(&sc, served.m);
    served.m.fused = false;
    model.resetState(twin);
    var at: usize = 0;
    for ([_]usize{ 20, 1, 1, 5, 1, 3, 1 }) |n| {
        var tra: model.Trace = .{ .gpa = testing.allocator };
        defer tra.deinit();
        var trb: model.Trace = .{ .gpa = testing.allocator };
        defer trb.deinit();
        served.m.trace = &tra;
        twin.trace = &trb;
        const a = try model.extend(served.m, ids[at..][0..n], .last);
        defer _ = mlx.mlx_array_free(a);
        const b = try model.extend(twin, ids[at..][0..n], .last);
        defer _ = mlx.mlx_array_free(b);
        served.m.trace = null;
        twin.trace = null;
        at += n;
        var flipped = false;
        const w: usize = served.m.cfg.window;
        for (0..served.m.cfg.layers) |li| {
            // A CED chunk's decoder layers hold its last window of rows.
            const rows = if (served.m.ced_split != null and li >= served.m.ced_split.? and n > w) w else n;
            for (0..rows) |row| flipped = (try selectionDiffers(&sc, li, &tra, row, &trb, row)) or (try routeDiffers(&sc, served.m, li, &tra, row, &trb, row)) or flipped;
            if (flipped) break;
        }
        // From a flip on the two caches hold other rows.
        if (flipped) return;
        const ga = try model.hostRows(&sc, a);
        defer testing.allocator.free(ga);
        const gb = try model.hostRows(&sc, b);
        defer testing.allocator.free(gb);
        if (std.c.getenv("SUSHI_DSV41_STQ_REPORT") != null) std.debug.print("dsv41 stq twin decode chunk {d}: logits cosine {d:.6}\n", .{ n, cosine(ga, gb) });
        try testing.expect(cosine(ga, gb) > 0.98);
    }
}

test "dsv41 stq: the STQ pack's K4 trunk and affine embedding keep the FP4 model's first attention (DSV41_STQ_TINY)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const dir = try stqDir();
    var pbuf: [1024]u8 = undefined;
    const src = try sourceOf(&pbuf, dir);
    var fx = try model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    const s = mlx.gpuStream();
    var sc = Scope.init(testing.allocator, s);
    defer sc.deinit();
    const ids = try model.tokens(&sc, &fx);
    defer testing.allocator.free(ids);
    const resident = try model.tinyModel(src, .{ .act = .float32 }, s);
    defer resident.deinit();
    var served: StqServed = undefined;
    try served.init(dir, .float32, true, false, 8, s);
    defer served.deinit();
    var tr: [2]model.Trace = .{ .{ .gpa = testing.allocator }, .{ .gpa = testing.allocator } };
    defer for (&tr) |*t| t.deinit();
    var logits: [2][]f32 = undefined;
    for ([_]*model.Model{ resident, served.m }, 0..) |m, k| {
        m.trace = &tr[k];
        const lg = try model.extendSpan(m, ids, .all, 64);
        defer _ = mlx.mlx_array_free(lg);
        m.trace = null;
        logits[k] = try model.hostRows(&sc, lg);
    }
    defer for (logits) |l| testing.allocator.free(l);
    const a = try traceOf(&sc, &tr[1], "attn_0");
    defer testing.allocator.free(a);
    const b = try traceOf(&sc, &tr[0], "attn_0");
    defer testing.allocator.free(b);
    // Random tiny weights at K1.5 to K4 move the routed experts by tens of percent, so past the first attention
    // the logits follow only loosely; the dense twin above is the exact bar.
    if (std.c.getenv("SUSHI_DSV41_STQ_REPORT") != null) std.debug.print("dsv41 stq vs fp4: attn_0 cosine {d:.5}, logits cosine {d:.5}\n", .{ cosine(a, b), cosine(logits[1], logits[0]) });
    try testing.expect(cosine(a, b) > 0.95);
}

fn argmax(row: []const f32) usize {
    var best: usize = 0;
    for (row, 0..) |v, i| if (v > row[best]) {
        best = i;
    };
    return best;
}

test "dsv41 stq: greedy DSpark rounds over the STQ pack emit the serial greedy tokens (DSV41_STQ_TINY)" {
    try dsparkMatchesSerial(null);
}

test "dsv41 drop: greedy DSpark rounds over the STQ pack with experts dropped emit the serial greedy tokens of the same mask (DSV41_STQ_TINY)" {
    var d = try model.testDrop(8);
    defer d.deinit();
    try dsparkMatchesSerial(&d);
}

fn dsparkMatchesSerial(drop: ?*const expert_usage.Drop) !void {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const dir = try stqDir();
    var fx = try model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    const s = mlx.gpuStream();
    var sc = Scope.init(testing.allocator, s);
    defer sc.deinit();
    const ids = try model.tokens(&sc, &fx);
    defer testing.allocator.free(ids);
    const P = 35;
    const G = 40;
    var serial: StqServed = undefined;
    // f32 with the QAT round trips off, the exact setting: in bf16 a round trip turns a row's last-ulp difference
    // between widths into a code flip, on the FP4 path as on this one.
    try serial.initDrop(dir, .float32, false, true, 8, s, drop, true);
    defer serial.deinit();
    try scaleRouters(&sc, serial.m);
    const V = serial.m.cfg.vocab;
    var chain: [G + 1]u32 = undefined;
    var rows: [G][]f32 = undefined;
    var made: usize = 0;
    defer for (rows[0..made]) |r| testing.allocator.free(r);
    {
        const lg = try model.extend(serial.m, ids[0..P], .last);
        defer _ = mlx.mlx_array_free(lg);
        chain[0] = @intCast(argmax(try sc_mod.hostF32(lg)));
    }
    var serial_tr: [G]model.Trace = undefined;
    for (&serial_tr) |*t| t.* = .{ .gpa = testing.allocator };
    defer for (&serial_tr) |*t| t.deinit();
    for (0..G) |i| {
        serial.m.trace = &serial_tr[i];
        const lg = try model.extend(serial.m, chain[i .. i + 1], .last);
        serial.m.trace = null;
        defer _ = mlx.mlx_array_free(lg);
        rows[i] = try testing.allocator.dupe(f32, try sc_mod.hostF32(lg));
        made = i + 1;
        chain[i + 1] = @intCast(argmax(rows[i]));
    }

    var spec: StqServed = undefined;
    try spec.initDrop(dir, .float32, false, true, 8, s, drop, true);
    defer spec.deinit();
    try scaleRouters(&sc, spec.m);
    {
        const lg = try model.extend(spec.m, ids[0..P], .last);
        defer _ = mlx.mlx_array_free(lg);
        try testing.expectEqual(chain[0], @as(u32, @intCast(argmax(try sc_mod.hostF32(lg)))));
    }
    var at: usize = 0;
    var accepted_total: usize = 0;
    // Twelve rounds: further on, the GPU forward's index selection (FP4 and STQ alike) can flip on a near-tie between
    // a verify chunk and a serial row, which is not this arm's to fix.
    while (at < 12) {
        var d = try model.draft(spec.m, chain[at], false);
        defer d.deinit();
        var round: [16]u32 = undefined;
        round[0] = chain[at];
        @memcpy(round[1 .. 1 + d.ids.len], d.ids);
        var verify_tr: model.Trace = .{ .gpa = testing.allocator };
        defer verify_tr.deinit();
        spec.m.trace = &verify_tr;
        const vl = try model.verify(spec.m, round[0 .. 1 + d.ids.len]);
        spec.m.trace = null;
        defer _ = mlx.mlx_array_free(vl);
        const vr = try sc_mod.hostF32(vl);
        var accepted: usize = 0;
        while (accepted < d.ids.len and argmax(vr[accepted * V ..][0..V]) == round[1 + accepted]) accepted += 1;
        // A tie broken the other way moves a row off its serial row, and the two sequences with it.
        for (0..spec.m.cfg.layers) |li| {
            var flipped = false;
            for (0..accepted + 1) |k| flipped = (try selectionDiffers(&sc, li, &verify_tr, k, &serial_tr[at + k], 0)) or (try routeDiffers(&sc, spec.m, li, &verify_tr, k, &serial_tr[at + k], 0)) or flipped;
            if (flipped) return;
        }
        // The round emits the serial tokens, each kept row near its serial row.
        for (0..accepted + 1) |k| {
            if (std.c.getenv("SUSHI_DSV41_STQ_REPORT") != null) std.debug.print("dspark at {d} k {d} serial {d} verify {d} cosine {d:.6}\n", .{ at, k, chain[at + 1 + k], argmax(vr[k * V ..][0..V]), cosine(rows[at + k], vr[k * V ..][0..V]) });
            try testing.expectEqual(chain[at + 1 + k], @as(u32, @intCast(argmax(vr[k * V ..][0..V]))));
            if (cosine(rows[at + k], vr[k * V ..][0..V]) > 0.9999) continue;
            // The two widths round differently, so a selection of compressed rows may tie the other way; from there the
            // two sequences differ.
            try expectSelectionTie(&sc, spec.m, &verify_tr, k, &serial_tr[at + k]);
            return;
        }
        try model.finishVerify(spec.m, accepted);
        accepted_total += accepted;
        at += accepted + 1;
    }
    try testing.expect(at >= 12);
    if (std.c.getenv("SUSHI_DSV41_STQ_REPORT") != null) std.debug.print("dsv41 stq dspark: {d} tokens over rounds, {d} drafts accepted\n", .{ at, accepted_total });
}

test "dsv41 stq: verify rows that decode each distinct expert once give the per-row chain's bytes, through hits, misses and the union (DSV41_STQ_TINY)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const dir = try stqDir();
    var fx = try model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    const s = mlx.gpuStream();
    var sc = Scope.init(testing.allocator, s);
    defer sc.deinit();
    const ids = try model.tokens(&sc, &fx);
    defer testing.allocator.free(ids);
    const groups = exl3.rate_groups;
    // The tiny pack's width is not the prepared chain's geometry, whose rows the grouping serves.
    groups.prepared_force = true;
    defer groups.prepared_force = null;
    defer groups.group_force = null;
    for ([_]u64{ 8, 3 }) |slots| {
        var per_row: StqServed = undefined;
        try per_row.init(dir, .bfloat16, true, true, slots, s);
        defer per_row.deinit();
        var grouped: StqServed = undefined;
        try grouped.init(dir, .bfloat16, true, true, slots, s);
        defer grouped.deinit();
        var at: usize = 0;
        for ([_]usize{ 20, 2, 5, 3, 4, 6, 1, 9 }) |n| {
            const take = @min(n, ids.len - at);
            if (take == 0) break;
            groups.group_force = 0;
            const a = try model.extend(per_row.m, ids[at..][0..take], .all);
            defer _ = mlx.mlx_array_free(a);
            groups.group_force = null;
            const b = try model.extend(grouped.m, ids[at..][0..take], .all);
            defer _ = mlx.mlx_array_free(b);
            at += take;
            const ga = try model.hostRows(&sc, a);
            defer testing.allocator.free(ga);
            const gb = try model.hostRows(&sc, b);
            defer testing.allocator.free(gb);
            try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(ga), std.mem.sliceAsBytes(gb));
        }
    }
}

test "dsv41 stq: tile-major pools give the pack-order pools' bytes, through hits, misses, the union and prefill rows (DSV41_STQ_TINY)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const dir = try stqDir();
    var fx = try model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    const s = mlx.gpuStream();
    var sc = Scope.init(testing.allocator, s);
    defer sc.deinit();
    const ids = try model.tokens(&sc, &fx);
    defer testing.allocator.free(ids);
    defer stq_tile_major = true;
    for ([_]u64{ 8, 3 }) |slots| {
        stq_tile_major = false;
        var pack: StqServed = undefined;
        try pack.init(dir, .bfloat16, true, true, slots, s);
        defer pack.deinit();
        stq_tile_major = true;
        var major: StqServed = undefined;
        try major.init(dir, .bfloat16, true, true, slots, s);
        defer major.deinit();
        var at: usize = 0;
        for ([_]usize{ 20, 2, 5, 3, 4, 6, 1, 9 }) |n| {
            const take = @min(n, ids.len - at);
            if (take == 0) break;
            const a = try model.extend(pack.m, ids[at..][0..take], .all);
            defer _ = mlx.mlx_array_free(a);
            const b = try model.extend(major.m, ids[at..][0..take], .all);
            defer _ = mlx.mlx_array_free(b);
            at += take;
            const ga = try model.hostRows(&sc, a);
            defer testing.allocator.free(ga);
            const gb = try model.hostRows(&sc, b);
            defer testing.allocator.free(gb);
            try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(ga), std.mem.sliceAsBytes(gb));
        }
    }
}

test "dsv41 stq: the submitted decode schedule gives the drained schedule's bytes, through hits, misses and the union (DSV41_STQ_TINY)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const dir = try stqDir();
    var fx = try model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    const s = mlx.gpuStream();
    var sc = Scope.init(testing.allocator, s);
    defer sc.deinit();
    const ids = try model.tokens(&sc, &fx);
    defer testing.allocator.free(ids);
    for ([_]u64{ 8, 3 }) |slots| {
        var drained: StqServed = undefined;
        try drained.init(dir, .bfloat16, true, false, slots, s);
        defer drained.deinit();
        drained.m.sync_layers = true;
        var submitted: StqServed = undefined;
        try submitted.init(dir, .bfloat16, true, false, slots, s);
        defer submitted.deinit();
        var at: usize = 0;
        var ns: [2]u64 = .{ 0, 0 };
        var cpu: [2]u64 = .{ 0, 0 };
        var ops: u64 = 0;
        for ([_]usize{ 20, 1, 1, 5, 1, 3, 1, 16, 1, 1, 1, 1 }) |n| {
            const take = @min(n, ids.len - at);
            if (take == 0) break;
            var sw = io_util.Stopwatch.init(std.Io.Threaded.global_single_threaded.io());
            var c0 = threadCpuNs();
            const a = try model.extend(drained.m, ids[at..][0..take], .all);
            defer _ = mlx.mlx_array_free(a);
            if (take == 1) ns[0] += sw.read();
            if (take == 1) cpu[0] += threadCpuNs() - c0;
            sw.reset();
            c0 = threadCpuNs();
            const o0 = mlx.op_count.load(.monotonic);
            const b = try model.extend(submitted.m, ids[at..][0..take], .all);
            if (take == 1) ops += mlx.op_count.load(.monotonic) - o0;
            defer _ = mlx.mlx_array_free(b);
            if (take == 1) ns[1] += sw.read();
            if (take == 1) cpu[1] += threadCpuNs() - c0;
            at += take;
            const ga = try model.hostRows(&sc, a);
            defer testing.allocator.free(ga);
            const gb = try model.hostRows(&sc, b);
            defer testing.allocator.free(gb);
            try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(ga), std.mem.sliceAsBytes(gb));
        }
        if (std.c.getenv("SUSHI_DSV41_STQ_REPORT") != null) std.debug.print("dsv41 stq schedule slots {d}: one-row steps drained {d:.2} ms (host cpu {d:.2}), submitted {d:.2} ms (host cpu {d:.2}), {d} mlx calls\n", .{ slots, @as(f64, @floatFromInt(ns[0])) / 1e6, @as(f64, @floatFromInt(cpu[0])) / 1e6, @as(f64, @floatFromInt(ns[1])) / 1e6, @as(f64, @floatFromInt(cpu[1])) / 1e6, ops });
    }
}

/// Rows of `a` and `b` (f32 logits) equal to the bit.
fn expectSameBits(sc: *Scope, a: A, b: A) !void {
    const ga = try model.hostRows(sc, a);
    defer testing.allocator.free(ga);
    const gb = try model.hostRows(sc, b);
    defer testing.allocator.free(gb);
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(ga), std.mem.sliceAsBytes(gb));
}

test "dsv41 stq: queueing the routed chains from the cache map gives the drained and the waiting schedules' bytes, through hits, misses, the union and rebuilt layers (DSV41_STQ_TINY)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const dir = try stqDir();
    var fx = try model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    const s = mlx.gpuStream();
    var sc = Scope.init(testing.allocator, s);
    defer sc.deinit();
    const ids = try model.tokens(&sc, &fx);
    defer testing.allocator.free(ids);
    // 8 holds every expert (resident: nothing to queue), the others stream through hits and misses.
    for ([_]u64{ 8, 4, 3 }) |slots| {
        var drained: StqServed = undefined;
        try drained.init(dir, .bfloat16, true, false, slots, s);
        defer drained.deinit();
        drained.m.sync_layers = true;
        var waiting: StqServed = undefined;
        try waiting.init(dir, .bfloat16, true, false, slots, s);
        defer waiting.deinit();
        waiting.m.route_spec = false;
        var queued: StqServed = undefined;
        try queued.init(dir, .bfloat16, true, false, slots, s);
        defer queued.deinit();
        // Every layer of a pass may be rebuilt.
        var eager: StqServed = undefined;
        try eager.init(dir, .bfloat16, true, false, slots, s);
        defer eager.deinit();
        eager.m.route_spec_rebuilds = std.math.maxInt(u32);
        const kept0 = model.spec_kept;
        const rebuilt0 = model.spec_rebuilt;
        var at: usize = 0;
        for ([_]usize{ 20, 1, 1, 5, 1, 3, 1, 16, 1, 1, 1, 1, 2, 1, 1 }) |n| {
            const take = @min(n, ids.len - at);
            if (take == 0) break;
            const a = try model.extend(drained.m, ids[at..][0..take], .all);
            defer _ = mlx.mlx_array_free(a);
            const b = try model.extend(waiting.m, ids[at..][0..take], .all);
            defer _ = mlx.mlx_array_free(b);
            const c = try model.extend(queued.m, ids[at..][0..take], .all);
            defer _ = mlx.mlx_array_free(c);
            const d = try model.extend(eager.m, ids[at..][0..take], .all);
            defer _ = mlx.mlx_array_free(d);
            at += take;
            try expectSameBits(&sc, a, b);
            try expectSameBits(&sc, a, c);
            try expectSameBits(&sc, a, d);
        }
        const kept = model.spec_kept - kept0;
        const rebuilt = model.spec_rebuilt - rebuilt0;
        if (std.c.getenv("SUSHI_DSV41_STQ_REPORT") != null) std.debug.print("dsv41 stq route-spec slots {d}: {d} layers kept, {d} rebuilt\n", .{ slots, kept, rebuilt });
        if (slots == 8) {
            try testing.expectEqual(@as(u64, 0), kept + rebuilt);
        } else if (slots <= 3) {
            try testing.expect(kept > 0);
            try testing.expect(rebuilt > 0);
        }
    }
}

test "dsv41 stq: queued routed chains leave a verify round's kept rows, the DSpark windows and the open groups as the drained schedule does (DSV41_STQ_TINY)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const dir = try stqDir();
    var fx = try model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    const s = mlx.gpuStream();
    var sc = Scope.init(testing.allocator, s);
    defer sc.deinit();
    const ids = try model.tokens(&sc, &fx);
    defer testing.allocator.free(ids);
    for ([_]u64{ 4, 3 }) |slots| {
        var drained: StqServed = undefined;
        try drained.init(dir, .bfloat16, true, true, slots, s);
        defer drained.deinit();
        drained.m.sync_layers = true;
        var queued: StqServed = undefined;
        try queued.init(dir, .bfloat16, true, true, slots, s);
        defer queued.deinit();
        const rebuilt0 = model.spec_rebuilt;
        var at: usize = 0;
        const prefill = try model.extend(drained.m, ids[0..20], .last);
        _ = mlx.mlx_array_free(prefill);
        const prefill_q = try model.extend(queued.m, ids[0..20], .last);
        _ = mlx.mlx_array_free(prefill_q);
        at = 20;
        // (rows, accepted) of a verify round; 0 rows is a one-row decode step.
        for ([_][2]usize{ .{ 5, 2 }, .{ 0, 0 }, .{ 6, 0 }, .{ 4, 3 }, .{ 0, 0 }, .{ 3, 1 }, .{ 0, 0 } }) |round| {
            const ref_m = drained.m;
            const cut_m = queued.m;
            if (round[0] == 0) {
                const a = try model.extend(ref_m, ids[at..][0..1], .all);
                defer _ = mlx.mlx_array_free(a);
                const c = try model.extend(cut_m, ids[at..][0..1], .all);
                defer _ = mlx.mlx_array_free(c);
                try expectSameBits(&sc, a, c);
                at += 1;
            } else {
                const a = try model.verify(ref_m, ids[at..][0..round[0]]);
                defer _ = mlx.mlx_array_free(a);
                const c = try model.verify(cut_m, ids[at..][0..round[0]]);
                defer _ = mlx.mlx_array_free(c);
                try expectSameBits(&sc, a, c);
                try model.finishVerify(ref_m, round[1]);
                try model.finishVerify(cut_m, round[1]);
                at += round[1] + 1;
            }
            var da = try model.draftHeads(ref_m, ids[at - 1], .full);
            defer da.deinit();
            var dc = try model.draftHeads(cut_m, ids[at - 1], .full);
            defer dc.deinit();
            try testing.expectEqualSlices(u32, da.ids, dc.ids);
            try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(da.logits), std.mem.sliceAsBytes(dc.logits));
        }
        try testing.expect(model.spec_rebuilt > rebuilt0);
    }
}

fn enablePrefetch(served: *StqServed, lead_rows: u8, n: u8, m: u8) !void {
    try served.engine.startPrefetch();
    served.m.prefetch_lead_rows = lead_rows;
    served.m.prefetch_n = n;
    served.m.prefetch_m = m;
}

test "dsv41 prefetch: experts read ahead leave the pool, the map and the slab bytes as a route to them does, and the route then hits (DSV41_STQ_TINY)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const dir = try stqDir();
    const s = mlx.gpuStream();
    var ahead: StqServed = undefined;
    try ahead.init(dir, .bfloat16, true, false, 4, s);
    defer ahead.deinit();
    try ahead.engine.startPrefetch();
    var routed: StqServed = undefined;
    try routed.init(dir, .bfloat16, true, false, 4, s);
    defer routed.deinit();
    const layer: u16 = 3;
    // Two rows of three ranked candidates; each asks for its first two: the experts 5, 1 and 6.
    const cand = [_]u16{ 5, 1, 7, 1, 6, 2 };
    ahead.engine.prefetch(layer, &cand, 2, 3, 2);
    ahead.engine.prefetchWait(layer);
    {
        var p = try routed.engine.prepareHost(layer, &.{ 5, 1, 6 });
        p.deinit();
    }
    const a = &ahead.engine.layers[layer];
    const r = &routed.engine.layers[layer];
    const gl = a.grouped.?;
    for (0..gl.groups.count) |g| {
        try testing.expectEqualSlices(i32, r.grouped.?.caches[g].expert_to_slot, gl.caches[g].expert_to_slot);
        try testing.expectEqualSlices(u16, r.grouped.?.caches[g].slot_to_expert, gl.caches[g].slot_to_expert);
        try testing.expectEqualSlices(bool, r.grouped.?.caches[g].ready, gl.caches[g].ready);
    }
    for (a.slabs, r.slabs) |x, y| try testing.expectEqualSlices(u8, y.slab.?.bytes, x.slab.?.bytes);
    // The route to the same experts finds each in its slot and reads nothing.
    const filled = ahead.engine.fill_experts_total;
    var p = try ahead.engine.prepareHost(layer, &.{ 6, 5, 1 });
    p.deinit();
    try testing.expectEqual(filled, ahead.engine.fill_experts_total);
    try testing.expectEqual(@as(u64, 3), ahead.engine.prefetcher.?.experts_read.load(.monotonic));
    // Experts the pool holds are not read again.
    ahead.engine.prefetch(layer, &cand, 2, 3, 2);
    ahead.engine.prefetchWait(layer);
    try testing.expectEqual(@as(u64, 3), ahead.engine.prefetcher.?.experts_read.load(.monotonic));
}

test "dsv41 prefetch: decode steps and verify rounds with the pools read ahead give the bytes of a model that reads nothing ahead, through hits, misses and rebuilt layers (DSV41_STQ_TINY)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const dir = try stqDir();
    var fx = try model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    const s = mlx.gpuStream();
    var sc = Scope.init(testing.allocator, s);
    defer sc.deinit();
    const ids = try model.tokens(&sc, &fx);
    defer testing.allocator.free(ids);
    for ([_]u64{ 4, 3 }) |slots| {
        var plain: StqServed = undefined;
        try plain.init(dir, .bfloat16, true, true, slots, s);
        defer plain.deinit();
        // The shortest lead, ranking every expert of a row.
        var wide: StqServed = undefined;
        try wide.init(dir, .bfloat16, true, true, slots, s);
        defer wide.deinit();
        try enablePrefetch(&wide, 2, 8, 2);
        var narrow: StqServed = undefined;
        try narrow.init(dir, .bfloat16, true, true, slots, s);
        defer narrow.deinit();
        try enablePrefetch(&narrow, 2, 3, 1);
        const pa = try model.extend(plain.m, ids[0..20], .last);
        _ = mlx.mlx_array_free(pa);
        for ([_]*StqServed{ &wide, &narrow }) |served| {
            const pb = try model.extend(served.m, ids[0..20], .last);
            _ = mlx.mlx_array_free(pb);
        }
        var at: usize = 20;
        // (rows, accepted) of a verify round; 0 rows is a one-row decode step.
        for ([_][2]usize{ .{ 0, 0 }, .{ 5, 2 }, .{ 0, 0 }, .{ 0, 0 }, .{ 6, 0 }, .{ 4, 3 }, .{ 0, 0 }, .{ 3, 1 }, .{ 0, 0 }, .{ 0, 0 }, .{ 2, 1 }, .{ 0, 0 } }) |round| {
            const a = if (round[0] == 0) try model.extend(plain.m, ids[at..][0..1], .all) else try model.verify(plain.m, ids[at..][0..round[0]]);
            defer _ = mlx.mlx_array_free(a);
            for ([_]*StqServed{ &wide, &narrow }) |served| {
                const c = if (round[0] == 0) try model.extend(served.m, ids[at..][0..1], .all) else try model.verify(served.m, ids[at..][0..round[0]]);
                defer _ = mlx.mlx_array_free(c);
                try expectSameBits(&sc, a, c);
            }
            if (round[0] == 0) {
                at += 1;
            } else {
                try model.finishVerify(plain.m, round[1]);
                try model.finishVerify(wide.m, round[1]);
                try model.finishVerify(narrow.m, round[1]);
                at += round[1] + 1;
            }
        }
        try testing.expect(wide.engine.prefetcher.?.experts_read.load(.monotonic) > 0);
        try testing.expect(narrow.engine.prefetcher.?.experts_read.load(.monotonic) > 0);
    }
}

test "dsv41 prefetch: one-row decode steps ask for nothing and keep the bytes (DSV41_STQ_TINY)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const dir = try stqDir();
    var fx = try model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    const s = mlx.gpuStream();
    var sc = Scope.init(testing.allocator, s);
    defer sc.deinit();
    const ids = try model.tokens(&sc, &fx);
    defer testing.allocator.free(ids);
    var plain: StqServed = undefined;
    try plain.init(dir, .bfloat16, true, false, 3, s);
    defer plain.deinit();
    var armed: StqServed = undefined;
    try armed.init(dir, .bfloat16, true, false, 3, s);
    defer armed.deinit();
    try enablePrefetch(&armed, 2, 8, 2);
    var at: usize = 0;
    for ([_]usize{ 20, 1, 1, 1, 1, 1, 1 }) |n| {
        const a = try model.extend(plain.m, ids[at..][0..n], .all);
        defer _ = mlx.mlx_array_free(a);
        const b = try model.extend(armed.m, ids[at..][0..n], .all);
        defer _ = mlx.mlx_array_free(b);
        try expectSameBits(&sc, a, b);
        at += n;
    }
    try testing.expectEqual(@as(u64, 0), armed.engine.prefetcher.?.jobs_queued.load(.monotonic));
}

test "dsv41 drop: the STQ pack with experts dropped gives the logits of the full pools under the same router mask, bit for bit, and never reads or caches a dropped expert (DSV41_STQ_TINY)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const dir = try stqDir();
    var fx = try model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    const s = mlx.gpuStream();
    var sc = Scope.init(testing.allocator, s);
    defer sc.deinit();
    const ids = try model.tokens(&sc, &fx);
    defer testing.allocator.free(ids);
    var d = try model.testDrop(8);
    defer d.deinit();
    // A cache that holds every kept expert, and one that streams through hits, misses and the union.
    for ([_]u64{ 8, 3 }) |slots| {
        var ref: StqServed = undefined;
        try ref.initDrop(dir, .bfloat16, true, false, slots, s, &d, false);
        defer ref.deinit();
        var cut: StqServed = undefined;
        try cut.initDrop(dir, .bfloat16, true, false, slots, s, &d, true);
        defer cut.deinit();
        // A pool that holds every expert serves them resident, with no union workspace to shrink.
        if (!ref.engine.plan.resident) {
            try testing.expect(cut.engine.plan.workspace_bytes < ref.engine.plan.workspace_bytes);
            var union_bytes: [2]usize = .{ 0, 0 };
            for ([_]*const stream.Engine{ &ref.engine, &cut.engine }, 0..) |e, k| {
                for (e.grouped_union) |operand| union_bytes[k] += operand.stride * operand.count;
            }
            try testing.expect(union_bytes[1] < union_bytes[0]);
        }
        try testing.expect(cut.engine.droppedBytes() > 0 and ref.engine.droppedBytes() == 0);
        var at: usize = 0;
        for ([_]usize{ 20, 1, 1, 5, 1, 3, 1, 16, 1, 1 }) |n| {
            const take = @min(n, ids.len - at);
            if (take == 0) break;
            const a = try model.extend(ref.m, ids[at..][0..take], .all);
            defer _ = mlx.mlx_array_free(a);
            const b = try model.extend(cut.m, ids[at..][0..take], .all);
            defer _ = mlx.mlx_array_free(b);
            at += take;
            const ga = try model.hostRows(&sc, a);
            defer testing.allocator.free(ga);
            const gb = try model.hostRows(&sc, b);
            defer testing.allocator.free(gb);
            try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(ga), std.mem.sliceAsBytes(gb));
        }
        for (cut.engine.layers, 0..) |layer, li| {
            const gl = layer.grouped orelse continue;
            for (0..gl.groups.count) |g| for (0..gl.groups.experts[g]) |local| {
                if (d.isDropped(li, gl.groups.base[g] + local)) try testing.expect(gl.caches[g].expert_to_slot[local] < 0);
            };
        }
        const gone: u16 = model.testDropIds(0)[0];
        try testing.expectError(error.ExpertDropped, cut.engine.prepareHost(0, &.{gone}));
        try testing.expectError(error.ExpertDropped, cut.engine.prepareHostWide(0, &.{ 1, gone }));
    }
}

/// Every entry of `fixture` linked into `tmp`.
fn linkPack(tmp: *testing.TmpDir, fixture: []const u8) !void {
    const io = std.Io.Threaded.global_single_threaded.io();
    var src = try std.Io.Dir.openDirAbsolute(io, fixture, .{ .iterate = true });
    defer src.close(io);
    var it = src.iterate();
    while (try it.next(io)) |entry| {
        var target: [1024]u8 = undefined;
        try tmp.dir.symLink(io, try std.fmt.bufPrint(&target, "{s}/{s}", .{ fixture, entry.name }), entry.name, .{});
    }
}

/// A scratch pack: `fixture` linked in, plus an `expert_usage.json` that counts the experts `model.testDropIds` names
/// least used in each of `layers` layers. The caller frees the path.
fn usageDir(tmp: *testing.TmpDir, fixture: []const u8, layers: usize, experts: usize) ![:0]u8 {
    const io = std.Io.Threaded.global_single_threaded.io();
    try linkPack(tmp, fixture);
    var json: std.ArrayList(u8) = .empty;
    defer json.deinit(testing.allocator);
    try json.appendSlice(testing.allocator, "{\"version\":1,\"source\":\"unit\",\"tokens\":1,\"top_k\":3,\"layers\":[");
    for (0..layers) |li| {
        try json.appendSlice(testing.allocator, if (li == 0) "[" else ",[");
        const gone = model.testDropIds(li);
        for (0..experts) |e| {
            var buf: [24]u8 = undefined;
            const least = std.mem.indexOfScalar(u16, &gone, @intCast(e)) != null;
            try json.appendSlice(testing.allocator, try std.fmt.bufPrint(&buf, "{s}{d}", .{ if (e == 0) "" else ",", if (least) @as(usize, 1) else 100 + e }));
        }
        try json.appendSlice(testing.allocator, "]");
    }
    try json.appendSlice(testing.allocator, "]}");
    try tmp.dir.writeFile(io, .{ .sub_path = expert_usage.file_name, .data = json.items });
    return tmp.dir.realPathFileAlloc(io, ".", testing.allocator);
}

test "dsv41 drop: the transformer loads the pack's usage list, bills the smaller expert set and serves the masked logits (DSV41_STQ_TINY)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const io = std.Io.Threaded.global_single_threaded.io();
    const dir = try stqDir();
    const model_mod = @import("model.zig");
    const scheduler_mod = @import("scheduler.zig");
    const Transformer = @import("transformer.zig").Transformer;
    var fx = try model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    const s = mlx.gpuStream();
    var sc = Scope.init(testing.allocator, s);
    defer sc.deinit();
    const ids = try model.tokens(&sc, &fx);
    defer testing.allocator.free(ids);
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const root = try usageDir(&tmp, dir, 8, 8);
    defer testing.allocator.free(root);

    var config = try model_mod.parseConfig(io, testing.allocator, dir);
    defer config.deinit(testing.allocator);
    try testing.expect(config.drop_expert == 0 and config.keptExperts() == 8);
    var plain = config;
    config.drop_expert = 3;
    try testing.expectEqual(@as(u32, 5), config.keptExperts());
    // The plan counts the dropped experts out of the union workspace and of the budget's fixed bill.
    const full_plan = try scheduler_mod.planExpertStreamingWith(io, testing.allocator, &plain, root, 2 << 30, false, 0);
    const cut_plan = try scheduler_mod.planExpertStreamingWith(io, testing.allocator, &config, root, 2 << 30, false, 0);
    // The tiny pack fits the budget whole, so the bill is the resident one: every kept expert and its meta table.
    try testing.expect(full_plan.cache.resident and cut_plan.cache.resident);
    try testing.expect(cut_plan.cache.cache_bytes < full_plan.cache.cache_bytes);
    try testing.expect(cut_plan.resolved.ledger.?.cache_bytes < full_plan.resolved.ledger.?.cache_bytes);
    // A pack without the list refuses before any weight loads.
    try testing.expectError(error.ExpertDropRefused, scheduler_mod.planExpertStreamingWith(io, testing.allocator, &config, dir, 2 << 30, false, 0));

    config.expert_streaming = true;
    config.expert_cache_bytes = 64 << 20;
    config.expert_source_dir = try testing.allocator.dupe(u8, root);
    var served: StqServed = undefined;
    var d = try model.testDrop(8);
    defer d.deinit();
    try served.initDrop(dir, .bfloat16, true, config.dsparkOn(), 8, s, &d, true);
    defer served.deinit();
    var weights = try model_mod.loadWeightsForConfig(io, testing.allocator, dir, &config, false);
    defer weights.deinit();
    var xfm = try Transformer.init(io, testing.allocator, config, &weights);
    defer xfm.deinit();
    const engine = xfm.expert_stream.?;
    try testing.expect(engine.geometry.drop != null and engine.geometry.drop_n == 3 and engine.droppedBytes() > 0);

    var at: usize = 0;
    for ([_]usize{ 25, 1, 1, 7, 1 }) |n| {
        const take = @min(n, ids.len - at);
        const chunk = try sc.ints(@as([]const i32, @ptrCast(ids[at .. at + take])), &.{ 1, @intCast(take) });
        const logits = try xfm.forward(chunk);
        defer _ = mlx.mlx_array_free(logits);
        const want = try model.extend(served.m, ids[at .. at + take], .last);
        defer _ = mlx.mlx_array_free(want);
        at += take;
        const got = try model.hostRows(&sc, logits);
        defer testing.allocator.free(got);
        const exp = try model.hostRows(&sc, want);
        defer testing.allocator.free(exp);
        try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(exp), std.mem.sliceAsBytes(got));
    }
}

test "dsv41 drop: a real STQ pack's headers and its usage file plan a drop of 8 and 16 without emptying a rate group (SUSHI_DSV41_REAL_PACK, SUSHI_DSV41_REAL_USAGE)" {
    const io = std.Io.Threaded.global_single_threaded.io();
    const pack = std.mem.span(std.c.getenv("SUSHI_DSV41_REAL_PACK") orelse return error.SkipZigTest);
    const usage = std.mem.span(std.c.getenv("SUSHI_DSV41_REAL_USAGE") orelse return error.SkipZigTest);
    const model_mod = @import("model.zig");
    const scheduler_mod = @import("scheduler.zig");
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try linkPack(&tmp, pack);
    var link: [1024]u8 = undefined;
    try tmp.dir.symLink(io, try std.fmt.bufPrint(&link, "{s}/{s}", .{ usage, expert_usage.file_name }), expert_usage.file_name, .{});
    const root = try tmp.dir.realPathFileAlloc(io, ".", testing.allocator);
    defer testing.allocator.free(root);
    var config = try model_mod.parseConfig(io, testing.allocator, pack);
    defer config.deinit(testing.allocator);
    const whole = try scheduler_mod.planExpertStreamingWith(io, testing.allocator, &config, root, 116 << 30, false, 0);
    for ([_]u16{ 8, 16 }) |n| {
        config.drop_expert = n;
        const cut = try scheduler_mod.planExpertStreamingWith(io, testing.allocator, &config, root, 116 << 30, false, 0);
        const freed = if (whole.cache.resident) whole.cache.cache_bytes - cut.cache.cache_bytes else whole.cache.workspace_bytes - cut.cache.workspace_bytes;
        if (std.c.getenv("SUSHI_DSV41_STQ_REPORT") != null) std.debug.print("dsv41 drop {d}: resident {} -> {}, union workspace {d:.3} -> {d:.3} GB, expert cache {d:.3} -> {d:.3} GB, {d} -> {d} slots/layer\n", .{
            n,
            whole.cache.resident,
            cut.cache.resident,
            @as(f64, @floatFromInt(whole.cache.workspace_bytes)) / 1e9,
            @as(f64, @floatFromInt(cut.cache.workspace_bytes)) / 1e9,
            @as(f64, @floatFromInt(whole.cache.cache_bytes)) / 1e9,
            @as(f64, @floatFromInt(cut.cache.cache_bytes)) / 1e9,
            whole.cache.slots_per_layer,
            cut.cache.slots_per_layer,
        });
        try testing.expect(freed > 0);
    }
}

test "dsv41 stq shell: the transformer loads an STQ pack through the one loader and serves the model's logits (DSV41_STQ_TINY)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const io = std.Io.Threaded.global_single_threaded.io();
    const dir = try stqDir();
    const model_mod = @import("model.zig");
    const Transformer = @import("transformer.zig").Transformer;
    var fx = try model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    const s = mlx.gpuStream();
    var sc = Scope.init(testing.allocator, s);
    defer sc.deinit();
    const ids = try model.tokens(&sc, &fx);
    defer testing.allocator.free(ids);

    var config = try model_mod.parseConfig(io, testing.allocator, dir);
    defer config.deinit(testing.allocator);
    try testing.expect(config.isDsv41() and config.expertStreamingRequired());
    try testing.expectEqual(@import("expert_quant.zig").Layout.exl3_k4, config.expert_layout);
    config.expert_streaming = true;
    config.expert_cache_bytes = 64 << 20;
    config.expert_source_dir = try testing.allocator.dupe(u8, dir);

    var served: StqServed = undefined;
    try served.init(dir, .bfloat16, true, config.dsparkOn(), 8, s);
    defer served.deinit();

    var weights = try model_mod.loadWeightsForConfig(io, testing.allocator, dir, &config, false);
    defer weights.deinit();
    var xfm = try Transformer.init(io, testing.allocator, config, &weights);
    defer xfm.deinit();
    try testing.expect(xfm.dsv41_stq_experts != null and xfm.dsv41.?.head.stq != null);

    var at: usize = 0;
    for ([_]usize{ 25, 1, 1, 7, 1 }) |n| {
        const take = @min(n, ids.len - at);
        const chunk = try sc.ints(@as([]const i32, @ptrCast(ids[at .. at + take])), &.{ 1, @intCast(take) });
        const logits = try xfm.forward(chunk);
        defer _ = mlx.mlx_array_free(logits);
        const want = try model.extend(served.m, ids[at .. at + take], .last);
        defer _ = mlx.mlx_array_free(want);
        at += take;
        const got = try model.hostRows(&sc, logits);
        defer testing.allocator.free(got);
        const exp = try model.hostRows(&sc, want);
        defer testing.allocator.free(exp);
        try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(exp), std.mem.sliceAsBytes(got));
    }
}

test "dsv41 borrow: a released pool gives MLX its bytes back, its routes read the union with the same bytes out, and a restore reads its experts back (DSV41_STQ_TINY)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const dir = try stqDir();
    var fx = try model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    const s = mlx.gpuStream();
    var sc = Scope.init(testing.allocator, s);
    defer sc.deinit();
    const ids = try model.tokens(&sc, &fx);
    defer testing.allocator.free(ids);
    var kept: StqServed = undefined;
    try kept.init(dir, .bfloat16, true, false, 3, s);
    defer kept.deinit();
    var lent: StqServed = undefined;
    try lent.init(dir, .bfloat16, true, false, 3, s);
    defer lent.deinit();
    const split = lent.m.ced_split.?;
    lent.engine.borrow_from = @intCast(split);
    var published: u64 = 0;
    lent.engine.borrow_publish = &published;
    lent.engine.publishBorrowable();
    var pools: u64 = 0;
    for (split..lent.m.cfg.layers) |li| pools += lent.engine.poolBytes(@intCast(li));
    try testing.expect(pools > 0);
    try testing.expectEqual(pools, published);

    var at: usize = 0;
    for ([_]usize{ 20, 9, 1, 1, 3, 1 }, 0..) |n, step| {
        const take = @min(n, ids.len - at);
        if (step == 1) {
            const li: u16 = @intCast(lent.m.cfg.layers - 1);
            const bytes = lent.engine.poolBytes(li);
            _ = mlx.mlx_synchronize(s);
            var before: usize = 0;
            _ = mlx.mlx_get_active_memory(&before);
            try testing.expectEqual(bytes, try lent.engine.borrowLayer(li));
            var after: usize = 0;
            _ = mlx.mlx_get_active_memory(&after);
            try testing.expectEqual(bytes, before - after);
            try testing.expectEqual(@as(u64, 0), lent.engine.poolBytes(li));
            try testing.expectEqual(pools - bytes, published);
        }
        if (step == 3) {
            const li: u16 = @intCast(lent.m.cfg.layers - 1);
            const saved = try testing.allocator.dupe(u16, lent.engine.layers[li].saved);
            defer testing.allocator.free(saved);
            try testing.expect(saved.len > 0);
            try lent.engine.restoreLayer(li);
            try testing.expectEqual(pools, published);
            const gl = lent.engine.layers[li].grouped.?;
            for (saved) |e| {
                const g = gl.groups.groupOf(e);
                const slot = gl.caches[g].expert_to_slot[e - gl.groups.base[g]];
                try testing.expect(slot >= 0 and gl.caches[g].ready[@intCast(slot)]);
            }
        }
        const a = try model.extend(kept.m, ids[at..][0..take], .last);
        defer _ = mlx.mlx_array_free(a);
        const b = try model.extend(lent.m, ids[at..][0..take], .last);
        defer _ = mlx.mlx_array_free(b);
        at += take;
        const ga = try model.hostRows(&sc, a);
        defer testing.allocator.free(ga);
        const gb = try model.hostRows(&sc, b);
        defer testing.allocator.free(gb);
        try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(ga), std.mem.sliceAsBytes(gb));
    }
}

const BorrowProbe = struct {
    var engine: ?*stream.Engine = null;
    var base: u64 = 0;

    /// A box whose free memory is `base` plus what the decoder pools have given back, less a resident pack's union.
    fn headroom(_: *const @import("model.zig").ModelConfig) u64 {
        const e = engine.?;
        var lent: u64 = 0;
        for (e.borrow_from..e.layers.len) |li| lent += e.lentBytes(@intCast(li));
        const made = if (e.plan.resident and e.grouped_union.len != 0) e.unionBytes() else 0;
        return (base + lent) -| made;
    }
};

test "dsv41 borrow: a prefill short of memory releases the latest decoder pools, serves the same bytes, and decode reads them back (DSV41_STQ_TINY)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    // A streamed pack, and a resident one whose first release also pays for the union workspace.
    for ([_]u64{ 3 * 8 * 256 * 128 * 3 * 64 / 128, 64 << 20 }) |cache_bytes| try borrowCase(cache_bytes);
}

fn borrowCase(cache_bytes: u64) !void {
    const io = std.Io.Threaded.global_single_threaded.io();
    const dir = try stqDir();
    const model_mod = @import("model.zig");
    const Transformer = @import("transformer.zig").Transformer;
    const borrow = @import("dsv41_borrow.zig");
    var fx = try model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    const s = mlx.gpuStream();
    var sc = Scope.init(testing.allocator, s);
    defer sc.deinit();
    const ids = try model.tokens(&sc, &fx);
    defer testing.allocator.free(ids);

    var config = try model_mod.parseConfig(io, testing.allocator, dir);
    defer config.deinit(testing.allocator);
    config.expert_streaming = true;
    config.expert_cache_bytes = cache_bytes;
    config.expert_source_dir = try testing.allocator.dupe(u8, dir);
    var served: StqServed = undefined;
    try served.init(dir, .bfloat16, true, config.dsparkOn(), 8, s);
    defer served.deinit();
    var weights = try model_mod.loadWeightsForConfig(io, testing.allocator, dir, &config, false);
    defer weights.deinit();
    var xfm = try Transformer.init(io, testing.allocator, config, &weights);
    defer xfm.deinit();
    const engine = xfm.expert_stream.?;
    const last: u16 = @intCast(config.dsv41.?.layers - 1);
    try testing.expectEqual(@as(u16, @intCast(xfm.dsv41.?.ced_split.?)), engine.borrow_from);

    // The first chunk finds the need less one and a half of the last two pools: those two go, the rest stay.
    const first = 25;
    const need = borrow.prefillNeed(&xfm.dsv41.?.cfg, first, first);
    BorrowProbe.engine = engine;
    try testing.expectEqual(cache_bytes == 64 << 20, engine.plan.resident);
    BorrowProbe.base = need + engine.unionPending() - engine.poolBytes(last) - engine.poolBytes(last - 1) / 2;
    borrow.headroom_fn = &BorrowProbe.headroom;
    defer {
        borrow.headroom_fn = null;
        BorrowProbe.engine = null;
    }

    var at: usize = 0;
    for ([_]usize{ first, 1, 1, 7, 1 }, 0..) |n, step| {
        const take = @min(n, ids.len - at);
        const chunk = try sc.ints(@as([]const i32, @ptrCast(ids[at .. at + take])), &.{ 1, @intCast(take) });
        const logits = try xfm.forward(chunk);
        defer _ = mlx.mlx_array_free(logits);
        for (engine.borrow_from..engine.layers.len) |li| {
            const lent = step == 0 and li + 2 >= engine.layers.len;
            try testing.expectEqual(lent, engine.isBorrowed(@intCast(li)));
        }
        const want = try model.extend(served.m, ids[at .. at + take], .last);
        defer _ = mlx.mlx_array_free(want);
        at += take;
        const got = try model.hostRows(&sc, logits);
        defer testing.allocator.free(got);
        const exp = try model.hostRows(&sc, want);
        defer testing.allocator.free(exp);
        try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(exp), std.mem.sliceAsBytes(got));
    }

    // A prefill that every pool cannot cover is refused before it releases anything.
    try xfm.resetCache();
    BorrowProbe.base = 0;
    const chunk = try sc.ints(@as([]const i32, @ptrCast(ids[0..first])), &.{ 1, first });
    try testing.expectError(error.PrefillDoesNotFit, xfm.forward(chunk));
    for (engine.borrow_from..engine.layers.len) |li| try testing.expect(!engine.isBorrowed(@intCast(li)));
}

fn expectSameStep(sc: *Scope, a: *model.Model, b: *model.Model, ids: []const u32) !void {
    const la = try model.extend(a, ids, .last);
    defer _ = mlx.mlx_array_free(la);
    const lb = try model.extend(b, ids, .last);
    defer _ = mlx.mlx_array_free(lb);
    const ga = try model.hostRows(sc, la);
    defer testing.allocator.free(ga);
    const gb = try model.hostRows(sc, lb);
    defer testing.allocator.free(gb);
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(ga), std.mem.sliceAsBytes(gb));
}

test "dsv41 resident: a budget that holds every expert reads them all at load, makes no union workspace, and serves the streamed bytes with no route read (DSV41_STQ_TINY)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const dir = try stqDir();
    var fx = try model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    const s = mlx.gpuStream();
    var sc = Scope.init(testing.allocator, s);
    defer sc.deinit();
    const ids = try model.tokens(&sc, &fx);
    defer testing.allocator.free(ids);
    var streamed: StqServed = undefined;
    try streamed.init(dir, .bfloat16, true, false, 3, s);
    defer streamed.deinit();
    var resident: StqServed = undefined;
    try resident.initBytes(dir, .bfloat16, true, false, 1 << 40, s);
    defer resident.deinit();
    const engine = &resident.engine;
    try testing.expect(engine.plan.resident and !streamed.engine.plan.resident);
    try testing.expectEqual(@as(usize, 0), engine.grouped_union.len);
    for (0..resident.m.cfg.layers) |li| try testing.expect(engine.residentRoute(@intCast(li)) != null);
    const loaded = engine.fill_bytes_total;
    try testing.expect(loaded > 0);

    const split = resident.m.ced_split.?;
    engine.borrow_from = @intCast(split);
    const last: u16 = @intCast(resident.m.cfg.layers - 1);
    var at: usize = 0;
    for ([_]usize{ 20, 9, 1, 1, 3, 1 }, 0..) |n, step| {
        const take = @min(n, ids.len - at);
        if (step == 1) {
            // A release pays for the union workspace out of the pool it gives back, and its routes read the union.
            const pool = engine.poolBytes(last);
            try testing.expectEqual(pool, try engine.borrowLayer(last));
            try testing.expectEqual(@as(usize, stream.quant.component_count), engine.grouped_union.len);
            try testing.expect(engine.residentRoute(last) == null);
        }
        if (step == 3) {
            // The last restore drops the workspace and the layer maps its picks on the GPU again.
            try engine.restoreLayer(last);
            try testing.expectEqual(@as(usize, 0), engine.grouped_union.len);
            try testing.expect(engine.residentRoute(last) != null);
        }
        try expectSameStep(&sc, streamed.m, resident.m, ids[at..][0..take]);
        at += take;
        // Only the released layer's picks and its restore read the SSD.
        if (step == 0) try testing.expectEqual(loaded, engine.fill_bytes_total);
    }
}

test "dsv41 drop: a budget that holds the kept experts alone serves them resident, and a dropped id is in no pool and no map (DSV41_STQ_TINY)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const dir = try stqDir();
    var fx = try model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    const s = mlx.gpuStream();
    var sc = Scope.init(testing.allocator, s);
    defer sc.deinit();
    const ids = try model.tokens(&sc, &fx);
    defer testing.allocator.free(ids);
    var d = try model.testDrop(8);
    defer d.deinit();
    // The bytes that hold every expert, and every kept one.
    var whole: StqServed = undefined;
    try whole.initBytesDrop(dir, .bfloat16, true, false, 1 << 40, s, &d, false);
    const all_bytes = whole.engine.store.quantized.?.shape().all_bytes;
    whole.deinit();
    var kept: StqServed = undefined;
    try kept.initBytesDrop(dir, .bfloat16, true, false, 1 << 40, s, &d, true);
    const kept_bytes = kept.engine.store.quantized.?.shape().all_bytes;
    kept.deinit();
    try testing.expect(kept_bytes < all_bytes);

    // Without the drop billed the budget is short of resident; with it, it holds.
    var streamed: StqServed = undefined;
    try streamed.initBytesDrop(dir, .bfloat16, true, false, kept_bytes, s, &d, false);
    defer streamed.deinit();
    try testing.expect(!streamed.engine.plan.resident);
    var resident: StqServed = undefined;
    try resident.initBytesDrop(dir, .bfloat16, true, false, kept_bytes, s, &d, true);
    defer resident.deinit();
    const engine = &resident.engine;
    try testing.expect(engine.plan.resident and engine.plan.cache_bytes == kept_bytes);
    try testing.expectEqual(@as(usize, 0), engine.grouped_union.len);
    for (engine.layers, 0..) |layer, li| {
        const gl = layer.grouped orelse continue;
        try testing.expect(engine.residentRoute(@intCast(li)) != null);
        var pooled: u32 = 0;
        for (0..gl.groups.count) |g| {
            try testing.expectEqual(@as(u32, gl.groups.keptOf(g)), gl.slots[g]);
            pooled += gl.slots[g];
            for (0..gl.groups.experts[g]) |local| {
                const gone = d.isDropped(li, gl.groups.base[g] + local);
                try testing.expectEqual(gone, gl.caches[g].expert_to_slot[local] < 0);
            }
        }
        try testing.expectEqual(@as(u32, 5), pooled);
        const map = try sc_mod.hostI32(layer.resident_map);
        var seen: [8]bool = @splat(false);
        for (map, 0..) |slot, e| {
            if (d.isDropped(li, e)) {
                try testing.expectEqual(@as(i32, -1), slot);
            } else {
                try testing.expect(slot >= 0 and slot < pooled and !seen[@intCast(slot)]);
                seen[@intCast(slot)] = true;
            }
        }
    }
    var at: usize = 0;
    for ([_]usize{ 20, 9, 1, 1, 3, 1 }) |n| {
        const take = @min(n, ids.len - at);
        try expectSameStep(&sc, streamed.m, resident.m, ids[at..][0..take]);
        at += take;
    }
}

test "dsv41 lossy pick: on rate-grouped layers a missed expert takes the best pooled one within the tolerance, from any group, and nothing at 0 (DSV41_STQ_TINY)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const dir = try stqDir();
    var fx = try model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    const s = mlx.gpuStream();
    var sc = Scope.init(testing.allocator, s);
    defer sc.deinit();
    const ids = try model.tokens(&sc, &fx);
    defer testing.allocator.free(ids);
    var served: StqServed = undefined;
    try served.init(dir, .bfloat16, true, false, 3, s);
    defer served.deinit();
    const engine = &served.engine;
    const warm = try model.extend(served.m, ids[0..20], .last);
    _ = mlx.mlx_array_free(warm);

    const E = served.m.cfg.experts;
    const gl = engine.layers[0].grouped.?;
    var missed: ?u16 = null;
    var pooled: ?u16 = null;
    for (0..E) |e| {
        const g = gl.groups.groupOf(@intCast(e));
        const slot = gl.caches[g].expert_to_slot[e - gl.groups.base[g]];
        if (slot >= 0 and gl.caches[g].ready[@intCast(slot)]) {
            if (pooled == null or (missed != null and gl.groups.groupOf(pooled.?) == gl.groups.groupOf(missed.?))) pooled = @intCast(e);
        } else if (missed == null) missed = @intCast(e);
    }
    const m_id = missed.?;
    const p_id = pooled.?;
    var scores = try testing.allocator.alloc(f32, E);
    defer testing.allocator.free(scores);
    for ([_]struct { tol: f32, pooled_ratio: f32, swaps: u32 }{
        .{ .tol = 0, .pooled_ratio = 0.99, .swaps = 0 },
        .{ .tol = 0.2, .pooled_ratio = 0.7, .swaps = 0 },
        .{ .tol = 0.2, .pooled_ratio = 0.9, .swaps = 1 },
    }) |case| {
        @memset(scores, -20);
        scores[m_id] = 0;
        scores[p_id] = @log(case.pooled_ratio);
        var row = [_]u16{m_id};
        const before = engine.swapped_ids;
        try testing.expectEqual(case.swaps, try engine.swapGroupedMisses(0, &row, scores, case.tol, 1));
        try testing.expectEqual(if (case.swaps == 1) p_id else m_id, row[0]);
        try testing.expectEqual(before + case.swaps, engine.swapped_ids);
    }

    // Through the forward the pick engages at decode width; at 0 the counter never moves.
    const saved = stream.pick_tolerance;
    defer stream.pick_tolerance = saved;
    var at: usize = 20;
    for ([_]f32{ 0, 0.6 }) |tol| {
        stream.pick_tolerance = tol;
        const seen = engine.swap_seen_ids;
        for (0..6) |_| {
            const out = try model.extend(served.m, ids[at..][0..1], .last);
            _ = mlx.mlx_array_free(out);
            at += 1;
        }
        try testing.expectEqual(tol > 0, engine.swap_seen_ids > seen);
    }

    // A DSpark verify block (2..16 rows) picks too; a wider span (prefill) stays exact.
    stream.pick_tolerance = 0.6;
    for ([_]struct { rows: usize, acts: bool }{ .{ .rows = 4, .acts = true }, .{ .rows = 17, .acts = false } }) |case| {
        const seen = engine.swap_seen_ids;
        // `.all` keeps a wide pass whole: the CED tail of a `.last` pass is a decode-width span on its own.
        const out = try model.extendSpan(served.m, ids[0..case.rows], .all, 64);
        _ = mlx.mlx_array_free(out);
        try testing.expectEqual(case.acts, engine.swap_seen_ids > seen);
    }
}

/// ||a - t|| / ||t||.
fn relErr(a: []const f32, t: []const f32) f64 {
    var d: f64 = 0;
    var n: f64 = 0;
    for (a, t) |x, y| {
        d += (@as(f64, x) - y) * (@as(f64, x) - y);
        n += @as(f64, y) * y;
    }
    return @sqrt(d / n);
}

test "dsv41 stq: the served path is no farther from the f32 CPU truth than the GPU twin, layers 0 to 4 (DSV41_STQ_TINY)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const dir = try stqDir();
    var fx = try model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    const gs = mlx.gpuStream();
    const cs = mlx.mlx_default_cpu_stream_new();
    var sc = Scope.init(testing.allocator, gs);
    defer sc.deinit();
    const ids = try model.tokens(&sc, &fx);
    defer testing.allocator.free(ids);
    model.exact_matmul = true;
    var truth_src: StqServed = undefined;
    try truth_src.init(dir, .float32, false, false, 8, cs);
    defer truth_src.deinit();
    try denseTwin(&sc, truth_src.m, dir);
    truth_src.m.fused = false;
    var tt: model.Trace = .{ .gpa = testing.allocator };
    defer tt.deinit();
    truth_src.m.trace = &tt;
    const tl = try model.extendSpan(truth_src.m, ids, .all, 64);
    truth_src.m.trace = null;
    defer _ = mlx.mlx_array_free(tl);
    model.exact_matmul = false;

    var twin_src: StqServed = undefined;
    try twin_src.init(dir, .float32, false, false, 8, gs);
    defer twin_src.deinit();
    try denseTwin(&sc, twin_src.m, dir);
    twin_src.m.fused = false;
    var wt: model.Trace = .{ .gpa = testing.allocator };
    defer wt.deinit();
    twin_src.m.trace = &wt;
    const wl = try model.extendSpan(twin_src.m, ids, .all, 64);
    twin_src.m.trace = null;
    defer _ = mlx.mlx_array_free(wl);

    var served: StqServed = undefined;
    try served.init(dir, .float32, false, false, 8, gs);
    defer served.deinit();
    served.m.fused = false;
    var st: model.Trace = .{ .gpa = testing.allocator };
    defer st.deinit();
    served.m.trace = &st;
    const sl = try model.extendSpan(served.m, ids, .all, 64);
    served.m.trace = null;
    defer _ = mlx.mlx_array_free(sl);

    for (0..5) |li| for ([_][]const u8{ "attn_in", "attn", "ffn", "stream" }) |nm| {
        var nb: [32]u8 = undefined;
        const name = try std.fmt.bufPrint(&nb, "{s}_{d}", .{ nm, li });
        const t = traceOf(&sc, &tt, name) catch continue;
        defer testing.allocator.free(t);
        const a = try traceOf(&sc, &wt, name);
        defer testing.allocator.free(a);
        const b = try traceOf(&sc, &st, name);
        defer testing.allocator.free(b);
        const twin_err = relErr(a, t);
        const served_err = relErr(b, t);
        if (std.c.getenv("SUSHI_DSV41_STQ_REPORT") != null) std.debug.print("dsv41 stq truth {s}: twin {e:.3} served {e:.3}\n", .{ name, twin_err, served_err });
        try testing.expect(served_err <= twin_err * 1.1 + 1e-6 and served_err < 2e-2);
    };
}

/// A request on `m` from a fresh state: `prompt` prefilled, then until position `end` a plain step and verify rounds of
/// every width up to the block, each keeping a different count.
fn roundsTo(m: *model.Model, prompt: []const u32, end: usize) !void {
    model.resetState(m);
    const V = m.cfg.vocab;
    var t: u32 = blk: {
        const lg = try model.extend(m, prompt, .last);
        defer _ = mlx.mlx_array_free(lg);
        break :blk @intCast(argmax(try sc_mod.hostF32(lg)));
    };
    const widest: usize = if (m.dspark != null) m.cfg.dspark.block + 1 else 1;
    var width: usize = 1;
    while ((try model.ensureState(m)).n < end) : (width = width % widest + 1) {
        if (width == 1) {
            const lg = try model.extend(m, &.{t}, .last);
            defer _ = mlx.mlx_array_free(lg);
            t = @intCast(argmax(try sc_mod.hostF32(lg)));
            continue;
        }
        var d = try model.draftHeads(m, t, .conf);
        defer d.deinit();
        var round: [16]u32 = undefined;
        round[0] = t;
        @memcpy(round[1..width], d.ids[0 .. width - 1]);
        const vl = try model.verify(m, round[0..width]);
        defer _ = mlx.mlx_array_free(vl);
        const keep = (try model.ensureState(m)).n % width;
        try model.finishVerify(m, keep);
        t = @intCast(argmax((try sc_mod.hostF32(vl))[keep * V ..][0..V]));
    }
}

test "dsv41 kernels: after the load's warm, prompts of any length, every verify width and a growing context compile no Metal library on the STQ pack, streamed and resident, DSpark on and off (DSV41_STQ_TINY)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const dir = try stqDir();
    var fx = try model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    const s = mlx.gpuStream();
    var sc = Scope.init(testing.allocator, s);
    defer sc.deinit();
    const ids = try model.tokens(&sc, &fx);
    defer testing.allocator.free(ids);
    const keys = mlx.kernel_keys;
    _ = keys.trackForTest();
    // The served setting (bf16, QAT round trips): a cache of 3 of 8 experts a layer (misses and the union), and one
    // that holds them all (resident).
    for ([_]u64{ 3 * 8 * 256 * 128 * 3 * 64 / 128, 1 << 40 }) |budget| for ([_]bool{ true, false }) |dspark| {
        var served: StqServed = undefined;
        try served.initBytes(dir, .bfloat16, true, dspark, budget, s);
        defer served.deinit();
        try model.warmKernels(served.m);
        const warm = keys.count();
        try roundsTo(served.m, ids[0..20], 21);
        try roundsTo(served.m, ids[0..3], 3 + 30);
        try roundsTo(served.m, ids[0..29], 110);
        try roundsTo(served.m, ids[0..17], 60);
        if (keys.count() != warm) {
            std.debug.print("budget {d} dspark {}: {d} libraries built after warmup, the last: {s}\n", .{ budget, dspark, keys.count() - warm, keys.last() });
            return error.KernelRebuilt;
        }
    };
}

test "dsv41 kernels: the load's warm leaves no state, and the served logits are the unwarmed model's bit for bit (DSV41_STQ_TINY)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const dir = try stqDir();
    var fx = try model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    const s = mlx.gpuStream();
    var sc = Scope.init(testing.allocator, s);
    defer sc.deinit();
    const ids = try model.tokens(&sc, &fx);
    defer testing.allocator.free(ids);
    var rows: [2][]f32 = undefined;
    for (&rows, [_]bool{ false, true }) |*out, warm| {
        var served: StqServed = undefined;
        try served.init(dir, .bfloat16, true, true, 3, s);
        defer served.deinit();
        if (warm) {
            try model.warmKernels(served.m);
            try testing.expect(served.m.dec_state == null);
        }
        var all: std.ArrayList(f32) = .empty;
        errdefer all.deinit(testing.allocator);
        var t: u32 = 0;
        for (0..8) |i| {
            const lg = try model.extend(served.m, if (i == 0) ids[0..20] else &.{t}, .last);
            defer _ = mlx.mlx_array_free(lg);
            const row = try sc_mod.hostF32(lg);
            try all.appendSlice(testing.allocator, row);
            t = @intCast(argmax(row));
        }
        out.* = try all.toOwnedSlice(testing.allocator);
    }
    defer for (rows) |r| testing.allocator.free(r);
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(rows[0]), std.mem.sliceAsBytes(rows[1]));
}

test "dsv41 DSpark exact: a one-row round gives the bytes of row 0 of a wider verify at every position, and leaves the state the wider round leaves (DSV41_STQ_TINY)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const dir = try stqDir();
    var fx = try model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    const s = mlx.gpuStream();
    var sc = Scope.init(testing.allocator, s);
    defer sc.deinit();
    const ids = try model.tokens(&sc, &fx);
    defer testing.allocator.free(ids);
    // The tiny width is not the prepared chain's geometry, which the served verify rows take.
    exl3.rate_groups.prepared_force = true;
    defer exl3.rate_groups.prepared_force = null;
    const P = 24;
    var one: StqServed = undefined;
    try one.init(dir, .bfloat16, true, true, 8, s);
    defer one.deinit();
    var wide: StqServed = undefined;
    try wide.init(dir, .bfloat16, true, true, 8, s);
    defer wide.deinit();
    const V = one.m.cfg.vocab;
    var t: u32 = undefined;
    for ([_]*StqServed{ &one, &wide }) |m| {
        const lg = try model.extend(m.m, ids[0..P], .last);
        defer _ = mlx.mlx_array_free(lg);
        t = @intCast(argmax(try sc_mod.hostF32(lg)));
    }
    var at: usize = P;
    for ([_]usize{ 3, 2, 6, 4, 5, 3, 6, 2 }) |width| {
        const a = try model.verify(one.m, &.{t});
        defer _ = mlx.mlx_array_free(a);
        var round: [6]u32 = undefined;
        round[0] = t;
        @memcpy(round[1..width], ids[at + 1 ..][0 .. width - 1]);
        const b = try model.verify(wide.m, round[0..width]);
        defer _ = mlx.mlx_array_free(b);
        const ra = try sc_mod.hostF32(a);
        const rb = try sc_mod.hostF32(b);
        try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(ra[0..V]), std.mem.sliceAsBytes(rb[0..V]));
        try model.finishVerify(one.m, 0);
        try model.finishVerify(wide.m, 0);
        t = @intCast(argmax(ra[0..V]));
        at += 1;
    }
}
