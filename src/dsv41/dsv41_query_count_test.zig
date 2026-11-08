//! The shape, size and dtype queries (`mlx.query_count`, each an FFI call) of a DeepSeek decode token on the tiny
//! fixtures: a trellis trunk linear, the shared expert riding the routed chain, and a whole token.
const std = @import("std");
const mlx = @import("../mlx.zig");
const exl3 = @import("sushi_exl3");
const trunk_mod = @import("../trunk.zig");
const model_mod = @import("../model.zig");
const dsv41_model = @import("dsv41_model.zig");
const scope_mod = @import("dsv41_scope.zig");
const Scope = scope_mod.Scope;
const Trunk = trunk_mod.Trunk;
const testing = std.testing;

fn tinyDir() ![]const u8 {
    return std.mem.span(std.c.getenv("DSV41_TINY") orelse return error.SkipZigTest);
}

fn stqDir() ![]const u8 {
    return std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
}

fn activations(sc: *Scope, shape: []const c_int) !mlx.mlx_array {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const data = try testing.allocator.alloc(f32, n);
    defer testing.allocator.free(data);
    for (data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 17)) * 0.05 - 0.4;
    return sc.astype(try sc.floats(data, shape), .bfloat16);
}

test "dsv41 queries: a trellis trunk linear and a grouped one (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    const m = try dsv41_model.tinyModel(try stqDir(), .{ .act = .bfloat16, .experts = false }, mlx.gpuStream());
    defer m.deinit();
    const ly = &m.layers[0];
    const q = ly.wq_a.stq.?;
    const in: c_int = @intCast(m.cfg.dim);
    var sc = Scope.plain(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    var got: [4]u64 = undefined;
    for ([_]c_int{ 1, 5 }, 0..) |rows, i| {
        sc.rows = .{ .exact = rows > 1 };
        const x = try activations(&sc, &.{ rows, in });
        for (0..2) |_| {
            const before = mlx.query_count;
            _ = try q.apply(&sc, x);
            got[i] = mlx.query_count - before;
        }
    }
    const g = ly.wo_a.stq.?;
    const og: c_int = @intCast(m.cfg.o_groups);
    for ([_]c_int{ 1, 5 }, 2..) |rows, i| {
        sc.rows = .{ .exact = rows > 1 };
        const x = try activations(&sc, &.{ og, rows, mlx.getShape(g.p.trellis)[1] * 16 });
        for (0..2) |_| {
            const before = mlx.query_count;
            _ = try g.applyGrouped(&sc, x);
            got[i] = mlx.query_count - before;
        }
    }
    // A plain decode's one-bank launch is bound at load (`Stq.geo`, non-exact); the exact five rows and the grouped
    // bank read their shapes.
    try testing.expectEqual([4]u64{ 7, 22, 19, 19 }, got);
}

test "dsv41 queries: an mxfp8 trunk linear reads nothing off its weight (DSV41_TINY)" {
    try mlx.requireGpu();
    const m = try dsv41_model.tinyModel(try tinyDir(), .{ .act = .bfloat16, .experts = false }, mlx.gpuStream());
    defer m.deinit();
    var sc = Scope.plain(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    const x = try activations(&sc, &.{ 1, @intCast(m.cfg.dim) });
    var got: u64 = undefined;
    for (0..2) |_| {
        const before = mlx.query_count;
        _ = try m.layers[0].wq_a.apply(&sc, x);
        got = mlx.query_count - before;
    }
    try testing.expectEqual(@as(u64, 0), got);
}

test "dsv41 queries: the shared expert riding the routed chain (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    exl3.rate_groups.prepared_force = true;
    defer exl3.rate_groups.prepared_force = null;
    const m = try dsv41_model.tinyModel(try stqDir(), .{ .act = .bfloat16, .experts = false, .dspark = true }, mlx.gpuStream());
    defer m.deinit();
    const c = &m.cfg;
    const li = c.layers;
    var sc = Scope.plain(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    const k: c_int = @intCast(m.layers[li].top_k);
    var got: [2]u64 = undefined;
    for ([_]c_int{ 1, 3 }, 0..) |rows, i| {
        const x = try activations(&sc, &.{ rows, @intCast(c.dim) });
        var ids: [3 * 8]i32 = undefined;
        var wts: [3 * 8]f32 = undefined;
        const n: usize = @intCast(rows * k);
        for (0..n) |j| {
            ids[j] = @intCast(j % 4);
            wts[j] = 0.5;
        }
        const slots = try sc.ints(ids[0..n], &.{ rows, k });
        const w = try sc.floats(wts[0..n], &.{ rows, k });
        for (0..2) |_| {
            const before = mlx.query_count;
            const y = try dsv41_model.stqRouted(&sc, c.expert_dec.?, @intFromFloat(c.swiglu_limit), &m.dspark.?.stq[0], x, slots, w, .{ .layer = .{ .m = m, .li = li } });
            _ = mlx.mlx_array_eval(y);
            got[i] = mlx.query_count - before;
        }
    }
    // The whole chain of a layer: the routed picks with the shared expert riding its launches, whose lane shapes the
    // layer read at load.
    try testing.expectEqual([2]u64{ 12, 12 }, got);
}

fn sameBits(a: mlx.mlx_array, b: mlx.mlx_array, sc: *Scope) !void {
    const ha = try scope_mod.hostF32(try sc.astype(a, .float32));
    const hb = try scope_mod.hostF32(try sc.astype(b, .float32));
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(hb), std.mem.sliceAsBytes(ha));
}

test "dsv41 stq: a one-bank trunk linear's launches bound at load give the shape-reading launch's bytes (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    const m = try dsv41_model.tinyModel(try stqDir(), .{ .act = .bfloat16, .experts = false }, mlx.gpuStream());
    defer m.deinit();
    const ly = &m.layers[0];
    var sc = Scope.plain(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    for ([_]dsv41_model.Stq{ ly.wq_a.stq.?, ly.wq_b.stq.?, ly.shared[2].stq.? }) |q| {
        try testing.expect(q.geo != null);
        var bare = q;
        bare.geo = null;
        const in = mlx.getShape(q.p.trellis)[1] * 16;
        for ([_]bool{ false, true }) |exact| for (1..10) |rows| {
            sc.rows = .{ .exact = exact };
            const x = try activations(&sc, &.{ @intCast(rows), in });
            try sameBits(try q.apply(&sc, x), try bare.apply(&sc, x), &sc);
        };
    }
}

test "dsv41 stq: the shared bank's lane read at load gives the bytes of the lane read per call (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    exl3.rate_groups.prepared_force = true;
    defer exl3.rate_groups.prepared_force = null;
    const m = try dsv41_model.tinyModel(try stqDir(), .{ .act = .bfloat16, .experts = false, .dspark = true }, mlx.gpuStream());
    defer m.deinit();
    const c = &m.cfg;
    const li = c.layers;
    const ly = &m.layers[li];
    const rb = &m.dspark.?.stq[0];
    const lane = ly.shared_lane orelse return error.TestExpectedLane;
    const sq = [3]dsv41_model.Stq{ ly.shared[0].stq.?, ly.shared[1].stq.?, ly.shared[2].stq.? };
    try testing.expectEqual(exl3.rate_groups.SharedLane.read(.{ .gate = sq[0].p, .up = sq[1].p, .down = sq[2].p }, rb), lane);
    var sc = Scope.plain(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    const k: c_int = @intCast(ly.top_k);
    for ([_]c_int{ 1, 2, 3 }) |rows| {
        const x = try activations(&sc, &.{ rows, @intCast(c.dim) });
        var ids: [3 * 8]i32 = undefined;
        var wts: [3 * 8]f32 = undefined;
        const n: usize = @intCast(rows * k);
        for (0..n) |j| {
            ids[j] = @intCast((j * 3) % 4);
            wts[j] = 0.25 + 0.125 * @as(f32, @floatFromInt(j % 3));
        }
        const slots = try sc.ints(ids[0..n], &.{ rows, k });
        const w = try sc.floats(wts[0..n], &.{ rows, k });
        const op: dsv41_model.SharedOp = .{ .layer = .{ .m = m, .li = li } };
        const bound = try dsv41_model.stqRouted(&sc, c.expert_dec.?, @intFromFloat(c.swiglu_limit), rb, x, slots, w, op);
        m.layers[li].shared_lane = null;
        const read = try dsv41_model.stqRouted(&sc, c.expert_dec.?, @intFromFloat(c.swiglu_limit), rb, x, slots, w, op);
        m.layers[li].shared_lane = lane;
        try sameBits(bound, read, &sc);
    }
}

fn forwardIds(xfm: *Trunk, ctx: *trunk_mod.ForwardCtx, ids: []const u32) !mlx.mlx_array {
    var ids32: [64]i32 = undefined;
    for (ids, ids32[0..ids.len]) |t, *o| o.* = @intCast(t);
    const shape = [_]c_int{ 1, @intCast(ids.len) };
    const arr = mlx.mlx_array_new_data(&ids32, &shape, 2, .int32);
    defer _ = mlx.mlx_array_free(arr);
    return xfm.forwardWith(ctx, arr);
}

/// Queries of six single-token forwards after a 30-token prompt, every expert resident.
fn decodeTokens(env: [:0]const u8) !?[6]u64 {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const dir = std.mem.span(std.c.getenv(env) orelse return null);
    var fx = try dsv41_model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    var sc = Scope.plain(a, mlx.gpuStream());
    defer sc.deinit();
    const ids = try dsv41_model.tokens(&sc, &fx);
    defer a.free(ids);
    var config = try model_mod.parseConfig(io, a, dir);
    defer config.deinit(a);
    config.expert_streaming = true;
    config.expert_cache_bytes = 64 << 20;
    config.expert_source_dir = try a.dupe(u8, dir);
    var weights = try model_mod.loadWeightsForConfig(io, a, dir, &config, false);
    defer weights.deinit();
    var xfm = try Trunk.init(io, a, config, &weights);
    defer xfm.deinit();
    var ctx = xfm.defaultCtx();
    _ = mlx.mlx_array_free(try forwardIds(&xfm, &ctx, ids[0..30]));
    var out: [6]u64 = undefined;
    for (ids[30..36], &out) |t, *n| {
        const before = mlx.query_count;
        _ = mlx.mlx_array_free(try forwardIds(&xfm, &ctx, &.{t}));
        n.* = mlx.query_count - before;
    }
    return out;
}

test "dsv41 queries: a decode token (DSV41_TINY, DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    var ran = false;
    for ([_][:0]const u8{ "DSV41_TINY", "DSV41_STQ_TINY" }) |env| {
        const got = (try decodeTokens(env)) orelse continue;
        ran = true;
        // The two counts alternate with the compressed-attention cycle.
        const want: [6]u64 = if (std.mem.eql(u8, env, "DSV41_TINY")) .{ 1169, 1218, 1169, 1218, 1169, 1218 } else .{ 2373, 2429, 2373, 2429, 2373, 2429 };
        try testing.expectEqual(want, got);
    }
    if (!ran) return error.SkipZigTest;
}
