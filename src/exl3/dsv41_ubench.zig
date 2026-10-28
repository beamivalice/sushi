//! DeepSeek-V4.1-Flash's shared expert beside the routed chain, timed until it moves into `sushi-ubench` with the
//! other shared-expert folds. The routed experts' bench is `ubench.zig`.
const std = @import("std");
const mlx = @import("mlx_host").mlx;
const exl3 = @import("expert_exl3.zig");
const kernels = @import("expert_exl3_kernels.zig");
const rate_groups = @import("rate_groups.zig");
const root = @import("root.zig");
const bench = @import("bench_util.zig");
const ub = @import("ubench.zig");

const HIDDEN = ub.HIDDEN;
const INTER = ub.INTER;
const LIMIT = ub.LIMIT;
const ROUTE_SETS = ub.ROUTE_SETS;
const LAYOUTS = ub.LAYOUTS;
const Ctx = ub.Ctx;
const makeRoutes = ub.makeRoutes;

fn medianOf(v: []f64) f64 {
    std.mem.sort(f64, v, {}, std.sort.asc(f64));
    return v[v.len / 2];
}

const SharedCtx = struct {
    base: Ctx,
    bank: root.Bank,
    dec: exl3.Decode,
    zeros: mlx.mlx_array,
    ones: mlx.mlx_array,
    arm: enum { separate, routed_only, shared_only, fused, fused_first },
};

fn sharedStep(c: SharedCtx, s: mlx.mlx_stream, i: usize, x: mlx.mlx_array) anyerror!mlx.mlx_array {
    const rows = c.base.rows;
    var x3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x3);
    try mlx.check(mlx.mlx_reshape(&x3, x, &[_]c_int{ 1, rows, @intCast(HIDDEN) }, 3, s));
    var sy3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sy3);
    if (c.arm == .separate or c.arm == .shared_only) {
        kernels.setDecodeParams(c.dec);
        sy3 = try root.moeClamped(s, x3, c.bank, c.zeros, c.ones, c.dec, LIMIT);
        if (c.arm == .shared_only) {
            const out = sy3;
            sy3 = mlx.mlx_array_new();
            return out;
        }
    }
    if (c.arm == .fused or c.arm == .fused_first) {
        kernels.setDecodeParams(.{ .codebook = .pair1, .window = .w16 });
        rate_groups.fused_shared_first = c.arm == .fused_first;
        const se: rate_groups.SharedExpert = .{ .bank = c.bank, .dec = c.dec };
        return (try rate_groups.moeDecodeClampedSharedFused(s, x, c.base.rb, c.base.routes[i % ROUTE_SETS], c.base.scores, c.base.topk, .float32, se, LIMIT)) orelse error.SharedFuseDeclined;
    }
    var sy = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sy);
    if (c.arm == .separate) try mlx.check(mlx.mlx_reshape(&sy, sy3, &[_]c_int{ rows, @intCast(HIDDEN) }, 2, s));
    kernels.setDecodeParams(.{ .codebook = .pair1, .window = .w16 });
    return rate_groups.moeDecodeClamped(s, x, c.base.rb, c.base.routes[i % ROUTE_SETS], c.base.scores, c.base.topk, .float32, if (c.arm == .separate) sy else null, null, LIMIT);
}

test "dsv41 shared expert beside the routed chain, interleaved (SUSHI_DSV41_SHARED_UBENCH=1)" {
    if (!bench.diagOn("SUSHI_DSV41_SHARED_UBENCH")) return error.SkipZigTest;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    defer kernels.setDecodeParams(.mul1);
    var prng = std.Random.DefaultPrng.init(0xd54e);
    const rnd = prng.random();
    kernels.setDecodeParams(.{ .codebook = .pair1, .window = .w16 });
    const lay = LAYOUTS[bench.envUsize("SUSHI_DSV41_SHARED_LAYOUT", 0)];
    var banks: [8]root.Bank = undefined;
    for (lay.groups, 0..) |g, i| banks[i] = try bench.makeBank(rnd, HIDDEN, INTER, g.n, g.experts, 0.02, 0.9);
    var rb = try rate_groups.Ragged.init(s, banks[0..lay.groups.len]);
    defer rb.deinit();
    const st = rb.storage();
    const vec = mlx.mlx_vector_array_new_data(&st, st.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_eval(vec));
    for (banks[0..lay.groups.len]) |*b| bench.freeBank(b);
    var sbank = try bench.makeBank(rnd, HIDDEN, INTER, .{ 64, 64, 64 }, 1, 0.02, 0.9);
    defer bench.freeBank(&sbank);
    const sdec = exl3.Decode.tabled(.spec1, .w15);
    for ([_]usize{ 1, 2, 3, 4, 6, 8 }) |rows| {
        var routes: [ROUTE_SETS]mlx.mlx_array = undefined;
        _ = makeRoutes(rnd, &rb, rows, lay.topk, .uniform, &routes);
        defer for (routes) |a| {
            _ = mlx.mlx_array_free(a);
        };
        const xh = try std.heap.page_allocator.alloc(u16, rows * HIDDEN);
        defer std.heap.page_allocator.free(xh);
        for (xh) |*v| v.* = @truncate(@as(u32, @bitCast((rnd.float(f32) * 2 - 1) * 0.5)) >> 16);
        const x0 = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(rows), @intCast(HIDDEN) }, 2, .bfloat16);
        defer _ = mlx.mlx_array_free(x0);
        var scv: [256 * 6]f32 = undefined;
        for (&scv) |*v| v.* = 0.05 + rnd.float(f32) * 0.2;
        const scores = mlx.mlx_array_new_data(&scv, &[_]c_int{@intCast(rows * lay.topk)}, 1, .float32);
        defer _ = mlx.mlx_array_free(scores);
        var zi: [16]i32 = @splat(0);
        var one: [16]f32 = @splat(1);
        const zeros = mlx.mlx_array_new_data(&zi, &[_]c_int{ 1, @intCast(rows), 1 }, 3, .int32);
        defer _ = mlx.mlx_array_free(zeros);
        const ones = mlx.mlx_array_new_data(&one, &[_]c_int{ 1, @intCast(rows), 1 }, 3, .float32);
        defer _ = mlx.mlx_array_free(ones);
        const base: Ctx = .{ .rb = &rb, .routes = &routes, .scores = scores, .rows = @intCast(rows), .topk = @intCast(lay.topk), .prefill = false };
        var arms: [5]SharedCtx = undefined;
        for (&arms, [_]@TypeOf(arms[0].arm){ .separate, .routed_only, .shared_only, .fused, .fused_first }) |*a, k| a.* = .{ .base = base, .bank = sbank, .dec = sdec, .zeros = zeros, .ones = ones, .arm = k };
        const steps = bench.envUsize("SUSHI_UBENCH_STEPS", 16);
        const rounds = bench.envUsize("SUSHI_UBENCH_ROUNDS", 9);
        var us: [5][]f64 = undefined;
        for (&us) |*u| u.* = try std.heap.page_allocator.alloc(f64, rounds);
        defer for (us) |u| std.heap.page_allocator.free(u);
        var counts: [5]u32 = undefined;
        for (arms, 0..) |c, ai| {
            kernels.resetFusedDispatchCount();
            const y = try sharedStep(c, s, 0, x0);
            try mlx.check(mlx.mlx_array_eval(y));
            _ = mlx.mlx_array_free(y);
            counts[ai] = kernels.fusedDispatchCount();
        }
        for (0..rounds + 1) |round| {
            const fl = try bench.chainNs(s, x0, steps, {}, bench.linkOnly);
            for (arms, 0..) |c, ai| {
                const ns = try bench.chainNs(s, x0, steps, c, sharedStep);
                if (round > 0) us[ai][round - 1] = (@as(f64, @floatFromInt(ns)) - @as(f64, @floatFromInt(fl))) / 1e3 / @as(f64, @floatFromInt(steps));
            }
        }
        var med: [5]f64 = undefined;
        for (&med, us) |*m, u| m.* = medianOf(u);
        std.debug.print("dsv41-shared {s} rows {d} (us/layer, median of {d}): separate {d:.1}, routed only {d:.1}, shared only {d:.1}, fused {d:.1}, fused shared-first {d:.1}; fused/separate {d:.3}x, ceiling {d:.1}; counted kernels {d}/{d}/{d}/{d}/{d}\n", .{ lay.name, rows, rounds, med[0], med[1], med[2], med[3], med[4], @min(med[3], med[4]) / med[0], med[0] - med[1], counts[0], counts[1], counts[2], counts[3], counts[4] });
    }
}
