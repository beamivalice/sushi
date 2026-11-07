//! DeepSeek-V4.1-Flash STQ kernels at the pack's own geometry, synthetic: the rate-group chain at 5120 -> 2304 -> 5120
//! over PAIR1 groups at n24/26/32/40 and the mixed class (gate/up n32, down n26), and the shared expert as a one-expert
//! chain, each scored against a dense f64 product of the host tile decode, the f32 truth. The real-pack cases are
//! `src/dsv41/dsv41_exl3_real_test.zig`.
const std = @import("std");
const mlx = @import("mlx_host").mlx;
const exl3 = @import("expert_exl3.zig");
const kernels = @import("expert_exl3_kernels.zig");
const rate_groups = @import("rate_groups.zig");
const root = @import("root.zig");
const bench = @import("bench_util.zig");
const truth = @import("f32_truth.zig");

const relRms = truth.relRms;
const toF32 = truth.toF32;
const activations = truth.activations;
const reshaped = truth.reshaped;
const routedErr = truth.routedErr;
const Dense = truth.Dense;
const LIMIT = truth.LIMIT;
const PAIR1: exl3.Decode = .{ .codebook = .pair1, .window = .w16 };

const Group = struct { n: [3]u32, experts: u32 };

// A 16-row batch runs the decode arm the 5-row verify block does.
const GEOMETRY_ROW_COUNTS = [_]usize{ 1, 5, 33 };

test "dsv41 geometry: the grouped chain at 5120x2304 over n24, n26, n32, n40 and a mixed class holds the f32-truth bar" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    kernels.setDecodeParams(PAIR1);
    defer kernels.setDecodeParams(.mul1);
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0xd54e);
    const rnd = prng.random();
    const groups = [_]Group{
        .{ .n = .{ 40, 40, 40 }, .experts = 1 },
        .{ .n = .{ 32, 32, 32 }, .experts = 1 },
        .{ .n = .{ 26, 26, 26 }, .experts = 1 },
        .{ .n = .{ 24, 24, 24 }, .experts = 1 },
        .{ .n = .{ 32, 32, 26 }, .experts = 1 },
    };
    const hidden: usize = 5120;
    const inter: usize = 2304;
    var banks: [groups.len]root.Bank = undefined;
    var dense: [groups.len]Dense = undefined;
    for (groups, 0..) |g, i| {
        banks[i] = try bench.makeBank(rnd, hidden, inter, g.n, g.experts, 0.0126, 1.03);
        dense[i] = try Dense.of(alloc, banks[i], 0, hidden, inter, g.n, PAIR1);
    }
    var rb = try rate_groups.Ragged.init(s, &banks);
    defer rb.deinit();
    defer for (&banks) |*b| bench.freeBank(b);
    for (GEOMETRY_ROW_COUNTS) |rows| {
        const err = try routedErr(alloc, s, &rb, &dense, rnd, hidden, inter, rows, 3, 4);
        if (err >= 0.005) std.debug.print("dsv41 geometry rows {d}: relRms {d}\n", .{ rows, err });
        try t.expect(err < 0.005);
    }
}

test "dsv41 geometry: the shared expert as a one-expert fused chain gives a row the same bytes at every decode width and holds the f32-truth bar" {
    try sharedChainCase(5120, 2304);
    // The tiny serving pack's geometry.
    try sharedChainCase(256, 128);
}

fn sharedChainCase(hidden: usize, inter: usize) !void {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const dec = exl3.Decode.tabled(.spec1, .w15);
    defer kernels.setDecodeParams(.mul1);
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x5a4e);
    const rnd = prng.random();
    var bank = try bench.makeBank(rnd, hidden, inter, .{ 64, 64, 64 }, 1, 0.0126, 1.03);
    defer bench.freeBank(&bank);
    const dense = try Dense.of(alloc, bank, 0, hidden, inter, .{ 64, 64, 64 }, dec);
    const act = try activations(alloc, rnd, 16, hidden, 4);
    defer _ = mlx.mlx_array_free(act.x);
    const zeros: [16]u32 = @splat(0);
    const unit: [16]f32 = @splat(1);
    for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dt| {
        var first: ?[]f32 = null;
        for ([_]usize{ 1, 2, 5, 16 }) |rows| {
            var xs = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(xs);
            try mlx.check(mlx.mlx_slice(&xs, act.x, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ @intCast(rows), @intCast(hidden) }, 2, &[_]c_int{ 1, 1 }, 2, s));
            var xr = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(xr);
            try mlx.check(mlx.mlx_astype(&xr, xs, dt, s));
            const x3 = try reshaped(s, xr, &.{ 1, @intCast(rows), @intCast(hidden) });
            defer _ = mlx.mlx_array_free(x3);
            const ids = mlx.mlx_array_new_data(&zeros, &[_]c_int{ 1, @intCast(rows), 1 }, 3, .uint32);
            defer _ = mlx.mlx_array_free(ids);
            const sc = mlx.mlx_array_new_data(&unit, &[_]c_int{ 1, @intCast(rows), 1 }, 3, .float32);
            defer _ = mlx.mlx_array_free(sc);
            const y = try root.moeClamped(s, x3, bank, ids, sc, dec, LIMIT);
            defer _ = mlx.mlx_array_free(y);
            var cy = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(cy);
            const got = try toF32(s, y, &cy);
            for (got) |v| try t.expect(std.math.isFinite(v));
            const want = try alloc.alloc(f64, got.len);
            const tmp = try alloc.alloc(f64, hidden);
            for (0..rows) |r| {
                try dense.run(alloc, act.vals[r * hidden ..][0..hidden], hidden, inter, LIMIT, tmp);
                @memcpy(want[r * hidden ..][0..hidden], tmp);
            }
            const err = relRms(got, want);
            if (err >= 0.005) std.debug.print("dsv41 shared chain rows {d}: relRms {d}\n", .{ rows, err });
            try t.expect(err < 0.005);
            if (first) |f| {
                if (!std.mem.eql(u8, std.mem.sliceAsBytes(f[0..hidden]), std.mem.sliceAsBytes(got[0..hidden]))) std.debug.print("dsv41 shared chain {s}: row 0 at {d} rows differs from its serial row\n", .{ @tagName(dt), rows });
                try t.expectEqualSlices(u8, std.mem.sliceAsBytes(f[0..hidden]), std.mem.sliceAsBytes(got[0..hidden]));
            } else first = try alloc.dupe(f32, got);
        }
    }
}
