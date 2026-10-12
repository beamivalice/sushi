//! A6 prefill expansion plus dense BF16 GEMM; no persistent weight copy.
const std = @import("std");
const mlx = @import("mlx.zig");
const Ops = @import("glm5_model.zig").Ops;
const Arr = mlx.mlx_array;

pub const expanded_weight_bytes: usize = 64 * 1024 * 1024;
// K -> N of every A6 g128 trunk projection served by the expansion at 2048 rows. A layer expands at most its four
// KDA banks (or the MLA projections, 140 MiB), which the bill covers. Left on qmm: the dense MLP (12288 -> 4096 is not
// byte-equal; 4096 -> 12288 would raise the bill) and the shared expert (its transient would shrink the default chunk).
const shapes = [_][2]c_int{ .{ 4096, 8192 }, .{ 8192, 4096 }, .{ 4096, 1536 }, .{ 1536, 16384 }, .{ 4096, 512 }, .{ 1536, 4096 } };
const banks_per_layer = 4;
var dispatches: usize = 0;
pub fn dispatchCount() usize {
    return dispatches;
}
pub fn resetDispatchCount() void {
    dispatches = 0;
}
fn geometry(xs: []const c_int, ws: []const c_int, ss: []const c_int, bs: []const c_int, dtypes: [4]mlx.mlx_dtype) bool {
    if (xs.len != 3 or xs[0] != 1 or xs[1] != 2048 or ws.len != 2) return false;
    const k = xs[2];
    const n = ws[0];
    const known = for (shapes) |shape| {
        if (shape[0] == k and shape[1] == n) break true;
    } else false;
    if (!known or ws[1] != @divExact(k * 3, 16)) return false;
    const scale_shape = [_]c_int{ n, @divExact(k, 128) };
    return std.mem.eql(c_int, &scale_shape, ss) and std.mem.eql(c_int, &scale_shape, bs) and
        std.mem.eql(mlx.mlx_dtype, &.{ .bfloat16, .uint32, .bfloat16, .bfloat16 }, &dtypes);
}
pub fn tryPrefill(ops: *Ops, x: Arr, w: Arr, scales: Arr, biases: Arr) !?Arr {
    if (!mlx.streamIsGpu(ops.s) or !@import("glm5_model.zig").naxArms()) return null;
    const arrays = [_]Arr{ x, w, scales, biases };
    for (arrays) |v| if (v.ctx == null) return null;
    var dtypes: [4]mlx.mlx_dtype = undefined;
    for (arrays, 0..) |v, i| dtypes[i] = mlx.mlx_array_dtype(v);
    if (!geometry(mlx.getShape(x), mlx.getShape(w), mlx.getShape(scales), mlx.getShape(biases), dtypes)) return null;
    const weights = try ops.dequant(w, scales, biases);
    const result = try ops.binary(.mm, x, try ops.transpose(weights, &.{ 1, 0 }));
    dispatches += 1;
    return result;
}
fn bill(pending_layers: usize) !usize {
    return std.math.mul(usize, expanded_weight_bytes * banks_per_layer, pending_layers);
}
pub fn transientBudget(chunk: usize, pending_layers: usize) !usize {
    return if (chunk == 2048 and @import("glm5_model.zig").naxArms()) bill(pending_layers) else 0;
}

const Input = struct { x: Arr, w: Arr, scales: Arr, biases: Arr };
fn fixture(ops: *Ops, k: c_int, n: c_int, seed: u64) !Input {
    const key = try ops.slot();
    try mlx.check(mlx.mlx_random_key(key, seed));
    const x = try ops.slot();
    try mlx.check(mlx.mlx_random_normal(x, &.{ 1, 2048, k }, 3, .bfloat16, 0, 0.2, key.*, ops.s));
    const w = try ops.slot();
    try mlx.check(mlx.mlx_random_bits(w, &.{ n, @divExact(k * 3, 16) }, 2, 4, key.*, ops.s));
    const scale = try ops.slot();
    try mlx.check(mlx.mlx_random_uniform(scale, try ops.scalar(0.001, .bfloat16), try ops.scalar(0.01, .bfloat16), &.{ n, @divExact(k, 128) }, 2, .bfloat16, key.*, ops.s));
    const bias = try ops.slot();
    try mlx.check(mlx.mlx_random_uniform(bias, try ops.scalar(-1.2, .bfloat16), try ops.scalar(-0.1, .bfloat16), &.{ n, @divExact(k, 128) }, 2, .bfloat16, key.*, ops.s));
    for ([_]Arr{ x.*, w.*, scale.*, bias.* }) |a| try mlx.check(mlx.mlx_array_eval(a));
    return .{ .x = x.*, .w = w.*, .scales = scale.*, .biases = bias.* };
}
fn expectBits(reference: Arr, candidate: Arr) !void {
    try mlx.check(mlx.mlx_array_eval(reference));
    try mlx.check(mlx.mlx_array_eval(candidate));
    try std.testing.expectEqualSlices(c_int, mlx.getShape(reference), mlx.getShape(candidate));
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(reference));
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(candidate));
    const count = mlx.mlx_array_size(reference);
    try std.testing.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(reference).?[0..count], mlx.mlx_array_data_bfloat16(candidate).?[0..count]);
}

test "GLM A6 dense once guard retains decode small tensor and affine8 paths" {
    const dtype = [_]mlx.mlx_dtype{ .bfloat16, .uint32, .bfloat16, .bfloat16 };
    try std.testing.expect(geometry(&.{ 1, 2048, 4096 }, &.{ 8192, 768 }, &.{ 8192, 32 }, &.{ 8192, 32 }, dtype));
    try std.testing.expect(geometry(&.{ 1, 2048, 8192 }, &.{ 4096, 1536 }, &.{ 4096, 64 }, &.{ 4096, 64 }, dtype));
    for (shapes) |shape| try std.testing.expect(geometry(&.{ 1, 2048, shape[0] }, &.{ shape[1], @divExact(shape[0] * 3, 16) }, &.{ shape[1], @divExact(shape[0], 128) }, &.{ shape[1], @divExact(shape[0], 128) }, dtype));
    try std.testing.expect(!geometry(&.{ 1, 2048, 12288 }, &.{ 4096, 2304 }, &.{ 4096, 96 }, &.{ 4096, 96 }, dtype));
    try std.testing.expect(!geometry(&.{ 1, 2048, 4096 }, &.{ 2048, 768 }, &.{ 2048, 32 }, &.{ 2048, 32 }, dtype));
    try std.testing.expect(!geometry(&.{ 1, 2048, 4096 }, &.{ 12288, 768 }, &.{ 12288, 32 }, &.{ 12288, 32 }, dtype));
    for ([_]c_int{ 1, 3, 512, 1024, 2047, 2049 }) |t| try std.testing.expect(!geometry(&.{ 1, t, 4096 }, &.{ 8192, 768 }, &.{ 8192, 32 }, &.{ 8192, 32 }, dtype));
    try std.testing.expect(!geometry(&.{ 2, 2048, 4096 }, &.{ 8192, 768 }, &.{ 8192, 32 }, &.{ 8192, 32 }, dtype));
    try std.testing.expect(!geometry(&.{ 1, 2048, 4096 }, &.{ 8192, 1024 }, &.{ 8192, 32 }, &.{ 8192, 32 }, dtype));
    try std.testing.expect(!geometry(&.{ 1, 2048, 4096 }, &.{ 128, 768 }, &.{ 128, 32 }, &.{ 128, 32 }, dtype));
    try std.testing.expect(!geometry(&.{ 1, 2048, 4096 }, &.{ 8192, 768 }, &.{ 8192, 64 }, &.{ 8192, 64 }, dtype));
    try std.testing.expect(!geometry(&.{ 1, 2048, 4096 }, &.{ 8192, 768 }, &.{ 8192, 32 }, &.{ 8192, 32 }, .{ .float32, .uint32, .bfloat16, .bfloat16 }));
}

test "GLM A6 dense once transient bill covers pending four projection layers" {
    try std.testing.expectEqual(@as(usize, 268435456), try bill(1));
    try std.testing.expectEqual(@as(usize, 536870912), try bill(2));
    try std.testing.expectEqual(@as(usize, 2147483648), try bill(8));
    var mla_layer: usize = 0;
    for (shapes, 0..) |shape, i| {
        const bytes = @as(usize, @intCast(shape[0])) * @as(usize, @intCast(shape[1])) * 2;
        try std.testing.expect(bytes <= expanded_weight_bytes);
        if (i >= 2) mla_layer += bytes;
    }
    try std.testing.expect(mla_layer + expanded_weight_bytes <= expanded_weight_bytes * banks_per_layer);
    try std.testing.expectError(error.Overflow, bill(std.math.maxInt(usize)));
    try std.testing.expectEqual(@as(usize, 0), try transientBudget(512, 2));
}

test "GLM A6 dense once Linear keeps native bits and declines decode rows" {
    if (!@import("glm5_model.zig").naxArms()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    for (shapes, 0..) |kn, shape| {
        var ops = Ops{ .s = s };
        defer ops.deinit();
        const k = kn[0];
        const n = kn[1];
        const input = try fixture(&ops, k, n, @intCast(1031 + shape));
        const linear = @import("glm5_model.zig").Linear{ .w = input.w, .scales = input.scales, .biases = input.biases, .input = k, .output = n };
        resetDispatchCount();
        const actual = try linear.apply(&ops, input.x);
        try expectBits(try ops.qmm(input.x, input.w, input.scales, input.biases, true), actual);
        try std.testing.expectEqual(@as(usize, 1), dispatchCount());
        const decode = try ops.slice(input.x, 1, 0, 3);
        const short_actual = try linear.apply(&ops, decode);
        try expectBits(try ops.qmm(decode, input.w, input.scales, input.biases, true), short_actual);
        try std.testing.expectEqual(@as(usize, 1), dispatchCount());
    }
    resetDispatchCount();
}

test "GLM A6 dense once trunk shapes: bytes and time against qmm (SUSHI_GLM_A6_SHAPES_UBENCH)" {
    if (!@import("transformer.zig").diagEnvOn("SUSHI_GLM_A6_SHAPES_UBENCH") or !@import("glm5_model.zig").naxArms()) return error.SkipZigTest;
    const io = std.Io.Threaded.global_single_threaded.io();
    const candidates = [_][2]c_int{ .{ 4096, 8192 }, .{ 8192, 4096 }, .{ 4096, 2048 }, .{ 2048, 4096 }, .{ 4096, 12288 }, .{ 12288, 4096 }, .{ 4096, 1536 }, .{ 1536, 16384 }, .{ 4096, 512 }, .{ 1536, 4096 } };
    const repeats = 6;
    for (candidates, 0..) |shape, index| {
        const k = shape[0];
        const n = shape[1];
        var fx = Ops{ .s = mlx.gpuStream() };
        defer fx.deinit();
        const input = try fixture(&fx, k, n, @intCast(7000 + index));
        var times: [2][9]f64 = undefined;
        var differ: usize = 0;
        for (0..11) |rep| for (0..2) |arm| {
            var ops = Ops{ .s = mlx.gpuStream() };
            defer ops.deinit();
            const sw = @import("io_util.zig").Stopwatch.init(io);
            var outs: [repeats]Arr = undefined;
            for (&outs) |*out| out.* = if (arm == 0) try ops.qmm(input.x, input.w, input.scales, input.biases, true) else try ops.binary(.mm, input.x, try ops.transpose(try ops.dequant(input.w, input.scales, input.biases), &.{ 1, 0 }));
            for (outs) |out| try mlx.check(mlx.mlx_array_eval(out));
            if (rep >= 2) times[arm][rep - 2] = @as(f64, @floatFromInt(sw.read())) / 1e6 / repeats;
            if (rep == 0 and arm == 1) {
                const reference = try ops.qmm(input.x, input.w, input.scales, input.biases, true);
                try mlx.check(mlx.mlx_array_eval(reference));
                const count = mlx.mlx_array_size(reference);
                const want = mlx.mlx_array_data_bfloat16(reference).?[0..count];
                const got = mlx.mlx_array_data_bfloat16(outs[0]).?[0..count];
                differ = 0;
                for (want, got) |a, b| differ += @intFromBool(a != b);
            }
        };
        for (&times) |*t| std.mem.sort(f64, t, {}, std.sort.asc(f64));
        std.debug.print("[a6-shapes] {d}->{d}: qmm {d:.3} ms, dequant+gemm {d:.3} ms ({d:.1}%), {d} values differ\n", .{ k, n, times[0][4], times[1][4], (times[1][4] / times[0][4] - 1) * 100, differ });
    }
    var fx = Ops{ .s = mlx.gpuStream() };
    defer fx.deinit();
    const gate = try fixture(&fx, 4096, 2048, 7100);
    const up = try fixture(&fx, 4096, 2048, 7101);
    var times: [2][9]f64 = undefined;
    var differ: usize = 0;
    for (0..11) |rep| for (0..2) |arm| {
        var ops = Ops{ .s = mlx.gpuStream() };
        defer ops.deinit();
        const sw = @import("io_util.zig").Stopwatch.init(io);
        var outs: [repeats][2]Arr = undefined;
        for (&outs) |*out| {
            if (arm == 0) {
                out.* = .{ try ops.binary(.mm, gate.x, try ops.transpose(try ops.dequant(gate.w, gate.scales, gate.biases), &.{ 1, 0 })), try ops.binary(.mm, gate.x, try ops.transpose(try ops.dequant(up.w, up.scales, up.biases), &.{ 1, 0 })) };
            } else {
                const w = try ops.concat(&.{ gate.w, up.w }, 0);
                const sc = try ops.concat(&.{ gate.scales, up.scales }, 0);
                const bi = try ops.concat(&.{ gate.biases, up.biases }, 0);
                const both = try ops.binary(.mm, gate.x, try ops.transpose(try ops.dequant(w, sc, bi), &.{ 1, 0 }));
                out.* = .{ try ops.slice(both, 2, 0, 2048), try ops.slice(both, 2, 2048, 4096) };
            }
        }
        for (outs) |out| for (out) |o| try mlx.check(mlx.mlx_array_eval(o));
        if (rep >= 2) times[arm][rep - 2] = @as(f64, @floatFromInt(sw.read())) / 1e6 / repeats;
        if (rep == 0 and arm == 1) {
            const reference = [2]Arr{ try ops.qmm(gate.x, gate.w, gate.scales, gate.biases, true), try ops.qmm(gate.x, up.w, up.scales, up.biases, true) };
            differ = 0;
            for (reference, outs[0]) |r, o| {
                const flat = try ops.contiguous(o);
                try mlx.check(mlx.mlx_array_eval(r));
                try mlx.check(mlx.mlx_array_eval(flat));
                const count = mlx.mlx_array_size(r);
                for (mlx.mlx_array_data_bfloat16(r).?[0..count], mlx.mlx_array_data_bfloat16(flat).?[0..count]) |a, b| differ += @intFromBool(a != b);
            }
        }
    };
    for (&times) |*t| std.mem.sort(f64, t, {}, std.sort.asc(f64));
    std.debug.print("[a6-shapes] shared gate+up: two banks {d:.3} ms, one 4096-wide bank {d:.3} ms ({d:.1}%), {d} values differ\n", .{ times[0][4], times[1][4], (times[1][4] / times[0][4] - 1) * 100, differ });
}
