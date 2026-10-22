//! MLA verify projections as one broadcast QMM per bank.
const std = @import("std");
const mlx = @import("mlx.zig");
const Ops = @import("glm5_model.zig").Ops;
const Arr = mlx.mlx_array;
pub const Direction = enum { query, value };
pub const Input = struct { x: Arr, w: Arr, scales: Arr, biases: Arr };
pub fn run(ops: *Ops, in: Input, dir: Direction) !?Arr {
    if (!mlx.streamIsGpu(ops.s) or !@import("glm5_model.zig").naxArms()) return null;
    for ([_]Arr{ in.x, in.w, in.scales, in.biases }) |a| if (a.ctx == null) return null;
    const xs = mlx.getShape(in.x);
    if (xs.len != 4 or xs[0] < 2 or xs[0] > 8 or
        !std.mem.eql(c_int, &.{ xs[0], 64, 1, if (dir == .query) @as(c_int, 256) else 512 }, xs) or
        mlx.mlx_array_dtype(in.x) != .bfloat16 or mlx.mlx_array_dtype(in.w) != .uint32 or
        mlx.mlx_array_dtype(in.scales) != .bfloat16 or mlx.mlx_array_dtype(in.biases) != .bfloat16 or
        !(std.mem.eql(c_int, &.{ 64, 256, 96 }, mlx.getShape(in.w)) or std.mem.eql(c_int, &.{ 64, 256, 80 }, mlx.getShape(in.w))) or !std.mem.eql(c_int, &.{ 64, 256, 4 }, mlx.getShape(in.scales)) or
        !std.mem.eql(c_int, mlx.getShape(in.scales), mlx.getShape(in.biases))) return null;
    if (dir == .value and xs[0] == 3) {
        const linear = @import("glm5_model.zig").Linear{ .w = in.w, .scales = in.scales, .biases = in.biases, .input = 512, .output = 256 };
        if (try @import("glm5_dflash_a6_hoist.zig").project(ops.s, in.x, linear)) |out| return try ops.own(out);
    }
    return try ops.qmm(in.x, in.w, in.scales, in.biases, dir == .value);
}
fn fixture(ops: *Ops, dir: Direction, rows: c_int, bits: c_int) !Input {
    const key = try ops.slot();
    try mlx.check(mlx.mlx_random_key(key, if (dir == .query) 1903 else 1913));
    const x = try ops.slot();
    try mlx.check(mlx.mlx_random_normal(x, &.{ rows, 64, 1, if (dir == .query) @as(c_int, 256) else 512 }, 4, .bfloat16, 0, 0.2, key.*, ops.s));
    const w = try ops.slot();
    try mlx.check(mlx.mlx_random_bits(w, &.{ 64, 256, 16 * bits }, 3, 4, key.*, ops.s));
    const sc = try ops.slot();
    try mlx.check(mlx.mlx_random_uniform(sc, try ops.scalar(0.001, .bfloat16), try ops.scalar(0.01, .bfloat16), &.{ 64, 256, 4 }, 3, .bfloat16, key.*, ops.s));
    const bi = try ops.slot();
    try mlx.check(mlx.mlx_random_uniform(bi, try ops.scalar(-0.3, .bfloat16), try ops.scalar(0.1, .bfloat16), &.{ 64, 256, 4 }, 3, .bfloat16, key.*, ops.s));
    const values = mlx.mlx_vector_array_new_data(&.{ x.*, w.*, sc.*, bi.* }, 4);
    defer _ = mlx.mlx_vector_array_free(values);
    try mlx.check(mlx.mlx_eval(values));
    return .{ .x = x.*, .w = w.*, .scales = sc.*, .biases = bi.* };
}

test "GLM MLA verify batch matches each serial row projection" {
    for ([_]c_int{ 6, 5 }) |bits| for ([_]Direction{ .query, .value }) |dir| for ([_]c_int{ 2, 3, 4, 5, 6, 7, 8 }) |rows| {
        var ops = Ops{ .s = mlx.gpuStream() };
        defer ops.deinit();
        const in = try fixture(&ops, dir, rows, bits);
        const hoist = @import("glm5_dflash_a6_hoist.zig");
        const before = hoist.dispatchCount();
        const got = (try run(&ops, in, dir)) orelse if (@import("glm5_model.zig").naxArms()) return error.TestExpectedMlaVerifyBatch else return error.SkipZigTest;
        try std.testing.expectEqual(before + @as(usize, if (dir == .value and rows == 3) 1 else 0), hoist.dispatchCount());
        try mlx.check(mlx.mlx_array_eval(got));
        for (0..@intCast(rows)) |row| {
            const r: c_int = @intCast(row);
            const want = try ops.qmm(try ops.slice(in.x, 0, r, r + 1), in.w, in.scales, in.biases, dir == .value);
            const have = try ops.contiguous(try ops.slice(got, 0, r, r + 1));
            try mlx.check(mlx.mlx_array_eval(want));
            try mlx.check(mlx.mlx_array_eval(have));
            const n = mlx.mlx_array_size(want);
            try std.testing.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(want).?[0..n], mlx.mlx_array_data_bfloat16(have).?[0..n]);
        }
    };
}
