//! Head-batched MLA prefill projections over the stored banks, affine or BF16; no weight or cache precision changes.
const std = @import("std");
const mlx = @import("mlx.zig");
const Ops = @import("glm5_model.zig").Ops;
const Arr = mlx.mlx_array;
pub const Direction = enum { query, value };
pub const Input = struct { x: Arr, w: Arr, scales: Arr = .{ .ctx = null }, biases: Arr = .{ .ctx = null } };
var dispatches: usize = 0;
pub fn dispatchCount() usize {
    return dispatches;
}

fn active() bool {
    const model = @import("glm5_model.zig");
    return model.naxArms() and !model.reference_numerics;
}

pub fn transientBudget(chunk: usize, pending_layers: usize) !usize {
    if (chunk < 128 or chunk > 2048 or !active()) return 0;
    // Input/output permutation copies total64H*(256+512+512+256)*2 bytes/row.
    return std.math.mul(usize, try std.math.mul(usize, chunk, 196608), pending_layers);
}

fn rowsOk(xs: []const c_int, dtype: mlx.mlx_dtype, dir: Direction) bool {
    const width: c_int = if (dir == .query) 256 else 512;
    return xs.len == 4 and xs[0] >= 128 and xs[0] <= 2048 and xs[1] == 64 and xs[2] == 1 and xs[3] == width and dtype == .bfloat16;
}

fn geometry(xs: []const c_int, ws: []const c_int, ss: []const c_int, bs: []const c_int, dtype: mlx.mlx_dtype, dir: Direction) bool {
    return rowsOk(xs, dtype, dir) and std.mem.eql(c_int, &.{ 64, 256, 96 }, ws) and std.mem.eql(c_int, &.{ 64, 256, 4 }, ss) and std.mem.eql(c_int, ss, bs);
}

/// Stored BF16 banks: `[64,256,512]` for both the key and the value side.
fn denseGeometry(xs: []const c_int, ws: []const c_int, x_dtype: mlx.mlx_dtype, w_dtype: mlx.mlx_dtype, dir: Direction) bool {
    return rowsOk(xs, x_dtype, dir) and w_dtype == .bfloat16 and std.mem.eql(c_int, &.{ 64, 256, 512 }, ws);
}

/// Affine or BF16 banks, each as stored; the BF16 arm is a batched GEMM.
pub fn run(ops: *Ops, input: Input, dir: Direction) !?Arr {
    if (!mlx.streamIsGpu(ops.s) or !active()) return null;
    if (input.x.ctx == null or input.w.ctx == null) return null;
    const affine = input.scales.ctx != null;
    if (affine) {
        if (input.biases.ctx == null or !geometry(mlx.getShape(input.x), mlx.getShape(input.w), mlx.getShape(input.scales), mlx.getShape(input.biases), mlx.mlx_array_dtype(input.x), dir) or
            mlx.mlx_array_dtype(input.w) != .uint32 or mlx.mlx_array_dtype(input.scales) != .bfloat16 or mlx.mlx_array_dtype(input.biases) != .bfloat16) return null;
    } else if (input.biases.ctx != null or !denseGeometry(mlx.getShape(input.x), mlx.getShape(input.w), mlx.mlx_array_dtype(input.x), mlx.mlx_array_dtype(input.w), dir)) return null;
    const xs = mlx.getShape(input.x);
    const out: c_int = if (dir == .query) 512 else 256;
    const heads = try ops.contiguous(try ops.reshape(try ops.transpose(input.x, &.{ 1, 0, 2, 3 }), &.{ 64, xs[0], xs[3] }));
    const y = if (affine)
        try ops.qmm(heads, input.w, input.scales, input.biases, dir == .value)
    else
        try ops.binary(.mm, heads, if (dir == .value) try ops.transpose(input.w, &.{ 0, 2, 1 }) else input.w);
    dispatches += 1;
    return try ops.reshape(try ops.contiguous(try ops.transpose(y, &.{ 1, 0, 2 })), &.{ xs[0], 64, 1, out });
}

test "GLM MLA headbatch prefill geometry guards original banks" {
    try std.testing.expect(geometry(&.{ 2048, 64, 1, 256 }, &.{ 64, 256, 96 }, &.{ 64, 256, 4 }, &.{ 64, 256, 4 }, .bfloat16, .query));
    try std.testing.expect(geometry(&.{ 128, 64, 1, 512 }, &.{ 64, 256, 96 }, &.{ 64, 256, 4 }, &.{ 64, 256, 4 }, .bfloat16, .value));
    for ([_]c_int{ 1, 3, 16, 32, 127, 2049 }) |t| try std.testing.expect(!geometry(&.{ t, 64, 1, 256 }, &.{ 64, 256, 96 }, &.{ 64, 256, 4 }, &.{ 64, 256, 4 }, .bfloat16, .query));
    try std.testing.expect(!geometry(&.{ 2048, 64, 1, 256 }, &.{ 64, 256, 128 }, &.{ 64, 256, 4 }, &.{ 64, 256, 4 }, .bfloat16, .query));
    try std.testing.expect(!geometry(&.{ 2048, 64, 1, 256 }, &.{ 64, 256, 96 }, &.{ 64, 256, 4 }, &.{ 64, 256, 4 }, .float32, .query));
}

test "GLM MLA headbatch prefill transient bill" {
    if (!@import("glm5_model.zig").naxArms()) return error.SkipZigTest;
    try std.testing.expectEqual(@as(usize, 402653184), try transientBudget(2048, 1));
    try std.testing.expectEqual(@as(usize, 805306368), try transientBudget(2048, 2));
    try std.testing.expectEqual(@as(usize, 0), try transientBudget(32, 2));
    try std.testing.expectError(error.Overflow, transientBudget(2048, std.math.maxInt(usize)));
}

fn bf16Normal(ops: *Ops, shape: []const c_int, seed: u64, deviation: f32) !Arr {
    const key = try ops.slot();
    try mlx.check(mlx.mlx_random_key(key, seed));
    const out = try ops.slot();
    try mlx.check(mlx.mlx_random_normal(out, shape.ptr, shape.len, .bfloat16, 0, deviation, key.*, ops.s));
    try mlx.check(mlx.mlx_array_eval(out.*));
    return out.*;
}

fn bf16Host(ops: *Ops, a: Arr) ![]const u16 {
    const flat = try ops.contiguous(a);
    try mlx.check(mlx.mlx_array_eval(flat));
    return mlx.mlx_array_data_bfloat16(flat).?[0..mlx.mlx_array_size(flat)];
}

fn widen(bits: u16) f64 {
    return @as(f32, @bitCast(@as(u32, bits) << 16));
}

test "GLM MLA head batch runs BF16 banks as stored and stays within one BF16 ulp of an FP64 oracle, as the staged chain does" {
    try mlx.requireGpu();
    if (!@import("glm5_model.zig").naxArms()) return error.SkipZigTest;
    const rows: c_int = 192;
    for ([_]Direction{ .query, .value }) |dir| {
        var ops = Ops{ .s = mlx.gpuStream() };
        defer ops.deinit();
        const width: c_int = if (dir == .query) 256 else 512;
        const out_width: usize = if (dir == .query) 512 else 256;
        const x = try bf16Normal(&ops, &.{ rows, 64, 1, width }, 11, 1.0);
        const w = try bf16Normal(&ops, &.{ 64, 256, 512 }, 12, 0.04);
        const before = dispatchCount();
        const got = (try run(&ops, .{ .x = x, .w = w }, dir)) orelse return error.ExpectedBf16HeadBatch;
        try std.testing.expectEqual(before + 1, dispatchCount());
        try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(got));
        try std.testing.expectEqualSlices(c_int, &.{ rows, 64, 1, @intCast(out_width) }, mlx.getShape(got));
        const staged = try ops.binary(.mm, x, if (dir == .value) try ops.transpose(w, &.{ 0, 2, 1 }) else w);
        const xs = try bf16Host(&ops, x);
        const ws = try bf16Host(&ops, w);
        const k: usize = @intCast(width);
        for ([_]Arr{ got, staged }) |candidate| {
            const ys = try bf16Host(&ops, candidate);
            for ([_]usize{ 0, 37, 127, 128, 191 }) |r| for (0..64) |h| for (0..out_width) |o| {
                var want: f64 = 0;
                for (0..k) |d| {
                    const wi = if (dir == .query) (h * 256 + d) * 512 + o else (h * 256 + o) * 512 + d;
                    want += widen(xs[(r * 64 + h) * k + d]) * widen(ws[wi]);
                }
                try std.testing.expect(@abs(widen(ys[(r * 64 + h) * out_width + o]) - want) <= @abs(want) * 0x1p-7 + 0x1p-14);
            };
        }
    }
}
