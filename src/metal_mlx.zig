//! Where our Metal queue meets MLX's: a buffer of ours over the memory of an MLX array, and the order the two
//! queues take turns in.
//!
//! One writer queue at a time. MLX's stream is drained (`drain`) before a pass of ours reads what MLX wrote, and a
//! pass of ours is waited on before any MLX op reads what it wrote. Neither queue sees the other's commands.

const std = @import("std");
const metal = @import("metal.zig");
const mlx = @import("mlx.zig");

pub const Error = metal.Error || error{ MlxError, NotEvaluated, NotRowContiguous };

/// MLX's own Metal buffer for `arr`, bound at the first byte `arr` views; `arr` is evaluated and may be a strided
/// view. Not retained: the caller keeps `arr`, and with it the buffer, while a pass binds it. What a pass writes
/// through the buffer is what MLX reads from `arr` afterwards.
pub fn bufferOf(arr: mlx.mlx_array) Error!metal.Bound {
    var available = false;
    try mlx.check(mlx._mlx_array_is_available(&available, arr));
    if (!available) return error.NotEvaluated;
    var raw: ?*anyopaque = null;
    var offset: usize = 0;
    try mlx.check(mlx._mlx_array_metal_buffer(&raw, &offset, arr));
    return .{ .buffer = .{ .raw = raw orelse return error.NotEvaluated }, .offset = offset };
}

/// Elements from `arr`'s first to its last, as its strides lay them out.
pub fn span(arr: mlx.mlx_array) usize {
    const shape = mlx.getShape(arr);
    const strides = mlx.mlx_array_strides(arr);
    var last: usize = 0;
    for (shape, 0..) |d, axis| {
        if (d == 0) return 0;
        last += @as(usize, @intCast(d - 1)) * strides[axis];
    }
    return last + 1;
}

/// Waits until MLX's stream has run everything submitted to it.
pub fn drain(s: mlx.mlx_stream) Error!void {
    try mlx.check(mlx.mlx_synchronize(s));
}

const scale_source =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\[[kernel]] void sushi_boundary_scale(
    \\  const device float* x [[buffer(0)]],
    \\  device float* out [[buffer(1)]],
    \\  const constant float& scale [[buffer(2)]],
    \\  uint3 at [[thread_position_in_grid]]) {
    \\  out[at.x] = x[at.x] * scale;
    \\}
;

/// `0, 1, .. n-1` as float32, computed by MLX on the GPU into memory MLX allocates.
fn mlxRamp(n: usize, s: mlx.mlx_stream) !mlx.mlx_array {
    var ramp = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(ramp);
    try mlx.check(mlx.mlx_arange(&ramp, 0, @floatFromInt(n), 1, .float32, s));
    try mlx.check(mlx.mlx_array_eval(ramp));
    return ramp;
}

test "a pass of ours reads an array MLX allocated and computed, whatever its size" {
    try mlx.requireGpu();
    const t = std.testing;
    const dev = try metal.device();
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    const state = try dev.compile(scale_source, "sushi_boundary_scale");
    defer state.release();

    // Under a page, a page and a bit, and many pages: MLX places these differently.
    for ([_]usize{ 8, 300, 5000, 200_000 }) |n| {
        const ramp = try mlxRamp(n, s);
        defer _ = mlx.mlx_array_free(ramp);
        try drain(s);
        const x = try bufferOf(ramp);
            const out = try dev.alloc(n * 4);
        defer out.release();

        const pass = dev.begin();
        pass.pipeline(state);
        pass.buffer(x.buffer, x.offset, 0);
        pass.buffer(out, 0, 1);
        pass.constant(@as(f32, 3), 2);
        pass.dispatch(.{ .x = n }, .{ .x = @min(n, 64) });
        try pass.commit().wait();
        const got: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, out.bytes()[0 .. n * 4]));
        for (got, 0..) |v, i| try t.expectEqual(@as(f32, @floatFromInt(i)) * 3, v);
    }
}

test "what a pass of ours writes into an MLX array is what MLX then computes on" {
    try mlx.requireGpu();
    const t = std.testing;
    const dev = try metal.device();
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    const state = try dev.compile(scale_source, "sushi_boundary_scale");
    defer state.release();

    const n = 50_000;
    const ramp = try mlxRamp(n, s);
    defer _ = mlx.mlx_array_free(ramp);
    try drain(s);
    const x = try bufferOf(ramp);

    // In place: the array's own bytes are the input and the output.
    const pass = dev.begin();
    pass.pipeline(state);
    pass.buffer(x.buffer, x.offset, 0);
    pass.buffer(x.buffer, x.offset, 1);
    pass.constant(@as(f32, 2), 2);
    pass.dispatch(.{ .x = n }, .{ .x = 64 });
    try pass.commit().wait();

    var total = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(total);
    try mlx.check(mlx.mlx_sum(&total, ramp, false, s));
    var sum: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&sum, total));
    // 2 * (0 + 1 + .. + n-1), exact in float32 at this n under any summation order MLX uses.
    try t.expectEqual(@as(f32, @floatFromInt(n * (n - 1))), sum);
}

test "an unevaluated array is refused by name, and a strided view is covered from its first element to its last" {
    try mlx.requireGpu();
    const t = std.testing;
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    const ramp = try mlxRamp(24, s);
    defer _ = mlx.mlx_array_free(ramp);
    var grid = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(grid);
    try mlx.check(mlx.mlx_reshape(&grid, ramp, &[_]c_int{ 4, 6 }, 2, s));
    var turned = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(turned);
    try mlx.check(mlx.mlx_transpose(&turned, grid, s));
    try mlx.check(mlx.mlx_array_eval(turned));
    try t.expectEqual(@as(usize, 24), span(turned));
    var column = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(column);
    try mlx.check(mlx.mlx_slice(&column, grid, &[_]c_int{ 1, 2 }, 2, &[_]c_int{ 4, 3 }, 2, &[_]c_int{ 1, 1 }, 2, s));
    try mlx.check(mlx.mlx_array_eval(column));
    try t.expectEqual(@as(usize, 13), span(column));
    const bound = try bufferOf(column);
    try t.expectEqual(@as(f32, 8), @as(*const f32, @ptrCast(@alignCast(bound.buffer.bytes().ptr + bound.offset))).*);
    var lazy = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(lazy);
    try mlx.check(mlx.mlx_multiply(&lazy, ramp, ramp, s));
    try t.expectError(error.NotEvaluated, bufferOf(lazy));
}
