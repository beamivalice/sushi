//! MLX array helpers with no model or architecture knowledge: scalars, views, owned copies, masks. Every
//! function takes the stream it runs on.
const std = @import("std");
const mlx = @import("mlx.zig");

/// Last-axis size of an array, as the input-dimension hint for
/// `quantParamsHinted`. Null for null-ctx handles or degenerate shapes.
pub inline fn lastDim(x: mlx.mlx_array) ?u32 {
    if (x.ctx == null) return null;
    const shape = mlx.getShape(x);
    if (shape.len == 0) return null;
    const d = shape[shape.len - 1];
    return if (d > 0) @as(u32, @intCast(d)) else null;
}

/// Pre-transpose a plain-BF16 weight stored as `[out, in]` to `[in, out]` so
/// `mlx_matmul(x, w_t)` lands the contraction over the input axis. Used by
/// Unsloth Dynamic checkpoints that leave linear-attention projections
/// unquantized while quantizing the rest. Caller owns the returned array.
pub fn transposeBf16Weight(w: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    // Swap the last two axes for any rank: 2D weights [out, in] → [in, out];
    // stacked MoE expert tensors [experts, out, in] → [experts, in, out].
    const ndim = mlx.getShape(w).len;
    var perm: [8]c_int = undefined; // mlx arrays never exceed 8 dims here
    for (0..ndim) |i| perm[i] = @intCast(i);
    perm[ndim - 2] = @intCast(ndim - 1);
    perm[ndim - 1] = @intCast(ndim - 2);
    var w_t = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_transpose_axes(&w_t, w, &perm, @intCast(ndim), s));
    return w_t;
}

/// A 0-d scalar in `dt` (a bare f32 scalar array promotes bf16 operands).
pub fn scalarOf(val: f32, dt: mlx.mlx_dtype, s: mlx.mlx_stream) !mlx.mlx_array {
    const raw = mlx.mlx_array_new_float(val);
    defer _ = mlx.mlx_array_free(raw);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, raw, dt, s));
    return out;
}

pub fn bf16Scalar(val: f32, s: mlx.mlx_stream) mlx.mlx_array {
    const f32_arr = mlx.mlx_array_new_float(val);
    defer _ = mlx.mlx_array_free(f32_arr);
    var bf16_arr = mlx.mlx_array_new();
    _ = mlx.mlx_astype(&bf16_arr, f32_arr, .bfloat16, s);
    return bf16_arr;
}

pub fn addOne(arr: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const one = bf16Scalar(1.0, s);
    defer _ = mlx.mlx_array_free(one);
    var result = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&result, one, arr, s));
    return result;
}

pub fn sliceAttentionSeq(s: mlx.mlx_stream, a: mlx.mlx_array, start: c_int, end: c_int) !mlx.mlx_array {
    const sh = mlx.getShape(a);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_slice(&out, a, &.{ 0, 0, start, 0 }, 4, &.{ sh[0], sh[1], end, sh[3] }, 4, &.{ 1, 1, 1, 1 }, 4, s));
    return out;
}

/// Force a REAL copy of `x` into a freshly allocated buffer. `mlx_copy` and
/// `mlx_contiguous` both take shared-buffer fast paths (Copy is a view op;
/// contiguous no-ops when the view's strides read as row-contiguous, which a
/// size-1-leading-dim slice does), so neither breaks the alias to a slice's
/// parent buffer. Adding a same-dtype scalar zero always runs a kernel with a
/// newly allocated output. Buffer donation can't alias either: the caller
/// still holds `x`, so its buffer is never donated.
pub fn materializedOwnedCopy(s: mlx.mlx_stream, x: mlx.mlx_array) !mlx.mlx_array {
    var zero = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(zero);
    const scalar_shape = [_]c_int{1};
    try mlx.check(mlx.mlx_zeros(&zero, &scalar_shape, 0, mlx.mlx_array_dtype(x), s));
    var neg_zero = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(neg_zero);
    try mlx.check(mlx.mlx_negative(&neg_zero, zero, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_add(&out, x, neg_zero, s));
    return out;
}

/// The additive `[1, 1, 1, kv_len]` bf16 mask a sliding decode row applies.
pub fn slidingDecodeMask(s: mlx.mlx_stream, kv_len: c_int, window: c_int) !mlx.mlx_array {
    var positions = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(positions);
    try mlx.check(mlx.mlx_arange(&positions, 0, @floatFromInt(kv_len), 1, .int32, s));
    const window_start = mlx.mlx_array_new_int(kv_len - window);
    defer _ = mlx.mlx_array_free(window_start);
    var too_old = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(too_old);
    try mlx.check(mlx.mlx_less(&too_old, positions, window_start, s));
    const zero = bf16Scalar(0.0, s);
    defer _ = mlx.mlx_array_free(zero);
    const neg_inf = bf16Scalar(-std.math.inf(f32), s);
    defer _ = mlx.mlx_array_free(neg_inf);
    var sw_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sw_mask);
    try mlx.check(mlx.mlx_where(&sw_mask, too_old, neg_inf, zero, s));
    const mask_shape = [_]c_int{ 1, 1, 1, kv_len };
    var mask_4d = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&mask_4d, sw_mask, &mask_shape, 4, s));
    return mask_4d;
}

test "lastDim reads the innermost size and refuses a null handle" {
    try std.testing.expectEqual(@as(?u32, null), lastDim(.{ .ctx = null }));
    try mlx.requireGpu();
    const shape = [_]c_int{ 2, 5 };
    const zeros: [10]f32 = @splat(0);
    const a = mlx.mlx_array_new_data(&zeros, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(a);
    try std.testing.expectEqual(@as(?u32, 5), lastDim(a));
}

test "a sliding decode mask hides exactly the keys older than the window" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const mask = try slidingDecodeMask(s, 6, 4);
    defer _ = mlx.mlx_array_free(mask);
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, mask, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f));
    const d = mlx.mlx_array_data_float32(f) orelse return error.MlxArrayDataNull;
    for (0..6) |pos| {
        if (pos < 2) try std.testing.expect(d[pos] == -std.math.inf(f32)) else try std.testing.expectEqual(@as(f32, 0), d[pos]);
    }
}

test "a materialized copy owns its buffer and a scalar takes the dtype asked" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const five = try scalarOf(5, .bfloat16, s);
    defer _ = mlx.mlx_array_free(five);
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(five));
    const copy = try materializedOwnedCopy(s, five);
    defer _ = mlx.mlx_array_free(copy);
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(copy));
}
