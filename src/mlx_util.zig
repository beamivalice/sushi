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
    errdefer _ = mlx.mlx_array_free(w_t);
    try mlx.check(mlx.mlx_transpose_axes(&w_t, w, &perm, @intCast(ndim), s));
    return w_t;
}

/// A 0-d scalar in `dt` (a bare f32 scalar array promotes bf16 operands).
pub fn scalarOf(val: f32, dt: mlx.mlx_dtype, s: mlx.mlx_stream) !mlx.mlx_array {
    // A float dtype is rounded on the host as the GPU's cast rounds (to nearest even): no dispatch for a constant.
    switch (dt) {
        .float32 => return mlx.newFloat(val),
        .float16 => {
            const h: f16 = @floatCast(val);
            return mlx.newData(&h, &.{}, .float16);
        },
        .bfloat16 => {
            const bits: u32 = @bitCast(val);
            const b: u16 = if (std.math.isNan(val)) @truncate((bits >> 16) | 0x40) else @truncate((bits +% 0x7FFF +% ((bits >> 16) & 1)) >> 16);
            return mlx.newData(&b, &.{}, .bfloat16);
        },
        else => {},
    }
    const raw = try mlx.newFloat(val);
    defer _ = mlx.mlx_array_free(raw);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_astype(&out, raw, dt, s));
    return out;
}

pub fn bf16Scalar(val: f32, s: mlx.mlx_stream) !mlx.mlx_array {
    return scalarOf(val, .bfloat16, s);
}

/// A 0-d f32 host scalar, kept as is. Made on the engine it takes a pool buffer, which can be refused.
pub fn f32Scalar(val: f32) !mlx.mlx_array {
    return mlx.newFloat(val);
}

pub fn addOne(arr: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const one = try bf16Scalar(1.0, s);
    defer _ = mlx.mlx_array_free(one);
    var result = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(result);
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
    // The engine has the copy itself: one dispatch into a buffer of its own.
    if (mlx.engine.isStream(s)) {
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.engine.copyOwned(&out, x));
        return out;
    }
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
    const window_start = try mlx.newInt(kv_len - window);
    defer _ = mlx.mlx_array_free(window_start);
    var too_old = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(too_old);
    try mlx.check(mlx.mlx_less(&too_old, positions, window_start, s));
    const zero = try bf16Scalar(0.0, s);
    defer _ = mlx.mlx_array_free(zero);
    const neg_inf = try bf16Scalar(-std.math.inf(f32), s);
    defer _ = mlx.mlx_array_free(neg_inf);
    var sw_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sw_mask);
    try mlx.check(mlx.mlx_where(&sw_mask, too_old, neg_inf, zero, s));
    const mask_shape = [_]c_int{ 1, 1, 1, kv_len };
    var mask_4d = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(mask_4d);
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

pub fn concatAxis0(s: mlx.mlx_stream, parts: []const mlx.mlx_array) !mlx.mlx_array {
    const vec = mlx.mlx_vector_array_new_data(parts.ptr, parts.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 0, s));
    return out;
}

test "a scalar the engine's pool refuses is an error, never a handle kept with the refusal latched" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer mlx.engine.enabled = was;
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    defer mlx.engine.fault.disarm();
    var said: [256]u8 = undefined;
    mlx.engine.fault.arm(1);
    try std.testing.expectError(error.HostArrayRefused, bf16Scalar(0.5, es));
    _ = mlx.takeError(&said);
    mlx.engine.fault.arm(1);
    try std.testing.expectError(error.HostArrayRefused, f32Scalar(1e-6));
    _ = mlx.takeError(&said);
    mlx.engine.fault.disarm();
    const ok = try f32Scalar(1e-6);
    defer _ = mlx.mlx_array_free(ok);
    try std.testing.expect(!mlx.errorPending());
}

test "shapeOf refuses an empty handle and another rank, where getShape would hand back a slice too short to index" {
    try std.testing.expectError(error.EmptyArray, mlx.shapeOf(.{}, 2));
    const ids = [_]i32{ 1, 2, 3 };
    const a = mlx.mlx_array_new_data(&ids, &[_]c_int{ 1, 3 }, 2, .int32);
    defer _ = mlx.mlx_array_free(a);
    try std.testing.expectEqual([2]c_int{ 1, 3 }, try mlx.shapeOf(a, 2));
    try std.testing.expectError(error.ShapeRank, mlx.shapeOf(a, 3));
}

test "on the engine a materialized copy is one dispatch into a buffer of its own, with the source's bytes, a view's too" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer mlx.engine.enabled = was;
    const es = mlx.engine.stream();
    const vals = [_]f32{ 1.5, -0.0, 3.25, -7.0, 0.0, 2.0 };
    const host = mlx.mlx_array_new_data(&vals, &[_]c_int{ 2, 3 }, 2, .float32);
    defer _ = mlx.mlx_array_free(host);
    var src = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(src);
    try mlx.check(mlx.mlx_contiguous(&src, host, false, es));
    try mlx.check(mlx.mlx_array_eval(src));
    var row = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(row);
    try mlx.check(mlx.mlx_slice(&row, src, &.{ 1, 0 }, 2, &.{ 2, 3 }, 2, &.{ 1, 1 }, 2, es));
    for ([_]mlx.mlx_array{ src, row }, [_]usize{ 0, 3 }) |x, at| {
        const d0 = mlx.engine.stats().dispatches;
        const copy = try materializedOwnedCopy(es, x);
        defer _ = mlx.mlx_array_free(copy);
        try std.testing.expectEqual(@as(u64, 1), mlx.engine.stats().dispatches - d0);
        try mlx.check(mlx.mlx_array_eval(copy));
        const got = mlx.mlx_array_data_float32(copy).?;
        try std.testing.expect(got != mlx.mlx_array_data_float32(x).?);
        for (0..mlx.mlx_array_size(copy)) |i| try std.testing.expectEqual(@as(u32, @bitCast(vals[at + i])), @as(u32, @bitCast(got[i])));
    }
}

test "a scalar in a float dtype is made on the host: no dispatch on the engine, and the GPU cast's bytes" {
    try mlx.requireGpu();
    const gpu = mlx.gpuStream();
    const values = [_]f32{ 1e-6, 1.0 / @sqrt(@as(f32, 2560)), 1.0 / @sqrt(@as(f32, 128)), 0.5, -3.0e38, 65520.0, 1.00390625, 1.01171875, -0.0, 7.0e-40 };
    const dtypes = [_]mlx.mlx_dtype{ .bfloat16, .float16, .float32 };
    var want: [dtypes.len][values.len][4]u8 = undefined;
    for (dtypes, 0..) |dt, di| for (values, 0..) |v, vi| {
        const raw = try mlx.newFloat(v);
        defer _ = mlx.mlx_array_free(raw);
        var cast = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cast);
        try mlx.check(mlx.mlx_astype(&cast, raw, dt, gpu));
        try mlx.check(mlx.mlx_array_eval(cast));
        @memcpy(want[di][vi][0..mlx.mlx_array_itemsize(cast)], mlx.mlx_array_data_uint8(cast).?[0..mlx.mlx_array_itemsize(cast)]);
    };
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer mlx.engine.enabled = was;
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    for (dtypes, 0..) |dt, di| for (values, 0..) |v, vi| {
        const d0 = mlx.engine.stats().dispatches;
        const mine = try scalarOf(v, dt, es);
        defer _ = mlx.mlx_array_free(mine);
        try std.testing.expectEqual(d0, mlx.engine.stats().dispatches);
        try std.testing.expectEqual(dt, mlx.mlx_array_dtype(mine));
        try mlx.check(mlx.mlx_array_eval(mine));
        const n = mlx.mlx_array_itemsize(mine);
        const got = mlx.mlx_array_data_uint8(mine).?[0..n];
        // A subnormal's cast is the GPU class's: the M5 flushes it to zero, the M1 keeps it. The host's is either.
        if (@abs(v) < std.math.floatMin(f32) and v != 0 and std.mem.allEqual(u8, want[di][vi][0..n], 0)) continue;
        try std.testing.expectEqualSlices(u8, want[di][vi][0..n], got);
    };
}
