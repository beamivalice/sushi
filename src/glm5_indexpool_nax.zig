//! IndexPool NAX scoring in 2048-pool tiles; top-512 retrieval policy is unchanged.
const std = @import("std");
const mlx = @import("mlx.zig");
const Ops = @import("glm5_model.zig").Ops;
const Arr = mlx.mlx_array;
pub const dot_limit: usize = 4 * 1024 * 1024;
pub const tile_pools: usize = 2048;
pub const max_rows: usize = 32;
/// Tiles stay lazy until their packed tile settles: two 32-row selections at 8192 pools, each
/// holding every dot plane (16 MiB), the tile outputs and the joined scores (1 MiB each).
pub const transient_bytes: usize = 2 * (16 * 1024 * 1024 + 2 * 1024 * 1024);
var calls: usize = 0;
pub fn enabled() bool {
    return !@import("glm5_model.zig").reference_numerics and @import("glm5_model.zig").naxArms();
}
pub fn resetDispatchCount() void { calls = 0; }
pub fn dispatchCount() usize { return calls; }
pub fn transientBudget(chunk: usize, pending_layers: usize) !usize {
    if (!enabled() or chunk <= 8) return 0;
    return std.math.mul(usize, transient_bytes, pending_layers);
}
pub fn dotBytes(rows: usize, pools: usize) !usize {
    return std.math.mul(usize, try std.math.mul(usize, rows, 32), try std.math.mul(usize, pools, 2));
}
pub fn outputBytes(rows: usize, pools: usize) !usize {
    return std.math.mul(usize, try std.math.mul(usize, rows, pools), 4);
}
// Dot output already has the original BF16(dot) boundary. Preserve product
// rounding, head order, final score rounding, and complete-pool causality.
const EPILOGUE: [:0]const u8 =
    \\#pragma clang fp contract(off)
    \\const uint i=thread_position_in_grid.x;
    \\if(i>=uint(ROWS)*uint(COLS)) return;
    \\const uint row=i/uint(COLS),col=i%uint(COLS),p=uint(first)+col;
    \\if((p+1u)*4u>uint(offset)+row+1u) {out[i]=-INFINITY;return;}
    \\float total=0.0f;
    \\for(uint h=0u;h<32u;++h) {
    \\ const float dot=float(dots[(row*32u+h)*uint(COLS)+col]);
    \\ total+=float(InT(max(dot,0.0f)*float(weights[row*32u+h])));
    \\}
    \\out[i]=float(InT(total));
;
fn kernel(comptime name: [:0]const u8, ins: []const [*:0]const u8, source: [:0]const u8) !mlx.mlx_fast_metal_kernel {
    const iv = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&.{"out"}, 1);
    defer _ = mlx.mlx_vector_string_free(ov);
    const result = mlx.mlx_fast_metal_kernel_new(name, iv, ov, source, "", true, false);
    if (result.ctx == null) return error.MetalKernelCompileFailed;
    return result;
}
var epi_kernel: ?mlx.mlx_fast_metal_kernel = null;
fn apply(ops: *Ops, k: mlx.mlx_fast_metal_kernel, cfg: mlx.mlx_fast_metal_kernel_config, inputs: []const Arr) !Arr {
    const iv = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, k, iv, cfg, ops.s));
    const out = try ops.slot();
    try mlx.check(mlx.mlx_vector_array_get(out, ov, 0));
    return out.*;
}
fn scalar(ops: *Ops, n: usize) !Arr {
    const value: u32 = @intCast(n);
    return ops.own(mlx.mlx_array_new_data(&value, &.{}, 0, .uint32));
}
fn geometry(q: []const c_int, keys: []const c_int, weights: []const c_int, offset: usize, pools: usize) bool {
    return q.len == 3 and q[0] > 0 and q[0] <= max_rows and q[1] == 32 and q[2] == 128 and
        keys.len == 2 and keys[0] >= pools and keys[1] == 128 and weights.len == 2 and weights[0] == q[0] and weights[1] == 32 and
        pools >= 512 and pools <= 8192 and offset <= pools * 4 and q[0] <= pools * 4 + 3 - offset;
}
/// Only the measured long-history prefill geometry is eligible. Decode/verify
/// and mixed precision retain the original scalar scorer.
pub fn tryScores(q: Arr, keys: Arr, weights: Arr, offset: usize, pools: usize, s: mlx.mlx_stream) !?Arr {
    if (!enabled() or !mlx.streamIsGpu(s) or pools < 3584) return null;
    for ([_]Arr{ q, keys, weights }) |a| if (a.ctx == null or mlx.mlx_array_dtype(a) != .bfloat16) return null;
    const sh = mlx.getShape(q);
    if (!geometry(sh, mlx.getShape(keys), mlx.getShape(weights), offset, pools) or sh[0] <= 8) return null;
    const result = try scores(q, keys, weights, offset, pools, s);
    calls += 1;
    return result;
}
/// Existing BF16 caches/queries/weights; returned FP32 scores contain rounded
/// BF16 values and -infinity for future pools, as in the original SCORE kernel.
pub fn scores(q: Arr, keys: Arr, weights: Arr, offset: usize, pools: usize, s: mlx.mlx_stream) !Arr {
    if (!mlx.streamIsGpu(s) or !@import("glm5_model.zig").naxArms()) return error.UnsupportedGlmIndexNax;
    for ([_]Arr{ q, keys, weights }) |a| if (a.ctx == null or mlx.mlx_array_dtype(a) != .bfloat16) return error.UnsupportedGlmIndexNax;
    if (!geometry(mlx.getShape(q), mlx.getShape(keys), mlx.getShape(weights), offset, pools)) return error.UnsupportedGlmIndexNax;
    const rows = mlx.getShape(q)[0];
    if (try outputBytes(@intCast(rows), pools) > dot_limit) return error.GlmIndexNaxScratchBudget;
    const parts = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(parts);
    var first: usize = 0;
    while (first < pools) {
        const end = @min(pools, first + tile_pools);
        const cols: c_int = @intCast(end - first);
        if (try dotBytes(@intCast(rows), @intCast(cols)) > dot_limit) return error.GlmIndexNaxScratchBudget;
        var ops = Ops{ .s = s };
        defer ops.deinit();
        const flat = try ops.reshape(try ops.contiguous(q), &.{ rows * 32, 128 });
        const pool = try ops.slice(keys, 0, @intCast(first), @intCast(end));
        const dots = try ops.binary(.mm, flat, try ops.transpose(pool, &.{ 1, 0 }));
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ rows, cols }, 2, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, rows * cols, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ROWS", rows));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "COLS", cols));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "InT", .bfloat16));
        if (epi_kernel == null) epi_kernel = try kernel("sushi_glm_index_nax_epilogue", &.{ "dots", "weights", "offset", "first" }, EPILOGUE);
        const tile = try apply(&ops, epi_kernel.?, cfg, &.{ dots, weights, try scalar(&ops, offset), try scalar(&ops, first) });
        try mlx.check(mlx.mlx_vector_array_append_value(parts, tile));
        first = end;
    }
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, parts, 1, s));
    return out;
}
test "GLM IndexPool NAX dot plane remains within original bound" {
    try std.testing.expectEqual(dot_limit, try dotBytes(32, 2048));
    try std.testing.expectEqual(@as(usize, 1048576), try outputBytes(32, 8192));
    try std.testing.expect(geometry(&.{ 32, 32, 128 }, &.{ 8192, 128 }, &.{ 32, 32 }, 32736, 8192));
    try std.testing.expect(!geometry(&.{ 33, 32, 128 }, &.{ 8192, 128 }, &.{ 33, 32 }, 32735, 8192));
}
test "GLM IndexPool NAX reserves transient copies per pending layer" {
    if (!@import("glm5_model.zig").naxArms()) return error.SkipZigTest;
    // Two packed tiles in flight hold two 32-row selections, each with every tile's dot plane,
    // its epilogue output and the joined scores.
    const selection = try dotBytes(max_rows, 8192) + 2 * try outputBytes(max_rows, 8192);
    try std.testing.expectEqual(2 * selection, transient_bytes);
    try std.testing.expectEqual(transient_bytes * 2, try transientBudget(2048, 2));
    try std.testing.expectEqual(@as(usize, 0), try transientBudget(8, 2));
    try std.testing.expectError(error.Overflow, transientBudget(2048, std.math.maxInt(usize)));
    const model = @import("glm5_model.zig");
    model.reference_numerics = true;
    defer model.reference_numerics = false;
    try std.testing.expectEqual(@as(usize, 0), try transientBudget(2048, 2));
}
