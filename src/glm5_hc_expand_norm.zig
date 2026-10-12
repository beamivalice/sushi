//! Preserve the BF16 residual while preparing its next HC collapse in the same dispatch.
const std = @import("std");
const mlx = @import("mlx.zig");
const Arr = mlx.mlx_array;

const SOURCE =
    \\#pragma clang fp contract(off)
    \\#pragma clang fp reassociate(off)
    \\const uint row=threadgroup_position_in_grid.x;
    \\const uint tid=thread_position_in_threadgroup.x;
    \\const uint lane=thread_index_in_simdgroup, sg=simdgroup_index_in_threadgroup;
    \\float vals[16];
    \\for(uint h=0;h<4;++h) for(uint i=0;i<4;++i) {
    \\ const uint d=tid*4+i;
    \\ float value=0.0f;
    \\ for(uint j=0;j<4;++j)value+=comb[row*16+j*4+h]*float(residual[(row*4+j)*4096+d]);
    \\ const float product=post[row*4+h]*float(branch[row*4096+d]);
    \\ const bfloat rounded=bfloat(product+value);
    \\ expanded[(row*4+h)*4096+d]=rounded;
    \\ vals[h*4+i]=float(rounded);
    \\}
    \\{
    \\#pragma clang fp contract(on)
    \\#pragma clang fp reassociate(on)
    \\ threadgroup float sums[32];
    \\ threadgroup float inv[1];
    \\ float acc=0.0f;
    \\ for(uint h=0;h<4;++h)for(uint i=0;i<4;++i)acc+=vals[h*4+i]*vals[h*4+i];
    \\ acc=simd_sum(acc);
    \\ if(lane==0)sums[sg]=acc;
    \\ threadgroup_barrier(mem_flags::mem_threadgroup);
    \\ if(sg==0){acc=simd_sum(sums[lane]);if(lane==0)inv[0]=metal::precise::rsqrt(acc/16384.0f+float(eps));}
    \\ threadgroup_barrier(mem_flags::mem_threadgroup);
    \\ for(uint h=0;h<4;++h)for(uint i=0;i<4;++i)normalized[(row*4+h)*4096+tid*4+i]=vals[h*4+i]*inv[0];
    \\}
;
var kernel: ?mlx.mlx_fast_metal_kernel = null;
/// One config per served row count: a serial decode row and the three-row verify.
var configs: [2]?mlx.mlx_fast_metal_kernel_config = @splat(null);
pub const Result = struct {
    expanded: Arr,
    normalized: Arr,
    pub fn deinit(self: Result) void {
        _ = mlx.mlx_array_free(self.expanded);
        _ = mlx.mlx_array_free(self.normalized);
    }
};
pub fn apply(s: mlx.mlx_stream, residual: Arr, branch: Arr, post: Arr, comb: Arr, epsilon: f32) !?Result {
    if (!mlx.streamIsGpu(s) or !@import("glm5_model.zig").naxArms() or !std.math.isFinite(epsilon) or epsilon <= 0) return null;
    for ([_]Arr{ residual, branch, post, comb }) |a| if (a.ctx == null) return null;
    if (mlx.mlx_array_dtype(residual) != .bfloat16 or mlx.mlx_array_dtype(branch) != .bfloat16 or mlx.mlx_array_dtype(post) != .float32 or mlx.mlx_array_dtype(comb) != .float32) return null;
    if (mlx.getShape(branch).len != 3) return null;
    const rows = mlx.getShape(branch)[1];
    if (rows != 1 and rows != 3) return null;
    if (!std.mem.eql(c_int, mlx.getShape(residual), &.{ 1, rows, 4, 4096 }) or !std.mem.eql(c_int, mlx.getShape(branch), &.{ 1, rows, 4096 }) or !std.mem.eql(c_int, mlx.getShape(post), &.{ 1, rows, 4 }) or !std.mem.eql(c_int, mlx.getShape(comb), &.{ 1, rows, 4, 4 })) return null;
    if (kernel == null) {
        const ins = mlx.mlx_vector_string_new_data(&.{ "residual", "branch", "post", "comb", "eps" }, 5);
        defer _ = mlx.mlx_vector_string_free(ins);
        const outs = mlx.mlx_vector_string_new_data(&.{ "expanded", "normalized" }, 2);
        defer _ = mlx.mlx_vector_string_free(outs);
        const k = mlx.mlx_fast_metal_kernel_new("sushi_glm_hc_expand_norm", ins, outs, SOURCE, "", true, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        kernel = k;
    }
    const config = &configs[@intFromBool(rows == 3)];
    if (config.* == null) {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &.{ 1, rows, 4, 4096 }, 4, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &.{ 1, rows, 16384 }, 3, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, rows * 1024, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 1024, 1, 1));
        config.* = c;
    }
    const eps = mlx.mlx_array_new_float(epsilon);
    defer _ = mlx.mlx_array_free(eps);
    const ins = mlx.mlx_vector_array_new_data(&.{ residual, branch, post, comb, eps }, 5);
    defer _ = mlx.mlx_vector_array_free(ins);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, kernel.?, ins, config.*.?, s));
    var out = Result{ .expanded = mlx.mlx_array_new(), .normalized = mlx.mlx_array_new() };
    errdefer out.deinit();
    try mlx.check(mlx.mlx_vector_array_get(&out.expanded, outs, 0));
    try mlx.check(mlx.mlx_vector_array_get(&out.normalized, outs, 1));
    return out;
}

test "GLM HC expand with FP32 normalization preserves both reference outputs" {
    if (mlx.noGpuBackend() or !@import("glm5_model.zig").naxArms()) return error.SkipZigTest;
    const Ops = @import("glm5_model.zig").Ops;
    const fixture = @import("dflash.zig").TinyFix;
    const s = mlx.gpuStream();
    for ([_]c_int{ 1, 3 }) |rows| for (0..8) |seed| {
        var ops = Ops{ .s = s };
        defer ops.deinit();
        const residual = if (seed == 0) try ops.zeros(&.{ 1, rows, 4, 4096 }, .bfloat16) else try ops.own(try fixture.bf16ArrShaped(&.{ 1, rows, 4, 4096 }, 781 + seed, s));
        const branch = if (seed == 0) try ops.zeros(&.{ 1, rows, 4096 }, .bfloat16) else try ops.own(try fixture.bf16ArrShaped(&.{ 1, rows, 4096 }, 801 + seed, s));
        const post = try ops.cast(try ops.own(try fixture.bf16ArrShaped(&.{ 1, rows, 4 }, 821 + seed, s)), .float32);
        const comb = try ops.cast(try ops.own(try fixture.bf16ArrShaped(&.{ 1, rows, 4, 4 }, 841 + seed, s)), .float32);
        const want = try ops.own(try @import("glm5_next.zig").hcExpand(residual, branch, post, comb, s));
        const norm = try ops.rms(try ops.reshape(try ops.cast(want, .float32), &.{ 1, rows, 16384 }), .{ .ctx = null }, 1e-6);
        const got = (try apply(s, residual, branch, post, comb, 1e-6)) orelse return error.TestExpectedFusedExpansion;
        defer got.deinit();
        for ([_]Arr{ want, norm, got.expanded, got.normalized }) |a| try mlx.check(mlx.mlx_array_eval(a));
        const n: usize = @intCast(rows * 16384);
        try std.testing.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(want).?[0..n], mlx.mlx_array_data_bfloat16(got.expanded).?[0..n]);
        try std.testing.expectEqualSlices(f32, mlx.mlx_array_data_float32(norm).?[0..n], mlx.mlx_array_data_float32(got.normalized).?[0..n]);
    };
}
