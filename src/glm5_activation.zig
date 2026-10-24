//! GLM dense/shared FFN activation with source BF16 rounding boundaries.
const std = @import("std");
const mlx = @import("mlx.zig");
const Arr = mlx.mlx_array;
var kernel: ?mlx.mlx_fast_metal_kernel = null;
var calls: usize = 0;
pub fn callCount() usize {
    return calls;
}
pub fn resetCallCount() void {
    calls = 0;
}

pub fn apply(s: mlx.mlx_stream, gate: Arr, up: Arr, limit: f32) !?Arr {
    if (!mlx.streamIsGpu(s) or gate.ctx == null or up.ctx == null or limit != 10 or
        mlx.mlx_array_dtype(gate) != .bfloat16 or mlx.mlx_array_dtype(up) != .bfloat16 or
        !std.mem.eql(c_int, mlx.getShape(gate), mlx.getShape(up))) return null;
    if (mlx.getShape(gate).len == 0) return null;
    const n = mlx.mlx_array_size(gate);
    if (n == 0 or n > std.math.maxInt(c_int)) return null;
    if (kernel == null) {
        const ins = mlx.mlx_vector_string_new_data(&.{ "gate", "up", "sigmoid" }, 3);
        defer _ = mlx.mlx_vector_string_free(ins);
        const outs = mlx.mlx_vector_string_new_data(&.{"out"}, 1);
        defer _ = mlx.mlx_vector_string_free(outs);
        const source: [:0]const u8 =
            \\uint i = thread_position_in_grid.x;
            \\if (i >= threads_per_grid.x) return;
            \\float g = float(gate[i]), u = float(up[i]);
            \\g = g > 10.0f ? 10.0f : g;
            \\u = u > 10.0f ? 10.0f : u;
            \\u = u < -10.0f ? -10.0f : u;
            \\bfloat16_t cg = bfloat16_t(g);
            \\bfloat16_t mid = bfloat16_t(g * float(sigmoid[as_type<ushort>(cg)]));
            \\out[i] = bfloat16_t(float(mid) * u);
        ;
        // ensure_row_contiguous makes strided inputs safe; the grid is the element count.
        const k = mlx.mlx_fast_metal_kernel_new("sushi_glm_clamped_bf16_swiglu", ins, outs, source, "", true, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        kernel = k;
    }
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const shape = mlx.getShape(gate);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, shape.ptr, shape.len, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @intCast(n), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    const inputs = mlx.mlx_vector_array_new_data(&.{ gate, up, try @import("hc_prefill.zig").sigmoidTable(s) }, 3);
    defer _ = mlx.mlx_vector_array_free(inputs);
    var outputs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs, kernel.?, inputs, cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs, 0));
    calls += 1;
    return out;
}

test "GLM fused dense activation exhaustive BF16 gates and clamp boundaries" {
    const Ops = @import("glm5_model.zig").Ops;
    var ops = Ops{ .s = mlx.gpuStream() };
    defer ops.deinit();
    var bits: [65536]u16 = undefined;
    for (&bits, 0..) |*v, i| v.* = @intCast(i);
    const values = try ops.own(mlx.mlx_array_new_data(&bits, &.{65536}, 1, .bfloat16));
    const hi = try ops.scalar(10, .bfloat16);
    const lo = try ops.scalar(-10, .bfloat16);
    for ([_]bool{ false, true }) |swap| {
        for ([_]f32{ -11, -10, -9.9375, -0.25, 0, 0.25, 9.9375, 10, 11 }) |u| {
            const fixed = try ops.broadcast(try ops.scalar(u, .bfloat16), &.{65536});
            const gate = if (swap) fixed else values;
            const up = if (swap) values else fixed;
            const cg = try ops.binary(.min, gate, hi);
            const silu = try ops.silu(cg);
            const cu = try ops.binary(.max, try ops.binary(.min, up, hi), lo);
            const reference = try ops.binary(.mul, silu, cu);
            const actual = try ops.own((try apply(ops.s, gate, up, 10)).?);
            try mlx.check(mlx.mlx_array_eval(reference));
            try mlx.check(mlx.mlx_array_eval(actual));
            const r: [*]const u16 = @ptrCast(mlx.mlx_array_data_bfloat16(reference).?);
            const a: [*]const u16 = @ptrCast(mlx.mlx_array_data_bfloat16(actual).?);
            for (0..65536) |i| {
                if ((r[i] & 0x7f80) == 0x7f80 and (r[i] & 0x7f) != 0) {
                    try std.testing.expect((a[i] & 0x7f80) == 0x7f80 and (a[i] & 0x7f) != 0);
                } else try std.testing.expectEqual(r[i], a[i]);
            }
        }
    }
    try std.testing.expect((try apply(ops.s, values, values, 9)) == null);
}

test "GLM fused dense activation rejects unsupported input shapes and dtypes" {
    const Ops = @import("glm5_model.zig").Ops;
    var ops = Ops{ .s = mlx.gpuStream() };
    defer ops.deinit();
    const scalar = try ops.scalar(1, .bfloat16);
    try std.testing.expect((try apply(ops.s, scalar, scalar, 10)) == null);
    const fp = try ops.ones(&.{4}, .float32);
    try std.testing.expect((try apply(ops.s, fp, fp, 10)) == null);
    const bf = try ops.cast(fp, .bfloat16);
    const matrix = try ops.reshape(bf, &.{ 2, 2 });
    try std.testing.expect((try apply(ops.s, bf, matrix, 10)) == null);
    try std.testing.expect((try apply(ops.s, bf, fp, 10)) == null);
}

test "GLM fused dense activation packs transposed inputs without changing values" {
    const Ops = @import("glm5_model.zig").Ops;
    var ops = Ops{ .s = mlx.gpuStream() };
    defer ops.deinit();
    const data = [_]f32{ -11, -9.5, 2, 12, 0.25, -0.5, 1, 3 };
    const raw = try ops.own(mlx.mlx_array_new_data(&data, &.{ 2, 4 }, 2, .float32));
    const bf = try ops.cast(raw, .bfloat16);
    const gate = try ops.transpose(bf, &.{ 1, 0 });
    const up = try ops.binary(.mul, gate, try ops.scalar(-1, .bfloat16));
    const hi = try ops.scalar(10, .bfloat16);
    const lo = try ops.scalar(-10, .bfloat16);
    const cg = try ops.binary(.min, gate, hi);
    const cu = try ops.binary(.max, try ops.binary(.min, up, hi), lo);
    const ref = try ops.binary(.mul, try ops.silu(cg), cu);
    const got = try ops.own((try apply(ops.s, gate, up, 10)).?);
    const equal = try ops.slot();
    try mlx.check(mlx.mlx_array_equal(equal, ref, got, false, ops.s));
    try mlx.check(mlx.mlx_array_eval(equal.*));
    try std.testing.expect(mlx.mlx_array_data_bool(equal.*).?[0]);
}
