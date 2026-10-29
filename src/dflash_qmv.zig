//! Reuse each A4 g64 FFN weight across the eight-row DFlash noise block.
const std = @import("std");
const mlx = @import("mlx.zig");
// Same subchunk and shuffle reduction as Apple's MLX qmv_wide_impl (MIT).
const SOURCE =
    \\uint lane = thread_index_in_simdgroup;
    \\uint sg = simdgroup_index_in_threadgroup;
    \\uint kl = lane % 8, r = threadgroup_position_in_grid.y * 8 + sg * 4 + lane / 8;
    \\uint v0 = threadgroup_position_in_grid.x * NV;
    \\uint row = min(r, uint(N - 1));
    \\device const uchar* wr = reinterpret_cast<device const uchar*>(w) + size_t(row) * (K / 2);
    \\float result[NV] = {0};
    \\for (uint g = kl; g < K / 64; g += 8) {
    \\    float scale = float(sc[size_t(row) * (K / 64) + g]);
    \\    float bias = float(bi[size_t(row) * (K / 64) + g]);
    \\    #pragma unroll
    \\    for (uint chunk = 0; chunk < 8; ++chunk) {
    \\        uint k0 = g * 64 + chunk * 8;
    \\        float dq[8];
    \\        #pragma unroll
    \\        for (uint i = 0; i < 4; ++i) {
    \\            uint packed = wr[(k0 / 2) + i];
    \\            dq[2*i] = scale * float(packed & 15) + bias;
    \\            dq[2*i+1] = (scale / 16.0f) * float(packed & 240) + bias;
    \\        }
    \\        #pragma unroll
    \\        for (uint v = 0; v < NV; ++v) {
    \\            uint input_row = min(v0 + v, uint(M - 1));
    \\            float acc = 0;
    \\            #pragma unroll
    \\            for (uint i = 0; i < 8; ++i) acc += float(x[size_t(input_row) * K + k0 + i]) * dq[i];
    \\            result[v] += acc;
    \\        }
    \\    }
    \\}
    \\for (uint v = 0; v < NV; ++v) {
    \\    result[v] += simd_shuffle_down(result[v], 4);
    \\    result[v] += simd_shuffle_down(result[v], 2);
    \\    result[v] += simd_shuffle_down(result[v], 1);
    \\}
    \\if (kl == 0 && r < N) {
    \\    for (uint v = 0; v < NV; ++v)
    \\        if (v0 + v < M) y[size_t(v0 + v) * N + r] = T(result[v]);
    \\}
;
var kernel_cache: ?mlx.mlx_fast_metal_kernel = null;
// Only the two admitted FFN geometries need configurations.
var configs: [2]?mlx.mlx_fast_metal_kernel_config = @splat(null);

pub fn matmul(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s) or !@import("transformer.zig").verifySharedHardware()) return null;
    if (x.ctx == null or w.ctx == null or sc.ctx == null or bi.ctx == null) return null;
    if (mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_dtype(w) != .uint32 or mlx.mlx_array_dtype(sc) != .bfloat16 or mlx.mlx_array_dtype(bi) != .bfloat16) return null;
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    const ss = mlx.getShape(sc);
    if (xs.len != 3 or xs[0] != 1 or xs[1] != 8 or ws.len != 2 or ss.len != 2) return null;
    const m = xs[1];
    const k = xs[2];
    const n = ws[0];
    if (!((k == 4096 and n == 12288) or (k == 12288 and n == 4096)) or ws[1] != @divExact(k, 8) or ss[0] != n or ss[1] != @divExact(k, 64) or !std.mem.eql(c_int, ss, mlx.getShape(bi))) return null;
    const kernel = blk: {
        if (kernel_cache) |value| break :blk value;
        const ins = [_][*:0]const u8{ "x", "w", "sc", "bi" };
        const outs = [_][*:0]const u8{"y"};
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const value = mlx.mlx_fast_metal_kernel_new("sushi_dflash_a4_wide", iv, ov, SOURCE, "", true, false);
        if (value.ctx == null) return error.MetalKernelCompileFailed;
        kernel_cache = value;
        break :blk value;
    };
    const slot: usize = if (k == 4096) 0 else 1;
    const config = blk: {
        if (configs[slot]) |hit| break :blk hit;
        const value = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(value);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(value, &.{ 1, m, n }, 3, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(value, 32, @intCast(@divExact(n, 8) * 2), 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(value, 32, 2, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(value, "T", .bfloat16));
        inline for (.{ "M", "N", "K", "NV" }, .{ m, n, k, 8 }) |name, number| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(value, name, number));
        configs[slot] = value;
        break :blk value;
    };
    const iv = mlx.mlx_vector_array_new_data(&.{ x, w, sc, bi }, 4);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kernel, iv, config, s));
    var result = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(result);
    try mlx.check(mlx.mlx_vector_array_get(&result, ov, 0));
    return result;
}

// The per-matrix body of SOURCE for one weight row; both projections keep its exact order.
const ACT_HEADER =
    \\template <int NV, int M, int K>
    \\METAL_FUNC void sushi_a4_rows(const device uint32_t* w, const device bfloat16_t* sc, const device bfloat16_t* bi, const device bfloat16_t* x, uint row, uint kl, uint v0, thread float* result) {
    \\    device const uchar* wr = reinterpret_cast<device const uchar*>(w) + size_t(row) * (K / 2);
    \\    for (uint g = kl; g < K / 64; g += 8) {
    \\        float scale = float(sc[size_t(row) * (K / 64) + g]);
    \\        float bias = float(bi[size_t(row) * (K / 64) + g]);
    \\        #pragma unroll
    \\        for (uint chunk = 0; chunk < 8; ++chunk) {
    \\            uint k0 = g * 64 + chunk * 8;
    \\            float dq[8];
    \\            #pragma unroll
    \\            for (uint i = 0; i < 4; ++i) {
    \\                uint packed = wr[(k0 / 2) + i];
    \\                dq[2*i] = scale * float(packed & 15) + bias;
    \\                dq[2*i+1] = (scale / 16.0f) * float(packed & 240) + bias;
    \\            }
    \\            #pragma unroll
    \\            for (uint v = 0; v < NV; ++v) {
    \\                uint input_row = min(v0 + v, uint(M - 1));
    \\                float acc = 0;
    \\                #pragma unroll
    \\                for (uint i = 0; i < 8; ++i) acc += float(x[size_t(input_row) * K + k0 + i]) * dq[i];
    \\                result[v] += acc;
    \\            }
    \\        }
    \\    }
    \\}
;
// Gate and up rows of one output column, then MLX's BF16 sigmoid, gate*sigmoid and *up roundings.
const ACT_SOURCE =
    \\uint lane = thread_index_in_simdgroup;
    \\uint sg = simdgroup_index_in_threadgroup;
    \\uint kl = lane % 8, r = threadgroup_position_in_grid.y * 8 + sg * 4 + lane / 8;
    \\uint v0 = threadgroup_position_in_grid.x * NV;
    \\uint row = min(r, uint(N - 1));
    \\float rg[NV] = {0};
    \\float ru[NV] = {0};
    \\sushi_a4_rows<NV, M, K>(gw, gsc, gbi, x, row, kl, v0, rg);
    \\sushi_a4_rows<NV, M, K>(uw, usc, ubi, x, row, kl, v0, ru);
    \\for (uint v = 0; v < NV; ++v) {
    \\    rg[v] += simd_shuffle_down(rg[v], 4);
    \\    rg[v] += simd_shuffle_down(rg[v], 2);
    \\    rg[v] += simd_shuffle_down(rg[v], 1);
    \\    ru[v] += simd_shuffle_down(ru[v], 4);
    \\    ru[v] += simd_shuffle_down(ru[v], 2);
    \\    ru[v] += simd_shuffle_down(ru[v], 1);
    \\}
    \\if (kl == 0 && r < N) {
    \\    for (uint v = 0; v < NV; ++v) {
    \\        if (v0 + v >= M) continue;
    \\        bfloat16_t g = bfloat16_t(rg[v]);
    \\        bfloat16_t mid = bfloat16_t(float(g) * float(sigmoid[as_type<ushort>(g)]));
    \\        y[size_t(v0 + v) * N + r] = bfloat16_t(float(mid) * float(bfloat16_t(ru[v])));
    \\    }
    \\}
;
var act_kernel: ?mlx.mlx_fast_metal_kernel = null;
var act_configs: [2]?mlx.mlx_fast_metal_kernel_config = @splat(null);

/// `silu(x·gateᵀ) · (x·upᵀ)` for the assistant's A4 g64 4096→12288 FFN at 8 or 3 rows, with the
/// bits of two MLX quantized matmuls followed by sigmoid, multiply, multiply.
pub fn gateUpAct(s: mlx.mlx_stream, x: mlx.mlx_array, gate: [3]mlx.mlx_array, up: [3]mlx.mlx_array) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s) or !@import("transformer.zig").verifySharedHardware()) return null;
    for ([_]mlx.mlx_array{ x, gate[0], gate[1], gate[2], up[0], up[1], up[2] }) |a| if (a.ctx == null) return null;
    if (mlx.mlx_array_dtype(x) != .bfloat16) return null;
    const xs = mlx.getShape(x);
    if (xs.len != 3 or xs[0] != 1 or (xs[1] != 8 and xs[1] != 3) or xs[2] != 4096) return null;
    for ([_][3]mlx.mlx_array{ gate, up }) |w| {
        if (mlx.mlx_array_dtype(w[0]) != .uint32 or mlx.mlx_array_dtype(w[1]) != .bfloat16 or mlx.mlx_array_dtype(w[2]) != .bfloat16) return null;
        if (!std.mem.eql(c_int, mlx.getShape(w[0]), &.{ 12288, 512 }) or !std.mem.eql(c_int, mlx.getShape(w[1]), &.{ 12288, 64 }) or !std.mem.eql(c_int, mlx.getShape(w[2]), &.{ 12288, 64 })) return null;
    }
    const m = xs[1];
    const n: c_int = 12288;
    const kernel = blk: {
        if (act_kernel) |value| break :blk value;
        const ins = [_][*:0]const u8{ "x", "gw", "gsc", "gbi", "uw", "usc", "ubi", "sigmoid" };
        const outs = [_][*:0]const u8{"y"};
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const value = mlx.mlx_fast_metal_kernel_new("sushi_dflash_a4_gate_up_act", iv, ov, ACT_SOURCE, ACT_HEADER, true, false);
        if (value.ctx == null) return error.MetalKernelCompileFailed;
        act_kernel = value;
        break :blk value;
    };
    const slot: usize = if (m == 8) 0 else 1;
    const config = blk: {
        if (act_configs[slot]) |hit| break :blk hit;
        const value = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(value);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(value, &.{ 1, m, n }, 3, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(value, 32, @intCast(@divExact(n, 8) * 2), 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(value, 32, 2, 1));
        inline for (.{ "M", "N", "K", "NV" }, .{ m, n, 4096, m }) |name, number| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(value, name, number));
        act_configs[slot] = value;
        break :blk value;
    };
    const iv = mlx.mlx_vector_array_new_data(&.{ x, gate[0], gate[1], gate[2], up[0], up[1], up[2], try @import("hc_prefill.zig").sigmoidTable(s) }, 8);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kernel, iv, config, s));
    var result = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(result);
    try mlx.check(mlx.mlx_vector_array_get(&result, ov, 0));
    return result;
}

test "DFlash A4 fused gate, up and SiLU equal two MLX quantized matmuls and the staged activation" {
    if (mlx.noGpuBackend() or !@import("transformer.zig").verifySharedHardware()) return error.SkipZigTest;
    const Ops = @import("glm5/glm5_model.zig").Ops;
    var ops = Ops{ .s = mlx.gpuStream() };
    defer ops.deinit();
    var weights: [2][3]mlx.mlx_array = undefined;
    for (&weights, 0..) |*w, i| {
        const key = try ops.slot();
        try mlx.check(mlx.mlx_random_key(key, 801 + i));
        const dense = try ops.slot();
        try mlx.check(mlx.mlx_random_normal(dense, &[_]c_int{ 12288, 4096 }, 2, .bfloat16, 0, 0.02, key.*, ops.s));
        var quant = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(quant);
        try mlx.check(mlx.mlx_quantize(&quant, dense.*, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{ .ctx = null }, ops.s));
        for (w, 0..) |*part, j| {
            const out = try ops.slot();
            try mlx.check(mlx.mlx_vector_array_get(out, quant, j));
            try mlx.check(mlx.mlx_array_eval(out.*));
            part.* = out.*;
        }
    }
    // Input scales from near-linear to both saturated tails of the sigmoid.
    for ([_]c_int{ 8, 3 }) |rows| for ([_]f32{ 0.25, 4, 64 }, 0..) |gain, case| {
        const key = try ops.slot();
        try mlx.check(mlx.mlx_random_key(key, 811 + case));
        const x = try ops.slot();
        try mlx.check(mlx.mlx_random_normal(x, &[_]c_int{ 1, rows, 4096 }, 3, .bfloat16, 0, gain, key.*, ops.s));
        var mm: [2]mlx.mlx_array = undefined;
        for (&mm, weights) |*out, w| {
            const value = try ops.slot();
            try mlx.check(mlx.mlx_quantized_matmul(value, x.*, w[0], w[1], w[2], true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", ops.s));
            out.* = value.*;
        }
        const sig = try ops.slot();
        try mlx.check(mlx.mlx_sigmoid(sig, mm[0], ops.s));
        const want = try ops.binary(.mul, try ops.binary(.mul, mm[0], sig.*), mm[1]);
        const got = try ops.own((try gateUpAct(ops.s, x.*, weights[0], weights[1])) orelse return error.TestExpectedFusedActivation);
        try mlx.check(mlx.mlx_array_eval(want));
        try mlx.check(mlx.mlx_array_eval(got));
        const count: usize = @intCast(rows * 12288);
        try std.testing.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(want).?[0..count], mlx.mlx_array_data_bfloat16(got).?[0..count]);
    };
    try std.testing.expect((try gateUpAct(ops.s, try ops.zeros(&.{ 1, 4, 4096 }, .bfloat16), weights[0], weights[1])) == null);
}

test "DFlash A4 eight-row FFN projection matches MLX exactly" {
    if (mlx.noGpuBackend() or !@import("transformer.zig").verifySharedHardware()) return error.SkipZigTest;
    const Ops = @import("glm5/glm5_model.zig").Ops;
    const fixture = @import("dflash.zig").TinyFix;
    for ([_][2]c_int{ .{ 4096, 12288 }, .{ 12288, 4096 } }) |nk| {
        const k = nk[0];
        const n = nk[1];
        var inputs = Ops{ .s = mlx.gpuStream() };
        defer inputs.deinit();
        const x = try inputs.own(try fixture.bf16ArrShaped(&.{ 1, 8, k }, 781, inputs.s));
        const dense = try inputs.own(try fixture.bf16ArrShaped(&.{ n, k }, 783, inputs.s));
        const w = try inputs.slot();
        const sc = try inputs.slot();
        const bi = try inputs.slot();
        var quant = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(quant);
        try mlx.check(mlx.mlx_quantize(&quant, dense, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{ .ctx = null }, inputs.s));
        try mlx.check(mlx.mlx_vector_array_get(w, quant, 0));
        try mlx.check(mlx.mlx_vector_array_get(sc, quant, 1));
        try mlx.check(mlx.mlx_vector_array_get(bi, quant, 2));
        for ([_]mlx.mlx_array{ x, w.*, sc.*, bi.* }) |a| try mlx.check(mlx.mlx_array_eval(a));
        const want = try inputs.slot();
        try mlx.check(mlx.mlx_quantized_matmul(want, x, w.*, sc.*, bi.*, true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", inputs.s));
        try mlx.check(mlx.mlx_array_eval(want.*));
        const got = try inputs.own((try matmul(inputs.s, x, w.*, sc.*, bi.*)).?);
        try mlx.check(mlx.mlx_array_eval(got));
        try std.testing.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(want.*).?[0..@intCast(8 * n)], mlx.mlx_array_data_bfloat16(got).?[0..@intCast(8 * n)]);
        try std.testing.expect((try matmul(inputs.s, try inputs.slice(x, 1, 0, 3), w.*, sc.*, bi.*)) == null);
        try std.testing.expect((try matmul(inputs.s, try inputs.cast(x, .float32), w.*, sc.*, bi.*)) == null);
    }
}
