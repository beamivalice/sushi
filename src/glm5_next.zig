//! GLM-5.3-Flash forward primitives: mHC collapse/expand and the KDA recurrence.
const std = @import("std");
const mlx = @import("mlx.zig");

pub const HcResult = struct {
    mixed: mlx.mlx_array,
    post: mlx.mlx_array,
    comb: mlx.mlx_array,

    pub fn deinit(self: HcResult) void {
        _ = mlx.mlx_array_free(self.mixed);
        _ = mlx.mlx_array_free(self.post);
        _ = mlx.mlx_array_free(self.comb);
    }
};

var hc_collapse_kernel: ?mlx.mlx_fast_metal_kernel = null;
var hc_expand_kernel: ?mlx.mlx_fast_metal_kernel = null;

fn makeKernel(slot: *?mlx.mlx_fast_metal_kernel, name: [*:0]const u8, inputs: []const [*:0]const u8, outputs: []const [*:0]const u8, source: [:0]const u8) !mlx.mlx_fast_metal_kernel {
    if (slot.*) |value| return value;
    const iv = mlx.mlx_vector_string_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(outputs.ptr, outputs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const value = mlx.mlx_fast_metal_kernel_new(name, iv, ov, source, "", true, false);
    if (value.ctx == null) return error.MetalKernelCompileFailed;
    slot.* = value;
    return value;
}

fn applyKernel(kernel: mlx.mlx_fast_metal_kernel, inputs: []const mlx.mlx_array, config: mlx.mlx_fast_metal_kernel_config, s: mlx.mlx_stream) !mlx.mlx_vector_array {
    const iv = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    errdefer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kernel, iv, config, s));
    return ov;
}

const HC_COLLAPSE: [:0]const u8 =
    \\#pragma clang fp contract(off)
    \\#pragma clang fp reassociate(off)
    \\const uint row = threadgroup_position_in_grid.x;
    \\const uint tid = thread_position_in_threadgroup.x;
    \\threadgroup float pre[4];
    \\threadgroup float matrix[16];
    \\const float epsilon = float(eps);
    \\if (tid == 0) {
    \\  for (uint j = 0; j < 4; ++j) {
    \\    float z = mixes[row * 24 + j] * scale[0] + base[j];
    \\    pre[j] = 1.0f / (1.0f + precise::exp(-z)) + epsilon;
    \\    z = mixes[row * 24 + 4 + j] * scale[1] + base[4 + j];
    \\    post[row * 4 + j] = 2.0f / (1.0f + precise::exp(-z));
    \\  }
    \\  for (uint j = 0; j < 4; ++j) {
    \\    float maximum = -INFINITY;
    \\    for (uint k = 0; k < 4; ++k) {
    \\      uint i = j * 4 + k;
    \\      matrix[i] = mixes[row * 24 + 8 + i] * scale[2] + base[8 + i];
    \\      maximum = max(maximum, matrix[i]);
    \\    }
    \\    float sum = 0.0f;
    \\    for (uint k = 0; k < 4; ++k) { matrix[j * 4 + k] = precise::exp(matrix[j * 4 + k] - maximum); sum += matrix[j * 4 + k]; }
    \\    for (uint k = 0; k < 4; ++k) matrix[j * 4 + k] = matrix[j * 4 + k] / sum + epsilon;
    \\  }
    \\  for (uint iter = 0; iter < uint(ITERS); ++iter) {
    \\    if (iter != 0) {
    \\      for (uint j = 0; j < 4; ++j) {
    \\        float sum = 0.0f;
    \\        for (uint k = 0; k < 4; ++k) sum += matrix[j * 4 + k];
    \\        for (uint k = 0; k < 4; ++k) matrix[j * 4 + k] /= sum + epsilon;
    \\      }
    \\    }
    \\    for (uint k = 0; k < 4; ++k) {
    \\      float sum = 0.0f;
    \\      for (uint j = 0; j < 4; ++j) sum += matrix[j * 4 + k];
    \\      for (uint j = 0; j < 4; ++j) matrix[j * 4 + k] /= sum + epsilon;
    \\    }
    \\  }
    \\  for (uint i = 0; i < 16; ++i) comb[row * 16 + i] = matrix[i];
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\for (uint d = tid; d < uint(D); d += 256) {
    \\  float value = 0.0f;
    \\  for (uint j = 0; j < 4; ++j) value += pre[j] * float(x[(row * 4 + j) * uint(D) + d]);
    \\  mixed[row * uint(D) + d] = OutT(value);
    \\}
;

const HC_EXPAND: [:0]const u8 =
    \\#pragma clang fp contract(off)
    \\#pragma clang fp reassociate(off)
    \\const uint i = thread_position_in_grid.x;
    \\if (i >= threads_per_grid.x) return;
    \\const uint row = i / (4u * uint(D));
    \\const uint h = (i / uint(D)) % 4u;
    \\const uint d = i % uint(D);
    \\float value = 0.0f;
    \\for (uint j = 0; j < 4; ++j) value += comb[row * 16u + j * 4u + h] * float(residual[(row * 4u + j) * uint(D) + d]);
    \\const float product = post[row * 4u + h] * float(branch[row * uint(D) + d]);
    \\out[i] = OutT(product + value);
;

pub fn hcCollapse(x: mlx.mlx_array, mixes: mlx.mlx_array, scale: mlx.mlx_array, base: mlx.mlx_array, iters: c_int, epsilon: f32, s: mlx.mlx_stream) !HcResult {
    const sh = mlx.getShape(x);
    const dtype = mlx.mlx_array_dtype(x);
    if (sh.len != 4 or sh[0] <= 0 or sh[1] <= 0 or sh[2] != 4 or sh[3] <= 0 or iters < 1 or iters > 100 or !(epsilon > 0)) return error.InvalidHcShape;
    const rows = try std.math.mul(c_int, sh[0], sh[1]);
    if (mlx.mlx_array_size(mixes) != @as(usize, @intCast(rows)) * 24 or mlx.mlx_array_size(scale) != 3 or mlx.mlx_array_size(base) != 24) return error.InvalidHcShape;
    if ((dtype != .float32 and dtype != .bfloat16) or mlx.mlx_array_dtype(mixes) != .float32 or mlx.mlx_array_dtype(scale) != .float32 or mlx.mlx_array_dtype(base) != .float32) return error.InvalidHcDtype;
    const cooperative = @import("glm5_hc_collapse_simd32.zig");
    if (cooperative.enabled()) {
        if (try cooperative.collapse(x, mixes, scale, base, iters, epsilon, s)) |out| return out;
    }
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ sh[0], sh[1], sh[3] }, 3, dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ sh[0], sh[1], 4 }, 3, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ sh[0], sh[1], 4, 4 }, 4, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, rows * 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "OutT", dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", sh[3]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ITERS", iters));
    const eps = mlx.mlx_array_new_float(epsilon);
    defer _ = mlx.mlx_array_free(eps);
    const kernel = try makeKernel(&hc_collapse_kernel, "sushi_glm_hc_collapse", &.{ "x", "mixes", "scale", "base", "eps" }, &.{ "mixed", "post", "comb" }, HC_COLLAPSE);
    const output = try applyKernel(kernel, &.{ x, mixes, scale, base, eps }, cfg, s);
    defer _ = mlx.mlx_vector_array_free(output);
    var result = HcResult{ .mixed = mlx.mlx_array_new(), .post = mlx.mlx_array_new(), .comb = mlx.mlx_array_new() };
    errdefer result.deinit();
    try mlx.check(mlx.mlx_vector_array_get(&result.mixed, output, 0));
    try mlx.check(mlx.mlx_vector_array_get(&result.post, output, 1));
    try mlx.check(mlx.mlx_vector_array_get(&result.comb, output, 2));
    return result;
}

pub fn hcExpand(residual: mlx.mlx_array, branch: mlx.mlx_array, post: mlx.mlx_array, comb: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const sh = mlx.getShape(residual);
    if (sh.len != 4 or sh[2] != 4 or sh[0] <= 0 or sh[1] <= 0 or sh[3] <= 0) return error.InvalidHcShape;
    const rows = try std.math.mul(c_int, sh[0], sh[1]);
    const expected = [_]c_int{ sh[0], sh[1], sh[3] };
    if (!std.mem.eql(c_int, &expected, mlx.getShape(branch)) or mlx.mlx_array_size(post) != @as(usize, @intCast(rows)) * 4 or mlx.mlx_array_size(comb) != @as(usize, @intCast(rows)) * 16) return error.InvalidHcShape;
    const dtype = mlx.mlx_array_dtype(branch);
    if (mlx.mlx_array_dtype(residual) != dtype or mlx.mlx_array_dtype(post) != .float32 or mlx.mlx_array_dtype(comb) != .float32) return error.InvalidHcDtype;
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, sh.ptr, 4, dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, rows * 4 * sh[3], 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "OutT", dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", sh[3]));
    const kernel = try makeKernel(&hc_expand_kernel, "sushi_glm_hc_expand", &.{ "residual", "branch", "post", "comb" }, &.{"out"}, HC_EXPAND);
    const output = try applyKernel(kernel, &.{ residual, branch, post, comb }, cfg, s);
    defer _ = mlx.mlx_vector_array_free(output);
    var result = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(result);
    try mlx.check(mlx.mlx_vector_array_get(&result, output, 0));
    return result;
}

fn sigmoidF32(v: f32) f32 {
    return 1.0 / (1.0 + @exp(-v));
}

const HcSplit = struct { pre: [8]f32, post: [8]f32, comb: [64]f32 };

/// Scalar reference for the mHC Sinkhorn split: pre = σ(m·s0+b)+eps; post = 2σ(m·s1+b);
/// comb = row-softmax(+eps) → colnorm → (iters-1)×(rownorm, colnorm).
/// hc ≤ 8. Returns fixed-size buffers; read the first hc / hc² entries.
fn hcSplitSinkhorn(mixes: []const f32, hc_scale: []const f32, hc_base: []const f32, hc: usize, iters: u32, eps: f32) HcSplit {
    std.debug.assert(hc <= 8 and mixes.len >= (2 + hc) * hc);
    var out: HcSplit = .{ .pre = @splat(0), .post = @splat(0), .comb = @splat(0) };
    for (0..hc) |j| {
        out.pre[j] = sigmoidF32(mixes[j] * hc_scale[0] + hc_base[j]) + eps;
        out.post[j] = 2.0 * sigmoidF32(mixes[hc + j] * hc_scale[1] + hc_base[hc + j]);
    }
    var comb = out.comb[0 .. hc * hc];
    for (0..hc) |j| {
        for (0..hc) |k| comb[j * hc + k] = mixes[2 * hc + j * hc + k] * hc_scale[2] + hc_base[2 * hc + j * hc + k];
    }
    // row softmax + eps
    for (0..hc) |j| {
        var m: f32 = -std.math.inf(f32);
        for (comb[j * hc ..][0..hc]) |v| m = @max(m, v);
        var sum: f32 = 0;
        for (comb[j * hc ..][0..hc]) |*v| {
            v.* = @exp(v.* - m);
            sum += v.*;
        }
        for (comb[j * hc ..][0..hc]) |*v| v.* = v.* / sum + eps;
    }
    var it: u32 = 0;
    while (it < iters) : (it += 1) {
        if (it > 0) {
            // row normalize (skipped on the first pass — softmax already did)
            for (0..hc) |j| {
                var sum: f32 = 0;
                for (comb[j * hc ..][0..hc]) |v| sum += v;
                for (comb[j * hc ..][0..hc]) |*v| v.* /= (sum + eps);
            }
        }
        // column normalize
        for (0..hc) |k| {
            var sum: f32 = 0;
            for (0..hc) |j| sum += comb[j * hc + k];
            for (0..hc) |j| comb[j * hc + k] /= (sum + eps);
        }
    }
    return out;
}

test "GLM mHC collapse and expand agree with scalar Sinkhorn" {
    const s = mlx.gpuStream();
    const rows = 3;
    const dim = 32;
    var x: [rows * 4 * dim]f32 = undefined;
    var mix: [rows * 24]f32 = undefined;
    var base: [24]f32 = undefined;
    const scale = [_]f32{ 0.7, 1.2, 0.4 };
    for (&x, 0..) |*v, i| v.* = (@as(f32, @floatFromInt(i % 29)) - 14) / 8;
    for (&mix, 0..) |*v, i| v.* = (@as(f32, @floatFromInt(i % 19)) - 9) / 4;
    for (&base, 0..) |*v, i| v.* = (@as(f32, @floatFromInt(i % 7)) - 3) / 16;
    const xa = mlx.mlx_array_new_data(&x, &[_]c_int{ 1, rows, 4, dim }, 4, .float32);
    defer _ = mlx.mlx_array_free(xa);
    const ma = mlx.mlx_array_new_data(&mix, &[_]c_int{ 1, rows, 24 }, 3, .float32);
    defer _ = mlx.mlx_array_free(ma);
    const sa = mlx.mlx_array_new_data(&scale, &[_]c_int{3}, 1, .float32);
    defer _ = mlx.mlx_array_free(sa);
    const ba = mlx.mlx_array_new_data(&base, &[_]c_int{24}, 1, .float32);
    defer _ = mlx.mlx_array_free(ba);
    const pre = try hcCollapse(xa, ma, sa, ba, 20, 1e-6, s);
    defer pre.deinit();
    const expanded = try hcExpand(xa, pre.mixed, pre.post, pre.comb, s);
    defer _ = mlx.mlx_array_free(expanded);
    try mlx.check(mlx.mlx_array_eval(pre.mixed));
    try mlx.check(mlx.mlx_array_eval(pre.post));
    try mlx.check(mlx.mlx_array_eval(pre.comb));
    try mlx.check(mlx.mlx_array_eval(expanded));
    const mixed = mlx.mlx_array_data_float32(pre.mixed).?;
    const post = mlx.mlx_array_data_float32(pre.post).?;
    const comb = mlx.mlx_array_data_float32(pre.comb).?;
    const output = mlx.mlx_array_data_float32(expanded).?;
    for (0..rows) |row| {
        const ref = hcSplitSinkhorn(mix[row * 24 ..][0..24], &scale, &base, 4, 20, 1e-6);
        for (0..4) |i| try std.testing.expectApproxEqAbs(ref.post[i], post[row * 4 + i], 1e-6);
        for (0..16) |i| try std.testing.expectApproxEqAbs(ref.comb[i], comb[row * 16 + i], 1e-6);
        for (0..dim) |d| {
            var want: f32 = 0;
            for (0..4) |j| want += ref.pre[j] * x[(row * 4 + j) * dim + d];
            try std.testing.expectApproxEqAbs(want, mixed[row * dim + d], 2e-6);
            for (0..4) |i| {
                var expanded_want = ref.post[i] * want;
                for (0..4) |j| expanded_want += ref.comb[j * 4 + i] * x[(row * 4 + j) * dim + d];
                try std.testing.expectApproxEqAbs(expanded_want, output[(row * 4 + i) * dim + d], 3e-6);
            }
        }
    }
}

test "GLM EXL3 clamp supports 2 to 4 bpw in decode and prefill" {
    const api = @import("sushi_exl3");
    const fmt = api.format;
    const a = std.testing.allocator;
    const s = mlx.gpuStream();
    const dim = 128;
    const experts = 3;
    const topk = 2;
    const dec = fmt.Decode{ .codebook = .mcg, .window = .w12 };
    for (0..17) |rate_index| {
        const n = 32 + rate_index * 2;
        const rate = fmt.Rate{ .n = @intCast(n) };
        const per = 64 * n;
        var codes: [3][]u16 = undefined;
        var su: [3][experts * dim]u16 = undefined;
        var sv: [3][experts * dim]u16 = undefined;
        var arrays: [9]mlx.mlx_array = undefined;
        var made: usize = 0;
        defer for (arrays[0..made]) |value| {
            _ = mlx.mlx_array_free(value);
        };
        var allocated: usize = 0;
        defer for (codes[0..allocated]) |value| a.free(value);
        var random = std.Random.DefaultPrng.init(517 + n);
        for (0..3) |p| {
            codes[p] = try a.alloc(u16, experts * per);
            allocated += 1;
            for (codes[p]) |*value| value.* = random.random().int(u16);
            @memset(&su[p], fmt.f32ToF16Bits(0.25));
            @memset(&sv[p], fmt.f32ToF16Bits(if (p == 2) 0.015625 else 0.25));
            arrays[p * 3] = mlx.mlx_array_new_data(codes[p].ptr, &[_]c_int{ experts, 8, 8, @intCast(n) }, 4, .uint16);
            arrays[p * 3 + 1] = mlx.mlx_array_new_data(&su[p], &[_]c_int{ experts, dim }, 2, .float16);
            arrays[p * 3 + 2] = mlx.mlx_array_new_data(&sv[p], &[_]c_int{ experts, dim }, 2, .float16);
            made += 3;
        }
        const bank = api.Bank{
            .gate = .{ .trellis = arrays[0], .suh = arrays[1], .svh = arrays[2] },
            .up = .{ .trellis = arrays[3], .suh = arrays[4], .svh = arrays[5] },
            .down = .{ .trellis = arrays[6], .suh = arrays[7], .svh = arrays[8] },
        };
        for ([_]usize{ 1, 17 }) |rows| {
            const x = try a.alloc(f32, rows * dim);
            defer a.free(x);
            const ids = try a.alloc(u32, rows * topk);
            defer a.free(ids);
            const scores = try a.alloc(f32, rows * topk);
            defer a.free(scores);
            for (x, 0..) |*value, i| value.* = @as(f32, @floatFromInt(i % 31)) - 15;
            for (ids, scores, 0..) |*id, *score, i| {
                id.* = @intCast(i % experts);
                score.* = if (i % topk == 0) 1 else 1.5;
            }
            const xa = mlx.mlx_array_new_data(x.ptr, &[_]c_int{ 1, @intCast(rows), dim }, 3, .float32);
            defer _ = mlx.mlx_array_free(xa);
            const ia = mlx.mlx_array_new_data(ids.ptr, &[_]c_int{ 1, @intCast(rows), topk }, 3, .uint32);
            defer _ = mlx.mlx_array_free(ia);
            const sa = mlx.mlx_array_new_data(scores.ptr, &[_]c_int{ 1, @intCast(rows), topk }, 3, .float32);
            defer _ = mlx.mlx_array_free(sa);
            const got = try api.moeClamped(s, xa, bank, ia, sa, dec, 10);
            defer _ = mlx.mlx_array_free(got);
            try mlx.check(mlx.mlx_array_eval(got));
            const actual = mlx.mlx_array_data_float32(got).?[0 .. rows * dim];
            var clipped: usize = 0;
            for (0..rows) |row| {
                var expected: [dim]f32 = @splat(0);
                for (0..topk) |slot| {
                    const eid = ids[row * topk + slot];
                    var tmp: [dim]f32 = undefined;
                    var inner: [dim]f32 = undefined;
                    var gate: [dim]f32 = undefined;
                    var up: [dim]f32 = undefined;
                    var down: [dim]f32 = undefined;
                    for (0..2) |p| fmt.project(x[row * dim ..][0..dim], codes[p][eid * per ..][0..per], su[p][eid * dim ..][0..dim], sv[p][eid * dim ..][0..dim], dim, dim, rate, dec, &tmp, &inner, if (p == 0) &gate else &up);
                    for (&gate, up) |*g, u| {
                        if (g.* > 10 or @abs(u) > 10) clipped += 1;
                        const cg = @min(g.*, 10);
                        g.* = cg / (1 + @exp(-cg)) * std.math.clamp(u, -10, 10);
                    }
                    fmt.project(&gate, codes[2][eid * per ..][0..per], su[2][eid * dim ..][0..dim], sv[2][eid * dim ..][0..dim], dim, dim, rate, dec, &tmp, &inner, &down);
                    for (&expected, down) |*value, d| value.* += d * scores[row * topk + slot];
                }
                for (expected, actual[row * dim ..][0..dim]) |want, value| try std.testing.expectApproxEqAbs(want, value, 0.025 + @abs(want) * 0.005);
            }
            try std.testing.expect(clipped > 0);
        }
    }
}

/// Q/K are already L2-normalized (Q also scaled by Dk^-0.5).
/// Decay is exp(log-gate), not the forget projection or log-gate itself.
pub const KdaInputs = struct {
    q: mlx.mlx_array, // [B,T,H,Dk]
    k: mlx.mlx_array,
    v: mlx.mlx_array, // [B,T,H,Dv]
    decay: mlx.mlx_array, // f32 [B,T,H,Dk]
    beta: mlx.mlx_array, // [B,T,H]
    state: mlx.mlx_array, // f32 [B,H,Dv,Dk]
};

pub const KdaResult = struct {
    y: mlx.mlx_array,
    state: mlx.mlx_array,

    pub fn deinit(self: KdaResult) void {
        _ = mlx.mlx_array_free(self.y);
        _ = mlx.mlx_array_free(self.state);
    }
};

pub fn kda(in: KdaInputs, s: mlx.mlx_stream) !KdaResult {
    if (!mlx.streamIsGpu(s)) return error.KdaGpuRequired;
    const q = mlx.getShape(in.q);
    const v = mlx.getShape(in.v);
    if (q.len != 4 or v.len != 4 or !std.mem.eql(c_int, q, mlx.getShape(in.k)) or
        !std.mem.eql(c_int, q, mlx.getShape(in.decay))) return error.InvalidKdaShape;
    if (q[0] <= 0 or q[1] <= 0 or q[2] <= 0 or q[3] <= 0 or @mod(q[3], 32) != 0 or
        v[3] <= 0 or @mod(v[3], 4) != 0 or !std.mem.eql(c_int, q[0..3], v[0..3]) or
        !std.mem.eql(c_int, q[0..3], mlx.getShape(in.beta))) return error.InvalidKdaShape;
    const state_shape = [_]c_int{ q[0], q[2], v[3], q[3] };
    if (!std.mem.eql(c_int, &state_shape, mlx.getShape(in.state))) return error.InvalidKdaShape;
    const dtype = mlx.mlx_array_dtype(in.q);
    if ((dtype != .bfloat16 and dtype != .float32) or mlx.mlx_array_dtype(in.k) != dtype or
        mlx.mlx_array_dtype(in.v) != dtype or mlx.mlx_array_dtype(in.state) != .float32 or
        mlx.mlx_array_dtype(in.decay) != .float32 or
        (mlx.mlx_array_dtype(in.beta) != .float32 and mlx.mlx_array_dtype(in.beta) != .bfloat16)) return error.InvalidKdaDtype;
    const heads = try std.math.mul(c_int, q[0], q[2]);
    const config = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, v.ptr, 4, dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &state_shape, 4, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32, v[3], heads));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, 4, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "InT", dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "OutT", dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "StT", .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dk", q[3]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dv", v[3]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hk", q[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hv", q[2]));
    const length = mlx.mlx_array_new_int(q[1]);
    defer _ = mlx.mlx_array_free(length);
    const arrays = [_]mlx.mlx_array{ in.q, in.k, in.v, in.decay, in.beta, in.state, length };
    const inputs = mlx.mlx_vector_array_new_data(&arrays, arrays.len);
    defer _ = mlx.mlx_vector_array_free(inputs);
    var outputs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs);
    const kernel = try @import("transformer.zig").getGdnKernel(true);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs, kernel, inputs, config, s));
    var result = KdaResult{ .y = mlx.mlx_array_new(), .state = mlx.mlx_array_new() };
    errdefer result.deinit();
    try mlx.check(mlx.mlx_vector_array_get(&result.y, outputs, 0));
    try mlx.check(mlx.mlx_vector_array_get(&result.state, outputs, 1));
    return result;
}

test "GLM KDA per-channel decay preserves FP32 state and agrees with a scalar reference" {
    const t = std.testing;
    const B = 2;
    const T = 3;
    const H = 2;
    const DK = 128;
    const DV = 4;
    var q: [B * T * H * DK]f32 = undefined;
    var k: @TypeOf(q) = undefined;
    var decay: @TypeOf(q) = undefined;
    var v: [B * T * H * DV]f32 = undefined;
    var beta: [B * T * H]f32 = undefined;
    var state: [B * H * DV * DK]f32 = undefined;
    for (&q, &k, &decay, 0..) |*a, *b, *g, i| {
        a.* = (@as(f32, @floatFromInt(i % 17)) - 8) / 128;
        b.* = (@as(f32, @floatFromInt(i % 13)) - 6) / 32;
        g.* = 0.2 + @as(f32, @floatFromInt(i % 19)) / 32;
    }
    for (&v, 0..) |*x, i| x.* = (@as(f32, @floatFromInt(i % 11)) - 5) / 8;
    for (&beta, 0..) |*x, i| x.* = 0.3 + @as(f32, @floatFromInt(i % 7)) / 16;
    for (&state, 0..) |*x, i| x.* = 0.00123 * @as(f32, @floatFromInt(i % 23));
    var expected_state: [state.len]f64 = undefined;
    for (state, &expected_state) |x, *out| out.* = x;
    var expected_y: [v.len]f64 = undefined;
    for (0..B) |b| for (0..T) |pos| for (0..H) |h| {
        const row = (b * T + pos) * H + h;
        for (0..DV) |dv| {
            const base = ((b * H + h) * DV + dv) * DK;
            var memory: f64 = 0;
            for (0..DK) |dk| {
                expected_state[base + dk] *= decay[row * DK + dk];
                memory += expected_state[base + dk] * k[row * DK + dk];
            }
            const delta = (@as(f64, v[row * DV + dv]) - memory) * beta[row];
            var out: f64 = 0;
            for (0..DK) |dk| {
                expected_state[base + dk] += k[row * DK + dk] * delta;
                out += expected_state[base + dk] * q[row * DK + dk];
            }
            expected_y[row * DV + dv] = out;
        }
    };
    const shape = [_]c_int{ B, T, H, DK };
    const arrs = [_]mlx.mlx_array{
        mlx.mlx_array_new_data(&q, &shape, 4, .float32),
        mlx.mlx_array_new_data(&k, &shape, 4, .float32),
        mlx.mlx_array_new_data(&v, &[_]c_int{ B, T, H, DV }, 4, .float32),
        mlx.mlx_array_new_data(&decay, &shape, 4, .float32),
        mlx.mlx_array_new_data(&beta, &[_]c_int{ B, T, H }, 3, .float32),
        mlx.mlx_array_new_data(&state, &[_]c_int{ B, H, DV, DK }, 4, .float32),
    };
    defer for (arrs) |a| {
        _ = mlx.mlx_array_free(a);
    };
    const s = mlx.gpuStream();
    const inputs = KdaInputs{ .q = arrs[0], .k = arrs[1], .v = arrs[2], .decay = arrs[3], .beta = arrs[4], .state = arrs[5] };
    const result = try kda(inputs, s);
    defer result.deinit();
    try mlx.check(mlx.mlx_array_eval(result.y));
    try mlx.check(mlx.mlx_array_eval(result.state));
    try t.expectEqual(mlx.mlx_dtype.float32, mlx.mlx_array_dtype(result.state));
    for (expected_y, mlx.mlx_array_data_float32(result.y).?[0..v.len]) |expected, actual| try t.expectApproxEqAbs(expected, actual, 1e-6);
    for (expected_state, mlx.mlx_array_data_float32(result.state).?[0..state.len]) |expected, actual| try t.expectApproxEqAbs(expected, actual, 1e-6);
}

fn expectBitsEqual(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !void {
    var ac = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ac);
    var bc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bc);
    try mlx.check(mlx.mlx_contiguous(&ac, a, false, s));
    try mlx.check(mlx.mlx_contiguous(&bc, b, false, s));
    try mlx.check(mlx.mlx_array_eval(ac));
    try mlx.check(mlx.mlx_array_eval(bc));
    try std.testing.expectEqual(mlx.mlx_array_dtype(ac), mlx.mlx_array_dtype(bc));
    try std.testing.expectEqual(mlx.mlx_array_size(ac), mlx.mlx_array_size(bc));
    if (mlx.mlx_array_dtype(ac) == .float32) {
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(mlx.mlx_array_data_float32(ac).?[0..mlx.mlx_array_size(ac)]), std.mem.sliceAsBytes(mlx.mlx_array_data_float32(bc).?[0..mlx.mlx_array_size(bc)]));
    } else {
        try std.testing.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(ac).?[0..mlx.mlx_array_size(ac)], mlx.mlx_array_data_bfloat16(bc).?[0..mlx.mlx_array_size(bc)]);
    }
}

fn timeSlice(a: mlx.mlx_array, pos: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const shape = mlx.getShape(a);
    var starts: [4]c_int = @splat(0);
    var stops: [4]c_int = @splat(1);
    const steps: [4]c_int = @splat(1);
    @memcpy(stops[0..shape.len], shape);
    starts[1] = pos;
    stops[1] = pos + 1;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_slice(&out, a, &starts, shape.len, &stops, shape.len, &steps, shape.len, s));
    return out;
}

test "GLM KDA chunked and serial execution preserve identical outputs and FP32 states" {
    const t = std.testing;
    const s = mlx.gpuStream();
    const shape = [_]c_int{ 2, 3, 2, 128 };
    var values: [2 * 3 * 2 * 128]f32 = undefined;
    for (&values, 0..) |*v, i| v.* = (@as(f32, @floatFromInt(i % 31)) - 15) / 64;
    const source = mlx.mlx_array_new_data(&values, &shape, 4, .float32);
    defer _ = mlx.mlx_array_free(source);
    var decay_values: [values.len]f32 = undefined;
    for (&decay_values, 0..) |*v, i| v.* = 0.1 + @as(f32, @floatFromInt(i % 29)) / 32;
    const decay = mlx.mlx_array_new_data(&decay_values, &shape, 4, .float32);
    defer _ = mlx.mlx_array_free(decay);
    const beta_values: [2 * 3 * 2]f32 = @splat(0.75);
    const beta = mlx.mlx_array_new_data(&beta_values, &[_]c_int{ 2, 3, 2 }, 3, .float32);
    defer _ = mlx.mlx_array_free(beta);
    var state_values: [2 * 2 * 128 * 128]f32 = undefined;
    for (&state_values, 0..) |*v, i| v.* = 0.000123 * @as(f32, @floatFromInt(i % 53));
    const initial = mlx.mlx_array_new_data(&state_values, &[_]c_int{ 2, 2, 128, 128 }, 4, .float32);
    defer _ = mlx.mlx_array_free(initial);
    for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dtype| {
        var input = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(input);
        try mlx.check(mlx.mlx_astype(&input, source, dtype, s));
        const args = KdaInputs{ .q = input, .k = input, .v = input, .decay = decay, .beta = beta, .state = initial };
        const full = try kda(args, s);
        defer full.deinit();
        var serial = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(serial);
        try mlx.check(mlx.mlx_array_set(&serial, initial));
        for (0..3) |pos| {
            const qi = try timeSlice(input, @intCast(pos), s);
            defer _ = mlx.mlx_array_free(qi);
            const gi = try timeSlice(decay, @intCast(pos), s);
            defer _ = mlx.mlx_array_free(gi);
            const bi = try timeSlice(beta, @intCast(pos), s);
            defer _ = mlx.mlx_array_free(bi);
            const step = try kda(.{ .q = qi, .k = qi, .v = qi, .decay = gi, .beta = bi, .state = serial }, s);
            defer step.deinit();
            const expected = try timeSlice(full.y, @intCast(pos), s);
            defer _ = mlx.mlx_array_free(expected);
            try expectBitsEqual(expected, step.y, s);
            try t.expectEqual(mlx.mlx_dtype.float32, mlx.mlx_array_dtype(step.state));
            try mlx.check(mlx.mlx_array_set(&serial, step.state));
        }
        try expectBitsEqual(full.state, serial, s);
        var rounded_state = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(rounded_state);
        try mlx.check(mlx.mlx_astype(&rounded_state, initial, .bfloat16, s));
        var invalid = args;
        invalid.state = rounded_state;
        try t.expectError(error.InvalidKdaDtype, kda(invalid, s));
        invalid = args;
        invalid.decay = beta;
        try t.expectError(error.InvalidKdaShape, kda(invalid, s));
    }
}

test "GLM HC expansion rounds residual contraction before branch addition" {
    const stream = mlx.gpuStream();
    const residual_data = [_]f32{ 0.041748046875, 2.828125, 0.07666015625, -1.265625 };
    const coefficients = [_]f32{ 0.3550497889518738, 0.7479683756828308, 0.7366215586662292, 0.0021052767988294363 };
    var comb_data: [16]f32 = undefined;
    for (0..4) |j| for (0..4) |i| {
        comb_data[j * 4 + i] = coefficients[j];
    };
    const post_data: [4]f32 = @splat(1.5905669927597046);
    const branch_data = [_]f32{-1.4140625};
    const rf = mlx.mlx_array_new_data(&residual_data, &[_]c_int{ 1, 1, 4, 1 }, 4, .float32);
    defer _ = mlx.mlx_array_free(rf);
    const bf = mlx.mlx_array_new_data(&branch_data, &[_]c_int{ 1, 1, 1 }, 3, .float32);
    defer _ = mlx.mlx_array_free(bf);
    var residual = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(residual);
    var branch = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(branch);
    try mlx.check(mlx.mlx_astype(&residual, rf, .bfloat16, stream));
    try mlx.check(mlx.mlx_astype(&branch, bf, .bfloat16, stream));
    const post = mlx.mlx_array_new_data(&post_data, &[_]c_int{ 1, 1, 4 }, 3, .float32);
    defer _ = mlx.mlx_array_free(post);
    const comb = mlx.mlx_array_new_data(&comb_data, &[_]c_int{ 1, 1, 4, 4 }, 4, .float32);
    defer _ = mlx.mlx_array_free(comb);
    const result = try hcExpand(residual, branch, post, comb, stream);
    defer _ = mlx.mlx_array_free(result);
    var out = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_astype(&out, result, .float32, stream));
    try mlx.check(mlx.mlx_array_eval(out));
    // Independently rounded FP32 residual dot product plus FP32 branch product,
    // then BF16 round-to-nearest-even. Starting the sum with the branch differs.
    for (mlx.mlx_array_data_float32(out).?[0..4]) |v| try std.testing.expectEqual(@as(f32, -0.0654296875), v);
}
