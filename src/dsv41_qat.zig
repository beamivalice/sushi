//! deepseek_v41 numerics that have no MLX op: the training-time activation round trips
//! (`act_quant` / `fp4_act_quant` in DeepSeek's `inference/kernel.py`), the hyper-connection split,
//! and rotary tables (`precompute_freqs_cis`) with the rotation itself.
const std = @import("std");
const mlx = @import("mlx.zig");
const sc_mod = @import("dsv41_scope.zig");
const A = sc_mod.A;
const Scope = sc_mod.Scope;
const dim = sc_mod.dim;
const ci = sc_mod.ci;
const evalAll = sc_mod.evalAll;

// ── fake quantization ────────────────────────────────────────────────────

/// fp8 e4m3 with a power-of-two scale per group, fp4 e2m1 with a power-of-two scale, fp4 e2m1 with an
/// e4m3 scale (the compressed KV, groups of 16).
pub const Qat = enum(c_int) { fp8_ue8m0 = 0, fp4_ue8m0 = 1, fp4_e4m3 = 2 };

pub const QAT_HEADER: [:0]const u8 =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\inline float round_scale(float v) {
    \\  uint b = as_type<uint>(v);
    \\  int e = int((b >> 23) & 0xFFu) - 127 + ((b & 0x7FFFFFu) != 0u ? 1 : 0);
    \\  return as_type<float>(uint(clamp(e + 127, 1, 254)) << 23);
    \\}
    \\inline float round_e4m3(float v) {
    \\  if (v == 0.0f) return 0.0f;
    \\  int e = max(int((as_type<uint>(v) >> 23) & 0xFFu) - 127, -6);
    \\  float step = as_type<float>(uint(e - 3 + 127) << 23);
    \\  return min(rint(v / step) * step, 448.0f);
    \\}
    \\inline float round_e2m1(float a) {
    \\  float step = a < 2.0f ? 0.5f : (a < 4.0f ? 1.0f : 2.0f);
    \\  return min(rint(a / step) * step, 6.0f);
    \\}
    \\template <int MODE>
    \\inline float qat_scale(float amax) {
    \\  if (MODE == 0) return round_scale(max(amax, 1e-4f) * (1.0f / 448.0f));
    \\  if (MODE == 1) return round_scale(max(amax, 6.0f * 1.1754943508222875e-38f) * (1.0f / 6.0f));
    \\  return round_e4m3(max(amax, 6.0f * 0.001953125f) / 6.0f);
    \\}
    \\template <int MODE>
    \\inline float qat_round(float v, float s) {
    \\  if (MODE == 0) {
    \\    const float t = clamp(precise::divide(v, s), -448.0f, 448.0f);
    \\    return copysign(round_e4m3(fabs(t)), t) * s;
    \\  }
    \\  const float t = clamp(precise::divide(v, s), -6.0f, 6.0f);
    \\  return copysign(round_e2m1(fabs(t)), t) * s;
    \\}
;

const QAT_SOURCE: [:0]const u8 =
    \\#pragma clang fp contract(off)
    \\#pragma clang fp reassociate(off)
    \\const uint gid = thread_position_in_grid.x;
    \\if (gid >= threads_per_grid.x) return;
    \\const uint base = gid * uint(G);
    \\float amax = 0.0f;
    \\for (uint k = 0; k < uint(G); ++k) amax = max(amax, fabs(float(x[base + k])));
    \\const float s = qat_scale<MODE>(amax);
    \\for (uint k = 0; k < uint(G); ++k) y[base + k] = static_cast<T>(qat_round<MODE>(float(x[base + k]), s));
;

var qat_kernel: ?mlx.mlx_fast_metal_kernel = null;

pub fn kernelNew(slot: *?mlx.mlx_fast_metal_kernel, name: [*:0]const u8, inputs: []const [*:0]const u8, outputs: []const [*:0]const u8, source: [:0]const u8, header: [:0]const u8) !mlx.mlx_fast_metal_kernel {
    if (slot.*) |k| return k;
    const iv = mlx.mlx_vector_string_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(outputs.ptr, outputs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new(name, iv, ov, source, header, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    slot.* = k;
    return k;
}

pub fn applyKernel(sc: *Scope, k: mlx.mlx_fast_metal_kernel, inputs: []const A, cfg: mlx.mlx_fast_metal_kernel_config, outs: []A) !void {
    const iv = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    // A Metal kernel runs on the GPU stream even when the scope's ops run on the CPU stream (the tests' exact one).
    const gpu = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(gpu);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, k, iv, cfg, if (mlx.streamIsGpu(sc.s)) sc.s else gpu));
    for (outs, 0..) |*o, i| {
        o.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(o, ov, i));
        _ = try sc.keep(o.*);
    }
}

/// `x` (any shape whose last dim is a multiple of `group`) through the round trip, in its own dtype.
pub fn qat(sc: *Scope, x: A, mode: Qat, group: usize) !A {
    const n = mlx.mlx_array_size(x);
    std.debug.assert(n % group == 0);
    const k = try kernelNew(&qat_kernel, "dsv41_qat", &.{"x"}, &.{"y"}, QAT_SOURCE, QAT_HEADER);
    const dt = mlx.mlx_array_dtype(x);
    const ng = n / group;
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const sh = mlx.getShape(x);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, sh.ptr, sh.len, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ci(ng), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, ci(@min(ng, 256)), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "G", ci(group)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "MODE", @intFromEnum(mode)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
    var o: [1]A = undefined;
    try applyKernel(sc, k, &.{x}, cfg, &o);
    return o[0];
}

// ── packed FP4 rows ──────────────────────────────────────────────────────

/// The stored form of an FP4 round trip's output: E2M1 codes two to a byte (low nibble first, bit 3 the sign) and one
/// scale byte per group, E8M0 (`fp4_ue8m0`) or E4M3 (`fp4_e4m3`). `unpack` gives back the round trip's values bit for
/// bit: each is a code times its group's scale, which f32 holds exactly and the activation dtype too.
pub const Packed = struct { codes: A, scales: A };

pub const PACK_HEADER: [:0]const u8 =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\constant float E2M1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    \\inline float scale_of(uint b, int MODE) {
    \\  if (MODE == 1) return as_type<float>(clamp(b, 1u, 254u) << 23);
    \\  const uint e = (b >> 3) & 15u, m = b & 7u;
    \\  return e == 0u ? float(m) * 0.001953125f : as_type<float>((e + 120u) << 23 | m << 20);
    \\}
    \\// The scale byte that holds `s` exactly, or 256.
    \\inline uint scale_byte(float s, int MODE) {
    \\  const uint bits = as_type<uint>(s);
    \\  const int e = int((bits >> 23) & 0xFFu) - 127;
    \\  if (MODE == 1) return ((bits & 0x807FFFFFu) == 0u && e >= -126 && e <= 127) ? uint(e + 127) : 256u;
    \\  if (s <= 0.0f || s > 448.0f) return 256u;
    \\  if (e >= -6) {
    \\    if ((bits & 0xFFFFFu) != 0u) return 256u;
    \\    return uint(e + 7) << 3 | ((bits >> 20) & 7u);
    \\  }
    \\  const float k = s * 512.0f;
    \\  return (k == rint(k) && k >= 1.0f && k <= 7.0f) ? uint(k) : 256u;
    \\}
    \\// The E2M1 code of `t`, or 16 when `t` is not one.
    \\inline uint code_of(float t) {
    \\  const float a = fabs(t);
    \\  for (uint i = 0; i < 8u; ++i) if (a == E2M1[i]) return i | (signbit(t) ? 8u : 0u);
    \\  return 16u;
    \\}
;

const PACK_SOURCE: [:0]const u8 =
    \\#pragma clang fp contract(off)
    \\const uint gid = thread_position_in_grid.x;
    \\if (gid >= uint(x_shape[0]) * uint(x_shape[1]) / uint(G)) return;
    \\const uint base = gid * uint(G);
    \\float amax = 0.0f;
    \\for (uint k = 0; k < uint(G); ++k) amax = max(amax, fabs(float(x[base + k])));
    \\const float cands[7] = {6.0f, 4.0f, 3.0f, 2.0f, 1.5f, 1.0f, 0.5f};
    \\uint sb = MODE == 1 ? 127u : 0x38u;
    \\if (amax > 0.0f) {
    \\  for (uint c = 0; c < 7u; ++c) {
    \\    const uint b = scale_byte(precise::divide(amax, cands[c]), MODE);
    \\    if (b > 255u) continue;
    \\    const float s = scale_of(b, MODE);
    \\    bool exact = true;
    \\    for (uint k = 0; k < uint(G) && exact; ++k) {
    \\      const float v = float(x[base + k]);
    \\      const uint code = code_of(precise::divide(v, s));
    \\      exact = code < 16u && E2M1[code & 7u] * s == fabs(v);
    \\    }
    \\    if (exact) { sb = b; break; }
    \\  }
    \\}
    \\scales[gid] = uchar(sb);
    \\const float s = scale_of(sb, MODE);
    \\for (uint k = 0; k < uint(G); k += 2u) {
    \\  uint lo = code_of(precise::divide(float(x[base + k]), s));
    \\  uint hi = code_of(precise::divide(float(x[base + k + 1u]), s));
    \\  codes[(base + k) / 2u] = uchar((lo & 15u) | (hi & 15u) << 4);
    \\}
;

const UNPACK_SOURCE: [:0]const u8 =
    \\#pragma clang fp contract(off)
    \\const uint i = thread_position_in_grid.x;
    \\if (i >= uint(codes_shape[0]) * uint(codes_shape[1]) * 2u) return;
    \\const uint byte = uint(codes[i / 2u]);
    \\const uint code = (i & 1u) ? byte >> 4 : byte & 15u;
    \\const float s = scale_of(uint(scales[i / uint(G)]), MODE);
    \\const float v = E2M1[code & 7u] * s;
    \\y[i] = static_cast<T>((code & 8u) ? -v : v);
;

var pack_kernel: ?mlx.mlx_fast_metal_kernel = null;
var unpack_kernel: ?mlx.mlx_fast_metal_kernel = null;

pub fn packMode(mode: Qat) c_int {
    return switch (mode) {
        .fp4_ue8m0 => 1,
        .fp4_e4m3 => 2,
        .fp8_ue8m0 => unreachable,
    };
}

/// Rows `x` `[n, w]` of an FP4 round trip (`mode`, `group`) in their stored form: codes `[n, w/2]`, scales `[n, w/group]`.
pub fn packFp4(sc: *Scope, x: A, mode: Qat, group: usize) !Packed {
    const n = dim(x, 0);
    const w = dim(x, 1);
    std.debug.assert(w % group == 0 and group % 2 == 0);
    const k = try kernelNew(&pack_kernel, "dsv41_pack_fp4", &.{"x"}, &.{ "codes", "scales" }, PACK_SOURCE, PACK_HEADER);
    const ng = n * w / group;
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(n), ci(w / 2) }, 2, .uint8));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(n), ci(w / group) }, 2, .uint8));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ci(@max(ng, 1)), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, ci(@min(@max(ng, 1), 256)), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "G", ci(group)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "MODE", packMode(mode)));
    var o: [2]A = undefined;
    try applyKernel(sc, k, &.{x}, cfg, &o);
    return .{ .codes = o[0], .scales = o[1] };
}

/// The values of packed rows (`codes` `[n, w/2]`, `scales` `[n, w/group]`) as `[n, w]` in `dt`.
pub fn unpackFp4(sc: *Scope, p: Packed, mode: Qat, group: usize, dt: mlx.mlx_dtype) !A {
    const n = dim(p.codes, 0);
    const w = dim(p.codes, 1) * 2;
    const k = try kernelNew(&unpack_kernel, "dsv41_unpack_fp4", &.{ "codes", "scales" }, &.{"y"}, UNPACK_SOURCE, PACK_HEADER);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(n), ci(w) }, 2, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ci(@max(n * w, 1)), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "G", ci(group)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "MODE", packMode(mode)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
    var o: [1]A = undefined;
    try applyKernel(sc, k, &.{ p.codes, p.scales }, cfg, &o);
    return o[0];
}

// ── hyper-connection split ───────────────────────────────────────────────

const HC_SPLIT_SOURCE: [:0]const u8 =
    \\#pragma clang fp contract(off)
    \\#pragma clang fp reassociate(off)
    \\const uint row = thread_position_in_grid.x;
    \\if (row >= threads_per_grid.x) return;
    \\const float e = eps[0];
    \\float m[16];
    \\for (uint j = 0; j < 4; ++j) {
    \\  float z = mixes[row * 24 + j] * scale[0] + base[j];
    \\  pre[row * 4 + j] = 1.0f / (1.0f + precise::exp(-z)) + e;
    \\  z = mixes[row * 24 + 4 + j] * scale[1] + base[4 + j];
    \\  post[row * 4 + j] = 2.0f / (1.0f + precise::exp(-z));
    \\}
    \\for (uint j = 0; j < 4; ++j) {
    \\  float mx = -INFINITY;
    \\  for (uint k = 0; k < 4; ++k) {
    \\    m[j * 4 + k] = mixes[row * 24 + 8 + j * 4 + k] * scale[2] + base[8 + j * 4 + k];
    \\    mx = max(mx, m[j * 4 + k]);
    \\  }
    \\  float sum = 0.0f;
    \\  for (uint k = 0; k < 4; ++k) { m[j * 4 + k] = precise::exp(m[j * 4 + k] - mx); sum += m[j * 4 + k]; }
    \\  for (uint k = 0; k < 4; ++k) m[j * 4 + k] = m[j * 4 + k] / sum + e;
    \\}
    \\for (uint it = 0; it < uint(ITERS); ++it) {
    \\  if (it != 0) {
    \\    for (uint j = 0; j < 4; ++j) {
    \\      float sum = 0.0f;
    \\      for (uint k = 0; k < 4; ++k) sum += m[j * 4 + k];
    \\      for (uint k = 0; k < 4; ++k) m[j * 4 + k] /= sum + e;
    \\    }
    \\  }
    \\  for (uint k = 0; k < 4; ++k) {
    \\    float sum = 0.0f;
    \\    for (uint j = 0; j < 4; ++j) sum += m[j * 4 + k];
    \\    for (uint j = 0; j < 4; ++j) m[j * 4 + k] /= sum + e;
    \\  }
    \\}
    \\for (uint i = 0; i < 16; ++i) comb[row * 16 + i] = m[i];
;

var hc_split_kernel: ?mlx.mlx_fast_metal_kernel = null;

pub const Split = struct { pre: A, post: A, comb: A };

/// `hcSplit` as the op chain: the CPU stream's arm, and the kernel's oracle.
pub fn hcSplitOps(sc: *Scope, mixes: A, scale: A, base: A, iters: u32, eps: f32) !Split {
    const rows = dim(mixes, 0);
    const e = try sc.f(eps);
    const s0 = try sc.slice(scale, 0, 0, 1);
    const s1 = try sc.slice(scale, 0, 1, 2);
    const s2 = try sc.slice(scale, 0, 2, 3);
    const pre = try sc.add(try sc.sigmoid(try sc.add(try sc.mul(try sc.slice(mixes, 1, 0, 4), s0), try sc.slice(base, 0, 0, 4))), e);
    const post = try sc.mul(try sc.f(2), try sc.sigmoid(try sc.add(try sc.mul(try sc.slice(mixes, 1, 4, 8), s1), try sc.slice(base, 0, 4, 8))));
    var comb = try sc.reshape(try sc.add(try sc.mul(try sc.slice(mixes, 1, 8, 24), s2), try sc.slice(base, 0, 8, 24)), &.{ ci(rows), 4, 4 });
    comb = try sc.add(try sc.softmax(comb, -1), e);
    comb = try sc.div(comb, try sc.add(try sc.sum(comb, 1, true), e));
    for (1..iters) |_| {
        comb = try sc.div(comb, try sc.add(try sc.sum(comb, 2, true), e));
        comb = try sc.div(comb, try sc.add(try sc.sum(comb, 1, true), e));
    }
    return .{ .pre = pre, .post = post, .comb = comb };
}

/// `hc_split_sinkhorn`: `mixes` `[rows, 24]` f32 into the pre and post gates `[rows, 4]` and the
/// doubly stochastic mix `[rows, 4, 4]`; `scale` `[3]`, `base` `[24]` f32.
pub fn hcSplit(sc: *Scope, mixes: A, scale: A, base: A, iters: u32, eps: f32) !Split {
    if (!mlx.streamIsGpu(sc.s)) return hcSplitOps(sc, mixes, scale, base, iters, eps);
    const rows = dim(mixes, 0);
    const k = try kernelNew(&hc_split_kernel, "dsv41_hc_split", &.{ "mixes", "scale", "base", "eps" }, &.{ "pre", "post", "comb" }, HC_SPLIT_SOURCE, "");
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(rows), 4 }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(rows), 4 }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(rows), 4, 4 }, 3, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ci(rows), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, ci(@min(rows, 64)), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ITERS", @intCast(iters)));
    var o: [3]A = undefined;
    try applyKernel(sc, k, &.{ mixes, scale, base, try sc.floats(&.{eps}, &.{1}) }, cfg, &o);
    return .{ .pre = o[0], .post = o[1], .comb = o[2] };
}

// ── rotary ───────────────────────────────────────────────────────────────

/// `1 / theta^(2i/dim)` for the `dim / 2` pairs of the rotated tail, with YaRN when `orig > 0`:
/// dimensions whose wavelength fits the training context keep their frequency, those far past it are
/// divided by `factor`, and the `beta_fast`..`beta_slow` band fades between (`precompute_freqs_cis`).
pub fn invFreqs(out: []f32, dim_: usize, orig: u32, theta: f64, factor: f64, beta_fast: f64, beta_slow: f64) void {
    const half = dim_ / 2;
    std.debug.assert(out.len == half);
    const base: f32 = @floatCast(theta);
    for (out, 0..) |*f, i| {
        const e: f32 = @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(dim_));
        f.* = 1.0 / std.math.pow(f32, base, e);
    }
    if (orig == 0) return;
    const d: f64 = @floatFromInt(dim_);
    const corrected = struct {
        fn at(d_: f64, orig_: f64, rot: f64, b: f64) f64 {
            return d_ * @log(orig_ / (rot * 2.0 * std.math.pi)) / (2.0 * @log(b));
        }
    }.at;
    const orig_f: f64 = @floatFromInt(orig);
    const low = @max(@floor(corrected(d, orig_f, beta_fast, theta)), 0);
    const high = @min(@ceil(corrected(d, orig_f, beta_slow, theta)), d - 1);
    const span: f32 = @floatCast(@max(high - low, 1e-3));
    const fac: f32 = @floatCast(factor);
    for (out, 0..) |*f, i| {
        const ramp = std.math.clamp((@as(f32, @floatFromInt(i)) - @as(f32, @floatCast(low))) / span, 0, 1);
        const smooth = 1 - ramp;
        f.* = f.* / fac * (1 - smooth) + f.* * smooth;
    }
}

/// The rotation of the last `rd` channels of `x` `[L, D]` or `[L, H, D]` by the angles `pos[row] * freqs`,
/// adjacent pairs as complex numbers, in f32, the result in `x`'s dtype. `inverse` conjugates it.
pub fn rotate(sc: *Scope, x: A, freqs: A, pos: A, rd: usize, inverse: bool) !A {
    const sh = mlx.getShape(x);
    const rank = sh.len;
    const L = dim(x, 0);
    const D = dim(x, -1);
    const half = rd / 2;
    const dt = mlx.mlx_array_dtype(x);
    const angle = try sc.mul(try sc.reshape(try sc.astype(pos, .float32), &.{ ci(L), 1 }), try sc.reshape(freqs, &.{ 1, ci(half) }));
    const shape_cs: []const c_int = if (rank == 3) &.{ ci(L), 1, ci(half), 1 } else &.{ ci(L), ci(half), 1 };
    const cos = try sc.reshape(try sc.op1(mlx.mlx_cos, angle), shape_cs);
    var sin = try sc.reshape(try sc.op1(mlx.mlx_sin, angle), shape_cs);
    if (inverse) sin = try sc.op1(mlx.mlx_negative, sin);
    const tail = try sc.astype(try sc.slice(x, -1, D - rd, D), .float32);
    var pair_shape: [4]c_int = undefined;
    @memcpy(pair_shape[0 .. rank - 1], sh[0 .. rank - 1]);
    pair_shape[rank - 1] = ci(half);
    pair_shape[rank] = 2;
    const pairs = try sc.reshape(tail, pair_shape[0 .. rank + 1]);
    const a = try sc.slice(pairs, -1, 0, 1);
    const b = try sc.slice(pairs, -1, 1, 2);
    const o0 = try sc.sub(try sc.mul(a, cos), try sc.mul(b, sin));
    const o1 = try sc.add(try sc.mul(a, sin), try sc.mul(b, cos));
    var out_shape: [3]c_int = undefined;
    @memcpy(out_shape[0 .. rank - 1], sh[0 .. rank - 1]);
    out_shape[rank - 1] = ci(rd);
    const rotated = try sc.astype(try sc.reshape(try sc.concat(&.{ o0, o1 }, -1), out_shape[0..rank]), dt);
    if (D == rd) return rotated;
    return sc.concat(&.{ try sc.slice(x, -1, 0, D - rd), rotated }, -1);
}

// ── tests ────────────────────────────────────────────────────────────────

const testing = std.testing;
const model = @import("model.zig");

fn fixtureDir() ![]const u8 {
    const raw = std.c.getenv("DSV41_TINY") orelse return error.SkipZigTest;
    return std.mem.span(raw);
}

fn loadFixture(dir: []const u8, name: []const u8) !model.Weights {
    const path = try std.fmt.allocPrintSentinel(testing.allocator, "{s}/{s}", .{ dir, name }, 0);
    defer testing.allocator.free(path);
    var w = model.Weights.init(testing.allocator);
    errdefer w.deinit();
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    try model.loadSafetensorsFile(testing.allocator, &w, path.ptr, s, true);
    return w;
}

/// The values of `a` as host f32s (any dtype the fixtures use), copied.
fn toHostF32(sc: *Scope, a: A) ![]f32 {
    const v = try sc_mod.hostF32(try sc.astype(a, .float32));
    return testing.allocator.dupe(f32, v);
}

test "dsv41 qat: the three round trips are bit-exact with the reference's, midpoints included (DSV41_TINY)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var w = try loadFixture(try fixtureDir(), "qat.safetensors");
    defer w.deinit();
    var sc = Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    const x = w.get("x").?;
    inline for (.{
        .{ "fp8_ue8m0_32", Qat.fp8_ue8m0, 32 },
        .{ "fp4_ue8m0_32", Qat.fp4_ue8m0, 32 },
        .{ "fp4_e4m3_16", Qat.fp4_e4m3, 16 },
    }) |c| {
        const got = try toHostF32(&sc, try qat(&sc, x, c[1], c[2]));
        defer testing.allocator.free(got);
        const want = try toHostF32(&sc, w.get(c[0]).?);
        defer testing.allocator.free(want);
        try testing.expectEqual(want.len, got.len);
        for (want, got, 0..) |a, b, i| {
            if (@as(u32, @bitCast(a)) != @as(u32, @bitCast(b)) and !(a == 0 and b == 0)) {
                std.debug.print("{s}[{d}]: want {e} got {e} (x {e})\n", .{ c[0], i, a, b, (try sc_mod.hostF32(x))[i] });
                return error.QatMismatch;
            }
        }
    }
}

test "dsv41 packed fp4: rows of both FP4 round trips come back from their stored form bit for bit, zeros and signs included" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var sc = Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    var prng = std.Random.DefaultPrng.init(0xf4);
    const rnd = prng.random();
    const n = 64;
    const w = 128;
    var host: [n * w]f32 = undefined;
    for (&host, 0..) |*v, i| {
        // Rows spanning many decades, a zero row, a row of negative zeros and exact grid midpoints.
        const row = i / w;
        const mag = std.math.pow(f32, 2, @as(f32, @floatFromInt(@as(i32, @intCast(row % 40)) - 24)));
        v.* = switch (row) {
            0 => 0,
            1 => -0.0,
            2 => @as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) * 0.25,
            else => rnd.floatNorm(f32) * mag,
        };
    }
    const x = try sc.floats(&host, &.{ n, w });
    inline for (.{ .{ Qat.fp4_ue8m0, 32 }, .{ Qat.fp4_e4m3, 16 } }) |m| for ([_]mlx.mlx_dtype{ .float32, .bfloat16 }) |dt| {
        const y = try qat(&sc, try sc.astype(x, dt), m[0], m[1]);
        const p = try packFp4(&sc, y, m[0], m[1]);
        try testing.expectEqual(@as(usize, n * w / 2), mlx.mlx_array_size(p.codes));
        try testing.expectEqual(@as(usize, n * w / m[1]), mlx.mlx_array_size(p.scales));
        const back = try unpackFp4(&sc, p, m[0], m[1], dt);
        const want = try sc.contiguous(try sc.view(y, .uint8));
        const got = try sc.contiguous(try sc.view(back, .uint8));
        try evalAll(&.{ want, got });
        const wb = mlx.mlx_array_data_uint8(want).?[0..mlx.mlx_array_size(want)];
        const gb = mlx.mlx_array_data_uint8(got).?[0..mlx.mlx_array_size(got)];
        try testing.expectEqualSlices(u8, wb, gb);
    };
}

test "dsv41 hc split: the sinkhorn coefficients match the reference formulas" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var sc = Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    var prng = std.Random.DefaultPrng.init(5);
    const r = prng.random();
    var mixes: [3 * 24]f32 = undefined;
    for (&mixes) |*v| v.* = (r.float(f32) - 0.5) * 4;
    const scale = [_]f32{ 0.7, 0.4, 0.9 };
    var base: [24]f32 = undefined;
    for (&base) |*v| v.* = (r.float(f32) - 0.5);
    const eps: f32 = 1e-6;
    const got = try hcSplit(&sc, try sc.floats(&mixes, &.{ 3, 24 }), try sc.floats(&scale, &.{3}), try sc.floats(&base, &.{24}), 20, eps);
    const pre = try sc_mod.hostF32(got.pre);
    const post = try sc_mod.hostF32(got.post);
    const comb = try sc_mod.hostF32(got.comb);
    for (0..3) |row| {
        const m = mixes[row * 24 ..][0..24];
        for (0..4) |j| {
            const p = 1.0 / (1.0 + @exp(-(m[j] * scale[0] + base[j]))) + eps;
            const q = 2.0 / (1.0 + @exp(-(m[4 + j] * scale[1] + base[4 + j])));
            try testing.expectApproxEqRel(p, pre[row * 4 + j], 1e-6);
            try testing.expectApproxEqRel(q, post[row * 4 + j], 1e-6);
        }
        var c: [16]f64 = undefined;
        for (0..4) |j| {
            var mx: f64 = -std.math.inf(f64);
            for (0..4) |k| {
                c[j * 4 + k] = m[8 + j * 4 + k] * scale[2] + base[8 + j * 4 + k];
                mx = @max(mx, c[j * 4 + k]);
            }
            var s: f64 = 0;
            for (0..4) |k| {
                c[j * 4 + k] = @exp(c[j * 4 + k] - mx);
                s += c[j * 4 + k];
            }
            for (0..4) |k| c[j * 4 + k] = c[j * 4 + k] / s + eps;
        }
        for (0..20) |it| {
            if (it != 0) for (0..4) |j| {
                var s: f64 = 0;
                for (0..4) |k| s += c[j * 4 + k];
                for (0..4) |k| c[j * 4 + k] /= s + eps;
            };
            for (0..4) |k| {
                var s: f64 = 0;
                for (0..4) |j| s += c[j * 4 + k];
                for (0..4) |j| c[j * 4 + k] /= s + eps;
            }
        }
        for (c, 0..) |v, i| try testing.expectApproxEqAbs(@as(f32, @floatCast(v)), comb[row * 16 + i], 2e-6);
    }
}

test "dsv41 hc split: the kernel and the op chain agree" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var sc = Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    var prng = std.Random.DefaultPrng.init(6);
    const r = prng.random();
    var mixes: [37 * 24]f32 = undefined;
    for (&mixes) |*v| v.* = (r.float(f32) - 0.5) * 6;
    const scale = [_]f32{ 0.3, 0.8, 1.1 };
    var base: [24]f32 = undefined;
    for (&base) |*v| v.* = (r.float(f32) - 0.5) * 2;
    const mx = try sc.floats(&mixes, &.{ 37, 24 });
    const sa = try sc.floats(&scale, &.{3});
    const ba = try sc.floats(&base, &.{24});
    const k = try hcSplit(&sc, mx, sa, ba, 20, 1e-6);
    const o = try hcSplitOps(&sc, mx, sa, ba, 20, 1e-6);
    inline for (.{ "pre", "post", "comb" }) |f| {
        const a = try sc_mod.hostF32(@field(k, f));
        const b = try sc_mod.hostF32(@field(o, f));
        for (a, b) |x, y| try testing.expectApproxEqAbs(x, y, 2e-6);
    }
}

test "dsv41 rotary: a forward rotation then the inverse gives the input, and positions rotate pairs" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var sc = Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    var freqs: [8]f32 = undefined;
    invFreqs(&freqs, 16, 64, 10000, 4, 32, 1);
    // pairs i < low keep 1/theta^(2i/dim); the slowest are divided by the factor
    try testing.expectEqual(@as(f32, 1.0), freqs[0]);
    try testing.expect(freqs[7] < 1.0 / std.math.pow(f32, 10000, 14.0 / 16.0) * 0.99 / 3.9);
    var prng = std.Random.DefaultPrng.init(9);
    const r = prng.random();
    var x: [5 * 3 * 32]f32 = undefined;
    for (&x) |*v| v.* = r.floatNorm(f32);
    const pos = [_]i32{ 0, 1, 7, 100, 4000 };
    const xa = try sc.floats(&x, &.{ 5, 3, 32 });
    const fr = try sc.floats(&freqs, &.{8});
    const pa = try sc.ints(&pos, &.{5});
    const fwd = try rotate(&sc, xa, fr, pa, 16, false);
    const back = try rotate(&sc, fwd, fr, pa, 16, true);
    const got = try sc_mod.hostF32(fwd);
    const bk = try sc_mod.hostF32(back);
    for (x, bk) |a, b| try testing.expectApproxEqAbs(a, b, 2e-5);
    // row 0 is at position 0: no rotation; the untouched head channels never move
    for (0..3) |h| for (0..32) |d| try testing.expectEqual(x[h * 32 + d], got[h * 32 + d]);
    for (0..5) |row| for (0..3) |h| for (0..16) |d| try testing.expectEqual(x[(row * 3 + h) * 32 + d], got[(row * 3 + h) * 32 + d]);
    // a pair at position p turns by p * freq
    const p: f32 = 7;
    const a0 = x[(2 * 3 + 1) * 32 + 16];
    const a1 = x[(2 * 3 + 1) * 32 + 17];
    const ang = p * freqs[0];
    try testing.expectApproxEqAbs(a0 * @cos(ang) - a1 * @sin(ang), got[(2 * 3 + 1) * 32 + 16], 2e-5);
    try testing.expectApproxEqAbs(a0 * @sin(ang) + a1 * @cos(ang), got[(2 * 3 + 1) * 32 + 17], 2e-5);
}
