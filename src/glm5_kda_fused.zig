//! GLM one-token KDA body; BF16 activations and FP32 vector decay/state.
//! Adapted from oMLX 6745c39c (Apache-2.0); sigmoid expressions derive from MLX (MIT).
const std = @import("std");
const mlx = @import("mlx.zig");
const Arr = mlx.mlx_array;
const HEADER: [:0]const u8 =
    \\#include <metal_stdlib>
    \\#include <metal_simdgroup>
    \\using namespace metal;
    \\template<typename T, bool precise> inline T glm_sigmoid(T x) {
    \\  auto y = 1 / (1 + (precise ? metal::precise::exp(metal::abs(x)) : metal::exp(metal::abs(x))));
    \\  return x < 0 ? y : 1 - y;
    \\}
    \\template<bool precise> inline float glm_exp(float x) { return precise ? metal::precise::exp(x) : metal::exp(x); }
    \\template<bool precise> inline float glm_rsqrt(float x) { return precise ? metal::precise::rsqrt(x) : metal::rsqrt(x); }
;
const SOURCE_HEAD =
    \\
    \\  constexpr int CK = 4;
    \\  constexpr int NROW = CK - 1 + TOK;
    \\  constexpr int NP = 3 * QKV;
    \\  const uint tid = thread_position_in_threadgroup.x;
    \\  const uint lane = thread_index_in_simdgroup;
    \\  const uint sg = simdgroup_index_in_threadgroup;
    \\  const int h = int(threadgroup_position_in_grid.x);
    \\  const float q_scale = consts[0];
    \\  const float l2_eps = consts[1];
    \\  const float norm_eps = consts[2];
    \\  const float lower = consts[3];
    \\  const float inv_n = consts[4];
    \\
    \\  threadgroup T qs[TOK][DK];
    \\  threadgroup T ks[TOK][DK];
    \\  threadgroup T vs[TOK][DK];
    \\  threadgroup T as_[TOK][DK];
    \\  threadgroup T gates[TOK][DK];
    \\  threadgroup T ys[TOK][DK];
    \\  threadgroup float gs[TOK][DK];
    \\  threadgroup T betas[TOK];
    \\
    \\  if (tid < uint(3 * DK)) {
    \\    const int part = int(tid) / DK;
    \\    const int i = int(tid) % DK;
    \\    const int gc = part * QKV + h * DK + i;
    \\    T win[NROW];
    \\    for (int r = 0; r < CK - 1; r++) {
    \\      win[r] = conv_state[r * NP + gc];
    \\    }
    \\    for (int t = 0; t < TOK; t++) {
    \\      win[CK - 1 + t] = proj[t * PROJ_W + gc];
    \\    }
    \\    const device T* w = conv_w + gc * CK;
    \\    for (int t = 0; t < TOK; t++) {
    \\      float acc = 0.0;
    \\      for (int j = 0; j < CK; ++j) {
    \\        acc += static_cast<float>(win[t + j]) * w[j];
    \\      }
    \\      T co = static_cast<T>(acc);
    \\      T sgm = glm_sigmoid<T, SIG_B>(co);
    \\      T sv = co * sgm;
    \\      if (part == 0) {
    \\        qs[t][i] = sv;
    \\      } else if (part == 1) {
    \\        ks[t][i] = sv;
    \\      } else {
    \\        vs[t][i] = sv;
    \\      }
    \\    }
    \\    for (int r = 0; r < CK - 1; r++) {
    \\      conv_state_out[r * NP + gc] = win[TOK + r];
    \\    }
    \\  }
    \\
;
const LOAD_SEPARATE =
    \\  for (int e = int(tid); e < TOK * DK; e += 1024) {
    \\    const int t = e / DK;
    \\    const int i = e % DK;
    \\    as_[t][i] = a_pre[t * QKV + h * DK + i];
    \\    gates[t][i] = gate_pre[t * QKV + h * DK + i];
    \\  }
    \\
;
// One token only. Mirrors MLX's BF16 `gemv` at K = 128: each lane takes four consecutive columns,
// accumulates them in order in FP32 (BF16 products are exact, so FMA contraction cannot change a
// bit), then the lanes reduce by shuffle-down 16, 8, 4, 2, 1 and the sum rounds to BF16.
const LOAD_PROJECTED =
    \\  {
    \\    float xa[4];
    \\    float xg[4];
    \\    for (int e = 0; e < 4; e++) {
    \\      xa[e] = static_cast<float>(a_low[4 * lane + e]);
    \\      xg[e] = static_cast<float>(g_low[4 * lane + e]);
    \\    }
    \\    for (int r = 0; r < 4; r++) {
    \\      const int i = int(sg) * 4 + r;
    \\      const size_t base = (size_t(h) * DK + i) * DK + 4 * lane;
    \\      float acc_a = 0.0f;
    \\      float acc_g = 0.0f;
    \\      for (int e = 0; e < 4; e++) {
    \\        acc_a += static_cast<float>(fb_w[base + e]) * xa[e];
    \\        acc_g += static_cast<float>(gb_w[base + e]) * xg[e];
    \\      }
    \\      for (ushort o = 16; o >= 1; o >>= 1) {
    \\        acc_a += simd_shuffle_down(acc_a, o);
    \\        acc_g += simd_shuffle_down(acc_g, o);
    \\      }
    \\      if (lane == 0) {
    \\        as_[0][i] = static_cast<T>(acc_a);
    \\        gates[0][i] = static_cast<T>(acc_g);
    \\      }
    \\    }
    \\  }
    \\
;
const SOURCE_TAIL =
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\  if (sg < uint(2 * TOK)) {
    \\    const int t = int(sg) / 2;
    \\    const bool is_q = (sg % 2) == 0;
    \\    threadgroup T* row = is_q ? qs[t] : ks[t];
    \\    float x[4];
    \\    float tot = 0.0f;
    \\    for (int e = 0; e < 4; e++) {
    \\      x[e] = static_cast<float>(row[4 * lane + e]);
    \\      float sq = x[e] * x[e];
    \\      tot = sq + tot;
    \\    }
    \\    tot = simd_sum(tot);
    \\    float u = tot + l2_eps;
    \\    float r = glm_rsqrt<RSQ_F>(u);
    \\    for (int e = 0; e < 4; e++) {
    \\      float xn = x[e] * r;
    \\      if (is_q) {
    \\        float xs = xn * q_scale;
    \\        row[4 * lane + e] = static_cast<T>(xs);
    \\      } else {
    \\        row[4 * lane + e] = static_cast<T>(xn);
    \\      }
    \\    }
    \\  }
    \\
    \\  if (tid < uint(TOK * DK)) {
    \\    const int t = int(tid) / DK;
    \\    const int i = int(tid) % DK;
    \\    float ea = exp_a[h];
    \\    float af = static_cast<float>(as_[t][i]);
    \\    float s1 = af + dt_bias[h * DK + i];
    \\    float s2 = ea * s1;
    \\    float s3 = glm_sigmoid<float, SIG_F>(s2);
    \\    float s4 = lower * s3;
    \\    gs[t][i] = glm_exp<EXP_F>(s4);
    \\  }
    \\  if (tid < uint(TOK)) {
    \\    betas[tid] = glm_sigmoid<T, SIG_B>(proj[tid * PROJ_W + OFF_B + h]);
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\  for (int j = 0; j < DK / 32; j++) {
    \\    const int dv_idx = int(sg) + 32 * j;
    \\    constexpr int n_per_t = DK / 32;
    \\    const int dk_idx = int(lane);
    \\    float state[n_per_t];
    \\    for (int i = 0; i < n_per_t; ++i) {
    \\      auto s_idx = n_per_t * dk_idx + i;
    \\      state[i] = static_cast<float>(state_in[(size_t(h) * DK + dv_idx) * DK + s_idx]);
    \\    }
    \\    for (int t = 0; t < TOK; ++t) {
    \\      float kv_mem = 0.0f;
    \\      for (int i = 0; i < n_per_t; ++i) {
    \\        auto s_idx = n_per_t * dk_idx + i;
    \\        state[i] = state[i] * gs[t][s_idx];
    \\        kv_mem += state[i] * ks[t][s_idx];
    \\      }
    \\      kv_mem = simd_sum(kv_mem);
    \\
    \\      auto delta = (vs[t][dv_idx] - kv_mem) * betas[t];
    \\
    \\      float out = 0.0f;
    \\      for (int i = 0; i < n_per_t; ++i) {
    \\        auto s_idx = n_per_t * dk_idx + i;
    \\        state[i] = state[i] + ks[t][s_idx] * delta;
    \\        out += state[i] * qs[t][s_idx];
    \\      }
    \\      out = simd_sum(out);
    \\      if (thread_index_in_simdgroup == 0) {
    \\        ys[t][dv_idx] = static_cast<T>(out);
    \\      }
    \\    }
    \\    for (int i = 0; i < n_per_t; ++i) {
    \\      auto s_idx = n_per_t * dk_idx + i;
    \\      state_out[(size_t(h) * DK + dv_idx) * DK + s_idx] = static_cast<float>(state[i]);
    \\    }
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\  if (sg < uint(TOK)) {
    \\    const int t = int(sg);
    \\    float x[4];
    \\    float tot = 0.0f;
    \\    for (int e = 0; e < 4; e++) {
    \\      x[e] = static_cast<float>(ys[t][4 * lane + e]);
    \\      float sq = x[e] * x[e];
    \\      tot = sq + tot;
    \\    }
    \\    tot = simd_sum(tot);
    \\    float var = tot * inv_n;
    \\    float u = var + norm_eps;
    \\    float r = glm_rsqrt<RSQ_F>(u);
    \\    for (int e = 0; e < 4; e++) {
    \\      const int c = 4 * lane + e;
    \\      float xn = x[e] * r;
    \\      float wf = static_cast<float>(norm_w[c]);
    \\      float wx = wf * xn;
    \\      float gf = static_cast<float>(gates[t][c]);
    \\      float gsg = glm_sigmoid<float, SIG_F>(gf);
    \\      float o = wx * gsg;
    \\      y[t * QKV + h * DK + c] = static_cast<T>(o);
    \\    }
    \\  }
;
const SOURCE: [:0]const u8 = SOURCE_HEAD ++ LOAD_SEPARATE ++ SOURCE_TAIL;
const SOURCE_PROJECTED: [:0]const u8 = SOURCE_HEAD ++ LOAD_PROJECTED ++ SOURCE_TAIL;

pub const Modes = struct { sig_b: bool, sig_f: bool, exp_f: bool, rsq_f: bool };
var modes_checked = false;
var modes_cached: ?Modes = null;
var kernel_cached: ?mlx.mlx_fast_metal_kernel = null;
var projected_cached: ?mlx.mlx_fast_metal_kernel = null;
var calls: usize = 0;
var projected_calls: usize = 0;
pub fn dispatchCount() usize {
    return calls;
}
pub fn projectedDispatchCount() usize {
    return projected_calls;
}
pub fn resetDispatchCount() void {
    calls = 0;
}

fn makeKernel(name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [:0]const u8) !mlx.mlx_fast_metal_kernel {
    const iv = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new(name, iv, ov, source, HEADER, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    return k;
}
fn applyKernel(k: mlx.mlx_fast_metal_kernel, inputs: []const Arr, cfg: mlx.mlx_fast_metal_kernel_config, s: mlx.mlx_stream) !mlx.mlx_vector_array {
    const iv = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    errdefer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, k, iv, cfg, s));
    return ov;
}

const PROBE: [:0]const u8 =
    \\const uint i=thread_position_in_grid.x;
    \\if(i>=uint(SIZE))return;
    \\if(OP==0) { fast[i]=glm_sigmoid<T,false>(x[i]); precise[i]=glm_sigmoid<T,true>(x[i]); }
    \\else if(OP==1) { fast[i]=T(glm_exp<false>(float(x[i]))); precise[i]=T(glm_exp<true>(float(x[i]))); }
    \\else { fast[i]=T(glm_rsqrt<false>(float(x[i]))); precise[i]=T(glm_rsqrt<true>(float(x[i]))); }
;
fn sameBytes(a: Arr, b: Arr) bool {
    const n = mlx.mlx_array_size(a);
    return if (mlx.mlx_array_dtype(a) == .bfloat16)
        std.mem.eql(u16, mlx.mlx_array_data_bfloat16(a).?[0..n], mlx.mlx_array_data_bfloat16(b).?[0..n])
    else
        std.mem.eql(u8, std.mem.sliceAsBytes(mlx.mlx_array_data_float32(a).?[0..n]), std.mem.sliceAsBytes(mlx.mlx_array_data_float32(b).?[0..n]));
}
fn probe(s: mlx.mlx_stream, dtype: mlx.mlx_dtype, operation: c_int) !?bool {
    const n = 65536;
    const a = std.heap.page_allocator;
    var input = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(input);
    if (dtype == .bfloat16) {
        const values = try a.alloc(u16, n);
        defer a.free(values);
        for (values, 0..) |*v, i| {
            const bits: u16 = @intCast(i);
            v.* = if (bits & 0x7f80 == 0x7f80) 0 else bits;
        }
        const raw = mlx.mlx_array_new_data(values.ptr, &[_]c_int{n}, 1, dtype);
        defer _ = mlx.mlx_array_free(raw);
        try mlx.check(mlx.mlx_array_set(&input, raw));
    } else {
        const values = try a.alloc(f32, n);
        defer a.free(values);
        for (values, 0..) |*v, i| {
            const u = @as(f32, @floatFromInt(i)) / @as(f32, n - 1);
            v.* = switch (operation) {
                0 => 48 * u - 24,
                1 => 5 * u - 5,
                else => @exp2(-20 + 40 * u),
            };
        }
        const raw = mlx.mlx_array_new_data(values.ptr, &[_]c_int{n}, 1, dtype);
        defer _ = mlx.mlx_array_free(raw);
        try mlx.check(mlx.mlx_array_set(&input, raw));
    }
    const k = try makeKernel("sushi_glm_kda_math_probe", &.{"x"}, &.{ "fast", "precise" }, PROBE);
    defer _ = mlx.mlx_fast_metal_kernel_free(k);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    for (0..2) |_| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{n}, 1, dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, n, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "SIZE", n));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "OP", operation));
    const outputs = try applyKernel(k, &.{input}, cfg, s);
    defer _ = mlx.mlx_vector_array_free(outputs);
    var candidates = [_]Arr{ mlx.mlx_array_new(), mlx.mlx_array_new() };
    defer for (candidates) |v| {
        _ = mlx.mlx_array_free(v);
    };
    for (&candidates, 0..) |*v, i| try mlx.check(mlx.mlx_vector_array_get(v, outputs, i));
    var expected = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(expected);
    try mlx.check(switch (operation) {
        0 => mlx.mlx_sigmoid(&expected, input, s),
        1 => mlx.mlx_exp(&expected, input, s),
        else => mlx.mlx_rsqrt(&expected, input, s),
    });
    const evals = mlx.mlx_vector_array_new_data(&[_]Arr{ expected, candidates[0], candidates[1] }, 3);
    defer _ = mlx.mlx_vector_array_free(evals);
    try mlx.check(mlx.mlx_eval(evals));
    if (sameBytes(expected, candidates[1])) return true;
    if (sameBytes(expected, candidates[0])) return false;
    return null;
}
fn modes(s: mlx.mlx_stream) !?Modes {
    if (modes_checked) return modes_cached;
    const result: ?Modes = blk: {
        break :blk .{ .sig_b = (try probe(s, .bfloat16, 0)) orelse break :blk null, .sig_f = (try probe(s, .float32, 0)) orelse break :blk null, .exp_f = (try probe(s, .float32, 1)) orelse break :blk null, .rsq_f = (try probe(s, .float32, 2)) orelse break :blk null };
    };
    modes_cached = result;
    modes_checked = true;
    return result;
}

pub fn unaryModes(s: mlx.mlx_stream) !?Modes {
    return modes(s);
}

pub fn sigmoidFloatMode(s: mlx.mlx_stream) !?bool {
    const found = (try modes(s)) orelse return null;
    return found.sig_f;
}

/// Second-stage gate projections `fb`/`gb` ([heads*128, 128] BF16) applied to the low-rank
/// outputs `a_low`/`gate_low` ([1, 1, 128]); stands in for `Inputs.a`/`gate`.
pub const SecondStage = struct { a_low: Arr, gate_low: Arr, fb: Arr, gb: Arr };
pub const Inputs = struct {
    qkv: Arr = .{ .ctx = null },
    beta: Arr = .{ .ctx = null },
    /// `[q|k|v|beta]` already joined; replaces `qkv` and `beta`.
    joined: Arr = .{ .ctx = null },
    a: Arr = .{ .ctx = null },
    gate: Arr = .{ .ctx = null },
    second_stage: ?SecondStage = null,
    conv_weight: Arr,
    exp_a: Arr,
    dt_bias: Arr,
    norm: Arr,
    conv_state: Arr,
    state: Arr,
    heads: c_int,
    norm_eps: f32,
    lower: f32,
};
pub const Result = struct {
    y: Arr,
    conv: Arr,
    state: Arr,
    pub fn deinit(self: Result) void {
        _ = mlx.mlx_array_free(self.y);
        _ = mlx.mlx_array_free(self.conv);
        _ = mlx.mlx_array_free(self.state);
    }
};
const Config = struct { value: mlx.mlx_fast_metal_kernel_config, cached: bool };
const ConfigEntry = struct { heads: c_int, value: mlx.mlx_fast_metal_kernel_config };
var configs: [8]?ConfigEntry = @splat(null);
fn configuration(heads: c_int, mode: Modes) !Config {
    for (configs) |entry| if (entry) |e| {
        if (e.heads == heads) return .{ .value = e.value, .cached = true };
    };
    const d = heads * 128;
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ 1, 1, d }, 3, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ 1, 3, d * 3 }, 3, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ 1, heads, 128, 128 }, 4, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 1024 * heads, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 1024, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    inline for (.{ "TOK", "DK", "QKV", "PROJ_W", "OFF_B" }, .{ @as(c_int, 1), @as(c_int, 128), d, d * 3 + heads, d * 3 }) |name, value| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, name, value));
    inline for (.{ "SIG_B", "SIG_F", "EXP_F", "RSQ_F" }, .{ mode.sig_b, mode.sig_f, mode.exp_f, mode.rsq_f }) |name, value| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, name, @intFromBool(value)));
    for (&configs) |*entry| if (entry.* == null) {
        entry.* = .{ .heads = heads, .value = cfg };
        return .{ .value = cfg, .cached = true };
    };
    return .{ .value = cfg, .cached = false };
}

pub fn step(s: mlx.mlx_stream, in: Inputs) !?Result {
    if (!mlx.streamIsGpu(s) or in.heads < 1 or in.heads > 1024 or !std.math.isFinite(in.norm_eps) or in.norm_eps <= 0 or !std.math.isFinite(in.lower) or in.lower != -5) return null;
    for ([_]Arr{ in.conv_weight, in.exp_a, in.dt_bias, in.norm, in.conv_state, in.state }) |value| if (value.ctx == null) return null;
    const d = in.heads * 128;
    if (in.joined.ctx != null) {
        if (!std.mem.eql(c_int, &.{ 1, 1, d * 3 + in.heads }, mlx.getShape(in.joined)) or mlx.mlx_array_dtype(in.joined) != .bfloat16) return null;
    } else {
        if (in.qkv.ctx == null or in.beta.ctx == null or
            !std.mem.eql(c_int, &.{ 1, 1, d * 3 }, mlx.getShape(in.qkv)) or mlx.mlx_array_dtype(in.qkv) != .bfloat16 or
            !std.mem.eql(c_int, &.{ 1, 1, in.heads }, mlx.getShape(in.beta)) or mlx.mlx_array_dtype(in.beta) != .bfloat16) return null;
    }
    if (in.second_stage) |st| {
        for ([_]Arr{ st.a_low, st.gate_low, st.fb, st.gb }) |value| if (value.ctx == null or mlx.mlx_array_dtype(value) != .bfloat16) return null;
        if (!std.mem.eql(c_int, &.{ 1, 1, 128 }, mlx.getShape(st.a_low)) or !std.mem.eql(c_int, &.{ 1, 1, 128 }, mlx.getShape(st.gate_low)) or
            !std.mem.eql(c_int, &.{ d, 128 }, mlx.getShape(st.fb)) or !std.mem.eql(c_int, &.{ d, 128 }, mlx.getShape(st.gb))) return null;
    } else {
        if (in.a.ctx == null or in.gate.ctx == null or
            !std.mem.eql(c_int, &.{ 1, 1, d }, mlx.getShape(in.a)) or mlx.mlx_array_dtype(in.a) != .bfloat16 or
            !std.mem.eql(c_int, &.{ 1, 1, d }, mlx.getShape(in.gate)) or mlx.mlx_array_dtype(in.gate) != .bfloat16) return null;
    }
    if (!std.mem.eql(c_int, &.{ d * 3, 4, 1 }, mlx.getShape(in.conv_weight)) or mlx.mlx_array_dtype(in.conv_weight) != .bfloat16 or
        mlx.mlx_array_size(in.exp_a) != @as(usize, @intCast(in.heads)) or mlx.mlx_array_dtype(in.exp_a) != .float32 or
        !std.mem.eql(c_int, &.{d}, mlx.getShape(in.dt_bias)) or mlx.mlx_array_dtype(in.dt_bias) != .float32 or
        !std.mem.eql(c_int, &.{128}, mlx.getShape(in.norm)) or mlx.mlx_array_dtype(in.norm) != .bfloat16 or
        !std.mem.eql(c_int, &.{ 1, 3, d * 3 }, mlx.getShape(in.conv_state)) or mlx.mlx_array_dtype(in.conv_state) != .bfloat16 or
        !std.mem.eql(c_int, &.{ 1, in.heads, 128, 128 }, mlx.getShape(in.state)) or mlx.mlx_array_dtype(in.state) != .float32) return null;
    const mode = (try modes(s)) orelse return null;
    var projection = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(projection);
    if (in.joined.ctx != null) {
        try mlx.check(mlx.mlx_array_set(&projection, in.joined));
    } else {
        const parts = mlx.mlx_vector_array_new_data(&[_]Arr{ in.qkv, in.beta }, 2);
        defer _ = mlx.mlx_vector_array_free(parts);
        try mlx.check(mlx.mlx_concatenate_axis(&projection, parts, -1, s));
    }
    const constants = [_]f32{ 1 / @sqrt(@as(f32, 128)), 1e-6, in.norm_eps, in.lower, 1.0 / 128.0 };
    const consts = mlx.mlx_array_new_data(&constants, &[_]c_int{5}, 1, .float32);
    defer _ = mlx.mlx_array_free(consts);
    const common = [_]Arr{ projection, in.conv_weight, in.exp_a, in.dt_bias, in.norm, consts, in.conv_state, in.state };
    const selected = try configuration(in.heads, mode);
    const cfg = selected.value;
    defer if (!selected.cached) {
        _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    };
    const outputs = if (in.second_stage) |st| blk: {
        if (projected_cached == null) projected_cached = try makeKernel("sushi_glm_kda_decode_body_proj", &.{ "proj", "conv_w", "exp_a", "dt_bias", "norm_w", "consts", "conv_state", "state_in", "a_low", "g_low", "fb_w", "gb_w" }, &.{ "y", "conv_state_out", "state_out" }, SOURCE_PROJECTED);
        break :blk try applyKernel(projected_cached.?, &(common ++ [_]Arr{ st.a_low, st.gate_low, st.fb, st.gb }), cfg, s);
    } else blk: {
        if (kernel_cached == null) kernel_cached = try makeKernel("sushi_glm_kda_decode_body", &.{ "proj", "conv_w", "exp_a", "dt_bias", "norm_w", "consts", "conv_state", "state_in", "a_pre", "gate_pre" }, &.{ "y", "conv_state_out", "state_out" }, SOURCE);
        break :blk try applyKernel(kernel_cached.?, &(common ++ [_]Arr{ in.a, in.gate }), cfg, s);
    };
    defer _ = mlx.mlx_vector_array_free(outputs);
    var result = Result{ .y = mlx.mlx_array_new(), .conv = mlx.mlx_array_new(), .state = mlx.mlx_array_new() };
    errdefer result.deinit();
    try mlx.check(mlx.mlx_vector_array_get(&result.y, outputs, 0));
    try mlx.check(mlx.mlx_vector_array_get(&result.conv, outputs, 1));
    try mlx.check(mlx.mlx_vector_array_get(&result.state, outputs, 2));
    calls += 1;
    if (calls == 1) @import("log.zig").info("[glm-kda] fused one-token body engaged\n", .{});
    if (in.second_stage != null) {
        projected_calls += 1;
        if (projected_calls == 1) @import("log.zig").info("[glm-kda] f_b/g_b projections run in the body\n", .{});
    }
    return result;
}

var post_kernel: ?mlx.mlx_fast_metal_kernel = null;
var post_calls: usize = 0;
pub fn postDispatchCount() usize {
    return post_calls;
}

fn postGeometryFits(tokens: c_int, heads: c_int) bool {
    if (tokens < 1 or heads < 1) return false;
    const rows = std.math.mul(u64, @intCast(tokens), @intCast(heads)) catch return false;
    const elements = std.math.mul(u64, rows, 128) catch return false;
    const grid = std.math.mul(u64, rows, 32) catch return false;
    return elements - 1 <= std.math.maxInt(u32) and grid <= std.math.maxInt(c_int);
}

pub fn post(s: mlx.mlx_stream, y: Arr, gate: Arr, norm: Arr, epsilon: f32) !?Arr {
    if (!mlx.streamIsGpu(s) or y.ctx == null or gate.ctx == null or norm.ctx == null or !std.math.isFinite(epsilon) or epsilon <= 0) return null;
    const sh = mlx.getShape(y);
    if (sh.len != 4 or sh[0] != 1 or sh[1] <= 0 or sh[2] <= 0 or sh[3] != 128 or
        !std.mem.eql(c_int, sh, mlx.getShape(gate)) or !std.mem.eql(c_int, &.{128}, mlx.getShape(norm)) or
        mlx.mlx_array_dtype(y) != .bfloat16 or mlx.mlx_array_dtype(gate) != .bfloat16 or
        (mlx.mlx_array_dtype(norm) != .bfloat16 and mlx.mlx_array_dtype(norm) != .float32)) return null;
    if (!postGeometryFits(sh[1], sh[2])) return null;
    const rows = std.math.mul(c_int, sh[1], sh[2]) catch return null;
    const grid = std.math.mul(c_int, rows, 32) catch return null;
    const mode = (try modes(s)) orelse return null;
    if (post_kernel == null) {
        const inputs = mlx.mlx_vector_string_new_data(&.{ "input", "gate", "weight", "eps" }, 4);
        defer _ = mlx.mlx_vector_string_free(inputs);
        const outputs = mlx.mlx_vector_string_new_data(&.{"out"}, 1);
        defer _ = mlx.mlx_vector_string_free(outputs);
        const source: [:0]const u8 =
            \\const uint row = threadgroup_position_in_grid.x;
            \\const uint lane = thread_index_in_simdgroup;
            \\float x[4];
            \\float total = 0.0f;
            \\for (int e=0;e<4;e++) {
            \\  x[e] = float(input[row*128 + 4*lane + e]);
            \\  float square = x[e] * x[e];
            \\  total = square + total;
            \\}
            \\total = simd_sum(total);
            \\float variance = total * (1.0f / 128.0f);
            \\float inv = glm_rsqrt<RSQ_F>(variance + eps[0]);
            \\for (int e=0;e<4;e++) {
            \\  const uint c = 4*lane + e;
            \\  float normalized = x[e] * inv;
            \\  float weighted = normalized * float(weight[c]);
            \\  float sigmoid = glm_sigmoid<float,SIG_F>(float(gate[row*128+c]));
            \\  out[row*128+c] = bfloat16_t(weighted * sigmoid);
            \\}
        ;
        const k = mlx.mlx_fast_metal_kernel_new("sushi_glm_kda_post", inputs, outputs, source, HEADER, true, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        post_kernel = k;
    }
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, sh.ptr, sh.len, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, grid, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "SIG_F", @intFromBool(mode.sig_f)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "RSQ_F", @intFromBool(mode.rsq_f)));
    const eps = mlx.mlx_array_new_data(&epsilon, &.{1}, 1, .float32);
    defer _ = mlx.mlx_array_free(eps);
    const inputs = mlx.mlx_vector_array_new_data(&.{ y, gate, norm, eps }, 4);
    defer _ = mlx.mlx_vector_array_free(inputs);
    var outputs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs, post_kernel.?, inputs, cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs, 0));
    post_calls += 1;
    return out;
}

test "GLM KDA fused post preserves staged BF16 output at prefill widths" {
    const Ops = @import("glm5_model.zig").Ops;
    const stream = mlx.gpuStream();
    for ([_]c_int{ 1, 17, 512 }) |rows| {
        var ops = Ops{ .s = stream };
        defer ops.deinit();
        const heads: c_int = if (rows == 512) 64 else 3;
        const shape = [_]c_int{ 1, rows, heads, 128 };
        const key = try ops.slot();
        try mlx.check(mlx.mlx_random_key(key, 791));
        const y = try ops.slot();
        try mlx.check(mlx.mlx_random_normal(y, &shape, 4, .bfloat16, 0, 0.7, key.*, stream));
        const gate = try ops.slot();
        try mlx.check(mlx.mlx_random_normal(gate, &shape, 4, .bfloat16, -0.3, 3, key.*, stream));
        const norm = try ops.slot();
        try mlx.check(mlx.mlx_random_normal(norm, &.{128}, 1, .bfloat16, 1, 0.2, key.*, stream));
        const yf = try ops.cast(y.*, .float32);
        const variance = try ops.reduce(try ops.binary(.mul, yf, yf), -1, true, true);
        const normalization = try ops.unary(.rsqrt, try ops.binary(.add, variance, try ops.scalar(1e-5, .float32)));
        const normalized = try ops.binary(.mul, try ops.binary(.mul, yf, normalization), try ops.cast(norm.*, .float32));
        const expected = try ops.cast(try ops.binary(.mul, normalized, try ops.unary(.sigmoid, try ops.cast(gate.*, .float32))), .bfloat16);
        const actual = try ops.own((try post(stream, y.*, gate.*, norm.*, 1e-5)) orelse return error.TestExpectedFusedPost);
        try mlx.check(mlx.mlx_array_eval(actual));
        try mlx.check(mlx.mlx_array_eval(expected));
        try std.testing.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(expected).?[0..mlx.mlx_array_size(expected)], mlx.mlx_array_data_bfloat16(actual).?[0..mlx.mlx_array_size(actual)]);
    }
}

fn stagedPost(ops: *@import("glm5_model.zig").Ops, input: Arr, gate: Arr, norm: Arr, epsilon: f32) !Arr {
    const y = try ops.cast(input, .float32);
    const variance = try ops.reduce(try ops.binary(.mul, y, y), -1, true, true);
    const normalization = try ops.unary(.rsqrt, try ops.binary(.add, variance, try ops.scalar(epsilon, .float32)));
    const normalized = try ops.binary(.mul, try ops.binary(.mul, y, normalization), try ops.cast(norm, .float32));
    return ops.cast(try ops.binary(.mul, normalized, try ops.unary(.sigmoid, try ops.cast(gate, .float32))), .bfloat16);
}

test "GLM KDA fused post validates guards and FP32 norm weights" {
    const Ops = @import("glm5_model.zig").Ops;
    var ops = Ops{ .s = mlx.gpuStream() };
    defer ops.deinit();
    const x = try ops.ones(&.{ 1, 17, 3, 128 }, .bfloat16);
    const gate = try ops.binary(.mul, x, try ops.scalar(-5, .bfloat16));
    const norm = try ops.ones(&.{128}, .float32);
    for ([_]f32{ 1e-6, 1e-5, 0.01 }) |epsilon| {
        const reference = try stagedPost(&ops, x, gate, norm, epsilon);
        const actual = try ops.own((try post(ops.s, x, gate, norm, epsilon)).?);
        try mlx.check(mlx.mlx_array_eval(reference));
        try mlx.check(mlx.mlx_array_eval(actual));
        try std.testing.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(reference).?[0..mlx.mlx_array_size(reference)], mlx.mlx_array_data_bfloat16(actual).?[0..mlx.mlx_array_size(actual)]);
    }
    try std.testing.expect((try post(ops.s, x, gate, norm, 0)) == null);
    try std.testing.expect((try post(ops.s, x, gate, norm, std.math.nan(f32))) == null);
    try std.testing.expect((try post(ops.s, try ops.cast(x, .float32), gate, norm, 1e-5)) == null);
    try std.testing.expect((try post(ops.s, x, try ops.reshape(gate, &.{ 1, 17, 384 }), norm, 1e-5)) == null);
}

test "GLM KDA post geometry guards every uint32 shader offset" {
    try std.testing.expect(postGeometryFits(512, 64));
    try std.testing.expect(postGeometryFits(524_288, 64));
    try std.testing.expect(!postGeometryFits(524_289, 64));
    try std.testing.expect(postGeometryFits(33_554_432, 1));
    try std.testing.expect(!postGeometryFits(33_554_433, 1));
    try std.testing.expect(!postGeometryFits(0, 64));
    try std.testing.expect(!postGeometryFits(1, 0));
    try std.testing.expect(!postGeometryFits(std.math.maxInt(c_int), std.math.maxInt(c_int)));
}

test "GLM SIMD KDA, prework and router arms serve GPUs without NAX" {
    const transformer = @import("transformer.zig");
    const saved = transformer.vqmm_nax_probe_override;
    defer transformer.vqmm_nax_probe_override = saved;
    transformer.vqmm_nax_probe_override = false;
    const s = mlx.gpuStream();
    var ops = @import("glm5_model.zig").Ops{ .s = s };
    defer ops.deinit();
    try std.testing.expect((try unaryModes(s)) != null);
    try std.testing.expect((try sigmoidFloatMode(s)) != null);
    const d = 128;
    const conv = try ops.zeros(&.{ 3 * d, 4, 1 }, .bfloat16);
    const exp_a = try ops.ones(&.{1}, .float32);
    const dt_bias = try ops.zeros(&.{d}, .float32);
    const body = (try step(s, .{
        .qkv = try ops.zeros(&.{ 1, 1, 3 * d }, .bfloat16), .beta = try ops.zeros(&.{ 1, 1, 1 }, .bfloat16),
        .a = try ops.zeros(&.{ 1, 1, d }, .bfloat16),       .gate = try ops.zeros(&.{ 1, 1, d }, .bfloat16),
        .conv_weight = conv,                                .exp_a = exp_a,
        .dt_bias = dt_bias,                                 .norm = try ops.ones(&.{128}, .bfloat16),
        .conv_state = try ops.zeros(&.{ 1, 3, 3 * d }, .bfloat16), .state = try ops.zeros(&.{ 1, 1, 128, 128 }, .float32),
        .heads = 1,                                         .norm_eps = 1e-5,
        .lower = -5,
    })) orelse return error.TestExpectedFusedKda;
    body.deinit();
    const y = try ops.zeros(&.{ 1, 2, 1, 128 }, .bfloat16);
    _ = try ops.own((try post(s, y, y, try ops.ones(&.{128}, .bfloat16), 1e-5)) orelse return error.TestExpectedFusedPost);
    const prework = @import("glm5_kda_prework.zig");
    const in = prework.Inputs{ .qkv = try ops.zeros(&.{ 1, 2, 3 * d }, .bfloat16), .a = try ops.zeros(&.{ 1, 2, d }, .bfloat16), .beta = try ops.zeros(&.{ 1, 2, 1 }, .bfloat16), .conv_weight = conv, .exp_a = exp_a, .dt_bias = dt_bias, .heads = 1 };
    (try prework.apply(s, in) orelse return error.TestExpectedPrework).deinit();
    (try prework.applyTree(s, in, &.{ -1, 0 }) orelse return error.TestExpectedPrework).deinit();
    const weights: [16 * 128]f32 = @splat(0.01);
    const bias: [16]f32 = @splat(0);
    const routed = (try @import("glm5_router.zig").route(s, try ops.zeros(&.{ 1, 1, 128 }, .bfloat16), try ops.own(mlx.mlx_array_new_data(&weights, &.{ 16, 128 }, 2, .float32)), try ops.own(mlx.mlx_array_new_data(&bias, &.{16}, 1, .float32)), 2, 2.5, true)) orelse return error.TestExpectedRouter;
    _ = try ops.own(routed.indices);
    _ = try ops.own(routed.scores);
}

fn randn(ops: *@import("glm5_model.zig").Ops, shape: []const c_int, dtype: mlx.mlx_dtype, loc: f32, scale: f32, seed: u64) !Arr {
    const key = try ops.slot();
    try mlx.check(mlx.mlx_random_key(key, seed));
    const out = try ops.slot();
    try mlx.check(mlx.mlx_random_normal(out, shape.ptr, shape.len, dtype, loc, scale, key.*, ops.s));
    return out.*;
}

fn expectSameBits(want: Arr, got: Arr) !void {
    try mlx.check(mlx.mlx_array_eval(want));
    try mlx.check(mlx.mlx_array_eval(got));
    try std.testing.expectEqual(mlx.mlx_array_size(want), mlx.mlx_array_size(got));
    try std.testing.expect(sameBytes(want, got));
}

test "GLM KDA body projects f_b and g_b in kernel with MLX's BF16 GEMV bits" {
    if (!mlx.streamIsGpu(mlx.gpuStream())) return error.SkipZigTest;
    const Ops = @import("glm5_model.zig").Ops;
    for ([_]c_int{ 1, 3, 64 }) |heads| {
        for ([_]u64{ 11, 12 }) |seed| {
            var ops = Ops{ .s = mlx.gpuStream() };
            defer ops.deinit();
            const d = heads * 128;
            const sd = seed * 100;
            const a_low = try randn(&ops, &.{ 1, 1, 128 }, .bfloat16, 0, 3, sd + 1);
            const g_low = try randn(&ops, &.{ 1, 1, 128 }, .bfloat16, 0.5, 3, sd + 2);
            const fb = try randn(&ops, &.{ d, 128 }, .bfloat16, 0, 0.1, sd + 3);
            const gb = try randn(&ops, &.{ d, 128 }, .bfloat16, 0, 0.1, sd + 4);
            const a = try ops.binary(.mm, a_low, try ops.transpose(fb, &.{ 1, 0 }));
            const gate = try ops.binary(.mm, g_low, try ops.transpose(gb, &.{ 1, 0 }));
            const base = Inputs{
                .qkv = try randn(&ops, &.{ 1, 1, 3 * d }, .bfloat16, 0, 1.5, sd + 5),
                .beta = try randn(&ops, &.{ 1, 1, heads }, .bfloat16, 0, 2, sd + 6),
                .a = a,
                .gate = gate,
                .conv_weight = try randn(&ops, &.{ 3 * d, 4, 1 }, .bfloat16, 0, 0.5, sd + 7),
                .exp_a = try ops.unary(.exp, try randn(&ops, &.{heads}, .float32, 0, 0.5, sd + 8)),
                .dt_bias = try randn(&ops, &.{d}, .float32, 0, 1, sd + 9),
                .norm = try randn(&ops, &.{128}, .bfloat16, 1, 0.2, sd + 10),
                .conv_state = try randn(&ops, &.{ 1, 3, 3 * d }, .bfloat16, 0, 1.5, sd + 11),
                .state = try randn(&ops, &.{ 1, heads, 128, 128 }, .float32, 0, 0.3, sd + 12),
                .heads = heads,
                .norm_eps = 1e-5,
                .lower = -5,
            };
            const want = (try step(ops.s, base)) orelse return error.TestExpectedFusedKda;
            defer want.deinit();
            var fused = base;
            fused.second_stage = .{ .a_low = a_low, .gate_low = g_low, .fb = fb, .gb = gb };
            const before = dispatchCount();
            const got = (try step(ops.s, fused)) orelse return error.TestExpectedFusedKda;
            defer got.deinit();
            try std.testing.expectEqual(before + 1, dispatchCount());
            try expectSameBits(want.y, got.y);
            try expectSameBits(want.conv, got.conv);
            try expectSameBits(want.state, got.state);

            // Controls: the FP32 state sees every f_b output and y sees every g_b output.
            var nudged = fused;
            nudged.second_stage.?.fb = try ops.binary(.mul, fb, try ops.scalar(1.0078125, .bfloat16));
            nudged.second_stage.?.gb = try ops.binary(.mul, gb, try ops.scalar(1.0078125, .bfloat16));
            const other = (try step(ops.s, nudged)) orelse return error.TestExpectedFusedKda;
            defer other.deinit();
            try mlx.check(mlx.mlx_array_eval(other.state));
            try mlx.check(mlx.mlx_array_eval(other.y));
            try std.testing.expect(!sameBytes(want.state, other.state));
            try std.testing.expect(!sameBytes(want.y, other.y));
        }
    }
}
