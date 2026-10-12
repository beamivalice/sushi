//! Fused GatedDeltaNet decode and verify step (B=1, per-head gate, bf16 activations): the
//! prework (conv, silu, q/k norm, gate, beta) and the recurrence in ONE dispatch,
//! then the norm-gate epilogue, or folds it and rollback history into verify.
//! Each head runs over SPLIT threadgroups in the unfolded kernel; each
//! recomputes its head's prework (cheaper than a barrier between kernels) before
//! its slice of the recurrence rows. Outputs are the bf16 y and the state and capture
//! buffers (bf16 or f32, as the incoming state) of `gdnPreworkFused -> gated_delta_step[_seq]`,
//! bit for bit.
//! Port of mlx-serve `src/gdn_decode.zig` (K1 by David Dalcu; the verify-width
//! kernel and host plumbing from mlx-serve #517 by Samuel Reed); its Hadamard
//! norm-gate-rotate kernel is not ported. The verify fold adapts mlx-serve #558.
const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");

const HEADER = @import("transformer.zig").GDN_KERNEL_HEADER;

const K1_SOURCE =
    \\constexpr int NSG = NT / 32;
    \\constexpr int RB = DV / SPLIT;       // dv rows per threadgroup
    \\constexpr int R = RB / NSG;          // dv rows per simdgroup
    \\constexpr int GRP = HV / HK;
    \\uint lane = thread_index_in_simdgroup;
    \\uint sg = simdgroup_index_in_threadgroup;
    \\uint hv = threadgroup_position_in_grid.x / SPLIT;
    \\uint part = threadgroup_position_in_grid.x % SPLIT;
    \\uint hk = hv / GRP;
    \\threadgroup float qs[DK], ks[DK], vs[DV];
    \\threadgroup float gb[2];
    \\uint row0 = part * RB + sg * R;
    \\float st[R][4];
    \\for (int j = 0; j < R; ++j) {
    \\  uint base = (hv * DV + row0 + j) * DK + lane * 4;
    \\  for (int i = 0; i < 4; ++i) st[j][i] = float(state_in[base + i]);
    \\}
    \\if (sg < 3) {
    \\  uint cb = sg == 0 ? hk * DK : (sg == 1 ? HK * DK + hk * DK : 2 * HK * DK + hv * DV);
    \\  T act[4];
    \\  float sumsq = 0.0f;
    \\  for (int i = 0; i < 4; ++i) {
    \\    uint ch = cb + lane * 4 + i;
    \\    float acc = 0.0f;
    \\    for (int tap = 0; tap < 3; ++tap) acc += float(conv_state[tap * C + ch]) * float(conv_w[ch * 4 + tap]);
    \\    acc += float(qkv[ch]) * float(conv_w[ch * 4 + 3]);
    \\    const T conv = T(acc);
    \\    T sy = T(1) / (T(1) + metal::exp(metal::abs(conv))); T sig = conv < T(0) ? sy : T(1) - sy;
    \\    act[i] = conv * sig;
    \\    float v = float(act[i]);
    \\    sumsq += v * v;
    \\  }
    \\  if (sg < 2) {
    \\    sumsq = simd_sum(sumsq);
    \\    float inv = metal::precise::rsqrt(sumsq / float(DK) + 1e-6f);
    \\    const T scale = sg == 0 ? q_scale : k_scale;
    \\    threadgroup float* dst = sg == 0 ? qs : ks;
    \\    for (int i = 0; i < 4; ++i) dst[lane * 4 + i] = float(scale * T(1) * T(float(act[i]) * inv));
    \\  } else {
    \\    for (int i = 0; i < 4; ++i) vs[lane * 4 + i] = float(act[i]);
    \\  }
    \\  if (part == 0 && (sg == 2 || hv % GRP == 0)) {
    \\    for (int i = 0; i < 4; ++i) {
    \\      uint ch = cb + lane * 4 + i;
    \\      conv_out[ch] = conv_state[C + ch];
    \\      conv_out[C + ch] = conv_state[2 * C + ch];
    \\      conv_out[2 * C + ch] = qkv[ch];
    \\    }
    \\  }
    \\}
    \\if (sg == (NSG > 3 ? 3 : 0) && lane == 31) {
    \\  const T bv = b_in[hv];
    \\  T by = T(1) / (T(1) + metal::exp(metal::abs(bv))); T bsig = bv < T(0) ? by : T(1) - by;
    \\  gb[1] = float(bsig);
    \\  const GT apd = GT(float(a_in[hv]) + float(dt_bias[hv]));
    \\  float sp = sushi_log1p(metal::precise::exp(float(apd)));
    \\  float ea = metal::precise::exp(float(A_log[hv]));
    \\  gb[0] = float(GT(metal::precise::exp(-(ea * sp))));
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\float kk[4], qq[4];
    \\for (int i = 0; i < 4; ++i) { kk[i] = ks[lane * 4 + i]; qq[i] = qs[lane * 4 + i]; }
    \\const float g = gb[0], beta = gb[1];
    \\for (int j = 0; j < R; ++j) {
    \\  uint dv = row0 + j;
    \\  float kv_mem = 0.0f;
    \\  for (int i = 0; i < 4; ++i) { st[j][i] = st[j][i] * g; kv_mem += st[j][i] * kk[i]; }
    \\  kv_mem = simd_sum(kv_mem);
    \\  float delta = (vs[dv] - kv_mem) * beta;
    \\  float out = 0.0f;
    \\  for (int i = 0; i < 4; ++i) { st[j][i] = st[j][i] + kk[i] * delta; out += st[j][i] * qq[i]; }
    \\  out = simd_sum(out);
    \\  uint base = (hv * DV + dv) * DK + lane * 4;
    \\  for (int i = 0; i < 4; ++i) state_out[base + i] = static_cast<StT>(st[j][i]);
    \\  if (lane == 0) y[hv * DV + dv] = static_cast<T>(out);
    \\}
;

// K1 over TL tokens (verify widths). Each captured step stores the state and
// carries the stored value on, as gated_delta_step_seq does, so state_seq[t]
// is the state serial decode holds after token t. state_seq[TL-1] is never
// written (the capture-tail trim).
const K1S_HEAD =
    \\constexpr int NSG = NT / 32;
    \\constexpr int RB = DV / SPLIT;
    \\constexpr int R = RB / NSG;
    \\constexpr int GRP = HV / HK;
    \\uint lane = thread_index_in_simdgroup;
    \\uint sg = simdgroup_index_in_threadgroup;
    \\uint hv = threadgroup_position_in_grid.x / SPLIT;
    \\uint part = threadgroup_position_in_grid.x % SPLIT;
    \\uint hk = hv / GRP;
    \\threadgroup float qs[TL][DK], ks[TL][DK], vs[TL][DV];
    \\threadgroup float gb[TL][2];
    \\uint row0 = part * RB + sg * R;
    \\float st[R][4];
    \\for (int j = 0; j < R; ++j) {
    \\  uint base = (hv * DV + row0 + j) * DK + lane * 4;
    \\  for (int i = 0; i < 4; ++i) st[j][i] = float(state_in[base + i]);
    \\}
    \\if (sg < 3) {
    \\  uint cb = sg == 0 ? hk * DK : (sg == 1 ? HK * DK + hk * DK : 2 * HK * DK + hv * DV);
    \\  for (int t = 0; t < TL; ++t) {
    \\    T act[4];
    \\    float sumsq = 0.0f;
    \\    for (int i = 0; i < 4; ++i) {
    \\      uint ch = cb + lane * 4 + i;
    \\      float acc = 0.0f;
    \\      for (int tap = 0; tap < 4; ++tap) {
    \\        int w = t + tap;
    \\        const T xv = w < 3 ? conv_state[w * C + ch] : qkv[(w - 3) * C + ch];
    \\        acc += float(xv) * float(conv_w[ch * 4 + tap]);
    \\      }
    \\      const T conv = T(acc);
    \\      T sy = T(1) / (T(1) + metal::exp(metal::abs(conv))); T sig = conv < T(0) ? sy : T(1) - sy;
    \\      act[i] = conv * sig;
    \\      float v = float(act[i]);
    \\      sumsq += v * v;
    \\    }
    \\    if (sg < 2) {
    \\      sumsq = simd_sum(sumsq);
    \\      float inv = metal::precise::rsqrt(sumsq / float(DK) + 1e-6f);
    \\      const T scale = sg == 0 ? q_scale : k_scale;
    \\      threadgroup float* dst = sg == 0 ? qs[t] : ks[t];
    \\      for (int i = 0; i < 4; ++i) dst[lane * 4 + i] = float(scale * T(1) * T(float(act[i]) * inv));
    \\    } else {
    \\      for (int i = 0; i < 4; ++i) vs[t][lane * 4 + i] = float(act[i]);
    \\    }
    \\  }
    \\  if (part == 0 && (sg == 2 || hv % GRP == 0)) {
    \\    for (int i = 0; i < 4; ++i) {
    \\      uint ch = cb + lane * 4 + i;
    \\      for (int j = 0; j < 3; ++j) {
    \\        int w = TL + j;
    \\        conv_out[j * C + ch] = w < 3 ? conv_state[w * C + ch] : qkv[(w - 3) * C + ch];
    \\      }
    \\    }
    \\  }
    \\}
    \\if (sg == (NSG > 3 ? 3 : 0) && lane == 31) {
    \\  float ea = metal::precise::exp(float(A_log[hv]));
    \\  for (int t = 0; t < TL; ++t) {
    \\    const T bv = b_in[t * HV + hv];
    \\    T by = T(1) / (T(1) + metal::exp(metal::abs(bv))); T bsig = bv < T(0) ? by : T(1) - by;
    \\    gb[t][1] = float(bsig);
    \\    const GT apd = GT(float(a_in[t * HV + hv]) + float(dt_bias[hv]));
    \\    float sp = sushi_log1p(metal::precise::exp(float(apd)));
    \\    gb[t][0] = float(GT(metal::precise::exp(-(ea * sp))));
    \\  }
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\for (int t = 0; t < TL; ++t) {
    \\  float kk[4], qq[4];
    \\  for (int i = 0; i < 4; ++i) { kk[i] = ks[t][lane * 4 + i]; qq[i] = qs[t][lane * 4 + i]; }
    \\  const float g = gb[t][0], beta = gb[t][1];
    \\  for (int j = 0; j < R; ++j) {
    \\    uint dv = row0 + j;
    \\    float kv_mem = 0.0f;
    \\    for (int i = 0; i < 4; ++i) { st[j][i] = st[j][i] * g; kv_mem += st[j][i] * kk[i]; }
    \\    kv_mem = simd_sum(kv_mem);
    \\    float delta = (vs[t][dv] - kv_mem) * beta;
    \\    float out = 0.0f;
    \\    for (int i = 0; i < 4; ++i) { st[j][i] = st[j][i] + kk[i] * delta; out += st[j][i] * qq[i]; }
    \\    out = simd_sum(out);
;
const K1S_TAIL =
    \\    if (t + 1 < TL) {
    \\      uint sbase = t * (HV * DV * DK) + (hv * DV + dv) * DK + lane * 4;
    \\      for (int i = 0; i < 4; ++i) {
    \\        state_seq[sbase + i] = static_cast<StT>(st[j][i]);
    \\        st[j][i] = static_cast<float>(state_seq[sbase + i]);
    \\      }
    \\    }
    \\  }
    \\}
    \\for (int j = 0; j < R; ++j) {
    \\  uint base = (hv * DV + row0 + j) * DK + lane * 4;
    \\  for (int i = 0; i < 4; ++i) state_out[base + i] = static_cast<StT>(st[j][i]);
    \\}
;

const K1S_SOURCE = K1S_HEAD ++ "\n" ++
    \\    if (lane == 0) y[(t * HV + hv) * DV + dv] = static_cast<T>(out);
++ "\n" ++ K1S_TAIL;

// One threadgroup owns a head. Its stored-precision y stays in shared memory
// for the norm-gate reduction; rollback also receives the convolution inputs.
const K1S_FOLD_SOURCE = "threadgroup float ys[TL][DV];\n" ++ K1S_HEAD ++ "\n" ++
    \\    if (lane == 0) ys[t][dv] = float(static_cast<T>(out));
++ "\n" ++ K1S_TAIL ++ "\n" ++
    \\if (sg < 3 && (sg == 2 || hv % GRP == 0)) {
    \\  uint cb = sg == 0 ? hk * DK : (sg == 1 ? HK * DK + hk * DK : 2 * HK * DK + hv * DV);
    \\  for (int i = 0; i < 4; ++i) {
    \\    uint ch = cb + lane * 4 + i;
    \\    for (int w = 0; w < 3 + TL; ++w) conv_in[w * C + ch] = w < 3 ? conv_state[w * C + ch] : qkv[(w - 3) * C + ch];
    \\  }
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (sg < TL) {
    \\  float xs[4];
    \\  float sumsq = 0.0f;
    \\  for (int i = 0; i < 4; ++i) { xs[i] = ys[sg][lane * 4 + i]; sumsq += xs[i] * xs[i]; }
    \\  sumsq = simd_sum(sumsq);
    \\  float inv = metal::precise::rsqrt(sumsq / float(DV) + eps);
    \\  uint base = (sg * HV + hv) * DV + lane * 4;
    \\  for (int i = 0; i < 4; ++i) {
    \\    const T normed = norm_w[lane * 4 + i] * T(xs[i] * inv);
    \\    const T zv = z[base + i];
    \\    T sy = T(1) / (T(1) + metal::exp(metal::abs(zv))); T sig = zv < T(0) ? sy : T(1) - sy;
    \\    gated[base + i] = SWISH ? (zv * sig) * normed : normed * sig;
    \\  }
    \\}
;

const SPLIT: c_int = 4;
const NT: c_int = 256; // 4 dv rows per simdgroup
pub const MAX_SEQ: c_int = 8;

pub const Geometry = struct { hk: c_int, hv: c_int, dk: c_int, dv: c_int };

/// The unprojected GDN inputs the chain's prework reads, for S tokens.
pub const Inputs = struct {
    qkv: mlx.mlx_array, // [1,S,C]
    a: mlx.mlx_array, // [1,S,Hv]
    b: mlx.mlx_array, // [1,S,Hv]
    conv_state: mlx.mlx_array, // [1,3,C]
    ssm_state: mlx.mlx_array, // [1,Hv,Dv,Dk]
    conv_w: mlx.mlx_array, // [C,4,1]
    A_log: mlx.mlx_array, // [Hv]
    dt_bias: mlx.mlx_array, // [Hv]
    q_scale: mlx.mlx_array, // 0-dim bf16
    k_scale: mlx.mlx_array, // 0-dim bf16
    gate_dtype: mlx.mlx_dtype = .bfloat16, // the decay gate's rounding: bf16 or f32
};

/// y [1,S,Hv,Dv], the next conv state [1,3,C] and the final state [1,Hv,Dv,Dk];
/// `state_seq` [S,1,Hv,Dv,Dk] (row S-1 unwritten) at verify widths only.
pub const Recur = struct {
    y: mlx.mlx_array,
    conv_state: mlx.mlx_array,
    ssm_state: mlx.mlx_array,
    state_seq: mlx.mlx_array = .{ .ctx = null },

    pub fn deinit(self: Recur) void {
        inline for (.{ self.y, self.conv_state, self.ssm_state, self.state_seq }) |a| {
            if (a.ctx != null) _ = mlx.mlx_array_free(a);
        }
    }
};

var k1_kernel: ?mlx.mlx_fast_metal_kernel = null;
var k1s_kernel: ?mlx.mlx_fast_metal_kernel = null;
const Widths = struct { g: Geometry, state: mlx.mlx_dtype, gate: mlx.mlx_dtype };
var cfg_key: ?Widths = null;
/// Index = token count; 1 is the decode kernel's, 2..MAX_SEQ the verify kernel's.
var cfgs: [MAX_SEQ + 1]?mlx.mlx_fast_metal_kernel_config = @splat(null);

fn makeKernel(name: [*:0]const u8, outs: []const [*:0]const u8, source: [*:0]const u8) !mlx.mlx_fast_metal_kernel {
    const ins = [_][*:0]const u8{ "qkv", "a_in", "b_in", "conv_state", "state_in", "conv_w", "A_log", "dt_bias", "q_scale", "k_scale" };
    const in_vec = mlx.mlx_vector_string_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, source, HEADER, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    return k;
}

fn buildConfig(g: Geometry, t_len: c_int, state: mlx.mlx_dtype, gate: mlx.mlx_dtype) !mlx.mlx_fast_metal_kernel_config {
    const c = 2 * g.hk * g.dk + g.hv * g.dv;
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, t_len, g.hv, g.dv }, 4, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, 3, c }, 3, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, g.hv, g.dv, g.dk }, 4, state));
    if (t_len > 1) try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ t_len, 1, g.hv, g.dv, g.dk }, 5, state));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, g.hv * SPLIT * NT, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, NT, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "StT", state));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "GT", gate));
    inline for (.{ .{ "HK", g.hk }, .{ "HV", g.hv }, .{ "DK", g.dk }, .{ "DV", g.dv }, .{ "C", c }, .{ "NT", NT }, .{ "SPLIT", SPLIT } }) |kv|
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, kv[0], kv[1]));
    if (t_len > 1) try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "TL", t_len));
    return cfg;
}

fn shapeIs(a: mlx.mlx_array, want: []const c_int) bool {
    return std.mem.eql(c_int, mlx.getShape(a), want);
}

fn isStateWidth(dt: mlx.mlx_dtype) bool {
    return dt == .bfloat16 or dt == .float32;
}

fn inputsFit(g: Geometry, t_len: c_int, in: Inputs) bool {
    if (t_len < 1 or t_len > MAX_SEQ) return false;
    if (g.dk != 128 or g.dv != 128 or g.hk < 1 or @rem(g.hv, g.hk) != 0) return false;
    for ([_]mlx.mlx_array{ in.qkv, in.a, in.b, in.conv_state, in.conv_w, in.dt_bias, in.q_scale, in.k_scale }) |arr|
        if (mlx.mlx_array_dtype(arr) != .bfloat16) return false;
    if (!isStateWidth(mlx.mlx_array_dtype(in.ssm_state)) or !isStateWidth(in.gate_dtype)) return false;
    const c = 2 * g.hk * g.dk + g.hv * g.dv;
    if (!shapeIs(in.qkv, &.{ 1, t_len, c }) or !shapeIs(in.a, &.{ 1, t_len, g.hv }) or !shapeIs(in.b, &.{ 1, t_len, g.hv }) or
        !shapeIs(in.conv_state, &.{ 1, 3, c }) or !shapeIs(in.ssm_state, &.{ 1, g.hv, g.dv, g.dk }) or
        mlx.mlx_array_size(in.conv_w) != @as(usize, @intCast(c * 4)) or mlx.getShape(in.conv_w)[0] != c) return false;

    return mlx.mlx_array_size(in.A_log) == @as(usize, @intCast(g.hv)) and
        mlx.mlx_array_size(in.dt_bias) == @as(usize, @intCast(g.hv)) and
        mlx.mlx_array_size(in.q_scale) == 1 and mlx.mlx_array_size(in.k_scale) == 1;
}

/// Prework + recurrence over `t_len` tokens (1 = decode, 2..MAX_SEQ = verify
/// with per-step state capture). Null outside the kernels' geometry, off bf16 activations (the
/// chain's prework and norm-gate serve no other) and for a state or gate that is neither bf16 nor f32.
pub fn step(g: Geometry, t_len: c_int, in: Inputs, s: mlx.mlx_stream) !?Recur {
    if (!mlx.streamIsGpu(s) or !inputsFit(g, t_len, in)) return null;

    const seq = t_len > 1;
    const kernel = if (seq)
        k1s_kernel orelse blk: {
            k1s_kernel = try makeKernel("sushi_gdn_decode_recur_seq", &.{ "y", "conv_out", "state_out", "state_seq" }, K1S_SOURCE);
            break :blk k1s_kernel.?;
        }
    else
        k1_kernel orelse blk: {
            k1_kernel = try makeKernel("sushi_gdn_decode_recur", &.{ "y", "conv_out", "state_out" }, K1_SOURCE);
            break :blk k1_kernel.?;
        };
    const widths = Widths{ .g = g, .state = mlx.mlx_array_dtype(in.ssm_state), .gate = in.gate_dtype };
    if (cfg_key == null or !std.meta.eql(cfg_key.?, widths)) {
        for (&cfgs) |*slot| if (slot.*) |cf| {
            _ = mlx.mlx_fast_metal_kernel_config_free(cf);
            slot.* = null;
        };
        cfg_key = widths;
    }
    const idx: usize = @intCast(t_len);
    if (cfgs[idx] == null) cfgs[idx] = try buildConfig(g, t_len, widths.state, widths.gate);

    const ins = [_]mlx.mlx_array{ in.qkv, in.a, in.b, in.conv_state, in.ssm_state, in.conv_w, in.A_log, in.dt_bias, in.q_scale, in.k_scale };
    const v = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(v);
    var o = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&o, kernel, v, cfgs[idx].?, s));
    var out = Recur{ .y = mlx.mlx_array_new(), .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new() };
    if (seq) out.state_seq = mlx.mlx_array_new();
    errdefer out.deinit();
    try mlx.check(mlx.mlx_vector_array_get(&out.y, o, 0));
    try mlx.check(mlx.mlx_vector_array_get(&out.conv_state, o, 1));
    try mlx.check(mlx.mlx_vector_array_get(&out.ssm_state, o, 2));
    if (seq) try mlx.check(mlx.mlx_vector_array_get(&out.state_seq, o, 3));
    return out;
}


pub const Fold = struct {
    gated: mlx.mlx_array,
    conv_state: mlx.mlx_array,
    ssm_state: mlx.mlx_array,
    state_seq: mlx.mlx_array,
    conv_input: mlx.mlx_array,

    pub fn deinit(self: Fold) void {
        inline for (.{ self.gated, self.conv_state, self.ssm_state, self.state_seq, self.conv_input }) |a| {
            if (a.ctx != null) _ = mlx.mlx_array_free(a);
        }
    }
};


const FOLD_NT: c_int = 1024;
pub var fold_nt_override: ?c_int = null;
var fold_kernel: ?mlx.mlx_fast_metal_kernel = null;
var fold_cfgs: [MAX_SEQ + 1]?mlx.mlx_fast_metal_kernel_config = @splat(null);
var fold_ok: [MAX_SEQ + 1]?bool = @splat(null);
const FoldKey = struct { g: Geometry, swish: bool, nt: c_int, state: mlx.mlx_dtype, gate: mlx.mlx_dtype };
var fold_key: ?FoldKey = null;

pub fn foldDeclined(t_len: c_int) bool {
    return t_len >= 2 and t_len <= MAX_SEQ and fold_ok[@intCast(t_len)] == false;
}

fn buildFoldConfig(key: FoldKey, t_len: c_int) !mlx.mlx_fast_metal_kernel_config {
    const g = key.g;
    const c = 2 * g.hk * g.dk + g.hv * g.dv;
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const out_dtypes = [_]mlx.mlx_dtype{ .bfloat16, .bfloat16, key.state, key.state, .bfloat16 };
    inline for (.{ &[_]c_int{ 1, t_len, g.hv * g.dv }, &[_]c_int{ 1, 3, c }, &[_]c_int{ 1, g.hv, g.dv, g.dk }, &[_]c_int{ t_len, 1, g.hv, g.dv, g.dk }, &[_]c_int{ 1, 3 + t_len, c } }, 0..) |shape, i|
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, shape, shape.len, out_dtypes[i]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, g.hv * key.nt, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, key.nt, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "StT", key.state));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "GT", key.gate));
    inline for (.{ .{ "HK", g.hk }, .{ "HV", g.hv }, .{ "DK", g.dk }, .{ "DV", g.dv }, .{ "C", c }, .{ "NT", key.nt }, .{ "SPLIT", @as(c_int, 1) }, .{ "TL", t_len }, .{ "SWISH", @as(c_int, @intFromBool(key.swish)) } }) |kv|
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, kv[0], kv[1]));
    return cfg;
}

/// Verify recurrence, norm-gate and rollback convolution history in one dispatch.
/// The recurrence retains the stored state between tokens, as serial decode does.
pub fn stepFold(g: Geometry, t_len: c_int, in: Inputs, z: mlx.mlx_array, norm_w: mlx.mlx_array, eps: mlx.mlx_array, swish: bool, s: mlx.mlx_stream) !?Fold {
    if (!mlx.streamIsGpu(s) or t_len < 2 or !inputsFit(g, t_len, in)) return null;
    if (mlx.mlx_array_dtype(z) != .bfloat16 or mlx.mlx_array_dtype(norm_w) != .bfloat16 or mlx.mlx_array_dtype(eps) != .float32) return null;
    if (!shapeIs(z, &.{ 1, t_len, g.hv * g.dv }) or !shapeIs(norm_w, &.{g.dv}) or mlx.mlx_array_size(eps) != 1) return null;
    if (fold_kernel == null) {
        const ins = [_][*:0]const u8{ "qkv", "a_in", "b_in", "conv_state", "state_in", "conv_w", "A_log", "dt_bias", "q_scale", "k_scale", "z", "norm_w", "eps" };
        const outs = [_][*:0]const u8{ "gated", "conv_out", "state_out", "state_seq", "conv_in" };
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const k = mlx.mlx_fast_metal_kernel_new("sushi_gdn_verify_fold", iv, ov, K1S_FOLD_SOURCE, HEADER, true, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        fold_kernel = k;
    }
    const key = FoldKey{ .g = g, .swish = swish, .nt = fold_nt_override orelse FOLD_NT, .state = mlx.mlx_array_dtype(in.ssm_state), .gate = in.gate_dtype };
    if (fold_key == null or !std.meta.eql(fold_key.?, key)) {
        for (&fold_cfgs) |*slot| if (slot.*) |cfg| {
            _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
            slot.* = null;
        };
        fold_key = key;
        fold_ok = @splat(null);
    }
    const idx: usize = @intCast(t_len);
    if (fold_ok[idx] == false) return null;
    if (fold_cfgs[idx] == null) fold_cfgs[idx] = try buildFoldConfig(key, t_len);
    const ins = [_]mlx.mlx_array{ in.qkv, in.a, in.b, in.conv_state, in.ssm_state, in.conv_w, in.A_log, in.dt_bias, in.q_scale, in.k_scale, z, norm_w, eps };
    // Probe with independent zeros: evaluating real inputs here could read a PLE
    // leaf before the caller fills it, or synchronize a still-lazy draft chain.
    if (fold_ok[idx] == null) {
        if (mlx.errorPending()) return error.MlxError;
        var dummy: [ins.len]mlx.mlx_array = @splat(.{ .ctx = null });
        defer for (dummy) |a| { if (a.ctx != null) _ = mlx.mlx_array_free(a); };
        for (ins, 0..) |a, i| {
            dummy[i] = mlx.mlx_array_new();
            const shape = mlx.getShape(a);
            try mlx.check(mlx.mlx_zeros(&dummy[i], shape.ptr, shape.len, mlx.mlx_array_dtype(a), s));
        }
        _ = mlx.mlx_array_free(dummy[12]);
        dummy[12] = mlx.mlx_array_new_float(1e-6);
        const pv = mlx.mlx_vector_array_new_data(&dummy, dummy.len);
        defer _ = mlx.mlx_vector_array_free(pv);
        var po = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(po);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&po, fold_kernel.?, pv, fold_cfgs[idx].?, s));
        const rc = mlx.mlx_eval(po);
        if (mlx.takeErrorIf("maximum allowed threads per threadgroup")) {
            fold_ok[idx] = false;
            log.info("[gdn-fold] declined at S={d}: pipeline threadgroup limit below {d}\n", .{ t_len, key.nt });
            return null;
        }
        try mlx.check(rc);
        if (mlx.errorPending()) return error.MlxError;
        fold_ok[idx] = true;
    }
    const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, fold_kernel.?, iv, fold_cfgs[idx].?, s));
    var out = Fold{ .gated = mlx.mlx_array_new(), .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .state_seq = mlx.mlx_array_new(), .conv_input = mlx.mlx_array_new() };
    errdefer out.deinit();
    inline for (.{ &out.gated, &out.conv_state, &out.ssm_state, &out.state_seq, &out.conv_input }, 0..) |a, i|
        try mlx.check(mlx.mlx_vector_array_get(a, ov, i));
    return out;
}
