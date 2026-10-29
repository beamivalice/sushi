const trf = @import("transformer.zig");
const std = @import("std");
const mlx = @import("mlx.zig");
const kv_quant = @import("kv_quant.zig");
const test_support = @import("test_support.zig");
const attn256MaxAbs = test_support.attn256MaxAbs;
const attn256MaxDiff = test_support.attn256MaxDiff;
const attn256RandBf16 = test_support.attn256RandBf16;
const attn256RandBf16Scaled = test_support.attn256RandBf16Scaled;
const attn256Reference = test_support.attn256Reference;
const KVQuantConfig = kv_quant.KVQuantConfig;
const model_mod = @import("model.zig");
const log = @import("log.zig");
const ModelConfig = model_mod.ModelConfig;
const nax_gate = @import("nax_gate.zig");
const naxSdpaPreferred = nax_gate.naxSdpaPreferred;
const qsaNaxOsOk = nax_gate.qsaNaxOsOk;
const qsaProbeAllFinite = nax_gate.qsaProbeAllFinite;
const qsaProbeCorruptFirst = nax_gate.qsaProbeCorruptFirst;
const qsaProbeLcgBf16 = nax_gate.qsaProbeLcgBf16;
const qsaProbeMaxAbsDiff = nax_gate.qsaProbeMaxAbsDiff;
const verifyQmmNaxAvailable = nax_gate.verifyQmmNaxAvailable;
const DenseKVView = trf.DenseKVView;
const KVCache = trf.KVCache;
const SWA_TEST_WINDOW = trf.SWA_TEST_WINDOW;
const ShapeKey = trf.ShapeKey;
const SwaGeom = trf.SwaGeom;
const TestQuantView = trf.TestQuantView;
const Transformer = trf.Transformer;
const absMax = trf.absMax;
const attnPdCausalTruth = trf.attnPdCausalTruth;
const attnPdTestArms = trf.attnPdTestArms;
const attnPdTruthF32 = trf.attnPdTruthF32;
const expectAttnPdNoWorse = trf.expectAttnPdNoWorse;
const sliceSeq = trf.sliceSeq;
const slidingViewFor = trf.slidingViewFor;
const swaBandMask = trf.swaBandMask;
const swaHostF32 = trf.swaHostF32;
const swaMaxErr = trf.swaMaxErr;
const swaSinkTestConfig = trf.swaSinkTestConfig;
const swaSinkTruth = trf.swaSinkTruth;

const ATTN_PD_KERNEL_HEADER =
    \\#include <metal_simdgroup_matrix>
    \\
    \\// Fragment layout mirrors MLX steel BaseMMAFrag<float,8,8>: each thread
    \\// of a simdgroup holds 2 adjacent elements of an 8x8 tile; the hardware
    \\// mma runs on simdgroup_float8x8 built from those elements. The 4 threads
    \\// holding one row differ in lane bits 0 and 3 (see sushi_coord).
    \\inline short2 sushi_coord(ushort lane) {
    \\  const short qid = lane / 4;
    \\  const short fm = (qid & 4) + ((lane / 2) % 4);
    \\  const short fn = (qid & 2) * 2 + (lane % 2) * 2;
    \\  return short2(fn, fm);
    \\}
    \\
    \\inline void sushi_mma(thread float2 &d, float2 a, float2 b) {
    \\  metal::simdgroup_float8x8 D, A, B, C;
    \\  A.thread_elements()[0] = a.x;
    \\  A.thread_elements()[1] = a.y;
    \\  B.thread_elements()[0] = b.x;
    \\  B.thread_elements()[1] = b.y;
    \\  C.thread_elements()[0] = d.x;
    \\  C.thread_elements()[1] = d.y;
    \\  simdgroup_multiply_accumulate(D, A, B, C);
    \\  d.x = D.thread_elements()[0];
    \\  d.y = D.thread_elements()[1];
    \\}
    \\
    \\inline float sushi_row_max(float2 v) {
    \\  float t = metal::max(v.x, v.y);
    \\  t = metal::max(t, metal::simd_shuffle_xor(t, ushort(1)));
    \\  t = metal::max(t, metal::simd_shuffle_xor(t, ushort(8)));
    \\  return t;
    \\}
    \\
    \\inline float sushi_row_sum(float2 v) {
    \\  float t = v.x + v.y;
    \\  t += metal::simd_shuffle_xor(t, ushort(1));
    \\  t += metal::simd_shuffle_xor(t, ushort(8));
    \\  return t;
    \\}
    \\
;

// v2 tile geometry (2026-07-12 micro-bench sweep on the M4 Max, 24q/4kv
// geometry): BQ=64 with 8 simdgroups halves the K/V staging walks
// per query row vs the v1 32-row tile; BK=32 halves the per-column softmax
// rescale + barrier overhead; Q lives in REGISTERS loaded straight from
// global (no Q smem at all — that memory goes to the wider K tile); staging
// runs on uint4 (8 bf16 per instruction, was scalar). Measured vs v1 at
// (qL x kL): (2048x16384) 250 -> 73 ms, (2048x65536) 999 -> 325 ms,
// (8192x8192) 279 -> 77 ms. vs the composed path the v2 kernel wins where
// kL <= ~4*qL (+63% at ratio 1, +15% at 2, +1.8% at 4) and loses beyond
// (-2.3% at 6, -6% at 8) — the causal arm's ratio gate (fusedSdpaPrefill)
// encodes exactly that envelope. Rejected variants (measured worse): BK=8
// ping-pong (per-block rescale doubles), V-from-global fragments (device
// loads too slow), K-only double-buffer (barrier structure unchanged),
// Q-smem BQ=32 with 2 threadgroups/core (SLC thrash at long kL).
const ATTN_PD_KERNEL_SOURCE =
    \\constexpr int BQ = 64;
    \\constexpr int BK = 32;
    \\constexpr int LDK = BK + 8;
    \\constexpr int LDV = BDV + 8;
    \\constexpr int NT = 256;
    \\// uint4 staging strides: 8 halves per instruction along each width.
    \\constexpr int NWK = BDK / 8;
    \\constexpr int NWV = BDV / 8;
    \\
    \\const int qL = q_shape[2];
    \\const int kL = k_shape[2];
    \\const int Hq = q_shape[1];
    \\const int Hk = k_shape[1];
    \\const int gqa = Hq / Hk;
    \\
    \\const int tqx = int(threadgroup_position_in_grid.x);
    \\const int hq = int(threadgroup_position_in_grid.y);
    \\const int bb = int(threadgroup_position_in_grid.z);
    \\const ushort lane = ushort(thread_index_in_simdgroup);
    \\const ushort warp = ushort(simdgroup_index_in_threadgroup);
    \\const int tix = int(thread_index_in_threadgroup);
    \\
    \\// exp2-based softmax (steel convention): fold log2(e) into the scale.
    \\const float scale_log2e = scl[0] * 1.44269504088896340736f;
    \\const int SW = win[0];
    \\
    \\// Budgeted kv-chunk dispatch (causal arm): this dispatch covers keys
    \\// [k_begin, k_end); online-softmax state (m/l/unnormalized O) rides fp32
    \\// carry buffers between dispatches — exact register precision, so the
    \\// chunked chain is bit-identical to one long dispatch. phase bit 0 =
    \\// carry-in present, bit 1 = final chunk (normalize + bf16 store).
    \\const int k_begin = kr[0];
    \\const int k_end = metal::min(kr[1], kL);
    \\// The buffer is not always the cache: a quantized cache hands each
    \\// dispatch a DEQUANTIZED SLICE, so `koff` is where this buffer's row 0
    \\// sits in the cache and `kL_abs` is the cache's own length. Every causal
    \\// and band comparison is in cache coordinates; staging is in buffer
    \\// coordinates. A dense caller passes {0, kL} and nothing below moves.
    \\const int koff = kr[2];
    \\const int kL_abs = kr[3];
    \\const bool has_carry = (phase[0] & 1) != 0;
    \\const bool is_final = (phase[0] & 2) != 0;
    \\
    \\// Bottom-right aligned causal: query row r of this chunk sits at
    \\// absolute KV position q_off + r.
    \\const int q_off = kL_abs - qL;
    \\
    \\const device T* Qp = q + bb * q_strides[0] + hq * q_strides[1]
    \\    + (long)(tqx * BQ) * q_strides[2];
    \\const device T* Kp = k + bb * k_strides[0] + (hq / gqa) * k_strides[1];
    \\const device T* Vp = v + bb * v_strides[0] + (hq / gqa) * v_strides[1];
    \\device T* Op = out + (((long)bb * Hq + hq) * (long)qL + (long)(tqx * BQ)) * BDV;
    \\
    \\// K^T tile [BDK][LDK] (20.5 KB at qk 256) is strictly larger than the
    \\// V tile [BK][LDV] (16.9 KB at v 256) — share one buffer, steel-style
    \\// (barriers separate the K reads from the V staging).
    \\constexpr int SMEM = (LDK * BDK > BK * LDV) ? (LDK * BDK) : (BK * LDV);
    \\threadgroup T KVs[SMEM];
    \\threadgroup T* Ks = KVs;
    \\threadgroup T* Vs = KVs;
    \\
    \\const int q_rows = metal::min(BQ, qL - tqx * BQ);
    \\
    \\const short2 sc = sushi_coord(lane);
    \\const short sn = sc.x;
    \\const short sm = sc.y;
    \\const short tm = 8 * short(warp);
    \\const int Ks_off = sm * LDK + sn;
    \\const int Vs_off = sm * LDV + sn;
    \\
    \\// Q fragment straight from global into registers, once per threadgroup:
    \\// row (tm+sm), element pairs (dd*8+sn, +1). sn is always even, so the
    \\// vec2 load is 4B-aligned for any row whose stride is even (structural:
    \\// a q row strides BDK). Rows past the ragged tail read zero — their
    \\// outputs are discarded by the store guard.
    \\float2 Qfrag[NWK];
    \\{
    \\  const int qr = tm + sm;
    \\  if (qr < q_rows) {
    \\    const device T* Qrow = Qp + (long)qr * q_strides[2];
    \\    for (int dd = 0; dd < NWK; ++dd) {
    \\      const vec<T, 2> p = *((const device vec<T, 2>*)(Qrow + dd * 8 + sn));
    \\      Qfrag[dd] = float2(float(p.x), float(p.y));
    \\    }
    \\  } else {
    \\    for (int dd = 0; dd < NWK; ++dd) Qfrag[dd] = float2(0.0f);
    \\  }
    \\}
    \\
    \\float2 Ofrag[NWV];
    \\for (int i = 0; i < NWV; ++i) Ofrag[i] = float2(0.0f);
    \\// Init max FINITE (not -inf) so the rescale factor exp2(old-new) can
    \\// never be exp2(nan); masked scores use true -INFINITY so a row whose
    \\// first blocks are fully banded out contributes exp2(-inf)=0, not 1.
    \\float max_score = -3.0e38f;
    \\float sum_score = 0.0f;
    \\
    \\// Resume carried softmax state from the previous kv chunk. Rows past
    \\// the ragged q tail never carry (their outputs are discarded anyway).
    \\// (m_in/l_in/o_in are indexed directly, never through a typed pointer:
    \\// MLX binds sub-4KB inputs — the no-carry dummies — in the `constant`
    \\// address space and real carries in `device`, and the source must
    \\// compile under both.)
    \\if (has_carry) {
    \\  const int qr = tm + sm;
    \\  if (qr < q_rows) {
    \\    const long crow = (((long)bb * Hq + hq) * (long)qL + (long)(tqx * BQ + qr));
    \\    max_score = m_in[crow];
    \\    sum_score = l_in[crow];
    \\    const long obase = crow * (long)BDV;
    \\    for (int dd = 0; dd < NWV; ++dd) {
    \\      Ofrag[dd] = float2(o_in[obase + dd * 8 + sn], o_in[obase + dd * 8 + sn + 1]);
    \\    }
    \\  }
    \\}
    \\// Learned per-head sink (template SINK=1): one extra softmax column that
    \\// joins the max and the sum and carries no value row, so it is exactly a
    \\// running state of (max = sink, sum = 1, O = 0) before the first key.
    \\if (SINK && !has_carry) {
    \\  max_score = sinks[hq] * 1.44269504088896340736f;
    \\  sum_score = 1.0f;
    \\}
    \\
    \\const int NK = (k_end + BK - 1) / BK;
    \\const int q_lo = tqx * BQ + q_off;
    \\const int q_hi = q_lo + BQ - 1;
    \\const int kb_lim = metal::min(NK, (q_hi - koff + BK) / BK);
    \\int kb = k_begin / BK;
    \\if (SW > 0) kb = metal::max(kb, metal::max(0, q_lo - SW + 1 - koff) / BK);
    \\const int kb_min_causal = metal::max(0, q_lo - koff) / BK;
    \\const int row_pos = q_lo + tm + sm;
    \\
    \\// QSA arm (template QSA=1): `mask` is the [B,1,qL,kL] bool visibility
    \\// (causal folded in), `skip` the [B, nq_tiles, NK] per-tile any-visible
    \\// table — a key block no row of this tile sees is never staged.
    \\const long skip_base = ((long)bb * ((qL + BQ - 1) / BQ) + tqx) * (long)((kL + BK - 1) / BK);
    \\const long mask_row0 = QSA ? ((long)bb * mask_strides[0] + (long)(tqx * BQ + tm + sm) * mask_strides[2]) : 0;
    \\const bool mask_row_ok = (tqx * BQ + tm + sm) < qL;
    \\
    \\for (; kb < kb_lim; kb++) {
    \\  if (QSA && !skip[skip_base + kb]) continue;
    \\  const int c0 = kb * BK;
    \\  const int rows_k = metal::min(BK, kL - c0);
    \\
    \\  // Stage K transposed (Ks[d][kk]): uint4 global loads (8 bf16 each),
    \\  // scalar transposed scatter into smem. Consecutive lanes take consecutive
    \\  // KEY rows, so a store lands on adjacent addresses of one Ks row; lanes
    \\  // spread over head-dim chunks would stride 8*LDK halves, one bank.
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  for (int i = tix; i < BK * NWK; i += NT) {
    \\    const int r = i % BK;
    \\    const int c8 = i / BK;
    \\    uint4 w = uint4(0);
    \\    if (r < rows_k) {
    \\      w = *((const device uint4*)(Kp + (long)(c0 + r) * k_strides[2]) + c8);
    \\    }
    \\    thread T* e = (thread T*)&w;
    \\    const int cb = c8 * 8;
    \\    Ks[(cb + 0) * LDK + r] = e[0];
    \\    Ks[(cb + 1) * LDK + r] = e[1];
    \\    Ks[(cb + 2) * LDK + r] = e[2];
    \\    Ks[(cb + 3) * LDK + r] = e[3];
    \\    Ks[(cb + 4) * LDK + r] = e[4];
    \\    Ks[(cb + 5) * LDK + r] = e[5];
    \\    Ks[(cb + 6) * LDK + r] = e[6];
    \\    Ks[(cb + 7) * LDK + r] = e[7];
    \\  }
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\
    \\  // S = Q @ K^T for this simdgroup's 8 query rows.
    \\  float2 Sfrag[BK / 8];
    \\  for (int i = 0; i < BK / 8; ++i) Sfrag[i] = float2(0.0f);
    \\  for (int dd = 0; dd < NWK; ++dd) {
    \\    const float2 qf = Qfrag[dd];
    \\    const int kbase = Ks_off + dd * 8 * LDK;
    \\    const float2 kf0 = float2(float(Ks[kbase]), float(Ks[kbase + 1]));
    \\    const float2 kf1 = float2(float(Ks[kbase + 8]), float(Ks[kbase + 9]));
    \\    const float2 kf2 = float2(float(Ks[kbase + 16]), float(Ks[kbase + 17]));
    \\    const float2 kf3 = float2(float(Ks[kbase + 24]), float(Ks[kbase + 25]));
    \\    sushi_mma(Sfrag[0], qf, kf0);
    \\    sushi_mma(Sfrag[1], qf, kf1);
    \\    sushi_mma(Sfrag[2], qf, kf2);
    \\    sushi_mma(Sfrag[3], qf, kf3);
    \\  }
    \\  Sfrag[0] *= scale_log2e;
    \\  Sfrag[1] *= scale_log2e;
    \\  Sfrag[2] *= scale_log2e;
    \\  Sfrag[3] *= scale_log2e;
    \\
    \\  // Masking: kL remainder + causal + sliding band, all element-wise.
    \\  const bool tail_k = (rows_k < BK);
    \\  const bool need_causal = (kb >= kb_min_causal);
    \\  const bool need_band = (SW > 0) && (koff + c0 <= q_hi - SW);
    \\  if (QSA || tail_k || need_causal || need_band) {
    \\    for (int kt = 0; kt < BK / 8; ++kt) {
    \\      for (int jj = 0; jj < 2; ++jj) {
    \\        const int col = c0 + kt * 8 + sn + jj;
    \\        const int col_abs = koff + col;
    \\        bool masked = false;
    \\        if (tail_k && col >= kL) masked = true;
    \\        if (need_causal && row_pos < col_abs) masked = true;
    \\        if (need_band && (row_pos - col_abs) >= SW) masked = true;
    \\        if (QSA && !masked && (!mask_row_ok || !mask[mask_row0 + (long)col_abs * mask_strides[3]])) masked = true;
    \\        if (masked) Sfrag[kt][jj] = -INFINITY;
    \\      }
    \\    }
    \\  }
    \\
    \\  // Stage V (same smem as K — K reads are done): uint4 on both sides.
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  for (int i = tix; i < BK * NWV; i += NT) {
    \\    const int r = i / NWV;
    \\    const int c8 = i % NWV;
    \\    uint4 w = uint4(0);
    \\    if (r < rows_k) {
    \\      w = *((const device uint4*)(Vp + (long)(c0 + r) * v_strides[2]) + c8);
    \\    }
    \\    *((threadgroup uint4*)(Vs + r * LDV) + c8) = w;
    \\  }
    \\
    \\  // Online softmax (registers only, overlaps the V staging above).
    \\  float new_max = max_score;
    \\  new_max = metal::max(new_max, sushi_row_max(Sfrag[0]));
    \\  new_max = metal::max(new_max, sushi_row_max(Sfrag[1]));
    \\  new_max = metal::max(new_max, sushi_row_max(Sfrag[2]));
    \\  new_max = metal::max(new_max, sushi_row_max(Sfrag[3]));
    \\  Sfrag[0] = metal::exp2(Sfrag[0] - new_max);
    \\  Sfrag[1] = metal::exp2(Sfrag[1] - new_max);
    \\  Sfrag[2] = metal::exp2(Sfrag[2] - new_max);
    \\  Sfrag[3] = metal::exp2(Sfrag[3] - new_max);
    \\  const float factor = metal::exp2(max_score - new_max);
    \\  max_score = new_max;
    \\  const float rowsum = sushi_row_sum(Sfrag[0]) + sushi_row_sum(Sfrag[1])
    \\      + sushi_row_sum(Sfrag[2]) + sushi_row_sum(Sfrag[3]);
    \\  sum_score = sum_score * factor + rowsum;
    \\  for (int i = 0; i < NWV; ++i) Ofrag[i] *= factor;
    \\
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\
    \\  // O += P @ V.
    \\  for (int id = 0; id < NWV; ++id) {
    \\    const int vbase = Vs_off + id * 8;
    \\    const float2 vf0 = float2(float(Vs[vbase]), float(Vs[vbase + 1]));
    \\    const float2 vf1 = float2(float(Vs[vbase + 8 * LDV]),
    \\                              float(Vs[vbase + 8 * LDV + 1]));
    \\    const float2 vf2 = float2(float(Vs[vbase + 16 * LDV]),
    \\                              float(Vs[vbase + 16 * LDV + 1]));
    \\    const float2 vf3 = float2(float(Vs[vbase + 24 * LDV]),
    \\                              float(Vs[vbase + 24 * LDV + 1]));
    \\    sushi_mma(Ofrag[id], Sfrag[0], vf0);
    \\    sushi_mma(Ofrag[id], Sfrag[1], vf1);
    \\    sushi_mma(Ofrag[id], Sfrag[2], vf2);
    \\    sushi_mma(Ofrag[id], Sfrag[3], vf3);
    \\  }
    \\}
    \\
    \\// Final chunk: normalize + store bf16 (output freshly allocated,
    \\// contiguous). Mid chunk: store the raw fp32 state for the next
    \\// dispatch (m/l written by all 4 lane-threads of a row — same value).
    \\const int local_row = tm + sm;
    \\if (local_row < q_rows) {
    \\  if (is_final) {
    \\    const float inv = 1.0f / sum_score;
    \\    device T* Optr = Op + (long)local_row * BDV + sn;
    \\    for (int id = 0; id < NWV; ++id) {
    \\      Optr[id * 8] = T(Ofrag[id].x * inv);
    \\      Optr[id * 8 + 1] = T(Ofrag[id].y * inv);
    \\    }
    \\  } else {
    \\    const long crow = (((long)bb * Hq + hq) * (long)qL + (long)(tqx * BQ + local_row));
    \\    m_out[crow] = max_score;
    \\    l_out[crow] = sum_score;
    \\    device float* Orow = o_out + crow * (long)BDV;
    \\    for (int id = 0; id < NWV; ++id) {
    \\      Orow[id * 8 + sn] = Ofrag[id].x;
    \\      Orow[id * 8 + sn + 1] = Ofrag[id].y;
    \\    }
    \\  }
    \\}
;

/// Which kernel a `sushi_attn_pd` dispatch runs: the SIMD-matrix kernel above, or
/// `sushi_attn_pd_nax` (src/kernels/attn_pd_nax.metal) on the M5 matrix units.
pub const AttnPdArm = enum { simd, nax };

/// [simd, nax]
var attn_pd_kernels: [2]?mlx.mlx_fast_metal_kernel = .{ null, null };

fn getAttnPdKernel(arm: AttnPdArm) !mlx.mlx_fast_metal_kernel {
    const slot = &attn_pd_kernels[@intFromEnum(arm)];
    if (slot.*) |kk| return kk;
    const kernel = try newAttnPdKernel(
        if (arm == .nax) "sushi_attn_pd_nax" else "sushi_attn_pd",
        if (arm == .nax) @embedFile("kernels/attn_pd_nax.metal") else ATTN_PD_KERNEL_SOURCE,
        if (arm == .nax) @embedFile("kernels/attn_pd_nax_header.metal") else ATTN_PD_KERNEL_HEADER,
    );
    slot.* = kernel;
    return kernel;
}

fn newAttnPdKernel(name: [*:0]const u8, source: [*:0]const u8, header: [*:0]const u8) !mlx.mlx_fast_metal_kernel {
    const input_names = [_][*:0]const u8{ "q", "k", "v", "scl", "win", "kr", "phase", "m_in", "l_in", "o_in", "mask", "skip", "sinks" };
    const output_names = [_][*:0]const u8{ "out", "m_out", "l_out", "o_out" };
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        name,
        in_vec,
        out_vec,
        source,
        header,
        false, // ensure_row_contiguous=false — K/V are cache VIEWS; a forced
        // contiguous copy of the full cache per layer would erase the win.
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    return kernel;
}

/// Test seam: the arm the LAST fusedSdpaPrefill call ran.
pub var attn_pd_last_arm: AttnPdArm = .simd;

/// null = not probed yet; the verdict holds for the process.
var attn_pd_nax_probe_ok: ?bool = null;
/// Test seam: perturb the probe's NAX output so the numeric check must fail.
pub var attn_pd_nax_probe_corrupt_override: bool = false;

pub fn attnPdNaxProbeReset() void {
    attn_pd_nax_probe_ok = null;
}

/// The NAX arm serves the MiMo global and sliding widths (192/128) without a QSA mask, on
/// matrix-unit silicon with cooperative input tensors (macOS 26.3, `qsaNaxOsOk`) whose
/// one-tile probe passed.
fn attnPdNaxServes(qk_dim: c_int, v_dim: c_int, masked: bool) bool {
    if (masked or qk_dim != 192 or v_dim != 128) return false;
    if (!(attn_pd_nax_override orelse true) or !verifyQmmNaxAvailable() or !qsaNaxOsOk()) return false;
    if (attn_pd_nax_probe_ok == null) attn_pd_nax_probe_ok = attnPdNaxRunProbe();
    return attn_pd_nax_probe_ok.?;
}

const ATTN_PD_NAX_PROBE_FLOOR: f32 = 4.9e-4;

/// Metal JIT-compiles at first EVAL, so both instantiations the engine dispatches (causal, and
/// band + sinks) run one ragged tile against an f32 composed reference before any prefill
/// trusts them: no worse than 1.5x the SIMD kernel's error. The latch a failure raised is dropped.
pub fn attnPdNaxRunProbe() bool {
    if (mlx.noGpuBackend()) return false;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return false;
    const had_error = mlx.errorPending();
    defer mlx.dropLatchedErrorUnless(had_error);
    for ([_]bool{ false, true }) |banded| {
        if (!attnPdNaxProbeCase(s, banded, had_error)) {
            log.info("[attn-pd] sushi_attn_pd_nax declined: probe failed ({s}); sushi_attn_pd serves\n", .{if (banded) "band + sinks" else "causal"});
            return false;
        }
    }
    return true;
}

fn attnPdNaxProbeCase(s: mlx.mlx_stream, banded: bool, had_error: bool) bool {
    // A ragged simdgroup (40 rows) and a partial key block (72 keys), gqa 2.
    const ql: c_int = 40;
    const kl: c_int = 72;
    const window: c_int = if (banded) 24 else 0;
    const scale: f32 = 1.0 / @sqrt(192.0);
    const q = qsaProbeLcgBf16(s, &[_]c_int{ 1, 2, ql, 192 }, 0xA77D01) orelse return false;
    defer _ = mlx.mlx_array_free(q);
    const k = qsaProbeLcgBf16(s, &[_]c_int{ 1, 1, kl, 192 }, 0xA77D02) orelse return false;
    defer _ = mlx.mlx_array_free(k);
    const v = qsaProbeLcgBf16(s, &[_]c_int{ 1, 1, kl, 128 }, 0xA77D03) orelse return false;
    defer _ = mlx.mlx_array_free(v);
    const sink_data = [_]f32{ 0.25, -0.5 };
    const sinks: mlx.mlx_array = if (banded) mlx.mlx_array_new_data(&sink_data, &[_]c_int{2}, 1, .float32) else .{ .ctx = null };
    defer if (sinks.ctx != null) {
        _ = mlx.mlx_array_free(sinks);
    };

    var dispatches: u32 = 0;
    const simd = (attnPdDispatch(s, .simd, q, k, v, scale, window, null, null, sinks, &dispatches) catch return false) orelse return false;
    defer _ = mlx.mlx_array_free(simd);
    var nax = (attnPdDispatch(s, .nax, q, k, v, scale, window, null, null, sinks, &dispatches) catch return false) orelse return false;
    defer _ = mlx.mlx_array_free(nax);
    if (mlx.mlx_array_eval(simd) != 0 or mlx.mlx_array_eval(nax) != 0) return false;
    if (!had_error and mlx.errorPending()) return false;
    if (attn_pd_nax_probe_corrupt_override) {
        const bad = qsaProbeCorruptFirst(s, nax) orelse return false;
        _ = mlx.mlx_array_free(nax);
        nax = bad;
    }

    var mask_buf: [ql * kl]bool = undefined;
    for (0..ql) |r| for (0..kl) |c| {
        const row: c_int = kl - ql + @as(c_int, @intCast(r));
        const col: c_int = @intCast(c);
        mask_buf[r * kl + c] = col <= row and (window == 0 or row - col < window);
    };
    const mask = mlx.mlx_array_new_data(&mask_buf, &[_]c_int{ 1, 1, ql, kl }, 4, .bool_);
    defer _ = mlx.mlx_array_free(mask);
    var wide: [3]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new() };
    defer for (wide) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for ([_]mlx.mlx_array{ q, k, v }, &wide) |src, *dst| {
        if (mlx.mlx_astype(dst, src, .float32, s) != 0) return false;
    }
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    if (mlx.mlx_fast_scaled_dot_product_attention(&ref, wide[0], wide[1], wide[2], scale, "array", mask, sinks, false, s) != 0) return false;
    if (!qsaProbeAllFinite(s, nax) or !qsaProbeAllFinite(s, simd) or !qsaProbeAllFinite(s, ref)) return false;
    const err_simd = qsaProbeMaxAbsDiff(s, simd, ref) orelse return false;
    const err_nax = qsaProbeMaxAbsDiff(s, nax, ref) orelse return false;
    return err_nax <= @max(1.5 * err_simd, ATTN_PD_NAX_PROBE_FLOOR);
}

/// Test seam: `fused256_override = false` forces the composed path on both arms (the parity oracle).
pub var fused256_override: ?bool = null;
/// Test seam: true = the NAX arm wherever the hardware and its probe allow it, false = SIMD.
pub var attn_pd_nax_override: ?bool = null;

pub fn fused256Enabled() bool {
    return fused256_override orelse true;
}

/// Plain-CAUSAL dispatch mode. Default `.all` since the kv-chunk dispatch
/// budget landed (2026-07-22): the historical net-LOSS of every pre-budget
/// ratio-gated variant (8K: fused-to-ratio-4 231.5, fused-to-ratio-2 232.9,
/// composed 234.7 tok/s — monotonic: more fused = slower, despite the v2
/// kernel winning the µbench at kL <= 2*qL) was the macOS IOGPU
/// interactivity-preemption class: one long dispatch scanning the whole KV
/// gets preempted, and the penalty scales with dispatch length. With the
/// key axis split into budget-sized dispatches (see
/// FUSED256_DEFAULT_DISPATCH_BUDGET) the causal arm WINS the same
/// same-session A/B on the 27B: +2.9%/+2.3%/+4.6% at 8K/16K/32K.
/// The SLIDING-BAND arm (window > 0) has no such competition — composed
/// computes full-width scores + a GB-scale mask while the kernel
/// block-skips outside the band — so it
/// rides the master switch alone and never chunks.
pub const Fused256CausalMode = enum { all, off };

// ── Dispatch work budget (kv-axis chunked dispatch) ──
// One long dispatch scanning the whole KV monopolizes the GPU long enough at
// high kv_len to trip macOS IOGPU interactivity preemption, collapsing
// long-context prefill (oMLX issue #2225: the M3 Max cliff sits near ~1.5e9
// work units ≈ 50 ms/dispatch; design ported from oMLX's
// qwen35_fa256_attention.py, Apache-2.0, oMLX by jundot). Above the budget
// (work = batch·Hq·qL·kL) the CAUSAL arm splits the key axis into
// separately-dispatched BK-aligned chunks with exact online-softmax carry
// (fp32 m/l/O buffers chained between dispatches — bit-identical to the
// single-dispatch result). The band arm never chunks: its in-kernel block
// skip already bounds per-dispatch work by the window.
// The default mirrors oMLX's 250M fallback (~23 ms/dispatch on the
// M4 Max at the kernel's measured ~1.1e10 work-units/s).
pub const FUSED256_DEFAULT_DISPATCH_BUDGET: i64 = 250_000_000;
/// `sushi_attn_pd_nax` does the same work in about a third of the time, and its lockstep simdgroups
/// keep a longer key chunk in cache, so fewer fp32 carries pay.
pub const ATTN_PD_NAX_DISPATCH_BUDGET: i64 = 1_000_000_000;

/// Rows of a quantized cache one dispatch may dequantize. The work budget
/// already chunks the key axis, but it scales with heads x q_len, so a narrow
/// forward would otherwise rebuild the whole prompt in one slice. 16384 rows
/// is ~21 MB at the widest cache geometry we serve.
pub const PACKED_KV_SLICE_MAX: c_int = 16384;

/// Test seam: forces the budget (0 = single dispatch).
pub var fused256_budget_override: ?i64 = null;

pub fn fused256DispatchBudget(arm: AttnPdArm) i64 {
    if (fused256_budget_override) |v| return v;
    return if (arm == .nax) ATTN_PD_NAX_DISPATCH_BUDGET else FUSED256_DEFAULT_DISPATCH_BUDGET;
}

/// kv-axis chunk length for the budgeted causal dispatch: the largest
/// BK(32)-multiple keeping work-per-dispatch = batch·hq·ql·chunk within
/// budget, floored at one BK block, capped at kl. budget <= 0 = one dispatch.
pub fn fused256KvChunkLen(batch: i64, hq: i64, ql: i64, kl: i64, budget: i64) c_int {
    if (budget <= 0) return @intCast(kl);
    const per_key = batch * hq * ql;
    var chunk = @divTrunc(budget, @max(per_key, 1));
    chunk = @divTrunc(chunk, 32) * 32;
    if (chunk < 32) chunk = 32;
    if (chunk > kl) chunk = kl;
    return @intCast(chunk);
}

/// Test seam: dispatches issued by the LAST fusedSdpaPrefill call
/// (engagement is counted, never inferred from output equality).
pub var fused256_last_dispatch_count: u32 = 0;

pub fn fused256CausalMode() Fused256CausalMode {
    return if (fused256Enabled()) .all else .off;
}

/// (query/key width, value width) pairs `sushi_attn_pd` is instantiated for.
/// MLX's own steel prefill kernel ships bd 64/96/128 plus a 256 dsplit, so
/// 192/128 — MiMo-V2's global layers, and the MLA shape that scores over
/// nope+rope while storing narrower values — has no fused arm anywhere else.
pub fn fusedPdDims(qk_dim: c_int, v_dim: c_int) bool {
    return (qk_dim == 256 and v_dim == 256) or (qk_dim == 192 and v_dim == 128);
}

/// ONE predicate consumed by the three prefill OOM guards AND the dispatch
/// sites: true when prefill attention at this head_dim will NOT materialize
/// the composed [heads, chunk, total_kv] score tensor — either MLX's own
/// fused kernel covers it (<= 128) or `sushi_attn_pd` does for EVERY arm.
/// Keyed on causal mode: at `.all` (the default since the budgeted-dispatch
/// flip) no score tensor materializes. Guards and dispatch must never drift
/// (the effectivePrefillChunk rule).
///
/// A width listed here owes a dispatch at EVERY prefill site that scores at
/// it: 192 covers mimo_v2's global layers and MLA. mimo_v2's SLIDING layers
/// (band + sinks) ride their own predicate, `slidingPrefillFused`, which
/// `server.slidingBandScoreBytes` reads too.
pub fn prefillHeadDimFused(head_dim: u32) bool {
    if (head_dim <= 128) return true;
    if (head_dim != 256 and head_dim != 192) return false;
    return fused256CausalMode() == .all;
}

/// Shortest query block the fused hd-256 kernel accepts. Below it MLX's
/// sdpa_vector wins (see the decline in fusedSdpaPrefill).
pub const FUSED256_MIN_Q_LEN: c_int = 16;

/// Try the fused flash prefill kernel (sushi_attn_pd, see `fusedPdDims`).
/// Returns null when a precondition doesn't hold — the caller falls back to
/// the composed path. `window` > 0 adds the sliding-band mask (sliding-window layers,
/// createSlidingWindowMask semantics: key masked when row_abs - col >=
/// window); 0 = plain bottom-right causal. Free function (stream param) so
/// the MTP head can use it too.
///
/// STRUCTURAL precondition (not checkable — lazy arrays carry no strides at
/// graph-build time): q/k/v innermost dim contiguous. True for fresh
/// rope/transpose outputs, cache views (slices along T of a contiguous
/// buffer), and dequantized dense views — i.e. every attention call site.
pub fn fusedSdpaPrefill(
    s: mlx.mlx_stream,
    q: mlx.mlx_array,
    k: mlx.mlx_array,
    v: mlx.mlx_array,
    scale: f32,
    window: c_int,
) !?mlx.mlx_array {
    if (!fusedPrefillArmOn(q, window)) return null;
    return fusedSdpaPrefillImpl(s, q, k, v, scale, window, null, null, .{ .ctx = null });
}

fn fusedPrefillArmOn(q: mlx.mlx_array, window: c_int) bool {
    // Band arm rides the master switch; plain causal is opt-in (see
    // fused256CausalMode — composed wins live on qwen chunked prefill).
    if (window > 0) return fused256Enabled();
    if (fused256CausalMode() == .off) return false;
    // The band arm stays ours (mlx's NAX kernel takes no band); the
    // causal arm yields to the stock fused kernel on NAX — which exists
    // at hd 256 only, so a narrower-value shape keeps our kernel.
    return !(naxSdpaPreferred() and mlx.mlx_array_ndim(q) == 4 and mlx.getShape(q)[3] == 256);
}

/// The causal arm over a KV cache view. When the cache is quantized the kernel
/// reads it one dispatch at a time (`PackedKv`) and `view.k`/`view.v` — the
/// lazy dense rebuild `denseView` set up — are never evaluated. Falls through
/// to the dense arm on an `off`-scheme cache or a shape the slicing cannot
/// serve, so the caller has one entry either way. The band arm reads the view
/// whole: a ringed sliding view is already one dispatch's rows. `sinks` ([Hq],
/// null = none) is a learned logit per head that joins the softmax as one
/// extra column with no value row.
pub fn fusedSdpaPrefillKv(
    s: mlx.mlx_stream,
    q: mlx.mlx_array,
    view: *const DenseKVView,
    scale: f32,
    window: c_int,
    sinks: mlx.mlx_array,
) !?mlx.mlx_array {
    if (window > 0 or !view.has_quant_triple or view.bits == 0 or view.group_size == 0) {
        if (!fusedPrefillArmOn(q, window)) return null;
        return fusedSdpaPrefillImpl(s, q, view.k, view.v, scale, window, null, null, sinks);
    }
    if (fused256CausalMode() == .off) return null;
    return fusedSdpaPrefillImpl(s, q, view.k, view.v, scale, 0, null, .{
        .k_q = view.k_triple_q,
        .k_scales = view.k_triple_scales,
        .k_biases = view.k_triple_biases,
        .v_q = view.v_triple_q,
        .v_scales = view.v_triple_scales,
        .v_biases = view.v_triple_biases,
        .bits = view.bits,
        .group_size = view.group_size,
    }, sinks);
}

/// Test seam for the switch half of `slidingPrefillFused` (never the shape half).
pub var sliding_prefill_fused_override: ?bool = null;

/// ONE predicate for a sliding layer's prefill dispatch (`slidingPrefillAttn`)
/// and `server.slidingBandScoreBytes`: true when the band + sink attention runs
/// in `sushi_attn_pd` and no [heads, q, window + q - 1] score sheet exists.
pub fn slidingPrefillFused(cfg: *const ModelConfig, q_len: u64) bool {
    const on = sliding_prefill_fused_override orelse fused256Enabled();
    const v_dim = if (cfg.v_head_dim > 0) cfg.v_head_dim else cfg.head_dim;
    return on and q_len >= FUSED256_MIN_Q_LEN and fusedPdDims(@intCast(cfg.head_dim), @intCast(v_dim));
}

var sliding_prefill_engaged_logged = false;

/// A sliding layer's prefill attention over its (ringed) view, sinks included.
/// Null = the composed path serves.
pub fn slidingPrefillAttn(
    s: mlx.mlx_stream,
    cfg: *const ModelConfig,
    q: mlx.mlx_array,
    view: *const DenseKVView,
    scale: f32,
    sinks: mlx.mlx_array,
) !?mlx.mlx_array {
    const q_len = mlx.getShape(q)[2];
    if (!slidingPrefillFused(cfg, @intCast(q_len))) return null;
    const out = try fusedSdpaPrefillKv(s, q, view, scale, @intCast(cfg.sliding_window), sinks);
    if (out != null and !sliding_prefill_engaged_logged) {
        sliding_prefill_engaged_logged = true;
        log.info("[attn-pd] sliding band engaged: {s} window={d} sinks={} qL={d} kL={d}\n", .{ attnPdKernelName(attn_pd_last_arm), cfg.sliding_window, sinks.ctx != null, q_len, mlx.getShape(view.k)[2] });
    }
    return out;
}

/// One line per (qk, value) width the kernel actually serves in this process:
/// an A/B arm is proven by an engagement line in its own log, not by the flag
/// it was launched with.
var attn_pd_engaged_logged: [2][2]bool = .{ .{ false, false }, .{ false, false } };

fn attnPdKernelName(arm: AttnPdArm) []const u8 {
    return if (arm == .nax) "sushi_attn_pd_nax" else "sushi_attn_pd";
}

fn logAttnPdEngaged(arm: AttnPdArm, qk_dim: c_int, v_dim: c_int, q_len: c_int, kv_len: c_int, dispatches: u32) void {
    const slot = &attn_pd_engaged_logged[@intFromEnum(arm)][if (qk_dim == 256) 0 else 1];
    if (slot.*) return;
    slot.* = true;
    log.info("[attn-pd] engaged: {s} qk={d} v={d} qL={d} kL={d} dispatches={d}\n", .{ attnPdKernelName(arm), qk_dim, v_dim, q_len, kv_len, dispatches });
}

/// True when a bf16 view's bound address is 16-byte aligned, which every `uint4` K/V staging load needs. A lazy
/// array has no address yet and passes: the stride gates are all that can be checked for it.
pub fn attn256Base16Aligned(a: mlx.mlx_array) bool {
    var avail = false;
    if (mlx._mlx_array_is_available(&avail, a) != 0 or !avail) return true;
    const p = mlx.mlx_array_data_uint32(a) orelse return true;
    return @intFromPtr(p) % 16 == 0;
}

/// Per-(batch, q tile, key block) any-visible table for the QSA arm:
/// [B, ceil(qL/64), ceil(kL/32)] bool.
fn qsaSkipTable(s: mlx.mlx_stream, mask: mlx.mlx_array) !mlx.mlx_array {
    const ms = mlx.getShape(mask);
    const nq: c_int = @divTrunc(ms[2] + 63, 64);
    const nk: c_int = @divTrunc(ms[3] + 31, 32);
    const axes = [_]c_int{ 2, 3 };
    const low = [_]c_int{ 0, 0 };
    const high = [_]c_int{ nq * 64 - ms[2], nk * 32 - ms[3] };
    const falsev = mlx.mlx_array_new_bool(false);
    defer _ = mlx.mlx_array_free(falsev);
    var padded = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(padded);
    try mlx.check(mlx.mlx_pad(&padded, mask, &axes, 2, &low, 2, &high, 2, falsev, "constant", s));
    var r5 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r5);
    try mlx.check(mlx.mlx_reshape(&r5, padded, &[_]c_int{ ms[0], nq, 64, nk, 32 }, 5, s));
    const red = [_]c_int{ 2, 4 };
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_any_axes(&out, r5, &red, 2, false, s));
    return out;
}

/// Virtual key index -> cache position for one query's block selection:
/// the `sel_len` tokens of its sorted blocks, then its own incomplete tail.
pub const ATTN_QSA256_KERNEL_HEADER = ATTN_PD_KERNEL_HEADER ++
    \\inline int sushi_qsa_pos(const device int* blk, int vi, int sel_len, int tail_start, int ratio) {
    \\  const int b = vi / ratio;
    \\  return (vi < sel_len) ? (blk[b] * ratio + (vi - b * ratio)) : (tail_start + (vi - sel_len));
    \\}
    \\
;

/// Dequantizes chunk `c8` (8 values) of one packed row into `ev`. 4 and 8 bits fill whole uint32
/// words; 6 bits packs four values into three bytes (little-endian bitstream, `mlx_quantize`'s
/// layout), so a chunk is six bytes and the row is read bytewise.
pub const KV_UNPACK_HEADER =
    \\template <typename T, int BITS>
    \\inline void sushi_kv_unpack8(const device uint32_t* wq, int c8, float sj, float bj, thread T* ev) {
    \\  constexpr uint MASKB = (1u << BITS) - 1u;
    \\  if constexpr (BITS == 6) {
    \\    const device uchar* p = ((const device uchar*)wq) + c8 * 6;
    \\    const uint lo = uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16);
    \\    const uint hi = uint(p[3]) | (uint(p[4]) << 8) | (uint(p[5]) << 16);
    \\    for (int u = 0; u < 4; ++u) {
    \\      ev[u] = T(float((lo >> (u * 6)) & MASKB) * sj + bj);
    \\      ev[4 + u] = T(float((hi >> (u * 6)) & MASKB) * sj + bj);
    \\    }
    \\  } else {
    \\    constexpr int VPW = 32 / BITS;
    \\    constexpr int NW = 8 / VPW;
    \\    for (int wj = 0; wj < NW; ++wj) {
    \\      const uint w = wq[c8 * NW + wj];
    \\      for (int u = 0; u < VPW; ++u) ev[wj * VPW + u] = T(float((w >> (u * BITS)) & MASKB) * sj + bj);
    \\    }
    \\  }
    \\}
    \\
    \\// Sixteen 6-bit values from three words (96 bits), as the matmul2d kernels load them.
    \\template <typename T>
    \\inline void sushi_kv_store16_b6(uint w0, uint w1, uint w2, float sj, float bj, threadgroup T* dst) {
    \\  const uint ws[4] = {w0, w1, w2, 0u};
    \\  #pragma clang loop unroll(full)
    \\  for (int u = 0; u < 16; ++u) {
    \\    const int bit = u * 6;
    \\    const int wi = bit >> 5;
    \\    const int sh = bit & 31;
    \\    uint v = ws[wi] >> sh;
    \\    if (sh + 6 > 32) v |= ws[wi + 1] << (32 - sh);
    \\    dst[u] = T(float(v & 63u) * sj + bj);
    \\  }
    \\}
    \\
;

pub const QSA_SCORE_KERNEL_HEADER =
    \\#include <metal_stdlib>
    \\#include <metal_simdgroup>
    \\#include <metal_simdgroup_matrix>
    \\#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
    \\using namespace metal;
    \\
;

const QSA_ATTN_Q_SOURCE =
    \\constexpr int KC = DK / 8;
    \\constexpr int VC = DV / 8;
    \\constexpr int LDK = BK + 8;
    \\constexpr int LDV = DV + 8;
    \\// Simdgroups past NSG only stage tiles: who loads an element never changes its value.
    \\constexpr int NT = 32 * NSGT;
    \\constexpr int KT = BK / 8;
    \\constexpr int KV_ELEMS = (DK * LDK > BK * LDV) ? DK * LDK : BK * LDV;
    \\// Per-length split counts ride the grid, never a template.
    \\const int NSPLIT = int(threadgroups_per_grid.z);
    \\
    \\const int qL = q_shape[2];
    \\const int kL = kq_shape[2];
    \\const int Hq = q_shape[1];
    \\const int Hk = kq_shape[1];
    \\const int gqa = Hq / Hk;
    \\const int KB = blocks_shape[2];
    \\
    \\const int s = int(threadgroup_position_in_grid.x);
    \\const int hk = int(threadgroup_position_in_grid.y);
    \\// The caller pins batch to 1 (qs[0] != 1 declines), so the grid's z is
    \\// free to carry the split index instead of the batch index.
    \\const int bb = 0;
    \\const int jsp = int(threadgroup_position_in_grid.z);
    \\const ushort lane = ushort(thread_index_in_simdgroup);
    \\const ushort warp = ushort(simdgroup_index_in_threadgroup);
    \\const int tix = int(thread_index_in_threadgroup);
    \\
    \\const float scale_log2e = scl[0] * 1.44269504088896340736f;
    \\// Query s sits at absolute key position p (bottom-right aligned). Its
    \\// visible keys: `count` complete blocks, then the tail [tail_start, p].
    \\// SPARSE=0 has no block list: its whole causal range [0, p] is the tail.
    \\const int p = (kL - qL) + s;
    \\const int complete = (p + 1) / RATIO;
    \\const int count = SPARSE ? metal::min(complete, KB) : 0;
    \\const int sel_len = count * RATIO;
    \\const int tail_start = SPARSE ? complete * RATIO : 0;
    \\const int L = sel_len + (p + 1 - tail_start);
    \\// Split-K. This threadgroup owns virtual keys [t_lo, t_hi) of the row's
    \\// own range [0, L). The virtual index already concatenates the row's
    \\// selected blocks and THEN its ragged tail, so a contiguous virtual
    \\// range is exactly a contiguous block range — plus, for the last
    \\// non-empty split, the tail. Boundaries are BK-aligned so every tile
    \\// but a split's last is full. A split starting past L is EMPTY and
    \\// writes the merge's identity.
    \\const int tiles = (L + BK - 1) / BK;
    \\// ceil(tiles/NSPLIT) per split, so the last splits can be short or empty; the
    \\// merge is exact over whatever the boundaries are.
    \\const int tiles_per = (tiles + NSPLIT - 1) / NSPLIT;
    \\const int t_lo = jsp * tiles_per * BK;
    \\const int t_hi = metal::min(L, t_lo + tiles_per * BK);
    \\
    \\const long block_base = (long)bb * blocks_strides[0] + (long)s * blocks_strides[1];
    \\const long kq_base  = (long)bb * kq_strides[0]  + (long)hk * kq_strides[1];
    \\const long ksc_base = (long)bb * ksc_strides[0] + (long)hk * ksc_strides[1];
    \\const long kbi_base = (long)bb * kbi_strides[0] + (long)hk * kbi_strides[1];
    \\const long vq_base  = (long)bb * vq_strides[0]  + (long)hk * vq_strides[1];
    \\const long vsc_base = (long)bb * vsc_strides[0] + (long)hk * vsc_strides[1];
    \\const long vbi_base = (long)bb * vbi_strides[0] + (long)hk * vbi_strides[1];
    \\
    \\threadgroup T KVs[KV_ELEMS];
    \\threadgroup T* Ks = KVs;
    \\threadgroup T* Vs = KVs;
    \\
    \\const short2 sc = sushi_coord(lane);
    \\const short sn = sc.x;
    \\const short sm = sc.y;
    \\const short tm = 8 * short(warp);
    \\const int Ks_off = sm * LDK + sn;
    \\const int Vs_off = sm * LDV + sn;
    \\const int row = tm + sm;
    \\const bool row_ok = row < gqa;
    \\const bool mma_sg = int(warp) < NSG;
    \\
    \\float2 Qfrag[KC];
    \\if (row_ok) {
    \\  const device T* Qrow = q + bb * q_strides[0] + (long)(hk * gqa + row) * q_strides[1] + (long)s * q_strides[2];
    \\  for (int dd = 0; dd < KC; ++dd) {
    \\    const vec<T, 2> pr = *((const device vec<T, 2>*)(Qrow + dd * 8 + sn));
    \\    Qfrag[dd] = float2(float(pr.x), float(pr.y));
    \\  }
    \\} else {
    \\  for (int dd = 0; dd < KC; ++dd) Qfrag[dd] = float2(0.0f);
    \\}
    \\float2 Ofrag[VC];
    \\for (int i = 0; i < VC; ++i) Ofrag[i] = float2(0.0f);
    \\float max_score = -3.0e38f;
    \\float sum_score = 0.0f;
    \\
    \\for (int t0 = t_lo; t0 < t_hi; t0 += BK) {
    \\  const int rows_k = metal::min(BK, t_hi - t0);
    \\
    \\  // K tile: dequantize into the SAME transposed staging the dense
    \\  // kernel fills, so the fragment math below is untouched. The 8 dims of
    \\  // one chunk share a quant group (GS is a multiple of 8), so the
    \\  // scale/bias pair is read once per chunk, not per element.
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  for (int i = tix; i < BK * KC; i += NT) {
    \\    const int r = i / KC;
    \\    const int c8 = i % KC;
    \\    const int cb = c8 * 8;
    \\    T ev[8];
    \\    if (r < rows_k) {
    \\      const int vi = t0 + r;
    \\      const int bi = vi / RATIO;
    \\      const int pos = vi < sel_len ? blocks[block_base + bi] * RATIO + (vi - bi * RATIO) : tail_start + vi - sel_len;
    \\      if (QUANT) {
    \\      const long wrow = kq_base + (long)pos * kq_strides[2];
    \\      const int g = cb / GS;
    \\      const float sj = float(ksc[ksc_base + (long)pos * ksc_strides[2] + (long)g]);
    \\      const float bj = float(kbi[kbi_base + (long)pos * kbi_strides[2] + (long)g]);
    \\      sushi_kv_unpack8<T, BITS>((const device uint32_t*)(kq + wrow), c8, sj, bj, ev);
    \\      } else {
    \\        for (int j = 0; j < 8; ++j) ev[j] = T(kq[kq_base + (long)pos * kq_strides[2] + cb + j]);
    \\      }
    \\    } else {
    \\      for (int j = 0; j < 8; ++j) ev[j] = T(0.0f);
    \\    }
    \\    for (int j = 0; j < 8; ++j) Ks[(cb + j) * LDK + r] = ev[j];
    \\  }
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\
    \\  float2 Sfrag[KT];
    \\  for (int i = 0; i < KT; ++i) Sfrag[i] = float2(0.0f);
    \\  if (mma_sg) {
    \\  for (int dd = 0; dd < KC; ++dd) {
    \\    const float2 qf = Qfrag[dd];
    \\    const int kbase = Ks_off + dd * 8 * LDK;
    \\    for (int kt = 0; kt < KT; ++kt) {
    \\      const float2 kf = float2(float(Ks[kbase + kt * 8]), float(Ks[kbase + kt * 8 + 1]));
    \\      sushi_mma(Sfrag[kt], qf, kf);
    \\    }
    \\  }
    \\  for (int kt = 0; kt < KT; ++kt) Sfrag[kt] *= scale_log2e;
    \\  if (rows_k < BK) {
    \\    for (int kt = 0; kt < KT; ++kt) {
    \\      if (kt * 8 + sn >= rows_k) Sfrag[kt].x = -INFINITY;
    \\      if (kt * 8 + sn + 1 >= rows_k) Sfrag[kt].y = -INFINITY;
    \\    }
    \\  }
    \\  }
    \\
    \\  // V tile: dequantize straight into the row-major staging.
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  for (int i = tix; i < BK * VC; i += NT) {
    \\    const int r = i / VC;
    \\    const int c8 = i % VC;
    \\    const int cb = c8 * 8;
    \\    T ev[8];
    \\    if (r < rows_k) {
    \\      const int vi = t0 + r;
    \\      const int bi = vi / RATIO;
    \\      const int pos = vi < sel_len ? blocks[block_base + bi] * RATIO + (vi - bi * RATIO) : tail_start + vi - sel_len;
    \\      if (QUANT) {
    \\      const long wrow = vq_base + (long)pos * vq_strides[2];
    \\      const int g = cb / GS;
    \\      const float sj = float(vsc[vsc_base + (long)pos * vsc_strides[2] + (long)g]);
    \\      const float bj = float(vbi[vbi_base + (long)pos * vbi_strides[2] + (long)g]);
    \\      sushi_kv_unpack8<T, BITS>((const device uint32_t*)(vq + wrow), c8, sj, bj, ev);
    \\      } else {
    \\        for (int j = 0; j < 8; ++j) ev[j] = T(vq[vq_base + (long)pos * vq_strides[2] + cb + j]);
    \\      }
    \\    } else {
    \\      for (int j = 0; j < 8; ++j) ev[j] = T(0.0f);
    \\    }
    \\    for (int j = 0; j < 8; ++j) Vs[r * LDV + cb + j] = ev[j];
    \\  }
    \\
    \\  if (mma_sg) {
    \\  float new_max = max_score;
    \\  for (int kt = 0; kt < KT; ++kt) new_max = metal::max(new_max, sushi_row_max(Sfrag[kt]));
    \\  float rowsum = 0.0f;
    \\  for (int kt = 0; kt < KT; ++kt) {
    \\    Sfrag[kt] = metal::exp2(Sfrag[kt] - new_max);
    \\    rowsum += sushi_row_sum(Sfrag[kt]);
    \\  }
    \\  const float factor = metal::exp2(max_score - new_max);
    \\  max_score = new_max;
    \\  sum_score = sum_score * factor + rowsum;
    \\  for (int i = 0; i < VC; ++i) Ofrag[i] *= factor;
    \\  }
    \\
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  if (mma_sg) {
    \\  for (int id = 0; id < VC; ++id) {
    \\    const int vbase = Vs_off + id * 8;
    \\    for (int kt = 0; kt < KT; ++kt) {
    \\      const float2 vf = float2(float(Vs[vbase + kt * 8 * LDV]), float(Vs[vbase + kt * 8 * LDV + 1]));
    \\      sushi_mma(Ofrag[id], Sfrag[kt], vf);
    \\    }
    \\  }
    \\  }
    \\}
    \\
    \\// Partials for the merge pass, UNNORMALIZED: the running max and the
    \\// running sum belong to this split alone, and only the merge knows the
    \\// row's global max. `sushi_row_max`/`sushi_row_sum` reduce across the four
    \\// lanes holding a row, so every lane of the row carries the same m/l
    \\// and exactly one of them (sn == 0) writes the pair.
    \\//
    \\// An empty split writes l = 0, which is the merge's skip flag: a
    \\// NON-empty split always has l >= 1, because its own maximum element
    \\// contributes exp2(max - max) = 1. So l == 0 identifies emptiness
    \\// exactly, and the merge never evaluates (-inf) - (-inf).
    \\if (row_ok) {
    \\  const int hq = hk * gqa + row;
    \\  const long pidx = ((long)hq * (long)qL + (long)s) * NSPLIT + jsp;
    \\  device float* Aptr = pacc + pidx * DV + sn;
    \\  for (int id = 0; id < VC; ++id) {
    \\    Aptr[id * 8] = Ofrag[id].x;
    \\    Aptr[id * 8 + 1] = Ofrag[id].y;
    \\  }
    \\  if (sn == 0) {
    \\    const bool empty = (t_lo >= t_hi);
    \\    pml[pidx * 2] = empty ? -INFINITY : max_score;
    \\    pml[pidx * 2 + 1] = empty ? 0.0f : sum_score;
    \\  }
    \\}
;

/// sushi_attn_qsa256_qmerge: the split-K reduction, one threadgroup per (q head, query row).
/// Standard flash-decoding merge in the log2 domain; `l_j == 0` marks an empty split and is
/// skipped. NSPLIT, qL and DV ride `pacc_shape`, so the config depends on nothing per token.
const QSA_ATTN_MERGE_SOURCE =
    \\const int DV = pacc_shape[3];
    \\const int d = int(thread_index_in_threadgroup);
    \\const int s = int(threadgroup_position_in_grid.y);
    \\const int hq = int(threadgroup_position_in_grid.z);
    \\const int qL = pacc_shape[1];
    \\const int NS = pacc_shape[2];
    \\const long pbase = ((long)hq * (long)qL + (long)s) * NS;
    \\
    \\// Every thread reads all NS (m, l) pairs. The pairs land in cache after
    \\// the first thread of the threadgroup touches them, which is
    \\// cheaper than a barrier plus threadgroup storage for two floats.
    \\float M = -INFINITY;
    \\for (int j = 0; j < NS; ++j) {
    \\  if (pml[(pbase + j) * 2 + 1] > 0.0f) M = metal::max(M, pml[(pbase + j) * 2]);
    \\}
    \\float Z = 0.0f;
    \\float acc = 0.0f;
    \\for (int j = 0; j < NS; ++j) {
    \\  const float lj = pml[(pbase + j) * 2 + 1];
    \\  if (lj <= 0.0f) continue;
    \\  const float w = metal::exp2(pml[(pbase + j) * 2] - M);
    \\  Z += w * lj;
    \\  acc += w * pacc[(pbase + j) * DV + d];
    \\}
    \\out[((long)hq * (long)qL + (long)s) * DV + d] = T(acc / Z);
;

var qsa_attn_merge_kernel_cached: ?mlx.mlx_fast_metal_kernel = null;

fn getQsaAttnMergeKernel() !mlx.mlx_fast_metal_kernel {
    if (qsa_attn_merge_kernel_cached) |kk| return kk;
    const input_names = [_][*:0]const u8{ "pacc", "pml" };
    const output_names = [_][*:0]const u8{"out"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "sushi_attn_qsa256_qmerge",
        in_vec,
        out_vec,
        QSA_ATTN_MERGE_SOURCE,
        ATTN_QSA256_KERNEL_HEADER,
        false,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    qsa_attn_merge_kernel_cached = kernel;
    return kernel;
}

/// Threadgroup tile depth: 16 keys stages 12 KiB, half the dense prefill kernel's footprint.
pub const QSA_ATTN_BK: c_int = 16;

var qsa_attn_q_kernel_cached: ?mlx.mlx_fast_metal_kernel = null;

fn getQsaAttnQKernel() !mlx.mlx_fast_metal_kernel {
    if (qsa_attn_q_kernel_cached) |kk| return kk;
    const input_names = [_][*:0]const u8{ "q", "scl", "blocks", "kq", "ksc", "kbi", "vq", "vsc", "vbi" };
    const output_names = [_][*:0]const u8{ "pacc", "pml" };
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "sushi_attn_qsa256_qsplit",
        in_vec,
        out_vec,
        QSA_ATTN_Q_SOURCE,
        KV_UNPACK_HEADER ++ ATTN_QSA256_KERNEL_HEADER,
        false, // packed triples are cache VIEWS — a contiguous copy erases the win
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    qsa_attn_q_kernel_cached = kernel;
    return kernel;
}

/// Keyed on everything the config depends on and nothing else: `kv`/`kb` are runtime shapes,
/// `NSPLIT` changes the grid and both output shapes.
/// A small LRU of metal_kernel configs: adaptive MTP alternates verify widths within one
/// request, and a single slot rebuilt on every alternation. Handles are borrowed.
pub fn QsaCfgCache(comptime Key: type, comptime NCFG: usize) type {
    return struct {
        const Self = @This();
        pub const SLOTS: usize = 4;
        /// Named so the default can be written as an array literal.
        const MaybeCfg = ?mlx.mlx_fast_metal_kernel_config;
        const Entry = struct {
            key: Key = std.mem.zeroes(Key),
            cfg: [NCFG]MaybeCfg = @splat(null),
            stamp: u64 = 0,
        };
        entries: [SLOTS]Entry = @splat(.{}),
        clock: u64 = 0,
        /// Build counter.
        builds: usize = 0,

        pub fn get(self: *Self, key: Key) ?[NCFG]mlx.mlx_fast_metal_kernel_config {
            for (&self.entries) |*e| {
                if (e.cfg[0] == null) continue;
                if (!std.meta.eql(e.key, key)) continue;
                self.clock += 1;
                e.stamp = self.clock;
                var out: [NCFG]mlx.mlx_fast_metal_kernel_config = undefined;
                for (e.cfg, 0..) |c, i| out[i] = c.?;
                return out;
            }
            return null;
        }

        pub fn put(self: *Self, key: Key, cfg: [NCFG]mlx.mlx_fast_metal_kernel_config) void {
            var victim: usize = 0;
            var oldest: u64 = std.math.maxInt(u64);
            for (&self.entries, 0..) |*e, i| {
                if (e.cfg[0] == null) {
                    victim = i;
                    oldest = 0;
                    break;
                }
                if (e.stamp < oldest) {
                    oldest = e.stamp;
                    victim = i;
                }
            }
            const e = &self.entries[victim];
            for (&e.cfg) |*c| {
                if (c.*) |cc| _ = mlx.mlx_fast_metal_kernel_config_free(cc);
                c.* = null;
            }
            e.key = key;
            for (cfg, 0..) |c, i| e.cfg[i] = c;
            self.clock += 1;
            e.stamp = self.clock;
            self.builds += 1;
        }
    };
}

pub const QsaAttnCfgKey = struct {
    q_shape: ShapeKey,
    /// Shapes the grid and is not recoverable from the q shape.
    h_kv: c_int,
    dv: c_int,
    bits: u8,
    gs: u32,
    quant: bool,
    /// False = no block list: every row attends its whole causal range.
    sparse: bool,
    nsg: c_int,
    /// Simdgroups per threadgroup, `nsg` of them computing.
    nsgt: c_int,
    nsplit: c_int,
    bk: c_int,
    ratio: c_int,
    dtype: mlx.mlx_dtype,
};
pub var qsa_attn_cfgs = QsaCfgCache(QsaAttnCfgKey, 2){};

/// One split-K pass plus its merge. `arrays` = the packed K and V triples, or dense K and V
/// three times each; `blocks` is read only when `key.sparse`. Null = a kernel failed to build.
pub fn splitKAttnDispatch(s: mlx.mlx_stream, q: mlx.mlx_array, blocks: mlx.mlx_array, arrays: [6]mlx.mlx_array, attn_scale: f32, key: QsaAttnCfgKey) !?mlx.mlx_array {
    const kernel = getQsaAttnQKernel() catch return null;
    const merge_kernel = getQsaAttnMergeKernel() catch return null;
    const qs = mlx.getShape(q);
    const seq_len = qs[2];
    const scl_data = [_]f32{attn_scale};
    const one = [_]c_int{1};
    const scl = mlx.mlx_array_new_data(&scl_data, &one, 1, .float32);
    defer _ = mlx.mlx_array_free(scl);

    const cfgs: [2]mlx.mlx_fast_metal_kernel_config = qsa_attn_cfgs.get(key) orelse blk: {
        // Split pass: one threadgroup per (row, kv head, split).
        const config = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(config);
        const acc_shape = [_]c_int{ qs[1], seq_len, key.nsplit, key.dv };
        const ml_shape = [_]c_int{ qs[1], seq_len, key.nsplit, 2 };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &acc_shape, 4, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &ml_shape, 4, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, seq_len * 32, key.h_kv * key.nsgt, key.nsplit));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, key.nsgt, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", key.dtype));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "NSG", key.nsg));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "NSGT", key.nsgt));        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BK", key.bk));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "RATIO", key.ratio));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BITS", @intCast(key.bits)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GS", @intCast(key.gs)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "QUANT", @intFromBool(key.quant)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "DK", qs[3]));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "DV", key.dv));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "SPARSE", @intFromBool(key.sparse)));

        // Merge pass: one threadgroup per (q head, row), one thread per value element.
        const mconfig = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(mconfig);
        const o_shape = [_]c_int{ qs[0], qs[1], seq_len, key.dv };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(mconfig, &o_shape, 4, key.dtype));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(mconfig, key.dv, seq_len, qs[1]));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(mconfig, key.dv, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(mconfig, "T", key.dtype));

        qsa_attn_cfgs.put(key, .{ config, mconfig });
        break :blk .{ config, mconfig };
    };

    const inputs_arr = [_]mlx.mlx_array{ q, scl, blocks } ++ arrays;
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var parts_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(parts_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&parts_vec, kernel, inputs_vec, cfgs[0], s));
    if (mlx.mlx_vector_array_size(parts_vec) != 2) return error.MetalKernelBadOutputCount;
    var pacc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pacc);
    var pml = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pml);
    try mlx.check(mlx.mlx_vector_array_get(&pacc, parts_vec, 0));
    try mlx.check(mlx.mlx_vector_array_get(&pml, parts_vec, 1));

    const merge_in = [_]mlx.mlx_array{ pacc, pml };
    const merge_vec = mlx.mlx_vector_array_new_data(&merge_in, merge_in.len);
    defer _ = mlx.mlx_vector_array_free(merge_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, merge_kernel, merge_vec, cfgs[1], s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 0));
    return out;
}

/// A quantized KV cache as the kernel reads it: the packed triples plus the
/// shape of the dense view they stand for. `KVCache.denseView` already borrows
/// these; taking them here is what keeps its lazy dense arrays unevaluated.
pub const PackedKv = struct {
    k_q: mlx.mlx_array,
    k_scales: mlx.mlx_array,
    k_biases: mlx.mlx_array,
    v_q: mlx.mlx_array,
    v_scales: mlx.mlx_array,
    v_biases: mlx.mlx_array,
    bits: u8,
    group_size: u32,
};

/// One dispatch's keys or values: slice the packed triple along the token axis
/// and dequantize the slice. The slice is a VIEW and mlx's quantized ops want
/// their own rows, so it is materialized first (a few MB of packed bytes).
fn dequantKvSlice(
    s: mlx.mlx_stream,
    q: mlx.mlx_array,
    scales: mlx.mlx_array,
    biases: mlx.mlx_array,
    k0: c_int,
    k1: c_int,
    group_size: u32,
    bits: u8,
) !mlx.mlx_array {
    var parts: [3]mlx.mlx_array = .{ .{ .ctx = null }, .{ .ctx = null }, .{ .ctx = null } };
    defer for (&parts) |*a| {
        if (a.ctx != null) _ = mlx.mlx_array_free(a.*);
    };
    for ([_]mlx.mlx_array{ q, scales, biases }, 0..) |src, i| {
        const sh = mlx.getShape(src);
        if (sh.len != 4) return error.InvalidKvSliceShape;
        var view = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(view);
        try mlx.check(mlx.mlx_slice(
            &view,
            src,
            &[_]c_int{ 0, 0, k0, 0 },
            4,
            &[_]c_int{ sh[0], sh[1], k1, sh[3] },
            4,
            &[_]c_int{ 1, 1, 1, 1 },
            4,
            s,
        ));
        parts[i] = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_contiguous(&parts[i], view, false, s));
    }
    return kv_quant.dequantizeAffine(s, parts[0], parts[1], parts[2], group_size, bits);
}

pub fn fusedSdpaPrefillImpl(
    s: mlx.mlx_stream,
    q: mlx.mlx_array,
    k: mlx.mlx_array,
    v: mlx.mlx_array,
    scale: f32,
    window: c_int,
    mask: ?mlx.mlx_array,
    packed_kv: ?PackedKv,
    sinks: mlx.mlx_array,
) !?mlx.mlx_array {
    if (mlx.mlx_array_ndim(q) != 4 or mlx.mlx_array_ndim(k) != 4 or mlx.mlx_array_ndim(v) != 4) return null;
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    const vs = mlx.getShape(v);
    if (ks[3] != qs[3] or !fusedPdDims(qs[3], vs[3])) return null;
    const has_sinks = sinks.ctx != null;
    if (has_sinks and (mlx.mlx_array_ndim(sinks) != 1 or mlx.getShape(sinks)[0] != qs[1])) return null;
    // Short sequences (< 16: decode AND spec-decode VERIFY forwards) belong
    // to MLX's sdpa_vector, which covers hd 256 natively and beats a 64-row
    // prefill tile walking the whole KV for a few-row query — dispatching
    // those here measured decode 48 -> 18 tok/s at 4K ctx on the 27B (MTP
    // verify is seq 1+depth, up to 9 at the NAX depth-8 cap — with causal
    // fused default-on the floor must clear it). 16 matches oMLX's
    // _MIN_ROUTE_Q_LEN; genuine prefill chunks are always far above it.
    if (qs[2] < FUSED256_MIN_Q_LEN) return null;
    if (ks[1] <= 0 or @rem(qs[1], ks[1]) != 0) return null;
    if (ks[2] < qs[2] or ks[2] != vs[2] or ks[1] != vs[1] or ks[0] != qs[0] or vs[0] != qs[0]) return null;
    if (mlx.mlx_array_dtype(q) != .bfloat16 or mlx.mlx_array_dtype(k) != .bfloat16 or mlx.mlx_array_dtype(v) != .bfloat16) return null;
    if (packed_kv == null and (!attn256Base16Aligned(k) or !attn256Base16Aligned(v))) return null;

    const arm: AttnPdArm = if (attnPdNaxServes(qs[3], vs[3], mask != null)) .nax else .simd;
    var dispatches: u32 = 0;
    const out = (try attnPdDispatch(s, arm, q, k, v, scale, window, mask, packed_kv, sinks, &dispatches)) orelse return null;
    fused256_last_dispatch_count = dispatches;
    attn_pd_last_arm = arm;
    logAttnPdEngaged(arm, qs[3], vs[3], qs[2], ks[2], dispatches);
    return out;
}

/// The dispatch chain of one validated `fusedSdpaPrefillImpl` call on the given arm. Null = the
/// kernel could not be built.
fn attnPdDispatch(
    s: mlx.mlx_stream,
    arm: AttnPdArm,
    q: mlx.mlx_array,
    k: mlx.mlx_array,
    v: mlx.mlx_array,
    scale: f32,
    window: c_int,
    mask: ?mlx.mlx_array,
    packed_kv: ?PackedKv,
    sinks: mlx.mlx_array,
    dispatches: *u32,
) !?mlx.mlx_array {
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    const vs = mlx.getShape(v);
    const has_sinks = sinks.ctx != null;
    const kernel = getAttnPdKernel(arm) catch return null;

    const one = [_]c_int{1};
    const scl_data = [_]f32{scale};
    const scl = mlx.mlx_array_new_data(&scl_data, &one, 1, .float32);
    defer _ = mlx.mlx_array_free(scl);
    const win_data = [_]i32{@intCast(window)};
    const win = mlx.mlx_array_new_data(&win_data, &one, 1, .int32);
    defer _ = mlx.mlx_array_free(win);

    // Budgeted kv-axis chunking (CAUSAL arm only — the band arm's in-kernel
    // block skip already bounds per-dispatch work by the window, so chunking
    // it would only add carry traffic). Chunk boundaries are BK(32)-aligned;
    // the fp32 m/l/O carry chained between dispatches keeps the result
    // bit-identical to a single dispatch (see the budget comment above).
    const kL: c_int = ks[2];
    const chunk_len: c_int = if (window > 0)
        kL
    else if (packed_kv != null)
        @min(PACKED_KV_SLICE_MAX, fused256KvChunkLen(@intCast(qs[0]), @intCast(qs[1]), @intCast(qs[2]), @intCast(kL), fused256DispatchBudget(arm)))
    else
        fused256KvChunkLen(@intCast(qs[0]), @intCast(qs[1]), @intCast(qs[2]), @intCast(kL), fused256DispatchBudget(arm));

    // Dummy 1-elem f32 stands in for absent carry inputs / unwritten outputs.
    const dummy_data = [_]f32{0};
    const dummy = mlx.mlx_array_new_data(&dummy_data, &one, 1, .float32);
    defer _ = mlx.mlx_array_free(dummy);
    // 4-D so `mask_strides[..3]` exists for the QSA=0 template too.
    const dummy_b_data = [_]bool{false};
    const dummy_b_shape = [_]c_int{ 1, 1, 1, 1 };
    const dummy_b = mlx.mlx_array_new_data(&dummy_b_data, &dummy_b_shape, 4, .bool_);
    defer _ = mlx.mlx_array_free(dummy_b);
    var skip = mlx.mlx_array{ .ctx = null };
    defer if (skip.ctx != null) {
        _ = mlx.mlx_array_free(skip);
    };
    if (mask) |m| skip = try qsaSkipTable(s, m);
    var sinks_f32 = mlx.mlx_array{ .ctx = null };
    defer if (sinks_f32.ctx != null) {
        _ = mlx.mlx_array_free(sinks_f32);
    };
    if (has_sinks) {
        sinks_f32 = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&sinks_f32, sinks, .float32, s));
    }

    var m_prev = mlx.mlx_array{ .ctx = null };
    var l_prev = mlx.mlx_array{ .ctx = null };
    var o_prev = mlx.mlx_array{ .ctx = null };
    defer {
        if (m_prev.ctx != null) _ = mlx.mlx_array_free(m_prev);
        if (l_prev.ctx != null) _ = mlx.mlx_array_free(l_prev);
        if (o_prev.ctx != null) _ = mlx.mlx_array_free(o_prev);
    }

    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    dispatches.* = 0;
    var k0: c_int = 0;
    while (k0 < kL) : (k0 += chunk_len) {
        const k1: c_int = @min(k0 + chunk_len, kL);
        const final = k1 == kL;
        const has_carry = k0 > 0;

        // {begin, end} in the BUFFER, then where the buffer starts in the
        // cache and how long the cache is. Dense: the buffer is the cache.
        const four = [_]c_int{4};
        const kr_data = [_]i32{
            if (packed_kv != null) 0 else @as(i32, @intCast(k0)),
            if (packed_kv != null) @as(i32, @intCast(k1 - k0)) else @as(i32, @intCast(k1)),
            if (packed_kv != null) @as(i32, @intCast(k0)) else 0,
            @intCast(kL),
        };
        const kr = mlx.mlx_array_new_data(&kr_data, &four, 1, .int32);
        defer _ = mlx.mlx_array_free(kr);

        // One dispatch's keys, dequantized here and freed as the graph runs
        // past them — the whole-prompt dense rebuild `denseView` would make
        // never exists (1.25 GiB per global layer at a 512k prompt).
        var k_slice: mlx.mlx_array = .{ .ctx = null };
        var v_slice: mlx.mlx_array = .{ .ctx = null };
        defer {
            if (k_slice.ctx != null) _ = mlx.mlx_array_free(k_slice);
            if (v_slice.ctx != null) _ = mlx.mlx_array_free(v_slice);
        }
        if (packed_kv) |pk| {
            k_slice = try dequantKvSlice(s, pk.k_q, pk.k_scales, pk.k_biases, k0, k1, pk.group_size, pk.bits);
            v_slice = try dequantKvSlice(s, pk.v_q, pk.v_scales, pk.v_biases, k0, k1, pk.group_size, pk.bits);
        }
        const phase_data = [_]i32{(if (has_carry) @as(i32, 1) else 0) | (if (final) @as(i32, 2) else 0)};
        const phase = mlx.mlx_array_new_data(&phase_data, &one, 1, .int32);
        defer _ = mlx.mlx_array_free(phase);

        const config = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
        if (final) {
            const o_shape = [_]c_int{ qs[0], qs[1], qs[2], vs[3] };
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &o_shape, 4, .bfloat16));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &one, 1, .float32));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &one, 1, .float32));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &one, 1, .float32));
        } else {
            const ml_shape = [_]c_int{ qs[0], qs[1], qs[2] };
            const oc_shape = [_]c_int{ qs[0], qs[1], qs[2], vs[3] };
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &one, 1, .bfloat16));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &ml_shape, 3, .float32));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &ml_shape, 3, .float32));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &oc_shape, 4, .float32));
        }
        // One threadgroup per 64-row q tile per head per batch; grid is in
        // THREADS (dispatch_threads), padded to whole tiles so every
        // threadgroup is full (the SIMD kernel's staging loops assume NT=256).
        // SIMD: (32,8,1), 8 rows per simdgroup; NAX: (128,1,1), 16 rows per simdgroup.
        const nq_tiles: c_int = @divTrunc(qs[2] + 63, 64);
        switch (arm) {
            .simd => {
                try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, nq_tiles * 32, qs[1] * 8, qs[0]));
                try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, 8, 1));
            },
            .nax => {
                try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, nq_tiles * 128, qs[1], qs[0]));
                try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 128, 1, 1));
            },
        }
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BDK", qs[3]));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BDV", vs[3]));
        if (arm == .simd) try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "QSA", if (mask != null) 1 else 0));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "SINK", if (has_sinks) 1 else 0));

        const inputs_arr = [_]mlx.mlx_array{
            q,                                if (packed_kv != null) k_slice else k,
            if (packed_kv != null) v_slice else v, scl,
            win,                              kr,
            phase,                            if (has_carry) m_prev else dummy,
            if (has_carry) l_prev else dummy, if (has_carry) o_prev else dummy,
            if (mask) |m| m else dummy_b,     if (mask != null) skip else dummy_b,
            if (has_sinks) sinks_f32 else dummy,
        };
        const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
        defer _ = mlx.mlx_vector_array_free(inputs_vec);

        var outputs_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(outputs_vec);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, config, s));
        if (mlx.mlx_vector_array_size(outputs_vec) != 4) return error.MetalKernelBadOutputCount;
        dispatches.* += 1;

        if (final) {
            try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 0));
        } else {
            var m_new = mlx.mlx_array_new();
            var l_new = mlx.mlx_array_new();
            var o_new = mlx.mlx_array_new();
            errdefer {
                if (m_new.ctx != null) _ = mlx.mlx_array_free(m_new);
                if (l_new.ctx != null) _ = mlx.mlx_array_free(l_new);
                if (o_new.ctx != null) _ = mlx.mlx_array_free(o_new);
            }
            try mlx.check(mlx.mlx_vector_array_get(&m_new, outputs_vec, 1));
            try mlx.check(mlx.mlx_vector_array_get(&l_new, outputs_vec, 2));
            try mlx.check(mlx.mlx_vector_array_get(&o_new, outputs_vec, 3));
            if (m_prev.ctx != null) _ = mlx.mlx_array_free(m_prev);
            if (l_prev.ctx != null) _ = mlx.mlx_array_free(l_prev);
            if (o_prev.ctx != null) _ = mlx.mlx_array_free(o_prev);
            m_prev = m_new;
            l_prev = l_new;
            o_prev = o_new;
            m_new = .{ .ctx = null };
            l_new = .{ .ctx = null };
            o_new = .{ .ctx = null };
        }
    }
    return out;
}

pub fn sliceAttentionSeq(s: mlx.mlx_stream, a: mlx.mlx_array, start: c_int, end: c_int) !mlx.mlx_array {
    const sh = mlx.getShape(a);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_slice(&out, a, &.{ 0, 0, start, 0 }, 4, &.{ sh[0], sh[1], end, sh[3] }, 4, &.{ 1, 1, 1, 1 }, 4, s));
    return out;
}

/// Parity seeds: a kernel bar is a sweep, never one chosen seed.
const ATTN_PD_PARITY_SEEDS: usize = 4;

/// One causal case at (qk, v) = (192, 128): the kernel must land no further
/// from fp32 truth than the composed bf16 path it replaces.
fn attnPd192CausalCase(seed: u64, hq: c_int, hkv: c_int, q_len: c_int, kv_len: c_int) !void {
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    const q_shape = [_]c_int{ 1, hq, q_len, 192 };
    const k_shape = [_]c_int{ 1, hkv, kv_len, 192 };
    const v_shape = [_]c_int{ 1, hkv, kv_len, 128 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &k_shape, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &v_shape, s);
    defer _ = mlx.mlx_array_free(v);
    const scale: f32 = 1.0 / @sqrt(192.0);

    const fused = (try fusedSdpaPrefill(s, q, k, v, scale, 0)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(fused);
    const composed = try attn256Reference(q, k, v, scale, "causal", .{ .ctx = null }, s);
    defer _ = mlx.mlx_array_free(composed);
    const truth = try attnPdTruthF32(q, k, v, scale, s);
    defer _ = mlx.mlx_array_free(truth);

    // Finiteness before the diff: an all-NaN pair diffs to NaN, which compares
    // false against every bar and would read as a pass under `!(d > bar)`.
    const scale_of = try attn256MaxAbs(truth, s);
    try std.testing.expect(std.math.isFinite(scale_of) and scale_of > 0);
    const fused_err = try attn256MaxDiff(fused, truth, s);
    const composed_err = try attn256MaxDiff(composed, truth, s);
    try std.testing.expect(std.math.isFinite(fused_err));
    // No worse than the path it replaces, plus one bf16 ulp of slack at the
    // reference's own scale (the two round in different places).
    try std.testing.expect(fused_err <= composed_err + scale_of * 0.00390625);
}

/// Walk a ringed sliding cache through `schedule` and hold every chunk's fused
/// band + sink attention to fp64 truth, no worse than the composed arm it replaces.
fn swaSinkParityWalk(kv_cfg: KVQuantConfig, seed: u64, schedule: []const c_int) !void {
    const alloc = std.testing.allocator;
    const s = mlx.gpuStream();
    const cfg = swaSinkTestConfig();
    const g = SwaGeom{ .hq = 16, .hk = 2, .qk = 192, .vd = 128, .t = blk: {
        var t: usize = 0;
        for (schedule) |w| t += @intCast(w);
        break :blk t;
    } };
    const t: c_int = @intCast(g.t);
    const scale: f32 = 1.0 / @sqrt(192.0);
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();

    // Peaked scores, so both the band edge and the sink move the output.
    const q_all = try attn256RandBf16Scaled(rnd, &[_]c_int{ 1, 16, t, 192 }, 8.0, s);
    defer _ = mlx.mlx_array_free(q_all);
    const k_all = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, t, 192 }, s);
    defer _ = mlx.mlx_array_free(k_all);
    const v_all = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, t, 128 }, s);
    defer _ = mlx.mlx_array_free(v_all);
    // bf16 like the checkpoint's `attention_sink_bias`, spread so some heads' sink dominates.
    var sink_f: [16]f32 = undefined;
    for (&sink_f) |*x| x.* = rnd.float(f32) * 6.0 - 2.0;
    const sink_shape = [_]c_int{16};
    const sinks_f32 = mlx.mlx_array_new_data(&sink_f, &sink_shape, 1, .float32);
    defer _ = mlx.mlx_array_free(sinks_f32);
    var sinks = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sinks);
    try mlx.check(mlx.mlx_astype(&sinks, sinks_f32, .bfloat16, s));

    // Truth reads the values the kernel reads: a quantized cache's are dequantized.
    var k_read = k_all;
    var v_read = v_all;
    var kq: ?kv_quant.QuantizedKV = null;
    var vq: ?kv_quant.QuantizedKV = null;
    defer if (kq) |*x| x.deinit();
    defer if (vq) |*x| x.deinit();
    defer if (kv_cfg.isQuant()) {
        _ = mlx.mlx_array_free(k_read);
        _ = mlx.mlx_array_free(v_read);
    };
    if (kv_cfg.isQuant()) {
        kq = try kv_quant.quantizeAffine(s, k_all, kv_cfg.group_size, kv_cfg.bits);
        vq = try kv_quant.quantizeAffine(s, v_all, kv_cfg.group_size, kv_cfg.bits);
        k_read = try kv_quant.dequantizeAffine(s, kq.?.q, kq.?.scales, kq.?.biases, kv_cfg.group_size, kv_cfg.bits);
        v_read = try kv_quant.dequantizeAffine(s, vq.?.q, vq.?.scales, vq.?.biases, kv_cfg.group_size, kv_cfg.bits);
    }
    const q_h = try swaHostF32(alloc, q_all, s);
    defer alloc.free(q_h);
    const k_h = try swaHostF32(alloc, k_read, s);
    defer alloc.free(k_h);
    const v_h = try swaHostF32(alloc, v_read, s);
    defer alloc.free(v_h);
    const sink_h = try swaHostF32(alloc, sinks, s);
    defer alloc.free(sink_h);

    var cache = try KVCache.initWithConfig(alloc, 1, kv_cfg);
    defer cache.deinit();
    cache.setSwaRing(SWA_TEST_WINDOW);

    var pos: c_int = 0;
    for (schedule) |ql| {
        const q_c = try sliceSeq(s, q_all, pos, pos + ql);
        defer _ = mlx.mlx_array_free(q_c);
        const k_c = try sliceSeq(s, k_all, pos, pos + ql);
        defer _ = mlx.mlx_array_free(k_c);
        const v_c = try sliceSeq(s, v_all, pos, pos + ql);
        defer _ = mlx.mlx_array_free(v_c);
        var view = try cache.update(0, k_c, v_c, s, slidingViewFor(&cfg, pos + ql, ql).span);
        defer view.deinit();

        const mask = try swaBandMask(ql, mlx.getShape(view.k)[2]);
        defer _ = mlx.mlx_array_free(mask);
        var composed = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(composed);
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&composed, q_c, view.k, view.v, scale, "array", mask, sinks, false, s));

        const q_ch = try swaHostF32(alloc, q_c, s);
        defer alloc.free(q_ch);
        const truth = try swaSinkTruth(alloc, g, q_ch, k_h, v_h, sink_h, @intCast(pos), @intCast(ql), scale);
        defer alloc.free(truth);
        const composed_h = try swaHostF32(alloc, composed, s);
        defer alloc.free(composed_h);

        var scale_of: f64 = 0;
        for (truth) |x| scale_of = @max(scale_of, @abs(x));
        try std.testing.expect(std.math.isFinite(scale_of) and scale_of > 0);
        const composed_err = swaMaxErr(composed_h, truth);
        try std.testing.expect(std.math.isFinite(composed_err));
        var simd_h: ?[]f32 = null;
        defer if (simd_h) |h| alloc.free(h);
        for (attnPdTestArms()) |arm| {
            attn_pd_nax_override = arm == .nax;
            defer attn_pd_nax_override = null;
            const fused = (try slidingPrefillAttn(s, &cfg, q_c, &view, scale, sinks)) orelse return error.FusedDeclined;
            defer _ = mlx.mlx_array_free(fused);
            try std.testing.expectEqual(arm, attn_pd_last_arm);
            const fused_h = try swaHostF32(alloc, fused, s);
            var kept = false;
            defer if (!kept) alloc.free(fused_h);
            const fused_err = swaMaxErr(fused_h, truth);
            try std.testing.expect(std.math.isFinite(fused_err));
            // One bf16 ulp of slack at the truth's own scale: the arms round in different places.
            try std.testing.expect(fused_err <= composed_err + scale_of * 0.00390625);
            if (simd_h) |simd| try expectAttnPdNoWorse(fused_h, simd, truth, absMax(v_h)) else {
                simd_h = fused_h;
                kept = true;
            }
        }
        pos += ql;
    }
    // The walk really wrapped: the ring holds a window, not the history.
    try std.testing.expect(cache.seqLen(0) < g.t);
}

// A first chunk inside the window, two chunks that push the ring past its cap
// (the next reads window rows an earlier chunk wrote), the arm's shortest q, a
// 2048 chunk, and a ragged tail — every key length a partial 32-key block.
const SWA_SINK_SCHEDULE = [_]c_int{ 100, 512, 512, 16, 2048, 37 };

/// One global-layer case (qk 192 / v 128) on every arm: `budget` > 0 forces fp32 carries across
/// dispatches, `kv8` reads a packed cache one dequantized slice per dispatch. Each arm lands no
/// further from fp64 truth than the composed bf16 path, and NAX no worse than the SIMD kernel.
fn attnPdGlobalCase(seed: u64, hq: c_int, hkv: c_int, q_len: c_int, kv_len: c_int, q_scale: f32, budget: i64, kv8: bool) !void {
    const alloc = std.testing.allocator;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    const q = try attn256RandBf16Scaled(rnd, &[_]c_int{ 1, hq, q_len, 192 }, q_scale, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &[_]c_int{ 1, hkv, kv_len, 192 }, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &[_]c_int{ 1, hkv, kv_len, 128 }, s);
    defer _ = mlx.mlx_array_free(v);
    const scale: f32 = 1.0 / @sqrt(192.0);
    var qv: ?TestQuantView = if (kv8) try TestQuantView.init(s, k, v, 8, 64) else null;
    defer if (qv) |*x| x.deinit();
    const dense = DenseKVView{ .k = k, .v = v, .owned = false };
    const view: *const DenseKVView = if (qv) |*x| &x.view else &dense;

    const g = SwaGeom{ .hq = @intCast(hq), .hk = @intCast(hkv), .qk = 192, .vd = 128, .t = @intCast(kv_len) };
    const q_h = try swaHostF32(alloc, q, s);
    defer alloc.free(q_h);
    const k_h = try swaHostF32(alloc, view.k, s);
    defer alloc.free(k_h);
    const v_h = try swaHostF32(alloc, view.v, s);
    defer alloc.free(v_h);
    const truth = try attnPdCausalTruth(alloc, g, q_h, k_h, v_h, @intCast(q_len), scale);
    defer alloc.free(truth);
    var scale_of: f64 = 0;
    for (truth) |x| scale_of = @max(scale_of, @abs(x));
    try std.testing.expect(std.math.isFinite(scale_of) and scale_of > 0);

    const composed = try attn256Reference(q, view.k, view.v, scale, "causal", .{ .ctx = null }, s);
    defer _ = mlx.mlx_array_free(composed);
    const composed_h = try swaHostF32(alloc, composed, s);
    defer alloc.free(composed_h);
    const composed_err = swaMaxErr(composed_h, truth);
    try std.testing.expect(std.math.isFinite(composed_err));

    fused256_budget_override = budget;
    defer fused256_budget_override = null;
    var simd_h: ?[]f32 = null;
    defer if (simd_h) |h| alloc.free(h);
    for (attnPdTestArms()) |arm| {
        attn_pd_nax_override = arm == .nax;
        defer attn_pd_nax_override = null;
        const out = (try fusedSdpaPrefillKv(s, q, view, scale, 0, .{ .ctx = null })) orelse return error.FusedDeclined;
        defer _ = mlx.mlx_array_free(out);
        try std.testing.expectEqual(arm, attn_pd_last_arm);
        if (budget > 0) try std.testing.expect(fused256_last_dispatch_count > 1);
        const out_h = try swaHostF32(alloc, out, s);
        var kept = false;
        defer if (!kept) alloc.free(out_h);
        const err = swaMaxErr(out_h, truth);
        try std.testing.expect(std.math.isFinite(err));
        try std.testing.expect(err <= composed_err + scale_of * 0.00390625);
        if (simd_h) |simd| try expectAttnPdNoWorse(out_h, simd, truth, absMax(v_h)) else {
            simd_h = out_h;
            kept = true;
        }
    }
}

test "prefillEvalCadence: small transients keep the coarse cadence" {
    const t = std.testing;
    fused256_override = false;
    defer fused256_override = null;
    // 26B geometry, short prompt: 16 heads x 2048 chunk x 2048 kv x 2B = 128 MB
    // of unfused scores — well inside the budget, keep today's cadence.
    try t.expectEqual(@as(u32, 4), Transformer.prefillEvalCadence(4, 256, 16, 8, 2048, 2048, false));
    // The LIVE-measured no-regression point: a 5,140-token agent prompt
    // (845 MB scores) must keep cadence 4 — flipping it cost a measured 4.5%
    // prefill for zero memory benefit (peak 18.1 GB both ways).
    try t.expectEqual(@as(u32, 4), Transformer.prefillEvalCadence(4, 256, 16, 8, 5140, 5140, false));
    // Fused head dim + dense KV: no transient at all, keep even the 48 default.
    try t.expectEqual(@as(u32, 48), Transformer.prefillEvalCadence(48, 128, 8, 8, 8192, 500_000, false));
}

test "prefillEvalCadence: fused hd-256 kernel drops the score term" {
    const t = std.testing;
    // With sushi_attn_p256 active (the default) the score transient never
    // exists — hd 256 keeps the coarse cadence at any context, exactly like
    // hd 128. The dequant term must still fire under --kv-quant.
    fused256_override = true;
    defer fused256_override = null;
    try t.expectEqual(@as(u32, 4), Transformer.prefillEvalCadence(4, 256, 16, 8, 8192, 102_448, false));
    try t.expectEqual(@as(u32, 1), Transformer.prefillEvalCadence(4, 256, 16, 8, 8192, 400_000, true));
}

test "prefillEvalCadence: big unfused score tensor forces eval-per-layer" {
    const t = std.testing;
    fused256_override = false;
    defer fused256_override = null;
    // 16 heads x 8192 chunk x 16384 kv x 2B = 4 GiB > 2 GiB budget: bounds a
    // ~13 GB windowed transient to ~4 GB. (8192 chunk x 8192 kv sits exactly
    // ON the 2 GiB budget and deliberately keeps the coarse cadence.)
    try t.expectEqual(@as(u32, 1), Transformer.prefillEvalCadence(4, 256, 16, 8, 8192, 16384, false));
    try t.expectEqual(@as(u32, 4), Transformer.prefillEvalCadence(4, 256, 16, 8, 8192, 8192, false));
    // The capped 1024 chunk at 102K ctx still materializes 3.4 GB — the chunk
    // cap (generate.zig) does NOT restore the coarse cadence at long context.
    // (Live: this cadence measured peak 27.0 GB and +14% prefill vs baseline.)
    try t.expectEqual(@as(u32, 1), Transformer.prefillEvalCadence(4, 256, 16, 8, 1024, 102_448, false));
}

test "fusedSdpaPrefill: causal parity vs composed SDPA (GQA, ragged shapes, chunk offset)" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    fused256_override = true;
    defer fused256_override = null;
    // The causal arm yields to the stock fused kernel on NAX silicon, so on an
    // M5-class Mac this parity check would DECLINE instead of running — the
    // kernel under test would never be exercised. Pin the arm, not the host.
    nax_gate.nax_sdpa_override = false;
    defer nax_gate.nax_sdpa_override = null;
    var prng = std.Random.DefaultPrng.init(0x256256);
    const rnd = prng.random();

    // qL=70 (partial 32-tile), kL=193 (partial 16-block), offset 123 (chunked
    // prefill bottom-right alignment), Hq=6/Hk=2 (gqa 3).
    const q_shape = [_]c_int{ 1, 6, 70, 256 };
    const kv_shape = [_]c_int{ 1, 2, 193, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(v);
    const scale: f32 = 1.0 / 16.0; // 1/sqrt(256)

    const fused = (try fusedSdpaPrefill(s, q, k, v, scale, 0)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(fused);
    const ref = try attn256Reference(q, k, v, scale, "causal", .{ .ctx = null }, s);
    defer _ = mlx.mlx_array_free(ref);

    // Not byte-parity: the composed path rounds softmax probs to bf16 before
    // the AV matmul, the fused kernel keeps float32 — bf16-rounding-scale
    // differences only.
    // Measured 0.00049 (one bf16 ULP at 0.5 magnitude) on this seed.
    const max_diff = try attn256MaxDiff(fused, ref, s);
    try std.testing.expect(max_diff < 0.005);
}

test "fusedSdpaPrefill: qk 192 / v 128 causal parity vs fp32 truth (MiMo global geometry)" {
    try mlx.requireGpu();
    fused256_override = true;
    defer fused256_override = null;
    for (0..ATTN_PD_PARITY_SEEDS) |i| {
        const seed: u64 = 0x192a + i;
        // Ragged 64-row q tile, ragged 32-key block, chunk offset, gqa 16 —
        // the mimo_v2 global layer's own head ratio.
        try attnPd192CausalCase(seed, 32, 2, 70, 193);
        try attnPd192CausalCase(seed + 1, 16, 1, 512, 512);
        try attnPd192CausalCase(seed + 2, 16, 1, 512, 4096);
    }
}

test "fusedSdpaPrefill: qk 192 / v 128 kv chunking carries the narrower value width" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    fused256_override = true;
    defer fused256_override = null;
    var prng = std.Random.DefaultPrng.init(0x19208);
    const rnd = prng.random();
    const q_shape = [_]c_int{ 1, 8, 64, 192 };
    const k_shape = [_]c_int{ 1, 2, 2048, 192 };
    const v_shape = [_]c_int{ 1, 2, 2048, 128 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &k_shape, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &v_shape, s);
    defer _ = mlx.mlx_array_free(v);
    const scale: f32 = 1.0 / @sqrt(192.0);

    fused256_budget_override = 0;
    const single = (try fusedSdpaPrefill(s, q, k, v, scale, 0)) orelse {
        fused256_budget_override = null;
        return error.FusedDeclined;
    };
    defer _ = mlx.mlx_array_free(single);
    try std.testing.expectEqual(@as(u32, 1), fused256_last_dispatch_count);

    // Small budget -> several kv chunks; the fp32 m/l/O carry makes the chain
    // bit-identical to the single dispatch.
    fused256_budget_override = 8 * 64 * 256;
    defer fused256_budget_override = null;
    const chunked = (try fusedSdpaPrefill(s, q, k, v, scale, 0)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(chunked);
    try std.testing.expect(fused256_last_dispatch_count > 1);
    try std.testing.expectEqual(@as(f32, 0), try attn256MaxDiff(single, chunked, s));
}

test "fusedSdpaPrefill: only the instantiated (qk, value) width pairs are served" {
    const s = mlx.gpuStream();
    fused256_override = true;
    defer fused256_override = null;
    nax_gate.nax_sdpa_override = false;
    defer nax_gate.nax_sdpa_override = null;
    var prng = std.Random.DefaultPrng.init(0x19288);
    const rnd = prng.random();
    try std.testing.expect(fusedPdDims(256, 256));
    try std.testing.expect(fusedPdDims(192, 128));
    try std.testing.expect(!fusedPdDims(192, 192));
    try std.testing.expect(!fusedPdDims(256, 128));
    try std.testing.expect(!fusedPdDims(128, 128));

    // A 192-wide value (no instantiation) declines rather than reading past
    // the tile it staged.
    const q = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, 32, 192 }, s);
    defer _ = mlx.mlx_array_free(q);
    const kv = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, 32, 192 }, s);
    defer _ = mlx.mlx_array_free(kv);
    try std.testing.expect((try fusedSdpaPrefill(s, q, kv, kv, 1.0, 0)) == null);
}

test "fusedSdpaPrefillKv: a quantized cache is read per dispatch, never rebuilt whole" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    fused256_override = true;
    defer fused256_override = null;
    nax_gate.nax_sdpa_override = false;
    defer nax_gate.nax_sdpa_override = null;
    var prng = std.Random.DefaultPrng.init(0x19282);
    const rnd = prng.random();

    // mimo_v2's global-layer geometry, ragged on both axes.
    const q = try attn256RandBf16(rnd, &[_]c_int{ 1, 16, 70, 192 }, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, 1993, 192 }, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, 1993, 128 }, s);
    defer _ = mlx.mlx_array_free(v);
    const scale: f32 = 1.0 / @sqrt(192.0);

    var qv = try TestQuantView.init(s, k, v, 8, 64);
    defer qv.deinit();

    // Force several slices so the absolute-offset arithmetic is exercised.
    fused256_budget_override = 16 * 70 * 256;
    defer fused256_budget_override = null;
    const sliced = (try fusedSdpaPrefillKv(s, q, &qv.view, scale, 0, .{ .ctx = null })) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(sliced);
    try std.testing.expect(fused256_last_dispatch_count > 1);

    // The dense arm over the SAME dequantized values, in one dispatch: the
    // fp32 carry makes the sliced chain exact, so this is byte equality.
    fused256_budget_override = 0;
    const whole = (try fusedSdpaPrefill(s, q, qv.view.k, qv.view.v, scale, 0)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(whole);
    try std.testing.expectEqual(@as(u32, 1), fused256_last_dispatch_count);
    try std.testing.expectEqual(@as(f32, 0), try attn256MaxDiff(sliced, whole, s));

    // And no worse than the composed path against fp32 truth.
    const truth = try attnPdTruthF32(q, qv.view.k, qv.view.v, scale, s);
    defer _ = mlx.mlx_array_free(truth);
    const ref = try attn256Reference(q, qv.view.k, qv.view.v, scale, "causal", .{ .ctx = null }, s);
    defer _ = mlx.mlx_array_free(ref);
    const scale_of = try attn256MaxAbs(truth, s);
    try std.testing.expect(std.math.isFinite(scale_of) and scale_of > 0);
    const sliced_err = try attn256MaxDiff(sliced, truth, s);
    try std.testing.expect(std.math.isFinite(sliced_err));
    try std.testing.expect(sliced_err <= (try attn256MaxDiff(ref, truth, s)) + scale_of * 0.00390625);
}

test "fusedSdpaPrefillKv: a kv6 cache is read per dispatch and matches the dense arm over its rebuild" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    fused256_override = true;
    defer fused256_override = null;
    nax_gate.nax_sdpa_override = false;
    defer nax_gate.nax_sdpa_override = null;
    var prng = std.Random.DefaultPrng.init(0x19286);
    const rnd = prng.random();
    const q = try attn256RandBf16(rnd, &[_]c_int{ 1, 16, 70, 192 }, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, 1993, 192 }, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, 1993, 128 }, s);
    defer _ = mlx.mlx_array_free(v);
    const scale: f32 = 1.0 / @sqrt(192.0);
    var qv = try TestQuantView.init(s, k, v, 6, 64);
    defer qv.deinit();

    fused256_budget_override = 16 * 70 * 256;
    defer fused256_budget_override = null;
    const sliced = (try fusedSdpaPrefillKv(s, q, &qv.view, scale, 0, .{ .ctx = null })) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(sliced);
    try std.testing.expect(fused256_last_dispatch_count > 1);
    fused256_budget_override = 0;
    const whole = (try fusedSdpaPrefill(s, q, qv.view.k, qv.view.v, scale, 0)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(whole);
    try std.testing.expectEqual(@as(f32, 0), try attn256MaxDiff(sliced, whole, s));
}

test "fusedSdpaPrefillKv: an unquantized cache view takes the dense arm unchanged" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    fused256_override = true;
    defer fused256_override = null;
    nax_gate.nax_sdpa_override = false;
    defer nax_gate.nax_sdpa_override = null;
    var prng = std.Random.DefaultPrng.init(0x19283);
    const rnd = prng.random();
    const q = try attn256RandBf16(rnd, &[_]c_int{ 1, 8, 64, 192 }, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, 512, 192 }, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, 512, 128 }, s);
    defer _ = mlx.mlx_array_free(v);
    const scale: f32 = 1.0 / @sqrt(192.0);
    const off = DenseKVView{ .k = k, .v = v, .owned = false };
    const via_view = (try fusedSdpaPrefillKv(s, q, &off, scale, 0, .{ .ctx = null })) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(via_view);
    const direct = (try fusedSdpaPrefill(s, q, k, v, scale, 0)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(direct);
    try std.testing.expectEqual(@as(f32, 0), try attn256MaxDiff(via_view, direct, s));
}

test "sliding band + sink prefill matches fp64 truth over a ringed dense cache" {
    try mlx.requireGpu();
    fused256_override = true;
    defer fused256_override = null;
    for (0..ATTN_PD_PARITY_SEEDS) |i| try swaSinkParityWalk(KVQuantConfig.dense, 0x5a1c + i, &SWA_SINK_SCHEDULE);
}

test "sliding band + sink prefill matches fp64 truth over a ringed kv8 cache" {
    try mlx.requireGpu();
    fused256_override = true;
    defer fused256_override = null;
    for (0..ATTN_PD_PARITY_SEEDS) |i| try swaSinkParityWalk(KVQuantConfig.affine(8), 0x5a2c + i, &SWA_SINK_SCHEDULE);
}

test "sushi_attn_pd every arm: qk 192 / v 128 causal vs fp64 truth (ragged, carries, peaked, kv8)" {
    try mlx.requireGpu();
    fused256_override = true;
    defer fused256_override = null;
    for (0..ATTN_PD_PARITY_SEEDS) |i| {
        const seed: u64 = 0x192f + 8 * i;
        // Ragged 16-row simdgroup, 64-row tile and 32-key block at a chunk offset.
        try attnPdGlobalCase(seed, 32, 2, 70, 193, 1.0, 0, false);
        // The shortest q, a partial last key block.
        try attnPdGlobalCase(seed + 1, 8, 1, 16, 48, 1.0, 0, false);
        // Peaked scores across fp32 carries: 96-key dispatches.
        try attnPdGlobalCase(seed + 2, 16, 1, 130, 1111, 8.0, 16 * 130 * 96, false);
        try attnPdGlobalCase(seed + 3, 4, 1, 300, 2300, 20.0, 4 * 300 * 256, false);
        // A packed kv8 cache, dequantized one slice per dispatch.
        try attnPdGlobalCase(seed + 4, 16, 1, 70, 1993, 8.0, 16 * 70 * 256, true);
    }
}

test "sushi_attn_pd every arm: a chunked dispatch chain is bit-identical to one dispatch" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    fused256_override = true;
    defer fused256_override = null;
    var prng = std.Random.DefaultPrng.init(0x19299);
    const rnd = prng.random();
    const q = try attn256RandBf16Scaled(rnd, &[_]c_int{ 1, 16, 70, 192 }, 8.0, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, 1993, 192 }, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, 1993, 128 }, s);
    defer _ = mlx.mlx_array_free(v);
    var qv = try TestQuantView.init(s, k, v, 8, 64);
    defer qv.deinit();
    const scale: f32 = 1.0 / @sqrt(192.0);
    defer fused256_budget_override = null;
    for (attnPdTestArms()) |arm| {
        attn_pd_nax_override = arm == .nax;
        defer attn_pd_nax_override = null;
        fused256_budget_override = 0;
        const whole = (try fusedSdpaPrefill(s, q, qv.view.k, qv.view.v, scale, 0)) orelse return error.FusedDeclined;
        defer _ = mlx.mlx_array_free(whole);
        try std.testing.expectEqual(@as(u32, 1), fused256_last_dispatch_count);
        fused256_budget_override = 16 * 70 * 256;
        const chunked = (try fusedSdpaPrefill(s, q, qv.view.k, qv.view.v, scale, 0)) orelse return error.FusedDeclined;
        defer _ = mlx.mlx_array_free(chunked);
        try std.testing.expect(fused256_last_dispatch_count > 1);
        const sliced = (try fusedSdpaPrefillKv(s, q, &qv.view, scale, 0, .{ .ctx = null })) orelse return error.FusedDeclined;
        defer _ = mlx.mlx_array_free(sliced);
        try std.testing.expectEqual(arm, attn_pd_last_arm);
        try std.testing.expectEqual(@as(f32, 0), try attn256MaxDiff(whole, chunked, s));
        try std.testing.expectEqual(@as(f32, 0), try attn256MaxDiff(whole, sliced, s));
    }
}

test "sushi_attn_pd_nax takes a 4x dispatch budget: fewer key chunks than the SIMD kernel" {
    if (mlx.noGpuBackend() or !verifyQmmNaxAvailable() or !qsaNaxOsOk()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    fused256_override = true;
    defer fused256_override = null;
    var prng = std.Random.DefaultPrng.init(0x1b4d);
    const rnd = prng.random();
    // 16 heads x 4096 rows: the SIMD budget covers 3808 keys per dispatch, the NAX one 15232.
    const q = try attn256RandBf16(rnd, &[_]c_int{ 1, 16, 4096, 192 }, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &[_]c_int{ 1, 1, 8192, 192 }, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &[_]c_int{ 1, 1, 8192, 128 }, s);
    defer _ = mlx.mlx_array_free(v);
    for ([_]AttnPdArm{ .simd, .nax }, [_]u32{ 3, 1 }) |arm, want| {
        attn_pd_nax_override = arm == .nax;
        defer attn_pd_nax_override = null;
        const out = (try fusedSdpaPrefill(s, q, k, v, 1.0 / @sqrt(192.0), 0)) orelse return error.FusedDeclined;
        defer _ = mlx.mlx_array_free(out);
        try std.testing.expectEqual(arm, attn_pd_last_arm);
        try std.testing.expectEqual(want, fused256_last_dispatch_count);
    }
}

test "sushi_attn_pd_nax: rows that start off an 8-byte boundary give the bytes of a contiguous copy" {
    if (mlx.noGpuBackend() or !verifyQmmNaxAvailable() or !qsaNaxOsOk()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    fused256_override = true;
    defer fused256_override = null;
    attn_pd_nax_override = true;
    defer attn_pd_nax_override = null;
    var prng = std.Random.DefaultPrng.init(0x1b4e);
    const rnd = prng.random();
    // Row strides of 193 and 129 elements: rows start on every 2-byte offset of an 8-byte word.
    var strided: [3]mlx.mlx_array = undefined;
    var dense: [3]mlx.mlx_array = undefined;
    var built: usize = 0;
    defer for (0..built) |i| {
        _ = mlx.mlx_array_free(strided[i]);
        _ = mlx.mlx_array_free(dense[i]);
    };
    for ([_][4]c_int{ .{ 1, 4, 40, 192 }, .{ 1, 1, 96, 192 }, .{ 1, 1, 96, 128 } }) |sh| {
        const wide = try attn256RandBf16(rnd, &[_]c_int{ sh[0], sh[1], sh[2], sh[3] + 1 }, s);
        defer _ = mlx.mlx_array_free(wide);
        strided[built] = mlx.mlx_array_new();
        dense[built] = mlx.mlx_array_new();
        built += 1;
        try mlx.check(mlx.mlx_slice(&strided[built - 1], wide, &[_]c_int{ 0, 0, 0, 0 }, 4, &sh, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
        try mlx.check(mlx.mlx_contiguous(&dense[built - 1], strided[built - 1], false, s));
    }
    const scale: f32 = 1.0 / @sqrt(192.0);
    const want = (try fusedSdpaPrefill(s, dense[0], dense[1], dense[2], scale, 0)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(want);
    try std.testing.expectEqual(AttnPdArm.nax, attn_pd_last_arm);
    const got = (try fusedSdpaPrefill(s, strided[0], strided[1], strided[2], scale, 0)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(got);
    try std.testing.expectEqual(AttnPdArm.nax, attn_pd_last_arm);
    try std.testing.expectEqual(@as(f32, 0), try attn256MaxDiff(got, want, s));
}

test "sushi_attn_pd_nax: a failed probe declines to the SIMD kernel by name" {
    if (mlx.noGpuBackend() or !verifyQmmNaxAvailable() or !qsaNaxOsOk()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    fused256_override = true;
    defer fused256_override = null;
    attn_pd_nax_override = true;
    defer attn_pd_nax_override = null;
    var prng = std.Random.DefaultPrng.init(0x19277);
    const rnd = prng.random();
    const q = try attn256RandBf16(rnd, &[_]c_int{ 1, 4, 32, 192 }, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &[_]c_int{ 1, 1, 80, 192 }, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &[_]c_int{ 1, 1, 80, 128 }, s);
    defer _ = mlx.mlx_array_free(v);
    defer attnPdNaxProbeReset();

    // The real probe passes on this machine, a corrupted NAX tile fails it.
    attnPdNaxProbeReset();
    try std.testing.expect(attnPdNaxRunProbe());
    attn_pd_nax_probe_corrupt_override = true;
    defer attn_pd_nax_probe_corrupt_override = false;
    try std.testing.expect(!attnPdNaxRunProbe());

    // A failed probe latches the SIMD arm for the process.
    attnPdNaxProbeReset();
    const out = (try fusedSdpaPrefill(s, q, k, v, 1.0 / @sqrt(192.0), 0)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(out);
    try std.testing.expectEqual(AttnPdArm.simd, attn_pd_last_arm);
    try std.testing.expect(!mlx.errorPending());
}

test "the sliding arm engages exactly where slidingPrefillFused says it does" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const cfg = swaSinkTestConfig();
    fused256_override = true;
    defer fused256_override = null;
    defer sliding_prefill_fused_override = null;
    var prng = std.Random.DefaultPrng.init(0x5a3c);
    const rnd = prng.random();
    const sinks = try attn256RandBf16(rnd, &[_]c_int{4}, s);
    defer _ = mlx.mlx_array_free(sinks);
    var engaged_any = false;
    for ([_]?bool{ null, false }) |ov| {
        sliding_prefill_fused_override = ov;
        for ([_]c_int{ 8, 15, 16, 64 }) |ql| {
            const q = try attn256RandBf16(rnd, &[_]c_int{ 1, 4, ql, 192 }, s);
            defer _ = mlx.mlx_array_free(q);
            const k = try attn256RandBf16(rnd, &[_]c_int{ 1, 1, ql + 127, 192 }, s);
            defer _ = mlx.mlx_array_free(k);
            const v = try attn256RandBf16(rnd, &[_]c_int{ 1, 1, ql + 127, 128 }, s);
            defer _ = mlx.mlx_array_free(v);
            const view = DenseKVView{ .k = k, .v = v, .owned = false };
            const out = try slidingPrefillAttn(s, &cfg, q, &view, 1.0 / @sqrt(192.0), sinks);
            defer if (out) |o| {
                _ = mlx.mlx_array_free(o);
            };
            try std.testing.expectEqual(slidingPrefillFused(&cfg, @intCast(ql)), out != null);
            engaged_any = engaged_any or out != null;
        }
    }
    try std.testing.expect(engaged_any);
}

test "fusedSdpaPrefill: sliding-band parity vs composed 'array' mask (sliding layers)" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    fused256_override = true;
    defer fused256_override = null;
    var prng = std.Random.DefaultPrng.init(0xBA9D);
    const rnd = prng.random();

    const qL: c_int = 70;
    const kL: c_int = 193;
    const sw: c_int = 40;
    const q_shape = [_]c_int{ 1, 4, qL, 256 };
    const kv_shape = [_]c_int{ 1, 4, kL, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(v);
    const scale: f32 = 1.0;

    // Reference mask: exactly createSlidingWindowMask semantics — masked when
    // col > row_abs (causal) or row_abs - col >= sw, row_abs = kL - qL + r.
    const mask_n: usize = @intCast(qL * kL);
    const mask_data = try std.testing.allocator.alloc(f32, mask_n);
    defer std.testing.allocator.free(mask_data);
    const off: c_int = kL - qL;
    var r: c_int = 0;
    while (r < qL) : (r += 1) {
        var c: c_int = 0;
        while (c < kL) : (c += 1) {
            const row_abs = off + r;
            const masked = (c > row_abs) or (row_abs - c >= sw);
            mask_data[@intCast(r * kL + c)] = if (masked) -std.math.inf(f32) else 0.0;
        }
    }
    const mask_shape = [_]c_int{ 1, 1, qL, kL };
    const mask_f32 = mlx.mlx_array_new_data(mask_data.ptr, &mask_shape, 4, .float32);
    defer _ = mlx.mlx_array_free(mask_f32);
    var mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask);
    try mlx.check(mlx.mlx_astype(&mask, mask_f32, .bfloat16, s));

    const fused = (try fusedSdpaPrefill(s, q, k, v, scale, sw)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(fused);
    const ref = try attn256Reference(q, k, v, scale, "array", mask, s);
    defer _ = mlx.mlx_array_free(ref);

    const max_diff = try attn256MaxDiff(fused, ref, s);
    try std.testing.expect(max_diff < 0.02);
}

test "fusedSdpaPrefill: declines cleanly outside its envelope" {
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0xDEC1);
    const rnd = prng.random();

    // Wrong head_dim -> null.
    const q128_shape = [_]c_int{ 1, 4, 64, 128 };
    const q128 = try attn256RandBf16(rnd, &q128_shape, s);
    defer _ = mlx.mlx_array_free(q128);
    fused256_override = true;
    defer fused256_override = null;
    try std.testing.expect((try fusedSdpaPrefill(s, q128, q128, q128, 1.0, 0)) == null);

    // Decode/verify shapes (q_len < 16) -> null (sdpa_vector owns hd 256
    // there; stealing MTP verify forwards measured decode 48 -> 18 tok/s;
    // the 16 floor matches oMLX's _MIN_ROUTE_Q_LEN — with causal fused
    // default-on, a depth-8 MTP verify is q_len 9 and must NOT route here).
    const q1_shape = [_]c_int{ 1, 4, 1, 256 };
    const q8_shape = [_]c_int{ 1, 4, 8, 256 };
    const q15_shape = [_]c_int{ 1, 4, 15, 256 };
    const kv_shape = [_]c_int{ 1, 4, 64, 256 };
    const q1 = try attn256RandBf16(rnd, &q1_shape, s);
    defer _ = mlx.mlx_array_free(q1);
    const q8 = try attn256RandBf16(rnd, &q8_shape, s);
    defer _ = mlx.mlx_array_free(q8);
    const q15 = try attn256RandBf16(rnd, &q15_shape, s);
    defer _ = mlx.mlx_array_free(q15);
    const k1 = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k1);
    try std.testing.expect((try fusedSdpaPrefill(s, q1, k1, k1, 1.0, 0)) == null);
    try std.testing.expect((try fusedSdpaPrefill(s, q8, k1, k1, 1.0, 0)) == null);
    try std.testing.expect((try fusedSdpaPrefill(s, q15, k1, k1, 1.0, 0)) == null);

    // Kill switch -> null even for a conforming call.
    fused256_override = false;
    const q_shape = [_]c_int{ 1, 4, 64, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    try std.testing.expect((try fusedSdpaPrefill(s, q, q, q, 1.0, 0)) == null);
}

test "fusedSdpaPrefill: both arms FUSED off NAX, causal yields to stock ON NAX" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x4A7E);
    const rnd = prng.random();

    // No master-switch override: the causal arm's historical net-loss (every
    // pre-budget ratio-gated variant lost same-boot on the 27B) was the IOGPU
    // preemption class — with the kv-chunk dispatch budget it wins live
    // (2026-07-22 same-session A/B: +2.9%/+2.3%/+4.6% at 8K/16K/32K on the
    // 27B), so causal is default-on now.
    std.debug.assert(fused256_override == null);
    const q_shape = [_]c_int{ 1, 6, 64, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const kv_shape = [_]c_int{ 1, 2, 64, 256 };
    const k = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k);

    // Off NAX both arms engage — the budgeted-dispatch flip this test was
    // written for. The preference is PINNED rather than read off the host:
    // `naxSdpaPreferred()` defaults to whether the machine has NAX, so leaving
    // it to the silicon makes this assertion true on an M4 and false on an M5.
    nax_gate.nax_sdpa_override = false;
    defer nax_gate.nax_sdpa_override = null;
    {
        const causal = try fusedSdpaPrefill(s, q, k, k, 1.0, 0);
        try std.testing.expect(causal != null);
        if (causal) |f| _ = mlx.mlx_array_free(f);
    }

    // On NAX the causal arm yields to the stock fused sdpa (mlx's own hd-256
    // kernel) while the BAND arm stays ours — mlx's NAX kernel takes no band.
    // Nothing pinned that asymmetry before; a decline is invisible from the
    // outside, since the composed fallback is output-equivalent.
    nax_gate.nax_sdpa_override = true;
    {
        const causal = try fusedSdpaPrefill(s, q, k, k, 1.0, 0);
        try std.testing.expect(causal == null);
    }

    const banded = try fusedSdpaPrefill(s, q, k, k, 1.0, 40);
    try std.testing.expect(banded != null);
    if (banded) |f| _ = mlx.mlx_array_free(f);
}

test "fused256KvChunkLen: BK alignment, one-block floor, kl cap, budget-off" {
    const t = std.testing;
    // budget <= 0: single dispatch covering the full key axis.
    try t.expectEqual(@as(c_int, 193), fused256KvChunkLen(1, 6, 70, 193, 0));
    try t.expectEqual(@as(c_int, 65536), fused256KvChunkLen(1, 24, 2048, 65536, -1));
    // 27B live shape (24 heads, 2048-chunk q) at the default budget:
    // 250M / (24*2048) = 5086 keys -> floored to the BK(32) multiple 5056.
    try t.expectEqual(@as(c_int, 5056), fused256KvChunkLen(1, 24, 2048, 65536, FUSED256_DEFAULT_DISPATCH_BUDGET));
    // Never below one BK block even at absurdly small budgets…
    try t.expectEqual(@as(c_int, 32), fused256KvChunkLen(1, 24, 2048, 65536, 1));
    // …and never above the actual key length.
    try t.expectEqual(@as(c_int, 193), fused256KvChunkLen(1, 6, 70, 193, 1 << 40));
}

test "fusedSdpaPrefill: budgeted kv chunking engages and is exact vs single dispatch" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    fused256_override = true;
    defer fused256_override = null;
    // Causal arm: declines on NAX silicon unless the preference is pinned.
    nax_gate.nax_sdpa_override = false;
    defer nax_gate.nax_sdpa_override = null;
    var prng = std.Random.DefaultPrng.init(0xC4A2);
    const rnd = prng.random();

    // Hq=6/Hk=2, qL=70 (ragged q tile), kL=193 (ragged kv block). Per-key
    // work = 6*70 = 420; budget 40000 -> chunk 64 -> dispatches over
    // [0,64) [64,128) [128,192) [192,193) — 4, incl. a 1-key ragged tail.
    const q_shape = [_]c_int{ 1, 6, 70, 256 };
    const kv_shape = [_]c_int{ 1, 2, 193, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(v);
    const scale: f32 = 1.0 / 16.0;

    fused256_budget_override = 0;
    const single = (try fusedSdpaPrefill(s, q, k, v, scale, 0)) orelse {
        fused256_budget_override = null;
        return error.FusedDeclined;
    };
    defer _ = mlx.mlx_array_free(single);
    try std.testing.expectEqual(@as(u32, 1), fused256_last_dispatch_count);

    fused256_budget_override = 40_000;
    defer fused256_budget_override = null;
    const chunked = (try fusedSdpaPrefill(s, q, k, v, scale, 0)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(chunked);
    // Engagement counted, not inferred: the budget must actually split.
    try std.testing.expectEqual(@as(u32, 4), fused256_last_dispatch_count);

    // The carry rides fp32 buffers — the exact register precision — so the
    // chunked result is BIT-IDENTICAL to the single dispatch, not just close.
    const max_diff = try attn256MaxDiff(single, chunked, s);
    try std.testing.expect(max_diff == 0.0);

    // And still correct in absolute terms vs the composed reference.
    const ref = try attn256Reference(q, k, v, scale, "causal", .{ .ctx = null }, s);
    defer _ = mlx.mlx_array_free(ref);
    const ref_diff = try attn256MaxDiff(chunked, ref, s);
    try std.testing.expect(ref_diff < 0.005);
}

test "fusedSdpaPrefill: band arm never chunks (single dispatch under a tiny budget)" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    fused256_override = true;
    defer fused256_override = null;
    fused256_budget_override = 1; // would force max chunking on the causal arm
    defer fused256_budget_override = null;
    var prng = std.Random.DefaultPrng.init(0xBA9D2);
    const rnd = prng.random();

    const qL: c_int = 70;
    const kL: c_int = 193;
    const sw: c_int = 40;
    const q_shape = [_]c_int{ 1, 4, qL, 256 };
    const kv_shape = [_]c_int{ 1, 4, kL, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(v);

    const fused = (try fusedSdpaPrefill(s, q, k, v, 1.0, sw)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(fused);
    // The band arm's block skip already bounds per-dispatch work; chunking it
    // would only add carry traffic. It must stay a single dispatch.
    try std.testing.expectEqual(@as(u32, 1), fused256_last_dispatch_count);

    // Unchanged band semantics (same reference as the band parity test).
    const mask_n: usize = @intCast(qL * kL);
    const mask_data = try std.testing.allocator.alloc(f32, mask_n);
    defer std.testing.allocator.free(mask_data);
    const off: c_int = kL - qL;
    var r: c_int = 0;
    while (r < qL) : (r += 1) {
        var c: c_int = 0;
        while (c < kL) : (c += 1) {
            const row_abs = off + r;
            const masked = (c > row_abs) or (row_abs - c >= sw);
            mask_data[@intCast(r * kL + c)] = if (masked) -std.math.inf(f32) else 0.0;
        }
    }
    const mask_shape = [_]c_int{ 1, 1, qL, kL };
    const mask_f32 = mlx.mlx_array_new_data(mask_data.ptr, &mask_shape, 4, .float32);
    defer _ = mlx.mlx_array_free(mask_f32);
    var mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask);
    try mlx.check(mlx.mlx_astype(&mask, mask_f32, .bfloat16, s));
    const ref = try attn256Reference(q, k, v, 1.0, "array", mask, s);
    defer _ = mlx.mlx_array_free(ref);
    const max_diff = try attn256MaxDiff(fused, ref, s);
    try std.testing.expect(max_diff < 0.02);
}

test "fusedSdpaPrefill: causal parity at Qwen 24q/4kv geometry (gqa 6, ragged 64-row tile)" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    fused256_override = true;
    defer fused256_override = null;
    // Causal arm: declines on NAX silicon unless the preference is pinned.
    nax_gate.nax_sdpa_override = false;
    defer nax_gate.nax_sdpa_override = null;
    var prng = std.Random.DefaultPrng.init(0x27B27B);
    const rnd = prng.random();

    // The production Qwen3.6-27B full-attention geometry: 24 q heads over 4
    // kv heads (gqa 6), hd 256. qL=97 exercises a partial 64-row q tile,
    // kL=250 a partial kv block; offset 153 = chunked-prefill alignment.
    const q_shape = [_]c_int{ 1, 24, 97, 256 };
    const kv_shape = [_]c_int{ 1, 4, 250, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(v);
    const scale: f32 = 1.0 / 16.0;

    const fused = (try fusedSdpaPrefill(s, q, k, v, scale, 0)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(fused);
    const ref = try attn256Reference(q, k, v, scale, "causal", .{ .ctx = null }, s);
    defer _ = mlx.mlx_array_free(ref);

    const max_diff = try attn256MaxDiff(fused, ref, s);
    try std.testing.expect(max_diff < 0.005);
}
