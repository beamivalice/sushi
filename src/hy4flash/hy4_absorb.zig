//! A decode row's (or a verify block's, up to eight rows, each row its own bytes) absorbed MLA products over `kv_b` as stored (BF16), each value widened in the register: the query
//! into the latent (`q_nope · W_k`, MLX's `gemv_t` over the widened weight) and the attended latent out (`u_lat · W_vᵀ`,
//! MLX's `gemv`). Each keeps the thread layout and f32 accumulation order of the MLX kernel the engine picks for the
//! f32 product at these shapes (`gemv.h`: `GEMVTKernel` bm1 bn4 sm8 sn4 tm4 tn4, `GEMVKernel` bm4 bn1 sm1 sn32 tm4
//! tn4), so each output is the bytes of the product over the f32 copy, which the weight then never needs.
const std = @import("std");
const mlx = @import("../mlx.zig");
const Arr = mlx.mlx_array;
const kcfg = @import("hy4_kcfg.zig");

/// `out[h, n] = Σ_k q[h, k] · W[h·ROWS + k, n]`: `q` [H, NOPE] f32, `W` = kv_b [H·ROWS, LAT] BF16. `gemv_t`'s tile: NSG
/// simdgroups of 8 x 4 lanes, 4 inputs by 4 outputs a thread, NSG · 16 outputs a threadgroup.
pub const q_source: [:0]const u8 =
    \\using W4 = metal::vec<TW, 4>;
    \\using F4 = metal::vec<float, 4>;
    \\constexpr int SM = 8;
    \\constexpr int SN = 4;
    \\constexpr int TM = 4;
    \\constexpr int TN = 4;
    \\constexpr int BLOCK_M = SM * TM;
    \\constexpr int MAX_RT = 8;
    \\const int BLOCK_N = NSG * SN * TN;
    \\uint lane = thread_index_in_simdgroup;
    \\uint sg = simdgroup_index_in_threadgroup;
    \\uint z = threadgroup_position_in_grid.z;
    \\uint h = z % uint(HEADS);
    \\int r0 = int(z / uint(HEADS)) * RT;
    \\const device TW* mat = kvb + (size_t)h * (size_t)ROWS * (size_t)LAT;
    \\const device float* q_h = q + (size_t)h * (size_t)q_strides[0];
    \\const size_t qs1 = (size_t)q_strides[1];
    \\const bool full_tile = (r0 + RT <= R);
    \\int thr_m = int(lane) / SN;
    \\int thr_n = int(lane) % SN;
    \\int bm = thr_m * TM;
    \\int out_col = int(threadgroup_position_in_grid.x) * BLOCK_N + (SN * int(sg) + thr_n) * TN;
    \\float result[MAX_RT][TN];
    \\#pragma clang loop unroll(full)
    \\for (int r = 0; r < MAX_RT; ++r) {
    \\  if (r >= RT) break;
    \\  #pragma clang loop unroll(full)
    \\  for (int tn = 0; tn < TN; ++tn) result[r][tn] = 0.0f;
    \\}
    \\const int n_iter = NOPE / BLOCK_M;
    \\#ifndef SUSHI_METALLIB
    \\static_assert(NOPE % BLOCK_M == 0 && LAT % BLOCK_N == 0, "an absorbed shape off the gemv_t tile");
    \\#endif
    \\for (int i = 0; i < n_iter; ++i) {
    \\  F4 qv[MAX_RT];
    \\  #pragma clang loop unroll(full)
    \\  for (int r = 0; r < MAX_RT; ++r) {
    \\    if (r >= RT) break;
    \\    if (full_tile || r0 + r < R) {
    \\      qv[r] = *(const device F4*)(q_h + (size_t)(r0 + r) * qs1 + bm);
    \\    } else {
    \\      qv[r] = F4(0.0f);
    \\    }
    \\  }
    \\  W4 mv[TM];
    \\  #pragma clang loop unroll(full)
    \\  for (int tm = 0; tm < TM; ++tm) {
    \\    mv[tm] = *(const device W4*)(mat + (size_t)(bm + tm) * (size_t)LAT + out_col);
    \\  }
    \\  #pragma clang loop unroll(full)
    \\  for (int tm = 0; tm < TM; ++tm) {
    \\    const float w0 = static_cast<float>(mv[tm][0]);
    \\    const float w1 = static_cast<float>(mv[tm][1]);
    \\    const float w2 = static_cast<float>(mv[tm][2]);
    \\    const float w3 = static_cast<float>(mv[tm][3]);
    \\    #pragma clang loop unroll(full)
    \\    for (int r = 0; r < MAX_RT; ++r) {
    \\      if (r >= RT) break;
    \\      const float vc = qv[r][tm];
    \\      result[r][0] += vc * w0;
    \\      result[r][1] += vc * w1;
    \\      result[r][2] += vc * w2;
    \\      result[r][3] += vc * w3;
    \\    }
    \\  }
    \\  bm += BLOCK_M;
    \\}
    \\#pragma clang loop unroll(full)
    \\for (int r = 0; r < MAX_RT; ++r) {
    \\  if (r >= RT) break;
    \\  #pragma clang loop unroll(full)
    \\  for (int tn = 0; tn < TN; ++tn) {
    \\    #pragma clang loop unroll(full)
    \\    for (ushort sm = SM / 2; sm >= 1; sm >>= 1) result[r][tn] += simd_shuffle_down(result[r][tn], ushort(SN) * sm);
    \\  }
    \\}
    \\if (thr_m == 0) {
    \\  #pragma clang loop unroll(full)
    \\  for (int r = 0; r < MAX_RT; ++r) {
    \\    if (r >= RT) break;
    \\    int rr = r0 + r;
    \\    if (rr < R) {
    \\      const size_t dst_row = ROW_MAJOR ? ((size_t)rr * (size_t)HEADS + (size_t)h) : ((size_t)h * (size_t)R + (size_t)rr);
    \\      device float* o = out + dst_row * (size_t)LAT + out_col;
    \\      #pragma clang loop unroll(full)
    \\      for (int tn = 0; tn < TN; ++tn) o[tn] = result[r][tn];
    \\    }
    \\  }
    \\}
;

/// `out[h, r] = Σ_k W[h·ROWS + NOPE + r, k] · u[h, k]`: `u` [H, LAT] f32. `gemv`'s tile: 4 simdgroups of 4 rows, 4
/// columns a lane, 16 rows a threadgroup.
const v_main: [:0]const u8 =
    \\using W4 = metal::vec<TW, 4>;
    \\using F4 = metal::vec<float, 4>;
    \\constexpr int TM = 4;
    \\constexpr int TN = 4;
    \\constexpr int BLOCK_N = 32 * TN;
    \\constexpr int MAX_RT = 8;
    \\uint lane = thread_index_in_simdgroup;
    \\uint sg = simdgroup_index_in_threadgroup;
    \\uint z = threadgroup_position_in_grid.z;
    \\uint h = z % uint(HEADS);
    \\int r0 = int(z / uint(HEADS)) * RT;
    \\int out_row = int(threadgroup_position_in_grid.x) * 4 * TM + int(sg) * TM;
    \\const device TW* mat = kvb + ((size_t)h * (size_t)ROWS + (size_t)NOPE + (size_t)out_row) * (size_t)LAT;
    \\const device float* u_h = u + (size_t)h * (size_t)u_strides[0];
    \\const size_t us1 = (size_t)u_strides[1];
    \\const bool full_tile = (r0 + RT <= R);
    \\#ifndef SUSHI_METALLIB
    \\static_assert(LAT % BLOCK_N == 0 && VDIM % (4 * TM) == 0, "an absorbed shape off the gemv tile");
    \\#endif
    \\float result[MAX_RT][TM];
    \\#pragma clang loop unroll(full)
    \\for (int r = 0; r < MAX_RT; ++r) {
    \\  if (r >= RT) break;
    \\  #pragma clang loop unroll(full)
    \\  for (int tm = 0; tm < TM; ++tm) result[r][tm] = 0.0f;
    \\}
    \\int bn = int(lane) * TN;
    \\const int n_iter = LAT / BLOCK_N;
    \\for (int i = 0; i < n_iter; ++i) {
    \\  F4 uv[MAX_RT];
    \\  #pragma clang loop unroll(full)
    \\  for (int r = 0; r < MAX_RT; ++r) {
    \\    if (r >= RT) break;
    \\    if (full_tile || r0 + r < R) {
    \\      uv[r] = *(const device F4*)(u_h + (size_t)(r0 + r) * us1 + bn);
    \\    } else {
    \\      uv[r] = F4(0.0f);
    \\    }
    \\  }
    \\  W4 mv[TM];
    \\  #pragma clang loop unroll(full)
    \\  for (int tm = 0; tm < TM; ++tm) {
    \\    mv[tm] = *(const device W4*)(mat + (size_t)tm * (size_t)LAT + bn);
    \\  }
    \\  #pragma clang loop unroll(full)
    \\  for (int tm = 0; tm < TM; ++tm) {
    \\    const float w0 = static_cast<float>(mv[tm][0]);
    \\    const float w1 = static_cast<float>(mv[tm][1]);
    \\    const float w2 = static_cast<float>(mv[tm][2]);
    \\    const float w3 = static_cast<float>(mv[tm][3]);
    \\    #pragma clang loop unroll(full)
    \\    for (int r = 0; r < MAX_RT; ++r) {
    \\      if (r >= RT) break;
    \\      result[r][tm] += w0 * uv[r][0];
    \\      result[r][tm] += w1 * uv[r][1];
    \\      result[r][tm] += w2 * uv[r][2];
    \\      result[r][tm] += w3 * uv[r][3];
    \\    }
    \\  }
    \\  bn += BLOCK_N;
    \\}
    \\#pragma clang loop unroll(full)
    \\for (int r = 0; r < MAX_RT; ++r) {
    \\  if (r >= RT) break;
    \\  #pragma clang loop unroll(full)
    \\  for (int tm = 0; tm < TM; ++tm) {
    \\    #pragma clang loop unroll(full)
    \\    for (ushort sn = 16; sn >= 1; sn >>= 1) result[r][tm] += simd_shuffle_down(result[r][tm], sn);
    \\  }
    \\}
;

const v_out_plain: [:0]const u8 =
    \\if (lane == 0u) {
    \\  #pragma clang loop unroll(full)
    \\  for (int r = 0; r < MAX_RT; ++r) {
    \\    if (r >= RT) break;
    \\    int rr = r0 + r;
    \\    if (rr < R) {
    \\      device float* o = out + ((size_t)h * (size_t)R + (size_t)rr) * (size_t)VDIM + out_row;
    \\      #pragma clang loop unroll(full)
    \\      for (int tm = 0; tm < TM; ++tm) o[tm] = result[r][tm];
    \\    }
    \\  }
    \\}
;

const v_out_gated: [:0]const u8 =
    \\threadgroup float tg_res[4][32];
    \\if (lane == 0u) {
    \\  #pragma clang loop unroll(full)
    \\  for (int r = 0; r < MAX_RT; ++r) {
    \\    if (r >= RT) break;
    \\    #pragma clang loop unroll(full)
    \\    for (int tm = 0; tm < TM; ++tm) tg_res[sg][r * TM + tm] = result[r][tm];
    \\  }
    \\}
    \\simdgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\if (lane < uint(RT * TM)) {
    \\  const int r = int(lane) >> 2;
    \\  const int tm = int(lane) & 3;
    \\  const int rr = r0 + r;
    \\  if (full_tile || rr < R) {
    \\    const size_t row_off = ((size_t)rr * (size_t)HEADS + (size_t)h) * (size_t)VDIM + (size_t)(out_row + tm);
    \\    const size_t g_off = (size_t)rr * (size_t)gate_strides[0] + (size_t)h * (size_t)VDIM + (size_t)(out_row + tm);
    \\    out[row_off] = tg_res[sg][lane] * hy4_sigmoid(static_cast<float>(gate[g_off]));
    \\  }
    \\}
;

pub const v_source = v_main ++ v_out_plain;
pub const v_source_gated = v_main ++ v_out_gated;

pub const sigmoid_header: [:0]const u8 =
    \\inline float hy4_sigmoid(float x) {
    \\  auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
    \\  return (x < 0) ? y : 1 - y;
    \\}
;

/// Wide-span absorbed query product `out[h, m, n] = Σ_k q[h, m, k] · float(kvb[h·ROWS + k, n])` on FP32
/// `simdgroup_float8x8` (BM=64, BN=64, BK=16, WM=2, WN=2): reads BF16 `kvb` directly (no f32 widening copy) and
/// strided f32 `q` directly (no slice copy), summing K in the exact 8-step `simdgroup_multiply_accumulate` order of
/// MLX's `steel_gemm_fused_nn_float32` so every output element matches byte-for-byte. Threadgroup memory stays at
/// 10.24 KB (<= 16 KB for 2+ TG/core occupancy), with double-buffered `Bs` (1 barrier / BK=16) and direct L1 `q`
/// loads on aligned tiles.
pub const q_sg_source: [:0]const u8 =
    \\using B4 = metal::vec<bfloat16_t, 4>;
    \\using F4 = metal::vec<float, 4>;
    \\constexpr int BM = 64, BN = 64, BK = 16;
    \\constexpr int A_STR = BK + 4;
    \\constexpr int B_STR = BN + 4;
    \\constexpr int BS_SZ = BK * B_STR;
    \\const uint tid = thread_index_in_threadgroup, sg = simdgroup_index_in_threadgroup;
    \\const int R = q_shape[1];
    \\const int h = int(threadgroup_position_in_grid.z);
    \\const int m0 = int(threadgroup_position_in_grid.y) * BM;
    \\const int n0 = int(threadgroup_position_in_grid.x) * BN;
    \\const int sg_m = int(sg) >> 1, sg_n = int(sg) & 1;
    \\threadgroup float sm[2560];
    \\simdgroup_float8x8 acc[4][4];
    \\#pragma clang loop unroll(full)
    \\for (int mi = 0; mi < 4; ++mi)
    \\  #pragma clang loop unroll(full)
    \\  for (int ni = 0; ni < 4; ++ni) acc[mi][ni] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    \\const int bk = int(tid) >> 3;
    \\const int bn = (int(tid) & 7) * 8;
    \\const size_t qs0 = size_t(q_strides[0]), qs1 = size_t(q_strides[1]);
    \\const device bfloat16_t* kvb_h = kvb + size_t(h) * size_t(ROWS) * size_t(LAT);
    \\if (m0 + BM <= R && q_strides[2] == 1) {
    \\  const device float* q_sg = q + size_t(h) * qs0 + size_t(m0 + sg_m * 32) * qs1;
    \\  {
    \\    const device B4* bp = (const device B4*)(kvb_h + size_t(bk) * size_t(LAT) + size_t(n0 + bn));
    \\    const B4 v0 = bp[0], v1 = bp[1];
    \\    threadgroup F4* bdst = (threadgroup F4*)(sm + bk * B_STR + bn);
    \\    bdst[0] = F4(float(v0[0]), float(v0[1]), float(v0[2]), float(v0[3]));
    \\    bdst[1] = F4(float(v1[0]), float(v1[1]), float(v1[2]), float(v1[3]));
    \\  }
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  int buf = 0;
    \\  for (int k0 = 0; k0 < NOPE; k0 += BK, buf ^= 1) {
    \\    threadgroup float* Bs_cur = sm + buf * BS_SZ;
    \\    if (k0 + BK < NOPE) {
    \\      const device B4* bp = (const device B4*)(kvb_h + size_t(k0 + BK + bk) * size_t(LAT) + size_t(n0 + bn));
    \\      const B4 v0 = bp[0], v1 = bp[1];
    \\      threadgroup F4* bdst = (threadgroup F4*)(sm + (buf ^ 1) * BS_SZ + bk * B_STR + bn);
    \\      bdst[0] = F4(float(v0[0]), float(v0[1]), float(v0[2]), float(v0[3]));
    \\      bdst[1] = F4(float(v1[0]), float(v1[1]), float(v1[2]), float(v1[3]));
    \\    }
    \\    #pragma clang loop unroll(full)
    \\    for (int kk = 0; kk < BK; kk += 8) {
    \\      simdgroup_float8x8 a[4], b[4];
    \\      #pragma clang loop unroll(full)
    \\      for (int mi = 0; mi < 4; ++mi) simdgroup_load(a[mi], q_sg + size_t(mi * 8) * qs1 + size_t(k0 + kk), qs1);
    \\      #pragma clang loop unroll(full)
    \\      for (int ni = 0; ni < 4; ++ni) simdgroup_load(b[ni], Bs_cur + kk * B_STR + sg_n * 32 + ni * 8, B_STR);
    \\      #pragma clang loop unroll(full)
    \\      for (int mi = 0; mi < 4; ++mi)
    \\        #pragma clang loop unroll(full)
    \\        for (int ni = 0; ni < 4; ++ni) simdgroup_multiply_accumulate(acc[mi][ni], a[mi], b[ni], acc[mi][ni]);
    \\    }
    \\    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  }
    \\  device float* dst = out + (size_t(h) * size_t(R) + size_t(m0 + sg_m * 32)) * size_t(LAT) + size_t(n0 + sg_n * 32);
    \\  #pragma clang loop unroll(full)
    \\  for (int mi = 0; mi < 4; ++mi)
    \\    #pragma clang loop unroll(full)
    \\    for (int ni = 0; ni < 4; ++ni) simdgroup_store(acc[mi][ni], dst + size_t(mi * 8) * size_t(LAT) + ni * 8, LAT);
    \\} else {
    \\  threadgroup float* As = sm;
    \\  threadgroup float* Bs = sm + BM * A_STR;
    \\  const int am = int(tid) >> 1;
    \\  const int ak = (int(tid) & 1) * 8;
    \\  const size_t q_row_base = size_t(h) * qs0 + size_t(m0 + am) * qs1;
    \\  const size_t qs2 = size_t(q_strides[2]);
    \\  const bool sg_active = (m0 + sg_m * 32) < R;
    \\  const bool hi_active = (m0 + sg_m * 32 + 16) < R;
    \\  for (int k0 = 0; k0 < NOPE; k0 += BK) {
    \\    threadgroup F4* adst = (threadgroup F4*)(As + am * A_STR + ak);
    \\    if (m0 + am < R) {
    \\      if (qs2 == 1) {
    \\        const device F4* qp4 = (const device F4*)(q + q_row_base + size_t(k0 + ak));
    \\        adst[0] = qp4[0];
    \\        adst[1] = qp4[1];
    \\      } else {
    \\        const device float* qp = q + q_row_base;
    \\        adst[0] = F4(qp[size_t(k0 + ak + 0) * qs2], qp[size_t(k0 + ak + 1) * qs2], qp[size_t(k0 + ak + 2) * qs2], qp[size_t(k0 + ak + 3) * qs2]);
    \\        adst[1] = F4(qp[size_t(k0 + ak + 4) * qs2], qp[size_t(k0 + ak + 5) * qs2], qp[size_t(k0 + ak + 6) * qs2], qp[size_t(k0 + ak + 7) * qs2]);
    \\      }
    \\    } else {
    \\      adst[0] = F4(0.0f);
    \\      adst[1] = F4(0.0f);
    \\    }
    \\    const device B4* bp = (const device B4*)(kvb_h + size_t(k0 + bk) * size_t(LAT) + size_t(n0 + bn));
    \\    const B4 v0 = bp[0], v1 = bp[1];
    \\    threadgroup F4* bdst = (threadgroup F4*)(Bs + bk * B_STR + bn);
    \\    bdst[0] = F4(float(v0[0]), float(v0[1]), float(v0[2]), float(v0[3]));
    \\    bdst[1] = F4(float(v1[0]), float(v1[1]), float(v1[2]), float(v1[3]));
    \\    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\    if (sg_active) {
    \\      #pragma clang loop unroll(full)
    \\      for (int kk = 0; kk < BK; kk += 8) {
    \\        simdgroup_float8x8 a[4], b[4];
    \\        #pragma clang loop unroll(full)
    \\        for (int ni = 0; ni < 4; ++ni) simdgroup_load(b[ni], Bs + kk * B_STR + sg_n * 32 + ni * 8, B_STR);
    \\        #pragma clang loop unroll(full)
    \\        for (int mi = 0; mi < 2; ++mi) simdgroup_load(a[mi], As + (sg_m * 32 + mi * 8) * A_STR + kk, A_STR);
    \\        #pragma clang loop unroll(full)
    \\        for (int mi = 0; mi < 2; ++mi)
    \\          #pragma clang loop unroll(full)
    \\          for (int ni = 0; ni < 4; ++ni) simdgroup_multiply_accumulate(acc[mi][ni], a[mi], b[ni], acc[mi][ni]);
    \\        if (hi_active) {
    \\          #pragma clang loop unroll(full)
    \\          for (int mi = 2; mi < 4; ++mi) simdgroup_load(a[mi], As + (sg_m * 32 + mi * 8) * A_STR + kk, A_STR);
    \\          #pragma clang loop unroll(full)
    \\          for (int mi = 2; mi < 4; ++mi)
    \\            #pragma clang loop unroll(full)
    \\            for (int ni = 0; ni < 4; ++ni) simdgroup_multiply_accumulate(acc[mi][ni], a[mi], b[ni], acc[mi][ni]);
    \\        }
    \\      }
    \\    }
    \\    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  }
    \\  const int c4 = (int(tid) & 15) * 4;
    \\  for (int half_m = 0; half_m < 2; ++half_m) {
    \\    if (m0 + half_m * 32 >= R) break;
    \\    if (sg_m == half_m) {
    \\      #pragma clang loop unroll(full)
    \\      for (int mi = 0; mi < 4; ++mi)
    \\        #pragma clang loop unroll(full)
    \\        for (int ni = 0; ni < 4; ++ni) simdgroup_store(acc[mi][ni], sm + (mi * 8) * BN + sg_n * 32 + ni * 8, BN);
    \\    }
    \\    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\    for (int r = int(tid) >> 4; r < 32; r += 8) {
    \\      const int mr = m0 + half_m * 32 + r;
    \\      if (mr < R) {
    \\        device F4* dst = (device F4*)(out + (size_t(h) * size_t(R) + size_t(mr)) * size_t(LAT) + size_t(n0 + c4));
    \\        *dst = *(const threadgroup F4*)(sm + r * BN + c4);
    \\      }
    \\    }
    \\    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  }
    \\}
;

/// Shared `u_lat @ W_v^T` FP32 `simdgroup_float8x8` core (BM=64, BN=64, BK=16, WM=2, WN=2): stages BF16 `kvb`
/// non-transposed via coalesced `F4` stores into `Bs[BN][BK+4]` and uses hardware `simdgroup_load(..., true)`
/// transpose. Keeps threadgroup memory at 10.24 KB (<= 16 KB) with double-buffered `Bs` (1 barrier / BK=16) and
/// direct L1 `u` loads on aligned tiles.
const v_sg_core: [:0]const u8 =
    \\using B4 = metal::vec<bfloat16_t, 4>;
    \\using F4 = metal::vec<float, 4>;
    \\constexpr int BM = 64, BN = 64, BK = 16;
    \\constexpr int A_STR = BK + 4;
    \\constexpr int B_STR = BK + 4;
    \\constexpr int BS_SZ = BN * B_STR;
    \\const uint tid = thread_index_in_threadgroup, sg = simdgroup_index_in_threadgroup;
    \\const int R = u_shape[1];
    \\const int h = int(threadgroup_position_in_grid.z);
    \\const int m0 = int(threadgroup_position_in_grid.y) * BM;
    \\const int n0 = int(threadgroup_position_in_grid.x) * BN;
    \\const int sg_m = int(sg) >> 1, sg_n = int(sg) & 1;
    \\threadgroup float sm[2 * BS_SZ];
    \\simdgroup_float8x8 acc[4][4];
    \\#pragma clang loop unroll(full)
    \\for (int mi = 0; mi < 4; ++mi)
    \\  #pragma clang loop unroll(full)
    \\  for (int ni = 0; ni < 4; ++ni) acc[mi][ni] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    \\const int bn = int(tid) >> 1;
    \\const int bk = (int(tid) & 1) * 8;
    \\const size_t us0 = size_t(u_strides[0]), us1 = size_t(u_strides[1]);
    \\const device bfloat16_t* wv_row = kvb + (size_t(h) * size_t(ROWS) + size_t(NOPE) + size_t(n0 + bn)) * size_t(LAT);
    \\if (m0 + BM <= R && u_strides[2] == 1) {
    \\  const device float* u_sg = u + size_t(h) * us0 + size_t(m0 + sg_m * 32) * us1;
    \\  {
    \\    const device B4* bp = (const device B4*)(wv_row + size_t(bk));
    \\    const B4 v0 = bp[0], v1 = bp[1];
    \\    threadgroup F4* bdst = (threadgroup F4*)(sm + bn * B_STR + bk);
    \\    bdst[0] = F4(float(v0[0]), float(v0[1]), float(v0[2]), float(v0[3]));
    \\    bdst[1] = F4(float(v1[0]), float(v1[1]), float(v1[2]), float(v1[3]));
    \\  }
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  int buf = 0;
    \\  for (int k0 = 0; k0 < LAT; k0 += BK, buf ^= 1) {
    \\    threadgroup float* Bs_cur = sm + buf * BS_SZ;
    \\    if (k0 + BK < LAT) {
    \\      const device B4* bp = (const device B4*)(wv_row + size_t(k0 + BK + bk));
    \\      const B4 v0 = bp[0], v1 = bp[1];
    \\      threadgroup F4* bdst = (threadgroup F4*)(sm + (buf ^ 1) * BS_SZ + bn * B_STR + bk);
    \\      bdst[0] = F4(float(v0[0]), float(v0[1]), float(v0[2]), float(v0[3]));
    \\      bdst[1] = F4(float(v1[0]), float(v1[1]), float(v1[2]), float(v1[3]));
    \\    }
    \\    #pragma clang loop unroll(full)
    \\    for (int kk = 0; kk < BK; kk += 8) {
    \\      simdgroup_float8x8 a[4], b[4];
    \\      #pragma clang loop unroll(full)
    \\      for (int mi = 0; mi < 4; ++mi) simdgroup_load(a[mi], u_sg + size_t(mi * 8) * us1 + size_t(k0 + kk), us1);
    \\      #pragma clang loop unroll(full)
    \\      for (int ni = 0; ni < 4; ++ni) simdgroup_load(b[ni], Bs_cur + (sg_n * 32 + ni * 8) * B_STR + kk, B_STR, ulong2(0, 0), true);
    \\      #pragma clang loop unroll(full)
    \\      for (int mi = 0; mi < 4; ++mi)
    \\        #pragma clang loop unroll(full)
    \\        for (int ni = 0; ni < 4; ++ni) simdgroup_multiply_accumulate(acc[mi][ni], a[mi], b[ni], acc[mi][ni]);
    \\    }
    \\    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  }
    \\} else {
    \\  threadgroup float* As = sm;
    \\  threadgroup float* Bs = sm + BM * A_STR;
    \\  const int am = bn;
    \\  const int ak = bk;
    \\  const size_t u_row_base = size_t(h) * us0 + size_t(m0 + am) * us1;
    \\  const size_t us2 = size_t(u_strides[2]);
    \\  const bool sg_active = (m0 + sg_m * 32) < R;
    \\  const bool hi_active = (m0 + sg_m * 32 + 16) < R;
    \\  for (int k0 = 0; k0 < LAT; k0 += BK) {
    \\    threadgroup F4* adst = (threadgroup F4*)(As + am * A_STR + ak);
    \\    if (m0 + am < R) {
    \\      if (us2 == 1) {
    \\        const device F4* up4 = (const device F4*)(u + u_row_base + size_t(k0 + ak));
    \\        adst[0] = up4[0];
    \\        adst[1] = up4[1];
    \\      } else {
    \\        const device float* up = u + u_row_base;
    \\        adst[0] = F4(up[size_t(k0 + ak + 0) * us2], up[size_t(k0 + ak + 1) * us2], up[size_t(k0 + ak + 2) * us2], up[size_t(k0 + ak + 3) * us2]);
    \\        adst[1] = F4(up[size_t(k0 + ak + 4) * us2], up[size_t(k0 + ak + 5) * us2], up[size_t(k0 + ak + 6) * us2], up[size_t(k0 + ak + 7) * us2]);
    \\      }
    \\    } else {
    \\      adst[0] = F4(0.0f);
    \\      adst[1] = F4(0.0f);
    \\    }
    \\    const device B4* bp = (const device B4*)(wv_row + size_t(k0 + bk));
    \\    const B4 v0 = bp[0], v1 = bp[1];
    \\    threadgroup F4* bdst = (threadgroup F4*)(Bs + bn * B_STR + bk);
    \\    bdst[0] = F4(float(v0[0]), float(v0[1]), float(v0[2]), float(v0[3]));
    \\    bdst[1] = F4(float(v1[0]), float(v1[1]), float(v1[2]), float(v1[3]));
    \\    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\    if (sg_active) {
    \\      #pragma clang loop unroll(full)
    \\      for (int kk = 0; kk < BK; kk += 8) {
    \\        simdgroup_float8x8 a[4], b[4];
    \\        #pragma clang loop unroll(full)
    \\        for (int ni = 0; ni < 4; ++ni) simdgroup_load(b[ni], Bs + (sg_n * 32 + ni * 8) * B_STR + kk, B_STR, ulong2(0, 0), true);
    \\        #pragma clang loop unroll(full)
    \\        for (int mi = 0; mi < 2; ++mi) simdgroup_load(a[mi], As + (sg_m * 32 + mi * 8) * A_STR + kk, A_STR);
    \\        #pragma clang loop unroll(full)
    \\        for (int mi = 0; mi < 2; ++mi)
    \\          #pragma clang loop unroll(full)
    \\          for (int ni = 0; ni < 4; ++ni) simdgroup_multiply_accumulate(acc[mi][ni], a[mi], b[ni], acc[mi][ni]);
    \\        if (hi_active) {
    \\          #pragma clang loop unroll(full)
    \\          for (int mi = 2; mi < 4; ++mi) simdgroup_load(a[mi], As + (sg_m * 32 + mi * 8) * A_STR + kk, A_STR);
    \\          #pragma clang loop unroll(full)
    \\          for (int mi = 2; mi < 4; ++mi)
    \\            #pragma clang loop unroll(full)
    \\            for (int ni = 0; ni < 4; ++ni) simdgroup_multiply_accumulate(acc[mi][ni], a[mi], b[ni], acc[mi][ni]);
    \\        }
    \\      }
    \\    }
    \\    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  }
    \\}
;

pub const v_sg_source: [:0]const u8 = v_sg_core ++ "\n" ++
    \\if (m0 + BM <= R) {
    \\  device float* dst = out + (size_t(h) * size_t(R) + size_t(m0 + sg_m * 32)) * size_t(VDIM) + size_t(n0 + sg_n * 32);
    \\  #pragma clang loop unroll(full)
    \\  for (int mi = 0; mi < 4; ++mi)
    \\    #pragma clang loop unroll(full)
    \\    for (int ni = 0; ni < 4; ++ni) simdgroup_store(acc[mi][ni], dst + size_t(mi * 8) * size_t(VDIM) + ni * 8, VDIM);
    \\} else {
    \\  const int c4 = (int(tid) & 15) * 4;
    \\  for (int half_m = 0; half_m < 2; ++half_m) {
    \\    if (m0 + half_m * 32 >= R) break;
    \\    if (sg_m == half_m) {
    \\      #pragma clang loop unroll(full)
    \\      for (int mi = 0; mi < 4; ++mi)
    \\        #pragma clang loop unroll(full)
    \\        for (int ni = 0; ni < 4; ++ni) simdgroup_store(acc[mi][ni], sm + (mi * 8) * BN + sg_n * 32 + ni * 8, BN);
    \\    }
    \\    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\    for (int r = int(tid) >> 4; r < 32; r += 8) {
    \\      const int mr = m0 + half_m * 32 + r;
    \\      if (mr < R) {
    \\        device F4* dst = (device F4*)(out + (size_t(h) * size_t(R) + size_t(mr)) * size_t(VDIM) + size_t(n0 + c4));
    \\        *dst = *(const threadgroup F4*)(sm + r * BN + c4);
    \\      }
    \\    }
    \\    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  }
    \\}
;

/// Fused wide-span `u_lat @ W_v^T` + `hy4_sigmoid(gate)` written directly row-major to `out[row0 + m, h·VDIM + n]`
/// as `OT` (bf16 or f32): eliminates both the `[H, R, VDIM]` f32 buffer and the `hy4_chain.gated`/`gatedInto` launch.
pub const v_sg_gated_source: [:0]const u8 = "#pragma clang fp contract(off)\n" ++ v_sg_core ++ "\n" ++
    \\const int c4 = (int(tid) & 15) * 4;
    \\const int base_row = row0[0];
    \\const size_t hv = size_t(HEADS) * size_t(VDIM);
    \\const size_t gcol = size_t(h) * size_t(VDIM) + size_t(n0 + c4);
    \\const size_t gs0 = size_t(gate_strides[0]), gs1 = size_t(gate_strides[1]);
    \\for (int half_m = 0; half_m < 2; ++half_m) {
    \\  if (m0 + half_m * 32 >= R) break;
    \\  if (sg_m == half_m) {
    \\    #pragma clang loop unroll(full)
    \\    for (int mi = 0; mi < 4; ++mi)
    \\      #pragma clang loop unroll(full)
    \\      for (int ni = 0; ni < 4; ++ni) simdgroup_store(acc[mi][ni], sm + (mi * 8) * BN + sg_n * 32 + ni * 8, BN);
    \\  }
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  for (int r = int(tid) >> 4; r < 32; r += 8) {
    \\    const int mr = m0 + half_m * 32 + r;
    \\    if (mr < R) {
    \\      const size_t g_off = size_t(mr) * gs0 + gcol * gs1;
    \\      const size_t out_off = size_t(base_row + mr) * hv + gcol;
    \\      #pragma clang loop unroll(full)
    \\      for (int e = 0; e < 4; ++e) {
    \\        const float uv = sm[r * BN + c4 + e];
    \\        const float gv = static_cast<float>(gate[g_off + size_t(e) * gs1]);
    \\        out[out_off + e] = static_cast<OT>(uv * hy4_sigmoid(gv));
    \\      }
    \\    }
    \\  }
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\}
;

/// Fused wide-span `u_lat @ W_v^T` + `hy4_sigmoid(gate)` + bf16 round + `suh` Hadamard-128 written directly into
/// `out[row0 + m, h·VDIM + n]` as `half` (BM=64, BN=128, BK=16, WM=2, WN=4, 256 threads): double-buffered `Bs`
/// (1 barrier / BK=16, 20 KB threadgroup memory) with register-prefetched BF16 loads, halving TG count and `kvb`
/// traffic vs BM=32, and a 2-pass 32×128 (16 KB) parallel 8-SG Hadamard-128 epilogue.
pub const v_sg_prepared_into_source: [:0]const u8 =
    \\#pragma clang fp contract(off)
    \\using B4 = metal::vec<bfloat16_t, 4>;
    \\using F4 = metal::vec<float, 4>;
    \\constexpr int BM = 64, BN = 128, BK = 16;
    \\constexpr int A_STR = BK + 4;
    \\constexpr int B_STR = BK + 4;
    \\constexpr int BS_SZ = BN * B_STR;
    \\const uint tid = thread_index_in_threadgroup, sg = simdgroup_index_in_threadgroup;
    \\const ushort lane = thread_index_in_simdgroup;
    \\const int R = u_shape[1];
    \\const int h = int(threadgroup_position_in_grid.z);
    \\const int m0 = int(threadgroup_position_in_grid.y) * BM;
    \\const int n0 = int(threadgroup_position_in_grid.x) * BN;
    \\const int sg_m = int(sg) >> 2, sg_n = int(sg) & 3;
    \\threadgroup float sm[2 * BS_SZ];
    \\simdgroup_float8x8 acc[4][4];
    \\#pragma clang loop unroll(full)
    \\for (int mi = 0; mi < 4; ++mi)
    \\  #pragma clang loop unroll(full)
    \\  for (int ni = 0; ni < 4; ++ni) acc[mi][ni] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    \\const int bn = int(tid) >> 1;
    \\const int bk = (int(tid) & 1) * 8;
    \\const size_t us0 = size_t(u_strides[0]), us1 = size_t(u_strides[1]);
    \\const device bfloat16_t* wv_row = kvb + (size_t(h) * size_t(ROWS) + size_t(NOPE) + size_t(n0 + bn)) * size_t(LAT);
    \\if (m0 + BM <= R && u_strides[2] == 1) {
    \\  const device float* u_sg = u + size_t(h) * us0 + size_t(m0 + sg_m * 32) * us1;
    \\  {
    \\    const device B4* bp = (const device B4*)(wv_row + size_t(bk));
    \\    const B4 v0 = bp[0], v1 = bp[1];
    \\    threadgroup F4* bdst = (threadgroup F4*)(sm + bn * B_STR + bk);
    \\    bdst[0] = F4(float(v0[0]), float(v0[1]), float(v0[2]), float(v0[3]));
    \\    bdst[1] = F4(float(v1[0]), float(v1[1]), float(v1[2]), float(v1[3]));
    \\  }
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  int buf = 0;
    \\  for (int k0 = 0; k0 < LAT; k0 += BK, buf ^= 1) {
    \\    threadgroup float* Bs_cur = sm + buf * BS_SZ;
    \\    B4 nv0 = B4(0), nv1 = B4(0);
    \\    if (k0 + BK < LAT) {
    \\      const device B4* bp = (const device B4*)(wv_row + size_t(k0 + BK + bk));
    \\      nv0 = bp[0];
    \\      nv1 = bp[1];
    \\    }
    \\    #pragma clang loop unroll(full)
    \\    for (int kk = 0; kk < BK; kk += 8) {
    \\      simdgroup_float8x8 a[4], b[4];
    \\      #pragma clang loop unroll(full)
    \\      for (int mi = 0; mi < 4; ++mi) simdgroup_load(a[mi], u_sg + size_t(mi * 8) * us1 + size_t(k0 + kk), us1);
    \\      #pragma clang loop unroll(full)
    \\      for (int ni = 0; ni < 4; ++ni) simdgroup_load(b[ni], Bs_cur + (sg_n * 32 + ni * 8) * B_STR + kk, B_STR, ulong2(0, 0), true);
    \\      #pragma clang loop unroll(full)
    \\      for (int mi = 0; mi < 4; ++mi)
    \\        #pragma clang loop unroll(full)
    \\        for (int ni = 0; ni < 4; ++ni) simdgroup_multiply_accumulate(acc[mi][ni], a[mi], b[ni], acc[mi][ni]);
    \\    }
    \\    if (k0 + BK < LAT) {
    \\      threadgroup F4* bdst = (threadgroup F4*)(sm + (buf ^ 1) * BS_SZ + bn * B_STR + bk);
    \\      bdst[0] = F4(float(nv0[0]), float(nv0[1]), float(nv0[2]), float(nv0[3]));
    \\      bdst[1] = F4(float(nv1[0]), float(nv1[1]), float(nv1[2]), float(nv1[3]));
    \\      threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\    }
    \\  }
    \\} else {
    \\  threadgroup float* As = sm;
    \\  threadgroup float* Bs = sm + BM * A_STR;
    \\  const int am = int(tid) >> 2;
    \\  const int ak = (int(tid) & 3) * 4;
    \\  const size_t u_row_base = size_t(h) * us0 + size_t(m0 + am) * us1;
    \\  const size_t us2 = size_t(u_strides[2]);
    \\  const bool sg_active = (m0 + sg_m * 32) < R;
    \\  const bool hi_active = (m0 + sg_m * 32 + 16) < R;
    \\  for (int k0 = 0; k0 < LAT; k0 += BK) {
    \\    threadgroup F4* adst = (threadgroup F4*)(As + am * A_STR + ak);
    \\    if (m0 + am < R) {
    \\      if (us2 == 1) {
    \\        *adst = *(const device F4*)(u + u_row_base + size_t(k0 + ak));
    \\      } else {
    \\        const device float* up = u + u_row_base;
    \\        *adst = F4(up[size_t(k0 + ak + 0) * us2], up[size_t(k0 + ak + 1) * us2], up[size_t(k0 + ak + 2) * us2], up[size_t(k0 + ak + 3) * us2]);
    \\      }
    \\    } else {
    \\      *adst = F4(0.0f);
    \\    }
    \\    const device B4* bp = (const device B4*)(wv_row + size_t(k0 + bk));
    \\    const B4 v0 = bp[0], v1 = bp[1];
    \\    threadgroup F4* bdst = (threadgroup F4*)(Bs + bn * B_STR + bk);
    \\    bdst[0] = F4(float(v0[0]), float(v0[1]), float(v0[2]), float(v0[3]));
    \\    bdst[1] = F4(float(v1[0]), float(v1[1]), float(v1[2]), float(v1[3]));
    \\    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\    if (sg_active) {
    \\      #pragma clang loop unroll(full)
    \\      for (int kk = 0; kk < BK; kk += 8) {
    \\        simdgroup_float8x8 a[4], b[4];
    \\        #pragma clang loop unroll(full)
    \\        for (int ni = 0; ni < 4; ++ni) simdgroup_load(b[ni], Bs + (sg_n * 32 + ni * 8) * B_STR + kk, B_STR, ulong2(0, 0), true);
    \\        #pragma clang loop unroll(full)
    \\        for (int mi = 0; mi < 2; ++mi) simdgroup_load(a[mi], As + (sg_m * 32 + mi * 8) * A_STR + kk, A_STR);
    \\        #pragma clang loop unroll(full)
    \\        for (int mi = 0; mi < 2; ++mi)
    \\          #pragma clang loop unroll(full)
    \\          for (int ni = 0; ni < 4; ++ni) simdgroup_multiply_accumulate(acc[mi][ni], a[mi], b[ni], acc[mi][ni]);
    \\        if (hi_active) {
    \\          #pragma clang loop unroll(full)
    \\          for (int mi = 2; mi < 4; ++mi) simdgroup_load(a[mi], As + (sg_m * 32 + mi * 8) * A_STR + kk, A_STR);
    \\          #pragma clang loop unroll(full)
    \\          for (int mi = 2; mi < 4; ++mi)
    \\            #pragma clang loop unroll(full)
    \\            for (int ni = 0; ni < 4; ++ni) simdgroup_multiply_accumulate(acc[mi][ni], a[mi], b[ni], acc[mi][ni]);
    \\        }
    \\      }
    \\    }
    \\    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  }
    \\}
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\const int base_row = row0[0];
    \\const size_t hv = size_t(HEADS) * size_t(VDIM);
    \\const size_t base_col = size_t(h) * size_t(VDIM) + size_t(n0);
    \\const size_t gs0 = size_t(gate_strides[0]), gs1 = size_t(gate_strides[1]);
    \\const float s_0 = float(suh[base_col + lane]);
    \\const float s_1 = float(suh[base_col + lane + 32u]);
    \\const float s_2 = float(suh[base_col + lane + 64u]);
    \\const float s_3 = float(suh[base_col + lane + 96u]);
    \\const float sc = 0.08838834764831845f;
    \\#pragma clang loop unroll(full)
    \\for (int half_m = 0; half_m < 2; ++half_m) {
    \\  if (m0 + half_m * 32 < R) {
    \\    if (sg_m == half_m) {
    \\      #pragma clang loop unroll(full)
    \\      for (int mi = 0; mi < 4; ++mi)
    \\        #pragma clang loop unroll(full)
    \\        for (int ni = 0; ni < 4; ++ni) simdgroup_store(acc[mi][ni], sm + (mi * 8) * BN + sg_n * 32 + ni * 8, BN);
    \\    }
    \\    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\    for (int ri = 0; ri < 4; ++ri) {
    \\      const int r = int(sg) * 4 + ri;
    \\      const int mr = m0 + half_m * 32 + r;
    \\      if (mr >= R) break;
    \\      const size_t g_row = size_t(mr) * gs0;
    \\      const float g0 = static_cast<float>(gate[g_row + (base_col + lane) * gs1]);
    \\      const float g1 = static_cast<float>(gate[g_row + (base_col + lane + 32u) * gs1]);
    \\      const float g2 = static_cast<float>(gate[g_row + (base_col + lane + 64u) * gs1]);
    \\      const float g3 = static_cast<float>(gate[g_row + (base_col + lane + 96u) * gs1]);
    \\      float4 v = float4(
    \\        float(bfloat16_t(sm[r * BN + lane] * hy4_sigmoid(g0))) * s_0,
    \\        float(bfloat16_t(sm[r * BN + lane + 32u] * hy4_sigmoid(g1))) * s_1,
    \\        float(bfloat16_t(sm[r * BN + lane + 64u] * hy4_sigmoid(g2))) * s_2,
    \\        float(bfloat16_t(sm[r * BN + lane + 96u] * hy4_sigmoid(g3))) * s_3);
    \\      for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    \\        const float p0 = simd_shuffle_xor(v.x, bit);
    \\        const float p1 = simd_shuffle_xor(v.y, bit);
    \\        const float p2 = simd_shuffle_xor(v.z, bit);
    \\        const float p3 = simd_shuffle_xor(v.w, bit);
    \\        const bool lower = (lane & bit) == 0u;
    \\        v.x = lower ? v.x + p0 : p0 - v.x;
    \\        v.y = lower ? v.y + p1 : p1 - v.y;
    \\        v.z = lower ? v.z + p2 : p2 - v.z;
    \\        v.w = lower ? v.w + p3 : p3 - v.w;
    \\      }
    \\      const float s0 = v.x + v.y;
    \\      const float s1 = v.x - v.y;
    \\      const float s2 = v.z + v.w;
    \\      const float s3 = v.z - v.w;
    \\      const size_t yb = size_t(base_row + mr) * hv + base_col;
    \\      out[yb + lane] = half((s0 + s2) * sc);
    \\      out[yb + lane + 32u] = half((s1 + s3) * sc);
    \\      out[yb + lane + 64u] = half((s0 - s2) * sc);
    \\      out[yb + lane + 96u] = half((s1 - s3) * sc);
    \\    }
    \\    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  }
    \\}
;

var q_kernel: ?mlx.mlx_fast_metal_kernel = null;
var v_kernel: ?mlx.mlx_fast_metal_kernel = null;
var v_gated_kernel: ?mlx.mlx_fast_metal_kernel = null;
var q_sg_kernel: ?mlx.mlx_fast_metal_kernel = null;
var v_sg_kernel: ?mlx.mlx_fast_metal_kernel = null;
var v_sg_gated_kernel: ?mlx.mlx_fast_metal_kernel = null;
var v_sg_prepared_into_kernel: ?mlx.mlx_fast_metal_kernel = null;

fn getKernel(slot: *?mlx.mlx_fast_metal_kernel, name: [*:0]const u8, inputs: []const [*:0]const u8, source: [:0]const u8, header: [:0]const u8, ensure_row_contiguous: bool) !mlx.mlx_fast_metal_kernel {
    if (slot.*) |k| return k;
    const output_names = [_][*:0]const u8{"out"};
    const in_vec = mlx.mlx_vector_string_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, source, header, ensure_row_contiguous, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    slot.* = k;
    return k;
}

/// The geometry of one layer's `kv_b` [heads · (nope + v), latent].
pub const Geometry = struct { heads: c_int, nope: c_int, v: c_int, latent: c_int };
pub const max_rows: c_int = 8;
pub const joined_max_rows: c_int = 16;

fn appliesCap(s: mlx.mlx_stream, kv_b: Arr, g: Geometry, rows: c_int, cap: c_int) bool {
    return rows >= 1 and rows <= cap and mlx.streamIsGpu(s) and mlx.mlx_array_dtype(kv_b) == .bfloat16 and
        g.nope < 8192 and @rem(g.nope, 32) == 0 and @rem(g.latent, gemvTSimdgroups(g.latent) * 16) == 0 and
        g.latent > 64 and @rem(g.latent, 128) == 0 and g.v < 4096 and @rem(g.v, 16) == 0 and g.latent < 16 * g.v;
}

/// Whether the products run here: one to `max_rows` rows, a BF16 `kv_b`, a GPU stream, and shapes on the tiles MLX picks for them
/// (`gemv_t` with 8 x 4 lanes, under 8192 inputs; `gemv` with 4 x 4 rows a simdgroup, more than 64 inputs and under 4096
/// rows, fewer inputs than 16 times its rows).
pub fn applies(s: mlx.mlx_stream, kv_b: Arr, g: Geometry, rows: c_int) bool {
    return appliesCap(s, kv_b, g, rows, max_rows);
}

/// Whether a joined/grouped tick of `rows` rows (up to `joined_max_rows`, tiled in `max_rows`-row chunks) runs here.
pub fn appliesJoined(s: mlx.mlx_stream, kv_b: Arr, g: Geometry, rows: c_int) bool {
    return appliesCap(s, kv_b, g, rows, joined_max_rows);
}

/// The simdgroups a `gemv_t` threadgroup spans for `out` outputs (MLX's `bn`).
fn gemvTSimdgroups(out: c_int) c_int {
    return if (out >= 2048) 16 else if (out >= 512) 4 else 2;
}

const Key = struct { g: Geometry, rows: c_int, qs: c_int, out_cols: c_int, groups_x: c_int, simdgroups: c_int, gated: bool = false, row_major: bool = false };
var cfgs: kcfg.Cache(Key) = .{};

fn build(k: Key) !kcfg.Config {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const g = k.g;
    if (k.gated) {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ k.rows, g.heads * k.out_cols }, 2, .float32));
    } else if (k.row_major) {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ k.rows, g.heads, k.out_cols }, 3, .float32));
    } else {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ g.heads, k.rows, k.out_cols }, 3, .float32));
    }
    const rt: c_int = @min(k.rows, max_rows);
    const r_tiles: c_int = @divTrunc(k.rows + rt - 1, rt);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32 * k.groups_x, k.simdgroups, g.heads * r_tiles));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, k.simdgroups, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NSG", k.simdgroups));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "TW", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "HEADS", g.heads));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ROWS", g.nope + g.v));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NOPE", g.nope));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "VDIM", g.v));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "LAT", g.latent));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "QS", k.qs));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "R", k.rows));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "RT", rt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "ROW_MAJOR", k.row_major));
    return cfg;
}

fn run(s: mlx.mlx_stream, kernel: mlx.mlx_fast_metal_kernel, inputs: []const Arr, g: Geometry, out_cols: c_int, groups_x: c_int, simdgroups: c_int, gated: bool, row_major: bool) !Arr {
    // The vector's [H, T, width] (a plain row [1, K] is one head's one row).
    const sh = mlx.getShape(inputs[1]);
    const rows: c_int = if (sh.len == 3) sh[1] else 1;
    if (rows < 1) return error.Hy4AbsorbRows;
    const cfg = try cfgs.get(.{ .g = g, .rows = rows, .qs = sh[sh.len - 1], .out_cols = out_cols, .groups_x = groups_x, .simdgroups = simdgroups, .gated = gated, .row_major = row_major }, build);
    const in_vec = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var outputs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs, kernel, in_vec, cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs, 0));
    return out;
}

/// The query's first `nope` dims of each of `q`'s [H, T, ≥ nope] f32 rows (strided along H/T, contiguous along last
/// axis; a rope part past them is not read) into the latent through `kv_b`'s key rows: [H, T, latent] f32.
pub fn queryLatent(s: mlx.mlx_stream, q_nope: Arr, kv_b: Arr, g: Geometry) !Arr {
    const k = try getKernel(&q_kernel, "sushi_hy4_absorb_q", &.{ "kvb", "q" }, q_source, "", false);
    const bn = gemvTSimdgroups(g.latent);
    return run(s, k, &.{ kv_b, q_nope }, g, g.latent, @divExact(g.latent, bn * 16), bn, false, false);
}

/// `queryLatent` laid out row-major [T, H, latent] f32.
pub fn queryLatentRows(s: mlx.mlx_stream, q_nope: Arr, kv_b: Arr, g: Geometry) !Arr {
    const k = try getKernel(&q_kernel, "sushi_hy4_absorb_q", &.{ "kvb", "q" }, q_source, "", false);
    const bn = gemvTSimdgroups(g.latent);
    return run(s, k, &.{ kv_b, q_nope }, g, g.latent, @divExact(g.latent, bn * 16), bn, false, true);
}

/// `u_lat` [H, T, latent] f32 out through `kv_b`'s value rows: [H, T, v] f32.
pub fn valueOut(s: mlx.mlx_stream, u_lat: Arr, kv_b: Arr, g: Geometry) !Arr {
    const k = try getKernel(&v_kernel, "sushi_hy4_absorb_v", &.{ "kvb", "u" }, v_source, "", false);
    return run(s, k, &.{ kv_b, u_lat }, g, g.v, @divExact(g.v, 16), 4, false, false);
}

fn valueOutGatedAppliesCap(s: mlx.mlx_stream, u_lat: Arr, kv_b: Arr, g: Geometry, gate: Arr, cap: c_int) bool {
    const us = mlx.getShape(u_lat);
    const gs = mlx.getShape(gate);
    const gd = mlx.mlx_array_dtype(gate);
    return us.len == 3 and us[0] == g.heads and us[2] == g.latent and appliesCap(s, kv_b, g, us[1], cap) and
        gs.len == 2 and gs[0] == us[1] and gs[1] == g.heads * g.v and (gd == .bfloat16 or gd == .float32);
}

/// Whether `valueOutGated` takes 1..`max_rows` rows: `valueOut`'s shapes and a BF16 or f32 gate as wide as the heads' values.
pub fn valueOutGatedApplies(s: mlx.mlx_stream, u_lat: Arr, kv_b: Arr, g: Geometry, gate: Arr) bool {
    return valueOutGatedAppliesCap(s, u_lat, kv_b, g, gate, max_rows);
}

/// Whether `valueOutGated` takes 1..`joined_max_rows` rows for a joined/grouped tick.
pub fn valueOutGatedJoinedApplies(s: mlx.mlx_stream, u_lat: Arr, kv_b: Arr, g: Geometry, gate: Arr) bool {
    return valueOutGatedAppliesCap(s, u_lat, kv_b, g, gate, joined_max_rows);
}

/// `valueOut` with the attention output gate in the same launch: each value times `σ(gate)`, written [T, H·v] as `W_o`
/// reads it, the bytes of `valueOut` then `hy4_chain.gated`.
pub fn valueOutGated(s: mlx.mlx_stream, u_lat: Arr, kv_b: Arr, g: Geometry, gate: Arr) !Arr {
    const k = try getKernel(&v_gated_kernel, "sushi_hy4_absorb_v_gated", &.{ "kvb", "u", "gate" }, v_source_gated, sigmoid_header, false);
    return run(s, k, &.{ kv_b, u_lat, gate }, g, g.v, @divExact(g.v, 16), 4, true, false);
}

/// Whether the FP32 `simdgroup_float8x8` wide-span absorb kernels serve a span of `rows` rows: BF16 `kv_b`,
/// wider than the decode/verify GEMV cap, and dimensions aligned to the BM=64/BN=64/BK=16 tiles.
pub fn wideSgApplies(s: mlx.mlx_stream, kv_b: Arr, g: Geometry, rows: c_int) bool {
    return rows > max_rows and mlx.streamIsGpu(s) and mlx.mlx_array_dtype(kv_b) == .bfloat16 and
        g.heads > 0 and g.nope > 0 and @rem(g.nope, 16) == 0 and
        g.latent > 0 and @rem(g.latent, 64) == 0 and
        g.v > 0 and @rem(g.v, 64) == 0;
}

/// Whether `valueOutWideSgGated` / `valueOutWideSgGatedInto` serves `u_lat`, `kv_b`, `gate`, and optional `dst`.
pub fn wideSgGatedApplies(s: mlx.mlx_stream, u_lat: Arr, kv_b: Arr, g: Geometry, gate: Arr, dst: ?Arr) bool {
    const us = mlx.getShape(u_lat);
    const gs = mlx.getShape(gate);
    const gd = mlx.mlx_array_dtype(gate);
    if (us.len != 3 or us[0] != g.heads or us[2] != g.latent or !wideSgApplies(s, kv_b, g, us[1])) return false;
    if (mlx.mlx_array_dtype(u_lat) != .float32) return false;
    if (gs.len != 2 or gs[0] != us[1] or gs[1] != g.heads * g.v or (gd != .bfloat16 and gd != .float32)) return false;
    if (dst) |d| {
        if (!mlx.engine.isStream(s)) return false;
        const ds = mlx.getShape(d);
        const dd = mlx.mlx_array_dtype(d);
        return ds.len == 2 and ds[1] == gs[1] and (dd == .bfloat16 or dd == .float32);
    }
    return true;
}

/// Whether `valueOutWideSgPreparedInto` serves `u_lat`, `kv_b`, `gate`, `suh`, and `dst` (f16, 128-wide Hadamard).
pub fn wideSgPreparedIntoApplies(s: mlx.mlx_stream, u_lat: Arr, kv_b: Arr, g: Geometry, gate: Arr, suh: Arr, dst: Arr) bool {
    if (!mlx.engine.isStream(s) or @rem(g.v, 128) != 0) return false;
    const us = mlx.getShape(u_lat);
    const gs = mlx.getShape(gate);
    const gd = mlx.mlx_array_dtype(gate);
    if (us.len != 3 or us[0] != g.heads or us[2] != g.latent or !wideSgApplies(s, kv_b, g, us[1])) return false;
    if (mlx.mlx_array_dtype(u_lat) != .float32) return false;
    if (gs.len != 2 or gs[0] != us[1] or gs[1] != g.heads * g.v or (gd != .bfloat16 and gd != .float32)) return false;
    const ds = mlx.getShape(dst);
    return ds.len == 2 and ds[1] == gs[1] and mlx.mlx_array_dtype(dst) == .float16 and
        mlx.mlx_array_dtype(suh) == .float16 and mlx.mlx_array_size(suh) == @as(usize, @intCast(gs[1]));
}

const SgKind = enum { q, v, v_gated, v_prepared_into };
const SgKey = struct { kind: SgKind, g: Geometry, rows: c_int, dst_rows: c_int = 0, out_dt: c_int = 0 };
var sg_cfgs: kcfg.Cache(SgKey) = .{};

fn buildSg(k: SgKey) !kcfg.Config {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const g = k.g;
    switch (k.kind) {
        .q => {
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ g.heads, k.rows, g.latent }, 3, .float32));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divExact(g.latent, 64) * 128, @divTrunc(k.rows + 63, 64), g.heads));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 128, 1, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ROWS", g.nope + g.v));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NOPE", g.nope));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "LAT", g.latent));
        },
        .v => {
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ g.heads, k.rows, g.v }, 3, .float32));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divExact(g.v, 64) * 128, @divTrunc(k.rows + 63, 64), g.heads));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 128, 1, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ROWS", g.nope + g.v));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NOPE", g.nope));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "VDIM", g.v));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "LAT", g.latent));
        },
        .v_gated => {
            const out_dt: mlx.mlx_dtype = @enumFromInt(k.out_dt);
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ k.dst_rows, g.heads * g.v }, 2, out_dt));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divExact(g.v, 64) * 128, @divTrunc(k.rows + 63, 64), g.heads));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 128, 1, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "OT", out_dt));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "HEADS", g.heads));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ROWS", g.nope + g.v));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NOPE", g.nope));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "VDIM", g.v));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "LAT", g.latent));
        },
        .v_prepared_into => {
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ k.dst_rows, g.heads * g.v }, 2, .float16));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divExact(g.v, 128) * 256, @divTrunc(k.rows + 63, 64), g.heads));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "HEADS", g.heads));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ROWS", g.nope + g.v));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NOPE", g.nope));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "VDIM", g.v));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "LAT", g.latent));
        },
    }
    return cfg;
}

/// Wide-span `q_nope · W_k` on FP32 `simdgroup_float8x8` (`[H, T, ≥ nope]` f32 × BF16 `kv_b` → `[H, T, latent]` f32).
pub fn queryLatentWideSg(s: mlx.mlx_stream, q_nope: Arr, kv_b: Arr, g: Geometry) !Arr {
    const k = try getKernel(&q_sg_kernel, "sushi_hy4_absorb_q_sg", &.{ "kvb", "q" }, q_sg_source, "", false);
    const rows: c_int = mlx.getShape(q_nope)[1];
    const cfg = try sg_cfgs.get(.{ .kind = .q, .g = g, .rows = rows }, buildSg);
    const ins = [_]Arr{ kv_b, q_nope };
    const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, k, iv, cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, ov, 0));
    return out;
}

/// Wide-span `u_lat · W_vᵀ` on FP32 `simdgroup_float8x8` (`[H, T, latent]` f32 × BF16 `kv_b` → `[H, T, v]` f32).
pub fn valueOutWideSg(s: mlx.mlx_stream, u_lat: Arr, kv_b: Arr, g: Geometry) !Arr {
    const k = try getKernel(&v_sg_kernel, "sushi_hy4_absorb_v_sg", &.{ "kvb", "u" }, v_sg_source, "", false);
    const rows: c_int = mlx.getShape(u_lat)[1];
    const cfg = try sg_cfgs.get(.{ .kind = .v, .g = g, .rows = rows }, buildSg);
    const ins = [_]Arr{ kv_b, u_lat };
    const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, k, iv, cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, ov, 0));
    return out;
}

/// Fused wide-span `u_lat · W_vᵀ ⊙ σ(gate)` → fresh `[T, H·v]` f32.
pub fn valueOutWideSgGated(s: mlx.mlx_stream, u_lat: Arr, kv_b: Arr, g: Geometry, gate: Arr) !Arr {
    const k = try getKernel(&v_sg_gated_kernel, "sushi_hy4_absorb_v_sg_gated", &.{ "kvb", "u", "gate", "row0" }, v_sg_gated_source, sigmoid_header, false);
    const rows: c_int = mlx.getShape(u_lat)[1];
    const cfg = try sg_cfgs.get(.{ .kind = .v_gated, .g = g, .rows = rows, .dst_rows = rows, .out_dt = @intFromEnum(mlx.mlx_dtype.float32) }, buildSg);
    const rv = [_]i32{0};
    const row0 = try mlx.newData(&rv, &.{1}, .int32);
    defer _ = mlx.mlx_array_free(row0);
    const ins = [_]Arr{ kv_b, u_lat, gate, row0 };
    const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, k, iv, cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, ov, 0));
    return out;
}

/// Fused wide-span `u_lat · W_vᵀ ⊙ σ(gate)` written directly into `dst[r0..r0+T, H·v]` in `dst`'s dtype (bf16/f32).
pub fn valueOutWideSgGatedInto(s: mlx.mlx_stream, u_lat: Arr, kv_b: Arr, g: Geometry, gate: Arr, dst: Arr, r0: c_int) !void {
    const k = try getKernel(&v_sg_gated_kernel, "sushi_hy4_absorb_v_sg_gated", &.{ "kvb", "u", "gate", "row0" }, v_sg_gated_source, sigmoid_header, false);
    const rows: c_int = mlx.getShape(u_lat)[1];
    const ds = mlx.getShape(dst);
    const cfg = try sg_cfgs.get(.{ .kind = .v_gated, .g = g, .rows = rows, .dst_rows = ds[0], .out_dt = @intFromEnum(mlx.mlx_array_dtype(dst)) }, buildSg);
    const rv = [_]i32{r0};
    const row0 = try mlx.newData(&rv, &.{1}, .int32);
    defer _ = mlx.mlx_array_free(row0);
    const ins = [_]Arr{ kv_b, u_lat, gate, row0 };
    const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.kernelApplyInto(&ov, k, iv, cfg, s, dst) orelse return error.MlxError);
}

/// Fused wide-span `u_lat · W_vᵀ ⊙ σ(gate)` + bf16 round + `suh` Hadamard-128 written directly into f16 `dst[r0..r0+T, H·v]`.
pub fn valueOutWideSgPreparedInto(s: mlx.mlx_stream, u_lat: Arr, kv_b: Arr, g: Geometry, gate: Arr, suh: Arr, dst: Arr, r0: c_int) !void {
    const k = try getKernel(&v_sg_prepared_into_kernel, "sushi_hy4_absorb_v_sg_prepared_into", &.{ "kvb", "u", "gate", "suh", "row0" }, v_sg_prepared_into_source, sigmoid_header, false);
    const rows: c_int = mlx.getShape(u_lat)[1];
    const ds = mlx.getShape(dst);
    const cfg = try sg_cfgs.get(.{ .kind = .v_prepared_into, .g = g, .rows = rows, .dst_rows = ds[0] }, buildSg);
    const rv = [_]i32{r0};
    const row0 = try mlx.newData(&rv, &.{1}, .int32);
    defer _ = mlx.mlx_array_free(row0);
    const ins = [_]Arr{ kv_b, u_lat, gate, suh, row0 };
    const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.kernelApplyInto(&ov, k, iv, cfg, s, dst) orelse return error.MlxError);
}

// ── tests ────────────────────────────────────────────────────────────────────────────────────────

const testing = std.testing;
const Ops = @import("../mlx_ops.zig").Ops;

fn randomBf16(gpa: std.mem.Allocator, rnd: std.Random, n: usize) ![]u16 {
    const out = try gpa.alloc(u16, n);
    for (out) |*v| v.* = @intCast(@as(u32, @bitCast(@as(f32, @floatCast(rnd.floatNorm(f64) * 0.05)))) >> 16);
    return out;
}

fn randomF32(gpa: std.mem.Allocator, rnd: std.Random, n: usize) ![]f32 {
    const out = try gpa.alloc(f32, n);
    for (out) |*v| v.* = @floatCast(rnd.floatNorm(f64));
    return out;
}

fn expectSameBytes(got: Arr, want: Arr) !void {
    try mlx.check(mlx.mlx_array_eval(got));
    try mlx.check(mlx.mlx_array_eval(want));
    try testing.expectEqual(mlx.mlx_array_dtype(want), mlx.mlx_array_dtype(got));
    try testing.expectEqualSlices(c_int, mlx.getShape(want), mlx.getShape(got));
    const n = mlx.mlx_array_size(want);
    switch (mlx.mlx_array_dtype(want)) {
        .float32 => {
            const a = (mlx.mlx_array_data_float32(want) orelse return error.MlxArrayDataNull)[0..n];
            const b = (mlx.mlx_array_data_float32(got) orelse return error.MlxArrayDataNull)[0..n];
            try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(a), std.mem.sliceAsBytes(b));
        },
        .bfloat16, .float16 => {
            const a = (mlx.mlx_array_data_uint16(want) orelse return error.MlxArrayDataNull)[0..n];
            const b = (mlx.mlx_array_data_uint16(got) orelse return error.MlxArrayDataNull)[0..n];
            try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(a), std.mem.sliceAsBytes(b));
        },
        else => return error.UnsupportedDtype,
    }
}

/// Both products on random bytes, against the forward's composed f32 products over the widened `kv_b`, byte for byte.
fn parityCase(s: mlx.mlx_stream, g: Geometry, seed: u64) !void {
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    const rows: usize = @intCast(g.heads * (g.nope + g.v));
    const lat: usize = @intCast(g.latent);
    const h: usize = @intCast(g.heads);
    const w_bits = try randomBf16(gpa, rnd, rows * lat);
    defer gpa.free(w_bits);
    const q_host = try randomF32(gpa, rnd, h * @as(usize, @intCast(g.nope + 64)));
    defer gpa.free(q_host);
    const u_host = try randomF32(gpa, rnd, h * lat);
    defer gpa.free(u_host);
    var o: Ops = .{ .s = s };
    defer o.deinit();
    const kv_b = try o.own(mlx.mlx_array_new_data(w_bits.ptr, &[_]c_int{ @intCast(rows), g.latent }, 2, .bfloat16));
    // The query as the forward holds it: [H, 1, nope + rope], its nope part a view.
    const q = try o.own(mlx.mlx_array_new_data(q_host.ptr, &[_]c_int{ g.heads, 1, g.nope + 64 }, 3, .float32));
    const u = try o.own(mlx.mlx_array_new_data(u_host.ptr, &[_]c_int{ g.heads, 1, g.latent }, 3, .float32));
    const kvb = try o.reshape(try o.cast(kv_b, .float32), &.{ g.heads, g.nope + g.v, g.latent });
    const q_nope = try o.slice(q, 2, 0, g.nope);
    const want_q = try o.binary(.mm, q_nope, try o.slice(kvb, 1, 0, g.nope));
    const want_v = try o.binary(.mm, u, try o.transpose(try o.slice(kvb, 1, g.nope, g.nope + g.v), &.{ 0, 2, 1 }));
    try expectSameBytes(try o.own(try queryLatent(s, try o.contiguous(q_nope), kv_b, g)), want_q);
    try expectSameBytes(try o.own(try valueOut(s, u, kv_b, g)), want_v);
}


test "hy4 absorb: a decode row's query and value products over kv_b as stored are the f32 products' bytes, at the release's sliding and sparse heads and at the tiny's" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const exact = mlx.engine.exactF32();
    defer exact.end();
    try parityCase(es, .{ .heads = 64, .nope = 192, .v = 256, .latent = 512 }, 1);
    try parityCase(es, .{ .heads = 32, .nope = 192, .v = 256, .latent = 512 }, 2);
    try parityCase(es, .{ .heads = 8, .nope = 32, .v = 32, .latent = 128 }, 3);
}

fn blockCase(s: mlx.mlx_stream, g: Geometry, t: c_int, seed: u64) !void {
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    const qk = g.nope + 64;
    const w_bits = try randomBf16(gpa, rnd, @intCast(g.heads * (g.nope + g.v) * g.latent));
    defer gpa.free(w_bits);
    const q_host = try randomF32(gpa, rnd, @intCast(g.heads * t * qk));
    defer gpa.free(q_host);
    const u_host = try randomF32(gpa, rnd, @intCast(g.heads * t * g.latent));
    defer gpa.free(u_host);
    var o: Ops = .{ .s = s };
    defer o.deinit();
    const kv_b = try o.own(mlx.mlx_array_new_data(w_bits.ptr, &[_]c_int{ g.heads * (g.nope + g.v), g.latent }, 2, .bfloat16));
    const q = try o.own(mlx.mlx_array_new_data(q_host.ptr, &[_]c_int{ g.heads, t, qk }, 3, .float32));
    const u = try o.own(mlx.mlx_array_new_data(u_host.ptr, &[_]c_int{ g.heads, t, g.latent }, 3, .float32));
    try testing.expect(applies(s, kv_b, g, t));
    try mlx.check(mlx.mlx_array_eval(kv_b));
    try mlx.check(mlx.mlx_array_eval(q));
    try mlx.check(mlx.mlx_array_eval(u));
    const before = mlx.engine.stats().dispatches;
    const q_lat = try o.own(try queryLatent(s, q, kv_b, g));
    const v_out = try o.own(try valueOut(s, u, kv_b, g));
    try mlx.check(mlx.mlx_array_eval(q_lat));
    try mlx.check(mlx.mlx_array_eval(v_out));
    try testing.expectEqual(@as(u64, 2), mlx.engine.stats().dispatches - before);
    for (0..@intCast(t)) |r| {
        const lo: c_int = @intCast(r);
        const q_nope = try o.contiguous(try o.slice(try o.slice(q, 1, lo, lo + 1), 2, 0, g.nope));
        try expectSameBytes(try o.contiguous(try o.slice(q_lat, 1, lo, lo + 1)), try o.own(try queryLatent(s, q_nope, kv_b, g)));
        try expectSameBytes(try o.contiguous(try o.slice(v_out, 1, lo, lo + 1)), try o.own(try valueOut(s, try o.contiguous(try o.slice(u, 1, lo, lo + 1)), kv_b, g)));
    }
}

test "hy4 absorb: 2, 6 and 8 rows through queryLatent and valueOut in one launch equal each row alone through the one-row kernel, at H 64, 32 and the tiny's shapes" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const geometries = [_]Geometry{
        .{ .heads = 64, .nope = 192, .v = 256, .latent = 512 },
        .{ .heads = 32, .nope = 192, .v = 256, .latent = 512 },
        .{ .heads = 8, .nope = 32, .v = 32, .latent = 128 },
    };
    for ([_]c_int{ 2, 6, 8 }, 0..) |t, ti| {
        for (geometries, 0..) |g, gi| {
            try blockCase(es, g, t, 200 + ti * 8 + gi);
        }
    }
    var o: Ops = .{ .s = es };
    defer o.deinit();
    const g = geometries[0];
    const kv_b = try o.zeros(&.{ g.heads * (g.nope + g.v), g.latent }, .bfloat16);
    try testing.expect(!applies(es, kv_b, g, 9));
}

test "hy4 absorb: a decode row's value product gated in the same launch is the bytes of valueOut then the gate's launch, BF16 and f32 gate rows, at the release's 64 and 32 heads" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const exact = mlx.engine.exactF32();
    defer exact.end();
    const chain = @import("hy4_chain.zig");
    const gpa = testing.allocator;
    for ([_]c_int{ 64, 32 }, 0..) |heads, ci| {
        const g: Geometry = .{ .heads = heads, .nope = 192, .v = 256, .latent = 512 };
        var prng = std.Random.DefaultPrng.init(0x9a7e + ci);
        const rnd = prng.random();
        const rows: usize = @intCast(g.heads * (g.nope + g.v));
        const w_bits = try randomBf16(gpa, rnd, rows * 512);
        defer gpa.free(w_bits);
        const u_host = try randomF32(gpa, rnd, @as(usize, @intCast(heads)) * 512);
        defer gpa.free(u_host);
        const gate_host = try randomF32(gpa, rnd, @as(usize, @intCast(heads)) * 256);
        defer gpa.free(gate_host);
        for (gate_host[0..4], [_]f32{ 0, -0.0, 120, -120 }) |*d, v| d.* = v;
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const kv_b = try o.own(mlx.mlx_array_new_data(w_bits.ptr, &[_]c_int{ @intCast(rows), g.latent }, 2, .bfloat16));
        const u = try o.own(mlx.mlx_array_new_data(u_host.ptr, &[_]c_int{ g.heads, 1, g.latent }, 3, .float32));
        const gate32 = try o.own(mlx.mlx_array_new_data(gate_host.ptr, &[_]c_int{ 1, g.heads * g.v }, 2, .float32));
        for ([_]Arr{ gate32, try o.cast(gate32, .bfloat16) }) |gate| {
            try mlx.check(mlx.mlx_array_eval(gate));
            const value = try o.own(try valueOut(es, u, kv_b, g));
            const want = try o.own(try chain.gated(es, value, gate));
            try testing.expect(valueOutGatedApplies(es, u, kv_b, g, gate));
            const before = mlx.engine.stats().dispatches;
            const got = try o.own(try valueOutGated(es, u, kv_b, g, gate));
            try testing.expectEqual(@as(u64, 1), mlx.engine.stats().dispatches - before);
            try expectSameBytes(got, want);
        }
    }
}

test "hy4 absorb: multi-row queryLatentRows (strided transposed input) and multi-row valueOutGated match single-row launches byte for byte up to 16 rows" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const exact = mlx.engine.exactF32();
    defer exact.end();
    const chain = @import("hy4_chain.zig");
    const gpa = testing.allocator;
    const g: Geometry = .{ .heads = 64, .nope = 192, .v = 256, .latent = 512 };
    const qk = g.nope + 64;
    for ([_]c_int{ 4, 8, 12, 16 }, 0..) |t, ti| {
        var prng = std.Random.DefaultPrng.init(0x51a7 + ti);
        const rnd = prng.random();
        const w_bits = try randomBf16(gpa, rnd, @intCast(g.heads * (g.nope + g.v) * g.latent));
        defer gpa.free(w_bits);
        const q_rm = try randomF32(gpa, rnd, @intCast(t * g.heads * qk));
        defer gpa.free(q_rm);
        const u_rm = try randomF32(gpa, rnd, @intCast(t * g.heads * g.latent));
        defer gpa.free(u_rm);
        const g_host = try randomF32(gpa, rnd, @intCast(t * g.heads * g.v));
        defer gpa.free(g_host);
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const kv_b = try o.own(mlx.mlx_array_new_data(w_bits.ptr, &[_]c_int{ g.heads * (g.nope + g.v), g.latent }, 2, .bfloat16));
        const q_rows = try o.own(mlx.mlx_array_new_data(q_rm.ptr, &[_]c_int{ t, g.heads, qk }, 3, .float32));
        const q_t = try o.transpose(q_rows, &.{ 1, 0, 2 }); // strided [H, T, qk]
        const u_rows = try o.own(mlx.mlx_array_new_data(u_rm.ptr, &[_]c_int{ t, g.heads, g.latent }, 3, .float32));
        const u_t = try o.transpose(u_rows, &.{ 1, 0, 2 }); // strided [H, T, LAT]
        const gate_bf = try o.cast(try o.own(mlx.mlx_array_new_data(g_host.ptr, &[_]c_int{ t, g.heads * g.v }, 2, .float32)), .bfloat16);
        try testing.expect(appliesJoined(es, kv_b, g, t));
        try testing.expect(valueOutGatedJoinedApplies(es, u_t, kv_b, g, gate_bf));
        try mlx.check(mlx.mlx_array_eval(kv_b));
        try mlx.check(mlx.mlx_array_eval(q_rows));
        try mlx.check(mlx.mlx_array_eval(u_rows));
        try mlx.check(mlx.mlx_array_eval(gate_bf));
        const before = mlx.engine.stats().dispatches;
        const ql_got = try o.own(try queryLatentRows(es, q_t, kv_b, g));
        const vg_got = try o.own(try valueOutGated(es, u_t, kv_b, g, gate_bf));
        try mlx.check(mlx.mlx_array_eval(ql_got));
        try mlx.check(mlx.mlx_array_eval(vg_got));
        try testing.expectEqual(@as(u64, 2), mlx.engine.stats().dispatches - before);
        for (0..@intCast(t)) |r| {
            const ri: c_int = @intCast(r);
            const q_one = try o.contiguous(try o.slice(q_t, 1, ri, ri + 1));
            const ql_want = try o.transpose(try o.own(try queryLatent(es, q_one, kv_b, g)), &.{ 1, 0, 2 });
            try expectSameBytes(try o.contiguous(try o.slice(ql_got, 0, ri, ri + 1)), try o.contiguous(ql_want));
            const u_one = try o.contiguous(try o.slice(u_t, 1, ri, ri + 1));
            const g_one = try o.contiguous(try o.slice(gate_bf, 0, ri, ri + 1));
            const vg_want = try o.own(try chain.gated(es, try o.own(try valueOut(es, u_one, kv_b, g)), g_one));
            try expectSameBytes(try o.contiguous(try o.slice(vg_got, 0, ri, ri + 1)), vg_want);
        }
    }
}

test "hy4 absorb wide SG: queryLatentWideSg, valueOutWideSg, valueOutWideSgGated/Into/PreparedInto match steel f32 GEMM + hy4_chain byte for byte" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const exact = mlx.engine.exactF32();
    defer exact.end();
    const chain = @import("hy4_chain.zig");
    const gpa = testing.allocator;

    for ([_]struct { heads: c_int, t: c_int }{ .{ .heads = 64, .t = 12 }, .{ .heads = 64, .t = 61 }, .{ .heads = 32, .t = 128 } }, 0..) |case, ci| {
        const g: Geometry = .{ .heads = case.heads, .nope = 192, .v = 256, .latent = 512 };
        const qk = g.nope + 64;
        const hv = g.heads * g.v;
        var prng = std.Random.DefaultPrng.init(0x7a1e + ci);
        const rnd = prng.random();
        const w_bits = try randomBf16(gpa, rnd, @intCast(g.heads * (g.nope + g.v) * g.latent));
        defer gpa.free(w_bits);
        const q_host = try randomF32(gpa, rnd, @intCast(g.heads * (case.t + 32) * qk));
        defer gpa.free(q_host);
        const u_host = try randomF32(gpa, rnd, @intCast(g.heads * case.t * g.latent));
        defer gpa.free(u_host);
        const gate_host = try randomF32(gpa, rnd, @intCast(case.t * hv));
        defer gpa.free(gate_host);
        const suh_host = try randomF32(gpa, rnd, @intCast(hv));
        defer gpa.free(suh_host);

        var o: Ops = .{ .s = es };
        defer o.deinit();
        const kv_b = try o.own(mlx.mlx_array_new_data(w_bits.ptr, &[_]c_int{ g.heads * (g.nope + g.v), g.latent }, 2, .bfloat16));
        const q_full = try o.own(mlx.mlx_array_new_data(q_host.ptr, &[_]c_int{ g.heads, case.t + 32, qk }, 3, .float32));
        // Non-zero offset strided slice along axis 1 (mirrors prefill span 2).
        const q_slice = try o.slice(q_full, 1, 32, case.t + 32);
        const u_f32 = try o.own(mlx.mlx_array_new_data(u_host.ptr, &[_]c_int{ g.heads, case.t, g.latent }, 3, .float32));
        const gate_bf = try o.cast(try o.own(mlx.mlx_array_new_data(gate_host.ptr, &[_]c_int{ case.t, hv }, 2, .float32)), .bfloat16);
        const suh_f16 = try o.cast(try o.own(mlx.mlx_array_new_data(suh_host.ptr, &[_]c_int{hv}, 1, .float32)), .float16);

        try testing.expect(wideSgApplies(es, kv_b, g, case.t));
        try testing.expect(wideSgGatedApplies(es, u_f32, kv_b, g, gate_bf, null));

        const kvb = try o.reshape(try o.cast(kv_b, .float32), &.{ g.heads, g.nope + g.v, g.latent });
        const w_k = try o.slice(kvb, 1, 0, g.nope);
        const w_v = try o.slice(kvb, 1, g.nope, g.nope + g.v);

        const want_q = try o.binary(.mm, try o.slice(q_slice, 2, 0, g.nope), w_k);
        const got_q = try o.own(try queryLatentWideSg(es, q_slice, kv_b, g));
        try expectSameBytes(got_q, want_q);

        const want_v = try o.binary(.mm, u_f32, try o.transpose(w_v, &.{ 0, 2, 1 }));
        const got_v = try o.own(try valueOutWideSg(es, u_f32, kv_b, g));
        try expectSameBytes(got_v, want_v);

        const want_vg = try o.own(try chain.gated(es, want_v, gate_bf));
        const got_vg = try o.own(try valueOutWideSgGated(es, u_f32, kv_b, g, gate_bf));
        try expectSameBytes(got_vg, want_vg);

        // GatedInto (BF16) at non-zero row offset r0 = 17.
        const total_rows: c_int = case.t + 25;
        const r0: c_int = 17;
        const dst_want = try o.slot();
        try mlx.check(mlx.mlx_zeros(dst_want, &[_]c_int{ total_rows, hv }, 2, .bfloat16, es));
        const dst_got = try o.slot();
        try mlx.check(mlx.mlx_zeros(dst_got, &[_]c_int{ total_rows, hv }, 2, .bfloat16, es));
        try testing.expect(wideSgGatedApplies(es, u_f32, kv_b, g, gate_bf, dst_got.*));
        try chain.gatedInto(es, want_v, gate_bf, dst_want.*, r0);
        try valueOutWideSgGatedInto(es, u_f32, kv_b, g, gate_bf, dst_got.*, r0);
        try expectSameBytes(dst_got.*, dst_want.*);

        // PreparedInto (F16 Hadamard-128) at non-zero row offset r0 = 17.
        const prep_want = try o.slot();
        try mlx.check(mlx.mlx_zeros(prep_want, &[_]c_int{ total_rows, hv }, 2, .float16, es));
        const prep_got = try o.slot();
        try mlx.check(mlx.mlx_zeros(prep_got, &[_]c_int{ total_rows, hv }, 2, .float16, es));
        try testing.expect(wideSgPreparedIntoApplies(es, u_f32, kv_b, g, gate_bf, suh_f16, prep_got.*));
        try chain.gatedPreparedInto(es, want_v, gate_bf, suh_f16, prep_want.*, r0);
        try valueOutWideSgPreparedInto(es, u_f32, kv_b, g, gate_bf, suh_f16, prep_got.*, r0);
        try expectSameBytes(prep_got.*, prep_want.*);
    }
}
