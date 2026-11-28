//! A decode row's identity-comb mix and its collapse and norm in one launch: the row's mixes (`hc_kernels.mixTiled`'s
//! arithmetic: thread `tid` the terms `k = tid + c·TG` in order, one `simd_sum`, the simdgroups in order), then the
//! split, the collapse by `pre` and the rms norm (`hc_kernels.normIdentity`'s), the mixes kept in threadgroup memory.
//! With `expand`, the stream it reads is `post·y + h` (`hc_kernels.mixExpandIdentity`'s two roundings), stored as `m`.
//! With `back`, the `y` it expands by is a layer's combine norm, which the launch takes first (`hy4_norm.rowsExpand`'s),
//! so a layer's write-back and the next layer's mix are one launch. On the engine's stream a row's mixes are summed
//! over `2·HC` threadgroups that count into the engine's counter page (`wide_sums`); elsewhere one threadgroup a row.
//! Each output is the bytes of the launches it replaces.
const std = @import("std");
const mlx = @import("../mlx.zig");
const hck = @import("../hc_kernels.zig");
const kcfg = @import("hy4_kcfg.zig");
const exl3 = @import("sushi_exl3");
const log = @import("../log.zig");
const Arr = mlx.mlx_array;

// The row, its thread and the threadgroup arrays, declared at the body's scope for every form.
const head =
    \\const int l = threadgroup_position_in_grid.y;
    \\const uint tid = thread_index_in_threadgroup, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    \\const int N = HC * D, DT = (D + TG - 1) / TG;
    \\#ifdef SUSHI_METALLIB
    \\threadgroup float t_d[MIXN][32], t_s[32], t_pre[HC], t_s2[32];
    \\#else
    \\threadgroup float t_d[MIXN][TG / 32], t_s[TG / 32], t_pre[HC], t_s2[TG / 32];
    \\#endif
    \\const float e0 = eps[0], e1 = eps[1], e2 = eps[2];
    \\const float zs = tid < uint(MIXN) ? scale[tid / uint(HC)] : 0.0f, zb = tid < uint(MIXN) ? float(base[tid]) : 0.0f;
    \\
;

// The mixes' simd sums into `t_d` and `t_s`: thread `tid` the terms `k = tid + c·TG` in order.
const mix_sums =
    \\constexpr int MAX_DT = 6;
    \\const bool cache_sv = (D % TG == 0) && (DT <= MAX_DT);
    \\float sv[HC][MAX_DT];
    \\#pragma clang loop unroll(full)
    \\for (int hc_i = 0; hc_i < HC; ++hc_i)
    \\  #pragma clang loop unroll(full)
    \\  for (int i = 0; i < MAX_DT; ++i) sv[hc_i][i] = 0.0f;
    \\float dd[MIXN], ss = 0.0f;
    \\#pragma clang loop unroll(full)
    \\for (int j = 0; j < MIXN; ++j) dd[j] = 0.0f;
    \\if (cache_sv) {
    \\  #pragma clang loop unroll(full)
    \\  for (int hc_i = 0; hc_i < HC; ++hc_i) {
    \\    #pragma clang loop unroll(full)
    \\    for (int i = 0; i < MAX_DT; ++i) {
    \\      if (i >= DT) break;
    \\      const int k = int(tid) + (hc_i * DT + i) * TG;
    \\      float fv[MIXN];
    \\      #pragma clang loop unroll(full)
    \\      for (int j = 0; j < MIXN; ++j) fv[j] = fn[size_t(j) * N + k];
    \\      const size_t o = size_t(l) * N + k;
    \\      const float v = STREAM(o, k);
    \\      STORE_M;
    \\      sv[hc_i][i] = v;
    \\      #pragma clang loop unroll(full)
    \\      for (int j = 0; j < MIXN; ++j) dd[j] = metal::fma(v, fv[j], dd[j]);
    \\      ss = metal::fma(v, v, ss);
    \\    }
    \\  }
    \\} else {
    \\  for (int k = int(tid); k < N; k += TG) {
    \\    float fv[MIXN];
    \\    #pragma clang loop unroll(full)
    \\    for (int j = 0; j < MIXN; ++j) fv[j] = fn[size_t(j) * N + k];
    \\    const size_t o = size_t(l) * N + k;
    \\    const float v = STREAM(o, k);
    \\    STORE_M;
    \\    #pragma clang loop unroll(full)
    \\    for (int j = 0; j < MIXN; ++j) dd[j] = metal::fma(v, fv[j], dd[j]);
    \\    ss = metal::fma(v, v, ss);
    \\  }
    \\}
    \\for (int j = 0; j < MIXN; ++j) { const float v = metal::simd_sum(dd[j]); if (lane == 0) t_d[j][sg] = v; }
    \\{ const float v = metal::simd_sum(ss); if (lane == 0) t_s[sg] = v; }
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\
;

// The split, the collapse by `pre` and the norm from the sums in `t_d` and `t_s`.
const mix_tail =
    \\if (tid < uint(MIXN)) {
    \\  const int j = int(tid), c = j % HC;
    \\  float a = 0.0f, b = 0.0f;
    \\#ifndef SUSHI_METALLIB
    \\  #pragma clang loop unroll(full)
    \\#endif
    \\  for (int g = 0; g < TG / 32; ++g) { a += t_d[j][g]; b += t_s[g]; }
    \\  const float z = a * metal::precise::rsqrt(b / float(N) + e0) * zs + zb;
    \\  if (j < HC) t_pre[c] = 1.0f / (1.0f + metal::precise::exp(-z)) + e1;
    \\  else post[l * HC + c] = (1.0f / (1.0f + metal::precise::exp(-z))) * e2 + e1;
    \\}
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\float cp[HC];
    \\#pragma clang loop unroll(full)
    \\for (int j = 0; j < HC; ++j) cp[j] = t_pre[j];
    \\if (DT <= MAX_DT) {
    \\  float s2 = 0.0f, hv[MAX_DT];
    \\  #pragma clang loop unroll(full)
    \\  for (int i = 0; i < MAX_DT; ++i) {
    \\    if (i >= DT) break;
    \\    const int d = int(tid) + i * TG;
    \\    hv[i] = 0.0f;
    \\    if (d < D) {
    \\      float h = 0.0f;
    \\      #pragma clang loop unroll(full)
    \\      for (int j = 0; j < HC; ++j) h += cp[j] * SV(j, i, d);
    \\      hv[i] = h;
    \\      s2 += h * h;
    \\    }
    \\  }
    \\  s2 = metal::simd_sum(s2);
    \\  if (lane == 0) t_s2[sg] = s2;
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  float tot = 0.0f;
    \\#ifndef SUSHI_METALLIB
    \\  #pragma clang loop unroll(full)
    \\#endif
    \\  for (int g = 0; g < TG / 32; ++g) tot += t_s2[g];
    \\  const float inv = metal::precise::rsqrt(tot / float(D) + e0);
    \\  #pragma clang loop unroll(full)
    \\  for (int i = 0; i < MAX_DT; ++i) {
    \\    if (i >= DT) break;
    \\    const int d = int(tid) + i * TG;
    \\    if (d < D) h_out[size_t(l) * D + d] = WV(i, d) * float(hv[i] * inv);
    \\  }
    \\} else {
    \\  float s2 = 0.0f;
    \\  for (int i = 0; i < DT; ++i) {
    \\    const int d = int(tid) + i * TG;
    \\    if (d < D) {
    \\      float h = 0.0f;
    \\      #pragma clang loop unroll(full)
    \\      for (int j = 0; j < HC; ++j) h += cp[j] * STREAM((size_t(l) * HC + j) * D + d, j * D + d);
    \\      h_out[size_t(l) * D + d] = h;
    \\      s2 += h * h;
    \\    }
    \\  }
    \\  s2 = metal::simd_sum(s2);
    \\  if (lane == 0) t_s2[sg] = s2;
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  float tot = 0.0f;
    \\#ifndef SUSHI_METALLIB
    \\  #pragma clang loop unroll(full)
    \\#endif
    \\  for (int g = 0; g < TG / 32; ++g) tot += t_s2[g];
    \\  const float inv = metal::precise::rsqrt(tot / float(D) + e0);
    \\  for (int i = 0; i < DT; ++i) {
    \\    const int d = int(tid) + i * TG;
    \\    if (d < D) {
    \\      const size_t o = size_t(l) * D + d;
    \\      h_out[o] = float(w[d]) * float(h_out[o] * inv);
    \\    }
    \\  }
    \\}
;

// The wide launch's `mix_sums`: MIXN threadgroups a row, each simdgroup one (virtual simdgroup `vs`, mix `j`) pair of
// the single threadgroup's, its lanes the same terms in the same order and the same `simd_sum`. Each sum goes to the
// row's partials in the counter page `ctr` (past its `CTR` counters) by relaxed atomics, fenced at device scope either
// side of the count; the row's last threadgroup to count itself in (`ctr[l]`, which it puts back to zero) reads them
// into `t_d` and `t_s` and runs `mix_tail`, the rest return, so none waits for another. Each threadgroup first loads
// the collapse's stream values and norm weight (`sv`, `wv`), so the last one's tail reads only the sums.
const wide_sums =
    \\constexpr int MAX_DT = 6;
    \\threadgroup bool t_last[1];
    \\float sv[HC][MAX_DT], wv[MAX_DT];
    \\#pragma clang loop unroll(full)
    \\for (int i = 0; i < MAX_DT; ++i) {
    \\  wv[i] = 0.0f;
    \\  #pragma clang loop unroll(full)
    \\  for (int j = 0; j < HC; ++j) sv[j][i] = 0.0f;
    \\}
    \\if (DT <= MAX_DT) {
    \\  #pragma clang loop unroll(full)
    \\  for (int i = 0; i < MAX_DT; ++i) {
    \\    if (i >= DT) break;
    \\    const int d = int(tid) + i * TG;
    \\    if (d < D) {
    \\      wv[i] = float(w[d]);
    \\      #pragma clang loop unroll(full)
    \\      for (int j = 0; j < HC; ++j) sv[j][i] = STREAM((size_t(l) * HC + j) * D + d, j * D + d);
    \\    }
    \\  }
    \\}
    \\{
    \\  device metal::atomic_float *pr = (device metal::atomic_float *)(ctr + CTR) + size_t(l) * (MIXN + 1) * (TG / 32);
    \\  {
    \\    const uint p = threadgroup_position_in_grid.x * uint(TG / 32) + sg;
    \\    const int j = int(p % uint(MIXN));
    \\    const uint vs = p / uint(MIXN);
    \\    float dd = 0.0f, ss = 0.0f;
    \\    const int n_steps = (N + TG - 1) / TG;
    \\#ifndef SUSHI_METALLIB
    \\    #pragma clang loop unroll(full)
    \\#endif
    \\    for (int c = 0; c < n_steps; ++c) {
    \\      const int k = int(vs * 32u + lane) + c * TG;
    \\      if (k < N) {
    \\        const float fv = fn[size_t(j) * N + k];
    \\        const size_t o = size_t(l) * N + k;
    \\        const float v = STREAM(o, k);
    \\        if (j == 0) { STORE_M; }
    \\        dd = metal::fma(v, fv, dd);
    \\        ss = metal::fma(v, v, ss);
    \\      }
    \\    }
    \\    dd = metal::simd_sum(dd);
    \\    ss = metal::simd_sum(ss);
    \\    if (lane == 0) {
    \\      metal::atomic_store_explicit(&pr[j * (TG / 32) + vs], dd, metal::memory_order_relaxed);
    \\      if (j == 0) metal::atomic_store_explicit(&pr[MIXN * (TG / 32) + vs], ss, metal::memory_order_relaxed);
    \\    }
    \\  }
    \\  metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst, metal::thread_scope_device);
    \\  threadgroup_barrier(metal::mem_flags::mem_device);
    \\  if (tid == 0) {
    \\    metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst, metal::thread_scope_device);
    \\    t_last[0] = metal::atomic_fetch_add_explicit((device metal::atomic_uint *)ctr + l, 1u, metal::memory_order_relaxed) == uint(MIXN - 1);
    \\    metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst, metal::thread_scope_device);
    \\  }
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  if (!t_last[0]) return;
    \\  metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst, metal::thread_scope_device);
    \\  for (int i = int(tid); i < (MIXN + 1) * (TG / 32); i += TG) {
    \\    const float v = metal::atomic_load_explicit(&pr[i], metal::memory_order_relaxed);
    \\    if (i < MIXN * (TG / 32)) t_d[i / (TG / 32)][i % (TG / 32)] = v; else t_s[i - MIXN * (TG / 32)] = v;
    \\  }
    \\  if (tid == 0) metal::atomic_store_explicit((device metal::atomic_uint *)ctr + l, 0u, metal::memory_order_relaxed);
    \\}
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\
;

// The back form's first part, outside the mix's contract(off): `hy4_norm.rowsExpand`'s norm of `a + b` by `wn`
// (MLX's `rms_single_row`: its RG threads of four columns each, here RR to a lane, the same simd sums in the same
// lanes), the normed row into `t_y`.
fn backNorm(comptime add: bool) []const u8 {
    return
        \\#ifdef SUSHI_METALLIB
        \\threadgroup float t_y[3072], t_ls[32], t_inv[1];
        \\#else
        \\threadgroup float t_y[D], t_ls[32], t_inv[1];
        \\#endif
        \\{
        \\  const uint axis_size = uint(D);
        \\  const int RG = 32 * ((((D + 3) / 4) + 31) / 32), RR = (RG + TG - 1) / TG;
        \\  if (sg == 0) t_ls[lane] = 0;
        \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        \\  if (RR <= 2) {
        \\    float thread_x[2][4];
        \\    #pragma clang loop unroll(full)
        \\    for (int r = 0; r < 2; ++r) {
        \\      if (r >= RR) break;
        \\      const uint vt = tid + uint(r * TG);
        \\      if (vt < uint(RG)) {
        \\        float acc = 0;
        \\        for (int i = 0; i < 4; i++) {
        \\          uint c = vt * 4 + i;
        \\          float v = 0;
        \\          if (c < axis_size) {
        \\            v = static_cast<float>(a[size_t(l) * D + c]);
        \\
    ++ (if (add)
        \\            v = v + static_cast<float>(b[size_t(l) * D + c]);
        \\
    else "") ++
        \\          }
        \\          thread_x[r][i] = v;
        \\          acc += thread_x[r][i] * thread_x[r][i];
        \\        }
        \\        acc = simd_sum(acc);
        \\        if (lane == 0) t_ls[vt / 32] = acc;
        \\      }
        \\    }
        \\    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        \\    if (sg == 0) {
        \\      float acc = simd_sum(t_ls[lane]);
        \\      if (lane == 0) t_inv[0] = metal::precise::rsqrt(acc / axis_size + as_type<float>(EPS));
        \\    }
        \\    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        \\    #pragma clang loop unroll(full)
        \\    for (int r = 0; r < 2; ++r) {
        \\      if (r >= RR) break;
        \\      const uint vt = tid + uint(r * TG);
        \\      for (int i = 0; i < 4; i++) {
        \\        uint c = vt * 4 + i;
        \\        if (vt < uint(RG) && c < axis_size) t_y[c] = static_cast<float>(wn[c]) * static_cast<float>(thread_x[r][i] * t_inv[0]);
        \\      }
        \\    }
        \\  } else {
        \\    for (int r = 0; r < RR; ++r) {
        \\      const uint vt = tid + uint(r * TG);
        \\      if (vt < uint(RG)) {
        \\        float acc = 0;
        \\        for (int i = 0; i < 4; i++) {
        \\          uint c = vt * 4 + i;
        \\          float v = 0;
        \\          if (c < axis_size) {
        \\            v = static_cast<float>(a[size_t(l) * D + c]);
        \\
    ++ (if (add)
        \\            v = v + static_cast<float>(b[size_t(l) * D + c]);
        \\
    else "") ++
        \\            t_y[c] = v;
        \\          }
        \\          acc += v * v;
        \\        }
        \\        acc = simd_sum(acc);
        \\        if (lane == 0) t_ls[vt / 32] = acc;
        \\      }
        \\    }
        \\    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        \\    if (sg == 0) {
        \\      float acc = simd_sum(t_ls[lane]);
        \\      if (lane == 0) t_inv[0] = metal::precise::rsqrt(acc / axis_size + as_type<float>(EPS));
        \\    }
        \\    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        \\    for (int r = 0; r < RR; ++r) {
        \\      const uint vt = tid + uint(r * TG);
        \\      for (int i = 0; i < 4; i++) {
        \\        uint c = vt * 4 + i;
        \\        if (vt < uint(RG) && c < axis_size) t_y[c] = static_cast<float>(wn[c]) * static_cast<float>(t_y[c] * t_inv[0]);
        \\      }
        \\    }
        \\  }
        \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        \\}
        \\
    ;
}

const back_norm = backNorm(true);

const back_inner_norm =
    \\#ifdef SUSHI_METALLIB
    \\threadgroup float t_y[3072], t_ls[32], t_inv[1];
    \\#else
    \\threadgroup float t_y[D], t_ls[32], t_inv[1];
    \\#endif
    \\{
    \\#pragma clang fp contract(off)
    \\  const uint NB = uint(D) / 128u;
    \\  const float psc = 0.08838834764831845f;
    \\  for (uint blk = sg; blk < NB; blk += uint(TG) / 32u) {
    \\    const uint col = blk * 128u + lane;
    \\    const size_t xb = (size_t)l * uint(D) + col;
    \\    float4 va = exl3_h128(float4(float(a[xb]), float(a[xb + 32u]), float(a[xb + 64u]), float(a[xb + 96u])), lane);
    \\    float4 vb = exl3_h128(float4(float(b[xb]), float(b[xb + 32u]), float(b[xb + 64u]), float(b[xb + 96u])), lane);
    \\    t_y[col] = float(bfloat16_t(va.x * psc * float(asvh[col]))) + float(bfloat16_t(vb.x * psc * float(bsvh[col])));
    \\    t_y[col + 32u] = float(bfloat16_t(va.y * psc * float(asvh[col + 32u]))) + float(bfloat16_t(vb.y * psc * float(bsvh[col + 32u])));
    \\    t_y[col + 64u] = float(bfloat16_t(va.z * psc * float(asvh[col + 64u]))) + float(bfloat16_t(vb.z * psc * float(bsvh[col + 64u])));
    \\    t_y[col + 96u] = float(bfloat16_t(va.w * psc * float(asvh[col + 96u]))) + float(bfloat16_t(vb.w * psc * float(bsvh[col + 96u])));
    \\  }
    \\}
    \\{
    \\  const uint axis_size = uint(D);
    \\  const int RG = 32 * ((((D + 3) / 4) + 31) / 32), RR = (RG + TG - 1) / TG;
    \\  if (sg == 0) t_ls[lane] = 0;
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  if (RR <= 2) {
    \\    float thread_x[2][4];
    \\    #pragma clang loop unroll(full)
    \\    for (int r = 0; r < 2; ++r) {
    \\      if (r >= RR) break;
    \\      const uint vt = tid + uint(r * TG);
    \\      if (vt < uint(RG)) {
    \\        float acc = 0;
    \\        for (int i = 0; i < 4; i++) {
    \\          uint c = vt * 4 + i;
    \\          float v = c < axis_size ? t_y[c] : 0;
    \\          thread_x[r][i] = v;
    \\          acc += thread_x[r][i] * thread_x[r][i];
    \\        }
    \\        acc = simd_sum(acc);
    \\        if (lane == 0) t_ls[vt / 32] = acc;
    \\      }
    \\    }
    \\    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\    if (sg == 0) {
    \\      float acc = simd_sum(t_ls[lane]);
    \\      if (lane == 0) t_inv[0] = metal::precise::rsqrt(acc / axis_size + as_type<float>(EPS));
    \\    }
    \\    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\    #pragma clang loop unroll(full)
    \\    for (int r = 0; r < 2; ++r) {
    \\      if (r >= RR) break;
    \\      const uint vt = tid + uint(r * TG);
    \\      for (int i = 0; i < 4; i++) {
    \\        uint c = vt * 4 + i;
    \\        if (vt < uint(RG) && c < axis_size) t_y[c] = static_cast<float>(wn[c]) * static_cast<float>(thread_x[r][i] * t_inv[0]);
    \\      }
    \\    }
    \\  } else {
    \\    for (int r = 0; r < RR; ++r) {
    \\      const uint vt = tid + uint(r * TG);
    \\      if (vt < uint(RG)) {
    \\        float acc = 0;
    \\        for (int i = 0; i < 4; i++) {
    \\          uint c = vt * 4 + i;
    \\          float v = c < axis_size ? t_y[c] : 0;
    \\          acc += v * v;
    \\        }
    \\        acc = simd_sum(acc);
    \\        if (lane == 0) t_ls[vt / 32] = acc;
    \\      }
    \\    }
    \\    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\    if (sg == 0) {
    \\      float acc = simd_sum(t_ls[lane]);
    \\      if (lane == 0) t_inv[0] = metal::precise::rsqrt(acc / axis_size + as_type<float>(EPS));
    \\    }
    \\    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\    for (int r = 0; r < RR; ++r) {
    \\      const uint vt = tid + uint(r * TG);
    \\      for (int i = 0; i < 4; i++) {
    \\        uint c = vt * 4 + i;
    \\        if (vt < uint(RG) && c < axis_size) t_y[c] = static_cast<float>(wn[c]) * static_cast<float>(t_y[c] * t_inv[0]);
    \\      }
    \\    }
    \\  }
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\}
    \\
;

/// The expand form's `y` as `W_o`'s launch's finish and norm give it (`rows_finish_rms_thru`), from that launch's k-split
/// partials `op` (`[L * OSPLIT, D]` f32), its `osvh` and the norm's `ow` and ε (`OEPS`, f32 bits): each simdgroup
/// finishes its 128-blocks into `t_y` (the parts summed in order, H128, `psc · svh`, the bf16 rounding), squares four
/// neighbours of a block by fused multiply-adds, sums them a block by `simd_sum` and the blocks' sums by simdgroup 0,
/// and leaves `ow · (y · inv)` in `t_y`, as that finish's one threadgroup does.
const o_norm =
    \\#ifdef SUSHI_METALLIB
    \\threadgroup float t_y[3072], t_ls[32], t_inv[1];
    \\#else
    \\threadgroup float t_y[D], t_ls[32], t_inv[1];
    \\#endif
    \\{
    \\#pragma clang fp contract(off)
    \\  const uint NB = uint(D) / 128u;
    \\  const float psc = 0.08838834764831845f;
    \\  if (sg == 0) t_ls[lane] = 0.0f;
    \\  for (uint blk = sg; blk < NB; blk += uint(TG) / 32u) {
    \\    const uint b = blk * 128u + lane;
    \\    float4 v = 0.0f;
    \\#ifndef SUSHI_METALLIB
    \\    #pragma clang loop unroll(full)
    \\#endif
    \\    for (uint s = 0u; s < uint(OSPLIT); s++) {
    \\      const device float *p = op + ((size_t)l * uint(OSPLIT) + s) * uint(D) + b;
    \\      v += float4(p[0], p[32], p[64], p[96]);
    \\    }
    \\    v = exl3_h128(v, lane);
    \\    t_y[b] = float(bfloat16_t(v.x * psc * float(osvh[b])));
    \\    t_y[b + 32u] = float(bfloat16_t(v.y * psc * float(osvh[b + 32u])));
    \\    t_y[b + 64u] = float(bfloat16_t(v.z * psc * float(osvh[b + 64u])));
    \\    t_y[b + 96u] = float(bfloat16_t(v.w * psc * float(osvh[b + 96u])));
    \\  }
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  for (uint blk = sg; blk < NB; blk += uint(TG) / 32u) {
    \\    float acc = 0.0f;
    \\    for (int i = 0; i < 4; i++) {
    \\      const float tx = t_y[blk * 128u + lane * 4u + uint(i)];
    \\      acc = metal::fma(tx, tx, acc);
    \\    }
    \\    acc = simd_sum(acc);
    \\    if (lane == 0) t_ls[blk] = acc;
    \\  }
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  if (sg == 0) {
    \\    const float acc = simd_sum(t_ls[lane]);
    \\    if (lane == 0) t_inv[0] = metal::precise::rsqrt(acc / uint(D) + as_type<float>(OEPS));
    \\  }
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  for (uint blk = sg; blk < NB; blk += uint(TG) / 32u) {
    \\    for (int i = 0; i < 4; i++) {
    \\      const uint c = blk * 128u + lane * 4u + uint(i);
    \\      t_y[c] = float(ow[c]) * (t_y[c] * t_inv[0]);
    \\    }
    \\  }
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\}
    \\
;

/// The expand form's `y` as `rows_finish_rms_thru` norms W_o's finished rows `oy` (`[L, D]` f32) by `ow` and ε (`OEPS`):
/// `o_norm` past its finish, each simdgroup's 128-blocks read from `oy` in that finish's lanes, into `t_y`.
const o_rows_norm =
    \\#ifdef SUSHI_METALLIB
    \\threadgroup float t_y[3072], t_ls[32], t_inv[1];
    \\#else
    \\threadgroup float t_y[D], t_ls[32], t_inv[1];
    \\#endif
    \\{
    \\#pragma clang fp contract(off)
    \\  const uint NB = uint(D) / 128u, SGN = uint(TG) / 32u, BR = (NB + SGN - 1u) / SGN;
    \\  const device float *yr = oy + size_t(l) * D;
    \\  if (sg == 0 && lane >= NB) t_ls[lane] = 0.0f;
    \\  if (BR <= 2u) {
    \\    float tx[2][4];
    \\    #pragma clang loop unroll(full)
    \\    for (uint i = 0u; i < 2u; i++) {
    \\      if (i >= BR) break;
    \\      const uint blk = sg + i * SGN;
    \\      if (blk < NB) {
    \\        float acc = 0.0f;
    \\        for (int q = 0; q < 4; q++) {
    \\          tx[i][q] = yr[blk * 128u + lane * 4u + uint(q)];
    \\          acc = metal::fma(tx[i][q], tx[i][q], acc);
    \\        }
    \\        acc = simd_sum(acc);
    \\        if (lane == 0) t_ls[blk] = acc;
    \\      }
    \\    }
    \\    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\    if (sg == 0) {
    \\      const float acc = simd_sum(t_ls[lane]);
    \\      if (lane == 0) t_inv[0] = metal::precise::rsqrt(acc / uint(D) + as_type<float>(OEPS));
    \\    }
    \\    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\    #pragma clang loop unroll(full)
    \\    for (uint i = 0u; i < 2u; i++) {
    \\      if (i >= BR) break;
    \\      const uint blk = sg + i * SGN;
    \\      if (blk < NB) for (int q = 0; q < 4; q++) {
    \\        const uint c = blk * 128u + lane * 4u + uint(q);
    \\        t_y[c] = float(ow[c]) * (tx[i][q] * t_inv[0]);
    \\      }
    \\    }
    \\  } else {
    \\    for (uint i = 0u; i < BR; i++) {
    \\      const uint blk = sg + i * SGN;
    \\      if (blk < NB) {
    \\        float acc = 0.0f;
    \\        for (int q = 0; q < 4; q++) {
    \\          const uint c = blk * 128u + lane * 4u + uint(q);
    \\          const float v = yr[c];
    \\          t_y[c] = v;
    \\          acc = metal::fma(v, v, acc);
    \\        }
    \\        acc = simd_sum(acc);
    \\        if (lane == 0) t_ls[blk] = acc;
    \\      }
    \\    }
    \\    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\    if (sg == 0) {
    \\      const float acc = simd_sum(t_ls[lane]);
    \\      if (lane == 0) t_inv[0] = metal::precise::rsqrt(acc / uint(D) + as_type<float>(OEPS));
    \\    }
    \\    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\    for (uint i = 0u; i < BR; i++) {
    \\      const uint blk = sg + i * SGN;
    \\      if (blk < NB) for (int q = 0; q < 4; q++) {
    \\        const uint c = blk * 128u + lane * 4u + uint(q);
    \\        t_y[c] = float(ow[c]) * (t_y[c] * t_inv[0]);
    \\      }
    \\    }
    \\  }
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\}
    \\
;

/// A launch's form: the plain mix, the mix of the stream expanded by a sublayer's output `y`, or by the combine norm of
/// `a + b` it computes first (`back`: the layer boundary); or expanded by `y` finished and normed from W_o's partials
/// in the launch (`expand_parts`), or normed from W_o's finished rows (`expand_rows`).
pub const Form = enum { plain, expand, back, back_inner, expand_norm, expand_parts, expand_rows };

var kernels: [2][7]?mlx.mlx_fast_metal_kernel = @splat(@splat(null));

/// The Metal body for `(form, wide)`. Shared between `kernel` and `metallib.zig`.
pub fn kernelSource(comptime f: Form, comptime w: bool) [:0]const u8 {
    const stream = switch (f) {
        .plain => "#define STREAM(o, k) float(x[o])\n#define STORE_M (void)0\n",
        .expand => "#define STREAM(o, k) (pin[l * HC + (k) / D] * float(y[size_t(l) * D + (k) % D]) + float(x[o]))\n#define STORE_M m[o] = v\n",
        .back, .back_inner, .expand_norm, .expand_parts, .expand_rows => "#define STREAM(o, k) (pin[l * HC + (k) / D] * t_y[(k) % D] + float(x[o]))\n#define STORE_M m[o] = v\n",
    };
    const first = switch (f) {
        .plain, .expand => "",
        .back => back_norm,
        .back_inner => back_inner_norm,
        .expand_norm => comptime backNorm(false),
        .expand_parts => o_norm,
        .expand_rows => o_rows_norm,
    };
    // Where the collapse reads its stream values and norm weight: the wide launch's prefetch, else the inputs.
    const sums = if (w) "#define SV(j, i, d) sv[(j)][(i)]\n#define WV(i, d) wv[i]\n" ++ wide_sums else "#define SV(j, i, d) (cache_sv ? sv[(j)][(i)] : STREAM((size_t(l) * HC + (j)) * D + (d), (j) * D + (d)))\n#define WV(i, d) float(w[d])\n" ++ mix_sums;
    // A wide grid's padding row (`build`) returns before it reads.
    const gate = if (w) "if (l >= x_shape[0]) return;\n" else "";
    return stream ++ head ++ gate ++ first ++ "{\n#pragma clang fp contract(off)\n" ++ sums ++ mix_tail ++ "}\n";
}

/// The form's kernel: one threadgroup a row, or `wide`, `2·HC` threadgroups a row that count into the counter page.
fn kernel(form: Form, wide: bool) !mlx.mlx_fast_metal_kernel {
    const slot = &kernels[@intFromBool(wide)][@backingInt(form)];
    if (slot.*) |k| return k;
    const ins: []const [*:0]const u8 = switch (form) {
        .plain => &.{ "x", "fn", "scale", "base", "eps", "w" },
        .expand => &.{ "x", "y", "pin", "fn", "scale", "base", "eps", "w" },
        .back => &.{ "x", "pin", "a", "b", "wn", "fn", "scale", "base", "eps", "w" },
        .back_inner => &.{ "x", "pin", "a", "asvh", "b", "bsvh", "wn", "fn", "scale", "base", "eps", "w" },
        .expand_norm => &.{ "x", "pin", "a", "wn", "fn", "scale", "base", "eps", "w" },
        .expand_parts => &.{ "x", "op", "osvh", "ow", "pin", "fn", "scale", "base", "eps", "w" },
        .expand_rows => &.{ "x", "oy", "ow", "pin", "fn", "scale", "base", "eps", "w" },
    };
    const all_outs = [_][*:0]const u8{ "post", "h_out", "m", "ctr" };
    var outs: [4][*:0]const u8 = undefined;
    var n_out: usize = 0;
    for (all_outs) |o| {
        if (o[0] == 'm' and form == .plain or o[0] == 'c' and !wide) continue;
        outs[n_out] = o;
        n_out += 1;
    }
    const iv = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&outs, n_out);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = switch (wide) {
        inline else => |w| switch (form) {
            inline else => |f| blk: {
                const src = comptime kernelSource(f, w);
                const name = switch (f) {
                    .plain => "sushi_hy4_hc_mix_norm",
                    .expand => "sushi_hy4_hc_mix_expand_norm",
                    .back => "sushi_hy4_hc_back_mix_norm",
                    .back_inner => "sushi_hy4_hc_back_inner_mix_norm",
                    .expand_norm => "sushi_hy4_hc_expand_norm_mix_norm",
                    .expand_parts => "sushi_hy4_hc_expand_parts_mix_norm",
                    .expand_rows => "sushi_hy4_hc_expand_rows_mix_norm",
                } ++ (if (w) "_wide" else "");
                break :blk mlx.mlx_fast_metal_kernel_new(name, iv, ov, src, if (f == .expand_parts or f == .back_inner) exl3.project_rows.ROWS_HELPERS else "", true, false);
            },
        },
    };
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    slot.* = k;
    return k;
}

/// The most rows a launch takes: a decode block (a prefill keeps the tiled mix, whose tiles share each weight load).
pub const rows_max = 8;

/// Whether the launch serves `x` [L, HC, D] (f32) with mixes `fn` [2·HC, HC·D] f32 (no comb), `scale` [2] and `w`
/// [D] on a GPU stream.
pub fn applies(s: mlx.mlx_stream, x: Arr, fn_w: Arr, scale: Arr) bool {
    const xs = mlx.getShape(x);
    const fs = mlx.getShape(fn_w);
    if (xs.len != 3 or fs.len != 2) return false;
    return mlx.streamIsGpu(s) and xs[0] >= 1 and xs[1] <= 8 and fs[0] == 2 * xs[1] and
        fs[1] == xs[1] * xs[2] and mlx.mlx_array_size(scale) >= 2 and mlx.mlx_array_dtype(x) == .float32 and
        mlx.mlx_array_dtype(fn_w) == .float32 and mlx.mlx_array_dtype(scale) == .float32;
}

/// Whether the engine's stream `s` runs the wide launch of `rows` rows of `hc` streams: its partials, `2·hc + 1` simd
/// sums a virtual simdgroup a row, fit the counter page past its counters, one counter a row.
fn takesWide(s: mlx.mlx_stream, rows: c_int, hc: c_int) bool {
    comptime std.debug.assert(rows_max <= mlx.engine.counter_words);
    if (rows > rows_max) return false;
    const parts: usize = @intCast(rows * (2 * hc + 1));
    return wide_on and mlx.engine.side(.{s}) == .engine and parts * (hck.hc_tg / 32) <= mlx.engine.counter_page_words - mlx.engine.counter_words;
}

/// Whether the engine's stream takes the wide launch where it fits (`takesWide`); off, every row is one threadgroup.
/// A test seam: both give the same bytes.
pub var wide_on = true;

const Key = struct { rows: c_int, hc: c_int, d: c_int, tg: c_int, form: Form, wide: bool, w_dt: mlx.mlx_dtype, base_dt: mlx.mlx_dtype, eps: c_int = 0, osplit: c_int = 0 };
var cfgs: kcfg.Cache(Key) = .{};

fn build(k: Key) !kcfg.Config {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ k.rows, k.hc }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ k.rows, k.d }, 2, .float32));
    if (k.form != .plain) try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ k.rows, k.hc, k.d }, 3, .float32));
    if (k.wide) try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{mlx.engine.counter_page_words}, 1, .uint32));
    // A wide grid of rows takes an even count (a padding row returns at once): a row whose threadgroups straddle the
    // middle of the grid counts in slower on the M5 Ultra (engine-kernels.md, Metal pitfalls).
    const grid_rows = if (k.wide and k.rows > 1) k.rows + @rem(k.rows, 2) else k.rows;
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, if (k.wide) 2 * k.hc * k.tg else k.tg, grid_rows, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, k.tg, 1, 1));
    inline for (.{ "TG", "HC", "MIXN", "D" }, .{ k.tg, k.hc, 2 * k.hc, k.d }) |n, v|
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, n, v));
    if (k.wide) try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "CTR", mlx.engine.counter_words));
    if (k.form == .back or k.form == .back_inner or k.form == .expand_norm) try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "EPS", k.eps));
    if (k.form == .expand_rows) try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "OEPS", k.eps));
    if (k.form == .expand_parts) inline for (.{ "OSPLIT", "OEPS" }, .{ k.osplit, k.eps }) |n, v|
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, n, v));
    return cfg;
}

pub const Normed = struct { post: Arr, h: Arr, m: ?Arr = null };

fn run(s: mlx.mlx_stream, inputs: []const Arr, x: Arr, w: Arr, base: Arr, form: Form, eps_bits: c_int) !Normed {
    return runSplit(s, inputs, x, w, base, form, eps_bits, 0);
}

fn runSplit(s: mlx.mlx_stream, inputs: []const Arr, x: Arr, w: Arr, base: Arr, form: Form, eps_bits: c_int, osplit: c_int) !Normed {
    const xs = mlx.getShape(x);
    if (form != .plain and form != .expand and xs[2] > 3072) return error.Hy4HcPartsShape;
    var key: Key = .{ .rows = xs[0], .hc = xs[1], .d = xs[2], .tg = @intCast(hck.hc_tg), .form = form, .wide = true, .w_dt = mlx.mlx_array_dtype(w), .base_dt = mlx.mlx_array_dtype(base), .eps = eps_bits, .osplit = osplit };
    const iv = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    const wide = takesWide(s, xs[0], xs[1]);
    const pipe = pipeOf(key, inputs);
    const granted = if (wide) grants.get(pipe) else null;
    // A dry recording looks no pipeline up, so it cannot learn a grant: the load's probe has, or its census notes it.
    if (wide and granted == null and mlx.engine.dryOn()) {
        const c = census orelse return error.Hy4HcWideUnprobed;
        try c.note(pipe, inputs);
    }
    if (!wide or granted == false or !try applyWide(&ov, form, iv, try cfgs.get(key, build), s, pipe)) {
        key.wide = false;
        try applyNarrow(&ov, form, iv, &key, s, pipeOf(key, inputs), inputs);
    }
    const expand = form != .plain;
    const n: usize = if (expand) 3 else 2;
    var out: [3]Arr = .{ mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new() };
    errdefer for (out) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (0..n) |i| try mlx.check(mlx.mlx_vector_array_get(&out[i], ov, i));
    if (!expand) _ = mlx.mlx_array_free(out[2]);
    return .{ .post = out[0], .h = out[1], .m = if (expand) out[2] else null };
}

/// One wide launch's pipeline as the engine instantiates it: the form, the template values its config sets and its
/// inputs' dtypes. A GPU grants each pipeline its own threads a group, so each holds its own grant.
const Pipe = struct { form: Form, wide: bool, hc: c_int, d: c_int, tg: c_int, eps: c_int, osplit: c_int, dts: [max_ins]u8 };
const max_ins = 12;

fn pipeOf(key: Key, inputs: []const Arr) Pipe {
    var p: Pipe = .{ .form = key.form, .wide = key.wide, .hc = key.hc, .d = key.d, .tg = key.tg, .eps = key.eps, .osplit = key.osplit, .dts = @splat(0xff) };
    // Every form takes at most `max_ins` inputs (`kernel`).
    for (inputs, p.dts[0..inputs.len]) |a, *dt| dt.* = @intCast(@intFromEnum(mlx.mlx_array_dtype(a)));
    return p;
}

const gpa = std.heap.c_allocator;

/// Each wide pipeline's grant, once a launch answered it (a served one, or the load's probe, `probeNoted`): a refused
/// pipeline runs each row as one threadgroup from then on.
var grants: std.AutoHashMapUnmanaged(Pipe, bool) = .empty;

/// Each one-threadgroup pipeline's threads a group, once a launch answered it: `hc_tg`, or the widest halving of it the
/// GPU grants (Metal caps a register-heavy pipeline below 1024 on an M1 or M2, each pipeline at its own count).
var narrow_tgs: std.AutoHashMapUnmanaged(Pipe, c_int) = .empty;

/// The wide pipelines a dry forward reached with no grant latched, each with its inputs' shapes: `Model.probeWide`'s
/// census, which `probeNoted` then launches.
pub const Census = struct {
    noted: std.ArrayList(Noted) = .empty,

    const Noted = struct { pipe: Pipe, n: usize, shapes: [max_ins][3]c_int = undefined, dims: [max_ins]usize = undefined };

    pub fn deinit(self: *Census) void {
        self.noted.deinit(gpa);
    }

    fn note(self: *Census, pipe: Pipe, inputs: []const Arr) !void {
        if (inputs.len > max_ins) return error.Hy4HcCensusInputs;
        for (inputs) |a| if (mlx.getShape(a).len > 3) return error.Hy4HcCensusInputs;
        for (self.noted.items) |n| if (std.meta.eql(n.pipe, pipe)) return;
        var n: Noted = .{ .pipe = pipe, .n = inputs.len };
        for (inputs, 0..) |a, i| {
            const sh = mlx.getShape(a);
            n.dims[i] = sh.len;
            @memcpy(n.shapes[i][0..sh.len], sh);
        }
        try self.noted.append(gpa, n);
    }
};

/// While set, a dry launch whose pipeline holds no grant is noted here instead of refused.
pub var census: ?*Census = null;

/// Each pipeline `c` noted launched once for real, on zeros of its inputs' shapes, so its grant latches as a served
/// launch latches it (the values make no difference to a grant).
pub fn probeNoted(s: mlx.mlx_stream, c: *const Census) !void {
    for (c.noted.items) |n| {
        if (if (n.pipe.wide) grants.contains(n.pipe) else narrow_tgs.contains(n.pipe)) continue;
        var o: Ops = .{ .s = s };
        defer o.deinit();
        var ins: [max_ins]Arr = undefined;
        for (ins[0..n.n], n.shapes[0..n.n], n.dims[0..n.n], n.pipe.dts[0..n.n]) |*x, sh, dims, dt|
            x.* = try o.contiguous(try o.zeros(sh[0..dims], @enumFromInt(dt)));
        // Every form's inputs end with the mix's base, its eps and the norm weight.
        const r = try runSplit(s, ins[0..n.n], ins[0], ins[n.n - 1], ins[n.n - 3], n.pipe.form, n.pipe.eps, n.pipe.osplit);
        _ = try o.own(r.post);
        _ = try o.own(r.h);
        if (r.m) |m| _ = try o.own(m);
    }
    try mlx.check(mlx.mlx_synchronize(s));
}

/// Every grant forgotten, as a fresh process holds none: a test seam, for a test that changes what the GPU grants.
pub fn forgetGrants() void {
    grants.clearRetainingCapacity();
    narrow_tgs.clearRetainingCapacity();
}

/// Wide launches made since the process began.
pub var wide_launches: u64 = 0;

/// The wide launch into `ov`; false, the pipeline's grant latched off, where the GPU grants `pipe` fewer than `hc_tg`
/// threads a group (the caller takes the one-threadgroup launch).
fn applyWide(ov: *mlx.mlx_vector_array, form: Form, iv: mlx.mlx_vector_array, cfg: kcfg.Config, s: mlx.mlx_stream, pipe: Pipe) !bool {
    mlx.expected_refusal = "Thread group size";
    defer mlx.expected_refusal = null;
    const rc = mlx.kernelApplyCounted(ov, try kernel(form, true), iv, cfg, s) orelse return error.Hy4HcWideStream;
    if (rc == 0) {
        // A dry launch runs nothing and learns nothing of the grant.
        if (!mlx.engine.dryOn()) {
            wide_launches += 1;
            const g = try grants.getOrPut(gpa, pipe);
            if (!g.found_existing) g.value_ptr.* = true;
        }
        return true;
    }
    if (!mlx.takeErrorIf("Thread group size")) return error.MlxError;
    try grants.put(gpa, pipe, false);
    log.warn("[engine] hy4 hc: this GPU does not grant the wide {s} mix {d} threads a group; its rows run one threadgroup each\n", .{ @tagName(form), hck.hc_tg });
    return false;
}

/// Lowers `hck.hc_tg` to the widest halving this GPU grants every form's one-threadgroup launch of one row of 4 streams
/// of `d` (an M1 caps the fused forms below 1024), so every hyper-connection kernel of the process, tiled or fused, runs
/// at one width and the two give the same bytes. The load runs it after `hck.probeIdentity`, before it records a plan.
pub fn probeForms(s: mlx.mlx_stream, d: c_int) !void {
    const scope = mlx.engine.enter(s);
    defer scope.end();
    const was = wide_on;
    wide_on = false;
    defer wide_on = was;
    while (true) {
        forgetGrants();
        var o: Ops = .{ .s = s };
        defer o.deinit();
        const c = try FormCase.zeros(&o, 1, d, if (d >= 3072) 32 else 16);
        var least: c_int = @intCast(hck.hc_tg);
        for (std.enums.values(Form)) |f| {
            switch (f) {
                .plain, .expand => {},
                .back, .expand_norm => if (d > 3072) continue,
                .back_inner, .expand_parts, .expand_rows => if (d > 3072 or @rem(d, 128) != 0) continue,
            }
            const r = try c.run(&o, f);
            try mlx.check(mlx.mlx_array_eval(r.h));
            var it = narrow_tgs.valueIterator();
            while (it.next()) |tg| least = @min(least, tg.*);
        }
        forgetGrants();
        if (least == hck.hc_tg) return;
        log.info("[engine] hy4 hc: this GPU grants the fused mixes {d} threads a group; every hyper-connection kernel runs at it\n", .{least});
        hck.hc_tg = @intCast(least);
    }
}

/// The one-threadgroup-a-row launch into `ov` at its pipeline's latched threads a group; unlatched, at `key.tg`, halved
/// on each size refusal until the GPU grants it (32 the last try). A dry recording learns no grant: it records at
/// `key.tg`, the width `probeForms` left every form granted, and a census notes the pipeline for its probe to latch.
fn applyNarrow(ov: *mlx.mlx_vector_array, form: Form, iv: mlx.mlx_vector_array, key: *Key, s: mlx.mlx_stream, pipe: Pipe, inputs: []const Arr) !void {
    if (narrow_tgs.get(pipe)) |tg| {
        key.tg = tg;
        return mlx.check(mlx.mlx_fast_metal_kernel_apply(ov, try kernel(form, false), iv, try cfgs.get(key.*, build), s));
    }
    if (mlx.engine.dryOn()) {
        if (census) |c| try c.note(pipe, inputs);
        return mlx.check(mlx.mlx_fast_metal_kernel_apply(ov, try kernel(form, false), iv, try cfgs.get(key.*, build), s));
    }
    defer mlx.expected_refusal = null;
    while (true) {
        // Only a halving left to take expects the refusal: the last one's is the caller's error.
        mlx.expected_refusal = if (key.tg > 32) "Thread group size" else null;
        if (mlx.mlx_fast_metal_kernel_apply(ov, try kernel(form, false), iv, try cfgs.get(key.*, build), s) == 0) {
            try narrow_tgs.put(gpa, pipe, key.tg);
            return;
        }
        if (key.tg <= 32 or !mlx.takeErrorIf("Thread group size")) return error.MlxError;
        log.warn("[engine] hy4 hc: this GPU does not grant the {s} mix {d} threads a group; its rows run at {d}\n", .{ @tagName(form), key.tg, @divExact(key.tg, 2) });
        key.tg = @divExact(key.tg, 2);
    }
}

/// `mixTiled` and `normIdentity` of the stream `x` [L, HC, D] f32 (`applies`): `post` [L, HC] and the normed row `h`
/// [L, D] f32. `eps` is `hc_kernels.identityEps`'s. The caller frees both.
pub fn mixNorm(s: mlx.mlx_stream, x: Arr, hw: hck.HcWeights, eps: Arr, norm_w: Arr) !Normed {
    return run(s, &.{ x, hw.fn_w, hw.scale, hw.base, eps, norm_w }, x, norm_w, hw.base, .plain, 0);
}

/// `mixExpandIdentity(h, y, post_in)` and `normIdentity` of its stream: also the expanded stream `m` [L, HC, D] f32.
/// The caller frees all three.
pub fn mixExpandNorm(s: mlx.mlx_stream, h: Arr, y: Arr, post_in: Arr, hw: hck.HcWeights, eps: Arr, norm_w: Arr) !Normed {
    return run(s, &.{ h, y, post_in, hw.fn_w, hw.scale, hw.base, eps, norm_w }, h, norm_w, hw.base, .expand, 0);
}

/// A layer's combine norm (`hy4_norm.rowsExpand(a, b, wn, combine_eps, h, post_in)`: its write-back into the streams
/// `h`) and the next layer's `mixNorm` of the streams it gives, in one launch: `m` is the streams, `post` and `h` the
/// next layer's. `a`, `b` [L, D] f32 or BF16, `wn` [D]; the rest as `mixExpandNorm`. The caller frees all three.
pub fn backMixNorm(s: mlx.mlx_stream, h: Arr, post_in: Arr, a: Arr, b: Arr, wn: Arr, combine_eps: f32, hw: hck.HcWeights, eps: Arr, norm_w: Arr) !Normed {
    return run(s, &.{ h, post_in, a, b, wn, hw.fn_w, hw.scale, hw.base, eps, norm_w }, h, norm_w, hw.base, .back, kcfg.bits(combine_eps));
}

/// `backMixNorm` reading the shared expert's down and `lat_up` as their unfinished GEMM rows and their `svh`: the two
/// finishes ride the mix's launch, their bytes.
pub fn backMixNormInner(s: mlx.mlx_stream, h: Arr, post_in: Arr, a: Arr, a_svh: Arr, b: Arr, b_svh: Arr, wn: Arr, combine_eps: f32, hw: hck.HcWeights, eps: Arr, norm_w: Arr) !Normed {
    return run(s, &.{ h, post_in, a, a_svh, b, b_svh, wn, hw.fn_w, hw.scale, hw.base, eps, norm_w }, h, norm_w, hw.base, .back_inner, kcfg.bits(combine_eps));
}

/// Whether `backMixNormInner` serves: `backMixNorm`'s shapes, `a` and `b` as [rows, D] unfinished rows with a D-wide
/// `svh`.
pub fn backInnerApplies(s: mlx.mlx_stream, h: Arr, post_in: Arr, a: Arr, a_svh: Arr, b: Arr, b_svh: Arr, wn: Arr, fn_w: Arr, scale: Arr) bool {
    if (!applies(s, h, fn_w, scale)) return false;
    const hs = mlx.getShape(h);
    const d = hs[2];
    const rows = [_]c_int{ hs[0], d };
    const lanes = [_]c_int{ hs[0], hs[1] };
    inline for (.{ a, b }) |r| {
        if (!std.mem.eql(c_int, mlx.getShape(r), &rows) or mlx.mlx_array_dtype(r) != .float16) return false;
    }
    inline for (.{ a_svh, b_svh }) |sv| {
        if (mlx.mlx_array_dtype(sv) != .float16 or mlx.mlx_array_size(sv) != @as(usize, @intCast(d))) return false;
    }
    const wd = mlx.mlx_array_dtype(wn);
    return @rem(d, 128) == 0 and d <= 3072 and std.mem.eql(c_int, mlx.getShape(post_in), &lanes) and mlx.mlx_array_dtype(post_in) == .float32 and
        mlx.getShape(wn).len == 1 and mlx.getShape(wn)[0] == d and (wd == .float32 or wd == .bfloat16);
}

/// The expand of `h` by the post mix and `a` normed by `wn`, then the next mix and its norm, in one launch: `a`'s norm
/// and the expand's write-back no longer launch apart.
pub fn expandNormMixNorm(s: mlx.mlx_stream, h: Arr, post_in: Arr, a: Arr, wn: Arr, norm_eps: f32, hw: hck.HcWeights, eps: Arr, norm_w: Arr) !Normed {
    return run(s, &.{ h, post_in, a, wn, hw.fn_w, hw.scale, hw.base, eps, norm_w }, h, norm_w, hw.base, .expand_norm, kcfg.bits(norm_eps));
}

/// Whether `expandNormMixNorm` takes these operands: `applies`' shapes, the attention rows `a` [T, d] f32 or BF16 with
/// their norm weight, the post mix [T, n] f32.
pub fn expandNormApplies(s: mlx.mlx_stream, h: Arr, post_in: Arr, a: Arr, wn: Arr, fn_w: Arr, scale: Arr) bool {
    if (!applies(s, h, fn_w, scale)) return false;
    const hs = mlx.getShape(h);
    const rows = [_]c_int{ hs[0], hs[2] };
    const lanes = [_]c_int{ hs[0], hs[1] };
    const ad = mlx.mlx_array_dtype(a);
    const wd = mlx.mlx_array_dtype(wn);
    return hs[2] <= 3072 and std.mem.eql(c_int, mlx.getShape(a), &rows) and (ad == .float32 or ad == .bfloat16) and
        std.mem.eql(c_int, mlx.getShape(post_in), &lanes) and mlx.mlx_array_dtype(post_in) == .float32 and
        mlx.getShape(wn).len == 1 and mlx.getShape(wn)[0] == hs[2] and (wd == .float32 or wd == .bfloat16);
}

/// `mixExpandNorm(h, y, post_in)` of `y = RMSNorm_ow(W_o's rows)` finished in the launch from that launch's k-split
/// partials `op` [L * osplit, D] f32 and its `osvh` [D] (f16), the norm over `ow` [D] with `o_eps` as
/// `rows_finish_rms_thru` takes it: the bytes of that finish and `mixExpandNorm`. The caller frees all three.
pub fn mixExpandNormParts(s: mlx.mlx_stream, h: Arr, post_in: Arr, op: Arr, osvh: Arr, osplit: c_int, ow: Arr, o_eps: f32, hw: hck.HcWeights, eps: Arr, norm_w: Arr) !Normed {
    if (!expandPartsApplies(s, h, post_in, op, osvh, osplit, ow, hw.fn_w, hw.scale)) return error.Hy4HcPartsShape;
    return runSplit(s, &.{ h, op, osvh, ow, post_in, hw.fn_w, hw.scale, hw.base, eps, norm_w }, h, norm_w, hw.base, .expand_parts, kcfg.bits(o_eps), osplit);
}

/// `mixExpandNormParts` of W_o's finished rows `oy` [L, D] f32 (the bytes its finish gives) in place of its partials:
/// the bytes of `rows_finish_rms_thru`'s norm over `ow` with `o_eps` and `mixExpandNorm`. The caller frees all three.
pub fn mixExpandNormRows(s: mlx.mlx_stream, h: Arr, post_in: Arr, oy: Arr, ow: Arr, o_eps: f32, hw: hck.HcWeights, eps: Arr, norm_w: Arr) !Normed {
    if (!expandRowsApplies(s, h, post_in, oy, ow, hw.fn_w, hw.scale)) return error.Hy4HcPartsShape;
    return run(s, &.{ h, oy, ow, post_in, hw.fn_w, hw.scale, hw.base, eps, norm_w }, h, norm_w, hw.base, .expand_rows, kcfg.bits(o_eps));
}

/// Whether `mixExpandNormRows` takes `h` [L, HC, D] (`applies`), `post_in` [L, HC] f32, `oy` [L, D] f32 and `ow` [D]
/// (f32 or BF16), D on the 128 grid within `t_y`'s 3,072-float cap.
pub fn expandRowsApplies(s: mlx.mlx_stream, h: Arr, post_in: Arr, oy: Arr, ow: Arr, fn_w: Arr, scale: Arr) bool {
    if (!applies(s, h, fn_w, scale)) return false;
    const hs = mlx.getShape(h);
    const d = hs[2];
    const lanes = [_]c_int{ hs[0], hs[1] };
    const rows = [_]c_int{ hs[0], d };
    const od = mlx.mlx_array_dtype(ow);
    return @rem(d, 128) == 0 and d <= 3072 and std.mem.eql(c_int, mlx.getShape(oy), &rows) and mlx.mlx_array_dtype(oy) == .float32 and
        mlx.mlx_array_size(ow) == @as(usize, @intCast(d)) and (od == .float32 or od == .bfloat16) and
        std.mem.eql(c_int, mlx.getShape(post_in), &lanes) and mlx.mlx_array_dtype(post_in) == .float32;
}

/// Whether `mixExpandNormParts` takes `h` [L, HC, D] (`applies`), `post_in` [L, HC] f32, `op` [L * osplit, D] f32,
/// `osvh` [D] f16 and `ow` [D] (f32 or BF16), D on the 128 grid within `t_y`'s 3,072-float cap.
pub fn expandPartsApplies(s: mlx.mlx_stream, h: Arr, post_in: Arr, op: Arr, osvh: Arr, osplit: c_int, ow: Arr, fn_w: Arr, scale: Arr) bool {
    if (!applies(s, h, fn_w, scale) or osplit < 1) return false;
    const hs = mlx.getShape(h);
    const d = hs[2];
    const lanes = [_]c_int{ hs[0], hs[1] };
    const parts = [_]c_int{ hs[0] * osplit, d };
    const od = mlx.mlx_array_dtype(ow);
    return @rem(d, 128) == 0 and d <= 3072 and std.mem.eql(c_int, mlx.getShape(op), &parts) and mlx.mlx_array_dtype(op) == .float32 and
        mlx.mlx_array_size(osvh) == @as(usize, @intCast(d)) and mlx.mlx_array_dtype(osvh) == .float16 and
        mlx.mlx_array_size(ow) == @as(usize, @intCast(d)) and (od == .float32 or od == .bfloat16) and
        std.mem.eql(c_int, mlx.getShape(post_in), &lanes) and mlx.mlx_array_dtype(post_in) == .float32;
}

/// Whether `backMixNorm` takes `h` [L, HC, D] (`applies`), `post_in` [L, HC] f32, the rows `a` and `b` [L, D] and the
/// norm weight `wn` [D] (f32 or BF16, D within `t_y`'s 3,072-float cap).
pub fn backApplies(s: mlx.mlx_stream, h: Arr, post_in: Arr, a: Arr, b: Arr, wn: Arr, fn_w: Arr, scale: Arr) bool {
    if (!applies(s, h, fn_w, scale)) return false;
    const hs = mlx.getShape(h);
    const rows = [_]c_int{ hs[0], hs[2] };
    const lanes = [_]c_int{ hs[0], hs[1] };
    inline for (.{ a, b }) |r| {
        const dt = mlx.mlx_array_dtype(r);
        if (!std.mem.eql(c_int, mlx.getShape(r), &rows) or (dt != .float32 and dt != .bfloat16)) return false;
    }
    const wd = mlx.mlx_array_dtype(wn);
    return hs[2] <= 3072 and std.mem.eql(c_int, mlx.getShape(post_in), &lanes) and mlx.mlx_array_dtype(post_in) == .float32 and
        mlx.getShape(wn).len == 1 and mlx.getShape(wn)[0] == hs[2] and (wd == .float32 or wd == .bfloat16);
}

// ── tests ────────────────────────────────────────────────────────────────────────────────────────

const testing = std.testing;
const Ops = @import("../mlx_ops.zig").Ops;

fn randomArr(o: *Ops, shape: []const c_int, dt: mlx.mlx_dtype, seed: u64, sc: f64) !Arr {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const v = try testing.allocator.alloc(f32, n);
    defer testing.allocator.free(v);
    var r = std.Random.DefaultPrng.init(seed);
    for (v) |*d| d.* = @floatCast(r.random().floatNorm(f64) * sc);
    return o.cast(try o.own(mlx.mlx_array_new_data(v.ptr, shape.ptr, @intCast(shape.len), .float32)), dt);
}

fn bytesOf(o: *Ops, a: Arr) ![]const u8 {
    const c = try o.contiguous(a);
    try mlx.check(mlx.mlx_array_eval(c));
    return std.mem.sliceAsBytes((mlx.mlx_array_data_float32(c) orelse return error.HostRead)[0..mlx.mlx_array_size(c)]);
}

test "hy4 hc: a decode row's mix and its collapse and norm in one launch give the bytes of the tiled mix and the identity norm after it, the expanded stream's too, at the release's 4 streams of 3,072 and the tiny's, one row and six" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const tg_was = hck.hc_tg;
    defer hck.hc_tg = tg_was;
    for ([_]c_int{ 1, 6 }) |t| for ([_]c_int{ 3072, 128 }) |d| {
        try probeForms(es, d);
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const seed: u64 = @intCast(3 * t + d);
        const h = try o.contiguous(try randomArr(&o, &.{ t, 4, d }, .float32, seed, 1));
        const y = try o.contiguous(try randomArr(&o, &.{ t, d }, .float32, seed + 1, 1));
        const pin = try o.contiguous(try randomArr(&o, &.{ t, 4 }, .float32, seed + 2, 1));
        const hw: hck.HcWeights = .{
            .fn_w = try o.contiguous(try randomArr(&o, &.{ 8, 4 * d }, .float32, seed + 3, 0.05)),
            .scale = try o.contiguous(try randomArr(&o, &.{3}, .float32, seed + 4, 1)),
            .base = try o.contiguous(try randomArr(&o, &.{8}, .float32, seed + 5, 1)),
        };
        const w = try o.contiguous(try randomArr(&o, &.{d}, .bfloat16, seed + 6, 0.3));
        const eps = try o.own(try hck.identityEps(1e-6, 1e-6, 2.0));
        try testing.expect(applies(es, h, hw.fn_w, hw.scale));
        // The plain mix.
        const z = try o.own(try hck.mixTiled(es, h, hw, eps, hck.hc_mix_tile));
        const want = try hck.normIdentity(es, z, h, w, eps);
        _ = try o.own(want.pre);
        const wp = try bytesOf(&o, try o.own(want.post));
        const wh = try bytesOf(&o, try o.own(want.h));
        const before = mlx.engine.stats().dispatches;
        const got = try mixNorm(es, h, hw, eps, w);
        const gp = try o.own(got.post);
        const gh = try o.own(got.h);
        try mlx.check(mlx.mlx_array_eval(gh));
        try testing.expectEqual(@as(u64, 1), mlx.engine.stats().dispatches - before);
        try testing.expectEqualSlices(u8, wp, try bytesOf(&o, gp));
        try testing.expectEqualSlices(u8, wh, try bytesOf(&o, gh));
        // The mix over the expanded stream.
        const ex = try hck.mixExpandIdentity(es, h, y, pin, hw, eps, hck.hc_mix_tile);
        const em = try bytesOf(&o, try o.own(ex.m));
        const wx = try hck.normIdentity(es, try o.own(ex.z), ex.m, w, eps);
        _ = try o.own(wx.pre);
        const xp = try bytesOf(&o, try o.own(wx.post));
        const xh = try bytesOf(&o, try o.own(wx.h));
        const gx = try mixExpandNorm(es, h, y, pin, hw, eps, w);
        try testing.expectEqualSlices(u8, xp, try bytesOf(&o, try o.own(gx.post)));
        try testing.expectEqualSlices(u8, xh, try bytesOf(&o, try o.own(gx.h)));
        try testing.expectEqualSlices(u8, em, try bytesOf(&o, try o.own(gx.m.?)));
    };
}

test "hy4 hc: a layer's combine norm and write-back and the next layer's mix and norm in one launch give the bytes of rowsExpand and the mix after it, at the release's 4 streams of 3,072 and the tiny's, one row and six, 1,024 lanes and 512" {
    try mlx.requireGpu();
    const hy4_norm = @import("hy4_norm.zig");
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const tg_was = hck.hc_tg;
    defer hck.hc_tg = tg_was;
    for ([_]usize{ 1024, 512 }) |tg| for ([_]c_int{ 1, 6 }) |t| for ([_]c_int{ 3072, 128 }) |d| {
        hck.hc_tg = tg;
        // A GPU that grants the fused mixes less (an M1) runs every kernel of the case at what it grants.
        try probeForms(es, d);
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const seed: u64 = @intCast(7 * t + d + @as(c_int, @intCast(tg)));
        const h = try o.contiguous(try randomArr(&o, &.{ t, 4, d }, .float32, seed, 1));
        const up = try o.contiguous(try randomArr(&o, &.{ t, d }, .float32, seed + 1, 0.7));
        const shared = try o.contiguous(try randomArr(&o, &.{ t, d }, .float32, seed + 2, 0.4));
        const pin = try o.contiguous(try randomArr(&o, &.{ t, 4 }, .float32, seed + 3, 1));
        const wn = try o.contiguous(try randomArr(&o, &.{d}, .bfloat16, seed + 4, 0.3));
        const hw: hck.HcWeights = .{
            .fn_w = try o.contiguous(try randomArr(&o, &.{ 8, 4 * d }, .float32, seed + 5, 0.05)),
            .scale = try o.contiguous(try randomArr(&o, &.{3}, .float32, seed + 6, 1)),
            .base = try o.contiguous(try randomArr(&o, &.{8}, .float32, seed + 7, 1)),
        };
        const w = try o.contiguous(try randomArr(&o, &.{d}, .bfloat16, seed + 8, 0.3));
        const eps = try o.own(try hck.identityEps(1e-5, 1e-6, 2.0));
        try testing.expect(backApplies(es, h, pin, up, shared, wn, hw.fn_w, hw.scale));
        const m = try o.own(try hy4_norm.rowsExpand(es, up, shared, wn, 1e-5, h, pin));
        const want = try mixNorm(es, m, hw, eps, w);
        const wm = try bytesOf(&o, m);
        const wp = try bytesOf(&o, try o.own(want.post));
        const wh = try bytesOf(&o, try o.own(want.h));
        const before = mlx.engine.stats().dispatches;
        const got = try backMixNorm(es, h, pin, up, shared, wn, 1e-5, hw, eps, w);
        const gm = try o.own(got.m.?);
        const gp = try o.own(got.post);
        const gh = try o.own(got.h);
        try mlx.check(mlx.mlx_array_eval(gh));
        try testing.expectEqual(@as(u64, 1), mlx.engine.stats().dispatches - before);
        try testing.expectEqualSlices(u8, wm, try bytesOf(&o, gm));
        try testing.expectEqualSlices(u8, wp, try bytesOf(&o, gp));
        try testing.expectEqualSlices(u8, wh, try bytesOf(&o, gh));
    };
}

/// The inputs of every form at `t` rows of 4 streams of `d`, W_o's partials `osplit` a row.
const FormCase = struct {
    h: Arr,
    y: Arr,
    pin: Arr,
    up: Arr,
    shared: Arr,
    up_in: Arr,
    shared_in: Arr,
    wn: Arr,
    op: Arr,
    osvh: Arr,
    ow: Arr,
    osplit: c_int,
    hw: hck.HcWeights,
    w: Arr,
    eps: Arr,

    fn init(o: *Ops, t: c_int, d: c_int, osplit: c_int, seed: u64) !FormCase {
        return .{
            .h = try o.contiguous(try randomArr(o, &.{ t, 4, d }, .float32, seed, 1)),
            .y = try o.contiguous(try randomArr(o, &.{ t, d }, .float32, seed + 1, 1)),
            .pin = try o.contiguous(try randomArr(o, &.{ t, 4 }, .float32, seed + 2, 1)),
            .up = try o.contiguous(try randomArr(o, &.{ t, d }, .float32, seed + 3, 0.7)),
            .shared = try o.contiguous(try randomArr(o, &.{ t, d }, .bfloat16, seed + 4, 0.4)),
            .up_in = try o.contiguous(try randomArr(o, &.{ t, d }, .float16, seed + 13, 0.7)),
            .shared_in = try o.contiguous(try randomArr(o, &.{ t, d }, .float16, seed + 14, 0.4)),
            .wn = try o.contiguous(try randomArr(o, &.{d}, .bfloat16, seed + 5, 0.3)),
            .op = try o.contiguous(try randomArr(o, &.{ t * osplit, d }, .float32, seed + 6, 1)),
            .osvh = try o.contiguous(try randomArr(o, &.{d}, .float16, seed + 7, 0.5)),
            .ow = try o.contiguous(try randomArr(o, &.{d}, .bfloat16, seed + 8, 0.3)),
            .osplit = osplit,
            .hw = .{
                .fn_w = try o.contiguous(try randomArr(o, &.{ 8, 4 * d }, .float32, seed + 9, 0.05)),
                .scale = try o.contiguous(try randomArr(o, &.{3}, .float32, seed + 10, 1)),
                .base = try o.contiguous(try randomArr(o, &.{8}, .float32, seed + 11, 1)),
            },
            .w = try o.contiguous(try randomArr(o, &.{d}, .bfloat16, seed + 12, 0.3)),
            .eps = try o.own(try hck.identityEps(1e-5, 1e-6, 2.0)),
        };
    }

    /// `init`'s shapes and dtypes filled with zeros: what a grant reads, outside a test.
    fn zeros(o: *Ops, t: c_int, d: c_int, osplit: c_int) !FormCase {
        const z = struct {
            fn of(od: *Ops, shape: []const c_int, dt: mlx.mlx_dtype) !Arr {
                return od.contiguous(try od.zeros(shape, dt));
            }
        }.of;
        return .{
            .h = try z(o, &.{ t, 4, d }, .float32),
            .y = try z(o, &.{ t, d }, .float32),
            .pin = try z(o, &.{ t, 4 }, .float32),
            .up = try z(o, &.{ t, d }, .float32),
            .shared = try z(o, &.{ t, d }, .bfloat16),
            .up_in = try z(o, &.{ t, d }, .float16),
            .shared_in = try z(o, &.{ t, d }, .float16),
            .wn = try z(o, &.{d}, .bfloat16),
            .op = try z(o, &.{ t * osplit, d }, .float32),
            .osvh = try z(o, &.{d}, .float16),
            .ow = try z(o, &.{d}, .bfloat16),
            .osplit = osplit,
            .hw = .{ .fn_w = try z(o, &.{ 8, 4 * d }, .float32), .scale = try z(o, &.{3}, .float32), .base = try z(o, &.{8}, .float32) },
            .w = try z(o, &.{d}, .bfloat16),
            .eps = try o.own(try hck.identityEps(1e-5, 1e-6, 2.0)),
        };
    }

    /// The form's launch, its outputs owned by `o`.
    fn run(c: FormCase, o: *Ops, form: Form) !Normed {
        const r = switch (form) {
            .plain => try mixNorm(o.s, c.h, c.hw, c.eps, c.w),
            .expand => try mixExpandNorm(o.s, c.h, c.y, c.pin, c.hw, c.eps, c.w),
            .back => try backMixNorm(o.s, c.h, c.pin, c.up, c.shared, c.wn, 1e-5, c.hw, c.eps, c.w),
            .back_inner => try backMixNormInner(o.s, c.h, c.pin, c.up_in, c.osvh, c.shared_in, c.osvh, c.wn, 1e-5, c.hw, c.eps, c.w),
            .expand_norm => try expandNormMixNorm(o.s, c.h, c.pin, c.y, c.wn, 1e-5, c.hw, c.eps, c.w),
            .expand_parts => try mixExpandNormParts(o.s, c.h, c.pin, c.op, c.osvh, c.osplit, c.ow, 1e-5, c.hw, c.eps, c.w),
            .expand_rows => try mixExpandNormRows(o.s, c.h, c.pin, c.y, c.ow, 1e-5, c.hw, c.eps, c.w),
        };
        return .{ .post = try o.own(r.post), .h = try o.own(r.h), .m = if (r.m) |m| try o.own(m) else null };
    }
};

test "hy4 hc: the wide launch gives the one-threadgroup launch's bytes for every form, one launch a site, at the release's 4 streams of 3,072 and the tiny's, one row, three (a padded grid) and eight, 1,024 lanes and 512" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const tg_was = hck.hc_tg;
    defer hck.hc_tg = tg_was;
    defer wide_on = true;
    forgetGrants();
    defer forgetGrants();
    for ([_]usize{ 1024, 512 }) |tg| for ([_]c_int{ 1, 3, 8 }) |t| for ([_]c_int{ 3072, 128 }) |d| {
        hck.hc_tg = tg;
        // A GPU that grants the fused mixes less (an M1) runs every kernel of the case at what it grants.
        try probeForms(es, d);
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const osplit: c_int = if (d == 3072) 32 else 16;
        const c = try FormCase.init(&o, t, d, osplit, @intCast(11 * t + d + @as(c_int, @intCast(tg))));
        try testing.expect(takesWide(es, t, 4));
        for ([_]Form{ .plain, .expand, .back, .expand_parts, .expand_rows }) |form| {
            wide_on = false;
            const want = try c.run(&o, form);
            const wp = try bytesOf(&o, want.post);
            const wh = try bytesOf(&o, want.h);
            wide_on = true;
            const before = mlx.engine.stats().dispatches;
            const wide_before = wide_launches;
            const got = try c.run(&o, form);
            try mlx.check(mlx.mlx_array_eval(got.h));
            try testing.expectEqual(@as(u64, 1), mlx.engine.stats().dispatches - before);
            // A form whose wide pipeline this GPU refuses at this width (an M1 at 1,024 lanes) runs one threadgroup a row.
            if (wide_launches == wide_before) try testing.expect(refusedGrants() > 0) else try testing.expectEqual(wide_before + 1, wide_launches);
            try testing.expectEqualSlices(u8, wp, try bytesOf(&o, got.post));
            try testing.expectEqualSlices(u8, wh, try bytesOf(&o, got.h));
            if (want.m) |m| try testing.expectEqualSlices(u8, try bytesOf(&o, m), try bytesOf(&o, got.m.?));
        }
    };
}

test "hy4 hc: where the GPU grants the wide mix fewer threads a group than hc_tg, each row runs one threadgroup with the same bytes, from then on" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const kn = mlx.engine.kernel;
    forgetGrants();
    defer {
        kn.grant_cap = std.math.maxInt(usize);
        kn.grant_cap_name = "";
        forgetGrants();
        wide_on = true;
    }
    var o: Ops = .{ .s = es };
    defer o.deinit();
    const c = try FormCase.init(&o, 1, 3072, 32, 5);
    wide_on = false;
    const want = try c.run(&o, .plain);
    wide_on = true;
    kn.grant_cap = hck.hc_tg / 2;
    kn.grant_cap_name = "sushi_hy4_hc_mix_norm_wide";
    for (0..2) |_| {
        const before = mlx.engine.stats().dispatches;
        const got = try c.run(&o, .plain);
        try mlx.check(mlx.mlx_array_eval(got.h));
        try testing.expectEqual(@as(u64, 1), mlx.engine.stats().dispatches - before);
        try testing.expect(grants.count() == 1 and refusedGrants() == 1 and !mlx.errorPending());
        try testing.expectEqualSlices(u8, try bytesOf(&o, want.post), try bytesOf(&o, got.post));
        try testing.expectEqualSlices(u8, try bytesOf(&o, want.h), try bytesOf(&o, got.h));
    }
}

test "hy4 hc: where the GPU grants a form's one-threadgroup launch fewer threads than hc_tg too, its rows run at the widest halving it grants, with that width's bytes, from then on" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const kn = mlx.engine.kernel;
    const tg_was = hck.hc_tg;
    defer {
        kn.grant_cap = std.math.maxInt(usize);
        kn.grant_cap_name = "";
        forgetGrants();
        wide_on = true;
        hck.hc_tg = tg_was;
    }
    var o: Ops = .{ .s = es };
    defer o.deinit();
    const c = try FormCase.init(&o, 1, 3072, 32, 11);
    const rung = hck.hc_tg / 4;
    for ([_]Form{ .plain, .expand, .back, .expand_norm, .expand_parts, .expand_rows }) |form| {
        // Both widths of every form are capped, as an M1 caps a register-heavy pipeline (640, 384) below 1024.
        forgetGrants();
        kn.grant_cap = rung;
        kn.grant_cap_name = "sushi_hy4_hc_";
        var got: [2]Normed = undefined;
        for (&got) |*g| {
            g.* = try c.run(&o, form);
            try mlx.check(mlx.mlx_array_eval(g.h));
            try testing.expect(!mlx.errorPending());
        }
        try testing.expectEqual(@as(usize, 1), narrow_tgs.count());
        var latched = narrow_tgs.valueIterator();
        try testing.expectEqual(@as(c_int, @intCast(rung)), latched.next().?.*);
        // What the form's one-threadgroup launch gives at that width on a GPU that grants it.
        kn.grant_cap = std.math.maxInt(usize);
        kn.grant_cap_name = "";
        forgetGrants();
        wide_on = false;
        hck.hc_tg = rung;
        const want = try c.run(&o, form);
        hck.hc_tg = tg_was;
        wide_on = true;
        for (got) |g| {
            try testing.expectEqualSlices(u8, try bytesOf(&o, want.post), try bytesOf(&o, g.post));
            try testing.expectEqualSlices(u8, try bytesOf(&o, want.h), try bytesOf(&o, g.h));
            if (want.m) |m| try testing.expectEqualSlices(u8, try bytesOf(&o, m), try bytesOf(&o, g.m.?));
        }
    }
}

test "hy4 hc: a dry recording that meets a one-threadgroup launch no probe latched (a prefill chunk's rows past rows_max) records it at the width the load's probe left, census or none, and plans what the real launch runs" {
    try mlx.requireGpu();
    const core = @import("../tensor/core.zig");
    const plan = @import("../tensor/plan/plan_api.zig");
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const tg_was = hck.hc_tg;
    defer hck.hc_tg = tg_was;
    defer forgetGrants();
    // The load's probe: hc_tg is then a width this GPU grants every form.
    try probeForms(es, 3072);
    var o: Ops = .{ .s = es };
    defer o.deinit();
    const c = try FormCase.init(&o, 16, 3072, 32, 13);
    try mlx.check(mlx.mlx_array_eval(c.h));
    try testing.expect(!takesWide(es, 16, 4));
    for ([_]Form{ .plain, .expand, .back }) |form| {
        forgetGrants();
        var dd: plan.Driver = .{ .gpa = testing.allocator };
        defer dd.deinit();
        var dry = try plan.Recorder.begin(testing.allocator, .{ .site = 1 });
        defer dry.deinit();
        {
            try core.beginRecording(.{ .dry = .{ .driver = &dd } }, &dry);
            defer core.endRecording() catch {};
            var od: Ops = .{ .s = es };
            defer od.deinit();
            _ = try c.run(&od, form);
        }
        var real = try plan.Recorder.begin(testing.allocator, .{ .site = 1 });
        defer real.deinit();
        {
            try core.beginRecording(.real, &real);
            defer core.endRecording() catch {};
            var od: Ops = .{ .s = es };
            defer od.deinit();
            const got = try c.run(&od, form);
            try mlx.check(mlx.mlx_array_eval(got.h));
        }
        try testing.expectEqual(@as(?plan.trace_mod.Difference, null), plan.trace_mod.firstDifference(real.instances.items, dry.instances.items));
    }
}

/// The grants latched off.
fn refusedGrants() usize {
    var n: usize = 0;
    var it = grants.valueIterator();
    while (it.next()) |g| n += @intFromBool(!g.*);
    return n;
}

test "hy4 hc: a GPU that refuses one form's wide pipeline: a dry recording with no grant is refused by name, a census notes each form's pipeline, the probe latches each grant, and each form then plans what it runs, the granted ones wide" {
    try mlx.requireGpu();
    const core = @import("../tensor/core.zig");
    const plan = @import("../tensor/plan/plan_api.zig");
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const kn = mlx.engine.kernel;
    defer {
        kn.grant_cap = std.math.maxInt(usize);
        kn.grant_cap_name = "";
        forgetGrants();
    }
    const forms = [_]Form{ .plain, .expand, .back, .expand_norm, .expand_parts, .expand_rows };
    var o: Ops = .{ .s = es };
    defer o.deinit();
    const c = try FormCase.init(&o, 1, 3072, 32, 7);
    try mlx.check(mlx.mlx_array_eval(c.h));
    // The forms whose wide pipeline this GPU itself refuses (an M1 refuses the heavier ones at 1,024 lanes).
    var own: [forms.len]bool = undefined;
    for (forms, &own) |form, *r| {
        forgetGrants();
        var od: Ops = .{ .s = es };
        defer od.deinit();
        const got = try c.run(&od, form);
        try mlx.check(mlx.mlx_array_eval(got.h));
        r.* = refusedGrants() > 0;
    }
    const Rec = struct {
        r: plan.Recorder,
        d: plan.Driver = .{ .gpa = testing.allocator },
        fn deinit(x: *@This()) void {
            x.r.deinit();
            x.d.deinit();
        }
        /// `form`'s launch recorded `dry` or real into `x`, its outputs owned by `od`.
        fn run(x: *@This(), cs: FormCase, od: *Ops, form: Form, dry: bool) !void {
            x.r = try plan.Recorder.begin(testing.allocator, .{ .site = 1 });
            try core.beginRecording(if (dry) .{ .dry = .{ .driver = &x.d } } else .real, &x.r);
            defer core.endRecording() catch {};
            const got = try cs.run(od, form);
            if (!dry) try mlx.check(mlx.mlx_array_eval(got.h));
        }
    };
    for ([_]Form{ .expand, .expand_parts }) |capped| {
        forgetGrants();
        kn.grant_cap = hck.hc_tg / 2;
        kn.grant_cap_name = switch (capped) {
            .expand => "sushi_hy4_hc_mix_expand_norm_wide",
            else => "sushi_hy4_hc_expand_parts_mix_norm_wide",
        };
        // No grant and no census: refused by name.
        for (forms) |form| {
            var x: Rec = .{ .r = undefined };
            defer x.deinit();
            var od: Ops = .{ .s = es };
            defer od.deinit();
            try testing.expectError(error.Hy4HcWideUnprobed, x.run(c, &od, form, true));
        }
        // The census notes each form's pipeline once; the probe latches each grant, the capped form's off.
        var cs: Census = .{};
        defer cs.deinit();
        {
            census = &cs;
            defer census = null;
            for (0..2) |_| for (forms) |form| {
                var x: Rec = .{ .r = undefined };
                defer x.deinit();
                var od: Ops = .{ .s = es };
                defer od.deinit();
                try x.run(c, &od, form, true);
            };
        }
        // Each form's wide pipeline, and the one-threadgroup pipeline of a form whose wide one is refused.
        var refused: [forms.len]bool = undefined;
        for (forms, own, &refused) |form, o_r, *r| r.* = o_r or form == capped;
        var wide_notes: usize = 0;
        for (cs.noted.items) |n| {
            if (n.pipe.wide) wide_notes += 1 else try testing.expect(refused[std.mem.indexOfScalar(Form, &forms, n.pipe.form).?]);
        }
        try testing.expectEqual(forms.len, wide_notes);
        try probeNoted(es, &cs);
        try testing.expectEqual(forms.len, grants.count());
        try testing.expectEqual(std.mem.count(bool, &refused, &.{true}), refusedGrants());
        for (forms, refused) |form, was_refused| {
            var dry: Rec = .{ .r = undefined };
            defer dry.deinit();
            var real: Rec = .{ .r = undefined };
            defer real.deinit();
            var od: Ops = .{ .s = es };
            defer od.deinit();
            try dry.run(c, &od, form, true);
            const before = wide_launches;
            try real.run(c, &od, form, false);
            try testing.expectEqual(!was_refused, wide_launches > before);
            try testing.expectEqual(@as(?plan.trace_mod.Difference, null), plan.trace_mod.firstDifference(real.r.instances.items, dry.r.instances.items));
        }
    }
}


test "hy4 hc: a census refuses by name an input it cannot note (more inputs than it holds, or a rank past three), and notes nothing" {
    try mlx.requireGpu();
    const one = [_]f32{0};
    const flat = try mlx.newData(&one, &[_]c_int{1}, .float32);
    defer _ = mlx.mlx_array_free(flat);
    const deep = try mlx.newData(&one, &[_]c_int{ 1, 1, 1, 1 }, .float32);
    defer _ = mlx.mlx_array_free(deep);
    var cs: Census = .{};
    defer cs.deinit();
    const pipe: Pipe = .{ .form = .plain, .wide = true, .hc = 4, .d = 1, .tg = 1, .eps = 0, .osplit = 0, .dts = @splat(0xff) };
    var many: [max_ins + 1]Arr = @splat(flat);
    try testing.expectError(error.Hy4HcCensusInputs, cs.note(pipe, &many));
    try testing.expectError(error.Hy4HcCensusInputs, cs.note(pipe, &.{ flat, deep }));
    try testing.expectEqual(@as(usize, 0), cs.noted.items.len);
}

test "hy4 hc: prefill rows past rows_max run plain, expand and back in one launch each, byte for byte" {
    try mlx.requireGpu();
    const hy4_norm = @import("hy4_norm.zig");
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const tg_was = hck.hc_tg;
    defer hck.hc_tg = tg_was;
    for ([_]struct { t: c_int, d: c_int }{ .{ .t = 33, .d = 128 }, .{ .t = 64, .d = 3072 } }) |cs| {
        try probeForms(es, cs.d);
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const seed: u64 = @intCast(11 * cs.t + cs.d);
        const h = try o.contiguous(try randomArr(&o, &.{ cs.t, 4, cs.d }, .float32, seed, 1));
        const y = try o.contiguous(try randomArr(&o, &.{ cs.t, cs.d }, .float32, seed + 1, 1));
        const pin = try o.contiguous(try randomArr(&o, &.{ cs.t, 4 }, .float32, seed + 2, 1));
        const up = try o.contiguous(try randomArr(&o, &.{ cs.t, cs.d }, .bfloat16, seed + 3, 0.7));
        const shared = try o.contiguous(try randomArr(&o, &.{ cs.t, cs.d }, .bfloat16, seed + 4, 0.4));
        const wn = try o.contiguous(try randomArr(&o, &.{cs.d}, .bfloat16, seed + 5, 0.3));
        const hw: hck.HcWeights = .{
            .fn_w = try o.contiguous(try randomArr(&o, &.{ 8, 4 * cs.d }, .float32, seed + 6, 0.05)),
            .scale = try o.contiguous(try randomArr(&o, &.{3}, .float32, seed + 7, 1)),
            .base = try o.contiguous(try randomArr(&o, &.{8}, .float32, seed + 8, 1)),
        };
        const w = try o.contiguous(try randomArr(&o, &.{cs.d}, .bfloat16, seed + 9, 0.3));
        const eps = try o.own(try hck.identityEps(1e-5, 1e-6, 2.0));
        try testing.expect(applies(es, h, hw.fn_w, hw.scale));
        try testing.expect(backApplies(es, h, pin, up, shared, wn, hw.fn_w, hw.scale));

        const z = try o.own(try hck.mixTiled(es, h, hw, eps, hck.hc_mix_tile));
        const want = try hck.normIdentity(es, z, h, w, eps);
        _ = try o.own(want.pre);
        const wp = try bytesOf(&o, try o.own(want.post));
        const wh = try bytesOf(&o, try o.own(want.h));
        const b0 = mlx.engine.stats().dispatches;
        const got = try mixNorm(es, h, hw, eps, w);
        const gp = try o.own(got.post);
        const gh = try o.own(got.h);
        try mlx.check(mlx.mlx_array_eval(gh));
        try testing.expectEqual(@as(u64, 1), mlx.engine.stats().dispatches - b0);
        try testing.expectEqualSlices(u8, wp, try bytesOf(&o, gp));
        try testing.expectEqualSlices(u8, wh, try bytesOf(&o, gh));

        const ex = try hck.mixExpandIdentity(es, h, y, pin, hw, eps, hck.hc_mix_tile);
        const em = try bytesOf(&o, try o.own(ex.m));
        const wx = try hck.normIdentity(es, try o.own(ex.z), ex.m, w, eps);
        _ = try o.own(wx.pre);
        const xp = try bytesOf(&o, try o.own(wx.post));
        const xh = try bytesOf(&o, try o.own(wx.h));
        const b1 = mlx.engine.stats().dispatches;
        const gx = try mixExpandNorm(es, h, y, pin, hw, eps, w);
        const gxm = try o.own(gx.m.?);
        const gxp = try o.own(gx.post);
        const gxh = try o.own(gx.h);
        try mlx.check(mlx.mlx_array_eval(gxh));
        try testing.expectEqual(@as(u64, 1), mlx.engine.stats().dispatches - b1);
        try testing.expectEqualSlices(u8, xp, try bytesOf(&o, gxp));
        try testing.expectEqualSlices(u8, xh, try bytesOf(&o, gxh));
        try testing.expectEqualSlices(u8, em, try bytesOf(&o, gxm));

        const bm = try o.own(try hy4_norm.rowsExpand(es, up, shared, wn, 1e-5, h, pin));
        const bw = try mixNorm(es, bm, hw, eps, w);
        const bwm = try bytesOf(&o, bm);
        const bwp = try bytesOf(&o, try o.own(bw.post));
        const bwh = try bytesOf(&o, try o.own(bw.h));
        const b2 = mlx.engine.stats().dispatches;
        const bg = try backMixNorm(es, h, pin, up, shared, wn, 1e-5, hw, eps, w);
        const bgm = try o.own(bg.m.?);
        const bgp = try o.own(bg.post);
        const bgh = try o.own(bg.h);
        try mlx.check(mlx.mlx_array_eval(bgh));
        try testing.expectEqual(@as(u64, 1), mlx.engine.stats().dispatches - b2);
        try testing.expectEqualSlices(u8, bwm, try bytesOf(&o, bgm));
        try testing.expectEqualSlices(u8, bwp, try bytesOf(&o, bgp));
        try testing.expectEqualSlices(u8, bwh, try bytesOf(&o, bgh));
    }
}

test "hy4 hc: expandNormMixNorm norms a single bf16 sublayer output, expands and mixes in one launch, byte for byte" {
    try mlx.requireGpu();
    const hy4_norm = @import("hy4_norm.zig");
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    var o: Ops = .{ .s = es };
    defer o.deinit();
    const t: c_int = 64;
    const d: c_int = 3072;
    const h = try o.contiguous(try randomArr(&o, &.{ t, 4, d }, .float32, 301, 1));
    const y_raw = try o.contiguous(try randomArr(&o, &.{ t, d }, .bfloat16, 302, 0.8));
    const pin = try o.contiguous(try randomArr(&o, &.{ t, 4 }, .float32, 303, 1));
    const wn = try o.contiguous(try randomArr(&o, &.{d}, .bfloat16, 304, 0.3));
    const hw: hck.HcWeights = .{
        .fn_w = try o.contiguous(try randomArr(&o, &.{ 8, 4 * d }, .float32, 305, 0.05)),
        .scale = try o.contiguous(try randomArr(&o, &.{3}, .float32, 306, 1)),
        .base = try o.contiguous(try randomArr(&o, &.{8}, .float32, 307, 1)),
    };
    const w = try o.contiguous(try randomArr(&o, &.{d}, .bfloat16, 308, 0.3));
    const eps = try o.own(try hck.identityEps(1e-5, 1e-6, 2.0));
    try testing.expect(expandNormApplies(es, h, pin, y_raw, wn, hw.fn_w, hw.scale));

    const y_norm = try o.own(try hy4_norm.rows(es, y_raw, null, wn, 1e-5, .float32));
    const want = try mixExpandNorm(es, h, y_norm, pin, hw, eps, w);
    const wm = try bytesOf(&o, try o.own(want.m.?));
    const wp = try bytesOf(&o, try o.own(want.post));
    const wh = try bytesOf(&o, try o.own(want.h));

    try mlx.check(mlx.mlx_array_eval(y_raw));
    const before = mlx.engine.stats().dispatches;
    const got = try expandNormMixNorm(es, h, pin, y_raw, wn, 1e-5, hw, eps, w);
    const gm = try o.own(got.m.?);
    const gp = try o.own(got.post);
    const gh = try o.own(got.h);
    try mlx.check(mlx.mlx_array_eval(gh));
    try testing.expectEqual(@as(u64, 1), mlx.engine.stats().dispatches - before);
    try testing.expectEqualSlices(u8, wm, try bytesOf(&o, gm));
    try testing.expectEqualSlices(u8, wp, try bytesOf(&o, gp));
    try testing.expectEqualSlices(u8, wh, try bytesOf(&o, gh));
}
