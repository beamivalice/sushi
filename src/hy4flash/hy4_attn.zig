//! Hy4-Flash's prefill attention core in one launch a span: the absorbed query's scores over a row's keys, the sink
//! softmax and the value sum, on the matrix units. A row's keys are a range of a key array (a sliding window, or every
//! key below the selection threshold) or the rows it picked from the layer's store, read where they lie: no gathered or
//! widened copy of them is made. The scores take the f32 query as three bf16 parts (f32's range and 24 bits), the value
//! sum the softmax weights P (in [0, 1]) as two; the keys are bf16, so exact; the softmax and the sums stay f32.
const std = @import("std");
const mlx = @import("../mlx.zig");
const metal_def = @import("../metal_def.zig");
const nax_gate = @import("../nax_gate.zig");
const kcfg = @import("hy4_kcfg.zig");
const Arr = mlx.mlx_array;

/// Query columns of one matrix-unit fragment: the heads of a head block (padded past the layer's heads).
const hb = 16;
/// Keys a tile takes on NAX (`matrix_g17`).
const kt = 32;
/// Keys a tile takes on non-NAX SIMD (`simdgroup_float8x8`).
const simd_kt = 16;
/// Channels a latent simdgroup owns on NAX (one affine group of the kv8 store); the last simdgroup owns the rope channels.
const slice = 64;
/// Channels a latent simdgroup owns on non-NAX SIMD (`DT=4`, halving register footprint for high occupancy).
const simd_slice = 32;
const head: [:0]const u8 =
    \\const uint tid = thread_index_in_threadgroup, sg = simdgroup_index_in_threadgroup;
    \\const ushort lane = thread_index_in_simdgroup;
    \\const int l = threadgroup_position_in_grid.y;
    \\const int h0 = threadgroup_position_in_grid.x * HB;
    \\const int H = ql_shape[0];
    \\const int p0 = meta[0], window = meta[1], N = meta[2], S = meta[3], row0 = meta[5];
    \\const float scale = as_type<float>(meta[4]);
    \\const int NL = L / 64;
    \\constexpr int QS = 2, NK = KT / 16, NJ = KT / 16;
    \\const bool rsg = int(sg) == NL, work = int(sg) < NSG;
    \\const int d0 = rsg ? 0 : int(sg) * 64;
    \\threadgroup float sp[NSG * HB * KT];
    \\threadgroup bfloat16_t ph[HB * KT], pl[HB * KT];
    \\threadgroup float cr[HB], rl[HB];
    \\threadgroup int rsrc[2][KT];
    \\const int hh = int(tid) / 16, sub = int(tid) % 16;
    \\const bool soft = int(tid) < HB * 16;
    \\float m = soft ? float(sink[metal::min(h0 + hh, H - 1)]) : 0.0f, lsum = 1.0f;
    \\bfloat16_t qh[QS][16], qm[QS][16], qo[QS][16];
    \\#pragma clang loop unroll(full)
    \\for (int ks = 0; ks < QS; ++ks)
    \\  #pragma clang loop unroll(full)
    \\  for (int i = 0; i < 16; ++i) {
    \\    const int ch = ks * 32 + (i / 8) * 16 + hy4_nax_r(lane, i % 8);
    \\    const int head = h0 + hy4_nax_c(lane, i % 8);
    \\    float v = 0.0f;
    \\    if (head < H && work) {
    \\      if (!rsg) v = ql[head * ql_strides[0] + (row0 + l) * ql_strides[1] + (d0 + ch) * ql_strides[2]];
    \\      else if (ch < R) v = qp[head * qp_strides[0] + (row0 + l) * qp_strides[1] + ch * qp_strides[2]];
    \\    }
    \\    const bfloat16_t hi = bfloat16_t(v);
    \\    const float r1 = v - float(hi);
    \\    const bfloat16_t md = bfloat16_t(r1);
    \\    qh[ks][i] = hi;
    \\    qm[ks][i] = md;
    \\    qo[ks][i] = bfloat16_t(r1 - float(md));
    \\  }
    \\float o[32];
    \\#pragma clang loop unroll(full)
    \\for (int i = 0; i < 32; ++i) o[i] = 0.0f;
;

/// The key range of a row: keys `c0 + j` for `j < total`, the window ending at the row (or every key to it).
const range_rows: [:0]const u8 =
    \\const int pos = p0 + l;
    \\const int c0 = window > 0 ? metal::max(0, pos - window + 1) : 0;
    \\const int c1 = metal::min(pos, N - 1);
    \\const int total = c1 >= c0 ? c1 - c0 + 1 : 0;
    \\#define SRC(t) ((t) < total ? c0 + (t) : -1)
    \\#define TSRC(jj) SRC(t0 + (jj))
    \\#define PRELUDE
    \\#define PREFETCH
;

/// The key rows a row picked: `rows[l, t]` where `keep[l, t]`, a tile's ids made while the tile before it is softmaxed.
const picks_rows: [:0]const u8 =
    \\const int total = S;
    \\#define SRC(t) ((t) < S && keep[size_t(l) * S + (t)] && N > 0 && rows[size_t(l) * S + (t)] < uint(N) ? int(rows[size_t(l) * S + (t)]) : -1)
    \\#define TSRC(jj) rsrc[bb][(jj)]
    \\#define PRELUDE if (tid < KT) rsrc[0][tid] = SRC(int(tid)); threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\#define PREFETCH if (tid < KT) rsrc[bb ^ 1][tid] = SRC(t0 + KT + int(tid));
;

const tile_open: [:0]const u8 =
    \\const short r0 = hy4_nax_r(lane, 0), c0f = hy4_nax_c(lane, 0);
    \\int bb = 0;
    \\PRELUDE
    \\for (int t0 = 0; t0 < total; t0 += KT, bb ^= 1) {
    \\  bfloat16_t kf[NK][QS][16];
    \\  if (work) {
    \\  #pragma clang loop unroll(full)
    \\  for (int kt = 0; kt < NK; ++kt) {
    \\    float s[8];
    \\    const int ra = TSRC(kt * 16 + r0), rb = TSRC(kt * 16 + r0 + 8);
    \\    #pragma clang loop unroll(full)
    \\    for (int i = 0; i < 8; ++i) s[i] = 0.0f;
    \\    #pragma clang loop unroll(full)
    \\    for (int ks = 0; ks < QS; ++ks) {
    \\      #pragma clang loop unroll(full)
    \\      for (int g = 0; g < 4; ++g) {
    \\        const int src = (g & 1) ? rb : ra;
    \\        const int ch0 = ks * 32 + (g >> 1) * 16 + c0f;
    \\        bfloat16_t v[4] = {bfloat16_t(0.0f), bfloat16_t(0.0f), bfloat16_t(0.0f), bfloat16_t(0.0f)};
    \\        if (src >= 0) {
    \\          if (rsg) {
    \\            if (ch0 < R) {
    \\              const uint2 w = *(const device uint2*)(kp + size_t(src) * R + ch0);
    \\              v[0] = as_type<bfloat16_t>(ushort(w.x & 0xffffu)); v[1] = as_type<bfloat16_t>(ushort(w.x >> 16));
    \\              v[2] = as_type<bfloat16_t>(ushort(w.y & 0xffffu)); v[3] = as_type<bfloat16_t>(ushort(w.y >> 16));
    \\            }
    \\          } else {
    \\            KEY4(src, ch0)
    \\          }
    \\        }
    \\        #pragma clang loop unroll(full)
    \\        for (int e = 0; e < 4; ++e) kf[kt][ks][g * 4 + e] = v[e];
    \\      }
    \\      hy4_nax_mma<16, 32>(s, kf[kt][ks], qo[ks]);
    \\      hy4_nax_mma<16, 32>(s, kf[kt][ks], qm[ks]);
    \\      hy4_nax_mma<16, 32>(s, kf[kt][ks], qh[ks]);
    \\    }
    \\    #pragma clang loop unroll(full)
    \\    for (int i = 0; i < 8; ++i) sp[(int(sg) * HB + hy4_nax_c(lane, i)) * KT + kt * 16 + hy4_nax_r(lane, i)] = s[i];
    \\  }
    \\  }
;

const tile_close: [:0]const u8 =
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  if (soft) {
    \\    float sv[NJ], tmax = -INFINITY;
    \\    #pragma clang loop unroll(full)
    \\    for (int jj = 0; jj < NJ; ++jj) {
    \\      const int j = sub + jj * 16;
    \\      float v = -INFINITY;
    \\      if (TSRC(j) != -1) {
    \\        float a = 0.0f;
    \\        #pragma clang loop unroll(full)
    \\        for (int g = 0; g < NSG; ++g) a += sp[(g * HB + hh) * KT + j];
    \\        v = a * scale;
    \\      }
    \\      sv[jj] = v;
    \\      tmax = metal::max(tmax, v);
    \\    }
    \\    PREFETCH
    \\    #pragma clang loop unroll(full)
    \\    for (int off = 8; off > 0; off >>= 1) tmax = metal::max(tmax, metal::simd_shuffle_xor(tmax, ushort(off)));
    \\    const float mn = metal::max(m, tmax), corr = metal::precise::exp(m - mn);
    \\    float psum = 0.0f;
    \\    #pragma clang loop unroll(full)
    \\    for (int jj = 0; jj < NJ; ++jj) {
    \\      const int j = sub + jj * 16;
    \\      const float p = metal::precise::exp(sv[jj] - mn);
    \\      const bfloat16_t hi = bfloat16_t(p);
    \\      psum += p;
    \\      ph[hh * KT + j] = hi;
    \\      pl[hh * KT + j] = bfloat16_t(p - float(hi));
    \\    }
    \\    #pragma clang loop unroll(full)
    \\    for (int off = 8; off > 0; off >>= 1) psum += metal::simd_shuffle_xor(psum, ushort(off));
    \\    lsum = lsum * corr + psum;
    \\    m = mn;
    \\    if (sub == 0) cr[hh] = corr;
    \\  }
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  if (work && !rsg) {
    \\    #pragma clang loop unroll(full)
    \\    for (int i = 0; i < 32; ++i) o[i] *= cr[hy4_nax_r(lane, i % 8)];
    \\    #pragma clang loop unroll(full)
    \\    for (int kt = 0; kt < NK; ++kt) {
    \\      bfloat16_t ah[8], al[8];
    \\      #pragma clang loop unroll(full)
    \\      for (int i = 0; i < 8; ++i) { ah[i] = ph[hy4_nax_r(lane, i) * KT + kt * 16 + hy4_nax_c(lane, i)]; al[i] = pl[hy4_nax_r(lane, i) * KT + kt * 16 + hy4_nax_c(lane, i)]; }
    \\      #pragma clang loop unroll(full)
    \\      for (int g = 0; g < QS; ++g) {
    \\        hy4_nax_mma<32, 16>(o + g * 16, al, kf[kt][g]);
    \\        hy4_nax_mma<32, 16>(o + g * 16, ah, kf[kt][g]);
    \\      }
    \\    }
    \\  }
    \\}
    \\if (soft && sub == 0) rl[hh] = 1.0f / lsum;
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\if (work && !rsg) {
    \\  #pragma clang loop unroll(full)
    \\  for (int hs = 0; hs < 2; ++hs) {
    \\    const int row = hy4_nax_r(lane, hs * 4);
    \\    const float inv = rl[row];
    \\    const int head = h0 + row;
    \\    if (head < H) {
    \\      device float* dst = out + (size_t(head) * meta[6] + l) * L + d0 + c0f;
    \\      #pragma clang loop unroll(full)
    \\      for (int f = 0; f < 2 * QS; ++f)
    \\        #pragma clang loop unroll(full)
    \\        for (int j = 0; j < 4; ++j) dst[f * 16 + j] = o[f * 8 + hs * 4 + j] * inv;
    \\    }
    \\  }
    \\}
;

/// A latent simdgroup's four channels of a bf16 key row.
const key4_bf16: [:0]const u8 =
    \\#define KEY4(src, ch0) { \
    \\  const uint2 w = *(const device uint2*)(kc + size_t(src) * L + d0 + ch0); \
    \\  v[0] = as_type<bfloat16_t>(ushort(w.x & 0xffffu)); v[1] = as_type<bfloat16_t>(ushort(w.x >> 16)); \
    \\  v[2] = as_type<bfloat16_t>(ushort(w.y & 0xffffu)); v[3] = as_type<bfloat16_t>(ushort(w.y >> 16)); }
;

/// The matrix-unit fragment layout and product of `metal_def.NAX_HEADER`.
pub const header: [:0]const u8 = metal_def.NAX_HEADER ++
    \\inline short hy4_nax_r(ushort lane, int i) { return dsv41_nax_r(lane, i); }
    \\inline short hy4_nax_c(ushort lane, int i) { return dsv41_nax_c(lane, i); }
    \\template <int N, int KK, typename AT, typename BT>
    \\inline void hy4_nax_mma(thread float* c, thread const AT* a, thread const BT* b) { dsv41_nax_mma<N, KK>(c, a, b); }
    \\
;

pub const range_source: [:0]const u8 = head ++ "\n" ++ range_rows ++ "\n" ++ key4_bf16 ++ "\n" ++ tile_open ++ "\n" ++ tile_close;
pub const picks_source: [:0]const u8 = head ++ "\n" ++ picks_rows ++ "\n" ++ key4_bf16 ++ "\n" ++ tile_open ++ "\n" ++ tile_close;

/// Non-NAX `simdgroup_float8x8` header: fragment coordinate map, bitwise BF16→F32 unpack, and IEEE f32 8x8 MMA.
pub const simd_header: [:0]const u8 =
    \\#include <metal_simdgroup_matrix>
    \\inline short2 hy4_simd_coord(ushort lane) {
    \\  const short qid = short(lane / 4u);
    \\  const short fm = (qid & 4) + ((short(lane) / 2) % 4);
    \\  const short fn = (qid & 2) * 2 + (short(lane) % 2) * 2;
    \\  return short2(fn, fm);
    \\}
    \\inline float2 hy4_bf16_u1_f2(uint w) {
    \\  return float2(as_type<float>((w & 0xffffu) << 16u), as_type<float>(w & 0xffff0000u));
    \\}
    \\inline void hy4_simd_mma(thread float2 &d, float2 a, float2 b) {
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
;

/// Non-NAX `simdgroup_float8x8` fused attention (V1-half): `NSG = L/32` simdgroups (`16` SGs = `512` threads at `L=512`),
/// `HB=16` heads, `KT=16` keys, `DT=4` (`32` latent cols/SG). Halves per-thread register state (`qf[2][4]` + `o[2][4]` +
/// `kf[2][4]` + `s[2][2]` = `48` regs), folds rope into SGs `0..R/8-1` (zero idle rope SG during `PV`), keeps `staticTG`
/// ~18.4 KiB (`<= 32 KiB`, with `sp[NSG*16*17]` padded stride eliminating 4-way bank conflicts at `NSG=16`), reads `K`
/// once per tile with bitwise BF16→F32 unpack, and overlaps `s[2][2]` (4 accumulators) across the inner `c` loop.
const simd_head: [:0]const u8 =
    \\const uint tid = thread_index_in_threadgroup, sg = simdgroup_index_in_threadgroup;
    \\const ushort lane = thread_index_in_simdgroup;
    \\const int l = int(threadgroup_position_in_grid.y);
    \\const int h0 = int(threadgroup_position_in_grid.x) * HB;
    \\const int H = ql_shape[0];
    \\const int p0 = meta[0], window = meta[1], N = meta[2], S = meta[3], row0 = meta[5], T = meta[6];
    \\const float scale2 = as_type<float>(meta[4]) * 1.4426950408889634f;
    \\constexpr int RT = HB / 8, KC = KT / 8, DT = 4;
    \\constexpr int KT_PAD = KT + 1, PF_STR = KT + 2;
    \\const bool work = int(sg) < NSG;
    \\const int d0 = int(sg) * 32;
    \\const short2 rc = hy4_simd_coord(lane);
    \\const int fn = int(rc.x), fm = int(rc.y);
    \\threadgroup float sp[NSG * HB * KT_PAD];
    \\threadgroup float pf[HB * PF_STR];
    \\threadgroup float cr[HB], rl[HB];
    \\threadgroup int rsrc[2][KT];
    \\const int hh = int(tid) >> 4, sub = int(tid) & 15;
    \\const bool soft = int(tid) < HB * KT;
    \\float m = soft ? float(sink[metal::min(h0 + hh, H - 1)]) * 1.4426950408889634f : 0.0f, lsum = 1.0f;
    \\float2 qf[RT][DT];
    \\float2 qr[RT];
    \\const bool has_rope = work && (int(sg) * 8 < R);
    \\{
    \\  const size_t qs0 = size_t(ql_strides[0]), qs1 = size_t(ql_strides[1]), qs2 = size_t(ql_strides[2]);
    \\  const size_t ps0 = size_t(qp_strides[0]), ps1 = size_t(qp_strides[1]), ps2 = size_t(qp_strides[2]);
    \\  #pragma clang loop unroll(full)
    \\  for (int r = 0; r < RT; ++r) {
    \\    const int head0 = h0 + r * 8 + fn, head1 = head0 + 1;
    \\    const size_t qrow0 = size_t(head0 < H ? head0 : 0) * qs0 + size_t(row0 + l) * qs1;
    \\    const size_t qrow1 = size_t(head1 < H ? head1 : 0) * qs0 + size_t(row0 + l) * qs1;
    \\    #pragma clang loop unroll(full)
    \\    for (int d = 0; d < DT; ++d) {
    \\      const int ch = d0 + d * 8 + fm;
    \\      float v0 = 0.0f, v1 = 0.0f;
    \\      if (work) {
    \\        if (head0 < H) v0 = ql[qrow0 + size_t(ch) * qs2];
    \\        if (head1 < H) v1 = ql[qrow1 + size_t(ch) * qs2];
    \\      }
    \\      qf[r][d] = float2(v0, v1);
    \\    }
    \\    float rv0 = 0.0f, rv1 = 0.0f;
    \\    const int rch = int(sg) * 8 + fm;
    \\    if (has_rope && rch < R) {
    \\      const size_t prow0 = size_t(head0 < H ? head0 : 0) * ps0 + size_t(row0 + l) * ps1 + size_t(rch) * ps2;
    \\      const size_t prow1 = size_t(head1 < H ? head1 : 0) * ps0 + size_t(row0 + l) * ps1 + size_t(rch) * ps2;
    \\      if (head0 < H) rv0 = qp[prow0];
    \\      if (head1 < H) rv1 = qp[prow1];
    \\    }
    \\    qr[r] = float2(rv0, rv1);
    \\  }
    \\}
    \\float2 o[RT][DT];
    \\#pragma clang loop unroll(full)
    \\for (int r = 0; r < RT; ++r)
    \\  #pragma clang loop unroll(full)
    \\  for (int d = 0; d < DT; ++d) o[r][d] = float2(0.0f);
;

const simd_range_rows: [:0]const u8 =
    \\const int pos = p0 + l;
    \\const int c0 = window > 0 ? metal::max(0, pos - window + 1) : 0;
    \\const int c1 = metal::min(pos, N - 1);
    \\const int total = c1 >= c0 ? c1 - c0 + 1 : 0;
    \\#define SSRC(t) ((t) < total ? c0 + (t) : -1)
    \\#define STSRC(jj) SSRC(t0 + (jj))
    \\#define SPRELUDE
    \\#define SPREFETCH
;

const simd_picks_rows: [:0]const u8 =
    \\const int total = S;
    \\const device uint* r_row = rows + size_t(l) * size_t(S);
    \\const device bool* k_row = keep + size_t(l) * size_t(S);
    \\#define SSRC(t) ((t) < S && k_row[(t)] && N > 0 && r_row[(t)] < uint(N) ? int(r_row[(t)]) : -1)
    \\#define STSRC(jj) rsrc[bb][(jj)]
    \\#define SPRELUDE if (tid < KT) rsrc[0][tid] = SSRC(int(tid)); threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\#define SPREFETCH if (tid < KT) rsrc[bb ^ 1][tid] = SSRC(t0 + KT + int(tid));
;

const simd_body: [:0]const u8 =
    \\int bb = 0;
    \\SPRELUDE
    \\for (int t0 = 0; t0 < total; t0 += KT, bb ^= 1) {
    \\  uint kf[KC][DT];
    \\  if (work) {
    \\    const int src0 = STSRC(fm);
    \\    const int src1 = STSRC(8 + fm);
    \\    const device bfloat16_t* krow0 = kc + size_t(src0 >= 0 ? src0 : 0) * size_t(L) + size_t(d0 + fn);
    \\    const device bfloat16_t* krow1 = kc + size_t(src1 >= 0 ? src1 : 0) * size_t(L) + size_t(d0 + fn);
    \\    if (src0 >= 0) {
    \\      #pragma clang loop unroll(full)
    \\      for (int d = 0; d < DT; ++d) kf[0][d] = *(const device uint*)(krow0 + d * 8);
    \\    } else {
    \\      #pragma clang loop unroll(full)
    \\      for (int d = 0; d < DT; ++d) kf[0][d] = 0u;
    \\    }
    \\    if (src1 >= 0) {
    \\      #pragma clang loop unroll(full)
    \\      for (int d = 0; d < DT; ++d) kf[1][d] = *(const device uint*)(krow1 + d * 8);
    \\    } else {
    \\      #pragma clang loop unroll(full)
    \\      for (int d = 0; d < DT; ++d) kf[1][d] = 0u;
    \\    }
    \\    uint rw0 = 0u, rw1 = 0u;
    \\    if (has_rope) {
    \\      const int rcol = int(sg) * 8 + fn;
    \\      if (rcol < R) {
    \\        rw0 = (src0 >= 0) ? *(const device uint*)(kp + size_t(src0) * size_t(R) + size_t(rcol)) : 0u;
    \\        rw1 = (src1 >= 0) ? *(const device uint*)(kp + size_t(src1) * size_t(R) + size_t(rcol)) : 0u;
    \\      }
    \\    }
    \\    float2 s[KC][RT];
    \\    #pragma clang loop unroll(full)
    \\    for (int c = 0; c < KC; ++c)
    \\      #pragma clang loop unroll(full)
    \\      for (int r = 0; r < RT; ++r) s[c][r] = float2(0.0f);
    \\    #pragma clang loop unroll(full)
    \\    for (int d = 0; d < DT; ++d) {
    \\      const float2 kv0 = hy4_bf16_u1_f2(kf[0][d]);
    \\      const float2 kv1 = hy4_bf16_u1_f2(kf[1][d]);
    \\      hy4_simd_mma(s[0][0], kv0, qf[0][d]);
    \\      hy4_simd_mma(s[0][1], kv0, qf[1][d]);
    \\      hy4_simd_mma(s[1][0], kv1, qf[0][d]);
    \\      hy4_simd_mma(s[1][1], kv1, qf[1][d]);
    \\    }
    \\    if (has_rope) {
    \\      const float2 kv0 = hy4_bf16_u1_f2(rw0);
    \\      const float2 kv1 = hy4_bf16_u1_f2(rw1);
    \\      hy4_simd_mma(s[0][0], kv0, qr[0]);
    \\      hy4_simd_mma(s[0][1], kv0, qr[1]);
    \\      hy4_simd_mma(s[1][0], kv1, qr[0]);
    \\      hy4_simd_mma(s[1][1], kv1, qr[1]);
    \\      if (NSG * 8 < R) {
    \\        const size_t ps0 = size_t(qp_strides[0]), ps1 = size_t(qp_strides[1]), ps2 = size_t(qp_strides[2]);
    \\        for (int rd = (int(sg) + NSG) * 8; rd < R; rd += NSG * 8) {
    \\          const int rch = rd + fm;
    \\          const int rc2 = rd + fn;
    \\          float2 qx[RT];
    \\          #pragma clang loop unroll(full)
    \\          for (int r = 0; r < RT; ++r) {
    \\            const int head0 = h0 + r * 8 + fn, head1 = head0 + 1;
    \\            float v0 = 0.0f, v1 = 0.0f;
    \\            if (rch < R) {
    \\              if (head0 < H) v0 = qp[size_t(head0) * ps0 + size_t(row0 + l) * ps1 + size_t(rch) * ps2];
    \\              if (head1 < H) v1 = qp[size_t(head1) * ps0 + size_t(row0 + l) * ps1 + size_t(rch) * ps2];
    \\            }
    \\            qx[r] = float2(v0, v1);
    \\          }
    \\          const uint xw0 = (src0 >= 0 && rc2 < R) ? *(const device uint*)(kp + size_t(src0) * size_t(R) + size_t(rc2)) : 0u;
    \\          const uint xw1 = (src1 >= 0 && rc2 < R) ? *(const device uint*)(kp + size_t(src1) * size_t(R) + size_t(rc2)) : 0u;
    \\          const float2 rkv0 = hy4_bf16_u1_f2(xw0);
    \\          const float2 rkv1 = hy4_bf16_u1_f2(xw1);
    \\          hy4_simd_mma(s[0][0], rkv0, qx[0]);
    \\          hy4_simd_mma(s[0][1], rkv0, qx[1]);
    \\          hy4_simd_mma(s[1][0], rkv1, qx[0]);
    \\          hy4_simd_mma(s[1][1], rkv1, qx[1]);
    \\        }
    \\      }
    \\    }
    \\    #pragma clang loop unroll(full)
    \\    for (int c = 0; c < KC; ++c)
    \\      #pragma clang loop unroll(full)
    \\      for (int r = 0; r < RT; ++r) {
    \\        sp[(int(sg) * HB + r * 8 + fn) * KT_PAD + c * 8 + fm] = s[c][r].x;
    \\        sp[(int(sg) * HB + r * 8 + fn + 1) * KT_PAD + c * 8 + fm] = s[c][r].y;
    \\      }
    \\  }
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  if (soft) {
    \\    float v = -INFINITY;
    \\    if (STSRC(sub) != -1) {
    \\      const threadgroup float* spp = sp + hh * KT_PAD + sub;
    \\      constexpr int SP_STEP = HB * KT_PAD;
    \\      float a;
    \\      if (NSG == 16) {
    \\        const float s0 = (spp[0 * SP_STEP] + spp[1 * SP_STEP]) + (spp[2 * SP_STEP] + spp[3 * SP_STEP]);
    \\        const float s1 = (spp[4 * SP_STEP] + spp[5 * SP_STEP]) + (spp[6 * SP_STEP] + spp[7 * SP_STEP]);
    \\        const float s2 = (spp[8 * SP_STEP] + spp[9 * SP_STEP]) + (spp[10 * SP_STEP] + spp[11 * SP_STEP]);
    \\        const float s3 = (spp[12 * SP_STEP] + spp[13 * SP_STEP]) + (spp[14 * SP_STEP] + spp[15 * SP_STEP]);
    \\        a = (s0 + s1) + (s2 + s3);
    \\      } else {
    \\        a = 0.0f;
    \\        #pragma clang loop unroll(full)
    \\        for (int g = 0; g < NSG; ++g) a += spp[g * SP_STEP];
    \\      }
    \\      v = a * scale2;
    \\    }
    \\    SPREFETCH
    \\    float tmax = v;
    \\    #pragma clang loop unroll(full)
    \\    for (int off = 8; off > 0; off >>= 1) tmax = metal::max(tmax, metal::simd_shuffle_xor(tmax, ushort(off)));
    \\    const float mn = metal::max(m, tmax);
    \\    const float corr = metal::exp2(m - mn);
    \\    const float p = metal::exp2(v - mn);
    \\    pf[hh * PF_STR + sub] = p;
    \\    float psum = p;
    \\    #pragma clang loop unroll(full)
    \\    for (int off = 8; off > 0; off >>= 1) psum += metal::simd_shuffle_xor(psum, ushort(off));
    \\    lsum = lsum * corr + psum;
    \\    m = mn;
    \\    if (sub == 0) cr[hh] = corr;
    \\  }
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  if (work) {
    \\    #pragma clang loop unroll(full)
    \\    for (int r = 0; r < RT; ++r) {
    \\      const float c_r = cr[r * 8 + fm];
    \\      #pragma clang loop unroll(full)
    \\      for (int d = 0; d < DT; ++d) o[r][d] *= c_r;
    \\    }
    \\    #pragma clang loop unroll(full)
    \\    for (int c = 0; c < KC; ++c) {
    \\      const int pi0 = fm * PF_STR + c * 8 + fn;
    \\      const int pi1 = (8 + fm) * PF_STR + c * 8 + fn;
    \\      const float2 a0 = float2(pf[pi0], pf[pi0 + 1]);
    \\      const float2 a1 = float2(pf[pi1], pf[pi1 + 1]);
    \\      #pragma clang loop unroll(full)
    \\      for (int d = 0; d < DT; ++d) {
    \\        const float2 kv = hy4_bf16_u1_f2(kf[c][d]);
    \\        hy4_simd_mma(o[0][d], a0, kv);
    \\        hy4_simd_mma(o[1][d], a1, kv);
    \\      }
    \\    }
    \\  }
    \\}
    \\if (soft && sub == 0) rl[hh] = 1.0f / lsum;
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\if (work) {
    \\  #pragma clang loop unroll(full)
    \\  for (int r = 0; r < RT; ++r) {
    \\    const int head = h0 + r * 8 + fm;
    \\    if (head < H) {
    \\      const float inv = rl[r * 8 + fm];
    \\      device float* dst = out + (size_t(head) * size_t(T) + size_t(l)) * size_t(L) + size_t(d0 + fn);
    \\      #pragma clang loop unroll(full)
    \\      for (int d = 0; d < DT; ++d) {
    \\        dst[d * 8] = o[r][d].x * inv;
    \\        dst[d * 8 + 1] = o[r][d].y * inv;
    \\      }
    \\    }
    \\  }
    \\}
;

pub const simd_range_source: [:0]const u8 = simd_head ++ "\n" ++ simd_range_rows ++ "\n" ++ simd_body;
pub const simd_picks_source: [:0]const u8 = simd_head ++ "\n" ++ simd_picks_rows ++ "\n" ++ simd_body;

const Kind = enum { range, picks };

fn defOf(kind: Kind, nax: bool) *metal_def.Def {
    const S = struct {
        var range: metal_def.Def = .{ .name = "sushi_hy4_attn_range", .ins = &.{ "ql", "qp", "kc", "kp", "sink", "meta" }, .outs = &.{"out"}, .source = range_source, .header = header };
        var picks: metal_def.Def = .{ .name = "sushi_hy4_attn_picks", .ins = &.{ "ql", "qp", "kc", "kp", "rows", "keep", "sink", "meta" }, .outs = &.{"out"}, .source = picks_source, .header = header };
        var simd_range: metal_def.Def = .{ .name = "sushi_hy4_attn_simd_range", .ins = &.{ "ql", "qp", "kc", "kp", "sink", "meta" }, .outs = &.{"out"}, .source = simd_range_source, .header = simd_header };
        var simd_picks: metal_def.Def = .{ .name = "sushi_hy4_attn_simd_picks", .ins = &.{ "ql", "qp", "kc", "kp", "rows", "keep", "sink", "meta" }, .outs = &.{"out"}, .source = simd_picks_source, .header = simd_header };
    };
    return if (nax) switch (kind) {
        .range => &S.range,
        .picks => &S.picks,
    } else switch (kind) {
        .range => &S.simd_range,
        .picks => &S.simd_picks,
    };
}

/// The query, as the absorb leaves it: `q_lat` [H, T, latent] f32 and the rope part `q_pe` [H, T, rope] f32, a view of
/// the query's own rows where they are (read through its strides).
pub const Query = struct { lat: Arr, pe: Arr };

/// A layer's key rows: the latent `c` [N, latent] and the rotated rope rows `k_pe` [N, rope], both bf16 (a kv8 store
/// is read to this width first: `hy4_latent.rangeBf16`).
pub const Keys = struct { c: Arr, k_pe: Arr };

/// Whether the kernel serves a layer of `heads` heads with a `latent` and `rope` width and `sinks`, on stream `s`:
/// GPU stream (NAX on `matrix_g17`, `simdgroup_float8x8` SIMD arm otherwise), a latent matching a compiled `NSG` bucket
/// (`{64, 128, 256, 512}`), a rope of at most 64 (multiple of 4), f32 or bf16 sinks.
pub fn fits(s: mlx.mlx_stream, heads: c_int, latent: c_int, rope: c_int, sinks: Arr) bool {
    if (!mlx.streamIsGpu(s)) return false;
    if (heads <= 0 or latent <= 0 or @rem(latent, slice) != 0 or rope <= 0 or rope > slice or @rem(rope, 4) != 0) return false;
    const nsg = @divExact(latent, slice) + 1;
    if (nsg != 2 and nsg != 3 and nsg != 5 and nsg != 9) return false;
    const sd = mlx.mlx_array_dtype(sinks);
    return (sd == .float32 or sd == .bfloat16) and mlx.getShape(sinks).len == 1 and mlx.getShape(sinks)[0] == heads;
}

/// Whether the kernel serves the query `q` over `keys`: the arrays' dtypes and shapes are the ones `fits` and the
/// stores take: f32 queries, bf16 latent and rope rows.
pub fn applies(s: mlx.mlx_stream, q: Query, keys: Keys, sinks: Arr) bool {
    const ls = mlx.getShape(q.lat);
    const ps = mlx.getShape(q.pe);
    if (ls.len != 3 or ps.len != 3 or ls[0] != ps[0] or ls[1] != ps[1] or ls[1] <= 0) return false;
    if (!fits(s, ls[0], ls[2], ps[2], sinks)) return false;
    const latent = ls[2];
    const rope = ps[2];
    if (mlx.mlx_array_dtype(q.lat) != .float32 or mlx.mlx_array_dtype(q.pe) != .float32) return false;
    if (mlx.mlx_array_dtype(keys.k_pe) != .bfloat16 or mlx.getShape(keys.k_pe).len != 2 or mlx.getShape(keys.k_pe)[1] != rope) return false;
    const cs = mlx.getShape(keys.c);
    if (cs.len != 2) return false;
    return mlx.mlx_array_dtype(keys.c) == .bfloat16 and cs[1] == latent and mlx.getShape(keys.k_pe)[0] == cs[0];
}

/// The launches made, a test's count of who reaches the core.
pub var launched: u64 = 0;

const Key = struct { kind: Kind, nax: bool, heads: c_int, rows: c_int, latent: c_int, rope: c_int };
var cfgs: kcfg.Cache(Key) = .{};

fn build(k: Key) !kcfg.Config {
    const nsg = if (k.nax) @divExact(k.latent, slice) + 1 else @divExact(k.latent, simd_slice);
    const blocks = @divTrunc(k.heads + hb - 1, hb);
    const tile_k: c_int = if (k.nax) kt else simd_kt;
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ k.heads, k.rows, k.latent }, 3, .float32));
    // The softmax runs 16 threads a head of the block, so a narrow latent's few simdgroups are joined by idle ones.
    const threads = @max(nsg * 32, hb * 16);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, blocks * threads, k.rows, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, threads, 1, 1));
    for ([_]struct { [*:0]const u8, c_int }{ .{ "L", k.latent }, .{ "R", k.rope }, .{ "HB", hb }, .{ "KT", tile_k }, .{ "NSG", nsg } }) |t|
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, t[0], t[1]));
    return cfg;
}

fn launch(s: mlx.mlx_stream, kind: Kind, q: Query, keys: Keys, picked: ?struct { rows: Arr, keep: Arr }, sinks: Arr, scale: f32, p0: c_int, window: c_int, row0: c_int, rows: c_int) !Arr {
    const ls = mlx.getShape(q.lat);
    const n: c_int = mlx.getShape(keys.c)[0];
    const sel: c_int = if (picked) |p| mlx.getShape(p.rows)[1] else 0;
    const mv = [_]i32{ p0, window, n, sel, kcfg.bits(scale), row0, rows };
    const meta = try mlx.newData(&mv, &.{mv.len}, .int32);
    defer _ = mlx.mlx_array_free(meta);
    const use_nax = nax_gate.verifyQmmNaxAvailable();
    const cfg = try cfgs.get(.{ .kind = kind, .nax = use_nax, .heads = ls[0], .rows = rows, .latent = ls[2], .rope = mlx.getShape(q.pe)[2] }, build);
    const def = defOf(kind, use_nax);
    const k = try getKernel(def);
    launched += 1;
    var ins: [8]Arr = undefined;
    var ni: usize = 0;
    for ([_]Arr{ q.lat, q.pe, keys.c, keys.k_pe }) |a| {
        ins[ni] = a;
        ni += 1;
    }
    if (picked) |p| for ([_]Arr{ p.rows, p.keep }) |a| {
        ins[ni] = a;
        ni += 1;
    };
    ins[ni] = sinks;
    ins[ni + 1] = meta;
    ni += 2;
    const iv = mlx.mlx_vector_array_new_data(&ins, ni);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, k, iv, cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, ov, 0));
    return out;
}

fn getKernel(def: *metal_def.Def) !mlx.mlx_fast_metal_kernel {
    if (def.built) |k| return k;
    const iv = mlx.mlx_vector_string_new_data(def.ins.ptr, def.ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(def.outs.ptr, def.outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    // The queries are read through their strides, the keys as row-major rows.
    const k = mlx.mlx_fast_metal_kernel_new(def.name, iv, ov, def.source, def.header, false, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    def.built = k;
    return k;
}

/// The attended latent [H, rows, latent] f32 of query rows `row0..row0 + rows` of `q`, each over the keys `max(0,
/// pos - window + 1)..pos` of `keys` (every key up to `pos` when `window` is 0), `pos = p0 + i` for the launch's row `i`,
/// each head's `sinks` logit beside them. The scores are scaled by `scale`. The caller frees it.
pub fn range(s: mlx.mlx_stream, q: Query, keys: Keys, sinks: Arr, scale: f32, p0: c_int, window: c_int, row0: c_int, rows: c_int) !Arr {
    return launch(s, .range, q, keys, null, sinks, scale, p0, window, row0, rows);
}

/// The attended latent [H, rows, latent] f32 of query rows `row0..row0 + rows` of `q`, each over the rows
/// `rows_idx[i, :]` of `keys` where `keep[i, :]` ([rows, S] uint32 and bool), the sinks beside them. The picks are read
/// where they lie, never gathered. The caller frees it.
pub fn picks(s: mlx.mlx_stream, q: Query, keys: Keys, rows_idx: Arr, keep: Arr, sinks: Arr, scale: f32, row0: c_int) !Arr {
    const rows = mlx.getShape(rows_idx)[0];
    return launch(s, .picks, q, keys, .{ .rows = rows_idx, .keep = keep }, sinks, scale, 0, 0, row0, rows);
}

// ── tests ────────────────────────────────────────────────────────────────────────────────────────

const testing = std.testing;
const Ops = @import("../mlx_ops.zig").Ops;
const hy4_latent = @import("hy4_latent.zig");
const kv_quant = @import("../kv_quant.zig");

/// How far the kernel's heads may sit from the f64 truth, as a fraction of the truth's peak, at a query of 4 sigma: three
/// bf16 parts hold the query to f32's own 24 bits over its whole range, two hold P (in [0, 1]) to 2^-18, and the sums are
/// f32. The op chain's own f32 sits 2e-6 to 7e-6 off the same truth, a query split in two bf16 parts 2e-5.
const bar = 5e-6;

fn hostF32(o: *Ops, a: Arr) ![]const f32 {
    const c = try o.contiguous(try o.cast(a, .float32));
    try mlx.check(mlx.mlx_array_eval(c));
    return (mlx.mlx_array_data_float32(c) orelse return error.HostRead)[0..mlx.mlx_array_size(c)];
}

/// `x` rounded to bf16 (nearest even), as f32.
fn bf16Round(x: f32) f32 {
    const u: u32 = @bitCast(x);
    return @bitCast((u +% 0x7fff +% ((u >> 16) & 1)) & 0xffff_0000);
}

/// The largest |got - want| over the largest |want|.
fn drift(got: []const f32, want: []const f64) f64 {
    var peak: f64 = 1e-30;
    var worst: f64 = 0;
    for (got, want) |g, w| {
        const e = @abs(@as(f64, g) - w);
        if (std.math.isNan(e)) return std.math.inf(f64);
        worst = @max(worst, e);
        peak = @max(peak, @abs(w));
    }
    return worst / peak;
}

fn trace() bool {
    return std.c.getenv("SUSHI_HY4_ATTN_TRACE") != null;
}

/// A case's host values: the query's latent and rope parts, the bf16 key rows and the sinks, as the arrays hold them.
const Host = struct {
    h: usize,
    t: usize,
    latent: usize,
    rope: usize,
    ql: []f32,
    qp: []f32,
    kc: []f32,
    kp: []f32,
    sinks: []f32,

    /// Random values: the query at `q_std`, the keys at 1, sinks at 1.5; `outlier` gives the keys' channel 3 30 times its
    /// size and the queries 4 times it, a dot product with one large term.
    fn init(rnd: std.Random, h: c_int, latent: c_int, rope: c_int, t: c_int, n: c_int, q_std: f32, outlier: bool) !Host {
        const a = testing.allocator;
        const hu: usize = @intCast(h);
        const tu: usize = @intCast(t);
        const lu: usize = @intCast(latent);
        const ru: usize = @intCast(rope);
        const nu: usize = @intCast(n);
        const x: Host = .{ .h = hu, .t = tu, .latent = lu, .rope = ru, .ql = try a.alloc(f32, hu * tu * lu), .qp = try a.alloc(f32, hu * tu * ru), .kc = try a.alloc(f32, nu * lu), .kp = try a.alloc(f32, nu * ru), .sinks = try a.alloc(f32, hu) };
        for (x.ql) |*e| e.* = rnd.floatNorm(f32) * q_std;
        for (x.qp) |*e| e.* = rnd.floatNorm(f32) * q_std;
        for (x.kc, 0..) |*e, i| e.* = bf16Round(rnd.floatNorm(f32) * if (outlier and i % lu == 3) @as(f32, 30) else 1);
        for (x.kp) |*e| e.* = bf16Round(rnd.floatNorm(f32));
        for (x.sinks) |*e| e.* = rnd.floatNorm(f32) * 1.5;
        if (outlier) for (0..hu * tu) |i| {
            x.ql[i * lu + 3] *= 4;
        };
        return x;
    }

    fn deinit(x: Host) void {
        for ([_][]f32{ x.ql, x.qp, x.kc, x.kp, x.sinks }) |v| testing.allocator.free(v);
    }

    /// The query and the keys as arrays; the rope part of the query is a view of a roped query three times as wide.
    fn arrays(x: Host, o: *Ops, wide_rope: bool) !struct { q: Query, keys: Keys, sinks: Arr } {
        const h: c_int = @intCast(x.h);
        const t: c_int = @intCast(x.t);
        const lat: c_int = @intCast(x.latent);
        const rp: c_int = @intCast(x.rope);
        const n: c_int = @intCast(x.kc.len / x.latent);
        const q_pe = if (wide_rope) blk: {
            // The rope part sits last in a roped query: the part's values at the tail, others unrelated.
            const wide = try testing.allocator.alloc(f32, x.h * x.t * 3 * x.rope);
            defer testing.allocator.free(wide);
            for (0..x.h * x.t) |i| {
                @memset(wide[i * 3 * x.rope ..][0 .. 2 * x.rope], 7);
                @memcpy(wide[i * 3 * x.rope + 2 * x.rope ..][0..x.rope], x.qp[i * x.rope ..][0..x.rope]);
            }
            break :blk try o.slice(try o.own(try mlx.newData(wide.ptr, &.{ h, t, 3 * rp }, .float32)), 2, 2 * rp, 3 * rp);
        } else try o.own(try mlx.newData(x.qp.ptr, &.{ h, t, rp }, .float32));
        return .{
            .q = .{ .lat = try o.own(try mlx.newData(x.ql.ptr, &.{ h, t, lat }, .float32)), .pe = q_pe },
            .keys = .{ .c = try o.cast(try o.own(try mlx.newData(x.kc.ptr, &.{ n, lat }, .float32)), .bfloat16), .k_pe = try o.cast(try o.own(try mlx.newData(x.kp.ptr, &.{ n, rp }, .float32)), .bfloat16) },
            .sinks = try o.own(try mlx.newData(x.sinks.ptr, &.{h}, .float32)),
        };
    }

    /// The f64 truth of the attended latent [H, T, latent]: row `l` reads the key rows `keys[l]` and each head's sink logit
    /// beside them, the scores scaled by `scale`.
    fn truth(x: Host, scale: f64, keys: []const []u32) ![]f64 {
        const out = try testing.allocator.alloc(f64, x.h * x.t * x.latent);
        errdefer testing.allocator.free(out);
        var score: []f64 = &.{};
        defer testing.allocator.free(score);
        for (0..x.h) |hh| for (0..x.t) |l| {
            const ks = keys[l];
            if (score.len < ks.len) score = try testing.allocator.realloc(score, ks.len);
            var m: f64 = x.sinks[hh];
            for (ks, 0..) |j, ji| {
                var a: f64 = 0;
                for (0..x.latent) |d| a += @as(f64, x.ql[(hh * x.t + l) * x.latent + d]) * x.kc[j * x.latent + d];
                for (0..x.rope) |d| a += @as(f64, x.qp[(hh * x.t + l) * x.rope + d]) * x.kp[j * x.rope + d];
                score[ji] = a * scale;
                m = @max(m, score[ji]);
            }
            const o = out[(hh * x.t + l) * x.latent ..][0..x.latent];
            @memset(o, 0);
            var den: f64 = @exp(@as(f64, x.sinks[hh]) - m);
            for (ks, 0..) |j, ji| {
                const p = @exp(score[ji] - m);
                den += p;
                for (0..x.latent) |d| o[d] += p * x.kc[j * x.latent + d];
            }
            for (o) |*e| e.* /= den;
        };
        return out;
    }
};

fn freeLists(lists: [][]u32) void {
    for (lists) |l| testing.allocator.free(l);
    testing.allocator.free(lists);
}

const RangeCase = struct { h: c_int, latent: c_int, rope: c_int, t: c_int, n: c_int, p0: c_int, window: c_int, q_std: f32 = 4, outlier: bool = false, bar: f64 = bar };

fn rangeCells(es: mlx.mlx_stream) !void {
    var prng = std.Random.DefaultPrng.init(0x4a77);
    const rnd = prng.random();
    for ([_]RangeCase{
        // A sliding span over S + 511 keys, a short last span, a sharp softmax, one large term, a query of 16 sigma.
        .{ .h = 64, .latent = 512, .rope = 64, .t = 32, .n = 543, .p0 = 511, .window = 512 },
        .{ .h = 64, .latent = 512, .rope = 64, .t = 22, .n = 533, .p0 = 511, .window = 512, .q_std = 1 },
        .{ .h = 64, .latent = 512, .rope = 64, .t = 16, .n = 527, .p0 = 511, .window = 512, .outlier = true },
        .{ .h = 64, .latent = 512, .rope = 64, .t = 16, .n = 527, .p0 = 511, .window = 512, .q_std = 16, .bar = 1.5e-5 },
        // A sparse layer below its selection threshold: every key to the row, the chunk's first rows and a later span.
        .{ .h = 32, .latent = 512, .rope = 64, .t = 20, .n = 980, .p0 = 960, .window = 0 },
        .{ .h = 32, .latent = 512, .rope = 64, .t = 20, .n = 20, .p0 = 0, .window = 0 },
        // The tiny: 4 and 2 heads (a head block of 16 padded), latent 64, rope 16, window 8.
        .{ .h = 4, .latent = 64, .rope = 16, .t = 50, .n = 50, .p0 = 0, .window = 0 },
        .{ .h = 2, .latent = 64, .rope = 16, .t = 50, .n = 50, .p0 = 0, .window = 8 },
        // The STQ tiny: 8 heads, latent 128 (two 64-channel groups), rope 16.
        .{ .h = 8, .latent = 128, .rope = 16, .t = 20, .n = 27, .p0 = 7, .window = 8 },
        .{ .h = 8, .latent = 128, .rope = 16, .t = 50, .n = 50, .p0 = 0, .window = 0 },
    }, 0..) |c, ci| {
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const x = try Host.init(rnd, c.h, c.latent, c.rope, c.t, c.n, c.q_std, c.outlier);
        defer x.deinit();
        const arr = try x.arrays(&o, true);
        try testing.expect(applies(es, arr.q, arr.keys, arr.sinks));
        const scale = std.math.pow(f32, @floatFromInt(3 * c.rope), -0.5);
        const got = try hostF32(&o, try o.own(try range(es, arr.q, arr.keys, arr.sinks, scale, c.p0, c.window, 0, c.t)));
        const lists = try testing.allocator.alloc([]u32, x.t);
        defer freeLists(lists);
        for (lists, 0..) |*l, i| {
            const pos: usize = @as(usize, @intCast(c.p0)) + i;
            const c0: usize = if (c.window > 0 and pos + 1 > @as(usize, @intCast(c.window))) pos + 1 - @as(usize, @intCast(c.window)) else 0;
            const c1 = @min(pos, @as(usize, @intCast(c.n)) - 1);
            l.* = try testing.allocator.alloc(u32, c1 + 1 - c0);
            for (l.*, 0..) |*e, j| e.* = @intCast(c0 + j);
        }
        const want = try x.truth(scale, lists);
        defer testing.allocator.free(want);
        const d = drift(got, want);
        if (trace()) std.debug.print("hy4 attn range case {d}: drift {e}\n", .{ ci, d });
        try testing.expect(d <= c.bar);
    }
}

test "hy4 attn range: a sliding window and a dense causal range give the f64 truth's heads at the release's shapes and the tiny's, a query a view of the roped query, at a query of 4 and 16 sigma and with one large term" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const exact = mlx.engine.exactF32();
    defer exact.end();
    rangeCells(es) catch |e| {
        var said: [8192]u8 = undefined;
        std.debug.print("\nhy4 attn failed ({t}): {s}\n", .{ e, mlx.takeError(&said) orelse "(no message)" });
        return e;
    };
    if (nax_gate.verifyQmmNaxAvailable()) {
        const device = @import("sushi_exl3").device;
        const simd = device.withProfile(device.simdOnly());
        defer simd.end();
        try rangeCells(es);
    }
}

const PicksCase = struct { h: c_int, latent: c_int, rope: c_int, t: c_int, n: c_int, s: c_int, kv8: bool, q_std: f32 = 4 };

/// The store's rows as the model keeps them (`kv8`: affine codes, scales and biases) read to bf16
/// (`hy4_latent.rangeBf16`), the picks indexing them where they lie; the truth reads the same rows from the host.
fn picksCells(es: mlx.mlx_stream) !void {
    var prng = std.Random.DefaultPrng.init(0x91c5);
    const rnd = prng.random();
    for ([_]PicksCase{
        // A sparse layer past its selection: 32 heads, 2,048 picked keys a row, kv8 as served, then the bf16 store.
        .{ .h = 32, .latent = 512, .rope = 64, .t = 16, .n = 9000, .s = 2048, .kv8 = true },
        .{ .h = 32, .latent = 512, .rope = 64, .t = 8, .n = 4096, .s = 2048, .kv8 = false, .q_std = 1 },
        // The tiny: 2 heads (a head block of 16 padded), 16 picks a row.
        .{ .h = 2, .latent = 64, .rope = 16, .t = 7, .n = 40, .s = 16, .kv8 = true },
        .{ .h = 2, .latent = 64, .rope = 16, .t = 7, .n = 40, .s = 16, .kv8 = false },
        // The STQ tiny: 8 heads, latent 128, 16 picks a row.
        .{ .h = 8, .latent = 128, .rope = 16, .t = 16, .n = 60, .s = 16, .kv8 = true },
        .{ .h = 8, .latent = 128, .rope = 16, .t = 16, .n = 60, .s = 16, .kv8 = false },
    }, 0..) |c, ci| {
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const x = try Host.init(rnd, c.h, c.latent, c.rope, c.t, c.n, c.q_std, false);
        defer x.deinit();
        const arr = try x.arrays(&o, true);
        var qk8 = if (c.kv8) try kv_quant.quantizeAffine(es, arr.keys.c, 64, 8) else null;
        defer if (qk8) |*q| q.deinit();
        const keys: Keys = .{ .c = if (qk8) |q| try o.own(try hy4_latent.rangeBf16(es, q.q, q.scales, q.biases, 64)) else arr.keys.c, .k_pe = arr.keys.k_pe };
        // The truth reads the rows the kernel reads: the dequantized rows, at the width the store gives them.
        @memcpy(x.kc, try hostF32(&o, keys.c));
        const scale = std.math.pow(f32, @floatFromInt(3 * c.rope), -0.5);
        const n_picks: usize = @intCast(c.t * c.s);
        const idx = try testing.allocator.alloc(u32, n_picks);
        defer testing.allocator.free(idx);
        const kept = try testing.allocator.alloc(bool, n_picks);
        defer testing.allocator.free(kept);
        for (idx, kept) |*d, *k| {
            d.* = rnd.uintLessThan(u32, @intCast(c.n));
            k.* = rnd.uintLessThan(u32, 10) != 0;
        }
        // A row with no key picked at all, and one with only its last pick kept: the sink alone, a tile of masked picks.
        @memset(kept[0..@intCast(c.s)], false);
        @memset(kept[@intCast(c.s)..@intCast(2 * c.s - 1)], false);
        const rows_idx = try o.own(try mlx.newData(idx.ptr, &.{ c.t, c.s }, .uint32));
        const keep = try o.own(try mlx.newData(kept.ptr, &.{ c.t, c.s }, .bool_));
        try testing.expect(applies(es, arr.q, keys, arr.sinks));
        const got = try hostF32(&o, try o.own(try picks(es, arr.q, keys, rows_idx, keep, arr.sinks, scale, 0)));
        const lists = try testing.allocator.alloc([]u32, x.t);
        defer freeLists(lists);
        for (lists, 0..) |*l, i| {
            var k: usize = 0;
            l.* = try testing.allocator.alloc(u32, @intCast(c.s));
            for (0..@as(usize, @intCast(c.s))) |j| if (kept[i * @as(usize, @intCast(c.s)) + j]) {
                l.*[k] = idx[i * @as(usize, @intCast(c.s)) + j];
                k += 1;
            };
            l.* = try testing.allocator.realloc(l.*, k);
        }
        const want = try x.truth(scale, lists);
        defer testing.allocator.free(want);
        const d = drift(got, want);
        if (trace()) std.debug.print("hy4 attn picks case {d}: drift {e}\n", .{ ci, d });
        try testing.expect(d <= bar);
    }
}

test "hy4 attn picks: the picked rows read from the kv8 or bf16 store give the f64 truth's heads, at the release's 32 heads of 2,048 picks and the tiny's, with rows that keep none or one" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const exact = mlx.engine.exactF32();
    defer exact.end();
    picksCells(es) catch |e| {
        var said: [8192]u8 = undefined;
        std.debug.print("\nhy4 attn failed ({t}): {s}\n", .{ e, mlx.takeError(&said) orelse "(no message)" });
        return e;
    };
    if (nax_gate.verifyQmmNaxAvailable()) {
        const device = @import("sushi_exl3").device;
        const simd = device.withProfile(device.simdOnly());
        defer simd.end();
        try picksCells(es);
    }
}

fn outlierCells(es: mlx.mlx_stream) !void {
    const fwd = @import("hy4_forward.zig");
    var prng = std.Random.DefaultPrng.init(0x0071_1e75);
    const rnd = prng.random();
    const Case = struct {
        name: []const u8,
        h: c_int,
        t: c_int,
        n: c_int,
        p0: c_int,
        window: c_int,
        q_std: f32,
        k_std: f32,
        q_out: f32,
        k_out_scale: f32,
        neg_k_out: bool,
        rope_out: f32,
        sink_bias: f32,
        bar: f64,
    };
    for ([_]Case{
        .{ .name = "sliding-sink-collapse", .h = 64, .t = 16, .n = 527, .p0 = 511, .window = 512, .q_std = 4, .k_std = 1, .q_out = 32896.0, .k_out_scale = 0.001, .neg_k_out = true, .rope_out = 0, .sink_bias = 2.0, .bar = 2e-5 },
        .{ .name = "sliding-mixed-outlier", .h = 64, .t = 16, .n = 527, .p0 = 511, .window = 512, .q_std = 4, .k_std = 1, .q_out = 50048.0, .k_out_scale = 0.001, .neg_k_out = false, .rope_out = 40064.0, .sink_bias = 0, .bar = 2e-5 },
        .{ .name = "sliding-small-q", .h = 64, .t = 16, .n = 527, .p0 = 511, .window = 512, .q_std = 1e-8, .k_std = 1e8, .q_out = 0, .k_out_scale = 0, .neg_k_out = false, .rope_out = 0, .sink_bias = 0, .bar = 2e-5 },
        .{ .name = "sparse-outlier", .h = 32, .t = 16, .n = 1000, .p0 = 984, .window = 0, .q_std = 4, .k_std = 1, .q_out = 32896.0, .k_out_scale = 0.001, .neg_k_out = true, .rope_out = 0, .sink_bias = 2.0, .bar = 2e-5 },
    }) |c| {
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const x = try Host.init(rnd, c.h, 512, 64, c.t, c.n, c.q_std, false);
        defer x.deinit();
        if (c.k_std != 1) {
            for (x.kc) |*e| e.* = bf16Round(e.* * c.k_std);
            for (x.kp) |*e| e.* = bf16Round(e.* * c.k_std);
        }
        if (c.q_out != 0) {
            for (0..x.h * x.t) |i| x.ql[i * x.latent + 7] = c.q_out;
            for (0..@as(usize, @intCast(c.n))) |i| {
                const v = @abs(rnd.floatNorm(f32)) * c.k_out_scale + c.k_out_scale * 0.25;
                x.kc[i * x.latent + 7] = bf16Round(if (c.neg_k_out) -v else if (i % 2 == 0) v else -v);
            }
        }
        if (c.rope_out != 0) {
            for (0..x.h * x.t) |i| x.qp[i * x.rope + 5] = c.rope_out;
            for (0..@as(usize, @intCast(c.n))) |i| x.kp[i * x.rope + 5] = bf16Round(rnd.floatNorm(f32) * c.k_out_scale);
        }
        if (c.sink_bias != 0) for (x.sinks) |*sv| {
            sv.* += c.sink_bias;
        };
        const arr = try x.arrays(&o, true);
        const scale: f32 = 1.0 / 16.0;
        const got = try hostF32(&o, try o.own(try range(es, arr.q, arr.keys, arr.sinks, scale, c.p0, c.window, 0, c.t)));
        const g: fwd.AttnGeometry = .{ .heads = c.h, .nope = 192, .rope = 64, .v = 256, .latent = 512, .theta = 1e4, .eps = 1e-5 };
        const w: fwd.Attn = .{ .q_a = undefined, .q_a_norm = undefined, .q_b = undefined, .kv_a = undefined, .kv_a_norm = undefined, .kv_b = undefined, .gate = undefined, .o = undefined, .post_norm = undefined, .sinks = arr.sinks };
        const chain_got = try hostF32(&o, try fwd.causalLatent(&o, arr.q.lat, arr.q.pe, arr.keys.c, arr.keys.k_pe, w, g, .{ .causal = .{ .p0 = c.p0, .window = c.window } }));
        const lists = try testing.allocator.alloc([]u32, x.t);
        defer freeLists(lists);
        for (lists, 0..) |*l, i| {
            const pos: usize = @as(usize, @intCast(c.p0)) + i;
            const c0: usize = if (c.window > 0 and pos + 1 > @as(usize, @intCast(c.window))) pos + 1 - @as(usize, @intCast(c.window)) else 0;
            const c1 = @min(pos, @as(usize, @intCast(c.n)) - 1);
            l.* = try testing.allocator.alloc(u32, c1 + 1 - c0);
            for (l.*, 0..) |*e, j| e.* = @intCast(c0 + j);
        }
        const want = try x.truth(scale, lists);
        defer testing.allocator.free(want);
        const d_core = drift(got, want);
        const d_chain = drift(chain_got, want);
        if (trace()) std.debug.print("hy4 attn outliers {s}: core {e} chain {e}\n", .{ c.name, d_core, d_chain });
        if (!(d_core <= c.bar and d_core <= @max(d_chain * 2.0, bar))) {
            std.debug.print("hy4 attn outliers {s}: core {e} > bar {e} (chain {e})\n", .{ c.name, d_core, c.bar, d_chain });
            return error.TestUnexpectedResult;
        }
    }
}

test "hy4 attn outliers: release shapes with large query outliers, constant-sign key channels, sinks and small queries stay within the chain's f64 error" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const exact = mlx.engine.exactF32();
    defer exact.end();
    try outlierCells(es);
    if (nax_gate.verifyQmmNaxAvailable()) {
        const device = @import("sushi_exl3").device;
        const simd = device.withProfile(device.simdOnly());
        defer simd.end();
        try outlierCells(es);
    }
}
