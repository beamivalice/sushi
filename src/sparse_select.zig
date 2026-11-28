//! The sparse-select kernels Sushi Cooler's archs share: the indexer's relu-weighted scores over cached keys (one
//! thread a key, or a matrix tile for a wide block) and the top-k of each row (one threadgroup a row, or split in
//! chunks and merged; ties to the lower index). Each launch takes a stream and returns an array the caller frees;
//! pipeline names keep their first arch's prefix, so the warm manifests' recipes do not move.
const std = @import("std");
const mlx = @import("mlx.zig");
const metal_def = @import("metal_def.zig");
const nax_gate = @import("nax_gate.zig");
const log = @import("log.zig");

const A = mlx.mlx_array;
const Def = metal_def.Def;
const OutSpec = metal_def.OutSpec;
const Tmpl = metal_def.Tmpl;
const launch = metal_def.launch;

fn dim(x: A, axis: usize) usize {
    return @intCast(mlx.getShape(x)[axis]);
}

fn ci(v: usize) c_int {
    return @intCast(v);
}

fn ints(vals: []const i32) !A {
    return mlx.newData(vals.ptr, &.{ci(vals.len)}, .int32);
}

/// The K largest of each row of `x` [L, n] f32 (K <= n), as their indices in ascending order, ties to the lower
/// index: a radix select of the order-preserving uint keys, 8 bits a pass, then one pass writing every key above
/// the threshold and the first ties. One threadgroup a row; the split variant runs it on G chunks of a row, then on
/// the chunks' candidates (ascending in the row's index, so a tie still goes to the lower index). A variant defines
/// `n`, `xr`, `ELEM(i)` (the key of element i) and `EMIT(p, i)` (pick p is element i).
pub const TOPK_BODY =
    \\threadgroup atomic_uint hist[8][256];
    \\threadgroup uint t_sel[3], t_gt[32], t_eq[32];
    \\const uint K = uint(kk[0]);
    \\const uint bank = sg & 7u;
    \\uint prefix = 0, krem = K;
    \\bool exact_cut = false;
    \\for (int pass = 0; pass < 4; ++pass) {
    \\  const int shift = 24 - 8 * pass;
    \\  const uint hmask = pass == 0 ? 0u : (0xFFFFFFFFu << (shift + 8));
    \\  for (uint b = tid; b < 8u * 256u; b += uint(TG)) atomic_store_explicit(&hist[b >> 8][b & 255u], 0u, metal::memory_order_relaxed);
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  for (int base = int(sg) * 32; base < n; base += TG) {
    \\    const int i = base + int(lane);
    \\    const bool valid = i < n;
    \\    const uint key = valid ? ELEM(i) : 0u;
    \\    const bool m = valid && ((key & hmask) == prefix);
    \\    const uint m_mask = uint(static_cast<metal::simd_vote::vote_t>(metal::simd_ballot(m)));
    \\    if (m_mask != 0u) {
    \\      const uint b = (key >> shift) & 255u;
    \\      const uint lead0 = metal::ctz(m_mask);
    \\      const uint b0 = metal::simd_broadcast(b, ushort(lead0));
    \\      const uint h0 = uint(static_cast<metal::simd_vote::vote_t>(metal::simd_ballot(m && b == b0)));
    \\      if (lane == lead0) atomic_fetch_add_explicit(&hist[bank][b0], metal::popcount(h0), metal::memory_order_relaxed);
    \\      const uint rem = m_mask & ~h0;
    \\      if (rem != 0u) {
    \\        const uint lead1 = metal::ctz(rem);
    \\        const uint b1 = metal::simd_broadcast(b, ushort(lead1));
    \\        const uint h1 = uint(static_cast<metal::simd_vote::vote_t>(metal::simd_ballot(((rem & (1u << lane)) != 0u) && b == b1)));
    \\        if (lane == lead1) atomic_fetch_add_explicit(&hist[bank][b1], metal::popcount(h1), metal::memory_order_relaxed);
    \\        if (((rem & ~h1) & (1u << lane)) != 0u) atomic_fetch_add_explicit(&hist[bank][b], 1u, metal::memory_order_relaxed);
    \\      }
    \\    }
    \\  }
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  if (sg == 0u) {
    \\    uint c[8], tot = 0;
    \\    for (int j = 0; j < 8; ++j) {
    \\      const uint idx_b = 255u - (lane * 8u + uint(j));
    \\      uint sum_b = 0u;
    \\      #pragma clang loop unroll(full)
    \\      for (uint bk = 0u; bk < 8u; ++bk) sum_b += atomic_load_explicit(&hist[bk][idx_b], metal::memory_order_relaxed);
    \\      c[j] = sum_b;
    \\      tot += sum_b;
    \\    }
    \\    const uint before = metal::simd_prefix_exclusive_sum(tot);
    \\    const bool mine = before < krem && before + tot >= krem;
    \\    if (mine) {
    \\      uint acc = before;
    \\      int j = 0;
    \\      for (; j < 8; ++j) { if (acc + c[j] >= krem) break; acc += c[j]; }
    \\      t_sel[0] = prefix | (uint(255u - (lane * 8u + uint(j))) << shift);
    \\      t_sel[1] = krem - acc;
    \\      t_sel[2] = (acc + c[j] == krem) ? 1u : 0u;
    \\    }
    \\  }
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  prefix = t_sel[0];
    \\  krem = t_sel[1];
    \\  exact_cut = t_sel[2] != 0u;
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  if (exact_cut) break;
    \\}
    \\const int nsg = TG / 32;
    \\const int sc = (n + nsg - 1) / nsg;
    \\const int s0 = int(sg) * sc, s1 = metal::min(n, s0 + sc);
    \\uint sg_gt = 0u, sg_eq = 0u;
    \\for (int base = s0; base < s1; base += 32) {
    \\  const int i = base + int(lane);
    \\  const bool valid = i < s1;
    \\  const uint key = valid ? ELEM(i) : 0u;
    \\  const bool is_gt = valid && (exact_cut ? (key >= prefix) : (key > prefix));
    \\  const bool is_eq = valid && !exact_cut && (key == prefix);
    \\  sg_gt += metal::popcount(uint(static_cast<metal::simd_vote::vote_t>(metal::simd_ballot(is_gt))));
    \\  sg_eq += metal::popcount(uint(static_cast<metal::simd_vote::vote_t>(metal::simd_ballot(is_eq))));
    \\}
    \\if (lane == 0u) { t_gt[sg] = sg_gt; t_eq[sg] = sg_eq; }
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\uint eq_before = 0u, pos = 0u;
    \\for (uint g = 0u; g < sg; ++g) {
    \\  const uint geq = t_eq[g];
    \\  const uint gtake = eq_before >= krem ? 0u : metal::min(geq, krem - eq_before);
    \\  eq_before += geq;
    \\  pos += t_gt[g] + gtake;
    \\}
    \\const uint take_sg = eq_before >= krem ? 0u : metal::min(sg_eq, krem - eq_before);
    \\if (sg_gt + take_sg > 0u) {
    \\  uint taken = 0u;
    \\  for (int base = s0; base < s1; base += 32) {
    \\    const int i = base + int(lane);
    \\    const bool valid = i < s1;
    \\    const uint key = valid ? ELEM(i) : 0u;
    \\    const bool is_gt = valid && (exact_cut ? (key >= prefix) : (key > prefix));
    \\    const bool is_eq = valid && !exact_cut && (key == prefix);
    \\    const uint eq_m = uint(static_cast<metal::simd_vote::vote_t>(metal::simd_ballot(is_eq)));
    \\    const uint eq_before_lane = metal::popcount(eq_m & ((1u << lane) - 1u));
    \\    const bool take_me = is_eq && (taken + eq_before_lane < take_sg);
    \\    const bool pick = is_gt || take_me;
    \\    const uint pk_m = uint(static_cast<metal::simd_vote::vote_t>(metal::simd_ballot(pick)));
    \\    if (pick) {
    \\      const uint off_lane = metal::popcount(pk_m & ((1u << lane) - 1u));
    \\      EMIT(pos + off_lane, i);
    \\    }
    \\    pos += metal::popcount(pk_m);
    \\    taken += metal::popcount(eq_m);
    \\  }
    \\}
;

pub const TOPK_PRE =
    \\const uint tid = thread_index_in_threadgroup, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
;

pub const TOPK_SOURCE = TOPK_PRE ++
    \\const int l = threadgroup_position_in_grid.y, n = x_shape[1];
    \\const auto xr = x + size_t(l) * n;
    \\#define ELEM(i) dsv41_key(xr[i])
    \\#define EMIT(p, i) out[size_t(l) * K + (p)] = (i)
    \\
++ TOPK_BODY;

/// Chunk g of G of each row: its K picks as row indices, `[L, G * K]`; kk = {K, G}.
pub const TOPK_CHUNK_SOURCE = TOPK_PRE ++
    \\const int l = threadgroup_position_in_grid.y, g = threadgroup_position_in_grid.x, G = kk[1], N = x_shape[1];
    \\const size_t c0 = size_t(N) * g / G;
    \\const int n = int(size_t(N) * (g + 1) / G - c0);
    \\const auto xr = x + size_t(l) * N + c0;
    \\#define ELEM(i) dsv41_key(xr[i])
    \\#define EMIT(p, i) out[(size_t(l) * G + g) * kk[0] + (p)] = int(c0) + (i)
    \\
++ TOPK_BODY;

/// `TOPK_SOURCE` that writes -1 for a pick not under its row's length `lens[l]`.
pub const TOPK_BELOW_SOURCE = TOPK_PRE ++
    \\const int l = threadgroup_position_in_grid.y, n = x_shape[1];
    \\const auto xr = x + size_t(l) * n;
    \\#define ELEM(i) dsv41_key(xr[i])
    \\#define EMIT(p, i) out[size_t(l) * K + (p)] = (int(i) < lens[l] ? int(i) : -1)
    \\
++ TOPK_BODY;

/// The K largest of each row of `x` among its candidate indices `ids` `[L, C]` (ascending).
pub const TOPK_MERGE_SOURCE = TOPK_PRE ++
    \\const int l = threadgroup_position_in_grid.y, n = ids_shape[1];
    \\const auto xr = x + size_t(l) * x_shape[1];
    \\const auto ir = ids + size_t(l) * n;
    \\#define ELEM(i) dsv41_key(xr[ir[i]])
    \\#define EMIT(p, i) out[size_t(l) * kk[0] + (p)] = ir[i]
    \\
++ TOPK_BODY;

pub const TOPK_HEADER =
    \\inline uint dsv41_key(float f) {
    \\  const uint u = as_type<uint>(f);
    \\  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    \\}
;

/// The indexer's scores `[L, nb]`: sum over heads of weight * relu(q_h . k_n), a weight being its projection times
/// `wsc`, keys read in their cached dtype, -inf past the groups row l's query sees ((p0 + l + 1) / ratio).
/// meta = {p0, ratio}.
pub const IDX_SCORE_FIRST =
    \\const int n = thread_position_in_grid.x, l = thread_position_in_grid.y;
    \\const uint tid = thread_index_in_threadgroup, tgn = threads_per_threadgroup.x;
;

pub const IDX_SCORE_SECOND =
    \\#ifdef SUSHI_METALLIB
    \\constexpr int IH_CAP = 64;
    \\threadgroup float qs[QS_CAP];
    \\#else
    \\constexpr int IH_CAP = IH;
    \\threadgroup float qs[IH * IHD];
    \\#endif
    \\for (uint k = tid; k < uint(IH * IHD); k += tgn) qs[k] = float(q[l * IH * IHD + k]);
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\if (n >= nb) return;
    \\float dots[IH_CAP];
    \\#pragma clang loop unroll(full)
    \\for (int h = 0; h < IH_CAP; ++h) { if (h >= IH) break; dots[h] = 0.0f; }
;

pub const IDX_SCORE_DOT =
    \\for (int j = 0; j < IHD; ++j) {
    \\  const float kv = KEY(j);
    \\  #pragma clang loop unroll(full)
    \\  for (int h = 0; h < IH_CAP; ++h) { if (h >= IH) break; dots[h] += qs[h * IHD + j] * kv; }
    \\}
    \\float s = 0.0f;
    \\#pragma clang loop unroll(full)
    \\for (int h = 0; h < IH_CAP; ++h) { if (h >= IH) break; s += (float(w[l * IH + h]) * wsc[0]) * metal::max(dots[h], 0.0f); }
;

/// `IDX_SCORE_SECOND` with q in four-wide rows, for `IDX_SCORE_DOT4`.
pub const IDX_SCORE_SECOND4 =
    \\threadgroup float4 qs[IH * IHD / 4];
    \\for (uint k = tid; k < IH * IHD / 4; k += tgn) {
    \\  const device float* qp = q + size_t(l) * IH * IHD + 4 * k;
    \\  qs[k] = float4(qp[0], qp[1], qp[2], qp[3]);
    \\}
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\if (n >= nb) return;
    \\float dots[IH];
    \\for (int h = 0; h < IH; ++h) dots[h] = 0.0f;
;

/// `IDX_SCORE_DOT` reading four channels of q a threadgroup load: each head still adds channel j, j + 1, j + 2, j + 3 in
/// that order, so the scores keep their bits.
pub const IDX_SCORE_DOT4 =
    \\for (int j = 0; j < IHD; j += 4) {
    \\  const float k0 = KEY(j), k1 = KEY(j + 1), k2 = KEY(j + 2), k3 = KEY(j + 3);
    \\  for (int h = 0; h < IH; ++h) {
    \\    const float4 qv = qs[(h * IHD + j) / 4];
    \\    dots[h] += qv.x * k0;
    \\    dots[h] += qv.y * k1;
    \\    dots[h] += qv.z * k2;
    \\    dots[h] += qv.w * k3;
    \\  }
    \\}
    \\float s = 0.0f;
    \\for (int h = 0; h < IH; ++h) s += (float(w[l * IH + h]) * wsc[0]) * metal::max(dots[h], 0.0f);
;

pub const IDX_SCORE_TAIL =
    \\const int seen = (meta[0] + l + 1) / meta[1];
    \\scores[size_t(l) * nb + n] = n < seen ? s : -INFINITY;
;

pub const IDX_SCORE_LAST = IDX_SCORE_DOT ++ IDX_SCORE_TAIL;

pub const IDX_SCORE_SOURCE: [:0]const u8 = IDX_SCORE_FIRST ++
    \\const int nb = keys_shape[0];
    \\
++ IDX_SCORE_SECOND ++
    \\const device KT* kr = keys + size_t(n) * IHD;
    \\#define KEY(j) float(kr[j])
    \\
++ IDX_SCORE_LAST;

/// `IDX_SCORE_SOURCE` for a wide block: one threadgroup a (query, KT-key tile), its four simdgroups each KT/4 keys x
/// every head on 8x8 matrix ops over the keys staged in threadgroup memory; a head tile is relu'd, weighted and
/// summed as it completes, so no [L, IH, nb] scratch. A tile the query cannot see is -inf without the work. `q`
/// arrives in the keys' dtype, so the cast is exact. meta = {p0, ratio, key offset}: `keys` may be a slice of a longer
/// key list starting at that offset.
pub const IDX_TILE_SOURCE =
    \\const int l = threadgroup_position_in_grid.y, n0 = threadgroup_position_in_grid.x * KT;
    \\const uint tid = thread_index_in_threadgroup, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    \\const int nb = keys_shape[0];
    \\const int seen = (meta[0] + l + 1) / meta[1] - meta[2];
    \\const int end = metal::min(nb, n0 + KT);
    \\if (n0 >= seen) {
    \\  for (int n = n0 + int(tid); n < end; n += 128) scores[size_t(l) * nb + n] = -INFINITY;
    \\  return;
    \\}
    \\#ifdef SUSHI_METALLIB
    \\constexpr int IH_CAP = 64, IHD_CAP = 128, RT_CAP = IH_CAP / 8;
    \\#else
    \\constexpr int IH_CAP = IH, IHD_CAP = IHD, RT_CAP = IH / 8;
    \\#endif
    \\const int RT = IH / 8, C4 = IHD / 4;
    \\constexpr int CT = KT / 32;
    \\using T4 = metal::vec<T, 4>;
    \\threadgroup T kt[KT * IHD_CAP];
    \\threadgroup float wt[IH_CAP], red[4][64];
    \\for (uint i = tid; i < uint(KT * C4); i += 128) {
    \\  const int n = n0 + int(i / uint(C4));
    \\  ((threadgroup T4*)kt)[i] = n < nb ? ((const device T4*)keys)[size_t(n) * C4 + i % uint(C4)] : T4(0);
    \\}
    \\if (int(tid) < IH) wt[tid] = float(w[size_t(l) * IH + tid]) * wsc[0];
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\const device T* qr = q + size_t(l) * IH * IHD;
    \\const threadgroup T* kr = kt + sg * (KT / 4) * IHD;
    \\metal::simdgroup_matrix<float, 8, 8> acc[RT_CAP][CT];
    \\#pragma clang loop unroll(full)
    \\for (int r = 0; r < RT_CAP; ++r) {
    \\  if (r >= RT) break;
    \\  #pragma clang loop unroll(full)
    \\  for (int c = 0; c < CT; ++c) acc[r][c] = metal::make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    \\}
    \\for (int d = 0; d < IHD; d += 8) {
    \\  metal::simdgroup_matrix<T, 8, 8> a[RT_CAP], b[CT];
    \\  #pragma clang loop unroll(full)
    \\  for (int r = 0; r < RT_CAP; ++r) { if (r >= RT) break; metal::simdgroup_load(a[r], qr + r * 8 * IHD + d, IHD); }
    \\  #pragma clang loop unroll(full)
    \\  for (int c = 0; c < CT; ++c) metal::simdgroup_load(b[c], kr + c * 8 * IHD + d, IHD, metal::ulong2(0, 0), true);
    \\  #pragma clang loop unroll(full)
    \\  for (int r = 0; r < RT_CAP; ++r) {
    \\    if (r >= RT) break;
    \\    #pragma clang loop unroll(full)
    \\    for (int c = 0; c < CT; ++c) metal::simdgroup_multiply_accumulate(acc[r][c], a[r], b[c], acc[r][c]);
    \\  }
    \\}
    \\// A finished tile's rows are heads: relu, weight and sum them through this simdgroup's scratch.
    \\// Threadgroup barriers: on M1 a simdgroup barrier let lanes read the scratch before the store landed.
    \\const int col = lane & 7, row = (lane >> 3) * 2;
    \\float s[CT];
    \\#pragma clang loop unroll(full)
    \\for (int c = 0; c < CT; ++c) s[c] = 0.0f;
    \\#pragma clang loop unroll(full)
    \\for (int r = 0; r < RT_CAP; ++r) {
    \\  if (r < RT) {
    \\    #pragma clang loop unroll(full)
    \\    for (int c = 0; c < CT; ++c) {
    \\      metal::simdgroup_store(acc[r][c], red[sg], 8);
    \\      metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\      float p = metal::max(red[sg][row * 8 + col], 0.0f) * wt[r * 8 + row] + metal::max(red[sg][(row + 1) * 8 + col], 0.0f) * wt[r * 8 + row + 1];
    \\      p += metal::simd_shuffle_xor(p, 8);
    \\      p += metal::simd_shuffle_xor(p, 16);
    \\      s[c] += p;
    \\      metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\    }
    \\  }
    \\}
    \\if (lane < 8)
    \\  for (int c = 0; c < CT; ++c) {
    \\    const int n = n0 + int(sg) * (KT / 4) + c * 8 + col;
    \\    if (n < nb) scores[size_t(l) * nb + n] = n < seen ? s[c] : -INFINITY;
    \\  }
;

var topk_def = Def{ .name = "dsv41_topk", .ins = &.{ "x", "kk" }, .outs = &.{"out"}, .source = TOPK_SOURCE, .header = TOPK_HEADER };
var topk_below_def = Def{ .name = "dsv41_topk_below", .ins = &.{ "x", "kk", "lens" }, .outs = &.{"out"}, .source = TOPK_BELOW_SOURCE, .header = TOPK_HEADER };
var topk_chunk_def = Def{ .name = "dsv41_topk_chunk", .ins = &.{ "x", "kk" }, .outs = &.{"out"}, .source = TOPK_CHUNK_SOURCE, .header = TOPK_HEADER };
var topk_merge_def = Def{ .name = "dsv41_topk_merge", .ins = &.{ "x", "ids", "kk" }, .outs = &.{"out"}, .source = TOPK_MERGE_SOURCE, .header = TOPK_HEADER };
var idx_score_def = Def{ .name = "dsv41_index_scores", .ins = &.{ "q", "keys", "w", "meta", "wsc" }, .outs = &.{"scores"}, .source = IDX_SCORE_SOURCE };
var idx_tile_def = Def{ .name = "dsv41_index_tile", .ins = &.{ "q", "keys", "w", "meta", "wsc" }, .outs = &.{"scores"}, .source = IDX_TILE_SOURCE };

/// `IDX_TILE_SOURCE` on the matrix units (`dsv41_nax_mma<16, 32>`) for IH=32, IHD=128, bf16 operands: 8 query rows
/// per threadgroup (1 per simdgroup) share each cooperatively loaded 32-key tile in transposed threadgroup memory,
/// keeping every query head in registers across IT=4 steps (128 keys) and reducing all 32 heads via 3 register
/// shuffles with zero threadgroup scratch and zero reduction barriers.
pub const IDX_TILE_NAX_SOURCE =
    \\using B4 = metal::vec<bfloat, 4>;
    \\const uint tid = thread_index_in_threadgroup, sg = simdgroup_index_in_threadgroup;
    \\const ushort lane = thread_index_in_simdgroup;
    \\const int L = q_shape[0], nb = keys_shape[0];
    \\const int l0 = threadgroup_position_in_grid.y * 8;
    \\const int l = l0 + int(sg);
    \\const int l_max = metal::min(l0 + 7, L - 1);
    \\const int tile0 = threadgroup_position_in_grid.x * (IT * 32);
    \\const int max_seen = (meta[0] + l_max + 1) / meta[1] - meta[2];
    \\if (tile0 >= max_seen) {
    \\  const int end = metal::min(nb, tile0 + IT * 32);
    \\  for (int rr = l0; rr <= l_max; ++rr)
    \\    for (int n = tile0 + int(tid); n < end; n += 256) scores[size_t(rr) * nb + n] = -INFINITY;
    \\  return;
    \\}
    \\const int seen = l < L ? ((meta[0] + l + 1) / meta[1] - meta[2]) : 0;
    \\const short r0 = dsv41_nax_r(lane, 0), cb = dsv41_nax_c(lane, 0);
    \\bfloat a[4][2][16];
    \\float w0 = 0.0f, w1 = 0.0f, w2 = 0.0f, w3 = 0.0f;
    \\if (l < L) {
    \\  const float sc = wsc[0];
    \\  const size_t wb = size_t(l) * 32;
    \\  w0 = float(w[wb + r0]) * sc;
    \\  w1 = float(w[wb + r0 + 8]) * sc;
    \\  w2 = float(w[wb + r0 + 16]) * sc;
    \\  w3 = float(w[wb + r0 + 24]) * sc;
    \\  const device bfloat* qr = q + size_t(l) * 32 * 128;
    \\  #pragma clang loop unroll(full)
    \\  for (int hh = 0; hh < 2; ++hh) {
    \\    const device bfloat* qh0 = qr + (hh * 16 + r0) * 128;
    \\    const device bfloat* qh1 = qr + (hh * 16 + r0 + 8) * 128;
    \\    #pragma clang loop unroll(full)
    \\    for (int g = 0; g < 4; ++g) {
    \\      const B4 v0 = *(const device B4*)(qh0 + g * 32 + cb);
    \\      const B4 v1 = *(const device B4*)(qh1 + g * 32 + cb);
    \\      const B4 v2 = *(const device B4*)(qh0 + g * 32 + 16 + cb);
    \\      const B4 v3 = *(const device B4*)(qh1 + g * 32 + 16 + cb);
    \\      a[g][hh][0] = v0[0]; a[g][hh][1] = v0[1]; a[g][hh][2] = v0[2]; a[g][hh][3] = v0[3];
    \\      a[g][hh][4] = v1[0]; a[g][hh][5] = v1[1]; a[g][hh][6] = v1[2]; a[g][hh][7] = v1[3];
    \\      a[g][hh][8] = v2[0]; a[g][hh][9] = v2[1]; a[g][hh][10] = v2[2]; a[g][hh][11] = v2[3];
    \\      a[g][hh][12] = v3[0]; a[g][hh][13] = v3[1]; a[g][hh][14] = v3[2]; a[g][hh][15] = v3[3];
    \\    }
    \\  }
    \\}
    \\threadgroup bfloat kt[128 * 36];
    \\const int k_local = int(tid) >> 3;
    \\const int c_chunk = (int(tid) & 7) * 16;
    \\for (int it = 0; it < IT; ++it) {
    \\  const int n0 = tile0 + it * 32;
    \\  if (n0 >= nb) break;
    \\  if (n0 >= max_seen) {
    \\    const int end = metal::min(nb, tile0 + IT * 32);
    \\    for (int rr = l0; rr <= l_max; ++rr)
    \\      for (int n = n0 + int(tid); n < end; n += 256) scores[size_t(rr) * nb + n] = -INFINITY;
    \\    break;
    \\  }
    \\  const int kn = n0 + k_local;
    \\  if (kn < nb) {
    \\    const device B4* src = (const device B4*)(keys + size_t(kn) * 128 + c_chunk);
    \\    #pragma clang loop unroll(full)
    \\    for (int v = 0; v < 4; ++v) {
    \\      const B4 vals = src[v];
    \\      kt[(c_chunk + v * 4 + 0) * 36 + k_local] = vals[0];
    \\      kt[(c_chunk + v * 4 + 1) * 36 + k_local] = vals[1];
    \\      kt[(c_chunk + v * 4 + 2) * 36 + k_local] = vals[2];
    \\      kt[(c_chunk + v * 4 + 3) * 36 + k_local] = vals[3];
    \\    }
    \\  } else {
    \\    #pragma clang loop unroll(full)
    \\    for (int j = 0; j < 16; ++j) kt[(c_chunk + j) * 36 + k_local] = bfloat(0.0f);
    \\  }
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  if (l < L) {
    \\    #pragma clang loop unroll(full)
    \\    for (int u = 0; u < 2; ++u) {
    \\      const int uoff = u * 16 + int(cb);
    \\      float c0[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    \\      float c1[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    \\      #pragma clang loop unroll(full)
    \\      for (int g = 0; g < 4; ++g) {
    \\        bfloat b[16];
    \\        const B4 b0 = *(const threadgroup B4*)(kt + (g * 32 + r0) * 36 + uoff);
    \\        const B4 b1 = *(const threadgroup B4*)(kt + (g * 32 + r0 + 8) * 36 + uoff);
    \\        const B4 b2 = *(const threadgroup B4*)(kt + (g * 32 + r0 + 16) * 36 + uoff);
    \\        const B4 b3 = *(const threadgroup B4*)(kt + (g * 32 + r0 + 24) * 36 + uoff);
    \\        b[0] = b0[0]; b[1] = b0[1]; b[2] = b0[2]; b[3] = b0[3];
    \\        b[4] = b1[0]; b[5] = b1[1]; b[6] = b1[2]; b[7] = b1[3];
    \\        b[8] = b2[0]; b[9] = b2[1]; b[10] = b2[2]; b[11] = b2[3];
    \\        b[12] = b3[0]; b[13] = b3[1]; b[14] = b3[2]; b[15] = b3[3];
    \\        dsv41_nax_mma<16, 32>(c0, a[g][0], b);
    \\        dsv41_nax_mma<16, 32>(c1, a[g][1], b);
    \\      }
    \\      float s[4];
    \\      #pragma clang loop unroll(full)
    \\      for (int j = 0; j < 4; ++j) {
    \\        float acc = metal::max(c0[j], 0.0f) * w0
    \\                  + metal::max(c0[4 + j], 0.0f) * w1
    \\                  + metal::max(c1[j], 0.0f) * w2
    \\                  + metal::max(c1[4 + j], 0.0f) * w3;
    \\        acc += metal::simd_shuffle_xor(acc, ushort(2));
    \\        acc += metal::simd_shuffle_xor(acc, ushort(4));
    \\        acc += metal::simd_shuffle_xor(acc, ushort(16));
    \\        s[j] = acc;
    \\      }
    \\      if (r0 == 0) {
    \\        #pragma clang loop unroll(full)
    \\        for (int j = 0; j < 4; ++j) {
    \\          const int n = n0 + u * 16 + int(cb) + j;
    \\          if (n < nb) scores[size_t(l) * nb + n] = n < seen ? s[j] : -INFINITY;
    \\        }
    \\      }
    \\    }
    \\  }
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\}
;
var idx_tile_nax_def = Def{ .name = "dsv41_index_tile_nax", .ins = &.{ "q", "keys", "w", "meta", "wsc" }, .outs = &.{"scores"}, .source = IDX_TILE_NAX_SOURCE, .header = metal_def.NAX_HEADER };

/// Threads a top-k threadgroup runs; `probeTopk` halves it where the GPU caps a kernel lower.
pub var topk_tg: usize = 1024;

/// Launches the top-k kernels (one threadgroup a row, chunked and merged, cut at each row's length) at `topk_tg`
/// threads, halving it until they run: Metal grants them fewer than 1024 threads a group on some GPUs (M1, M2). Every
/// arch whose forward launches one runs this at its load, before a plan records a launch; all four are launched, so
/// the width is the GPU's, whichever arch loads first.
pub fn probeTopk(s: mlx.mlx_stream) !void {
    defer mlx.expected_refusal = null;
    const widest = topk_tg;
    while (true) {
        // The floor's refusal is the load's error, logged as one.
        mlx.expected_refusal = if (topk_tg > 32) "Thread group size" else null;
        probeTopkOnce(s) catch |e| {
            if (e != error.MlxError or topk_tg <= 32 or !mlx.takeErrorIf("Thread group size")) return e;
            topk_tg /= 2;
            continue;
        };
        if (topk_tg < widest) log.info("[engine] top-k kernels run {d} threads a group on this GPU (widest {d})\n", .{ topk_tg, widest });
        return;
    }
}

fn probeTopkOnce(s: mlx.mlx_stream) !void {
    const host = mlx.engine.enter(s);
    defer host.end();
    const zeros: [64]f32 = @splat(0);
    const x = try mlx.newData(&zeros, &.{ 1, 64 }, .float32);
    defer _ = mlx.mlx_array_free(x);
    const lens = try mlx.newData(&[_]i32{64}, &.{ 1, 1 }, .int32);
    defer _ = mlx.mlx_array_free(lens);
    var outs: [3]A = undefined;
    var made: usize = 0;
    defer for (outs[0..made]) |o| {
        _ = mlx.mlx_array_free(o);
    };
    outs[0] = try topkIndicesSplit(s, x, 4, 1);
    made = 1;
    outs[1] = try topkIndicesSplit(s, x, 4, 2);
    made = 2;
    outs[2] = try topkIndicesBelow(s, x, 1, 4, lens);
    made = 3;
    const vec = mlx.mlx_vector_array_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_eval(vec));
}

/// Chunks a row of `n` scores splits into for its top-k. Past these sizes one threadgroup a row leaves the GPU idle;
/// below them, or at a wide k, the merge costs what the split saves (`sushi-ubench run dsv41.topk`).
pub fn topkSplits(rows: usize, n: usize, k: usize) usize {
    if (rows > topk_split_rows or k > topk_split_max_k) return 1;
    return if (n >= (if (rows == 1) topk_split_min_n_one_row else topk_split_min_n)) topk_splits else 1;
}

const topk_split_rows = 16;
const topk_split_max_k = 512;
pub const topk_split_min_n_one_row: usize = 131072;
pub const topk_split_min_n: usize = 262144;
pub const topk_splits: usize = 8;

/// The indices of the `k` largest of each row of `x` `[L, n]` (f32), ascending, ties to the lower index.
pub fn topkIndices(s: mlx.mlx_stream, x: A, k: usize) !A {
    return topkIndicesSplit(s, x, k, topkSplits(dim(x, 0), dim(x, 1), k));
}

/// `topkIndices` of `x` `[rows, n]`, which `topkSplits` leaves in one threadgroup a row, with every pick not under its
/// row's `lens[row]` (int32 `[rows, 1]`) given as -1, in the one launch.
pub fn topkIndicesBelow(s: mlx.mlx_stream, x: A, rows: usize, k: usize, lens: A) !A {
    const kk = try ints(&.{@intCast(k)});
    defer _ = mlx.mlx_array_free(kk);
    var o: [1]A = undefined;
    try launch(s, &topk_below_def, &.{ x, kk, lens }, &.{.{ .shape = &.{ ci(rows), ci(k) }, .dt = .int32 }}, .{ topk_tg, rows, 1 }, .{ topk_tg, 1, 1 }, &.{.{ .name = "TG", .v = ci(topk_tg) }}, &.{}, &o);
    return o[0];
}

/// `topkIndices` over `splits` chunks of each row (1: one threadgroup a row); each chunk holds at least `k` scores.
pub fn topkIndicesSplit(s: mlx.mlx_stream, x: A, k: usize, splits: usize) !A {
    const L = dim(x, 0);
    var o: [1]A = undefined;
    const tmpl: []const Tmpl = &.{.{ .name = "TG", .v = ci(topk_tg) }};
    const spec: []const OutSpec = &.{.{ .shape = &.{ ci(L), ci(k) }, .dt = .int32 }};
    // k rides in a buffer: as a template arg every index width up to index_topk was its own Metal compile.
    const kk = try ints(&.{@intCast(k)});
    defer _ = mlx.mlx_array_free(kk);
    if (splits <= 1) {
        try launch(s, &topk_def, &.{ x, kk }, spec, .{ topk_tg, L, 1 }, .{ topk_tg, 1, 1 }, tmpl, &.{}, &o);
        return o[0];
    }
    const ks = try ints(&.{ @intCast(k), @intCast(splits) });
    defer _ = mlx.mlx_array_free(ks);
    var cand: [1]A = undefined;
    try launch(s, &topk_chunk_def, &.{ x, ks }, &.{.{ .shape = &.{ ci(L), ci(splits * k) }, .dt = .int32 }}, .{ topk_tg * splits, L, 1 }, .{ topk_tg, 1, 1 }, tmpl, &.{}, &cand);
    defer _ = mlx.mlx_array_free(cand[0]);
    try launch(s, &topk_merge_def, &.{ x, cand[0], kk }, spec, .{ topk_tg, L, 1 }, .{ topk_tg, 1, 1 }, tmpl, &.{}, &o);
    return o[0];
}

/// Keys one threadgroup of `indexScoresTile` scores for keys of `dt`: a slice scored at an offset must start on one.
pub fn indexKeyTile(dt: mlx.mlx_dtype) usize {
    return if (dt == .float32) 32 else 64;
}

/// Whether `indexScoresTile` serves `ih` heads of `ihd` channels (the matrix tiles need multiples of 8).
pub fn indexTileFits(ih: usize, ihd: usize, nb: usize) bool {
    return nb > 0 and ih >= 8 and ih <= 64 and ih % 8 == 0 and ihd >= 8 and ihd <= 128 and ihd % 8 == 0;
}

/// Threadgroup SRAM bucket for `dsv41_index_scores` (`qs[QS_CAP]`), keeping Hy4 at 8 KiB and DS41/GLM-5 at 16 KiB.
pub fn indexScoresQsCap(ih: usize, ihd: usize) usize {
    const elems = ih * ihd;
    return if (elems <= 2048) 2048 else if (elems <= 4096) 4096 else 8192;
}

/// Scores `[L, nb]` of `q` `[L, IH, IHD]` (f32) against `keys` `[nb, IHD]`, one thread a key: the decode arm. `w`
/// `[L, IH]` are the head weights, times `wscale`; a key past the groups row l sees, `(p0 + l + 1) / ratio`, is -inf.
pub fn indexScores(s: mlx.mlx_stream, q: A, keys: A, w: A, wscale: f32, p0: i32, ratio: usize) !A {
    const L = dim(q, 0);
    const ih = dim(q, 1);
    const ihd = dim(q, 2);
    if (ih == 0 or ih > 64 or ihd == 0 or ihd > 128 or ih * ihd > 8192) return error.UnsupportedIndexGeometry;
    const nb = dim(keys, 0);
    const meta = try ints(&.{ p0, @intCast(ratio) });
    defer _ = mlx.mlx_array_free(meta);
    const wsc = try mlx.newData(&[_]f32{wscale}, &.{1}, .float32);
    defer _ = mlx.mlx_array_free(wsc);
    const tg: usize = 256;
    var o: [1]A = undefined;
    try launch(s, &idx_score_def, &.{ q, keys, w, meta, wsc }, &.{.{ .shape = &.{ ci(L), ci(nb) }, .dt = .float32 }}, .{ (nb + tg - 1) / tg * tg, L, 1 }, .{ tg, 1, 1 }, &.{
        .{ .name = "QS_CAP", .v = ci(indexScoresQsCap(ih, ihd)) }, .{ .name = "IH", .v = ci(ih) }, .{ .name = "IHD", .v = ci(ihd) },
    }, &.{.{ .name = "KT", .dt = mlx.mlx_array_dtype(keys) }}, &o);
    return o[0];
}

/// `indexScores` for a wide block on 8x8 matrix ops; `q` in the keys' dtype. `keys` are the list's rows from `key0`
/// on, a multiple of the kernel's key tile, so a slice scores the bits the whole list's columns would.
pub fn indexScoresTile(s: mlx.mlx_stream, q: A, keys: A, w: A, wscale: f32, p0: i32, ratio: usize, key0: usize) !A {
    const L = dim(q, 0);
    const ih = dim(q, 1);
    const ihd = dim(keys, 1);
    const nb = dim(keys, 0);
    if (!indexTileFits(ih, ihd, nb)) return error.UnsupportedIndexGeometry;
    const dt = mlx.mlx_array_dtype(keys);
    const kt = indexKeyTile(dt);
    std.debug.assert(key0 % kt == 0 and mlx.mlx_array_dtype(q) == dt);
    const meta = try ints(&.{ p0, @intCast(ratio), @intCast(key0) });
    defer _ = mlx.mlx_array_free(meta);
    const wsc = try mlx.newData(&[_]f32{wscale}, &.{1}, .float32);
    defer _ = mlx.mlx_array_free(wsc);
    var o: [1]A = undefined;
    if (dt == .bfloat16 and ih == 32 and ihd == 128 and mlx.engine.isStream(s) and nax_gate.verifyQmmNaxAvailable()) {
        const it: usize = 4;
        const step = it * 32;
        try launch(s, &idx_tile_nax_def, &.{ q, keys, w, meta, wsc }, &.{.{ .shape = &.{ ci(L), ci(nb) }, .dt = .float32 }}, .{ (nb + step - 1) / step * 256, (L + 7) / 8, 1 }, .{ 256, 1, 1 }, &.{
            .{ .name = "IT", .v = ci(it) },
        }, &.{}, &o);
        return o[0];
    }
    try launch(s, &idx_tile_def, &.{ q, keys, w, meta, wsc }, &.{.{ .shape = &.{ ci(L), ci(nb) }, .dt = .float32 }}, .{ (nb + kt - 1) / kt * 128, L, 1 }, .{ 128, 1, 1 }, &.{
        .{ .name = "IH", .v = ci(ih) }, .{ .name = "IHD", .v = ci(ihd) }, .{ .name = "KT", .v = ci(kt) },
    }, &.{.{ .name = "T", .dt = dt }}, &o);
    return o[0];
}
