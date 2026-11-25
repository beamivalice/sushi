//! The sparse-select kernels Sushi Cooler's archs share: the indexer's relu-weighted scores over cached keys (one
//! thread a key, or a matrix tile for a wide block) and the top-k of each row (one threadgroup a row, or split in
//! chunks and merged; ties to the lower index). Each launch takes a stream and returns an array the caller frees;
//! pipeline names keep their first arch's prefix, so the warm manifests' recipes do not move.
const std = @import("std");
const mlx = @import("mlx.zig");
const metal_def = @import("metal_def.zig");

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
    \\threadgroup atomic_uint hist[256];
    \\threadgroup uint t_sel[2], t_part[TG / 32];
    \\const uint K = uint(kk[0]);
    \\uint prefix = 0, krem = K;
    \\for (int pass = 0; pass < 4; ++pass) {
    \\  const int shift = 24 - 8 * pass;
    \\  const uint hmask = pass == 0 ? 0u : (0xFFFFFFFFu << (shift + 8));
    \\  for (uint b = tid; b < 256; b += TG) atomic_store_explicit(&hist[b], 0u, metal::memory_order_relaxed);
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  for (int i = int(tid); i < n; i += TG) {
    \\    const uint key = ELEM(i);
    \\    if ((key & hmask) == prefix) atomic_fetch_add_explicit(&hist[(key >> shift) & 255u], 1u, metal::memory_order_relaxed);
    \\  }
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  if (sg == 0) {
    \\    uint c[8], tot = 0;
    \\    for (int j = 0; j < 8; ++j) { c[j] = atomic_load_explicit(&hist[255 - (lane * 8 + j)], metal::memory_order_relaxed); tot += c[j]; }
    \\    const uint before = metal::simd_prefix_exclusive_sum(tot);
    \\    const bool mine = before < krem && before + tot >= krem;
    \\    if (mine) {
    \\      uint acc = before;
    \\      int j = 0;
    \\      for (; j < 8; ++j) { if (acc + c[j] >= krem) break; acc += c[j]; }
    \\      t_sel[0] = prefix | (uint(255 - (lane * 8 + j)) << shift);
    \\      t_sel[1] = krem - acc;
    \\    }
    \\  }
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  prefix = t_sel[0];
    \\  krem = t_sel[1];
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\}
    \\const int C = (n + TG - 1) / TG, i0 = int(tid) * C, i1 = metal::min(n, i0 + C);
    \\uint gt = 0, eq = 0;
    \\for (int i = i0; i < i1; ++i) { const uint key = ELEM(i); gt += key > prefix; eq += key == prefix; }
    \\uint s = metal::simd_prefix_exclusive_sum(eq);
    \\if (lane == 31) t_part[sg] = s + eq;
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\uint off = 0;
    \\for (uint g = 0; g < sg; ++g) off += t_part[g];
    \\const uint eq_before = s + off;
    \\const uint take = eq_before >= krem ? 0u : metal::min(eq, krem - eq_before);
    \\const uint sel = gt + take;
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\s = metal::simd_prefix_exclusive_sum(sel);
    \\if (lane == 31) t_part[sg] = s + sel;
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\off = 0;
    \\for (uint g = 0; g < sg; ++g) off += t_part[g];
    \\uint pos = s + off, taken = 0;
    \\for (int i = i0; i < i1; ++i) {
    \\  const uint key = ELEM(i);
    \\  if (key > prefix || (key == prefix && taken < take)) {
    \\    taken += key == prefix;
    \\    EMIT(pos, i);
    \\    ++pos;
    \\  }
    \\}
;

pub const TOPK_PRE =
    \\const uint tid = thread_index_in_threadgroup, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
;

const TOPK_SOURCE = TOPK_PRE ++
    \\const int l = threadgroup_position_in_grid.y, n = x_shape[1];
    \\const auto xr = x + size_t(l) * n;
    \\#define ELEM(i) dsv41_key(xr[i])
    \\#define EMIT(p, i) out[size_t(l) * K + (p)] = (i)
    \\
++ TOPK_BODY;

/// Chunk g of G of each row: its K picks as row indices, `[L, G * K]`; kk = {K, G}.
const TOPK_CHUNK_SOURCE = TOPK_PRE ++
    \\const int l = threadgroup_position_in_grid.y, g = threadgroup_position_in_grid.x, G = kk[1], N = x_shape[1];
    \\const size_t c0 = size_t(N) * g / G;
    \\const int n = int(size_t(N) * (g + 1) / G - c0);
    \\const auto xr = x + size_t(l) * N + c0;
    \\#define ELEM(i) dsv41_key(xr[i])
    \\#define EMIT(p, i) out[(size_t(l) * G + g) * kk[0] + (p)] = int(c0) + (i)
    \\
++ TOPK_BODY;

/// `TOPK_SOURCE` that writes -1 for a pick not under its row's length `lens[l]`.
const TOPK_BELOW_SOURCE = TOPK_PRE ++
    \\const int l = threadgroup_position_in_grid.y, n = x_shape[1];
    \\const auto xr = x + size_t(l) * n;
    \\#define ELEM(i) dsv41_key(xr[i])
    \\#define EMIT(p, i) out[size_t(l) * K + (p)] = (int(i) < lens[l] ? int(i) : -1)
    \\
++ TOPK_BODY;

/// The K largest of each row of `x` among its candidate indices `ids` `[L, C]` (ascending).
const TOPK_MERGE_SOURCE = TOPK_PRE ++
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
    \\threadgroup float qs[IH * IHD];
    \\for (uint k = tid; k < IH * IHD; k += tgn) qs[k] = float(q[l * IH * IHD + k]);
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\if (n >= nb) return;
    \\float dots[IH];
    \\for (int h = 0; h < IH; ++h) dots[h] = 0.0f;
;

pub const IDX_SCORE_DOT =
    \\for (int j = 0; j < IHD; ++j) {
    \\  const float kv = KEY(j);
    \\  for (int h = 0; h < IH; ++h) dots[h] += qs[h * IHD + j] * kv;
    \\}
    \\float s = 0.0f;
    \\for (int h = 0; h < IH; ++h) s += (float(w[l * IH + h]) * wsc[0]) * metal::max(dots[h], 0.0f);
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

const IDX_SCORE_SOURCE: [:0]const u8 = IDX_SCORE_FIRST ++
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
const IDX_TILE_SOURCE =
    \\const int l = threadgroup_position_in_grid.y, n0 = threadgroup_position_in_grid.x * KT;
    \\const uint tid = thread_index_in_threadgroup, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    \\const int nb = keys_shape[0];
    \\const int seen = (meta[0] + l + 1) / meta[1] - meta[2];
    \\const int end = metal::min(nb, n0 + KT);
    \\if (n0 >= seen) {
    \\  for (int n = n0 + int(tid); n < end; n += 128) scores[size_t(l) * nb + n] = -INFINITY;
    \\  return;
    \\}
    \\constexpr int RT = IH / 8, CT = KT / 32, C4 = IHD / 4;
    \\using T4 = metal::vec<T, 4>;
    \\threadgroup T kt[KT * IHD];
    \\threadgroup float wt[IH], red[4][64];
    \\for (uint i = tid; i < KT * C4; i += 128) {
    \\  const int n = n0 + int(i / C4);
    \\  ((threadgroup T4*)kt)[i] = n < nb ? ((const device T4*)keys)[size_t(n) * C4 + i % C4] : T4(0);
    \\}
    \\if (tid < IH) wt[tid] = float(w[size_t(l) * IH + tid]) * wsc[0];
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\const device T* qr = q + size_t(l) * IH * IHD;
    \\const threadgroup T* kr = kt + sg * (KT / 4) * IHD;
    \\metal::simdgroup_matrix<float, 8, 8> acc[RT][CT];
    \\#pragma clang loop unroll(full)
    \\for (int r = 0; r < RT; ++r)
    \\  #pragma clang loop unroll(full)
    \\  for (int c = 0; c < CT; ++c) acc[r][c] = metal::make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    \\for (int d = 0; d < IHD; d += 8) {
    \\  metal::simdgroup_matrix<T, 8, 8> a[RT], b[CT];
    \\  #pragma clang loop unroll(full)
    \\  for (int r = 0; r < RT; ++r) metal::simdgroup_load(a[r], qr + r * 8 * IHD + d, IHD);
    \\  #pragma clang loop unroll(full)
    \\  for (int c = 0; c < CT; ++c) metal::simdgroup_load(b[c], kr + c * 8 * IHD + d, IHD, metal::ulong2(0, 0), true);
    \\  #pragma clang loop unroll(full)
    \\  for (int r = 0; r < RT; ++r)
    \\    #pragma clang loop unroll(full)
    \\    for (int c = 0; c < CT; ++c) metal::simdgroup_multiply_accumulate(acc[r][c], a[r], b[c], acc[r][c]);
    \\}
    \\// A finished tile's rows are heads: relu, weight and sum them through this simdgroup's scratch.
    \\// Threadgroup barriers: on M1 a simdgroup barrier let lanes read the scratch before the store landed.
    \\const int col = lane & 7, row = (lane >> 3) * 2;
    \\float s[CT];
    \\#pragma clang loop unroll(full)
    \\for (int c = 0; c < CT; ++c) s[c] = 0.0f;
    \\#pragma clang loop unroll(full)
    \\for (int r = 0; r < RT; ++r)
    \\  #pragma clang loop unroll(full)
    \\  for (int c = 0; c < CT; ++c) {
    \\    metal::simdgroup_store(acc[r][c], red[sg], 8);
    \\    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\    float p = metal::max(red[sg][row * 8 + col], 0.0f) * wt[r * 8 + row] + metal::max(red[sg][(row + 1) * 8 + col], 0.0f) * wt[r * 8 + row + 1];
    \\    p += metal::simd_shuffle_xor(p, 8);
    \\    p += metal::simd_shuffle_xor(p, 16);
    \\    s[c] += p;
    \\    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  }
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

/// Threads a top-k threadgroup runs; an arch's probe halves it where the GPU caps the kernel lower.
pub var topk_tg: usize = 1024;

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
    return nb > 0 and ih % 8 == 0 and ih <= 128 and ihd % 8 == 0 and ihd % 4 == 0;
}

/// Scores `[L, nb]` of `q` `[L, IH, IHD]` (f32) against `keys` `[nb, IHD]`, one thread a key: the decode arm. `w`
/// `[L, IH]` are the head weights, times `wscale`; a key past the groups row l sees, `(p0 + l + 1) / ratio`, is -inf.
pub fn indexScores(s: mlx.mlx_stream, q: A, keys: A, w: A, wscale: f32, p0: i32, ratio: usize) !A {
    const L = dim(q, 0);
    const nb = dim(keys, 0);
    const meta = try ints(&.{ p0, @intCast(ratio) });
    defer _ = mlx.mlx_array_free(meta);
    const wsc = try mlx.newData(&[_]f32{wscale}, &.{1}, .float32);
    defer _ = mlx.mlx_array_free(wsc);
    const tg: usize = 256;
    var o: [1]A = undefined;
    try launch(s, &idx_score_def, &.{ q, keys, w, meta, wsc }, &.{.{ .shape = &.{ ci(L), ci(nb) }, .dt = .float32 }}, .{ (nb + tg - 1) / tg * tg, L, 1 }, .{ tg, 1, 1 }, &.{
        .{ .name = "IH", .v = ci(dim(q, 1)) }, .{ .name = "IHD", .v = ci(dim(q, 2)) },
    }, &.{.{ .name = "KT", .dt = mlx.mlx_array_dtype(keys) }}, &o);
    return o[0];
}

/// `indexScores` for a wide block on 8x8 matrix ops; `q` in the keys' dtype. `keys` are the list's rows from `key0`
/// on, a multiple of the kernel's key tile, so a slice scores the bits the whole list's columns would.
pub fn indexScoresTile(s: mlx.mlx_stream, q: A, keys: A, w: A, wscale: f32, p0: i32, ratio: usize, key0: usize) !A {
    const L = dim(q, 0);
    const nb = dim(keys, 0);
    const dt = mlx.mlx_array_dtype(keys);
    const kt = indexKeyTile(dt);
    std.debug.assert(key0 % kt == 0 and mlx.mlx_array_dtype(q) == dt);
    const meta = try ints(&.{ p0, @intCast(ratio), @intCast(key0) });
    defer _ = mlx.mlx_array_free(meta);
    const wsc = try mlx.newData(&[_]f32{wscale}, &.{1}, .float32);
    defer _ = mlx.mlx_array_free(wsc);
    var o: [1]A = undefined;
    try launch(s, &idx_tile_def, &.{ q, keys, w, meta, wsc }, &.{.{ .shape = &.{ ci(L), ci(nb) }, .dt = .float32 }}, .{ (nb + kt - 1) / kt * 128, L, 1 }, .{ 128, 1, 1 }, &.{
        .{ .name = "IH", .v = ci(dim(q, 1)) }, .{ .name = "IHD", .v = ci(dim(q, 2)) }, .{ .name = "KT", .v = ci(kt) },
    }, &.{.{ .name = "T", .dt = dt }}, &o);
    return o[0];
}
