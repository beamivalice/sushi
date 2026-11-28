//! The compressed indexer's decode launches. `scores`: the query's and every pooled key's tail rotated as they are
//! staged (`hy4_rope.tail`'s arithmetic, keys at their groups' first slots), then `sparse.indexScores`' relu-weighted
//! sum, one key a group of `IH` lanes (a head each, its dot over the channels in order) and the heads summed in order
//! by the key's first lane: the bytes of the two rope launches and the score launch, with no rotated copy of the keys.
const std = @import("std");
const mlx = @import("../mlx.zig");
const Arr = mlx.mlx_array;
const kcfg = @import("hy4_kcfg.zig");
const sparse = @import("../sparse_select.zig");

/// Threads a score threadgroup runs: `TG / IH` keys.
const score_tg = 256;

// No contract pragma: the rope and the dots are written as `hy4_rope.tail` and `sparse.indexScores` write them, under
// the same default.
pub const score_source: [:0]const u8 =
    \\const int L = q_shape[0];
    \\const uint tid = thread_index_in_threadgroup, lane = thread_index_in_simdgroup;
    \\const int KPT = TG / IH, P = IHD + 1;
    \\const int W = NOPE + HALF;
    \\const int nb = keys_shape[0];
    \\const int n0 = int(threadgroup_position_in_grid.x) * KPT;
    \\const int kk = int(tid) / IH, h = int(tid) % IH, n = n0 + kk;
    \\const int max_seen = (meta[0] + L - 1) / meta[1];
    \\if (n0 >= max_seen) {
    \\  if (h == 0 && n < nb) {
    \\    for (int l = 0; l < L; ++l) scores[size_t(l) * nb + n] = -INFINITY;
    \\  }
    \\  return;
    \\}
    \\#ifdef SUSHI_METALLIB
    \\threadgroup float sm[5160];
    \\threadgroup float* qs = sm;
    \\threadgroup float* ks = sm + IH * P;
    \\#else
    \\threadgroup float qs[IH * P], ks[KPT * P];
    \\#endif
    \\for (int e = int(tid); e < KPT * W; e += TG) {
    \\  const int kki = e / W, ni = n0 + kki;
    \\  const uint j = uint(e % W);
    \\  threadgroup float* dst = ks + kki * P;
    \\  if (ni >= nb) continue;
    \\  const device float* row = keys + size_t(ni) * IHD;
    \\  if (j < uint(NOPE)) {
    \\    dst[j] = static_cast<float>(row[j]);
    \\    continue;
    \\  }
    \\  uint i = j - uint(NOPE);
    \\  float d = static_cast<float>(i) / static_cast<float>(HALF);
    \\  float inv_freq = metal::exp2(-d * as_type<float>(BASE));
    \\  float Lpos = as_type<float>(KSCALE) * static_cast<float>(0 + ni);
    \\  float theta = Lpos * inv_freq;
    \\  float costheta = metal::fast::cos(theta);
    \\  float sintheta = metal::fast::sin(theta);
    \\  float x1 = static_cast<float>(row[NOPE + 2 * i]);
    \\  float x2 = static_cast<float>(row[NOPE + 2 * i + 1]);
    \\  float rx1 = x1 * costheta - x2 * sintheta;
    \\  float rx2 = x1 * sintheta + x2 * costheta;
    \\  dst[NOPE + 2 * i] = rx1;
    \\  dst[NOPE + 2 * i + 1] = rx2;
    \\}
    \\const threadgroup float* qh = qs + h * P;
    \\const threadgroup float* kr = ks + kk * P;
    \\const uint first = lane - uint(h);
    \\for (int l = 0; l < L; ++l) {
    \\  if (l > 0) threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  for (int e = int(tid); e < IH * W; e += TG) {
    \\    const int hh = e / W;
    \\    const uint j = uint(e % W);
    \\    const device float* row = q + (size_t(l) * IH + hh) * IHD;
    \\    threadgroup float* dst = qs + hh * P;
    \\    if (j < uint(NOPE)) {
    \\      dst[j] = row[j];
    \\      continue;
    \\    }
    \\    uint i = j - uint(NOPE);
    \\    float d = static_cast<float>(i) / static_cast<float>(HALF);
    \\    float inv_freq = metal::exp2(-d * as_type<float>(BASE));
    \\    float Lpos = as_type<float>(QSCALE) * static_cast<float>(meta[0] + l);
    \\    float theta = Lpos * inv_freq;
    \\    float costheta = metal::fast::cos(theta);
    \\    float sintheta = metal::fast::sin(theta);
    \\    float x1 = static_cast<float>(row[NOPE + 2 * i]);
    \\    float x2 = static_cast<float>(row[NOPE + 2 * i + 1]);
    \\    float rx1 = x1 * costheta - x2 * sintheta;
    \\    float rx2 = x1 * sintheta + x2 * costheta;
    \\    dst[NOPE + 2 * i] = rx1;
    \\    dst[NOPE + 2 * i + 1] = rx2;
    \\  }
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  float dot = 0.0f;
    \\  for (int j = 0; j < IHD; ++j) {
    \\    const float kv = kr[j];
    \\    dot += qh[j] * kv;
    \\  }
    \\  float s = 0.0f;
    \\  for (int hh = 0; hh < IH; ++hh) {
    \\    const float dh = metal::simd_shuffle(dot, ushort(first + uint(hh)));
    \\    s += (float(w[l * IH + hh]) * wsc[0]) * metal::max(dh, 0.0f);
    \\  }
    \\  const int seen = (meta[0] + l) / meta[1];
    \\  if (h == 0 && n < nb) scores[size_t(l) * nb + n] = n < seen ? s : -INFINITY;
    \\}
;

var score_kernel: ?mlx.mlx_fast_metal_kernel = null;

fn scoreKernel() !mlx.mlx_fast_metal_kernel {
    if (score_kernel) |k| return k;
    const ins = [_][*:0]const u8{ "q", "keys", "w", "meta", "wsc" };
    const outs = [_][*:0]const u8{"scores"};
    const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new("sushi_hy4_index_scores", iv, ov, score_source, "", true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    score_kernel = k;
    return k;
}

/// The most rows a score launch takes: a decode or copy-draft block. Each row's threadgroups rotate their keys again,
/// which a prefill span's rows would repeat where the chain rotates once a span.
pub const rows_max = 16;

/// Whether `scores` takes the query rows `q` [T, IH·IHD] f32 (unrotated) against the pooled keys `keys` [G, IHD] f32
/// (unrotated: the pool's; BF16 keys rotate through MLX's rope, which rounds them), `heads` lanes a key and the rotated
/// tail within a head.
pub fn scoresApply(s: mlx.mlx_stream, q: Arr, keys: Arr, heads: c_int, rope_dims: c_int) bool {
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(keys);
    if (qs.len != 2 or ks.len != 2 or qs[0] < 1 or qs[0] > rows_max or heads <= 0 or heads > 32 or @rem(32, heads) != 0) return false;
    const d = ks[1];
    const sm_need: i64 = @as(i64, heads + @divExact(score_tg, heads)) * (@as(i64, d) + 1);
    return mlx.streamIsGpu(s) and qs[1] == heads * d and ks[0] > 0 and @rem(rope_dims, 2) == 0 and rope_dims > 0 and
        rope_dims <= d and sm_need <= 5160 and mlx.mlx_array_dtype(q) == .float32 and mlx.mlx_array_dtype(keys) == .float32;
}

const Key = struct { t: c_int, nb: c_int, ih: c_int, ihd: c_int, nope: c_int, half: c_int, base: c_int, kscale: c_int };
var cfgs: kcfg.Cache(Key) = .{};

fn build(k: Key) !kcfg.Config {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const kpt = @divExact(score_tg, k.ih);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ k.t, k.nb }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divTrunc(k.nb + kpt - 1, kpt) * score_tg, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, score_tg, 1, 1));
    inline for (.{ "TG", "IH", "IHD", "NOPE", "HALF", "BASE", "QSCALE", "KSCALE" }, .{ score_tg, k.ih, k.ihd, k.nope, k.half, k.base, kcfg.bits(1.0), k.kscale }) |n, v|
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, n, v));
    return cfg;
}

/// The indexer's scores [T, G] f32 of the query rows `q` [T, heads·d] at positions `p0..` against the pooled keys
/// `keys` [G, d], group g's key at position `g · ratio` (`scoresApply`): head h's query and each key with their
/// trailing `rope_dims` rotated (θ `theta`), then `Σ_h w_h · wscale · relu(q_h · k_g)` with `w` [T, heads] f32, -inf at
/// and past each row's own group (`(p0 + t) / ratio`). The bytes of `fwd.recallAt`'s rope, key rotation and
/// `sparse.indexScores`. The caller frees the result.
pub fn scores(s: mlx.mlx_stream, q: Arr, keys: Arr, w: Arr, heads: c_int, rope_dims: c_int, theta: f32, wscale: f32, p0: c_int, ratio: c_int) !Arr {
    const ks = mlx.getShape(keys);
    const d = ks[1];
    if (heads <= 0 or heads > 32 or @rem(32, heads) != 0) return error.Hy4IndexScratch;
    const sm_need: i64 = @as(i64, heads + @divExact(score_tg, heads)) * (@as(i64, d) + 1);
    if (sm_need > 5160) return error.Hy4IndexScratch;
    const half = @divExact(rope_dims, 2);
    const cfg = try cfgs.get(.{ .t = mlx.getShape(q)[0], .nb = ks[0], .ih = heads, .ihd = d, .nope = d - rope_dims, .half = half, .base = kcfg.bits(std.math.log2(theta)), .kscale = kcfg.bits(@floatFromInt(ratio)) }, build);
    const mv = [_]i32{ p0, ratio };
    const meta = try mlx.newData(&mv, &.{2}, .int32);
    defer _ = mlx.mlx_array_free(meta);
    const sv = [_]f32{wscale};
    const wsc = try mlx.newData(&sv, &.{1}, .float32);
    defer _ = mlx.mlx_array_free(wsc);
    const ins = [_]Arr{ q, keys, w, meta, wsc };
    const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, try scoreKernel(), iv, cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, ov, 0));
    return out;
}

/// `pick`: the top-k of each row's scores (`sparse.TOPK_BODY`, one threadgroup a row: the picks in ascending order,
/// ties to the lower index), each pick written where `hy4_chain.recallGroups` and `recalledRows` write it (its group,
/// its key rows and keep flags, in their f32 order), and the row's own group last.
pub const pick_source: [:0]const u8 = "#pragma clang fp contract(off)\n" ++ sparse.TOPK_PRE ++
    \\const int l = threadgroup_position_in_grid.y, n = x_shape[1];
    \\const auto xr = x + size_t(l) * n;
    \\const int KW = kk[0] + 1, ratio = meta[1];
    \\const float own_group = metal::floor(static_cast<float>(meta[0] + l) / static_cast<float>(meta[1]));
    \\auto emit = [&](uint j, float g) {
    \\  groups[size_t(l) * KW + j] = g;
    \\  const uint width = uint(KW * ratio);
    \\  for (int r = 0; r < ratio; ++r) {
    \\    const uint at = j * uint(ratio) + uint(r);
    \\    bool ok = (g < own_group) || (static_cast<float>(KW - 1) <= static_cast<float>(j));
    \\    float base = g * static_cast<float>(ratio);
    \\    float p = base + static_cast<float>(r);
    \\    bool reads = (p <= static_cast<float>(meta[0] + l)) && ok;
    \\    float last = static_cast<float>(meta[2] - 1);
    \\    float clamped = metal::isnan(p) ? p : (p < last ? p : last);
    \\    rows[size_t(l) * width + at] = static_cast<uint>(clamped);
    \\    keep[size_t(l) * width + at] = reads;
    \\  }
    \\};
    \\#define ELEM(i) dsv41_key(xr[i])
    \\#define EMIT(p, i) emit(uint(p), static_cast<float>(i))
    \\
++ sparse.TOPK_BODY ++
    \\if (tid == 0) emit(uint(KW - 1), own_group);
;

var pick_kernel: ?mlx.mlx_fast_metal_kernel = null;

fn pickKernel() !mlx.mlx_fast_metal_kernel {
    if (pick_kernel) |k| return k;
    const ins = [_][*:0]const u8{ "x", "kk", "meta" };
    const outs = [_][*:0]const u8{ "groups", "rows", "keep" };
    const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new("sushi_hy4_index_pick", iv, ov, pick_source, sparse.TOPK_HEADER, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    pick_kernel = k;
    return k;
}

/// Threads a pick threadgroup runs (the picks do not depend on it); null takes `sparse.topk_tg`. A test seam.
pub var pick_tg: ?usize = null;

/// Whether `pick` takes `k` picks of each row of `scores` [T, G]: the rows `sparse.topkIndices` leaves in one
/// threadgroup a row.
pub fn pickApplies(s: mlx.mlx_stream, scores_: Arr, k: c_int) bool {
    const sh = mlx.getShape(scores_);
    return mlx.streamIsGpu(s) and sh.len == 2 and k >= 1 and k <= sh[1] and mlx.mlx_array_dtype(scores_) == .float32 and
        sparse.topkSplits(@intCast(sh[0]), @intCast(sh[1]), @intCast(k)) == 1;
}

const PickKey = struct { t: c_int, k: c_int, ratio: c_int, tg: c_int };
var pick_cfgs: kcfg.Cache(PickKey) = .{};

fn buildPick(k: PickKey) !kcfg.Config {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const width = (k.k + 1) * k.ratio;
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ k.t, k.k + 1 }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ k.t, width }, 2, .uint32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ k.t, width }, 2, .bool_));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, k.tg, k.t, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, k.tg, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "TG", k.tg));
    return cfg;
}

pub const Picked = struct { groups: Arr, rows: Arr, keep: Arr };

/// The `k` best groups of each row of `scores` [T, G] and its own group, for rows at positions `p0..` over `n` keys
/// (`pickApplies`): `groups` [T, k + 1] f32 (`fwd.recallAt`'s), and the key rows [T, (k + 1)·ratio] uint32 and their
/// keep mask (`fwd.recalledRows`'). The bytes of `sparse.topkIndices`, `hy4_chain.recallGroups` and
/// `hy4_chain.recalledRows`. The caller frees all three.
pub fn pick(s: mlx.mlx_stream, scores_: Arr, k: c_int, p0: c_int, n: c_int, ratio: c_int) !Picked {
    const tg: c_int = @intCast(pick_tg orelse sparse.topk_tg);
    const cfg = try pick_cfgs.get(.{ .t = mlx.getShape(scores_)[0], .k = k, .ratio = ratio, .tg = tg }, buildPick);
    // k rides in a buffer, as `sparse.topkIndices` takes it.
    const kv = [_]i32{k};
    const kk = try mlx.newData(&kv, &.{1}, .int32);
    defer _ = mlx.mlx_array_free(kk);
    const mv = [_]i32{ p0, ratio, n };
    const meta = try mlx.newData(&mv, &.{3}, .int32);
    defer _ = mlx.mlx_array_free(meta);
    const ins = [_]Arr{ scores_, kk, meta };
    const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, try pickKernel(), iv, cfg, s));
    var out: [3]Arr = .{ mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new() };
    errdefer for (out) |x| {
        _ = mlx.mlx_array_free(x);
    };
    for (&out, 0..) |*x, i| try mlx.check(mlx.mlx_vector_array_get(x, ov, i));
    return .{ .groups = out[0], .rows = out[1], .keep = out[2] };
}

// 8 rows by 32 keys a threadgroup (a simdgroup a row, a lane a key), the queries and keys staged 16 channels at a
// time: each head's dot sums its channels in order, as `sparse.indexScores` does, so the bytes are its. A tile past
// every row's seen groups writes -inf and reads nothing.
pub const wide_score_source: [:0]const u8 =
    \\constexpr int QL = 8, KN = 32, BK = 16, TG = 256, BK4 = BK / 4, IH4 = IH / 4;
    \\const int l0 = int(threadgroup_position_in_grid.y) * QL;
    \\const int n0 = int(threadgroup_position_in_grid.x) * KN;
    \\const uint tid = thread_index_in_threadgroup;
    \\const uint sg = simdgroup_index_in_threadgroup;
    \\const uint lane = thread_index_in_simdgroup;
    \\const int L = q_shape[0], nb = keys_shape[0];
    \\const int l = l0 + int(sg);
    \\const int n = n0 + int(lane);
    \\const int max_l = metal::min(l0 + QL - 1, L - 1);
    \\const int max_seen = (meta[0] + max_l + 1) / meta[1];
    \\if (n0 >= max_seen) {
    \\  if (l < L && n < nb) scores[size_t(l) * nb + n] = -INFINITY;
    \\  return;
    \\}
    \\threadgroup float qs[QL * BK * IH];
    \\threadgroup float ks[BK * KN];
    \\float dots[IH];
    \\#pragma clang loop unroll(full)
    \\for (int h = 0; h < IH; ++h) dots[h] = 0.0f;
    \\for (int k0 = 0; k0 < IHD; k0 += BK) {
    \\  for (int e = int(tid); e < QL * IH * BK4; e += TG) {
    \\    const int d4 = e % BK4;
    \\    const int tmp = e / BK4;
    \\    const int h = tmp % IH;
    \\    const int ql = tmp / IH;
    \\    const int row = l0 + ql;
    \\    float4 v = float4(0.0f);
    \\    if (row < L) {
    \\      const device float* qp = q + (size_t(row) * IH + h) * IHD + k0 + d4 * 4;
    \\      v = float4(qp[0], qp[1], qp[2], qp[3]);
    \\    }
    \\    qs[(ql * BK + d4 * 4 + 0) * IH + h] = v.x;
    \\    qs[(ql * BK + d4 * 4 + 1) * IH + h] = v.y;
    \\    qs[(ql * BK + d4 * 4 + 2) * IH + h] = v.z;
    \\    qs[(ql * BK + d4 * 4 + 3) * IH + h] = v.w;
    \\  }
    \\  if (int(tid) < KN * BK4) {
    \\    const int kn = int(tid) / BK4;
    \\    const int d4 = int(tid) % BK4;
    \\    const int key_idx = n0 + kn;
    \\    float4 v = float4(0.0f);
    \\    if (key_idx < nb) {
    \\      const device float* kp = keys + size_t(key_idx) * IHD + k0 + d4 * 4;
    \\      v = float4(kp[0], kp[1], kp[2], kp[3]);
    \\    }
    \\    ks[(d4 * 4 + 0) * KN + kn] = v.x;
    \\    ks[(d4 * 4 + 1) * KN + kn] = v.y;
    \\    ks[(d4 * 4 + 2) * KN + kn] = v.z;
    \\    ks[(d4 * 4 + 3) * KN + kn] = v.w;
    \\  }
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  const threadgroup float4* qrow4 = (const threadgroup float4*)(qs + int(sg) * BK * IH);
    \\  #pragma clang loop unroll(full)
    \\  for (int j = 0; j < BK; ++j) {
    \\    const float kv = ks[j * KN + lane];
    \\    #pragma clang loop unroll(full)
    \\    for (int h4 = 0; h4 < IH4; ++h4) {
    \\      const float4 qv = qrow4[j * IH4 + h4];
    \\      dots[h4 * 4 + 0] += qv.x * kv;
    \\      dots[h4 * 4 + 1] += qv.y * kv;
    \\      dots[h4 * 4 + 2] += qv.z * kv;
    \\      dots[h4 * 4 + 3] += qv.w * kv;
    \\    }
    \\  }
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\}
    \\if (l < L && n < nb) {
    \\  const int seen = (meta[0] + l + 1) / meta[1];
    \\  if (n < seen) {
    \\    float s = 0.0f;
    \\    for (int h = 0; h < IH; ++h) s += (float(w[size_t(l) * IH + h]) * wsc[0]) * metal::max(dots[h], 0.0f);
    \\    scores[size_t(l) * nb + n] = s;
    \\  } else {
    \\    scores[size_t(l) * nb + n] = -INFINITY;
    \\  }
    \\}
;

var wide_score_kernel: ?mlx.mlx_fast_metal_kernel = null;

fn wideScoreKernel() !mlx.mlx_fast_metal_kernel {
    if (wide_score_kernel) |k| return k;
    const ins = [_][*:0]const u8{ "q", "keys", "w", "meta", "wsc" };
    const outs = [_][*:0]const u8{"scores"};
    const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new("sushi_hy4_index_wide", iv, ov, wide_score_source, "", true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    wide_score_kernel = k;
    return k;
}

const WideKey = struct { l: c_int, nb: c_int, ih: c_int, ihd: c_int };
var wide_cfgs: kcfg.Cache(WideKey) = .{};

fn buildWide(k: WideKey) !kcfg.Config {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ k.l, k.nb }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divTrunc(k.nb + 31, 32) * 256, @divTrunc(k.l + 7, 8), 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "IH", k.ih));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "IHD", k.ihd));
    return cfg;
}

/// Whether `wideScores` serves `q` [T, heads, dim] against `keys` [groups, dim] with head weights `w` [T, heads]:
/// f32, a GPU stream, heads in fours up to 32, dim in sixteens.
pub fn wideScoresApply(s: mlx.mlx_stream, q: Arr, keys: Arr, w: Arr) bool {
    if (!mlx.streamIsGpu(s)) return false;
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(keys);
    const ws = mlx.getShape(w);
    if (qs.len != 3 or ks.len != 2 or ws.len != 2) return false;
    if (qs[0] <= 0 or ks[0] <= 0 or qs[1] <= 0 or qs[1] > 32 or @rem(qs[1], 4) != 0 or qs[2] <= 0 or @rem(qs[2], 16) != 0) return false;
    if (ks[1] != qs[2] or ws[0] != qs[0] or ws[1] != qs[1]) return false;
    return mlx.mlx_array_dtype(q) == .float32 and mlx.mlx_array_dtype(keys) == .float32 and mlx.mlx_array_dtype(w) == .float32;
}

/// `sparse.indexScores` of a prefill span, its bytes, as one tiled launch: each row's score of every group it sees
/// (from `p0`, `ratio` tokens a group), -inf past them.
pub fn wideScores(s: mlx.mlx_stream, q: Arr, keys: Arr, w: Arr, wscale: f32, p0: i32, ratio: usize) !Arr {
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(keys);
    const cfg = try wide_cfgs.get(.{ .l = qs[0], .nb = ks[0], .ih = qs[1], .ihd = qs[2] }, buildWide);
    const mv = [_]i32{ p0, @intCast(ratio) };
    const meta = try mlx.newData(&mv, &.{2}, .int32);
    defer _ = mlx.mlx_array_free(meta);
    const sv = [_]f32{wscale};
    const wsc = try mlx.newData(&sv, &.{1}, .float32);
    defer _ = mlx.mlx_array_free(wsc);
    const ins = [_]Arr{ q, keys, w, meta, wsc };
    const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, try wideScoreKernel(), iv, cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, ov, 0));
    return out;
}

/// Strict f32 `simdgroup_float8x8` prefill indexer scores: 8 query rows × 32 keys per threadgroup (1 simdgroup/row),
/// keeping every dot product and head sum in f32 while accelerating `Q @ K^T` on the FP32 matrix unit.
pub const wide_score_sg_source: [:0]const u8 =
    \\constexpr int QL = 8, KN = 32, BK = 32, HH = IH / 8;
    \\const int l0 = int(threadgroup_position_in_grid.y) * QL;
    \\const int n0 = int(threadgroup_position_in_grid.x) * KN;
    \\const uint tid = thread_index_in_threadgroup;
    \\const uint sg = simdgroup_index_in_threadgroup;
    \\const uint lane = thread_index_in_simdgroup;
    \\const int L = q_shape[0], nb = keys_shape[0];
    \\const int l = l0 + int(sg);
    \\const int n = n0 + int(lane);
    \\const int max_l = metal::min(l0 + QL - 1, L - 1);
    \\const int max_seen = (meta[0] + max_l + 1) / meta[1];
    \\if (n0 >= max_seen) {
    \\  if (l < L && n < nb) scores[size_t(l) * nb + n] = -INFINITY;
    \\  return;
    \\}
    \\threadgroup float ks[KN * BK];
    \\threadgroup float ds[QL * IH * KN];
    \\simdgroup_float8x8 acc[HH][4];
    \\#pragma clang loop unroll(full)
    \\for (int hh = 0; hh < HH; ++hh) {
    \\  #pragma clang loop unroll(full)
    \\  for (int nn = 0; nn < 4; ++nn) {
    \\    acc[hh][nn] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    \\  }
    \\}
    \\const int kn = int(tid) / 8;
    \\const int c4 = int(tid) % 8;
    \\const int key_idx = n0 + kn;
    \\for (int k0 = 0; k0 < IHD; k0 += BK) {
    \\  float4 v = float4(0.0f);
    \\  if (key_idx < nb) {
    \\    const device float* kp = keys + size_t(key_idx) * IHD + k0 + c4 * 4;
    \\    v = float4(kp[0], kp[1], kp[2], kp[3]);
    \\  }
    \\  threadgroup float* kd = ks + kn * BK + c4 * 4;
    \\  kd[0] = v.x; kd[1] = v.y; kd[2] = v.z; kd[3] = v.w;
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  if (l < L) {
    \\    const device float* qrow = q + size_t(l) * size_t(IH) * size_t(IHD) + k0;
    \\    #pragma clang loop unroll(full)
    \\    for (int kk = 0; kk < BK; kk += 8) {
    \\      simdgroup_float8x8 qa[HH];
    \\      #pragma clang loop unroll(full)
    \\      for (int hh = 0; hh < HH; ++hh) {
    \\        simdgroup_load(qa[hh], qrow + size_t(hh * 8) * IHD + kk, IHD);
    \\      }
    \\      #pragma clang loop unroll(full)
    \\      for (int nn = 0; nn < 4; ++nn) {
    \\        simdgroup_float8x8 kb;
    \\        simdgroup_load(kb, ks + (nn * 8) * BK + kk, BK, ulong2(0, 0), true);
    \\        #pragma clang loop unroll(full)
    \\        for (int hh = 0; hh < HH; ++hh) {
    \\          simdgroup_multiply_accumulate(acc[hh][nn], qa[hh], kb, acc[hh][nn]);
    \\        }
    \\      }
    \\    }
    \\  }
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\}
    \\if (l < L) {
    \\  threadgroup float* dsg = ds + int(sg) * (IH * KN);
    \\  #pragma clang loop unroll(full)
    \\  for (int hh = 0; hh < HH; ++hh) {
    \\    #pragma clang loop unroll(full)
    \\    for (int nn = 0; nn < 4; ++nn) {
    \\      simdgroup_store(acc[hh][nn], dsg + (hh * 8) * KN + nn * 8, KN);
    \\    }
    \\  }
    \\  simdgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  if (n < nb) {
    \\    const int seen = (meta[0] + l + 1) / meta[1];
    \\    if (n < seen) {
    \\      float s = 0.0f;
    \\      const float sc = wsc[0];
    \\      const device float* wl = w + size_t(l) * IH;
    \\      #pragma clang loop unroll(full)
    \\      for (int h = 0; h < IH; ++h) {
    \\        s += (wl[h] * sc) * metal::max(dsg[h * KN + int(lane)], 0.0f);
    \\      }
    \\      scores[size_t(l) * nb + n] = s;
    \\    } else {
    \\      scores[size_t(l) * nb + n] = -INFINITY;
    \\    }
    \\  }
    \\}
;

var wide_score_sg_kernel: ?mlx.mlx_fast_metal_kernel = null;

fn wideScoreSgKernel() !mlx.mlx_fast_metal_kernel {
    if (wide_score_sg_kernel) |k| return k;
    const ins = [_][*:0]const u8{ "q", "keys", "w", "meta", "wsc" };
    const outs = [_][*:0]const u8{"scores"};
    const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new("sushi_hy4_index_wide_sg", iv, ov, wide_score_sg_source, "", true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    wide_score_sg_kernel = k;
    return k;
}

pub fn wideScoresSgApply(s: mlx.mlx_stream, q: Arr, keys: Arr, w: Arr) bool {
    if (!wideScoresApply(s, q, keys, w)) return false;
    const qs = mlx.getShape(q);
    return (qs[1] == 8 or qs[1] == 16) and @rem(qs[2], 32) == 0;
}

pub fn wideScoresSg(s: mlx.mlx_stream, q: Arr, keys: Arr, w: Arr, wscale: f32, p0: i32, ratio: usize) !Arr {
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(keys);
    const cfg = try wide_cfgs.get(.{ .l = qs[0], .nb = ks[0], .ih = qs[1], .ihd = qs[2] }, buildWide);
    const mv = [_]i32{ p0, @intCast(ratio) };
    const meta = try mlx.newData(&mv, &.{2}, .int32);
    defer _ = mlx.mlx_array_free(meta);
    const sv = [_]f32{wscale};
    const wsc = try mlx.newData(&sv, &.{1}, .float32);
    defer _ = mlx.mlx_array_free(wsc);
    const ins = [_]Arr{ q, keys, w, meta, wsc };
    const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, try wideScoreSgKernel(), iv, cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, ov, 0));
    return out;
}

// ── tests ────────────────────────────────────────────────────────────────────────────────────────

const testing = std.testing;
const Ops = @import("../mlx_ops.zig").Ops;
const hy4_rope = @import("hy4_rope.zig");

fn randomArr(o: *Ops, shape: []const c_int, seed: u64, sc: f64) !Arr {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const v = try testing.allocator.alloc(f32, n);
    defer testing.allocator.free(v);
    var r = std.Random.DefaultPrng.init(seed);
    for (v) |*d| d.* = @floatCast(r.random().floatNorm(f64) * sc);
    return o.own(mlx.mlx_array_new_data(v.ptr, shape.ptr, @intCast(shape.len), .float32));
}

fn bytesOf(o: *Ops, a: Arr) ![]const u8 {
    const c = try o.contiguous(a);
    try mlx.check(mlx.mlx_array_eval(c));
    return std.mem.sliceAsBytes((mlx.mlx_array_data_float32(c) orelse return error.HostRead)[0..mlx.mlx_array_size(c)]);
}

test "hy4 index: a decode row's indexer scores with the query's and the pooled keys' rotations in the launch give the bytes of the two rope launches and the score kernel, at the release's 16 heads of 128 and the tiny's 4 of 32, one row and three, 1,024 to 3,750 groups" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const Case = struct { h: c_int, d: c_int, rope: c_int, g: c_int, t: c_int };
    const cases = [_]Case{
        .{ .h = 16, .d = 128, .rope = 64, .g = 1024, .t = 1 },
        .{ .h = 16, .d = 128, .rope = 64, .g = 3750, .t = 1 },
        .{ .h = 16, .d = 128, .rope = 64, .g = 1001, .t = 3 },
        .{ .h = 4, .d = 32, .rope = 16, .g = 150, .t = 1 },
        .{ .h = 4, .d = 32, .rope = 16, .g = 37, .t = 3 },
    };
    const ratio: c_int = 4;
    for (cases, 0..) |c, ci| {
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const seed: u64 = 11 * ci + 3;
        // The query projection's rows, the pooled keys (rms-normed: unit scale) and the head weights.
        const q = try randomArr(&o, &.{ c.t, c.h * c.d }, seed, 0.6);
        const keys = try randomArr(&o, &.{ c.g, c.d }, seed + 1, 1);
        const w = try randomArr(&o, &.{ c.t, c.h }, seed + 2, 0.5);
        const scale = std.math.pow(f32, @floatFromInt(c.h * c.d), -0.5);
        // At and past each row's group the scores are -inf: the last row sits in the last group.
        const p0: c_int = ratio * c.g - c.t + 1;
        // The chain recallAt runs.
        const qv = try o.transpose(try o.reshape(q, &.{ c.t, c.h, c.d }), &.{ 1, 0, 2 });
        const qr = try o.own(try hy4_rope.tail(es, qv, c.rope, 1e7, p0, 1, .float32));
        const qt = try o.contiguous(try o.transpose(qr, &.{ 1, 0, 2 }));
        const kr = try o.own(try hy4_rope.tail(es, try o.reshape(keys, &.{ 1, c.g, c.d }), c.rope, 1e7, 0, ratio, .float32));
        const want = try o.own(try sparse.indexScores(es, qt, try o.reshape(kr, &.{ c.g, c.d }), w, scale, p0 - 1, @intCast(ratio)));
        const wb = try bytesOf(&o, want);
        try testing.expect(scoresApply(es, q, keys, c.h, c.rope));
        const before = mlx.engine.stats().dispatches;
        const got = try o.own(try scores(es, q, keys, w, c.h, c.rope, 1e7, scale, p0, ratio));
        try mlx.check(mlx.mlx_array_eval(got));
        try testing.expectEqual(@as(u64, 1), mlx.engine.stats().dispatches - before);
        try testing.expectEqualSlices(u8, wb, try bytesOf(&o, got));
    }
}

/// Host scores [T, G]: ReLU of normal draws (many exact zeros, so ties), -inf at and past row t's own group.
fn pickScores(o: *Ops, t: c_int, g: c_int, p0: c_int, ratio: c_int, seed: u64) !Arr {
    const n: usize = @intCast(t * g);
    const v = try testing.allocator.alloc(f32, n);
    defer testing.allocator.free(v);
    var r = std.Random.DefaultPrng.init(seed);
    for (v, 0..) |*d, i| {
        const row: c_int = @intCast(i / @as(usize, @intCast(g)));
        const col: c_int = @intCast(i % @as(usize, @intCast(g)));
        d.* = if (col < @divTrunc(p0 + row, ratio)) @max(0, @as(f32, @floatCast(r.random().floatNorm(f64)))) else -std.math.inf(f32);
    }
    return o.own(mlx.mlx_array_new_data(v.ptr, &[_]c_int{ t, g }, 2, .float32));
}

/// An integer or bool array's values as f32 bytes: exact for the key rows (under 2^24) and the mask.
fn valueBytes(o: *Ops, a: Arr) ![]const u8 {
    return bytesOf(o, try o.cast(a, .float32));
}

test "hy4 index: a decode row's group picks written as their key rows and keep mask in the top-k launch give the bytes of the top-k, recallGroups and recalledRows, at the release's 511 of 1,024 to 3,750 groups and the tiny's 3, with ties, one row and three, 1,024 lanes and 256" {
    try mlx.requireGpu();
    const hy4_chain = @import("hy4_chain.zig");
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const tg_was = pick_tg;
    defer pick_tg = tg_was;
    const Case = struct { g: c_int, k: c_int, t: c_int };
    const cases = [_]Case{ .{ .g = 1024, .k = 511, .t = 1 }, .{ .g = 3750, .k = 511, .t = 1 }, .{ .g = 1001, .k = 511, .t = 3 }, .{ .g = 37, .k = 3, .t = 3 } };
    const ratio: c_int = 4;
    for ([_]usize{ 1024, 256 }) |tg| for (cases, 0..) |c, ci| {
        pick_tg = tg;
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const p0: c_int = ratio * c.g - c.t + 1;
        const n = p0 + c.t;
        const scores_ = try pickScores(&o, c.t, c.g, p0, ratio, 5 * ci + tg);
        const order = try o.own(try sparse.topkIndicesSplit(es, scores_, @intCast(c.k), 1));
        const groups = try o.own(try hy4_chain.recallGroups(es, order, p0, ratio));
        const want = try hy4_chain.recalledRows(es, groups, p0, n, ratio);
        const wr = try valueBytes(&o, try o.own(want.rows));
        const wk = try valueBytes(&o, try o.own(want.keep));
        const wg = try bytesOf(&o, groups);
        try testing.expect(pickApplies(es, scores_, c.k));
        const before = mlx.engine.stats().dispatches;
        const got = try pick(es, scores_, c.k, p0, n, ratio);
        const gg = try o.own(got.groups);
        const gr = try o.own(got.rows);
        const gk = try o.own(got.keep);
        try mlx.check(mlx.mlx_array_eval(gk));
        try testing.expectEqual(@as(u64, 1), mlx.engine.stats().dispatches - before);
        try testing.expectEqualSlices(u8, wg, try bytesOf(&o, gg));
        try testing.expectEqualSlices(u8, wr, try valueBytes(&o, gr));
        try testing.expectEqualSlices(u8, wk, try valueBytes(&o, gk));
    };
}

test "hy4 index: wide prefill scores match sparse.indexScores byte for byte across release and tiny shapes" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const Case = struct { h: c_int, d: c_int, g: c_int, t: c_int };
    const cases = [_]Case{
        .{ .h = 16, .d = 128, .g = 1024, .t = 256 },
        .{ .h = 16, .d = 128, .g = 1001, .t = 37 },
        .{ .h = 4, .d = 32, .g = 150, .t = 19 },
    };
    const ratio: c_int = 4;
    for (cases, 0..) |c, ci| {
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const seed: u64 = 101 * ci + 7;
        const q = try randomArr(&o, &.{ c.t, c.h, c.d }, seed, 0.6);
        const keys = try randomArr(&o, &.{ c.g, c.d }, seed + 1, 1);
        const w = try randomArr(&o, &.{ c.t, c.h }, seed + 2, 0.5);
        const scale = std.math.pow(f32, @floatFromInt(c.h * c.d), -0.5);
        const p0: c_int = ratio * c.g - c.t + 1;
        const want = try o.own(try sparse.indexScores(es, q, keys, w, scale, p0 - 1, @intCast(ratio)));
        const wb = try bytesOf(&o, want);
        try testing.expect(wideScoresApply(es, q, keys, w));
        const before = mlx.engine.stats().dispatches;
        const got = try o.own(try wideScores(es, q, keys, w, scale, p0 - 1, @intCast(ratio)));
        try mlx.check(mlx.mlx_array_eval(got));
        try testing.expectEqual(@as(u64, 1), mlx.engine.stats().dispatches - before);
        try testing.expectEqualSlices(u8, wb, try bytesOf(&o, got));
    }
}

test "hy4 index: simdgroup_float8x8 wide prefill scores match wideScores to f32 rounding across release and tail shapes" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const Case = struct { h: c_int, d: c_int, g: c_int, t: c_int };
    const cases = [_]Case{
        .{ .h = 16, .d = 128, .g = 1024, .t = 256 },
        .{ .h = 16, .d = 128, .g = 1001, .t = 37 },
        .{ .h = 8, .d = 64, .g = 150, .t = 19 },
    };
    const ratio: c_int = 4;
    for (cases, 0..) |c, ci| {
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const seed: u64 = 201 * ci + 13;
        const q = try randomArr(&o, &.{ c.t, c.h, c.d }, seed, 0.6);
        const keys = try randomArr(&o, &.{ c.g, c.d }, seed + 1, 1);
        const w = try randomArr(&o, &.{ c.t, c.h }, seed + 2, 0.5);
        const scale = std.math.pow(f32, @floatFromInt(c.h * c.d), -0.5);
        const p0: c_int = ratio * c.g - c.t + 1;
        const want = try o.own(try wideScores(es, q, keys, w, scale, p0 - 1, @intCast(ratio)));
        try mlx.check(mlx.mlx_array_eval(want));
        try testing.expect(wideScoresSgApply(es, q, keys, w));
        const before = mlx.engine.stats().dispatches;
        const got = try o.own(try wideScoresSg(es, q, keys, w, scale, p0 - 1, @intCast(ratio)));
        try mlx.check(mlx.mlx_array_eval(got));
        try testing.expectEqual(@as(u64, 1), mlx.engine.stats().dispatches - before);
        const n = mlx.mlx_array_size(want);
        const wp = (mlx.mlx_array_data_float32(want) orelse return error.HostRead)[0..n];
        const gp = (mlx.mlx_array_data_float32(got) orelse return error.HostRead)[0..n];
        var max_err: f32 = 0;
        for (wp, gp) |a, b| {
            if (std.math.isInf(a)) {
                try testing.expect(std.math.isInf(b) and (a < 0) == (b < 0));
            } else {
                const diff = @abs(a - b);
                if (diff > max_err) max_err = diff;
            }
        }
        try testing.expect(max_err < 1e-4);
    }
}
