//! A sampled draw as one kernel: temperature, top-k by rank, the nucleus and the draw, a threadgroup a row.
//!
//! The rule (`hostDraw` is its reference):
//! - a value is `logit * (1 / temperature)`; values order high to low, equal values by lower id first;
//! - top-k keeps the first `top_k` in that order (0 or >= vocab: all);
//! - the nucleus keeps, of those, each one whose mass strictly above it in that order is under `top_p` of their mass
//!   (the top one always);
//! - the draw is the kept id with the largest `value + gumbel(key, id)`, the lower id on a tie.
//! A row's draw is a function of that row, the parameters and its key alone: the same alone and beside other rows.
//! A row with no finite value draws id 0.

const std = @import("std");
const mlx = @import("mlx.zig");

const Arr = mlx.mlx_array;

/// Rows of one launch: a verify block's.
pub const max_rows = 16;
/// A row's threadgroup. A GPU without matrix units grants this kernel 768, so it launches only where the engine serves.
const group_threads = 1024;

// `sushi_draw_mix` is lowbias32, ported from hash-prospector (public domain): NOTICE.
const HEADER =
    \\static inline uint sushi_draw_key(float x) {
    \\  const uint u = as_type<uint>(x);
    \\  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    \\}
    \\static inline float sushi_draw_unkey(uint k) {
    \\  return as_type<float>((k & 0x80000000u) ? (k ^ 0x80000000u) : ~k);
    \\}
    \\static inline float sushi_draw_val(float x) { return x == x ? x : -INFINITY; }
    \\static inline uint sushi_draw_mix(uint h) {
    \\  h ^= h >> 16; h *= 0x7FEB352Du; h ^= h >> 15; h *= 0x846CA68Bu; h ^= h >> 16;
    \\  return h;
    \\}
    \\static inline float sushi_draw_gumbel(uint id, uint k0, uint k1) {
    \\  const uint h = sushi_draw_mix(sushi_draw_mix(id * 0x9E3779B1u + k0) + k1);
    \\  const float u = (float(h >> 8) + 0.5f) * 5.9604644775390625e-8f;
    \\  return -metal::precise::log(-metal::precise::log(u));
    \\}
    \\// Whole nats below the row's top value, the last bin taking everything from 31 down.
    \\static inline uint sushi_draw_bin(float m, float v) { return uint(metal::min(m - v, 31.0f)); }
    \\#define SD_CAP 1024u
    \\#define SD_VAL(j) sushi_draw_val(float(x[j]) * inv_t)
    \\#define SD_INK ((!do_k) || k > kth || (k == kth && (base_k + ko++) < tk))
    \\#define SD_BARRIER threadgroup_barrier(metal::mem_flags::mem_threadgroup)
    \\#define SD_SHIFT(r) ((r) < 6u ? 27u - 5u * (r) : 0u)
    \\#define SD_BINS(r) ((r) < 6u ? 32u : 4u)
    \\#define SD_UP(r) ((r) < 6u ? 32u - 5u * (r) : 2u)
    \\#define SD_BASE(out, expr, with_ink) { uint c_ = 0u; uint ko = 0u; for (uint j = 0u; j < n; ++j) { const uint k = sushi_draw_key(SD_VAL(j)); const bool ink = with_ink && SD_INK; c_ += (expr) ? 1u : 0u; } \
    \\  const uint pre_ = metal::simd_prefix_exclusive_sum(c_); const uint tot_ = metal::simd_sum(c_); if (lane == 0u) pu[0][sg] = tot_; SD_BARRIER; \
    \\  uint b_ = pre_; for (uint g = 0u; g < sg; ++g) b_ += pu[0][g]; out = b_; SD_BARRIER; }
;

/// One thread owns `STRIPE` consecutive ids, in simdgroup then lane order, so "lower id first" is "earlier thread
/// first". A filtered row takes one of two routes to the same rule:
/// - the shortlist: one walk counts the ids by whole nats below the row's top value, the bins from the top that hold
///   the top-k's count (or, by the bins' bounds, surely the nucleus's mass) are collected in id order, at most
///   `SD_CAP` ids, and every thread ranks one of them against the rest;
/// - the descents, when the shortlist would not fit: a level (the top-k's last key, the nucleus's last key) is found
///   five key bits a round, every thread binning its ids of the live bucket.
/// A barrier follows every shared write and every shared read.
const SOURCE =
    \\const uint sg = simdgroup_index_in_threadgroup;
    \\const uint lane = thread_index_in_simdgroup;
    \\const uint row = threadgroup_position_in_grid.x;
    \\constexpr uint STRIPE = (uint(V) + 1023u) / 1024u;
    \\const uint i0 = (sg * 32u + lane) * STRIPE;
    \\const uint n = i0 < uint(V) ? metal::min(STRIPE, uint(V) - i0) : 0u;
    \\const device T* x = logits + size_t(row) * V + i0;
    \\const float inv_t = fparams[0];
    \\const float top_p = fparams[1];
    \\const uint top_k = uparams[0];
    \\const uint key0 = keys[row * 2u], key1 = keys[row * 2u + 1u];
    \\threadgroup uint pu[32][32];
    \\threadgroup float pf[32][32];
    \\threadgroup uint su[3];
    \\threadgroup float sf[2];
    \\threadgroup uint pi[32];
    \\threadgroup uint ck[SD_CAP];
    \\threadgroup float ce[SD_CAP];
    \\threadgroup uint ci[SD_CAP];
    \\const bool do_k = top_k > 0u && top_k < uint(V);
    \\const bool do_p = top_p < 1.0f;
    \\
    \\// The row's top value, which only a filter reads.
    \\float m = -INFINITY;
    \\if (do_k || do_p) {
    \\  float s = -INFINITY;
    \\  for (uint j = 0u; j < n; ++j) s = metal::max(s, SD_VAL(j));
    \\  s = metal::simd_max(s);
    \\  if (lane == 0u) pf[0][sg] = s;
    \\  SD_BARRIER;
    \\  for (uint g = 0u; g < 32u; ++g) m = metal::max(m, pf[0][g]);
    \\  SD_BARRIER;
    \\}
    \\
    \\float best = -INFINITY;
    \\uint bi = 0xFFFFFFFFu;
    \\uint route = 0u;
    \\if (m > -INFINITY) {
    \\  uint lc[32];
    \\  for (uint d = 0u; d < 32u; ++d) lc[d] = 0u;
    \\  for (uint j = 0u; j < n; ++j) lc[sushi_draw_bin(m, SD_VAL(j))] += 1u;
    \\  for (uint d = 0u; d < 32u; ++d) { const uint c = metal::simd_sum(lc[d]); if (lane == 0u) pu[d][sg] = c; }
    \\  SD_BARRIER;
    \\  if (sg == 0u) {
    \\    // Lane d holds bin d. Top-k cuts at the first bin down to which the count is the k. A nucleus cuts where the
    \\    // bins above surely hold its mass: an id of bin d weighs between e^-(d+1) and e^-d of the top one.
    \\    uint c = 0u;
    \\    for (uint g = 0u; g < 32u; ++g) c += pu[lane][g];
    \\    const uint cc = metal::simd_prefix_exclusive_sum(c) + c;
    \\    const float lo = lane < 31u ? float(c) * metal::precise::exp(-float(lane + 1u)) : 0.0f;
    \\    const float up = float(c) * metal::precise::exp(-float(lane));
    \\    const float head = metal::simd_prefix_exclusive_sum(lo) + lo;
    \\    const float all = metal::simd_sum(up);
    \\    const float tail = all - (metal::simd_prefix_exclusive_sum(up) + up);
    \\    const float p = metal::max(top_p, 1.17549435e-38f);
    \\    const bool enough = do_k ? cc >= top_k : head * (1.0f - p) >= p * (tail * 1.001f + all * 1e-6f);
    \\    const uint cut = metal::simd_min(enough ? lane : 0xFFu);
    \\    const uint total = metal::simd_sum(lane == cut ? cc : 0u);
    \\    if (lane == 0u) { su[0] = cut; su[1] = total; }
    \\  }
    \\  SD_BARRIER;
    \\  const uint cut = su[0];
    \\  const uint total = su[1];
    \\  route = (cut < 31u && total <= SD_CAP) ? 1u : 2u;
    \\  if (route == 1u) {
    \\    uint mine = 0u;
    \\    for (uint d = 0u; d <= cut; ++d) mine += lc[d];
    \\    const uint pre = metal::simd_prefix_exclusive_sum(mine);
    \\    const uint tot = metal::simd_sum(mine);
    \\    if (lane == 0u) pu[0][sg] = tot;
    \\    SD_BARRIER;
    \\    uint at = pre;
    \\    for (uint g = 0u; g < sg; ++g) at += pu[0][g];
    \\    SD_BARRIER;
    \\    // The shortlist in id order, and for a nucleus alone the mass of every id left out of it.
    \\    float rest = 0.0f;
    \\    for (uint j = 0u; j < n; ++j) {
    \\      const float v = SD_VAL(j);
    \\      if (sushi_draw_bin(m, v) <= cut) { ck[at] = sushi_draw_key(v); ce[at] = metal::precise::exp(v - m); ci[at] = i0 + j; ++at; }
    \\      else if (!do_k) rest += metal::precise::exp(v - m);
    \\    }
    \\    rest = metal::simd_sum(rest);
    \\    if (lane == 0u) pf[1][sg] = rest;
    \\    SD_BARRIER;
    \\    // Thread t ranks the shortlist's t-th id: how many come before it, and their mass.
    \\    const uint t = sg * 32u + lane;
    \\    const bool have = t < total;
    \\    const uint kt = have ? ck[t] : 0u;
    \\    uint rank = 0u;
    \\    float above = 0.0f;
    \\    if (have) for (uint q = 0u; q < total; ++q) {
    \\      const uint kq = ck[q];
    \\      const bool before = kq > kt || (kq == kt && q < t);
    \\      rank += before ? 1u : 0u;
    \\      above += before ? ce[q] : 0.0f;
    \\    }
    \\    const bool ink = have && (!do_k || rank < top_k);
    \\    float zs = 0.0f;
    \\    if (do_p) {
    \\      const float s = metal::simd_sum(ink ? ce[t] : 0.0f);
    \\      if (lane == 0u) pf[2][sg] = s;
    \\      SD_BARRIER;
    \\      for (uint g = 0u; g < 32u; ++g) zs += pf[2][g];
    \\      if (!do_k) for (uint g = 0u; g < 32u; ++g) zs += pf[1][g];
    \\    }
    \\    if (ink && (!do_p || above < metal::max(top_p, 1.17549435e-38f) * zs)) {
    \\      best = sushi_draw_unkey(kt) + sushi_draw_gumbel(ci[t], key0, key1);
    \\      bi = ci[t];
    \\    }
    \\  }
    \\}
    \\
    \\if (route != 1u) {
    \\  // Top-k: the key of the k-th value, how many ids sit above it, and how many of its ties the k takes.
    \\  uint kth = 0u, tk = 0u, base_k = 0u;
    \\  if (do_k) {
    \\    uint above = 0u;
    \\    for (uint r = 0u; r < 7u; ++r) {
    \\      const uint shift = SD_SHIFT(r), nb = SD_BINS(r), up = SD_UP(r);
    \\      uint bins[32];
    \\      for (uint d = 0u; d < nb; ++d) bins[d] = 0u;
    \\      for (uint j = 0u; j < n; ++j) {
    \\        const uint k = sushi_draw_key(SD_VAL(j));
    \\        if (r == 0u || (k >> up) == (kth >> up)) bins[(k >> shift) & (nb - 1u)] += 1u;
    \\      }
    \\      for (uint d = 0u; d < nb; ++d) { const uint c = metal::simd_sum(bins[d]); if (lane == 0u) pu[d][sg] = c; }
    \\      SD_BARRIER;
    \\      if (sg == 0u) {
    \\        uint t = 0u;
    \\        if (lane < nb) for (uint g = 0u; g < 32u; ++g) t += pu[nb - 1u - lane][g];
    \\        const uint exc = above + metal::simd_prefix_exclusive_sum(t);
    \\        const uint sel = metal::min(metal::simd_min((lane < nb && exc + t >= top_k) ? lane : 0xFFu), nb - 1u);
    \\        above = metal::simd_sum(lane == sel ? exc : 0u);
    \\        const uint eq = metal::simd_sum(lane == sel ? t : 0u);
    \\        if (lane == 0u) { su[0] = nb - 1u - sel; su[1] = above; su[2] = eq; }
    \\      }
    \\      SD_BARRIER;
    \\      kth |= su[0] << shift;
    \\    }
    \\    tk = top_k - su[1];
    \\    if (tk < su[2]) SD_BASE(base_k, k == kth, false);
    \\  }
    \\
    \\  // Nucleus: the last key whose ids, with all above them, hold the threshold's mass; its ties are kept one by
    \\  // one while the mass above each stays under.
    \\  const bool nuc = do_p && m > -INFINITY;
    \\  uint tau = 0u, base_p = 0u;
    \\  float s_gt = 0.0f, e_tau = 0.0f, pz = 0.0f;
    \\  if (nuc) {
    \\    float above = 0.0f, lim = 0.0f;
    \\    for (uint r = 0u; r < 7u; ++r) {
    \\      const uint shift = SD_SHIFT(r), nb = SD_BINS(r), up = SD_UP(r);
    \\      float bins[32];
    \\      for (uint d = 0u; d < nb; ++d) bins[d] = 0.0f;
    \\      {
    \\        uint ko = 0u;
    \\        for (uint j = 0u; j < n; ++j) {
    \\          const float v = SD_VAL(j);
    \\          const uint k = sushi_draw_key(v);
    \\          const bool ink = SD_INK;
    \\          if (ink && (r == 0u || (k >> up) == (tau >> up))) bins[(k >> shift) & (nb - 1u)] += metal::precise::exp(v - m);
    \\        }
    \\      }
    \\      for (uint d = 0u; d < nb; ++d) { const float c = metal::simd_sum(bins[d]); if (lane == 0u) pf[d][sg] = c; }
    \\      SD_BARRIER;
    \\      if (sg == 0u) {
    \\        float t = 0.0f;
    \\        if (lane < nb) for (uint g = 0u; g < 32u; ++g) t += pf[nb - 1u - lane][g];
    \\        if (r == 0u) lim = metal::max(top_p, 1.17549435e-38f) * metal::simd_sum(t);
    \\        const float exc = above + metal::simd_prefix_exclusive_sum(t);
    \\        const uint sel = metal::min(metal::simd_min((lane < nb && exc + t >= lim) ? lane : 0xFFu), nb - 1u);
    \\        above = metal::simd_sum(lane == sel ? exc : 0.0f);
    \\        if (lane == 0u) { su[0] = nb - 1u - sel; sf[0] = above; sf[1] = lim; }
    \\      }
    \\      SD_BARRIER;
    \\      tau |= su[0] << shift;
    \\    }
    \\    s_gt = sf[0];
    \\    pz = sf[1];
    \\    e_tau = metal::precise::exp(sushi_draw_unkey(tau) - m);
    \\    SD_BASE(base_p, ink && k == tau, true);
    \\  }
    \\
    \\  uint ko = 0u, po = 0u;
    \\  for (uint j = 0u; j < n; ++j) {
    \\    const float v = SD_VAL(j);
    \\    const uint k = sushi_draw_key(v);
    \\    const bool ink = SD_INK;
    \\    const bool keep = ink && (!nuc || k > tau || (k == tau && s_gt + float(base_p + po++) * e_tau < pz));
    \\    if (keep && v > -INFINITY) {
    \\      const float s = v + sushi_draw_gumbel(i0 + j, key0, key1);
    \\      if (s > best) { best = s; bi = i0 + j; }
    \\    }
    \\  }
    \\}
    \\
    \\const float sb = metal::simd_max(best);
    \\const uint cand = metal::simd_min(best == sb ? bi : 0xFFFFFFFFu);
    \\if (lane == 0u) { pf[0][sg] = sb; pi[sg] = cand; }
    \\SD_BARRIER;
    \\if (sg == 0u && lane == 0u) {
    \\  float fb = -INFINITY;
    \\  uint fi = 0xFFFFFFFFu;
    \\  for (uint g = 0u; g < 32u; ++g) {
    \\    const float s = pf[0][g];
    \\    const uint c = pi[g];
    \\    if (c != 0xFFFFFFFFu && (fi == 0xFFFFFFFFu || s > fb || (s == fb && c < fi))) { fb = s; fi = c; }
    \\  }
    \\  tok[row] = fi == 0xFFFFFFFFu ? 0u : fi;
    \\  route_of[row] = route;
    \\}
;

var kernel: ?mlx.mlx_fast_metal_kernel = null;

const Key = struct { rows: c_int, vocab: c_int, dtype: mlx.mlx_dtype, own: bool };
const Entry = struct { key: Key, value: mlx.mlx_fast_metal_kernel_config };
/// One config a (rows, vocab, dtype): a load draws at one vocab and dtype, at up to `max_rows` widths.
var configs: [4 * max_rows]?Entry = @splat(null);

/// Whether the kernel serves `rows` rows of `dtype` logits.
pub fn fits(n_rows: usize, dtype: mlx.mlx_dtype) bool {
    return n_rows >= 1 and n_rows <= max_rows and (dtype == .float32 or dtype == .bfloat16 or dtype == .float16);
}

/// The draws of `logits` (`[rows, vocab]`, already masked and penalized: an id at -inf is never drawn), one per row
/// under `keys[row]`, as a uint32 `[rows]` not yet evaluated.
pub fn rows(logits: Arr, temperature: f32, top_p: f32, top_k: u32, keys: []const [2]u32, s: mlx.mlx_stream) !Arr {
    const both = try routed(logits, temperature, top_p, top_k, keys, s);
    _ = mlx.mlx_array_free(both.route);
    return both.tok;
}

/// A launch's two outputs: the draws, and the route each row took (0 no filter, 1 the shortlist, 2 the descents).
pub const Routed = struct { tok: Arr, route: Arr };

/// `rows` with the route of each row beside its draw.
pub fn routed(logits: Arr, temperature: f32, top_p: f32, top_k: u32, keys: []const [2]u32, s: mlx.mlx_stream) !Routed {
    const shape = mlx.getShape(logits);
    const dtype = mlx.mlx_array_dtype(logits);
    if (shape.len != 2 or shape[0] != @as(c_int, @intCast(keys.len)) or !fits(keys.len, dtype) or !(temperature > 0)) return error.SamplerDrawUnsupported;
    const scope = mlx.engine.enter(s);
    defer scope.end();
    if (kernel == null) {
        const ins = mlx.mlx_vector_string_new_data(&.{ "logits", "fparams", "uparams", "keys" }, 4);
        defer _ = mlx.mlx_vector_string_free(ins);
        const outs = mlx.mlx_vector_string_new_data(&.{ "tok", "route_of" }, 2);
        defer _ = mlx.mlx_vector_string_free(outs);
        const k = mlx.mlx_fast_metal_kernel_new("sushi_sampler_draw", ins, outs, SOURCE, HEADER, true, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        kernel = k;
    }
    const key = Key{ .rows = shape[0], .vocab = shape[1], .dtype = dtype, .own = scope.own };
    var cached: ?mlx.mlx_fast_metal_kernel_config = null;
    for (configs) |item| if (item) |entry| if (std.meta.eql(key, entry.key)) {
        cached = entry.value;
        break;
    };
    const cfg = cached orelse mlx.mlx_fast_metal_kernel_config_new();
    var retained = cached != null;
    defer if (!retained) {
        _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    };
    if (cached == null) {
        for (0..2) |_| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{key.rows}, 1, .uint32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, key.rows * group_threads, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, group_threads, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "V", key.vocab));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dtype));
        for (&configs) |*item| if (item.* == null) {
            item.* = .{ .key = key, .value = cfg };
            retained = true;
            break;
        };
    }
    const fp = [2]f32{ 1.0 / temperature, top_p };
    const up = [1]u32{top_k};
    const fparams = mlx.mlx_array_new_data(&fp, &[_]c_int{2}, 1, .float32);
    defer _ = mlx.mlx_array_free(fparams);
    const uparams = mlx.mlx_array_new_data(&up, &[_]c_int{1}, 1, .uint32);
    defer _ = mlx.mlx_array_free(uparams);
    const key_arr = mlx.mlx_array_new_data(keys.ptr, &[_]c_int{ key.rows, 2 }, 2, .uint32);
    defer _ = mlx.mlx_array_free(key_arr);
    const ins = mlx.mlx_vector_array_new_data(&.{ logits, fparams, uparams, key_arr }, 4);
    defer _ = mlx.mlx_vector_array_free(ins);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, kernel.?, ins, cfg, s));
    var tok = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(tok);
    try mlx.check(mlx.mlx_vector_array_get(&tok, outs, 0));
    var route = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(route);
    try mlx.check(mlx.mlx_vector_array_get(&route, outs, 1));
    return .{ .tok = tok, .route = route };
}

fn keyOf(x: f32) u32 {
    const u: u32 = @bitCast(x);
    return if (u & 0x8000_0000 != 0) ~u else u | 0x8000_0000;
}

fn mix(h0: u32) u32 {
    var h = h0;
    h ^= h >> 16;
    h *%= 0x7FEB_352D;
    h ^= h >> 15;
    h *%= 0x846C_A68B;
    h ^= h >> 16;
    return h;
}

/// The kernel's Gumbel of id `id` under `key`, on the host.
pub fn gumbel(id: u32, key: [2]u32) f32 {
    const h = mix(mix(id *% 0x9E37_79B1 +% key[0]) +% key[1]);
    const u: f32 = (@as(f32, @floatFromInt(h >> 8)) + 0.5) * 5.9604644775390625e-8;
    return -@log(-@log(u));
}

/// What the rule says of one row, with how far its two closest calls were from going the other way.
pub const HostDraw = struct {
    id: u32,
    /// The runner-up of the draw, and the gap between the two scores.
    second: ?u32,
    gap: f32,
    /// How close, as a share of the kept mass, the nucleus came to keeping one id more or one fewer.
    nucleus_margin: f64,
    kept: usize,
};

/// The rule on the host: `row` is the logits as f32, `order` is scratch of the row's length.
pub fn hostDraw(row: []const f32, order: []u32, temperature: f32, top_p: f32, top_k: u32, key: [2]u32) HostDraw {
    const inv_t: f32 = 1.0 / temperature;
    const Ctx = struct {
        row: []const f32,
        inv_t: f32,
        fn val(self: @This(), i: u32) f32 {
            const v = self.row[i] * self.inv_t;
            return if (v == v) v else -std.math.inf(f32);
        }
        fn before(self: @This(), a: u32, b: u32) bool {
            const ka = keyOf(self.val(a));
            const kb = keyOf(self.val(b));
            return ka > kb or (ka == kb and a < b);
        }
    };
    const ctx = Ctx{ .row = row, .inv_t = inv_t };
    for (order, 0..) |*o, i| o.* = @intCast(i);
    std.mem.sort(u32, order, ctx, Ctx.before);
    var kept: usize = if (top_k > 0 and top_k < row.len) top_k else row.len;
    var margin: f64 = std.math.inf(f64);
    const m = ctx.val(order[0]);
    if (top_p < 1.0 and m > -std.math.inf(f32)) {
        var z: f64 = 0;
        for (order[0..kept]) |i| z += @exp(@as(f64, ctx.val(i)) - m);
        const pz = @as(f64, @max(top_p, std.math.floatMin(f32))) * z;
        var above: f64 = 0;
        var n: usize = 0;
        for (order[0..kept]) |i| {
            // The top one is kept whatever the threshold: it has no call to be close on.
            if (n > 0 or above > 0) margin = @min(margin, @abs(above - pz) / z);
            if (above < pz) n += 1;
            above += @exp(@as(f64, ctx.val(i)) - m);
        }
        kept = n;
    }
    var best: f32 = -std.math.inf(f32);
    var second: f32 = -std.math.inf(f32);
    var id: ?u32 = null;
    var runner: ?u32 = null;
    for (order[0..kept]) |i| {
        const v = ctx.val(i);
        if (!(v > -std.math.inf(f32))) continue;
        const sc = v + gumbel(i, key);
        if (id == null or sc > best or (sc == best and i < id.?)) {
            second = best;
            runner = id;
            best = sc;
            id = i;
        } else if (runner == null or sc > second) {
            second = sc;
            runner = i;
        }
    }
    return .{ .id = id orelse 0, .second = runner, .gap = best - second, .nucleus_margin = margin, .kept = kept };
}

const testing = std.testing;

/// Deterministic test logits: a few high ids over a wide floor, rounded to bf16 when `ties` (equal values at the top).
/// An odd seed adds a head 20 nats over the floor that falls a third of a nat a rank, as a model's row does.
fn testRow(out: []f32, seed: u64, ties: bool) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    for (out) |*x| x.* = r.floatNorm(f32) * 2.5 + (if (r.uintLessThan(u32, 200) == 0) @as(f32, 9.0) else 0.0);
    if (seed % 2 == 1) for (0..60) |rank| {
        out[(rank * 67 + seed) % out.len] = 30.0 - @as(f32, @floatFromInt(rank)) / 3.0;
    };
    if (ties) for (out) |*x| {
        x.* = @bitCast(@as(u32, @bitCast(x.*)) & 0xFFFF_0000);
    };
}

/// Rows a test's draws sent down each route, by filter: none, top-k, the nucleus, both.
var test_routes: [4][3]usize = @splat(@splat(0));

fn drawHost(a: std.mem.Allocator, s: mlx.mlx_stream, flat: []const f32, n_rows: usize, vocab: usize, dtype: mlx.mlx_dtype, temperature: f32, top_p: f32, top_k: u32, keys: []const [2]u32) ![]u32 {
    const scope = mlx.engine.enter(s);
    defer scope.end();
    const f32_arr = mlx.mlx_array_new_data(flat.ptr, &[_]c_int{ @intCast(n_rows), @intCast(vocab) }, 2, .float32);
    defer _ = mlx.mlx_array_free(f32_arr);
    var typed = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(typed);
    try mlx.check(mlx.mlx_astype(&typed, f32_arr, dtype, s));
    const both = try routed(typed, temperature, top_p, top_k, keys, s);
    const drawn = both.tok;
    defer _ = mlx.mlx_array_free(drawn);
    defer _ = mlx.mlx_array_free(both.route);
    try mlx.check(mlx.mlx_array_eval(drawn));
    try mlx.check(mlx.mlx_array_eval(both.route));
    const filter = @as(usize, @intFromBool(top_k > 0 and top_k < vocab)) | @as(usize, @intFromBool(top_p < 1.0)) << 1;
    for ((mlx.mlx_array_data_uint32(both.route) orelse return error.TestUnexpectedResult)[0..n_rows]) |r| test_routes[filter][r] += 1;
    const out = try a.alloc(u32, n_rows);
    const data = mlx.mlx_array_data_uint32(drawn) orelse return error.TestUnexpectedResult;
    @memcpy(out, data[0..n_rows]);
    return out;
}

test "sampler draw: the kernel draws what the rule draws, at every filter, with and without ties" {
    try mlx.requireGpu();
    // The engine draws where it serves a model, a GPU with matrix units; one without grants this kernel 768 threads a
    // group, under the 1,024 it is laid out on. A failed launch takes its raise with it.
    if (!mlx.engine.matrixUnits()) return error.SkipZigTest;
    const had_error = mlx.errorPending();
    errdefer mlx.dropLatchedErrorUnless(had_error);
    const a = testing.allocator;
    const vocab = 4099;
    const n_rows = 5;
    const flat = try a.alloc(f32, n_rows * vocab);
    defer a.free(flat);
    const order = try a.alloc(u32, vocab);
    defer a.free(order);
    var checked: usize = 0;
    var close: usize = 0;
    test_routes = @splat(@splat(0));
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("test_support.zig").restoreEngine(was);
    for ([_]bool{ true, false }) |own| for ([_]bool{ false, true }) |ties| {
        const s = if (own) mlx.engine.stream() else mlx.gpuStream();
        for (0..n_rows) |r| testRow(flat[r * vocab ..][0..vocab], 0x5a17 + r + @as(u64, @intFromBool(ties)) * 100, ties);
        for ([_]f32{ 0.7, 1.0, 1.4 }) |temperature| for ([_]u32{ 0, 1, 5, 40, 4098, 4099, 9000 }) |top_k| for ([_]f32{ 1.0, 0.95, 0.5, 0.0 }) |top_p| {
            var keys: [n_rows][2]u32 = undefined;
            for (&keys, 0..) |*k, r| k.* = .{ @intCast(0x9000 + r * 7 + top_k), @intCast(0x1234 + r) };
            const got = try drawHost(a, s, flat, n_rows, vocab, if (ties) .bfloat16 else .float32, temperature, top_p, top_k, &keys);
            defer a.free(got);
            for (got, 0..) |g, r| {
                const want = hostDraw(flat[r * vocab ..][0..vocab], order, temperature, top_p, top_k, keys[r]);
                // A call the two float orders can make differently is not a verdict: the nucleus within 1e-5 of the
                // kept mass of its threshold, or two scores within 1e-4.
                if (want.nucleus_margin < 1e-5) {
                    close += 1;
                    continue;
                }
                if (want.gap < 1e-4) {
                    close += 1;
                    try testing.expect(g == want.id or g == want.second.?);
                    continue;
                }
                try testing.expectEqual(want.id, g);
                checked += 1;
            }
        };
    };
    try testing.expect(checked > 20 * close);
    // Every route answered for some of them: an unfiltered row its own, each filter the shortlist and the descents.
    try testing.expect(test_routes[0][0] > 0);
    for (test_routes[1..]) |by_route| for (by_route[1..]) |seen| try testing.expect(seen > 0);
}

test "sampler draw: a row's draw is the same alone, beside other rows and at any place among them" {
    try mlx.requireGpu();
    // The engine draws where it serves a model, a GPU with matrix units; one without grants this kernel 768 threads a
    // group, under the 1,024 it is laid out on. A failed launch takes its raise with it.
    if (!mlx.engine.matrixUnits()) return error.SkipZigTest;
    const had_error = mlx.errorPending();
    errdefer mlx.dropLatchedErrorUnless(had_error);
    const a = testing.allocator;
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("test_support.zig").restoreEngine(was);
    const s = mlx.engine.stream();
    const vocab = 4099;
    const flat = try a.alloc(f32, max_rows * vocab);
    defer a.free(flat);
    for (0..max_rows) |r| testRow(flat[r * vocab ..][0..vocab], 0x77 + r, r % 2 == 0);
    var keys: [max_rows][2]u32 = undefined;
    for (&keys, 0..) |*k, r| k.* = .{ @intCast(31 * r + 5), @intCast(r ^ 0xabc) };
    const all = try drawHost(a, s, flat, max_rows, vocab, .float32, 0.8, 0.9, 40, &keys);
    defer a.free(all);
    for (0..max_rows) |r| {
        const alone = try drawHost(a, s, flat[r * vocab ..][0..vocab], 1, vocab, .float32, 0.8, 0.9, 40, keys[r .. r + 1]);
        defer a.free(alone);
        try testing.expectEqual(all[r], alone[0]);
    }
    // The same rows in reverse order, each with its own key.
    const rev = try a.alloc(f32, max_rows * vocab);
    defer a.free(rev);
    var rkeys: [max_rows][2]u32 = undefined;
    for (0..max_rows) |r| {
        @memcpy(rev[r * vocab ..][0..vocab], flat[(max_rows - 1 - r) * vocab ..][0..vocab]);
        rkeys[r] = keys[max_rows - 1 - r];
    }
    const back = try drawHost(a, s, rev, max_rows, vocab, .float32, 0.8, 0.9, 40, &rkeys);
    defer a.free(back);
    for (0..max_rows) |r| try testing.expectEqual(all[max_rows - 1 - r], back[r]);
}

test "sampler draw: an id at -inf is never drawn, top-k 1 and a zero nucleus are the lowest top id, and no finite value draws id 0" {
    try mlx.requireGpu();
    // The engine draws where it serves a model, a GPU with matrix units; one without grants this kernel 768 threads a
    // group, under the 1,024 it is laid out on. A failed launch takes its raise with it.
    if (!mlx.engine.matrixUnits()) return error.SkipZigTest;
    const had_error = mlx.errorPending();
    errdefer mlx.dropLatchedErrorUnless(had_error);
    const a = testing.allocator;
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("test_support.zig").restoreEngine(was);
    const s = mlx.engine.stream();
    const vocab = 2500;
    const row = try a.alloc(f32, vocab);
    defer a.free(row);
    @memset(row, 1.5);
    // Every id ties: the lowest wins under top-k 1 and under a zero nucleus, whatever the key.
    for ([_][2]u32{ .{ 1, 2 }, .{ 77, 0 }, .{ 0xffff_ffff, 9 } }) |key| {
        for ([_]struct { k: u32, p: f32 }{ .{ .k = 1, .p = 1.0 }, .{ .k = 0, .p = 0.0 }, .{ .k = 7, .p = 0.0 } }) |f| {
            const got = try drawHost(a, s, row, 1, vocab, .bfloat16, 1.0, f.p, f.k, &.{key});
            defer a.free(got);
            try testing.expectEqual(@as(u32, 0), got[0]);
        }
    }
    // Only ids 1700 and 2400 are finite: every draw is one of them, and both are drawn over a few keys.
    @memset(row, -std.math.inf(f32));
    row[1700] = 0.25;
    row[2400] = 0.5;
    var seen = [2]usize{ 0, 0 };
    for (0..64) |i| {
        const got = try drawHost(a, s, row, 1, vocab, .float32, 1.0, 1.0, 0, &.{.{ @intCast(i), 3 }});
        defer a.free(got);
        try testing.expect(got[0] == 1700 or got[0] == 2400);
        seen[@intFromBool(got[0] == 2400)] += 1;
    }
    try testing.expect(seen[0] > 8 and seen[1] > 8);
    @memset(row, -std.math.inf(f32));
    const none = try drawHost(a, s, row, 1, vocab, .float32, 1.0, 0.9, 5, &.{.{ 4, 4 }});
    defer a.free(none);
    try testing.expectEqual(@as(u32, 0), none[0]);
}

test "sampler draw: over many keys the kept ids come out at their share of the kept mass" {
    try mlx.requireGpu();
    // The engine draws where it serves a model, a GPU with matrix units; one without grants this kernel 768 threads a
    // group, under the 1,024 it is laid out on. A failed launch takes its raise with it.
    if (!mlx.engine.matrixUnits()) return error.SkipZigTest;
    const had_error = mlx.errorPending();
    errdefer mlx.dropLatchedErrorUnless(had_error);
    const a = testing.allocator;
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("test_support.zig").restoreEngine(was);
    const s = mlx.engine.stream();
    const vocab = 3000;
    const draws = 16 * 250;
    const flat = try a.alloc(f32, max_rows * vocab);
    defer a.free(flat);
    for (0..max_rows) |r| {
        const row = flat[r * vocab ..][0..vocab];
        @memset(row, -4.0);
        row[11] = 2.0;
        row[1200] = 1.0;
        row[2999] = 0.0;
        row[5] = -1.0;
    }
    // top-k 3 at temperature 1: the shares are e^2 : e^1 : e^0.
    const ids = [3]u32{ 11, 1200, 2999 };
    var counts = [3]f64{ 0, 0, 0 };
    var keys: [max_rows][2]u32 = undefined;
    var call: u32 = 0;
    while (call < draws / max_rows) : (call += 1) {
        for (&keys, 0..) |*k, r| k.* = .{ call * 16 + @as(u32, @intCast(r)), 0xfeed };
        const got = try drawHost(a, s, flat, max_rows, vocab, .float32, 1.0, 1.0, 3, &keys);
        defer a.free(got);
        for (got) |g| {
            const which = std.mem.indexOfScalar(u32, &ids, g) orelse return error.TestUnexpectedResult;
            counts[which] += 1;
        }
    }
    const z = @exp(2.0) + @exp(1.0) + 1.0;
    for ([_]f64{ @exp(2.0) / z, @exp(1.0) / z, 1.0 / z }, counts) |p, c| {
        const sigma = @sqrt(p * (1 - p) / @as(f64, draws));
        try testing.expect(@abs(c / @as(f64, draws) - p) < 4 * sigma);
    }
}
