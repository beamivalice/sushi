//! Fused Metal kernels of the deepseek_v41 decode and prefill path, each replacing a chain of MLX ops of
//! `dsv41_model.zig` (the composed path, which stays the reference of every test below).
//! Ported from mlx-serve's `deepseek_v41.zig` (MIT) and adapted to the original checkpoint layout: routed
//! experts are `mxfp4` banks with separate gate and up, the shared expert and the trunk are `mxfp8`.
const std = @import("std");
const mlx = @import("mlx.zig");
const sc_mod = @import("dsv41_scope.zig");
const qat_mod = @import("dsv41_qat.zig");

const A = sc_mod.A;
const Scope = sc_mod.Scope;
const dim = sc_mod.dim;
const ci = sc_mod.ci;

// ── launching ────────────────────────────────────────────────────────────

const Tmpl = struct { name: [*:0]const u8, v: c_int };
const TDtype = struct { name: [*:0]const u8, dt: mlx.mlx_dtype };
const OutSpec = struct { shape: []const c_int, dt: mlx.mlx_dtype };

/// A kernel's source and signature; the pipeline is built on first use.
const Def = struct {
    name: [*:0]const u8,
    ins: []const [*:0]const u8,
    outs: []const [*:0]const u8,
    source: [:0]const u8,
    header: [:0]const u8 = "",
    built: ?mlx.mlx_fast_metal_kernel = null,
};

const Grid = [3]usize;

fn launch(sc: *Scope, def: *Def, inputs: []const A, outs: []const OutSpec, grid: Grid, tg: Grid, ints: []const Tmpl, dts: []const TDtype, result: []A) !void {
    const k = try qat_mod.kernelNew(&def.built, def.name, def.ins, def.outs, def.source, def.header);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    for (outs) |o| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, o.shape.ptr, o.shape.len, o.dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ci(grid[0]), ci(grid[1]), ci(grid[2])));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, ci(tg[0]), ci(tg[1]), ci(tg[2])));
    for (ints) |t| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, t.name, t.v));
    for (dts) |t| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, t.name, t.dt));
    try qat_mod.applyKernel(sc, k, inputs, cfg, result);
}

fn f32Const(sc: *Scope, vals: []const f32) !A {
    return sc.floats(vals, &.{ci(vals.len)});
}

// ── routed and shared experts ────────────────────────────────────────────

/// Dot products of `R` rows of each of `NB` banks (`w0`/`s0`, then `w1`/`s1`) against `x` (K wide) on one
/// simdgroup, a lane taking packed words `lane, lane + 32, ...`. `FP4`: mxfp4 (e2m1 nibbles, eight a word, low
/// first); else mxfp8 (e4m3 bytes, four a word). One e8m0 scale per 32 columns; `rows` index every bank alike.
const EXPERT_HEADER =
    \\inline float dsv41_e2m1(uint n) {
    \\  const uint m = n & 7u, e = m >> 1, f = m & 1u;
    \\  const float v = e == 0u ? 0.5f * float(f) : (1.0f + 0.5f * float(f)) * float(1u << (e - 1u));
    \\  return (n & 8u) ? -v : v;
    \\}
    \\inline float dsv41_e4m3(uint b) {
    \\  const ushort v = ushort(b & 127u);
    \\  const ushort u = ushort((v << 7) | (((v + 1) >> 7) << 14) | ((b >> 7) << 15));
    \\  return float(as_type<half>(u)) * 256.0f;
    \\}
    \\inline float dsv41_e8m0(uint e) {
    \\  return as_type<float>(e == 0u ? 0x400000u : (e << 23));
    \\}
    \\template <int K, int R, int NB, bool FP4, typename X>
    \\inline void dsv41_dots(const device uint32_t* w0, const device uint8_t* s0, const device uint32_t* w1, const device uint8_t* s1,
    \\                       thread const size_t* rows, const device X* x, uint lane, thread float* acc) {
    \\  constexpr int VPW = FP4 ? 8 : 4;
    \\  constexpr int KP = K / VPW, KG = K / 32, KW = K / VPW;
    \\  for (int i = 0; i < NB * R; ++i) acc[i] = 0.0f;
    \\  for (int pk = int(lane); pk < KP; pk += 32) {
    \\    const int kb = pk * VPW, gi = kb / 32;
    \\    float xp[VPW];
    \\    for (int i = 0; i < VPW; ++i) xp[i] = float(x[kb + i]);
    \\    for (int b = 0; b < NB; ++b) {
    \\      const device uint32_t* w = b == 0 ? w0 : w1;
    \\      const device uint8_t* sc = b == 0 ? s0 : s1;
    \\      for (int r = 0; r < R; ++r) {
    \\        const uint32_t q = w[rows[r] * KW + pk];
    \\        float d = 0.0f;
    \\        for (int i = 0; i < VPW; ++i) d += xp[i] * (FP4 ? dsv41_e2m1((q >> (4 * i)) & 15u) : dsv41_e4m3((q >> (8 * i)) & 255u));
    \\        acc[b * R + r] += dsv41_e8m0(uint(sc[rows[r] * KG + gi])) * d;
    \\      }
    \\    }
    \\  }
    \\  for (int i = 0; i < NB * R; ++i) acc[i] = metal::simd_sum(acc[i]);
    \\}
;

/// One token's routed gate and up (slots 0..TOPK-1, experts `inds`) and, as slot TOPK, the shared expert's; RPS
/// output rows a simdgroup. Each dot rounds to T where `gather_qmm` writes it, then the clamped SwiGLU times the
/// slot's routing weight (1 for the shared) in f32, rounded once to T.
const MOE_GU_SOURCE =
    \\const uint lane = thread_index_in_simdgroup;
    \\const uint n0 = thread_position_in_grid.y * RPS, e = thread_position_in_grid.z;
    \\float a[2 * RPS];
    \\size_t rows[RPS];
    \\if (e < TOPK) {
    \\  for (int r = 0; r < RPS; ++r) rows[r] = size_t(inds[e]) * N + n0 + r;
    \\  dsv41_dots<K, RPS, 2, true>(gw, gs, uw, us, rows, x, lane, a);
    \\} else {
    \\  for (int r = 0; r < RPS; ++r) rows[r] = n0 + r;
    \\  dsv41_dots<K, RPS, 2, false>(sgw, sgs, suw, sus, rows, x, lane, a);
    \\}
    \\const float wt = e < TOPK ? wts[e] : 1.0f;
    \\if (lane == 0) for (int r = 0; r < RPS; ++r) {
    \\  float g = float(T(a[r])), u = float(T(a[RPS + r]));
    \\  if (lim[0] > 0.0f) { g = metal::min(g, lim[0]); u = metal::clamp(u, -lim[0], lim[0]); }
    \\  const float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(g)));
    \\  out[size_t(e) * N + n0 + r] = T((g * (g < 0.0f ? y : 1.0f - y)) * u * wt);
    \\}
;

/// One token's routed down projections summed in f32 in slot order, plus the shared expert's, rounded to T
/// where `moe` adds them, then the hyper-connection expand of the stream `res` into the new one; RPS rows a
/// simdgroup.
const MOE_DN_SOURCE =
    \\const uint lane = thread_index_in_simdgroup;
    \\const uint n0 = thread_position_in_grid.y * RPS;
    \\float tot[RPS], a[RPS];
    \\size_t rows[RPS];
    \\for (int r = 0; r < RPS; ++r) tot[r] = 0.0f;
    \\for (int e = 0; e < TOPK; ++e) {
    \\  for (int r = 0; r < RPS; ++r) rows[r] = size_t(inds[e]) * N + n0 + r;
    \\  dsv41_dots<K, RPS, 1, true>(dw, ds, dw, ds, rows, x + size_t(e) * K, lane, a);
    \\  for (int r = 0; r < RPS; ++r) tot[r] += float(T(a[r]));
    \\}
    \\for (int r = 0; r < RPS; ++r) rows[r] = n0 + r;
    \\dsv41_dots<K, RPS, 1, false>(sdw, sds, sdw, sds, rows, x + size_t(TOPK) * K, lane, a);
    \\if (lane == 0) for (int r = 0; r < RPS; ++r) {
    \\  const uint d = n0 + r;
    \\  const float y = float(T(tot[r] + float(T(a[r]))));
    \\  for (int i = 0; i < HC; ++i) {
    \\    float acc = 0.0f;
    \\    for (int j = 0; j < HC; ++j) acc = metal::fma(comb[j * HC + i], float(res[j * N + d]), acc);
    \\    out[i * N + d] = T(metal::fma(post[i], y, acc));
    \\  }
    \\}
;

/// A prefill MoE call's SwiGLU, one element a thread: row `r` of the sorted gate and up gathers `g` and `u`
/// (`[n, N]`) times its routing weight, the f32 math of `clampedSwiglu` in its order, rounded once to T.
const MOE_ACT_SOURCE =
    \\const uint i = thread_position_in_grid.x;
    \\const uint row = i / N;
    \\if (row >= uint(g_shape[0])) return;
    \\float gv = float(g[i]), uv = float(u[i]);
    \\if (lim[0] > 0.0f) { gv = metal::min(gv, lim[0]); uv = metal::clamp(uv, -lim[0], lim[0]); }
    \\const float s = 1.0f / (1.0f + metal::precise::exp(metal::abs(gv)));
    \\out[i] = T(gv * (gv < 0.0f ? s : 1.0f - s) * uv * wts[row]);
;

/// A prefill MoE call's combine, one element a thread: the row's K sorted down rows (`inv` names them) summed in
/// f32 plus the shared expert, rounded once to T.
const MOE_COMBINE_SOURCE =
    \\const uint i = thread_position_in_grid.x;
    \\const uint row = i / D, c = i % D;
    \\if (row >= uint(sh_shape[0])) return;
    \\float acc = 0.0f;
    \\for (int j = 0; j < K; ++j) acc += float(dn[size_t(inv[row * K + j]) * D + c]);
    \\out[i] = T(acc + float(sh[i]));
;

var moe_gu_def = Def{ .name = "dsv41_moe_gu", .ins = &.{ "x", "gw", "gs", "uw", "us", "sgw", "sgs", "suw", "sus", "inds", "wts", "lim" }, .outs = &.{"out"}, .source = MOE_GU_SOURCE, .header = EXPERT_HEADER };
var moe_dn_def = Def{ .name = "dsv41_moe_dn", .ins = &.{ "x", "dw", "ds", "sdw", "sds", "inds", "res", "post", "comb" }, .outs = &.{"out"}, .source = MOE_DN_SOURCE, .header = EXPERT_HEADER };
var moe_act_def = Def{ .name = "dsv41_moe_act", .ins = &.{ "g", "u", "wts", "lim" }, .outs = &.{"out"}, .source = MOE_ACT_SOURCE };
var moe_combine_def = Def{ .name = "dsv41_moe_combine", .ins = &.{ "dn", "inv", "sh" }, .outs = &.{"out"}, .source = MOE_COMBINE_SOURCE };

/// Output rows a simdgroup in the decode MoE kernels.
const moe_rps = 2;

/// An mxfp4 bank (`w` u32 `[E, N, K / 8]`, `s` u8 `[E, N, K / 32]`) or an mxfp8 linear (`w` u32 `[N, K / 4]`).
pub const Quant = struct { w: A, s: A };

pub const Experts = struct { gate: Quant, up: Quant, down: Quant };
pub const SharedLins = struct { gate: Quant, up: Quant, down: Quant };

/// Whether the decode MoE kernels serve this geometry.
pub fn moeDecodeFits(k_in: usize, inter: usize) bool {
    return k_in % 32 == 0 and inter % 32 == 0 and inter % moe_rps == 0 and k_in % moe_rps == 0;
}

/// The routed and shared SwiGLU activations of one token `x` `[1, K]`: `[topk + 1, inter]`, the last row the
/// shared expert's. `inds` `[topk]` int32 bank slots, `wts` `[topk]` f32.
pub fn moeGateUp(sc: *Scope, x: A, e: Experts, sh: SharedLins, inds: A, wts: A, limit: f32) !A {
    const K = dim(x, 1);
    const N = dim(e.gate.w, 1);
    const topk = dim(inds, 0);
    const dt = mlx.mlx_array_dtype(x);
    const lim = try f32Const(sc, &.{limit});
    var o: [1]A = undefined;
    try launch(sc, &moe_gu_def, &.{ x, e.gate.w, e.gate.s, e.up.w, e.up.s, sh.gate.w, sh.gate.s, sh.up.w, sh.up.s, inds, wts, lim }, &.{.{ .shape = &.{ ci(topk + 1), ci(N) }, .dt = dt }}, .{ 32, N / moe_rps, topk + 1 }, .{ 32, 1, 1 }, &.{
        .{ .name = "RPS", .v = moe_rps }, .{ .name = "K", .v = ci(K) }, .{ .name = "N", .v = ci(N) }, .{ .name = "TOPK", .v = ci(topk) },
    }, &.{.{ .name = "T", .dt = dt }}, &o);
    return o[0];
}

/// The new stream `[hc, D]` of one token: the routed down projections of `act` (from `moeGateUp`) and the
/// shared one summed, expanded into the stream `res` (`hc * D` values) by `post` `[hc]` and `comb` `[hc, hc]` (f32).
pub fn moeDownExpand(sc: *Scope, act: A, e: Experts, sh: SharedLins, inds: A, res: A, post: A, comb: A) !A {
    const K = dim(act, 1);
    const N = dim(e.down.w, 1);
    const topk = dim(inds, 0);
    const hc = mlx.mlx_array_size(res) / N;
    const dt = mlx.mlx_array_dtype(res);
    var o: [1]A = undefined;
    try launch(sc, &moe_dn_def, &.{ act, e.down.w, e.down.s, sh.down.w, sh.down.s, inds, res, post, comb }, &.{.{ .shape = &.{ ci(hc), ci(N) }, .dt = dt }}, .{ 32, N / moe_rps, 1 }, .{ 32, 1, 1 }, &.{
        .{ .name = "RPS", .v = moe_rps }, .{ .name = "K", .v = ci(K) }, .{ .name = "N", .v = ci(N) }, .{ .name = "TOPK", .v = ci(topk) }, .{ .name = "HC", .v = ci(hc) },
    }, &.{.{ .name = "T", .dt = dt }}, &o);
    return o[0];
}

/// One thread an output element, 256 a threadgroup.
fn flatLaunch(sc: *Scope, def: *Def, inputs: []const A, shape: []const c_int, dt: mlx.mlx_dtype, total: usize, ints: []const Tmpl) !A {
    var o: [1]A = undefined;
    try launch(sc, def, inputs, &.{.{ .shape = shape, .dt = dt }}, .{ (total + 255) / 256 * 256, 1, 1 }, .{ 256, 1, 1 }, ints, &.{.{ .name = "T", .dt = dt }}, &o);
    return o[0];
}

/// `silu(g) * u * w` of a sorted prefill gather: `g`, `u` `[n, 1, N]`, `wts` `[n]` f32.
pub fn moeAct(sc: *Scope, g: A, u: A, wts: A, limit: f32) !A {
    const n = dim(g, 0);
    const N = dim(g, 2);
    return flatLaunch(sc, &moe_act_def, &.{ g, u, wts, try f32Const(sc, &.{limit}) }, &.{ ci(n), 1, ci(N) }, mlx.mlx_array_dtype(g), n * N, &.{.{ .name = "N", .v = ci(N) }});
}

/// The rows' routed sums and shared expert: `dn` `[n, D]` sorted down rows, `inv` `[n]` the rank of each
/// (row, slot) pair in the sort, `sh` `[L, D]`.
pub fn moeCombine(sc: *Scope, dn: A, inv: A, sh: A, k: usize) !A {
    const L = dim(sh, 0);
    const D = dim(sh, 1);
    return flatLaunch(sc, &moe_combine_def, &.{ dn, inv, sh }, &.{ ci(L), ci(D) }, mlx.mlx_array_dtype(sh), L * D, &.{ .{ .name = "D", .v = ci(D) }, .{ .name = "K", .v = ci(k) } });
}

// ── router, top-k, dense f32 ─────────────────────────────────────────────

/// Router logits `[L, E]` in f32 from rows `x`: one simdgroup an expert, 8 inputs a lane per step.
const GATE_SOURCE =
    \\const uint lane = thread_index_in_simdgroup;
    \\const int e = threadgroup_position_in_grid.x * SGS + simdgroup_index_in_threadgroup;
    \\const int l = threadgroup_position_in_grid.y;
    \\const int E = w_shape[0], D = w_shape[1];
    \\if (e >= E) return;
    \\const device auto* wr = w + size_t(e) * D;
    \\const device auto* xr = x + size_t(l) * D;
    \\float acc = 0.0f;
    \\for (int k = lane * 8; k < D; k += 256)
    \\  for (int j = 0; j < 8; ++j) acc += float(xr[k + j]) * float(wr[k + j]);
    \\acc = metal::simd_sum(acc);
    \\if (lane == 0) logits[size_t(l) * E + e] = acc;
;

/// One row a threadgroup, one expert a thread: sqrt-softplus scores, the bias-picked top K (lowest index on ties),
/// their scores normalized (NORM) and scaled by scal[0].
const ROUTE_SOURCE =
    \\const int e = thread_position_in_threadgroup.x, l = threadgroup_position_in_grid.y;
    \\const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    \\const int E = logits_shape[1];
    \\threadgroup float t_s[1024], t_v[32];
    \\threadgroup int t_i[32], t_pick[K];
    \\float s = 0.0f, v = -INFINITY;
    \\if (e < E) {
    \\  const float z = logits[size_t(l) * E + e];
    \\  const float y = metal::exp(-metal::abs(z)), yp = 1.0f + y;
    \\  const float lp = yp == 1.0f ? y : y * (metal::log(yp) / (yp - 1.0f));
    \\  s = metal::sqrt(metal::max(z, 0.0f) + lp);
    \\  v = s + bias[(bias_shape[0] == 1 ? 0 : l) * E + e];
    \\  t_s[e] = s;
    \\}
    \\for (int r = 0; r < K; ++r) {
    \\  const float bv = metal::simd_max(v);
    \\  const int bi = metal::simd_min(v == bv && e < E ? e : 0x7fffffff);
    \\  if (lane == 0) { t_v[sg] = bv; t_i[sg] = bi; }
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  if (e == 0) {
    \\    const int nsg = (threads_per_threadgroup.x + 31) / 32;
    \\    float best = t_v[0]; int at = t_i[0];
    \\    for (int g = 1; g < nsg; ++g) if (t_v[g] > best || (t_v[g] == best && t_i[g] < at)) { best = t_v[g]; at = t_i[g]; }
    \\    t_pick[r] = at;
    \\  }
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  if (e == t_pick[r]) v = -INFINITY;
    \\}
    \\if (e == 0) {
    \\  float sum = 0.0f;
    \\  for (int r = 0; r < K; ++r) sum += t_s[t_pick[r]];
    \\  const float div = NORM ? sum + 1e-20f : 1.0f;
    \\  for (int r = 0; r < K; ++r) {
    \\    idx[l * K + r] = uint(t_pick[r]);
    \\    wts[l * K + r] = t_s[t_pick[r]] / div * scal[0];
    \\  }
    \\}
;

/// The K largest of each row of `x` [L, n] f32 (K <= n), as their indices in ascending order, ties to the lower
/// index: a radix select of the order-preserving uint keys, 8 bits a pass, then one pass writing every key above
/// the threshold and the first ties. One threadgroup a row.
const TOPK_SOURCE =
    \\const uint tid = thread_index_in_threadgroup, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    \\const int l = threadgroup_position_in_grid.y, n = x_shape[1];
    \\const auto xr = x + size_t(l) * n;
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
    \\    const uint key = dsv41_key(xr[i]);
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
    \\for (int i = i0; i < i1; ++i) { const uint key = dsv41_key(xr[i]); gt += key > prefix; eq += key == prefix; }
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
    \\  const uint key = dsv41_key(xr[i]);
    \\  if (key > prefix || (key == prefix && taken < take)) {
    \\    taken += key == prefix;
    \\    out[size_t(l) * K + pos++] = i;
    \\  }
    \\}
;

const TOPK_HEADER =
    \\inline uint dsv41_key(float f) {
    \\  const uint u = as_type<uint>(f);
    \\  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    \\}
;

/// `x` [L, K] f32 times dense rows `w` [N, K] (any float dtype) in f32, RPS rows a simdgroup: the product the
/// reference computes after widening w.
const DENSE_F32X_SOURCE =
    \\const uint lane = thread_index_in_simdgroup;
    \\const int n0 = thread_position_in_grid.y * RPS, l = thread_position_in_grid.z;
    \\const int K = x_shape[1], N = w_shape[0];
    \\float acc[RPS];
    \\for (int r = 0; r < RPS; ++r) acc[r] = 0.0f;
    \\const device float* xr = x + size_t(l) * K;
    \\for (int k = int(lane) * 4; k < K; k += 128) {
    \\  const float4 xv = *(const device float4*)(xr + k);
    \\  for (int r = 0; r < RPS; ++r) {
    \\    const device auto* wr = w + size_t(n0 + r) * K + k;
    \\    acc[r] += xv.x * float(wr[0]) + xv.y * float(wr[1]) + xv.z * float(wr[2]) + xv.w * float(wr[3]);
    \\  }
    \\}
    \\for (int r = 0; r < RPS; ++r) {
    \\  const float v = metal::simd_sum(acc[r]);
    \\  if (lane == 0 && n0 + r < N) out[size_t(l) * N + n0 + r] = v;
    \\}
;

var gate_def = Def{ .name = "dsv41_gate", .ins = &.{ "x", "w" }, .outs = &.{"logits"}, .source = GATE_SOURCE };
var route_def = Def{ .name = "dsv41_route", .ins = &.{ "logits", "bias", "scal" }, .outs = &.{ "idx", "wts" }, .source = ROUTE_SOURCE };
var topk_def = Def{ .name = "dsv41_topk", .ins = &.{ "x", "kk" }, .outs = &.{"out"}, .source = TOPK_SOURCE, .header = TOPK_HEADER };
var dense_f32x_def = Def{ .name = "dsv41_dense_f32x", .ins = &.{ "x", "w" }, .outs = &.{"out"}, .source = DENSE_F32X_SOURCE };

const gate_sgs = 8;
const topk_tg = 1024;
const dense_f32x_rps = 4;

/// `x` `[L, D]` times the router rows `w` `[E, D]` in f32 (`D` a multiple of 8).
pub fn gateLogits(sc: *Scope, x: A, w: A) !A {
    const L = dim(x, 0);
    const E = dim(w, 0);
    var o: [1]A = undefined;
    try launch(sc, &gate_def, &.{ x, w }, &.{.{ .shape = &.{ ci(L), ci(E) }, .dt = .float32 }}, .{ (E + gate_sgs - 1) / gate_sgs * 32, gate_sgs * L, 1 }, .{ 32, gate_sgs, 1 }, &.{.{ .name = "SGS", .v = gate_sgs }}, &.{}, &o);
    return o[0];
}

/// `logits` (f32 `[L, E]`, the router's pre-score output) rides along for a streamed layer's lossy pick.
pub const Routes = struct { idx: A, wts: A, logits: A = .{ .ctx = null } };

/// The `k` experts of each row of `logits` `[L, E]` (E <= 1024): picked by sqrt-softplus score plus `bias`
/// (`[1, E]` or `[L, E]`, f32), weighted by the unbiased score, normalized when `norm`, times `scale`.
pub fn route(sc: *Scope, logits: A, bias: A, k: usize, norm: bool, scale: f32) !Routes {
    const L = dim(logits, 0);
    const E = dim(logits, 1);
    const tg = (E + 31) / 32 * 32;
    var o: [2]A = undefined;
    try launch(sc, &route_def, &.{ logits, bias, try f32Const(sc, &.{scale}) }, &.{
        .{ .shape = &.{ ci(L), ci(k) }, .dt = .uint32 },
        .{ .shape = &.{ ci(L), ci(k) }, .dt = .float32 },
    }, .{ tg, L, 1 }, .{ tg, 1, 1 }, &.{ .{ .name = "K", .v = ci(k) }, .{ .name = "NORM", .v = @intFromBool(norm) } }, &.{}, &o);
    return .{ .idx = o[0], .wts = o[1] };
}

/// The indices of the `k` largest of each row of `x` `[L, n]` (f32), ascending, ties to the lower index.
pub fn topkIndices(sc: *Scope, x: A, k: usize) !A {
    var o: [1]A = undefined;
    // k rides in a buffer: as a template arg every index width up to index_topk was its own Metal compile.
    try launch(sc, &topk_def, &.{ x, try sc.ints(&.{ci(k)}, &.{1}) }, &.{.{ .shape = &.{ ci(dim(x, 0)), ci(k) }, .dt = .int32 }}, .{ topk_tg, dim(x, 0), 1 }, .{ topk_tg, 1, 1 }, &.{
        .{ .name = "TG", .v = topk_tg },
    }, &.{}, &o);
    return o[0];
}

/// Whether `denseF32x` serves a `[N, K]` weight.
pub fn denseF32xFits(n: usize, k: usize) bool {
    return k % 128 == 0 and n % dense_f32x_rps == 0;
}

/// `x` `[L, K]` (f32) times the rows of `w` `[N, K]` widened to f32, summed in f32.
pub fn denseF32x(sc: *Scope, x: A, w: A) !A {
    const L = dim(x, 0);
    const N = dim(w, 0);
    var o: [1]A = undefined;
    try launch(sc, &dense_f32x_def, &.{ x, w }, &.{.{ .shape = &.{ ci(L), ci(N) }, .dt = .float32 }}, .{ 32, N / dense_f32x_rps, L }, .{ 32, 1, 1 }, &.{.{ .name = "RPS", .v = dense_f32x_rps }}, &.{}, &o);
    return o[0];
}

// ── rotary, kv row ───────────────────────────────────────────────────────

/// One element a thread: the last RD channels of each D-wide vector rotated by `pos[row] * freqs` (adjacent pairs,
/// `rotate`'s math uncontracted, cos and sin as MLX's), rounded to T, then QK's activation round trip over blocks
/// inside one simdgroup (0 none, 1 fp8 / 32 / ue8m0, 2 fp4 / 32 / ue8m0, 3 fp4 / 16 / e4m3), written as OT.
/// INV turns the rotation back.
const ROPE_TAIL_SOURCE =
    \\#pragma clang fp contract(off)
    \\const uint i = thread_position_in_grid.x;
    \\if (QK == 0 && i >= threads_per_grid.x) return;
    \\const int c = i % D, row = i / ROW;
    \\const int head = D - RD;
    \\float r = float(x[i]);
    \\if (c >= head) {
    \\  const int p = (c - head) / 2;
    \\  const size_t b = i - (c - head) % 2;
    \\  const float x1 = float(x[b]), x2 = float(x[b + 1]);
    \\  const float theta = float(pos[row]) * freqs[p];
    \\  const float cs = metal::precise::cos(theta), sn = (INV ? -1.0f : 1.0f) * metal::precise::sin(theta);
    \\  r = float(T((c - head) % 2 == 0 ? x1 * cs - x2 * sn : x1 * sn + x2 * cs));
    \\}
    \\if (QK != 0) {
    \\  float amax = metal::abs(r);
    \\  for (ushort m = (QK == 3 ? 8 : 16); m > 0; m >>= 1) amax = metal::max(amax, metal::simd_shuffle_xor(amax, m));
    \\  r = qat_round<QK - 1>(r, qat_scale<QK - 1>(amax));
    \\}
    \\out[i] = OT(T(r));
;

/// `kvRow` after its projection: MLX's rms_norm (four elements a thread, its accumulation order), the tail
/// rotation, then (QAT) the fp8 / 32 / ue8m0 round trip. One threadgroup a row.
const KV_ROW_SOURCE =
    \\#pragma clang fp contract(off)
    \\const int l = threadgroup_position_in_grid.y;
    \\const uint t = thread_position_in_threadgroup.x, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    \\threadgroup float local_sums[32], inv_s[1];
    \\const size_t base = size_t(l) * D;
    \\float xs[4], acc = 0.0f;
    \\for (int i = 0; i < 4; ++i) {
    \\  const int idx = int(t) * 4 + i;
    \\  xs[i] = idx < D ? float(x[base + idx]) : 0.0f;
    \\  acc = metal::fma(xs[i], xs[i], acc);
    \\}
    \\acc = metal::simd_sum(acc);
    \\if (sg == 0) local_sums[lane] = 0.0f;
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\if (lane == 0) local_sums[sg] = acc;
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\if (sg == 0) {
    \\  acc = metal::simd_sum(local_sums[lane]);
    \\  if (lane == 0) inv_s[0] = metal::precise::rsqrt(acc / float(D) + eps[0]);
    \\}
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\const float inv = inv_s[0];
    \\float r[4];
    \\for (int i = 0; i < 4; ++i) {
    \\  const int idx = int(t) * 4 + i;
    \\  r[i] = idx < D ? float(w[idx] * T(xs[i] * inv)) : 0.0f;
    \\}
    \\const int head = D - RD;
    \\for (int i = 0; i < 4; i += 2) {
    \\  const int idx = int(t) * 4 + i;
    \\  if (idx >= head && idx < D) {
    \\    const float theta = float(pos[l]) * freqs[(idx - head) / 2];
    \\    const float cs = metal::precise::cos(theta), sn = metal::precise::sin(theta);
    \\    const float x1 = r[i], x2 = r[i + 1];
    \\    r[i] = float(T(x1 * cs - x2 * sn));
    \\    r[i + 1] = float(T(x1 * sn + x2 * cs));
    \\  }
    \\}
    \\if (QAT) {
    \\  float amax = 0.0f;
    \\  for (int i = 0; i < 4; ++i) amax = metal::max(amax, metal::abs(r[i]));
    \\  for (ushort m = 4; m > 0; m >>= 1) amax = metal::max(amax, metal::simd_shuffle_xor(amax, m));
    \\  const float s = qat_scale<0>(amax);
    \\  for (int i = 0; i < 4; ++i) r[i] = qat_round<0>(r[i], s);
    \\}
    \\for (int i = 0; i < 4; ++i) if (int(t) * 4 + i < D) out[base + int(t) * 4 + i] = T(r[i]);
;

var rope_tail_def = Def{ .name = "dsv41_rope_tail", .ins = &.{ "x", "freqs", "pos" }, .outs = &.{"out"}, .source = ROPE_TAIL_SOURCE, .header = qat_mod.QAT_HEADER };
var kv_row_def = Def{ .name = "dsv41_kv_row", .ins = &.{ "x", "w", "freqs", "pos", "eps" }, .outs = &.{"out"}, .source = KV_ROW_SOURCE, .header = qat_mod.QAT_HEADER };

/// Which activation round trip follows a rotation (`Qat` of `dsv41_qat.zig`, or none).
pub const RopeQat = enum(c_int) { none = 0, fp8_ue8m0 = 1, fp4_ue8m0 = 2, fp4_e4m3 = 3 };

/// `x` `[L, D]` or `[L, H, D]` with its last `rd` channels rotated by `pos[row] * freqs` (`inverse` turns them
/// back), then `qat`'s round trip, in `x`'s dtype (`out_dt` if given).
pub fn ropeTail(sc: *Scope, x: A, freqs: A, pos: A, rd: usize, inverse: bool, qat: RopeQat, out_dt: ?mlx.mlx_dtype) !A {
    const n = mlx.mlx_array_size(x);
    const dt = mlx.mlx_array_dtype(x);
    const odt = out_dt orelse dt;
    const D = dim(x, -1);
    var o: [1]A = undefined;
    try launch(sc, &rope_tail_def, &.{ x, freqs, pos }, &.{.{ .shape = mlx.getShape(x), .dt = odt }}, .{ n, 1, 1 }, .{ @min(n, 256), 1, 1 }, &.{
        .{ .name = "RD", .v = ci(rd) },
        .{ .name = "D", .v = ci(D) },
        .{ .name = "ROW", .v = ci(n / dim(x, 0)) },
        .{ .name = "QK", .v = @backingInt(qat) },
        .{ .name = "INV", .v = @intFromBool(inverse) },
    }, &.{ .{ .name = "T", .dt = dt }, .{ .name = "OT", .dt = odt } }, &o);
    return o[0];
}

/// Whether `kvRow` serves rows of `D` channels.
pub fn kvRowFits(d: usize) bool {
    return d % 32 == 0 and d <= 4096;
}

/// The projected window row `kv` `[L, D]`: rms-normed by `w`, its tail rotated at `pos[row]`, and (`qat`) the fp8
/// round trip.
pub fn kvRow(sc: *Scope, kv: A, w: A, freqs: A, pos: A, rd: usize, eps: f32, qat: bool) !A {
    const L = dim(kv, 0);
    const D = dim(kv, 1);
    const dt = mlx.mlx_array_dtype(kv);
    const tg = ((D + 3) / 4 + 31) / 32 * 32;
    var o: [1]A = undefined;
    try launch(sc, &kv_row_def, &.{ kv, w, freqs, pos, try f32Const(sc, &.{eps}) }, &.{.{ .shape = &.{ ci(L), ci(D) }, .dt = dt }}, .{ tg, L, 1 }, .{ tg, 1, 1 }, &.{
        .{ .name = "RD", .v = ci(rd) }, .{ .name = "D", .v = ci(D) }, .{ .name = "QAT", .v = @intFromBool(qat) },
    }, &.{.{ .name = "T", .dt = dt }}, &o);
    return o[0];
}

// ── attention ────────────────────────────────────────────────────────────

/// Decode attention for one (row, head) per threadgroup: the window rows the query sees (read from the ring after
/// this step's write) then its selected compressed rows, f32 online softmax split over NSG simdgroups, the sink in
/// the denominator only. meta = {p0, L, K, window, cap, n_comp}; `topk` entries < 0 or >= n_comp are skipped.
const ATTN_DEC_BODY =
    \\const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    \\const int i = threadgroup_position_in_grid.y, h = threadgroup_position_in_grid.x;
    \\const int p0 = meta[0], K = meta[2], window = meta[3], cap = meta[4], nc = meta[5];
    \\const int H = q_shape[1];
    \\constexpr int V = D / 32;
    \\const int pos = p0 + i;
    \\const int lo = metal::max(0, pos - window + 1);
    \\const int nw = pos - lo + 1;
    \\float qv[V], acc[V];
    \\for (int v = 0; v < V; ++v) { qv[v] = float(q[(i * H + h) * D + lane * V + v]); acc[v] = 0.0f; }
    \\float mx = -INFINITY, den = 0.0f;
    \\const float scale = scal[0];
    \\const int S = int(threadgroups_per_grid.z), sp = threadgroup_position_in_grid.z, chunk = (nw + K + S - 1) / S;
    \\const int t_lo = sp * chunk, t_hi = metal::min(nw + K, t_lo + chunk);
    \\for (int t = t_lo + sg; t < t_hi; t += NSG) {
    \\  const device KT* row;
    \\  if (t < nw) row = ring + size_t((lo + t) % cap) * D;
    \\  else {
    \\    const int c = topk[i * K + t - nw];
    \\    if (c < 0 || c >= nc) continue;
    \\    row = comp + size_t(c) * D;
    \\  }
    \\  float kv[V], dot = 0.0f;
    \\  for (int v = 0; v < V; ++v) { kv[v] = float(row[lane * V + v]); dot += qv[v] * kv[v]; }
    \\  const float s = metal::simd_sum(dot) * scale;
    \\  const float nm = metal::max(mx, s);
    \\  const float corr = metal::precise::exp(mx - nm), p = metal::precise::exp(s - nm);
    \\  den = den * corr + p;
    \\  for (int v = 0; v < V; ++v) acc[v] = acc[v] * corr + p * kv[v];
    \\  mx = nm;
    \\}
    \\threadgroup float t_acc[NSG * D];
    \\threadgroup float t_mx[NSG], t_den[NSG];
    \\for (int v = 0; v < V; ++v) t_acc[sg * D + lane * V + v] = acc[v];
    \\if (lane == 0) { t_mx[sg] = mx; t_den[sg] = den; }
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
;
const ATTN_DEC_SOURCE = ATTN_DEC_BODY ++
    \\float M = sink[h];
    \\for (int g = 0; g < NSG; ++g) M = metal::max(M, t_mx[g]);
    \\float total = metal::precise::exp(sink[h] - M);
    \\float w[NSG];
    \\for (int g = 0; g < NSG; ++g) { w[g] = t_den[g] > 0.0f ? metal::precise::exp(t_mx[g] - M) : 0.0f; total += t_den[g] * w[g]; }
    \\for (int d = sg * 32 + lane; d < D; d += NSG * 32) {
    \\  float o = 0.0f;
    \\  for (int g = 0; g < NSG; ++g) o += t_acc[g * D + d] * w[g];
    \\  out[(i * H + h) * D + d] = dsv41_unrope<OT, RD, D>(OT(o / total), d, p0 + i, freqs);
    \\}
;

/// The inverse rotation of the tail channel `d` of a head row the attention just wrote (`v`, at position `pos`):
/// pairs are adjacent lanes, so every lane of the simdgroup must call it. `P`: MLX passes a small input as a
/// constant pointer, else a device one.
const UNROPE_HEADER =
    \\template <typename OT, int RD, int D, typename P>
    \\inline OT dsv41_unrope(OT v, int d, int pos, P freqs) {
    \\  #pragma clang fp contract(off)
    \\  const float pv = metal::simd_shuffle_xor(float(v), ushort(1));
    \\  if (RD == 0 || d < D - RD) return v;
    \\  const bool even = (d - (D - RD)) % 2 == 0;
    \\  const float x1 = even ? float(v) : pv, x2 = even ? pv : float(v);
    \\  const float theta = float(pos) * freqs[(d - (D - RD)) / 2];
    \\  const float cs = metal::precise::cos(theta), sn = -metal::precise::sin(theta);
    \\  return OT(even ? x1 * cs - x2 * sn : x1 * sn + x2 * cs);
    \\}
;

/// `ATTN_DEC_SOURCE` over one of S splits of its rows: the split's max, sum and unnormalized output
/// (`ATTN_MERGE_SOURCE` finishes it).
const ATTN_SPLIT_SOURCE = ATTN_DEC_BODY ++
    \\float M = -INFINITY, l = 0.0f, w[NSG];
    \\for (int g = 0; g < NSG; ++g) M = metal::max(M, t_mx[g]);
    \\for (int g = 0; g < NSG; ++g) { w[g] = t_den[g] > 0.0f ? metal::precise::exp(t_mx[g] - M) : 0.0f; l += t_den[g] * w[g]; }
    \\const size_t base = (size_t(i) * H + h) * S + sp;
    \\for (int d = sg * 32 + lane; d < D; d += NSG * 32) {
    \\  float o = 0.0f;
    \\  for (int g = 0; g < NSG; ++g) o += t_acc[g * D + d] * w[g];
    \\  pacc[base * D + d] = o;
    \\}
    \\if (sg == 0 && lane == 0) { pm[base] = M; pl[base] = l; }
;

/// The S partials of one (row, head) `ATTN_SPLIT_SOURCE` wrote, merged with the sink in the denominator, then the
/// inverse rotation (as `ATTN_DEC_SOURCE` ends); one thread a channel.
const ATTN_MERGE_SOURCE =
    \\const int d = thread_position_in_threadgroup.x, h = threadgroup_position_in_grid.y, i = threadgroup_position_in_grid.z;
    \\const int H = pm_shape[1], S = pm_shape[2];
    \\const size_t base = (size_t(i) * H + h) * S;
    \\float M = sink[h];
    \\for (int j = 0; j < S; ++j) M = metal::max(M, pm[base + j]);
    \\float total = metal::precise::exp(sink[h] - M), o = 0.0f;
    \\for (int j = 0; j < S; ++j) {
    \\  const float w = pl[base + j] > 0.0f ? metal::precise::exp(pm[base + j] - M) : 0.0f;
    \\  total += pl[base + j] * w;
    \\  o += pacc[(base + j) * D + d] * w;
    \\}
    \\out[(size_t(i) * H + h) * D + d] = dsv41_unrope<OT, RD, D>(OT(o / total), d, meta[0] + i, freqs);
;

/// A wide span's softmax, one simdgroup per (row, head): the raw window scores `lw` [L*H, Wk] under the band
/// (query l sees columns (wp + l - window, wp + l]), the raw selected scores `lc` [L, H, K] where the pick is >= 0,
/// scaled by `scal` here in f32, max and sum in f32 with the sink in the denominator only, the normalized
/// probabilities written in `OT`. NJ columns a lane. meta = {wp, window, Wk, K}.
const ATTN_SOFTMAX_SOURCE =
    \\const uint lane = thread_index_in_simdgroup;
    \\const int h = thread_position_in_grid.x / 32, l = thread_position_in_grid.y;
    \\const int H = sink_shape[1];
    \\const int wp = meta[0], window = meta[1], Wk = meta[2], K = meta[3];
    \\const int N = Wk + K;
    \\const float scale = scal[0];
    \\const size_t rw = size_t(l) * H + h;
    \\float s[NJ];
    \\float mx = -INFINITY;
    \\for (int j = 0; j < NJ; ++j) {
    \\  const int c = lane + 32 * j;
    \\  float v = -INFINITY;
    \\  if (c < Wk) { if (c <= wp + l && c + window > wp + l) v = scale * float(lw[rw * Wk + c]); }
    \\  else if (c < N) { if (topk[size_t(l) * K + c - Wk] >= 0) v = scale * float(lc[rw * K + c - Wk]); }
    \\  s[j] = v;
    \\  mx = metal::max(mx, v);
    \\}
    \\mx = metal::simd_max(mx);
    \\const float sk = sink[h];
    \\const float M = metal::max(mx, sk);
    \\float den = 0.0f;
    \\for (int j = 0; j < NJ; ++j) { const float p = metal::precise::exp(s[j] - M); s[j] = p; den += p; }
    \\den = metal::simd_sum(den) + metal::precise::exp(sk - M);
    \\const float inv = 1.0f / den;
    \\for (int j = 0; j < NJ; ++j) {
    \\  const int c = lane + 32 * j;
    \\  if (c < Wk) pw[rw * Wk + c] = OT(s[j] * inv);
    \\  else if (c < N) pc[rw * K + c - Wk] = OT(s[j] * inv);
    \\}
;

var attn_dec_def = Def{ .name = "dsv41_attn_decode", .ins = &.{ "q", "ring", "comp", "topk", "sink", "meta", "scal", "freqs" }, .outs = &.{"out"}, .source = ATTN_DEC_SOURCE, .header = UNROPE_HEADER };
var attn_split_def = Def{ .name = "dsv41_attn_split", .ins = &.{ "q", "ring", "comp", "topk", "sink", "meta", "scal" }, .outs = &.{ "pacc", "pm", "pl" }, .source = ATTN_SPLIT_SOURCE };
var attn_merge_def = Def{ .name = "dsv41_attn_merge", .ins = &.{ "pacc", "pm", "pl", "sink", "meta", "freqs" }, .outs = &.{"out"}, .source = ATTN_MERGE_SOURCE, .header = UNROPE_HEADER };
var attn_softmax_def = Def{ .name = "dsv41_attn_softmax", .ins = &.{ "lw", "lc", "topk", "sink", "meta", "scal" }, .outs = &.{ "pw", "pc" }, .source = ATTN_SOFTMAX_SOURCE };

const attn_dec_nsg = 8;

/// Threadgroups a (row, head) splits its `rows` (window + selected) into: one per 160 rows.
fn attnSplits(rows: usize) usize {
    return @max(1, @min(8, rows / 160));
}

pub const DecodeArgs = struct {
    /// `[L, H, D]` roped queries.
    q: A,
    /// The window ring `[cap, D]`, already holding this step's rows.
    ring: A,
    /// `[n_comp, D]` compressed rows with `topk` `[L, K]` int32 picks (-1 = none), or none.
    comp: A = sc_mod.none,
    topk: A = sc_mod.none,
    /// `[1, H, 1]` f32.
    sink: A,
    p0: usize,
    window: usize,
    scale: f32,
    /// The inverse rotation of the heads' tail (`rd` channels, `freqs`), or `rd` 0.
    freqs: A = sc_mod.none,
    rd: usize = 0,
    out_dt: mlx.mlx_dtype,
};

/// Whether `attnDecode` serves head rows of `d` channels.
pub fn attnDecodeFits(d: usize) bool {
    return d % 32 == 0 and d <= 1024;
}

/// The heads `[L, H, D]` of a decode block, softmax over window and selected rows with the sink, inverse-rotated.
pub fn attnDecode(sc: *Scope, a: DecodeArgs) !A {
    return attnDecodeSplit(sc, a, attnSplits(a.window + (if (a.topk.ctx != null) dim(a.topk, 1) else 0)));
}

pub fn attnDecodeSplit(sc: *Scope, a: DecodeArgs, splits: usize) !A {
    const L = dim(a.q, 0);
    const H = dim(a.q, 1);
    const D = dim(a.q, 2);
    const K = if (a.topk.ctx != null) dim(a.topk, 1) else 0;
    const kt = mlx.mlx_array_dtype(a.ring);
    const meta_v = [_]i32{ ci(a.p0), ci(L), ci(K), ci(a.window), ci(dim(a.ring, 0)), if (K > 0) ci(dim(a.comp, 0)) else 0 };
    const meta = try sc.ints(&meta_v, &.{meta_v.len});
    const scal = try f32Const(sc, &.{a.scale});
    const comp = if (K > 0) try sc.astype(a.comp, kt) else a.ring;
    const ids = if (K > 0) try sc.astype(a.topk, .int32) else meta;
    const freqs = if (a.rd > 0) a.freqs else try sc.zeros(&.{1}, .float32);
    const grid: Grid = .{ H * attn_dec_nsg * 32, L, splits };
    const tg: Grid = .{ attn_dec_nsg * 32, 1, 1 };
    // The split count is the grid's depth, read by the kernels, so a context crossing a split step compiles nothing.
    const common = [_]Tmpl{ .{ .name = "D", .v = ci(D) }, .{ .name = "NSG", .v = attn_dec_nsg } };
    if (splits == 1) {
        var o: [1]A = undefined;
        try launch(sc, &attn_dec_def, &.{ a.q, a.ring, comp, ids, a.sink, meta, scal, freqs }, &.{.{ .shape = &.{ ci(L), ci(H), ci(D) }, .dt = a.out_dt }}, grid, tg, &(common ++ [_]Tmpl{.{ .name = "RD", .v = ci(a.rd) }}), &.{ .{ .name = "KT", .dt = kt }, .{ .name = "OT", .dt = a.out_dt } }, &o);
        return o[0];
    }
    var parts: [3]A = undefined;
    try launch(sc, &attn_split_def, &.{ a.q, a.ring, comp, ids, a.sink, meta, scal }, &.{
        .{ .shape = &.{ ci(L), ci(H), ci(splits), ci(D) }, .dt = .float32 },
        .{ .shape = &.{ ci(L), ci(H), ci(splits) }, .dt = .float32 },
        .{ .shape = &.{ ci(L), ci(H), ci(splits) }, .dt = .float32 },
    }, grid, tg, &common, &.{.{ .name = "KT", .dt = kt }}, &parts);
    var o: [1]A = undefined;
    try launch(sc, &attn_merge_def, &.{ parts[0], parts[1], parts[2], a.sink, meta, freqs }, &.{.{ .shape = &.{ ci(L), ci(H), ci(D) }, .dt = a.out_dt }}, .{ D, H, L }, .{ D, 1, 1 }, &.{
        .{ .name = "D", .v = ci(D) }, .{ .name = "RD", .v = ci(a.rd) },
    }, &.{.{ .name = "OT", .dt = a.out_dt }}, &o);
    return o[0];
}

/// The most window + selected columns `attnWide` scores (NJ <= 32 floats a lane).
pub const attn_softmax_max_cols = 1024;

pub const Probs = struct { pw: A, pc: A };

/// Normalized f16 probabilities of a wide span: `lw` `[L * H, Wk]` raw window scores under the band that starts
/// at `wp`, `lc` `[L, H, K]` raw selected scores where `topk` `[L, K]` >= 0 (else `lw`, `topk` unread), `scale`
/// applied here in f32, the sink in the denominator only.
pub fn attnSoftmax(sc: *Scope, lw: A, lc: A, topk: A, sink: A, wp: usize, window: usize, scale: f32, L: usize, H: usize, K: usize) !Probs {
    const Wk = dim(lw, 1);
    const meta_v = [_]i32{ ci(wp), ci(window), ci(Wk), ci(K) };
    const meta = try sc.ints(&meta_v, &.{meta_v.len});
    const h16 = mlx.mlx_dtype.float16;
    var o: [2]A = undefined;
    try launch(sc, &attn_softmax_def, &.{ lw, lc, if (K > 0) topk else meta, sink, meta, try f32Const(sc, &.{scale}) }, &.{
        .{ .shape = &.{ ci(L * H), ci(Wk) }, .dt = h16 },
        .{ .shape = &.{ ci(if (K > 0) L else 1), ci(if (K > 0) H else 1), ci(@max(K, 1)) }, .dt = h16 },
    }, .{ H * 32, L, 1 }, .{ @min(H * 32, 128), 1, 1 }, &.{.{ .name = "NJ", .v = ci((Wk + K + 127) / 128 * 4) }}, &.{.{ .name = "OT", .dt = h16 }}, &o);
    return .{ .pw = o[0], .pc = o[1] };
}

/// A wide span's attention with f16 GEMMs and the softmax kernel: `attnCore` with the window band starting at `wp`
/// (query i sees window rows (wp + i - window, wp + i]). The operands convert to f16 exactly (fp8/fp4 codes times
/// powers of two) and the keys are rms-normed, so nothing nears f16's maximum; q carries the power of two under
/// `scale` so the raw scores sit within 2x of the logits, and the kernel applies the rest in f32. Result `[L, H, D]`.
pub fn attnWide(sc: *Scope, sink: A, q: A, keys_w: A, wp: usize, window: usize, comp_rows: A, topk: A, scale: f32, out_dt: mlx.mlx_dtype) !A {
    const L = dim(q, 0);
    const H = dim(q, 1);
    const D = dim(q, 2);
    const K = if (comp_rows.ctx != null) dim(topk, 1) else 0;
    const h16 = mlx.mlx_dtype.float16;
    const pow2 = @exp2(@floor(@log2(scale)));
    const qh = try sc.astype(try sc.mul(q, try sc.astype(try sc.f(pow2), mlx.mlx_array_dtype(q))), h16);
    const kw = try sc.astype(keys_w, h16);
    const lw = try sc.matmul(try sc.reshape(qh, &.{ ci(L * H), ci(D) }), try sc.t2(kw));
    var g = sc_mod.none;
    var lc = lw;
    var picks = sc_mod.none;
    if (K > 0) {
        picks = try sc.astype(topk, .int32);
        g = try sc.take(try sc.astype(comp_rows, h16), try sc.maximum(picks, try sc.i(0)), 0);
        lc = try sc.matmul(qh, try sc.transpose(g, &.{ 0, 2, 1 }));
    }
    const p = try attnSoftmax(sc, lw, lc, picks, sink, wp, window, scale / pow2, L, H, K);
    var o = try sc.reshape(try sc.matmul(p.pw, kw), &.{ ci(L), ci(H), ci(D) });
    if (K > 0) o = try sc.add(o, try sc.matmul(p.pc, g));
    return sc.astype(o, out_dt);
}

/// A wide span's attention in one pass, one threadgroup a (query row, HB heads): the keys of row l (its band of
/// window rows from `wp`, then its K picked compressed rows by index, `topk` < 0 or >= nc skipped) stream through
/// threadgroup memory KT at a time, shared by the HB heads. Each of NSG simdgroups owns D / NSG channels of both
/// matrix products: it scores its channels (`S` partials, summed in simdgroup order) and accumulates its slice of
/// the output. The softmax is online in f32, seeded by the sink (max = sink, sum = 1, zero value), so rows with
/// no key leave 0. P reaches the bf16 matrix unit as a high and a low part, which keeps ~16 bits of it.
/// meta = {wp, window, K, nc}.
const ATTN_FUSED_SOURCE =
    \\const uint tid = thread_index_in_threadgroup, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    \\const int l = threadgroup_position_in_grid.y, h0 = threadgroup_position_in_grid.x * HB;
    \\const int wp = meta[0], window = meta[1], K = meta[2], nc = meta[3];
    \\const int H = q_shape[1];
    \\const int c1 = wp + l, c0 = metal::max(0, c1 - window + 1), nw = c1 - c0 + 1, total = nw + K;
    \\constexpr int RT = HB / 8, DS = D / NSG, DT = DS / 8, KC = KT / 8, NT = NSG * 32, TPH = NT / HB, NJ = KT / TPH;
    \\const int d0 = int(sg) * DS;
    \\const short qid = short(lane) / 4;
    \\const short fm = (qid & 4) + ((short(lane) / 2) % 4), fn = (qid & 2) * 2 + (short(lane) % 2) * 2;
    \\threadgroup T kt[KT * D];
    \\threadgroup float sp[NSG * HB * KT];
    \\threadgroup T ph[HB * KT], pl[HB * KT];
    \\threadgroup float cr[HB], rl[HB];
    \\threadgroup int rsrc[KT];
    \\const int hh = int(tid) / TPH, sub = int(tid) % TPH;
    \\float m = sink[h0 + hh], lsum = 1.0f;
    \\const float scale = scal[0];
    \\metal::simdgroup_matrix<T, 8, 8> qf[RT][DT];
    \\metal::simdgroup_matrix<float, 8, 8> o[RT][DT];
    \\#pragma clang loop unroll(full)
    \\for (int r = 0; r < RT; ++r)
    \\  #pragma clang loop unroll(full)
    \\  for (int d = 0; d < DT; ++d) {
    \\    metal::simdgroup_load(qf[r][d], q + (size_t(l) * H + h0 + r * 8) * D + d0 + d * 8, D);
    \\    o[r][d] = metal::make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    \\  }
    \\for (int t0 = 0; t0 < total; t0 += KT) {
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  if (tid < KT) {
    \\    const int t = t0 + int(tid);
    \\    int src = -1;
    \\    if (t < nw) src = c0 + t;
    \\    else if (t < total) { const int c = topk[size_t(l) * K + t - nw]; if (c >= 0 && c < nc) src = -2 - c; }
    \\    rsrc[tid] = src;
    \\  }
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  bool any = false;
    \\  for (int j = 0; j < KT; ++j) any = any || rsrc[j] != -1;
    \\  if (!any) continue;
    \\  for (int i = int(tid); i < KT * (D / 8); i += NT) {
    \\    const int j = i / (D / 8), cc = i % (D / 8), src = rsrc[j];
    \\    metal::uint4 v = metal::uint4(0);
    \\    if (src >= 0) v = ((const device metal::uint4*)(keys + size_t(src) * D))[cc];
    \\    else if (src != -1) v = ((const device metal::uint4*)(comp + size_t(-2 - src) * D))[cc];
    \\    ((threadgroup metal::uint4*)kt)[i] = v;
    \\  }
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  metal::simdgroup_matrix<float, 8, 8> s[RT][KC];
    \\  #pragma clang loop unroll(full)
    \\  for (int r = 0; r < RT; ++r)
    \\    #pragma clang loop unroll(full)
    \\    for (int c = 0; c < KC; ++c) s[r][c] = metal::make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    \\  #pragma clang loop unroll(full)
    \\  for (int d = 0; d < DT; ++d) {
    \\    metal::simdgroup_matrix<T, 8, 8> b[KC];
    \\    #pragma clang loop unroll(full)
    \\    for (int c = 0; c < KC; ++c) metal::simdgroup_load(b[c], kt + c * 8 * D + d0 + d * 8, D, metal::ulong2(0, 0), true);
    \\    #pragma clang loop unroll(full)
    \\    for (int r = 0; r < RT; ++r)
    \\      #pragma clang loop unroll(full)
    \\      for (int c = 0; c < KC; ++c) metal::simdgroup_multiply_accumulate(s[r][c], qf[r][d], b[c], s[r][c]);
    \\  }
    \\  #pragma clang loop unroll(full)
    \\  for (int r = 0; r < RT; ++r)
    \\    #pragma clang loop unroll(full)
    \\    for (int c = 0; c < KC; ++c) metal::simdgroup_store(s[r][c], sp + (int(sg) * HB + r * 8) * KT + c * 8, KT);
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  float sv[NJ], tmax = -INFINITY;
    \\  for (int jj = 0; jj < NJ; ++jj) {
    \\    const int j = sub + jj * TPH;
    \\    float v = -INFINITY;
    \\    if (rsrc[j] != -1) {
    \\      float a = 0.0f;
    \\      for (int g = 0; g < NSG; ++g) a += sp[(g * HB + hh) * KT + j];
    \\      v = a * scale;
    \\    }
    \\    sv[jj] = v;
    \\    tmax = metal::max(tmax, v);
    \\  }
    \\  for (int off = TPH / 2; off > 0; off >>= 1) tmax = metal::max(tmax, metal::simd_shuffle_xor(tmax, ushort(off)));
    \\  const float mn = metal::max(m, tmax), corr = metal::precise::exp(m - mn);
    \\  float psum = 0.0f;
    \\  for (int jj = 0; jj < NJ; ++jj) {
    \\    const int j = sub + jj * TPH;
    \\    const float p = metal::precise::exp(sv[jj] - mn);
    \\    const T hi = T(p);
    \\    psum += p;
    \\    ph[hh * KT + j] = hi;
    \\    pl[hh * KT + j] = T(p - float(hi));
    \\  }
    \\  for (int off = TPH / 2; off > 0; off >>= 1) psum += metal::simd_shuffle_xor(psum, ushort(off));
    \\  lsum = lsum * corr + psum;
    \\  m = mn;
    \\  if (sub == 0) cr[hh] = corr;
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  #pragma clang loop unroll(full)
    \\  for (int r = 0; r < RT; ++r) {
    \\    const float c = cr[r * 8 + fm];
    \\    #pragma clang loop unroll(full)
    \\    for (int d = 0; d < DT; ++d) {
    \\      o[r][d].thread_elements()[0] *= c;
    \\      o[r][d].thread_elements()[1] *= c;
    \\    }
    \\  }
    \\  #pragma clang loop unroll(full)
    \\  for (int c = 0; c < KC; ++c) {
    \\    metal::simdgroup_matrix<T, 8, 8> a_hi[RT], a_lo[RT];
    \\    #pragma clang loop unroll(full)
    \\    for (int r = 0; r < RT; ++r) {
    \\      metal::simdgroup_load(a_hi[r], ph + r * 8 * KT + c * 8, KT);
    \\      metal::simdgroup_load(a_lo[r], pl + r * 8 * KT + c * 8, KT);
    \\    }
    \\    #pragma clang loop unroll(full)
    \\    for (int d = 0; d < DT; ++d) {
    \\      metal::simdgroup_matrix<T, 8, 8> v;
    \\      metal::simdgroup_load(v, kt + c * 8 * D + d0 + d * 8, D);
    \\      #pragma clang loop unroll(full)
    \\      for (int r = 0; r < RT; ++r) {
    \\        metal::simdgroup_multiply_accumulate(o[r][d], a_lo[r], v, o[r][d]);
    \\        metal::simdgroup_multiply_accumulate(o[r][d], a_hi[r], v, o[r][d]);
    \\      }
    \\    }
    \\  }
    \\}
    \\if (sub == 0) rl[hh] = 1.0f / lsum;
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\#pragma clang loop unroll(full)
    \\for (int r = 0; r < RT; ++r) {
    \\  const float inv = rl[r * 8 + fm];
    \\  device OT* dst = out + (size_t(l) * H + h0 + r * 8 + fm) * D + d0 + fn;
    \\  #pragma clang loop unroll(full)
    \\  for (int d = 0; d < DT; ++d) {
    \\    const auto e = o[r][d].thread_elements();
    \\    dst[d * 8] = OT(e[0] * inv);
    \\    dst[d * 8 + 1] = OT(e[1] * inv);
    \\  }
    \\}
;

var attn_fused_def = Def{ .name = "dsv41_attn_fused", .ins = &.{ "q", "keys", "comp", "topk", "sink", "meta", "scal" }, .outs = &.{"out"}, .source = ATTN_FUSED_SOURCE };

const fused_hb = 16;
const fused_nsg = 8;
const fused_kt = 16;

/// Whether `attnFused` serves `h` heads of `d` channels.
pub fn attnFusedFits(h: usize, d: usize) bool {
    return h % fused_hb == 0 and d % (fused_nsg * 8) == 0;
}

/// `attnWide` without the intermediates or the f16: `attnCore`'s heads `[L, H, D]` for `q` `[L, H, D]` (bf16)
/// against the window rows `keys_w` (query i sees rows (wp + i - window, wp + i]) and the compressed rows
/// `comp_rows` it picked (`topk` `[L, K]`), f32 throughout past the bf16 operands.
pub fn attnFused(sc: *Scope, sink: A, q: A, keys_w: A, wp: usize, window: usize, comp_rows: A, topk: A, scale: f32, out_dt: mlx.mlx_dtype) !A {
    const L = dim(q, 0);
    const H = dim(q, 1);
    const D = dim(q, 2);
    const dt = mlx.mlx_array_dtype(q);
    const K = if (comp_rows.ctx != null) dim(topk, 1) else 0;
    const meta_v = [_]i32{ ci(wp), ci(window), ci(K), if (K > 0) ci(dim(comp_rows, 0)) else 0 };
    const meta = try sc.ints(&meta_v, &.{meta_v.len});
    const keys = try sc.astype(keys_w, dt);
    const comp = if (K > 0) try sc.astype(comp_rows, dt) else keys;
    const picks = if (K > 0) try sc.astype(topk, .int32) else meta;
    var o: [1]A = undefined;
    try launch(sc, &attn_fused_def, &.{ q, keys, comp, picks, sink, meta, try f32Const(sc, &.{scale}) }, &.{.{ .shape = &.{ ci(L), ci(H), ci(D) }, .dt = out_dt }}, .{ H / fused_hb * fused_nsg * 32, L, 1 }, .{ fused_nsg * 32, 1, 1 }, &.{
        .{ .name = "D", .v = ci(D) }, .{ .name = "HB", .v = fused_hb }, .{ .name = "NSG", .v = fused_nsg }, .{ .name = "KT", .v = fused_kt },
    }, &.{ .{ .name = "T", .dt = dt }, .{ .name = "OT", .dt = out_dt } }, &o);
    return o[0];
}

// ── indexer ──────────────────────────────────────────────────────────────

/// The indexer's scores `[L, nb]`: sum over heads of weight * relu(q_h . k_n), a weight being its projection times
/// `wsc`, keys read in their cached dtype, -inf past the groups row l's query sees ((p0 + l + 1) / ratio).
/// meta = {p0, ratio}.
const IDX_SCORE_FIRST =
    \\const int n = thread_position_in_grid.x, l = thread_position_in_grid.y;
    \\const uint tid = thread_index_in_threadgroup, tgn = threads_per_threadgroup.x;
;

const IDX_SCORE_SECOND =
    \\threadgroup float qs[IH * IHD];
    \\for (uint k = tid; k < IH * IHD; k += tgn) qs[k] = float(q[l * IH * IHD + k]);
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\if (n >= nb) return;
    \\float dots[IH];
    \\for (int h = 0; h < IH; ++h) dots[h] = 0.0f;
;

const IDX_SCORE_LAST =
    \\for (int j = 0; j < IHD; ++j) {
    \\  const float kv = KEY(j);
    \\  for (int h = 0; h < IH; ++h) dots[h] += qs[h * IHD + j] * kv;
    \\}
    \\float s = 0.0f;
    \\for (int h = 0; h < IH; ++h) s += (float(w[l * IH + h]) * wsc[0]) * metal::max(dots[h], 0.0f);
    \\const int seen = (meta[0] + l + 1) / meta[1];
    \\scores[size_t(l) * nb + n] = n < seen ? s : -INFINITY;
;

const IDX_SCORE_SOURCE: [:0]const u8 = IDX_SCORE_FIRST ++
    \\const int nb = keys_shape[0];
    \\
++ IDX_SCORE_SECOND ++
    \\const device KT* kr = keys + size_t(n) * IHD;
    \\#define KEY(j) float(kr[j])
    \\
++ IDX_SCORE_LAST;

/// The same scores from packed index keys (`codes` `[nb, IHD / 2]`, `scales` `[nb, IHD / G]`, `qat_mod.packFp4`'s
/// form), each key value decoded as `qat_mod.unpackFp4` does, so no unpacked copy of the keys exists.
const IDX_SCORE_PACKED_SOURCE: [:0]const u8 = IDX_SCORE_FIRST ++
    \\const int nb = codes_shape[0];
    \\
++ IDX_SCORE_SECOND ++
    \\const auto kc = codes + size_t(n) * (IHD / 2);
    \\const auto ks = scales + size_t(n) * (IHD / G);
    \\#define KEY(j) dsv41_fp4_at(kc, ks, j, G, MODE)
    \\
++ IDX_SCORE_LAST;

const IDX_PACKED_HEADER: [:0]const u8 = qat_mod.PACK_HEADER ++
    \\template <typename C, typename S>
    \\inline float dsv41_fp4_at(C codes, S scales, int j, int G, int MODE) {
    \\  const uint byte = uint(codes[j / 2]);
    \\  const uint code = (j & 1) ? byte >> 4 : byte & 15u;
    \\  const float v = E2M1[code & 7u] * scale_of(uint(scales[j / G]), MODE);
    \\  return (code & 8u) ? -v : v;
    \\}
;

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

var idx_score_def = Def{ .name = "dsv41_index_scores", .ins = &.{ "q", "keys", "w", "meta", "wsc" }, .outs = &.{"scores"}, .source = IDX_SCORE_SOURCE };
var idx_packed_def = Def{ .name = "dsv41_index_scores_fp4", .ins = &.{ "q", "codes", "scales", "w", "meta", "wsc" }, .outs = &.{"scores"}, .source = IDX_SCORE_PACKED_SOURCE, .header = IDX_PACKED_HEADER };
var idx_tile_def = Def{ .name = "dsv41_index_tile", .ins = &.{ "q", "keys", "w", "meta", "wsc" }, .outs = &.{"scores"}, .source = IDX_TILE_SOURCE };

/// Keys one threadgroup of `indexScoresTile` scores for keys of `dt`: a slice scored at an offset must start on one.
pub fn indexKeyTile(dt: mlx.mlx_dtype) usize {
    return if (dt == .float32) 32 else 64;
}

/// Whether `indexScoresTile` serves `ih` heads of `ihd` channels (the matrix tiles need multiples of 8).
pub fn indexTileFits(ih: usize, ihd: usize, nb: usize) bool {
    return nb > 0 and ih % 8 == 0 and ih <= 128 and ihd % 8 == 0 and ihd % 4 == 0;
}

/// Scores `[L, nb]` of `q` `[L, IH, IHD]` (f32) against `keys` `[nb, IHD]`, one thread a key: the decode arm.
pub fn indexScores(sc: *Scope, q: A, keys: A, w: A, wscale: f32, p0: usize, ratio: usize) !A {
    const L = dim(q, 0);
    const nb = dim(keys, 0);
    const meta = try sc.ints(&.{ ci(p0), ci(ratio) }, &.{2});
    const tg: usize = 256;
    var o: [1]A = undefined;
    try launch(sc, &idx_score_def, &.{ q, keys, w, meta, try f32Const(sc, &.{wscale}) }, &.{.{ .shape = &.{ ci(L), ci(nb) }, .dt = .float32 }}, .{ (nb + tg - 1) / tg * tg, L, 1 }, .{ tg, 1, 1 }, &.{
        .{ .name = "IH", .v = ci(dim(q, 1)) }, .{ .name = "IHD", .v = ci(dim(q, 2)) },
    }, &.{.{ .name = "KT", .dt = mlx.mlx_array_dtype(keys) }}, &o);
    return o[0];
}

/// `indexScores` over packed keys (`codes`, `scales`: `packFp4`'s form of `mode` and `group`), the bits of the unpacked
/// keys' scores.
pub fn indexScoresPacked(sc: *Scope, q: A, codes: A, scales: A, w: A, wscale: f32, p0: usize, ratio: usize, mode: qat_mod.Qat, group: usize) !A {
    const L = dim(q, 0);
    const nb = dim(codes, 0);
    const meta = try sc.ints(&.{ ci(p0), ci(ratio) }, &.{2});
    const tg: usize = 256;
    var o: [1]A = undefined;
    try launch(sc, &idx_packed_def, &.{ q, codes, scales, w, meta, try f32Const(sc, &.{wscale}) }, &.{.{ .shape = &.{ ci(L), ci(nb) }, .dt = .float32 }}, .{ (nb + tg - 1) / tg * tg, L, 1 }, .{ tg, 1, 1 }, &.{
        .{ .name = "IH", .v = ci(dim(q, 1)) }, .{ .name = "IHD", .v = ci(dim(codes, 1) * 2) }, .{ .name = "G", .v = ci(group) }, .{ .name = "MODE", .v = qat_mod.packMode(mode) },
    }, &.{}, &o);
    return o[0];
}

/// `indexScores` for a wide block on 8x8 matrix ops; `q` is cast to the keys' dtype. `keys` are the list's rows from
/// `key0` on, a multiple of the kernel's key tile, so a slice scores the bits the whole list's columns would.
pub fn indexScoresTile(sc: *Scope, q: A, keys: A, w: A, wscale: f32, p0: usize, ratio: usize, key0: usize) !A {
    const L = dim(q, 0);
    const nb = dim(keys, 0);
    const dt = mlx.mlx_array_dtype(keys);
    const kt = indexKeyTile(dt);
    std.debug.assert(key0 % kt == 0);
    const meta = try sc.ints(&.{ ci(p0), ci(ratio), ci(key0) }, &.{3});
    var o: [1]A = undefined;
    try launch(sc, &idx_tile_def, &.{ try sc.astype(q, dt), keys, w, meta, try f32Const(sc, &.{wscale}) }, &.{.{ .shape = &.{ ci(L), ci(nb) }, .dt = .float32 }}, .{ (nb + kt - 1) / kt * 128, L, 1 }, .{ 128, 1, 1 }, &.{
        .{ .name = "IH", .v = ci(dim(q, 1)) }, .{ .name = "IHD", .v = ci(dim(q, 2)) }, .{ .name = "KT", .v = ci(kt) },
    }, &.{.{ .name = "T", .dt = dt }}, &o);
    return o[0];
}

// ── hyper-connections ────────────────────────────────────────────────────

/// A mix's index into the three scales (pre, post, comb) the checkpoint ships.
const HC_HEADER =
    \\template <int HC>
    \\inline int dsv41_scale_at(int m) { return m < HC ? 0 : (m < 2 * HC ? 1 : 2); }
;

/// `hcMixes`' projection: per row, the mixing logits of the flattened stream, rms-scaled, times scale plus base.
/// One threadgroup an R-row x M-mix tile, so each fn and stream load serves the tile; every (row, mix) dot keeps
/// `HC_STAGE_SOURCE`'s order (thread tid the terms k = tid + c * TG in c order, one simd sum, the simdgroups in
/// order), which is what makes the two agree to the bit.
const HC_MIX_SOURCE =
    \\#pragma clang fp contract(off)
    \\const int m0 = threadgroup_position_in_grid.x * M, l0 = threadgroup_position_in_grid.y * R;
    \\const uint tid = thread_index_in_threadgroup, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    \\const int N = fn_shape[1], MIX = fn_shape[0], L = x_shape[0];
    \\float d[R][M], ss[R];
    \\for (int r = 0; r < R; ++r) { ss[r] = 0.0f; for (int j = 0; j < M; ++j) d[r][j] = 0.0f; }
    \\for (int k = tid; k < N; k += TG) {
    \\  float fv[M];
    \\  #pragma clang loop unroll(full)
    \\  for (int j = 0; j < M; ++j) fv[j] = fn[size_t(m0 + j) * N + k];
    \\  #pragma clang loop unroll(full)
    \\  for (int r = 0; r < R; ++r) {
    \\    const float v = l0 + r < L ? float(x[size_t(l0 + r) * N + k]) : 0.0f;
    \\    #pragma clang loop unroll(full)
    \\    for (int j = 0; j < M; ++j) d[r][j] = metal::fma(v, fv[j], d[r][j]);
    \\    ss[r] = metal::fma(v, v, ss[r]);
    \\  }
    \\}
    \\threadgroup float t_d[R][M][TG / 32], t_s[R][TG / 32];
    \\for (int r = 0; r < R; ++r) {
    \\  for (int j = 0; j < M; ++j) { const float v = metal::simd_sum(d[r][j]); if (lane == 0) t_d[r][j][sg] = v; }
    \\  const float v = metal::simd_sum(ss[r]);
    \\  if (lane == 0) t_s[r][sg] = v;
    \\}
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\if (tid < R * M) {
    \\  const int r = tid / M, j = tid % M;
    \\  float a = 0.0f, b = 0.0f;
    \\  for (int g = 0; g < TG / 32; ++g) { a += t_d[r][j][g]; b += t_s[r][g]; }
    \\  if (l0 + r < L) z[(l0 + r) * MIX + m0 + j] = a * metal::precise::rsqrt(b / float(N) + eps[0]) * scale[dsv41_scale_at<HC>(m0 + j)] + base[m0 + j];
    \\}
;

/// One hyper-connection stage as two kernels. The staggered collapse reads the PREVIOUS mix (`cpre`), so in
/// `HC_STAGE_SOURCE` threadgroup 0 collapses and rms-norms the stream without waiting for this stage's mixes, while
/// threadgroups 1..MIX each take one mix's dot; the FFN side rebuilds the stream from the block output in every
/// threadgroup and threadgroup 0 stores it. The split (`HC_SPLIT_SOURCE`) is read only later.
const HC_STAGE_HEAD =
    \\#pragma clang fp contract(off)
    \\const uint role = threadgroup_position_in_grid.x;
    \\const int l = threadgroup_position_in_grid.y;
    \\const uint tid = thread_index_in_threadgroup, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    \\const int D = x_shape[2], N = fn_shape[1];
    \\threadgroup float t_a[TG / 32], t_b[TG / 32];
    \\const device auto* xr = x + size_t(l) * HC * D;
    \\float v[HC][ND];
    \\for (int c = 0; c < ND; ++c) {
    \\  const int d = int(tid) + c * TG;
    \\  for (int i = 0; i < HC; ++i) v[i][c] = 0.0f;
;

/// The stream is `x` itself.
const HC_STAGE_LOAD_PLAIN =
    \\  if (d < D) for (int i = 0; i < HC; ++i) v[i][c] = float(xr[i * D + d]);
    \\}
;

/// The stream is `hcExpand(x, y, previous split)`, rounded to T where the expand rounds it.
const HC_STAGE_LOAD_EXPAND =
    \\  if (d < D) {
    \\    const auto pr = spost + l * HC;
    \\    const auto cb = scomb + l * HC * HC;
    \\    float sv[HC];
    \\    for (int j = 0; j < HC; ++j) sv[j] = float(xr[j * D + d]);
    \\    const float yv = float(y[size_t(l) * D + d]);
    \\    for (int i = 0; i < HC; ++i) {
    \\      float acc = 0.0f;
    \\      for (int j = 0; j < HC; ++j) acc = metal::fma(cb[j * HC + i], sv[j], acc);
    \\      const T o = T(metal::fma(pr[i], yv, acc));
    \\      v[i][c] = float(o);
    \\      if (role == 0) x1[(size_t(l) * HC + i) * D + d] = o;
    \\    }
    \\  }
    \\}
;

const HC_STAGE_TAIL =
    \\if (role == 0) {
    \\  float cp[HC];
    \\  for (int j = 0; j < HC; ++j) cp[j] = cpre[l * HC + j];
    \\  float ss = 0.0f;
    \\  T hv[ND];
    \\  #pragma clang loop unroll(full)
    \\  for (int c = 0; c < ND; ++c) {
    \\    float h = 0.0f;
    \\    for (int j = 0; j < HC; ++j) h += cp[j] * v[j][c];
    \\    hv[c] = T(h);
    \\    ss += float(hv[c]) * float(hv[c]);
    \\  }
    \\  ss = metal::simd_sum(ss);
    \\  if (lane == 0) t_a[sg] = ss;
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  float tot = 0.0f;
    \\  for (int g = 0; g < TG / 32; ++g) tot += t_a[g];
    \\  const float inv = metal::precise::rsqrt(tot / float(D) + eps[0]);
    \\  #pragma clang loop unroll(full)
    \\  for (int c = 0; c < ND; ++c) {
    \\    const int d = int(tid) + c * TG;
    \\    if (d < D) h_out[size_t(l) * D + d] = w[d] * T(float(hv[c]) * inv);
    \\  }
    \\  return;
    \\}
    \\const int m = int(role) - 1;
    \\const device float* fr = fn + size_t(m) * N;
    \\float dd = 0.0f, sq = 0.0f;
    \\#pragma clang loop unroll(full)
    \\for (int i = 0; i < HC; ++i)
    \\  #pragma clang loop unroll(full)
    \\  for (int c = 0; c < ND; ++c) {
    \\    const int d = int(tid) + c * TG;
    \\    const float u = v[i][c];
    \\    if (d < D) dd = metal::fma(u, fr[i * D + d], dd);
    \\    sq = metal::fma(u, u, sq);
    \\  }
    \\dd = metal::simd_sum(dd); sq = metal::simd_sum(sq);
    \\if (lane == 0) { t_a[sg] = dd; t_b[sg] = sq; }
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\if (tid == 0) {
    \\  float a = 0.0f, b = 0.0f;
    \\  for (int g = 0; g < TG / 32; ++g) { a += t_a[g]; b += t_b[g]; }
    \\  z[l * ((2 + HC) * HC) + m] = a * metal::precise::rsqrt(b / float(N) + eps[0]) * scale[dsv41_scale_at<HC>(m)] + base[m];
    \\}
;

/// The hc split of the mixes `z` (pre = sigmoid + eps, post = 2 sigmoid, Sinkhorn comb); one simdgroup a row, lane
/// a * HC + b holding comb[a][b] and each sum gathering its HC terms in the serial order.
const HC_SPLIT_BODY =
    \\const uint lane = thread_index_in_simdgroup;
    \\constexpr int MIX = (2 + HC) * HC;
    \\const float e = eps[1];
    \\const device float* zr = z + l * MIX;
    \\if (lane < HC) {
    \\  pre[l * HC + lane] = 1.0f / (1.0f + metal::precise::exp(-zr[lane])) + e;
    \\  post[l * HC + lane] = 2.0f / (1.0f + metal::precise::exp(-zr[HC + lane]));
    \\}
    \\const bool act = lane < HC * HC;
    \\const ushort a = act ? lane / HC : 0, b = act ? lane % HC : 0;
    \\float v = zr[2 * HC + a * HC + b];
    \\float mx = metal::simd_shuffle(v, a * HC);
    \\for (ushort k = 1; k < HC; ++k) mx = metal::max(mx, metal::simd_shuffle(v, a * HC + k));
    \\v = metal::precise::exp(v - mx);
    \\float sm = 0.0f;
    \\for (ushort k = 0; k < HC; ++k) sm += metal::simd_shuffle(v, a * HC + k);
    \\v = v / sm + e;
    \\for (int t = 0; t < 2 * ITERS - 1; ++t) {
    \\  const bool rows = t % 2 == 1;
    \\  sm = 0.0f;
    \\  for (ushort k = 0; k < HC; ++k) sm += metal::simd_shuffle(v, rows ? a * HC + k : k * HC + b);
    \\  v = v / (sm + e);
    \\}
    \\if (act) comb[l * HC * HC + lane] = v;
;
const HC_SPLIT_SOURCE = "#pragma clang fp contract(off)\nconst int l = threadgroup_position_in_grid.y;\n" ++ HC_SPLIT_BODY;

/// The hc split, then the stream collapsed with `cpre` (f32, rounded to T like `hcCollapse`) and rms-normed like
/// MLX's `rms_norm`. One threadgroup a row.
const HC_NORM_SOURCE =
    \\#pragma clang fp contract(off)
    \\const int l = threadgroup_position_in_grid.y;
    \\const uint tid = thread_index_in_threadgroup, sg = simdgroup_index_in_threadgroup;
    \\const int D = x_shape[2];
    \\if (sg == 0) {
++ "\n" ++ HC_SPLIT_BODY ++
    \\}
    \\float cp[HC];
    \\for (int j = 0; j < HC; ++j) cp[j] = cpre[l * HC + j];
    \\float ss = 0.0f;
    \\for (int d = tid; d < D; d += TG) {
    \\  float h = 0.0f;
    \\  for (int j = 0; j < HC; ++j) h += cp[j] * float(x[(size_t(l) * HC + j) * D + d]);
    \\  const T hb = T(h);
    \\  h_out[size_t(l) * D + d] = hb;
    \\  ss += float(hb) * float(hb);
    \\}
    \\threadgroup float t_s[TG / 32];
    \\ss = metal::simd_sum(ss);
    \\if (thread_index_in_simdgroup == 0) t_s[sg] = ss;
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\float tot = 0.0f;
    \\for (int g = 0; g < TG / 32; ++g) tot += t_s[g];
    \\const float inv = metal::precise::rsqrt(tot / float(D) + eps[0]);
    \\for (int d = tid; d < D; d += TG) {
    \\  const size_t o = size_t(l) * D + d;
    \\  h_out[o] = w[d] * T(float(h_out[o]) * inv);
    \\}
;

const HC_STAGE_SOURCE = HC_STAGE_HEAD ++ HC_STAGE_LOAD_PLAIN ++ HC_STAGE_TAIL;
const HC_STAGE_X_SOURCE = HC_STAGE_HEAD ++ HC_STAGE_LOAD_EXPAND ++ HC_STAGE_TAIL;

var hc_mix_def = Def{ .name = "dsv41_hc_mix", .ins = &.{ "x", "fn", "scale", "base", "eps" }, .outs = &.{"z"}, .source = HC_MIX_SOURCE, .header = HC_HEADER };
var hc_stage_def = Def{ .name = "dsv41_hc_stage", .ins = &.{ "x", "cpre", "w", "fn", "scale", "base", "eps" }, .outs = &.{ "h_out", "z" }, .source = HC_STAGE_SOURCE, .header = HC_HEADER };
var hc_stage_x_def = Def{ .name = "dsv41_hc_stage_x", .ins = &.{ "x", "y", "spost", "scomb", "cpre", "w", "fn", "scale", "base", "eps" }, .outs = &.{ "h_out", "z", "x1" }, .source = HC_STAGE_X_SOURCE, .header = HC_HEADER };
var hc_split_def = Def{ .name = "dsv41_hc_split", .ins = &.{ "z", "eps" }, .outs = &.{ "pre", "post", "comb" }, .source = HC_SPLIT_SOURCE };
var hc_norm_def = Def{ .name = "dsv41_hc_norm", .ins = &.{ "z", "x", "cpre", "w", "eps" }, .outs = &.{ "pre", "post", "comb", "h_out" }, .source = HC_NORM_SOURCE };

/// Threads a hyper-connection threadgroup runs; `probeHc` halves it where the GPU caps register-heavy kernels lower.
pub var hc_tg: usize = 1024;
const hc_mix_rows = 4;
const hc_mix_cols = [_]usize{ 6, 4, 3, 2, 1 };

pub const HcWeights = struct { fn_w: A, scale: A, base: A };
pub const HcSplit = struct { pre: A, post: A, comb: A };
pub const HcStaged = struct { h: A, z: A, x1: A };
/// The stream `hcExpand(x, y, post, comb)` that a stage rebuilds in place of `x`.
pub const HcExpand = struct { y: A, post: A, comb: A };

pub fn hcEps(sc: *Scope, rms_eps: f32, hc_eps: f32) !A {
    return f32Const(sc, &.{ rms_eps, hc_eps });
}

/// A decode-width stage: `h = rms(collapse(x, cpre))` and the mixes `z` `[L, mix]` of `x` (of the expanded stream
/// with `ex`, returned as `x1`).
pub fn hcStage(sc: *Scope, x: A, ex: ?HcExpand, cpre: A, norm_w: A, hw: HcWeights, eps: A) !HcStaged {
    const L = dim(x, 0);
    const hc = dim(x, 1);
    const D = dim(x, 2);
    const mix = dim(hw.fn_w, 0);
    const dt = mlx.mlx_array_dtype(x);
    const tg = hc_tg;
    var o: [3]A = undefined;
    const outs3 = [_]OutSpec{
        .{ .shape = &.{ ci(L), ci(D) }, .dt = dt },
        .{ .shape = &.{ ci(L), ci(mix) }, .dt = .float32 },
        .{ .shape = mlx.getShape(x), .dt = dt },
    };
    const ints = [_]Tmpl{ .{ .name = "TG", .v = ci(tg) }, .{ .name = "HC", .v = ci(hc) }, .{ .name = "ND", .v = ci((D + tg - 1) / tg) } };
    const dts = [_]TDtype{.{ .name = "T", .dt = dt }};
    if (ex) |e| {
        try launch(sc, &hc_stage_x_def, &.{ x, try sc.astype(e.y, dt), e.post, e.comb, cpre, norm_w, hw.fn_w, hw.scale, hw.base, eps }, &outs3, .{ (mix + 1) * tg, L, 1 }, .{ tg, 1, 1 }, &ints, &dts, &o);
        return .{ .h = o[0], .z = o[1], .x1 = o[2] };
    }
    try launch(sc, &hc_stage_def, &.{ x, cpre, norm_w, hw.fn_w, hw.scale, hw.base, eps }, outs3[0..2], .{ (mix + 1) * tg, L, 1 }, .{ tg, 1, 1 }, &ints, &dts, o[0..2]);
    return .{ .h = o[0], .z = o[1], .x1 = x };
}

/// The split of mixes `z` `[L, (2 + hc) * hc]` (scale and base already applied).
pub fn hcSplitZ(sc: *Scope, z: A, hc: usize, iters: usize, eps: A) !HcSplit {
    const L = dim(z, 0);
    var o: [3]A = undefined;
    try launch(sc, &hc_split_def, &.{ z, eps }, &.{
        .{ .shape = &.{ ci(L), ci(hc) }, .dt = .float32 },
        .{ .shape = &.{ ci(L), ci(hc) }, .dt = .float32 },
        .{ .shape = &.{ ci(L), ci(hc), ci(hc) }, .dt = .float32 },
    }, .{ 32, L, 1 }, .{ 32, 1, 1 }, &.{ .{ .name = "HC", .v = ci(hc) }, .{ .name = "ITERS", .v = ci(iters) } }, &.{}, &o);
    return .{ .pre = o[0], .post = o[1], .comb = o[2] };
}

/// The mixes `z` `[L, mix]` of the stream `x` `[L, hc, D]`, a tile of rows and mixes a threadgroup (wide blocks).
pub fn hcMix(sc: *Scope, x: A, hw: HcWeights, eps: A) !A {
    const L = dim(x, 0);
    const hc = dim(x, 1);
    const mix = dim(hw.fn_w, 0);
    const cols = for (hc_mix_cols) |c| {
        if (mix % c == 0) break c;
    } else unreachable;
    var o: [1]A = undefined;
    try launch(sc, &hc_mix_def, &.{ x, hw.fn_w, hw.scale, hw.base, eps }, &.{.{ .shape = &.{ ci(L), ci(mix) }, .dt = .float32 }}, .{ mix / cols * hc_tg, (L + hc_mix_rows - 1) / hc_mix_rows, 1 }, .{ hc_tg, 1, 1 }, &.{
        .{ .name = "TG", .v = ci(hc_tg) }, .{ .name = "R", .v = hc_mix_rows }, .{ .name = "M", .v = ci(cols) }, .{ .name = "HC", .v = ci(hc) },
    }, &.{}, &o);
    return o[0];
}

/// The split of `z` and `h = rms(collapse(x, cpre))`, one threadgroup a row.
pub fn hcNorm(sc: *Scope, z: A, x: A, cpre: A, norm_w: A, iters: usize, eps: A) !struct { sp: HcSplit, h: A } {
    const L = dim(x, 0);
    const hc = dim(x, 1);
    const D = dim(x, 2);
    const dt = mlx.mlx_array_dtype(x);
    var o: [4]A = undefined;
    try launch(sc, &hc_norm_def, &.{ z, x, cpre, norm_w, eps }, &.{
        .{ .shape = &.{ ci(L), ci(hc) }, .dt = .float32 },
        .{ .shape = &.{ ci(L), ci(hc) }, .dt = .float32 },
        .{ .shape = &.{ ci(L), ci(hc), ci(hc) }, .dt = .float32 },
        .{ .shape = &.{ ci(L), ci(D) }, .dt = dt },
    }, .{ hc_tg, L, 1 }, .{ hc_tg, 1, 1 }, &.{ .{ .name = "TG", .v = ci(hc_tg) }, .{ .name = "HC", .v = ci(hc) }, .{ .name = "ITERS", .v = ci(iters) } }, &.{.{ .name = "T", .dt = dt }}, &o);
    return .{ .sp = .{ .pre = o[0], .post = o[1], .comb = o[2] }, .h = o[3] };
}

/// Metal caps a register-heavy kernel's threadgroup below 1024 on some GPUs (M1, M2): launches every hyper-connection
/// kernel once at `hc_tg` threads and halves it until they run.
pub fn probeHc(gpa: std.mem.Allocator, s: mlx.mlx_stream, hc: usize, D: usize, iters: usize) !void {
    while (true) {
        probeHcOnce(gpa, s, hc, D, iters) catch |e| {
            if (e != error.MlxError or hc_tg <= 32 or !mlx.takeErrorIf("Thread group size")) return e;
            hc_tg /= 2;
            continue;
        };
        return;
    }
}

fn probeHcOnce(gpa: std.mem.Allocator, s: mlx.mlx_stream, hc: usize, D: usize, iters: usize) !void {
    var sc = Scope.init(gpa, s);
    defer sc.deinit();
    const dt = mlx.mlx_dtype.bfloat16;
    const mix = (2 + hc) * hc;
    const x = try sc.zeros(&.{ 1, ci(hc), ci(D) }, dt);
    const hw = HcWeights{ .fn_w = try sc.zeros(&.{ ci(mix), ci(hc * D) }, .float32), .scale = try sc.zeros(&.{3}, .float32), .base = try sc.zeros(&.{ci(mix)}, .float32) };
    const cpre = try sc.zeros(&.{ 1, ci(hc) }, .float32);
    const w = try sc.zeros(&.{ci(D)}, dt);
    const eps = try hcEps(&sc, 1e-6, 1e-6);
    const plain = try hcStage(&sc, x, null, cpre, w, hw, eps);
    const ex = HcExpand{ .y = try sc.zeros(&.{ 1, ci(D) }, dt), .post = try sc.zeros(&.{ 1, ci(hc) }, .float32), .comb = try sc.zeros(&.{ 1, ci(hc), ci(hc) }, .float32) };
    const expanded = try hcStage(&sc, x, ex, cpre, w, hw, eps);
    const wide_z = try hcMix(&sc, x, hw, eps);
    const wide = try hcNorm(&sc, wide_z, x, cpre, w, iters, eps);
    const sp = try hcSplitZ(&sc, plain.z, hc, iters, eps);
    try sc_mod.evalAll(&.{ plain.h, expanded.h, expanded.x1, wide.h, wide.sp.comb, sp.comb });
}

// ── Engram ───────────────────────────────────────────────────────────────

/// The threads MLX's sum gives a contiguous row of `n` floats (`threadgroup_size_from_row_size`); at 64 or fewer one
/// thread adds the row in order (`row_reduce_small`).
fn mlxRowReduceThreads(n: usize) usize {
    if (n <= 64) return 1;
    if (n <= 512) return 32;
    if (n <= 1024) return 128;
    return @min(1024, ((n + 3) / 4 + 31) / 32 * 32);
}

/// `engramGateOps` for one (row l, copy j) of the stream `x` [L, hc, D], `kv` the `wkv` rows [L, (hc + 1) * D]: the
/// three row sums in MLX's order (`mlxRowReduceThreads`: each thread its 4-wide blocks then its tail, simd sums, then
/// the simdgroup partials), then every op of the chain in its order, uncontracted, so the bits are the chain's.
/// cst = {1 / D, eps, 1 / sqrt(D)}.
const ENGRAM_SOURCE =
    \\#pragma clang fp contract(off)
    \\const uint t = thread_position_in_threadgroup.x, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    \\const int TG = threads_per_threadgroup.x, j = threadgroup_position_in_grid.y, l = threadgroup_position_in_grid.z;
    \\const int HC = qk_shape[0], D = qk_shape[1];
    \\const size_t xo = (size_t(l) * HC + j) * D, ko = size_t(l) * (HC + 1) * D + size_t(j) * D, vo = size_t(l) * (HC + 1) * D + size_t(HC) * D;
    \\threadgroup float part[3][32];
    \\threadgroup float gate_s;
    \\float sh = 0.0f, sk = 0.0f, sd = 0.0f;
    \\const int blocks = D / (TG * 4), extra = D - blocks * TG * 4;
    \\for (int b = 0; b <= blocks; ++b) {
    \\  const int c0 = b * TG * 4 + int(t) * 4;
    \\  const int n = b < blocks ? 4 : metal::clamp(extra - int(t) * 4, 0, 4);
    \\  for (int i = 0; i < n; ++i) {
    \\    const float hv = float(x[xo + c0 + i]), kk = float(kv[ko + c0 + i]);
    \\    const float h2 = hv * hv, k2 = kk * kk, hq = hv * qk[size_t(j) * D + c0 + i];
    \\    const float p = hq * kk;
    \\    sh = h2 + sh;
    \\    sk = k2 + sk;
    \\    sd = p + sd;
    \\  }
    \\}
    \\if (TG > 1) {
    \\  sh = metal::simd_sum(sh);
    \\  sk = metal::simd_sum(sk);
    \\  sd = metal::simd_sum(sd);
    \\}
    \\if (TG > 32) {
    \\  if (lane == 0) { part[0][sg] = sh; part[1][sg] = sk; part[2][sg] = sd; }
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  const bool own = int(t) < TG / 32;
    \\  sh = metal::simd_sum(own ? part[0][t] : 0.0f);
    \\  sk = metal::simd_sum(own ? part[1][t] : 0.0f);
    \\  sd = metal::simd_sum(own ? part[2][t] : 0.0f);
    \\}
    \\if (t == 0) {
    \\  const float mh = sh * cst[0], mk = sk * cst[0];
    \\  const float eh = mh + cst[1], ek = mk + cst[1];
    \\  const float rh = metal::precise::rsqrt(eh), rk = metal::precise::rsqrt(ek);
    \\  const float rr = rh * rk;
    \\  const float d1 = sd * rr;
    \\  const float dot = d1 * cst[2];
    \\  const float ad = metal::abs(dot);
    \\  const float mg = metal::precise::sqrt(metal::isnan(ad) || ad > 1e-6f ? ad : 1e-6f);
    \\  const float s = dot < 0.0f ? -mg : mg;
    \\  const float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(s)));
    \\  gate_s = s < 0.0f ? y : 1.0f - y;
    \\}
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\const float g = gate_s;
    \\for (int c = t; c < D; c += TG) {
    \\  const float gv = g * float(kv[vo + c]);
    \\  out[xo + c] = OT(float(x[xo + c]) + gv);
    \\}
;

var engram_def = Def{ .name = "dsv41_engram", .ins = &.{ "x", "kv", "qk", "cst" }, .outs = &.{"out"}, .source = ENGRAM_SOURCE };

/// The Engram gate on the stream `x` `[L, hc, D]` from the `wkv` rows `kv` and `qk` `[hc, D]` (f32): the stream plus
/// the gated shared value, in `x`'s dtype.
pub fn engramGate(sc: *Scope, x: A, kv: A, qk: A, eps: f32) !A {
    const D = dim(x, 2);
    const tg = mlxRowReduceThreads(D);
    const dt = mlx.mlx_array_dtype(x);
    const cst = try f32Const(sc, &.{ @floatCast(1.0 / @as(f64, @floatFromInt(D))), eps, 1.0 / @sqrt(@as(f32, @floatFromInt(D))) });
    var o: [1]A = undefined;
    try launch(sc, &engram_def, &.{ x, kv, qk, cst }, &.{.{ .shape = mlx.getShape(x), .dt = dt }}, .{ tg, dim(x, 1), dim(x, 0) }, .{ tg, 1, 1 }, &.{}, &.{.{ .name = "OT", .dt = dt }}, &o);
    return o[0];
}

// ── tests ────────────────────────────────────────────────────────────────

const testing = std.testing;
const model = @import("dsv41_model.zig");
const Config = @import("dsv41_config.zig").Config;

fn randn(sc: *Scope, shape: []const c_int, dt: mlx.mlx_dtype, seed: u64, std_dev: f32) !A {
    var key = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_random_key(&key, seed));
    _ = try sc.keep(key);
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_random_normal(&o, shape.ptr, shape.len, dt, 0, std_dev, key, sc.s), &o);
}

fn quantize(sc: *Scope, w: A, bits: c_int, mode: [*:0]const u8) !Quant {
    var vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_quantize(&vec, w, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(bits), mode, sc_mod.none, sc.s));
    var q = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&q, vec, 0));
    _ = try sc.keep(q);
    var s = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&s, vec, 1));
    _ = try sc.keep(s);
    return .{ .w = q, .s = s };
}

/// The f32 values a quantized array stands for, exactly (e2m1/e4m3 times a power of two).
fn dequantHost(sc: *Scope, q: Quant, bits: c_int, mode: [*:0]const u8) ![]const f32 {
    var o = mlx.mlx_array_new();
    const d = try sc.res(mlx.mlx_dequantize(&o, q.w, q.s, sc_mod.none, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(bits), mode, sc_mod.none, .{ .value = .float32, .has_value = true }, sc.s), &o);
    return sc_mod.hostF32(d);
}

fn hostOf(sc: *Scope, a: A) ![]const f32 {
    return sc_mod.hostF32(try sc.astype(a, .float32));
}

/// Max |got - want| / peak(|want|), the error measure of every kernel test below.
fn relErr(got: []const f32, want: []const f64) f64 {
    var peak: f64 = 1e-30;
    for (want) |v| peak = @max(peak, @abs(v));
    var worst: f64 = 0;
    for (got, want) |g, w| {
        const e = @abs(@as(f64, g) - w);
        if (std.math.isNan(e)) return std.math.inf(f64);
        worst = @max(worst, e);
    }
    return worst / peak;
}

fn expectLe(label: []const u8, got: f64, bound: f64) !void {
    if (got <= bound) return;
    std.debug.print("{s}: {e} > {e}\n", .{ label, got, bound });
    return error.KernelOutsideBar;
}

fn relErrF(got: []const f32, want: []const f32) f64 {
    var peak: f64 = 1e-30;
    for (want) |v| peak = @max(peak, @abs(v));
    var worst: f64 = 0;
    for (got, want) |g, w| {
        const e = @abs(@as(f64, g) - @as(f64, w));
        if (std.math.isNan(e)) return std.math.inf(f64);
        worst = @max(worst, e);
    }
    return worst / peak;
}

const FfnCase = struct {
    d: usize = 5120,
    inter: usize = 2304,
    experts: usize = 12,
    topk: usize = 6,
    hc: usize = 4,
    limit: f32 = 10,
};

const FfnData = struct {
    x: A,
    res: A,
    post: A,
    comb: A,
    idx: A,
    wts: A,
    e: Experts,
    sh: SharedLins,
    c: FfnCase,
};

fn ffnData(sc: *Scope, c: FfnCase, dt: mlx.mlx_dtype) !FfnData {
    const d: c_int = ci(c.d);
    const n: c_int = ci(c.inter);
    const E: c_int = ci(c.experts);
    const wstd: f32 = 0.05;
    const f32t = mlx.mlx_dtype.float32;
    const e = Experts{
        .gate = try quantize(sc, try randn(sc, &.{ E, n, d }, f32t, 11, wstd), 4, "mxfp4"),
        .up = try quantize(sc, try randn(sc, &.{ E, n, d }, f32t, 12, wstd), 4, "mxfp4"),
        .down = try quantize(sc, try randn(sc, &.{ E, d, n }, f32t, 13, wstd), 4, "mxfp4"),
    };
    const sh = SharedLins{
        .gate = try quantize(sc, try randn(sc, &.{ n, d }, f32t, 14, wstd), 8, "mxfp8"),
        .up = try quantize(sc, try randn(sc, &.{ n, d }, f32t, 15, wstd), 8, "mxfp8"),
        .down = try quantize(sc, try randn(sc, &.{ d, n }, f32t, 16, wstd), 8, "mxfp8"),
    };
    var picks: [8]i32 = undefined;
    var wv: [8]f32 = undefined;
    for (0..c.topk) |i| {
        picks[i] = @intCast((i * 5 + 3) % c.experts);
        wv[i] = 0.4 + 0.15 * @as(f32, @floatFromInt(i));
    }
    const hc: c_int = ci(c.hc);
    return .{
        .x = try sc.astype(try randn(sc, &.{ 1, d }, f32t, 17, 2.0), dt),
        .res = try sc.astype(try randn(sc, &.{ 1, hc, d }, f32t, 18, 1.5), dt),
        .post = try randn(sc, &.{ 1, hc }, f32t, 19, 0.7),
        .comb = try randn(sc, &.{ 1, hc, hc }, f32t, 20, 0.5),
        .idx = try sc.ints(picks[0..c.topk], &.{ 1, ci(c.topk) }),
        .wts = try sc.floats(wv[0..c.topk], &.{ 1, ci(c.topk) }),
        .e = e,
        .sh = sh,
        .c = c,
    };
}

/// The composed chain `dsv41_model.moe` and `block` run around one token's FFN: `routedSum`, the shared
/// expert, and the hyper-connection expand.
fn composedFfn(sc: *Scope, f: *const FfnData, dt: mlx.mlx_dtype) !A {
    var cfg = Config{};
    cfg.swiglu_limit = f.c.limit;
    const banks = model.BankSet{
        .gate = .{ .w = f.e.gate.w, .s = f.e.gate.s },
        .up = .{ .w = f.e.up.w, .s = f.e.up.s },
        .down = .{ .w = f.e.down.w, .s = f.e.down.s },
    };
    const routed = try model.routedSum(sc, &cfg, dt, f.x, banks, f.idx, f.wts);
    const g = try (model.Lin{ .w = f.sh.gate.w, .s = f.sh.gate.s }).apply(sc, f.x);
    const u = try (model.Lin{ .w = f.sh.up.w, .s = f.sh.up.s }).apply(sc, f.x);
    const sy = try (model.Lin{ .w = f.sh.down.w, .s = f.sh.down.s }).apply(sc, try sc.astype(try model.clampedSwiglu(sc, g, u, f.c.limit), dt));
    const y = try sc.astype(try sc.add(routed, try sc.astype(sy, .float32)), dt);
    return model.hcExpand(sc, f.res, y, f.post, f.comb);
}

/// The FFN in f64 from the exact dequantized weights and the same inputs, no rounding between stages.
fn truthFfn(sc: *Scope, f: *const FfnData, out: []f64) !void {
    const gpa = testing.allocator;
    const d = f.c.d;
    const n = f.c.inter;
    const hc = f.c.hc;
    const gw = try dequantHost(sc, f.e.gate, 4, "mxfp4");
    const uw = try dequantHost(sc, f.e.up, 4, "mxfp4");
    const dw = try dequantHost(sc, f.e.down, 4, "mxfp4");
    const sg = try dequantHost(sc, f.sh.gate, 8, "mxfp8");
    const su = try dequantHost(sc, f.sh.up, 8, "mxfp8");
    const sd = try dequantHost(sc, f.sh.down, 8, "mxfp8");
    const x = try hostOf(sc, f.x);
    const picks = try sc_mod.hostI32(f.idx);
    const wts = try sc_mod.hostF32(f.wts);
    const post = try sc_mod.hostF32(f.post);
    const comb = try sc_mod.hostF32(f.comb);
    const res = try hostOf(sc, f.res);
    const acts = try gpa.alloc(f64, (f.c.topk + 1) * n);
    defer gpa.free(acts);
    const swiglu = struct {
        fn at(g_: f64, u_: f64, lim: f64) f64 {
            const g = @min(g_, lim);
            const u = std.math.clamp(u_, -lim, lim);
            return g / (1.0 + @exp(-g)) * u;
        }
    }.at;
    for (0..f.c.topk + 1) |e| {
        const routed = e < f.c.topk;
        for (0..n) |r| {
            var g: f64 = 0;
            var u: f64 = 0;
            const base = if (routed) (@as(usize, @intCast(picks[e])) * n + r) * d else r * d;
            const gp = if (routed) gw else sg;
            const up = if (routed) uw else su;
            for (0..d) |k| {
                g += @as(f64, x[k]) * gp[base + k];
                u += @as(f64, x[k]) * up[base + k];
            }
            acts[e * n + r] = swiglu(g, u, f.c.limit) * (if (routed) @as(f64, wts[e]) else 1.0);
        }
    }
    const y = try gpa.alloc(f64, d);
    defer gpa.free(y);
    for (0..d) |r| {
        var s: f64 = 0;
        for (0..f.c.topk + 1) |e| {
            const routed = e < f.c.topk;
            const base = if (routed) (@as(usize, @intCast(picks[e])) * d + r) * n else r * n;
            const wp = if (routed) dw else sd;
            for (0..n) |k| s += acts[e * n + k] * wp[base + k];
        }
        y[r] = s;
    }
    for (0..hc) |i| for (0..d) |c| {
        var acc: f64 = @as(f64, post[i]) * y[c];
        for (0..hc) |j| acc += @as(f64, comb[j * hc + i]) * res[j * d + c];
        out[i * d + c] = acc;
    };
}

fn kernelFfn(sc: *Scope, f: *const FfnData) !A {
    const inds = try sc.reshape(f.idx, &.{ci(f.c.topk)});
    const wts = try sc.reshape(f.wts, &.{ci(f.c.topk)});
    const act = try moeGateUp(sc, f.x, f.e, f.sh, inds, wts, f.c.limit);
    return moeDownExpand(sc, act, f.e, f.sh, inds, f.res, try sc.reshape(f.post, &.{ci(f.c.hc)}), try sc.reshape(f.comb, &.{ ci(f.c.hc), ci(f.c.hc) }));
}

test "dsv41 moe decode: the gate-up and down-expand kernels are no further from the f64 truth than the composed chain, f32 sums in another order" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var sc = Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    const f = try ffnData(&sc, .{}, .float32);
    const truth = try testing.allocator.alloc(f64, f.c.hc * f.c.d);
    defer testing.allocator.free(truth);
    try truthFfn(&sc, &f, truth);
    const want = try composedFfn(&sc, &f, .float32);
    const got = try kernelFfn(&sc, &f);
    const err_kernel = relErr(try hostOf(&sc, got), truth);
    const err_composed = relErr(try hostOf(&sc, want), truth);
    try expectLe("kernel vs truth", err_kernel, 1e-5);
    try expectLe("kernel vs composed's error", err_kernel, err_composed);
}

test "dsv41 moe decode: in bf16 the kernels give the composed chain's stream to a few bf16 ulps" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var sc = Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    const f = try ffnData(&sc, .{}, .bfloat16);
    const want = try hostOf(&sc, try composedFfn(&sc, &f, .bfloat16));
    const got = try hostOf(&sc, try kernelFfn(&sc, &f));
    try expectLe("kernel vs composed", relErrF(got, want), 1.0 / 128.0);
}

test "dsv41 gate: the router logits are no further from the f64 truth than the f32 matmul" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var sc = Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    const E = 384;
    const D = 5120;
    const w = try randn(&sc, &.{ E, D }, .float32, 31, 0.05);
    for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dt| {
        const x = try sc.astype(try randn(&sc, &.{ 3, D }, .float32, 32, 3.0), dt);
        const got = try hostOf(&sc, try gateLogits(&sc, x, w));
        const comp = try hostOf(&sc, try sc.matmul(try sc.astype(x, .float32), try sc.t2(w)));
        const xh = try hostOf(&sc, x);
        const wh = try sc_mod.hostF32(w);
        const truth = try testing.allocator.alloc(f64, 3 * E);
        defer testing.allocator.free(truth);
        for (0..3) |l| for (0..E) |e| {
            var s: f64 = 0;
            for (0..D) |k| s += @as(f64, xh[l * D + k]) * wh[e * D + k];
            truth[l * E + e] = s;
        };
        try expectLe("kernel vs truth", relErr(got, truth), @max(relErr(comp, truth), 2e-6));
    }
}

fn composedRoute(sc: *Scope, logits: A, bias: A, k: usize, norm: bool, scale: f32) !Routes {
    const scores = try sc.op1(mlx.mlx_sqrt, try sc.op2(mlx.mlx_logaddexp, logits, try sc.f(0)));
    const idx = try sc.astype(try sc.topkIdx(try sc.add(scores, bias), k), .int32);
    var wts = try sc.takeAlong(scores, idx, -1);
    if (norm and k > 1) wts = try sc.div(wts, try sc.add(try sc.sum(wts, -1, true), try sc.f(1e-20)));
    return .{ .idx = idx, .wts = try sc.mul(wts, try sc.f(scale)) };
}

test "dsv41 route: the experts are the composed chain's, their weights its scores to f32 rounding, with and without normalizing" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var sc = Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    const E = 384;
    const K = 6;
    const L = 5;
    const logits = try randn(&sc, &.{ L, E }, .float32, 41, 2.5);
    const bias = try randn(&sc, &.{ 1, E }, .float32, 42, 0.3);
    for ([_]bool{ true, false }) |norm| {
        const want = try composedRoute(&sc, logits, bias, K, norm, 1.5);
        const got = try route(&sc, logits, bias, K, norm, 1.5);
        const gi = try sc_mod.hostI32(try sc.astype(got.idx, .int32));
        const wi = try sc_mod.hostI32(want.idx);
        const gw = try sc_mod.hostF32(got.wts);
        const ww = try sc_mod.hostF32(want.wts);
        for (0..L) |l| for (0..K) |r| {
            const e = gi[l * K + r];
            const at = for (0..K) |q| {
                if (wi[l * K + q] == e) break q;
            } else return error.RouteExpertMissing;
            try testing.expectApproxEqRel(ww[l * K + at], gw[l * K + r], 2e-6);
        };
    }
}

test "dsv41 topk: the picks are the k largest in index order, ties to the lower index" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var sc = Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    const n = 1500;
    const k = 512;
    const x = try randn(&sc, &.{ 3, n }, .float32, 51, 1.0);
    const got = try sc_mod.hostI32(try topkIndices(&sc, x, k));
    const want = try sc_mod.hostI32(try sc.astype(try sc.topkIdx(x, k), .int32));
    for (0..3) |l| {
        const w = try testing.allocator.dupe(i32, want[l * k ..][0..k]);
        defer testing.allocator.free(w);
        std.mem.sort(i32, w, {}, std.sort.asc(i32));
        try testing.expectEqualSlices(i32, w, got[l * k ..][0..k]);
    }
    // Every score equal but four large ones and a -inf: the large ones, then the lowest indices.
    var tie: [n]f32 = @splat(1.0);
    for (n - 4..n) |i| tie[i] = 9.0;
    tie[7] = -std.math.inf(f32);
    const t = try sc_mod.hostI32(try topkIndices(&sc, try sc.floats(&tie, &.{ 1, n }), 10));
    try testing.expectEqualSlices(i32, &.{ 0, 1, 2, 3, 4, 5, n - 4, n - 3, n - 2, n - 1 }, t[0..10]);
}

test "dsv41 dense f32x: the product is no further from the f64 truth than the widened f32 matmul" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var sc = Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    const N = 1024;
    const K = 5120;
    const x = try randn(&sc, &.{ 2, K }, .float32, 61, 2.0);
    for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |wdt| {
        const w = try sc.astype(try randn(&sc, &.{ N, K }, .float32, 62, 0.04), wdt);
        const got = try hostOf(&sc, try denseF32x(&sc, x, w));
        const comp = try hostOf(&sc, try sc.matmul(x, try sc.t2(w)));
        const xh = try sc_mod.hostF32(x);
        const wh = try hostOf(&sc, w);
        const truth = try testing.allocator.alloc(f64, 2 * N);
        defer testing.allocator.free(truth);
        for (0..2) |l| for (0..N) |n| {
            var s: f64 = 0;
            for (0..K) |k| s += @as(f64, xh[l * K + k]) * wh[n * K + k];
            truth[l * N + n] = s;
        };
        try expectLe("kernel vs truth", relErr(got, truth), @max(relErr(comp, truth), 2e-6));
    }
}

fn expectBitEqual(label: []const u8, got: []const f32, want: []const f32) !void {
    try testing.expectEqual(want.len, got.len);
    for (want, got, 0..) |w, g, i| {
        if (@as(u32, @bitCast(w)) != @as(u32, @bitCast(g))) {
            std.debug.print("{s}[{d}]: want {e} got {e}\n", .{ label, i, w, g });
            return error.NotBitIdentical;
        }
    }
}

test "dsv41 moe act: the clamped SwiGLU times the routing weight is the composed chain's, bit for bit" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var sc = Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    const n = 61;
    const N = 2304;
    for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dt| {
        const g = try sc.astype(try randn(&sc, &.{ n, 1, N }, .float32, 71, 6.0), dt);
        const u = try sc.astype(try randn(&sc, &.{ n, 1, N }, .float32, 72, 6.0), dt);
        const w = try randn(&sc, &.{n}, .float32, 73, 0.5);
        const want = try sc.astype(try sc.mul(try model.clampedSwiglu(&sc, g, u, 10), try sc.reshape(w, &.{ n, 1, 1 })), dt);
        const got = try moeAct(&sc, g, u, w, 10);
        try expectBitEqual("moe act", try hostOf(&sc, got), try hostOf(&sc, want));
    }
}

test "dsv41 moe combine: unsort, the f32 sum over the slots and the shared add are the composed chain's, bit for bit" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var sc = Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    const L = 37;
    const k = 6;
    const D = 5120;
    for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dt| {
        var key = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_random_key(&key, 81));
        _ = try sc.keep(key);
        var picks = mlx.mlx_array_new();
        const lo = try sc.i(0);
        const hi = try sc.i(40);
        const flat = try sc.res(mlx.mlx_random_randint(&picks, lo, hi, &[_]c_int{L * k}, 1, .uint32, key, sc.s), &picks);
        var oo = mlx.mlx_array_new();
        const order = try sc.res(mlx.mlx_argsort_axis(&oo, flat, 0, sc.s), &oo);
        var oi = mlx.mlx_array_new();
        const inv = try sc.res(mlx.mlx_argsort_axis(&oi, order, 0, sc.s), &oi);
        const dn = try sc.astype(try randn(&sc, &.{ L * k, D }, .float32, 82, 1.0), dt);
        const sh = try sc.astype(try randn(&sc, &.{ L, D }, .float32, 83, 1.0), dt);
        const rows = try sc.reshape(try sc.take(dn, inv, 0), &.{ L, k, D });
        const want = try sc.astype(try sc.add(try sc.sum(try sc.astype(rows, .float32), 1, false), try sc.astype(sh, .float32)), dt);
        const got = try moeCombine(&sc, dn, inv, sh, k);
        try expectBitEqual("moe combine", try hostOf(&sc, got), try hostOf(&sc, want));
    }
}

fn ropeFreqs(sc: *Scope) !A {
    var f: [32]f32 = undefined;
    qat_mod.invFreqs(&f, 64, 65536, 160000, 16, 32, 1);
    return sc.floats(&f, &.{32});
}

fn positions(sc: *Scope, first: i32, n: usize, step: i32) !A {
    var p: [64]i32 = undefined;
    for (0..n) |i| p[i] = first + @as(i32, @intCast(i)) * step;
    return sc.ints(p[0..n], &.{ci(n)});
}

test "dsv41 rope tail: the rotation and the activation round trip are the composed chain's, bit for bit" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var sc = Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    const freqs = try ropeFreqs(&sc);
    const cases = [_]struct { shape: []const c_int, qat: RopeQat, inverse: bool }{
        .{ .shape = &.{ 5, 64, 512 }, .qat = .none, .inverse = false },
        .{ .shape = &.{ 5, 64, 512 }, .qat = .none, .inverse = true },
        .{ .shape = &.{ 7, 512 }, .qat = .fp8_ue8m0, .inverse = false },
        .{ .shape = &.{ 7, 512 }, .qat = .fp4_e4m3, .inverse = false },
        .{ .shape = &.{ 3, 64, 128 }, .qat = .fp4_ue8m0, .inverse = false },
    };
    for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dt| for (cases) |c| {
        const rows: usize = @intCast(c.shape[0]);
        const x = try sc.astype(try randn(&sc, c.shape, .float32, 91, 3.0), dt);
        const pos = try positions(&sc, 1000, rows, 7);
        const rotated = try qat_mod.rotate(&sc, x, freqs, pos, 64, c.inverse);
        const want = switch (c.qat) {
            .none => rotated,
            .fp8_ue8m0 => try qat_mod.qat(&sc, rotated, .fp8_ue8m0, 32),
            .fp4_ue8m0 => try qat_mod.qat(&sc, rotated, .fp4_ue8m0, 32),
            .fp4_e4m3 => try qat_mod.qat(&sc, rotated, .fp4_e4m3, 16),
        };
        const got = try ropeTail(&sc, x, freqs, pos, 64, c.inverse, c.qat, null);
        try expectBitEqual("rope tail", try hostOf(&sc, got), try hostOf(&sc, want));
    };
}

test "dsv41 kv row: the rms norm, rotation and fp8 round trip are the composed chain's, bit for bit" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var sc = Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    const freqs = try ropeFreqs(&sc);
    for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dt| for ([_]bool{ true, false }) |qat| {
        const kv = try sc.astype(try randn(&sc, &.{ 9, 512 }, .float32, 92, 5.0), dt);
        const w = try sc.astype(try sc.add(try randn(&sc, &.{512}, .float32, 93, 0.3), try sc.f(1.0)), dt);
        const pos = try positions(&sc, 4000, 9, 1);
        const normed = try qat_mod.rotate(&sc, try sc.rms(kv, w, 1e-6), freqs, pos, 64, false);
        const want = if (qat) try qat_mod.qat(&sc, normed, .fp8_ue8m0, 32) else normed;
        const got = try kvRow(&sc, kv, w, freqs, pos, 64, 1e-6, qat);
        try expectBitEqual("kv row", try hostOf(&sc, got), try hostOf(&sc, want));
    };
}

const AttnCase = struct {
    heads: usize = 64,
    d: usize = 512,
    window: usize = 128,
    p0: usize = 300,
    n_comp: usize = 100,
    k: usize = 64,
};

const AttnData = struct { q: A, ring: A, comp: A, topk: A, sink: A, c: AttnCase };

fn attnData(sc: *Scope, c: AttnCase, dt: mlx.mlx_dtype) !AttnData {
    var picks: [256]i32 = undefined;
    for (0..c.k) |i| picks[i] = if (i % 11 == 5) -1 else @intCast((i * 7 + 3) % c.n_comp);
    return .{
        .q = try sc.astype(try randn(sc, &.{ 1, ci(c.heads), ci(c.d) }, .float32, 101, 1.5), dt),
        .ring = try sc.astype(try randn(sc, &.{ ci(c.window), ci(c.d) }, .float32, 102, 1.5), dt),
        .comp = try sc.astype(try randn(sc, &.{ ci(c.n_comp), ci(c.d) }, .float32, 103, 1.5), dt),
        .topk = try sc.ints(picks[0..c.k], &.{ 1, ci(c.k) }),
        .sink = try randn(sc, &.{ 1, ci(c.heads), 1 }, .float32, 104, 1.0),
        .c = c,
    };
}

fn decodeArgs(a: *const AttnData, dt: mlx.mlx_dtype, freqs: ?A) DecodeArgs {
    return .{
        .q = a.q,
        .ring = a.ring,
        .comp = a.comp,
        .topk = a.topk,
        .sink = a.sink,
        .p0 = a.c.p0,
        .window = a.c.window,
        .scale = 1.0 / @sqrt(@as(f32, @floatFromInt(a.c.d))),
        .freqs = freqs orelse sc_mod.none,
        .rd = if (freqs != null) 64 else 0,
        .out_dt = dt,
    };
}

/// The decode heads in f64 from the same f32/bf16 values, no rounding between stages.
fn truthAttn(sc: *Scope, a: *const AttnData, out: []f64) !void {
    const gpa = testing.allocator;
    const c = a.c;
    const q = try hostOf(sc, a.q);
    const ring = try hostOf(sc, a.ring);
    const comp = try hostOf(sc, a.comp);
    const picks = try sc_mod.hostI32(a.topk);
    const sink = try sc_mod.hostF32(a.sink);
    const scale: f64 = 1.0 / @sqrt(@as(f64, @floatFromInt(c.d)));
    const lo = if (c.p0 + 1 > c.window) c.p0 + 1 - c.window else 0;
    const rows = try gpa.alloc([]const f32, c.window + c.k);
    defer gpa.free(rows);
    var n: usize = 0;
    for (lo..c.p0 + 1) |p| {
        rows[n] = ring[(p % c.window) * c.d ..][0..c.d];
        n += 1;
    }
    for (picks[0..c.k]) |pk| if (pk >= 0) {
        rows[n] = comp[@as(usize, @intCast(pk)) * c.d ..][0..c.d];
        n += 1;
    };
    const w = try gpa.alloc(f64, n);
    defer gpa.free(w);
    for (0..c.heads) |h| {
        var mx: f64 = sink[h];
        for (rows[0..n], 0..) |r, i| {
            var s: f64 = 0;
            for (0..c.d) |d| s += @as(f64, q[h * c.d + d]) * r[d];
            w[i] = s * scale;
            mx = @max(mx, w[i]);
        }
        var den: f64 = @exp(@as(f64, sink[h]) - mx);
        for (w[0..n]) |*x| {
            x.* = @exp(x.* - mx);
            den += x.*;
        }
        for (0..c.d) |d| {
            var o: f64 = 0;
            for (rows[0..n], 0..) |r, i| o += w[i] * r[d];
            out[h * c.d + d] = o / den;
        }
    }
}

test "dsv41 decode attention: every split count is no further from the f64 truth than the composed core" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var sc = Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    const a = try attnData(&sc, .{}, .float32);
    const truth = try testing.allocator.alloc(f64, a.c.heads * a.c.d);
    defer testing.allocator.free(truth);
    try truthAttn(&sc, &a, truth);
    // the composed core over the window rows in position order and the picked compressed rows
    const lo = a.c.p0 + 1 - a.c.window;
    var slots: [128]i32 = undefined;
    for (lo..a.c.p0 + 1, 0..) |p, i| slots[i] = @intCast(p % a.c.window);
    const keys_w = try sc.take(a.ring, try sc.ints(slots[0..a.c.window], &.{ci(a.c.window)}), 0);
    const band = try sc.broadcast(try sc.keep(mlx.mlx_array_new_bool(true)), &.{ 1, ci(a.c.window) });
    const comp = try hostOf(&sc, try model.attnCore(&sc, a.sink, a.q, keys_w, band, a.comp, a.topk, 1.0 / @sqrt(@as(f32, 512)), .float32));
    const err_comp = relErr(comp, truth);
    for ([_]usize{ 1, 2, 4, 5 }) |splits| {
        const got = try hostOf(&sc, try attnDecodeSplit(&sc, decodeArgs(&a, .float32, null), splits));
        try expectLe("attn decode vs truth", relErr(got, truth), @max(err_comp, 2e-6));
    }
}

test "dsv41 decode attention: the folded inverse rotation is the rotation of the heads, bit for bit" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var sc = Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    const freqs = try ropeFreqs(&sc);
    for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dt| for ([_]usize{ 1, 4 }) |splits| {
        const a = try attnData(&sc, .{}, dt);
        const plain = try attnDecodeSplit(&sc, decodeArgs(&a, dt, null), splits);
        const want = try qat_mod.rotate(&sc, plain, freqs, try positions(&sc, @intCast(a.c.p0), 1, 1), 64, true);
        const got = try attnDecodeSplit(&sc, decodeArgs(&a, dt, freqs), splits);
        try expectBitEqual("attn unrope", try hostOf(&sc, got), try hostOf(&sc, want));
    };
}

test "dsv41 decode attention: a block of rows is the rows of single-row blocks, bit for bit, over a ring with slack" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var sc = Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    const freqs = try ropeFreqs(&sc);
    const rows = 6;
    const c: AttnCase = .{ .p0 = 300 };
    var a = try attnData(&sc, c, .bfloat16);
    a.q = try sc.astype(try randn(&sc, &.{ rows, ci(c.heads), ci(c.d) }, .float32, 111, 1.5), .bfloat16);
    a.ring = try sc.astype(try randn(&sc, &.{ ci(c.window + 16), ci(c.d) }, .float32, 112, 1.5), .bfloat16);
    var picks: [rows * 64]i32 = undefined;
    for (0..rows) |r| for (0..c.k) |i| {
        picks[r * c.k + i] = if ((i + r) % 11 == 5) -1 else @intCast((i * 7 + r * 5 + 3) % c.n_comp);
    };
    a.topk = try sc.ints(&picks, &.{ rows, ci(c.k) });
    var args = decodeArgs(&a, .bfloat16, freqs);
    const got = try hostOf(&sc, try attnDecode(&sc, args));
    const row = c.heads * c.d;
    for (0..rows) |r| {
        args.q = try sc.slice(a.q, 0, r, r + 1);
        args.topk = try sc.slice(a.topk, 0, r, r + 1);
        args.p0 = c.p0 + r;
        try expectBitEqual("attn rows", got[r * row ..][0..row], try hostOf(&sc, try attnDecode(&sc, args)));
    }
}

test "dsv41 wide attention: the f16 core gives the f32 chain's heads at the release's span shape" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var sc = Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    const L = 256;
    const H = 64;
    const D = 512;
    const window = 128;
    const wp = 127;
    const K = 512;
    const n_comp = 700;
    const q = try sc.astype(try randn(&sc, &.{ L, H, D }, .float32, 111, 1.5), .bfloat16);
    const keys = try sc.astype(try randn(&sc, &.{ wp + L, D }, .float32, 112, 1.5), .bfloat16);
    const comp = try sc.astype(try randn(&sc, &.{ n_comp, D }, .float32, 113, 1.5), .bfloat16);
    const sink = try randn(&sc, &.{ 1, H, 1 }, .float32, 114, 1.0);
    var picks: [L * K]i32 = undefined;
    for (0..L) |l| for (0..K) |i| {
        picks[l * K + i] = if ((l + i) % 9 == 4) -1 else @intCast((i * 3 + l) % n_comp);
    };
    const topk = try sc.ints(&picks, &.{ L, K });
    const scale: f32 = 1.0 / @sqrt(@as(f32, D));
    const qpos = try sc.add(try sc.reshape(try sc.arange(0, L), &.{ L, 1 }), try sc.i(wp));
    const kpos = try sc.reshape(try sc.arange(0, wp + L), &.{ 1, wp + L });
    const band = try sc.op2(mlx.mlx_logical_and, try sc.op2(mlx.mlx_less_equal, kpos, qpos), try sc.op2(mlx.mlx_greater, kpos, try sc.sub(qpos, try sc.i(window))));
    const want = try hostOf(&sc, try model.attnCore(&sc, sink, q, keys, band, comp, topk, scale, .bfloat16));
    const got = try hostOf(&sc, try attnWide(&sc, sink, q, keys, wp, window, comp, topk, scale, .bfloat16));
    try expectLe("f16 core vs f32 chain", relErrF(got, want), 1.0 / 64.0);
}

fn composedIndexScores(sc: *Scope, q: A, keys: A, w: A, wscale: f32, p0: usize, ratio: usize) !A {
    const L = dim(q, 0);
    const ih = dim(q, 1);
    const nb = dim(keys, 0);
    const weights = try sc.mul(try sc.astype(w, .float32), try sc.f(wscale));
    const dots = try sc.reshape(try sc.matmul(try sc.reshape(q, &.{ ci(L * ih), ci(dim(q, 2)) }), try sc.t2(try sc.astype(keys, .float32))), &.{ ci(L), ci(ih), ci(nb) });
    const s = try sc.sum(try sc.mul(try sc.maximum(dots, try sc.f(0)), try sc.expand(weights, 2)), 1, false);
    var lens: [64]i32 = undefined;
    for (0..L) |l| lens[l] = @intCast((p0 + l + 1) / ratio);
    const seen = try sc.ints(lens[0..L], &.{ ci(L), 1 });
    return sc.where(try sc.lt(try sc.reshape(try sc.arange(0, nb), &.{ 1, ci(nb) }), seen), s, try sc.f(-std.math.inf(f32)));
}

test "dsv41 index scores: the scalar and the matrix kernels are no further from the f64 truth than the composed chain, -inf past the groups seen" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var sc = Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    const IH = 64;
    const IHD = 128;
    const nb = 1500;
    const ratio = 4;
    for ([_]struct { L: usize, p0: usize, tile: bool }{ .{ .L = 1, .p0 = 5000, .tile = false }, .{ .L = 40, .p0 = 5000, .tile = true }, .{ .L = 40, .p0 = 5000, .tile = false } }) |c| {
        const q = try sc.astype(try sc.astype(try randn(&sc, &.{ ci(c.L), IH, IHD }, .float32, 121, 1.5), .bfloat16), .float32);
        const keys = try sc.astype(try randn(&sc, &.{ nb, IHD }, .float32, 122, 1.5), .bfloat16);
        const w = try sc.astype(try randn(&sc, &.{ ci(c.L), IH }, .float32, 123, 1.0), .bfloat16);
        const wscale: f32 = 1.0 / @sqrt(@as(f32, IHD)) / @sqrt(@as(f32, IH));
        const got = try hostOf(&sc, if (c.tile) try indexScoresTile(&sc, q, keys, w, wscale, c.p0, ratio, 0) else try indexScores(&sc, q, keys, w, wscale, c.p0, ratio));
        const comp = try hostOf(&sc, try composedIndexScores(&sc, q, keys, w, wscale, c.p0, ratio));
        const qh = try sc_mod.hostF32(q);
        const kh = try hostOf(&sc, keys);
        const wh = try hostOf(&sc, w);
        const truth = try testing.allocator.alloc(f64, c.L * nb);
        defer testing.allocator.free(truth);
        var gv: std.ArrayList(f32) = .empty;
        defer gv.deinit(testing.allocator);
        var cv: std.ArrayList(f32) = .empty;
        defer cv.deinit(testing.allocator);
        var tv: std.ArrayList(f64) = .empty;
        defer tv.deinit(testing.allocator);
        for (0..c.L) |l| for (0..nb) |n| {
            const seen = (c.p0 + l + 1) / ratio;
            const i = l * nb + n;
            if (n >= seen) {
                try testing.expect(std.math.isNegativeInf(got[i]) and std.math.isNegativeInf(comp[i]));
                continue;
            }
            var s: f64 = 0;
            for (0..IH) |h| {
                var d: f64 = 0;
                for (0..IHD) |j| d += @as(f64, qh[(l * IH + h) * IHD + j]) * kh[n * IHD + j];
                s += @as(f64, wh[l * IH + h]) * wscale * @max(d, 0);
            }
            try gv.append(testing.allocator, got[i]);
            try cv.append(testing.allocator, comp[i]);
            try tv.append(testing.allocator, s);
        };
        try expectLe("index scores vs truth", relErr(gv.items, tv.items), @max(relErr(cv.items, tv.items), 2e-6));
    }
}

test "dsv41 index scores: key slices scored at their offset are the whole list's columns bit for bit" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var sc = Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    const IH = 32;
    const IHD = 128;
    const L = 40;
    const nb = 2000;
    const ratio = 2;
    for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dt| {
        const q = try sc.astype(try randn(&sc, &.{ L, IH, IHD }, .float32, 141, 1.5), dt);
        const keys = try sc.astype(try randn(&sc, &.{ nb, IHD }, .float32, 142, 1.5), dt);
        const w = try sc.astype(try randn(&sc, &.{ L, IH }, .float32, 143, 1.0), .bfloat16);
        const wscale: f32 = 1.0 / @sqrt(@as(f32, IHD)) / @sqrt(@as(f32, IH));
        const p0 = 1700;
        const whole = try hostOf(&sc, try indexScoresTile(&sc, q, keys, w, wscale, p0, ratio, 0));
        var t0: usize = 0;
        while (t0 < nb) : (t0 += 640) {
            const t1 = @min(nb, t0 + 640);
            const part = try hostOf(&sc, try indexScoresTile(&sc, q, try sc.slice(keys, 0, t0, t1), w, wscale, p0, ratio, t0));
            for (0..L) |l| for (t0..t1) |n| {
                try testing.expectEqual(@as(u32, @bitCast(whole[l * nb + n])), @as(u32, @bitCast(part[l * (t1 - t0) + n - t0])));
            };
        }
    }
}

test "dsv41 index scores: the scalar kernel over packed keys gives the unpacked keys' scores bit for bit" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var sc = Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    const ratio = 1;
    // A few keys, one scale byte a key: arrays MLX passes in the constant address space.
    for ([_]struct { ih: usize, ihd: usize }{ .{ .ih = 32, .ihd = 128 }, .{ .ih = 8, .ihd = 32 } }) |g| for ([_]usize{ 4, 1537 }) |nb| for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dt| for ([_]usize{ 1, 5, 16 }) |L| {
        const x = try qat_mod.qat(&sc, try sc.astype(try randn(&sc, &.{ ci(nb), ci(g.ihd) }, .float32, 151, 2.0), dt), .fp4_ue8m0, 32);
        const packed_keys = try qat_mod.packFp4(&sc, x, .fp4_ue8m0, 32);
        const keys = try qat_mod.unpackFp4(&sc, packed_keys, .fp4_ue8m0, 32, dt);
        const q = try sc.astype(try sc.astype(try randn(&sc, &.{ ci(L), ci(g.ih), ci(g.ihd) }, .float32, 152, 1.5), .bfloat16), .float32);
        const w = try sc.astype(try randn(&sc, &.{ ci(L), ci(g.ih) }, .float32, 153, 1.0), .bfloat16);
        const wscale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(g.ihd))) / @sqrt(@as(f32, @floatFromInt(g.ih)));
        const p0 = nb - 1;
        const want = try hostOf(&sc, try indexScores(&sc, q, keys, w, wscale, p0, ratio));
        const got = try hostOf(&sc, try indexScoresPacked(&sc, q, packed_keys.codes, packed_keys.scales, w, wscale, p0, ratio, .fp4_ue8m0, 32));
        try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(want), std.mem.sliceAsBytes(got));
    };
}

const HcData = struct { x: A, y: A, cpre: A, w: A, hw: HcWeights, ex: HcExpand, eps: A, L: usize, hc: usize, d: usize };

fn hcData(sc: *Scope, L: usize, dt: mlx.mlx_dtype) !HcData {
    const hc = 4;
    const D = 5120;
    const mix = (2 + hc) * hc;
    const f32t = mlx.mlx_dtype.float32;
    const l: c_int = ci(L);
    return .{
        .x = try sc.astype(try randn(sc, &.{ l, hc, D }, f32t, 131, 1.5), dt),
        .y = try sc.astype(try randn(sc, &.{ l, D }, f32t, 132, 1.0), dt),
        .cpre = try sc.op1(mlx.mlx_abs, try randn(sc, &.{ l, hc }, f32t, 133, 0.6)),
        .w = try sc.astype(try sc.add(try randn(sc, &.{D}, f32t, 134, 0.2), try sc.f(1.0)), dt),
        .hw = .{ .fn_w = try randn(sc, &.{ mix, hc * D }, f32t, 135, 0.02), .scale = try f32Const(sc, &.{ 0.3, 0.5, 0.7 }), .base = try randn(sc, &.{mix}, f32t, 136, 0.5) },
        .ex = .{ .y = undefined, .post = try randn(sc, &.{ l, hc }, f32t, 137, 0.7), .comb = try randn(sc, &.{ l, hc, hc }, f32t, 138, 0.5) },
        .eps = try hcEps(sc, 1e-6, 1e-6),
        .L = L,
        .hc = hc,
        .d = D,
    };
}

/// `hcMixes`' projection and `rms(hcCollapse)` as the op chains of `dsv41_model.zig`.
fn composedHc(sc: *Scope, d: *const HcData, x: A) !struct { z: A, h: A } {
    const flat = try sc.reshape(try sc.astype(x, .float32), &.{ ci(d.L), ci(d.hc * d.d) });
    const rs = try sc.op1(mlx.mlx_rsqrt, try sc.add(try sc.mean(try sc.op1(mlx.mlx_square, flat), -1, true), try sc.f(1e-6)));
    const mixes = try sc.mul(try sc.matmul(flat, try sc.t2(d.hw.fn_w)), rs);
    const s0 = try sc.slice(d.hw.scale, 0, 0, 1);
    const s1 = try sc.slice(d.hw.scale, 0, 1, 2);
    const s2 = try sc.slice(d.hw.scale, 0, 2, 3);
    const z = try sc.concat(&.{
        try sc.add(try sc.mul(try sc.slice(mixes, 1, 0, 4), s0), try sc.slice(d.hw.base, 0, 0, 4)),
        try sc.add(try sc.mul(try sc.slice(mixes, 1, 4, 8), s1), try sc.slice(d.hw.base, 0, 4, 8)),
        try sc.add(try sc.mul(try sc.slice(mixes, 1, 8, 24), s2), try sc.slice(d.hw.base, 0, 8, 24)),
    }, 1);
    return .{ .z = z, .h = try sc.rms(try model.hcCollapse(sc, x, d.cpre), d.w, 1e-6) };
}

/// The same in f64 for row 0: z and h of the stream `x`.
fn truthHc(sc: *Scope, d: *const HcData, x: A, z: []f64, h: []f64) !void {
    const xh = try hostOf(sc, x);
    const fnw = try sc_mod.hostF32(d.hw.fn_w);
    const scale = try sc_mod.hostF32(d.hw.scale);
    const base = try sc_mod.hostF32(d.hw.base);
    const cpre = try sc_mod.hostF32(d.cpre);
    const w = try hostOf(sc, d.w);
    const N = d.hc * d.d;
    var ss: f64 = 0;
    for (xh[0..N]) |v| ss += @as(f64, v) * v;
    const rs = 1.0 / @sqrt(ss / @as(f64, @floatFromInt(N)) + 1e-6);
    for (0..z.len) |m| {
        var a: f64 = 0;
        for (0..N) |k| a += @as(f64, xh[k]) * fnw[m * N + k];
        z[m] = a * rs * scale[if (m < 4) 0 else if (m < 8) 1 else 2] + base[m];
    }
    var hs: f64 = 0;
    for (0..d.d) |c| {
        var v: f64 = 0;
        for (0..d.hc) |j| v += @as(f64, cpre[j]) * xh[j * d.d + c];
        h[c] = v;
        hs += v * v;
    }
    const inv = 1.0 / @sqrt(hs / @as(f64, @floatFromInt(d.d)) + 1e-6);
    for (h, 0..) |*v, c| v.* = v.* * inv * w[c];
}

test "dsv41 hc stage: the mixes and the normed collapse are no further from the f64 truth than the chain, the expanded stream too" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var sc = Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    var d = try hcData(&sc, 1, .float32);
    d.ex.y = d.y;
    const z_truth = try testing.allocator.alloc(f64, 24);
    defer testing.allocator.free(z_truth);
    const h_truth = try testing.allocator.alloc(f64, d.d);
    defer testing.allocator.free(h_truth);
    const ref = try composedHc(&sc, &d, d.x);
    try truthHc(&sc, &d, d.x, z_truth, h_truth);
    const st = try hcStage(&sc, d.x, null, d.cpre, d.w, d.hw, d.eps);
    try expectLe("stage z", relErr(try hostOf(&sc, st.z), z_truth), @max(relErr(try hostOf(&sc, ref.z), z_truth), 2e-6));
    try expectLe("stage h", relErr(try hostOf(&sc, st.h), h_truth), @max(relErr(try hostOf(&sc, ref.h), h_truth), 2e-6));
    // the expanded stream: the chain's `hcExpand`, then the same two outputs of its mixes
    const x1_ref = try model.hcExpand(&sc, d.x, d.y, d.ex.post, d.ex.comb);
    const sx = try hcStage(&sc, d.x, d.ex, d.cpre, d.w, d.hw, d.eps);
    const xh = try hostOf(&sc, d.x);
    const yh = try hostOf(&sc, d.y);
    const ph = try sc_mod.hostF32(d.ex.post);
    const ch = try sc_mod.hostF32(d.ex.comb);
    const x1_truth = try testing.allocator.alloc(f64, d.hc * d.d);
    defer testing.allocator.free(x1_truth);
    for (0..d.hc) |i| for (0..d.d) |c| {
        var acc: f64 = @as(f64, ph[i]) * yh[c];
        for (0..d.hc) |j| acc += @as(f64, ch[j * d.hc + i]) * xh[j * d.d + c];
        x1_truth[i * d.d + c] = acc;
    };
    try expectLe("expanded stream", relErr(try hostOf(&sc, sx.x1), x1_truth), @max(relErr(try hostOf(&sc, x1_ref), x1_truth), 2e-6));
    const ref1 = try composedHc(&sc, &d, x1_ref);
    try truthHc(&sc, &d, sx.x1, z_truth, h_truth);
    try expectLe("expand stage z", relErr(try hostOf(&sc, sx.z), z_truth), @max(relErr(try hostOf(&sc, ref1.z), z_truth), 2e-6));
    try expectLe("expand stage h", relErr(try hostOf(&sc, sx.h), h_truth), @max(relErr(try hostOf(&sc, ref1.h), h_truth), 2e-6));
}

test "dsv41 hc stage: in bf16 the stage gives the chain's normed collapse and expanded stream to bf16 ulps" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var sc = Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    var d = try hcData(&sc, 1, .bfloat16);
    d.ex.y = d.y;
    const x1_ref = try model.hcExpand(&sc, d.x, d.y, d.ex.post, d.ex.comb);
    const sx = try hcStage(&sc, d.x, d.ex, d.cpre, d.w, d.hw, d.eps);
    try expectLe("bf16 expanded stream", relErrF(try hostOf(&sc, sx.x1), try hostOf(&sc, x1_ref)), 1.0 / 128.0);
    const ref = try composedHc(&sc, &d, x1_ref);
    try expectLe("bf16 normed collapse", relErrF(try hostOf(&sc, sx.h), try hostOf(&sc, ref.h)), 1.0 / 64.0);
}

test "dsv41 hc split: the lane-per-element split gives the thread-per-row split of the mixes" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var sc = Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    const mixes = try randn(&sc, &.{ 37, 24 }, .float32, 141, 3.0);
    const scale = try f32Const(&sc, &.{ 0.3, 0.8, 1.1 });
    const base = try randn(&sc, &.{24}, .float32, 142, 1.0);
    const want = try qat_mod.hcSplitOps(&sc, mixes, scale, base, 20, 1e-6);
    const z = try sc.concat(&.{
        try sc.add(try sc.mul(try sc.slice(mixes, 1, 0, 4), try sc.slice(scale, 0, 0, 1)), try sc.slice(base, 0, 0, 4)),
        try sc.add(try sc.mul(try sc.slice(mixes, 1, 4, 8), try sc.slice(scale, 0, 1, 2)), try sc.slice(base, 0, 4, 8)),
        try sc.add(try sc.mul(try sc.slice(mixes, 1, 8, 24), try sc.slice(scale, 0, 2, 3)), try sc.slice(base, 0, 8, 24)),
    }, 1);
    const got = try hcSplitZ(&sc, z, 4, 20, try hcEps(&sc, 1e-6, 1e-6));
    inline for (.{ "pre", "post", "comb" }) |f| {
        const a = try sc_mod.hostF32(@field(got, f));
        const b = try sc_mod.hostF32(@field(want, f));
        for (a, b) |x, y| try testing.expectApproxEqAbs(x, y, 2e-6);
    }
}

test "dsv41 hc wide: the mix and norm kernels give the decode stage's mixes, split and normed collapse, bit for bit" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var sc = Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dt| {
        const d = try hcData(&sc, 9, dt);
        const st = try hcStage(&sc, d.x, null, d.cpre, d.w, d.hw, d.eps);
        const sp = try hcSplitZ(&sc, st.z, 4, 20, d.eps);
        const z = try hcMix(&sc, d.x, d.hw, d.eps);
        const wide = try hcNorm(&sc, z, d.x, d.cpre, d.w, 20, d.eps);
        try expectBitEqual("hc mix z", try sc_mod.hostF32(z), try sc_mod.hostF32(st.z));
        try expectBitEqual("hc norm h", try hostOf(&sc, wide.h), try hostOf(&sc, st.h));
        try expectBitEqual("hc norm pre", try sc_mod.hostF32(wide.sp.pre), try sc_mod.hostF32(sp.pre));
        try expectBitEqual("hc norm post", try sc_mod.hostF32(wide.sp.post), try sc_mod.hostF32(sp.post));
        try expectBitEqual("hc norm comb", try sc_mod.hostF32(wide.sp.comb), try sc_mod.hostF32(sp.comb));
    }
}

test "dsv41 engram: the gate kernel writes the op chain's stream, bit for bit" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var sc = Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    var cfg = Config{};
    cfg.hc_mult = 4;
    cfg.eps = 1e-6;
    for ([_]struct { d: u32, L: usize }{ .{ .d = 5120, .L = 1 }, .{ .d = 5120, .L = 5 }, .{ .d = 1000, .L = 2 }, .{ .d = 128, .L = 3 }, .{ .d = 40, .L = 2 } }) |c| {
        cfg.dim = c.d;
        for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dt| {
            const d: c_int = @intCast(c.d);
            const l: c_int = ci(c.L);
            const x = try sc.astype(try randn(&sc, &.{ l, 4, d }, .float32, 151, 1.5), dt);
            const kv = try sc.astype(try randn(&sc, &.{ l, 5 * d }, .float32, 152, 1.0), dt);
            const qk = try randn(&sc, &.{ 4, d }, .float32, 153, 0.8);
            const want = try model.engramGateOps(&sc, &cfg, qk, x, kv);
            const got = try engramGate(&sc, x, kv, qk, 1e-6);
            try expectBitEqual("engram gate", try hostOf(&sc, got), try hostOf(&sc, want));
        }
    }
}

test "dsv41 kernels: a new row count, pick count, split count or context width compiles no new Metal library" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var sc = Scope.init(testing.allocator, mlx.gpuStream());
    defer sc.deinit();
    const keys = mlx.kernel_keys;
    _ = keys.trackForTest();
    const freqs = try ropeFreqs(&sc);
    const Arm = enum { rope, rope_qat, topk, qat, pack, attn };
    for (std.enums.values(Arm)) |arm| {
        var seen: usize = 0;
        // The first size builds the arm's libraries; every later size must reuse them (all of 8 rows or more: MLX binds
        // an input under 8 elements as `constant`, a second library per binding).
        for ([_]usize{ 9, 12, 13, 20 }, 0..) |n, i| {
            const before = keys.count();
            const out = switch (arm) {
                .rope => try ropeTail(&sc, try randn(&sc, &.{ ci(n), 4, 128 }, .bfloat16, 1, 1.0), freqs, try positions(&sc, 9, n, 1), 64, false, .none, null),
                .rope_qat => try ropeTail(&sc, try randn(&sc, &.{ ci(n), 128 }, .bfloat16, 2, 1.0), freqs, try positions(&sc, 9, n, 1), 64, false, .fp4_e4m3, null),
                .topk => try topkIndices(&sc, try randn(&sc, &.{ 2, 40 }, .float32, 3, 1.0), n),
                .qat => try qat_mod.qat(&sc, try randn(&sc, &.{ ci(n), 64 }, .bfloat16, 4, 1.0), .fp8_ue8m0, 32),
                .pack => blk: {
                    const p = try qat_mod.packFp4(&sc, try qat_mod.qat(&sc, try randn(&sc, &.{ ci(n), 64 }, .bfloat16, 5, 1.0), .fp4_ue8m0, 32), .fp4_ue8m0, 32);
                    break :blk try qat_mod.unpackFp4(&sc, p, .fp4_ue8m0, 32, .bfloat16);
                },
                .attn => blk: {
                    const a = try attnData(&sc, .{ .k = 8 * n }, .bfloat16);
                    break :blk try attnDecodeSplit(&sc, decodeArgs(&a, .bfloat16, freqs), 2 + i);
                },
            };
            try mlx.check(mlx.mlx_array_eval(out));
            if (i == 0) seen = keys.count() - before else if (keys.count() != before) {
                std.debug.print("{t} at size {d} built {d} libraries after the first size's {d}: {s}\n", .{ arm, n, keys.count() - before, seen, keys.last() });
                return error.KernelRebuilt;
            }
        }
    }
}
