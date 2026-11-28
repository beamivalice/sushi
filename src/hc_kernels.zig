//! The hyper-connection stage kernels (mix, collapse and norm, split, expand) Sushi Cooler's archs share. Each
//! launch takes a stream and returns arrays the caller frees; pipeline names keep their first arch's prefix, so the
//! warm manifests' recipes do not move.
const std = @import("std");
const mlx = @import("mlx.zig");
const metal_def = @import("metal_def.zig");
const nax_gate = @import("nax_gate.zig");

const A = mlx.mlx_array;
const Def = metal_def.Def;
const OutSpec = metal_def.OutSpec;
const Grid = metal_def.Grid;
const Tmpl = metal_def.Tmpl;
const TDtype = metal_def.TDtype;
const launch = metal_def.launch;

fn dim(x: A, axis: usize) usize {
    return @intCast(mlx.getShape(x)[axis]);
}

fn ci(v: usize) c_int {
    return @intCast(v);
}


/// A mix's index into the three scales (pre, post, comb) the checkpoint ships.
pub const HC_HEADER =
    \\inline int dsv41_scale_at(int m, int hc) { return m < hc ? 0 : (m < 2 * hc ? 1 : 2); }
;

/// `hcMixes`' projection: per row, the mixing logits of the flattened stream, rms-scaled, times scale plus base.
/// One threadgroup an R-row x M-mix tile, so each fn and stream load serves the tile; every (row, mix) dot keeps
/// `HC_STAGE_SOURCE`'s order (thread tid the terms k = tid + c * TG in c order, one simd sum, the simdgroups in
/// order), which is what makes the two agree to the bit.
pub const HC_MIX_SOURCE =
    \\#pragma clang fp contract(off)
    \\const int m0 = threadgroup_position_in_grid.x * M, l0 = threadgroup_position_in_grid.y * R;
    \\const uint tid = thread_index_in_threadgroup, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    \\const int N = fn_shape[1], MIX = fn_shape[0], L = x_shape[0];
    \\#ifdef SUSHI_METALLIB
    \\constexpr int R_CAP = 4;
    \\constexpr int M_CAP = 12;
    \\#else
    \\constexpr int R_CAP = R;
    \\constexpr int M_CAP = M;
    \\#endif
    \\float d[R_CAP][M_CAP], ss[R_CAP];
    \\#pragma clang loop unroll(full)
    \\for (int r = 0; r < R_CAP; ++r) {
    \\  if (r >= R) break;
    \\  ss[r] = 0.0f;
    \\  #pragma clang loop unroll(full)
    \\  for (int j = 0; j < M_CAP; ++j) { if (j >= M) break; d[r][j] = 0.0f; }
    \\}
    \\for (int k = tid; k < N; k += TG) {
    \\  float fv[M_CAP];
    \\  #pragma clang loop unroll(full)
    \\  for (int j = 0; j < M_CAP; ++j) { if (j >= M) break; fv[j] = fn[size_t(m0 + j) * N + k]; }
    \\  #pragma clang loop unroll(full)
    \\  for (int r = 0; r < R_CAP; ++r) {
    \\    if (r >= R) break;
    \\    const float v = l0 + r < L ? float(x[size_t(l0 + r) * N + k]) : 0.0f;
    \\    #pragma clang loop unroll(full)
    \\    for (int j = 0; j < M_CAP; ++j) { if (j >= M) break; d[r][j] = metal::fma(v, fv[j], d[r][j]); }
    \\    ss[r] = metal::fma(v, v, ss[r]);
    \\  }
    \\}
    \\threadgroup float t_d[R_CAP][M_CAP][32], t_s[R_CAP][32];
    \\#pragma clang loop unroll(full)
    \\for (int r = 0; r < R_CAP; ++r) {
    \\  if (r >= R) break;
    \\  #pragma clang loop unroll(full)
    \\  for (int j = 0; j < M_CAP; ++j) { if (j >= M) break; const float v = metal::simd_sum(d[r][j]); if (lane == 0) t_d[r][j][sg] = v; }
    \\  const float v = metal::simd_sum(ss[r]);
    \\  if (lane == 0) t_s[r][sg] = v;
    \\}
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\if (int(tid) < R * M) {
    \\  const int r = int(tid) / M, j = int(tid) % M;
    \\  float a = 0.0f, b = 0.0f;
    \\  for (int g = 0; g < TG / 32; ++g) { a += t_d[r][j][g]; b += t_s[r][g]; }
    \\  if (l0 + r < L) z[(l0 + r) * MIX + m0 + j] = a * metal::precise::rsqrt(b / float(N) + eps[0]) * scale[dsv41_scale_at(m0 + j, HC)] + base[m0 + j];
    \\}
;

/// `HC_MIX_SOURCE` on the matrix units: a threadgroup takes 16 rows and every mix, its simdgroups a k range each of
/// 16 rows x 32 mix columns over 16-wide k steps, `fn` split into a high and a low half so the product keeps f32's
/// bits. The sums' order differs from the ALU kernel's.
pub const HC_MIX_NAX_SOURCE =
    \\const uint tid = thread_index_in_threadgroup, sg = simdgroup_index_in_threadgroup;
    \\const ushort lane = thread_index_in_simdgroup;
    \\const int MIX = fn_shape[0], L = x_shape[0];
    \\const int l0 = threadgroup_position_in_grid.x * 16;
    \\const int STEPS = NK / 16 / NSG;
    \\const int k0 = int(sg) * STEPS * 16;
    \\const int ra = metal::min(l0 + int(dsv41_nax_r(lane, 0)), L - 1), rb = metal::min(l0 + int(dsv41_nax_r(lane, 4)), L - 1);
    \\const device T* xa = x + size_t(ra) * NK + k0 + dsv41_nax_c(lane, 0);
    \\const device T* xb = x + size_t(rb) * NK + k0 + dsv41_nax_c(lane, 0);
    \\float d[16];
    \\#pragma clang loop unroll(full)
    \\for (int i = 0; i < 16; ++i) d[i] = 0.0f;
    \\float ssa = 0.0f, ssb = 0.0f;
    \\for (int s = 0; s < STEPS; ++s) {
    \\  T a[8];
    \\  #pragma clang loop unroll(full)
    \\  for (int j = 0; j < 4; ++j) { a[j] = xa[s * 16 + j]; a[4 + j] = xb[s * 16 + j]; }
    \\  half bh[16], bl[16];
    \\  #pragma clang loop unroll(full)
    \\  for (int i = 0; i < 16; ++i) {
    \\    const int mix = (i / 8) * 16 + dsv41_nax_c(lane, i % 8);
    \\    const float f = mix < MIXN ? fn[size_t(mix) * NK + k0 + s * 16 + dsv41_nax_r(lane, i % 8)] * 1024.0f : 0.0f;
    \\    const half hi = half(f);
    \\    bh[i] = hi;
    \\    bl[i] = half(f - float(hi));
    \\  }
    \\  dsv41_nax_mma<32, 16>(d, a, bl);
    \\  dsv41_nax_mma<32, 16>(d, a, bh);
    \\  #pragma clang loop unroll(full)
    \\  for (int j = 0; j < 4; ++j) {
    \\    const float va = float(a[j]), vb = float(a[4 + j]);
    \\    ssa = metal::fma(va, va, ssa);
    \\    ssb = metal::fma(vb, vb, ssb);
    \\  }
    \\}
    \\ssa += metal::simd_shuffle_xor(ssa, ushort(1));
    \\ssa += metal::simd_shuffle_xor(ssa, ushort(8));
    \\ssb += metal::simd_shuffle_xor(ssb, ushort(1));
    \\ssb += metal::simd_shuffle_xor(ssb, ushort(8));
    \\threadgroup float t_d[NSG][16][MIXN], t_s[NSG][16];
    \\#pragma clang loop unroll(full)
    \\for (int i = 0; i < 16; ++i) {
    \\  const int col = (i / 8) * 16 + dsv41_nax_c(lane, i % 8);
    \\  if (col < MIXN) t_d[sg][dsv41_nax_r(lane, i % 8)][col] = d[i];
    \\}
    \\if ((lane & 9) == 0) { t_s[sg][dsv41_nax_r(lane, 0)] = ssa; t_s[sg][dsv41_nax_r(lane, 4)] = ssb; }
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\if (tid < 16 * MIXN) {
    \\  const int r = tid / MIXN, j = tid % MIXN;
    \\  float a = 0.0f, b = 0.0f;
    \\  for (int g = 0; g < NSG; ++g) { a += t_d[g][r][j]; b += t_s[g][r]; }
    \\  if (l0 + r < L) z[(l0 + r) * MIX + j] = (a * (1.0f / 1024.0f)) * metal::precise::rsqrt(b / float(NK) + eps[0]) * scale[dsv41_scale_at(j, HC)] + base[j];
    \\}
;

/// One hyper-connection stage as two kernels. The staggered collapse reads the PREVIOUS mix (`HC_PRE`), so in
/// `HC_STAGE_SOURCE` threadgroup 0 collapses and rms-norms the stream without waiting for this stage's mixes, while
/// threadgroups 1..MIX each take one mix's dot; the FFN side rebuilds the stream from the block output in every
/// threadgroup and threadgroup 0 stores it. The split (`HC_SPLIT_SOURCE`) is read only later, except where
/// `HC_STAGE_Z_SOURCE` derives the previous split itself.
const HC_STAGE_HEAD =
    \\#pragma clang fp contract(off)
    \\const uint role = threadgroup_position_in_grid.x;
    \\const int l = threadgroup_position_in_grid.y;
    \\const uint tid = thread_index_in_threadgroup, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    \\const int D = x_shape[2], N = fn_shape[1];
    \\#ifdef SUSHI_METALLIB
    \\constexpr int HC_CAP = 4;
    \\constexpr int ND_CAP = 32;
    \\#else
    \\constexpr int HC_CAP = HC;
    \\constexpr int ND_CAP = ND;
    \\#endif
    \\threadgroup float t_a[32], t_b[32];
    \\const device auto* xr = x + size_t(l) * HC * D;
    \\float v[HC_CAP][ND_CAP];
    \\
;

const HC_STAGE_LOOP =
    \\#pragma clang loop unroll(full)
    \\for (int c = 0; c < ND_CAP; ++c) {
    \\  if (c >= ND) break;
    \\  const int d = int(tid) + c * TG;
    \\  #pragma clang loop unroll(full)
    \\  for (int i = 0; i < HC_CAP; ++i) { if (i >= HC) break; v[i][c] = 0.0f; }
;

/// The collapse weights are `cpre`.
const HC_COEF_PLAIN =
    \\#define HC_PRE(j) cpre[l * HC + (j)]
    \\
;

/// The previous split arrives as arrays.
const HC_COEF_ARRAYS = HC_COEF_PLAIN ++
    \\#define HC_POST(i) spost[l * HC + (i)]
    \\#define HC_COMB(j, i) scomb[l * HC * HC + (j) * HC + (i)]
    \\
;

/// The previous mixes `za` arrive unsplit: simdgroup 0 splits them into threadgroup memory.
const HC_COEF_Z =
    \\threadgroup float t_pre[HC_CAP], t_post[HC_CAP], t_comb[HC_CAP * HC_CAP];
    \\#define HC_PRE(j) t_pre[(j)]
    \\#define HC_POST(i) t_post[(i)]
    \\#define HC_COMB(j, i) t_comb[(j) * HC + (i)]
    \\#define HC_SPLIT_PRE(i, v) t_pre[(i)] = (v)
    \\#define HC_SPLIT_POST(i, v) t_post[(i)] = (v)
    \\#define HC_SPLIT_COMB(i, v) t_comb[(i)] = (v)
    \\if (sg == 0) {
    \\  const device float* zr = za + l * ((2 + HC) * HC);
++ "\n" ++ HC_SPLIT_BODY ++
    \\}
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\
;

/// The stream is `x` itself.
const HC_STAGE_LOAD_PLAIN =
    \\  if (d < D) {
    \\    #pragma clang loop unroll(full)
    \\    for (int i = 0; i < HC_CAP; ++i) { if (i >= HC) break; v[i][c] = float(xr[i * D + d]); }
    \\  }
    \\}
;

/// The stream is `hcExpand(x, y, previous split)`, rounded to T where the expand rounds it.
const HC_STAGE_LOAD_EXPAND =
    \\  if (d < D) {
    \\    float sv[HC_CAP];
    \\    #pragma clang loop unroll(full)
    \\    for (int j = 0; j < HC_CAP; ++j) { if (j >= HC) break; sv[j] = float(xr[j * D + d]); }
    \\    const float yv = float(y[size_t(l) * D + d]);
    \\    #pragma clang loop unroll(full)
    \\    for (int i = 0; i < HC_CAP; ++i) {
    \\      if (i >= HC) break;
    \\      float acc = 0.0f;
    \\      #pragma clang loop unroll(full)
    \\      for (int j = 0; j < HC_CAP; ++j) { if (j >= HC) break; acc = metal::fma(HC_COMB(j, i), sv[j], acc); }
    \\      const T o = T(metal::fma(HC_POST(i), yv, acc));
    \\      v[i][c] = float(o);
    \\      if (role == 0) x1[(size_t(l) * HC + i) * D + d] = o;
    \\    }
    \\  }
    \\}
;

const HC_STAGE_TAIL =
    \\if (role == 0) {
    \\  float cp[HC_CAP];
    \\  #pragma clang loop unroll(full)
    \\  for (int j = 0; j < HC_CAP; ++j) { if (j >= HC) break; cp[j] = HC_PRE(j); }
    \\  float ss = 0.0f;
    \\  T hv[ND_CAP];
    \\  #pragma clang loop unroll(full)
    \\  for (int c = 0; c < ND_CAP; ++c) {
    \\    if (c >= ND) break;
    \\    float h = 0.0f;
    \\    #pragma clang loop unroll(full)
    \\    for (int j = 0; j < HC_CAP; ++j) { if (j >= HC) break; h += cp[j] * v[j][c]; }
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
    \\  for (int c = 0; c < ND_CAP; ++c) {
    \\    if (c >= ND) break;
    \\    const int d = int(tid) + c * TG;
    \\    if (d < D) h_out[size_t(l) * D + d] = w[d] * T(float(hv[c]) * inv);
    \\  }
    \\  return;
    \\}
    \\const int m = int(role) - 1;
    \\const device float* fr = fn + size_t(m) * N;
    \\float dd = 0.0f, sq = 0.0f;
    \\#pragma clang loop unroll(full)
    \\for (int i = 0; i < HC_CAP; ++i) {
    \\  if (i >= HC) break;
    \\  #pragma clang loop unroll(full)
    \\  for (int c = 0; c < ND_CAP; ++c) {
    \\    if (c >= ND) break;
    \\    const int d = int(tid) + c * TG;
    \\    const float u = v[i][c];
    \\    if (d < D) dd = metal::fma(u, fr[i * D + d], dd);
    \\    sq = metal::fma(u, u, sq);
    \\  }
    \\}
    \\dd = metal::simd_sum(dd); sq = metal::simd_sum(sq);
    \\if (lane == 0) { t_a[sg] = dd; t_b[sg] = sq; }
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\if (tid == 0) {
    \\  float a = 0.0f, b = 0.0f;
    \\  for (int g = 0; g < TG / 32; ++g) { a += t_a[g]; b += t_b[g]; }
    \\  z[l * ((2 + HC) * HC) + m] = a * metal::precise::rsqrt(b / float(N) + eps[0]) * scale[dsv41_scale_at(m, HC)] + base[m];
    \\}
;

/// The hc split of the mixes `zr` (pre = sigmoid + eps, post = 2 sigmoid, Sinkhorn comb); one simdgroup a row, lane
/// a * HC + b holding comb[a][b] and each sum gathering its HC terms in the serial order. The includer declares
/// `lane` and `zr` and defines the stores `HC_SPLIT_PRE`, `HC_SPLIT_POST` and `HC_SPLIT_COMB`.
pub const HC_SPLIT_BODY =
    \\const float e = eps[1];
    \\if (lane < HC) {
    \\  HC_SPLIT_PRE(lane, 1.0f / (1.0f + metal::precise::exp(-zr[lane])) + e);
    \\  HC_SPLIT_POST(lane, 2.0f / (1.0f + metal::precise::exp(-zr[HC + lane])));
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
    \\if (act) HC_SPLIT_COMB(lane, v);
;

/// The split of row `l` of `z` into the output arrays.
pub const HC_SPLIT_GLOBAL =
    \\const uint lane = thread_index_in_simdgroup;
    \\const device float* zr = z + l * ((2 + HC) * HC);
    \\#define HC_SPLIT_PRE(i, v) pre[l * HC + (i)] = (v)
    \\#define HC_SPLIT_POST(i, v) post[l * HC + (i)] = (v)
    \\#define HC_SPLIT_COMB(i, v) comb[l * HC * HC + (i)] = (v)
    \\
;
pub const HC_SPLIT_SOURCE = "#pragma clang fp contract(off)\nconst int l = threadgroup_position_in_grid.y;\n" ++ HC_SPLIT_GLOBAL ++ HC_SPLIT_BODY;

/// The hc split, then the stream collapsed with `cpre` (f32, rounded to T like `hcCollapse`) and rms-normed like
/// MLX's `rms_norm`. One threadgroup a row.
pub const HC_NORM_SOURCE =
    \\#pragma clang fp contract(off)
    \\const int l = threadgroup_position_in_grid.y;
    \\const uint tid = thread_index_in_threadgroup, sg = simdgroup_index_in_threadgroup;
    \\const int D = x_shape[2];
    \\if (sg == 0) {
++ "\n" ++ HC_SPLIT_GLOBAL ++ HC_SPLIT_BODY ++
    \\}
    \\float cp[8];
    \\for (int j = 0; j < HC; ++j) cp[j] = cpre[l * HC + j];
    \\float ss = 0.0f;
    \\for (int d = tid; d < D; d += TG) {
    \\  float h = 0.0f;
    \\  for (int j = 0; j < HC; ++j) h += cp[j] * float(x[(size_t(l) * HC + j) * D + d]);
    \\  const T hb = T(h);
    \\  h_out[size_t(l) * D + d] = hb;
    \\  ss += float(hb) * float(hb);
    \\}
    \\threadgroup float t_s[32];
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

pub const HC_STAGE_SOURCE = HC_STAGE_HEAD ++ HC_COEF_PLAIN ++ HC_STAGE_LOOP ++ HC_STAGE_LOAD_PLAIN ++ HC_STAGE_TAIL;
pub const HC_STAGE_X_SOURCE = HC_STAGE_HEAD ++ HC_COEF_ARRAYS ++ HC_STAGE_LOOP ++ HC_STAGE_LOAD_EXPAND ++ HC_STAGE_TAIL;
pub const HC_STAGE_Z_SOURCE = HC_STAGE_HEAD ++ HC_COEF_Z ++ HC_STAGE_LOOP ++ HC_STAGE_LOAD_EXPAND ++ HC_STAGE_TAIL;

var hc_mix_def = Def{ .name = "dsv41_hc_mix", .ins = &.{ "x", "fn", "scale", "base", "eps" }, .outs = &.{"z"}, .source = HC_MIX_SOURCE, .header = HC_HEADER };
var hc_mix_nax_def = Def{ .name = "dsv41_hc_mix_nax", .ins = &.{ "x", "fn", "scale", "base", "eps" }, .outs = &.{"z"}, .source = HC_MIX_NAX_SOURCE, .header = metal_def.NAX_HEADER ++ HC_HEADER };
var hc_stage_def = Def{ .name = "dsv41_hc_stage", .ins = &.{ "x", "cpre", "w", "fn", "scale", "base", "eps" }, .outs = &.{ "h_out", "z" }, .source = HC_STAGE_SOURCE, .header = HC_HEADER };
var hc_stage_x_def = Def{ .name = "dsv41_hc_stage_x", .ins = &.{ "x", "y", "spost", "scomb", "cpre", "w", "fn", "scale", "base", "eps" }, .outs = &.{ "h_out", "z", "x1" }, .source = HC_STAGE_X_SOURCE, .header = HC_HEADER };
var hc_stage_z_def = Def{ .name = "dsv41_hc_stage_z", .ins = &.{ "x", "y", "za", "w", "fn", "scale", "base", "eps" }, .outs = &.{ "h_out", "z", "x1" }, .source = HC_STAGE_Z_SOURCE, .header = HC_HEADER };
var hc_split_def = Def{ .name = "dsv41_hc_split", .ins = &.{ "z", "eps" }, .outs = &.{ "pre", "post", "comb" }, .source = HC_SPLIT_SOURCE };
var hc_norm_def = Def{ .name = "dsv41_hc_norm", .ins = &.{ "z", "x", "cpre", "w", "eps" }, .outs = &.{ "pre", "post", "comb", "h_out" }, .source = HC_NORM_SOURCE };

/// `hcExpand`'s op chain as one pass: `out[k] = post[k] * y + sum_j comb[j, k] * x[j]`. The chain's f32 matmul is
/// TF32-class on a NAX GPU: both operands keep their top 10 mantissa bits (truncated), the four products are exact, and
/// the sum is `(p0 + p1) + (p2 + p3)`; a GPU whose matmul is true f32 fails the probe and keeps the chain. `COPY` sets
/// how a thread maps to output: one element (`k` from the grid's z), or all HC copies of one column.
pub const HC_EXPAND_SOURCE =
    \\#pragma clang fp contract(off)
    \\const int d = int(thread_position_in_grid.x), l = int(thread_position_in_grid.y);
    \\const int D = x_shape[2];
    \\if (d >= D) return;
    \\const size_t xo = size_t(l) * HC * D + d;
    \\float xv[HC];
    \\for (int j = 0; j < HC; ++j) xv[j] = as_type<float>(as_type<uint>(float(x[xo + size_t(j) * D])) & 0xFFFFE000u);
    \\const float yv = float(y[size_t(l) * D + d]);
    \\const int k0 = COPY ? int(thread_position_in_grid.z) : 0, k1 = COPY ? k0 + 1 : HC;
    \\for (int k = k0; k < k1; ++k) {
    \\  float p[HC];
    \\  for (int j = 0; j < HC; ++j) p[j] = as_type<float>(as_type<uint>(comb[l * HC * HC + j * HC + k]) & 0xFFFFE000u) * xv[j];
    \\  out[xo + size_t(k) * D] = T(post[l * HC + k] * yv + ((p[0] + p[1]) + (p[2] + p[3])));
    \\}
;

var hc_expand_def = Def{ .name = "dsv41_hc_expand", .ins = &.{ "x", "y", "post", "comb" }, .outs = &.{"out"}, .source = HC_EXPAND_SOURCE };

/// Threads a hyper-connection threadgroup runs; an arch's probe halves it where the GPU caps register-heavy kernels lower.
pub const hc_tg_widest: usize = 1024;
pub var hc_tg: usize = hc_tg_widest;
/// The row x mix tile a `mix` threadgroup takes. Every tile gives the same bytes (each dot keeps its order); the tile
/// only moves the time, and `hc_mix_tile` is its test seam.
pub const HcMixTile = struct { rows: usize, cols: usize };
pub var hc_mix_tile: HcMixTile = .{ .rows = 2, .cols = 12 };

pub const HcWeights = struct { fn_w: A, scale: A, base: A };
pub const HcSplit = struct { pre: A, post: A, comb: A };
pub const HcStaged = struct { h: A, z: A, x1: A };
/// The previous mixes of a stage that expands: their split, or the mixes `z` it splits itself.
pub const HcPrev = union(enum) { split: HcSplit, z: struct { z: A, iters: usize } };
/// The stream `expand(x, y, previous post, previous comb)` that a stage rebuilds in place of `x` and collapses with the
/// previous pre; `y` in `x`'s dtype.
pub const HcExpand = struct { y: A, prev: HcPrev };

/// A decode-width stage: `h = rms(collapse(x, cpre))` and the mixes `z` `[L, mix]` of `x` (of the expanded stream
/// with `ex`, returned as `x1`). `dt` is `x`'s dtype, which the caller has read to cast `ex.y`: a dtype read is an FFI
/// call on the decode path. The caller frees `h`, `z`, and `x1` when `ex` is set (else `x1` is `x`).
pub fn stage(s: mlx.mlx_stream, dt: mlx.mlx_dtype, x: A, ex: ?HcExpand, cpre: A, norm_w: A, hw: HcWeights, eps: A) !HcStaged {
    const L = dim(x, 0);
    const hc = dim(x, 1);
    const D = dim(x, 2);
    const mix = dim(hw.fn_w, 0);
    const tg = hc_tg;
    const nd = (D + tg - 1) / tg;
    if (hc == 0 or hc > 4 or nd == 0 or nd > 32 or tg < 32 or tg > 1024 or tg % 32 != 0) return error.UnsupportedHcGeometry;
    var o: [3]A = undefined;
    const outs3 = [_]OutSpec{
        .{ .shape = &.{ ci(L), ci(D) }, .dt = dt },
        .{ .shape = &.{ ci(L), ci(mix) }, .dt = .float32 },
        .{ .shape = mlx.getShape(x), .dt = dt },
    };
    const ints = [_]Tmpl{ .{ .name = "TG", .v = ci(tg) }, .{ .name = "HC", .v = ci(hc) }, .{ .name = "ND", .v = ci(nd) } };
    const dts = [_]TDtype{.{ .name = "T", .dt = dt }};
    if (ex) |e| {
        const grid: Grid = .{ (mix + 1) * tg, L, 1 };
        switch (e.prev) {
            .split => |sp| try launch(s, &hc_stage_x_def, &.{ x, e.y, sp.post, sp.comb, sp.pre, norm_w, hw.fn_w, hw.scale, hw.base, eps }, &outs3, grid, .{ tg, 1, 1 }, &ints, &dts, &o),
            .z => |p| {
                if (p.iters == 0) return error.UnsupportedHcGeometry;
                try launch(s, &hc_stage_z_def, &.{ x, e.y, p.z, norm_w, hw.fn_w, hw.scale, hw.base, eps }, &outs3, grid, .{ tg, 1, 1 }, &(ints ++ [_]Tmpl{.{ .name = "ITERS", .v = ci(p.iters) }}), &dts, &o);
            },
        }
        return .{ .h = o[0], .z = o[1], .x1 = o[2] };
    }
    try launch(s, &hc_stage_def, &.{ x, cpre, norm_w, hw.fn_w, hw.scale, hw.base, eps }, outs3[0..2], .{ (mix + 1) * tg, L, 1 }, .{ tg, 1, 1 }, &ints, &dts, o[0..2]);
    return .{ .h = o[0], .z = o[1], .x1 = x };
}

/// The split of mixes `z` `[L, (2 + hc) * hc]` (scale and base already applied).
pub fn splitZ(s: mlx.mlx_stream, z: A, hc: usize, iters: usize, eps: A) !HcSplit {
    const L = dim(z, 0);
    var o: [3]A = undefined;
    try launch(s, &hc_split_def, &.{ z, eps }, &.{
        .{ .shape = &.{ ci(L), ci(hc) }, .dt = .float32 },
        .{ .shape = &.{ ci(L), ci(hc) }, .dt = .float32 },
        .{ .shape = &.{ ci(L), ci(hc), ci(hc) }, .dt = .float32 },
    }, .{ 32, L, 1 }, .{ 32, 1, 1 }, &.{ .{ .name = "HC", .v = ci(hc) }, .{ .name = "ITERS", .v = ci(iters) } }, &.{}, &o);
    return .{ .pre = o[0], .post = o[1], .comb = o[2] };
}

/// The mixes `z` `[L, mix]` of the stream `x` `[L, hc, D]`, a tile of rows and mixes a threadgroup (wide blocks).
pub fn mixTiled(s: mlx.mlx_stream, x: A, hw: HcWeights, eps: A, tile: HcMixTile) !A {
    const L = dim(x, 0);
    const hc = dim(x, 1);
    const mix = dim(hw.fn_w, 0);
    const cols = if (tile.cols != 0 and mix % tile.cols == 0) tile.cols else 1;
    if (tile.rows == 0 or tile.rows > 4 or cols == 0 or cols > 12 or hc == 0 or hc > 8 or hc_tg < 32 or hc_tg % 32 != 0 or tile.rows * cols > hc_tg) return error.UnsupportedHcGeometry;
    var o: [1]A = undefined;
    try launch(s, &hc_mix_def, &.{ x, hw.fn_w, hw.scale, hw.base, eps }, &.{.{ .shape = &.{ ci(L), ci(mix) }, .dt = .float32 }}, .{ mix / cols * hc_tg, (L + tile.rows - 1) / tile.rows, 1 }, .{ hc_tg, 1, 1 }, &.{
        .{ .name = "TG", .v = ci(hc_tg) }, .{ .name = "R", .v = ci(tile.rows) }, .{ .name = "M", .v = ci(cols) }, .{ .name = "HC", .v = ci(hc) },
    }, &.{}, &o);
    return o[0];
}

/// Simdgroups a matrix-unit mix threadgroup splits the flattened stream over.
const hc_mix_nsg = 16;

/// Whether `mixNax` serves `hc` copies of `d` channels into `mix` mixes here: the stream splits into whole 16-wide k
/// steps a simdgroup, a tile's 32 columns hold every mix, and the GPU has the units.
pub fn mixNaxFits(hc: usize, d: usize, mix: usize) bool {
    return (hc * d) % (16 * hc_mix_nsg) == 0 and mix <= 32 and nax_gate.verifyQmmNaxAvailable();
}

/// The tiled mix on the matrix units for a bf16 stream (`mixNaxFits` must hold): the same mixes with the sums in
/// another order. A row's bytes do not depend on the block's width or on where the row sits in it.
pub fn mixNax(s: mlx.mlx_stream, x: A, hw: HcWeights, eps: A) !A {
    const L = dim(x, 0);
    const mix = dim(hw.fn_w, 0);
    const n = dim(hw.fn_w, 1);
    std.debug.assert(mlx.mlx_array_dtype(x) == .bfloat16 and n % (16 * hc_mix_nsg) == 0 and mix <= 32);
    var o: [1]A = undefined;
    try launch(s, &hc_mix_nax_def, &.{ x, hw.fn_w, hw.scale, hw.base, eps }, &.{.{ .shape = &.{ ci(L), ci(mix) }, .dt = .float32 }}, .{ (L + 15) / 16 * hc_mix_nsg * 32, 1, 1 }, .{ hc_mix_nsg * 32, 1, 1 }, &.{
        .{ .name = "NK", .v = ci(n) }, .{ .name = "NSG", .v = hc_mix_nsg }, .{ .name = "MIXN", .v = ci(mix) }, .{ .name = "HC", .v = ci(dim(x, 1)) },
    }, &.{.{ .name = "T", .dt = .bfloat16 }}, &o);
    return o[0];
}

/// The split of `z` and `h = rms(collapse(x, cpre))`, one threadgroup a row.
pub fn norm(s: mlx.mlx_stream, z: A, x: A, cpre: A, norm_w: A, iters: usize, eps: A) !struct { sp: HcSplit, h: A } {
    const L = dim(x, 0);
    const hc = dim(x, 1);
    const D = dim(x, 2);
    const dt = mlx.mlx_array_dtype(x);
    var o: [4]A = undefined;
    try launch(s, &hc_norm_def, &.{ z, x, cpre, norm_w, eps }, &.{
        .{ .shape = &.{ ci(L), ci(hc) }, .dt = .float32 },
        .{ .shape = &.{ ci(L), ci(hc) }, .dt = .float32 },
        .{ .shape = &.{ ci(L), ci(hc), ci(hc) }, .dt = .float32 },
        .{ .shape = &.{ ci(L), ci(D) }, .dt = dt },
    }, .{ hc_tg, L, 1 }, .{ hc_tg, 1, 1 }, &.{ .{ .name = "TG", .v = ci(hc_tg) }, .{ .name = "HC", .v = ci(hc) }, .{ .name = "ITERS", .v = ci(iters) } }, &.{.{ .name = "T", .dt = dt }}, &o);
    return .{ .sp = .{ .pre = o[0], .post = o[1], .comb = o[2] }, .h = o[3] };
}

/// The expand of four copies in one kernel, in `x`'s dtype: a thread an output element (`per_element`), or a column
/// writing all four copies.
pub fn expand(s: mlx.mlx_stream, x: A, y: A, post: A, comb: A, per_element: bool) !A {
    const L = dim(x, 0);
    const D = dim(x, 2);
    const dt = mlx.mlx_array_dtype(x);
    var o: [1]A = undefined;
    const grid: Grid = .{ (D + 255) / 256 * 256, L, if (per_element) 4 else 1 };
    try launch(s, &hc_expand_def, &.{ x, y, post, comb }, &.{.{ .shape = mlx.getShape(x), .dt = dt }}, grid, .{ 256, 1, 1 }, &.{ .{ .name = "HC", .v = 4 }, .{ .name = "COPY", .v = @intFromBool(per_element) } }, &.{.{ .name = "T", .dt = dt }}, &o);
    return o[0];
}

// ── identity comb (iHC) ──────────────────────────────────────────────────────────────────────────────────────────────

/// The constants an identity-comb kernel reads: the rms eps (the mixes' and the norm's), the pre/post eps, and the
/// post's magnitude.
pub fn identityEps(rms_eps: f32, hc_eps: f32, magnitude: f32) !A {
    return mlx.newData(&[_]f32{ rms_eps, hc_eps, magnitude }, &.{3}, .float32);
}

pub const IdentityNormed = struct { pre: A, post: A, h: A };

/// The identity-comb split of row `l` (pre, and post when the mixes carry it) into threadgroup `t_pre` and the outputs.
fn identitySplit(comptime with_post: bool) []const u8 {
    return
        \\#pragma clang fp contract(off)
        \\const int l = threadgroup_position_in_grid.y;
        \\const uint tid = thread_index_in_threadgroup, sg = simdgroup_index_in_threadgroup;
        \\const int D = x_shape[2];
        \\threadgroup float t_pre[8];
        \\if (tid < HC) {
        \\  const float pv = 1.0f / (1.0f + metal::precise::exp(-z[l * ZW + tid])) + eps[1];
        \\  t_pre[tid] = pv;
        \\  pre[l * HC + tid] = pv;
        \\
    ++ (if (with_post)
        \\  post[l * HC + tid] = (1.0f / (1.0f + metal::precise::exp(-z[l * ZW + HC + tid]))) * eps[2] + eps[1];
        \\
    else "") ++
        \\}
        \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        \\float cp[8];
        \\for (int j = 0; j < HC; ++j) cp[j] = t_pre[j];
        \\
    ;
}

/// The collapse by `cp` and the rms norm, as `HC_NORM_SOURCE` takes them.
const HC_COLLAPSE_NORM =
    \\float ss = 0.0f;
    \\for (int d = tid; d < D; d += TG) {
    \\  float h = 0.0f;
    \\  for (int j = 0; j < HC; ++j) h += cp[j] * float(x[(size_t(l) * HC + j) * D + d]);
    \\  const T hb = T(h);
    \\  h_out[size_t(l) * D + d] = hb;
    \\  ss += float(hb) * float(hb);
    \\}
    \\threadgroup float t_s[32];
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

pub const HC_IDENT_NORM_SOURCE = identitySplit(true) ++ HC_COLLAPSE_NORM;
pub const HC_IDENT_HEAD_SOURCE = identitySplit(false) ++ HC_COLLAPSE_NORM;

/// One thread an output element: `out = post·y + h`, a product and a sum each rounded to f32.
pub const HC_IDENT_EXPAND_SOURCE =
    \\#pragma clang fp contract(off)
    \\const int d = int(thread_position_in_grid.x), l = int(thread_position_in_grid.y), k = int(thread_position_in_grid.z);
    \\const int D = h_shape[2];
    \\if (d >= D) return;
    \\const size_t o = (size_t(l) * HC + k) * D + d;
    \\out[o] = post[l * HC + k] * float(y[size_t(l) * D + d]) + float(h[o]);
;

var hc_ident_norm_def = Def{ .name = "hc_ident_norm", .ins = &.{ "z", "x", "w", "eps" }, .outs = &.{ "pre", "post", "h_out" }, .source = HC_IDENT_NORM_SOURCE };
var hc_ident_head_def = Def{ .name = "hc_ident_head", .ins = &.{ "z", "x", "w", "eps" }, .outs = &.{ "pre", "h_out" }, .source = HC_IDENT_HEAD_SOURCE };
var hc_ident_expand_def = Def{ .name = "hc_ident_expand", .ins = &.{ "h", "y", "post" }, .outs = &.{"out"}, .source = HC_IDENT_EXPAND_SOURCE };


/// With no comb: from the mixes `z` [L, 2·HC], `pre = σ(z[:HC]) + ε` and `post = magnitude·σ(z[HC:]) + ε`, and the
/// stream `x` [L, HC, D] collapsed by this `pre` and rms-normed by `norm_w`. The caller frees all three.
pub fn normIdentity(s: mlx.mlx_stream, z: A, x: A, norm_w: A, eps: A) !IdentityNormed {
    const L = dim(x, 0);
    const hc = dim(x, 1);
    var o: [3]A = undefined;
    try launch(s, &hc_ident_norm_def, &.{ z, x, norm_w, eps }, &.{
        .{ .shape = &.{ ci(L), ci(hc) }, .dt = .float32 },
        .{ .shape = &.{ ci(L), ci(hc) }, .dt = .float32 },
        .{ .shape = &.{ ci(L), ci(dim(x, 2)) }, .dt = mlx.mlx_array_dtype(x) },
    }, .{ hc_tg, L, 1 }, .{ hc_tg, 1, 1 }, &identityInts(z, hc), &.{.{ .name = "T", .dt = mlx.mlx_array_dtype(x) }}, &o);
    return .{ .pre = o[0], .post = o[1], .h = o[2] };
}

fn identityInts(z: A, hc: usize) [3]Tmpl {
    return .{ .{ .name = "TG", .v = ci(hc_tg) }, .{ .name = "HC", .v = ci(hc) }, .{ .name = "ZW", .v = ci(dim(z, 1)) } };
}

/// The head's form: `pre = σ(z) + ε` from `z` [L, HC], and `x` collapsed by it and rms-normed. The caller frees both.
pub fn headIdentity(s: mlx.mlx_stream, z: A, x: A, norm_w: A, eps: A) !struct { pre: A, h: A } {
    const L = dim(x, 0);
    const hc = dim(x, 1);
    var o: [2]A = undefined;
    try launch(s, &hc_ident_head_def, &.{ z, x, norm_w, eps }, &.{
        .{ .shape = &.{ ci(L), ci(hc) }, .dt = .float32 },
        .{ .shape = &.{ ci(L), ci(dim(x, 2)) }, .dt = mlx.mlx_array_dtype(x) },
    }, .{ hc_tg, L, 1 }, .{ hc_tg, 1, 1 }, &identityInts(z, hc), &.{.{ .name = "T", .dt = mlx.mlx_array_dtype(x) }}, &o);
    return .{ .pre = o[0], .h = o[1] };
}

/// `h'_c = post_c · y + h_c` in f32, the op chain's two roundings: the expand with no comb.
pub fn expandIdentity(s: mlx.mlx_stream, h: A, y: A, post: A) !A {
    const L = dim(h, 0);
    const hc = dim(h, 1);
    var o: [1]A = undefined;
    try launch(s, &hc_ident_expand_def, &.{ h, y, post }, &.{.{ .shape = mlx.getShape(h), .dt = .float32 }}, .{ (dim(h, 2) + 255) / 256 * 256, L, hc }, .{ 256, 1, 1 }, &.{.{ .name = "HC", .v = ci(hc) }}, &.{}, &o);
    return o[0];
}

/// `src` with its one `old` replaced: a kernel built from another's text.
fn sub(comptime src: []const u8, comptime old: []const u8, comptime new: []const u8) [:0]const u8 {
    const at = std.mem.indexOf(u8, src, old) orelse @compileError("kernel text not found: " ++ old);
    return src[0..at] ++ new ++ src[at + old.len ..] ++ "";
}

/// `HC_MIX_SOURCE` over the stream `post·y + h` it expands as it reads, `HC_IDENT_EXPAND_SOURCE`'s two roundings; the
/// mix tile at x 0 stores it as `m`.
pub const HC_IDENT_MIX_EXPAND_SOURCE = sub(sub(HC_MIX_SOURCE, "L = x_shape[0];", "L = h_shape[0], D = h_shape[2];"),
    "const float v = l0 + r < L ? float(x[size_t(l0 + r) * N + k]) : 0.0f;",
    \\float v = 0.0f;
    \\    if (l0 + r < L) {
    \\      const size_t o = size_t(l0 + r) * N + k;
    \\      v = post[(l0 + r) * HC + k / D] * float(y[size_t(l0 + r) * D + k % D]) + float(h[o]);
    \\      if (threadgroup_position_in_grid.x == 0) m[o] = v;
    \\    }
);

var hc_ident_mix_expand_def = Def{ .name = "hc_ident_mix_expand", .ins = &.{ "h", "y", "post", "fn", "scale", "base", "eps" }, .outs = &.{ "m", "z" }, .source = HC_IDENT_MIX_EXPAND_SOURCE, .header = HC_HEADER };

pub const MixExpanded = struct { m: A, z: A };

/// `expandIdentity(h, y, post)` and `mixTiled` of it in one launch: the expanded stream `m` (f32) and its mixes `z`,
/// each the two launches' bytes. The caller frees both.
pub fn mixExpandIdentity(s: mlx.mlx_stream, h: A, y: A, post: A, hw: HcWeights, eps: A, tile: HcMixTile) !MixExpanded {
    const L = dim(h, 0);
    const hc = dim(h, 1);
    const mix = dim(hw.fn_w, 0);
    const cols = if (tile.cols != 0 and mix % tile.cols == 0) tile.cols else 1;
    if (tile.rows == 0 or tile.rows > 4 or cols == 0 or cols > 12 or hc == 0 or hc > 8 or hc_tg < 32 or hc_tg % 32 != 0 or tile.rows * cols > hc_tg) return error.UnsupportedHcGeometry;
    var o: [2]A = undefined;
    try launch(s, &hc_ident_mix_expand_def, &.{ h, y, post, hw.fn_w, hw.scale, hw.base, eps }, &.{
        .{ .shape = mlx.getShape(h), .dt = .float32 },
        .{ .shape = &.{ ci(L), ci(mix) }, .dt = .float32 },
    }, .{ mix / cols * hc_tg, (L + tile.rows - 1) / tile.rows, 1 }, .{ hc_tg, 1, 1 }, &.{
        .{ .name = "TG", .v = ci(hc_tg) }, .{ .name = "R", .v = ci(tile.rows) }, .{ .name = "M", .v = ci(cols) }, .{ .name = "HC", .v = ci(hc) },
    }, &.{}, &o);
    return .{ .m = o[0], .z = o[1] };
}

/// Launches the identity-comb kernels once at `hc_tg` threads for `hc` streams of `d` and halves it until they run:
/// Metal caps a register-heavy kernel's threadgroup below 1024 on some GPUs (M1, M2).
pub fn probeIdentity(s: mlx.mlx_stream, hc: usize, d: usize) !void {
    defer mlx.expected_refusal = null;
    while (true) {
        // The floor's refusal is the load's error, logged as one.
        mlx.expected_refusal = if (hc_tg > 32) "Thread group size" else null;
        probeIdentityOnce(s, hc, d) catch |e| {
            if (e != error.MlxError or hc_tg <= 32 or !mlx.takeErrorIf("Thread group size")) return e;
            hc_tg /= 2;
            continue;
        };
        return;
    }
}

fn probeIdentityOnce(s: mlx.mlx_stream, hc: usize, d: usize) !void {
    const gpa = std.heap.page_allocator;
    const zeros = try gpa.alloc(f32, 2 * hc * hc * d);
    defer gpa.free(zeros);
    @memset(zeros, 0);
    var held: [10]A = undefined;
    var n: usize = 0;
    defer for (held[0..n]) |a| {
        _ = mlx.mlx_array_free(a);
    };
    const x = try mlx.newData(zeros.ptr, &.{ 1, ci(hc), ci(d) }, .float32);
    held[n] = x;
    n += 1;
    const w: HcWeights = .{ .fn_w = try mlx.newData(zeros.ptr, &.{ ci(2 * hc), ci(hc * d) }, .float32), .scale = try mlx.newData(zeros.ptr, &.{2}, .float32), .base = try mlx.newData(zeros.ptr, &.{ci(2 * hc)}, .float32) };
    held[n..][0..3].* = .{ w.fn_w, w.scale, w.base };
    n += 3;
    const norm_w = try mlx.newData(zeros.ptr, &.{ci(d)}, .float32);
    held[n] = norm_w;
    n += 1;
    const eps = try identityEps(1e-5, 1e-6, 2.0);
    held[n] = eps;
    n += 1;
    const z = try mixTiled(s, x, w, eps, hc_mix_tile);
    held[n] = z;
    n += 1;
    const r = try normIdentity(s, z, x, norm_w, eps);
    held[n..][0..3].* = .{ r.pre, r.post, r.h };
    n += 3;
    for ([_]A{ z, r.pre, r.post, r.h }) |a| try mlx.check(mlx.mlx_array_eval(a));
}

const testing = std.testing;
const Ops = @import("mlx_ops.zig").Ops;

fn testValues(buf: []f32, seed: u64, scale: f32) void {
    var st = seed;
    for (buf) |*v| {
        st = st *% 6364136223846793005 +% 1442695040888963407;
        v.* = (@as(f32, @floatFromInt(@as(u32, @truncate(st >> 32)))) / 2147483648.0 - 1.0) * scale;
    }
}

fn hostF32(a: A) ![]const f32 {
    try mlx.check(mlx.mlx_array_eval(a));
    const p = mlx.mlx_array_data_float32(a) orelse return error.HostRead;
    return p[0..mlx.mlx_array_size(a)];
}

fn sigmoid64(v: f64) f64 {
    return 1.0 / (1.0 + @exp(-v));
}

test "hc identity norm: pre, post (magnitude and eps) and the stream collapsed by its own pre and rms-normed are the f64 truth to f32 rounding, at the layers' 2·HC mixes and the head's HC" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const L = 5;
    const HC = 4;
    const D = 320;
    var zb: [L * 2 * HC]f32 = undefined;
    var xb: [L * HC * D]f32 = undefined;
    var wb: [D]f32 = undefined;
    testValues(&zb, 11, 3.0);
    testValues(&xb, 12, 2.0);
    testValues(&wb, 13, 1.0);
    const rms_eps: f32 = 1e-5;
    const hc_eps: f32 = 1e-6;
    const mag: f32 = 2.0;
    const eps = try identityEps(rms_eps, hc_eps, mag);
    defer _ = mlx.mlx_array_free(eps);
    const x = try mlx.newData(&xb, &.{ L, HC, D }, .float32);
    defer _ = mlx.mlx_array_free(x);
    const w = try mlx.newData(&wb, &.{D}, .float32);
    defer _ = mlx.mlx_array_free(w);
    for ([_]bool{ true, false }) |layer| {
        const zw: usize = if (layer) 2 * HC else HC;
        var zl: [L * 2 * HC]f32 = undefined;
        for (0..L) |l| for (0..zw) |j| {
            zl[l * zw + j] = zb[l * 2 * HC + j];
        };
        const z = try mlx.newData(&zl, &.{ L, @intCast(zw) }, .float32);
        defer _ = mlx.mlx_array_free(z);
        var pre: A = undefined;
        var post: ?A = null;
        var h: A = undefined;
        if (layer) {
            const r = try normIdentity(s, z, x, w, eps);
            pre = r.pre;
            post = r.post;
            h = r.h;
        } else {
            const r = try headIdentity(s, z, x, w, eps);
            pre = r.pre;
            h = r.h;
        }
        defer for ([_]?A{ pre, post, h }) |a| if (a) |v| {
            _ = mlx.mlx_array_free(v);
        };
        const pre_h = try hostF32(pre);
        const h_h = try hostF32(h);
        var worst: f64 = 0;
        var peak: f64 = 0;
        for (0..L) |l| {
            var p64: [HC]f64 = undefined;
            for (0..HC) |c| {
                p64[c] = sigmoid64(zl[l * zw + c]) + hc_eps;
                try testing.expectApproxEqRel(p64[c], @as(f64, pre_h[l * HC + c]), 1e-6);
                if (post) |pa| {
                    const want = @as(f64, mag) * sigmoid64(zl[l * zw + HC + c]) + hc_eps;
                    try testing.expectApproxEqRel(want, @as(f64, (try hostF32(pa))[l * HC + c]), 1e-6);
                }
            }
            var col: [D]f64 = undefined;
            var ss: f64 = 0;
            for (0..D) |d| {
                var acc: f64 = 0;
                for (0..HC) |c| acc += p64[c] * xb[(l * HC + c) * D + d];
                col[d] = acc;
                ss += acc * acc;
            }
            const inv = 1.0 / @sqrt(ss / D + rms_eps);
            for (0..D) |d| {
                const want = col[d] * inv * wb[d];
                peak = @max(peak, @abs(want));
                worst = @max(worst, @abs(want - h_h[l * D + d]));
            }
        }
        try testing.expect(worst <= 4e-6 * peak);
    }
}

test "hc identity expand: post·y + h is the op chain's bytes" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const L = 3;
    const HC = 4;
    const D = 200;
    var hb: [L * HC * D]f32 = undefined;
    var yb: [L * D]f32 = undefined;
    var pb: [L * HC]f32 = undefined;
    testValues(&hb, 21, 4.0);
    testValues(&yb, 22, 3.0);
    testValues(&pb, 23, 2.0);
    const h = try mlx.newData(&hb, &.{ L, HC, D }, .float32);
    defer _ = mlx.mlx_array_free(h);
    const post = try mlx.newData(&pb, &.{ L, HC }, .float32);
    defer _ = mlx.mlx_array_free(post);
    for ([_]mlx.mlx_dtype{ .float32, .bfloat16 }) |ydt| {
        var o: Ops = .{ .s = s };
        defer o.deinit();
        const y = try o.cast(try o.own(try mlx.newData(&yb, &.{ L, D }, .float32)), ydt);
        const want = try o.binary(.add, try o.binary(.mul, try o.reshape(post, &.{ L, HC, 1 }), try o.reshape(try o.cast(y, .float32), &.{ L, 1, D })), h);
        const got = try o.own(try expandIdentity(s, h, y, post));
        try testing.expectEqualSlices(f32, try hostF32(want), try hostF32(got));
    }
}

test "hc identity expand into the mix: one launch writes post·y + h and its mixes, the expand's and the mix's bytes, at Hy4's streams, by tile, on MLX's stream and the engine's" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    defer mlx.engine.enabled = was;
    for ([_]bool{ false, true }) |on_engine| {
        mlx.engine.enabled = on_engine;
        try mixExpandCase(if (on_engine) mlx.engine.stream() else mlx.gpuStream());
    }
}

fn mixExpandCase(s: mlx.mlx_stream) !void {
    const HC = 4;
    const D = 3072;
    const MIX = 2 * HC;
    const a = testing.allocator;
    const fnb = try a.alloc(f32, MIX * HC * D);
    defer a.free(fnb);
    testValues(fnb, 31, 0.05);
    var sb = [_]f32{ 0.7, 1.3 };
    var bb: [MIX]f32 = undefined;
    testValues(&bb, 32, 1.0);
    const fn_w = try mlx.newData(fnb.ptr, &.{ MIX, HC * D }, .float32);
    defer _ = mlx.mlx_array_free(fn_w);
    const scale = try mlx.newData(&sb, &.{2}, .float32);
    defer _ = mlx.mlx_array_free(scale);
    const base = try mlx.newData(&bb, &.{MIX}, .float32);
    defer _ = mlx.mlx_array_free(base);
    const eps = try identityEps(1e-5, 1e-6, 2.0);
    defer _ = mlx.mlx_array_free(eps);
    const hw: HcWeights = .{ .fn_w = fn_w, .scale = scale, .base = base };
    for ([_]usize{ 1, 3 }) |L| for ([_]mlx.mlx_dtype{ .float32, .bfloat16 }) |ydt| for ([_]HcMixTile{ .{ .rows = 2, .cols = 12 }, .{ .rows = 1, .cols = 8 } }) |tile| {
        const hb = try a.alloc(f32, L * HC * D);
        defer a.free(hb);
        const yb = try a.alloc(f32, L * D);
        defer a.free(yb);
        const pb = try a.alloc(f32, L * HC);
        defer a.free(pb);
        testValues(hb, 33 + L, 4.0);
        testValues(yb, 34 + L, 3.0);
        testValues(pb, 35 + L, 2.0);
        var o: Ops = .{ .s = s };
        defer o.deinit();
        const h = try o.own(try mlx.newData(hb.ptr, &.{ @intCast(L), HC, D }, .float32));
        const post = try o.own(try mlx.newData(pb.ptr, &.{ @intCast(L), HC }, .float32));
        const y = try o.cast(try o.own(try mlx.newData(yb.ptr, &.{ @intCast(L), D }, .float32)), ydt);
        const m = try o.own(try expandIdentity(s, h, y, post));
        const z = try o.own(try mixTiled(s, m, hw, eps, tile));
        const got = try mixExpandIdentity(s, h, y, post, hw, eps, tile);
        _ = try o.own(got.m);
        _ = try o.own(got.z);
        try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(try hostF32(m)), std.mem.sliceAsBytes(try hostF32(got.m)));
        try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(try hostF32(z)), std.mem.sliceAsBytes(try hostF32(got.z)));
    };
}

test "hc identity probe: a threadgroup above the GPU's cap is halved until the identity kernels run" {
    try mlx.requireGpu();
    const saved = hc_tg;
    defer hc_tg = saved;
    hc_tg = 4096;
    try probeIdentity(mlx.gpuStream(), 4, 256);
    try testing.expect(hc_tg >= 32 and hc_tg < 4096);
}
