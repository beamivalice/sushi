//! Flash-Next quantized sparse attention: block pooling, scoring and top-block selection.

const trf = @import("../transformer.zig");
const qwen4_qsa_gather = @import("qwen4_qsa_gather.zig");
const Transformer = trf.Transformer;
const std = @import("std");
const mlx = @import("../mlx.zig");
const qwen4_forward = @import("qwen4_forward.zig");
const testing = std.testing;
const test_support = @import("../test_support.zig");
const attn256MaxDiff = test_support.attn256MaxDiff;

const DenseKVView = trf.DenseKVView;
const FUSED256_MIN_Q_LEN = trf.FUSED256_MIN_Q_LEN;
const ForwardCtx = trf.ForwardCtx;
const FullAttnWeights = trf.FullAttnWeights;
const KVCache = trf.KVCache;
const KvPrefixView = trf.KvPrefixView;
const OneShotBits = trf.OneShotBits;
const ProfClock = trf.ProfClock;
const QSA_PACKED_GATHER_MAX_REUSE = trf.QSA_PACKED_GATHER_MAX_REUSE;
const QsaArm = trf.QsaArm;
const QsaSelectCfgKey = trf.QsaSelectCfgKey;
const QsaSelectSplitCfgKey = trf.QsaSelectSplitCfgKey;
const SSMCacheEntry = trf.SSMCacheEntry;
const capBufAppend = trf.capBufAppend;
const diagEnvOnCached = trf.diagEnvOnCached;
const diagEnvValueOn = trf.diagEnvValueOn;
const fusedSdpa256Masked = trf.fusedSdpa256Masked;
const gatherQsa256 = trf.gatherQsa256;
const gatherQsa256Packed = trf.gatherQsa256Packed;
const getQsaSelectKernel = qwen4_forward.getQsaSelectKernel;
const getQsaSelectSplitLocalKernel = qwen4_forward.getQsaSelectSplitLocalKernel;
const getQsaSelectSplitMergeKernel = qwen4_forward.getQsaSelectSplitMergeKernel;
const log = trf.log;
const markPosOf = qwen4_forward.markPosOf;
const qsaAttnKernelEnabled = trf.qsaAttnKernelEnabled;
const qsaBatchedGatherEnabled = trf.qsaBatchedGatherEnabled;
const qsaDecodeGatherAttn = trf.qsaDecodeGatherAttn;
const qsaDecodeGatherEnabled = trf.qsaDecodeGatherEnabled;
const qsaEntrySatisfiesForward = trf.qsaEntrySatisfiesForward;
const qsaGatherEnabled = trf.qsaGatherEnabled;
const qsaGatherMinKv = trf.qsaGatherMinKv;
const qsaMaskFromBlockSel = trf.qsaMaskFromBlockSel;
const qsaMaskFromBlocks = trf.qsaMaskFromBlocks;
const qsaNaxEnabled = trf.qsaNaxEnabled;
const qsaNaxOsOk = trf.qsaNaxOsOk;
const qsaPoolNormRopeFused = trf.qsaPoolNormRopeFused;
const qsaPrefillGatherMinKvFrom = trf.qsaPrefillGatherMinKvFrom;
const qsaScoreFusedActiveFor = trf.qsaScoreFusedActiveFor;
const qsaScoreFusedArm = trf.qsaScoreFusedArm;
const qsaScoreFusedDispatch = trf.qsaScoreFusedDispatch;
const qsaScoreRowsPerChunkFused = trf.qsaScoreRowsPerChunkFused;
const qsaScoreSheetComposed = trf.qsaScoreSheetComposed;
const qsaSelectKernelEnabled = trf.qsaSelectKernelEnabled;
const qsaSelectSplitEnabled = trf.qsaSelectSplitEnabled;
const qsaSparseAttn = trf.qsaSparseAttn;
const qsaSparseAttnServes = trf.qsaSparseAttnServes;
const qsaVerifyGatherAttn = trf.qsaVerifyGatherAttn;
const qsaVerifyGatherEnabled = trf.qsaVerifyGatherEnabled;
const qsaVerifyGatherMinKvFor = trf.qsaVerifyGatherMinKvFor;
const qsaWidthBucket = trf.qsaWidthBucket;
const qwen4Standin = qwen4_forward.qwen4Standin;
const sliceAttentionSeq = trf.sliceAttentionSeq;
const spanPreservingDropIndex = trf.spanPreservingDropIndex;
const splitMaskedSdpa256 = trf.splitMaskedSdpa256;
const verifyQmmNaxAvailable = trf.verifyQmmNaxAvailable;

pub fn qsaPrefillGatherMinKv() c_int {
    if (qwen4_qsa_gather.qsa_gather_min_kv_override != null) return qsaGatherMinKv();
    const serving = qsaNaxEnabled() and qsaNaxOsOk() and verifyQmmNaxAvailable();
    return qsaPrefillGatherMinKvFrom(FUSED256_MIN_Q_LEN, serving, null);
}

pub var qsa_batched_gather_logged: bool = false;

// ── Fused QSA top-k block select (sushi_qsa_select) ──
//
// The composed arm is ~10 dependent dispatches per attention layer (~0.31 ms/layer at kv
// 62.7k); this kernel is one dispatch: one threadgroup per query row, an MSD radix select
// over the f32 score's monotone ordinal, then an index-ordered compaction. Semantics are
// `qsaChunkSelect`'s: top-K visible blocks per row, ties to the LOWER index (torch.topk),
// emitted ascending, INT_MAX past the row's count. The tie rule is exact where the composed
// arm's `b * 1e-7` bias only approximates it.
pub const QSA_SELECT_KERNEL_HEADER =
    \\// Monotone f32 -> uint32: ascending order preserved, NaN above every
    \\// number, -0.0 and +0.0 the SAME key (torch compares them equal, and a
    \\// relu sum can produce either).
    \\inline uint sushi_qsa_ord(float v) {
    \\  if (metal::isnan(v)) { return 0xFFFFFFFFu; }
    \\  if (v == 0.0f) { return 0x80000000u; }
    \\  uint u = as_type<uint>(v);
    \\  return (u & 0x80000000u) ? (~u) : (u | 0x80000000u);
    \\}
    \\
;

/// One threadgroup per query row. 1. MSD radix select over the 32-bit ordinal (the DIGITS
/// template picks 11/11/10-bit digits at prefill widths or four 8-bit digits at decode widths,
/// where each level's fixed cost, not the walk, is the bill) counting only visible blocks,
/// stopping early when a bin's count is exactly what is still needed. 2. Compaction: selected
/// iff `ord > T`, or `ord == T` with rank below `need_eq`; the output slot is the count of
/// selected elements with a smaller index (a simd prefix scan per chunk), so the output is
/// born sorted. 3. Rows with `bounds[row] <= K` are `0 .. bounds-1` then INT_MAX.
/// Threadgroup memory ~8.3 KiB at 2048 bins, ~1.3 KiB at 256.
pub const QSA_SELECT_SHARED =
    \\constexpr uint TGN   = (uint)TGS;
    \\constexpr uint SIMDW = 32u;
    \\constexpr uint NSIMD = TGN / SIMDW;
    \\constexpr uint BINS  = (DIGITS == 8) ? 256u : 2048u;
    \\constexpr uint NLEV  = (DIGITS == 8) ? 4u : 3u;
    \\constexpr uint KTOP  = (uint)K;
    \\constexpr int  SENTINEL = 2147483647;
    \\static_assert(NSIMD <= SIMDW);
    \\
    \\threadgroup metal::atomic_uint hist[BINS];
    \\threadgroup uint sgs[2u * NSIMD];
    \\threadgroup uint sh[4];
    \\
    \\const uint row  = threadgroup_position_in_grid.y;
    \\const uint tid  = thread_position_in_threadgroup.x;
    \\const uint lane = thread_index_in_simdgroup;
    \\const uint sg   = simdgroup_index_in_threadgroup;
    \\
;

// One radix-select body composed into three kernels (single row, row slice, merge) so a
// fix to the digit walk or the tie rule lands once; the kernels differ only in setup.
pub const QSA_SELECT_RADIX_BODY =
    \\uint pref = 0u;
    \\uint fixed = 0u;
    \\uint need = KTOP;
    \\uint T = 0u;
    \\uint need_eq = 0u;
    \\
    \\for (uint lv = 0u; lv < NLEV; ++lv) {
    \\  const uint width = (DIGITS == 8) ? 8u : ((lv == 2u) ? 10u : 11u);
    \\  const uint nbins = 1u << width;
    \\  const uint shift = 32u - fixed - width;
    \\  const uint hi_b  = (fixed == 0u) ? 31u : (32u - fixed);
    \\
    \\  for (uint b = tid; b < nbins; b += TGN) metal::atomic_store_explicit(&hist[b], 0u, metal::memory_order_relaxed);
    \\  if (tid == 0u) { sh[0] = 0u; sh[1] = 0u; sh[2] = 0u; }
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\
    \\  uint last_d = 0xFFFFFFFFu;
    \\  uint run = 0u;
    \\  for (uint base = SEL_LO; base < SEL_HI; base += TGN) {
    \\    const uint i = base + tid;
    \\    uint u = 0u;
    \\    int out_idx = SENTINEL;
    \\    uint ok = 0u;
    \\    SEL_LOAD(i, u, out_idx, ok)
    \\    if (ok != 0u) {
    \\      if (!(fixed != 0u && (u >> hi_b) != pref)) {
    \\        const uint d = (u >> shift) & (nbins - 1u);
    \\        if (d == last_d) { run += 1u; }
    \\        else {
    \\          if (run != 0u) metal::atomic_fetch_add_explicit(&hist[last_d], run, metal::memory_order_relaxed);
    \\          last_d = d;
    \\          run = 1u;
    \\        }
    \\      }
    \\    }
    \\  }
    \\  if (run != 0u) metal::atomic_fetch_add_explicit(&hist[last_d], run, metal::memory_order_relaxed);
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\
    \\  if (sg == 0u) {
    \\    const uint chunk = nbins / SIMDW;
    \\    const uint base  = nbins - (lane + 1u) * chunk;
    \\    uint tot = 0u;
    \\    for (uint j = 0u; j < chunk; ++j) tot += metal::atomic_load_explicit(&hist[base + j], metal::memory_order_relaxed);
    \\    const uint pre = metal::simd_prefix_exclusive_sum(tot);
    \\    if (pre < need && need <= pre + tot) {
    \\      uint acc = pre;
    \\      for (uint jj = chunk; jj > 0u; --jj) {
    \\        const uint bidx = base + jj - 1u;
    \\        const uint c = metal::atomic_load_explicit(&hist[bidx], metal::memory_order_relaxed);
    \\        if (acc + c >= need) { sh[0] = bidx; sh[1] = acc; sh[2] = c; break; }
    \\        acc += c;
    \\      }
    \\    }
    \\  }
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  const uint d_sel   = sh[0];
    \\  const uint above_w = sh[1];
    \\  const uint cnt_d   = sh[2];
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\
    \\  need = need - above_w;
    \\  pref = (pref << width) | d_sel;
    \\  fixed += width;
    \\  if (cnt_d == need || fixed >= 32u) {
    \\    T = (fixed >= 32u) ? pref : (pref << (32u - fixed));
    \\    need_eq = need;
    \\    break;
    \\  }
    \\}
    \\
    \\for (uint base = 0u; base < KTOP; base += TGN) {
    \\  const uint i = base + tid;
    \\  if (i < KTOP) outp[i] = SENTINEL;
    \\}
    \\threadgroup_barrier(metal::mem_flags::mem_device);
    \\
    \\uint run_gt = 0u;
    \\uint run_eq = 0u;
    \\for (uint base = SEL_LO; base < SEL_HI; base += TGN) {
    \\  const uint i = base + tid;
    \\  uint gtf = 0u;
    \\  uint e = 0u;
    \\  int out_idx = SENTINEL;
    \\  uint u = 0u;
    \\  uint ok = 0u;
    \\  SEL_LOAD(i, u, out_idx, ok)
    \\  if (ok != 0u) {
    \\    gtf = (u > T) ? 1u : 0u;
    \\    e = (u == T) ? 1u : 0u;
    \\  }
    \\  const uint pg = metal::simd_prefix_exclusive_sum(gtf);
    \\  const uint pe = metal::simd_prefix_exclusive_sum(e);
    \\  const uint sg_g = metal::simd_sum(gtf);
    \\  const uint sg_e = metal::simd_sum(e);
    \\  if (lane == 0u) { sgs[sg] = sg_g; sgs[NSIMD + sg] = sg_e; }
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  uint off_g = 0u, off_e = 0u, tot_g = 0u, tot_e = 0u;
    \\  {
    \\    const uint a = (lane < NSIMD) ? sgs[lane] : 0u;
    \\    const uint b = (lane < NSIMD) ? sgs[NSIMD + lane] : 0u;
    \\    const uint pa = metal::simd_prefix_exclusive_sum(a);
    \\    const uint pb = metal::simd_prefix_exclusive_sum(b);
    \\    off_g = metal::simd_shuffle(pa, sg);
    \\    off_e = metal::simd_shuffle(pb, sg);
    \\    tot_g = metal::simd_sum(a);
    \\    tot_e = metal::simd_sum(b);
    \\  }
    \\  const uint gb = run_gt + off_g + pg;
    \\  const uint eb = run_eq + off_e + pe;
    \\  if (ok != 0u) {
    \\    if (gtf != 0u) {
    \\      const uint pos = gb + metal::min(eb, need_eq);
    \\      if (pos < KTOP) outp[pos] = out_idx;
    \\    } else if (e != 0u && eb < need_eq) {
    \\      const uint pos = gb + eb;
    \\      if (pos < KTOP) outp[pos] = out_idx;
    \\    }
    \\  }
    \\  run_gt += tot_g;
    \\  run_eq += tot_e;
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  if (run_gt + metal::min(run_eq, need_eq) >= KTOP) break;
    \\}
;

pub const QSA_SELECT_SINGLE_SETUP =
    \\const uint nb = (uint)scores_shape[2];
    \\const int  vbi = bounds[row];
    \\const uint vb  = (vbi > 0) ? (uint)vbi : 0u;
    \\const device float* sc = scores + (ulong)row * (ulong)nb;
    \\device int* outp = ids + (ulong)row * (ulong)KTOP;
    \\if (vb <= KTOP) {
    \\  for (uint i = tid; i < KTOP; i += TGN) outp[i] = (i < vb) ? int(i) : SENTINEL;
    \\  return;
    \\}
    \\#define SEL_LO 0u
    \\#define SEL_HI vb
    \\#define SEL_LOAD(i, u, out_idx, ok) { ok = 0u; out_idx = SENTINEL; if ((i) < SEL_HI) { out_idx = int(i); u = sushi_qsa_ord(sc[i]); ok = 1u; } }
    \\
;

pub const QSA_SELECT_SLICE_SETUP =
    \\const uint g    = threadgroup_position_in_grid.x;
    \\const uint G    = threadgroups_per_grid.x;
    \\const uint nb = (uint)scores_shape[2];
    \\const int  vbi = bounds[row];
    \\const uint vb  = (vbi > 0) ? (uint)vbi : 0u;
    \\const uint lo  = (uint)(((ulong)g * (ulong)vb) / (ulong)G);
    \\const uint hi  = (uint)(((ulong)(g + 1u) * (ulong)vb) / (ulong)G);
    \\const uint nsl = hi - lo;
    \\const device float* sc = scores + (ulong)row * (ulong)nb;
    \\device int* outp = ids + (((ulong)row * (ulong)G) + (ulong)g) * (ulong)KTOP;
    \\if (nsl <= KTOP) {
    \\  for (uint i = tid; i < KTOP; i += TGN) outp[i] = (i < nsl) ? int(lo + i) : SENTINEL;
    \\  return;
    \\}
    \\#define SEL_LO lo
    \\#define SEL_HI hi
    \\#define SEL_LOAD(i, u, out_idx, ok) { ok = 0u; out_idx = SENTINEL; if ((i) < SEL_HI) { out_idx = int(i); u = sushi_qsa_ord(sc[i]); ok = 1u; } }
    \\
;

// Candidates arrive ascending by ORIGINAL index with SENTINEL slots skipped, never sorted
// in; that is what resolves a cross-slice tie at the K-th ordinal to the lowest index. Every
// simd reduction below runs in a uniform base-stepped loop: a divergent simd_sum is undefined.
pub const QSA_SELECT_MERGE_SETUP =
    \\const uint nb = (uint)scores_shape[2];
    \\const uint nc = (uint)local_ids_shape[2];
    \\const device float* sc = scores + (ulong)row * (ulong)nb;
    \\const device int* loc = local_ids + (ulong)row * (ulong)nc;
    \\device int* outp = ids + (ulong)row * (ulong)KTOP;
    \\
    \\uint nvalid = 0u;
    \\for (uint base = 0u; base < nc; base += TGN) {
    \\  const uint i = base + tid;
    \\  uint v = 0u;
    \\  if (i < nc) v = (loc[i] != SENTINEL) ? 1u : 0u;
    \\  nvalid += metal::simd_sum(v);
    \\}
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\if (lane == 0u) sgs[sg] = nvalid;
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\{
    \\  const uint a = (lane < NSIMD) ? sgs[lane] : 0u;
    \\  nvalid = metal::simd_sum(a);
    \\}
    \\
    \\if (nvalid <= KTOP) {
    \\  for (uint i = tid; i < KTOP; i += TGN) outp[i] = SENTINEL;
    \\  threadgroup_barrier(metal::mem_flags::mem_device);
    \\  uint run = 0u;
    \\  for (uint base = 0u; base < nc; base += TGN) {
    \\    const uint i = base + tid;
    \\    uint gtf = 0u;
    \\    int idx = SENTINEL;
    \\    if (i < nc) {
    \\      idx = loc[i];
    \\      gtf = (idx != SENTINEL) ? 1u : 0u;
    \\    }
    \\    const uint pg = metal::simd_prefix_exclusive_sum(gtf);
    \\    const uint sg_g = metal::simd_sum(gtf);
    \\    if (lane == 0u) sgs[sg] = sg_g;
    \\    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\    uint off_g = 0u, tot_g = 0u;
    \\    {
    \\      const uint a = (lane < NSIMD) ? sgs[lane] : 0u;
    \\      const uint pa = metal::simd_prefix_exclusive_sum(a);
    \\      off_g = metal::simd_shuffle(pa, sg);
    \\      tot_g = metal::simd_sum(a);
    \\    }
    \\    const uint pos = run + off_g + pg;
    \\    if (gtf != 0u && pos < KTOP) outp[pos] = idx;
    \\    run += tot_g;
    \\    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  }
    \\  return;
    \\}
    \\#define SEL_LO 0u
    \\#define SEL_HI nc
    \\#define SEL_LOAD(i, u, out_idx, ok) { ok = 0u; out_idx = SENTINEL; if ((i) < SEL_HI) { out_idx = loc[i]; if (out_idx != SENTINEL) { u = sushi_qsa_ord(sc[out_idx]); ok = 1u; } } }
    \\
;

pub const QSA_SELECT_KERNEL_SOURCE = std.fmt.comptimePrint("{s}{s}{s}", .{ QSA_SELECT_SHARED, QSA_SELECT_SINGLE_SETUP, QSA_SELECT_RADIX_BODY });

pub const QSA_SELECT_SPLIT_LOCAL_SOURCE = std.fmt.comptimePrint("{s}{s}{s}", .{ QSA_SELECT_SHARED, QSA_SELECT_SLICE_SETUP, QSA_SELECT_RADIX_BODY });

pub const QSA_SELECT_SPLIT_MERGE_SOURCE = std.fmt.comptimePrint("{s}{s}{s}", .{ QSA_SELECT_SHARED, QSA_SELECT_MERGE_SETUP, QSA_SELECT_RADIX_BODY });

/// Threads per row-threadgroup.
pub const QSA_SELECT_TG_DEFAULT: c_int = 1024;

// A row's threadgroup is a per-shape choice: the wide passes of a prefill chunk run more
// rows concurrently at 256 threads, a narrow pass wants 512, and the decode widths keep
// 1024 (the split kernels size their own slices). Outputs are identical at every width.
pub fn qsaSelectTgFor(rows: c_int) c_int {
    if (rows >= 256) return 256;
    if (rows >= 64) return 512;
    return QSA_SELECT_TG_DEFAULT;
}

pub var qsa_select_kernel_cached: ?mlx.mlx_fast_metal_kernel = null;

pub var qsa_select_engaged_bits: OneShotBits = .{};

pub var qsa_index_score_graphs: usize = 0;

pub var qsa_index_select_graphs: usize = 0;

/// Per-row first-not-fully-visible block index (`b < (p+1)/ratio`). `all_vis` is the caller's
/// own `qsaAllBlocksVisible` claim, honored verbatim so both arms answer the same question.
pub fn qsaVisibleBoundsHost(out: []i32, row0: c_int, rows: c_int, nb: c_int, ratio: c_int, all_vis: bool) void {
    const n: usize = @intCast(rows);
    var r: usize = 0;
    while (r < n) : (r += 1) {
        if (all_vis) {
            out[r] = nb;
            continue;
        }
        const p: c_int = row0 + @as(c_int, @intCast(r));
        const complete: c_int = if (p < 0) 0 else @divTrunc(p + 1, ratio);
        out[r] = @min(nb, complete);
    }
}

/// Fused replacement for the composed selection tail: f32 `[1, rows, nb]` scores + int32
/// `[rows]` bounds -> sorted int32 `[1, rows, kb]` selection. Null = declined.
pub fn qsaSelectTopBlocks(
    s: mlx.mlx_stream,
    scores: mlx.mlx_array,
    bounds: mlx.mlx_array,
    kb: c_int,
) !?mlx.mlx_array {
    if (!qsaSelectKernelEnabled()) return null;
    if (!mlx.streamIsGpu(s)) return null;
    if (scores.ctx == null or bounds.ctx == null) return null;
    if (mlx.mlx_array_dtype(scores) != .float32 or mlx.mlx_array_dtype(bounds) != .int32) return null;
    if (mlx.mlx_array_ndim(scores) != 3 or mlx.mlx_array_ndim(bounds) != 1) return null;
    const ssh = mlx.getShape(scores);
    const bsh = mlx.getShape(bounds);
    if (ssh[0] != 1) return null;
    const rows = ssh[1];
    const nb = ssh[2];
    if (rows <= 0 or nb <= 0 or bsh[0] != rows) return null;
    if (kb <= 0 or kb > nb) return null;

    qsa_select_used_split = false;
    if (qsaSelectSplitEnabled() and qsaSelectSplitServes(rows, nb, kb)) {
        if (try qsaSelectTopBlocksSplit(s, scores, bounds, kb, rows, nb)) |picks| {
            qsa_select_used_split = true;
            return picks;
        }
    }

    const kernel = getQsaSelectKernel() catch return null;
    const tg = qsaSelectTgFor(rows);
    const digits = qsaSelectDigitBits(rows);
    const key = QsaSelectCfgKey{ .rows = rows, .k = kb, .tg = tg, .digits = digits };
    const sel_cfgs: [1]mlx.mlx_fast_metal_kernel_config = trf.qsa_select_cfgs.get(key) orelse blk: {
        const config = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(config);
        const o_shape = [_]c_int{ 1, rows, kb };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &o_shape, 3, .int32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, tg, rows, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, tg, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "TGS", tg));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "K", kb));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "DIGITS", digits));
        trf.qsa_select_cfgs.put(key, .{config});
        break :blk .{config};
    };

    const inputs_arr = [_]mlx.mlx_array{ scores, bounds };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, sel_cfgs[0], s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 0));
    qsa_index_select_graphs += 1;
    if (qsa_select_engaged_bits.take(qsaWidthBucket(rows))) {
        log.info(
            "[qsa-select] engaged (S={d} nb={d} k={d} tg={d})\n",
            .{ rows, nb, kb, tg },
        );
    }
    return out;
}

// One threadgroup per row is latency-bound at decode widths (0.72 ms at 215k blocks; ~20x
// under the bandwidth floor), so rows split over G slices, each running the exact select,
// and a second dispatch selects over the G*K candidates. Exact because the order is total
// (ordinal, then lowest original index): every global top-K element is in its slice's
// top-K. The floor is the crossover of the marginal in-graph cost (probe: the split pair
// beats the single kernel from ~24k blocks once the compaction scan and 8-bit digits landed;
// live, +4% at 40k blocks and +18% serial at 200k); 2*G*K <= nb (16k blocks at G=16) keeps
// the candidate list smaller than the row it replaces and is the design's hard minimum.
pub const QSA_SELECT_SPLIT_MIN_NB: c_int = 24576;

pub const QSA_SELECT_SPLIT_MAX_ROWS: c_int = 15;

pub const QSA_SELECT_SPLIT_G: c_int = 16;

pub var qsa_select_digits_override: ?c_int = null;

pub fn qsaSelectDigitBits(rows: c_int) c_int {
    if (qsa_select_digits_override) |v| return v;
    if (rows <= QSA_SELECT_SPLIT_MAX_ROWS) return 8;
    return 11;
}

pub var qsa_select_used_split: bool = false;

pub fn qsaSelectSplitServes(rows: c_int, nb: c_int, kb: c_int) bool {
    if (rows < 1 or rows > QSA_SELECT_SPLIT_MAX_ROWS) return false;
    if (nb < QSA_SELECT_SPLIT_MIN_NB) return false;
    if (kb <= 0 or kb > nb) return false;
    const span = std.math.mul(c_int, kb, QSA_SELECT_SPLIT_G) catch return false;
    const twice = std.math.mul(c_int, span, 2) catch return false;
    return twice <= nb;
}

pub var qsa_select_split_local_kernel_cached: ?mlx.mlx_fast_metal_kernel = null;

pub var qsa_select_split_merge_kernel_cached: ?mlx.mlx_fast_metal_kernel = null;

pub var qsa_select_split_engaged_bits: OneShotBits = .{};

pub fn qsaSelectTopBlocksSplit(
    s: mlx.mlx_stream,
    scores: mlx.mlx_array,
    bounds: mlx.mlx_array,
    kb: c_int,
    rows: c_int,
    nb: c_int,
) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return null;
    const g = QSA_SELECT_SPLIT_G;
    const loc_count = std.math.mul(c_int, kb, g) catch return null;
    const local_kernel = getQsaSelectSplitLocalKernel() catch return null;
    const merge_kernel = getQsaSelectSplitMergeKernel() catch return null;
    const slice_tg: c_int = 512;
    const merge_tg: c_int = QSA_SELECT_TG_DEFAULT;
    const digits = qsaSelectDigitBits(rows);
    const key = QsaSelectSplitCfgKey{ .rows = rows, .k = kb, .slice_tg = slice_tg, .merge_tg = merge_tg, .digits = digits };
    const cfgs: [2]mlx.mlx_fast_metal_kernel_config = trf.qsa_select_split_cfgs.get(key) orelse blk: {
        const local_cfg = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(local_cfg);
        const local_shape = [_]c_int{ 1, rows, loc_count };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(local_cfg, &local_shape, 3, .int32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(local_cfg, slice_tg * g, rows, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(local_cfg, slice_tg, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(local_cfg, "TGS", slice_tg));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(local_cfg, "K", kb));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(local_cfg, "DIGITS", digits));
        const merge_cfg = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(merge_cfg);
        const o_shape = [_]c_int{ 1, rows, kb };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(merge_cfg, &o_shape, 3, .int32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(merge_cfg, merge_tg, rows, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(merge_cfg, merge_tg, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(merge_cfg, "TGS", merge_tg));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(merge_cfg, "K", kb));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(merge_cfg, "DIGITS", digits));
        trf.qsa_select_split_cfgs.put(key, .{ local_cfg, merge_cfg });
        break :blk .{ local_cfg, merge_cfg };
    };

    const local_in = [_]mlx.mlx_array{ scores, bounds };
    const local_in_vec = mlx.mlx_vector_array_new_data(&local_in, local_in.len);
    defer _ = mlx.mlx_vector_array_free(local_in_vec);
    var local_out_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(local_out_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&local_out_vec, local_kernel, local_in_vec, cfgs[0], s));
    if (mlx.mlx_vector_array_size(local_out_vec) != 1) return error.MetalKernelBadOutputCount;
    var local_ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(local_ids);
    try mlx.check(mlx.mlx_vector_array_get(&local_ids, local_out_vec, 0));

    const merge_in = [_]mlx.mlx_array{ scores, local_ids };
    const merge_in_vec = mlx.mlx_vector_array_new_data(&merge_in, merge_in.len);
    defer _ = mlx.mlx_vector_array_free(merge_in_vec);
    var merge_out_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(merge_out_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&merge_out_vec, merge_kernel, merge_in_vec, cfgs[1], s));
    if (mlx.mlx_vector_array_size(merge_out_vec) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&out, merge_out_vec, 0));
    if (qsa_select_split_engaged_bits.take(qsaWidthBucket(rows))) {
        log.info(
            "[qsa-select] engaged (split G={d} S={d} nb={d} k={d} tg={d})\n",
            .{ g, rows, nb, kb, slice_tg },
        );
    }
    return out;
}

pub var qsa_dump_blocks_path_override: ?[]const u8 = null;

pub var qsa_dump_blocks_cached: ?[]const u8 = null;

pub var qsa_dump_blocks_read: bool = false;

pub fn qsaScoreFusedEligible(batch: c_int, n_idx: c_int, idx_hd: c_int, q_dt: mlx.mlx_dtype, pooled_dt: mlx.mlx_dtype) bool {
    return qsaScoreFusedActiveFor(batch, n_idx, idx_hd) and
        q_dt == .bfloat16 and pooled_dt == .bfloat16;
}

pub fn qsaScoreFusedEligibleFrom(q: mlx.mlx_array, pooled: mlx.mlx_array) bool {
    if (q.ctx == null or pooled.ctx == null) return false;
    if (mlx.mlx_array_ndim(q) != 4 or mlx.mlx_array_ndim(pooled) != 3) return false;
    const qs = mlx.getShape(q);
    const ps = mlx.getShape(pooled);
    if (qs[2] <= 0 or ps[1] <= 0) return false;
    if (qs[3] != ps[2]) return false;
    return qsaScoreFusedEligible(qs[0], qs[1], qs[3], mlx.mlx_array_dtype(q), mlx.mlx_array_dtype(pooled));
}

pub fn qsaPooledRowContiguous(pooled: mlx.mlx_array) bool {
    if (mlx.mlx_array_ndim(pooled) != 3) return false;
    const ps = mlx.getShape(pooled);
    if (ps[0] != 1 or ps[2] != 128) return false;
    const st = mlx.mlx_array_strides(pooled);
    return st[2] == 1 and st[1] == 128;
}

pub fn qsaScoreFused(s: mlx.mlx_stream, q: mlx.mlx_array, pooled: mlx.mlx_array) !?mlx.mlx_array {
    qsaScoreFusedArm();
    if (!qsaScoreFusedEligibleFrom(q, pooled)) return null;
    if (!mlx.streamIsGpu(s)) return null;
    if (!qsaPooledRowContiguous(pooled)) return null;
    return try qsaScoreFusedDispatch(s, q, pooled);
}

pub fn qsaScoreSheet(s: mlx.mlx_stream, q: mlx.mlx_array, pooled: mlx.mlx_array, k32t: mlx.mlx_array, fused: bool) !mlx.mlx_array {
    if (fused) {
        return (try qsaScoreFused(s, q, pooled)) orelse error.QsaScoreFusedDeclined;
    }
    if (k32t.ctx == null) return error.QsaScoreBankMissing;
    return qsaScoreSheetComposed(s, q, k32t);
}

pub fn qsaDumpBlocksPath() ?[]const u8 {
    if (qsa_dump_blocks_path_override) |p| {
        if (p.len == 0 or p[0] == '0') return null;
        return p;
    }
    if (qsa_dump_blocks_read) return qsa_dump_blocks_cached;
    qsa_dump_blocks_read = true;
    const raw = std.c.getenv("QWEN4_DUMP_QSA_BLOCKS") orelse return null;
    if (!diagEnvValueOn(raw)) return null;
    const s = std.mem.sliceTo(raw, 0);
    qsa_dump_blocks_cached = s;
    return s;
}

pub fn qsaDumpQsaBlocks(layer: u32, row0: c_int, rows: c_int, kb: c_int, blocks: mlx.mlx_array) void {
    const path = qsaDumpBlocksPath() orelse return;
    if (blocks.ctx == null) return;
    if (mlx.mlx_array_eval(blocks) != 0) return;
    var contig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(contig);
    const s = mlx.gpuStream();
    if (mlx.mlx_contiguous(&contig, blocks, false, s) != 0) return;
    if (mlx.mlx_array_eval(contig) != 0) return;
    const data = mlx.mlx_array_data_int32(contig) orelse return;
    const n = mlx.mlx_array_size(contig);
    var path_z: [1024]u8 = undefined;
    if (path.len == 0 or path.len >= path_z.len) return;
    @memcpy(path_z[0..path.len], path);
    path_z[path.len] = 0;
    const fd = std.c.open(@ptrCast(&path_z), .{
        .ACCMODE = .WRONLY,
        .CREAT = true,
        .APPEND = true,
    }, @as(std.c.mode_t, 0o644));
    if (fd < 0) return;
    defer _ = std.c.close(fd);
    var hdr: [128]u8 = undefined;
    const header = std.fmt.bufPrint(&hdr, "layer={d} row0={d} rows={d} kb={d}\n", .{ layer, row0, rows, kb }) catch return;
    _ = std.c.write(fd, header.ptr, header.len);
    var buf: [32]u8 = undefined;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const piece = if (i == 0)
            std.fmt.bufPrint(&buf, "{d}", .{data[i]}) catch return
        else
            std.fmt.bufPrint(&buf, " {d}", .{data[i]}) catch return;
        _ = std.c.write(fd, piece.ptr, piece.len);
    }
    _ = std.c.write(fd, "\n", 1);
}

// ── The composed selection arm (the kernel's reference) ──
// Free functions so the parity test can run both arms on the same scores.

/// argpartition of the visibility-masked scores; a tiny index-ascending bias reproduces
/// torch.topk's lower-index-wins. A null-ctx `vis3` skips the mask op (all visible).
pub fn qsaTopBlocksOps(s: mlx.mlx_stream, scores: mlx.mlx_array, vis3: mlx.mlx_array, tie_bias: mlx.mlx_array, nb: c_int, block_topk: c_int) !mlx.mlx_array {
    var biased = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(biased);
    try mlx.check(mlx.mlx_subtract(&biased, scores, tie_bias, s));
    if (vis3.ctx == null) {
        var part_all = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_argpartition_axis(&part_all, biased, nb - block_topk, -1, s));
        return part_all;
    }
    const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer _ = mlx.mlx_array_free(neg_inf);
    var masked = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(masked);
    try mlx.check(mlx.mlx_where(&masked, vis3, biased, neg_inf, s));
    var part = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_argpartition_axis(&part, masked, nb - block_topk, -1, s));
    return part;
}

/// Sort a `[1, rows, kb]` pick list ascending, mapping rejected picks to INT_MAX first.
pub fn qsaSortWithInvisible(s: mlx.mlx_stream, top_idx: mlx.mlx_array, vis3: mlx.mlx_array, all_vis: bool) !mlx.mlx_array {
    var sorted = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(sorted);
    if (all_vis) {
        try mlx.check(mlx.mlx_sort_axis(&sorted, top_idx, -1, s));
        return sorted;
    }
    const int_max = mlx.mlx_array_new_int(std.math.maxInt(i32));
    defer _ = mlx.mlx_array_free(int_max);
    var picked_vis = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(picked_vis);
    try mlx.check(mlx.mlx_take_along_axis(&picked_vis, vis3, top_idx, -1, s));
    var sel = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sel);
    try mlx.check(mlx.mlx_where(&sel, picked_vis, top_idx, int_max, s));
    try mlx.check(mlx.mlx_sort_axis(&sorted, sel, -1, s));
    return sorted;
}

/// The composed arm the fused kernel replaces, end to end.
pub fn qsaSelectComposedOps(s: mlx.mlx_stream, scores: mlx.mlx_array, vis3: mlx.mlx_array, tie_bias: mlx.mlx_array, rows: c_int, nb: c_int, kb: c_int, block_topk: c_int, all_vis: bool) !mlx.mlx_array {
    const strides3 = [_]c_int{ 1, 1, 1 };
    const part = try qsaTopBlocksOps(s, scores, vis3, tie_bias, nb, block_topk);
    defer _ = mlx.mlx_array_free(part);
    var tail = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tail);
    try mlx.check(mlx.mlx_slice(&tail, part, &[_]c_int{ 0, 0, nb - kb }, 3, &[_]c_int{ 1, rows, nb }, 3, &strides3, 3, s));
    var top_idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(top_idx);
    try mlx.check(mlx.mlx_astype(&top_idx, tail, .int32, s));
    return qsaSortWithInvisible(s, top_idx, vis3, all_vis);
}

/// Past the split-K widths: gather the packed cache in place, or from its rebuild when the
/// chunk's rows would re-dequantize each row more often than a rebuild does once.
pub fn qsaPackedGatherServes(seq_len: c_int, kv: c_int, kb: c_int, ratio: c_int) bool {
    if (seq_len <= 0 or kv <= 0 or kb <= 0 or ratio <= 0) return false;
    const staged: u64 = @as(u64, @intCast(seq_len)) * @as(u64, @intCast(kb)) * @as(u64, @intCast(ratio));
    return staged <= QSA_PACKED_GATHER_MAX_REUSE * @as(u64, @intCast(kv));
}

/// One batched call serves every slot in the group with the SAME arm (the
/// stacked-mask SDPA), so it counts once on each participating slot's own
/// tally — a single shared counter would report the group as one request.
pub fn noteQsaArmForSlots(slots: []const *ForwardCtx, arm: QsaArm) void {
    for (slots) |slot_ctx| slot_ctx.qsa_arms.note(arm);
}

pub fn qsaAlignedSparseAttn(s: mlx.mlx_stream, q: mlx.mlx_array, view: *const DenseKVView, blocks: mlx.mlx_array, ratio: c_int, scale: f32, budget: c_int) !?mlx.mlx_array {
    const width = mlx.getShape(q)[2];
    const length = mlx.getShape(view.k)[2];
    const cut = std.math.clamp(budget + ratio - 1 - (length - width), 0, width);
    if (budget <= 0 or cut == 0) return qsaSparseAttn(s, q, view, blocks, ratio, scale);
    if (width >= FUSED256_MIN_Q_LEN) return null;
    var parts: [FUSED256_MIN_Q_LEN]mlx.mlx_array = @splat(.{});
    var count: usize = 0;
    defer for (parts[0..count]) |part| {
        _ = mlx.mlx_array_free(part);
    };
    var row: c_int = 0;
    while (row < cut) : (row += 1) {
        const qr = try sliceAttentionSeq(s, q, row, row + 1);
        defer _ = mlx.mlx_array_free(qr);
        var prefix = try KvPrefixView.init(s, view.*, length - width + row + 1);
        defer prefix.deinit();
        parts[count] = mlx.mlx_array_new();
        count += 1;
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&parts[count - 1], qr, prefix.view.k, prefix.view.v, scale, "", .{ .ctx = null }, .{ .ctx = null }, false, s));
    }
    if (cut < width) {
        const qr = try sliceAttentionSeq(s, q, cut, width);
        defer _ = mlx.mlx_array_free(qr);
        var br = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(br);
        try mlx.check(mlx.mlx_slice(&br, blocks, &[_]c_int{ 0, cut, 0 }, 3, &[_]c_int{ 1, width, mlx.getShape(blocks)[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        parts[count] = (try qsaSparseAttn(s, qr, view, br, ratio, scale)) orelse return null;
        count += 1;
    }
    const vector = mlx.mlx_vector_array_new_data(&parts, count);
    defer _ = mlx.mlx_vector_array_free(vector);
    var result = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(result);
    try mlx.check(mlx.mlx_concatenate_axis(&result, vector, 2, s));
    return result;
}

pub fn qsaBatchedGatherAttn(
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,
    q_rope: mlx.mlx_array,
    views: []const DenseKVView,
    blocks: []const mlx.mlx_array,
    masks: []const mlx.mlx_array,
    ratio: c_int,
    attn_scale: f32,
    out_arms: []QsaArm,
    budget: c_int,
) !mlx.mlx_array {
    const qs = mlx.getShape(q_rope);
    const n: usize = @intCast(qs[0]);
    const seq_len: c_int = qs[2];
    const h_count: c_int = qs[1];
    const hd: c_int = qs[3];
    if (views.len != n or blocks.len != n) return error.QsaBatchedGatherLen;
    if (masks.len != 0 and masks.len != n) return error.QsaBatchedGatherLen;
    if (out_arms.len != 0 and out_arms.len != n) return error.QsaBatchedGatherLen;
    const standin_sdpa = qwen4Standin().attn_sdpa;
    var outs = try allocator.alloc(mlx.mlx_array, n);
    defer allocator.free(outs);
    var built: usize = 0;
    errdefer {
        var fi: usize = 0;
        while (fi < built) : (fi += 1) _ = mlx.mlx_array_free(outs[fi]);
    }
    for (views, blocks, 0..) |*kv_view, blk, i| {
        const slot_mask: mlx.mlx_array = if (i < masks.len) masks[i] else .{ .ctx = null };
        const i_c: c_int = @intCast(i);
        var q_slot = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_slot);
        try mlx.check(mlx.mlx_slice(
            &q_slot,
            q_rope,
            &[_]c_int{ i_c, 0, 0, 0 },
            4,
            &[_]c_int{ i_c + 1, h_count, seq_len, hd },
            4,
            &[_]c_int{ 1, 1, 1, 1 },
            4,
            s,
        ));
        var gathered: ?mlx.mlx_array = null;
        const qsa_ok = blk.ctx != null and slot_mask.ctx == null and !standin_sdpa;
        if (qsa_ok and qsaSparseAttnServes(kv_view.has_quant_triple, seq_len)) {
            gathered = try qsaAlignedSparseAttn(s, q_slot, kv_view, blk, ratio, attn_scale, budget);
        }
        if (gathered == null and qsa_ok and seq_len == 1) {
            gathered = try qsaDecodeGatherAttn(s, q_slot, kv_view, blk, ratio, attn_scale);
        }
        if (gathered == null and qsa_ok and seq_len >= 2 and seq_len < FUSED256_MIN_Q_LEN) {
            gathered = try qsaVerifyGatherAttn(s, q_slot, kv_view, blk, ratio, attn_scale);
        }
        if (gathered == null and qsa_ok and seq_len >= FUSED256_MIN_Q_LEN and kv_view.has_quant_triple) {
            gathered = if (qsaPackedGatherServes(seq_len, mlx.getShape(kv_view.k)[2], mlx.getShape(blk)[2], ratio))
                try gatherQsa256Packed(s, q_slot, kv_view, attn_scale, blk, ratio)
            else
                try gatherQsa256(s, q_slot, kv_view.k, kv_view.v, attn_scale, blk, ratio);
        }
        if (gathered) |g| {
            outs[i] = g;
            built = i + 1;
            if (out_arms.len == n) out_arms[i] = switch (qsaWidthBucket(seq_len)) {
                0 => .decode_gather,
                1 => .verify_gather,
                else => .prefill_gather,
            };
            continue;
        }
        var mask_tmp: mlx.mlx_array = .{ .ctx = null };
        defer if (mask_tmp.ctx != null) {
            _ = mlx.mlx_array_free(mask_tmp);
        };
        const mask: mlx.mlx_array = if (slot_mask.ctx != null) slot_mask else if (blk.ctx != null) blk: {
            mask_tmp = try qsaMaskFromBlocks(s, blk, mlx.getShape(kv_view.k)[2], ratio);
            break :blk mask_tmp;
        } else .{ .ctx = null };
        outs[i] = mlx.mlx_array_new();
        built = i + 1;
        if (out_arms.len == n) out_arms[i] = .mask;
        if (mask.ctx != null) {
            if (try fusedSdpa256Masked(s, q_slot, kv_view.k, kv_view.v, attn_scale, mask)) |fused| {
                _ = mlx.mlx_array_free(outs[i]);
                outs[i] = fused;
            } else if (try splitMaskedSdpa256(s, q_slot, kv_view.k, kv_view.v, attn_scale, mask)) |split_out| {
                _ = mlx.mlx_array_free(outs[i]);
                outs[i] = split_out;
            } else {
                try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&outs[i], q_slot, kv_view.k, kv_view.v, attn_scale, "array", mask, .{ .ctx = null }, false, s));
            }
        } else {
            const none = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(none);
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&outs[i], q_slot, kv_view.k, kv_view.v, attn_scale, "causal", none, .{ .ctx = null }, false, s));
        }
    }
    const vec = mlx.mlx_vector_array_new_data(outs.ptr, n);
    defer _ = mlx.mlx_vector_array_free(vec);
    var stacked = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(stacked);
    try mlx.check(mlx.mlx_concatenate_axis(&stacked, vec, 0, s));
    var fi: usize = 0;
    while (fi < built) : (fi += 1) _ = mlx.mlx_array_free(outs[fi]);
    return stacked;
}

/// The `pos % ratio` raw indexer rows ending at MTP-head row `pos`, copied while the ring
/// still held them. The trunk restores at one of its checkpoint positions and the head is
/// clamped to that same row; by commit time the head's ring is a whole generated tail past
/// it, so the rows the next append must re-pool are kept here instead.
pub const QsaHeadMark = struct {
    pos: c_int = 0,
    rows: mlx.mlx_array = .{ .ctx = null },
};

/// Marks a head keeps. A prefill's checkpoints are capped at `ssm_checkpoint_max` (16) and a
/// warm turn merges the donor's; past this the list thins like the checkpoints themselves.
pub const QSA_HEAD_MARKS_MAX: usize = 32;

/// One owner of a head's marks: the live head holds one, its committed snapshot a
/// refcount-sharing copy. Positions ascend.
pub const QsaHeadMarkSet = struct {
    items: [QSA_HEAD_MARKS_MAX]QsaHeadMark = @splat(.{}),
    len: usize = 0,

    pub fn slice(self: *const QsaHeadMarkSet) []const QsaHeadMark {
        return self.items[0..self.len];
    }

    /// A second owner for every mark; both sets then free independently.
    pub fn share(src: []const QsaHeadMark) QsaHeadMarkSet {
        var out: QsaHeadMarkSet = .{};
        for (src) |m| {
            if (m.rows.ctx == null or out.len == QSA_HEAD_MARKS_MAX) continue;
            var arr = mlx.mlx_array_new();
            _ = mlx.mlx_array_set(&arr, m.rows);
            out.items[out.len] = .{ .pos = m.pos, .rows = arr };
            out.len += 1;
        }
        return out;
    }

    pub fn deinit(self: *QsaHeadMarkSet) void {
        for (self.items[0..self.len]) |*m| {
            if (m.rows.ctx != null) _ = mlx.mlx_array_free(m.rows);
            m.* = .{};
        }
        self.len = 0;
    }

    pub fn bytes(self: *const QsaHeadMarkSet) u64 {
        var total: u64 = 0;
        for (self.items[0..self.len]) |m| {
            if (m.rows.ctx == null) continue;
            total += @as(u64, mlx.mlx_array_size(m.rows)) * @as(u64, mlx.mlx_array_itemsize(m.rows));
        }
        return total;
    }

    pub fn find(self: *const QsaHeadMarkSet, pos: c_int) ?mlx.mlx_array {
        for (self.items[0..self.len]) |m| {
            if (m.pos == pos and m.rows.ctx != null) return m.rows;
        }
        return null;
    }

    /// Marks above `pos` describe rows a clamp just discarded — the next append writes
    /// different tokens there.
    pub fn dropAbove(self: *QsaHeadMarkSet, pos: c_int) void {
        var i = self.len;
        while (i > 0 and self.items[i - 1].pos > pos) : (i -= 1) {
            if (self.items[i - 1].rows.ctx != null) _ = mlx.mlx_array_free(self.items[i - 1].rows);
            self.items[i - 1] = .{};
        }
        self.len = i;
    }

    pub fn dropAt(self: *QsaHeadMarkSet, idx: usize) void {
        if (self.items[idx].rows.ctx != null) _ = mlx.mlx_array_free(self.items[idx].rows);
        var i = idx;
        while (i + 1 < self.len) : (i += 1) self.items[i] = self.items[i + 1];
        self.items[self.len - 1] = .{};
        self.len -= 1;
    }

    /// Takes ownership of `rows`. A repeat at `pos` replaces; a full set thins its interior.
    pub fn put(self: *QsaHeadMarkSet, pos: c_int, rows: mlx.mlx_array) void {
        self.dropAbove(pos);
        if (self.len > 0 and self.items[self.len - 1].pos == pos) self.dropAt(self.len - 1);
        if (self.len == QSA_HEAD_MARKS_MAX) {
            self.dropAt(spanPreservingDropIndex(QsaHeadMark, self.slice(), markPosOf, .min_span_recency, null));
        }
        self.items[self.len] = .{ .pos = pos, .rows = rows };
        self.len += 1;
    }
};

pub var qsa_score_incremental_appends: usize = 0;

pub fn qsaPooledTableEq(a: ?[]const i32, b: ?[]const i32) bool {
    const aa = a orelse return b == null;
    const bb = b orelse return false;
    return aa.ptr == bb.ptr and aa.len == bb.len;
}

/// Does every complete-block visibility check pass for every query row of a call starting at
/// cache position `offset`? Block `b` is complete at `p` iff `b*ratio + ratio - 1 <= p`, so
/// the whole sheet is all-true iff `offset >= nb*ratio-1`. Always true at decode width.
pub fn qsaAllBlocksVisible(offset: c_int, nb: c_int, ratio: c_int) bool {
    return offset >= nb * ratio - 1;
}

/// `n` columns of the block-score matmul's key operand: `[B, 1, hd, n]` f32, the pooled
/// bank transposed and up-cast.
pub fn qsaScoreCols(self: *Transformer, pooled: mlx.mlx_array, batch: c_int, n: c_int, idx_hd: c_int) !mlx.mlx_array {
    var k_rope = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_rope);
    try mlx.check(mlx.mlx_reshape(&k_rope, pooled, &[_]c_int{ batch, 1, n, idx_hd }, 4, self.s));
    const kperm = [_]c_int{ 0, 1, 3, 2 };
    var kt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kt);
    try mlx.check(mlx.mlx_transpose_axes(&kt, k_rope, &kperm, 4, self.s));
    var k32 = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(k32);
    try mlx.check(mlx.mlx_astype(&k32, kt, .float32, self.s));
    return k32;
}

/// The score operand for `nb` blocks, borrowed. Built only on the composed arm.
/// A completed block appends columns; from-scratch rebuilds after restore, a cold
/// entry, or rollback.
pub fn qsaScoreBank(self: *Transformer, entry: *SSMCacheEntry, batch: c_int, nb: c_int, idx_hd: c_int) !mlx.mlx_array {
    if (entry.qsa_score_bank.ctx != null and entry.qsa_score_blocks == nb) return entry.qsa_score_bank;
    const old_nb = entry.qsa_score_blocks;
    const ratio: usize = @max(@as(usize, @intCast(entry.qsa_ratio)), 1);
    const reserve = entry.qsa_reserve_rows / ratio;
    if (entry.qsa_score_buf.ctx != null and old_nb > 0 and nb > old_nb and entry.qsa_pooled.ctx != null) {
        const psh = mlx.getShape(entry.qsa_pooled);
        if (psh[1] >= nb) {
            const n_new = nb - old_nb;
            var new_pooled = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(new_pooled);
            try mlx.check(mlx.mlx_slice(
                &new_pooled,
                entry.qsa_pooled,
                &[_]c_int{ 0, old_nb, 0 },
                3,
                &[_]c_int{ psh[0], nb, psh[2] },
                3,
                &[_]c_int{ 1, 1, 1 },
                3,
                self.s,
            ));
            const cols = try self.qsaScoreCols(new_pooled, batch, n_new, idx_hd);
            defer _ = mlx.mlx_array_free(cols);
            try capBufAppend(self.s, &entry.qsa_score_buf, &entry.qsa_score_bank, &entry.qsa_score_blocks, cols, 3, reserve);
            self.qsa_score_bank_builds += 1;
            qsa_score_incremental_appends += 1;
            return entry.qsa_score_bank;
        }
    }
    if (entry.qsa_score_bank.ctx != null) _ = mlx.mlx_array_free(entry.qsa_score_bank);
    if (entry.qsa_score_buf.ctx != null) _ = mlx.mlx_array_free(entry.qsa_score_buf);
    entry.qsa_score_bank = .{ .ctx = null };
    entry.qsa_score_buf = .{ .ctx = null };
    entry.qsa_score_blocks = 0;
    const cols = try self.qsaScoreCols(entry.qsa_pooled, batch, nb, idx_hd);
    defer _ = mlx.mlx_array_free(cols);
    try capBufAppend(self.s, &entry.qsa_score_buf, &entry.qsa_score_bank, &entry.qsa_score_blocks, cols, 3, reserve);
    self.qsa_score_bank_builds += 1;
    return entry.qsa_score_bank;
}

pub fn qsaScoreOperand(self: *Transformer, entry: *SSMCacheEntry, batch: c_int, nb: c_int, idx_hd: c_int, fused: bool) !mlx.mlx_array {
    if (fused) return .{ .ctx = null };
    return self.qsaScoreBank(entry, batch, nb, idx_hd);
}

/// The pooled-block cos/sin for this forward, built on first ask. Borrowed handles.
pub fn qsaPooledCosSin(self: *Transformer, ctx: *ForwardCtx, rope_dims: c_int, base: c_int, step: c_int, n: c_int, dt: mlx.mlx_dtype) !Transformer.MropeCosSin {
    const is_mrope = ctx.mrope_pos != null;
    const c = &self.qsa_pooled_rope;
    if (c.stale.load(.acquire)) c.deinit();
    if (c.cos.ctx != null and
        c.base == base and c.step == step and c.n == n and
        c.dtype == dt and c.mrope == is_mrope and
        qsaPooledTableEq(c.mrope_pos, ctx.mrope_pos) and
        c.mrope_total == ctx.mrope_total and c.mrope_delta == ctx.mrope_delta)
        return .{ .cos = c.cos, .sin = c.sin };
    c.deinit();
    const cs = if (is_mrope)
        try self.mropeCosSinAt(Transformer.mropeContext(ctx), @intCast(base), @intCast(step), @intCast(n), dt)
    else
        try self.ropeCosSinFromFreqs(rope_dims, try self.ropeInvFreq(rope_dims, self.config.rope_theta), @floatFromInt(base), @floatFromInt(step), n, dt);
    c.cos = cs.cos;
    c.sin = cs.sin;
    c.base = base;
    c.step = step;
    c.n = n;
    c.dtype = dt;
    c.mrope = is_mrope;
    c.mrope_pos = ctx.mrope_pos;
    c.mrope_total = ctx.mrope_total;
    c.mrope_delta = ctx.mrope_delta;
    c.builds += 1;
    return cs;
}

/// Pooled block keys `[B, nb, D]` from raw index keys `kb4 [B, nb, ratio, D]`: f32 block mean,
/// bf16, idx_k_norm, then partial RoPE at the block starts `base + i*ratio`. Text turns take
/// one fused kernel, bit-identical to the chain; M-RoPE and other shapes keep the chain.
pub fn qsaPooledKeys(self: *Transformer, ctx: *ForwardCtx, kb4: mlx.mlx_array, norm_w: mlx.mlx_array, rope_dims: c_int, base: c_int) !mlx.mlx_array {
    const sh = mlx.getShape(kb4);
    if (ctx.mrope_pos == null and qsaPoolRopeFusedEnabled() and mlx.mlx_array_dtype(norm_w) == .bfloat16) {
        const cs = try self.qsaPooledCosSin(ctx, rope_dims, base, sh[2], sh[1], .bfloat16);
        if (try qsaPoolNormRopeFused(self.s, kb4, norm_w, self.config.rms_norm_eps, cs.cos, cs.sin, rope_dims)) |out| return out;
    }
    const pn = try self.qsaPoolNorm(kb4, norm_w);
    defer _ = mlx.mlx_array_free(pn);
    const cs = try self.qsaPooledCosSin(ctx, rope_dims, base, sh[2], sh[1], mlx.mlx_array_dtype(pn));
    if (ctx.mrope_pos == null) return self.qsaPooledRopeComposed(pn, rope_dims, cs.cos, cs.sin);
    var pn4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pn4);
    try mlx.check(mlx.mlx_expand_dims(&pn4, pn, 1, self.s));
    const roped = try self.applyMrope(pn4, cs.cos, cs.sin, rope_dims);
    defer _ = mlx.mlx_array_free(roped);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&out, roped, &[_]c_int{ sh[0], sh[1], sh[3] }, 3, self.s));
    return out;
}

/// QSA: append this chunk's raw index keys to the layer's history and,
/// past the token budget, build the block-selection mask. Returns the
/// bool `[B,1,S,kv]` mask or null-ctx for dense attention.
/// `cache_len` = key rows already in the history, `pos_base` = absolute
/// position of key row 0 (0 for the trunk; the MTP head's cache starts at
/// the first draft position). Block/tail/causal arithmetic is in cache
/// coordinates; RoPE angles use absolute positions — the SAME M-RoPE table
/// as attention on an image request (queries from the chunk's cos/sin,
/// pooled block keys at 3-D block-start positions), scalar rope otherwise.
pub fn qsaSlotEntry(sc: *ForwardCtx, layer: u32) *SSMCacheEntry {
    if (sc.qsa_entry) |e| return e;
    return &sc.ssm_entries.?[layer];
}

pub fn qsaMask(self: *Transformer, ctx: *ForwardCtx, x: mlx.mlx_array, fa: *const FullAttnWeights, entry: *SSMCacheEntry, layer: u32, cache_len: c_int, pos_base: c_int, batch: c_int, seq_len: c_int) !mlx.mlx_array {
    const qk = try self.qmatmul(x, fa.idx_qk_w, fa.idx_qk_s, fa.idx_qk_b); // [B,S,(n+1)*hd]
    defer _ = mlx.mlx_array_free(qk);
    if (ctx.batch_slots) |slots| return self.qsaMaskBatched(slots, qk, fa, layer);
    return self.qsaMaskFromQk(ctx, qk, fa, entry, cache_len, pos_base, batch, seq_len, layer);
}

/// Batched decode: the indexer projection ran once on `[N,1,·]`; each row
/// then walks the serial per-slot body against ITS slot's key history and
/// position, and the per-slot masks are false-padded to the group's
/// longest kv and stacked to `[N,1,1,kv_max]`. Null-ctx when no slot is
/// past the budget (dense group — the plain batched mask suffices).
pub fn qsaMaskBatched(self: *Transformer, slots: []const *ForwardCtx, qk: mlx.mlx_array, fa: *const FullAttnWeights, layer: u32) !mlx.mlx_array {
    const N = slots.len;
    // The per-slot ctxs have no other net: without this one throw orphaned a slot's selection.
    errdefer for (slots) |sc| {
        if (sc.qsa_blocks.ctx != null) _ = mlx.mlx_array_free(sc.qsa_blocks);
        sc.qsa_blocks = .{ .ctx = null };
    };
    const qk_sh = mlx.getShape(qk);
    const seq_len: c_int = qk_sh[1];
    const w: c_int = qk_sh[2];
    const masks = try self.allocator.alloc(mlx.mlx_array, N);
    defer self.allocator.free(masks);
    for (masks) |*m| m.* = .{ .ctx = null };
    defer for (masks) |m| {
        if (m.ctx != null) _ = mlx.mlx_array_free(m);
    };
    const kv_lens = try self.allocator.alloc(c_int, N);
    defer self.allocator.free(kv_lens);
    var kv_max: c_int = 0;
    var any = false;
    var any_blocks = false;
    const ratio: c_int = @intCast(self.config.indexer_compress_ratio);
    for (slots, 0..) |sc, i| {
        const e = qsaSlotEntry(sc, layer);
        const cache_len: c_int = @intCast(sc.moe_seq_offset.*);
        const i_c: c_int = @intCast(i);
        var qk_i = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(qk_i);
        try mlx.check(mlx.mlx_slice(&qk_i, qk, &[_]c_int{ i_c, 0, 0 }, 3, &[_]c_int{ i_c + 1, seq_len, w }, 3, &[_]c_int{ 1, 1, 1 }, 3, self.s));
        masks[i] = try self.qsaMaskFromQk(sc, qk_i, fa, e, cache_len, sc.qsa_pos_base, 1, seq_len, layer);
        kv_lens[i] = cache_len + seq_len;
        if (kv_lens[i] > kv_max) kv_max = kv_lens[i];
        if (masks[i].ctx != null) any = true;
        if (sc.qsa_blocks.ctx != null) any_blocks = true;
    }
    const keep_blocks = qsaBatchedGatherEnabled() and self.qwen4 != null and any_blocks;
    if (keep_blocks) {
        for (slots, 0..) |sc, i| {
            if (sc.qsa_blocks.ctx == null and masks[i].ctx != null) {
                sc.qsa_mask = masks[i];
                masks[i] = .{ .ctx = null };
            }
        }
        return mlx.mlx_array_new();
    }
    for (slots, 0..) |sc, i| {
        if (sc.qsa_blocks.ctx != null) {
            // Decode-width selection is per-slot unusable in the stacked
            // mask path — expand it to the equivalent dense mask (the same
            // fallback the serial arm uses when its gatherer declines).
            const m = try qsaMaskFromBlocks(self.s, sc.qsa_blocks, kv_lens[i], ratio);
            _ = mlx.mlx_array_free(sc.qsa_blocks);
            sc.qsa_blocks = .{ .ctx = null };
            _ = mlx.mlx_array_free(masks[i]);
            masks[i] = m;
            any = true;
        }
    }
    if (!any) return mlx.mlx_array_new();
    const padded = try self.allocator.alloc(mlx.mlx_array, N);
    defer self.allocator.free(padded);
    for (padded) |*m| m.* = .{ .ctx = null };
    defer for (padded) |m| {
        if (m.ctx != null) _ = mlx.mlx_array_free(m);
    };
    const false_v = mlx.mlx_array_new_bool(false);
    defer _ = mlx.mlx_array_free(false_v);
    for (0..N) |i| {
        var row = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(row);
        if (masks[i].ctx != null) {
            try mlx.check(mlx.mlx_array_set(&row, masks[i]));
        } else {
            try mlx.check(mlx.mlx_ones(&row, &[_]c_int{ 1, 1, seq_len, kv_lens[i] }, 4, .bool_, self.s));
        }
        const pad_axes = [_]c_int{3};
        const low = [_]c_int{0};
        const high = [_]c_int{kv_max - kv_lens[i]};
        try mlx.check(mlx.mlx_pad(&padded[i], row, &pad_axes, 1, &low, 1, &high, 1, false_v, "constant", self.s));
    }
    return Transformer.concatAxis0(self.s, padded);
}

/// Serial per-slot body of `qsaMask` over the projected `qk` rows.
pub fn qsaMaskFromQk(self: *Transformer, ctx: *ForwardCtx, qk: mlx.mlx_array, fa: *const FullAttnWeights, entry: *SSMCacheEntry, cache_len: c_int, pos_base: c_int, batch: c_int, seq_len: c_int, layer: u32) !mlx.mlx_array {
    const offset = cache_len;
    const cfg = &self.config;
    const n_idx: c_int = @intCast(cfg.indexer_n_heads);
    const idx_hd: c_int = @intCast(cfg.indexer_head_dim);
    const ratio: c_int = @intCast(cfg.indexer_compress_ratio);
    const budget: c_int = @intCast(cfg.indexer_budget);
    const block_topk: c_int = @divTrunc(budget, ratio);
    const rope_dims: c_int = @intFromFloat(@as(f32, @floatFromInt(cfg.head_dim)) * cfg.partial_rotary_factor);
    const strides3 = [_]c_int{ 1, 1, 1 };
    var k_raw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_raw);
    try mlx.check(mlx.mlx_slice(&k_raw, qk, &[_]c_int{ 0, 0, n_idx * idx_hd }, 3, &[_]c_int{ batch, seq_len, (n_idx + 1) * idx_hd }, 3, &strides3, 3, self.s));

    const kv: c_int = offset + seq_len;
    entry.qsa_ratio = ratio;
    if (!qsaEntrySatisfiesForward(entry, offset)) return error.QsaHistoryGap;
    if (diagEnvOnCached(&qwen4_forward.qwen4_no_pooled_env, "QWEN4_NO_POOLED")) return error.QsaPooledRequired;
    const nb: c_int = @divTrunc(kv, ratio);
    const nb_cached: c_int = if (entry.qsa_pooled.ctx != null) mlx.getShape(entry.qsa_pooled)[1] else 0;
    if (nb > nb_cached) {
        const leftover_n = offset - nb_cached * ratio;
        if (leftover_n < 0) return error.QsaHistoryGap;
        const held: c_int = if (entry.aux_state.ctx != null) mlx.getShape(entry.aux_state)[1] else 0;
        if (leftover_n > held) return error.QsaHistoryGap;
        const n_new = nb - nb_cached;
        const need = n_new * ratio;
        const from_k = need - leftover_n;
        if (from_k < 0 or from_k > seq_len) return error.QsaHistoryGap;
        var kb_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(kb_flat);
        if (leftover_n == 0) {
            try mlx.check(mlx.mlx_slice(&kb_flat, k_raw, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ batch, need, idx_hd }, 3, &strides3, 3, self.s));
        } else {
            const ash = mlx.getShape(entry.aux_state);
            var leftover = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(leftover);
            try mlx.check(mlx.mlx_slice(&leftover, entry.aux_state, &[_]c_int{ 0, held - leftover_n, 0 }, 3, &[_]c_int{ ash[0], held, ash[2] }, 3, &strides3, 3, self.s));
            var k_part = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(k_part);
            try mlx.check(mlx.mlx_slice(&k_part, k_raw, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ batch, from_k, idx_hd }, 3, &strides3, 3, self.s));
            const parts = [_]mlx.mlx_array{ leftover, k_part };
            const vec = mlx.mlx_vector_array_new_data(&parts, 2);
            defer _ = mlx.mlx_vector_array_free(vec);
            try mlx.check(mlx.mlx_concatenate_axis(&kb_flat, vec, 1, self.s));
        }
        const kb_shape = [_]c_int{ batch, n_new, ratio, idx_hd };
        const want: usize = @as(usize, @intCast(batch)) * @as(usize, @intCast(n_new)) * @as(usize, @intCast(ratio)) * @as(usize, @intCast(idx_hd));
        if (mlx.mlx_array_size(kb_flat) != want) return error.QsaHistoryGap;
        var kb4 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(kb4);
        try mlx.check(mlx.mlx_reshape(&kb4, kb_flat, &kb_shape, 4, self.s));
        const new3 = try self.qsaPooledKeys(ctx, kb4, fa.idx_k_norm, rope_dims, pos_base + nb_cached * ratio);
        defer _ = mlx.mlx_array_free(new3);
        try self.qsaAppendPooled(entry, new3, nb_cached);
    }

    try self.qsaAppendKeys(entry, k_raw, offset);
    var keys = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(keys);
    try mlx.check(mlx.mlx_array_set(&keys, entry.aux_state));

    if (kv <= budget + ratio - 1) return mlx.mlx_array_new();
    std.debug.assert(batch == 1);
    if (!Transformer.qsa_engaged_logged) {
        Transformer.qsa_engaged_logged = true;
        log.info("[qsa] sparse attention engaged: kv={d} blocks={d} top-{d} (budget {d}, ratio {d})\n", .{ kv, @divTrunc(kv, ratio), block_topk, budget, ratio });
    }

    // Index queries: per-head norm → [B, n, S, hd] → partial RoPE at the
    // chunk's positions.
    var q = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q);
    try mlx.check(mlx.mlx_slice(&q, qk, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ batch, seq_len, n_idx * idx_hd }, 3, &strides3, 3, self.s));
    const q_shape = [_]c_int{ batch, seq_len, n_idx, idx_hd };
    var q4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q4);
    try mlx.check(mlx.mlx_reshape(&q4, q, &q_shape, 4, self.s));
    const qn = try self.rmsNorm(q4, fa.idx_q_norm);
    defer _ = mlx.mlx_array_free(qn);
    const perm = [_]c_int{ 0, 2, 1, 3 };
    var qt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(qt);
    try mlx.check(mlx.mlx_transpose_axes(&qt, qn, &perm, 4, self.s));
    var q_rope = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_rope);
    if (ctx.mrope_cos_cur) |cos| {
        _ = mlx.mlx_array_free(q_rope);
        q_rope = try self.applyMrope(qt, cos, ctx.mrope_sin_cur.?, rope_dims);
    } else {
        const eff_off: c_int = pos_base + offset + (if (ctx.mrope_pos != null) ctx.mrope_delta else 0);
        // The SAME spectrum attention rotates with (scaled when the config
        // says YaRN): the index queries and keys are the only place the
        // model asks "how far apart are these two blocks", and a block
        // pick made on unscaled angles at 1M positions would select
        // against a position notion the attention layers no longer have.
        // Indexer is 128-wide / 64 rotated, so score = ms²·A+B and mscale
        // CAN change a top-k. M-RoPE tables already carry mscale via
        // fillCosSin; this scalar arm must too. Never yarnScaleQK here
        // (that table is [head_dim] = 256). Keep mlx_fast_rope scale=1.0.
        const use_yarn = self.yarnActive();
        try mlx.check(mlx.mlx_fast_rope(&q_rope, qt, rope_dims, false, mlx.mlx_optional_float{
            .value = cfg.rope_theta,
            .has_value = !use_yarn,
        }, 1.0, eff_off, if (use_yarn) self.rope_freqs_yarn.? else .{ .ctx = null }, self.s));
        if (use_yarn) try self.yarnScaleRotated(&q_rope, rope_dims);
    }

    // Pooled block keys: mean of leftover + this chunk, already appended above.
    if (diagEnvOnCached(&qwen4_forward.qwen4_debug_scores_env, "QWEN4_DEBUG_SCORES")) {
        const qs = mlx.getShape(entry.qsa_pooled);
        std.debug.print("[qsa] kv={d} nb={d} cached={d} pooled shape {any} keys {any}\n", .{ kv, nb, nb_cached, qs, mlx.getShape(keys) });
    }
    // scores[b, s, blk] = sum_h relu(q_h . k_blk) in f32 like the reference; the 1/sqrt(hd)
    // scale is dropped because it is monotone for the top-k. Borrowed from the entry.
    const fused = qsaScoreFusedEligibleFrom(q_rope, entry.qsa_pooled) and
        qsaPooledRowContiguous(entry.qsa_pooled) and mlx.streamIsGpu(self.s);
    const k32 = try self.qsaScoreOperand(entry, batch, nb, idx_hd, fused);

    // A packed cache gathers at every width (split-K, packed, or from its rebuild), never the mask arm.
    const quantized = ctx.cache.config.scheme == .affine;
    const row_sparse = qsaAttnKernelEnabled() and
        (qsaSparseAttnServes(quantized, seq_len) or (quantized and seq_len >= FUSED256_MIN_Q_LEN));
    const nax_geometry = seq_len >= FUSED256_MIN_Q_LEN and ratio == 4 and block_topk == 512 and cfg.head_dim == 256 and
        cfg.num_attention_heads == 24 and cfg.num_key_value_heads == 2 and mlx.mlx_array_dtype(qk) == .bfloat16;
    const gather_floor = if (nax_geometry) qsaPrefillGatherMinKv() else qsaGatherMinKv();
    const legacy_blocks = kv > gather_floor and
        (seq_len >= FUSED256_MIN_Q_LEN or (seq_len == 1 and qsaDecodeGatherEnabled()) or
            (seq_len >= 2 and seq_len < FUSED256_MIN_Q_LEN and qsaVerifyGatherEnabled() and kv > qsaVerifyGatherMinKvFor(quantized)));
    const want_blocks = batch == 1 and qsaGatherEnabled() and (row_sparse or legacy_blocks);
    if (want_blocks) {
        // Prefill: sorted per-row block indices for the gather kernel;
        // the dense [S, kv] mask is never built. Decode (S==1): the same
        // single-row selection for the decode gatherer.
        const prof = diagEnvOnCached(&qwen4_forward.qwen4_profile_qsa_env, "QWEN4_PROFILE_QSA");
        var clk: ProfClock = undefined;
        if (prof) {
            if (k32.ctx != null) try mlx.check(mlx.mlx_array_eval(k32));
            clk = ProfClock.init();
        }
        // Free before assign: a `defer` registered below a fallible call leaked a handle per tick.
        if (ctx.qsa_blocks.ctx != null) _ = mlx.mlx_array_free(ctx.qsa_blocks);
        ctx.qsa_blocks = .{ .ctx = null };
        ctx.qsa_blocks = try self.qsaSelectBlocks(q_rope, k32, entry.qsa_pooled, offset, seq_len, nb, block_topk, fused);
        if (ctx.qsa_blocks.ctx != null) {
            const kb_dim: c_int = if (mlx.mlx_array_ndim(ctx.qsa_blocks) == 3) mlx.getShape(ctx.qsa_blocks)[2] else block_topk;
            qsaDumpQsaBlocks(layer, offset, seq_len, kb_dim, ctx.qsa_blocks);
        }
        if (prof) {
            try mlx.check(mlx.mlx_array_eval(ctx.qsa_blocks));
            log.info("[qsa-prof] select S={d} nb={d}: {d:.2} ms\n", .{ seq_len, nb, @as(f64, @floatFromInt(clk.lap())) / 1e6 });
        }
        return mlx.mlx_array_new();
    }

    const scores = try qsaScoreSheet(self.s, q_rope, entry.qsa_pooled, k32, fused);
    defer _ = mlx.mlx_array_free(scores);

    if (diagEnvOnCached(&qwen4_forward.qwen4_debug_scores_env, "QWEN4_DEBUG_SCORES")) {
        var f = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(f);
        try mlx.check(mlx.mlx_astype(&f, scores, .float32, self.s));
        try mlx.check(mlx.mlx_array_eval(f));
        const d = mlx.mlx_array_data_float32(f).?;
        const nbu: usize = @intCast(nb);
        for (0..@intCast(seq_len)) |r| {
            std.debug.print("[qsa scores] row {d}:", .{r});
            for (0..nbu) |c| std.debug.print(" {d:.4}", .{d[r * nbu + c]});
            std.debug.print("\n", .{});
        }
    }
    // Past the budget an all-visible call skips the sheet; the `nb <= block_topk` arm always builds it.
    const all_vis = qsaAllBlocksVisible(offset, nb, ratio);
    var vis3 = mlx.mlx_array{ .ctx = null };
    defer if (vis3.ctx != null) {
        _ = mlx.mlx_array_free(vis3);
    };
    if (!all_vis or nb <= block_topk) vis3 = try self.qsaBlockVisibility(offset, seq_len, nb, ratio);

    try self.qsa_consts.ensure(self.s, nb, ratio);
    return qsaSelectMaskOps(self.allocator, self.s, scores, vis3, self.qsa_consts.tie_bias, offset, seq_len, kv, nb, ratio, block_topk, batch);
}

/// Block selection -> dense `[B,1,S,kv]` QSA mask, as pure ops. `vis3` may be null-ctx
/// (every block complete); every consumer must then skip its own op, since mlx-c aborts
/// on a null handle. Split out so the all-visible claim is testable without a Transformer.
pub fn qsaSelectMaskOps(
    alloc: std.mem.Allocator,
    s: mlx.mlx_stream,
    scores: mlx.mlx_array,
    vis3: mlx.mlx_array,
    tie_bias: mlx.mlx_array,
    offset: c_int,
    seq_len: c_int,
    kv: c_int,
    nb: c_int,
    ratio: c_int,
    block_topk: c_int,
    batch: c_int,
) !mlx.mlx_array {
    const strides3 = [_]c_int{ 1, 1, 1 };
    var blk_sel = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(blk_sel);
    if (nb <= block_topk) {
        // This arm is the sheet; a null sheet means every block is both visible and selected.
        if (vis3.ctx == null) {
            try mlx.check(mlx.mlx_ones(&blk_sel, &[_]c_int{ batch, seq_len, nb }, 3, .bool_, s));
        } else {
            try mlx.check(mlx.mlx_array_set(&blk_sel, vis3));
        }
    } else {
        // One tie rule: the gather path selects with the exact radix-select kernel, so this
        // arm prefers the same kernel and keeps the argpartition chain as its decline path.
        var top_idx = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(top_idx);
        var have_exact = false;
        if (batch == 1) {
            const bounds_host = try alloc.alloc(i32, @intCast(seq_len));
            defer alloc.free(bounds_host);
            qsaVisibleBoundsHost(bounds_host, offset, seq_len, nb, ratio, vis3.ctx == null);
            const bounds = mlx.mlx_array_new_data(bounds_host.ptr, &[_]c_int{seq_len}, 1, .int32);
            defer _ = mlx.mlx_array_free(bounds);
            if (qsaSelectTopBlocks(s, scores, bounds, block_topk) catch null) |picks| {
                defer _ = mlx.mlx_array_free(picks);
                // The kernel pads a short row with INT_MAX; clamping those to 0 is safe because
                // the `logical_and` with the sheet below clears block 0 when it is not visible.
                const nb_v = mlx.mlx_array_new_int(nb);
                defer _ = mlx.mlx_array_free(nb_v);
                var valid = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(valid);
                try mlx.check(mlx.mlx_less(&valid, picks, nb_v, s));
                const zero_i = mlx.mlx_array_new_int(0);
                defer _ = mlx.mlx_array_free(zero_i);
                try mlx.check(mlx.mlx_where(&top_idx, valid, picks, zero_i, s));
                have_exact = true;
            }
        }
        var part = mlx.mlx_array{ .ctx = null };
        defer if (part.ctx != null) {
            _ = mlx.mlx_array_free(part);
        };
        if (!have_exact) {
            part = try qsaTopBlocksOps(s, scores, vis3, tie_bias, nb, block_topk);
            try mlx.check(mlx.mlx_slice(&top_idx, part, &[_]c_int{ 0, 0, nb - block_topk }, 3, &[_]c_int{ batch, seq_len, nb }, 3, &strides3, 3, s));
        }
        var falses = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(falses);
        try mlx.check(mlx.mlx_zeros(&falses, &[_]c_int{ batch, seq_len, nb }, 3, .bool_, s));
        const true_v = mlx.mlx_array_new_bool(true);
        defer _ = mlx.mlx_array_free(true_v);
        var picked = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(picked);
        try mlx.check(mlx.mlx_put_along_axis(&picked, falses, top_idx, true_v, -1, s));
        if (vis3.ctx == null) {
            // `picked AND all-true` is `picked`; passing the null handle to mlx aborted the process.
            try mlx.check(mlx.mlx_array_set(&blk_sel, picked));
        } else {
            try mlx.check(mlx.mlx_logical_and(&blk_sel, picked, vis3, s));
        }
    }
    return qsaMaskFromBlockSel(s, blk_sel, offset, seq_len, kv, nb, ratio, batch);
}

/// Block visibility [1,S,nb]: block blk is complete for the query at
/// cache position p iff blk*ratio + ratio - 1 <= p. The block-end column comes from `qsa_consts`.
pub fn qsaBlockVisibility(self: *Transformer, offset: c_int, seq_len: c_int, nb: c_int, ratio: c_int) !mlx.mlx_array {
    const s = self.s;
    try self.qsa_consts.ensure(s, nb, ratio);
    var pos_col = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pos_col);
    try mlx.check(mlx.mlx_arange(&pos_col, @floatFromInt(offset), @floatFromInt(offset + seq_len), 1.0, .int32, s));
    var pos2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pos2);
    try mlx.check(mlx.mlx_reshape(&pos2, pos_col, &[_]c_int{ seq_len, 1 }, 2, s));
    var vis = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vis);
    try mlx.check(mlx.mlx_less_equal(&vis, self.qsa_consts.blk_end2, pos2, s)); // [S, nb]
    var vis3 = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_expand_dims(&vis3, vis, 0, s)); // [1,S,nb]
    return vis3;
}

/// One row-chunk of the block selection: the top-`kb` visible blocks per row, ascending,
/// INT_MAX past the row's count. Under `all_vis` the visibility sheet and both `where`s are
/// identities and are skipped; the fused arm reads the same claim through `qsaVisibleBoundsHost`.
pub fn qsaChunkSelect(self: *Transformer, scores: mlx.mlx_array, canonical: mlx.mlx_array, row0: c_int, rows: c_int, nb: c_int, ratio: c_int, kb: c_int, block_topk: c_int, all_vis: bool) !mlx.mlx_array {
    if (canonical.ctx == null and qsaSelectKernelEnabled() and mlx.streamIsGpu(self.s)) {
        // The whole composed tail in one dispatch; bounds are host arithmetic.
        const bounds_host = try self.allocator.alloc(i32, @intCast(rows));
        defer self.allocator.free(bounds_host);
        qsaVisibleBoundsHost(bounds_host, row0, rows, nb, ratio, all_vis);
        const bounds = mlx.mlx_array_new_data(bounds_host.ptr, &[_]c_int{rows}, 1, .int32);
        defer _ = mlx.mlx_array_free(bounds);
        if (try qsaSelectTopBlocks(self.s, scores, bounds, kb)) |fused| return fused;
    }
    var vis3 = mlx.mlx_array{ .ctx = null };
    defer if (vis3.ctx != null) {
        _ = mlx.mlx_array_free(vis3);
    };
    if (!all_vis) vis3 = try self.qsaBlockVisibility(row0, rows, nb, ratio);
    if (canonical.ctx == null) {
        try self.qsa_consts.ensure(self.s, nb, ratio);
        return qsaSelectComposedOps(self.s, scores, vis3, self.qsa_consts.tie_bias, rows, nb, kb, block_topk, all_vis);
    }
    // `nb <= block_topk`: every block is a pick, only visibility can remove one.
    var top_idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(top_idx);
    try mlx.check(mlx.mlx_broadcast_to(&top_idx, canonical, &[_]c_int{ 1, rows, nb }, 3, self.s));
    return qsaSortWithInvisible(self.s, top_idx, vis3, all_vis);
}

/// Prefill block selection for the gather kernel: per row the top-k
/// visible blocks ascending, INT_MAX past the row's count ([1,S,kb]
/// int32). Row-chunked so the [n_idx, rows, nb] f32 score sheet stays
/// under `qsaScoreSheetBudget()`. `k32t` is the pooled key bank
/// [1,1,hd,nb] f32.
pub fn qsaSelectBlocks(self: *Transformer, q_rope: mlx.mlx_array, k32t: mlx.mlx_array, pooled: mlx.mlx_array, offset: c_int, seq_len: c_int, nb: c_int, block_topk: c_int, fused: bool) !mlx.mlx_array {
    const n_idx: c_int = @intCast(self.config.indexer_n_heads);
    const idx_hd: c_int = @intCast(self.config.indexer_head_dim);
    const ratio: c_int = @intCast(self.config.indexer_compress_ratio);
    const kb: c_int = @min(nb, block_topk);
    const rows_per: c_int = @intCast(qsaScoreRowsPerChunkFused(@intCast(n_idx), @intCast(nb), @intCast(seq_len), fused));
    const strides4 = [_]c_int{ 1, 1, 1, 1 };
    var canonical = mlx.mlx_array{ .ctx = null };
    defer if (canonical.ctx != null) {
        _ = mlx.mlx_array_free(canonical);
    };
    if (nb <= block_topk) {
        var ar = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ar);
        try mlx.check(mlx.mlx_arange(&ar, 0, @floatFromInt(nb), 1.0, .int32, self.s));
        canonical = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&canonical, ar, &[_]c_int{ 1, 1, nb }, 3, self.s));
    }

    var parts: std.ArrayList(mlx.mlx_array) = .empty;
    defer {
        for (parts.items) |a| _ = mlx.mlx_array_free(a);
        parts.deinit(self.allocator);
    }
    var r0: c_int = 0;
    while (r0 < seq_len) : (r0 += rows_per) {
        const r1: c_int = @min(r0 + rows_per, seq_len);
        const rows = r1 - r0;
        var q_chunk = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_chunk);
        try mlx.check(mlx.mlx_slice(&q_chunk, q_rope, &[_]c_int{ 0, 0, r0, 0 }, 4, &[_]c_int{ 1, n_idx, r1, idx_hd }, 4, &strides4, 4, self.s));
        qsa_index_score_graphs += 1;
        const scores = try qsaScoreSheet(self.s, q_chunk, pooled, k32t, fused);
        defer _ = mlx.mlx_array_free(scores);
        const sorted = try self.qsaChunkSelect(scores, canonical, offset + r0, rows, nb, ratio, kb, block_topk, qsaAllBlocksVisible(offset + r0, nb, ratio));
        errdefer _ = mlx.mlx_array_free(sorted);
        try parts.append(self.allocator, sorted);
    }
    if (parts.items.len == 1) return parts.pop().?;
    const vec = mlx.mlx_vector_array_new_data(parts.items.ptr, parts.items.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var merged = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&merged, vec, 1, self.s));
    return merged;
}

pub fn qsaBatchedAttn(
    self: *Transformer,
    ctx: *ForwardCtx,
    slots: []const *ForwardCtx,
    q_rope: mlx.mlx_array,
    layer: u32,
    seq_len: c_int,
    attn_scale: f32,
) !?mlx.mlx_array {
    if (!qsaBatchedGatherEnabled()) return null;
    if (self.qwen4 == null) return null;
    if (ctx.qsa_mask.ctx != null) return null;
    var any_blocks = false;
    for (slots) |sc| if (sc.qsa_blocks.ctx != null) {
        any_blocks = true;
    };
    if (!any_blocks) return null;
    const ratio: c_int = @intCast(self.config.indexer_compress_ratio);
    var views = try self.allocator.alloc(DenseKVView, slots.len);
    defer {
        for (views) |*dv| dv.deinit();
        self.allocator.free(views);
    }
    var blks = try self.allocator.alloc(mlx.mlx_array, slots.len);
    defer self.allocator.free(blks);
    var slot_masks = try self.allocator.alloc(mlx.mlx_array, slots.len);
    defer self.allocator.free(slot_masks);
    const arms = try self.allocator.alloc(QsaArm, slots.len);
    defer self.allocator.free(arms);
    for (slots, 0..) |slot_ctx, i| {
        views[i] = try slot_ctx.cache.denseView(layer, self.s);
        blks[i] = slot_ctx.qsa_blocks;
        slot_masks[i] = slot_ctx.qsa_mask;
    }
    if (!qsa_batched_gather_logged) {
        qsa_batched_gather_logged = true;
        log.info("[qsa-batched-gather] engaged (slots={d} S={d})\n", .{ slots.len, seq_len });
    }
    const stacked = try qsaBatchedGatherAttn(self.allocator, self.s, q_rope, views, blks, slot_masks, ratio, attn_scale, arms, @intCast(self.config.indexer_budget));
    for (slots, arms) |slot_ctx, arm| {
        slot_ctx.qsa_arms.note(arm);
        if (self.cost_trace_active) self.cost_attention |= switch (arm) {
            .mask => @as(u8, 16),
            .decode_gather => 2,
            .verify_gather => 4,
            .prefill_gather => 8,
        };
    }
    return stacked;
}

/// Arm seam for the parity tests and the fwd-ubench A/B: false runs the composed chain.
pub var qsa_pool_rope_fused_override: ?bool = null;

pub fn qsaPoolRopeFusedEnabled() bool {
    return qsa_pool_rope_fused_override orelse true;
}

test "qsa pooled rope: one cos/sin build per forward, not one per full-attention layer" {
    // The tables are a property of the forward, not of the layer.
    try mlx.requireGpu();
    const s = mlx.gpuStream();

    var t: Transformer = undefined;
    t.s = s;
    t.allocator = testing.allocator;
    t.config = .{};
    t.config.head_dim = 256;
    t.config.partial_rotary_factor = 0.5;
    t.config.rope_theta = 10_000_000.0;
    t.config.mrope_section = .{ 11, 11, 10 };
    t.rope_freqs_yarn = null;
    t.yarn_inv_freq = null;
    t.fwd_gen = 0;
    t.qsa_pooled_rope = .{};
    defer t.qsa_pooled_rope.deinit();

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    var seq_off: usize = 0;
    var ctx: ForwardCtx = .{ .cache = &cache, .moe_seq_offset = &seq_off, .ssm_entries = null, .capture_hidden = null, .vision_embeddings = null };

    const rope_dims: c_int = 128;
    const base: c_int = 96;
    const step: c_int = 4; // = indexer_compress_ratio
    const n: c_int = 5;

    const cs = try t.qsaPooledCosSin(&ctx, rope_dims, base, step, n, .bfloat16);
    try testing.expectEqual(@as(usize, 1), t.qsa_pooled_rope.builds);
    for (0..11) |_| {
        const again = try t.qsaPooledCosSin(&ctx, rope_dims, base, step, n, .bfloat16);
        try testing.expect(again.cos.ctx == cs.cos.ctx and again.sin.ctx == cs.sin.ctx);
    }
    try testing.expectEqual(@as(usize, 1), t.qsa_pooled_rope.builds);

    var data: [@as(usize, @intCast(n)) * 128]f32 = undefined;
    for (&data, 0..) |*d, i| d.* = @floatFromInt((i % 37) + 1);
    const xshape = [_]c_int{ 1, 1, n, 128 };
    const xf = mlx.mlx_array_new_data(&data, &xshape, 4, .float32);
    defer _ = mlx.mlx_array_free(xf);
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_astype(&x, xf, .bfloat16, s));
    const hoisted = try t.ropeApplyCosSin(x, rope_dims, cs.cos, cs.sin);
    defer _ = mlx.mlx_array_free(hoisted);
    const per_layer = try t.ropeAtPositions(x, rope_dims, t.config.rope_theta, @floatFromInt(base), @floatFromInt(step), n);
    defer _ = mlx.mlx_array_free(per_layer);
    try testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(hoisted, per_layer, s));

    // A new forward at the same position hits: the forward counter is not in the key.
    t.fwd_gen += 1;
    _ = try t.qsaPooledCosSin(&ctx, rope_dims, base, step, n, .bfloat16);
    try testing.expectEqual(@as(usize, 1), t.qsa_pooled_rope.builds);
    _ = try t.qsaPooledCosSin(&ctx, rope_dims, base + step, step, n, .bfloat16);
    try testing.expectEqual(@as(usize, 2), t.qsa_pooled_rope.builds);

    // The M-RoPE arm is a different table at the same (base, step, n).
    const pos = try testing.allocator.alloc(i32, 3 * 256);
    defer testing.allocator.free(pos);
    for (pos, 0..) |*p, i| p.* = @intCast(i % 256);
    ctx.mrope_pos = pos;
    ctx.mrope_total = 256;
    const m = try t.qsaPooledCosSin(&ctx, rope_dims, base + step, step, n, .bfloat16);
    try testing.expectEqual(@as(usize, 3), t.qsa_pooled_rope.builds);
    const direct = try t.mropeCosSinAt(.{ .pos = pos, .total = 256, .delta = 0 }, @intCast(base + step), @intCast(step), @intCast(n), .bfloat16);
    defer _ = mlx.mlx_array_free(direct.cos);
    defer _ = mlx.mlx_array_free(direct.sin);
    try testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(m.cos, direct.cos, s));
    try testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(m.sin, direct.sin, s));
    for (0..11) |_| _ = try t.qsaPooledCosSin(&ctx, rope_dims, base + step, step, n, .bfloat16);
    try testing.expectEqual(@as(usize, 3), t.qsa_pooled_rope.builds);
}

test "qsa pooled rope: a mark from another thread is observed by the next lookup" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();

    var t: Transformer = undefined;
    t.s = s;
    t.allocator = testing.allocator;
    t.config = .{};
    t.config.head_dim = 256;
    t.config.partial_rotary_factor = 0.5;
    t.config.rope_theta = 10_000_000.0;
    t.config.mrope_section = .{ 11, 11, 10 };
    t.rope_freqs_yarn = null;
    t.yarn_inv_freq = null;
    t.fwd_gen = 0;
    t.qsa_pooled_rope = .{};
    defer t.qsa_pooled_rope.deinit();

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    var seq_off: usize = 0;
    var ctx: ForwardCtx = .{ .cache = &cache, .moe_seq_offset = &seq_off, .ssm_entries = null, .capture_hidden = null, .vision_embeddings = null };

    const rope_dims: c_int = 128;
    const base: c_int = 96;
    const step: c_int = 4;
    const n: c_int = 5;

    _ = try t.qsaPooledCosSin(&ctx, rope_dims, base, step, n, .bfloat16);
    try testing.expectEqual(@as(usize, 1), t.qsa_pooled_rope.builds);

    const Worker = struct {
        fn run(xfm: *Transformer) void {
            xfm.markQsaPooledRopeStale();
        }
    };
    const th = try std.Thread.spawn(.{}, Worker.run, .{&t});
    th.join();
    try testing.expect(t.qsa_pooled_rope.stale.load(.acquire));
    _ = try t.qsaPooledCosSin(&ctx, rope_dims, base, step, n, .bfloat16);
    try testing.expectEqual(@as(usize, 2), t.qsa_pooled_rope.builds);
}
