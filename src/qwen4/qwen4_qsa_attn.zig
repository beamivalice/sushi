const trf = @import("../transformer.zig");
const qwen4_qsa_select = @import("qwen4_qsa_select.zig");
const qwen4_qsa_gather = @import("qwen4_qsa_gather.zig");
const std = @import("std");
const mlx = @import("../mlx.zig");
const kv_quant = @import("../kv_quant.zig");
const qwen4_qsa = @import("qwen4_qsa.zig");
const noteQsaArmForSlots = qwen4_qsa.noteQsaArmForSlots;
const qsaAlignedSparseAttn = qwen4_qsa.qsaAlignedSparseAttn;
const qsaBatchedGatherAttn = qwen4_qsa.qsaBatchedGatherAttn;
const qsaPooledRowContiguous = qwen4_qsa.qsaPooledRowContiguous;
const qsaScoreFusedEligibleFrom = qwen4_qsa.qsaScoreFusedEligibleFrom;
const qsaScoreSheet = qwen4_qsa.qsaScoreSheet;
const KVQuantConfig = kv_quant.KVQuantConfig;
const model_mod = @import("../model.zig");
const log = @import("../log.zig");
const io_util_mod = @import("../io_util.zig");
const diagEnvOn = io_util_mod.diagEnvOn;
const testing = std.testing;
const test_support = @import("../test_support.zig");
const attn256MaxAbs = test_support.attn256MaxAbs;
const attn256MaxDiff = test_support.attn256MaxDiff;
const attn256RandBf16 = test_support.attn256RandBf16;
const attn256Reference = test_support.attn256Reference;
const qkvExpectBitsEqual = test_support.qkvExpectBitsEqual;
const qkvIdentityArray = test_support.qkvIdentityArray;
const testRandWeightBf16 = test_support.testRandWeightBf16;
const DenseKVView = trf.DenseKVView;
const FUSED256_MIN_Q_LEN = trf.FUSED256_MIN_Q_LEN;
const ForwardCtx = trf.ForwardCtx;
const KVCache = trf.KVCache;
const KvPrefixView = trf.KvPrefixView;
const ProfClock = trf.ProfClock;
const QSA_ATTN_BK = trf.QSA_ATTN_BK;
const QsaAttnCfgKey = trf.QsaAttnCfgKey;
const QsaBlockFixture = trf.QsaBlockFixture;
const SSMCacheEntry = trf.SSMCacheEntry;
const ShapeKey = trf.ShapeKey;
const Transformer = trf.Transformer;
const bf16Scalar = trf.bf16Scalar;
const gatherQsa256 = trf.gatherQsa256;
const maxAbsDiffF32 = trf.maxAbsDiffF32;
const qsaDecodeGatherEnabled = trf.qsaDecodeGatherEnabled;
const qsaDecodeGatherIdsHost = trf.qsaDecodeGatherIdsHost;
const qsaGatherEnabled = trf.qsaGatherEnabled;
const qsaScoreFusedActiveFor = trf.qsaScoreFusedActiveFor;
const qsaScoreRowsPerChunk = trf.qsaScoreRowsPerChunk;
const qsaScoreSheetComposed = trf.qsaScoreSheetComposed;
const qsaSelectKernelEnabled = trf.qsaSelectKernelEnabled;
const sliceAttentionSeq = trf.sliceAttentionSeq;
const splitKAttnDispatch = trf.splitKAttnDispatch;
const splitMaskedSdpa256 = trf.splitMaskedSdpa256;
const ssmFreeQsaState = trf.ssmFreeQsaState;

/// Measured split count (M5 Max, qwen4_exp, kv 62k, 8-bit KV, 12 counterbalanced boots):
/// every count 8..64 beats the union gather by 5-13%, only S=6's 16-vs-32 clears the noise
/// floor, so one flat constant. The un-split kernel (12 threadgroups at S=6) ran +15.6%.
pub const QSA_ATTN_NSPLIT_DEFAULT: c_int = 16;

/// Ceiling on the split count; past it the f32 partial buffer costs more traffic than it buys.
pub const QSA_ATTN_MAX_NSPLIT: c_int = 64;

/// Test and microbench seam: forces the split count.
pub var qsa_attn_nsplit_override: ?c_int = null;

/// Resolve every lazily-cached QSA env read once, from the main thread, before any other
/// thread exists: first touch of a `?bool`/`?c_int` from two threads is a race.
pub fn warmQsaEnvCaches() void {
    _ = qsaScoreSheetBudget();
}

pub fn qsaAttnNSplit() c_int {
    const forced = qsa_attn_nsplit_override orelse 0;
    if (forced > 0) {
        var f: c_int = 1;
        while (f < forced and f < QSA_ATTN_MAX_NSPLIT) f *= 2;
        return f;
    }
    return QSA_ATTN_NSPLIT_DEFAULT;
}

/// Test and microbench seam: forces the simdgroups per threadgroup (floored at the computing ones).
pub var qsa_attn_simdgroups_override: ?c_int = null;

/// Simdgroups per QSA split-K threadgroup: a small grid stages each tile with loader simdgroups.
pub fn qsaAttnSimdgroups(seq_len: c_int, h_kv: c_int, nsg: c_int, nsplit: c_int) c_int {
    const fallback: c_int = if (seq_len * h_kv * nsplit <= 32) 16 else nsg;
    return @max(nsg, @min(qsa_attn_simdgroups_override orelse fallback, 32));
}

/// Widest query width the split-K kernel serves on a PACKED cache: past it the gathers win
/// (measured crossover, 8k-256k keys).
pub const QSA_ATTN_PACKED_MAX_S: c_int = 40;
/// µbench seam: measure the kernel past its cap.
pub var qsa_attn_packed_max_s_override: ?c_int = null;

/// Times a packed gather may stage each cache row on average before one whole rebuild plus the
/// dense gather is cheaper (measured crossover, 4k-256k keys).
pub const QSA_PACKED_GATHER_MAX_REUSE: u64 = 4;


/// Rows a QSA forward of width `fwd` on a packed cache can rebuild: the dense span below the
/// indexer budget (plus a verify block's cut rows), and past the split-K widths every cache the
/// packed gather declines. Null = the packed arms are switched off: the whole cache rebuilds.
pub fn qsaDenseRebuildRows(budget: u32, ratio: u32, fwd: u64) ?u64 {
    if (budget == 0 or ratio == 0 or !qsaAttnKernelEnabled() or !qsaGatherEnabled()) return null;
    const span: u64 = @as(u64, budget) + ratio - 1 + FUSED256_MIN_Q_LEN - 1;
    if (fwd <= @as(u64, @intCast(QSA_ATTN_PACKED_MAX_S))) return span;
    return @max(span, fwd * budget / QSA_PACKED_GATHER_MAX_REUSE);
}

pub fn qsaSparseAttnServes(has_quant_triple: bool, seq_len: c_int) bool {
    if (seq_len < 1) return false;
    if (seq_len < FUSED256_MIN_Q_LEN) return true;
    return has_quant_triple and seq_len <= (qsa_attn_packed_max_s_override orelse QSA_ATTN_PACKED_MAX_S);
}

/// Microbench and test seam: forces the tile depth.
pub var qsa_attn_bk_override: ?c_int = null;

/// Threadgroup tile depth; a multiple of 8 (`KT = BK / 8`).
pub fn qsaAttnBk() c_int {
    const v = qsa_attn_bk_override orelse QSA_ATTN_BK;
    if (v <= 0 or @rem(v, 8) != 0 or v > 32) return QSA_ATTN_BK;
    return v;
}

pub var qsa_attn_kernel_override: ?bool = null;

pub fn qsaAttnKernelEnabled() bool {
    return qsa_attn_kernel_override orelse true;
}

pub fn qsaAttnCfgBuilds() usize {
    return trf.qsa_attn_cfgs.builds;
}
var qsa_attn_engaged_bits: OneShotBits = .{};

/// Every position uses the same row body for dense and quantized caches.
pub fn qsaSparseAttn(
    s: mlx.mlx_stream,
    q_rope: mlx.mlx_array, // [1, Hq, S, 256] bf16, post-RoPE
    kv_view: *const DenseKVView, // cache AFTER this block's append
    blocks: mlx.mlx_array, // [1, S, kb] int32 ascending, INT_MAX padded
    ratio: c_int,
    attn_scale: f32,
) !?mlx.mlx_array {
    if (!qsaAttnKernelEnabled()) return null;
    if (!mlx.streamIsGpu(s)) return null;
    if (mlx.mlx_array_ndim(q_rope) != 4 or mlx.mlx_array_ndim(blocks) != 3) return null;
    const qs = mlx.getShape(q_rope);
    const seq_len = qs[2];
    if (!qsaSparseAttnServes(kv_view.has_quant_triple, seq_len)) return null;
    if (qs[0] != 1 or qs[3] != 256) return null;
    if (mlx.mlx_array_dtype(q_rope) != .bfloat16) return null;
    if (mlx.mlx_array_dtype(blocks) != .int32) return null;
    const bs = mlx.getShape(blocks);
    if (bs[0] != 1 or bs[1] != seq_len or bs[2] <= 0) return null;
    // `kb` rides `blocks_shape[2]` into the kernel.
    _ = bs[2];
    if (ratio <= 0) return null;

    const quant = kv_view.has_quant_triple;
    const bits: u8 = if (quant) kv_view.bits else 8;
    const gs: u32 = if (quant) kv_view.group_size else 64;
    if (!KVQuantConfig.supportedBits(bits)) return null;
    if (gs == 0 or @rem(@as(u32, 256), gs) != 0 or gs % 8 != 0) return null;
    const arrays = if (quant)
        [_]mlx.mlx_array{ kv_view.k_triple_q, kv_view.k_triple_scales, kv_view.k_triple_biases, kv_view.v_triple_q, kv_view.v_triple_scales, kv_view.v_triple_biases }
    else
        [_]mlx.mlx_array{ kv_view.k, kv_view.k, kv_view.k, kv_view.v, kv_view.v, kv_view.v };
    for (arrays) |array| {
        if (array.ctx == null or mlx.mlx_array_ndim(array) != 4 or mlx.mlx_array_strides(array)[3] != 1) return null;
    }
    if (!quant and (mlx.mlx_array_dtype(arrays[0]) != .bfloat16 or mlx.mlx_array_dtype(arrays[3]) != .bfloat16)) return null;
    const ks = mlx.getShape(arrays[0]);
    const vs = mlx.getShape(arrays[3]);
    const row_cols: c_int = if (quant) kv_quant.packedWords(256, bits) else 256;
    if (ks[0] != 1 or vs[0] != 1 or ks[3] != row_cols or vs[3] != row_cols) return null;
    const h_kv = ks[1];
    const kv = ks[2];
    if (h_kv <= 0 or kv < seq_len or vs[1] != h_kv or vs[2] != kv or @rem(qs[1], h_kv) != 0) return null;
    const gqa = @divExact(qs[1], h_kv);
    if (gqa <= 0 or gqa > 64) return null;
    if (quant) {
        for ([_]usize{ 1, 2, 4, 5 }) |i| {
            const shape = mlx.getShape(arrays[i]);
            if (shape[0] != 1 or shape[1] != h_kv or shape[2] != kv or shape[3] != @as(c_int, @intCast(256 / gs))) return null;
        }
    }
    if (mlx.mlx_array_strides(q_rope)[3] != 1 or mlx.mlx_array_strides(blocks)[2] != 1) return null;

    const key = QsaAttnCfgKey{
        .q_shape = ShapeKey.from(qs),
        .h_kv = h_kv,
        .dv = 256,
        .bits = bits,
        .gs = gs,
        .quant = quant,
        .sparse = true,
        .nsg = @divTrunc(gqa + 7, 8),
        .nsgt = qsaAttnSimdgroups(seq_len, h_kv, @divTrunc(gqa + 7, 8), qsaAttnNSplit()),        .nsplit = qsaAttnNSplit(),
        .bk = qsaAttnBk(),
        .ratio = ratio,
        .dtype = .bfloat16,
    };
    const out = (try splitKAttnDispatch(s, q_rope, blocks, arrays, attn_scale, key)) orelse return null;
    if (qsa_attn_engaged_bits.take(qsaWidthBucket(seq_len))) {
        log.info(
            "[qsa-attn] engaged (S={d} kv={d} quant={d}/gs{d} gqa={d} bk={d} nsplit={d} tgs={d})\n",
            .{ seq_len, kv, kv_view.bits, kv_view.group_size, gqa, key.bk, key.nsplit, seq_len * h_kv * key.nsplit },
        );
    }
    return out;
}

/// One-shot gate keyed by a small bit index. `take` returns true the FIRST
/// time each bit is asked for.
pub const OneShotBits = struct {
    seen: u32 = 0,

    pub fn take(self: *OneShotBits, bit: u5) bool {
        const m: u32 = @as(u32, 1) << bit;
        if (self.seen & m != 0) return false;
        self.seen |= m;
        return true;
    }
    pub fn reset(self: *OneShotBits) void {
        self.seen = 0;
    }
};

/// Which arm served one qwen4_exp full-attention call.
pub const QsaArm = enum(u2) { mask = 0, decode_gather = 1, verify_gather = 2, prefill_gather = 3 };

/// Query-width bucket: decode (1), verify (2 .. FUSED256_MIN_Q_LEN-1),
/// prefill (>= FUSED256_MIN_Q_LEN). The meters are one-shot PER BUCKET so a
/// decode-width engagement at 16k cannot vouch for a verify-width call at
/// 62k — which is exactly the confusion a single global bool created.
pub fn qsaWidthBucket(seq_len: c_int) u2 {
    if (seq_len <= 1) return 0;
    if (seq_len < FUSED256_MIN_Q_LEN) return 1;
    return 2;
}

/// Why a gather arm handed the call back to the dense mask.
pub const QsaDeclineReason = enum(u5) {
    disabled,
    width,
    geometry,
    dtype,
    kv_floor,
    no_win,
    quant_layout,
    gather_shape,

    pub fn text(self: QsaDeclineReason) []const u8 {
        return switch (self) {
            .disabled => "kill switch",
            .width => "query width not this arm's",
            .geometry => "geometry",
            .dtype => "dtype",
            .kv_floor => "kv below the arm's floor",
            .no_win => "no sparsity win (gathered rows >= kv)",
            .quant_layout => "quant triple layout",
            .gather_shape => "gathered slab shape",
        };
    }
};

/// Decline reporter threaded through one gather call: `no(reason)` logs at
/// most once per (arm, reason) for the process and hands the caller the
/// decline. `s`/`kv` are filled in as soon as the guards learn them (-1 =
/// not yet known at that guard).
pub const QsaDecline = struct {
    bits: *OneShotBits,
    arm: []const u8,
    s: c_int = -1,
    kv: c_int = -1,

    pub fn no(self: *QsaDecline, r: QsaDeclineReason) ?mlx.mlx_array {
        if (self.bits.take(@backingInt(r))) {
            log.info(
                "[{s}] declined: {s} (S={d} kv={d}) — the dense [S, kv] mask arm serves this call\n",
                .{ self.arm, r.text(), self.s, self.kv },
            );
        }
        return null;
    }
};

/// Per-request tally of which arm served each full-attention call. One
/// increment per attention layer; read and reset where the request's
/// `[qsa-arms]` line is emitted. It lives on the REQUEST's `ForwardCtx`
/// (the Generator's own), never on the shared Transformer: qwen4 text slots
/// batch-decode on one Transformer and a cold prefill interleaves with other
/// slots' decode ticks, so a Transformer-level tally prints the union of
/// every concurrent slot and leaves the next finisher an empty one.
pub const QsaArmCounts = struct {
    mask: u32 = 0,
    decode: u32 = 0,
    verify: u32 = 0,
    prefill: u32 = 0,

    pub fn note(self: *QsaArmCounts, arm: QsaArm) void {
        switch (arm) {
            .mask => self.mask +|= 1,
            .decode_gather => self.decode +|= 1,
            .verify_gather => self.verify +|= 1,
            .prefill_gather => self.prefill +|= 1,
        }
    }
    /// A gather served the call: which gather is the query WIDTH's business.
    pub fn noteGather(self: *QsaArmCounts, seq_len: c_int) void {
        self.note(switch (qsaWidthBucket(seq_len)) {
            0 => .decode_gather,
            1 => .verify_gather,
            else => .prefill_gather,
        });
    }
    pub fn total(self: QsaArmCounts) u32 {
        return self.mask +| self.decode +| self.verify +| self.prefill;
    }
    /// Read and clear this request's tally. Null when nothing was counted
    /// (not a QSA arch, or below the sparse budget) so a caller never prints
    /// a meaningless `qsa=mask 0/0`.
    pub fn take(self: *QsaArmCounts) ?QsaArmCounts {
        const c = self.*;
        self.reset();
        if (c.total() == 0) return null;
        return c;
    }
    /// The arm that served the most calls; null when nothing was counted.
    /// Ties break toward the CHEAPEST arm that could have run so a tie never
    /// reads as "the mask arm is fine".
    pub fn majority(self: QsaArmCounts) ?QsaArm {
        if (self.total() == 0) return null;
        var best: QsaArm = .mask;
        var best_n: u32 = self.mask;
        if (self.decode >= best_n) {
            best = .decode_gather;
            best_n = self.decode;
        }
        if (self.verify >= best_n) {
            best = .verify_gather;
            best_n = self.verify;
        }
        if (self.prefill >= best_n) {
            best = .prefill_gather;
            best_n = self.prefill;
        }
        return best;
    }
    pub fn reset(self: *QsaArmCounts) void {
        self.* = .{};
    }
};


/// Engagement meters, one-shot per (arm, width bucket): bit = arm*4 + bucket.
pub var qsa_engaged_bits: OneShotBits = .{};
var qsa_decode_decline_bits: OneShotBits = .{};
var qsa_verify_decline_bits: OneShotBits = .{};

pub fn qsaEngagedBit(arm: QsaArm, seq_len: c_int) u5 {
    return @as(u5, @backingInt(arm)) * 4 + @as(u5, qsaWidthBucket(seq_len));
}

/// take_axis + materialized copy: quantized kernels (dequantize included)
/// misread the strided view take hands back, so the gather-then-dequantize
/// path must own contiguous rows first. NOTE: not the `+0` trick — an
/// add-materialized take of packed rows still reads back wrong, so this uses
/// the canonical contiguous primitive.
fn takeContig(s: mlx.mlx_stream, src: mlx.mlx_array, idx: mlx.mlx_array, axis: c_int) !mlx.mlx_array {
    // One owner for `view`: an `errdefer` here plus a `defer` after the take double-freed it
    // when `mlx_contiguous` failed (reachable since #353 turned exit(-1) into an unwind).
    var view = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(view);
    try mlx.check(mlx.mlx_take_axis(&view, src, idx, axis, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_contiguous(&out, view, false, s));
    return out;
}

/// Decode-width QSA gather (S==1, sorted `blocks`, no mask): expand the
/// selection + incomplete tail into ascending token indices, gather those rows
/// (quant triples when the view carries them — dequantizing ONLY the subset —
/// else dense rows), then dense SDPA under an additive validity mask. Null =
/// declined (kill switch, dtype, geometry) — the caller keeps the dense mask.
///
/// Parity argument: the mask arm attends exactly the selected complete blocks'
/// tokens ∪ the incomplete tail ∧ causal. At decode the query is the last
/// cache row, so every gathered token is causal-visible and the ascending
/// gather order matches the masked row's summation order; the masked-out slots
/// contributed exact zeros there. Remaining deltas are kernel tiling order.
pub fn qsaDecodeGatherAttn(
    s: mlx.mlx_stream,
    q_rope: mlx.mlx_array, // [1,Hq,1,D] bf16
    kv_view: *const DenseKVView, // full cache AFTER this token's append
    blocks: mlx.mlx_array, // [1,1,kb] int32 sorted asc, INT_MAX sentinel
    ratio: c_int,
    attn_scale: f32,
) !?mlx.mlx_array {
    var dq = QsaDecline{ .bits = &qsa_decode_decline_bits, .arm = "qsa-decode-gather" };
    if (!qsaDecodeGatherEnabled()) return dq.no(.disabled);
    if (mlx.mlx_array_ndim(q_rope) != 4 or mlx.mlx_array_ndim(blocks) != 3) return dq.no(.geometry);
    const qs = mlx.getShape(q_rope);
    dq.s = qs[2];
    if (qs[0] != 1) return dq.no(.geometry);
    if (qs[2] != 1) return dq.no(.width);
    if (mlx.mlx_array_dtype(q_rope) != .bfloat16) return dq.no(.dtype);
    if (mlx.mlx_array_dtype(blocks) != .int32) return dq.no(.dtype);
    const bs = mlx.getShape(blocks);
    if (bs[0] != 1 or bs[1] != 1) return dq.no(.geometry);
    const kb: c_int = bs[2];
    if (kb <= 0 or ratio <= 0) return dq.no(.geometry);
    const h_q: c_int = qs[1];
    const head_dim: c_int = qs[3];
    if (h_q <= 0 or head_dim <= 0) return dq.no(.geometry);

    // Cache geometry off the dense K (present in every scheme; shapes are
    // known without executing the lazy dequant graph).
    if (kv_view.k.ctx == null or kv_view.v.ctx == null) return dq.no(.geometry);
    const ks = mlx.getShape(kv_view.k);
    if (ks.len != 4 or ks[0] != 1) return dq.no(.geometry);
    const h_kv: c_int = ks[1];
    const kv: c_int = ks[2];
    dq.kv = kv;
    if (h_kv <= 0 or kv <= 0 or @rem(h_q, h_kv) != 0) return dq.no(.geometry);
    if (ks[3] != head_dim) return dq.no(.geometry);
    const vs = mlx.getShape(kv_view.v);
    if (vs.len != 4 or vs[0] != 1 or vs[1] != h_kv or vs[2] != kv or vs[3] != head_dim) return dq.no(.geometry);

    const nb: c_int = @divTrunc(kv, ratio);
    const tail_start: c_int = nb * ratio;
    const tail_len: c_int = kv - tail_start;
    if (tail_len < 0) return dq.no(.geometry);
    const sel_len: c_int = kb * ratio;
    const gathered_len: c_int = sel_len + tail_len;
    // No sparsity win (or degenerate): keep the mask arm.
    if (gathered_len <= 0 or gathered_len >= kv) return dq.no(.no_win);

    // Token ids: selected blocks' ratio tokens each (sentinel → 0, masked
    // below) then the incomplete tail. Reachable decode rows have no
    // sentinels — every complete block is visible from the last position, so
    // visible == nb >= kb — and this stays ascending; the mask is the
    // backstop, not the plan.
    const nb_arr = mlx.mlx_array_new_int(nb);
    defer _ = mlx.mlx_array_free(nb_arr);
    var valid_b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(valid_b);
    try mlx.check(mlx.mlx_less(&valid_b, blocks, nb_arr, s)); // [1,1,kb] bool
    const zero_i = mlx.mlx_array_new_int(0);
    defer _ = mlx.mlx_array_free(zero_i);
    var safe_b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(safe_b);
    try mlx.check(mlx.mlx_where(&safe_b, valid_b, blocks, zero_i, s));
    var blk4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(blk4);
    try mlx.check(mlx.mlx_expand_dims(&blk4, safe_b, -1, s)); // [1,1,kb,1]
    var rvec = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rvec);
    try mlx.check(mlx.mlx_arange(&rvec, 0, @floatFromInt(ratio), 1.0, .int32, s)); // [ratio]
    var r4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r4);
    try mlx.check(mlx.mlx_reshape(&r4, rvec, &[_]c_int{ 1, 1, 1, ratio }, 4, s));
    const ratio_arr = mlx.mlx_array_new_int(ratio);
    defer _ = mlx.mlx_array_free(ratio_arr);
    var blk_scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(blk_scaled);
    try mlx.check(mlx.mlx_multiply(&blk_scaled, blk4, ratio_arr, s));
    var tok4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tok4);
    try mlx.check(mlx.mlx_add(&tok4, blk_scaled, r4, s)); // [1,1,kb,ratio]
    var tok_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tok_flat);
    try mlx.check(mlx.mlx_reshape(&tok_flat, tok4, &[_]c_int{sel_len}, 1, s));
    var valid4e = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(valid4e);
    try mlx.check(mlx.mlx_expand_dims(&valid4e, valid_b, -1, s)); // [1,1,kb,1]
    var valid4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(valid4);
    try mlx.check(mlx.mlx_broadcast_to(&valid4, valid4e, &[_]c_int{ 1, 1, kb, ratio }, 4, s));
    var valid_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(valid_flat);
    try mlx.check(mlx.mlx_reshape(&valid_flat, valid4, &[_]c_int{sel_len}, 1, s));
    var tail_ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tail_ids);
    var tail_ok = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tail_ok);
    if (tail_len > 0) {
        try mlx.check(mlx.mlx_arange(&tail_ids, @floatFromInt(tail_start), @floatFromInt(kv), 1.0, .int32, s));
        try mlx.check(mlx.mlx_ones(&tail_ok, &[_]c_int{tail_len}, 1, .bool_, s));
    } else {
        const eshp = [_]c_int{0};
        try mlx.check(mlx.mlx_zeros(&tail_ids, &eshp, 1, .int32, s));
        try mlx.check(mlx.mlx_zeros(&tail_ok, &eshp, 1, .bool_, s));
    }
    const idx_parts = [_]mlx.mlx_array{ tok_flat, tail_ids };
    const idx_vec = mlx.mlx_vector_array_new_data(&idx_parts, 2);
    defer _ = mlx.mlx_vector_array_free(idx_vec);
    var idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(idx);
    try mlx.check(mlx.mlx_concatenate_axis(&idx, idx_vec, 0, s)); // [K'] int32
    const vok_parts = [_]mlx.mlx_array{ valid_flat, tail_ok };
    const vok_vec = mlx.mlx_vector_array_new_data(&vok_parts, 2);
    defer _ = mlx.mlx_vector_array_free(vok_vec);
    var vok = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vok);
    try mlx.check(mlx.mlx_concatenate_axis(&vok, vok_vec, 0, s)); // [K'] bool
    // take_axis wants unsigned indices; every id is >= 0 by construction.
    var idx_u32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(idx_u32);
    try mlx.check(mlx.mlx_astype(&idx_u32, idx, .uint32, s));

    // Additive validity mask, same bf16 convention as the sliding masks.
    const zero_f = bf16Scalar(0.0, s);
    defer _ = mlx.mlx_array_free(zero_f);
    const neg_inf = bf16Scalar(-std.math.inf(f32), s);
    defer _ = mlx.mlx_array_free(neg_inf);
    var addv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(addv);
    try mlx.check(mlx.mlx_where(&addv, vok, zero_f, neg_inf, s));
    var add4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(add4);
    try mlx.check(mlx.mlx_reshape(&add4, addv, &[_]c_int{ 1, 1, 1, gathered_len }, 4, s));

    // Gather rows: quant triples when the view carries affine 4/6/8-bit (then
    // dequantize ONLY the subset), else dense rows. Dequantize is per-token
    // along D, so subset-first is exactly full-first-restricted.
    var gk = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gk);
    var gv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gv);
    if (kv_view.has_quant_triple and KVQuantConfig.supportedBits(kv_view.bits)) {
        const kq = kv_view.k_triple_q;
        const vq = kv_view.v_triple_q;
        if (kq.ctx == null or vq.ctx == null) return dq.no(.quant_layout);
        const kqs = mlx.getShape(kq);
        const vqs = mlx.getShape(vq);
        if (kqs.len != 4 or vqs.len != 4 or kqs[0] != 1 or vqs[0] != 1 or
            kqs[1] != h_kv or vqs[1] != h_kv or kqs[2] != kv or vqs[2] != kv) return dq.no(.quant_layout);
        // Contiguous copies: dequantize misreads the strided take views.
        const gkq = try takeContig(s, kq, idx_u32, 2);
        defer _ = mlx.mlx_array_free(gkq);
        const gks = try takeContig(s, kv_view.k_triple_scales, idx_u32, 2);
        defer _ = mlx.mlx_array_free(gks);
        const gkb = try takeContig(s, kv_view.k_triple_biases, idx_u32, 2);
        defer _ = mlx.mlx_array_free(gkb);
        const gvq = try takeContig(s, vq, idx_u32, 2);
        defer _ = mlx.mlx_array_free(gvq);
        const gvs = try takeContig(s, kv_view.v_triple_scales, idx_u32, 2);
        defer _ = mlx.mlx_array_free(gvs);
        const gvb = try takeContig(s, kv_view.v_triple_biases, idx_u32, 2);
        defer _ = mlx.mlx_array_free(gvb);
        const gs: u32 = kv_view.group_size;
        if (gs == 0) return dq.no(.quant_layout);
        const gkd = try kv_quant.dequantizeAffine(s, gkq, gks, gkb, gs, kv_view.bits);
        defer _ = mlx.mlx_array_free(gkd);
        const gvd = try kv_quant.dequantizeAffine(s, gvq, gvs, gvb, gs, kv_view.bits);
        defer _ = mlx.mlx_array_free(gvd);
        try mlx.check(mlx.mlx_array_set(&gk, gkd));
        try mlx.check(mlx.mlx_array_set(&gv, gvd));
    } else {
        if (kv_view.k.ctx == null or kv_view.v.ctx == null) return dq.no(.geometry);
        try mlx.check(mlx.mlx_take_axis(&gk, kv_view.k, idx_u32, 2, s));
        try mlx.check(mlx.mlx_take_axis(&gv, kv_view.v, idx_u32, 2, s));
    }
    if (mlx.mlx_array_dtype(gk) != mlx.mlx_array_dtype(q_rope)) return dq.no(.dtype);
    const gks2 = mlx.getShape(gk);
    const gvs2 = mlx.getShape(gv);
    if (gks2.len != 4 or gvs2.len != 4 or gks2[0] != 1 or gvs2[0] != 1 or
        gks2[1] != h_kv or gvs2[1] != h_kv or gks2[2] != gathered_len or gvs2[2] != gathered_len or
        gks2[3] != head_dim or gvs2[3] != head_dim) return dq.no(.gather_shape);

    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&out, q_rope, gk, gv, attn_scale, "array", add4, .{ .ctx = null }, false, s));
    if (qsa_engaged_bits.take(qsaEngagedBit(.decode_gather, qs[2]))) {
        log.info("[qsa-decode-gather] engaged (S={d} kv={d} rows={d})\n", .{ qs[2], kv, gathered_len });
    }
    return out;
}

pub var qsa_verify_gather_override: ?bool = null;

/// Verify-width QSA gather (2 <= S < FUSED256_MIN_Q_LEN — an MTP/DFlash
/// verify block): read the UNION of the S rows' selections + the union tail
/// instead of the whole cache under a dense [S, kv] mask. With `--kv-quant 8`
/// the dense mask arm also dequantizes the whole stored range per layer per
/// forward; this arm dequantizes only the gathered rows.
/// The override restores the dense mask arm.
pub fn qsaVerifyGatherEnabled() bool {
    return qsa_verify_gather_override orelse true;
}

/// The union is FIXED-size (S * budget blocks): at the production budget
/// (512 blocks of 4) S=7 is 14336 rows. On dense KV the gather COPIES those
/// rows while the mask arm reads the cache in place, so the copy only pays
/// once the union is well under half the cache (M4 Max, S=7: gather loses 7%
/// at 17k keys, breaks even at 34k). Quantized KV dequantizes only the
/// gathered rows, so its floor stays lower.
pub const QSA_VERIFY_GATHER_MIN_KV_DENSE: c_int = 32768;
pub const QSA_VERIFY_GATHER_MIN_KV_QUANT: c_int = 16384;
pub var qsa_verify_gather_min_kv_override: ?c_int = null;

/// `quantized` = the arm will dequantize only the gathered rows, i.e. the
/// view carries affine triples (`DenseKVView.has_quant_triple`).
pub fn qsaVerifyGatherMinKvFor(quantized: bool) c_int {
    if (qsa_verify_gather_min_kv_override) |v| return v;
    return if (quantized) QSA_VERIFY_GATHER_MIN_KV_QUANT else QSA_VERIFY_GATHER_MIN_KV_DENSE;
}

/// Geometry of the verify-width QSA gather — pure arithmetic, the ONE owner
/// of R (the gate, the MLX gatherer and the host reference plan all read it).
///
/// The gathered set is two DISJOINT parts:
///   A. the union tail `[tail_start, kv)`, where `tail_start` is row 0's own
///      tail start `ratio * floor((offset+1)/ratio)`. Tail starts are
///      non-decreasing in position and tails are SUFFIXES, so row 0's (the
///      SMALLEST start) is the largest and covers every other row's.
///   B. the union of the rows' selected blocks that lie ENTIRELY below
///      `tail_start` (`b < nb_lo`), deduplicated and padded to a fixed
///      `k_blocks` slots with blocks no row selected. A padded block's tokens
///      are below every row's tail start and in no row's selection, so the
///      per-row mask hides them everywhere — no host sync is needed to learn
///      the real union size.
/// Selected blocks at/above `tail_start` need no slot: their tokens are in A.
pub const QsaVerifyGeom = struct {
    /// Complete blocks in the cache.
    nb: c_int,
    /// Blocks lying entirely below the union tail (`tail_start / ratio`).
    nb_lo: c_int,
    /// Row 0's tail start = the union tail's first token.
    tail_start: c_int,
    tail_len: c_int,
    /// Fixed union slot count (part B).
    k_blocks: c_int,
    /// R — gathered rows total.
    rows: c_int,

    pub fn compute(seq_len: c_int, kv: c_int, ratio: c_int, kb: c_int) ?QsaVerifyGeom {
        if (ratio <= 0 or kb <= 0) return null;
        if (seq_len < 2 or seq_len >= FUSED256_MIN_Q_LEN) return null;
        if (kv <= seq_len) return null;
        const offset = kv - seq_len;
        const nb = @divTrunc(kv, ratio);
        const tail_start = @divTrunc(offset + 1, ratio) * ratio;
        if (tail_start > kv) return null;
        const nb_lo = @divTrunc(tail_start, ratio);
        const k_blocks = @min(nb_lo, seq_len * kb);
        return .{
            .nb = nb,
            .nb_lo = nb_lo,
            .tail_start = tail_start,
            .tail_len = kv - tail_start,
            .k_blocks = k_blocks,
            .rows = k_blocks * ratio + (kv - tail_start),
        };
    }
};

/// Host reference plan for the verify gather: the ascending gathered token
/// ids and the `[S, R]` visibility mask. Executable spec of the MLX arm —
/// `qsaVerifyGatherAttn` builds the same two arrays in ops, and the parity
/// test pins its row ids against this.
pub const QsaVerifyPlan = struct {
    rows: []i32,
    mask: []bool, // [seq_len, rows.len], row-major

    pub fn deinit(self: *QsaVerifyPlan, alloc: std.mem.Allocator) void {
        alloc.free(self.rows);
        alloc.free(self.mask);
        self.rows = &.{};
        self.mask = &.{};
    }
};

pub fn qsaVerifyPlanHost(
    alloc: std.mem.Allocator,
    blocks: []const i32, // [seq_len, kb] ascending, INT_MAX past each row's count
    seq_len: c_int,
    kb: c_int,
    kv: c_int,
    ratio: c_int,
) !QsaVerifyPlan {
    const geom = QsaVerifyGeom.compute(seq_len, kv, ratio, kb) orelse return error.QsaVerifyGeometry;
    const su: usize = @intCast(seq_len);
    const kbu: usize = @intCast(kb);
    if (blocks.len != su * kbu) return error.QsaVerifyBadSelection;
    const ru: usize = @intCast(geom.rows);
    const ku: usize = @intCast(geom.k_blocks);

    const in_union = try alloc.alloc(bool, @intCast(geom.nb_lo));
    defer alloc.free(in_union);
    @memset(in_union, false);
    for (0..su) |i| {
        for (0..kbu) |c| {
            const b = blocks[i * kbu + c];
            if (b >= 0 and b < geom.nb_lo) in_union[@intCast(b)] = true;
        }
    }
    // Union members ascending, then non-union fillers ascending, truncated to
    // the fixed slot count — the `where(u, b, nb_lo + b)` argsort key the MLX
    // arm uses — then sorted so the gathered rows come out ascending.
    const ids = try alloc.alloc(i32, ku);
    defer alloc.free(ids);
    var w: usize = 0;
    for (in_union, 0..) |u, b| {
        if (u and w < ku) {
            ids[w] = @intCast(b);
            w += 1;
        }
    }
    for (in_union, 0..) |u, b| {
        if (!u and w < ku) {
            ids[w] = @intCast(b);
            w += 1;
        }
    }
    std.mem.sort(i32, ids, {}, std.sort.asc(i32));

    const rows = try alloc.alloc(i32, ru);
    errdefer alloc.free(rows);
    var r: usize = 0;
    for (ids) |b| {
        var j: c_int = 0;
        while (j < ratio) : (j += 1) {
            rows[r] = b * ratio + j;
            r += 1;
        }
    }
    var t: c_int = geom.tail_start;
    while (t < kv) : (t += 1) {
        rows[r] = t;
        r += 1;
    }

    const mask = try alloc.alloc(bool, su * ru);
    errdefer alloc.free(mask);
    @memset(mask, false);
    const offset = kv - seq_len;
    for (0..su) |i| {
        const p: c_int = offset + @as(c_int, @intCast(i));
        const ts: c_int = @divTrunc(p + 1, ratio) * ratio;
        for (rows, 0..) |tok, slot| {
            if (tok > p) continue;
            var vis = tok >= ts;
            if (!vis) {
                const b: i32 = @divTrunc(tok, ratio);
                if (b < geom.nb) {
                    for (0..kbu) |c| {
                        if (blocks[i * kbu + c] == b) {
                            vis = true;
                            break;
                        }
                    }
                }
            }
            mask[i * ru + slot] = vis;
        }
    }
    return .{ .rows = rows, .mask = mask };
}

pub const QsaSlotPlan = struct {
    rows: []i32,
    mask: []bool,

    pub fn deinit(self: *QsaSlotPlan, alloc: std.mem.Allocator) void {
        alloc.free(self.rows);
        if (self.mask.len != 0) alloc.free(self.mask);
        self.rows = &.{};
        self.mask = &.{};
    }
};

pub fn qsaBatchedPlanHost(
    alloc: std.mem.Allocator,
    slot_blocks: []const []const i32,
    seq_len: c_int,
    kb: c_int,
    kv_lens: []const c_int,
    ratio: c_int,
) ![]QsaSlotPlan {
    if (slot_blocks.len != kv_lens.len) return error.QsaBatchedPlanLen;
    const out = try alloc.alloc(QsaSlotPlan, slot_blocks.len);
    var built: usize = 0;
    errdefer {
        for (out[0..built]) |*p| p.deinit(alloc);
        alloc.free(out);
    }
    for (slot_blocks, kv_lens, 0..) |blocks, kv, i| {
        if (seq_len == 1) {
            const rows = try qsaDecodeGatherIdsHost(alloc, blocks, kb, kv, ratio);
            out[i] = .{ .rows = rows, .mask = &.{} };
        } else {
            var plan = try qsaVerifyPlanHost(alloc, blocks, seq_len, kb, kv, ratio);
            out[i] = .{ .rows = plan.rows, .mask = plan.mask };
            plan.rows = &.{};
            plan.mask = &.{};
        }
        built = i + 1;
    }
    return out;
}

/// Verify-width QSA gather (2 <= S < FUSED256_MIN_Q_LEN, sorted `blocks`):
/// gather the fixed-size union of the S rows' selections + the union tail
/// (see `QsaVerifyGeom`), dequantizing ONLY those rows on a quantized cache,
/// then dense SDPA under the per-row `[S, R]` additive mask. Null = declined
/// (kill switch, kv floor, no sparsity win, dtype, geometry) — the caller
/// falls back to the dense `[S, kv]` mask arm.
///
/// Parity argument: the gathered rows are STRICTLY ASCENDING (part B blocks
/// are all below the tail's first token and the block ids are sorted), so no
/// key can be counted twice; the mask is the dense formula evaluated at the
/// gathered tokens; and every token the dense mask shows a row is present
/// (a token below the union tail that a row sees is in that row's selection,
/// hence in the union, and `k_blocks >= |union|` by construction). Remaining
/// deltas are kernel tiling order.
pub fn qsaVerifyGatherAttn(
    s: mlx.mlx_stream,
    q_rope: mlx.mlx_array, // [1,Hq,S,D] bf16
    kv_view: *const DenseKVView, // full cache AFTER this block's append
    blocks: mlx.mlx_array, // [1,S,kb] int32 sorted asc, INT_MAX sentinel
    ratio: c_int,
    attn_scale: f32,
) !?mlx.mlx_array {
    var dq = QsaDecline{ .bits = &qsa_verify_decline_bits, .arm = "qsa-verify-gather" };
    if (!qsaVerifyGatherEnabled()) return dq.no(.disabled);
    if (mlx.mlx_array_ndim(q_rope) != 4 or mlx.mlx_array_ndim(blocks) != 3) return dq.no(.geometry);
    if (mlx.mlx_array_dtype(q_rope) != .bfloat16 or mlx.mlx_array_dtype(blocks) != .int32) return dq.no(.dtype);
    const qs = mlx.getShape(q_rope);
    if (qs[0] != 1) return dq.no(.geometry);
    const seq_len: c_int = qs[2];
    dq.s = seq_len;
    const head_dim: c_int = qs[3];
    const h_q: c_int = qs[1];
    if (h_q <= 0 or head_dim <= 0) return dq.no(.geometry);
    const bs = mlx.getShape(blocks);
    if (bs[0] != 1 or bs[1] != seq_len or bs[2] <= 0) return dq.no(.geometry);
    const kb: c_int = bs[2];

    if (kv_view.k.ctx == null or kv_view.v.ctx == null) return dq.no(.geometry);
    const ks = mlx.getShape(kv_view.k);
    if (ks.len != 4 or ks[0] != 1) return dq.no(.geometry);
    const h_kv: c_int = ks[1];
    const kv: c_int = ks[2];
    dq.kv = kv;
    if (h_kv <= 0 or kv <= 0 or @rem(h_q, h_kv) != 0) return dq.no(.geometry);
    if (ks[3] != head_dim) return dq.no(.geometry);
    const vs = mlx.getShape(kv_view.v);
    if (vs.len != 4 or vs[0] != 1 or vs[1] != h_kv or vs[2] != kv or vs[3] != head_dim) return dq.no(.geometry);
    if (kv <= qsaVerifyGatherMinKvFor(kv_view.has_quant_triple)) return dq.no(.kv_floor);

    const geom = QsaVerifyGeom.compute(seq_len, kv, ratio, kb) orelse return dq.no(.width);
    if (geom.k_blocks <= 0 or geom.rows <= 0 or geom.rows >= kv) return dq.no(.no_win);
    const offset: c_int = kv - seq_len;
    const nb = geom.nb;
    const sel_w: c_int = nb + 2; // [.. nb-1] real, `nb` = the always-false
    // membership sink for tokens past the last complete block, `nb+1` the
    // sentinel sink.

    // Per-row block selection [S, nb+2] bool.
    const nb_arr = mlx.mlx_array_new_int(nb);
    defer _ = mlx.mlx_array_free(nb_arr);
    const sink_arr = mlx.mlx_array_new_int(nb + 1);
    defer _ = mlx.mlx_array_free(sink_arr);
    var valid_b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(valid_b);
    try mlx.check(mlx.mlx_less(&valid_b, blocks, nb_arr, s));
    var safe_b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(safe_b);
    try mlx.check(mlx.mlx_where(&safe_b, valid_b, blocks, sink_arr, s));
    var falses = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(falses);
    try mlx.check(mlx.mlx_zeros(&falses, &[_]c_int{ 1, seq_len, sel_w }, 3, .bool_, s));
    const true_v = mlx.mlx_array_new_bool(true);
    defer _ = mlx.mlx_array_free(true_v);
    var sel3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sel3);
    try mlx.check(mlx.mlx_put_along_axis(&sel3, falses, safe_b, true_v, -1, s));
    var sel = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sel);
    try mlx.check(mlx.mlx_reshape(&sel, sel3, &[_]c_int{ seq_len, sel_w }, 2, s)); // [S, nb+2]

    // Union over rows of the blocks BELOW the union tail, then a fixed-size
    // ascending id list: argsort a key that ranks union members first
    // (`b` vs `nb_lo + b`), keep k_blocks, sort so the rows come out
    // ascending. No host sync — the padding blocks mask out everywhere.
    const strides2 = [_]c_int{ 1, 1 };
    var sel_lo = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sel_lo);
    try mlx.check(mlx.mlx_slice(&sel_lo, sel, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ seq_len, geom.nb_lo }, 2, &strides2, 2, s));
    var union_b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(union_b);
    try mlx.check(mlx.mlx_any_axes(&union_b, sel_lo, &[_]c_int{0}, 1, false, s)); // [nb_lo]
    var lo_idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(lo_idx);
    try mlx.check(mlx.mlx_arange(&lo_idx, 0, @floatFromInt(geom.nb_lo), 1.0, .int32, s));
    const nb_lo_arr = mlx.mlx_array_new_int(geom.nb_lo);
    defer _ = mlx.mlx_array_free(nb_lo_arr);
    var lo_shift = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(lo_shift);
    try mlx.check(mlx.mlx_add(&lo_shift, lo_idx, nb_lo_arr, s));
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_where(&key, union_b, lo_idx, lo_shift, s));
    var order = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(order);
    try mlx.check(mlx.mlx_argsort_axis(&order, key, -1, s));
    var order_i32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(order_i32);
    try mlx.check(mlx.mlx_astype(&order_i32, order, .int32, s));
    var picked = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(picked);
    try mlx.check(mlx.mlx_slice(&picked, order_i32, &[_]c_int{0}, 1, &[_]c_int{geom.k_blocks}, 1, &[_]c_int{1}, 1, s));
    var ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids);
    try mlx.check(mlx.mlx_sort_axis(&ids, picked, -1, s)); // [K] ascending

    // Token ids: each picked block's `ratio` tokens, then the union tail.
    const ratio_arr = mlx.mlx_array_new_int(ratio);
    defer _ = mlx.mlx_array_free(ratio_arr);
    var ids2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids2);
    try mlx.check(mlx.mlx_reshape(&ids2, ids, &[_]c_int{ geom.k_blocks, 1 }, 2, s));
    var ids_scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids_scaled);
    try mlx.check(mlx.mlx_multiply(&ids_scaled, ids2, ratio_arr, s));
    var rvec = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rvec);
    try mlx.check(mlx.mlx_arange(&rvec, 0, @floatFromInt(ratio), 1.0, .int32, s));
    var r2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r2);
    try mlx.check(mlx.mlx_reshape(&r2, rvec, &[_]c_int{ 1, ratio }, 2, s));
    var tok2d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tok2d);
    try mlx.check(mlx.mlx_add(&tok2d, ids_scaled, r2, s)); // [K, ratio]
    var sel_tok = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sel_tok);
    try mlx.check(mlx.mlx_reshape(&sel_tok, tok2d, &[_]c_int{geom.k_blocks * ratio}, 1, s));
    var tok = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tok);
    if (geom.tail_len > 0) {
        var tail_ids = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(tail_ids);
        try mlx.check(mlx.mlx_arange(&tail_ids, @floatFromInt(geom.tail_start), @floatFromInt(kv), 1.0, .int32, s));
        const parts = [_]mlx.mlx_array{ sel_tok, tail_ids };
        const vec = mlx.mlx_vector_array_new_data(&parts, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&tok, vec, 0, s));
    } else {
        try mlx.check(mlx.mlx_array_set(&tok, sel_tok));
    }
    var tok_u32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tok_u32);
    try mlx.check(mlx.mlx_astype(&tok_u32, tok, .uint32, s));

    // [S, R] visibility: the dense formula evaluated at the gathered tokens.
    var blk_of = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(blk_of);
    try mlx.check(mlx.mlx_floor_divide(&blk_of, tok, ratio_arr, s)); // [R], <= nb
    var blk_of_u32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(blk_of_u32);
    try mlx.check(mlx.mlx_astype(&blk_of_u32, blk_of, .uint32, s));
    var member = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(member);
    try mlx.check(mlx.mlx_take_axis(&member, sel, blk_of_u32, 1, s)); // [S, R]
    var pos = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pos);
    try mlx.check(mlx.mlx_arange(&pos, @floatFromInt(offset), @floatFromInt(kv), 1.0, .int32, s));
    var pos2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pos2);
    try mlx.check(mlx.mlx_reshape(&pos2, pos, &[_]c_int{ seq_len, 1 }, 2, s));
    const one_i = mlx.mlx_array_new_int(1);
    defer _ = mlx.mlx_array_free(one_i);
    var p1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(p1);
    try mlx.check(mlx.mlx_add(&p1, pos2, one_i, s));
    var nb_p = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(nb_p);
    try mlx.check(mlx.mlx_floor_divide(&nb_p, p1, ratio_arr, s));
    var ts = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ts);
    try mlx.check(mlx.mlx_multiply(&ts, nb_p, ratio_arr, s)); // [S,1]
    var tok_row = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tok_row);
    try mlx.check(mlx.mlx_reshape(&tok_row, tok, &[_]c_int{ 1, geom.rows }, 2, s));
    var tailvis = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tailvis);
    try mlx.check(mlx.mlx_greater_equal(&tailvis, tok_row, ts, s));
    var causal = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(causal);
    try mlx.check(mlx.mlx_less_equal(&causal, tok_row, pos2, s));
    var any_vis = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(any_vis);
    try mlx.check(mlx.mlx_logical_or(&any_vis, member, tailvis, s));
    var vis = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vis);
    try mlx.check(mlx.mlx_logical_and(&vis, any_vis, causal, s));
    const zero_f = bf16Scalar(0.0, s);
    defer _ = mlx.mlx_array_free(zero_f);
    const neg_inf = bf16Scalar(-std.math.inf(f32), s);
    defer _ = mlx.mlx_array_free(neg_inf);
    var addv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(addv);
    try mlx.check(mlx.mlx_where(&addv, vis, zero_f, neg_inf, s));
    var add4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(add4);
    try mlx.check(mlx.mlx_reshape(&add4, addv, &[_]c_int{ 1, 1, seq_len, geom.rows }, 4, s));

    // Gather rows: quant triples when the view carries affine 4/6/8-bit (then
    // dequantize ONLY the subset — the whole point at `--kv-quant 8`), else
    // dense rows. Dequantize groups along D, so subset-first is exactly
    // full-first-restricted (pinned by the take-then-dequantize test).
    var gk = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gk);
    var gv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gv);
    if (kv_view.has_quant_triple and KVQuantConfig.supportedBits(kv_view.bits)) {
        const kq = kv_view.k_triple_q;
        const vq = kv_view.v_triple_q;
        if (kq.ctx == null or vq.ctx == null) return dq.no(.quant_layout);
        const kqs = mlx.getShape(kq);
        const vqs = mlx.getShape(vq);
        if (kqs.len != 4 or vqs.len != 4 or kqs[0] != 1 or vqs[0] != 1 or
            kqs[1] != h_kv or vqs[1] != h_kv or kqs[2] != kv or vqs[2] != kv) return dq.no(.quant_layout);
        const gs: u32 = kv_view.group_size;
        if (gs == 0) return dq.no(.quant_layout);
        // Contiguous copies: dequantize misreads the strided take views.
        const gkq = try takeContig(s, kq, tok_u32, 2);
        defer _ = mlx.mlx_array_free(gkq);
        const gks = try takeContig(s, kv_view.k_triple_scales, tok_u32, 2);
        defer _ = mlx.mlx_array_free(gks);
        const gkb = try takeContig(s, kv_view.k_triple_biases, tok_u32, 2);
        defer _ = mlx.mlx_array_free(gkb);
        const gvq = try takeContig(s, vq, tok_u32, 2);
        defer _ = mlx.mlx_array_free(gvq);
        const gvs = try takeContig(s, kv_view.v_triple_scales, tok_u32, 2);
        defer _ = mlx.mlx_array_free(gvs);
        const gvb = try takeContig(s, kv_view.v_triple_biases, tok_u32, 2);
        defer _ = mlx.mlx_array_free(gvb);
        const gkd = try kv_quant.dequantizeAffine(s, gkq, gks, gkb, gs, kv_view.bits);
        defer _ = mlx.mlx_array_free(gkd);
        const gvd = try kv_quant.dequantizeAffine(s, gvq, gvs, gvb, gs, kv_view.bits);
        defer _ = mlx.mlx_array_free(gvd);
        try mlx.check(mlx.mlx_array_set(&gk, gkd));
        try mlx.check(mlx.mlx_array_set(&gv, gvd));
    } else {
        try mlx.check(mlx.mlx_take_axis(&gk, kv_view.k, tok_u32, 2, s));
        try mlx.check(mlx.mlx_take_axis(&gv, kv_view.v, tok_u32, 2, s));
    }
    if (mlx.mlx_array_dtype(gk) != .bfloat16 or mlx.mlx_array_dtype(gv) != .bfloat16) return dq.no(.dtype);
    const gks2 = mlx.getShape(gk);
    const gvs2 = mlx.getShape(gv);
    if (gks2.len != 4 or gvs2.len != 4 or gks2[0] != 1 or gvs2[0] != 1 or
        gks2[1] != h_kv or gvs2[1] != h_kv or gks2[2] != geom.rows or gvs2[2] != geom.rows or
        gks2[3] != head_dim or gvs2[3] != head_dim) return dq.no(.gather_shape);

    // hd 256 + an ARRAY mask has no fused arm in MLX: the same row-group
    // split the dense mask arm uses, now over [R] instead of [kv].
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    if (try splitMaskedSdpa256(s, q_rope, gk, gv, attn_scale, add4)) |split_out| {
        _ = mlx.mlx_array_free(out);
        out = split_out;
    } else {
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&out, q_rope, gk, gv, attn_scale, "array", add4, .{ .ctx = null }, false, s));
    }
    if (qsa_engaged_bits.take(qsaEngagedBit(.verify_gather, seq_len))) {
        log.info("[qsa-verify-gather] engaged (S={d} kv={d} rows={d})\n", .{ seq_len, kv, geom.rows });
    }
    return out;
}

/// Dense [B,1,S,kv] bool mask equivalent to a block selection (fallback arm
/// + fixture trace): the selected blocks' tokens ∪ each row's own incomplete
/// tail, ∧ causal. Sentinel entries land in a spare column and are sliced off.
pub fn qsaMaskFromBlocks(s: mlx.mlx_stream, blocks: mlx.mlx_array, kv: c_int, ratio: c_int) !mlx.mlx_array {
    const bs = mlx.getShape(blocks);
    const batch = bs[0];
    const seq_len = bs[1];
    const nb: c_int = @divTrunc(kv, ratio);
    const nb_i = mlx.mlx_array_new_int(nb);
    defer _ = mlx.mlx_array_free(nb_i);
    var valid = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(valid);
    try mlx.check(mlx.mlx_less(&valid, blocks, nb_i, s));
    var safe = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(safe);
    try mlx.check(mlx.mlx_where(&safe, valid, blocks, nb_i, s));
    var falses = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(falses);
    try mlx.check(mlx.mlx_zeros(&falses, &[_]c_int{ batch, seq_len, nb + 1 }, 3, .bool_, s));
    const true_v = mlx.mlx_array_new_bool(true);
    defer _ = mlx.mlx_array_free(true_v);
    var put = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(put);
    try mlx.check(mlx.mlx_put_along_axis(&put, falses, safe, true_v, -1, s));
    var blk_sel = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(blk_sel);
    const strides3 = [_]c_int{ 1, 1, 1 };
    try mlx.check(mlx.mlx_slice(&blk_sel, put, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ batch, seq_len, nb }, 3, &strides3, 3, s));
    return qsaMaskFromBlockSel(s, blk_sel, kv - seq_len, seq_len, kv, nb, ratio, batch);
}

/// Expand a [B,S,nb] bool block selection to the [B,1,S,kv] token mask:
/// selected blocks' tokens (zero-padded to kv) ∪ each query's own incomplete
/// tail (tokens at/after ratio*floor((p+1)/ratio) — always visible), ∧ the
/// causal bound. `offset` is the cache position of row 0.
pub fn qsaMaskFromBlockSel(s: mlx.mlx_stream, blk_sel: mlx.mlx_array, offset: c_int, seq_len: c_int, kv: c_int, nb: c_int, ratio: c_int, batch: c_int) !mlx.mlx_array {
    var pos_col = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pos_col);
    try mlx.check(mlx.mlx_arange(&pos_col, @floatFromInt(offset), @floatFromInt(offset + seq_len), 1.0, .int32, s));
    var pos2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pos2);
    try mlx.check(mlx.mlx_reshape(&pos2, pos_col, &[_]c_int{ seq_len, 1 }, 2, s));
    var sel4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sel4);
    try mlx.check(mlx.mlx_expand_dims(&sel4, blk_sel, -1, s)); // [B,S,nb,1]
    var sel_b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sel_b);
    try mlx.check(mlx.mlx_broadcast_to(&sel_b, sel4, &[_]c_int{ batch, seq_len, nb, ratio }, 4, s));
    var sel_tok = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sel_tok);
    try mlx.check(mlx.mlx_reshape(&sel_tok, sel_b, &[_]c_int{ batch, seq_len, nb * ratio }, 3, s));
    var sel_full = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sel_full);
    const pad: c_int = kv - nb * ratio;
    if (pad > 0) {
        var zeros_pad = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(zeros_pad);
        try mlx.check(mlx.mlx_zeros(&zeros_pad, &[_]c_int{ batch, seq_len, pad }, 3, .bool_, s));
        const parts = [_]mlx.mlx_array{ sel_tok, zeros_pad };
        const vec = mlx.mlx_vector_array_new_data(&parts, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&sel_full, vec, 2, s));
    } else {
        try mlx.check(mlx.mlx_array_set(&sel_full, sel_tok));
    }
    var kv_idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kv_idx);
    try mlx.check(mlx.mlx_arange(&kv_idx, 0, @floatFromInt(kv), 1.0, .int32, s));
    var kv_idx2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kv_idx2);
    try mlx.check(mlx.mlx_reshape(&kv_idx2, kv_idx, &[_]c_int{ 1, kv }, 2, s));
    // tail_start[p] = ratio * floor((p+1)/ratio), via integer arithmetic.
    const one_i = mlx.mlx_array_new_int(1);
    defer _ = mlx.mlx_array_free(one_i);
    const ratio_i = mlx.mlx_array_new_int(ratio);
    defer _ = mlx.mlx_array_free(ratio_i);
    var p1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(p1);
    try mlx.check(mlx.mlx_add(&p1, pos2, one_i, s));
    var nb_p = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(nb_p);
    try mlx.check(mlx.mlx_floor_divide(&nb_p, p1, ratio_i, s));
    var tail_start = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tail_start);
    try mlx.check(mlx.mlx_multiply(&tail_start, nb_p, ratio_i, s)); // [S,1]
    var tail_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tail_mask);
    try mlx.check(mlx.mlx_greater_equal(&tail_mask, kv_idx2, tail_start, s)); // [S,kv]
    var tok_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tok_mask);
    try mlx.check(mlx.mlx_logical_or(&tok_mask, sel_full, tail_mask, s));
    var causal = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(causal);
    try mlx.check(mlx.mlx_less_equal(&causal, kv_idx2, pos2, s)); // [S, kv]
    var mask3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask3);
    try mlx.check(mlx.mlx_logical_and(&mask3, tok_mask, causal, s));
    var mask = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&mask, mask3, &[_]c_int{ batch, 1, seq_len, kv }, 4, s));
    return mask;
}

/// Bound on the per-chunk [n_idx, rows, nb] f32 indexer score sheet: the
/// prefill selection runs in row chunks so 4096 rows x 64k blocks x 4 heads
/// (4.3 GB at 256k) never materializes at once.
///
/// A lever because at long context it decides the prefill's row chunk: at kv 383k (nb ~95.8k)
/// 256 MB leaves room for ~171 rows, so a 4096-token chunk is re-split into 24 indexer passes.
/// `SUSHI_QSA_SCORE_SHEET_MB` moves it; the forward and the memory bill both read it
/// through `qsaScoreRowsPerChunk`, so a wider sheet is billed, never just spent.
pub const QSA_SCORE_SHEET_BUDGET_MB_DEFAULT: u64 = 256;

/// Upper bound on the lever: a fat-fingered value must become a named 400, not a Metal OOM.
pub const QSA_SCORE_SHEET_BUDGET_MB_MAX: u64 = 4096;

var qsa_score_sheet_mb_cached: ?u64 = null;
/// Test seam: bypass the env read entirely.
pub var qsa_score_sheet_mb_override: ?u64 = null;
var qsa_score_sheet_logged: bool = false;

/// Resolve the raw env value to a megabyte count: absent/0 = default, above the cap clamps.
pub fn qsaScoreSheetMbFrom(raw: ?[]const u8) u64 {
    const text = raw orelse return QSA_SCORE_SHEET_BUDGET_MB_DEFAULT;
    if (text.len == 0) return QSA_SCORE_SHEET_BUDGET_MB_DEFAULT;
    const parsed = std.fmt.parseInt(u64, text, 10) catch return QSA_SCORE_SHEET_BUDGET_MB_DEFAULT;
    if (parsed == 0) return QSA_SCORE_SHEET_BUDGET_MB_DEFAULT;
    return @min(parsed, QSA_SCORE_SHEET_BUDGET_MB_MAX);
}

/// The live sheet budget in bytes. Read once, logged once when not the default.
pub fn qsaScoreSheetBudget() u64 {
    if (qsa_score_sheet_mb_override) |mb| return mb << 20;
    if (qsa_score_sheet_mb_cached) |mb| return mb << 20;
    const raw = std.c.getenv("SUSHI_QSA_SCORE_SHEET_MB");
    const mb = qsaScoreSheetMbFrom(if (raw) |r| std.mem.sliceTo(r, 0) else null);
    qsa_score_sheet_mb_cached = mb;
    if (mb != QSA_SCORE_SHEET_BUDGET_MB_DEFAULT and !qsa_score_sheet_logged) {
        qsa_score_sheet_logged = true;
        log.info("[qsa] score sheet budget: {d} MB (SUSHI_QSA_SCORE_SHEET_MB; default {d})\n", .{ mb, QSA_SCORE_SHEET_BUDGET_MB_DEFAULT });
    }
    return mb << 20;
}

pub fn qsaScoreRowsPerChunkFor(per_row: u64, rows: u64) u64 {
    if (per_row == 0) return rows;
    return @max(16, @min(rows, qsaScoreSheetBudget() / per_row));
}

pub fn qsaScoreRowsPerChunkFused(n_idx: u64, nb: u64, rows: u64, fused: bool) u64 {
    const per_row: u64 = if (fused) nb * 4 else n_idx * nb * 4;
    return qsaScoreRowsPerChunkFor(per_row, rows);
}

/// Prefill transient of the block-gathered QSA path for ONE live layer.
/// Composed: matmul out + relu per head + scores + argpartition temps + selection.
/// Fused: the [rows,nb] f32 sheet (×4 when the select kernel is off) + selection.
pub fn qsaPrefillTransientBytes(n_idx: u64, fwd: u64, kv: u64, ratio: u64, idx_hd: u64) u64 {
    if (ratio == 0) return 0;
    const nb = kv / ratio;
    const fused = qsaScoreFusedActiveFor(1, @intCast(n_idx), @intCast(idx_hd));
    const rows = qsaScoreRowsPerChunkFused(n_idx, nb, fwd, fused);
    if (fused) {
        const sheet = rows * nb * 4;
        const select_mult: u64 = if (qsaSelectKernelEnabled()) 1 else 4;
        return sheet * select_mult + fwd * @min(nb, 512) * 4;
    }
    return rows * nb * 4 * (n_idx * 2 + 4) + fwd * @min(nb, 512) * 4;
}

/// Two bf16 ulps at the reference's own scale.
pub fn attn256UlpBar(ref: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    const m = try attn256MaxAbs(ref, s);
    return 2.0 * m * 0.00390625 + 1.0e-6;
}

/// Cosine of two attention outputs. A max-abs bar alone cannot see a uniform
/// scale error (a dropped or double-counted key shifts every row a little),
/// so the gather parity tests assert BOTH.
pub fn attn256Cosine(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    var a_c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a_c);
    var b_c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b_c);
    try mlx.check(mlx.mlx_contiguous(&a_c, a, false, s));
    try mlx.check(mlx.mlx_contiguous(&b_c, b, false, s));
    var a32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a32);
    var b32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b32);
    try mlx.check(mlx.mlx_astype(&a32, a_c, .float32, s));
    try mlx.check(mlx.mlx_astype(&b32, b_c, .float32, s));
    try mlx.check(mlx.mlx_array_eval(a32));
    try mlx.check(mlx.mlx_array_eval(b32));
    const ad = mlx.mlx_array_data_float32(a32) orelse return error.InvalidDtype;
    const bd = mlx.mlx_array_data_float32(b32) orelse return error.InvalidDtype;
    const n = mlx.mlx_array_size(a32);
    if (n != mlx.mlx_array_size(b32)) return error.ShapeMismatch;
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (0..n) |i| {
        dot += @as(f64, ad[i]) * @as(f64, bd[i]);
        na += @as(f64, ad[i]) * @as(f64, ad[i]);
        nb += @as(f64, bd[i]) * @as(f64, bd[i]);
    }
    if (na == 0 or nb == 0) return error.ZeroNorm;
    return @floatCast(dot / (@sqrt(na) * @sqrt(nb)));
}

fn qsaGatherRowL(p: c_int, kb: c_int, ratio: c_int) c_int {
    const complete: c_int = @divTrunc(p + 1, ratio);
    const sel_len: c_int = @min(complete, kb) * ratio;
    const tail_start: c_int = complete * ratio;
    return sel_len + p + 1 - tail_start;
}

fn qsaGatherRowPos(blk_row: []const i32, vi: c_int, p: c_int, kb: c_int, ratio: c_int) u32 {
    const complete: c_int = @divTrunc(p + 1, ratio);
    const sel_len: c_int = @min(complete, kb) * ratio;
    const tail_start: c_int = complete * ratio;
    if (vi < sel_len) {
        const b: c_int = @divTrunc(vi, ratio);
        return @intCast(blk_row[@intCast(b)] * ratio + (vi - b * ratio));
    }
    return @intCast(tail_start + (vi - sel_len));
}

pub fn qsaGatherF32Reference(
    q: mlx.mlx_array,
    k: mlx.mlx_array,
    v: mlx.mlx_array,
    scale: f32,
    blocks: mlx.mlx_array,
    ratio: c_int,
    stream: mlx.mlx_stream,
) !mlx.mlx_array {
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    const bs = mlx.getShape(blocks);
    const B = qs[0];
    const Hq = qs[1];
    const qL = qs[2];
    const hd = qs[3];
    const Hk = ks[1];
    const kL = ks[2];
    const kb = bs[2];
    const gqa: c_int = @divTrunc(Hq, Hk);
    if (B != 1) return error.ShapeMismatch;
    try mlx.check(mlx.mlx_array_eval(blocks));
    const bp = mlx.mlx_array_data_int32(blocks) orelse return error.InvalidDtype;
    const nq: usize = @intCast(qL);
    const nkb: usize = @intCast(kb);
    var lmax: c_int = 1;
    for (0..nq) |si| {
        const p: c_int = kL - qL + @as(c_int, @intCast(si));
        lmax = @max(lmax, qsaGatherRowL(p, kb, ratio));
    }
    const nl: usize = @intCast(lmax);
    const ta = std.heap.page_allocator;
    const idx = try ta.alloc(u32, nq * nl);
    defer ta.free(idx);
    const vis = try ta.alloc(bool, nq * nl);
    defer ta.free(vis);
    @memset(idx, 0);
    @memset(vis, false);
    for (0..nq) |si| {
        const p: c_int = kL - qL + @as(c_int, @intCast(si));
        const L = qsaGatherRowL(p, kb, ratio);
        const row = bp[si * nkb .. si * nkb + nkb];
        var vi: c_int = 0;
        while (vi < L) : (vi += 1) {
            idx[si * nl + @as(usize, @intCast(vi))] = qsaGatherRowPos(row, vi, p, kb, ratio);
            vis[si * nl + @as(usize, @intCast(vi))] = true;
        }
    }
    const ishape = [_]c_int{ qL, lmax };
    const vshape = [_]c_int{ 1, 1, qL, lmax };
    const idx_arr = mlx.mlx_array_new_data(idx.ptr, &ishape, 2, .uint32);
    defer _ = mlx.mlx_array_free(idx_arr);
    const vis_arr = mlx.mlx_array_new_data(vis.ptr, &vshape, 4, .bool_);
    defer _ = mlx.mlx_array_free(vis_arr);

    var q32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q32);
    var k32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k32);
    var v32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v32);
    try mlx.check(mlx.mlx_astype(&q32, q, .float32, stream));
    try mlx.check(mlx.mlx_astype(&k32, k, .float32, stream));
    try mlx.check(mlx.mlx_astype(&v32, v, .float32, stream));

    var kg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kg);
    var vg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vg);
    try mlx.check(mlx.mlx_take_axis(&kg, k32, idx_arr, 2, stream));
    try mlx.check(mlx.mlx_take_axis(&vg, v32, idx_arr, 2, stream));

    var kge = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kge);
    var vge = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vge);
    try mlx.check(mlx.mlx_expand_dims(&kge, kg, 2, stream));
    try mlx.check(mlx.mlx_expand_dims(&vge, vg, 2, stream));
    const bcast_shape = [_]c_int{ B, Hk, gqa, qL, lmax, hd };
    var kgb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kgb);
    var vgb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vgb);
    try mlx.check(mlx.mlx_broadcast_to(&kgb, kge, &bcast_shape, 6, stream));
    try mlx.check(mlx.mlx_broadcast_to(&vgb, vge, &bcast_shape, 6, stream));
    const gshape = [_]c_int{ B, Hq, qL, lmax, hd };
    var kgr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kgr);
    var vgr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vgr);
    try mlx.check(mlx.mlx_reshape(&kgr, kgb, &gshape, 5, stream));
    try mlx.check(mlx.mlx_reshape(&vgr, vgb, &gshape, 5, stream));

    var qe = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(qe);
    try mlx.check(mlx.mlx_expand_dims(&qe, q32, 3, stream));
    var dots = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dots);
    try mlx.check(mlx.mlx_multiply(&dots, qe, kgr, stream));
    var scores_u = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scores_u);
    try mlx.check(mlx.mlx_sum_axis(&scores_u, dots, 4, false, stream));
    const scl = mlx.mlx_array_new_float(scale);
    defer _ = mlx.mlx_array_free(scl);
    var scores = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scores);
    try mlx.check(mlx.mlx_multiply(&scores, scores_u, scl, stream));

    const ninf = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer _ = mlx.mlx_array_free(ninf);
    var masked = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(masked);
    try mlx.check(mlx.mlx_where(&masked, vis_arr, scores, ninf, stream));
    var probs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(probs);
    try mlx.check(mlx.mlx_softmax_axis(&probs, masked, 3, true, stream));
    var pe = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pe);
    try mlx.check(mlx.mlx_expand_dims(&pe, probs, 4, stream));
    var wv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wv);
    try mlx.check(mlx.mlx_multiply(&wv, pe, vgr, stream));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_sum_axis(&out, wv, 3, false, stream));
    return out;
}

pub fn qsaBoostLastTileK(
    q: mlx.mlx_array,
    k: mlx.mlx_array,
    blocks: mlx.mlx_array,
    ratio: c_int,
    stream: mlx.mlx_stream,
) !mlx.mlx_array {
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    const bs = mlx.getShape(blocks);
    const Hq = qs[1];
    const qL = qs[2];
    const hd = qs[3];
    const Hk = ks[1];
    const kL = ks[2];
    const kb = bs[2];
    const gqa: c_int = @divTrunc(Hq, Hk);
    var qc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(qc);
    var kc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kc);
    try mlx.check(mlx.mlx_contiguous(&qc, q, false, stream));
    try mlx.check(mlx.mlx_contiguous(&kc, k, false, stream));
    var q32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q32);
    var k32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k32);
    try mlx.check(mlx.mlx_astype(&q32, qc, .float32, stream));
    try mlx.check(mlx.mlx_astype(&k32, kc, .float32, stream));
    try mlx.check(mlx.mlx_array_eval(q32));
    try mlx.check(mlx.mlx_array_eval(k32));
    try mlx.check(mlx.mlx_array_eval(blocks));
    const qp = mlx.mlx_array_data_float32(q32) orelse return error.InvalidDtype;
    const kp = mlx.mlx_array_data_float32(k32) orelse return error.InvalidDtype;
    const bp = mlx.mlx_array_data_int32(blocks) orelse return error.InvalidDtype;
    const nq: usize = @intCast(qL);
    const nkv: usize = @intCast(kL);
    const nd: usize = @intCast(hd);
    const nkb: usize = @intCast(kb);
    const nhk: usize = @intCast(Hk);
    const buf = try std.testing.allocator.alloc(f32, nhk * nkv * nd);
    defer std.testing.allocator.free(buf);
    @memcpy(buf, kp[0 .. nhk * nkv * nd]);
    const bk_tile: c_int = 32;
    for (0..nq) |si| {
        const p: c_int = kL - qL + @as(c_int, @intCast(si));
        const L = qsaGatherRowL(p, kb, ratio);
        const t0: c_int = L - @min(bk_tile, L);
        const row = bp[si * nkb .. si * nkb + nkb];
        var hk: usize = 0;
        while (hk < nhk) : (hk += 1) {
            const h: usize = hk * @as(usize, @intCast(gqa));
            const qbase = (h * nq + si) * nd;
            var vi: c_int = t0;
            while (vi < L) : (vi += 1) {
                const pos: usize = @intCast(qsaGatherRowPos(row, vi, p, kb, ratio));
                const kbase = (hk * nkv + pos) * nd;
                for (0..nd) |d| buf[kbase + d] += 40.0 * qp[qbase + d];
            }
        }
    }
    const kshape = [_]c_int{ 1, Hk, kL, hd };
    const k_host = mlx.mlx_array_new_data(buf.ptr, &kshape, 4, .float32);
    defer _ = mlx.mlx_array_free(k_host);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, k_host, .bfloat16, stream));
    return out;
}

/// Host `[S, kb]` block selection in the shape `qsaSelectBlocks` hands the
/// gatherer: ascending VISIBLE block ids per row, `INT_MAX` past the count.
pub fn qsaVerifyBlocksHost(alloc: std.mem.Allocator, rnd: std.Random, seq_len: c_int, kv: c_int, kb: c_int, ratio: c_int) ![]i32 {
    const su: usize = @intCast(seq_len);
    const kbu: usize = @intCast(kb);
    const nb: c_int = @divTrunc(kv, ratio);
    const out = try alloc.alloc(i32, su * kbu);
    errdefer alloc.free(out);
    const picked = try alloc.alloc(bool, @intCast(nb + 1));
    defer alloc.free(picked);
    for (0..su) |i| {
        const p: c_int = kv - seq_len + @as(c_int, @intCast(i));
        const complete: c_int = @min(@divTrunc(p + 1, ratio), nb);
        const count: c_int = @min(complete, kb);
        @memset(picked, false);
        var n: c_int = 0;
        while (n < count) {
            const b = rnd.uintLessThan(usize, @intCast(complete));
            if (picked[b]) continue;
            picked[b] = true;
            n += 1;
        }
        var w: usize = 0;
        for (0..@intCast(complete)) |b| if (picked[b]) {
            out[i * kbu + w] = @intCast(b);
            w += 1;
        };
        while (w < kbu) : (w += 1) out[i * kbu + w] = std.math.maxInt(i32);
    }
    return out;
}

/// Brute-force `[S, kv]` visibility — the contract `qsaMaskFromBlockSel`
/// implements: the row's selected blocks' tokens ∪ its own incomplete tail
/// (tokens at/after `ratio * floor((p+1)/ratio)`), ∧ causal.
fn qsaVerifyDenseHost(alloc: std.mem.Allocator, blocks: []const i32, seq_len: c_int, kb: c_int, kv: c_int, ratio: c_int) ![]bool {
    const su: usize = @intCast(seq_len);
    const kvu: usize = @intCast(kv);
    const kbu: usize = @intCast(kb);
    const nb: c_int = @divTrunc(kv, ratio);
    const out = try alloc.alloc(bool, su * kvu);
    errdefer alloc.free(out);
    @memset(out, false);
    for (0..su) |i| {
        const p: c_int = kv - seq_len + @as(c_int, @intCast(i));
        const ts: c_int = @divTrunc(p + 1, ratio) * ratio;
        for (0..kvu) |t| {
            const ti: c_int = @intCast(t);
            if (ti > p) continue;
            if (ti >= ts) {
                out[i * kvu + t] = true;
                continue;
            }
            if (ti >= nb * ratio) continue;
            const b: i32 = @intCast(@divTrunc(ti, ratio));
            for (0..kbu) |c| {
                if (blocks[i * kbu + c] == b) {
                    out[i * kvu + t] = true;
                    break;
                }
            }
        }
    }
    return out;
}

/// Build a KVCache whose view is a strided slice of a larger capacity buffer (the shape the serving path hands the kernel).
pub fn qsaAttnCacheFixture(
    alloc: std.mem.Allocator,
    s: mlx.mlx_stream,
    k_dense: mlx.mlx_array,
    v_dense: mlx.mlx_array,
    kv: c_int,
    split: c_int,
    cfg: kv_quant.KVQuantConfig,
    cache: *KVCache,
) !DenseKVView {
    _ = alloc;
    const st = [_]c_int{ 1, 1, 1, 1 };
    var k0 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k0);
    var v0 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v0);
    const ks = mlx.getShape(k_dense);
    try mlx.check(mlx.mlx_slice(&k0, k_dense, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ ks[0], ks[1], split, ks[3] }, 4, &st, 4, s));
    try mlx.check(mlx.mlx_slice(&v0, v_dense, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ ks[0], ks[1], split, ks[3] }, 4, &st, 4, s));
    var first = try cache.update(0, k0, v0, s, 0);
    first.deinit();
    var k1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k1);
    var v1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v1);
    try mlx.check(mlx.mlx_slice(&k1, k_dense, &[_]c_int{ 0, 0, split, 0 }, 4, &[_]c_int{ ks[0], ks[1], kv, ks[3] }, 4, &st, 4, s));
    try mlx.check(mlx.mlx_slice(&v1, v_dense, &[_]c_int{ 0, 0, split, 0 }, 4, &[_]c_int{ ks[0], ks[1], kv, ks[3] }, 4, &st, 4, s));
    _ = cfg;
    return try cache.update(0, k1, v1, s, 0);
}

fn qsaIdentityServingArm(s: mlx.mlx_stream, q: mlx.mlx_array, view: *const DenseKVView, blocks: mlx.mlx_array) !mlx.mlx_array {
    if (try qsaSparseAttn(s, q, view, blocks, 4, 1.0 / 16.0)) |out| return out;
    if (mlx.getShape(q)[2] == 1) return (try qsaDecodeGatherAttn(s, q, view, blocks, 4, 1.0 / 16.0)) orelse error.QsaTickDeclined;
    return (try qsaVerifyGatherAttn(s, q, view, blocks, 4, 1.0 / 16.0)) orelse error.QsaVerifyDeclined;
}

test "qsa verify gather floor: dense KV 32768, quantized 16384, override wins" {
    const prev_o = qsa_verify_gather_min_kv_override;
    defer qsa_verify_gather_min_kv_override = prev_o;
    qsa_verify_gather_min_kv_override = null;
    try std.testing.expectEqual(@as(c_int, 32768), qsaVerifyGatherMinKvFor(false));
    try std.testing.expectEqual(@as(c_int, 16384), qsaVerifyGatherMinKvFor(true));
    qsa_verify_gather_min_kv_override = 4096;
    try std.testing.expectEqual(@as(c_int, 4096), qsaVerifyGatherMinKvFor(false));
    try std.testing.expectEqual(@as(c_int, 4096), qsaVerifyGatherMinKvFor(true));
}

test "qsa decode gather: take-then-dequantize equals dequantize-then-take" {
    // The subset-dequant primitive the decode gatherer is built on: per-token
    // groups along D make the T axis independent, so gathering rows before or
    // after dequantization must agree bitwise. Production-like sizes (T=256,
    // take 64 scattered rows).
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0xdec0de);
    const rnd = prng.random();
    const dense = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, 256, 64 }, s);
    defer _ = mlx.mlx_array_free(dense);
    var qq = try kv_quant.quantizeAffine(s, dense, 32, 4);
    defer qq.deinit();
    const n_take = 64;
    const idx_host = blk: {
        var tmp: [n_take]i32 = undefined;
        var i: usize = 0;
        while (i < n_take) : (i += 1) tmp[i] = @intCast((i * 37 + 11) % 256);
        break :blk tmp;
    };
    const idx_sh = [_]c_int{n_take};
    const idx_i32 = mlx.mlx_array_new_data(&idx_host, &idx_sh, 1, .int32);
    defer _ = mlx.mlx_array_free(idx_i32);
    // take_axis wants unsigned indices; every id is >= 0 by construction.
    var idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(idx);
    try mlx.check(mlx.mlx_astype(&idx, idx_i32, .uint32, s));
    // take-then-dequantize, through the same contiguous helper the decode
    // gatherer uses (dequantize misreads strided take views).
    const tq = try takeContig(s, qq.q, idx, 2);
    defer _ = mlx.mlx_array_free(tq);
    const ts = try takeContig(s, qq.scales, idx, 2);
    defer _ = mlx.mlx_array_free(ts);
    const tb = try takeContig(s, qq.biases, idx, 2);
    defer _ = mlx.mlx_array_free(tb);
    // Exact check of the taken packed words against the source rows they
    // name (u32 — float32 cannot distinguish large packed words; raw pointer
    // walks go through mlx_contiguous first since a take result need not lay
    // out densely). This pins the +0-materializer footgun below: an add of a
    // strided take view reads back wrong, so takeContig uses mlx_contiguous.
    {
        var takenc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(takenc);
        try mlx.check(mlx.mlx_contiguous(&takenc, tq, false, s));
        const evx = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(evx);
        _ = mlx.mlx_vector_array_append_value(evx, takenc);
        _ = mlx.mlx_vector_array_append_value(evx, qq.q);
        try mlx.check(mlx.mlx_eval(evx));
        const td = mlx.mlx_array_data_uint32(takenc) orelse return error.TestUnexpectedNullData;
        const qd = mlx.mlx_array_data_uint32(qq.q) orelse return error.TestUnexpectedNullData;
        var bad: usize = 0;
        for (0..2) |h| {
            for (0..n_take) |i| {
                const src_row: usize = @intCast(idx_host[i]);
                for (0..8) |w| {
                    if (td[(h * n_take + i) * 8 + w] != qd[(h * 256 + src_row) * 8 + w]) bad += 1;
                }
            }
        }
        try std.testing.expectEqual(@as(usize, 0), bad);
    }
    const sub = try kv_quant.dequantizeAffine(s, tq, ts, tb, 32, 4);
    defer _ = mlx.mlx_array_free(sub);
    const full = try kv_quant.dequantizeAffine(s, qq.q, qq.scales, qq.biases, 32, 4);
    defer _ = mlx.mlx_array_free(full);
    var want = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(want);
    try mlx.check(mlx.mlx_take_axis(&want, full, idx, 2, s));
    // Compare through explicit contiguous copies on BOTH sides: the shared
    // comparator's +0 materializer may itself misread the strided take view.
    var subc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(subc);
    try mlx.check(mlx.mlx_contiguous(&subc, sub, false, s));
    var wantc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wantc);
    try mlx.check(mlx.mlx_contiguous(&wantc, want, false, s));
    const diff = try maxAbsDiffF32(subc, wantc, s);
    errdefer std.debug.print("[qsa-take-dequant] max abs diff {d}\n", .{diff});
    try std.testing.expectEqual(@as(f32, 0), diff);
}

test "qsa decode gather: subset SDPA matches masked full SDPA at decode width" {
    // End-to-end parity for `qsaDecodeGatherAttn` on synthetic geometry:
    // same selection + tail, ascending order, causal-visible by construction.
    // Remaining deltas are kernel tiling order only.
    const s = mlx.gpuStream();
    qwen4_qsa_gather.qsa_decode_gather_override = true;
    defer qwen4_qsa_gather.qsa_decode_gather_override = null;
    var prng = std.Random.DefaultPrng.init(0x60d31);
    const rnd = prng.random();
    const kL: c_int = 50;
    const kb: c_int = 3;
    const ratio: c_int = 4;
    const q_shape = [_]c_int{ 1, 4, 1, 64 };
    const kv_shape = [_]c_int{ 1, 2, kL, 64 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k_dense = try testRandWeightBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k_dense);
    const v_dense = try testRandWeightBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(v_dense);
    var fx = try QsaBlockFixture.build(rnd, 1, kL, kb, ratio);
    defer fx.deinit();
    const scale: f32 = 1.0 / 8.0;

    // Quant path: the view carries affine 4-bit triples AND dense arrays.
    var kq = try kv_quant.quantizeAffine(s, k_dense, 32, 4);
    defer kq.deinit();
    var vq = try kv_quant.quantizeAffine(s, v_dense, 32, 4);
    defer vq.deinit();
    var view = DenseKVView{
        .k = k_dense,
        .v = v_dense,
        .owned = false,
        .k_triple_q = kq.q,
        .k_triple_scales = kq.scales,
        .k_triple_biases = kq.biases,
        .v_triple_q = vq.q,
        .v_triple_scales = vq.scales,
        .v_triple_biases = vq.biases,
        .has_quant_triple = true,
        .bits = 4,
        .group_size = 32,
    };
    const got = (try qsaDecodeGatherAttn(s, q, &view, fx.blocks, ratio, scale)) orelse return error.GatherDeclined;
    defer _ = mlx.mlx_array_free(got);
    // Reference: masked SDPA over the dense dequantized full KV (what the
    // mask arm computes, up to its own kernel tiling).
    const dk_full = try kv_quant.dequantizeAffine(s, kq.q, kq.scales, kq.biases, 32, 4);
    defer _ = mlx.mlx_array_free(dk_full);
    const dv_full = try kv_quant.dequantizeAffine(s, vq.q, vq.scales, vq.biases, 32, 4);
    defer _ = mlx.mlx_array_free(dv_full);
    const ref = try attn256Reference(q, dk_full, dv_full, scale, "array", fx.mask, s);
    defer _ = mlx.mlx_array_free(ref);
    const diff = try attn256MaxDiff(got, ref, s);
    errdefer std.debug.print("[qsa-decode-gather] quant path max diff {d:.6}\n", .{diff});
    try std.testing.expect(diff < 0.01);

    // Dense path (no triple): same answer off the dense rows.
    var dview = DenseKVView{ .k = k_dense, .v = v_dense, .owned = false };
    const got_dense = (try qsaDecodeGatherAttn(s, q, &dview, fx.blocks, ratio, scale)) orelse return error.GatherDeclined;
    defer _ = mlx.mlx_array_free(got_dense);
    const ref_dense = try attn256Reference(q, k_dense, v_dense, scale, "array", fx.mask, s);
    defer _ = mlx.mlx_array_free(ref_dense);
    const diff_dense = try attn256MaxDiff(got_dense, ref_dense, s);
    errdefer std.debug.print("[qsa-decode-gather] dense path max diff {d:.6}\n", .{diff_dense});
    try std.testing.expect(diff_dense < 0.01);

    // Kill switch + wrong-dtype blocks decline to the mask arm.
    qwen4_qsa_gather.qsa_decode_gather_override = false;
    try std.testing.expect((try qsaDecodeGatherAttn(s, q, &view, fx.blocks, ratio, scale)) == null);
}

test "qsa batched gather S=2 and S=4: N=1 is byte-identical to solo verify, N=2 matches per-slot solos" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const ta = std.testing.allocator;
    qsa_verify_gather_override = true;
    defer qsa_verify_gather_override = null;
    qsa_verify_gather_min_kv_override = 16;
    defer qsa_verify_gather_min_kv_override = null;
    qwen4_qsa_gather.qsa_batched_gather_override = true;
    defer qwen4_qsa_gather.qsa_batched_gather_override = null;
    var prng = std.Random.DefaultPrng.init(0xc0de);
    const rnd = prng.random();
    const ratio: c_int = 4;
    const kb: c_int = 3;
    const hq: c_int = 4;
    const hkv: c_int = 2;
    const hd: c_int = 256;
    const scale: f32 = 1.0 / 16.0;
    const kL0: c_int = 96;
    const kL1: c_int = 80;
    for ([_]c_int{ 2, 4 }) |seq| {
        const q0 = try attn256RandBf16(rnd, &[_]c_int{ 1, hq, seq, hd }, s);
        defer _ = mlx.mlx_array_free(q0);
        const q1 = try attn256RandBf16(rnd, &[_]c_int{ 1, hq, seq, hd }, s);
        defer _ = mlx.mlx_array_free(q1);
        const k0 = try attn256RandBf16(rnd, &[_]c_int{ 1, hkv, kL0, hd }, s);
        defer _ = mlx.mlx_array_free(k0);
        const v0 = try attn256RandBf16(rnd, &[_]c_int{ 1, hkv, kL0, hd }, s);
        defer _ = mlx.mlx_array_free(v0);
        const k1 = try attn256RandBf16(rnd, &[_]c_int{ 1, hkv, kL1, hd }, s);
        defer _ = mlx.mlx_array_free(k1);
        const v1 = try attn256RandBf16(rnd, &[_]c_int{ 1, hkv, kL1, hd }, s);
        defer _ = mlx.mlx_array_free(v1);
        var fx0 = try QsaBlockFixture.build(rnd, seq, kL0, kb, ratio);
        defer fx0.deinit();
        var fx1 = try QsaBlockFixture.build(rnd, seq, kL1, kb, ratio);
        defer fx1.deinit();
        const view0 = DenseKVView{ .k = k0, .v = v0, .owned = false };
        const view1 = DenseKVView{ .k = k1, .v = v1, .owned = false };
        const solo0 = (try qsaSparseAttn(s, q0, &view0, fx0.blocks, ratio, scale)) orelse return error.GatherDeclined;
        defer _ = mlx.mlx_array_free(solo0);
        const bat1 = try qsaBatchedGatherAttn(ta, s, q0, &.{view0}, &.{fx0.blocks}, &.{}, ratio, scale, &.{}, 0);
        defer _ = mlx.mlx_array_free(bat1);
        try std.testing.expectEqual(@as(f32, 0), try attn256MaxDiff(bat1, solo0, s));

        var q_n2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_n2);
        const qparts = [_]mlx.mlx_array{ q0, q1 };
        const qvec = mlx.mlx_vector_array_new_data(&qparts, 2);
        defer _ = mlx.mlx_vector_array_free(qvec);
        try mlx.check(mlx.mlx_concatenate_axis(&q_n2, qvec, 0, s));
        const views = [_]DenseKVView{ view0, view1 };
        const blks = [_]mlx.mlx_array{ fx0.blocks, fx1.blocks };
        const bat2 = try qsaBatchedGatherAttn(ta, s, q_n2, &views, &blks, &.{}, ratio, scale, &.{}, 0);
        defer _ = mlx.mlx_array_free(bat2);
        const solo1 = (try qsaSparseAttn(s, q1, &view1, fx1.blocks, ratio, scale)) orelse return error.GatherDeclined;
        defer _ = mlx.mlx_array_free(solo1);
        var want = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(want);
        const wparts = [_]mlx.mlx_array{ solo0, solo1 };
        const wvec = mlx.mlx_vector_array_new_data(&wparts, 2);
        defer _ = mlx.mlx_vector_array_free(wvec);
        try mlx.check(mlx.mlx_concatenate_axis(&want, wvec, 0, s));
        try std.testing.expectEqual(@as(f32, 0), try attn256MaxDiff(bat2, want, s));
    }
}

test "qsa verify gather plan: gathered rows carry exactly the dense [S, kv] mask, no duplicates" {
    // (a) The pure planner is the executable spec of the MLX gatherer: the
    // fixed-size union of the S rows' selections (dedup'd; padded with
    // non-union blocks that every row masks out) plus the union tail. Every
    // (row, token) the dense mask shows must be reachable through EXACTLY
    // one gathered slot, and nothing else may be — a token gathered twice and
    // visible twice would be double-counted under the softmax.
    const ta = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xbe1f7);
    const rnd = prng.random();
    const cases = [_]struct { s: c_int, kv: c_int, kb: c_int, ratio: c_int }{
        .{ .s = 2, .kv = 4096, .kb = 32, .ratio = 4 },
        .{ .s = 5, .kv = 4096, .kb = 64, .ratio = 4 },
        .{ .s = 7, .kv = 20000, .kb = 128, .ratio = 4 },
        .{ .s = 15, .kv = 20000, .kb = 128, .ratio = 4 },
        .{ .s = 3, .kv = 61, .kb = 3, .ratio = 4 }, // ragged tail (kv % ratio != 0)
        .{ .s = 4, .kv = 64, .kb = 2, .ratio = 4 }, // kv divisible by ratio
        .{ .s = 8, .kv = 257, .kb = 5, .ratio = 8 }, // ratio != 4
    };
    for (cases) |c| {
        const blocks = try qsaVerifyBlocksHost(ta, rnd, c.s, c.kv, c.kb, c.ratio);
        defer ta.free(blocks);
        const dense = try qsaVerifyDenseHost(ta, blocks, c.s, c.kb, c.kv, c.ratio);
        defer ta.free(dense);
        var plan = try qsaVerifyPlanHost(ta, blocks, c.s, c.kb, c.kv, c.ratio);
        defer plan.deinit(ta);

        for (1..plan.rows.len) |r| try std.testing.expect(plan.rows[r] > plan.rows[r - 1]);
        for (plan.rows) |t| try std.testing.expect(t >= 0 and t < c.kv);

        const su: usize = @intCast(c.s);
        const kvu: usize = @intCast(c.kv);
        const slot_of = try ta.alloc(isize, kvu);
        defer ta.free(slot_of);
        @memset(slot_of, -1);
        for (plan.rows, 0..) |t, r| slot_of[@intCast(t)] = @intCast(r);
        for (0..su) |i| {
            for (0..kvu) |t| {
                const want = dense[i * kvu + t];
                const sl = slot_of[t];
                const got = sl >= 0 and plan.mask[i * plan.rows.len + @as(usize, @intCast(sl))];
                if (want != got) {
                    std.debug.print("[qsa-verify-plan] S={d} kv={d} row {d} token {d}: dense={} gathered={}\n", .{ c.s, c.kv, i, t, want, got });
                    return error.VerifyPlanMismatch;
                }
            }
        }
    }
}

test "qsa batched gather plan: per-slot rows match the solo planner at S=1 and S=2..8" {
    const ta = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xa11c3);
    const rnd = prng.random();
    const ratio: c_int = 4;
    const kb: c_int = 32;
    const kvs = [_]c_int{ 4096, 20000 };
    for (kvs) |kv_a| {
        const kv_b: c_int = @divTrunc(kv_a, 2) + 17;
        {
            const s: c_int = 1;
            const ba = try qsaVerifyBlocksHost(ta, rnd, s, kv_a, kb, ratio);
            defer ta.free(ba);
            const bb = try qsaVerifyBlocksHost(ta, rnd, s, kv_b, kb, ratio);
            defer ta.free(bb);
            const slot_blocks = [_][]const i32{ ba, bb };
            const kv_lens = [_]c_int{ kv_a, kv_b };
            var plans = try qsaBatchedPlanHost(ta, &slot_blocks, s, kb, &kv_lens, ratio);
            defer {
                var pi: usize = 0;
                while (pi < plans.len) : (pi += 1) plans[pi].deinit(ta);
                ta.free(plans);
            }
            try std.testing.expectEqual(@as(usize, 2), plans.len);
            const ea = try qsaDecodeGatherIdsHost(ta, ba, kb, kv_a, ratio);
            defer ta.free(ea);
            const eb = try qsaDecodeGatherIdsHost(ta, bb, kb, kv_b, ratio);
            defer ta.free(eb);
            try std.testing.expectEqualSlices(i32, ea, plans[0].rows);
            try std.testing.expectEqualSlices(i32, eb, plans[1].rows);
        }
        for ([_]c_int{ 2, 4, 5, 8 }) |s| {
            const ba = try qsaVerifyBlocksHost(ta, rnd, s, kv_a, kb, ratio);
            defer ta.free(ba);
            const bb = try qsaVerifyBlocksHost(ta, rnd, s, kv_b, kb, ratio);
            defer ta.free(bb);
            const slot_blocks = [_][]const i32{ ba, bb };
            const kv_lens = [_]c_int{ kv_a, kv_b };
            var plans = try qsaBatchedPlanHost(ta, &slot_blocks, s, kb, &kv_lens, ratio);
            defer {
                var pi: usize = 0;
                while (pi < plans.len) : (pi += 1) plans[pi].deinit(ta);
                ta.free(plans);
            }
            var pa = try qsaVerifyPlanHost(ta, ba, s, kb, kv_a, ratio);
            defer pa.deinit(ta);
            var pb = try qsaVerifyPlanHost(ta, bb, s, kb, kv_b, ratio);
            defer pb.deinit(ta);
            try std.testing.expectEqualSlices(i32, pa.rows, plans[0].rows);
            try std.testing.expectEqualSlices(bool, pa.mask, plans[0].mask);
            try std.testing.expectEqualSlices(i32, pb.rows, plans[1].rows);
            try std.testing.expectEqualSlices(bool, pb.mask, plans[1].mask);
        }
    }
}

test "qsa arm meters: one-shot bits, width buckets, and the per-request tally" {
    // The meters exist because a process-global `engaged` bool cannot answer
    // "which arm served THIS request at THIS context length". Each property
    // it needs is pinned here: a bit fires once, decode/verify/prefill get
    // SEPARATE engagement bits (so a 16k decode engagement cannot vouch for a
    // 62k verify call), and an empty tally reports nothing rather than a
    // misleading `qsa=mask`.
    var bits: OneShotBits = .{};
    try std.testing.expect(bits.take(3));
    try std.testing.expect(!bits.take(3));
    try std.testing.expect(bits.take(4));
    try std.testing.expect(!bits.take(4));
    bits.reset();
    try std.testing.expect(bits.take(3));

    try std.testing.expectEqual(@as(u2, 0), qsaWidthBucket(1));
    try std.testing.expectEqual(@as(u2, 1), qsaWidthBucket(2));
    try std.testing.expectEqual(@as(u2, 1), qsaWidthBucket(FUSED256_MIN_Q_LEN - 1));
    try std.testing.expectEqual(@as(u2, 2), qsaWidthBucket(FUSED256_MIN_Q_LEN));
    try std.testing.expectEqual(@as(u2, 2), qsaWidthBucket(4096));

    // Every (arm, bucket) pair owns a distinct bit — a collision would let one
    // arm's engagement silence another's.
    var seen = std.AutoHashMap(u5, void).init(std.testing.allocator);
    defer seen.deinit();
    for ([_]QsaArm{ .mask, .decode_gather, .verify_gather, .prefill_gather }) |arm| {
        for ([_]c_int{ 1, 5, 64 }) |s| {
            const bit = qsaEngagedBit(arm, s);
            try std.testing.expect(!seen.contains(bit));
            try seen.put(bit, {});
        }
    }

    var counts: QsaArmCounts = .{};
    try std.testing.expect(counts.majority() == null);
    try std.testing.expectEqual(@as(u32, 0), counts.total());
    counts.note(.mask);
    counts.note(.mask);
    counts.note(.decode_gather);
    try std.testing.expectEqual(@as(u32, 3), counts.total());
    try std.testing.expectEqual(QsaArm.mask, counts.majority().?);
    counts.note(.decode_gather);
    // Tie breaks toward the gather: a tie must never read as "the mask is fine".
    try std.testing.expectEqual(QsaArm.decode_gather, counts.majority().?);
    counts.reset();
    try std.testing.expectEqual(@as(u32, 0), counts.total());

    // Every reason carries distinct, non-empty text (the log's whole value).
    var texts = std.StringHashMap(void).init(std.testing.allocator);
    defer texts.deinit();
    inline for (@typeInfo(QsaDeclineReason).@"enum".field_values) |v| {
        const t = (@as(QsaDeclineReason, @fromBackingInt(@intCast(v)))).text();
        try std.testing.expect(t.len > 0);
        try std.testing.expect(!texts.contains(t));
        try texts.put(t, {});
        // Reasons index OneShotBits directly — they must fit its 32 bits.
        try std.testing.expect(v < 32);
    }
}

test "qsa arm tally is owned by the request: two ForwardCtx owners never mix" {
    // The `[qsa-arms]` line is a PER-REQUEST meter, so its tally belongs to
    // the seam that logs it (the Generator's own ForwardCtx), never to the
    // shared Transformer. qwen4 text slots batch-decode on ONE Transformer
    // (`forwardMoeBatchedDecode`) and a cold prefill interleaves with other
    // slots' decode ticks (`scheduler.interleaveDecodeTick`), so a
    // Transformer-level counter printed the union of every concurrent slot
    // since the last reset and left the next finisher a near-empty tally — a
    // short chat claimed `qsa=verify_gather` while the 40k chat later logged
    // `qsa=mask`. Two owners here stand in for two concurrent slots.
    //
    // `cache` / `moe_seq_offset` are never read on this path — the tally is
    // pure bookkeeping on the ctx — so the ctxs are built as shells.
    var cache_a: KVCache = undefined;
    var cache_b: KVCache = undefined;
    var off_a: usize = 0;
    var off_b: usize = 0;
    var a: ForwardCtx = .{ .cache = &cache_a, .moe_seq_offset = &off_a, .ssm_entries = null, .capture_hidden = null, .vision_embeddings = null };
    var b: ForwardCtx = .{ .cache = &cache_b, .moe_seq_offset = &off_b, .ssm_entries = null, .capture_hidden = null, .vision_embeddings = null };

    // A serves a decode-width gather and a prefill-width gather; B serves the
    // dense mask arm in between (the interleave the shared counter merged).
    a.qsa_arms.noteGather(1);
    b.qsa_arms.note(.mask);
    a.qsa_arms.noteGather(FUSED256_MIN_Q_LEN);
    b.qsa_arms.note(.mask);
    a.qsa_arms.noteGather(4);

    try std.testing.expectEqual(@as(u32, 3), a.qsa_arms.total());
    try std.testing.expectEqual(@as(u32, 0), a.qsa_arms.mask);
    try std.testing.expectEqual(@as(u32, 1), a.qsa_arms.decode);
    try std.testing.expectEqual(@as(u32, 1), a.qsa_arms.verify);
    try std.testing.expectEqual(@as(u32, 1), a.qsa_arms.prefill);
    try std.testing.expectEqual(@as(u32, 2), b.qsa_arms.total());
    try std.testing.expectEqual(@as(u32, 2), b.qsa_arms.mask);

    // A finishing request takes its OWN tally; B's survives untouched (the
    // shared field reset whichever slot finished first).
    const taken = a.qsa_arms.take().?;
    try std.testing.expectEqual(@as(u32, 3), taken.total());
    try std.testing.expect(a.qsa_arms.take() == null);
    try std.testing.expectEqual(@as(u32, 2), b.qsa_arms.total());
    try std.testing.expectEqual(QsaArm.mask, b.qsa_arms.take().?.majority().?);

    // One batched call serves the whole group with the SAME arm: it counts
    // once on every participating slot, not once on a shared counter.
    const slots = [_]*ForwardCtx{ &a, &b };
    noteQsaArmForSlots(&slots, .mask);
    try std.testing.expectEqual(@as(u32, 1), a.qsa_arms.mask);
    try std.testing.expectEqual(@as(u32, 1), b.qsa_arms.mask);
}

test "qsa verify gather geometry: rows, the union tail, and the no-win decline" {
    // The ONE owner of R. Production budget is 512 blocks of 4: at S=7 the
    // union is 14336 rows + tail, so the arm needs kv well past that; at
    // S=15 it is 30720 and only pays past ~31k.
    const g7 = QsaVerifyGeom.compute(7, 65536, 4, 512).?;
    try std.testing.expectEqual(@as(c_int, 3584), g7.k_blocks);
    try std.testing.expectEqual(@as(c_int, 65528), g7.tail_start); // ratio*floor((65529+1)/4)
    try std.testing.expectEqual(@as(c_int, 8), g7.tail_len);
    try std.testing.expectEqual(@as(c_int, 14344), g7.rows);
    try std.testing.expectEqual(@as(c_int, 30736), QsaVerifyGeom.compute(15, 65536, 4, 512).?.rows);
    // 20k at the production budget: S=7 wins, S=15 gathers the whole cache.
    try std.testing.expect(QsaVerifyGeom.compute(7, 20000, 4, 512).?.rows < 20000);
    try std.testing.expectEqual(@as(c_int, 20000), QsaVerifyGeom.compute(15, 20000, 4, 512).?.rows);
    // Decode and prefill widths are not this arm's business.
    try std.testing.expect(QsaVerifyGeom.compute(1, 65536, 4, 512) == null);
    try std.testing.expect(QsaVerifyGeom.compute(FUSED256_MIN_Q_LEN, 65536, 4, 512) == null);
}

test "qsa verify gather: subset SDPA matches masked full SDPA at verify widths (dense + affine-8 cache)" {
    // (b) End-to-end parity of `qsaVerifyGatherAttn` against the arm it
    // replaces: the production dense [S, kv] expansion (`qsaMaskFromBlocks`)
    // over the FULL cache. The quant arm reads a real affine-8 `KVCache`
    // (strided views, capacity > logical) so the subset dequant sees the live
    // layout, and its reference is the same cache's FULL dense view.
    const s = mlx.gpuStream();
    qsa_verify_gather_override = true;
    qsa_verify_gather_min_kv_override = 1024;
    defer qsa_verify_gather_override = null;
    defer qsa_verify_gather_min_kv_override = null;
    var prng = std.Random.DefaultPrng.init(0x7e21f);
    const rnd = prng.random();
    const ratio: c_int = 4;
    const cases = [_]struct { s: c_int, kv: c_int, kb: c_int }{
        .{ .s = 2, .kv = 4096, .kb = 32 },
        .{ .s = 5, .kv = 4096, .kb = 32 },
        .{ .s = 7, .kv = 20000, .kb = 128 },
        .{ .s = 15, .kv = 20000, .kb = 128 },
    };
    for (cases) |c| {
        const q_shape = [_]c_int{ 1, 24, c.s, 256 }; // gqa 12, hd 256
        const kv_shape = [_]c_int{ 1, 2, c.kv, 256 };
        const q = try attn256RandBf16(rnd, &q_shape, s);
        defer _ = mlx.mlx_array_free(q);
        const k_dense = try testRandWeightBf16(rnd, &kv_shape, s);
        defer _ = mlx.mlx_array_free(k_dense);
        const v_dense = try testRandWeightBf16(rnd, &kv_shape, s);
        defer _ = mlx.mlx_array_free(v_dense);

        const blocks_host = try qsaVerifyBlocksHost(std.testing.allocator, rnd, c.s, c.kv, c.kb, ratio);
        defer std.testing.allocator.free(blocks_host);
        const bshape = [_]c_int{ 1, c.s, c.kb };
        const blocks = mlx.mlx_array_new_data(blocks_host.ptr, &bshape, 3, .int32);
        defer _ = mlx.mlx_array_free(blocks);
        const mask = try qsaMaskFromBlocks(s, blocks, c.kv, ratio);
        defer _ = mlx.mlx_array_free(mask);
        const scale: f32 = 1.0 / 16.0;

        var dview = DenseKVView{ .k = k_dense, .v = v_dense, .owned = false };
        const got = (try qsaVerifyGatherAttn(s, q, &dview, blocks, ratio, scale)) orelse return error.VerifyGatherDeclined;
        defer _ = mlx.mlx_array_free(got);
        const ref = try attn256Reference(q, k_dense, v_dense, scale, "array", mask, s);
        defer _ = mlx.mlx_array_free(ref);
        const d_diff = try attn256MaxDiff(got, ref, s);
        const d_cos = try attn256Cosine(got, ref, s);
        errdefer std.debug.print("[qsa-verify-gather] S={d} kv={d} dense: max diff {d:.6} cos {d:.7}\n", .{ c.s, c.kv, d_diff, d_cos });
        try std.testing.expect(d_diff < 0.005);
        try std.testing.expect(d_cos > 0.9999);

        var cache = try KVCache.initWithConfig(std.testing.allocator, 1, kv_quant.KVQuantConfig.affine(8));
        defer cache.deinit();
        var view = try cache.update(0, k_dense, v_dense, s, 0);
        defer view.deinit();
        try std.testing.expect(view.has_quant_triple);
        const gotq = (try qsaVerifyGatherAttn(s, q, &view, blocks, ratio, scale)) orelse return error.VerifyGatherDeclined;
        defer _ = mlx.mlx_array_free(gotq);
        const refq = try attn256Reference(q, view.k, view.v, scale, "array", mask, s);
        defer _ = mlx.mlx_array_free(refq);
        const q_diff = try attn256MaxDiff(gotq, refq, s);
        const q_cos = try attn256Cosine(gotq, refq, s);
        errdefer std.debug.print("[qsa-verify-gather] S={d} kv={d} affine-8: max diff {d:.6} cos {d:.7}\n", .{ c.s, c.kv, q_diff, q_cos });
        try std.testing.expect(q_diff < 0.01);
        try std.testing.expect(q_cos > 0.9999);
    }

    // Kill switch and the kv floor both decline back to the dense mask arm.
    const q_shape = [_]c_int{ 1, 24, 5, 256 };
    const kv_shape = [_]c_int{ 1, 2, 4096, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const kd = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(kd);
    const vd = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(vd);
    const bh = try qsaVerifyBlocksHost(std.testing.allocator, rnd, 5, 4096, 32, ratio);
    defer std.testing.allocator.free(bh);
    const bsh = [_]c_int{ 1, 5, 32 };
    const bl = mlx.mlx_array_new_data(bh.ptr, &bsh, 3, .int32);
    defer _ = mlx.mlx_array_free(bl);
    var dv2 = DenseKVView{ .k = kd, .v = vd, .owned = false };
    qsa_verify_gather_override = false;
    try std.testing.expect((try qsaVerifyGatherAttn(s, q, &dv2, bl, ratio, 1.0)) == null);
    qsa_verify_gather_override = true;
    qsa_verify_gather_min_kv_override = 16384;
    try std.testing.expect((try qsaVerifyGatherAttn(s, q, &dv2, bl, ratio, 1.0)) == null);
}

test "qsa sparse attn: dense and affine strided caches retain the reference error bar" {
    // The bar is two bf16 ulps at the reference's own scale (an ulp is relative, so a constant
    // bar only works at one output scale), plus cosine against a wrong key set.
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const ta = std.testing.allocator;
    const ratio: c_int = 4;
    var prng = std.Random.DefaultPrng.init(0x5A17_C0DE);
    const rnd = prng.random();

    const cases = [_]struct { s: c_int, kv: c_int, kb: c_int, hq: c_int, hkv: c_int, bits: u8 = 8 }{
        // Decode width: NSPLIT hits the ceiling, most splits are empty.
        .{ .s = 1, .kv = 4096, .kb = 64, .hq = 24, .hkv = 2 },
        .{ .s = 1, .kv = 70000, .kb = 512, .hq = 24, .hkv = 2 },
        .{ .s = 3, .kv = 62000, .kb = 512, .hq = 24, .hkv = 2 },
        // Tile count far below NSPLIT.
        .{ .s = 2, .kv = 1024, .kb = 512, .hq = 24, .hkv = 2 },
        .{ .s = 15, .kv = 70000, .kb = 512, .hq = 24, .hkv = 2 },
        .{ .s = 2, .kv = 4096, .kb = 64, .hq = 24, .hkv = 2 },
        .{ .s = 6, .kv = 4096, .kb = 64, .hq = 24, .hkv = 2 },
        .{ .s = 15, .kv = 4096, .kb = 64, .hq = 24, .hkv = 2 },
        .{ .s = 6, .kv = 20000, .kb = 128, .hq = 24, .hkv = 2 },
        .{ .s = 15, .kv = 20000, .kb = 128, .hq = 12, .hkv = 1 },
        // Fewer visible blocks than the budget: the INT_MAX pads must never be read.
        .{ .s = 6, .kv = 4096, .kb = 2048, .hq = 24, .hkv = 2 },
        .{ .s = 1, .kv = 70000, .kb = 512, .hq = 24, .hkv = 2, .bits = 6 },
        .{ .s = 6, .kv = 20000, .kb = 128, .hq = 24, .hkv = 2, .bits = 6 },
        .{ .s = 15, .kv = 20001, .kb = 128, .hq = 12, .hkv = 1, .bits = 6 },
        .{ .s = 6, .kv = 4096, .kb = 2048, .hq = 24, .hkv = 2, .bits = 6 },
    };

    for (cases) |c| {
        const q_shape = [_]c_int{ 1, c.hq, c.s, 256 };
        const kv_shape = [_]c_int{ 1, c.hkv, c.kv, 256 };
        const q = try attn256RandBf16(rnd, &q_shape, s);
        defer _ = mlx.mlx_array_free(q);
        const k_dense = try testRandWeightBf16(rnd, &kv_shape, s);
        defer _ = mlx.mlx_array_free(k_dense);
        const v_dense = try testRandWeightBf16(rnd, &kv_shape, s);
        defer _ = mlx.mlx_array_free(v_dense);

        const blocks_host = try qsaVerifyBlocksHost(ta, rnd, c.s, c.kv, c.kb, ratio);
        defer ta.free(blocks_host);
        const bshape = [_]c_int{ 1, c.s, c.kb };
        const blocks = mlx.mlx_array_new_data(blocks_host.ptr, &bshape, 3, .int32);
        defer _ = mlx.mlx_array_free(blocks);
        const mask = try qsaMaskFromBlocks(s, blocks, c.kv, ratio);
        defer _ = mlx.mlx_array_free(mask);
        const scale: f32 = 1.0 / 16.0;

        var dview = DenseKVView{ .k = k_dense, .v = v_dense, .owned = false };
        const sparse_dense = (try qsaSparseAttn(s, q, &dview, blocks, ratio, scale)) orelse return error.SparseAttnDeclined;
        defer _ = mlx.mlx_array_free(sparse_dense);
        const ref_d = try attn256Reference(q, k_dense, v_dense, scale, "array", mask, s);
        defer _ = mlx.mlx_array_free(ref_d);
        try std.testing.expect(try attn256Cosine(sparse_dense, ref_d, s) > 0.99999);
        try std.testing.expect(try attn256MaxDiff(sparse_dense, ref_d, s) <= try attn256UlpBar(ref_d, s));
        if (try qsaVerifyGatherAttn(s, q, &dview, blocks, ratio, scale)) |got_d| {
            defer _ = mlx.mlx_array_free(got_d);
            const dd = try attn256MaxDiff(got_d, ref_d, s);
            const dc = try attn256Cosine(got_d, ref_d, s);
            const d_bar = try attn256UlpBar(ref_d, s);
            errdefer std.debug.print("[qsa-attn] S={d} kv={d} kb={d} gqa={d} dense->union gather: max diff {d:.6} cos {d:.7} (bar {d:.6})\n", .{ c.s, c.kv, c.kb, @divExact(c.hq, c.hkv), dd, dc, d_bar });
            try std.testing.expect(dc > 0.99999);
            try std.testing.expect(dd <= d_bar);
        }

        // Packed arm: a real affine cache, filled in two appends so the view is strided.
        var cache = try KVCache.initWithConfig(ta, 1, kv_quant.KVQuantConfig.affine(c.bits));
        defer cache.deinit();
        var view = try qsaAttnCacheFixture(ta, s, k_dense, v_dense, c.kv, @divTrunc(c.kv, 2), kv_quant.KVQuantConfig.affine(c.bits), &cache);
        defer view.deinit();
        try std.testing.expect(view.has_quant_triple);
        try std.testing.expectEqual(c.kv, mlx.getShape(view.k)[2]);

        const got_q = (try qsaSparseAttn(s, q, &view, blocks, ratio, scale)) orelse return error.SparseAttnDeclined;
        defer _ = mlx.mlx_array_free(got_q);
        // Reference reads the cache's own dequantized rows.
        const ref_q = try attn256Reference(q, view.k, view.v, scale, "array", mask, s);
        defer _ = mlx.mlx_array_free(ref_q);
        const qd = try attn256MaxDiff(got_q, ref_q, s);
        const qc = try attn256Cosine(got_q, ref_q, s);
        const q_bar = try attn256UlpBar(ref_q, s);
        errdefer std.debug.print("[qsa-attn] S={d} kv={d} kb={d} gqa={d} affine-{d}: max diff {d:.6} cos {d:.7} (bar {d:.6})\n", .{ c.s, c.kv, c.kb, @divExact(c.hq, c.hkv), c.bits, qd, qc, q_bar });
        try std.testing.expect(qc > 0.99999);
        try std.testing.expect(qd <= q_bar);

        // The two shipping arms must not disagree with each other.
        if (qsaVerifyGatherAttn(s, q, &view, blocks, ratio, scale) catch null) |gather| {
            defer _ = mlx.mlx_array_free(gather);
            const ac = try attn256Cosine(got_q, gather, s);
            try std.testing.expect(ac > 0.99999);
        }
    }
}

test "qsa sparse attn: packed caches serve widths past 15 within the error bar of today's dense-rebuild route" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const ta = std.testing.allocator;
    const ratio: c_int = 4;
    var prng = std.Random.DefaultPrng.init(0x5A17_0016);
    const rnd = prng.random();
    defer qsa_attn_nsplit_override = null;

    const cases = [_]struct { s: c_int, kv: c_int, kb: c_int, bits: u8, nsplit: ?c_int = null }{
        .{ .s = 16, .kv = 70001, .kb = 512, .bits = 8 },
        .{ .s = 24, .kv = 20002, .kb = 128, .bits = 8 },
        .{ .s = 32, .kv = 62003, .kb = 512, .bits = 4 },
        .{ .s = QSA_ATTN_PACKED_MAX_S, .kv = 70000, .kb = 512, .bits = 8 },
        .{ .s = QSA_ATTN_PACKED_MAX_S, .kv = 33333, .kb = 512, .bits = 4 },
        .{ .s = 24, .kv = 62003, .kb = 512, .bits = 6 },
        .{ .s = QSA_ATTN_PACKED_MAX_S, .kv = 33335, .kb = 512, .bits = 6, .nsplit = 64 },
        // Fewer visible blocks than the budget: the INT_MAX pads must never be read.
        .{ .s = 32, .kv = 4097, .kb = 2048, .bits = 8 },
        .{ .s = 16, .kv = 70002, .kb = 512, .bits = 8, .nsplit = 64 },
        .{ .s = QSA_ATTN_PACKED_MAX_S, .kv = 20001, .kb = 512, .bits = 4, .nsplit = 64 },
    };
    for (cases) |c| {
        qsa_attn_nsplit_override = c.nsplit;
        const q = try attn256RandBf16(rnd, &[_]c_int{ 1, 24, c.s, 256 }, s);
        defer _ = mlx.mlx_array_free(q);
        const k_dense = try testRandWeightBf16(rnd, &[_]c_int{ 1, 2, c.kv, 256 }, s);
        defer _ = mlx.mlx_array_free(k_dense);
        const v_dense = try testRandWeightBf16(rnd, &[_]c_int{ 1, 2, c.kv, 256 }, s);
        defer _ = mlx.mlx_array_free(v_dense);
        const blocks_host = try qsaVerifyBlocksHost(ta, rnd, c.s, c.kv, c.kb, ratio);
        defer ta.free(blocks_host);
        const blocks = mlx.mlx_array_new_data(blocks_host.ptr, &[_]c_int{ 1, c.s, c.kb }, 3, .int32);
        defer _ = mlx.mlx_array_free(blocks);
        const mask = try qsaMaskFromBlocks(s, blocks, c.kv, ratio);
        defer _ = mlx.mlx_array_free(mask);
        var cache = try KVCache.initWithConfig(ta, 1, kv_quant.KVQuantConfig.affine(c.bits));
        defer cache.deinit();
        var view = try qsaAttnCacheFixture(ta, s, k_dense, v_dense, c.kv, @divTrunc(c.kv, 2), kv_quant.KVQuantConfig.affine(c.bits), &cache);
        defer view.deinit();
        const scale: f32 = 1.0 / 16.0;

        const ref = try attn256Reference(q, view.k, view.v, scale, "array", mask, s);
        defer _ = mlx.mlx_array_free(ref);
        const bar = try attn256UlpBar(ref, s);
        // Today's route at these widths: the prefill gather over the dense rebuild.
        const old = (try gatherQsa256(s, q, view.k, view.v, scale, blocks, ratio)) orelse return error.GatherDeclined;
        defer _ = mlx.mlx_array_free(old);
        try std.testing.expect(try attn256MaxDiff(old, ref, s) <= bar);

        const got = (try qsaSparseAttn(s, q, &view, blocks, ratio, scale)) orelse return error.SparseAttnDeclined;
        defer _ = mlx.mlx_array_free(got);
        const d = try attn256MaxDiff(got, ref, s);
        const cs = try attn256Cosine(got, ref, s);
        errdefer std.debug.print("[qsa-attn] S={d} kv={d} kb={d} bits={d}: max diff {d:.6} cos {d:.7} (bar {d:.6})\n", .{ c.s, c.kv, c.kb, c.bits, d, cs, bar });
        try std.testing.expect(cs > 0.99999);
        try std.testing.expect(d <= bar);
    }
}

test "qsa sparse attn: the split-K merge is invariant in the split count (1, 8, 64)" {
    // Splitting a row's key range and recombining the per-split online-softmax states must be
    // exact: a wrong merge would move the answer with NSPLIT. NSPLIT = 1 is the control.
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const ta = std.testing.allocator;
    const ratio: c_int = 4;
    var prng = std.Random.DefaultPrng.init(0x5B17_C0DE);
    const rnd = prng.random();

    const seq: c_int = 6;
    const kv: c_int = 20000;
    const kb: c_int = 128;
    const q = try attn256RandBf16(rnd, &[_]c_int{ 1, 24, seq, 256 }, s);
    defer _ = mlx.mlx_array_free(q);
    const k_dense = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, kv, 256 }, s);
    defer _ = mlx.mlx_array_free(k_dense);
    const v_dense = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, kv, 256 }, s);
    defer _ = mlx.mlx_array_free(v_dense);
    const blocks_host = try qsaVerifyBlocksHost(ta, rnd, seq, kv, kb, ratio);
    defer ta.free(blocks_host);
    const blocks = mlx.mlx_array_new_data(blocks_host.ptr, &[_]c_int{ 1, seq, kb }, 3, .int32);
    defer _ = mlx.mlx_array_free(blocks);
    const mask = try qsaMaskFromBlocks(s, blocks, kv, ratio);
    defer _ = mlx.mlx_array_free(mask);
    const scale: f32 = 1.0 / 16.0;

    var cache = try KVCache.initWithConfig(ta, 1, kv_quant.KVQuantConfig.affine(8));
    defer cache.deinit();
    var view = try qsaAttnCacheFixture(ta, s, k_dense, v_dense, kv, @divTrunc(kv, 2), kv_quant.KVQuantConfig.affine(8), &cache);
    defer view.deinit();
    const ref = try attn256Reference(q, view.k, view.v, scale, "array", mask, s);
    defer _ = mlx.mlx_array_free(ref);
    const bar = try attn256UlpBar(ref, s);

    defer qsa_attn_nsplit_override = null;
    var base: ?mlx.mlx_array = null;
    defer if (base) |b| {
        _ = mlx.mlx_array_free(b);
    };
    for ([_]c_int{ 1, 8, 64 }) |ns| {
        qsa_attn_nsplit_override = ns;
        try std.testing.expectEqual(ns, qsaAttnNSplit());
        const got = (try qsaSparseAttn(s, q, &view, blocks, ratio, scale)) orelse return error.SparseAttnDeclined;
        defer _ = mlx.mlx_array_free(got);
        const d = try attn256MaxDiff(got, ref, s);
        const c = try attn256Cosine(got, ref, s);
        errdefer std.debug.print("[qsa-attn] nsplit={d}: max diff {d:.6} cos {d:.7} (bar {d:.6})\n", .{ ns, d, c, bar });
        try std.testing.expect(c > 0.99999);
        try std.testing.expect(d <= bar);
        if (base) |b| {
            const cc = try attn256Cosine(got, b, s);
            try std.testing.expect(cc > 0.99999);
        } else {
            var keep = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_set(&keep, got));
            base = keep;
        }
    }
}

test "qsa sparse attn: NSPLIT is one MEASURED constant, and the width floor binds the DISPATCH" {
    defer qsa_attn_nsplit_override = null;
    qsa_attn_nsplit_override = null;
    try std.testing.expectEqual(QSA_ATTN_NSPLIT_DEFAULT, qsaAttnNSplit());
    try std.testing.expectEqual(@as(c_int, 16), QSA_ATTN_NSPLIT_DEFAULT);
    qsa_attn_nsplit_override = 8;
    try std.testing.expectEqual(@as(c_int, 8), qsaAttnNSplit());
    qsa_attn_nsplit_override = 5;
    try std.testing.expectEqual(@as(c_int, 8), qsaAttnNSplit());
    qsa_attn_nsplit_override = 1000;
    try std.testing.expectEqual(QSA_ATTN_MAX_NSPLIT, qsaAttnNSplit());
    qsa_attn_nsplit_override = null;

    defer qsa_attn_bk_override = null;
    try std.testing.expectEqual(QSA_ATTN_BK, qsaAttnBk());
    // BK must be a multiple of 8; 12 is rejected back to the default.
    qsa_attn_bk_override = 12;
    try std.testing.expectEqual(QSA_ATTN_BK, qsaAttnBk());
    qsa_attn_bk_override = 0;
    try std.testing.expectEqual(QSA_ATTN_BK, qsaAttnBk());
    qsa_attn_bk_override = 64;
    try std.testing.expectEqual(QSA_ATTN_BK, qsaAttnBk());
    qsa_attn_bk_override = 8;
    try std.testing.expectEqual(@as(c_int, 8), qsaAttnBk());
    try std.testing.expect(256 * (8 + 8) * 2 <= 10 * 1024);
    qsa_attn_bk_override = null;
}

test "qsa dispatch: dense and quantized decode and verify share the same row body" {
    for ([_]bool{ false, true }) |quant| {
        for (1..16) |width| try std.testing.expect(qsaSparseAttnServes(quant, @intCast(width)));
        try std.testing.expect(!qsaSparseAttnServes(quant, 0));
        try std.testing.expectEqual(quant, qsaSparseAttnServes(quant, FUSED256_MIN_Q_LEN));
        try std.testing.expectEqual(quant, qsaSparseAttnServes(quant, QSA_ATTN_PACKED_MAX_S));
        try std.testing.expect(!qsaSparseAttnServes(quant, QSA_ATTN_PACKED_MAX_S + 1));
    }
}

test "qsa sparse attn: gqa tile reuse is invariant — one kv head serving 12 q heads equals 12 serving one each" {
    // The kernel dequantizes each K/V element once per threadgroup and feeds it to all `gqa`
    // rows: gqa 12 (tile reuse) vs gqa 1 (12 identical kv heads) must agree exactly.
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const ta = std.testing.allocator;
    const ratio: c_int = 4;
    const kv: c_int = 4096;
    const seq: c_int = 6;
    const kb: c_int = 64;
    var prng = std.Random.DefaultPrng.init(0x7EA_1EAF);
    const rnd = prng.random();

    const q = try attn256RandBf16(rnd, &[_]c_int{ 1, 12, seq, 256 }, s);
    defer _ = mlx.mlx_array_free(q);
    const k1 = try attn256RandBf16(rnd, &[_]c_int{ 1, 1, kv, 256 }, s);
    defer _ = mlx.mlx_array_free(k1);
    const v1 = try attn256RandBf16(rnd, &[_]c_int{ 1, 1, kv, 256 }, s);
    defer _ = mlx.mlx_array_free(v1);
    var k12 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k12);
    var v12 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v12);
    try mlx.check(mlx.mlx_broadcast_to(&k12, k1, &[_]c_int{ 1, 12, kv, 256 }, 4, s));
    try mlx.check(mlx.mlx_broadcast_to(&v12, v1, &[_]c_int{ 1, 12, kv, 256 }, 4, s));
    var k12c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k12c);
    var v12c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v12c);
    try mlx.check(mlx.mlx_contiguous(&k12c, k12, false, s));
    try mlx.check(mlx.mlx_contiguous(&v12c, v12, false, s));

    const bh = try qsaVerifyBlocksHost(ta, rnd, seq, kv, kb, ratio);
    defer ta.free(bh);
    const blocks = mlx.mlx_array_new_data(bh.ptr, &[_]c_int{ 1, seq, kb }, 3, .int32);
    defer _ = mlx.mlx_array_free(blocks);
    const scale: f32 = 1.0 / 16.0;

    // Both fixtures must be real quantized caches (a dense view declines).
    const qcfg = kv_quant.KVQuantConfig.affine(8);
    var cache_r = try KVCache.initWithConfig(ta, 1, qcfg);
    defer cache_r.deinit();
    var v_reuse = try qsaAttnCacheFixture(ta, s, k1, v1, kv, @divTrunc(kv, 2), qcfg, &cache_r); // gqa 12
    defer v_reuse.deinit();
    try std.testing.expect(v_reuse.has_quant_triple);
    var cache_n = try KVCache.initWithConfig(ta, 1, qcfg);
    defer cache_n.deinit();
    var v_noreuse = try qsaAttnCacheFixture(ta, s, k12c, v12c, kv, @divTrunc(kv, 2), qcfg, &cache_n); // gqa 1
    defer v_noreuse.deinit();
    const a = (try qsaSparseAttn(s, q, &v_reuse, blocks, ratio, scale)) orelse return error.SparseAttnDeclined;
    defer _ = mlx.mlx_array_free(a);
    const b = (try qsaSparseAttn(s, q, &v_noreuse, blocks, ratio, scale)) orelse return error.SparseAttnDeclined;
    defer _ = mlx.mlx_array_free(b);
    try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(a, b, s));
}

test "qsa sparse attn: loader simdgroups leave every output bit unchanged" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const ta = std.testing.allocator;
    const ratio: c_int = 4;
    var prng = std.Random.DefaultPrng.init(0x10AD_5165);
    const rnd = prng.random();
    defer qsa_attn_simdgroups_override = null;

    const cases = [_]struct { s: c_int, kv: c_int, kb: c_int, bits: u8 }{
        .{ .s = 1, .kv = 4096, .kb = 512, .bits = 8 },
        .{ .s = 1, .kv = 70001, .kb = 512, .bits = 8 },
        .{ .s = 1, .kv = 9000, .kb = 512, .bits = 4 },
        .{ .s = 3, .kv = 20002, .kb = 512, .bits = 8 },
        .{ .s = 8, .kv = 4097, .kb = 2048, .bits = 8 },
    };
    for (cases) |c| {
        const q = try attn256RandBf16(rnd, &[_]c_int{ 1, 24, c.s, 256 }, s);
        defer _ = mlx.mlx_array_free(q);
        const k_dense = try testRandWeightBf16(rnd, &[_]c_int{ 1, 2, c.kv, 256 }, s);
        defer _ = mlx.mlx_array_free(k_dense);
        const v_dense = try testRandWeightBf16(rnd, &[_]c_int{ 1, 2, c.kv, 256 }, s);
        defer _ = mlx.mlx_array_free(v_dense);
        const blocks_host = try qsaVerifyBlocksHost(ta, rnd, c.s, c.kv, c.kb, ratio);
        defer ta.free(blocks_host);
        const blocks = mlx.mlx_array_new_data(blocks_host.ptr, &[_]c_int{ 1, c.s, c.kb }, 3, .int32);
        defer _ = mlx.mlx_array_free(blocks);
        var cache = try KVCache.initWithConfig(ta, 1, kv_quant.KVQuantConfig.affine(c.bits));
        defer cache.deinit();
        var view = try qsaAttnCacheFixture(ta, s, k_dense, v_dense, c.kv, @divTrunc(c.kv, 2), kv_quant.KVQuantConfig.affine(c.bits), &cache);
        defer view.deinit();

        qsa_attn_simdgroups_override = 2;
        const base = (try qsaSparseAttn(s, q, &view, blocks, ratio, 1.0 / 16.0)) orelse return error.SparseAttnDeclined;
        defer _ = mlx.mlx_array_free(base);
        for ([_]c_int{ 4, 8, 16 }) |sgs| {
            qsa_attn_simdgroups_override = sgs;
            const got = (try qsaSparseAttn(s, q, &view, blocks, ratio, 1.0 / 16.0)) orelse return error.SparseAttnDeclined;
            defer _ = mlx.mlx_array_free(got);
            try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(got, base, s));
        }
    }
    // The decode grid (2 kv heads x 16 splits) is too small to hide the tile loads alone.
    qsa_attn_simdgroups_override = null;
    try std.testing.expect(qsaAttnSimdgroups(1, 2, 2, QSA_ATTN_NSPLIT_DEFAULT) > 2);
}

test "qsa sparse attn µbench: decode width, 12 dependent layers (SUSHI_QSA_ATTN_UBENCH=1)" {
    if (!diagEnvOn("SUSHI_QSA_ATTN_UBENCH")) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const ta = std.testing.allocator;
    const ratio: c_int = 4;
    const layers: usize = 12;
    const chain: usize = 8;
    const reps: usize = 7;
    var prng = std.Random.DefaultPrng.init(0x0B5E_7A11);
    const rnd = prng.random();
    defer qsa_attn_simdgroups_override = null;
    const arms = [_]c_int{ 2, 16 };
    for ([_]c_int{ 4096, 32768 }) |kv| {
        var caches: [layers]KVCache = undefined;
        var views: [layers]DenseKVView = undefined;
        var built: usize = 0;
        defer for (0..built) |i| {
            views[i].deinit();
            caches[i].deinit();
        };
        for (0..layers) |i| {
            const k = try testRandWeightBf16(rnd, &[_]c_int{ 1, 2, kv, 256 }, s);
            defer _ = mlx.mlx_array_free(k);
            const v = try testRandWeightBf16(rnd, &[_]c_int{ 1, 2, kv, 256 }, s);
            defer _ = mlx.mlx_array_free(v);
            caches[i] = try KVCache.initWithConfig(ta, 1, kv_quant.KVQuantConfig.affine(8));
            views[i] = try qsaAttnCacheFixture(ta, s, k, v, kv, kv - 1, kv_quant.KVQuantConfig.affine(8), &caches[i]);
            built += 1;
        }
        const blocks_host = try qsaVerifyBlocksHost(ta, rnd, 1, kv, 512, ratio);
        defer ta.free(blocks_host);
        const blocks = mlx.mlx_array_new_data(blocks_host.ptr, &[_]c_int{ 1, 1, 512 }, 3, .int32);
        defer _ = mlx.mlx_array_free(blocks);
        const q0 = try attn256RandBf16(rnd, &[_]c_int{ 1, 24, 1, 256 }, s);
        defer _ = mlx.mlx_array_free(q0);
        var us: [arms.len][reps]f64 = undefined;
        for (0..reps + 1) |rep| for (arms, 0..) |arm, ai| {
            qsa_attn_simdgroups_override = arm;
            var q = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_set(&q, q0));
            for (0..chain) |_| for (&views) |*view| {
                const out = (try qsaSparseAttn(s, q, view, blocks, ratio, 1.0 / 16.0)) orelse return error.SparseAttnDeclined;
                _ = mlx.mlx_array_free(q);
                q = out;
            };
            var clk = ProfClock.init();
            try mlx.check(mlx.mlx_array_eval(q));
            const ns = clk.lap();
            _ = mlx.mlx_array_free(q);
            if (rep > 0) us[ai][rep - 1] = @as(f64, @floatFromInt(ns)) / 1e3 / @as(f64, @floatFromInt(layers * chain));
        };
        for (arms, 0..) |arm, ai| {
            std.mem.sort(f64, &us[ai], {}, std.sort.asc(f64));
            std.debug.print("[qsa-attn-ubench] kv={d} simdgroups={d}: {d:.1} us/call (median of {d})\n", .{ kv, arm, us[ai][reps / 2], reps });
        }
    }
}

test "qsa sparse attn: one fused dispatch replaces the arm's op chain (>= 100 ops over 12 layers)" {
    // A dispatch-count bar: twelve calls stand in for qwen4's twelve full-attention layers.
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const ta = std.testing.allocator;
    const ratio: c_int = 4;
    const kv: c_int = 20000;
    const seq: c_int = 6;
    const kb: c_int = 128;
    var prng = std.Random.DefaultPrng.init(0x0F5_C0DE);
    const rnd = prng.random();

    const q = try attn256RandBf16(rnd, &[_]c_int{ 1, 24, seq, 256 }, s);
    defer _ = mlx.mlx_array_free(q);
    const kd = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, kv, 256 }, s);
    defer _ = mlx.mlx_array_free(kd);
    const vd = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, kv, 256 }, s);
    defer _ = mlx.mlx_array_free(vd);
    const bh = try qsaVerifyBlocksHost(ta, rnd, seq, kv, kb, ratio);
    defer ta.free(bh);
    const blocks = mlx.mlx_array_new_data(bh.ptr, &[_]c_int{ 1, seq, kb }, 3, .int32);
    defer _ = mlx.mlx_array_free(blocks);
    const scale: f32 = 1.0 / 16.0;

    var cache = try KVCache.initWithConfig(ta, 1, kv_quant.KVQuantConfig.affine(8));
    defer cache.deinit();
    var view = try qsaAttnCacheFixture(ta, s, kd, vd, kv, @divTrunc(kv, 2), kv_quant.KVQuantConfig.affine(8), &cache);
    defer view.deinit();

    // Warm both arms first: the first call builds its cached config.
    if (try qsaSparseAttn(s, q, &view, blocks, ratio, scale)) |w| _ = mlx.mlx_array_free(w);
    if (qsaVerifyGatherAttn(s, q, &view, blocks, ratio, scale) catch null) |w| _ = mlx.mlx_array_free(w);

    const before_k = mlx.op_count.load(.monotonic);
    for (0..12) |_| {
        const o = (try qsaSparseAttn(s, q, &view, blocks, ratio, scale)) orelse return error.SparseAttnDeclined;
        _ = mlx.mlx_array_free(o);
    }
    const kernel_ops = mlx.op_count.load(.monotonic) - before_k;

    const before_g = mlx.op_count.load(.monotonic);
    for (0..12) |_| {
        const o = (qsaVerifyGatherAttn(s, q, &view, blocks, ratio, scale) catch null) orelse return error.VerifyGatherDeclined;
        _ = mlx.mlx_array_free(o);
    }
    const gather_ops = mlx.op_count.load(.monotonic) - before_g;

    errdefer std.debug.print("[qsa-attn] 12 layers S={d} kv={d}: kernel {d} ops, union gather {d} ops (saved {d})\n", .{ seq, kv, kernel_ops, gather_ops, gather_ops -| kernel_ops });
    try std.testing.expect(gather_ops > kernel_ops);
    try std.testing.expect(gather_ops - kernel_ops >= 100);
}

test "qsa sparse attn: the width gate, the kill switch and the quant preconditions all decline cleanly" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const ta = std.testing.allocator;
    const ratio: c_int = 4;
    const kv: c_int = 4096;
    var prng = std.Random.DefaultPrng.init(0xDEC_11E);
    const rnd = prng.random();
    const kd = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, kv, 256 }, s);
    defer _ = mlx.mlx_array_free(kd);
    const vd = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, kv, 256 }, s);
    defer _ = mlx.mlx_array_free(vd);
    // The width gate and the kill switch are only observable on a cache the kernel would serve.
    const qcfg = kv_quant.KVQuantConfig.affine(8);
    var qcache = try KVCache.initWithConfig(ta, 1, qcfg);
    defer qcache.deinit();
    var dv = try qsaAttnCacheFixture(ta, s, kd, vd, kv, @divTrunc(kv, 2), qcfg, &qcache);
    defer dv.deinit();
    try std.testing.expect(dv.has_quant_triple);

    const H = struct {
        fn blocksFor(alloc: std.mem.Allocator, r: std.Random, seq: c_int, kvv: c_int, kb: c_int, rat: c_int) ![]i32 {
            return qsaVerifyBlocksHost(alloc, r, seq, kvv, kb, rat);
        }
    };

    {
        const q1 = try attn256RandBf16(rnd, &[_]c_int{ 1, 24, 1, 256 }, s);
        defer _ = mlx.mlx_array_free(q1);
        const b1 = try H.blocksFor(ta, rnd, 1, kv, 64, ratio);
        defer ta.free(b1);
        const bl1 = mlx.mlx_array_new_data(b1.ptr, &[_]c_int{ 1, 1, 64 }, 3, .int32);
        defer _ = mlx.mlx_array_free(bl1);
        const one_row = (try qsaSparseAttn(s, q1, &dv, bl1, ratio, 1.0)) orelse return error.SparseAttnDeclined;
        _ = mlx.mlx_array_free(one_row);
    }
    {
        const q16 = try attn256RandBf16(rnd, &[_]c_int{ 1, 24, FUSED256_MIN_Q_LEN, 256 }, s);
        defer _ = mlx.mlx_array_free(q16);
        const b16 = try H.blocksFor(ta, rnd, FUSED256_MIN_Q_LEN, kv, 64, ratio);
        defer ta.free(b16);
        const bl16 = mlx.mlx_array_new_data(b16.ptr, &[_]c_int{ 1, FUSED256_MIN_Q_LEN, 64 }, 3, .int32);
        defer _ = mlx.mlx_array_free(bl16);
        var dense_view = DenseKVView{ .k = kd, .v = vd, .owned = false };
        try std.testing.expect((try qsaSparseAttn(s, q16, &dense_view, bl16, ratio, 1.0)) == null);

        const wide: c_int = QSA_ATTN_PACKED_MAX_S + 1;
        const qw = try attn256RandBf16(rnd, &[_]c_int{ 1, 24, wide, 256 }, s);
        defer _ = mlx.mlx_array_free(qw);
        const bw = try H.blocksFor(ta, rnd, wide, kv, 64, ratio);
        defer ta.free(bw);
        const blw = mlx.mlx_array_new_data(bw.ptr, &[_]c_int{ 1, wide, 64 }, 3, .int32);
        defer _ = mlx.mlx_array_free(blw);
        try std.testing.expect((try qsaSparseAttn(s, qw, &dv, blw, ratio, 1.0)) == null);
    }
    {
        const q6 = try attn256RandBf16(rnd, &[_]c_int{ 1, 24, 6, 256 }, s);
        defer _ = mlx.mlx_array_free(q6);
        const b6 = try H.blocksFor(ta, rnd, 6, kv, 64, ratio);
        defer ta.free(b6);
        const bl6 = mlx.mlx_array_new_data(b6.ptr, &[_]c_int{ 1, 6, 64 }, 3, .int32);
        defer _ = mlx.mlx_array_free(bl6);
        const ok = (try qsaSparseAttn(s, q6, &dv, bl6, ratio, 1.0)) orelse return error.SparseAttnDeclined;
        _ = mlx.mlx_array_free(ok);
        {
            qsa_attn_kernel_override = false;
            defer qsa_attn_kernel_override = null;
            try std.testing.expect((try qsaSparseAttn(s, q6, &dv, bl6, ratio, 1.0)) == null);
        }
        var dense_view = DenseKVView{ .k = kd, .v = vd, .owned = false };
        try std.testing.expect(!dense_view.has_quant_triple);
        const dense_out = (try qsaSparseAttn(s, q6, &dense_view, bl6, ratio, 1.0)) orelse return error.SparseAttnDeclined;
        _ = mlx.mlx_array_free(dense_out);
    }
    {
        const big_kv: c_int = QSA_VERIFY_GATHER_MIN_KV_DENSE + 4000;
        const kbig = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, big_kv, 256 }, s);
        defer _ = mlx.mlx_array_free(kbig);
        const vbig = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, big_kv, 256 }, s);
        defer _ = mlx.mlx_array_free(vbig);
        const q4 = try attn256RandBf16(rnd, &[_]c_int{ 1, 24, 4, 256 }, s);
        defer _ = mlx.mlx_array_free(q4);
        const b4 = try H.blocksFor(ta, rnd, 4, big_kv, 128, ratio);
        defer ta.free(b4);
        const bl4 = mlx.mlx_array_new_data(b4.ptr, &[_]c_int{ 1, 4, 128 }, 3, .int32);
        defer _ = mlx.mlx_array_free(bl4);
        var big_dense = DenseKVView{ .k = kbig, .v = vbig, .owned = false };
        const sparse_out = (try qsaSparseAttn(s, q4, &big_dense, bl4, ratio, 1.0)) orelse return error.SparseAttnDeclined;
        _ = mlx.mlx_array_free(sparse_out);
        const via_gather = (try qsaVerifyGatherAttn(s, q4, &big_dense, bl4, ratio, 1.0)) orelse return error.VerifyGatherDeclined;
        _ = mlx.mlx_array_free(via_gather);
    }
}

test "qsa sparse attn: the config key is kv- and kb-INDEPENDENT (one build per width)" {
    // Nothing in the config reads kv or kb (the kernel takes them at runtime), and kv grows
    // every decode round: same width, any kv, any kb => exactly one build.
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const ta = std.testing.allocator;
    const ratio: c_int = 4;
    var prng = std.Random.DefaultPrng.init(0xC0F_9E15);
    const rnd = prng.random();

    const Run = struct {
        fn once(st: mlx.mlx_stream, alloc: std.mem.Allocator, r: std.Random, seq: c_int, kv: c_int, kb: c_int) !void {
            const q = try attn256RandBf16(r, &[_]c_int{ 1, 24, seq, 256 }, st);
            defer _ = mlx.mlx_array_free(q);
            const kd = try attn256RandBf16(r, &[_]c_int{ 1, 2, kv, 256 }, st);
            defer _ = mlx.mlx_array_free(kd);
            const vd = try attn256RandBf16(r, &[_]c_int{ 1, 2, kv, 256 }, st);
            defer _ = mlx.mlx_array_free(vd);
            const bh = try qsaVerifyBlocksHost(alloc, r, seq, kv, kb, 4);
            defer alloc.free(bh);
            const bl = mlx.mlx_array_new_data(bh.ptr, &[_]c_int{ 1, seq, kb }, 3, .int32);
            defer _ = mlx.mlx_array_free(bl);
            var cache = try KVCache.initWithConfig(alloc, 1, kv_quant.KVQuantConfig.affine(8));
            defer cache.deinit();
            var view = try qsaAttnCacheFixture(alloc, st, kd, vd, kv, @divTrunc(kv, 2), kv_quant.KVQuantConfig.affine(8), &cache);
            defer view.deinit();
            const o = (try qsaSparseAttn(st, q, &view, bl, 4, 1.0)) orelse return error.SparseAttnDeclined;
            defer _ = mlx.mlx_array_free(o);
            try mlx.check(mlx.mlx_array_eval(o));
        }
    };
    _ = ratio;

    // Normalize first: one global cache shared with every other test in the file.
    try Run.once(s, ta, rnd, 6, 4096, 64);
    const base = qsaAttnCfgBuilds();
    for (0..3) |_| try Run.once(s, ta, rnd, 6, 4096, 64);
    try std.testing.expectEqual(base, qsaAttnCfgBuilds());
    // A growing kv at a fixed width is the served decode round.
    try Run.once(s, ta, rnd, 6, 8192, 64);
    try Run.once(s, ta, rnd, 6, 12288, 64);
    try Run.once(s, ta, rnd, 6, 20000, 64);
    try std.testing.expectEqual(base, qsaAttnCfgBuilds());
    try Run.once(s, ta, rnd, 6, 20000, 128);
    try std.testing.expectEqual(base, qsaAttnCfgBuilds());
    // A new width is a new grid, new output shapes and a new NSPLIT.
    try Run.once(s, ta, rnd, 7, 4096, 64);
    try std.testing.expectEqual(base + 1, qsaAttnCfgBuilds());
    // Coming back to a width already built is free (the LRU).
    try Run.once(s, ta, rnd, 6, 4096, 64);
    try std.testing.expectEqual(base + 1, qsaAttnCfgBuilds());
    try Run.once(s, ta, rnd, 3, 4096, 64);
    const after_three = qsaAttnCfgBuilds();
    for (0..3) |_| {
        try Run.once(s, ta, rnd, 6, 4096, 64);
        try Run.once(s, ta, rnd, 7, 4096, 64);
        try Run.once(s, ta, rnd, 3, 4096, 64);
    }
    try std.testing.expectEqual(after_three, qsaAttnCfgBuilds());
    // Past SLOTS distinct widths the LRU evicts.
    try std.testing.expect(@TypeOf(trf.qsa_attn_cfgs).SLOTS == 4);
}

test "takeContig releases `view` exactly once on every faulted op (CPU stream, no GPU)" {
    // Driven with the fault injector at `mlx.check`: `arm(k)` fails the k-th checked call, so
    // the sweep walks every error path. On the pre-fix shape k = 2 double-frees and aborts.
    const s = mlx.mlx_default_cpu_stream_new();
    const rows = 8;
    const cols = 4;
    var host: [rows * cols]f32 = undefined;
    for (&host, 0..) |*v, n| v.* = @floatFromInt(n);
    const shape = [_]c_int{ rows, cols };
    const srcarr = mlx.mlx_array_new_data(&host, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(srcarr);
    var idx_host = [_]u32{ 5, 0, 3, 3 };
    const ish = [_]c_int{idx_host.len};
    const idx = mlx.mlx_array_new_data(&idx_host, &ish, 1, .uint32);
    defer _ = mlx.mlx_array_free(idx);

    const c0 = mlx.op_count.load(.monotonic);
    const warm = try takeContig(s, srcarr, idx, 0);
    const n_ops = mlx.op_count.load(.monotonic) - c0;
    try mlx.check(mlx.mlx_array_eval(warm));
    try testing.expectEqual(@as(c_int, idx_host.len), mlx.getShape(warm)[0]);
    _ = mlx.mlx_array_free(warm);
    try testing.expectEqual(@as(u64, 2), n_ops);

    var fired: usize = 0;
    var k: u64 = 1;
    while (k <= n_ops) : (k += 1) {
        mlx.fault.arm(k);
        const r = takeContig(s, srcarr, idx, 0);
        const did = mlx.fault.didFire();
        mlx.fault.disarm();
        if (r) |ok| {
            _ = mlx.mlx_array_free(ok);
        } else |err| {
            try testing.expectEqual(error.MlxError, err);
            try testing.expect(did);
            fired += 1;
        }
    }
    try testing.expectEqual(@as(usize, 2), fired);
}

test "qsa score sheet budget: the env parser, and the row chunk it buys at nb 95k" {
    const t = std.testing;
    qwen4_qsa_select.qsa_score_fused_override = false;
    defer qwen4_qsa_select.qsa_score_fused_override = null;
    try t.expectEqual(QSA_SCORE_SHEET_BUDGET_MB_DEFAULT, qsaScoreSheetMbFrom(null));
    try t.expectEqual(QSA_SCORE_SHEET_BUDGET_MB_DEFAULT, qsaScoreSheetMbFrom(""));
    try t.expectEqual(QSA_SCORE_SHEET_BUDGET_MB_DEFAULT, qsaScoreSheetMbFrom("wide"));
    try t.expectEqual(QSA_SCORE_SHEET_BUDGET_MB_DEFAULT, qsaScoreSheetMbFrom("0"));
    try t.expectEqual(@as(u64, 1024), qsaScoreSheetMbFrom("1024"));
    try t.expectEqual(QSA_SCORE_SHEET_BUDGET_MB_MAX, qsaScoreSheetMbFrom("999999"));

    // Flash-Next at the 374k rung: nb 95801, n_idx 4, chunk 4096. At 256 MB the sheet fits
    // ~175 rows, so one chunk is 24 indexer passes; at 1024 MB it is 6.
    const n_idx: u64 = 4;
    const nb: u64 = 383205 / 4;
    const chunk: u64 = 4096;
    const prev = qsa_score_sheet_mb_override;
    defer qsa_score_sheet_mb_override = prev;

    qsa_score_sheet_mb_override = 256;
    const rows_256 = qsaScoreRowsPerChunk(n_idx, nb, chunk, 128);
    qsa_score_sheet_mb_override = 1024;
    const rows_1024 = qsaScoreRowsPerChunk(n_idx, nb, chunk, 128);

    try t.expectEqual(@as(u64, (256 << 20) / (4 * nb * 4)), rows_256);
    try t.expectEqual(rows_256 * 4, rows_1024);
    try t.expect(rows_256 < chunk and rows_1024 < chunk); // both still row-chunked
    const passes = struct {
        fn f(rows: u64, c: u64) u64 {
            return (c + rows - 1) / rows;
        }
    }.f;
    try t.expectEqual(@as(u64, 24), passes(rows_256, chunk));
    try t.expectEqual(@as(u64, 6), passes(rows_1024, chunk));

    // A wider sheet must be billed: the admission guard reads the same helper.
    qsa_score_sheet_mb_override = 256;
    const bill_256 = qsaPrefillTransientBytes(n_idx, chunk, 383205, 4, 128);
    qsa_score_sheet_mb_override = 1024;
    const bill_1024 = qsaPrefillTransientBytes(n_idx, chunk, 383205, 4, 128);
    try t.expect(bill_1024 > bill_256);
    try t.expectEqual((rows_1024 - rows_256) * nb * 4 * (n_idx * 2 + 4), bill_1024 - bill_256);

    qsa_score_sheet_mb_override = 4096;
    try t.expectEqual(@as(u64, 64), qsaScoreRowsPerChunk(n_idx, 1024, 64, 128));
    try t.expectEqual(@as(u64, 16), qsaScoreRowsPerChunk(n_idx, 1 << 30, 4096, 128));
}

test "qsa score fused: unsupported indexer geometry falls back to the composed chain" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var xfm: Transformer = undefined;
    xfm.s = s;
    xfm.allocator = testing.allocator;
    xfm.qsa_score_bank_builds = 0;
    const saved_fused = qwen4_qsa_select.qsa_score_fused_override;
    const saved_nax = trf.vqmm_nax_probe_override;
    const saved_sheet = qsa_score_sheet_mb_override;
    defer {
        qwen4_qsa_select.qsa_score_fused_override = saved_fused;
        trf.vqmm_nax_probe_override = saved_nax;
        qsa_score_sheet_mb_override = saved_sheet;
    }
    qsa_score_sheet_mb_override = QSA_SCORE_SHEET_BUDGET_MB_DEFAULT;
    qwen4_qsa_select.qsa_score_fused_override = true;

    const fwd: u64 = 4096;
    const kv: u64 = 262144 * 4;
    const ratio: u64 = 4;
    qwen4_qsa_select.qsa_score_fused_override = false;
    const composed_n2 = qsaPrefillTransientBytes(2, fwd, kv, ratio, 128);
    const composed_hd64 = qsaPrefillTransientBytes(4, fwd, kv, ratio, 64);
    const composed_rows_n2 = qsaScoreRowsPerChunk(2, kv / ratio, fwd, 128);
    const composed_rows_hd64 = qsaScoreRowsPerChunk(4, kv / ratio, fwd, 64);
    qwen4_qsa_select.qsa_score_fused_override = true;
    try testing.expect(!qsaScoreFusedActiveFor(1, 2, 128));
    try testing.expect(!qsaScoreFusedActiveFor(1, 4, 64));
    try testing.expectEqual(composed_n2, qsaPrefillTransientBytes(2, fwd, kv, ratio, 128));
    try testing.expectEqual(composed_hd64, qsaPrefillTransientBytes(4, fwd, kv, ratio, 64));
    try testing.expectEqual(composed_rows_n2, qsaScoreRowsPerChunk(2, kv / ratio, fwd, 128));
    try testing.expectEqual(composed_rows_hd64, qsaScoreRowsPerChunk(4, kv / ratio, fwd, 64));

    var cfg2 = model_mod.ModelConfig{};
    cfg2.num_hidden_layers = 48;
    cfg2.full_attention_interval = 4;
    cfg2.indexer_budget = 2048;
    cfg2.indexer_n_heads = 2;
    cfg2.indexer_head_dim = 128;
    cfg2.indexer_compress_ratio = 4;
    const bank2 = cfg2.qsaScoreBankBytesPerToken();
    try testing.expect(bank2 > 0);
    var cfg64 = cfg2;
    cfg64.indexer_n_heads = 4;
    cfg64.indexer_head_dim = 64;
    try testing.expect(cfg64.qsaScoreBankBytesPerToken() > 0);

    var prng = std.Random.DefaultPrng.init(0x6E01);
    const rnd = prng.random();
    const nb: c_int = 32;
    const rows: c_int = 4;
    {
        var entry: SSMCacheEntry = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true };
        defer ssmFreeQsaState(&entry);
        entry.qsa_ratio = 4;
        entry.qsa_reserve_rows = 256;
        const pooled = try attn256RandBf16(rnd, &[_]c_int{ 1, nb, 128 }, s);
        defer _ = mlx.mlx_array_free(pooled);
        try xfm.qsaAppendPooled(&entry, pooled, 0);
        const fused = qsaScoreFusedActiveFor(1, 2, 128);
        try testing.expect(!fused);
        _ = try xfm.qsaScoreOperand(&entry, 1, nb, 128, fused);
        try testing.expect(entry.qsa_score_bank.ctx != null);
        const q = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, rows, 128 }, s);
        defer _ = mlx.mlx_array_free(q);
        const sheet = try qsaScoreSheet(s, q, pooled, entry.qsa_score_bank, fused);
        defer _ = mlx.mlx_array_free(sheet);
        const want = try qsaScoreSheetComposed(s, q, entry.qsa_score_bank);
        defer _ = mlx.mlx_array_free(want);
        try testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(sheet, want, s));
    }
    {
        var entry: SSMCacheEntry = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true };
        defer ssmFreeQsaState(&entry);
        entry.qsa_ratio = 4;
        entry.qsa_reserve_rows = 256;
        const pooled = try attn256RandBf16(rnd, &[_]c_int{ 1, nb, 64 }, s);
        defer _ = mlx.mlx_array_free(pooled);
        try xfm.qsaAppendPooled(&entry, pooled, 0);
        const fused = qsaScoreFusedActiveFor(1, 4, 64);
        try testing.expect(!fused);
        _ = try xfm.qsaScoreOperand(&entry, 1, nb, 64, fused);
        try testing.expect(entry.qsa_score_bank.ctx != null);
        const q = try attn256RandBf16(rnd, &[_]c_int{ 1, 4, rows, 64 }, s);
        defer _ = mlx.mlx_array_free(q);
        const sheet = try qsaScoreSheet(s, q, pooled, entry.qsa_score_bank, fused);
        defer _ = mlx.mlx_array_free(sheet);
        const want = try qsaScoreSheetComposed(s, q, entry.qsa_score_bank);
        defer _ = mlx.mlx_array_free(want);
        try testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(sheet, want, s));
    }
    {
        var entry: SSMCacheEntry = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true };
        defer ssmFreeQsaState(&entry);
        entry.qsa_ratio = 4;
        entry.qsa_reserve_rows = 256;
        const pooled = try attn256RandBf16(rnd, &[_]c_int{ 1, nb, 128 }, s);
        defer _ = mlx.mlx_array_free(pooled);
        try xfm.qsaAppendPooled(&entry, pooled, 0);
        try mlx.check(mlx.mlx_array_eval(pooled));
        const q_bf = try attn256RandBf16(rnd, &[_]c_int{ 1, 4, rows, 128 }, s);
        defer _ = mlx.mlx_array_free(q_bf);
        var q32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q32);
        try mlx.check(mlx.mlx_astype(&q32, q_bf, .float32, s));
        const fused = qsaScoreFusedEligibleFrom(q32, entry.qsa_pooled) and
            qsaPooledRowContiguous(entry.qsa_pooled) and mlx.streamIsGpu(s);
        try testing.expect(!fused);
        _ = try xfm.qsaScoreOperand(&entry, 1, nb, 128, fused);
        try testing.expect(entry.qsa_score_bank.ctx != null);
        const sheet = try qsaScoreSheet(s, q32, pooled, entry.qsa_score_bank, fused);
        defer _ = mlx.mlx_array_free(sheet);
        const want = try qsaScoreSheetComposed(s, q32, entry.qsa_score_bank);
        defer _ = mlx.mlx_array_free(want);
        try testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(sheet, want, s));
    }
}

test "QSA verify rows are bit-identical to decode at both KV tile sizes" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const a = testing.allocator;
    const old_kernel = qsa_attn_kernel_override;
    const old_decode = qwen4_qsa_gather.qsa_decode_gather_override;
    const old_verify = qsa_verify_gather_override;
    const old_bk = qsa_attn_bk_override;
    const old_split = qsa_attn_nsplit_override;
    qsa_attn_kernel_override = true;
    qwen4_qsa_gather.qsa_decode_gather_override = true;
    qsa_verify_gather_override = true;
    qsa_attn_nsplit_override = 16;
    defer qsa_attn_kernel_override = old_kernel;
    defer qwen4_qsa_gather.qsa_decode_gather_override = old_decode;
    defer qsa_verify_gather_override = old_verify;
    defer qsa_attn_bk_override = old_bk;
    defer qsa_attn_nsplit_override = old_split;
    const length: c_int = 40003;
    const kb: c_int = 512;
    const kd = try qkvIdentityArray(s, 2, length, 256, 0.1, .bfloat16);
    defer _ = mlx.mlx_array_free(kd);
    const vd = try qkvIdentityArray(s, 2, length, 256, 3.7, .bfloat16);
    defer _ = mlx.mlx_array_free(vd);
    var random = std.Random.DefaultPrng.init(0x05a51);
    for ([_]KVQuantConfig{ KVQuantConfig.dense, KVQuantConfig.affine(8), .{ .scheme = .affine, .bits = 8, .group_size = 32 } }) |config| {
        var cache = try KVCache.initWithConfig(a, 1, config);
        defer cache.deinit();
        var view = try qsaAttnCacheFixture(a, s, kd, vd, length, 20000, config, &cache);
        defer view.deinit();
        for ([_]c_int{ 16, 32 }) |bk| {
            qsa_attn_bk_override = bk;
            for (2..9) |wi| {
                const width: c_int = @intCast(wi);
                const q = try qkvIdentityArray(s, 24, width, 256, 1.3, .bfloat16);
                defer _ = mlx.mlx_array_free(q);
                const block_ids = try qsaVerifyBlocksHost(a, random.random(), width, length, kb, 4);
                defer a.free(block_ids);
                const blocks = mlx.mlx_array_new_data(block_ids.ptr, &[_]c_int{ 1, width, kb }, 3, .int32);
                defer _ = mlx.mlx_array_free(blocks);
                const together = try qsaIdentityServingArm(s, q, &view, blocks);
                defer _ = mlx.mlx_array_free(together);
                for (0..wi) |ri| {
                    const row: c_int = @intCast(ri);
                    var prefix = try KvPrefixView.init(s, view, length - width + row + 1);
                    defer prefix.deinit();
                    const qr = try sliceAttentionSeq(s, q, row, row + 1);
                    defer _ = mlx.mlx_array_free(qr);
                    var br = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(br);
                    try mlx.check(mlx.mlx_slice(&br, blocks, &[_]c_int{ 0, row, 0 }, 3, &[_]c_int{ 1, row + 1, kb }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
                    const serial = try qsaIdentityServingArm(s, qr, &prefix.view, br);
                    defer _ = mlx.mlx_array_free(serial);
                    const actual = try sliceAttentionSeq(s, together, row, row + 1);
                    defer _ = mlx.mlx_array_free(actual);
                    qkvExpectBitsEqual(s, serial, actual) catch |err| {
                        std.debug.print("QSA kv={d} gs={d} BK={d} width={d} row={d}\n", .{ config.bits, config.group_size, bk, width, row });
                        return err;
                    };
                }
            }
        }
    }
}

test "QSA verify preserves serial arithmetic across the sparse budget boundary" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const a = testing.allocator;
    const budget: c_int = 64;
    const ratio: c_int = 4;
    const width: c_int = 8;
    const length: c_int = 71;
    var random = std.Random.DefaultPrng.init(0xb0d9e7);
    for ([_]KVQuantConfig{ KVQuantConfig.dense, KVQuantConfig.affine(8) }) |config| {
        var cache = try KVCache.initWithConfig(a, 1, config);
        defer cache.deinit();
        const kd = try qkvIdentityArray(s, 2, length, 256, 0.7, .bfloat16);
        defer _ = mlx.mlx_array_free(kd);
        const vd = try qkvIdentityArray(s, 2, length, 256, 1.7, .bfloat16);
        defer _ = mlx.mlx_array_free(vd);
        var view = try qsaAttnCacheFixture(a, s, kd, vd, length, 32, config, &cache);
        defer view.deinit();
        const q = try qkvIdentityArray(s, 24, width, 256, 0.31, .bfloat16);
        defer _ = mlx.mlx_array_free(q);
        const bh = try qsaVerifyBlocksHost(a, random.random(), width, length, budget / ratio, ratio);
        defer a.free(bh);
        const blocks = mlx.mlx_array_new_data(bh.ptr, &[_]c_int{ 1, width, budget / ratio }, 3, .int32);
        defer _ = mlx.mlx_array_free(blocks);
        const together = (try qsaAlignedSparseAttn(s, q, &view, blocks, ratio, 1.0 / 16.0, budget)) orelse return error.QsaVerifyDeclined;
        defer _ = mlx.mlx_array_free(together);
        for (0..@intCast(width)) |ri| {
            const row: c_int = @intCast(ri);
            const length_r = length - width + row + 1;
            var prefix = try KvPrefixView.init(s, view, length_r);
            defer prefix.deinit();
            const qr = try sliceAttentionSeq(s, q, row, row + 1);
            defer _ = mlx.mlx_array_free(qr);
            var br = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(br);
            try mlx.check(mlx.mlx_slice(&br, blocks, &[_]c_int{ 0, row, 0 }, 3, &[_]c_int{ 1, row + 1, budget / ratio }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
            var serial = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(serial);
            if (length_r <= budget + ratio - 1) {
                try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&serial, qr, prefix.view.k, prefix.view.v, 1.0 / 16.0, "", .{ .ctx = null }, .{ .ctx = null }, false, s));
            } else {
                const result = try qsaIdentityServingArm(s, qr, &prefix.view, br);
                _ = mlx.mlx_array_free(serial);
                serial = result;
            }
            const actual = try sliceAttentionSeq(s, together, row, row + 1);
            defer _ = mlx.mlx_array_free(actual);
            try qkvExpectBitsEqual(s, serial, actual);
        }
    }
}
