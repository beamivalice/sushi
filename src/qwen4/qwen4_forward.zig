//! Flash-Next (qwen4_exp) forward: attention, GDN wiring, native MTP head, decode/verify rows.

const trf = @import("../transformer.zig");
const qwen4_gdn = @import("qwen4_gdn.zig");
const Transformer = trf.Transformer;
const sliceBatchRows = Transformer.sliceBatchRows;
const std = @import("std");
const sushi_exl3 = @import("sushi_exl3");
const expert_exl3_kernels = sushi_exl3.kernels;
const mlx = @import("../mlx.zig");
const mrope = @import("../mrope.zig");
const kv_quant = @import("../kv_quant.zig");
pub const KVQuantConfig = kv_quant.KVQuantConfig;
const qwen4_qsa = @import("qwen4_qsa.zig");
const qwen4_hc = @import("qwen4_hc.zig");
const qwen4_mod = @import("qwen4_exp.zig");
const expert_stream_mod = @import("../expert_stream.zig");
const imatrix_capture = @import("../imatrix.zig");
const expert_exl3 = sushi_exl3.format;
const expert_quant_mod = @import("../expert_quant.zig");
const model_mod = @import("../model.zig");
const testing = std.testing;
const qwen4_test_support = @import("qwen4_test_support.zig");
const Qwen4TestSlot = qwen4_test_support.Qwen4TestSlot;
const attn256QsaMask = qwen4_test_support.attn256QsaMask;
const expectRowsByteEqual = qwen4_test_support.expectRowsByteEqual;
const expectSlotStateEqual = qwen4_test_support.expectSlotStateEqual;
const qwen4BatchedPleForward = qwen4_test_support.qwen4BatchedPleForward;
const qwen4ExpectMoeBitsEqual = qwen4_test_support.qwen4ExpectMoeBitsEqual;
const test_support = @import("../test_support.zig");
const Exl3MoeHarness = test_support.Exl3MoeHarness;
const attn256MaxDiff = test_support.attn256MaxDiff;
const attn256RandBf16 = test_support.attn256RandBf16;
const attn256RandBf16Scaled = test_support.attn256RandBf16Scaled;
const attn256Reference = test_support.attn256Reference;
const qkvExpectBitsEqual = test_support.qkvExpectBitsEqual;
const qkvIdentityArray = test_support.qkvIdentityArray;
const qwen4ReadF32 = test_support.qwen4ReadF32;

const DenseKVView = trf.DenseKVView;
const DenseMlpWeights = trf.DenseMlpWeights;
const FUSED256_MIN_Q_LEN = trf.FUSED256_MIN_Q_LEN;
const ForwardCtx = trf.ForwardCtx;
const FullAttnWeights = trf.FullAttnWeights;
const HC_FUSED_D_SOURCE = qwen4_hc.HC_FUSED_D_SOURCE;
const HC_FUSED_N_SOURCE = qwen4_hc.HC_FUSED_N_SOURCE;
const HC_FUSED_U_SOURCE = qwen4_hc.HC_FUSED_U_SOURCE;
const HcPending = qwen4_hc.HcPending;
const HcWeights = trf.HcWeights;
const KVCache = trf.KVCache;
const KVCacheSnapshot = trf.KVCacheSnapshot;
const LinearAttnWeights = trf.LinearAttnWeights;
const MAX_BATCH_ROWS = trf.MAX_BATCH_ROWS;
const ModelConfig = trf.ModelConfig;
const MoeDecodeDispatchArm = trf.MoeDecodeDispatchArm;
const MoeLayerWeights = trf.MoeLayerWeights;
const MoeMlpWeights = trf.MoeMlpWeights;
const PleWeights = trf.PleWeights;
const ProfClock = trf.ProfClock;
const QSA_SELECT_KERNEL_HEADER = qwen4_qsa.QSA_SELECT_KERNEL_HEADER;
const QSA_SELECT_KERNEL_SOURCE = qwen4_qsa.QSA_SELECT_KERNEL_SOURCE;
const QSA_SELECT_SPLIT_LOCAL_SOURCE = qwen4_qsa.QSA_SELECT_SPLIT_LOCAL_SOURCE;
const QSA_SELECT_SPLIT_MERGE_SOURCE = qwen4_qsa.QSA_SELECT_SPLIT_MERGE_SOURCE;
const QsaHeadMark = qwen4_qsa.QsaHeadMark;
const QsaHeadMarkSet = qwen4_qsa.QsaHeadMarkSet;
const SSMCacheEntry = trf.SSMCacheEntry;
const SSMCacheEntrySnapshot = trf.SSMCacheEntrySnapshot;
const SingleVerifyFeature = trf.SingleVerifyFeature;
const SsmMergeKind = trf.SsmMergeKind;
const VerifyProjectionKind = trf.VerifyProjectionKind;
const Weights = trf.Weights;
const appendFullAttnWeights = trf.appendFullAttnWeights;
const appendHybridMlpWeights = trf.appendHybridMlpWeights;
const appendLinearAttnWeights = trf.appendLinearAttnWeights;
const appendStructArrays = trf.appendStructArrays;
const bf16Scalar = trf.bf16Scalar;
const capturePrefillHidden = trf.capturePrefillHidden;
const capturePrefillHiddenLast = trf.capturePrefillHiddenLast;
const constTableAs = trf.constTableAs;
const decodeProfileEnabled = trf.decodeProfileEnabled;
const decodeRowsProjection = trf.decodeRowsProjection;
const diagEnvOn = trf.diagEnvOn;
const diagEnvOnCached = trf.diagEnvOnCached;
const fusedSdpa256Masked = trf.fusedSdpa256Masked;
const fusedSdpaPrefill = trf.fusedSdpaPrefill;
const gatherQsa256 = trf.gatherQsa256;
const gatherQsa256Packed = trf.gatherQsa256Packed;
const gdnGateChain = trf.gdnGateChain;
const initMoeLayers = trf.initMoeLayers;
const layerCap = trf.layerCap;
const loadHcWeights = trf.loadHcWeights;
const log = trf.log;
const materializedOwnedCopy = trf.materializedOwnedCopy;
const moeDecodeDispatchArm = trf.moeDecodeDispatchArm;
const moeDumpBeginForward = trf.moeDumpBeginForward;
const moeDumpForwardDone = trf.moeDumpForwardDone;
const moeRouterFusedEnabled = trf.moeRouterFusedEnabled;
const mtp_mod = trf.mtp_mod;
const naxSdpaPreferred = trf.naxSdpaPreferred;
const scalarOf = trf.scalarOf;
const splitSpecCaptures = trf.splitSpecCaptures;
const ssmBindSlotRefs = trf.ssmBindSlotRefs;
const ssmBindTickState = trf.ssmBindTickState;
const ssmFreeMergedLayer = trf.ssmFreeMergedLayer;
const ssmFreeSpecCapture = trf.ssmFreeSpecCapture;
const ssmSplitAll = trf.ssmSplitAll;
const noteQsaArmForSlots = qwen4_qsa.noteQsaArmForSlots;
const qsaAlignedSparseAttn = qwen4_qsa.qsaAlignedSparseAttn;
const qsaDecodeGatherAttn = trf.qsaDecodeGatherAttn;
const qsaHistoryRows = trf.qsaHistoryRows;
const qsaLeftoverAt = trf.qsaLeftoverAt;
const qsaMaskFromBlocks = trf.qsaMaskFromBlocks;
const qsaPackedGatherServes = qwen4_qsa.qsaPackedGatherServes;
const qsaResliceToKeep = trf.qsaResliceToKeep;
const qsaSparseAttnServes = trf.qsaSparseAttnServes;
const qsaVerifyGatherAttn = trf.qsaVerifyGatherAttn;
const ropeAngleRows = trf.ropeAngleRows;
const sliceAttentionSeq = trf.sliceAttentionSeq;
const ssmFreeQsaState = trf.ssmFreeQsaState;
const ssmRestore = trf.ssmRestore;
const ssmSnapshotDeinit = trf.ssmSnapshotDeinit;
const standinRef = trf.standinRef;
const truncatePooled = trf.truncatePooled;
const verifyJoinedProjection = trf.verifyJoinedProjection;
const weightTriple = trf.weightTriple;



pub fn getQsaSelectKernel() !mlx.mlx_fast_metal_kernel {
    if (qwen4_qsa.qsa_select_kernel_cached) |kk| return kk;
    const input_names = [_][*:0]const u8{ "scores", "bounds" };
    const output_names = [_][*:0]const u8{"ids"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "sushi_qsa_select",
        in_vec,
        out_vec,
        QSA_SELECT_KERNEL_SOURCE,
        QSA_SELECT_KERNEL_HEADER,
        true, // the row walk indexes linearly; let mlx materialize a view
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    qwen4_qsa.qsa_select_kernel_cached = kernel;
    return kernel;
}

pub var qwen4_mtp_head_graphs: usize = 0;

pub var mtp_verify_moe_group_graphs: u64 = 0;

pub var mtp_verify_moe_group_last_ops: u64 = 0;

pub var mtp_head_force_batched_override: ?bool = null;

pub var mtp_verify_shared_rows_calls: u64 = 0;

pub var mtp_verify_shared_gate_rows_calls: u64 = 0;

pub var mtp_verify_gdn_rows_calls: u64 = 0;

pub var mtp_verify_gdn_output_calls: u64 = 0;

pub var mtp_verify_attn_rows_calls: u64 = 0;

pub var mtp_verify_attn_output_calls: u64 = 0;

pub var mtp_verify_lmhead_rows_calls: u64 = 0;

pub fn sharedGateRowsEligible(widths: []const c_int) bool {
    if (widths.len < 2 or widths.len > 8) return false;
    var total: c_int = 0;
    for (widths) |width| {
        if (width < 1 or width > 7) return false;
        total += width;
    }
    return total <= 24;
}

pub var mtp_verify_hc_prepared_active = false;

pub var mtp_verify_hc_prepared_calls: u64 = 0;

pub fn mtpHeadRowsFit(widths: []const c_int, hc: u32) bool {
    if (widths.len == 0 or widths.len > 8 or hc == 0) return false;
    if (@as(u64, hc) * widths.len > 16) return false;
    for (widths) |width| if (width != 1) return false;
    return true;
}

pub fn getQsaSelectSplitLocalKernel() !mlx.mlx_fast_metal_kernel {
    if (qwen4_qsa.qsa_select_split_local_kernel_cached) |kk| return kk;
    const input_names = [_][*:0]const u8{ "scores", "bounds" };
    const output_names = [_][*:0]const u8{"ids"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "sushi_qsa_select_split_local",
        in_vec,
        out_vec,
        QSA_SELECT_SPLIT_LOCAL_SOURCE,
        QSA_SELECT_KERNEL_HEADER,
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    qwen4_qsa.qsa_select_split_local_kernel_cached = kernel;
    return kernel;
}

pub fn getQsaSelectSplitMergeKernel() !mlx.mlx_fast_metal_kernel {
    if (qwen4_qsa.qsa_select_split_merge_kernel_cached) |kk| return kk;
    const input_names = [_][*:0]const u8{ "scores", "local_ids" };
    const output_names = [_][*:0]const u8{"ids"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "sushi_qsa_select_split_merge",
        in_vec,
        out_vec,
        QSA_SELECT_SPLIT_MERGE_SOURCE,
        QSA_SELECT_KERNEL_HEADER,
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    qwen4_qsa.qsa_select_split_merge_kernel_cached = kernel;
    return kernel;
}

pub var qwen4_rows_logged: bool = false;

pub var verify_fault_after_ple_row: ?usize = null;

pub fn markPosOf(m: *const QsaHeadMark) usize {
    return @intCast(@max(m.pos, 0));
}

/// qwen4_exp MTP head (vLLM #53896 / SGLang #36497 `residual_linear_shared`):
///   x = fc_hidden(rms_{4H}(stream)) per stream + fc_embedding(rms_H(embed))
///   → one full-attention decoder layer (own hc/QSA/MoE) → mixer → lm_head.
/// Keeps its own KV cache (index `num_hidden_layers`) + indexer key history.
pub const Qwen4Mtp = struct {
    layer: MoeLayerWeights,
    pre_norm_emb: mlx.mlx_array,
    pre_norm_hidden: mlx.mlx_array,
    fc_emb_w: mlx.mlx_array,
    fc_emb_s: mlx.mlx_array,
    fc_emb_b: mlx.mlx_array,
    fc_hid_w: mlx.mlx_array,
    fc_hid_s: mlx.mlx_array,
    fc_hid_b: mlx.mlx_array,
    mixer: HcWeights,
    owned: []mlx.mlx_array,
    cache: KVCache,
    entry: SSMCacheEntry,
    seq_offset: usize = 0,
    /// The head's QSA leftovers at the trunk's checkpoint positions
    /// (`qwen4MtpMarkQsaLeftover`): what a warm clamp reads once the ring has
    /// moved past them. Per REQUEST — `qwen4MtpReset` clears it and an adopt
    /// replaces it with the committed set.
    qsa_marks: QsaHeadMarkSet = .{},
    /// Absolute position of the head's key row 0 (the first draft position
    /// after a reset).
    pos_base: c_int = 0,
    /// Cross-request EV controller seed — the in-checkpoint head's twin of
    /// MiMo's `Head.ev_seed_*`: the last HEALTHY request's per-index
    /// acceptance EMAs and base depth, written by `Generator.deinit` and read
    /// by the first `nextMtp` round of the next request. Without it every
    /// request re-warms the controller (MTP_EV_WARMUP_ROUNDS legacy rounds
    /// plus a +1/round base climb), which is a sizeable share of a short
    /// generation. Per-LOADED-MODEL state, exactly like the sidecar's: it
    /// lives on the head, so it dies with the head and can never reach
    /// another model. `qwen4MtpReset` deliberately does not clear it — a
    /// reset starts a new REQUEST, which is precisely who should inherit it.
    ev_seed_accept: ?[mtp_mod.MAX_DEPTH]f32 = null,
    ev_seed_m_lo: u32 = 1,
    /// Draft rerank (`mtp.rerankSelect`): a low-bit copy of the TRUNK lm_head
    /// that shortlists 32 candidates, which the real head then re-scores.
    /// Built at LOAD by `qwen4BuildDraftRerank` — not in `loadQwen4Mtp`, which
    /// runs while the Transformer is still under construction and `lm_head_w`
    /// is not assigned yet, but from the scheduler's load path once it is.
    /// `tried` makes it (and the refusal) one-shot, so the draft path's own
    /// ask is a pure read; a null `rerank` after it means the full readout is
    /// this head's draft path.
    rerank: ?mtp_mod.RerankCoarse = null,
    rerank_logged: bool = false,
    rerank_tried: bool = false,

    /// `allocator` is the one `loadQwen4Mtp` took; the layer's weights are the trunk's `Weights`.
    pub fn deinit(self: *Qwen4Mtp, allocator: std.mem.Allocator) void {
        for (self.owned) |a| _ = mlx.mlx_array_free(a);
        allocator.free(self.owned);
        self.cache.deinit();
        _ = mlx.mlx_array_free(self.entry.conv_state);
        _ = mlx.mlx_array_free(self.entry.ssm_state);
        ssmFreeQsaState(&self.entry);
        self.qsa_marks.deinit();
        if (self.rerank) |*rc| rc.deinit();
    }
};

/// Compile the GatedDeltaNet gating chain (astype→exp→add→astype→exp→log1p→
/// multiply→negative→exp→astype, ~10 dispatches) into one fused kernel, at the dtype
/// `config.gdnGateDtype()` holds when the first call traces it.
/// shapeless=true: pure elementwise chain, same trace for any [B,S,Hv].
pub fn compileGdnGate(self: *Transformer) void {
    const raw_closure = mlx.mlx_closure_new_func_payload(
        &gdnGateClosureCallback,
        @ptrCast(self),
        null,
    );
    var compiled = mlx.mlx_closure{ .ctx = null };
    const rc = mlx.mlx_compile(&compiled, raw_closure, true);
    _ = mlx.mlx_closure_free(raw_closure);
    if (rc == 0 and compiled.ctx != null) {
        self.compiled_gdn_gate = compiled;
        log.info("GDN gate compiled (kernel fusion enabled), g {s}\n", .{@tagName(self.config.gdnGateDtype())});
    }
}

pub fn gdnGateClosureCallback(res: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
    const self: *Transformer = @ptrCast(@alignCast(payload.?));
    var A_log = mlx.mlx_array_new();
    if (mlx.mlx_vector_array_get(&A_log, input, 0) != 0) return -1;
    defer _ = mlx.mlx_array_free(A_log);
    var a = mlx.mlx_array_new();
    if (mlx.mlx_vector_array_get(&a, input, 1) != 0) return -1;
    defer _ = mlx.mlx_array_free(a);
    var dt_bias = mlx.mlx_array_new();
    if (mlx.mlx_vector_array_get(&dt_bias, input, 2) != 0) return -1;
    defer _ = mlx.mlx_array_free(dt_bias);

    const g = gdnGateChain(A_log, a, dt_bias, self.config.gdnGateDtype(), self.s) catch return -1;
    const out_arr = [_]mlx.mlx_array{g};
    res.* = mlx.mlx_vector_array_new_data(&out_arr, 1);
    _ = mlx.mlx_array_free(g);
    return 0;
}

/// qwen4_exp: compile the four hyper-connection elementwise tails.
pub fn compileQwen4Hc(self: *Transformer) void {
    const specs = .{
        .{ &self.compiled_hc_silu, &Transformer.hcSiluCallback, true },
        .{ &self.compiled_hc_mix, &Transformer.hcMixCallback, false },
        .{ &self.compiled_hc_inj, &Transformer.hcInjCallback, true },
        .{ &self.compiled_hc_write, &Transformer.hcWriteCallback, true },
    };
    var n: usize = 0;
    inline for (specs) |sp| {
        const raw = mlx.mlx_closure_new_func_payload(sp[1], @ptrCast(self), null);
        var compiled = mlx.mlx_closure{ .ctx = null };
        const rc = mlx.mlx_compile(&compiled, raw, sp[2]);
        _ = mlx.mlx_closure_free(raw);
        if (rc == 0 and compiled.ctx != null) {
            sp[0].* = compiled;
            n += 1;
        }
    }
    if (n > 0) log.info("[qwen4] hyper-connection tails compiled ({d}/4 kernels)\n", .{n});
}

pub fn closureIn(input: mlx.mlx_vector_array, i: usize) ?mlx.mlx_array {
    var a = mlx.mlx_array_new();
    if (mlx.mlx_vector_array_get(&a, input, i) != 0) return null;
    return a;
}

pub fn closureOut(res: *mlx.mlx_vector_array, a: mlx.mlx_array) c_int {
    const arr = [_]mlx.mlx_array{a};
    res.* = mlx.mlx_vector_array_new_data(&arr, 1);
    _ = mlx.mlx_array_free(a);
    return 0;
}

pub fn applyClosure(compiled: ?mlx.mlx_closure, inputs: []const mlx.mlx_array) !?mlx.mlx_array {
    const c = compiled orelse return null;
    const in_vec = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var out_vec = mlx.mlx_vector_array{ .ctx = null };
    try mlx.check(mlx.mlx_closure_apply(&out_vec, c, in_vec));
    defer _ = mlx.mlx_vector_array_free(out_vec);
    if (mlx.mlx_vector_array_size(out_vec) != 1) return null;
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&out, out_vec, 0));
    return out;
}

/// Compile MoE routing (negate→argpartition→slice→softmax→take_along_axis→sum→expand→divide)
/// into a single fused kernel. Input: router_logits. Outputs: inds, norm_scores.
/// shapeless=false: slice bounds derive from input ndim, so the closure must
/// re-trace per input shape. MoE inference only sees two shapes in practice
/// (decode seq_len=1, prefill seq_len=N), so the trace cost amortizes after
/// the first prefill + first decode.
pub fn compileMoeRouting(self: *Transformer) void {
    // MiMo's sigmoid chain takes the per-layer expert_bias as a second input and runs uncompiled.
    if (self.config.moe_sigmoid_router) return;
    const raw_closure = mlx.mlx_closure_new_func_payload(
        &moeRoutingClosureCallback,
        @ptrCast(self),
        null,
    );
    var compiled = mlx.mlx_closure{ .ctx = null };
    const rc = mlx.mlx_compile(&compiled, raw_closure, false);
    _ = mlx.mlx_closure_free(raw_closure);
    if (rc == 0 and compiled.ctx != null) {
        self.compiled_moe_routing = compiled;
        log.info("MoE routing compiled (kernel fusion enabled)\n", .{});
    }
}

/// Pure subgraph for MoE routing. Inputs:
///   [0] router_logits — shape [..., num_experts]
/// Outputs:
///   [0] inds         — shape [..., K], int32 expert indices (top-K)
///   [1] norm_scores  — shape [..., K], renormalized top-K softmax weights
///
/// The sigma-MoE per-expert-scale path stays outside the closure because it
/// branches on per-layer weights at model-load time.
pub fn moeRoutingClosureCallback(res: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
    const self: *Transformer = @ptrCast(@alignCast(payload.?));
    const k: c_int = @intCast(self.config.num_experts_per_tok);

    var router_logits = mlx.mlx_array_new();
    if (mlx.mlx_vector_array_get(&router_logits, input, 0) != 0) return -1;
    defer _ = mlx.mlx_array_free(router_logits);

    const inds_norm = self.moeRoutingUncompiled(router_logits, k) catch return -1;
    defer _ = mlx.mlx_array_free(inds_norm.inds);
    defer _ = mlx.mlx_array_free(inds_norm.norm_scores);

    const out_arr = [_]mlx.mlx_array{ inds_norm.inds, inds_norm.norm_scores };
    res.* = mlx.mlx_vector_array_new_data(&out_arr, 2);
    return 0;
}

/// Batched Qwen4 overlaps graph construction with execution; an explicit setting still wins.
pub fn qwen4DecodeLadderStride(raw: ?[]const u8) u32 {
    return if (raw == null) 4 else Transformer.decodeAsyncLadderStride(raw);
}

// A PREFILL-side async-eval ladder (mlxfast-challenge prefillLadder:
// asyncEval every 3rd-4th layer at seq >= 512) was ported and measured
// 2026-08-16 (qwen3.8-27B-4bit, 43.7k-token prompt, M4 Max): off 227.9
// vs ladder 225.9 tok/s median — null-to-negative, same reason the
// decode ladder loses here: chunked prefill + the eval cadence already
// collect the overlap. Removed rather than shipped as a dead knob.

/// Plain decode rows of one forward, each byte-identical to its solo tick (`forwardQwen4DecodeRows`).
pub fn supportsBatchedQwen4Rows(self: *const Transformer) bool {
    return self.qwen4 != null and self.moe_layers != null and self.expert_stream == null and self.config.supportsBatchedQwen4Rows();
}

/// Full-attention layer with the QSA mask threaded through
/// `gatedFullAttnWith` (q-gate, QK norm, partial RoPE, KV cache all shared).
pub fn qwen4AttnWith(self: *Transformer, ctx: *ForwardCtx, x: mlx.mlx_array, fa: *const FullAttnWeights, entry: *SSMCacheEntry, layer: u32, cache_len: c_int, pos_base: c_int, batch: c_int, seq_len: c_int, is_prefill: bool) !mlx.mlx_array {
    return self.qwen4AttnProjected(ctx, x, fa, entry, layer, cache_len, pos_base, batch, seq_len, is_prefill, null, false);
}

pub fn qwen4AttnProjected(self: *Transformer, ctx: *ForwardCtx, x: mlx.mlx_array, fa: *const FullAttnWeights, entry: *SSMCacheEntry, layer: u32, cache_len: c_int, pos_base: c_int, batch: c_int, seq_len: c_int, is_prefill: bool, projected: ?*const AttnVerifyInputs, skip_output: bool) !mlx.mlx_array {
    // Registered above the fallible builders: `qsaMask` can throw with `ctx.qsa_blocks` already written.
    defer {
        if (ctx.qsa_mask.ctx != null) _ = mlx.mlx_array_free(ctx.qsa_mask);
        ctx.qsa_mask = .{ .ctx = null };
        if (ctx.qsa_blocks.ctx != null) _ = mlx.mlx_array_free(ctx.qsa_blocks);
        ctx.qsa_blocks = .{ .ctx = null };
        if (ctx.batch_slots) |slots| {
            for (slots) |sc| {
                if (sc.qsa_mask.ctx != null) _ = mlx.mlx_array_free(sc.qsa_mask);
                sc.qsa_mask = .{ .ctx = null };
                if (sc.qsa_blocks.ctx != null) _ = mlx.mlx_array_free(sc.qsa_blocks);
                sc.qsa_blocks = .{ .ctx = null };
            }
        }
    }
    const prof = Qwen4AttnProf.begin(x, seq_len);
    defer Qwen4AttnProf.active = false;
    errdefer Qwen4AttnProf.reset();
    if (fa.idx_qk_w.ctx != null and !qwen4Standin().attn_qsa) {
        ctx.qsa_mask = try self.qsaMask(ctx, x, fa, entry, layer, cache_len, pos_base, batch, seq_len);
    }
    if (prof) Qwen4AttnProf.lap(if (ctx.qsa_blocks.ctx != null) ctx.qsa_blocks else ctx.qsa_mask, .indexer);
    if (layer == 3) if (Transformer.qwen4_trace) |tr| {
        if (ctx.qsa_mask.ctx != null) {
            Qwen4Trace.set(&tr.qsa_mask, ctx.qsa_mask);
        } else if (ctx.qsa_blocks.ctx != null) {
            const m = try qsaMaskFromBlocks(self.s, ctx.qsa_blocks, cache_len + seq_len, @intCast(self.config.indexer_compress_ratio));
            defer _ = mlx.mlx_array_free(m);
            Qwen4Trace.set(&tr.qsa_mask, m);
        }
    };
    const out = try self.gatedFullAttnProjected(ctx, x, fa, layer, pos_base + cache_len, batch, seq_len, is_prefill, projected, skip_output);
    if (prof) Qwen4AttnProf.lap(out, .tail);
    return out;
}

/// Load the qwen4_exp MTP head when the pack ships `mtp.*` (null otherwise).
pub fn loadQwen4Mtp(allocator: std.mem.Allocator, config: ModelConfig, weights: *const Weights, name_buf: *[256]u8, s: mlx.mlx_stream) !?Qwen4Mtp {
    const mtp_prefix = "language_model.mtp";
    if (weights.get(mtp_prefix ++ ".fc_hidden.weight") == null) return null;
    var mcfg = config;
    mcfg.weight_prefix = mtp_prefix;
    mcfg.num_hidden_layers = 1;
    mcfg.full_attention_interval = 1; // layer 0 is the full-attention layer
    mcfg.ple_layer_idx = -1;
    mcfg.expert_streaming = false;
    const ml = try initMoeLayers(allocator, mcfg, weights, name_buf, s);
    const layer = ml.moe_layers[0];
    allocator.free(ml.moe_layers);
    var entry = ml.ssm_entries[0];
    allocator.free(ml.ssm_entries);
    var owned: std.ArrayList(mlx.mlx_array) = .fromOwnedSlice(ml.owned_bf16);
    errdefer {
        for (owned.items) |a| _ = mlx.mlx_array_free(a);
        owned.deinit(allocator);
        _ = mlx.mlx_array_free(entry.conv_state);
        _ = mlx.mlx_array_free(entry.ssm_state);
        ssmFreeQsaState(&entry);
    }
    const fe = try weightTriple(weights, mtp_prefix ++ ".fc_embedding", &owned, allocator, s);
    const fh = try weightTriple(weights, mtp_prefix ++ ".fc_hidden", &owned, allocator, s);
    const mixer = try loadHcWeights(weights, mtp_prefix ++ ".hyper_connection_mixer", false, config.hc_count, config.hidden_size, &owned, allocator, s);
    const pne = weights.get(mtp_prefix ++ ".pre_fc_norm_embedding.weight") orelse return error.MissingWeight;
    const pnh = weights.get(mtp_prefix ++ ".pre_fc_norm_hidden.weight") orelse return error.MissingWeight;
    var cache = try KVCache.init(allocator, config.num_hidden_layers + 1);
    errdefer cache.deinit();
    log.info("[qwen4] MTP head loaded (1 hyper-connected QSA+MoE layer; drafts armed by --mtp)\n", .{});
    return .{
        .layer = layer,
        .pre_norm_emb = pne,
        .pre_norm_hidden = pnh,
        .fc_emb_w = fe.w,
        .fc_emb_s = fe.sc,
        .fc_emb_b = fe.bi,
        .fc_hid_w = fh.w,
        .fc_hid_s = fh.sc,
        .fc_hid_b = fh.bi,
        .mixer = mixer,
        .owned = try owned.toOwnedSlice(allocator),
        .cache = cache,
        .entry = entry,
    };
}

/// The head's PER-REQUEST half. Every request owns one and installs it on the
/// module (`qwen4MtpActivate`) before any head operation; the module's own copy
/// is whatever the last owner swapped out.
pub const Qwen4MtpState = struct {
    cache: KVCache,
    entry: SSMCacheEntry,
    seq_offset: usize = 0,
    qsa_marks: QsaHeadMarkSet = .{},
    pos_base: c_int = -1,

    pub const Meta = struct {
        stash_len: usize,
        origin: c_int,
        cache_step: usize,
    };

    pub fn meta(self: *const Qwen4MtpState) Meta {
        return .{
            .stash_len = self.seq_offset,
            .origin = self.pos_base,
            .cache_step = self.cache.step,
        };
    }

    pub fn restoreMeta(self: *Qwen4MtpState, m: Meta) void {
        self.seq_offset = m.stash_len;
        self.pos_base = m.origin;
        self.cache.step = m.cache_step;
    }

    pub fn deinit(self: *Qwen4MtpState) void {
        self.cache.deinit();
        ssmFreeQsaState(&self.entry);
        self.qsa_marks.deinit();
    }
};

/// Pointers to one head's live runtime. The tick holds N of these without a module swap.
pub const Qwen4MtpLive = struct {
    cache: *KVCache,
    entry: *SSMCacheEntry,
    seq_offset: *usize,
    pos_base: *c_int,
    qsa_marks: *QsaHeadMarkSet,
};

pub fn qwen4MtpLiveState(st: *Qwen4MtpState) Qwen4MtpLive {
    return .{
        .cache = &st.cache,
        .entry = &st.entry,
        .seq_offset = &st.seq_offset,
        .pos_base = &st.pos_base,
        .qsa_marks = &st.qsa_marks,
    };
}

pub fn qwen4MtpLiveModule(m: *Qwen4Mtp) Qwen4MtpLive {
    return .{
        .cache = &m.cache,
        .entry = &m.entry,
        .seq_offset = &m.seq_offset,
        .pos_base = &m.pos_base,
        .qsa_marks = &m.qsa_marks,
    };
}

/// A fresh, empty per-request head state (what `qwen4MtpReset` leaves behind).
pub fn qwen4MtpStateNew(self: *Transformer) !Qwen4MtpState {
    const m = &(self.qwen4_mtp orelse return error.NoMtpHead);
    var entry = m.entry;
    entry.conv_state = .{ .ctx = null };
    entry.ssm_state = .{ .ctx = null };
    entry.initialized = false;
    entry.aux_state = .{ .ctx = null };
    entry.qsa_key_buf = .{ .ctx = null };
    entry.qsa_key_rows = 0;
    entry.qsa_hist_rows = 0;
    entry.qsa_pooled = .{ .ctx = null };
    entry.qsa_pooled_buf = .{ .ctx = null };
    entry.qsa_pooled_blocks = 0;
    entry.qsa_score_bank = .{ .ctx = null };
    entry.qsa_score_buf = .{ .ctx = null };
    entry.qsa_score_blocks = 0;
    entry.spec_state_seq = .{ .ctx = null };
    entry.spec_replay = .{};
    entry.spec_conv_input = .{ .ctx = null };
    entry.spec_ple_input = .{ .ctx = null };
    const cache = try KVCache.initWithConfig(self.allocator, self.config.num_hidden_layers + 1, m.cache.config);
    return .{ .cache = cache, .entry = entry };
}

pub fn qwen4MtpSwapState(m: *Qwen4Mtp, st: *Qwen4MtpState) void {
    std.mem.swap(KVCache, &m.cache, &st.cache);
    std.mem.swap(SSMCacheEntry, &m.entry, &st.entry);
    std.mem.swap(usize, &m.seq_offset, &st.seq_offset);
    std.mem.swap(QsaHeadMarkSet, &m.qsa_marks, &st.qsa_marks);
    std.mem.swap(c_int, &m.pos_base, &st.pos_base);
}

/// Install `st` as the head's live state (swapping the previous owner's out). Idempotent.
pub fn qwen4MtpActivate(self: *Transformer, st: *Qwen4MtpState) void {
    const m = &(self.qwen4_mtp orelse return);
    if (self.qwen4_mtp_owner == st) return;
    if (self.qwen4_mtp_owner) |prev| qwen4MtpSwapState(m, prev);
    qwen4MtpSwapState(m, st);
    self.qwen4_mtp_owner = st;
}

/// Take `st` back off the module before it is freed.
pub fn qwen4MtpRelease(self: *Transformer, st: *Qwen4MtpState) void {
    const m = &(self.qwen4_mtp orelse return);
    if (self.qwen4_mtp_owner != st) return;
    qwen4MtpSwapState(m, st);
    self.qwen4_mtp_owner = null;
}

/// Reset the MTP head's per-request state (KV, indexer keys, position).
pub fn qwen4MtpReset(self: *Transformer) !void {
    const m = &(self.qwen4_mtp orelse return);
    try m.cache.reinit(self.config.num_hidden_layers + 1, m.cache.config);
    ssmFreeQsaState(&m.entry);
    m.qsa_marks.deinit();
    m.seq_offset = 0;
    m.pos_base = -1;
}

pub fn qwen4MtpResetOwned(self: *Transformer, enable_mtp: bool) void {
    if (enable_mtp) self.qwen4MtpReset() catch {};
}

pub fn mtpHeadKvQuantEnabled() bool {
    if (Transformer.mtp_head_kv_quant_override) |v| return v;
    return Transformer.mtp_head_kv_quant_flag;
}

/// The head's KV scheme: dense unless `--mtp-head-kv-quant` is set, then the trunk's.
/// Fixed at load at both construction sites, so a session never mixes schemes and the
/// bill prices the head at THIS scheme, never at a request's kv_quant override.
pub fn qwen4MtpHeadKvConfig(trunk: KVQuantConfig) KVQuantConfig {
    if (mtpHeadKvQuantEnabled()) return trunk;
    return KVQuantConfig.dense;
}

/// Re-creates the head cache under the chosen scheme with the HEAD's own geometry
/// (layers + 1, its key head dim). Acceptance measured within noise of dense at
/// 4k-128k; 960 B/token saved with MTP on.
pub fn qwen4MtpApplyKvQuant(self: *Transformer, trunk: KVQuantConfig) !void {
    const m = &(self.qwen4_mtp orelse return);
    const cfg = qwen4MtpHeadKvConfig(trunk);
    const n_layers: u32 = self.config.num_hidden_layers + 1;
    if (std.meta.eql(m.cache.config, cfg) and m.cache.entries.len == n_layers) return;
    try m.cache.reinit(n_layers, cfg);
}

/// Record the head's QSA leftover at `row`, the trunk's checkpoint position in the head's
/// own row space (absolute minus the head's base): a warm restore clamps the head to
/// exactly that row, by then a whole generated tail below its ring. A head that is not at
/// `row` — one whose history window skipped these chunks — marks nothing. Best effort; a
/// missing mark is a blind head, never a wrong one.
pub fn qwen4MtpMarkQsaLeftover(self: *Transformer, row: usize) !void {
    const m = &(self.qwen4_mtp orelse return);
    if (row != m.seq_offset) return;
    const pos: c_int = @intCast(m.seq_offset);
    const ratio = @max(m.entry.qsa_ratio, 1);
    // An aligned position needs no rows at all: the pooled bank is the whole history there.
    if (pos == 0 or @mod(pos, ratio) == 0 or m.entry.aux_state.ctx == null) return;
    const rows = try qsaLeftoverAt(m.entry.aux_state, qsaHistoryRows(&m.entry), pos, ratio, self.s);
    if (rows.ctx == null) return;
    // The copy is lazy and its parent is the ring the next append rewrites.
    {
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        _ = mlx.mlx_vector_array_append_value(vec, rows);
        _ = mlx.mlx_eval(vec);
    }
    m.qsa_marks.put(pos, rows);
}

/// The marks a commit stores beside the head's QSA half.
pub fn qwen4MtpMarks(self: *const Transformer) []const QsaHeadMark {
    if (self.qwen4_mtp) |*m| return m.qsa_marks.slice();
    return &.{};
}

/// MTP draft logits: `stream_prev` `[B,S,hc*H]` is the trunk's pre-mixer
/// stream at positions p.., `token_ids` `[B,S]` the tokens at p+1.., and
/// `pos_offset` = p+1 of the first row (the draft query's own position).
/// Returns `[B,S,V]` logits for positions p+2... Caller frees.
/// `logits` is the last-row projection (null unless `.all_rows`/`.last_row`),
/// `stream` the pre-mixer hyper-connection stream the next chain step reads,
/// and `mixed` the MIXER output — the vector the lm_head consumes — which
/// only `.mixed_last_row` fills. The three are distinct spaces: `stream` is
/// `[B,S,hc*H]`, `mixed` is `[B,1,H]`, and handing the wrong one to the
/// lm_head is a silent shape error, not a wrong answer.
pub const Qwen4MtpOut = struct {
    logits: mlx.mlx_array,
    stream: mlx.mlx_array,
    mixed: mlx.mlx_array = .{ .ctx = null },
};

/// Owns the rows built so far, so a failure part-way frees exactly them.
pub const Qwen4MtpOutBuf = struct {
    items: []Qwen4MtpOut,
    built: usize = 0,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, n: usize) !Qwen4MtpOutBuf {
        return .{ .items = try allocator.alloc(Qwen4MtpOut, n), .allocator = allocator };
    }

    pub fn deinit(self: *Qwen4MtpOutBuf) void {
        for (self.items[0..self.built]) |o| {
            if (o.logits.ctx != null) _ = mlx.mlx_array_free(o.logits);
            if (o.stream.ctx != null) _ = mlx.mlx_array_free(o.stream);
            if (o.mixed.ctx != null) _ = mlx.mlx_array_free(o.mixed);
        }
        self.allocator.free(self.items);
    }

    pub fn abandon(self: *Qwen4MtpOutBuf) []Qwen4MtpOut {
        return self.items;
    }
};

/// What `qwen4MtpForward` owes its caller past the head's own layer. The
/// mixer read + the 248320-wide lm_head run over whatever rows `stream`
/// carries, so a caller that consumes ONE row says so instead of paying
/// for `S` of them and slicing (the merged history step after a partial
/// accept is `1 + accepted` rows wide, of which the drafter wants the
/// last).
pub const Qwen4MtpProject = enum {
    /// Every row's logits + the full `[B,S,hc*H]` stream (fixture oracles).
    all_rows,
    /// The LAST row only: `h` is sliced BEFORE the mixer, so `logits` is
    /// `[B,1,V]` and `stream` is that one row.
    last_row,
    /// No logits at all (history append): mixer + lm_head are skipped and
    /// `logits` comes back null; `stream` is the full `[B,S,hc*H]`.
    none,
    /// The LAST row's mixer output in `mixed`, and NO lm_head: the draft
    /// rerank path shortlists through its own coarse head, so the 248320-
    /// wide projection — 675 MB, the single biggest read in a draft step —
    /// never runs. `logits` comes back null; `stream` is the last row.
    mixed_last_row,
};

/// Truncate the head's committed history to `len` rows.
pub fn qwen4MtpTruncate(self: *Transformer, len: usize) !void {
    const m = &(self.qwen4_mtp orelse return);
    var live = qwen4MtpLiveModule(m);
    try self.qwen4MtpTruncateLive(&live, len);
}

pub fn qwen4MtpTruncateOn(self: *Transformer, st: *Qwen4MtpState, len: usize) !void {
    var live = qwen4MtpLiveState(st);
    try self.qwen4MtpTruncateLive(&live, len);
}

pub fn qwen4MtpTruncateLive(self: *Transformer, live: *Qwen4MtpLive, len: usize) !void {
    if (len >= live.seq_offset.*) return;
    try live.cache.truncate(len, self.s);
    if (live.entry.aux_state.ctx != null or live.entry.qsa_pooled.ctx != null) {
        if (len == 0) {
            ssmFreeQsaState(live.entry);
        } else {
            const keep: c_int = @intCast(len);
            if (!try qsaResliceToKeep(live.entry, keep, self.s)) {
                if (live.entry.aux_state.ctx == null) {
                    live.entry.qsa_hist_rows = keep;
                    try truncatePooled(&live.entry.qsa_pooled, @intCast(len), live.entry.qsa_ratio, self.s, true);
                } else {
                    const hist = qsaHistoryRows(live.entry);
                    const drop_tail = hist - keep;
                    const ks = mlx.getShape(live.entry.aux_state);
                    const new_held = ks[1] - drop_tail;
                    if (new_held < @mod(keep, @max(live.entry.qsa_ratio, 1))) {
                        try self.qwen4MtpTruncateToMarkLive(live, keep);
                    } else {
                        const start = [_]c_int{ 0, 0, 0 };
                        const stop = [_]c_int{ ks[0], new_held, ks[2] };
                        const strides = [_]c_int{ 1, 1, 1 };
                        var view = mlx.mlx_array_new();
                        defer _ = mlx.mlx_array_free(view);
                        try mlx.check(mlx.mlx_slice(&view, live.entry.aux_state, &start, 3, &stop, 3, &strides, 3, self.s));
                        const owned = try materializedOwnedCopy(self.s, view);
                        const pooled = live.entry.qsa_pooled;
                        live.entry.qsa_pooled = .{ .ctx = null };
                        ssmFreeQsaState(live.entry);
                        live.entry.qsa_pooled = pooled;
                        live.entry.aux_state = owned;
                        live.entry.qsa_key_rows = new_held;
                        live.entry.qsa_hist_rows = keep;
                        try truncatePooled(&live.entry.qsa_pooled, @intCast(len), live.entry.qsa_ratio, self.s, true);
                    }
                }
            }
        }
    }
    live.qsa_marks.dropAbove(@intCast(len));
    live.seq_offset.* = len;
}

pub fn qwen4MtpTruncateToMarkLive(self: *Transformer, live: *Qwen4MtpLive, keep: c_int) !void {
    const want_lv = @mod(keep, @max(live.entry.qsa_ratio, 1));
    var rows: mlx.mlx_array = .{ .ctx = null };
    if (want_lv > 0) {
        const marked = live.qsa_marks.find(keep) orelse return error.MtpHeadQsaRingGap;
        if (mlx.getShape(marked)[1] != want_lv) return error.MtpHeadQsaRingGap;
        rows = mlx.mlx_array_new();
        mlx.check(mlx.mlx_array_set(&rows, marked)) catch |err| {
            _ = mlx.mlx_array_free(rows);
            return err;
        };
    }
    const pooled = live.entry.qsa_pooled;
    live.entry.qsa_pooled = .{ .ctx = null };
    ssmFreeQsaState(live.entry);
    live.entry.qsa_pooled = pooled;
    live.entry.aux_state = rows;
    live.entry.qsa_key_rows = want_lv;
    live.entry.qsa_hist_rows = keep;
    try truncatePooled(&live.entry.qsa_pooled, keep, live.entry.qsa_ratio, self.s, true);
}

/// Adopt a committed head history from the prefix cache: the head's KV plus the QSA aux
/// entry it is only valid with, trimmed to `want` rows. Any failure resets the head to blank.
pub fn qwen4MtpAdopt(
    self: *Transformer,
    kv_snap: *const KVCacheSnapshot,
    aux: *const SSMCacheEntrySnapshot,
    marks: []const QsaHeadMark,
    pos_base: c_int,
    want: usize,
) !void {
    const m = &(self.qwen4_mtp orelse return error.NoMtpHead);
    if (kv_snap.step > 0 and pos_base < 0) return error.MtpHeadInvalidOrigin;
    // The head's row count IS its cache's step (`qwen4MtpAdvance`).
    if (m.cache.step != m.seq_offset) return error.MtpHeadStepGap;
    // A history that is not exactly as long as the KV is not adoptable at any length.
    if (aux.aux_state.ctx == null and aux.qsa_pooled.ctx == null) return error.MtpHeadNoQsaHistory;
    const hist: c_int = if (aux.qsa_rows > 0) aux.qsa_rows else blk: {
        if (aux.aux_state.ctx == null) break :blk 0;
        const sh = mlx.getShape(aux.aux_state);
        break :blk if (sh.len >= 2) sh[1] else 0;
    };
    if (hist != @as(c_int, @intCast(kv_snap.step))) return error.MtpHeadQsaHistoryGap;
    errdefer self.qwen4MtpReset() catch {};
    try m.cache.restore(kv_snap);
    // `ssmRestore` replaces the aux state, so a stale history cannot survive.
    try ssmRestore(&m.entry, aux);
    // Same discipline for the marks: the clamp below reads the committed set, never a
    // previous request's.
    m.qsa_marks.deinit();
    m.qsa_marks = QsaHeadMarkSet.share(marks);
    m.seq_offset = kv_snap.step;
    m.pos_base = pos_base;
    // Unconditional clamp: a snapshot's buffer can be longer than the matched prefix.
    try self.qwen4MtpTruncate(want);
    if (m.seq_offset != want) return error.MtpHeadTrimGap;
}

/// Commit a forward's `seq_len` rows to the head's own bookkeeping: `KVCache.update`
/// advances `step` only at layer 0, and the head's layer is `num_hidden_layers`. Without
/// this the committed snapshot's step disagreed with its QSA history and every restore declined.
pub fn qwen4MtpAdvance(cache: *KVCache, seq_offset: *usize, seq_len: c_int) void {
    seq_offset.* += @intCast(seq_len);
    cache.step = seq_offset.*;
}

/// `mrope_ctx` (image turns): the head's rows sit at ABSOLUTE positions
/// `pos_base + seq_offset ..`, so its attention + QSA indexer read the
/// slot's 3-D table there (prompt rows spanning the image) and the
/// scalar `offset + delta` past it — the same two arms as the trunk.
pub fn qwen4MtpForward(self: *Transformer, stream_prev: mlx.mlx_array, token_ids_in: mlx.mlx_array, pos_offset: c_int, mrope_ctx: ?mrope.PositionContext, project: Qwen4MtpProject) !Qwen4MtpOut {
    const m = &(self.qwen4_mtp orelse return error.NoMtpHead);
    var live = qwen4MtpLiveModule(m);
    return self.qwen4MtpForwardLive(m, &live, stream_prev, token_ids_in, pos_offset, mrope_ctx, project);
}

pub fn qwen4MtpForwardOn(self: *Transformer, st: *Qwen4MtpState, stream_prev: mlx.mlx_array, token_ids_in: mlx.mlx_array, pos_offset: c_int, mrope_ctx: ?mrope.PositionContext, project: Qwen4MtpProject) !Qwen4MtpOut {
    const m = &(self.qwen4_mtp orelse return error.NoMtpHead);
    var live = qwen4MtpLiveState(st);
    return self.qwen4MtpForwardLive(m, &live, stream_prev, token_ids_in, pos_offset, mrope_ctx, project);
}

pub const Qwen4MtpBatchRow = struct {
    st: *Qwen4MtpState,
    stream: mlx.mlx_array,
    token_ids: mlx.mlx_array,
    pos_offset: c_int,
    mrope_ctx: ?mrope.PositionContext = null,
};

pub fn mtpPadSeq(self: *Transformer, arr: mlx.mlx_array, want: c_int) !mlx.mlx_array {
    const sh = mlx.getShape(arr);
    if (sh[1] == want) {
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&out, arr));
        return out;
    }
    const high = [_]c_int{want - sh[1]};
    const z = mlx.mlx_array_new_int(0);
    defer _ = mlx.mlx_array_free(z);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_pad(&out, arr, &[_]c_int{1}, 1, &[_]c_int{0}, 1, &high, 1, z, "constant", self.s));
    return out;
}

pub fn mtpHeadForceBatched() bool {
    return mtp_head_force_batched_override orelse false;
}

pub fn mtpMoeRows(self: *Transformer, x: mlx.mlx_array, mw: *const MoeMlpWeights) !mlx.mlx_array {
    if (self.config.expert_layout == .exl3_k4) {
        if (expert_exl3_kernels.usesPrefillArm(expert_exl3_kernels.rowsOfShape(mlx.getShape(x))))
            return error.Exl3MtpRowsExceedDecode;
    }
    if (mw.shared_expert_gate_w == null or qwen4Standin().moe_shared) return self.moeMLP(x, mw);
    var routed = mw.*;
    routed.shared_expert_gate_w = null;
    const expert_sum = try self.moeMLP(x, &routed);
    defer _ = mlx.mlx_array_free(expert_sum);
    const down = (try self.moeSharedDown(.mtp_rows, x, mw, &.{})).?;
    defer _ = mlx.mlx_array_free(down);
    const gate_logit = try self.mtpProjectionRows(x, mw.shared_expert_gate_w.?, mw.shared_expert_gate_s.?, mw.shared_expert_gate_b.?, false);
    defer _ = mlx.mlx_array_free(gate_logit);
    return self.moeSharedGateTail(expert_sum, gate_logit, down);
}

pub fn qwen4MtpForwardBatched(
    self: *Transformer,
    rows: []const Qwen4MtpBatchRow,
    project: Qwen4MtpProject,
    depth: u32,
) ![]Qwen4MtpOut {
    const m = &(self.qwen4_mtp orelse return error.NoMtpHead);
    if (rows.len == 0) return error.EmptyMtpBatch;
    if (rows.len > MAX_BATCH_ROWS) return error.MtpBatchTooWide;
    var any_mrope = false;
    var widths: [MAX_BATCH_ROWS]c_int = undefined;
    for (rows, 0..) |r, i| {
        const shape = mlx.getShape(r.stream);
        if (shape.len != 3) return error.MtpHeadStreamShape;
        widths[i] = shape[1];
        if (r.mrope_ctx != null) any_mrope = true;
    }
    const fits = mtpHeadRowsFit(widths[0..rows.len], self.config.hc_count);
    const use_batch = fits and !any_mrope and (rows.len > 1 or mtpHeadForceBatched());
    if (!use_batch) {
        var buf = try Qwen4MtpOutBuf.init(self.allocator, rows.len);
        errdefer buf.deinit();
        for (rows, 0..) |r, i| {
            buf.items[i] = try self.qwen4MtpForwardOn(r.st, r.stream, r.token_ids, r.pos_offset, r.mrope_ctx, project);
            buf.built = i + 1;
        }
        return buf.abandon();
    }

    const Once = struct {
        var logged: bool = false;
    };
    if (!Once.logged) {
        Once.logged = true;
        log.info("[batched] mtp head engaged (slots={d} depth={d})\n", .{ rows.len, depth });
    }

    qwen4_mtp_head_graphs += 1;
    self.fwd_gen +%= 1;
    const cfg = &self.config;
    const hc: c_int = @intCast(cfg.hc_count);
    const hidden: c_int = @intCast(cfg.hidden_size);
    const N: c_int = @intCast(rows.len);
    var seqs: [MAX_BATCH_ROWS]c_int = undefined;
    var s_max: c_int = 0;
    for (rows, 0..) |r, i| {
        const sh = mlx.getShape(r.stream);
        seqs[i] = sh[1];
        if (seqs[i] > s_max) s_max = seqs[i];
        if (r.st.seq_offset == 0) r.st.pos_base = r.pos_offset;
        if (r.pos_offset != r.st.pos_base + @as(c_int, @intCast(r.st.seq_offset))) return error.MtpPositionGap;
    }

    var tok_parts: [MAX_BATCH_ROWS]mlx.mlx_array = undefined;
    var str_parts: [MAX_BATCH_ROWS]mlx.mlx_array = undefined;
    for (tok_parts[0..rows.len], 0..) |*p, i| {
        p.* = .{ .ctx = null };
        str_parts[i] = .{ .ctx = null };
    }
    defer for (tok_parts[0..rows.len], 0..) |p, i| {
        if (p.ctx != null) _ = mlx.mlx_array_free(p);
        if (str_parts[i].ctx != null) _ = mlx.mlx_array_free(str_parts[i]);
    };
    for (rows, 0..) |r, i| {
        var tok2 = mlx.mlx_array_new();
        errdefer if (tok2.ctx != null) {
            _ = mlx.mlx_array_free(tok2);
        };
        try mlx.check(mlx.mlx_reshape(&tok2, r.token_ids, &[_]c_int{ 1, seqs[i] }, 2, self.s));
        tok_parts[i] = try self.mtpPadSeq(tok2, s_max);
        _ = mlx.mlx_array_free(tok2);
        tok2 = .{ .ctx = null };
        str_parts[i] = try self.mtpPadSeq(r.stream, s_max);
    }
    const token_ids = try Transformer.concatAxis0(self.s, tok_parts[0..rows.len]);
    defer _ = mlx.mlx_array_free(token_ids);
    const stream_prev = try Transformer.concatAxis0(self.s, str_parts[0..rows.len]);
    defer _ = mlx.mlx_array_free(stream_prev);

    const e = try self.embedding(token_ids);
    defer _ = mlx.mlx_array_free(e);
    const en = try self.rmsNorm(e, m.pre_norm_emb);
    defer _ = mlx.mlx_array_free(en);
    const ep = try self.mtpProjectionRows(en, m.fc_emb_w, m.fc_emb_s, m.fc_emb_b, false);
    defer _ = mlx.mlx_array_free(ep);
    const hn = try self.rmsNorm(stream_prev, m.pre_norm_hidden);
    defer _ = mlx.mlx_array_free(hn);
    var hn4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hn4);
    try mlx.check(mlx.mlx_reshape(&hn4, hn, &[_]c_int{ N, s_max, hc, hidden }, 4, self.s));
    const hp = try self.mtpProjectionRows(hn4, m.fc_hid_w, m.fc_hid_s, m.fc_hid_b, false);
    defer _ = mlx.mlx_array_free(hp);
    var ep4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ep4);
    try mlx.check(mlx.mlx_reshape(&ep4, ep, &[_]c_int{ N, s_max, 1, hidden }, 4, self.s));
    var x4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x4);
    try mlx.check(mlx.mlx_add(&x4, hp, ep4, self.s));
    var h = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(h);
    try mlx.check(mlx.mlx_reshape(&h, x4, &[_]c_int{ N, s_max, hc * hidden }, 3, self.s));

    var slot_ctxs: [MAX_BATCH_ROWS]ForwardCtx = undefined;
    var slot_ptrs: [MAX_BATCH_ROWS]*ForwardCtx = undefined;
    for (rows, 0..) |r, i| {
        slot_ctxs[i] = .{
            .cache = &r.st.cache,
            .moe_seq_offset = &r.st.seq_offset,
            .ssm_entries = null,
            .capture_hidden = null,
            .vision_embeddings = null,
            .qsa_entry = &r.st.entry,
            .qsa_pos_base = r.st.pos_base,
        };
        slot_ptrs[i] = &slot_ctxs[i];
    }

    const li: u32 = cfg.num_hidden_layers;
    const lw = &m.layer;
    const fa = switch (lw.attn) {
        .full => |w| w,
        .linear => return error.MtpLayerNotAttention,
    };

    var h_parts: [MAX_BATCH_ROWS]mlx.mlx_array = undefined;
    var mlp_mix: [MAX_BATCH_ROWS]mlx.mlx_array = undefined;
    var mlp_inj: [MAX_BATCH_ROWS]mlx.mlx_array = undefined;
    for (h_parts[0..rows.len], 0..) |*p, i| {
        p.* = .{ .ctx = null };
        mlp_mix[i] = .{ .ctx = null };
        mlp_inj[i] = .{ .ctx = null };
    }
    defer for (h_parts[0..rows.len], 0..) |p, i| {
        if (p.ctx != null) _ = mlx.mlx_array_free(p);
        if (mlp_mix[i].ctx != null) _ = mlx.mlx_array_free(mlp_mix[i]);
        if (mlp_inj[i].ctx != null) _ = mlx.mlx_array_free(mlp_inj[i]);
    };
    {
        const hsh = mlx.getShape(h);
        for (rows, 0..) |r, i| {
            const i_c: c_int = @intCast(i);
            var h_i = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(h_i);
            try mlx.check(mlx.mlx_slice(&h_i, h, &[_]c_int{ i_c, 0, 0 }, 3, &[_]c_int{ i_c + 1, seqs[i], hsh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, self.s));
            var pre = try self.hcRead(h_i, &lw.hc_attn.?, 1, seqs[i]);
            defer pre.deinit();
            const ao = try self.qwen4AttnWith(slot_ptrs[i], pre.mixed, &fa, &r.st.entry, li, @intCast(r.st.seq_offset), r.st.pos_base, 1, seqs[i], seqs[i] > 1);
            defer _ = mlx.mlx_array_free(ao);
            h_i = try self.hcWrite(h_i, ao, pre.inj, 1, seqs[i]);
            const pre2 = try self.hcRead(h_i, &lw.hc_mlp.?, 1, seqs[i]);
            mlp_mix[i] = pre2.mixed;
            mlp_inj[i] = pre2.inj;
            h_parts[i] = h_i;
        }
    }
    _ = mlx.mlx_array_free(h);
    h = .{ .ctx = null };
    const mix_all = try Transformer.concatAxis0(self.s, mlp_mix[0..rows.len]);
    defer _ = mlx.mlx_array_free(mix_all);
    const mo_all = switch (lw.mlp) {
        .moe => |*mw| try self.mtpMoeRows(mix_all, mw),
        .dense => |*dw| try self.denseMLP(mix_all, dw),
    };
    defer _ = mlx.mlx_array_free(mo_all);
    const msh = mlx.getShape(mo_all);
    for (0..rows.len) |i| {
        const i_c: c_int = @intCast(i);
        var mo_i = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(mo_i);
        try mlx.check(mlx.mlx_slice(&mo_i, mo_all, &[_]c_int{ i_c, 0, 0 }, 3, &[_]c_int{ i_c + 1, seqs[i], msh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, self.s));
        h_parts[i] = try self.hcWrite(h_parts[i], mo_i, mlp_inj[i], 1, seqs[i]);
    }
    for (rows, 0..) |r, i| {
        qwen4MtpAdvance(&r.st.cache, &r.st.seq_offset, seqs[i]);
    }

    var buf = try Qwen4MtpOutBuf.init(self.allocator, rows.len);
    errdefer buf.deinit();

    if (project == .none) {
        for (0..rows.len) |i| {
            buf.items[i] = .{ .logits = .{ .ctx = null }, .stream = h_parts[i] };
            h_parts[i] = .{ .ctx = null };
            buf.built = i + 1;
        }
        return buf.abandon();
    }

    var last_parts: [MAX_BATCH_ROWS]mlx.mlx_array = undefined;
    for (last_parts[0..rows.len]) |*p| p.* = .{ .ctx = null };
    defer for (last_parts[0..rows.len]) |p| {
        if (p.ctx != null) _ = mlx.mlx_array_free(p);
    };
    for (0..rows.len) |i| {
        var stream_i = h_parts[i];
        h_parts[i] = .{ .ctx = null };
        errdefer _ = mlx.mlx_array_free(stream_i);
        const ssh = mlx.getShape(stream_i);
        if ((project == .last_row or project == .mixed_last_row) and ssh[1] > 1) {
            var last = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(last);
            try mlx.check(mlx.mlx_slice(&last, stream_i, &[_]c_int{ 0, ssh[1] - 1, 0 }, 3, &[_]c_int{ 1, ssh[1], ssh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, self.s));
            _ = mlx.mlx_array_free(stream_i);
            stream_i = last;
        }
        last_parts[i] = stream_i;
    }
    const last_h = try Transformer.concatAxis0(self.s, last_parts[0..rows.len]);
    defer _ = mlx.mlx_array_free(last_h);
    for (last_parts[0..rows.len]) |*part| {
        if (part.ctx != null) _ = mlx.mlx_array_free(part.*);
        part.* = .{ .ctx = null };
    }
    const n_rows: c_int = @intCast(rows.len);
    const mix = try self.hcRead(last_h, &m.mixer, n_rows, 1);
    if (mix.inj.ctx != null) _ = mlx.mlx_array_free(mix.inj);
    defer if (mix.mixed.ctx != null) {
        _ = mlx.mlx_array_free(mix.mixed);
    };
    var logits_all: mlx.mlx_array = .{ .ctx = null };
    defer if (logits_all.ctx != null) {
        _ = mlx.mlx_array_free(logits_all);
    };
    if (project != .mixed_last_row) {
        logits_all = try self.mtpProjectionRows(mix.mixed, self.lm_head_w, self.lm_head_s, self.lm_head_b, true);
    }
    const lsh = if (logits_all.ctx != null) mlx.getShape(logits_all) else mlx.getShape(mix.mixed);
    for (0..rows.len) |i| {
        const i_c: c_int = @intCast(i);
        var stream_i = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(stream_i);
        try mlx.check(mlx.mlx_slice(&stream_i, last_h, &[_]c_int{ i_c, 0, 0 }, 3, &[_]c_int{ i_c + 1, 1, mlx.getShape(last_h)[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, self.s));
        if (project == .mixed_last_row) {
            var mixed_i = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(mixed_i);
            try mlx.check(mlx.mlx_slice(&mixed_i, mix.mixed, &[_]c_int{ i_c, 0, 0 }, 3, &[_]c_int{ i_c + 1, 1, mlx.getShape(mix.mixed)[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, self.s));
            buf.items[i] = .{ .logits = .{ .ctx = null }, .stream = stream_i, .mixed = mixed_i };
        } else {
            var logits = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(logits);
            try mlx.check(mlx.mlx_slice(&logits, logits_all, &[_]c_int{ i_c, 0, 0 }, 3, &[_]c_int{ i_c + 1, 1, lsh[lsh.len - 1] }, 3, &[_]c_int{ 1, 1, 1 }, 3, self.s));
            buf.items[i] = .{ .logits = logits, .stream = stream_i };
        }
        buf.built = i + 1;
    }
    return buf.abandon();
}

pub fn qwen4MtpForwardLive(self: *Transformer, m: *Qwen4Mtp, live: *Qwen4MtpLive, stream_prev: mlx.mlx_array, token_ids_in: mlx.mlx_array, pos_offset: c_int, mrope_ctx: ?mrope.PositionContext, project: Qwen4MtpProject) !Qwen4MtpOut {
    qwen4_mtp_head_graphs += 1;
    self.fwd_gen +%= 1; // per-forward QSA scratch key
    const cfg = &self.config;
    const hc: c_int = @intCast(cfg.hc_count);
    const hidden: c_int = @intCast(cfg.hidden_size);
    const shape = mlx.getShape(stream_prev);
    const batch: c_int = shape[0];
    const seq_len: c_int = shape[1];
    const is_prefill = seq_len > 1;
    if (live.seq_offset.* == 0) live.pos_base.* = pos_offset;
    if (pos_offset != live.pos_base.* + @as(c_int, @intCast(live.seq_offset.*))) return error.MtpPositionGap;

    var token_ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(token_ids);
    try mlx.check(mlx.mlx_reshape(&token_ids, token_ids_in, &[_]c_int{ batch, seq_len }, 2, self.s));
    const e = try self.embedding(token_ids);
    defer _ = mlx.mlx_array_free(e);
    const en = try self.rmsNorm(e, m.pre_norm_emb);
    defer _ = mlx.mlx_array_free(en);
    const ep = try self.qmatmul(en, m.fc_emb_w, m.fc_emb_s, m.fc_emb_b);
    defer _ = mlx.mlx_array_free(ep);
    const hn = try self.rmsNorm(stream_prev, m.pre_norm_hidden); // ONE rms over all hc*H
    defer _ = mlx.mlx_array_free(hn);
    const shape4 = [_]c_int{ batch, seq_len, hc, hidden };
    var hn4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hn4);
    try mlx.check(mlx.mlx_reshape(&hn4, hn, &shape4, 4, self.s));
    const hp = try self.qmatmul(hn4, m.fc_hid_w, m.fc_hid_s, m.fc_hid_b);
    defer _ = mlx.mlx_array_free(hp);
    var ep4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ep4);
    try mlx.check(mlx.mlx_reshape(&ep4, ep, &[_]c_int{ batch, seq_len, 1, hidden }, 4, self.s));
    var x4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x4);
    try mlx.check(mlx.mlx_add(&x4, hp, ep4, self.s));
    var h = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(h);
    try mlx.check(mlx.mlx_reshape(&h, x4, &[_]c_int{ batch, seq_len, hc * hidden }, 3, self.s));

    var ctx: ForwardCtx = .{ .cache = live.cache, .moe_seq_offset = live.seq_offset, .ssm_entries = null, .capture_hidden = null, .vision_embeddings = null };
    if (mrope_ctx) |mc| {
        ctx.mrope_pos = mc.pos;
        ctx.mrope_total = mc.total;
        ctx.mrope_delta = mc.delta;
    }
    try self.beginMropeChunk(&ctx, @intCast(live.pos_base.* + @as(c_int, @intCast(live.seq_offset.*))), @intCast(seq_len), mlx.mlx_array_dtype(h));
    defer Transformer.endMropeChunk(&ctx);
    const li: u32 = cfg.num_hidden_layers;
    const lw = &m.layer;
    var pre = try self.hcRead(h, &lw.hc_attn.?, batch, seq_len);
    defer pre.deinit();
    const attn_out = switch (lw.attn) {
        .full => |fa| try self.qwen4AttnWith(&ctx, pre.mixed, &fa, live.entry, li, @intCast(live.seq_offset.*), live.pos_base.*, batch, seq_len, is_prefill),
        .linear => return error.MtpLayerNotAttention,
    };
    defer _ = mlx.mlx_array_free(attn_out);
    h = try self.hcWrite(h, attn_out, pre.inj, batch, seq_len);
    var pre2 = try self.hcRead(h, &lw.hc_mlp.?, batch, seq_len);
    defer pre2.deinit();
    const mlp_out = switch (lw.mlp) {
        .moe => |*mw| try self.moeMLP(pre2.mixed, mw),
        .dense => |*dw| try self.denseMLP(pre2.mixed, dw),
    };
    defer _ = mlx.mlx_array_free(mlp_out);
    h = try self.hcWrite(h, mlp_out, pre2.inj, batch, seq_len);
    qwen4MtpAdvance(live.cache, live.seq_offset, seq_len);

    // History append wants the stream and nothing else: the mixer read and
    // the vocab-wide projection are pure, so skipping them is free.
    if (project == .none) return .{ .logits = .{ .ctx = null }, .stream = h };
    // Last-row-only: cut `h` HERE, so the mixer + lm_head see one row
    // instead of S. Same values as slicing the outputs afterwards.
    var out_seq: c_int = seq_len;
    if ((project == .last_row or project == .mixed_last_row) and seq_len > 1) {
        const hs = mlx.getShape(h);
        var h_last = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(h_last);
        try mlx.check(mlx.mlx_slice(
            &h_last,
            h,
            &[_]c_int{ 0, seq_len - 1, 0 },
            3,
            &[_]c_int{ hs[0], seq_len, hs[2] },
            3,
            &[_]c_int{ 1, 1, 1 },
            3,
            self.s,
        ));
        _ = mlx.mlx_array_free(h);
        h = h_last;
        out_seq = 1;
    }

    const mix = try self.hcRead(h, &m.mixer, batch, out_seq);
    if (mix.inj.ctx != null) _ = mlx.mlx_array_free(mix.inj);
    // Rerank draft: the mixer output IS the answer. Skipping
    // `lmHeadProject` here is the whole point — the 248320-wide 8-bit
    // projection is 675 MB, more than every other read in a draft step put
    // together, and a greedy draft only ever needed its argmax.
    if (project == .mixed_last_row) {
        return .{ .logits = .{ .ctx = null }, .stream = h, .mixed = mix.mixed };
    }
    defer _ = mlx.mlx_array_free(mix.mixed);
    const logits = try self.lmHeadProject(mix.mixed, false);
    return .{ .logits = logits, .stream = h };
}

/// Build the qwen4_exp head's coarse rerank head. ONE-SHOT (`rerank_tried`
/// caches the refusal too) and infallible — false keeps the full-vocab
/// draft projection.
///
/// Called at LOAD (`scheduler.doLoadOnInferenceThread`, the site both the
/// boot load and the `/v1/load` cold load route through), the sidecar
/// arm's `maybeBuildDraftRerank`-at-bind twin: a full `requantizeRows` of
/// the trunk lm_head plus a synchronous eval of its ~240 MB is a LOAD
/// cost, and inside the first request's draft chain it was first-token
/// latency on the inference thread with the stream drained mid-round.
pub fn qwen4BuildDraftRerank(self: *Transformer) bool {
    const m = &(self.qwen4_mtp orelse return false);
    if (m.rerank_tried) return m.rerank != null;
    m.rerank_tried = true;
    const io = std.Io.Threaded.global_single_threaded.io();
    const t0 = std.Io.Timestamp.now(io, .awake);
    // The ONE env read: the built head carries its width from here on.
    const rc = mtp_mod.buildRerankCoarse(self.s, self, mtp_mod.rerankCoarseBits()) orelse return false;
    m.rerank = rc;
    const bytes = mtp_mod.rerankCoarseBytes(rc.rows, @intCast(self.config.hidden_size), rc.bits);
    const ms: u64 = @intCast(@divTrunc(t0.untilNow(io, .awake).nanoseconds, std.time.ns_per_ms));
    log.info(
        "[qwen4] MTP draft rerank: coarse lm_head {d}-bit ({d} MB, {d} ms) + exact top-32 rescoring\n",
        .{ rc.bits, bytes / (1024 * 1024), ms },
    );
    return true;
}

/// Is the qwen4_exp head's draft rerank available? A pure read once the
/// load-time build has run; the build is only reached here when it did
/// NOT (an offline path, a directly-constructed Transformer), and it is
/// one-shot either way.
pub fn qwen4DraftRerankReady(self: *Transformer) bool {
    const m = &(self.qwen4_mtp orelse return false);
    if (m.rerank_tried) return m.rerank != null;
    return self.qwen4BuildDraftRerank();
}

/// One greedy draft id from the mixer output `x` `[B,1,H]`. Shares the
/// sidecar head's scheme verbatim; a coarse-head absence or a shortlist
/// failure falls through to the full trunk-head readout, which is exactly
/// what this path replaced.
pub fn qwen4DraftSelect(self: *Transformer, x: mlx.mlx_array, suppress_mask: ?mlx.mlx_array) !mlx.mlx_array {
    const m = &(self.qwen4_mtp orelse return error.NoMtpHead);
    if (try mtp_mod.rerankSelect(self.s, self, &m.rerank, &m.rerank_logged, x, suppress_mask)) |tok| return tok;
    return mtp_mod.fullReadoutArgmax(self.s, self, x, suppress_mask);
}

/// The exact re-scored top-32 shortlist for a SAMPLED draft. Null = the
/// coarse head is unavailable and the caller falls back to a greedy draft.
pub fn qwen4DraftShortlist(self: *Transformer, x: mlx.mlx_array, suppress_mask: ?mlx.mlx_array) !?mtp_mod.Shortlist {
    const m = &(self.qwen4_mtp orelse return error.NoMtpHead);
    return mtp_mod.rerankShortlist(self.s, self, &m.rerank, &m.rerank_logged, x, suppress_mask);
}

/// Per-row exact re-scored shortlists for a batched SAMPLED draft step.
/// False = no coarse head; the caller drafts greedily instead.
pub fn qwen4DraftShortlistsBatched(
    self: *Transformer,
    x: mlx.mlx_array,
    suppress_mask: ?mlx.mlx_array,
    out: []mtp_mod.Shortlist,
) !bool {
    const m = &(self.qwen4_mtp orelse return error.NoMtpHead);
    return mtp_mod.rerankShortlistsBatched(self.s, self, &m.rerank, &m.rerank_logged, x, suppress_mask, out);
}

pub fn qwen4DraftSelectBatched(self: *Transformer, x: mlx.mlx_array, suppress_mask: ?mlx.mlx_array) !mlx.mlx_array {
    const m = &(self.qwen4_mtp orelse return error.NoMtpHead);
    if (try mtp_mod.rerankSelectBatched(self.s, self, &m.rerank, &m.rerank_logged, x, suppress_mask)) |tok| return tok;
    return mtp_mod.fullReadoutArgmax(self.s, self, x, suppress_mask);
}

/// Test hook: layer-0 intermediates of the qwen4 forward (fixture bisect).
pub const Qwen4Trace = struct {
    mixed_attn: mlx.mlx_array = .{ .ctx = null },
    inj_attn: mlx.mlx_array = .{ .ctx = null },
    attn_out: mlx.mlx_array = .{ .ctx = null },
    mixed_mlp: mlx.mlx_array = .{ .ctx = null },
    inj_mlp: mlx.mlx_array = .{ .ctx = null },
    mlp_out: mlx.mlx_array = .{ .ctx = null },
    ple_emb: mlx.mlx_array = .{ .ctx = null },
    ple_out: mlx.mlx_array = .{ .ctx = null },
    qsa_mask: mlx.mlx_array = .{ .ctx = null },
    attn3_out: mlx.mlx_array = .{ .ctx = null },
    attn3_input: mlx.mlx_array = .{ .ctx = null },
    q3_raw: mlx.mlx_array = .{ .ctx = null },
    k3_raw: mlx.mlx_array = .{ .ctx = null },
    v3_raw: mlx.mlx_array = .{ .ctx = null },
    q3_rope: mlx.mlx_array = .{ .ctx = null },
    k3_rope: mlx.mlx_array = .{ .ctx = null },
    v3_t: mlx.mlx_array = .{ .ctx = null },
    kv3_k: mlx.mlx_array = .{ .ctx = null },
    kv3_v: mlx.mlx_array = .{ .ctx = null },
    attn3_pre_tail: mlx.mlx_array = .{ .ctx = null },
    gate3: mlx.mlx_array = .{ .ctx = null },
    pub fn set(dst: *mlx.mlx_array, src: mlx.mlx_array) void {
        if (dst.ctx != null) _ = mlx.mlx_array_free(dst.*);
        dst.* = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(dst, src);
    }
    pub fn deinit(self: *Qwen4Trace) void {
        inline for (.{ &self.mixed_attn, &self.inj_attn, &self.attn_out, &self.mixed_mlp, &self.inj_mlp, &self.mlp_out, &self.ple_emb, &self.ple_out, &self.qsa_mask, &self.attn3_out, &self.attn3_input, &self.q3_raw, &self.k3_raw, &self.v3_raw, &self.q3_rope, &self.k3_rope, &self.v3_t, &self.kv3_k, &self.kv3_v, &self.attn3_pre_tail, &self.gate3 }) |f| {
            if (f.ctx != null) _ = mlx.mlx_array_free(f.*);
        }
    }
};

pub fn qwen4VerifyRowsEligible(self: *Transformer, token_rows: []const mlx.mlx_array, ctxs: []const *ForwardCtx) bool {
    if (self.qwen4 == null or !mlx.streamIsGpu(self.s) or !moeRouterFusedEnabled() or
        token_rows.len < 2 or token_rows.len > 8 or token_rows.len != ctxs.len or
        self.config.hidden_size < 32 or self.config.num_experts_per_tok >= 32)
    {
        return false;
    }
    if (Transformer.qwen4_trace != null or diagEnvOnCached(&qwen4_profile_fwd_env, "QWEN4_PROFILE_FWD")) return false;
    // The grouped verify reads resident expert banks; a streamed trunk has none.
    if (self.expert_stream != null) return false;
    const si = qwen4Standin();
    if (si.gdn or si.attn or si.mlp or si.gdn_recur or si.gdn_proj or si.attn_qsa or si.attn_sdpa or si.hc or si.moe_shared or si.moe_router or si.moe_gateup or si.moe_down) return false;
    for (self.moe_layers.?) |lw| switch (lw.mlp) {
        .moe => |mw| if (mw.expert_bias != null) return false,
        else => return false,
    };
    var total_tokens: u64 = 0;
    for (token_rows, ctxs) |tokens, ctx| {
        const shape = mlx.getShape(tokens);
        if (shape.len != 2 or shape[0] != 1 or shape[1] < 1 or ctx.ssm_entries == null or ctx.mrope_pos != null or ctx.vision_embeddings != null or
            ctx.capture_hidden != null or ctx.capture_hidden_all != null or ctx.capture_stream_all != null or ctx.capture_layers != null or ctx.batch_slots != null)
        {
            return false;
        }
        total_tokens += @intCast(shape[1]);
    }
    if (self.config.num_experts == 0 or self.config.num_experts_per_tok == 0) return false;
    return (total_tokens * self.config.num_experts_per_tok) / self.config.num_experts < 4;
}

pub fn forwardQwen4VerifyRows(
    self: *Transformer,
    token_rows: []const mlx.mlx_array,
    ctxs: []const *ForwardCtx,
    out_logits: []mlx.mlx_array,
    out_last: []mlx.mlx_array,
    out_all: []mlx.mlx_array,
) !void {
    const previous_hc_prepared = mtp_verify_hc_prepared_active;
    mtp_verify_hc_prepared_active = true;
    defer mtp_verify_hc_prepared_active = previous_hc_prepared;
    const op_start = mlx.op_count.load(.monotonic);
    const cfg = &self.config;
    const layers = self.moe_layers.?;
    const hc: c_int = @intCast(cfg.hc_count);
    const Row = struct {
        ctx: *ForwardCtx,
        tokens: mlx.mlx_array,
        entries: []SSMCacheEntry,
        offset: usize,
        seq: c_int,
        generation: u64,
        h: mlx.mlx_array,
        pending: ?HcPending = null,
    };
    var rows: [MAX_BATCH_ROWS]Row = undefined;
    var initialized: usize = 0;
    defer for (rows[0..initialized]) |*row| {
        if (row.pending) |*pending| pending.deinit();
        if (row.h.ctx != null) _ = mlx.mlx_array_free(row.h);
    };
    var generation = self.fwd_gen;
    for (token_rows, ctxs, 0..) |tokens, ctx, i| {
        generation +%= 1;
        self.fwd_gen = generation;
        const seq = mlx.getShape(tokens)[1];
        const emb = try self.embedding(tokens);
        defer _ = mlx.mlx_array_free(emb);
        var h = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_tile(&h, emb, &.{ 1, 1, hc }, 3, self.s));
        errdefer _ = mlx.mlx_array_free(h);
        if (self.qwen4_stream_f32 or diagEnvOnCached(&qwen4_stream_f32_env, "QWEN4_STREAM_F32")) {
            var h32 = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&h32, h, .float32, self.s));
            _ = mlx.mlx_array_free(h);
            h = h32;
        }
        rows[i] = .{
            .ctx = ctx,
            .tokens = tokens,
            .entries = ctx.ssm_entries.?,
            .offset = ctx.moe_seq_offset.*,
            .seq = seq,
            .generation = generation,
            .h = h,
        };
        initialized += 1;
    }

    self.spec_capture_ssm = true;
    defer self.spec_capture_ssm = false;
    for (0..layerCap(cfg.num_hidden_layers)) |layer_idx| {
        const layer = &layers[layer_idx];
        var group_projection = true;
        if (group_projection) {
            var total: c_int = 0;
            for (rows[0..initialized]) |row| {
                group_projection = group_projection and row.seq >= 2 and row.seq <= 6;
                total += row.seq;
            }
            group_projection = group_projection and total <= 12;
        }
        var attn_reads: [MAX_BATCH_ROWS]Transformer.HcRead = undefined;
        var attn_count: usize = 0;
        defer for (attn_reads[0..attn_count]) |*read| read.deinit();
        var attn_inputs: [MAX_BATCH_ROWS]mlx.mlx_array = undefined;
        var attn_entries: [MAX_BATCH_ROWS]*SSMCacheEntry = undefined;
        var generations: [MAX_BATCH_ROWS]u64 = undefined;
        for (rows[0..initialized], 0..) |*row, i| {
            self.fwd_gen = row.generation;
            if (layer.ple) |*pw| try self.addVerifyPle(row, pw, layer_idx, i);
        }
        const joined_attn = try self.hcReadVerifyRows(rows[0..initialized], &layer.hc_attn.?);
        defer if (joined_attn) |values| {
            for (values) |*value| if (value.mixed.ctx != null) value.deinit();
            self.allocator.free(values);
        };
        for (rows[0..initialized], 0..) |*row, i| {
            self.fwd_gen = row.generation;
            var read = if (joined_attn) |values| blk: {
                const value = values[i];
                values[i] = .{ .mixed = .{}, .inj = .{} };
                break :blk value;
            } else try self.hcReadPending(&row.h, &layer.hc_attn.?, 1, row.seq, &row.pending);
            if (group_projection) {
                attn_reads[i] = read;
                attn_count += 1;
                attn_inputs[i] = read.mixed;
                attn_entries[i] = &row.entries[layer_idx];
                generations[i] = row.generation;
            } else {
                defer read.deinit();
                const attn = switch (layer.attn) {
                    .linear => |la| try self.gatedDeltaNet(read.mixed, &la, &row.entries[layer_idx], layer_idx, 1, row.seq, true),
                    .full => |fa| try self.qwen4AttnWith(row.ctx, read.mixed, &fa, &row.entries[layer_idx], @intCast(layer_idx), @intCast(row.offset), 0, 1, row.seq, true),
                };
                defer _ = mlx.mlx_array_free(attn);
                try self.hcWriteOrDefer(&row.h, attn, read.inj, 1, row.seq, .{ .ctx = null }, &row.pending);
            }
        }
        if (group_projection) {
            const projected = switch (layer.attn) {
                .linear => |*la| try self.gdnVerifyRows(attn_inputs[0..initialized], attn_entries[0..initialized], generations[0..initialized], layer_idx, la),
                .full => |*fa| try self.attnVerifyRows(attn_inputs[0..initialized], attn_entries[0..initialized], ctxs, generations[0..initialized], layer_idx, fa),
            };
            defer if (projected) |values| {
                for (values) |value| _ = mlx.mlx_array_free(value);
                self.allocator.free(values);
            };
            for (rows[0..initialized], attn_reads[0..initialized], 0..) |*row, read, i| {
                self.fwd_gen = row.generation;
                const attn = if (projected) |values| try standinRef(values[i]) else switch (layer.attn) {
                    .linear => |*la| try self.gatedDeltaNet(read.mixed, la, &row.entries[layer_idx], layer_idx, 1, row.seq, true),
                    .full => |*fa| try self.qwen4AttnWith(row.ctx, read.mixed, fa, &row.entries[layer_idx], @intCast(layer_idx), @intCast(row.offset), 0, 1, row.seq, true),
                };
                defer _ = mlx.mlx_array_free(attn);
                try self.hcWriteOrDefer(&row.h, attn, read.inj, 1, row.seq, .{ .ctx = null }, &row.pending);
            }
        }

        var reads: [MAX_BATCH_ROWS]Transformer.HcRead = undefined;
        var read_count: usize = 0;
        defer for (reads[0..read_count]) |*read| read.deinit();
        var inputs: [MAX_BATCH_ROWS]mlx.mlx_array = undefined;
        const joined_mlp = try self.hcReadVerifyRows(rows[0..initialized], &layer.hc_mlp.?);
        defer if (joined_mlp) |values| {
            for (values) |*value| if (value.mixed.ctx != null) value.deinit();
            self.allocator.free(values);
        };
        for (rows[0..initialized], 0..) |*row, i| {
            self.fwd_gen = row.generation;
            reads[i] = if (joined_mlp) |values| blk: {
                const value = values[i];
                values[i] = .{ .mixed = .{}, .inj = .{} };
                break :blk value;
            } else try self.hcReadPending(&row.h, &layer.hc_mlp.?, 1, row.seq, &row.pending);
            read_count += 1;
            inputs[i] = reads[i].mixed;
        }
        const mw = switch (layer.mlp) {
            .moe => |*weights| weights,
            else => unreachable,
        };
        const mlp = try self.moeVerifyRows(inputs[0..initialized], mw);
        defer {
            for (mlp) |value| _ = mlx.mlx_array_free(value);
            self.allocator.free(mlp);
        }
        for (rows[0..initialized], mlp, reads[0..initialized]) |*row, value, read| {
            self.fwd_gen = row.generation;
            try self.hcWriteOrDefer(&row.h, value, read.inj, 1, row.seq, .{ .ctx = null }, &row.pending);
        }
    }

    var join_lmhead = true;
    var lmhead_widths: [MAX_BATCH_ROWS]c_int = undefined;
    if (join_lmhead) {
        var total: c_int = 0;
        for (rows[0..initialized], 0..) |row, i| {
            join_lmhead = join_lmhead and !row.ctx.skip_lm_head and row.seq >= 2 and row.seq <= 6;
            lmhead_widths[i] = row.seq;
            total += row.seq;
        }
        join_lmhead = join_lmhead and total <= 12;
    }
    var lmhead_inputs: [MAX_BATCH_ROWS]mlx.mlx_array = @splat(.{});
    defer for (lmhead_inputs[0..initialized]) |value| {
        if (value.ctx != null) _ = mlx.mlx_array_free(value);
    };
    var completed: usize = 0;
    errdefer for (0..completed) |i| {
        _ = mlx.mlx_array_free(out_logits[i]);
        _ = mlx.mlx_array_free(out_last[i]);
        _ = mlx.mlx_array_free(out_all[i]);
    };
    for (rows[0..initialized], 0..) |*row, i| {
        self.fwd_gen = row.generation;
        try self.hcFlush(&row.h, 1, row.seq, &row.pending);
        row.ctx.moe_seq_offset.* += @intCast(row.seq);
        var all = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&all, row.h));
        errdefer _ = mlx.mlx_array_free(all);
        const shape = mlx.getShape(row.h);
        var last = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice(&last, row.h, &.{ 0, shape[1] - 1, 0 }, 3, &.{ 1, shape[1], shape[2] }, 3, &.{ 1, 1, 1 }, 3, self.s));
        errdefer _ = mlx.mlx_array_free(last);
        const mix = try self.hcReadPending(&row.h, &self.qwen4_mixer.?, 1, row.seq, &row.pending);
        _ = mlx.mlx_array_free(row.h);
        row.h = .{ .ctx = null };
        if (mix.inj.ctx != null) _ = mlx.mlx_array_free(mix.inj);
        const logits: mlx.mlx_array = if (join_lmhead) blk: {
            lmhead_inputs[i] = mix.mixed;
            break :blk .{};
        } else if (row.ctx.skip_lm_head)
            mix.mixed
        else blk: {
            defer _ = mlx.mlx_array_free(mix.mixed);
            break :blk try self.lmHeadProject(mix.mixed, row.ctx.argmax_only);
        };
        out_logits[i] = logits;
        out_last[i] = last;
        out_all[i] = all;
        completed += 1;
    }
    if (join_lmhead) {
        const vec = mlx.mlx_vector_array_new_data(&lmhead_inputs, initialized);
        defer _ = mlx.mlx_vector_array_free(vec);
        var joined = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(joined);
        try mlx.check(mlx.mlx_concatenate_axis(&joined, vec, 1, self.s));
        if (try verifyLmHeadProjection(self.s, joined, self.lm_head_w, self.lm_head_s, self.lm_head_b, lmhead_widths[0..initialized])) |logits| {
            // The block stays published for the accept side; `out_logits`
            // are views of it either way.
            self.verify_joined_logits = logits;
            var offset: c_int = 0;
            for (rows[0..initialized], 0..) |row, i| {
                try mlx.check(mlx.mlx_slice(&out_logits[i], logits, &.{ 0, offset, 0 }, 3, &.{ 1, offset + row.seq, mlx.getShape(logits)[2] }, 3, &.{ 1, 1, 1 }, 3, self.s));
                offset += row.seq;
            }
            mtp_verify_lmhead_rows_calls +%= 1;
            const Once = struct {
                var logged = false;
            };
            if (!Once.logged) {
                Once.logged = true;
                log.info("[batched] vocabulary verify projection engaged (slots={d})\n", .{initialized});
            }
        } else {
            for (rows[0..initialized], 0..) |row, i| out_logits[i] = try self.lmHeadProject(lmhead_inputs[i], row.ctx.argmax_only);
        }
    }
    self.fwd_gen = generation;
    const group_ops = mlx.op_count.load(.monotonic) - op_start;
    mtp_verify_moe_group_graphs +%= 1;
    mtp_verify_moe_group_last_ops = group_ops;
}

/// Plain batched decode on qwen4_exp: row i is slot i's decode tick. A row's recurrence, attention, PLE and
/// KV append run as the slot's solo tick runs them, on its own state at its own position. The ops that read
/// the same weights for every row (hyper-connection reads, projections, routed experts, lm_head) take one
/// pass through kernels whose per-row arithmetic is the single-row one. Returns N logits `[1, 1, V]`; the
/// caller owns each and the slice. `hidden_rows`, when given, receives every row's pre-mixer stream.
pub fn forwardQwen4DecodeRows(
    self: *Transformer,
    next_tokens: []const u32,
    ctxs: []const *ForwardCtx,
    hidden_rows: ?*?[]mlx.mlx_array,
) ![]mlx.mlx_array {
    const n = ctxs.len;
    std.debug.assert(n == next_tokens.len and n >= 1 and n <= MAX_BATCH_ROWS);
    if (!qwen4_rows_logged) {
        qwen4_rows_logged = true;
        log.info("[batched] qwen4 per-row decode engaged (slots={d})\n", .{n});
    }
    const cfg = &self.config;
    const layers = self.moe_layers.?;
    const hc: c_int = @intCast(cfg.hc_count);
    const Row = struct {
        ctx: *ForwardCtx,
        tokens: mlx.mlx_array,
        entries: []SSMCacheEntry,
        offset: usize,
        seq: c_int,
        generation: u64,
        h: mlx.mlx_array,
        pending: ?HcPending = null,
    };
    var rows: [MAX_BATCH_ROWS]Row = undefined;
    var initialized: usize = 0;
    defer for (rows[0..initialized]) |*row| {
        if (row.pending) |*pending| pending.deinit();
        if (row.h.ctx != null) _ = mlx.mlx_array_free(row.h);
        _ = mlx.mlx_array_free(row.tokens);
    };
    for (ctxs) |ctx| ctx.ple_defer = true;
    defer for (ctxs) |ctx| {
        ctx.ple_defer = false;
    };
    errdefer for (ctxs) |ctx| self.discardDeferredPle(ctx);
    self.spec_capture_ssm = false;
    var generation = self.fwd_gen;
    for (ctxs, next_tokens) |ctx, token| {
        try self.ssmGroupRelease(ctx);
        generation +%= 1;
        self.fwd_gen = generation;
        const id: i32 = @intCast(token);
        const tokens = mlx.mlx_array_new_data(&id, &[_]c_int{ 1, 1 }, 2, .int32);
        errdefer _ = mlx.mlx_array_free(tokens);
        const emb = try self.embedding(tokens);
        defer _ = mlx.mlx_array_free(emb);
        var h = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_tile(&h, emb, &.{ 1, 1, hc }, 3, self.s));
        errdefer _ = mlx.mlx_array_free(h);
        if (self.qwen4_stream_f32 or diagEnvOnCached(&qwen4_stream_f32_env, "QWEN4_STREAM_F32")) {
            var h32 = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&h32, h, .float32, self.s));
            _ = mlx.mlx_array_free(h);
            h = h32;
        }
        rows[initialized] = .{
            .ctx = ctx,
            .tokens = tokens,
            .entries = ctx.ssm_entries orelse return error.MissingSsmEntries,
            .offset = ctx.moe_seq_offset.*,
            .seq = 1,
            .generation = generation,
            .h = h,
        };
        initialized += 1;
    }
    const live = rows[0..initialized];

    for (0..layerCap(cfg.num_hidden_layers)) |layer_idx| {
        const layer = &layers[layer_idx];
        const li: u32 = @intCast(layer_idx);
        if (layer.ple) |*pw| for (live, 0..) |*row, i| {
            self.fwd_gen = row.generation;
            try self.addVerifyPle(row, pw, layer_idx, i);
        };

        var attn_reads: [MAX_BATCH_ROWS]Transformer.HcRead = undefined;
        try self.hcReadRows(live, &layer.hc_attn.?, &attn_reads);
        defer for (attn_reads[0..n]) |*read| read.deinit();
        var inputs: [MAX_BATCH_ROWS]mlx.mlx_array = undefined;
        var entries: [MAX_BATCH_ROWS]*SSMCacheEntry = undefined;
        var generations: [MAX_BATCH_ROWS]u64 = undefined;
        for (live, 0..) |*row, i| {
            inputs[i] = attn_reads[i].mixed;
            entries[i] = &row.entries[layer_idx];
            generations[i] = row.generation;
        }
        const projected = switch (layer.attn) {
            .linear => |*la| try self.gdnDecodeRows(inputs[0..n], entries[0..n], generations[0..n], layer_idx, la),
            .full => |*fa| try self.attnDecodeRows(inputs[0..n], entries[0..n], ctxs, generations[0..n], layer_idx, fa),
        };
        defer if (projected) |values| {
            for (values) |value| _ = mlx.mlx_array_free(value);
            self.allocator.free(values);
        };
        for (live, attn_reads[0..n], 0..) |*row, read, i| {
            self.fwd_gen = row.generation;
            const attn = if (projected) |values| try standinRef(values[i]) else switch (layer.attn) {
                .linear => |*la| try self.gatedDeltaNet(read.mixed, la, entries[i], layer_idx, 1, 1, false),
                .full => |*fa| try self.qwen4AttnWith(row.ctx, read.mixed, fa, entries[i], li, @intCast(row.offset), 0, 1, 1, false),
            };
            defer _ = mlx.mlx_array_free(attn);
            try self.hcWriteOrDefer(&row.h, attn, read.inj, 1, 1, layer.hc_mlp.?.inject_flat, &row.pending);
        }

        var mlp_reads: [MAX_BATCH_ROWS]Transformer.HcRead = undefined;
        try self.hcReadRows(live, &layer.hc_mlp.?, &mlp_reads);
        defer for (mlp_reads[0..n]) |*read| read.deinit();
        for (live, 0..) |_, i| inputs[i] = mlp_reads[i].mixed;
        const mlp: []mlx.mlx_array = switch (layer.mlp) {
            .moe => |*mw| try self.moeVerifyRows(inputs[0..n], mw),
            .dense => |*dw| blk: {
                const out = try self.allocator.alloc(mlx.mlx_array, n);
                var built: usize = 0;
                errdefer {
                    for (out[0..built]) |value| _ = mlx.mlx_array_free(value);
                    self.allocator.free(out);
                }
                for (inputs[0..n], out) |input, *slot| {
                    slot.* = try self.denseMLP(input, dw);
                    built += 1;
                }
                break :blk out;
            },
        };
        defer {
            for (mlp) |value| _ = mlx.mlx_array_free(value);
            self.allocator.free(mlp);
        }
        const next_inject: mlx.mlx_array = if (layer_idx + 1 < layerCap(cfg.num_hidden_layers)) layers[layer_idx + 1].hc_attn.?.inject_flat else .{ .ctx = null };
        for (live, mlp, mlp_reads[0..n]) |*row, value, read| {
            self.fwd_gen = row.generation;
            try self.hcWriteOrDefer(&row.h, value, read.inj, 1, 1, next_inject, &row.pending);
        }
    }

    var hidden_out: ?[]mlx.mlx_array = null;
    errdefer if (hidden_out) |values| {
        for (values) |value| _ = mlx.mlx_array_free(value);
        self.allocator.free(values);
    };
    if (hidden_rows != null) {
        hidden_out = try self.allocator.alloc(mlx.mlx_array, n);
        @memset(hidden_out.?, .{ .ctx = null });
    }
    var mixed: [MAX_BATCH_ROWS]mlx.mlx_array = @splat(.{ .ctx = null });
    defer for (mixed[0..n]) |value| {
        if (value.ctx != null) _ = mlx.mlx_array_free(value);
    };
    for (live, 0..) |*row, i| {
        self.fwd_gen = row.generation;
        try self.hcFlush(&row.h, 1, 1, &row.pending);
        row.ctx.moe_seq_offset.* += 1;
        if (hidden_out) |values| values[i] = try materializedOwnedCopy(self.s, row.h);
        const mix = try self.hcReadPending(&row.h, &self.qwen4_mixer.?, 1, 1, &row.pending);
        _ = mlx.mlx_array_free(row.h);
        row.h = .{ .ctx = null };
        if (mix.inj.ctx != null) _ = mlx.mlx_array_free(mix.inj);
        mixed[i] = mix.mixed;
    }

    const logits = try self.allocator.alloc(mlx.mlx_array, n);
    var made: usize = 0;
    errdefer {
        for (logits[0..made]) |value| _ = mlx.mlx_array_free(value);
        self.allocator.free(logits);
    }
    var joined_logits: ?mlx.mlx_array = null;
    defer if (joined_logits) |value| {
        _ = mlx.mlx_array_free(value);
    };
    if (n >= 2) {
        const vec = mlx.mlx_vector_array_new_data(&mixed, n);
        defer _ = mlx.mlx_vector_array_free(vec);
        var joined = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(joined);
        try mlx.check(mlx.mlx_concatenate_axis(&joined, vec, 1, self.s));
        const widths: [MAX_BATCH_ROWS]c_int = @splat(1);
        joined_logits = try decodeRowsProjection(self.s, joined, self.lm_head_w, self.lm_head_s, self.lm_head_b, widths[0..n]);
    }
    for (live, 0..) |row, i| {
        logits[i] = if (joined_logits) |value| blk: {
            var slice = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(slice);
            try mlx.check(mlx.mlx_slice(&slice, value, &.{ 0, @intCast(i), 0 }, 3, &.{ 1, @as(c_int, @intCast(i)) + 1, mlx.getShape(value)[2] }, 3, &.{ 1, 1, 1 }, 3, self.s));
            break :blk slice;
        } else try self.lmHeadProject(mixed[i], row.ctx.argmax_only);
        made += 1;
    }
    self.fwd_gen = generation;
    try self.flushDeferredPleGroup(ctxs);
    if (hidden_rows) |dst| dst.* = hidden_out;
    hidden_out = null;
    return logits;
}

/// Qwen3.8-Flash-Next forward: embeddings tiled into `hc` residual
/// streams; every block reads a gated mix of the streams and writes back
/// through per-stream scalar gates; the n-gram PLE adds to the streams
/// before its layer; the final mixer replaces model.norm.
pub fn forwardQwen4With(self: *Transformer, ctx: *ForwardCtx, token_ids: mlx.mlx_array) !mlx.mlx_array {
    Qwen4AttnProf.reset();
    errdefer Qwen4AttnProf.reset();
    const dumping = moeDumpBeginForward();
    defer if (dumping) moeDumpForwardDone();
    self.fwd_gen +%= 1; // per-forward QSA scratch key
    if (self.expert_stream) |engine| engine.beginForward();
    const ml = self.moe_layers.?;
    const cfg = &self.config;
    const offset = ctx.moe_seq_offset.*;
    const entries = ctx.ssm_entries orelse return error.MissingSsmEntries;
    const hc: c_int = @intCast(cfg.hc_count);

    self.spec_capture_ssm = ctx.capture_ssm_seq;
    defer self.spec_capture_ssm = false;

    // Vision rows splice into the `hidden`-wide embeddings BEFORE the
    // hyper-connection tile (the reference masked_scatters, then repeats).
    var emb = if (ctx.stream_in == null) try self.embedding(token_ids) else mlx.mlx_array_new();
    if (ctx.stream_in == null) {
        errdefer _ = mlx.mlx_array_free(emb);
        emb = try self.applyVisionEmbeddingsWith(ctx, emb, token_ids);
    }
    defer _ = mlx.mlx_array_free(emb);
    const x_shape = if (ctx.stream_in == null) mlx.getShape(emb) else mlx.getShape(token_ids);
    const batch: c_int = x_shape[0];
    const seq_len: c_int = x_shape[1];
    const is_prefill = seq_len > 1;

    const reps = [_]c_int{ 1, 1, hc };
    var h = mlx.mlx_array_new();
    if (ctx.stream_in) |stream| {
        try mlx.check(mlx.mlx_array_set(&h, stream));
    } else {
        try mlx.check(mlx.mlx_tile(&h, emb, &reps, 3, self.s));
    }
    errdefer _ = mlx.mlx_array_free(h);
    if (self.qwen4_stream_f32 or diagEnvOnCached(&qwen4_stream_f32_env, "QWEN4_STREAM_F32")) {
        var h32 = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&h32, h, .float32, self.s));
        _ = mlx.mlx_array_free(h);
        h = h32;
    }
    // Boundary zero includes all HC streams, matching the layer outputs.
    if (ctx.capture_layers) |cl| if (cl.input) |slot| {
        _ = mlx.mlx_array_set(slot, h);
    };
    // M-RoPE chunk tables: read by every full-attn layer AND the QSA
    // indexer's queries (its pooled block keys take a strided build).
    try self.beginMropeChunk(ctx, @intCast(offset), @intCast(seq_len), mlx.mlx_array_dtype(h));
    defer Transformer.endMropeChunk(ctx);
    var dt = mlx.DtypeTrace.begin("qwen4", h, switch (ml[0].attn) {
        .full => |f| f.q_w,
        .linear => |la| la.qkv_w,
    });

    const eval_cadence = Transformer.prefillEvalCadence(
        Transformer.PREFILL_EVAL_CADENCE_DEFAULT,
        cfg.head_dim,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        @intCast(seq_len),
        @as(u64, @intCast(offset)) + @as(u64, @intCast(seq_len)),
        ctx.cache.config.scheme != .off,
    );

    var prof = try Qwen4FwdProf.init(seq_len, h);
    var pending: ?HcPending = null;
    defer if (pending) |*pd| pd.deinit();

    const ladder = if (ctx.batch_slots != null) self.decode_async_ladder_qwen4 else self.decode_async_ladder;
    var ladder_blocked = false;

    const defer_moe = self.expert_stream != null and self.imatrix == null and batch == 1 and seq_len == 1 and
        !prof.timing and !prof.ops and !dt.on and !dumping and Transformer.qwen4_trace == null and ctx.capture_layers == null and
        @as(u16, @bitCast(qwen4Standin())) == 0 and !decodeProfileEnabled() and
        !diagEnvOnCached(&expert_defer_sync_env, "SUSHI_EXPERT_DEFER_SYNC");
    var deferred: ?DeferredQwenMoe = null;
    defer if (deferred) |*d| d.deinit();
    var layer_idx: usize = 0;
    while (layer_idx < layerCap(cfg.num_hidden_layers)) {
        const li: u32 = @intCast(layer_idx);
        if (dumping) trf.moe_dump_layer = li;
        const lw = &ml[layer_idx];
        const entry = &entries[layer_idx];
        const next_safe = defer_moe and layer_idx + 1 < layerCap(cfg.num_hidden_layers) and streamedSuccessorSafe(&ml[layer_idx + 1]);
        if (deferred != null and !streamedSuccessorSafe(lw))
            _ = try self.verifyQwenMoe(&deferred, &h, &pending, batch, seq_len);
        var undo: ?StreamGdnUndo = if (deferred != null) try StreamGdnUndo.init(entry) else null;
        defer if (undo) |*u| u.deinit();
        var submitted: ?Transformer.StreamMoe = null;
        defer if (submitted) |*work| work.deinit();

        if (lw.ple != null and !ctx.skip_ple) {
            const pw = &lw.ple.?;
            try self.hcFlush(&h, batch, seq_len, &pending);
            const add = try self.pleForward(ctx, h, token_ids, pw, entry, layer_idx, batch, seq_len);
            defer _ = mlx.mlx_array_free(add);
            var h_ple = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&h_ple, h, add, self.s));
            _ = mlx.mlx_array_free(h);
            h = h_ple;
            if (prof.timing) try self.plePreEval(ctx);
            try prof.lap(h, .ple);
        }

        const si = qwen4Standin();
        var pre: Transformer.HcRead = if (si.hc) try self.hcReadStandin(h, batch, seq_len) else try self.hcReadPending(&h, &lw.hc_attn.?, batch, seq_len, &pending);
        defer pre.deinit();
        try prof.lap(pre.mixed, .hc_read);
        const attn_out = switch (lw.attn) {
            .linear => |la| if (si.gdn) try standinRef(pre.mixed) else try self.gatedDeltaNet(pre.mixed, &la, entry, layer_idx, batch, seq_len, is_prefill),
            .full => |fa| if (si.attn) try standinRef(pre.mixed) else try self.qwen4AttnWith(ctx, pre.mixed, &fa, entry, li, @intCast(offset), 0, batch, seq_len, is_prefill),
        };
        defer _ = mlx.mlx_array_free(attn_out);
        try prof.lap(attn_out, if (lw.attn == .linear) .gdn else .attn);
        if (layer_idx == 3) if (Transformer.qwen4_trace) |tr| {
            Qwen4Trace.set(&tr.attn3_input, pre.mixed);
            Qwen4Trace.set(&tr.attn3_out, attn_out);
        };
        if (layer_idx == 0) if (Transformer.qwen4_trace) |tr| {
            Qwen4Trace.set(&tr.mixed_attn, pre.mixed);
            Qwen4Trace.set(&tr.inj_attn, pre.inj);
            Qwen4Trace.set(&tr.attn_out, attn_out);
        };
        if (si.hc) h = try self.hcWrite(h, attn_out, pre.inj, batch, seq_len) else try self.hcWriteOrDefer(&h, attn_out, pre.inj, batch, seq_len, lw.hc_mlp.?.inject_flat, &pending);
        if (prof.timing) try self.hcFlush(&h, batch, seq_len, &pending);
        try prof.lap(h, .hc_write);

        var pre2: Transformer.HcRead = if (si.hc) try self.hcReadStandin(h, batch, seq_len) else try self.hcReadPending(&h, &lw.hc_mlp.?, batch, seq_len, &pending);
        defer pre2.deinit();
        try prof.lap(pre2.mixed, .hc_read);
        var mlp_out = if (si.mlp) try standinRef(pre2.mixed) else switch (lw.mlp) {
            .moe => |*mw| if (self.expert_stream != null)
                try self.moeMLP2WithRouter(pre2.mixed, pre2.mixed, mw, null, false, .{
                    .ctx = ctx,
                    .layer = @intCast(layer_idx),
                    .deferred = if (next_safe or deferred != null) &submitted else null,
                }, null)
            else
                try self.moeMLP(pre2.mixed, mw),
            .dense => |*dw| try self.denseMLP(pre2.mixed, dw),
        };
        defer _ = mlx.mlx_array_free(mlp_out);
        if (deferred != null) {
            if (!try self.verifyQwenMoe(&deferred, &h, &pending, batch, seq_len)) {
                undo.?.restore(entry);
                continue;
            }
        }
        if (submitted) |*work| {
            if (next_safe) {
                const saved_h = try standinRef(h);
                errdefer _ = mlx.mlx_array_free(saved_h);
                const saved_inj = try standinRef(pre2.inj);
                deferred = .{ .moe = work.*, .h = saved_h, .inj = saved_inj };
                submitted = null;
            } else {
                const exact = try self.finishStreamedMoe(work, &lw.mlp.moe, null);
                _ = mlx.mlx_array_free(mlp_out);
                mlp_out = exact;
            }
        }
        try prof.lap(mlp_out, .mlp);
        if (layer_idx == 0) if (Transformer.qwen4_trace) |tr| {
            Qwen4Trace.set(&tr.mixed_mlp, pre2.mixed);
            Qwen4Trace.set(&tr.inj_mlp, pre2.inj);
            Qwen4Trace.set(&tr.mlp_out, mlp_out);
        };
        if (si.hc) h = try self.hcWrite(h, mlp_out, pre2.inj, batch, seq_len) else try self.hcWriteOrDefer(&h, mlp_out, pre2.inj, batch, seq_len, if (layer_idx + 1 < layerCap(cfg.num_hidden_layers)) ml[layer_idx + 1].hc_attn.?.inject_flat else .{ .ctx = null }, &pending);
        if (prof.timing) try self.hcFlush(&h, batch, seq_len, &pending);
        try prof.lap(h, .hc_write);
        prof.endLayer(if (lw.ple != null) .ple else if (lw.attn == .linear) .gdn else .attn);
        if (ladder != 0 and seq_len == 1 and (layer_idx + 1) % ladder == 0 and !ladder_blocked) {
            // The host-filled PLE leaf must be ready before early evaluation.
            // A lazy token sample keeps this forward on its normal terminal eval.
            if (ctx.ple_pending) |pp| {
                var ids_available = false;
                if (mlx._mlx_array_is_available(&ids_available, pp.token_ids) == 0 and ids_available)
                    try self.flushDeferredPle(ctx)
                else
                    ladder_blocked = true;
            }
            if (!ladder_blocked) {
                // The HC write of mlp_out is deferred to the next read.
                Transformer.ladderStepMulti(&.{ h, mlp_out }, ladder, layer_idx, seq_len);
                if (ctx.batch_slots != null and !qwen4_batched_ladder_engaged) {
                    qwen4_batched_ladder_engaged = true;
                    log.info("[qwen4] batched decode ladder engaged: slots={d} stride={d}\n", .{ batch, ladder });
                }
            }
        }

        if (ctx.capture_layers) |cl| {
            for (cl.ids, cl.out) |cid, *slot| {
                if (cid == li) {
                    try self.hcFlush(&h, batch, seq_len, &pending);
                    if (cl.hc_mean == 0) {
                        _ = mlx.mlx_array_set(slot, h);
                    } else {
                        const tap = try hcStreamMean(self.s, h, cl.hc_mean);
                        _ = mlx.mlx_array_free(slot.*);
                        slot.* = tap;
                    }
                }
            }
        }
        if (is_prefill and Transformer.prefillEvalCadenceApplies(seq_len) and ((layer_idx + 1) % eval_cadence == 0 or layer_idx + 1 == layerCap(cfg.num_hidden_layers))) {
            try self.hcFlush(&h, batch, seq_len, &pending);
            try self.plePreEval(ctx);
            try Transformer.evalCadencePoint(h, ctx.ssm_entries);
        }
        dt.layer(h, layer_idx);
        layer_idx += 1;
    }
    std.debug.assert(deferred == null);

    if (ctx.capture_stream_all != null or ctx.capture_hidden != null or ctx.capture_hidden_all != null) {
        try self.hcFlush(&h, batch, seq_len, &pending);
        try self.plePreEval(ctx);
    }
    ctx.moe_seq_offset.* += @intCast(seq_len);
    dt.end(h);
    prof.report(seq_len, @as(usize, @intCast(offset)) + @as(usize, @intCast(seq_len)), ctx.capture_ssm_seq, cfg.num_hidden_layers - cfg.attnCacheLayerCount(), cfg.attnCacheLayerCount());
    Qwen4AttnProf.flush(seq_len, @as(usize, @intCast(offset)) + @as(usize, @intCast(seq_len)));
    if (ctx.capture_stream_all) |target| try capturePrefillHidden(self.s, target, h, true);
    // On this arch the spec "hidden" IS the pre-mixer stream: the MTP head
    // consumes `[B, L, hc*hidden]`, never the mixed 2560 (vLLM/SGLang).
    if (ctx.capture_hidden) |target| {
        try capturePrefillHiddenLast(self.s, target, h, true);
    }
    if (ctx.capture_hidden_all) |target_all| try capturePrefillHidden(self.s, target_all, h, true);

    const mix = try self.hcReadPending(&h, &self.qwen4_mixer.?, batch, seq_len, &pending);
    // The function-scope errdefer reads `h` at unwind time, so surrender the handle where it is released.
    _ = mlx.mlx_array_free(h);
    h = .{ .ctx = null };
    if (mix.inj.ctx != null) _ = mlx.mlx_array_free(mix.inj);
    const final = mix.mixed;
    if (ctx.skip_lm_head) {
        if (self.expert_stream) |engine| engine.finishForward(@intCast(batch * seq_len));
        return final;
    }
    errdefer _ = mlx.mlx_array_free(final);
    const logits = try self.lmHeadProject(final, ctx.argmax_only);
    _ = mlx.mlx_array_free(final);
    if (self.expert_stream) |engine| engine.finishForward(@intCast(batch * seq_len));
    return logits;
}

/// Average `[B, L, hc*H]` over its `hc` streams into a bf16 `[B, L, H]` tap.
fn hcStreamMean(s: mlx.mlx_stream, h: mlx.mlx_array, hc: u32) !mlx.mlx_array {
    const shape = mlx.getShape(h);
    const split = [_]c_int{ shape[0], shape[1], @intCast(hc), @divExact(shape[2], @as(c_int, @intCast(hc))) };
    var streams = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(streams);
    try mlx.check(mlx.mlx_reshape(&streams, h, &split, 4, s));
    var mean = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mean);
    try mlx.check(mlx.mlx_mean_axis(&mean, streams, 2, false, s));
    var tap = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(tap);
    try mlx.check(mlx.mlx_astype(&tap, mean, .bfloat16, s));
    return tap;
}

pub fn gatedFullAttnProjected(
    self: *Transformer,
    ctx: *ForwardCtx,
    x: mlx.mlx_array,
    fa: *const FullAttnWeights,
    layer: u32,
    offset: c_int,
    batch: c_int,
    seq_len: c_int,
    is_prefill: bool,
    projected: ?*const AttnVerifyInputs,
    skip_output: bool,
) !mlx.mlx_array {
    const cache = ctx.cache;
    const cfg = &self.config;
    const h_count: c_int = @intCast(cfg.num_attention_heads);
    const kv_h: c_int = @intCast(cfg.num_key_value_heads);
    const hd: c_int = @intCast(cfg.head_dim);
    const attn_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.query_pre_attn_scalar)));
    const rope_dims: c_int = @intFromFloat(@as(f32, @floatFromInt(cfg.head_dim)) * cfg.partial_rotary_factor);
    const flat_shape = [_]c_int{ batch, seq_len, h_count * hd };

    // Q projection
    const q_proj = if (projected) |values| values.q else try self.qmatmul(x, fa.q_w, fa.q_s, fa.q_b);
    defer if (projected == null) {
        _ = mlx.mlx_array_free(q_proj);
    };

    // With output gate: q_proj outputs [B, S, 2*H*D], split into queries + gate
    // Without: q_proj outputs [B, S, H*D], used directly as queries
    var queries: mlx.mlx_array = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(queries);
    var gate: mlx.mlx_array = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gate);

    if (cfg.attn_output_gate) {
        // Reshape the Q-proj output to [B, S, H, D*2]
        // then `mx.split(_, 2, axis=-1)` into (queries, gate). The single split op
        // replaces our prior two-slice pattern (2 dispatches → 1 dispatch). Adds up
        // across all `full_attention_interval` layers.
        const q_gate_shape = [_]c_int{ batch, seq_len, h_count, hd * 2 };
        var q_gate_r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_gate_r);
        try mlx.check(mlx.mlx_reshape(&q_gate_r, q_proj, &q_gate_shape, 4, self.s));

        var split_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(split_vec);
        try mlx.check(mlx.mlx_split(&split_vec, q_gate_r, 2, -1, self.s));
        if (mlx.mlx_vector_array_size(split_vec) != 2) return error.UnexpectedSplitCount;

        try mlx.check(mlx.mlx_vector_array_get(&queries, split_vec, 0));

        // The gate STAYS 4-D: flattening the split view here merges the
        // head axis across the packed q/gate interleave — a REAL Copy
        // kernel per call. Elementwise sigmoid/multiply below take the
        // strided view copy-free; the element pairing (h, d) <-> flat
        // h*D+d is identical either way.
        try mlx.check(mlx.mlx_vector_array_get(&gate, split_vec, 1));
    } else {
        const q_shape = [_]c_int{ batch, seq_len, h_count, hd };
        try mlx.check(mlx.mlx_reshape(&queries, q_proj, &q_shape, 4, self.s));
    }

    // K, V projections
    const k_proj = if (projected) |values| values.k else try self.qmatmul(x, fa.k_w, fa.k_s, fa.k_b);
    defer if (projected == null) {
        _ = mlx.mlx_array_free(k_proj);
    };
    const v_proj = if (projected) |values| values.v else try self.qmatmul(x, fa.v_w, fa.v_s, fa.v_b);
    defer if (projected == null) {
        _ = mlx.mlx_array_free(v_proj);
    };

    if (layer == 3) if (Transformer.qwen4_trace) |tr| {
        Qwen4Trace.set(&tr.q3_raw, q_proj);
        Qwen4Trace.set(&tr.k3_raw, k_proj);
        Qwen4Trace.set(&tr.v3_raw, v_proj);
        Qwen4Trace.set(&tr.gate3, gate);
    };
    const kv_shape = [_]c_int{ batch, seq_len, kv_h, hd };
    var k_r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_r);
    var v_r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v_r);
    try mlx.check(mlx.mlx_reshape(&k_r, k_proj, &kv_shape, 4, self.s));
    try mlx.check(mlx.mlx_reshape(&v_r, v_proj, &kv_shape, 4, self.s));

    // V transpose (independent of the q/k chain).
    const perm = [_]c_int{ 0, 2, 1, 3 };
    var v_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v_t);
    try mlx.check(mlx.mlx_transpose_axes(&v_t, v_r, &perm, 4, self.s));

    // Fused decode QK-norm+RoPE (hd-128 partial-rotary):
    // one dispatch replaces per-head RMSNorm → transpose → partial RoPE for
    // q AND k. Bit-identical to the composed chain below (parity-pinned
    // incl. the rd=32 qwen geometry). With `attn_output_gate` the q input
    // is the post-split strided view — ensure_row_contiguous materializes
    // it, which is the one extra (tiny) copy this path pays. M-RoPE prefill
    // chunks keep the composed path; decode is scalar rope at
    // offset+delta, which the probe row reproduces exactly.
    //
    // YaRN (a `rope_type: "yarn"` rope_parameters): the scaled denominator
    // array goes to every rope here (mlx_fast_rope takes base OR freqs, so
    // `has_value` flips) and the mscale rides along — post-rotation on q/k
    // for the composed arms, folded into the probe's cos|sin rows for the
    // fused ones (scaling both halves of a rotary pair is the same
    // multiply). `family` 1 keeps the angle cache from serving an unscaled
    // row to a scaled layer.
    const use_yarn = self.yarnActive();
    const rope_family: usize = if (use_yarn) 1 else 0;
    const rope_base = mlx.mlx_optional_float{
        .value = self.config.rope_theta,
        .has_value = !use_yarn,
    };
    const rope_freqs: mlx.mlx_array = if (use_yarn) self.rope_freqs_yarn.? else .{ .ctx = null };
    const rope_mscale: f32 = if (use_yarn) self.config.yarn_attention_factor else 1.0;
    var q_rope = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_rope);
    var k_rope = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_rope);
    var fused_qk = false;
    if (batch == 1 and seq_len == 1 and hd == 128 and ctx.mrope_cos_cur == null and
        fa.q_norm.ctx != null and fa.k_norm.ctx != null and
        self.rms_eps_arr.ctx != null)
    blk: {
        const eff_off: c_int = offset + (if (ctx.mrope_pos != null) ctx.mrope_delta else 0);
        const angles = self.qkAngleFor(rope_family, rope_dims, rope_base, rope_freqs, eff_off, 1.0) catch break :blk;
        const ms: ?mlx.mlx_array = if (use_yarn)
            try constTableAs(self.yarn_mscale.?, mlx.mlx_array_dtype(queries), &self.yarn_mscale_cast, self.s)
        else
            null;
        const pair = (fusedQkNormRope(self.s, queries, k_r, fa.q_norm, fa.k_norm, angles, ms, self.rms_eps_arr, h_count, kv_h, rope_dims) catch null) orelse break :blk;
        _ = mlx.mlx_array_free(q_rope);
        _ = mlx.mlx_array_free(k_rope);
        q_rope = pair[0];
        k_rope = pair[1];
        fused_qk = true;
    }
    // hd-256 sibling (full-attention layers), decode AND
    // spec-verify widths (S 1..16) — the hd-128 gate above never fires
    // there, so those layers paid the composed chain on every step.
    if (!fused_qk and batch == 1 and hd == 256 and seq_len <= 32 and
        ctx.mrope_cos_cur == null and
        fa.q_norm.ctx != null and fa.k_norm.ctx != null and
        self.rms_eps_arr.ctx != null)
    blk: {
        const eff_off: c_int = offset + (if (ctx.mrope_pos != null) ctx.mrope_delta else 0);
        // The kernel has no mscale slot: YaRN's factor is uniform across the
        // rotated slice, so folding it into the angle rows scales exactly the
        // rotation and nothing else (the pass-through dims never appear).
        const angles = self.qkAngleRowsFor(rope_family, rope_dims, rope_base, rope_freqs, eff_off, seq_len, rope_mscale) catch break :blk;
        const pair = (fusedQkNormRope256(self.s, queries, k_r, fa.q_norm, fa.k_norm, angles, self.rms_eps_arr, h_count, kv_h, seq_len, rope_dims) catch null) orelse break :blk;
        _ = mlx.mlx_array_free(q_rope);
        _ = mlx.mlx_array_free(k_rope);
        q_rope = pair[0];
        k_rope = pair[1];
        fused_qk = true;
    }
    if (!fused_qk) {
        // Q/K norms
        const q_normed = try self.rmsNorm(queries, fa.q_norm);
        defer _ = mlx.mlx_array_free(q_normed);
        const k_normed = try self.rmsNorm(k_r, fa.k_norm);
        defer _ = mlx.mlx_array_free(k_normed);

        // Transpose to [B, H, S, D]
        var q_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_t);
        var k_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_t);
        try mlx.check(mlx.mlx_transpose_axes(&q_t, q_normed, &perm, 4, self.s));
        try mlx.check(mlx.mlx_transpose_axes(&k_t, k_normed, &perm, 4, self.s));

        // Partial RoPE. Qwen3-VL image requests use interleaved M-RoPE: manual
        // per-token cos/sin on the prefill chunk (3D t/h/w angles at image
        // tokens), and scalar RoPE at offset+delta on decode (decode tokens are
        // text → t=h=w). Plain scalar partial RoPE otherwise (zero-cost for
        // text-only). YaRN needs nothing here: the M-RoPE tables are
        // filled from the scaled spectrum + mscale at the one choke point
        // (`mropeCosSinAt`), and the scalar arms take the freqs array below.
        if (ctx.mrope_cos_cur) |cos| {
            const sin = ctx.mrope_sin_cur.?;
            _ = mlx.mlx_array_free(q_rope);
            _ = mlx.mlx_array_free(k_rope);
            q_rope = try self.applyMrope(q_t, cos, sin, rope_dims);
            k_rope = try self.applyMrope(k_t, cos, sin, rope_dims);
        } else if (ctx.batch_rope_offsets) |off_arr| {
            // Batched decode: every slot sits at its own position, so the
            // offset is an [N] array, not a scalar. The driver already
            // folded each slot's M-RoPE delta into it.
            try mlx.check(mlx.mlx_fast_rope_dynamic(&q_rope, q_t, rope_dims, false, rope_base, 1.0, off_arr, rope_freqs, self.s));
            try mlx.check(mlx.mlx_fast_rope_dynamic(&k_rope, k_t, rope_dims, false, rope_base, 1.0, off_arr, rope_freqs, self.s));
            if (use_yarn) try self.yarnScaleQK(&q_rope, &k_rope);
        } else {
            const eff_offset: c_int = offset + (if (ctx.mrope_pos != null) ctx.mrope_delta else 0);
            try mlx.check(mlx.mlx_fast_rope(&q_rope, q_t, rope_dims, false, rope_base, 1.0, eff_offset, rope_freqs, self.s));
            try mlx.check(mlx.mlx_fast_rope(&k_rope, k_t, rope_dims, false, rope_base, 1.0, eff_offset, rope_freqs, self.s));
            if (use_yarn) try self.yarnScaleQK(&q_rope, &k_rope);
        }
    }

    // Batched decode: each slot owns its own KV cache and its own kv_len,
    // so the update happens per slot at B=1 (a slice of the stacked k/v)
    // and the reads are padded to a common width and stacked — the same
    // shape the standard batched path builds. Everything downstream (gate,
    // o_proj, MLP) is batch-invariant and stays shared.
    if (ctx.batch_slots) |slots| {
        var attn_out_b = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn_out_b);
        {
            var cache_buf: [MAX_BATCH_ROWS]*KVCache = undefined;
            for (slots, 0..) |slot_ctx, i| cache_buf[i] = slot_ctx.cache;
            try KVCache.appendFromStacked(
                k_rope,
                v_t,
                cache_buf[0..slots.len],
                layer,
                self.s,
                0,
                self.batched_kv_append_override orelse true,
            );

            if (try self.qsaBatchedAttn(ctx, slots, q_rope, layer, seq_len, attn_scale)) |gathered| {
                _ = mlx.mlx_array_free(attn_out_b);
                attn_out_b = gathered;
            } else {
                const dense_views = try self.allocator.alloc(DenseKVView, slots.len);
                defer {
                    for (dense_views) |*dv| dv.deinit();
                    self.allocator.free(dense_views);
                }
                for (dense_views) |*dv| dv.* = .{ .k = .{ .ctx = null }, .v = .{ .ctx = null }, .owned = false };
                const kv_len_buf = try self.allocator.alloc(i32, slots.len);
                defer self.allocator.free(kv_len_buf);
                var kv_max: c_int = 0;
                for (slots, 0..) |slot_ctx, i| {
                    dense_views[i] = try slot_ctx.cache.denseView(layer, self.s);
                    const klen: c_int = mlx.getShape(dense_views[i].k)[2];
                    kv_len_buf[i] = klen;
                    if (klen > kv_max) kv_max = klen;
                }

                const stacked_k = try self.padAndStackBatchedKV(dense_views, true, kv_max);
                defer _ = mlx.mlx_array_free(stacked_k);
                const stacked_v = try self.padAndStackBatchedKV(dense_views, false, kv_max);
                defer _ = mlx.mlx_array_free(stacked_v);
                var stacked_mask = try self.buildBatchedDecodeMask(kv_len_buf, kv_max, seq_len);
                defer _ = mlx.mlx_array_free(stacked_mask);
                if (ctx.qsa_mask.ctx != null) {
                    // qwen4 QSA: the per-slot block selection (bool, false-
                    // padded to kv_max) narrows the additive pad mask. One
                    // dense-mask call serves the whole group, so it is billed
                    // to every slot in it — the tally is per REQUEST.
                    noteQsaArmForSlots(slots, .mask);
                    if (self.cost_trace_active) self.cost_attention |= 16;
                    const neg_inf = bf16Scalar(-std.math.inf(f32), self.s);
                    defer _ = mlx.mlx_array_free(neg_inf);
                    var narrowed = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_where(&narrowed, ctx.qsa_mask, stacked_mask, neg_inf, self.s));
                    _ = mlx.mlx_array_free(stacked_mask);
                    stacked_mask = narrowed;
                }

                try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out_b, q_rope, stacked_k, stacked_v, attn_scale, "array", stacked_mask, .{ .ctx = null }, false, self.s));
            }
        }
        return self.gatedAttnTail(attn_out_b, gate, fa, flat_shape, skip_output);
    }

    var kv_view = try cache.update(layer, k_rope, v_t, self.s, 0);
    defer kv_view.deinit();
    const full_k = kv_view.k;
    const full_v = kv_view.v;
    if (layer == 3) if (Transformer.qwen4_trace) |tr| {
        Qwen4Trace.set(&tr.q3_rope, q_rope);
        Qwen4Trace.set(&tr.k3_rope, k_rope);
        Qwen4Trace.set(&tr.v3_t, v_t);
        Qwen4Trace.set(&tr.kv3_k, full_k);
        Qwen4Trace.set(&tr.kv3_v, full_v);
    };

    // Attention
    var attn_out = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn_out);
    const none_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(none_mask);

    if (Qwen4AttnProf.active) {
        if (kv_view.has_quant_triple) {
            for ([_]mlx.mlx_array{ kv_view.k_triple_q, kv_view.k_triple_scales, kv_view.k_triple_biases, kv_view.v_triple_q, kv_view.v_triple_scales, kv_view.v_triple_biases }) |a| Qwen4AttnProf.sync(a);
        } else {
            Qwen4AttnProf.sync(full_k);
            Qwen4AttnProf.sync(full_v);
        }
        Qwen4AttnProf.lap(q_rope, .proj);
    }
    if (ctx.qsa_blocks.ctx != null or ctx.qsa_mask.ctx != null) {
        // qwen4_exp QSA: the indexer already chose the visible blocks
        // (causal folded in). Prefill gathers them by index; decode/
        // verify widths (and a declined gather) run under the bool mask.
        const ratio: c_int = @intCast(self.config.indexer_compress_ratio);
        var gathered: ?mlx.mlx_array = null;
        const qsa_ok = ctx.qsa_blocks.ctx != null and ctx.qsa_mask.ctx == null and !qwen4Standin().attn_sdpa;
        if (qsa_ok and qsaSparseAttnServes(kv_view.has_quant_triple, seq_len)) {
            // One fused dispatch per layer: each row indexes its own selection + tail.
            gathered = try qsaAlignedSparseAttn(self.s, q_rope, &kv_view, ctx.qsa_blocks, ratio, attn_scale, @intCast(self.config.indexer_budget));
            if (self.cost_trace_active and gathered != null) self.cost_attention |= 1;
        }
        if (gathered == null and qsa_ok and seq_len == 1) {
            // Decode width: subset triples (or dense rows) → subset
            // dequant -> dense SDPA over K'<<kv. Declines to the mask arm.
            gathered = try qsaDecodeGatherAttn(self.s, q_rope, &kv_view, ctx.qsa_blocks, ratio, attn_scale);
            if (self.cost_trace_active and gathered != null) self.cost_attention |= 2;
        }
        if (gathered == null and qsa_ok and seq_len >= 2 and seq_len < FUSED256_MIN_Q_LEN) {
            // Verify widths 2..15: the union of the block's rows' selections + tail. The
            // fused kernel's fallback for any width it declines.
            gathered = try qsaVerifyGatherAttn(self.s, q_rope, &kv_view, ctx.qsa_blocks, ratio, attn_scale);
            if (self.cost_trace_active and gathered != null) self.cost_attention |= 4;
        }
        // The prefill kernel has no q_len floor of its own; a verify-width
        // selection must never fall into it (it walks the WHOLE cache).
        if (gathered == null and seq_len >= FUSED256_MIN_Q_LEN and ctx.qsa_blocks.ctx != null and !qwen4Standin().attn_sdpa) {
            const prof = diagEnvOnCached(&qwen4_profile_qsa_env, "QWEN4_PROFILE_QSA");
            var clk: ProfClock = undefined;
            if (prof) {
                try mlx.check(mlx.mlx_array_eval(q_rope));
                clk = ProfClock.init();
            }
            if (kv_view.has_quant_triple and qsaPackedGatherServes(seq_len, mlx.getShape(full_k)[2], mlx.getShape(ctx.qsa_blocks)[2], ratio))
                gathered = try gatherQsa256Packed(self.s, q_rope, &kv_view, attn_scale, ctx.qsa_blocks, ratio);
            if (gathered == null) gathered = try gatherQsa256(self.s, q_rope, full_k, full_v, attn_scale, ctx.qsa_blocks, ratio);
            if (self.cost_trace_active and gathered != null) self.cost_attention |= 8;
            if (prof) if (gathered) |g| {
                try mlx.check(mlx.mlx_array_eval(g));
                log.info("[qsa-prof] gather S={d} kv={d}: {d:.2} ms\n", .{ seq_len, mlx.getShape(full_k)[2], @as(f64, @floatFromInt(clk.lap())) / 1e6 });
            };
        }
        if (gathered) |g| {
            ctx.qsa_arms.noteGather(seq_len);
            _ = mlx.mlx_array_free(attn_out);
            attn_out = g;
        } else {
            // The dense [S, kv] mask arm reads `full_k`/`full_v` — on a
            // quantized cache that is what EVALUATES the whole-range
            // dequant graph `updateAffine` built, per layer per forward.
            ctx.qsa_arms.note(.mask);
            if (self.cost_trace_active) self.cost_attention |= 16;
            if (ctx.qsa_mask.ctx == null) ctx.qsa_mask = try qsaMaskFromBlocks(self.s, ctx.qsa_blocks, mlx.getShape(full_k)[2], ratio);
            if (qwen4Standin().attn_sdpa) {
                _ = mlx.mlx_array_free(attn_out);
                attn_out = try standinRef(q_rope);
            } else if (try fusedSdpa256Masked(self.s, q_rope, full_k, full_v, attn_scale, ctx.qsa_mask)) |fused| {
                _ = mlx.mlx_array_free(attn_out);
                attn_out = fused;
            } else if (try splitMaskedSdpa256(self.s, q_rope, full_k, full_v, attn_scale, ctx.qsa_mask)) |split_out| {
                _ = mlx.mlx_array_free(attn_out);
                attn_out = split_out;
            } else {
                try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "array", ctx.qsa_mask, .{ .ctx = null }, false, self.s));
            }
        }
    } else if (is_prefill) {
        if (try fusedSdpaPrefill(self.s, q_rope, full_k, full_v, attn_scale, 0)) |fused| {
            _ = mlx.mlx_array_free(attn_out);
            attn_out = fused;
        } else if (try splitCausalSdpa(self.s, q_rope, full_k, full_v, attn_scale)) |split_out| {
            // Verify-width (6..9) dense blocks: two vector-path halves
            // beat MLX's internal hd-256 fallback.
            _ = mlx.mlx_array_free(attn_out);
            attn_out = split_out;
        } else {
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "causal", none_mask, .{ .ctx = null }, sdpaForceFused(q_rope, full_k), self.s));
        }
    } else {
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, full_k, full_v, attn_scale, "", none_mask, .{ .ctx = null }, false, self.s));
    }

    if (Qwen4AttnProf.active) Qwen4AttnProf.lap(attn_out, .qsa);

    if (layer == 3) if (Transformer.qwen4_trace) |tr| Qwen4Trace.set(&tr.attn3_pre_tail, attn_out);
    return self.gatedAttnTail(attn_out, gate, fa, flat_shape, skip_output);
}

/// Shared tail of `gatedFullAttnWith`: transpose the attention output back
/// to [B,S,H,D], apply the optional sigmoid output gate, flatten, project.
/// One copy so the batched-decode branch cannot drift from the serial one.
pub fn gatedAttnTail(
    self: *Transformer,
    attn_out: mlx.mlx_array,
    gate: mlx.mlx_array,
    fa: *const FullAttnWeights,
    flat_shape: [3]c_int,
    skip_output: bool,
) !mlx.mlx_array {
    // Transpose back [B,H,S,D] -> [B,S,H,D] (view)
    const perm_back = [_]c_int{ 0, 2, 1, 3 };
    var attn_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn_t);
    try mlx.check(mlx.mlx_transpose_axes(&attn_t, attn_out, &perm_back, 4, self.s));

    // Optional output gating: multiply 4-D (strided inputs are copy-free
    // in the elementwise kernel), then flatten the CONTIGUOUS product —
    // a free view. Flattening attn_t/gate first paid two REAL Copy
    // kernels per call (reshape of a transpose/split view).
    if (self.config.attn_output_gate) {
        var gate_sig = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gate_sig);
        try mlx.check(mlx.mlx_sigmoid(&gate_sig, gate, self.s));
        var gated_4d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gated_4d);
        try mlx.check(mlx.mlx_multiply(&gated_4d, attn_t, gate_sig, self.s));
        var gated = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gated);
        try mlx.check(mlx.mlx_reshape(&gated, gated_4d, &flat_shape, 3, self.s));
        if (skip_output) return standinRef(gated);
        return self.qmatmul(gated, fa.o_w, fa.o_s, fa.o_b);
    }

    var attn_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn_flat);
    try mlx.check(mlx.mlx_reshape(&attn_flat, attn_t, &flat_shape, 3, self.s));
    if (skip_output) return standinRef(attn_flat);
    return self.qmatmul(attn_flat, fa.o_w, fa.o_s, fa.o_b);
}

pub const AttnVerifyInputs = struct {
    q: mlx.mlx_array = .{},
    k: mlx.mlx_array = .{},
    v: mlx.mlx_array = .{},

    pub fn deinit(self: *AttnVerifyInputs) void {
        inline for (.{ "q", "k", "v" }) |name| {
            if (@field(self, name).ctx != null) _ = mlx.mlx_array_free(@field(self, name));
        }
    }

    pub fn slice(self: *const AttnVerifyInputs, s: mlx.mlx_stream, start: c_int, width: c_int) !AttnVerifyInputs {
        var result: AttnVerifyInputs = .{};
        errdefer result.deinit();
        inline for (.{ "q", "k", "v" }) |name| {
            const value = @field(self, name);
            try mlx.check(mlx.mlx_slice(&@field(result, name), value, &.{ 0, start, 0 }, 3, &.{ 1, start + width, mlx.getShape(value)[2] }, 3, &.{ 1, 1, 1 }, 3, s));
        }
        return result;
    }
};

pub const VerifyRowsKind = enum { attention, gdn };

/// `decode`: every input is one slot's single-token tick (`forwardQwen4DecodeRows`), not a verify block.
pub fn verifyRowsJoined(
    self: *Transformer,
    comptime kind: VerifyRowsKind,
    comptime decode: bool,
    inputs: []const mlx.mlx_array,
    entries: []const *SSMCacheEntry,
    ctxs: []const *ForwardCtx,
    generations: []const u64,
    layer: usize,
    w: switch (kind) {
        .attention => *const FullAttnWeights,
        .gdn => *const LinearAttnWeights,
    },
) !?[]mlx.mlx_array {
    if (inputs.len < 2 or inputs.len > 8) return null;
    var widths: [MAX_BATCH_ROWS]c_int = undefined;
    var total: c_int = 0;
    for (inputs, 0..) |input, i| {
        const shape = mlx.getShape(input);
        if (shape.len != 3 or shape[0] != 1 or shape[2] != 2560 or shape[1] < (if (decode) 1 else 2) or shape[1] > 6) return null;
        widths[i] = shape[1];
        total += shape[1];
    }
    if (total > 12) return null;
    const vec = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var joined = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(joined);
    try mlx.check(mlx.mlx_concatenate_axis(&joined, vec, 1, self.s));
    var projections: switch (kind) {
        .attention => AttnVerifyInputs,
        .gdn => Transformer.GdnVerifyInputs,
    } = .{};
    defer projections.deinit();
    inline for (switch (kind) {
        .attention => .{ "q", "k", "v" },
        .gdn => .{ "qkv", "z", "a", "b" },
    }) |name| {
        const part = switch (kind) {
            .attention => try projectRows(self.s, decode, .attention, joined, @field(w, name ++ "_w"), @field(w, name ++ "_s"), @field(w, name ++ "_b"), widths[0..inputs.len]),
            .gdn => try projectRows(self.s, decode, .gdn, joined, @field(w, name ++ "_w"), @field(w, name ++ "_s"), @field(w, name ++ "_b"), widths[0..inputs.len]),
        };
        @field(projections, name) = part orelse return null;
    }
    var flat: [MAX_BATCH_ROWS]mlx.mlx_array = @splat(.{});
    defer for (flat[0..inputs.len]) |value| {
        if (value.ctx != null) _ = mlx.mlx_array_free(value);
    };
    var offset: c_int = 0;
    for (inputs, entries, generations, 0..) |input, entry, generation, i| {
        flat[i] = switch (kind) {
            .attention => blk: {
                var values = try projections.slice(self.s, offset, widths[i]);
                defer values.deinit();
                self.fwd_gen = generation;
                const ctx = ctxs[i];
                break :blk try self.qwen4AttnProjected(ctx, input, w, entry, @intCast(layer), @intCast(ctx.moe_seq_offset.*), 0, 1, widths[i], !decode, &values, true);
            },
            .gdn => blk: {
                const values = try projections.slice(self.s, offset, widths[i]);
                self.fwd_gen = generation;
                break :blk try self.gatedDeltaNetProjected(input, w, entry, layer, 1, widths[i], !decode, values, true);
            },
        };
        offset += widths[i];
    }
    const out_w, const out_s, const out_b = switch (kind) {
        .attention => .{ w.o_w, w.o_s, w.o_b },
        .gdn => .{ w.out_w, w.out_s, w.out_b },
    };
    const flat_vec = mlx.mlx_vector_array_new_data(&flat, inputs.len);
    defer _ = mlx.mlx_vector_array_free(flat_vec);
    var joined_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(joined_flat);
    try mlx.check(mlx.mlx_concatenate_axis(&joined_flat, flat_vec, 1, self.s));
    const projected = try projectRows(self.s, decode, kind, joined_flat, out_w, out_s, out_b, widths[0..inputs.len]);
    defer if (projected) |value| {
        _ = mlx.mlx_array_free(value);
    };
    const result = try self.allocator.alloc(mlx.mlx_array, inputs.len);
    var built: usize = 0;
    errdefer {
        for (result[0..built]) |value| _ = mlx.mlx_array_free(value);
        self.allocator.free(result);
    }
    offset = 0;
    for (inputs, 0..) |_, i| {
        if (projected) |value| {
            result[i] = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(result[i]);
            try mlx.check(mlx.mlx_slice(&result[i], value, &.{ 0, offset, 0 }, 3, &.{ 1, offset + widths[i], mlx.getShape(value)[2] }, 3, &.{ 1, 1, 1 }, 3, self.s));
        } else result[i] = switch (kind) {
            .attention => try self.qmatmul(flat[i], out_w, out_s, out_b),
            .gdn => try self.qmatmul(flat[i], out_w, out_s, out_b),
        };
        built += 1;
        offset += widths[i];
    }
    if (!decode) switch (kind) {
        .attention => {
            mtp_verify_attn_rows_calls +%= 1;
            if (projected != null) mtp_verify_attn_output_calls +%= 1;
        },
        .gdn => {
            mtp_verify_gdn_rows_calls +%= 1;
            if (projected != null) mtp_verify_gdn_output_calls +%= 1;
        },
    };
    const Once = struct {
        const tag = kind;
        const mode = decode;
        var logged = false;
    };
    if (!Once.logged) {
        Once.logged = true;
        log.info("[batched] " ++ switch (kind) {
            .attention => "attention",
            .gdn => "GDN",
        } ++ (if (decode) " decode-row projections engaged (slots={d})\n" else " verify projections engaged (slots={d})\n"), .{inputs.len});
    }
    return result;
}

pub fn attnDecodeRows(self: *Transformer, inputs: []const mlx.mlx_array, entries: []const *SSMCacheEntry, ctxs: []const *ForwardCtx, generations: []const u64, layer: usize, fa: *const FullAttnWeights) !?[]mlx.mlx_array {
    return self.verifyRowsJoined(.attention, true, inputs, entries, ctxs, generations, layer, fa);
}

pub fn attnVerifyRows(self: *Transformer, inputs: []const mlx.mlx_array, entries: []const *SSMCacheEntry, ctxs: []const *ForwardCtx, generations: []const u64, layer: usize, fa: *const FullAttnWeights) !?[]mlx.mlx_array {
    return self.verifyRowsJoined(.attention, false, inputs, entries, ctxs, generations, layer, fa);
}

pub fn projectRows(s: mlx.mlx_stream, comptime decode: bool, comptime kind: VerifyRowsKind, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, widths: []const c_int) !?mlx.mlx_array {
    const pk: VerifyProjectionKind = switch (kind) {
        .attention => .attention,
        .gdn => .gdn,
    };
    return if (decode) decodeRowsProjection(s, x, w, sc, bi, widths) else verifyJoinedProjection(s, x, w, sc, bi, widths, pk);
}

pub fn gdnDecodeRows(self: *Transformer, inputs: []const mlx.mlx_array, entries: []const *SSMCacheEntry, generations: []const u64, layer: usize, la: *const LinearAttnWeights) !?[]mlx.mlx_array {
    return self.verifyRowsJoined(.gdn, true, inputs, entries, &.{}, generations, layer, la);
}

pub fn gdnVerifyRows(self: *Transformer, inputs: []const mlx.mlx_array, entries: []const *SSMCacheEntry, generations: []const u64, layer: usize, la: *const LinearAttnWeights) !?[]mlx.mlx_array {
    return self.verifyRowsJoined(.gdn, false, inputs, entries, &.{}, generations, layer, la);
}

// ── GatedDeltaNet (linear attention layers) ──

pub fn gatedDeltaNet(
    self: *Transformer,
    x: mlx.mlx_array,
    la: *const LinearAttnWeights,
    ssm: *SSMCacheEntry,
    layer_idx: usize,
    batch: c_int,
    seq_len: c_int,
    is_prefill: bool,
) !mlx.mlx_array {
    return self.gatedDeltaNetProjected(x, la, ssm, layer_idx, batch, seq_len, is_prefill, null, false);
}

pub const DeferredQwenMoe = struct {
    moe: Transformer.StreamMoe,
    h: mlx.mlx_array,
    inj: mlx.mlx_array,

    pub fn deinit(self: *DeferredQwenMoe) void {
        self.moe.deinit();
        _ = mlx.mlx_array_free(self.h);
        _ = mlx.mlx_array_free(self.inj);
    }
};

pub fn restoreStreamH(h: *mlx.mlx_array, saved: *mlx.mlx_array) void {
    // Transfer the rollback handle so a later MLX failure leaves cleanup
    // with one live owner and needs no fallible reference allocation.
    _ = mlx.mlx_array_free(h.*);
    h.* = saved.*;
    saved.* = .{ .ctx = null };
}

pub fn verifyQwenMoe(self: *Transformer, deferred: *?DeferredQwenMoe, h: *mlx.mlx_array, pending: *?HcPending, batch: c_int, seq: c_int) !bool {
    const d = &(deferred.* orelse return true);
    // The successor owns different slabs; no cache state from its wrong route
    // is committed until this layer has been verified.
    var kept = false;
    const mw = &self.moe_layers.?[d.moe.stream_ctx.layer].mlp.moe;
    const exact = try self.finishStreamedMoe(&d.moe, mw, &kept);
    defer _ = mlx.mlx_array_free(exact);
    defer {
        d.deinit();
        deferred.* = null;
    }
    noteExpertDeferred(kept);
    if (!kept) {
        if (pending.*) |*pd| pd.deinit();
        pending.* = null;
        restoreStreamH(h, &d.h);
        const next = @as(usize, d.moe.stream_ctx.layer) + 1;
        try self.hcWriteOrDefer(h, exact, d.inj, batch, seq, self.moe_layers.?[next].hc_attn.?.inject_flat, pending);
    }
    return kept;
}

pub fn evalQwen4MtpResident(mtp_head: *const Qwen4Mtp) !void {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    const layer = &mtp_head.layer;
    if (layer.hc_attn) |*weights| appendStructArrays(vec, weights);
    if (layer.hc_mlp) |*weights| appendStructArrays(vec, weights);
    appendHybridMlpWeights(vec, &layer.mlp);
    switch (layer.attn) {
        .full => |weights| appendFullAttnWeights(vec, &weights),
        .linear => |weights| appendLinearAttnWeights(vec, &weights),
    }
    inline for (.{
        mtp_head.pre_norm_emb,
        mtp_head.pre_norm_hidden,
        mtp_head.fc_emb_w,
        mtp_head.fc_emb_s,
        mtp_head.fc_emb_b,
        mtp_head.fc_hid_w,
        mtp_head.fc_hid_s,
        mtp_head.fc_hid_b,
    }) |array| {
        if (array.ctx != null) _ = mlx.mlx_vector_array_append_value(vec, array);
    }
    appendStructArrays(vec, &mtp_head.mixer);
    try mlx.check(mlx.mlx_eval(vec));
}

// ── Utility functions ──

// Self-contained monotonic lap timer (Zig 0.17 has no std.time.Timer;
// the repo times via std.Io). `lap()` returns ns since the previous lap.
/// QWEN4_STANDIN=gdn,attn,mlp,gdn_recur,gdn_proj,attn_qsa,attn_sdpa,hc,moe_shared,moe_router,moe_gateup,moe_down — replace a block
/// with a free stand-in (a +1 ref of its input / a cached ones array) so the
/// in-situ fwd-ubench reports what that block costs. Diagnostic only.
pub const Standin = packed struct(u16) { gdn: bool = false, attn: bool = false, mlp: bool = false, gdn_recur: bool = false, gdn_proj: bool = false, attn_qsa: bool = false, attn_sdpa: bool = false, hc: bool = false, moe_shared: bool = false, moe_router: bool = false, moe_gateup: bool = false, moe_down: bool = false, _pad: u4 = 0 };

pub var standin_cached: ?Standin = null;

pub var qwen4_standin_override: ?Standin = null;

pub fn qwen4Standin() Standin {
    if (qwen4_standin_override) |v| return v;
    if (standin_cached) |v| return v;
    var v = Standin{};
    if (std.c.getenv("QWEN4_STANDIN")) |raw| {
        var it = std.mem.splitScalar(u8, std.mem.sliceTo(raw, 0), ',');
        while (it.next()) |tok| {
            if (std.mem.eql(u8, tok, "gdn")) v.gdn = true;
            if (std.mem.eql(u8, tok, "attn")) v.attn = true;
            if (std.mem.eql(u8, tok, "mlp")) v.mlp = true;
            if (std.mem.eql(u8, tok, "gdn_recur")) v.gdn_recur = true;
            if (std.mem.eql(u8, tok, "gdn_proj")) v.gdn_proj = true;
            if (std.mem.eql(u8, tok, "attn_qsa")) v.attn_qsa = true;
            if (std.mem.eql(u8, tok, "attn_sdpa")) v.attn_sdpa = true;
            if (std.mem.eql(u8, tok, "hc")) v.hc = true;
            if (std.mem.eql(u8, tok, "moe_shared")) v.moe_shared = true;
            if (std.mem.eql(u8, tok, "moe_router")) v.moe_router = true;
            if (std.mem.eql(u8, tok, "moe_gateup")) v.moe_gateup = true;
            if (std.mem.eql(u8, tok, "moe_down")) v.moe_down = true;
        }
    }
    standin_cached = v;
    return v;
}

/// Forward-pass diagnostics for qwen4_exp. QWEN4_PROFILE_FWD=1 (S 2..16) or
/// =all: per-block GPU ms with a sync per block, never on by default;
/// QWEN4_OPCOUNT=1: FFI op attribution per block, no syncs.
pub var qwen4_stream_f32_env: ?bool = null;

pub const Qwen4FwdProf = struct {
    const Block = enum(u8) { ple, hc_read, gdn, attn, hc_write, mlp };
    timing: bool,
    ops: bool,
    clock: ProfClock,
    ns: [6]u64 = @splat(0),
    layer_ns: u64 = 0,
    kind_ns: [3]u64 = @splat(0), // ple layer, gdn layers, attn layers
    op_mark: u64 = 0,
    op_ns: [6]u64 = @splat(0),

    pub fn init(seq_len: c_int, h: mlx.mlx_array) !Qwen4FwdProf {
        const env = if (diagEnvOnCached(&qwen4_profile_fwd_env, "QWEN4_PROFILE_FWD")) std.c.getenv("QWEN4_PROFILE_FWD") else null;
        // `=1`: S 2..16 (verify). `=all`: any S, including decode S=1 and prefill.
        const all = env != null and env.?[0] == 'a';
        const timing = env != null and (all or (seq_len >= 2 and seq_len <= 16));
        var p: Qwen4FwdProf = .{ .timing = timing, .ops = diagEnvOnCached(&qwen4_opcount_env, "QWEN4_OPCOUNT"), .clock = ProfClock.init() };
        if (p.ops) {
            trf.moe_rows_fused_layers = 0;
            trf.moe_sorted_layers = 0;
        }
        if (timing) {
            try mlx.check(mlx.mlx_array_eval(h));
            _ = p.clock.lap();
        }
        if (p.ops) p.op_mark = mlx.op_count.load(.monotonic);
        return p;
    }

    pub fn lap(self: *Qwen4FwdProf, arr: mlx.mlx_array, block: Block) !void {
        if (self.ops) {
            const now = mlx.op_count.load(.monotonic);
            self.op_ns[@backingInt(block)] += now - self.op_mark;
            self.op_mark = now;
        }
        if (!self.timing) return;
        try mlx.check(mlx.mlx_array_eval(arr));
        const ns = self.clock.lap();
        self.ns[@backingInt(block)] += ns;
        self.layer_ns += ns;
    }

    pub fn endLayer(self: *Qwen4FwdProf, kind: Block) void {
        const slot: usize = switch (kind) {
            .ple => 0,
            .gdn => 1,
            else => 2,
        };
        self.kind_ns[slot] += self.layer_ns;
        self.layer_ns = 0;
    }

    pub fn report(self: *const Qwen4FwdProf, seq_len: c_int, kv: usize, capture: bool, gdn_layers: u32, attn_layers: u32) void {
        if (self.ops) log.info("[qwen4-ops] S={d} ple {d} hcRead {d} gdn {d} attn {d} hcWrite {d} mlp {d} moeFused {d} moeSorted {d}\n", .{ seq_len, self.op_ns[0], self.op_ns[1], self.op_ns[2], self.op_ns[3], self.op_ns[4], self.op_ns[5], trf.moe_rows_fused_layers, trf.moe_sorted_layers });
        if (!self.timing) return;
        const ms = struct {
            fn f(n: u64) f64 {
                return @as(f64, @floatFromInt(n)) / 1e6;
            }
        }.f;
        log.info("[qwen4-prof] S={d} kv={d} capture={} gdn {d:.3} ms ({d} us, {d} layers, {d:.3}/layer) attn {d:.3} ms ({d} us, {d} layers, {d:.3}/layer) ple-layer {d:.3} ms ({d} us)\n", .{
            seq_len,
            kv,
            capture,
            ms(self.kind_ns[1]),
            self.kind_ns[1] / 1000,
            gdn_layers,
            ms(self.kind_ns[1]) / @as(f64, @floatFromInt(@max(gdn_layers, 1))),
            ms(self.kind_ns[2]),
            self.kind_ns[2] / 1000,
            attn_layers,
            ms(self.kind_ns[2]) / @as(f64, @floatFromInt(@max(attn_layers, 1))),
            ms(self.kind_ns[0]),
            self.kind_ns[0] / 1000,
        });
        log.info("[qwen4-prof] blocks (whole forward): ple {d:.3} ms / {d} us  hcRead {d:.3} ms / {d} us  gdn {d:.3} ms / {d} us  attn {d:.3} ms / {d} us  hcWrite {d:.3} ms / {d} us  mlp {d:.3} ms / {d} us\n", .{
            ms(self.ns[0]), self.ns[0] / 1000,
            ms(self.ns[1]), self.ns[1] / 1000,
            ms(self.ns[2]), self.ns[2] / 1000,
            ms(self.ns[3]), self.ns[3] / 1000,
            ms(self.ns[4]), self.ns[4] / 1000,
            ms(self.ns[5]), self.ns[5] / 1000,
        });
    }
};

pub const Qwen4AttnProf = struct {
    pub const Stage = enum(u2) { indexer, proj, qsa, tail };
    pub var on: ?bool = null;
    pub var active = false;
    pub var clock: ProfClock = undefined;
    pub var ns: [4]u64 = @splat(0);
    pub var layers: u32 = 0;

    pub fn sync(arr: mlx.mlx_array) void {
        if (arr.ctx == null) return;
        mlx.check(mlx.mlx_array_eval(arr)) catch {};
    }

    pub fn begin(x: mlx.mlx_array, seq_len: c_int) bool {
        if (on == null) on = diagEnvOn("SUSHI_PROFILE_ATTN");
        if (!on.? or seq_len <= 16) return false;
        sync(x);
        clock = ProfClock.init();
        active = true;
        layers += 1;
        return true;
    }

    pub fn lap(arr: mlx.mlx_array, stage: Stage) void {
        sync(arr);
        ns[@backingInt(stage)] += clock.lap();
    }

    pub fn flush(seq_len: c_int, kv: usize) void {
        if (layers == 0) return;
        const ms = struct {
            fn f(n: u64) f64 {
                return @as(f64, @floatFromInt(n)) / 1e6;
            }
        }.f;
        log.info("[qwen4-attn] S={d} kv={d} layers={d} indexer {d:.3} ms  proj+rope+kv {d:.3} ms  qsa {d:.3} ms  gate+o_proj {d:.3} ms\n", .{ seq_len, kv, layers, ms(ns[0]), ms(ns[1]), ms(ns[2]), ms(ns[3]) });
        reset();
    }

    pub fn reset() void {
        active = false;
        ns = @splat(0);
        layers = 0;
    }
};

/// Caches for the qwen4 per-layer diagnostic switches: a `getenv` per layer per forward is a libc scan of the environ block.
pub var qwen4_no_pooled_env: ?bool = null;

pub var qwen4_debug_scores_env: ?bool = null;

pub var qwen4_profile_fwd_env: ?bool = null;

pub var qwen4_profile_qsa_env: ?bool = null;

var qwen4_opcount_env: ?bool = null;

pub fn getHcFusedKernel(which: usize) !mlx.mlx_fast_metal_kernel {
    if (qwen4_hc.hc_fused_kernels[which]) |k| return k;
    const n_inputs = [_][*:0]const u8{ "x_in", "nw", "iw", "eps", "wo_in", "wi_in" };
    const n_outputs = [_][*:0]const u8{ "xn_out", "ipart_out", "xs_out" };
    const d_inputs = [_][*:0]const u8{ "xn_in", "dw_q", "dw_s", "dw_b", "ipart_in" };
    const d_outputs = [_][*:0]const u8{ "act_out", "inj_out" };
    const u_inputs = [_][*:0]const u8{ "xn_in", "act_in", "uw_q", "uw_s", "uw_b" };
    const u_outputs = [_][*:0]const u8{"mixed_out"};
    const inputs: []const [*:0]const u8 = switch (which) {
        0 => &n_inputs,
        1 => &d_inputs,
        else => &u_inputs,
    };
    const outputs: []const [*:0]const u8 = switch (which) {
        0 => &n_outputs,
        1 => &d_outputs,
        else => &u_outputs,
    };
    const in_vec = mlx.mlx_vector_string_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(outputs.ptr, outputs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        switch (which) {
            0 => "sushi_hc_read_n",
            1 => "sushi_hc_read_d",
            else => "sushi_hc_read_u",
        },
        in_vec,
        out_vec,
        switch (which) {
            0 => HC_FUSED_N_SOURCE,
            1 => HC_FUSED_D_SOURCE,
            else => HC_FUSED_U_SOURCE,
        },
        "",
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    qwen4_hc.hc_fused_kernels[which] = kernel;
    return kernel;
}

pub var qwen4_batched_ladder_engaged = false;

pub fn verifyLmHeadProjection(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, widths: []const c_int) !?mlx.mlx_array {
    return verifyJoinedProjection(s, x, w, sc, bi, widths, .lm_head);
}

/// force_fused is legal only where MLX has a fused kernel for the shape:
/// hd 256 (<= 128 fuses on its own, 192 is not ours), more than 8 query
/// rows (<= 8 is the vector kernel and its q*gqa <= 32 wall makes
/// force_fused THROW), q <= k (the full kernel's causal precondition).
pub fn sdpaForceFusedFor(preferred: bool, q_len: c_int, k_len: c_int, head_dim: c_int) bool {
    return preferred and head_dim == 256 and q_len > 8 and q_len <= k_len;
}

pub fn sdpaForceFused(q: mlx.mlx_array, k: mlx.mlx_array) bool {
    if (!naxSdpaPreferred()) return false;
    if (mlx.mlx_array_ndim(q) != 4 or mlx.mlx_array_ndim(k) != 4) return false;
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    const on = sdpaForceFusedFor(true, qs[2], ks[2], qs[3]);
    if (on and !nax_sdpa_dispatch_logged) {
        nax_sdpa_dispatch_logged = true;
        log.info("[nax-sdpa] force_fused dispatch engaged: qL={d} kL={d} Hq={d} Hkv={d} hd={d}\n", .{ qs[2], ks[2], qs[1], ks[1], qs[3] });
    }
    return on;
}
var nax_sdpa_dispatch_logged = false;

/// One engagement log per width (6..9): the first engagement is usually the
/// warmup's own 8-token prefill (kL==qL==8) — a per-width line is what
/// witnesses the split at REAL verify shapes in an A/B arm's log.
var sdpa_split_logged: [4]bool = .{ false, false, false, false };

fn sdpaTickBoundary(length: c_int) bool {
    return switch (length) {
        1024, 1025, 4096, 8193, 16384, 32769, 65536, 65537 => true,
        else => false,
    };
}

var sdpa_tick_identical_logged = false;

// Each group must retain the decode tick's one/two-pass kernel and split count.
fn sdpaTickIdenticalGroups(s: mlx.mlx_stream, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, scale: f32) !mlx.mlx_array {
    const width = mlx.getShape(q)[2];
    const kv_len = mlx.getShape(k)[2];
    var parts: [9]mlx.mlx_array = undefined;
    var count: usize = 0;
    defer for (parts[0..count]) |part| {
        _ = mlx.mlx_array_free(part);
    };
    var start: c_int = 0;
    while (start < width) {
        var end = @min(start + 2, width);
        const first_kv = kv_len - width + start + 1;
        if (end - start == 2 and sdpaTickBoundary(first_kv + 1)) end = start + 1;
        const end_kv = kv_len - width + end;
        const qr = try sliceAttentionSeq(s, q, start, end);
        defer _ = mlx.mlx_array_free(qr);
        const kr = try sliceAttentionSeq(s, k, 0, end_kv);
        defer _ = mlx.mlx_array_free(kr);
        const vr = try sliceAttentionSeq(s, v, 0, end_kv);
        defer _ = mlx.mlx_array_free(vr);
        parts[count] = mlx.mlx_array_new();
        count += 1;
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&parts[count - 1], qr, kr, vr, scale, if (end - start == 1) "" else "causal", .{ .ctx = null }, .{ .ctx = null }, false, s));
        start = end;
    }
    const vec = mlx.mlx_vector_array_new_data(&parts, count);
    defer _ = mlx.mlx_vector_array_free(vec);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 2, s));
    if (!sdpa_tick_identical_logged) {
        sdpa_tick_identical_logged = true;
        log.info("[sdpa-split] row-identical causal arm engaged: qL={d} kL={d} GQA=12\n", .{ width, kv_len });
    }
    return out;
}

pub fn splitCausalSdpa(
    s: mlx.mlx_stream,
    q: mlx.mlx_array,
    k: mlx.mlx_array,
    v: mlx.mlx_array,
    scale: f32,
) !?mlx.mlx_array {
    if (mlx.mlx_array_ndim(q) != 4 or mlx.mlx_array_ndim(k) != 4 or mlx.mlx_array_ndim(v) != 4) return null;
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    const vs = mlx.getShape(v);
    if (qs[3] != 256 or ks[3] != 256 or vs[3] != 256) return null;
    if (qs[0] != 1 or ks[0] != 1 or vs[0] != 1) return null;
    if (ks[2] < qs[2] or ks[2] != vs[2] or ks[1] != vs[1]) return null;
    if (ks[1] > 0 and qs[1] == 12 * ks[1] and qs[2] >= 2 and qs[2] <= 9 and mlx.streamIsGpu(s)) {
        return try sdpaTickIdenticalGroups(s, q, k, v, scale);
    }
    if (qs[2] < 6 or qs[2] > 9) return null;

    const qL = qs[2];
    const kL = ks[2];
    const split: c_int = 5;
    const k_split: c_int = kL - (qL - split);
    const strides = [_]c_int{ 1, 1, 1, 1 };

    // Chunk A: rows 0..<5 vs keys[0 .. kL-(qL-5)] — row i's bottom-right
    // causal window is exactly the one a width-5 round at this offset gets.
    var q_a = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_a);
    var k_a = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_a);
    var v_a = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v_a);
    const zero4 = [_]c_int{ 0, 0, 0, 0 };
    const qa_stop = [_]c_int{ 1, qs[1], split, 256 };
    const ka_stop = [_]c_int{ 1, ks[1], k_split, 256 };
    try mlx.check(mlx.mlx_slice(&q_a, q, &zero4, 4, &qa_stop, 4, &strides, 4, s));
    try mlx.check(mlx.mlx_slice(&k_a, k, &zero4, 4, &ka_stop, 4, &strides, 4, s));
    try mlx.check(mlx.mlx_slice(&v_a, v, &zero4, 4, &ka_stop, 4, &strides, 4, s));

    // Chunk B: rows 5.. vs the full keys — the follow-up width-(qL-5) round.
    var q_b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_b);
    const qb_start = [_]c_int{ 0, 0, split, 0 };
    const qb_stop = [_]c_int{ 1, qs[1], qL, 256 };
    try mlx.check(mlx.mlx_slice(&q_b, q, &qb_start, 4, &qb_stop, 4, &strides, 4, s));

    const none_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(none_mask);
    var out_a = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(out_a);
    var out_b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(out_b);
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&out_a, q_a, k_a, v_a, scale, "causal", none_mask, .{ .ctx = null }, false, s));
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&out_b, q_b, k, v, scale, "causal", none_mask, .{ .ctx = null }, false, s));

    const parts = [_]mlx.mlx_array{ out_a, out_b };
    const vec = mlx.mlx_vector_array_new_data(&parts, parts.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 2, s));

    const log_idx: usize = @intCast(qL - 6);
    if (!sdpa_split_logged[log_idx]) {
        sdpa_split_logged[log_idx] = true;
        log.info("[sdpa-split] engaged: qL={d} kL={d} Hq={d} Hkv={d}\n", .{ qL, kL, qs[1], ks[1] });
    }
    return out;
}

var sdpa_mask_split_logged = false;

/// Array-mask verify widths on hd 256 (qwen4_exp QSA past the budget): MLX's
/// vector kernel serves `qL * gqa <= 32` rows, anything wider (S >= 3 at
/// gqa 12) and every array-masked qL 9..15 is the unfused fallback. Rows are
/// independent under a per-row mask, so slice the block into vector-sized row
/// groups.
pub fn splitMaskedSdpa256(
    s: mlx.mlx_stream,
    q: mlx.mlx_array,
    k: mlx.mlx_array,
    v: mlx.mlx_array,
    scale: f32,
    mask: mlx.mlx_array,
) !?mlx.mlx_array {
    if (mlx.mlx_array_ndim(q) != 4 or mlx.mlx_array_ndim(k) != 4 or mlx.mlx_array_ndim(mask) != 4) return null;
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    const ms = mlx.getShape(mask);
    if (qs[3] != 256 or ks[3] != 256) return null;
    if (qs[0] != 1 or ks[0] != 1 or ms[0] != 1 or ms[1] != 1 or ms[2] != qs[2] or ms[3] != ks[2]) return null;
    const qL = qs[2];
    if (qL < 2 or qL >= FUSED256_MIN_Q_LEN) return null;
    const gqa: c_int = @divTrunc(qs[1], ks[1]);
    if (qL * gqa <= 32) return null;
    const group: c_int = @max(1, @divTrunc(32, gqa));

    var parts = std.ArrayList(mlx.mlx_array).empty;
    defer {
        for (parts.items) |a| _ = mlx.mlx_array_free(a);
        parts.deinit(std.heap.c_allocator);
    }
    const strides = [_]c_int{ 1, 1, 1, 1 };
    var r0: c_int = 0;
    while (r0 < qL) : (r0 += group) {
        const r1: c_int = @min(r0 + group, qL);
        var q_g = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_g);
        var m_g = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(m_g);
        try mlx.check(mlx.mlx_slice(&q_g, q, &[_]c_int{ 0, 0, r0, 0 }, 4, &[_]c_int{ 1, qs[1], r1, 256 }, 4, &strides, 4, s));
        try mlx.check(mlx.mlx_slice(&m_g, mask, &[_]c_int{ 0, 0, r0, 0 }, 4, &[_]c_int{ 1, 1, r1, ms[3] }, 4, &strides, 4, s));
        var o_g = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(o_g);
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&o_g, q_g, k, v, scale, "array", m_g, .{ .ctx = null }, false, s));
        try parts.append(std.heap.c_allocator, o_g);
    }
    const vec = mlx.mlx_vector_array_new_data(parts.items.ptr, parts.items.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 2, s));
    if (!sdpa_mask_split_logged) {
        sdpa_mask_split_logged = true;
        log.info("[sdpa-split] masked arm engaged: qL={d} kL={d} gqa={d} rows/group={d}\n", .{ qL, ks[2], gqa, group });
    }
    return out;
}

/// This checkpoint's YaRN spectrum: the config's rope geometry (theta, the
/// rotating slice, the pre-trained window) resolved into `mrope.Yarn`, the one
/// implementation of HF's `_compute_yarn_parameters` in the engine. Only
/// meaningful when `config.rope_yarn`.
pub fn yarnSpec(config: *const ModelConfig) mrope.Yarn {
    return .{
        .theta = config.rope_theta,
        .factor = config.yarn_factor,
        .orig_max = config.yarn_orig_max_pos,
        .rotary_dim = config.yarnRotaryDim(),
        .beta_fast = config.yarn_beta_fast,
        .beta_slow = config.yarn_beta_slow,
        .truncate = config.yarn_truncate,
    };
}

/// The mscale the host cos/sin tables are multiplied by: HF's `attention_factor`
/// when the checkpoint scales its rope, 1.0 (a no-op) when it does not.
pub fn yarnMscale(config: *const ModelConfig) f64 {
    return if (config.rope_yarn) config.yarn_attention_factor else 1.0;
}

/// HF `_compute_yarn_parameters` → the per-dim RoPE DENOMINATOR array (`freqs`)
/// that mlx_fast_rope consumes: mlx computes angle = position / freqs[i], so we
/// return 1/inv_freq where inv_freq is the YaRN-corrected inverse frequency.
/// `out.len` must equal the rotary half-dim (= int(head_dim * partial) / 2).
/// Pure f64 math (no MLX) so it unit-tests directly against reference values.
/// The spectrum itself lives in `mrope.Yarn` — shared with the host cos/sin
/// tables, which need inv_freq and not the reciprocal — so the device and host
/// rope paths can never drift apart. `truncate` is the HF flag (an omitted
/// flag means true).
pub fn computeYarnFreqs(
    out: []f64,
    head_dim: u32,
    partial: f32,
    base: f64,
    factor: f64,
    beta_fast: f64,
    beta_slow: f64,
    orig_max: u32,
    truncate: bool,
) void {
    const spec = mrope.Yarn{
        .theta = base,
        .factor = factor,
        .orig_max = orig_max,
        .rotary_dim = @intFromFloat(@floor(@as(f64, @floatFromInt(head_dim)) * @as(f64, partial))),
        .beta_fast = beta_fast,
        .beta_slow = beta_slow,
        .truncate = truncate,
    };
    spec.invFreq(out);
    for (out) |*v| v.* = 1.0 / v.*;
}

var qk_fused_kernel: ?mlx.mlx_fast_metal_kernel = null;
var qk_fused_engaged: bool = false;

// The RMS half mirrors `rms_single_row` at axis 128 / N_READS 4 / one
// simdgroup (lane l owns [4l, 4l+4), f32 squares in index order, `simd_sum`,
// `precise::rsqrt`); the T(x*inv) rounding inside the w-multiply is the value
// the separate kernel would have written and the rotation would have read
// back. The rotation mirrors `rope_single_impl<T, false>`: pair p couples
// elements p and p+RD/2, partner values arrive by simd_shuffle (RD/8 lanes
// apart), f32 arithmetic, one T rounding — then the optional mscale multiply
// reproduces the composed path's separate broadcast multiply (one more T
// rounding, table value 1.0 on the pass-through tail).
const QK_NORM_ROPE_SOURCE =
    \\constexpr uint HD = 128;
    \\uint head = threadgroup_position_in_grid.x;
    \\uint lane = thread_position_in_threadgroup.x;
    \\bool is_q = head < uint(HQ);
    \\const device T* src = is_q ? (q + head * HD) : (k + (head - uint(HQ)) * HD);
    \\const device T* w = is_q ? qw : kw;
    \\device T* dst = is_q ? (oq + head * HD) : (ok + (head - uint(HQ)) * HD);
    \\uint base = lane * 4;
    \\float acc = 0.0f;
    \\for (uint i = 0; i < 4; ++i) {
    \\    float v = float(src[base + i]);
    \\    acc += v * v;
    \\}
    \\acc = simd_sum(acc);
    \\float inv = metal::precise::rsqrt(acc / float(HD) + float(eps));
    \\T nrm[4];
    \\for (uint i = 0; i < 4; ++i) {
    \\    nrm[i] = w[base + i] * T(float(src[base + i]) * inv);
    \\}
    \\constexpr uint half_rd = uint(RD) / 2;
    \\float partner[4];
    \\for (uint i = 0; i < 4; ++i) {
    \\    partner[i] = simd_shuffle(float(nrm[i]), lane ^ (half_rd / 4));
    \\}
    \\for (uint i = 0; i < 4; ++i) {
    \\    uint j = base + i;
    \\    T r;
    \\    if (j < half_rd) {
    \\        r = T(float(nrm[i]) * angles[j] - partner[i] * angles[half_rd + j]);
    \\    } else if (j < uint(RD)) {
    \\        r = T(partner[i] * angles[j] + float(nrm[i]) * angles[j - half_rd]);
    \\    } else {
    \\        r = nrm[i];
    \\    }
    \\    if (SCALED != 0) {
    \\        r = r * ms[j];
    \\    }
    \\    dst[j] = r;
    \\}
;

fn getQkNormRopeKernel() !mlx.mlx_fast_metal_kernel {
    if (qk_fused_kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "q", "k", "qw", "kw", "angles", "ms", "eps" };
    const output_names = [_][*:0]const u8{ "oq", "ok" };
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "sushi_qk_norm_rope",
        in_vec,
        out_vec,
        QK_NORM_ROPE_SOURCE,
        "",
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    qk_fused_kernel = kernel;
    return kernel;
}

const QkFusedCfgKey = struct { hq: c_int, hk: c_int, rd: c_int, scaled: bool, dtype: mlx.mlx_dtype };
var qk_fused_cfg: ?mlx.mlx_fast_metal_kernel_config = null;
var qk_fused_cfg_key: QkFusedCfgKey = std.mem.zeroes(QkFusedCfgKey);

/// One fused dispatch: per-head RMSNorm + RoPE (+ optional mscale) for q AND
/// k, emitting the transposed [1,H,1,128] layout attention consumes (at L==1
/// that is the flat buffer relabeled, so the transposes vanish too).
/// `angles` is the [RD] f32 cos|sin row from `ropeAngleRow`. Null → caller
/// keeps the composed chain. Shared with tts.zig's talker decode (hd 128).
pub fn fusedQkNormRope(
    s: mlx.mlx_stream,
    q_flat: mlx.mlx_array, // [1, 1, HQ*128]
    k_flat: mlx.mlx_array, // [1, 1, HK*128]
    q_norm_w: mlx.mlx_array,
    k_norm_w: mlx.mlx_array,
    angles: mlx.mlx_array,
    ms: ?mlx.mlx_array, // bf16 [128] mscale table, null = unscaled
    eps_arr: mlx.mlx_array, // 0-dim f32 SCALAR (binds `constant float&`; a [1] array binds a pointer and the kernel fails to compile)
    hq: c_int,
    hk: c_int,
    rd: c_int,
) !?[2]mlx.mlx_array {
    const dt = mlx.mlx_array_dtype(q_flat);
    if (dt != .bfloat16 or mlx.mlx_array_dtype(k_flat) != .bfloat16) return null;
    // rd=32 is a partial-rotary geometry (0.25 x 128); the
    // shuffle mapping (lane ^ (half_rd/4)) is exact for all three widths.
    if (rd != 32 and rd != 64 and rd != 128) return null;
    if (q_norm_w.ctx == null or k_norm_w.ctx == null) return null;

    const key = QkFusedCfgKey{ .hq = hq, .hk = hk, .rd = rd, .scaled = ms != null, .dtype = dt };
    if (qk_fused_cfg == null or !std.meta.eql(qk_fused_cfg_key, key)) {
        if (qk_fused_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const config = mlx.mlx_fast_metal_kernel_config_new();
        const oq_shape = [_]c_int{ 1, hq, 1, 128 };
        const ok_shape = [_]c_int{ 1, hk, 1, 128 };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &oq_shape, 4, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &ok_shape, 4, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, (hq + hk) * 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "HQ", hq));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "RD", rd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "SCALED", if (ms != null) 1 else 0));
        qk_fused_cfg = config;
        qk_fused_cfg_key = key;
    }

    const kernel = try getQkNormRopeKernel();
    // The unscaled variant never reads `ms` (SCALED == 0 guards the index),
    // but the input slot must be bound — the norm weight stands in.
    const inputs_arr = [_]mlx.mlx_array{ q_flat, k_flat, q_norm_w, k_norm_w, angles, ms orelse q_norm_w, eps_arr };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, qk_fused_cfg.?, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 2) return error.MetalKernelBadOutputCount;
    var oq = mlx.mlx_array_new();
    var ok = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&oq, outputs_vec, 0));
    try mlx.check(mlx.mlx_vector_array_get(&ok, outputs_vec, 1));
    if (!qk_fused_engaged) {
        qk_fused_engaged = true;
        log.info("[attn] fused QK-norm+RoPE engaged: hq={d} hk={d} rd={d} scaled={}\n", .{ hq, hk, rd, ms != null });
    }
    return .{ oq, ok };
}







/// Extract the exact cos|sin row the stock RoPE kernel computes at `offset`:
/// a [1,1,1,rd] f32 probe with ones in the first half rotates to
/// [cos(theta_p) | sin(theta_p)] — the same floats, not a re-derivation (the
/// angle math inside `mlx_fast_rope` is float regardless of the tensor
/// dtype, so the f32 probe reads it out exactly).
pub fn ropeAngleRow(
    s: mlx.mlx_stream,
    rd: c_int,
    base: mlx.mlx_optional_float,
    freqs: mlx.mlx_array,
    offset: c_int,
    allocator: std.mem.Allocator,
) !mlx.mlx_array {
    return ropeAngleRows(s, rd, base, freqs, offset, 1, allocator, 1.0);
}

const QK_NORM_ROPE_256_SOURCE =
    \\constexpr uint HD = 256;
    \\constexpr uint half_rd = uint(RD) / 2;
    \\uint row = threadgroup_position_in_grid.x;
    \\uint tid = thread_position_in_threadgroup.x;
    \\uint simd_lane = thread_index_in_simdgroup;
    \\uint simd_group = simdgroup_index_in_threadgroup;
    \\uint q_rows = uint(HQ) * uint(S);
    \\bool is_q = row < q_rows;
    \\uint local_row = is_q ? row : row - q_rows;
    \\uint head = local_row / uint(S);
    \\uint seq = local_row % uint(S);
    \\// Inputs are row-contiguous [1, S, H, 256]; outputs head-major [1, H, S, 256].
    \\const device T* src = is_q ? (q + (seq * uint(HQ) + head) * HD)
    \\                           : (k + (seq * uint(HK) + head) * HD);
    \\const device T* w = is_q ? qw : kw;
    \\device T* dst = is_q ? (oq + local_row * HD) : (ok + local_row * HD);
    \\uint base = tid * 4;
    \\threadgroup float local_sums[32];
    \\threadgroup float local_inv[1];
    \\threadgroup T nrm[HD];
    \\float acc = 0.0f;
    \\for (uint i = 0; i < 4; ++i) {
    \\    float v = float(src[base + i]);
    \\    acc += v * v;
    \\}
    \\acc = simd_sum(acc);
    \\if (simd_group == 0) {
    \\    local_sums[simd_lane] = 0.0f;
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (simd_lane == 0) {
    \\    local_sums[simd_group] = acc;
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (simd_group == 0) {
    \\    acc = simd_sum(local_sums[simd_lane]);
    \\    if (simd_lane == 0) {
    \\        local_inv[0] = metal::precise::rsqrt(acc / float(HD) + float(eps));
    \\    }
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\float inv = local_inv[0];
    \\for (uint i = 0; i < 4; ++i) {
    \\    nrm[base + i] = w[base + i] * T(float(src[base + i]) * inv);
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\const device float* arow = angles + seq * uint(RD);
    \\for (uint i = 0; i < 4; ++i) {
    \\    uint j = base + i;
    \\    T r;
    \\    if (j < half_rd) {
    \\        r = T(float(nrm[j]) * arow[j] - float(nrm[j + half_rd]) * arow[half_rd + j]);
    \\    } else if (j < uint(RD)) {
    \\        r = T(float(nrm[j - half_rd]) * arow[j] + float(nrm[j]) * arow[j - half_rd]);
    \\    } else {
    \\        r = nrm[j];
    \\    }
    \\    dst[j] = r;
    \\}
;

var qk256_fused_kernel: ?mlx.mlx_fast_metal_kernel = null;
var qk256_fused_engaged: bool = false;

fn getQkNormRope256Kernel() !mlx.mlx_fast_metal_kernel {
    if (qk256_fused_kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "q", "k", "qw", "kw", "angles", "eps" };
    const output_names = [_][*:0]const u8{ "oq", "ok" };
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "sushi_qk_norm_rope_256",
        in_vec,
        out_vec,
        QK_NORM_ROPE_256_SOURCE,
        "",
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    qk256_fused_kernel = kernel;
    return kernel;
}

const Qk256CfgKey = struct { hq: c_int, hk: c_int, seq: c_int, rd: c_int, dtype: mlx.mlx_dtype };
var qk256_fused_cfg: ?mlx.mlx_fast_metal_kernel_config = null;
var qk256_fused_cfg_key: Qk256CfgKey = std.mem.zeroes(Qk256CfgKey);

/// One fused dispatch for hd-256 heads at S in 1..16: per-head RMSNorm +
/// partial RoPE for q AND k, emitting the transposed [1,H,S,256] layout.
/// `angles` is the [S*RD] f32 table from `ropeAngleRows` (row p = position
/// offset+p). Null → caller keeps the composed chain.
pub fn fusedQkNormRope256(
    s: mlx.mlx_stream,
    q_in: mlx.mlx_array, // [1, S, HQ, 256] (view ok — ensure_row_contiguous materializes)
    k_in: mlx.mlx_array, // [1, S, HK, 256]
    q_norm_w: mlx.mlx_array,
    k_norm_w: mlx.mlx_array,
    angles: mlx.mlx_array,
    eps_arr: mlx.mlx_array, // 0-dim f32 scalar
    hq: c_int,
    hk: c_int,
    seq: c_int,
    rd: c_int,
) !?[2]mlx.mlx_array {
    if (seq < 1 or seq > 32) return null;
    const dt = mlx.mlx_array_dtype(q_in);
    if (dt != .bfloat16 or mlx.mlx_array_dtype(k_in) != .bfloat16) return null;
    // The shuffle-free staged rotation is exact for any even rd <= 256 whose
    // pair partner stays inside the row; the served geometry is rd=64
    // (partial factor 0.25 x 256).
    if (rd != 32 and rd != 64 and rd != 128) return null;
    if (q_norm_w.ctx == null or k_norm_w.ctx == null) return null;

    const key = Qk256CfgKey{ .hq = hq, .hk = hk, .seq = seq, .rd = rd, .dtype = dt };
    if (qk256_fused_cfg == null or !std.meta.eql(qk256_fused_cfg_key, key)) {
        if (qk256_fused_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const config = mlx.mlx_fast_metal_kernel_config_new();
        const oq_shape = [_]c_int{ 1, hq, seq, 256 };
        const ok_shape = [_]c_int{ 1, hk, seq, 256 };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &oq_shape, 4, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &ok_shape, 4, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, (hq + hk) * seq * 64, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 64, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "HQ", hq));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "HK", hk));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "S", seq));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "RD", rd));
        qk256_fused_cfg = config;
        qk256_fused_cfg_key = key;
    }

    const kernel = try getQkNormRope256Kernel();
    const inputs_arr = [_]mlx.mlx_array{ q_in, k_in, q_norm_w, k_norm_w, angles, eps_arr };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, qk256_fused_cfg.?, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 2) return error.MetalKernelBadOutputCount;
    var oq = mlx.mlx_array_new();
    var ok = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&oq, outputs_vec, 0));
    try mlx.check(mlx.mlx_vector_array_get(&ok, outputs_vec, 1));
    if (!qk256_fused_engaged) {
        qk256_fused_engaged = true;
        log.info("[attn] fused QK-norm+RoPE (hd-256) engaged: hq={d} hk={d} S={d} rd={d}\n", .{ hq, hk, seq, rd });
    }
    return .{ oq, ok };
}

/// Like attn256MaxDiff but errors on any non-finite element on either side —
/// NaN > max_err is false, so an all-NaN pair would otherwise diff to 0.
fn attn256FiniteMaxDiff(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    var a32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a32);
    var b32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b32);
    try mlx.check(mlx.mlx_astype(&a32, a, .float32, s));
    try mlx.check(mlx.mlx_astype(&b32, b, .float32, s));
    try mlx.check(mlx.mlx_array_eval(a32));
    try mlx.check(mlx.mlx_array_eval(b32));
    const ad = mlx.mlx_array_data_float32(a32) orelse return error.InvalidDtype;
    const bd = mlx.mlx_array_data_float32(b32) orelse return error.InvalidDtype;
    const n = mlx.mlx_array_size(a32);
    if (n != mlx.mlx_array_size(b32)) return error.ShapeMismatch;
    var max_diff: f32 = 0;
    for (0..n) |i| {
        if (!std.math.isFinite(ad[i]) or !std.math.isFinite(bd[i])) return error.NonFinite;
        max_diff = @max(max_diff, @abs(ad[i] - bd[i]));
    }
    return max_diff;
}

/// Composed reference for the fused QK-norm+RoPE tests: the EXACT shipped
/// chain (reshape → fast_rms_norm → transpose → fast_rope → optional bf16
/// mscale multiply), returning [1, H, 1, 128].
fn qkComposedRef(
    s: mlx.mlx_stream,
    flat: mlx.mlx_array,
    norm_w: mlx.mlx_array,
    ms: ?mlx.mlx_array,
    h: c_int,
    rd: c_int,
    base: mlx.mlx_optional_float,
    freqs: mlx.mlx_array,
    offset: c_int,
    eps: f32,
) !mlx.mlx_array {
    const shape4 = [_]c_int{ 1, 1, h, 128 };
    var r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r);
    try mlx.check(mlx.mlx_reshape(&r, flat, &shape4, 4, s));
    var normed = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(normed);
    try mlx.check(mlx.mlx_fast_rms_norm(&normed, r, norm_w, eps, s));
    const perm = [_]c_int{ 0, 2, 1, 3 };
    var t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(t);
    try mlx.check(mlx.mlx_transpose_axes(&t, normed, &perm, 4, s));
    var roped = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rope(&roped, t, rd, false, base, 1.0, offset, freqs, s));
    if (ms) |m| {
        var scaled = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&scaled, roped, m, s));
        _ = mlx.mlx_array_free(roped);
        return scaled;
    }
    return roped;
}

const Qwen4LadderHostFixture = struct {
    bytes: []align(std.heap.page_size_min) u8,
    st: qwen4_mod.Qwen4State,
    xfm_bytes: [@sizeOf(Transformer)]u8 align(@alignOf(Transformer)),
    cache_bytes: [@sizeOf(KVCache)]u8 align(@alignOf(KVCache)),

    pub fn init(self: *Qwen4LadderHostFixture) !void {
        const hash = try qwen4_mod.NgramHash.init(1000, 3, 8, 500, 1, 1234, 0, 999);
        const dim = 64;
        self.bytes = try testing.allocator.alignedAlloc(u8, .fromByteUnits(std.heap.page_size_min), @intCast(hash.total_rows * dim * 2));
        var prng = std.Random.DefaultPrng.init(21);
        for (0..self.bytes.len / 2) |i| {
            const v = (prng.random().float(f32) - 0.5) * 2;
            std.mem.writeInt(u16, self.bytes[i * 2 ..][0..2], @truncate(@as(u32, @bitCast(v)) >> 16), .little);
        }
        self.st = .{ .hash = hash, .table = .{ .map = self.bytes, .rows = hash.total_rows,
            .dim = dim, .bits = 16, .group_size = 0, .w_off = 0, .s_off = 0, .b_off = 0,
            .wcols = dim, .scols = 0 } };
        self.xfm_bytes = @splat(0);
        self.cache_bytes = @splat(0);
        const x = self.xfm();
        x.allocator = testing.allocator;
        x.s = mlx.gpuStream();
        x.qwen4 = &self.st;
    }

    fn xfm(self: *Qwen4LadderHostFixture) *Transformer {
        return @ptrCast(&self.xfm_bytes);
    }

    pub fn deinit(self: *Qwen4LadderHostFixture) void {
        testing.allocator.free(self.bytes);
    }
};
fn ladderPleEntry() SSMCacheEntry {
    return .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = false };
}

fn ladderPleIds(ids: []const u32, batch: c_int) mlx.mlx_array {
    var buf: [64]i32 = undefined;
    for (ids, 0..) |v, i| buf[i] = @intCast(v);
    const shape = [_]c_int{ batch, @divExact(@as(c_int, @intCast(ids.len)), batch) };
    return mlx.mlx_array_new_data(&buf, &shape, 2, .int32);
}

fn expectLadderPleHistoryEqual(want: *const SSMCacheEntry, got: *const SSMCacheEntry) !void {
    try testing.expectEqual(want.ple_prev_valid, got.ple_prev_valid);
    try testing.expectEqualSlices(u32, &want.ple_prev, &got.ple_prev);
    try testing.expectEqual(want.spec_ple_len, got.spec_ple_len);
    try testing.expectEqualSlices(u32, want.spec_ple_tokens[0..want.spec_ple_len], got.spec_ple_tokens[0..got.spec_ple_len]);
}


const Qwen4LadderRig = struct {
    fx: Qwen4LadderHostFixture,
    layer_bytes: [LAYERS * @sizeOf(MoeLayerWeights)]u8 align(@alignOf(MoeLayerWeights)),
    arrs: [16]mlx.mlx_array,
    n: usize,

    const LAYERS = 4;
    const HIDDEN = 128;
    const HC = 2;
    const RANK = 16;
    const VOCAB = 64;

    const Slot = struct {
        entries: [LAYERS]SSMCacheEntry,
        off: usize,

        pub fn init() Slot {
            var s: Slot = .{ .entries = undefined, .off = 0 };
            for (&s.entries) |*e| e.* = ladderPleEntry();
            return s;
        }

        pub fn deinit(self: *Slot) void {
            for (&self.entries) |*e| ssmFreeQsaState(e);
        }
    };

    const Step = struct { logits: []f32, pending: bool };

    pub fn init(self: *Qwen4LadderRig) !void {
        try self.fx.init();
        self.n = 0;
        const x = self.fx.xfm();
        const st = x.qwen4.?;
        const emb_dim: c_int = @intCast(st.table.dim * st.hash.n_heads);
        x.config = .{ .model_type = "qwen4_exp", .hidden_size = HIDDEN, .hc_count = HC, .num_hidden_layers = LAYERS, .vocab_size = VOCAB };
        var prng = std.Random.DefaultPrng.init(59);
        const r = prng.random();
        x.emb_w = try self.weight(r, &.{ VOCAB, HIDDEN });
        x.lm_head_w = try self.weight(r, &.{ VOCAB, HIDDEN });
        var ones = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_ones(&ones, &[_]c_int{HIDDEN}, 1, .bfloat16, x.s));
        self.keep(ones);
        x.ones_hidden = ones;
        const none: mlx.mlx_array = .{ .ctx = null };
        x.qwen4_mixer = .{
            .norm_w = try self.weight(r, &.{ HC, HIDDEN }),
            .down_w = try self.weight(r, &.{ HC * HIDDEN, RANK }),
            .down_s = none,
            .down_b = none,
            .up_w = try self.weight(r, &.{ RANK, HC * HIDDEN }),
            .up_s = none,
            .up_b = none,
        };
        self.layer_bytes = @splat(0);
        const layers: [*]MoeLayerWeights = @ptrCast(&self.layer_bytes);
        for (layers[0..LAYERS]) |*lw| {
            lw.hc_attn = null;
            lw.hc_mlp = null;
            lw.ple = null;
        }
        layers[0].ple = .{
            .key_w = try self.weight(r, &.{ emb_dim, HC * HIDDEN }),
            .key_s = none,
            .key_b = none,
            .value_w = try self.weight(r, &.{ emb_dim, HIDDEN }),
            .value_s = none,
            .value_b = none,
            .norm_key = try self.weight(r, &.{ HC, HIDDEN }),
            .norm_query = try self.weight(r, &.{ HC, HIDDEN }),
            .norm_conv = try self.weight(r, &.{ HC, HIDDEN }),
            .conv_w = try self.weight(r, &.{ HC * HIDDEN, @intCast(x.config.ple_conv_kernel), 1 }),
        };
        x.moe_layers = layers[0..LAYERS];
        x.decode_async_ladder = 0;
        x.decode_async_ladder_qwen4 = Transformer.qwen4DecodeLadderStride(null);
    }

    pub fn deinit(self: *Qwen4LadderRig) void {
        for (self.arrs[0..self.n]) |a| _ = mlx.mlx_array_free(a);
        self.fx.deinit();
    }

    fn keep(self: *Qwen4LadderRig, a: mlx.mlx_array) void {
        self.arrs[self.n] = a;
        self.n += 1;
    }

    fn weight(self: *Qwen4LadderRig, r: std.Random, shape: []const c_int) !mlx.mlx_array {
        const a = try attn256RandBf16(r, shape, self.fx.xfm().s);
        self.keep(a);
        return a;
    }

    /// One serial decode step on a deferred PLE leaf at serial ladder stride `ladder`; reports whether the leaf was
    /// still pending when the build returned, then fills it and reads the logits.
    fn step(self: *Qwen4LadderRig, slot: *Slot, ids: mlx.mlx_array, ladder: u32) !Step {
        const x = self.fx.xfm();
        x.decode_async_ladder = ladder;
        defer x.decode_async_ladder = 0;
        var ctx: ForwardCtx = .{ .cache = @ptrCast(&self.fx.cache_bytes), .moe_seq_offset = &slot.off, .ssm_entries = &slot.entries, .capture_hidden = null, .vision_embeddings = null, .ple_defer = true };
        const logits = try x.forwardQwen4With(&ctx, ids);
        defer _ = mlx.mlx_array_free(logits);
        const pending = ctx.ple_pending != null;
        try x.flushDeferredPle(&ctx);
        return .{ .logits = try qwen4ReadF32(testing.allocator, logits, x.s), .pending = pending };
    }

    fn prefill(self: *Qwen4LadderRig, sl: *Qwen4TestSlot, ids: []const i32) !void {
        const shape = [_]c_int{ 1, @intCast(ids.len) };
        const arr = mlx.mlx_array_new_data(@ptrCast(ids.ptr), &shape, 2, .int32);
        defer _ = mlx.mlx_array_free(arr);
        const logits = try self.fx.xfm().forwardQwen4With(&sl.ctx, arr);
        defer _ = mlx.mlx_array_free(logits);
        try mlx.check(mlx.mlx_array_eval(logits));
    }

    /// One batched decode step over `slots` at batched ladder stride `ladder`, filling (or discarding) the leaf.
    fn batchedStep(self: *Qwen4LadderRig, slots: []const *Qwen4TestSlot, ids: []const i32, ladder: u32, fill: bool) !Step {
        const x = self.fx.xfm();
        x.decode_async_ladder_qwen4 = ladder;
        defer x.decode_async_ladder_qwen4 = Transformer.qwen4DecodeLadderStride(null);
        const run = try qwen4BatchedPleForward(x, slots, ids, 1, false, true, fill);
        defer _ = mlx.mlx_array_free(run.logits);
        return .{ .logits = try qwen4ReadF32(testing.allocator, run.logits, x.s), .pending = run.pending };
    }
};

fn expectLogitBitsEqual(want: []const f32, got: []const f32) !void {
    try testing.expectEqual(want.len, got.len);
    if (std.mem.eql(u8, std.mem.sliceAsBytes(want), std.mem.sliceAsBytes(got))) return;
    var n: usize = 0;
    var max: f32 = 0;
    for (want, got) |a, b| {
        if (@as(u32, @bitCast(a)) == @as(u32, @bitCast(b))) continue;
        n += 1;
        max = @max(max, @abs(a - b));
    }
    std.debug.print("logits differ in {d}/{d} entries, max |diff| {d}\n", .{ n, want.len, max });
    return error.TestExpectedEqual;
}

pub var expert_defer_sync_env: ?bool = null;
var expert_deferred_layers: u64 = 0;
var expert_deferred_rollbacks: u64 = 0;
pub fn noteExpertDeferred(kept: bool) void {
    expert_deferred_layers += 1;
    if (!kept) expert_deferred_rollbacks += 1;
    if (expert_deferred_layers == 1 or expert_deferred_layers % 20_000 == 0)
        log.info("[expert-defer] verified {d} layers, rolled back {d} GDN successors\n", .{ expert_deferred_layers, expert_deferred_rollbacks });
}

pub const StreamGdnUndo = struct {
    conv: mlx.mlx_array,
    state: mlx.mlx_array,
    initialized: bool,

    pub fn init(entry: *const SSMCacheEntry) !StreamGdnUndo {
        const conv = if (entry.conv_state.ctx != null) try standinRef(entry.conv_state) else entry.conv_state;
        errdefer if (conv.ctx != null) { _ = mlx.mlx_array_free(conv); };
        const state = if (entry.ssm_state.ctx != null) try standinRef(entry.ssm_state) else entry.ssm_state;
        return .{ .conv = conv, .state = state, .initialized = entry.initialized };
    }

    pub fn restore(self: *StreamGdnUndo, entry: *SSMCacheEntry) void {
        if (entry.conv_state.ctx != null) _ = mlx.mlx_array_free(entry.conv_state);
        if (entry.ssm_state.ctx != null) _ = mlx.mlx_array_free(entry.ssm_state);
        entry.conv_state = self.conv;
        entry.ssm_state = self.state;
        entry.initialized = self.initialized;
        self.conv = .{ .ctx = null };
        self.state = .{ .ctx = null };
    }

    pub fn deinit(self: *StreamGdnUndo) void {
        if (self.conv.ctx != null) _ = mlx.mlx_array_free(self.conv);
        if (self.state.ctx != null) _ = mlx.mlx_array_free(self.state);
    }
};

pub fn streamedSuccessorSafe(layer: *const MoeLayerWeights) bool {
    return layer.attn == .linear and layer.ple == null and layer.mlp == .moe;
}

const StreamDeferredRig = struct {
    fx: Qwen4LadderHostFixture,
    layers: [4]MoeLayerWeights,
    arrays: [96]mlx.mlx_array = undefined,
    count: usize = 0,

    fn weight(self: *StreamDeferredRig, r: std.Random, shape: []const c_int) !mlx.mlx_array {
        const a = try attn256RandBf16(r, shape, self.fx.xfm().s);
        self.arrays[self.count] = a;
        self.count += 1;
        return a;
    }

    pub fn init(self: *StreamDeferredRig, pack: *const expert_stream_mod.TinyQuantPack) !void {
        try self.fx.init();
        const x = self.fx.xfm();
        x.config = .{
            .model_type = "qwen4_exp", .hidden_size = 64, .hc_count = 4, .num_hidden_layers = 4, .vocab_size = 16,
            .num_experts = 4, .num_experts_per_tok = 2, .moe_intermediate_size = 64, .hidden_act = .silu,
            .linear_num_key_heads = 1, .linear_num_value_heads = 1, .linear_key_head_dim = 32, .linear_value_head_dim = 32,
            .linear_conv_kernel_dim = 4, .quant_group_size = 32, .quant_bits = 4,
        };
        var prng = std.Random.DefaultPrng.init(183);
        const r = prng.random();
        const none: mlx.mlx_array = .{ .ctx = null };
        x.emb_w = try self.weight(r, &.{ 16, 64 });
        x.lm_head_w = try self.weight(r, &.{ 16, 64 });
        x.one = mlx.mlx_array_new_float(1);
        x.ones_hidden = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_ones(&x.ones_hidden.?, &[_]c_int{64}, 1, .bfloat16, x.s));
        const hc_weights = HcWeights{
            .norm_w = try self.weight(r, &.{ 4, 64 }),
            .down_w = try self.weight(r, &.{ 256, 8 }), .down_s = none, .down_b = none,
            .up_w = try self.weight(r, &.{ 8, 256 }), .up_s = none, .up_b = none,
            .inject_w = try self.weight(r, &.{ 256, 4 }), .inject_s = none, .inject_b = none,
        };
        x.qwen4_mixer = hc_weights;
        var mw: MoeMlpWeights = std.mem.zeroes(MoeMlpWeights);
        mw.router_w = try self.weight(r, &.{ 64, 4 });
        mw.switch_gate_w = pack.residentBank(4, 0);
        mw.switch_gate_s = pack.residentBank(4, 1);
        mw.switch_gate_b = pack.residentBank(4, 2);
        mw.switch_up_w = pack.residentBank(4, 3);
        mw.switch_up_s = pack.residentBank(4, 4);
        mw.switch_up_b = pack.residentBank(4, 5);
        mw.switch_down_w = pack.residentBank(4, 6);
        mw.switch_down_s = pack.residentBank(4, 7);
        mw.switch_down_b = pack.residentBank(4, 8);
        for ([_]mlx.mlx_array{ mw.switch_gate_w, mw.switch_gate_s, mw.switch_gate_b, mw.switch_up_w, mw.switch_up_s, mw.switch_up_b, mw.switch_down_w, mw.switch_down_s, mw.switch_down_b }) |a| {
            self.arrays[self.count] = a;
            self.count += 1;
        }
        const la = LinearAttnWeights{
            .qkv_w = try self.weight(r, &.{ 64, 96 }), .qkv_s = none, .qkv_b = none,
            .z_w = try self.weight(r, &.{ 64, 32 }), .z_s = none, .z_b = none,
            .a_w = try self.weight(r, &.{ 64, 1 }), .a_s = none, .a_b = none,
            .b_w = try self.weight(r, &.{ 64, 1 }), .b_s = none, .b_b = none,
            .out_w = try self.weight(r, &.{ 32, 64 }), .out_s = none, .out_b = none,
            .conv1d_w = try self.weight(r, &.{ 96, 4, 1 }),
            .norm_w = try self.weight(r, &.{32}),
            .A_log = try self.weight(r, &.{1}), .dt_bias = try self.weight(r, &.{1}),
            };
        for (&self.layers) |*layer| layer.* = .{
            .input_norm = none, .post_attn_norm = none, .is_linear = true,
            .attn = .{ .linear = la }, .mlp = .{ .moe = mw },
            .hc_attn = hc_weights, .hc_mlp = hc_weights,
        };
        x.moe_layers = &self.layers;
    }

    pub fn deinit(self: *StreamDeferredRig) void {
        const x = self.fx.xfm();
        for (self.arrays[0..self.count]) |a| _ = mlx.mlx_array_free(a);
        inline for (.{ "gdn_q_scale", "gdn_k_scale", "gdn_eps", "gdn_ones_w", "ones_hidden" }) |name|
            if (@field(x, name)) |a| { _ = mlx.mlx_array_free(a); };
        _ = mlx.mlx_array_free(x.one);
        self.fx.deinit();
    }
};

fn streamDeferredCheckpoint(dir: std.Io.Dir, pack: *const expert_stream_mod.TinyQuantPack) !void {
    const a = testing.allocator;
    var index: std.ArrayList(u8) = .empty;
    defer index.deinit(a);
    try index.appendSlice(a, "{\"weight_map\":{");
    const header_len: usize = @intCast(std.mem.readInt(u64, pack.file[0..8], .little));
    for (0..4) |layer| {
        const copy = try a.dupe(u8, pack.file);
        defer a.free(copy);
        var pos: usize = 8;
        while (std.mem.indexOfPos(u8, copy[0 .. 8 + header_len], pos, "layers.0")) |found| {
            copy[found + 7] = @intCast('0' + layer);
            pos = found + 8;
        }
        var buf: [32]u8 = undefined;
        const file = try std.fmt.bufPrint(&buf, "layer-{d}.safetensors", .{layer});
        try dir.writeFile(testing.io, .{ .sub_path = file, .data = copy });
        for (0..expert_quant_mod.component_count) |ci| {
            var key_buf: [192]u8 = undefined;
            const key = try expert_quant_mod.tensorKey(&key_buf, @intCast(layer), @fromBackingInt(@intCast(ci)));
            const item = try std.fmt.allocPrint(a, "{s}\"{s}\":\"{s}\"", .{ if (layer == 0 and ci == 0) "" else ",", key, file });
            defer a.free(item);
            try index.appendSlice(a, item);
        }
    }
    try index.appendSlice(a, "}}");
    try dir.writeFile(testing.io, .{ .sub_path = "model.safetensors.index.json", .data = index.items });
}

// ── Batched-decode forward pass ──
//
// One forward call computes next-token logits for N concurrent requests at
// decode step (q_len=1 each). Each request owns its own KVCache via its
// ForwardCtx; weights are shared. Returns N logits arrays of shape [1,1,V],
// one per slot in the input order. Caller owns the returned slice and the
// inner arrays (free each via mlx_array_free, then allocator.free the slice).
//
// Restrictions (enforced upstream by `Scheduler.batchable`):
//   - Standard arch only (no MoE, hybrid SSM).
//   - Decode only (each slot contributes exactly one new token).
//   - No grammar-constrained sampling, no in-flight speculative round.
//
// Per-layer flow:
//   embed → input_norm → Q/K/V proj (B=N, batch-invariant)
//   → Q/K-norm → transpose → mlx_fast_rope_dynamic (per-slot offset)
//   → per-slot cache.update at B=1 (each ctx's cache owns its own state)
//   → gather views, pad to common kv_max, concat axis=0
//   → build [N,1,1,kv_max] additive mask via positions < kv_lens
//   → SDPA → o_proj → MLP → final_norm → lm_head → softcap → demux.
/// Batched decode for a GatedDeltaNet trunk, the hybrid
/// twin of `forwardBatchedDecode`.
///
/// Why this exists: without it, concurrent requests on a GDN model decode
/// strictly serially, so N streams read the whole weight set N times.
/// Batching reads it ONCE — measured against oMLX/mlx-lm, which batch
/// these models and were ~2x our aggregate throughput at 4 streams.
///
/// The trick is that `forwardMoeWith` is already batch-generic: every
/// projection, norm, MLP and the GDN recurrence kernel (whose grid is
/// `B*Hv` on the z axis) take B=N unchanged. Only two things are per-slot:
///
///   - the GDN recurrent state — merged here into ONE [N, …] entry per
///     layer and split back after, so the layers never learn about slots;
///   - the KV cache of the full-attention layers — left in place and
///     handled by the `ctx.batch_slots` branch in `gatedFullAttnWith`,
///     because kv_len differs per slot and cannot be a single tensor.
///
/// Returns N logits arrays `[1,1,V]`, caller owns each and the slice.
/// `hidden_rows`: when given, receives every slot's post-norm hidden `[1,1,H]`
/// (owned slice), for a slot that keeps an MTP head resumable across plain ticks.
pub fn forwardMoeBatchedDecode(
    self: *Transformer,
    next_tokens: []const u32,
    ctxs: []const *ForwardCtx,
    rope_offsets: []const u32,
    hidden_rows: ?*?[]mlx.mlx_array,
) ![]mlx.mlx_array {
    const N: c_int = @intCast(next_tokens.len);
    std.debug.assert(next_tokens.len == ctxs.len);
    if (!qwen4_gdn.gdn_batched_logged) {
        qwen4_gdn.gdn_batched_logged = true;
        log.info("[batched] gdn batched decode engaged (slots={d})\n", .{next_tokens.len});
    }
    // [N,1] tokens.
    var token_buf = try self.allocator.alloc(i32, next_tokens.len);
    defer self.allocator.free(token_buf);
    for (next_tokens, 0..) |t, i| token_buf[i] = @intCast(t);
    const tok_shape = [_]c_int{ N, 1 };
    const token_arr = mlx.mlx_array_new_data(token_buf.ptr, &tok_shape, 2, .int32);
    defer _ = mlx.mlx_array_free(token_arr);

    var hidden = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hidden);
    const logits = try self.forwardMoeBatchedRows(token_arr, ctxs, rope_offsets, false, if (hidden_rows != null) &hidden else null, null);
    defer _ = mlx.mlx_array_free(logits);
    if (hidden_rows) |out| out.* = try sliceBatchRows(self.allocator, self.s, hidden, next_tokens.len);
    return sliceBatchRows(self.allocator, self.s, logits, next_tokens.len);
}

/// Batched spec VERIFY: `[N, S]` rows (every slot's `[t1, drafts…]`, padded to the
/// group's widest), per-position SSM capture ON so each slot rolls back on its own,
/// post-norm hidden captured at the last row (`hidden_last` `[N,1,H]`) and at every
/// row (`hidden_all` `[N,S,H]`). Returns logits `[N, S, V]`; caller owns everything.
pub fn forwardMoeBatchedVerify(
    self: *Transformer,
    token_arr: mlx.mlx_array,
    ctxs: []const *ForwardCtx,
    rope_offsets: []const u32,
    hidden_last: *mlx.mlx_array,
    hidden_all: *mlx.mlx_array,
) !mlx.mlx_array {
    if (!qwen4_gdn.gdn_batched_verify_logged) {
        qwen4_gdn.gdn_batched_verify_logged = true;
        log.info("[batched] gdn batched verify engaged (slots={d}, width={d})\n", .{ ctxs.len, mlx.getShape(token_arr)[1] });
    }
    return self.forwardMoeBatchedRows(token_arr, ctxs, rope_offsets, true, hidden_last, hidden_all);
}

pub fn forwardMoeBatchedRows(
    self: *Transformer,
    token_arr: mlx.mlx_array,
    ctxs: []const *ForwardCtx,
    rope_offsets: []const u32,
    capture_ssm: bool,
    hidden_last: ?*mlx.mlx_array,
    hidden_all: ?*mlx.mlx_array,
) !mlx.mlx_array {
    self.fwd_gen +%= 1; // per-forward QSA scratch key
    const N: c_int = @intCast(ctxs.len);
    std.debug.assert(ctxs.len == rope_offsets.len);
    std.debug.assert(N >= 1);
    const ml = self.moe_layers orelse return error.BatchedGdnRequiresMoeLayers;

    // Per-slot RoPE offsets, each already carrying its own M-RoPE delta.
    var off_buf = try self.allocator.alloc(i32, rope_offsets.len);
    defer self.allocator.free(off_buf);
    for (rope_offsets, 0..) |o, i| {
        const d: i32 = if (ctxs[i].mrope_pos != null) ctxs[i].mrope_delta else 0;
        off_buf[i] = @as(i32, @intCast(o)) + d;
    }
    const off_shape = [_]c_int{N};
    const rope_offset_arr = mlx.mlx_array_new_data(off_buf.ptr, &off_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(rope_offset_arr);

    // Merge each linear layer's per-slot GDN state into one B=N entry.
    // A layer whose state is not initialized on EVERY slot cannot be
    // merged (widths would disagree) — the caller gates on
    // `batchedGdnReady`.
    const persist = self.persistent_group_state_override orelse true;
    self.ssm_group.allocator = self.allocator;
    const stable = persist and self.ssm_group.live and self.ssm_group.matches(ctxs);
    var kinds_buf: []SsmMergeKind = &.{};
    defer if (kinds_buf.len != 0) self.allocator.free(kinds_buf);
    const kinds: []const SsmMergeKind = if (stable) self.ssm_group.kinds else blk: {
        kinds_buf = try self.allocator.alloc(SsmMergeKind, ml.len);
        for (ml, 0..) |*lw, l| kinds_buf[l] = .{ .linear = lw.is_linear, .ple = lw.ple != null };
        break :blk kinds_buf;
    };
    const bind = try ssmBindTickState(self.allocator, self.s, &self.ssm_group, ctxs, kinds, persist);
    const merged = bind.merged;
    defer if (bind.owned) {
        for (merged) |*m| ssmFreeMergedLayer(m);
        self.allocator.free(merged);
    };

    // Scratch ctx. `cache` is never read on this path — the batched branch
    // in `gatedFullAttnWith` goes to the slot caches — but the field is
    // non-optional, so slot 0's stands in. `moe_seq_offset` likewise only
    // feeds the scalar-RoPE path the batched branch replaces.
    var scratch_offset: usize = rope_offsets[0];
    var bctx: ForwardCtx = .{
        .cache = ctxs[0].cache,
        .moe_seq_offset = &scratch_offset,
        .ssm_entries = merged,
        .capture_hidden = hidden_last,
        .capture_hidden_all = hidden_all,
        .capture_ssm_seq = capture_ssm,
        .vision_embeddings = null,
        .batch_slots = ctxs,
        .batch_rope_offsets = rope_offset_arr,
        .ple_defer = self.qwen4 != null,
    };
    // A verify's captures live on the merged entries until split below.
    defer if (capture_ssm) {
        for (merged) |*m| ssmFreeSpecCapture(m);
    };

    // qwen4_exp rides the same driver: GDN + PLE window merged here, QSA
    // key histories reached per slot through `batch_slots` (kv differs).
    const logits = (if (self.qwen4 != null) self.forwardQwen4With(&bctx, token_arr) else self.forwardMoeWith(&bctx, token_arr)) catch |e| {
        self.discardDeferredPle(&bctx);
        return e;
    };
    errdefer {
        self.discardDeferredPle(&bctx);
        _ = mlx.mlx_array_free(logits);
    }
    try self.flushDeferredPle(&bctx);

    if (persist) {
        if (capture_ssm) {
            for (merged, 0..) |*m, li| {
                if (!kinds[li].linear and !kinds[li].ple) continue;
                for (ctxs, 0..) |c, i| try splitSpecCaptures(self.s, m, &c.ssm_entries.?[li], i);
            }
        }
        if (!stable) ssmBindSlotRefs(&self.ssm_group, ctxs);
    } else {
        try ssmSplitAll(self.s, merged, ctxs, kinds);
    }
    return logits;
}

/// Can these slots take ONE batched GDN decode tick? Every slot must
/// already carry initialized state for every linear layer — that is true
/// after any prefill, and false for a slot that has not run one, whose
/// zero-state would otherwise be merged at the wrong width.
pub fn batchedGdnReady(self: *const Transformer, ctxs: []const *ForwardCtx) bool {
    const ml = self.moe_layers orelse return false;
    for (ctxs) |c| {
        const entries = c.ssm_entries orelse return false;
        if (entries.len != ml.len) return false;
        for (ml, 0..) |*lw, i| {
            // The PLE layer merges its conv window too and reads the
            // slot's token history — both exist after any prefill.
            if (lw.ple != null and (entries[i].aux_state.ctx == null or !entries[i].ple_prev_valid)) return false;
            if (!lw.is_linear) continue;
            if (!entries[i].initialized) return false;
            if (entries[i].conv_state.ctx == null or entries[i].ssm_state.ctx == null) return false;
        }
    }
    return true;
}

pub fn supportsBatchedGdnDecode(self: *const Transformer) bool {
    // Share the architecture gate with the server's concurrency clamp.
    if (!self.config.supportsBatchedGdnDecode()) return false;
    // The loaded trunk must supply MoE layers and per-slot GDN state.
    if (self.moe_layers == null) return false;
    if (self.glm5 != null) return false;
    return true;
}

/// YaRN mscale on rope'd q and k: multiply by the [head_dim] table
/// (attention_factor on the rotated slice, 1.0 on the pass-through tail).
/// That is the reference's `cos, sin = cos*af, sin*af` — every rotary pair
/// shares one frequency and both of its dims carry it, so scaling the
/// rotation and scaling q/k are the same product. The table MUST be read in
/// the tensors' dtype: an f32 operand promotes the result, and the f32 then
/// rides SDPA → o_proj → residual, upcasting every weight read downstream.
pub fn yarnScaleQK(self: *Transformer, q: *mlx.mlx_array, k: *mlx.mlx_array) !void {
    const ms = try constTableAs(self.yarn_mscale.?, mlx.mlx_array_dtype(q.*), &self.yarn_mscale_cast, self.s);
    var qs = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&qs, q.*, ms, self.s));
    _ = mlx.mlx_array_free(q.*);
    q.* = qs;
    var ks = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&ks, k.*, ms, self.s));
    _ = mlx.mlx_array_free(k.*);
    k.* = ks;
}

/// YaRN mscale on a rope'd array whose last dim may not be `head_dim`
/// (the QSA indexer is 128-wide / 64 rotated). Slice the first `rope_dims`,
/// multiply by a same-dtype scalar, concat the pass-through tail. NEVER
/// `yarnScaleQK` here: that table is `[head_dim]` (256 on qwen4).
pub fn yarnScaleRotated(self: *Transformer, q: *mlx.mlx_array, rope_dims: c_int) !void {
    const sh = mlx.getShape(q.*);
    const dt = mlx.mlx_array_dtype(q.*);
    const st = [_]c_int{ 1, 1, 1, 1 };
    var rot = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rot);
    try mlx.check(mlx.mlx_slice(&rot, q.*, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ sh[0], sh[1], sh[2], rope_dims }, 4, &st, 4, self.s));
    var pass = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pass);
    try mlx.check(mlx.mlx_slice(&pass, q.*, &[_]c_int{ 0, 0, 0, rope_dims }, 4, &[_]c_int{ sh[0], sh[1], sh[2], sh[3] }, 4, &st, 4, self.s));
    const ms = try scalarOf(self.config.yarn_attention_factor, dt, self.s);
    defer _ = mlx.mlx_array_free(ms);
    var scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scaled);
    try mlx.check(mlx.mlx_multiply(&scaled, rot, ms, self.s));
    const parts = [_]mlx.mlx_array{ scaled, pass };
    const vec = mlx.mlx_vector_array_new_data(&parts, 2);
    defer _ = mlx.mlx_vector_array_free(vec);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, -1, self.s));
    _ = mlx.mlx_array_free(q.*);
    q.* = out;
}

/// The angle row for `offset` in the given rope family, rebuilt when the
/// offset moves (per decode token: 38-of-40 layers hit the cache).
pub fn qkAngleFor(self: *Transformer, family: usize, rd: c_int, base: mlx.mlx_optional_float, freqs: mlx.mlx_array, offset: c_int, mscale: f32) !mlx.mlx_array {
    return self.qkAngleRowsFor(family, rd, base, freqs, offset, 1, mscale);
}

test "shared expert gate tail admits the joined verifier envelope only" {
    try testing.expect(sharedGateRowsEligible(&.{ 2, 2, 2, 2 }));
    try testing.expect(sharedGateRowsEligible(&.{ 5, 5, 5, 5 }));
    try testing.expect(sharedGateRowsEligible(&.{ 2, 3 }));
    try testing.expect(!sharedGateRowsEligible(&.{ 7, 7, 7, 7 }));
    try testing.expect(!sharedGateRowsEligible(&.{1}));
}

test "qwen4 MTP head: two explicit states with different origins and stash lengths round-trip" {
    try mlx.requireGpu();
    const t = std.testing;
    var xfm: Transformer = undefined;
    xfm.s = mlx.gpuStream();
    xfm.allocator = t.allocator;
    xfm.qwen4_mtp_owner = null;
    const head_cache = try KVCache.init(t.allocator, 1);
    var head: Qwen4Mtp = undefined;
    head.cache = head_cache;
    head.entry = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true };
    head.qsa_marks = .{};
    head.seq_offset = 0;
    head.pos_base = 0;
    xfm.qwen4_mtp = head;
    const m = &xfm.qwen4_mtp.?;
    defer m.cache.deinit();

    const ca = try KVCache.init(t.allocator, 1);
    var a: Transformer.Qwen4MtpState = .{ .cache = ca, .entry = m.entry, .seq_offset = 7, .pos_base = 3 };
    a.cache.step = 7;
    defer a.deinit();
    const cb = try KVCache.init(t.allocator, 1);
    var b: Transformer.Qwen4MtpState = .{ .cache = cb, .entry = m.entry, .seq_offset = 2, .pos_base = 11 };
    b.cache.step = 2;
    defer b.deinit();

    const live_a = Transformer.qwen4MtpLiveState(&a);
    const live_b = Transformer.qwen4MtpLiveState(&b);
    try t.expect(live_a.cache == &a.cache);
    try t.expect(live_b.cache == &b.cache);
    try t.expect(xfm.qwen4_mtp_owner == null);
    try t.expectEqual(@as(usize, 0), m.seq_offset);

    const sa = a.meta();
    const sb = b.meta();
    try t.expectEqual(@as(usize, 7), sa.stash_len);
    try t.expectEqual(@as(c_int, 3), sa.origin);
    try t.expectEqual(@as(usize, 2), sb.stash_len);
    try t.expectEqual(@as(c_int, 11), sb.origin);

    live_a.seq_offset.* = 99;
    live_b.pos_base.* = -1;
    try t.expectEqual(@as(usize, 0), m.seq_offset);
    try t.expectEqual(@as(c_int, 0), m.pos_base);

    a.restoreMeta(sa);
    b.restoreMeta(sb);
    try t.expectEqual(sa, a.meta());
    try t.expectEqual(sb, b.meta());
    try t.expect(xfm.qwen4_mtp_owner == null);
}

test "qwen4 MTP head: two requests keep their own state through activate/release" {
    // Bar: after A appends and B activates, A's rows are not on the module; A re-activating
    // sees them again; releasing hands the module its own copy back.
    try mlx.requireGpu();
    const t = std.testing;
    var xfm: Transformer = undefined;
    xfm.s = mlx.gpuStream();
    xfm.allocator = t.allocator;
    xfm.qwen4_mtp_owner = null;
    const head_cache = try KVCache.init(t.allocator, 1);
    var head: Qwen4Mtp = undefined;
    head.cache = head_cache;
    head.entry = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true };
    head.qsa_marks = .{};
    head.seq_offset = 0;
    head.pos_base = 0;
    xfm.qwen4_mtp = head;
    const m = &xfm.qwen4_mtp.?;
    defer m.cache.deinit();

    const ca = try KVCache.init(t.allocator, 1);
    var a: Transformer.Qwen4MtpState = .{ .cache = ca, .entry = m.entry };
    defer a.deinit();
    const cb = try KVCache.init(t.allocator, 1);
    var b: Transformer.Qwen4MtpState = .{ .cache = cb, .entry = m.entry };
    defer b.deinit();

    xfm.qwen4MtpActivate(&a);
    m.seq_offset = 7;
    m.pos_base = 3;
    xfm.qwen4MtpActivate(&a); // idempotent
    try t.expectEqual(@as(usize, 7), m.seq_offset);
    xfm.qwen4MtpActivate(&b);
    try t.expectEqual(@as(usize, 0), m.seq_offset);
    try t.expectEqual(@as(c_int, -1), m.pos_base);
    m.seq_offset = 2;
    xfm.qwen4MtpActivate(&a);
    try t.expectEqual(@as(usize, 7), m.seq_offset);
    try t.expectEqual(@as(c_int, 3), m.pos_base);
    try t.expectEqual(@as(usize, 2), b.seq_offset);
    xfm.qwen4MtpRelease(&a);
    try t.expectEqual(@as(usize, 0), m.seq_offset);
    try t.expectEqual(@as(usize, 7), a.seq_offset);
    try t.expect(xfm.qwen4_mtp_owner == null);
}

test "exl3 MTP rows wider than the decode arm refuse by Exl3MtpRowsExceedDecode" {
    const t = std.testing;
    const s = mlx.gpuStream();
    var xfm: Transformer = undefined;
    xfm.s = s;
    xfm.config = .{
        .expert_layout = .exl3_k4,
        .num_experts = 4,
        .num_experts_per_tok = 4,
        .hidden_size = 8,
        .moe_intermediate_size = 8,
        .quant_bits = 0,
        .quant_group_size = 64,
        .quant_mode = .affine,
    };
    xfm.one = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(xfm.one);
    xfm.bits_cache = .{};
    xfm.compiled_moe_routing = null;
    xfm.cost_trace_active = false;
    var mw: MoeMlpWeights = .{
        .router_w = mlx.mlx_array_new(),
        .router_s = mlx.mlx_array_new(),
        .router_b = mlx.mlx_array_new(),
        .switch_gate_w = mlx.mlx_array_new(),
        .switch_gate_s = mlx.mlx_array_new(),
        .switch_gate_b = mlx.mlx_array_new(),
        .switch_up_w = mlx.mlx_array_new(),
        .switch_up_s = mlx.mlx_array_new(),
        .switch_up_b = mlx.mlx_array_new(),
        .switch_down_w = mlx.mlx_array_new(),
        .switch_down_s = mlx.mlx_array_new(),
        .switch_down_b = mlx.mlx_array_new(),
        .shared_gate_w = mlx.mlx_array_new(),
        .shared_gate_s = mlx.mlx_array_new(),
        .shared_gate_b = mlx.mlx_array_new(),
        .shared_up_w = mlx.mlx_array_new(),
        .shared_up_s = mlx.mlx_array_new(),
        .shared_up_b = mlx.mlx_array_new(),
        .shared_down_w = mlx.mlx_array_new(),
        .shared_down_s = mlx.mlx_array_new(),
        .shared_down_b = mlx.mlx_array_new(),
    };
    defer {
        const arrs = [_]mlx.mlx_array{
            mw.router_w,      mw.router_s,      mw.router_b,
            mw.switch_gate_w, mw.switch_gate_s, mw.switch_gate_b,
            mw.switch_up_w,   mw.switch_up_s,   mw.switch_up_b,
            mw.switch_down_w, mw.switch_down_s, mw.switch_down_b,
            mw.shared_gate_w, mw.shared_gate_s, mw.shared_gate_b,
            mw.shared_up_w,   mw.shared_up_s,   mw.shared_up_b,
            mw.shared_down_w, mw.shared_down_s, mw.shared_down_b,
        };
        for (arrs) |a| _ = mlx.mlx_array_free(a);
    }
    const shape = [_]c_int{ 17, 1, 8 };
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_zeros(&x, &shape, 3, .float16, s));
    try t.expectError(error.Exl3MtpRowsExceedDecode, xfm.mtpMoeRows(x, &mw));
}

test "exl3 MTP fused rows match N solo calls on the same kernel" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const exl3 = expert_exl3;
    const fixture = exl3.fixtures.k4;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const header_len = std.mem.readInt(u64, fixture[0..8], .little);
    const header = fixture[8 .. 8 + header_len];
    const data = fixture[8 + header_len ..];
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
    defer parsed.deinit();
    const trellis_meta = parsed.value.object.get("trellis").?.object;
    const suh_meta = parsed.value.object.get("suh").?.object;
    const svh_meta = parsed.value.object.get("svh").?.object;
    const t0: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[0].integer);
    const t1: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[1].integer);
    const s0: usize = @intCast(suh_meta.get("data_offsets").?.array.items[0].integer);
    const s1: usize = @intCast(suh_meta.get("data_offsets").?.array.items[1].integer);
    const v0: usize = @intCast(svh_meta.get("data_offsets").?.array.items[0].integer);
    const v1: usize = @intCast(svh_meta.get("data_offsets").?.array.items[1].integer);
    const trellis_bits = std.mem.bytesAsSlice(u16, data[t0..t1]);
    const suh_bits = std.mem.bytesAsSlice(u16, data[s0..s1]);
    const svh_bits = std.mem.bytesAsSlice(u16, data[v0..v1]);
    const E: usize = 4;
    const dim: usize = 128;
    const rows: usize = 8;
    const tile_n = 8 * 8 * 64;
    const stacked_t = try alloc.alloc(u16, E * tile_n);
    const stacked_suh = try alloc.alloc(u16, E * dim);
    const stacked_svh = try alloc.alloc(u16, E * dim);
    for (0..E) |e| {
        @memcpy(stacked_t[e * tile_n ..][0..tile_n], trellis_bits);
        @memcpy(stacked_suh[e * dim ..][0..dim], suh_bits);
        @memcpy(stacked_svh[e * dim ..][0..dim], svh_bits);
    }
    var prng = std.Random.DefaultPrng.init(47);
    const rnd = prng.random();
    const xh = try alloc.alloc(u16, rows * dim);
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
    const rw = try alloc.alloc(u16, dim * E);
    for (rw) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 0.05);
    var xfm: Transformer = undefined;
    xfm.s = s;
    xfm.config = .{
        .expert_layout = .exl3_k4,
        .num_experts = @intCast(E),
        .num_experts_per_tok = @intCast(E),
        .hidden_size = @intCast(dim),
        .moe_intermediate_size = @intCast(dim),
        .quant_bits = 0,
        .quant_group_size = 64,
        .quant_mode = .affine,
        .hidden_act = .silu,
    };
    xfm.one = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(xfm.one);
    xfm.bits_cache = .{};
    xfm.compiled_moe_routing = null;
    xfm.compiled_gelu = null;
    xfm.compiled_geglu = null;
    xfm.cost_trace_active = false;
    const prev_st = qwen4_standin_override;
    qwen4_standin_override = .{ .moe_shared = true };
    defer qwen4_standin_override = prev_st;
    const tr = mlx.mlx_array_new_data(stacked_t.ptr, &[_]c_int{ @intCast(E), 8, 8, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr);
    const suh = mlx.mlx_array_new_data(stacked_suh.ptr, &[_]c_int{ @intCast(E), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(suh);
    const svh = mlx.mlx_array_new_data(stacked_svh.ptr, &[_]c_int{ @intCast(E), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(svh);
    const router = mlx.mlx_array_new_data(rw.ptr, &[_]c_int{ @intCast(dim), @intCast(E) }, 2, .float16);
    defer _ = mlx.mlx_array_free(router);
    const none = mlx.mlx_array{ .ctx = null };
    const ident = try alloc.alloc(u16, dim * dim);
    @memset(ident, 0);
    for (0..dim) |i| ident[i * dim + i] = exl3.f32ToF16Bits(1.0);
    const sh_gate = mlx.mlx_array_new_data(ident.ptr, &[_]c_int{ @intCast(dim), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(sh_gate);
    const sh_up = mlx.mlx_array_new_data(ident.ptr, &[_]c_int{ @intCast(dim), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(sh_up);
    const sh_down = mlx.mlx_array_new_data(ident.ptr, &[_]c_int{ @intCast(dim), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(sh_down);
    const gbits = try alloc.alloc(u16, dim);
    @memset(gbits, 0);
    const sh_eg = mlx.mlx_array_new_data(gbits.ptr, &[_]c_int{ @intCast(dim), 1 }, 2, .float16);
    defer _ = mlx.mlx_array_free(sh_eg);
    var mw: MoeMlpWeights = .{
        .router_w = router,
        .router_s = none,
        .router_b = none,
        .switch_gate_w = tr,
        .switch_gate_s = suh,
        .switch_gate_b = svh,
        .switch_up_w = tr,
        .switch_up_s = suh,
        .switch_up_b = svh,
        .switch_down_w = tr,
        .switch_down_s = suh,
        .switch_down_b = svh,
        .shared_gate_w = sh_gate,
        .shared_gate_s = none,
        .shared_gate_b = none,
        .shared_up_w = sh_up,
        .shared_up_s = none,
        .shared_up_b = none,
        .shared_down_w = sh_down,
        .shared_down_s = none,
        .shared_down_b = none,
        .shared_expert_gate_w = sh_eg,
        .shared_expert_gate_s = none,
        .shared_expert_gate_b = none,
    };
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(rows), 1, @intCast(dim) }, 3, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    try mlx.check(mlx.mlx_array_eval(tr));
    try mlx.check(mlx.mlx_array_eval(suh));
    try mlx.check(mlx.mlx_array_eval(svh));
    try mlx.check(mlx.mlx_array_eval(router));
    try mlx.check(mlx.mlx_array_eval(sh_gate));
    try mlx.check(mlx.mlx_array_eval(sh_up));
    try mlx.check(mlx.mlx_array_eval(sh_down));
    try mlx.check(mlx.mlx_array_eval(sh_eg));
    try mlx.check(mlx.mlx_array_eval(x_arr));
    expert_exl3_kernels.resetFusedDispatchCount();
    expert_exl3_kernels.setPairSplitsForTest(1);
    defer expert_exl3_kernels.setPairSplitsForTest(null);
    const fused = try xfm.mtpMoeRows(x_arr, &mw);
    defer _ = mlx.mlx_array_free(fused);
    const n_disp = expert_exl3_kernels.fusedDispatchCount();
    var c_f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c_f);
    try mlx.check(mlx.mlx_contiguous(&c_f, fused, false, s));
    try mlx.check(mlx.mlx_array_eval(c_f));
    const sf = mlx.mlx_array_data_float16(c_f) orelse return error.F16Unreadable;
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        const x1 = mlx.mlx_array_new_data(xh[r * dim ..][0..dim].ptr, &[_]c_int{ 1, 1, @intCast(dim) }, 3, .float16);
        defer _ = mlx.mlx_array_free(x1);
        const solo = try xfm.mtpMoeRows(x1, &mw);
        defer _ = mlx.mlx_array_free(solo);
        var c_s = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(c_s);
        try mlx.check(mlx.mlx_contiguous(&c_s, solo, false, s));
        try mlx.check(mlx.mlx_array_eval(c_s));
        const ss = mlx.mlx_array_data_float16(c_s) orelse return error.F16Unreadable;
        for (0..dim) |i| {
            const a: u16 = @bitCast(sf[r * dim + i]);
            const b: u16 = @bitCast(ss[i]);
            try t.expectEqual(b, a);
        }
    }
    try t.expectEqual(@as(u32, 3), n_disp);
}

test "exl3 MoE prefill chunks return live mlx bytes to baseline" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var h = try Exl3MoeHarness.init(arena.allocator(), s, expert_exl3_kernels.DECODE_ROWS_MAX * 2, 53);
    defer h.deinit();
    const prev_st = qwen4_standin_override;
    qwen4_standin_override = .{};
    defer qwen4_standin_override = prev_st;
    const chunk = struct {
        fn run(hh: *Exl3MoeHarness) !void {
            const y = try hh.xfm.moeMLP(hh.x, &hh.mw);
            defer _ = mlx.mlx_array_free(y);
            try mlx.check(mlx.mlx_array_eval(y));
        }
    };
    try chunk.run(&h);
    try chunk.run(&h);
    try mlx.check(mlx.mlx_synchronize(s));
    _ = mlx.mlx_clear_cache();
    var base: usize = 0;
    try mlx.check(mlx.mlx_get_active_memory(&base));
    try chunk.run(&h);
    try chunk.run(&h);
    try chunk.run(&h);
    try mlx.check(mlx.mlx_synchronize(s));
    _ = mlx.mlx_clear_cache();
    var after: usize = 0;
    try mlx.check(mlx.mlx_get_active_memory(&after));
    try t.expectEqual(base, after);
}

test "exl3 MoE taps the layer's input rows and routing for a calibration capture" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const rows: usize = 6;
    var h = try Exl3MoeHarness.init(arena.allocator(), s, rows, 61);
    defer h.deinit();
    const prev_st = qwen4_standin_override;
    defer qwen4_standin_override = prev_st;
    qwen4_standin_override = .{};
    const col = try imatrix_capture.Collector.init(t.allocator, s, "/dev/null", 1, Exl3MoeHarness.E, .qwen4_exp);
    defer col.deinit();
    col.armRows(0);

    var ids_host: [rows * Exl3MoeHarness.E]i32 = undefined;
    var w_host: [rows * Exl3MoeHarness.E]u16 = undefined;
    for (0..rows) |r| for (0..Exl3MoeHarness.E) |k| {
        ids_host[r * Exl3MoeHarness.E + k] = @intCast((r + k) % Exl3MoeHarness.E);
        w_host[r * Exl3MoeHarness.E + k] = expert_exl3.f32ToF16Bits(0.25);
    };
    const shape3 = [_]c_int{ 1, rows, Exl3MoeHarness.E };
    const inds = mlx.mlx_array_new_data(&ids_host, &shape3, 3, .int32);
    defer _ = mlx.mlx_array_free(inds);
    const globals = mlx.mlx_array_new_data(&ids_host, &[_]c_int{ rows, Exl3MoeHarness.E }, 2, .int32);
    defer _ = mlx.mlx_array_free(globals);
    const scores = mlx.mlx_array_new_data(&w_host, &shape3, 3, .float16);
    defer _ = mlx.mlx_array_free(scores);
    const y = try h.xfm.moeMLP2WithRouter(h.x, h.x, &h.mw, null, true, null, .{
        .inds = inds,
        .norm_scores = scores,
        .imatrix = .{ .collector = col, .layer = 0, .global_ids = globals },
    });
    defer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_array_eval(y));

    const taken = try col.takeRows();
    defer _ = mlx.mlx_array_free(taken.x);
    defer t.allocator.free(taken.ids);
    try t.expectEqualSlices(c_int, &[_]c_int{ rows, Exl3MoeHarness.dim }, mlx.getShape(taken.x));
    try t.expectEqual(@as(usize, Exl3MoeHarness.E), taken.topk);
    try t.expectEqualSlices(i32, &ids_host, taken.ids);
}

test "exl3 MoE answers the shared-expert standin with the routed sum" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var h = try Exl3MoeHarness.init(arena.allocator(), s, expert_exl3_kernels.DECODE_ROWS_MAX * 2, 59);
    defer h.deinit();
    const prev_st = qwen4_standin_override;
    defer qwen4_standin_override = prev_st;
    qwen4_standin_override = .{};
    const routed = try h.xfm.moeMLP2WithRouter(h.x, h.x, &h.mw, null, true, null, null);
    defer _ = mlx.mlx_array_free(routed);
    qwen4_standin_override = .{ .moe_shared = true };
    const got = try h.xfm.moeMLP(h.x, &h.mw);
    defer _ = mlx.mlx_array_free(got);
    var c_r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c_r);
    var c_g = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c_g);
    try mlx.check(mlx.mlx_contiguous(&c_r, routed, false, s));
    try mlx.check(mlx.mlx_contiguous(&c_g, got, false, s));
    try mlx.check(mlx.mlx_array_eval(c_r));
    try mlx.check(mlx.mlx_array_eval(c_g));
    const pr = mlx.mlx_array_data_float16(c_r) orelse return error.F16Unreadable;
    const pg = mlx.mlx_array_data_float16(c_g) orelse return error.F16Unreadable;
    const n = Exl3MoeHarness.dim * expert_exl3_kernels.DECODE_ROWS_MAX * 2;
    for (0..n) |i| {
        const a: u16 = @bitCast(pr[i]);
        const b: u16 = @bitCast(pg[i]);
        try t.expectEqual(a, b);
    }
}

test "exl3 resident decode folds the shared-expert gate tail into the finish reduce" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const prev_st = qwen4_standin_override;
    defer qwen4_standin_override = prev_st;
    qwen4_standin_override = .{};
    for ([_]u64{ 71, 72, 73 }) |seed| {
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        var h = try Exl3MoeHarness.initDtype(arena.allocator(), s, 1, seed, .bfloat16);
        defer h.deinit();
        var rng = std.Random.DefaultPrng.init(seed);
        const seg = try attn256RandBf16Scaled(rng.random(), &.{ Exl3MoeHarness.dim, 1 }, 8.0, s);
        defer _ = mlx.mlx_array_free(seg);
        h.mw.shared_expert_gate_w = seg;

        for ([_]bool{ true, false }) |skip| _ = mlx.mlx_array_free(try h.xfm.moeMLP2WithRouter(h.x, h.x, &h.mw, null, skip, null, null));
        const ops0 = mlx.op_count.load(.monotonic);
        const routed = try h.xfm.moeMLP2WithRouter(h.x, h.x, &h.mw, null, true, null, null);
        defer _ = mlx.mlx_array_free(routed);
        const down = (try h.xfm.moeSharedDown(.qmatmul, h.x, &h.mw, &.{})).?;
        defer _ = mlx.mlx_array_free(down);
        const logit = try h.xfm.moeSharedGateLogit(h.x, &h.mw);
        defer _ = mlx.mlx_array_free(logit);
        const ops2 = mlx.op_count.load(.monotonic);
        const want = try h.xfm.moeSharedGateTail(routed, logit, down);
        defer _ = mlx.mlx_array_free(want);

        const ops3 = mlx.op_count.load(.monotonic);
        const got = try h.xfm.moeMLP(h.x, &h.mw);
        defer _ = mlx.mlx_array_free(got);
        // The routed sum plus the shared projections, and nothing for sigmoid, multiply or add.
        try t.expectEqual(ops2 - ops0, mlx.op_count.load(.monotonic) - ops3);
        try qwen4ExpectMoeBitsEqual(want, got);
    }
}

test "qwen4 PLE spec capture follows the verify flag, streamed or not" {
    var state: qwen4_mod.Qwen4State = undefined;
    state.hash.ngram_size = 3;
    var xfm: Transformer = undefined;
    xfm.qwen4 = &state;
    xfm.expert_stream = null;
    xfm.spec_capture_ssm = false;
    var entry = SSMCacheEntry{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = false };
    entry.spec_ple_len = 17;
    try testing.expect(!xfm.pleClaimSpecCapture(&entry, 4));
    try testing.expectEqual(@as(u32, 0), entry.spec_ple_len);
    xfm.spec_capture_ssm = true;
    try testing.expect(xfm.pleClaimSpecCapture(&entry, 4));
    try testing.expectEqual(@as(u32, 6), entry.spec_ple_len);
    var engine: expert_stream_mod.Engine = undefined;
    xfm.expert_stream = &engine;
    entry.spec_ple_len = 17;
    try testing.expect(xfm.pleClaimSpecCapture(&entry, 4));
    try testing.expectEqual(@as(u32, 6), entry.spec_ple_len);
    xfm.spec_capture_ssm = false;
    try testing.expect(!xfm.pleClaimSpecCapture(&entry, 4));
    try testing.expectEqual(@as(u32, 0), entry.spec_ple_len);
}

test "moe decode dispatch: rows vs sorted vs gatherQmv" {
    trf.moe_rows_fused_override = true;
    defer trf.moe_rows_fused_override = null;
    qwen4_standin_override = .{};
    defer qwen4_standin_override = null;
    const K: c_int = 10;
    var b: c_int = 2;
    while (b <= 8) : (b += 1) {
        try std.testing.expectEqual(MoeDecodeDispatchArm.rows, moeDecodeDispatchArm(b, 1, K));
    }
    try std.testing.expectEqual(MoeDecodeDispatchArm.sorted, moeDecodeDispatchArm(9, 1, K));
    try std.testing.expectEqual(MoeDecodeDispatchArm.rows, moeDecodeDispatchArm(2, 2, K));
    for (2..17) |width| try std.testing.expectEqual(MoeDecodeDispatchArm.rows, moeDecodeDispatchArm(1, @intCast(width), K));
    try std.testing.expectEqual(MoeDecodeDispatchArm.rows, moeDecodeDispatchArm(8, 2, K));
    try std.testing.expectEqual(MoeDecodeDispatchArm.sorted, moeDecodeDispatchArm(8, 3, K));
    try std.testing.expectEqual(MoeDecodeDispatchArm.sorted, moeDecodeDispatchArm(1, 17, K));
    try std.testing.expectEqual(MoeDecodeDispatchArm.gather_qmv, moeDecodeDispatchArm(1, 1, K));
    qwen4_standin_override = .{ .moe_gateup = true };
    try std.testing.expectEqual(MoeDecodeDispatchArm.sorted, moeDecodeDispatchArm(2, 1, K));
    qwen4_standin_override = .{ .moe_down = true };
    try std.testing.expectEqual(MoeDecodeDispatchArm.sorted, moeDecodeDispatchArm(2, 1, K));
    qwen4_standin_override = .{};
}

test "ropeAtPositions YaRN: cos/sin carry mscale on the rotated slice" {
    // Power-of-two mscale so the rotated slice is an exact doubling of the
    // mscale-1.0 arm (same yarn_inv_freq, yarnActive stays on). The pass-through
    // tail must not move: those dims never enter the rotation.
    try mlx.requireGpu();
    const s = mlx.gpuStream();

    var c = ModelConfig{};
    c.model_type = "qwen4_exp";
    c.head_dim = 256;
    c.partial_rotary_factor = 0.25;
    c.rope_theta = 10_000_000.0;
    c.rope_yarn = true;
    c.yarn_factor = 4.0;
    c.yarn_orig_max_pos = 262_144;
    c.yarn_attention_factor = 2.0;
    c.yarn_beta_fast = 32.0;
    c.yarn_beta_slow = 1.0;

    const rotary: c_int = 64;
    const half: usize = 32;
    const inv = try testing.allocator.alloc(f64, half);
    defer testing.allocator.free(inv);
    yarnSpec(&c).invFreq(inv);

    var den_f32: [half]f32 = undefined;
    for (inv, 0..) |v, i| den_f32[i] = @floatCast(1.0 / v);
    const fshape = [_]c_int{@intCast(half)};
    const freqs = mlx.mlx_array_new_data(&den_f32, &fshape, 1, .float32);
    defer _ = mlx.mlx_array_free(freqs);

    var t: Transformer = undefined;
    t.s = s;
    t.allocator = testing.allocator;
    t.config = c;
    t.rope_freqs_yarn = freqs;
    t.yarn_inv_freq = inv;
    try testing.expect(t.yarnActive());

    var ones: [256]f32 = @splat(1.0);
    const xshape = [_]c_int{ 1, 1, 1, 256 };
    const x = mlx.mlx_array_new_data(&ones, &xshape, 4, .float32);
    defer _ = mlx.mlx_array_free(x);

    const scaled = try t.ropeAtPositions(x, rotary, c.rope_theta, 1234.0, 1.0, 1);
    defer _ = mlx.mlx_array_free(scaled);

    t.config.yarn_attention_factor = 1.0;
    try testing.expect(t.yarnActive());
    const unscaled = try t.ropeAtPositions(x, rotary, c.rope_theta, 1234.0, 1.0, 1);
    defer _ = mlx.mlx_array_free(unscaled);

    var sf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sf);
    var uf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(uf);
    try mlx.check(mlx.mlx_astype(&sf, scaled, .float32, s));
    try mlx.check(mlx.mlx_astype(&uf, unscaled, .float32, s));
    try mlx.check(mlx.mlx_array_eval(sf));
    try mlx.check(mlx.mlx_array_eval(uf));
    const sv = mlx.mlx_array_data_float32(sf).?;
    const uv = mlx.mlx_array_data_float32(uf).?;
    for (0..64) |i| try testing.expectEqual(uv[i] * 2.0, sv[i]);
    for (64..256) |i| try testing.expectEqual(uv[i], sv[i]);
}

test "qwen4_exp yarn: the device denominator and the host spectrum are one spectrum" {
    // The engine spells the SAME YaRN spectrum twice: mlx_fast_rope and the
    // fused-angle probe take a DENOMINATOR array (angle = position / freqs),
    // while the host cos/sin tables (`fillCosSin`, `ropeAtPositions`) take
    // inv_freq (angle = position × inv_freq). If the two ever drift, prefill and
    // decode rotate differently — the model would hold one position for the
    // prompt and another for everything generated after it, and nothing errors.
    var c = ModelConfig{};
    c.model_type = "qwen4_exp";
    c.head_dim = 256;
    c.partial_rotary_factor = 0.25;
    c.rope_theta = 10_000_000.0;
    c.rope_yarn = true;
    c.yarn_factor = 4.0;
    c.yarn_orig_max_pos = 262_144;
    c.yarn_attention_factor = 1.138629436111989;
    c.yarn_beta_fast = 32.0;
    c.yarn_beta_slow = 1.0;
    c.max_position_embeddings = 1_048_576;

    // qwen4's ONE trunk rope: 64 of 256 dims, 32 frequencies — which is also
    // the width the QSA indexer rotates (it takes rope_dims from head_dim, not
    // from indexer_head_dim), so a single table serves every consumer.
    try testing.expectEqual(@as(u32, 64), c.yarnRotaryDim());
    try testing.expectApproxEqAbs(@as(f32, 0.25), c.yarnPartial(), 1e-9);
    try testing.expectEqual(@as(u32, 1_048_576), c.contextCap());

    var inv: [32]f64 = undefined;
    yarnSpec(&c).invFreq(&inv);
    var den: [32]f64 = undefined;
    computeYarnFreqs(
        &den,
        c.head_dim,
        c.yarnPartial(),
        c.rope_theta,
        c.yarn_factor,
        c.yarn_beta_fast,
        c.yarn_beta_slow,
        c.yarn_orig_max_pos,
        c.yarn_truncate,
    );
    for (inv, den) |a, b| try testing.expectApproxEqRel(@as(f64, 1.0), a * b, 1e-12);
    // Wavelengths run from one token (the fastest rotary dim) to the whole
    // stretched window (the slowest), strictly increasing — the shape every
    // consumer assumes when it indexes the table by dim.
    try testing.expectApproxEqRel(@as(f64, 1.0), den[0], 1e-12);
    for (den[1..], 1..) |b, i| try testing.expect(b > den[i - 1]);
    // The table's width is what the composed path multiplies: attention_factor
    // across exactly the first `yarnRotaryDim` dims of the head, 1.0 across the
    // 192 pass-through dims (scaling those would change values RoPE never wrote).
    const rotary = c.yarnRotaryDim();
    try testing.expectEqual(@as(usize, rotary), inv.len * 2);
    try testing.expect(rotary < c.head_dim); // partial rotary ⇒ a real pass-through tail
    try testing.expect(c.yarn_attention_factor > 1.0); // factor 4 ⇒ mscale 1.1386…
}

test "computeYarnFreqs matches HF _compute_yarn_parameters (full-attn rope)" {
    // Golden values from the reference YaRN math and the pure-Python
    // cross-check: head_dim 128, partial 0.5 → dim 64 →
    // 32 freqs; base 5e5, factor 32, beta_fast 32, beta_slow 1, orig_max 8192.
    // low/high correction dims truncate to 9/18.
    var freqs: [32]f64 = undefined;
    computeYarnFreqs(&freqs, 128, 0.5, 500000.0, 32.0, 32.0, 1.0, 8192, true);
    // Spot-check the denominator array mlx_fast_rope consumes (angle = pos/freqs).
    const golden = [_]struct { idx: usize, val: f64 }{
        .{ .idx = 0, .val = 1.000000000e+00 },
        .{ .idx = 1, .val = 1.506929076e+00 },
        .{ .idx = 2, .val = 2.270835240e+00 },
        .{ .idx = 7, .val = 1.764613870e+01 },
        .{ .idx = 15, .val = 1.324904288e+03 },
        .{ .idx = 16, .val = 2.868264127e+03 },
        .{ .idx = 23, .val = 3.992865387e+05 },
        .{ .idx = 31, .val = 1.061761980e+07 },
    };
    for (golden) |g| {
        try testing.expectApproxEqRel(g.val, freqs[g.idx], 1e-7);
    }
    // Below the low correction dim (9): pure extrapolation → freqs = base^(2i/64).
    try testing.expectApproxEqRel(std.math.pow(f64, 500000.0, 0.0), freqs[0], 1e-9);
    // Above the high correction dim (18): pure interpolation → freqs = factor·base^(2i/64).
    try testing.expectApproxEqRel(32.0 * std.math.pow(f64, 500000.0, @as(f64, 2 * 31) / 64.0), freqs[31], 1e-6);
}

test "splitMaskedSdpa256: verify-width array-mask rows split through the vector kernel match one dispatch" {
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x5b5b);
    const rnd = prng.random();
    // gqa 12 like qwen4_exp (24/2): qL 4 is the unfused fallback in stock MLX.
    const q_shape = [_]c_int{ 1, 24, 4, 256 };
    const kv_shape = [_]c_int{ 1, 2, 300, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(v);
    const mask = try attn256QsaMask(rnd, 4, 300);
    defer _ = mlx.mlx_array_free(mask);
    const scale: f32 = 1.0 / 16.0;
    const split = (try splitMaskedSdpa256(s, q, k, v, scale, mask)) orelse return error.SplitDeclined;
    defer _ = mlx.mlx_array_free(split);
    const ref = try attn256Reference(q, k, v, scale, "array", mask, s);
    defer _ = mlx.mlx_array_free(ref);
    try std.testing.expect(try attn256MaxDiff(split, ref, s) < 0.005);
    // qL 2 already fits the vector kernel: declined.
    const q2_shape = [_]c_int{ 1, 24, 2, 256 };
    const q2 = try attn256RandBf16(rnd, &q2_shape, s);
    defer _ = mlx.mlx_array_free(q2);
    const m2 = try attn256QsaMask(rnd, 2, 300);
    defer _ = mlx.mlx_array_free(m2);
    try std.testing.expect((try splitMaskedSdpa256(s, q2, k, v, scale, m2)) == null);
}

test "causal verify attention rows are bit-identical to decode at Qwen3.8 geometry" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    for ([_]KVQuantConfig{ KVQuantConfig.dense, KVQuantConfig.affine(8) }) |config| {
        for ([_]c_int{ 37, 253, 1022, 2045, 8190 }) |prefix| {
            for (2..9) |wi| {
                const width: c_int = @intCast(wi);
                var cache = try KVCache.initWithConfig(testing.allocator, 1, config);
                defer cache.deinit();
                const k = try qkvIdentityArray(s, 2, prefix + width, 256, 0.17, .bfloat16);
                defer _ = mlx.mlx_array_free(k);
                const v = try qkvIdentityArray(s, 2, prefix + width, 256, 3.37, .bfloat16);
                defer _ = mlx.mlx_array_free(v);
                var view = try cache.update(0, k, v, s, 0);
                defer view.deinit();
                const q = try qkvIdentityArray(s, 24, width, 256, 9.13, .bfloat16);
                defer _ = mlx.mlx_array_free(q);
                var verify = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(verify);
                if (try splitCausalSdpa(s, q, view.k, view.v, 1.0 / 16.0)) |out| {
                    _ = mlx.mlx_array_free(verify);
                    verify = out;
                } else {
                    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&verify, q, view.k, view.v, 1.0 / 16.0, "causal", .{ .ctx = null }, .{ .ctx = null }, false, s));
                }
                for (0..wi) |ri| {
                    const row: c_int = @intCast(ri);
                    const qr = try sliceAttentionSeq(s, q, row, row + 1);
                    defer _ = mlx.mlx_array_free(qr);
                    const kr = try sliceAttentionSeq(s, view.k, 0, prefix + row + 1);
                    defer _ = mlx.mlx_array_free(kr);
                    const vr = try sliceAttentionSeq(s, view.v, 0, prefix + row + 1);
                    defer _ = mlx.mlx_array_free(vr);
                    var tick = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(tick);
                    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&tick, qr, kr, vr, 1.0 / 16.0, "", .{ .ctx = null }, .{ .ctx = null }, false, s));
                    const got = try sliceAttentionSeq(s, verify, row, row + 1);
                    defer _ = mlx.mlx_array_free(got);
                    qkvExpectBitsEqual(s, tick, got) catch |err| {
                        std.debug.print("causal Qwen3.8 kv bits={d} prefix={d} width={d} row={d}\n", .{ config.bits, prefix, width, row });
                        return err;
                    };
                }
            }
        }
    }
}

test "splitCausalSdpa: parity vs single causal sdpa at every verify width (q 6..9)" {
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x5D9A);
    const rnd = prng.random();

    // Qwen3.6-27B verify geometry: gqa 6 (24/4 scaled to 6/2 hd 256), long-ish
    // KV so the chunk-A window (kL - (qL-5)) differs measurably from full kL.
    const kL: c_int = 193;
    var qL: c_int = 6;
    while (qL <= 9) : (qL += 1) {
        const q_shape = [_]c_int{ 1, 6, qL, 256 };
        const kv_shape = [_]c_int{ 1, 2, kL, 256 };
        const q = try attn256RandBf16(rnd, &q_shape, s);
        defer _ = mlx.mlx_array_free(q);
        const k = try attn256RandBf16(rnd, &kv_shape, s);
        defer _ = mlx.mlx_array_free(k);
        const v = try attn256RandBf16(rnd, &kv_shape, s);
        defer _ = mlx.mlx_array_free(v);
        const scale: f32 = 1.0 / 16.0;

        const split = (try splitCausalSdpa(s, q, k, v, scale)) orelse return error.SplitDeclined;
        defer _ = mlx.mlx_array_free(split);
        const ref = try attn256Reference(q, k, v, scale, "causal", .{ .ctx = null }, s);
        defer _ = mlx.mlx_array_free(ref);

        // Tolerance, not byte-identity: the split changes which sdpa kernel
        // runs and therefore the reduction order (the verify-lane rule).
        const max_diff = try attn256FiniteMaxDiff(split, ref, s);
        try std.testing.expect(max_diff < 0.005);
    }
}

test "splitCausalSdpa: declines outside its envelope" {
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x5DEC);
    const rnd = prng.random();

    const kv_shape = [_]c_int{ 1, 2, 64, 256 };
    const k = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k);

    // q_len 5 (vector path already serves it) and 10 (above the split window).
    const q5_shape = [_]c_int{ 1, 6, 5, 256 };
    const q5 = try attn256RandBf16(rnd, &q5_shape, s);
    defer _ = mlx.mlx_array_free(q5);
    try std.testing.expect((try splitCausalSdpa(s, q5, k, k, 1.0)) == null);
    const q10_shape = [_]c_int{ 1, 6, 10, 256 };
    const q10 = try attn256RandBf16(rnd, &q10_shape, s);
    defer _ = mlx.mlx_array_free(q10);
    try std.testing.expect((try splitCausalSdpa(s, q10, k, k, 1.0)) == null);

    // Batch 2 -> null (their gate; the exactness argument is B==1 only).
    const qb_shape = [_]c_int{ 2, 6, 7, 256 };
    const qb = try attn256RandBf16(rnd, &qb_shape, s);
    defer _ = mlx.mlx_array_free(qb);
    const kb_shape = [_]c_int{ 2, 2, 64, 256 };
    const kb = try attn256RandBf16(rnd, &kb_shape, s);
    defer _ = mlx.mlx_array_free(kb);
    try std.testing.expect((try splitCausalSdpa(s, qb, kb, kb, 1.0)) == null);

    // head_dim 128 -> null (MLX's own full kernel already covers it).
    const q128_shape = [_]c_int{ 1, 6, 7, 128 };
    const q128 = try attn256RandBf16(rnd, &q128_shape, s);
    defer _ = mlx.mlx_array_free(q128);
    const k128_shape = [_]c_int{ 1, 2, 64, 128 };
    const k128 = try attn256RandBf16(rnd, &k128_shape, s);
    defer _ = mlx.mlx_array_free(k128);
    try std.testing.expect((try splitCausalSdpa(s, q128, k128, k128, 1.0)) == null);

    // KV shorter than q -> null (no window to split).
    const kshort_shape = [_]c_int{ 1, 2, 5, 256 };
    const kshort = try attn256RandBf16(rnd, &kshort_shape, s);
    defer _ = mlx.mlx_array_free(kshort);
    const q7_shape = [_]c_int{ 1, 6, 7, 256 };
    const q7 = try attn256RandBf16(rnd, &q7_shape, s);
    defer _ = mlx.mlx_array_free(q7);
    try std.testing.expect((try splitCausalSdpa(s, q7, kshort, kshort, 1.0)) == null);
}

test "qk norm rope fused: bit-identical at the qwen geometry (rd=32, strided post-split q)" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x40FD);
    const rnd = prng.random();

    const hq: c_int = 8;
    const hk: c_int = 2;
    const eps: f32 = 1.0e-6;
    const eps_arr = mlx.mlx_array_new_float(eps);
    defer _ = mlx.mlx_array_free(eps_arr);
    const base = mlx.mlx_optional_float{ .value = 5000000.0, .has_value = true };
    const no_freqs = mlx.mlx_array{ .ctx = null };

    for ([_]c_int{ 0, 4093 }) |offset| {
        // q arrives as the attn_output_gate SPLIT view: [1,1,hq,256] with the
        // first 128 of each 256 being the query half — a strided view, which
        // ensure_row_contiguous must materialize without changing values.
        const qg_shape = [_]c_int{ 1, 1, hq, 256 };
        const q_gate = try attn256RandBf16(rnd, &qg_shape, s);
        defer _ = mlx.mlx_array_free(q_gate);
        var split_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(split_vec);
        try mlx.check(mlx.mlx_split(&split_vec, q_gate, 2, -1, s));
        var q_view = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_view);
        try mlx.check(mlx.mlx_vector_array_get(&q_view, split_vec, 0)); // [1,1,hq,128] strided

        const k_shape = [_]c_int{ 1, 1, hk * 128 };
        const k = try attn256RandBf16(rnd, &k_shape, s);
        defer _ = mlx.mlx_array_free(k);
        const w_shape = [_]c_int{128};
        const qw = try attn256RandBf16(rnd, &w_shape, s);
        defer _ = mlx.mlx_array_free(qw);
        const kw = try attn256RandBf16(rnd, &w_shape, s);
        defer _ = mlx.mlx_array_free(kw);

        const angles = try ropeAngleRow(s, 32, base, no_freqs, offset, std.testing.allocator);
        defer _ = mlx.mlx_array_free(angles);
        const pair = (try fusedQkNormRope(s, q_view, k, qw, kw, angles, null, eps_arr, hq, hk, 32)) orelse return error.FusedDeclined;
        defer _ = mlx.mlx_array_free(pair[0]);
        defer _ = mlx.mlx_array_free(pair[1]);

        // Composed reference over the SAME strided view (the shipped chain
        // norms the view directly).
        var qn = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(qn);
        try mlx.check(mlx.mlx_fast_rms_norm(&qn, q_view, qw, eps, s));
        const perm = [_]c_int{ 0, 2, 1, 3 };
        var qt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(qt);
        try mlx.check(mlx.mlx_transpose_axes(&qt, qn, &perm, 4, s));
        var rq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(rq);
        try mlx.check(mlx.mlx_fast_rope(&rq, qt, 32, false, base, 1.0, offset, no_freqs, s));

        const rk = try qkComposedRef(s, k, kw, null, hk, 32, base, no_freqs, offset, eps);
        defer _ = mlx.mlx_array_free(rk);

        try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(pair[0], rq, s));
        try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(pair[1], rk, s));
    }
}

test "qk norm rope fused hd-256: bit-identical at qwen 24q/4kv rd=64, S 1..16 incl. strided q" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x256A);
    const rnd = prng.random();

    const hq: c_int = 24;
    const hk: c_int = 4;
    const rd: c_int = 64;
    const eps: f32 = 1.0e-6;
    const eps_arr = mlx.mlx_array_new_float(eps);
    defer _ = mlx.mlx_array_free(eps_arr);
    const base = mlx.mlx_optional_float{ .value = 10000000.0, .has_value = true };
    const no_freqs = mlx.mlx_array{ .ctx = null };
    const perm = [_]c_int{ 0, 2, 1, 3 };

    for ([_]c_int{ 1, 3, 9, 16, 32 }) |seq| {
        for ([_]c_int{ 0, 8191 }) |offset| {
            // q arrives as the attn_output_gate SPLIT view: [1,S,hq,512] with
            // the first 256 of each 512 being the query half.
            const qg_shape = [_]c_int{ 1, seq, hq, 512 };
            const q_gate = try attn256RandBf16(rnd, &qg_shape, s);
            defer _ = mlx.mlx_array_free(q_gate);
            var split_vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(split_vec);
            try mlx.check(mlx.mlx_split(&split_vec, q_gate, 2, -1, s));
            var q_view = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(q_view);
            try mlx.check(mlx.mlx_vector_array_get(&q_view, split_vec, 0)); // [1,S,hq,256] strided

            const k_shape = [_]c_int{ 1, seq, hk, 256 };
            const k = try attn256RandBf16(rnd, &k_shape, s);
            defer _ = mlx.mlx_array_free(k);
            const w_shape = [_]c_int{256};
            const qw = try attn256RandBf16(rnd, &w_shape, s);
            defer _ = mlx.mlx_array_free(qw);
            const kw = try attn256RandBf16(rnd, &w_shape, s);
            defer _ = mlx.mlx_array_free(kw);

            const angles = try ropeAngleRows(s, rd, base, no_freqs, offset, seq, std.testing.allocator, 1.0);
            defer _ = mlx.mlx_array_free(angles);
            const pair = (try fusedQkNormRope256(s, q_view, k, qw, kw, angles, eps_arr, hq, hk, seq, rd)) orelse return error.FusedDeclined;
            defer _ = mlx.mlx_array_free(pair[0]);
            defer _ = mlx.mlx_array_free(pair[1]);

            // Composed reference over the SAME inputs (norm → transpose → rope).
            inline for (.{ .{ q_view, qw, pair[0] }, .{ k, kw, pair[1] } }) |case| {
                var n = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(n);
                try mlx.check(mlx.mlx_fast_rms_norm(&n, case[0], case[1], eps, s));
                var t = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(t);
                try mlx.check(mlx.mlx_transpose_axes(&t, n, &perm, 4, s));
                var r = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(r);
                try mlx.check(mlx.mlx_fast_rope(&r, t, rd, false, base, 1.0, offset, no_freqs, s));
                try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(case[2], r, s));
            }
        }
    }
}

test "qk norm rope fused: bit-identical to the composed chain (sliding, rd=128)" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x40FE);
    const rnd = prng.random();

    const hq: c_int = 8;
    const hk: c_int = 2;
    const eps: f32 = 1.0e-6;
    const eps_arr = mlx.mlx_array_new_float(eps);
    defer _ = mlx.mlx_array_free(eps_arr);
    const base = mlx.mlx_optional_float{ .value = 10000.0, .has_value = true };
    const no_freqs = mlx.mlx_array{ .ctx = null };

    for ([_]c_int{ 0, 977, 65533 }) |offset| {
        const q_shape = [_]c_int{ 1, 1, hq * 128 };
        const k_shape = [_]c_int{ 1, 1, hk * 128 };
        const w_shape = [_]c_int{128};
        const q = try attn256RandBf16(rnd, &q_shape, s);
        defer _ = mlx.mlx_array_free(q);
        const k = try attn256RandBf16(rnd, &k_shape, s);
        defer _ = mlx.mlx_array_free(k);
        const qw = try attn256RandBf16(rnd, &w_shape, s);
        defer _ = mlx.mlx_array_free(qw);
        const kw = try attn256RandBf16(rnd, &w_shape, s);
        defer _ = mlx.mlx_array_free(kw);

        const angles = try ropeAngleRow(s, 128, base, no_freqs, offset, std.testing.allocator);
        defer _ = mlx.mlx_array_free(angles);
        const pair = (try fusedQkNormRope(s, q, k, qw, kw, angles, null, eps_arr, hq, hk, 128)) orelse return error.FusedDeclined;
        defer _ = mlx.mlx_array_free(pair[0]);
        defer _ = mlx.mlx_array_free(pair[1]);

        const rq = try qkComposedRef(s, q, qw, null, hq, 128, base, no_freqs, offset, eps);
        defer _ = mlx.mlx_array_free(rq);
        const rk = try qkComposedRef(s, k, kw, null, hk, 128, base, no_freqs, offset, eps);
        defer _ = mlx.mlx_array_free(rk);

        try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(pair[0], rq, s));
        try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(pair[1], rk, s));
    }
}

test "qk norm rope fused: bit-identical incl. YaRN freqs + mscale (rd=64)" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x40FF);
    const rnd = prng.random();

    const hq: c_int = 48;
    const hk: c_int = 8;
    const eps: f32 = 1.0e-6;
    const eps_arr = mlx.mlx_array_new_float(eps);
    defer _ = mlx.mlx_array_free(eps_arr);

    // YaRN-shaped freqs vector [rd/2] f32 and a YaRN mscale table:
    // the YaRN factor on the rotated 64, exactly 1.0 on the pass-through tail.
    var freq_buf: [32]f32 = undefined;
    for (&freq_buf, 0..) |*f, i| f.* = std.math.pow(f32, 500000.0, @as(f32, @floatFromInt(i)) / 32.0);
    const f_shape = [_]c_int{32};
    const freqs = mlx.mlx_array_new_data(&freq_buf, &f_shape, 1, .float32);
    defer _ = mlx.mlx_array_free(freqs);
    const no_base = mlx.mlx_optional_float{ .value = 0, .has_value = false };

    var ms_buf: [128]f32 = undefined;
    for (&ms_buf, 0..) |*m, i| m.* = if (i < 64) 1.3465735912322998 else 1.0;
    const ms_shape = [_]c_int{128};
    const ms_f32 = mlx.mlx_array_new_data(&ms_buf, &ms_shape, 1, .float32);
    defer _ = mlx.mlx_array_free(ms_f32);
    var ms_bf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ms_bf);
    try mlx.check(mlx.mlx_astype(&ms_bf, ms_f32, .bfloat16, s));
    try mlx.check(mlx.mlx_array_eval(ms_bf));

    for ([_]c_int{ 0, 12345 }) |offset| {
        const q_shape = [_]c_int{ 1, 1, hq * 128 };
        const k_shape = [_]c_int{ 1, 1, hk * 128 };
        const w_shape = [_]c_int{128};
        const q = try attn256RandBf16(rnd, &q_shape, s);
        defer _ = mlx.mlx_array_free(q);
        const k = try attn256RandBf16(rnd, &k_shape, s);
        defer _ = mlx.mlx_array_free(k);
        const qw = try attn256RandBf16(rnd, &w_shape, s);
        defer _ = mlx.mlx_array_free(qw);
        const kw = try attn256RandBf16(rnd, &w_shape, s);
        defer _ = mlx.mlx_array_free(kw);

        const angles = try ropeAngleRow(s, 64, no_base, freqs, offset, std.testing.allocator);
        defer _ = mlx.mlx_array_free(angles);
        const pair = (try fusedQkNormRope(s, q, k, qw, kw, angles, ms_bf, eps_arr, hq, hk, 64)) orelse return error.FusedDeclined;
        defer _ = mlx.mlx_array_free(pair[0]);
        defer _ = mlx.mlx_array_free(pair[1]);

        const rq = try qkComposedRef(s, q, qw, ms_bf, hq, 64, no_base, freqs, offset, eps);
        defer _ = mlx.mlx_array_free(rq);
        const rk = try qkComposedRef(s, k, kw, ms_bf, hk, 64, no_base, freqs, offset, eps);
        defer _ = mlx.mlx_array_free(rk);

        try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(pair[0], rq, s));
        try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(pair[1], rk, s));
    }
}

test "sdpaForceFusedFor: only shapes MLX has a fused kernel for (hd 256, > 8 rows, q <= k)" {
    try testing.expect(sdpaForceFusedFor(true, 16, 4096, 256));
    try testing.expect(sdpaForceFusedFor(true, 9, 9, 256));
    // <= 8 rows is the vector kernel; its q*gqa <= 32 wall makes force_fused THROW.
    try testing.expect(!sdpaForceFusedFor(true, 8, 4096, 256));
    // hd <= 128 already fuses on its own; hd 192/exotic dims are not ours to force.
    try testing.expect(!sdpaForceFusedFor(true, 16, 4096, 128));
    try testing.expect(!sdpaForceFusedFor(true, 16, 4096, 192));
    try testing.expect(!sdpaForceFusedFor(true, 16, 8, 256));
    try testing.expect(!sdpaForceFusedFor(false, 16, 4096, 256));
}

test "sdpa force_fused: hd-256 causal blocks the gate admits match the unforced call (plumbing + shape gate)" {
    // The shapes sdpaForceFusedFor admits (hd 256, > 8 rows, q <= k): the
    // forced kernel (steel on M1-M4, the NAX variant on M5) must agree with
    // the unforced dispatch (unfused at hd 256 on non-NAX) and never throw.
    // <= 8 rows is NOT covered on purpose: that is the vector kernel, and
    // force_fused on a gqa-8 block there throws (the gate's whole point).
    const s = mlx.mlx_default_gpu_stream_new();
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    const none = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(none);
    const cases = [_][3]c_int{ .{ 16, 4096, 8 }, .{ 9, 512, 8 }, .{ 64, 64, 2 } }; // q, k, gqa
    for (cases) |c| {
        const q_shape = [_]c_int{ 1, 2 * c[2], c[0], 256 };
        const k_shape = [_]c_int{ 1, 2, c[1], 256 };
        try std.testing.expect(sdpaForceFusedFor(true, c[0], c[1], 256));
        const q = try attn256RandBf16(rnd, &q_shape, s);
        defer _ = mlx.mlx_array_free(q);
        const k = try attn256RandBf16(rnd, &k_shape, s);
        defer _ = mlx.mlx_array_free(k);
        const v = try attn256RandBf16(rnd, &k_shape, s);
        defer _ = mlx.mlx_array_free(v);
        var plain = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(plain);
        var forced = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(forced);
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&plain, q, k, v, 0.0625, "causal", none, .{ .ctx = null }, false, s));
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&forced, q, k, v, 0.0625, "causal", none, .{ .ctx = null }, true, s));
        const diff = try attn256MaxDiff(plain, forced, s);
        try std.testing.expect(diff < 5e-3);
    }
}

test "qwen4 MTP head KV stays dense under the engine-default trunk scheme unless opted in" {
    const t = std.testing;
    const saved = Transformer.mtp_head_kv_quant_override;
    defer Transformer.mtp_head_kv_quant_override = saved;
    Transformer.mtp_head_kv_quant_override = false;
    var xfm: Transformer = undefined;
    xfm.config = .{};
    xfm.config.num_hidden_layers = 4;
    xfm.config.head_dim = 64;
    xfm.allocator = t.allocator;
    var head: Qwen4Mtp = undefined;
    const head_cache = try KVCache.init(t.allocator, xfm.config.num_hidden_layers + 1);
    head.cache = head_cache;
    xfm.qwen4_mtp = head;
    defer xfm.qwen4_mtp.?.cache.deinit();
    try t.expectEqual(kv_quant.Scheme.off, xfm.qwen4_mtp.?.cache.config.scheme);
    const trunk = kv_quant.KVQuantConfig.engine_default;
    try t.expect(trunk.isQuant());
    try xfm.qwen4MtpApplyKvQuant(trunk);
    try t.expectEqual(kv_quant.Scheme.off, xfm.qwen4_mtp.?.cache.config.scheme);
    Transformer.mtp_head_kv_quant_override = true;
    try xfm.qwen4MtpApplyKvQuant(trunk);
    try t.expectEqual(trunk.scheme, xfm.qwen4_mtp.?.cache.config.scheme);
    try t.expectEqual(trunk.bits, xfm.qwen4_mtp.?.cache.config.bits);
}

test "head row coverage declines virtual widths before any stacked dispatch" {
    try testing.expect(mtpHeadRowsFit(&.{1}, 4));
    try testing.expect(mtpHeadRowsFit(&.{ 1, 1 }, 4));
    try testing.expect(mtpHeadRowsFit(&.{ 1, 1, 1, 1 }, 4));
    try testing.expect(!mtpHeadRowsFit(&.{ 1, 1, 1, 1, 1 }, 4));
    try testing.expect(!mtpHeadRowsFit(&.{ 1, 1, 1, 1, 1, 1, 1, 1 }, 4));
    try testing.expect(!mtpHeadRowsFit(&.{ 2, 3 }, 4));
    try testing.expect(!mtpHeadRowsFit(&.{ 2, 2 }, 4));
    try testing.expect(!mtpHeadRowsFit(&.{}, 4));
}

test "verify MoE rows preserve every value with fewer graph operations" {
    const path = std.c.getenv("MTP_TEST_MODEL") orelse return error.SkipZigTest;
    try mlx.requireGpu();
    const allocator = testing.allocator;
    const stream = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();
    var config = try model_mod.parseConfig(io, allocator, std.mem.span(path));
    defer if (config.ngram_table_path) |name| allocator.free(name);
    var weights = try model_mod.loadWeights(io, allocator, std.mem.span(path));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, allocator, config, &weights);
    defer xfm.deinit();
    try testing.expect(xfm.qwen4 != null);
    xfm.compileMoeRouting();
    var prng = std.Random.DefaultPrng.init(924);
    for ([_]usize{ 0, config.num_hidden_layers / 2, config.num_hidden_layers - 1 }) |layer| {
        const mw = &xfm.moe_layers.?[layer].mlp.moe;
        for ([_]usize{ 2, 4 }) |n| for ([_][4]c_int{ .{ 2, 2, 2, 2 }, .{ 5, 5, 5, 5 }, .{ 5, 3, 2, 4 } }) |lengths| {
            var inputs: [4]mlx.mlx_array = undefined;
            defer for (inputs[0..n]) |input| {
                _ = mlx.mlx_array_free(input);
            };
            for (0..n) |row| {
                inputs[row] = try attn256RandBf16(prng.random(), &.{ 1, lengths[row], @intCast(config.hidden_size) }, stream);
                const warm = try xfm.moeMLP(inputs[row], mw);
                try mlx.check(mlx.mlx_array_eval(warm));
                _ = mlx.mlx_array_free(warm);
            }
            var solo: [4]mlx.mlx_array = undefined;
            defer for (solo[0..n]) |value| {
                _ = mlx.mlx_array_free(value);
            };
            const start_solo = mlx.op_count.load(.monotonic);
            for (0..n) |row| solo[row] = try xfm.moeMLP(inputs[row], mw);
            const solo_ops = mlx.op_count.load(.monotonic) - start_solo;
            const shared_before = mtp_verify_shared_rows_calls;
            const start_group = mlx.op_count.load(.monotonic);
            const group = try xfm.moeVerifyRows(inputs[0..n], mw);
            const group_ops = mlx.op_count.load(.monotonic) - start_group;
            try testing.expectEqual(shared_before + 1, mtp_verify_shared_rows_calls);
            defer {
                for (group) |value| _ = mlx.mlx_array_free(value);
                allocator.free(group);
            }
            for (0..n) |row| try expectRowsByteEqual(allocator, stream, solo[row], group[row], "verify MoE rows");
            errdefer std.debug.print("[verify-moe] layer={d} N={d} widths={any} ops solo={d} grouped={d}\n", .{ layer, n, lengths[0..n], solo_ops, group_ops });
            if (group_ops >= solo_ops) {
                const reused_start = mlx.op_count.load(.monotonic);
                const reused = try xfm.moeVerifyRows(inputs[0..n], mw);
                const reused_ops = mlx.op_count.load(.monotonic) - reused_start;
                for (reused) |value| _ = mlx.mlx_array_free(value);
                allocator.free(reused);
                std.debug.print("[verify-moe] reused configurations: grouped={d} solo={d}\n", .{ reused_ops, solo_ops });
            }
            try testing.expect(group_ops < solo_ops);
        };
    }
}

test "qwen4 MTP adopt rejects a negative origin before mutating nonempty state" {
    const allocator = testing.allocator;
    var source = try KVCache.init(allocator, 1);
    defer source.deinit();
    var snapshot = try source.snapshot();
    defer snapshot.deinit();
    snapshot.step = 1;

    const pooled_data = [_]f32{0};
    const pooled = mlx.mlx_array_new_data(&pooled_data, &.{ 1, 1, 1 }, 3, .float32);
    var aux = SSMCacheEntrySnapshot{
        .conv_state = mlx.mlx_array_new(),
        .ssm_state = mlx.mlx_array_new(),
        .initialized = false,
        .qsa_pooled = pooled,
        .qsa_rows = 1,
    };
    defer ssmSnapshotDeinit(&aux);

    var xfm: Transformer = undefined;
    xfm.allocator = allocator;
    xfm.s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(xfm.s);
    const head_cache = try KVCache.init(allocator, 1);
    var head: Qwen4Mtp = undefined;
    head.cache = head_cache;
    head.entry = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false };
    head.seq_offset = 0;
    head.pos_base = 7;
    head.qsa_marks = .{};
    xfm.qwen4_mtp = head;
    defer {
        const live = &xfm.qwen4_mtp.?;
        live.cache.deinit();
        _ = mlx.mlx_array_free(live.entry.conv_state);
        _ = mlx.mlx_array_free(live.entry.ssm_state);
        ssmFreeQsaState(&live.entry);
        live.qsa_marks.deinit();
    }

    try testing.expectError(error.MtpHeadInvalidOrigin, xfm.qwen4MtpAdopt(&snapshot, &aux, &.{}, -1, 1));
    try testing.expectEqual(@as(usize, 0), xfm.qwen4_mtp.?.cache.step);
    try testing.expectEqual(@as(usize, 0), xfm.qwen4_mtp.?.seq_offset);
    try testing.expectEqual(@as(c_int, 7), xfm.qwen4_mtp.?.pos_base);
    try xfm.qwen4MtpAdopt(&snapshot, &aux, &.{}, 8, 1);
    try testing.expectEqual(@as(usize, 1), xfm.qwen4_mtp.?.cache.step);
    try testing.expectEqual(@as(usize, 1), xfm.qwen4_mtp.?.seq_offset);
    try testing.expectEqual(@as(c_int, 8), xfm.qwen4_mtp.?.pos_base);
}

test "single-stream verify feature selection excludes prefill and ordinary decoding" {
    var xfm: Transformer = undefined;
    var qwen4: qwen4_mod.Qwen4State = undefined;
    const old_features = trf.single_verify_test_features;
    defer trf.single_verify_test_features = old_features;
    trf.single_verify_test_features = @splat(true);
    inline for (.{ SingleVerifyFeature.routing, SingleVerifyFeature.down_reduce, SingleVerifyFeature.hc_graph }) |feature| {
        for ([_]bool{ false, true }) |is_qwen4| {
            xfm.qwen4 = if (is_qwen4) &qwen4 else null;
            for ([_]bool{ false, true }) |capture| {
                xfm.spec_capture_ssm = capture;
                for ([_]c_int{ 1, 2, 4 }) |batch| {
                    for ([_]c_int{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 4096 }) |width| {
                        try testing.expectEqual(is_qwen4 and capture and batch == 1 and width >= 2 and width <= 8, xfm.verifyFeatureEnabled(feature, false, batch, width));
                        try testing.expectEqual(is_qwen4, xfm.verifyFeatureEnabled(feature, true, batch, width));
                    }
                }
            }
        }
        xfm.qwen4 = &qwen4;
        xfm.spec_capture_ssm = true;
        trf.single_verify_test_features[@backingInt(feature)] = false;
        try testing.expect(!xfm.verifyFeatureEnabled(feature, false, 1, 5));
        try testing.expect(xfm.verifyFeatureEnabled(feature, true, 1, 5));
        trf.single_verify_test_features[@backingInt(feature)] = true;
    }
}

test "qwen4 decode ladder: stride resolution keeps an explicit off and leaves other arches off" {
    try testing.expectEqual(@as(u32, 4), Transformer.qwen4DecodeLadderStride(null));
    try testing.expectEqual(@as(u32, 0), Transformer.qwen4DecodeLadderStride("0"));
    try testing.expectEqual(@as(u32, 0), Transformer.qwen4DecodeLadderStride("off"));
    try testing.expectEqual(@as(u32, 0), Transformer.qwen4DecodeLadderStride(""));
    try testing.expectEqual(@as(u32, 2), Transformer.qwen4DecodeLadderStride("2"));
    try testing.expectEqual(@as(u32, 8), Transformer.qwen4DecodeLadderStride("auto"));
    try testing.expectEqual(@as(u32, 0), Transformer.decodeAsyncLadderStride(null));
}

test "qwen4 decode ladder: serial decode keeps the default off; an explicit stride fills host ids first, skips lazy ones" {
    try mlx.requireGpu();
    qwen4_standin_override = .{ .gdn = true, .attn = true, .mlp = true, .hc = true };
    defer qwen4_standin_override = null;
    var rig: Qwen4LadderRig = undefined;
    try rig.init();
    defer rig.deinit();
    const s = rig.fx.xfm().s;
    const strides = [_]u32{ 1, 4 };

    var off = Qwen4LadderRig.Slot.init();
    defer off.deinit();
    var on: [strides.len]Qwen4LadderRig.Slot = .{ Qwen4LadderRig.Slot.init(), Qwen4LadderRig.Slot.init() };
    defer for (&on) |*sl| sl.deinit();
    var lazy = Qwen4LadderRig.Slot.init();
    defer lazy.deinit();
    const zero = mlx.mlx_array_new_int(0);
    defer _ = mlx.mlx_array_free(zero);
    for ([_]u32{ 11, 42, 7 }) |tok| {
        const ids = ladderPleIds(&[_]u32{tok}, 1);
        defer _ = mlx.mlx_array_free(ids);
        // The batched default does not reach a serial forward.
        trf.decode_ladder_engaged = false;
        const want = try rig.step(&off, ids, 0);
        defer testing.allocator.free(want.logits);
        try testing.expect(want.pending);
        try testing.expect(!trf.decode_ladder_engaged);

        // Host-readable ids: the guard fills the leaf mid-build and the ladder fires.
        for (strides, &on) |stride, *sl| {
            trf.decode_ladder_engaged = false;
            const got = try rig.step(sl, ids, stride);
            defer testing.allocator.free(got.logits);
            try expectLogitBitsEqual(want.logits, got.logits);
            try testing.expect(!got.pending);
            try testing.expect(trf.decode_ladder_engaged);
        }

        // Ids still a lazy sample (pipelined decode): the ladder stands down instead of syncing on them.
        var lazy_ids = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(lazy_ids);
        try mlx.check(mlx.mlx_add(&lazy_ids, ids, zero, s));
        trf.decode_ladder_engaged = false;
        const got = try rig.step(&lazy, lazy_ids, 1);
        defer testing.allocator.free(got.logits);
        try expectLogitBitsEqual(want.logits, got.logits);
        try testing.expect(got.pending);
        try testing.expect(!trf.decode_ladder_engaged);
    }
}

test "qwen4 decode ladder: batched N=2 decode fills the PLE leaf first, logits and per-slot history bit-identical to off" {
    try mlx.requireGpu();
    qwen4_standin_override = .{ .gdn = true, .attn = true, .mlp = true, .hc = true };
    defer qwen4_standin_override = null;
    var rig: Qwen4LadderRig = undefined;
    try rig.init();
    defer rig.deinit();
    const alloc = testing.allocator;
    // Arms: off, on at stride 1, on at the shipped stride, and an unfilled leaf.
    const strides = [_]u32{ 0, 1, Transformer.qwen4DecodeLadderStride(null) };
    const prompts = [2][]const i32{ &.{ 5, 17, 42, 9 }, &.{ 11, 4, 19 } };
    var slots: [strides.len + 1][2]*Qwen4TestSlot = undefined;
    var made: usize = 0;
    defer for (0..made) |k| slots[k / 2][k % 2].deinit(alloc);
    for (&slots) |*arm| for (arm, prompts) |*sl, p| {
        sl.* = try Qwen4TestSlot.init(alloc, Qwen4LadderRig.LAYERS);
        made += 1;
        try rig.prefill(sl.*, p);
    };

    const steps = [_][2]i32{ .{ 13, 6 }, .{ 27, 3 }, .{ 40, 12 } };
    for (steps, 0..) |ids, k| {
        const want = try rig.batchedStep(&slots[0], &ids, strides[0], true);
        defer alloc.free(want.logits);
        try testing.expect(want.pending);
        if (k == 0) {
            // The harness has teeth: a leaf left unfilled changes the logits.
            const zeroed = try rig.batchedStep(&slots[strides.len], &ids, strides[0], false);
            defer alloc.free(zeroed.logits);
            try testing.expect(!std.mem.eql(u8, std.mem.sliceAsBytes(want.logits), std.mem.sliceAsBytes(zeroed.logits)));
        }
        for (strides[1..], slots[1..strides.len]) |stride, *arm| {
            trf.decode_ladder_engaged = false;
            const got = try rig.batchedStep(arm, &ids, stride, true);
            defer alloc.free(got.logits);
            try expectLogitBitsEqual(want.logits, got.logits);
            try testing.expect(!got.pending);
            try testing.expect(trf.decode_ladder_engaged);
            for (slots[0], arm.*) |a, b| {
                for (a.entries, b.entries) |*ea, *eb| {
                    try expectLadderPleHistoryEqual(ea, eb);
                    if (ea.aux_state.ctx != null or eb.aux_state.ctx != null) {
                        try testing.expect(ea.aux_state.ctx != null and eb.aux_state.ctx != null);
                        try qkvExpectBitsEqual(rig.fx.xfm().s, ea.aux_state, eb.aux_state);
                    }
                }
            }
        }
    }
}

test "qwen4 attention profiler: off and narrow widths stay idle, reset clears forward totals" {
    const saved_on = Qwen4AttnProf.on;
    defer Qwen4AttnProf.on = saved_on;
    defer Qwen4AttnProf.reset();
    Qwen4AttnProf.reset();
    Qwen4AttnProf.on = false;
    try std.testing.expect(!Qwen4AttnProf.begin(.{ .ctx = null }, 8192));
    Qwen4AttnProf.on = true;
    for ([_]c_int{ 1, 2, 15, 16 }) |width| {
        try std.testing.expect(!Qwen4AttnProf.begin(.{ .ctx = null }, width));
    }
    try std.testing.expectEqual(@as(u32, 0), Qwen4AttnProf.layers);
    try std.testing.expect(Qwen4AttnProf.begin(.{ .ctx = null }, 17));
    try std.testing.expect(Qwen4AttnProf.active);
    try std.testing.expectEqual(@as(u32, 1), Qwen4AttnProf.layers);
    Qwen4AttnProf.ns = .{ 1, 2, 3, 4 };
    Qwen4AttnProf.reset();
    try std.testing.expect(!Qwen4AttnProf.active);
    try std.testing.expectEqual(@as(u32, 0), Qwen4AttnProf.layers);
    try std.testing.expectEqual([_]u64{ 0, 0, 0, 0 }, Qwen4AttnProf.ns);
}

test "streamed deferred verification crosses only GDN MoE layers without PLE" {
    var layer: MoeLayerWeights = undefined;
    layer.attn = .{ .linear = std.mem.zeroes(LinearAttnWeights) };
    layer.mlp = .{ .moe = std.mem.zeroes(MoeMlpWeights) };
    layer.ple = null;
    try testing.expect(streamedSuccessorSafe(&layer));
    layer.ple = std.mem.zeroes(PleWeights);
    try testing.expect(!streamedSuccessorSafe(&layer));
    layer.ple = null;
    layer.attn = .{ .full = std.mem.zeroes(FullAttnWeights) };
    try testing.expect(!streamedSuccessorSafe(&layer));
    layer.attn = .{ .linear = std.mem.zeroes(LinearAttnWeights) };
    layer.mlp = .{ .dense = std.mem.zeroes(DenseMlpWeights) };
    try testing.expect(!streamedSuccessorSafe(&layer));
}

test "streamed deferred verification preserves outputs states and accounting on hits misses and workspace" {
    try mlx.requireGpu();
    const a = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var pack = try expert_stream_mod.writeTinyQuantCheckpoint(a, tmp.dir, 32, 4, 64, 64, 32, .{ 4, 4, 4 });
    defer pack.deinit();
    try streamDeferredCheckpoint(tmp.dir, &pack);
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try tmp.dir.realPath(testing.io, &path_buf);
    const path = path_buf[0..path_len];
    var rig: StreamDeferredRig = .{ .fx = undefined, .layers = undefined };
    try rig.init(&pack);
    defer rig.deinit();
    const x = rig.fx.xfm();
    const geom = expert_stream_mod.Geometry{ .layers = 4, .experts = 4, .hidden = 64, .intermediate = 64 };
    const per = try expert_stream_mod.expertBytesFor(a, path, geom, .quantized_split);
    const old_sync = expert_defer_sync_env;
    defer expert_defer_sync_env = old_sync;
    const warm = [_]u16{ 0, 1, 2, 3 };
    for ([_]u16{ 4, 2, 1 }) |slots| {
        var baseline = try expert_stream_mod.Engine.initWithOptions(a, path, geom, per * 4 * slots, x.s, .{ .layout = .quantized_split, .bounce_size = 1 << 20 });
        defer baseline.deinit();
        var candidate = try expert_stream_mod.Engine.initWithOptions(a, path, geom, per * 4 * slots, x.s, .{ .layout = .quantized_split, .bounce_size = 1 << 20 });
        defer candidate.deinit();
        if (slots == 4) for (0..4) |layer| {
            var b = try baseline.prepareHost(@intCast(layer), &warm);
            b.deinit();
            var c = try candidate.prepareHost(@intCast(layer), &warm);
            c.deinit();
        };
        const bs = try Qwen4TestSlot.init(a, 4);
        defer bs.deinit(a);
        const cs = try Qwen4TestSlot.init(a, 4);
        defer cs.deinit(a);
        const verified = expert_deferred_layers;
        const rolled_back = expert_deferred_rollbacks;
        for ([_]i32{ 1, 2, 3, 6, 9, 12, 4, 1 }) |token| {
            x.expert_stream = &baseline;
            expert_defer_sync_env = true;
            const expected = try bs.forward(x, &.{token});
            defer _ = mlx.mlx_array_free(expected);
            const ev = try qwen4ReadF32(a, expected, x.s);
            defer a.free(ev);
            x.expert_stream = &candidate;
            expert_defer_sync_env = false;
            const got = try cs.forward(x, &.{token});
            defer _ = mlx.mlx_array_free(got);
            const gv = try qwen4ReadF32(a, got, x.s);
            defer a.free(gv);
            try testing.expectEqualSlices(f32, ev, gv);
            try expectSlotStateEqual(a, x.s, bs, cs, 0);
            try testing.expectEqual(baseline.fill_experts_total, candidate.fill_experts_total);
            try testing.expectEqual(baseline.breakdown().hits, candidate.breakdown().hits);
            try testing.expectEqual(baseline.breakdown().union_members, candidate.breakdown().union_members);
        }
        try testing.expect(expert_deferred_layers > verified);
        if (slots == 4) try testing.expectEqual(rolled_back, expert_deferred_rollbacks);
        if (slots == 1) try testing.expect(expert_deferred_rollbacks > rolled_back);
        x.expert_stream = null;
    }
}

test "streamed rollback owns HC handle after injected MLX failure" {
    const current_data = [_]f32{3};
    const saved_data = [_]f32{7};
    var h = mlx.mlx_array_new_data(&current_data, &[_]c_int{1}, 1, .float32);
    var saved = mlx.mlx_array_new_data(&saved_data, &[_]c_int{1}, 1, .float32);
    const saved_ctx = saved.ctx;
    // Check ownership before reading h; the source owns cleanup until
    // transfer succeeds, including an injected reference failure.
    defer if (saved.ctx != null) {
        _ = mlx.mlx_array_free(saved);
    } else {
        _ = mlx.mlx_array_free(h);
    };
    const Rollback = struct {
        fn run(dst: *mlx.mlx_array, src: *mlx.mlx_array) !void {
            Transformer.restoreStreamH(dst, src);
            try mlx.check(mlx.mlx_array_eval(dst.*));
        }
    };
    mlx.fault.arm(1);
    const result = Rollback.run(&h, &saved);
    const fired = mlx.fault.didFire();
    mlx.fault.disarm();
    try testing.expectError(error.MlxError, result);
    try testing.expect(fired);
    try testing.expectEqual(saved_ctx, h.ctx);
    try testing.expect(saved.ctx == null);
    try mlx.check(mlx.mlx_array_eval(h));
    try testing.expectEqual(@as(f32, 7), mlx.mlx_array_data_float32(h).?[0]);
}

test "qsa indexer scalar rope applies yarn mscale to first rope_dims only" {
    // The QSA indexer is 128-wide with 64 rotary dims. mlx_fast_rope scale
    // stays 1.0 (never the mscale); yarnScaleRotated then scales the rotated
    // slice only. NOT applyMrope — that path already carries mscale via
    // fillCosSin. A 256-wide yarnScaleQK table would silently scale the
    // pass-through tail (or crash on a shape mismatch).
    try mlx.requireGpu();
    const s = mlx.gpuStream();

    var t: Transformer = undefined;
    t.s = s;
    t.config = .{};
    t.config.yarn_attention_factor = 2.0;
    t.config.rope_yarn = true;
    t.config.rope_theta = 10_000_000.0;

    var data: [128]f32 = undefined;
    for (&data, 0..) |*d, i| d.* = @floatFromInt(i + 1);
    const shape = [_]c_int{ 1, 1, 1, 128 };
    const x = mlx.mlx_array_new_data(&data, &shape, 4, .float32);
    defer _ = mlx.mlx_array_free(x);

    var roped = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(roped);
    try mlx.check(mlx.mlx_fast_rope(&roped, x, 64, false, mlx.mlx_optional_float{
        .value = t.config.rope_theta,
        .has_value = true,
    }, 1.0, 0, .{ .ctx = null }, s));

    var unscaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(unscaled);
    try mlx.check(mlx.mlx_array_set(&unscaled, roped));

    try t.yarnScaleRotated(&roped, 64);

    var a = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a);
    var b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b);
    try mlx.check(mlx.mlx_astype(&a, roped, .float32, s));
    try mlx.check(mlx.mlx_astype(&b, unscaled, .float32, s));
    try mlx.check(mlx.mlx_array_eval(a));
    try mlx.check(mlx.mlx_array_eval(b));
    const av = mlx.mlx_array_data_float32(a).?;
    const bv = mlx.mlx_array_data_float32(b).?;
    for (0..64) |i| try testing.expectEqual(bv[i] * 2.0, av[i]);
    for (64..128) |i| try testing.expectEqual(bv[i], av[i]);
}
