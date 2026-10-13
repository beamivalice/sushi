//! Flash-Next (qwen4_exp) forward: attention, GDN wiring, native MTP head, decode/verify rows.

const trf = @import("transformer.zig");
const Transformer = trf.Transformer;
const std = @import("std");
const sushi_exl3 = @import("sushi_exl3");
const expert_exl3_kernels = sushi_exl3.kernels;
const mlx = @import("mlx.zig");
const mrope = @import("mrope.zig");
const kv_quant = @import("kv_quant.zig");
pub const KVQuantConfig = kv_quant.KVQuantConfig;
const qwen4_qsa = @import("qwen4_qsa.zig");
const qwen4_hc = @import("qwen4_hc.zig");

const DenseKVView = trf.DenseKVView;
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
const MoeLayerWeights = trf.MoeLayerWeights;
const MoeMlpWeights = trf.MoeMlpWeights;
const ProfClock = trf.ProfClock;
const QSA_SELECT_KERNEL_HEADER = qwen4_qsa.QSA_SELECT_KERNEL_HEADER;
const QSA_SELECT_KERNEL_SOURCE = qwen4_qsa.QSA_SELECT_KERNEL_SOURCE;
const QSA_SELECT_SPLIT_LOCAL_SOURCE = qwen4_qsa.QSA_SELECT_SPLIT_LOCAL_SOURCE;
const QSA_SELECT_SPLIT_MERGE_SOURCE = qwen4_qsa.QSA_SELECT_SPLIT_MERGE_SOURCE;
const QsaHeadMark = qwen4_qsa.QsaHeadMark;
const QsaHeadMarkSet = qwen4_qsa.QsaHeadMarkSet;
const SSMCacheEntry = trf.SSMCacheEntry;
const SSMCacheEntrySnapshot = trf.SSMCacheEntrySnapshot;
const StreamGdnUndo = trf.StreamGdnUndo;
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
const fusedQkNormRope = trf.fusedQkNormRope;
const fusedQkNormRope256 = trf.fusedQkNormRope256;
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
const moeDumpBeginForward = trf.moeDumpBeginForward;
const moeDumpForwardDone = trf.moeDumpForwardDone;
const moeRouterFusedEnabled = trf.moeRouterFusedEnabled;
const mtp_mod = trf.mtp_mod;
const noteExpertDeferred = trf.noteExpertDeferred;
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
const sdpaForceFused = trf.sdpaForceFused;
const splitCausalSdpa = trf.splitCausalSdpa;
const splitMaskedSdpa256 = trf.splitMaskedSdpa256;
const ssmFreeQsaState = trf.ssmFreeQsaState;
const ssmRestore = trf.ssmRestore;
const standinRef = trf.standinRef;
const streamedSuccessorSafe = trf.streamedSuccessorSafe;
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
    /// `mtp.MtpModel.ev_seed_*`: the last HEALTHY request's per-index
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
    mcfg.linear_attn_tail_from = 0;
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

    var join_lmhead = !self.embedding_mode;
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
        } else if (self.embedding_mode or row.ctx.skip_lm_head)
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
    var emb = try self.embedding(token_ids);
    {
        errdefer _ = mlx.mlx_array_free(emb);
        emb = try self.applyVisionEmbeddingsWith(ctx, emb, token_ids);
    }
    defer _ = mlx.mlx_array_free(emb);
    const x_shape = mlx.getShape(emb);
    const batch: c_int = x_shape[0];
    const seq_len: c_int = x_shape[1];
    const is_prefill = seq_len > 1;

    const reps = [_]c_int{ 1, 1, hc };
    var h = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_tile(&h, emb, &reps, 3, self.s));
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
        !diagEnvOnCached(&trf.expert_defer_sync_env, "SUSHI_EXPERT_DEFER_SYNC");
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

        if (lw.ple) |*pw| {
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
                    _ = mlx.mlx_array_set(slot, h);
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
    if (self.embedding_mode or ctx.skip_lm_head) {
        if (self.expert_stream) |engine| engine.finishForward(@intCast(batch * seq_len));
        return final;
    }
    errdefer _ = mlx.mlx_array_free(final);
    const logits = try self.lmHeadProject(final, ctx.argmax_only);
    _ = mlx.mlx_array_free(final);
    if (self.expert_stream) |engine| engine.finishForward(@intCast(batch * seq_len));
    return logits;
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
        // Mirror mlx-lm qwen3_next.py:130-134: reshape Q-proj output to [B, S, H, D*2]
        // then `mx.split(_, 2, axis=-1)` into (queries, gate). The single split op
        // replaces our prior two-slice pattern (2 dispatches → 1 dispatch). Adds up
        // across all `full_attention_interval` layers — was the dominant Qwen 3.5/3.6
        // hybrid decode gap vs mlx-lm (5.7% → ~tied).
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

    // Fused decode QK-norm+RoPE (the laguna kernel, hd-128 partial-rotary):
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
    // row to a scaled layer, exactly as laguna's yarn split does.
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
    // hd-256 sibling (qwen3.5/3.6 full-attention layers), decode AND
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
        // text-only qwen3_5). YaRN needs nothing here: the M-RoPE tables are
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
    if (la.combined_proj) return null;
    return self.verifyRowsJoined(.gdn, true, inputs, entries, &.{}, generations, layer, la);
}

pub fn gdnVerifyRows(self: *Transformer, inputs: []const mlx.mlx_array, entries: []const *SSMCacheEntry, generations: []const u64, layer: usize, la: *const LinearAttnWeights) !?[]mlx.mlx_array {
    if (la.combined_proj) return null;
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
