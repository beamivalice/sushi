//! The engine's handle on a loaded model: the per-request `ForwardCtx` and the operations that dispatch over the
//! loaded arch through the registry. The trunk math lives in `trunk.zig`.
const std = @import("std");
const mlx = @import("mlx.zig");
const kv_cache = @import("kv_cache.zig");
const ssm_cache = @import("ssm_cache.zig");
const qwen4_qsa_attn = @import("qwen4/qwen4_qsa_attn.zig");
const qwen4_hc = @import("qwen4/qwen4_hc.zig");
const KVCache = kv_cache.KVCache;
const SSMCacheEntry = ssm_cache.SSMCacheEntry;
const QsaArmCounts = qwen4_qsa_attn.QsaArmCounts;
const PlePending = qwen4_hc.PlePending;
const trunk = @import("trunk.zig");
const Trunk = trunk.Trunk;
const arch_forward = @import("arch/forward.zig");
const arch_life = @import("arch/life.zig");
const arch_request = @import("arch/request.zig");
const arch_registry = @import("arch/registry.zig");
const arch_state = @import("arch/arch_state.zig");
const dsv41_state = @import("dsv41/dsv41_state.zig");

// ── Forward context ──
//
// Routes per-request mutable state (KV cache, MoE seq offset, SSM entries,
// hidden-state capture target, vision embeddings) into the forward pass via
// a single struct. The legacy single-slot path uses `Trunk.defaultCtx()`
// which points at the Trunk's own fields — semantically identical to the
// pre-refactor code. Concurrent batching (Phase 1+) constructs one ctx per
// in-flight request so multiple slots can share a Trunk's weights while
// owning their own caches.
/// DFlash capture request: intermediate layer OUTPUTS, `hidden_states[i+1]`
/// semantics (the residual stream AFTER layer `ids[k]` completes, before the
/// next layer's input norm). Standard forward path only (v1).
pub const CaptureLayers = struct {
    /// Trunk layer indices to capture, ascending.
    ids: []const u32,
    /// Filled per forward: `out[k]` = h after layer `ids[k]`, ALL positions
    /// `[B, L, H]`, refcount-shared with the live graph. Caller owns the
    /// handles (init with `mlx_array_new`, free after use).
    out: []mlx.mlx_array,
    /// The residual entering the first layer (`hidden_states[0]`), same
    /// ownership as `out`. Honored by `forwardMoeWith` only.
    input: ?*mlx.mlx_array = null,
    /// qwen4_exp: average the layer output over this many hyper-connection
    /// streams, giving `[B, L, H]` taps (0 = the raw `[B, L, hc*H]` stream).
    hc_mean: u32 = 0,
};

pub const ForwardCtx = struct {
    cache: *KVCache,
    moe_seq_offset: *usize,
    ssm_entries: ?[]SSMCacheEntry,
    ssm_member_gen: u64 = 0,
    /// The arch's own per-forward context (`arch/arch_state.zig`): a field per enabled arch.
    arch_ctx: arch_state.Ctx = .{},
    capture_hidden: ?*mlx.mlx_array,
    /// Like `capture_hidden` but receives the FULL post-final-norm hidden
    /// `[B, L, H]` (all positions, refcount-shared) instead of the last
    /// position only. Used by the Qwen MTP head, whose committed-history
    /// cache needs the trunk hidden at every verify/prefill position.
    capture_hidden_all: ?*mlx.mlx_array = null,
    /// qwen4_exp: receives the PRE-mixer hyper-connection stream `[B, L,
    /// hc*hidden]` (all positions) — what the MTP head consumes.
    capture_stream_all: ?*mlx.mlx_array = null,
    /// DFlash: capture the residual stream at these layer OUTPUTS (all
    /// positions). Honored by `forwardStandardWith` only — DflashModel.bind
    /// refuses non-standard-path targets so this can never be silently
    /// ignored on an engaged path.
    capture_layers: ?*CaptureLayers = null,
    /// PLD spec-decode: when true, the GatedDeltaNet trunk records per-position
    /// SSM/conv state during the verify forward (see `SSMCacheEntry.spec_*`),
    /// so partial-accept rollback needs no re-forward. Set only by `nextPld`
    /// for the verify pass on a GDN model; the flag reaches the layers via
    /// `Trunk.spec_capture_ssm`.
    capture_ssm_seq: bool = false,
    /// MiMo spec verify: every row of this multi-row forward computes with its
    /// own decode tick's arithmetic (`mimoVerifyRowsAttn`), so greedy MTP is serial.
    verify_rows: bool = false,
    vision_embeddings: ?mlx.mlx_array,
    /// Chunked vision prefill (issue #197): number of image/audio placeholder
    /// tokens already spliced by earlier chunks of this prefill. The splice's
    /// row lookup is a cumsum over the CURRENT forward's placeholder mask, so
    /// without this every chunk would restart at vision row 0. Set per chunk
    /// by the generate.zig prefill loop; 0 for unchunked/decode forwards.
    vision_splice_offset: usize = 0,
    /// DeepSeek-V4.1 chunked prefill: the prompt tokens after this chunk, whose Engram rows the model reads while the
    /// chunk runs. Empty for the last chunk and for decode.
    ahead_ids: []const i32 = &.{},
    /// DeepSeek-V4.1 chunked prefill: the fewest rows the chunk after this one can have, 0 when none follows in the
    /// prefill loop. A model whose decoder tail nothing reads may leave state only (`dsv41_model.extendState`).
    ahead_min_rows: usize = 0,
    /// A restore point is taken at this chunk's end.
    holds_point: bool = false,
    /// Qwen3-VL interleaved M-RoPE. `mrope_pos` is the server-computed flat
    /// [3 × mrope_total] i32 position-id table (axis-major: t,h,w rows), borrowed
    /// from the slot. `mrope_delta` shifts the scalar decode RoPE (decode tokens
    /// are text → t=h=w). `mrope_cos/sin_cur` are the per-prefill-chunk cos/sin
    /// tables, built once in forwardMoeWith and consumed by every full-attn layer
    /// that chunk, then freed. All null/0 for non-Qwen / text-only requests.
    mrope_pos: ?[]const i32 = null,
    mrope_total: usize = 0,
    mrope_delta: i32 = 0,
    mrope_cos_cur: ?mlx.mlx_array = null,
    mrope_sin_cur: ?mlx.mlx_array = null,
    /// qwen4_exp QSA: the bool `[B, 1, S, kv]` mask (indexer-selected blocks
    /// ∧ causal) `gatedFullAttnWith` must apply this layer. Null-ctx = dense.
    qsa_mask: mlx.mlx_array = .{ .ctx = null },
    /// qwen4_exp QSA: the sorted `[B, S, kb]` int32 block selection (INT_MAX
    /// past each row's count) for the three gather arms — `gatherQsa256`
    /// (prefill), `qsaVerifyGatherAttn` (S = 2..15) and `qsaDecodeGatherAttn`
    /// (S == 1); set INSTEAD of `qsa_mask`. Null-ctx = not selected.
    qsa_blocks: mlx.mlx_array = .{ .ctx = null },
    /// Which QSA arm served each full-attention call of THIS request. The
    /// `[qsa-arms]` line is a per-request meter, so the tally is owned by the
    /// seam that logs it (`Generator.logQsaArms` over the Generator's own
    /// ctx) — a Trunk-level counter merged every concurrent slot's
    /// calls (batched decode + interleaved prefill) into whichever request
    /// finished first. A batched forward attributes each call to every
    /// participating slot's own ctx (`noteQsaArmForSlots`).
    ///
    /// Scoped to the TRUNK: the qwen4 MTP head's forward builds its own ctx
    /// and its single QSA layer is not part of this tally.
    qsa_arms: QsaArmCounts = .{},
    /// Return the post-final-norm hidden instead of logits (caller frees
    /// either way), skipping the lm_head projection.
    skip_lm_head: bool = false,
    stream_in: ?mlx.mlx_array = null,
    skip_ple: bool = false,
    /// Set by the Generator when this request consumes ONLY the argmax of
    /// the logits (greedy or top-1 sampling, no logit-modifying penalties,
    /// no per-token logprobs, no grammar mask). Gates the certified lm_head
    /// prune, whose output row is bit-identical at every candidate slot and
    /// certified-below-the-winner elsewhere — exactly enough for an argmax,
    /// not for a tail distribution. Default false = dense path everywhere.
    argmax_only: bool = false,
    /// The caller's next act is to evaluate a value computed from the returned logits, so the forward may hand them
    /// back unevaluated (DeepSeek's one-row decode). Nothing else may read the logits' bytes first.
    lazy_logits: bool = false,
    /// Batched decode over a GatedDeltaNet trunk (`forwardMoeBatchedDecode`).
    /// Non-null means: this forward runs at `batch == batch_slots.len`,
    /// `seq_len == 1`, and every per-slot KV cache lives on the slot ctxs
    /// here — NOT on `ctx.cache`, which is only a shape donor in this mode.
    /// The GDN layers need no branch (their state is merged into this ctx's
    /// `ssm_entries` at B=N and split back out by the driver); only the
    /// full-attention layers do, because K/V lengths differ per slot and
    /// each slot ropes at its own offset.
    ///
    /// `batch_rope_offsets` is the matching int32 `[N]` offset array for
    /// `mlx_fast_rope_dynamic` — already carrying each slot's M-RoPE delta,
    /// so the attention layer never re-derives a per-slot offset.
    batch_slots: ?[]const *ForwardCtx = null,
    batch_rope_offsets: ?mlx.mlx_array = null,
    /// MiMo batched decode (`forwardMimoBatchedDecode`): row i of this `[1, N]` forward is
    /// slot i's decode tick, its attention and KV append on `batch_rows[i]`'s cache at
    /// `batch_row_offsets[i]`; every other op already computes rows independently.
    batch_rows: ?[]const *ForwardCtx = null,
    batch_row_offsets: []const u32 = &.{},
    /// MTP head: one QSA entry, not `ssm_entries[layer]`. Null on the trunk.
    qsa_entry: ?*SSMCacheEntry = null,
    /// Absolute position of this slot's QSA key row 0 (0 on the trunk).
    qsa_pos_base: c_int = 0,
    /// qwen4_exp pipelined decode: the n-gram PLE rows are a HOST gather keyed
    /// on the token ids, and the pipelined step builds its graph on a token
    /// that is still a lazy sample. With `ple_defer` set, `pleEmbedding` hands
    /// the graph an unfilled leaf and records it here; the caller fills it via
    /// `Trunk.flushDeferredPle` (syncing on the token THEN) right before
    /// the graph is evaluated. Set by `generate.lazyForward` (decode) and by
    /// the MTP verify build in `Generator.nextMtp` (whose `[1, 1+m]` input is
    /// the still-lazy draft chain); a forward evaluated with `ple_pending`
    /// still set reads zero rows.
    ple_defer: bool = false,
    ple_pending: ?PlePending = null,
};


// ── The loaded model's operations: each dispatches over the arch the model was loaded as ──

/// Default forward context, routing through the Trunk's own state.
/// Used by the single-slot legacy path and by Phase-2 prefill on a slot
/// that has had its KVCache temporarily swapped onto the Trunk.
pub fn defaultCtx(self: *Trunk) ForwardCtx {
    return .{
        .cache = &self.cache,
        .moe_seq_offset = &self.moe_seq_offset,
        .ssm_entries = self.ssm_entries,
        .capture_hidden = null,
        .vision_embeddings = self.vision_embeddings,
        .arch_ctx = arch_state.ctxOf(&self.arch_state),
    };
}

/// Module pointer fields that hold READ-ONLY per-model state (qwen4: the
/// n-gram hash + mmapped table). Every per-request thing lives on the
/// slot's `SSMCacheEntry` (GDN pair, PLE window + tokens, QSA keys +
/// pooled blocks), so slots interleave AND batch freely. The one
/// module-owned piece on this arch is the MTP head's cache, and the
/// scheduler makes the SLOT that uses it exclusive (`slotExclusiveDecode`).
/// An arch's own state may share read-only weights too (`arch_state.sharesReadonly`).
pub const module_shared_readonly_fields = [_][]const u8{ "qwen4", "expert_stream" };

/// Whether this model is single-flight at admission (`slot_admission.modelExclusiveDecode`).
pub fn ownsModuleDecodeState(self: *const Trunk) bool {
    return arch_request.ownsModuleDecodeState(arch_registry.handleOf(@constCast(self)));
}

pub fn sharesModuleReadonlyState(self: *const Trunk) bool {
    inline for (module_shared_readonly_fields) |f| {
        if (@field(self, f) != null) return true;
    }
    return arch_state.sharesReadonly(&self.arch_state);
}

/// Spec wiring class: module archs run only their OWN MTP head
/// (`moduleStateSpecRollback`), PLD/drafters stay off — unmeasured there.
/// Covers both the owned and the shared-readonly lists so a shared arch
/// does not fall onto the generic wiring and pick up PLD by default.
pub fn moduleSpecWiring(self: *const Trunk) bool {
    return self.ownsModuleDecodeState() or self.sharesModuleReadonlyState();
}

/// Can this arch's MODULE-owned decode state be rolled back across a
/// speculative verify?
///
/// This is what `ownsModuleDecodeState` actually stood in for: the reason
/// module-owned archs had every spec mode forced off is that a rejected
/// draft tail cannot be rewound, not the ownership itself. An arch earns
/// a true here by shipping per-position state capture plus a rewind,
/// pinned against fresh forwards at every accepted count. dsv4 cannot:
/// its rings and compressed caches have no per-position capture, and it
/// has DSpark for drafting anyway.
pub fn moduleStateSpecRollback(self: *const Trunk) bool {
    // qwen4_exp: GDN per-position capture + PLE window/token capture +
    // positional QSA key histories (`ssmRollbackFromCapture`), so the MTP
    // head's verify can rewind. dsv4 cannot. deepseek_v41 cuts its own rows
    // back to the accepted prefix (`dsv41_model.finishVerify`).
    if (dsv41_state.modelOf(self)) |m| return m.dspark != null;
    return self.qwen4_mtp != null;
}

/// Is this arch's MTP head the checkpoint's OWN trained layer AND measured
/// no-worse-than-serial across the whole context ladder on this family?
///
/// `defaultEnableMtp`'s `is_moe` caution is about a BOLTED-ON sidecar
/// paying expert routing in a verify it was never designed around. It does
/// not describe a head that ships inside the checkpoint — dsv4's DSpark
/// stages already get the same exemption through `dsv4_stages`.
///
/// The bar for adding an arch here is a MEASUREMENT, not the head's
/// provenance: no losing cell on the context ladder against the same
/// model's serial decode, on prompt text that is not trivially draftable.
pub fn nativeMoeMtpHeadMeasured(self: *const Trunk) bool {
    _ = self;
    return false;
}

pub fn forward(self: *Trunk, token_ids: mlx.mlx_array) !mlx.mlx_array {
    var ctx = self.defaultCtx();
    return self.forwardWith(&ctx, token_ids);
}

pub fn forwardWith(self: *Trunk, ctx: *ForwardCtx, token_ids: mlx.mlx_array) !mlx.mlx_array {
    return arch_forward.forward(arch_registry.handleOf(self), ctx, token_ids);
}

/// Project a hidden state through the trunk's lm_head, dense (no argmax
/// prune). DFlash draft logits come from the assistant's hidden through
/// THIS head — the reference applies the bare Linear (no softcap /
/// output_multiplier; both are monotone, so draft argmax is unaffected).
pub fn lmHeadForDraft(self: *const Trunk, x: mlx.mlx_array) !mlx.mlx_array {
    return arch_forward.lmHeadForDraft(arch_registry.handleOf(@constCast(self)), x);
}

/// Frees the model: the arch's own state first, then the trunk's.
pub fn deinit(self: *Trunk) void {
    arch_life.deinit(arch_registry.handleOf(self));
}

/// Empties every cache for a new request.
pub fn resetCache(self: *Trunk) !void {
    return arch_life.resetCache(arch_registry.handleOf(self));
}

/// Pre-faults the weights and compiles the first-touch kernels before the first request.
pub fn warmup(self: *Trunk) !void {
    return arch_life.warmup(arch_registry.handleOf(self));
}
