//! The engine's handle on a loaded model: the per-request `ForwardCtx` and the operations that dispatch over the
//! loaded arch through the registry. The trunk math lives in `trunk.zig`.
const mlx = @import("mlx.zig");
const kv_cache = @import("kv_cache.zig");
const ssm_cache = @import("ssm_cache.zig");
const KVCache = kv_cache.KVCache;
const SSMCacheEntry = ssm_cache.SSMCacheEntry;
const trunk = @import("trunk.zig");
const Trunk = trunk.Trunk;
const arch_forward = @import("arch/forward.zig");
const arch_life = @import("arch/life.zig");
const arch_request = @import("arch/request.zig");
const arch_registry = @import("arch/registry.zig");
const arch_state = @import("arch/arch_state.zig");

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
    /// Return the post-final-norm hidden instead of logits (caller frees
    /// either way), skipping the lm_head projection.
    skip_lm_head: bool = false,
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
    /// The ids are still a lazy sample: an arch's host-side lookup keyed on them waits for `arch_state.flushDeferred`.
    lazy_ids: bool = false,
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


/// `arch_state.setLayerCapture` on `ctx`, for a caller that reaches the engine only through this file.
pub fn setLayerCapture(ctx: *ForwardCtx, input: ?mlx.mlx_array, skip_ple: bool) void {
    arch_state.setLayerCapture(&ctx.arch_ctx, input, skip_ple);
}

/// Whether this model is single-flight at admission (`slot_admission.modelExclusiveDecode`).
pub fn ownsModuleDecodeState(self: *const Trunk) bool {
    return arch_request.ownsModuleDecodeState(arch_registry.handleOf(@constCast(self)));
}

/// A streamed expert engine is read-only per-model state every slot reads; so may an arch's own state be
/// (`arch_state.sharesReadonly`).
pub fn sharesModuleReadonlyState(self: *const Trunk) bool {
    if (self.expert_stream != null) return true;
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
    return arch_state.specRollsBack(&self.arch_state);
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
