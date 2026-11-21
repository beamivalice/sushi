//! Flash-Next's own terms of the bill: the MTP head's KV, ring and state, and the thought-point checkpoint.
const qwen4_forward = @import("qwen4_forward.zig");
const model_mod = @import("../model.zig");
const bills = @import("../bills.zig");
const trunk_mod = @import("../trunk.zig");
const qwen4_exp = @import("qwen4_exp.zig");
const qwen4_qsa_attn = @import("qwen4_qsa_attn.zig");
const qwen4_qsa_gather = @import("qwen4_qsa_gather.zig");
const qwen4_gdn = @import("qwen4_gdn.zig");

// Hooks: bills.zig reaches these through the arch's `bill` declaration; a hook returns null where it has no say.

/// The mapped n-gram table's GPU bytes, billed as taken whichever model asks.
/// A Qwen request commits its whole cache at admission (`bills.cacheCommitBytes`), as GLM's does: admission counts
/// what it is still to grow beside the requests it admits after it.
pub fn commitBytes(config: *const model_mod.ModelConfig, prompt_len: usize, max_tokens: u32, kv_bits: u64, chunk: u64, mtp_on: bool) ?u64 {
    return bills.cacheCommitBytes(config, prompt_len, max_tokens, kv_bits, chunk, mtp_on);
}

/// The recurrent state sets a speculative round keeps past its new state from the verify to the rollback, at its
/// dearest verify width: a fold verify's per-row states (`qwen4_gdn.gdnVerifyFoldFor`), else the replay's pre-state.
/// A round of joined rows also holds each row's state in the group's merged state and that state's successor.
pub fn roundCaptureStates(config: *const model_mod.ModelConfig) ?u64 {
    var solo: u64 = 1;
    var rows: c_int = 2;
    while (rows <= 1 + @as(c_int, @intCast(@import("../mtp.zig").MAX_DEPTH))) : (rows += 1) {
        if (qwen4_gdn.gdnVerifyFoldFor(rows)) solo = @max(solo, @as(u64, @intCast(rows)));
    }
    const joins = @import("../cooler_side.zig").onEngine(config) and @import("qwen4_config.zig").supportsBatchedRows(config);
    return solo + (if (joins) joined_round_states else 0);
}

/// The states a round of joined rows holds a row past a solo round's (`qwen4_forward.forwardMoeBatchedRows` keeps the
/// group's merged state and its captures beside the rows'): two, measured at depths 1 and 6 by the contract cell
/// `a served request, plain or speculative, solo or joined...`.
pub const joined_round_states: u64 = 2;

/// On Sushi Cooler, the bytes each prefill row holds at the peak of each kind of layer, f16 unless named, from the
/// buffers alive there (`SUSHI_ENGINE_POOL_PEAK_TRACE`). A chunk peaks in one kind; the trunk's rows are held across all.
pub const PrefillRows = struct {
    /// A GatedDeltaNet layer: its qkv projection and the prework's copy, the f32 decay and beta rows, and four value-wide
    /// planes (the recurrence's output, the gate projection, its normed product, the output projection's input).
    gdn: u64,
    /// A full-attention layer: the gated query projection, four query-wide planes (the roped query, its gather, the
    /// attention output and the gated tail) and the rope's angle rows.
    attn: u64,
    /// A rate-grouped EXL3 MoE layer at its down combine (`bills.rateGroupedPrefillBytesPerRow`).
    moe: u64,
    /// The n-gram PLE layer's entry: nine stream-wide planes (key, its norm, the normed query, their product, the
    /// gated value, its norm, the conv's input, output and activation), the value and the n-gram embedding.
    ple: u64,
    /// The MTP head's forward over the chunk's rows (its drafter history), where the head runs: its MoE layer, and
    /// four stream-wide rows past the trunk's (the history row, its projection, the gated mix and the hc write).
    mtp: u64,
    /// The hc stream pending its write, and five hidden-wide rows: the token embedding, the read's mixed and injected
    /// rows, and the two trunk projections that outlive their layer.
    trunk: u64,
    /// The down GEMM's two planes a MoE layer frees while the GPU still binds them (its SwiGLU input and its output),
    /// when the next layer asks for its own: room for them lets that layer go on without waiting for the pass (a cap
    /// without them waits about once a layer: +1.3% at 4,096 rows on the stq4 cut).
    inflight: u64,

    pub fn peak(r: PrefillRows) u64 {
        return @max(@max(@max(r.gdn, r.attn), @max(r.moe, r.ple)), r.mtp) + r.trunk + r.inflight;
    }
};

pub fn coolerPrefillRows(config: *const model_mod.ModelConfig) PrefillRows {
    const hidden: u64 = config.hidden_size;
    const stream: u64 = @as(u64, config.hc_count) * hidden * 2;
    const value: u64 = @as(u64, config.linear_num_value_heads) * config.linear_value_head_dim;
    const qkv: u64 = 2 * @as(u64, config.linear_num_key_heads) * config.linear_key_head_dim + value;
    const query: u64 = @as(u64, config.num_attention_heads) * config.head_dim;
    return .{
        .gdn = 2 * qkv * 2 + 2 * @as(u64, config.linear_num_value_heads) * 4 + 4 * value * 2,
        .attn = 2 * query * 2 + 4 * query * 2 + 2 * @as(u64, config.head_dim) * 2,
        .moe = bills.rateGroupedPrefillBytesPerRow(config),
        .ple = if (config.ple_layer_idx >= 0) 9 * stream + hidden * 2 + @as(u64, config.ple_embed_dim) * 2 else 0,
        .mtp = if (bills.mtpHeadDefaultOn(config)) bills.rateGroupedPrefillBytesPerRow(config) + 4 * stream else 0,
        .trunk = stream + 5 * hidden * 2,
        .inflight = @as(u64, @max(config.num_experts_per_tok, 1)) * (hidden + @as(u64, config.moe_intermediate_size)) * 2,
    };
}

/// The exact Cooler prefill rows where every layer kind has its account: an STQ (rate-grouped EXL3) pack; null keeps
/// the fitted envelope.
pub fn coolerPrefillRowBytes(config: *const model_mod.ModelConfig) ?u64 {
    if (bills.rateGroupedPrefillBytesPerRow(config) == 0) return null;
    return coolerPrefillRows(config).peak();
}

/// The most rows a trellis trunk linear's matmul holds beside its decoded weight (`bills.trunkDecodeScratchBytes`): its
/// own layer kind's (attention, GatedDeltaNet, PLE) or a MoE layer's shared expert's, whose gate, up and SwiGLU planes
/// and output sit beside the router's f32 logits, ids and scores and the routed sum but never the routed planes; in the
/// MTP head, past its four stream-wide rows.
pub fn coolerPrefillDecodeSiteRowBytes(config: *const model_mod.ModelConfig) ?u64 {
    if (bills.rateGroupedPrefillBytesPerRow(config) == 0) return null;
    const r = coolerPrefillRows(config);
    const hidden: u64 = config.hidden_size;
    const top_k: u64 = @max(config.num_experts_per_tok, 1);
    const shared = 3 * @as(u64, config.shared_expert_intermediate_size) * 2 + 2 * hidden * 2 + @as(u64, config.num_experts) * 4 + top_k * 8;
    const head = if (bills.mtpHeadDefaultOn(config)) 4 * @as(u64, config.hc_count) * hidden * 2 + @max(r.attn, shared) else 0;
    return @max(@max(@max(r.gdn, r.attn), @max(r.ple, shared)), head) + r.trunk + r.inflight;
}

/// The rows held where a full-attention layer selects its QSA blocks: the trunk's alone.
pub fn coolerPrefillEntryRowBytes(config: *const model_mod.ModelConfig) ?u64 {
    if (bills.rateGroupedPrefillBytesPerRow(config) == 0) return null;
    return coolerPrefillRows(config).trunk;
}

pub fn pageCacheClaim() u64 {
    return qwen4_exp.page_cache_claim.load(.acquire);
}

/// The keys one QSA layer rebuilds dense from a quantized cache: what a forward of `fwd` rows cannot gather in place.
pub fn denseRebuildRows(config: *const model_mod.ModelConfig, fwd: u64) ?u64 {
    return qwen4_qsa_attn.qsaDenseRebuildRows(config.indexer_budget, config.indexer_compress_ratio, fwd);
}

/// A prefill width that gathers by block index holds a row-chunked score sheet, never the `[S, kv]` mask.
pub fn gatheredMaskBytes(config: *const model_mod.ModelConfig, fwd: u64, kv: u64) ?u64 {
    if (fwd < trunk_mod.FUSED256_MIN_Q_LEN or !qwen4_qsa_gather.qsaGatherEnabled()) return null;
    if (@import("../cooler_side.zig").onEngine(config))
        return qwen4_qsa_attn.qsaPrefillTransientBytesEngine(config.indexer_n_heads, fwd, kv, config.indexer_compress_ratio, config.indexer_head_dim);
    return qwen4_qsa_attn.qsaPrefillTransientBytes(config.indexer_n_heads, fwd, kv, config.indexer_compress_ratio, config.indexer_head_dim);
}

/// Per-token bytes of the MTP head's own KV at the scheme it was LOADED with, billed once per request when MTP is on.
pub fn headKvPerToken(config: *const model_mod.ModelConfig) ?u64 {
    const n = config.attnCacheLayerCount();
    if (n == 0) return 0;
    return bills.kvBytesPerTokenAtBits(config.kvBytesPerToken() / n, headKvBits(config).?);
}

/// The bits the MTP head's cache stores at (16: dense), which its rows grow on (`bills.cacheStorage`).
pub fn headKvBits(config: *const model_mod.ModelConfig) ?u64 {
    const head_cfg = qwen4_forward.qwen4MtpHeadKvConfig(bills.configuredKvQuantFor(config));
    return if (head_cfg.scheme == .off) 16 else head_cfg.bits;
}

pub fn headQsaRing(config: *const model_mod.ModelConfig) ?u64 {
    const n = config.attnCacheLayerCount();
    if (n == 0) return 0;
    return config.qsaRingBytes() / n;
}

pub fn headStatePerToken(config: *const model_mod.ModelConfig) ?u64 {
    const n = config.attnCacheLayerCount();
    if (n == 0) return 0;
    return bills.statePerTokenBilled(config) / n;
}

/// The checkpoint state a thinking request holds past its prefill's: the restore point behind its thought's closer, taken
/// mid-decode and held to the commit (`generate.ThoughtPoint`).
pub fn thoughtPoint(config: *const model_mod.ModelConfig, warm: bills.WarmPrefix) ?u64 {
    if (!warm.thought or bills.launchCheckpointStride(config) == 0) return 0;
    return config.ssmCheckpointBytes();
}

/// The most checkpoint states a request holds at once when its model runs an SSD tier: the write-through
/// frees each one once its file is written, and a request it cannot serve is thinned to the same number
/// (`scheduler.ssmCheckpointCap`). Null = every checkpoint stays in RAM.
pub fn residentCheckpointCap(config: *const model_mod.ModelConfig) ?u64 {
    if (!bills.diskTierOn(config)) return null;
    return trunk_mod.SSM_RESIDENT_MAX;
}

