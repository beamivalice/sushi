const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const model_discovery = @import("model_discovery.zig");
const expert_quant = @import("expert_quant.zig");
const expert_io = @import("expert_io.zig");
const sushi_exl3 = @import("sushi_exl3");
const expert_exl3 = sushi_exl3.format;
const tokenizer_mod = @import("tokenizer.zig");
const qwen4_exp = @import("qwen4_exp.zig");
const kv_quant_mod = @import("kv_quant.zig");
const mtp_acceptance_mod = @import("mtp_acceptance.zig");

pub const HiddenAct = enum { gelu_approx, gelu, silu, relu_sq };

/// MLX quantization mode from config.json's `quantization.mode`. All
/// non-affine modes store NO `.biases` tensors (per-group fp8-encoded uint8
/// scales only) but share the packed-u32 weight layout, so supporting them is
/// a matter of skipping the biases fetch and passing the right mode string to
/// the mlx quantized ops. Tag names match the mlx-c mode strings exactly.
/// Upper bound on a vision tower's per-layer type table (muse ships 50).
pub const MAX_VISION_LAYERS = 64;

/// A MiMo-ViT block's attention: all patches, or a band over the image's
/// patches in row-major or column-major merge-unit order.
pub const MimoVitAttn = enum { full, row, col };

/// `MuseGlimmerImageProcessor.max_image_tokens` — MERGED tokens, not pixels.

pub const QuantMode = enum {
    affine,
    nvfp4,
    mxfp4,
    mxfp8,

    pub fn fromString(name: []const u8) ?QuantMode {
        return std.meta.stringToEnum(QuantMode, name);
    }

    /// Mode string for mlx_quantized_matmul / mlx_gather_qmm / mlx_dequantize.
    pub fn cstr(self: QuantMode) [*:0]const u8 {
        return switch (self) {
            .affine => "affine",
            .nvfp4 => "nvfp4",
            .mxfp4 => "mxfp4",
            .mxfp8 => "mxfp8",
        };
    }

    /// Affine is the only mode whose checkpoints carry per-group biases.
    pub fn hasBiases(self: QuantMode) bool {
        return self == .affine;
    }
};

pub const LayerBlockType = enum { attention, gated_conv, mamba2, mlp, moe };

/// Sentence-transformers pooling operation for embedding requests (issue
/// #116): masked mean over real positions, the CLS token (position 0), or the
/// last real (non-padding) token. Every mode is followed by L2 normalization.
pub const PoolingMode = enum {
    mean,
    cls,
    last_token,

    pub fn fromString(s: []const u8) ?PoolingMode {
        if (std.mem.eql(u8, s, "mean")) return .mean;
        if (std.mem.eql(u8, s, "cls")) return .cls;
        if (std.mem.eql(u8, s, "last_token")) return .last_token;
        return null;
    }
};

/// Parse a sentence-transformers `1_Pooling/config.json`. Returns the pooling
/// mode when the file declares one we implement, null when the content isn't a
/// pooling config at all (malformed JSON, unrelated object — best-effort, like
/// generation_config.json), and `error.UnsupportedPoolingMode` when the file
/// DOES declare pooling but only modes we don't implement (weighted-mean,
/// max): serving those checkpoints mean-pooled would be silent corruption.
pub fn parsePoolingSidecar(content: []const u8) !?PoolingMode {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const parsed = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), content, .{}) catch return null;
    if (parsed != .object) return null;
    const obj = parsed.object;

    const getBool = struct {
        fn get(o: std.json.ObjectMap, key: []const u8) bool {
            if (o.get(key)) |v| {
                if (v == .bool) return v.bool;
            }
            return false;
        }
    }.get;
    // ST configs set exactly one mode true; check ours most-specific first.
    if (getBool(obj, "pooling_mode_lasttoken")) return .last_token;
    if (getBool(obj, "pooling_mode_cls_token")) return .cls;
    if (getBool(obj, "pooling_mode_mean_tokens")) return .mean;
    // Declares pooling, but none we support → refuse rather than mean-pool.
    var it = obj.iterator();
    while (it.next()) |e| {
        if (std.mem.startsWith(u8, e.key_ptr.*, "pooling_mode_")) return error.UnsupportedPoolingMode;
    }
    return null;
}

/// Known-family pooling fallback for checkpoints that ship neither an explicit
/// `pooling_mode` nor the ST sidecar (the mlx-community conversions strip it).
/// Gated on the arch so a directory name can never flip an unrelated model:
/// qwen3* named *embedding* → last-token (Qwen3-Embedding's contract), BERT
/// bge-/mxbai-embed → CLS (their model cards' contract). Everything else null
/// → the mean default.
pub fn poolingFromDirName(dir_basename: []const u8, model_type: []const u8) ?PoolingMode {
    var lower_buf: [256]u8 = undefined;
    if (dir_basename.len > lower_buf.len) return null;
    const lower = std.ascii.lowerString(&lower_buf, dir_basename);
    if (std.mem.startsWith(u8, model_type, "qwen3")) {
        if (std.mem.indexOf(u8, lower, "embedding") != null) return .last_token;
        return null;
    }
    if (std.mem.eql(u8, model_type, "bert")) {
        if (std.mem.indexOf(u8, lower, "bge-") != null) return .cls;
        if (std.mem.indexOf(u8, lower, "mxbai-embed") != null) return .cls;
        return null;
    }
    return null;
}

pub const isExpertStreamingArch = expert_quant.isExpertStreamingArch;

pub const ModelConfig = struct {
    // Architecture identity
    model_type: []const u8 = "gemma3",
    weight_prefix: []const u8 = "language_model.model",

    // Core dimensions
    vocab_size: u32 = 262208,
    hidden_size: u32 = 3840,
    intermediate_size: u32 = 15360,
    /// Whether `intermediate_size` came from the JSON or is the struct default
    /// above. MoE checkpoints routinely omit the key (DSV4 ships none at all),
    /// and a consumer that cannot tell the two apart bills the 15360 default as
    /// though the model declared it — 3.96 GB of phantom MLP transient in the
    /// prefill guard. Only readers that need the DISTINCTION should look here;
    /// everyone else keeps using `intermediate_size` and its fallback value.
    intermediate_size_declared: bool = false,
    num_hidden_layers: u32 = 48,
    num_attention_heads: u32 = 16,
    num_key_value_heads: u32 = 8,
    head_dim: u32 = 256,
    v_head_dim: u32 = 0, // 0 = the layer's query/key width
    global_v_head_dim: u32 = 0,
    attention_value_scale: f32 = 1.0, // Applied before the KV cache write.
    rms_norm_eps: f32 = 1e-6,

    // RoPE
    rope_theta: f32 = 1000000.0,
    rope_local_base_freq: f32 = 10000.0,
    rope_scaling_factor: f32 = 1.0,

    // Sliding window attention
    has_sliding_window: bool = true,
    sliding_window: u32 = 1024,
    sliding_window_pattern: u32 = 6,

    // Quantization. 0 = dense bf16 (config.json has no "quantization" key);
    // quantized checkpoints always set this from that key (see parseConfig).
    quant_bits: u32 = 0,
    quant_group_size: u32 = 64,
    quant_mode: QuantMode = .affine,
    expert_streaming: bool = false,
    /// A streamed load that serves `--mtp` keeps the head, and its own routed experts, resident.
    stream_mtp_head: bool = false,
    expert_layout: expert_quant.Layout = .bf16_fused,
    expert_quant_rate: expert_exl3.Rate = .{ .n = 64 },
    expert_quant_codebook: expert_exl3.Codebook = .mul1,
    expert_quant_window: expert_exl3.Window = .w16,
    expert_source_dir: ?[]u8 = null,
    /// `SUSHI_NGRAM_BF16_DIR`: serve the PLE n-gram table from the ORIGINAL bf16
    /// shards in this HF checkpoint dir instead of the pack's quantized `ngram_table.bin`
    /// (a two-arm lever: it isolates the table's quantization cost under `kld compare`).
    ngram_bf16_dir: ?[]u8 = null,
    expert_cache_bytes: u64 = 0,
    expert_ssd_budget_bytes: u64 = 0,
    expert_workspace_bytes: u64 = 0,
    expert_bounce_bytes: u64 = 0,
    expert_fill_peak_bytes: u64 = 0,
    /// A streamed load's resident vision tower (in the ssd budget) and its largest single-image
    /// encode (beside the budget, at the wired-limit admission); 0 = tower off.
    expert_vision_tower_bytes: u64 = 0,
    expert_vision_encode_bytes: u64 = 0,

    // Attention scale: 1/sqrt(query_pre_attn_scalar) for Gemma, 1/sqrt(head_dim) for others
    query_pre_attn_scalar: u32 = 256,

    // Architectural differences between model families
    tie_word_embeddings: bool = false,
    hidden_act: HiddenAct = .gelu_approx,
    norm_has_offset: bool = true,
    scale_embeddings: bool = true,
    has_pre_ff_norm: bool = true,
    has_qk_norm: bool = true,
    // Learned per-head attention sinks (gpt_oss `self_attn.sinks`, [n_heads]):
    // one extra logit column that lands in the softmax DENOMINATOR only, so
    // every head can attend to "nothing". mlx's fused SDPA takes them
    // natively (`mlx_fast_scaled_dot_product_attention`'s `sinks` argument);
    // this flag is what makes the loader fetch the weight and the forward
    // pass it instead of the null array.
    has_attn_sinks: bool = false,
    attn_sinks_global: bool = true,
    attn_sinks_sliding: bool = true,

    // MoE
    num_experts: u32 = 0,
    num_experts_per_tok: u32 = 0,
    moe_intermediate_size: u32 = 0,
    shared_expert_intermediate_size: u32 = 0,
    // DeepSeek-V3-style sigmoid routing (hy_v3): scores = sigmoid(logits) in
    // f32; top-k SELECTED on scores + expert_bias but WEIGHTED by the unbiased
    // scores; optional renorm (/(sum+1e-20)) then × router_scaling_factor.
    moe_sigmoid_router: bool = false,
    moe_route_norm: bool = true,
    router_scaling_factor: f32 = 1.0,
    // Layers [0, first_k_dense_replace) use a dense MLP instead of MoE
    // (hy_v3: layer 0 dense at intermediate_size, the rest MoE).
    first_k_dense_replace: u32 = 0,

    // Grouped ("noaux_tc") expert routing: split the biased scores into
    // moe_n_group equal groups, keep the moe_topk_group best by their top-2
    // sum, then take the global top-k inside the survivors. 1/1 = ungrouped.
    moe_n_group: u32 = 1,
    moe_topk_group: u32 = 1,

    // Linear attention (GatedDeltaNet)
    linear_num_key_heads: u32 = 0,
    linear_num_value_heads: u32 = 0,
    linear_key_head_dim: u32 = 128,
    linear_value_head_dim: u32 = 128,
    linear_conv_kernel_dim: u32 = 4,
    /// Store the recurrent state in f32 between tokens (`SUSHI_GDN_STATE_F32=1`); bf16 by default.
    gdn_state_f32: bool = false,

    // KDA (Kimi Delta Attention, bailing_hybrid) variations on the
    // GatedDeltaNet recurrence:
    //   - the forget gate is PER CHANNEL ([B,T,H,Dk]) rather than per head, so
    //     the fused kernel indexes `g` by the key channel (kda_vector_gate);
    //   - a non-zero lower bound replaces the softplus gate entirely with
    //     `g = bound * sigmoid(exp(A_log) * (a + dt_bias))`, which is bounded
    //     in (bound, 0) instead of (-inf, 0) — it is NOT a clamp on the
    //     softplus form (fla/ops/kda/fused_recurrent.py);
    //   - the output gate is a plain sigmoid, not SiLU/swish.
    kda_vector_gate: bool = false,
    kda_gate_lower_bound: f32 = 0.0, // 0 = plain -exp(A_log)·softplus form
    kda_sigmoid_out_gate: bool = false,
    glm_hc_sinkhorn_iters: u32 = 20,
    glm_hc_eps: f32 = 1e-6,
    glm_swiglu_limit: f32 = 10,
    glm_index_tail: bool = true,
    glm_mtp_layers: u32 = 0,
    glm_fp8_trunk: bool = false,

    // Multi-head Latent Attention (bailing_hybrid's full-attention layers,
    // DeepSeek-V3 shape): low-rank Q (q_a_proj → q_a_layernorm → q_b_proj) and
    // a single compressed KV latent (kv_a_proj_with_mqa → kv_lora_rank latent
    // + qk_rope_head_dim shared rope key) expanded per head by kv_b_proj.
    // Query/key head dim is nope+rope; the value head dim is SMALLER, so the
    // KV cache holds asymmetric K/V (MLX's SDPA has a 192/128 vector kernel).
    // mla_head_gate: per-head sigmoid gate on the attention output (the
    // checkpoint's `head_wise` gated_attention_proj_granularity_type).
    mla_q_lora_rank: u32 = 0, // 0 = not an MLA arch
    mla_kv_lora_rank: u32 = 0,
    mla_qk_nope_head_dim: u32 = 0,
    mla_qk_rope_head_dim: u32 = 0,
    mla_v_head_dim: u32 = 0,

    // Hybrid attention
    full_attention_interval: u32 = 0,
    // Hybrid archs: layers at or past this index are ALWAYS full
    // attention, whatever the interval says. The reference's rule is
    // `(idx+1) % group == 0 OR idx >= n_layers // group * group`, i.e. the
    // ragged tail after the last WHOLE group never gets linear attention.
    // 0 = no tail bound, which is every other hybrid arch (qwen3_next, lfm2).
    linear_attn_tail_from: u32 = 0,
    partial_rotary_factor: f32 = 1.0,
    attn_output_gate: bool = false,

    // Qwen4-Exp (Qwen3.8-Flash-Next): gated residual streams ("hyper
    // connections", hc_count x hidden wide), a hashed n-gram embedding
    // injected at ONE layer (PLE), and Qwen Sparse Attention (indexer-selected
    // 4-token blocks past `indexer_budget` tokens). hc_count 0 = none.
    hc_count: u32 = 0,
    hc_lowrank: u32 = 0,
    ple_layer_idx: i32 = -1, // 0-based; the config lists 1-based ids
    ple_embed_dim: u32 = 0,
    ple_conv_kernel: u32 = 4,
    ngram_size: u32 = 3,
    heads_per_ngram: u32 = 8,
    ngram_vocab_base: u64 = 20_000_000,
    ngram_vocab_divisor: u32 = 128,
    ngram_seed: u64 = 1234,
    indexer_n_heads: u32 = 0, // 0 = dense attention
    indexer_head_dim: u32 = 0,
    indexer_budget: u32 = 0,
    indexer_compress_ratio: u32 = 0,
    /// The TEXT config's own eos (its first entry): the n-gram hash's segment
    /// reset token, independent of the generation-time stop set.
    ngram_eos: u32 = 0,
    /// `<model_dir>/ngram_table.bin` for the PLE table (mmapped by the
    /// engine, never mlx-loaded). Set by `parseConfig`; lives as long as the
    /// config does.
    ngram_table_path: ?[]const u8 = null,

    // Laguna: per-layer Q-head count (full-attention layers 48, sliding 72;
    // KV heads uniform at num_key_value_heads). 0 = uniform num_attention_heads.
    num_attention_heads_per_layer: [128]u32 = @splat(0),
    has_per_layer_heads: bool = false,


    // Laguna YaRN RoPE (full-attention layers only; sliding layers use default
    // RoPE at rope_local_base_freq). rope_yarn gates the freqs + mscale
    // precompute at model load; the sliding/full split is by isGlobalLayer.
    rope_yarn: bool = false,
    yarn_factor: f32 = 1.0,
    yarn_orig_max_pos: u32 = 0,
    yarn_beta_fast: f32 = 32.0,
    yarn_beta_slow: f32 = 1.0,
    yarn_attention_factor: f32 = 1.0,
    /// HF's `truncate` (default true): floor/ceil the ramp correction bounds to
    /// whole dims. Only the flat-`rope_parameters` readers set it.
    yarn_truncate: bool = true,



    // BERT encoder-only
    is_encoder_only: bool = false,

    /// Sentence-transformers pooling for /v1/embeddings (issue #116). null =
    /// no explicit signal → masked mean (the historical behavior, correct for
    /// MiniLM-class BERTs and EmbeddingGemma). Set from config.json
    /// `pooling_mode`, the ST `1_Pooling/config.json` sidecar, or the
    /// known-family name fallback (`poolingFromDirName`). A non-null mode on a
    /// decoder arch (Qwen3-Embedding) also advertises the `embeddings`
    /// capability WITHOUT flipping `is_encoder_only` — the forward stays the
    /// arch's own causal pass.
    pooling_mode: ?PoolingMode = null,

    // BOS id from config.json (embedding models wrap inputs <bos>…<eos>).
    bos_token_id: ?u32 = null,

    // Context length from config.json (0 = unknown)
    max_position_embeddings: u32 = 0,

    /// Auto-context, FROZEN at model-load time (`server.pinAutoContext`).
    /// 0 = not pinned yet.
    ///
    /// Without `--ctx-size` the effective context used to be recomputed from
    /// LIVE memory on every request, so the number the server advertised drifted
    /// as other processes took RAM (measured: 92,387–94,883 across one session).
    /// Agent CLIs budget their own `max_tokens` against that advertised value,
    /// so it has to hold still for the model's whole residency. Explicit
    /// `--ctx-size` still wins over this.
    pinned_context: u32 = 0,

    /// Per-model settings from `model-settings.json`, set at the load
    /// construction site; an explicit launch flag outranks each (`model_settings.pick`).
    /// 0/null = unset; `mtp_override` true = head loaded AND on by default (`--mtp`, per model).
    ctx_override: u32 = 0,
    kv_quant_override: ?kv_quant_mod.KVQuantConfig = null,
    mtp_override: ?bool = null,
    mtp_acceptance_override: ?mtp_acceptance_mod.Mode = null,
    mtp_greedy_tail_override: ?bool = null,
    /// Per-model `ssd_budget_gb` (GiB, the `--ssd-budget-gb` unit) from model-settings.json; 0 = none.
    ssd_budget_gb_override: u32 = 0,
    preserve_thinking_override: ?bool = null,
    think_penalty_override: ?f32 = null,
    logit_bias_file_override: ?@import("logit_bias.zig").FilePath = null,
    vision_override: ?bool = null,

    /// The prefill chunk this model was sized for, FROZEN at load
    /// (`server.pinPrefillChunk`). 0 = not pinned yet, which keeps the
    /// launch/base chunk.
    ///
    /// The chunk is the multiplier on the biggest transient in the memory bill
    /// (`8 x chunk x max(hidden, ffn) x 2`, three of them). Nothing used to size
    /// it to the MACHINE, so a 16 GB Mac reserved the same 5-7 GB envelope a
    /// 128 GB one does, which is most of its budget: the sizer then reported a
    /// 1024-token context and the admission guard refused prompts whose real
    /// peak was a third of the bill. The sizer, `checkAttentionMemory` and
    /// `generate.effectivePrefillChunk` all read THIS field, so the bill and the
    /// forward can never disagree. Explicit `--prefill-chunk` still wins.
    pinned_prefill_chunk: u32 = 0,

    // Stop tokens (populated from config.json)
    eos_token_ids: [8]u32 = @splat(0),
    num_eos_tokens: u32 = 0,

    // Model-author sampling recommendations from generation_config.json
    // (e.g. Qwen 3.6: temp 1.0 / top_p 0.95 / top_k 20; Gemma 4: top_k 64).
    // null = the file or key is absent. Used as defaults for request fields
    // the client OMITTED — Claude Code sends no sampling params at all, and
    // pre-2026-06 it sampled the full untruncated distribution at temp 1.0,
    // well outside the model card's intended envelope.
    gen_temperature: ?f32 = null,
    gen_top_p: ?f32 = null,
    gen_top_k: ?u32 = null,

    // The checkpoint's OWN thinking default, from generation_config.json's
    // `default_chat_template_kwargs.enable_thinking`. null = the file or key
    // is absent. Read by `defaultEnableThinking` for requests that name no
    // thinking preference; an explicit request value still outranks it.
    gen_enable_thinking: ?bool = null,

    /// Native GLM DFlash2 request terms, stamped from the selected assistant's
    /// config before load. Its stored weights are in the resident load bill.
    glm_dflash_loaded: bool = false,
    glm_dflash_window_bytes: u64 = 0,
    glm_dflash_capture_bytes_per_token: u64 = 0,

    /// Set at load when this model's SSD prefix tier was wanted and did not come up (no room on the
    /// volume, an unreadable fingerprint, a failed init): every bill and budget then runs as without a disk.
    prefix_cache_disk_declined: bool = false,

    // Gemma 4: explicit layer type map (bit = 1 means full/global attention)
    has_explicit_layer_types: bool = false,
    layer_is_global: [128]bool = @splat(false),

    // Vision encoder (Gemma 4 SigLIP)
    has_vision: bool = false,
    image_token_id: u32 = 0, // 0 = no image token


    // Qwen3.5/3.6 vision (Qwen3-VL ViT). Distinct from the Gemma SigLIP fields
    // above: Qwen ships a fused-qkv ViT with a patch merger, and the text trunk
    // uses INTERLEAVED M-RoPE (image tokens get 2D grid positions). Populated in
    // the qwen3_5 arm below. Encoder: src/qwen_vision.zig; M-RoPE: src/mrope.zig.
    qwen_vision: bool = false,
    qv_depth: u32 = 0, // ViT transformer blocks
    qv_hidden: u32 = 0, // ViT hidden size
    qv_heads: u32 = 0, // ViT attention heads
    qv_head_dim: u32 = 0, // = qv_hidden / qv_heads
    qv_intermediate: u32 = 0, // ViT MLP intermediate
    qv_patch: u32 = 16, // pixel patch size
    qv_temporal_patch: u32 = 2, // frames folded per patch (still image duplicated)
    qv_merge: u32 = 2, // spatial merge: merge×merge patches → one LLM token
    qv_num_pos_emb: u32 = 0, // learned pos table entries (e.g. 2304 = 48×48)
    qv_out_hidden: u32 = 0, // merger output dim (= text hidden_size)
    // Image-area bounds from processor_config.json / preprocessor_config.json.
    // 0 means absent: the Qwen processor defaults remain the fallback.
    qv_min_pixels: u32 = 0,
    qv_max_pixels: u32 = 0,
    // GLM-5.3 shares patch-grid geometry, with a separate RMSNorm ViT and padded CLIP processor.
    glm5_vision: bool = false,
    glmv_projection_intermediate: u32 = 10240,
    glmv_eps: f32 = 1e-5,
    glmv_swiglu_limit: f32 = 10,
    glmv_rope_theta: f32 = 10000,
    glmv_min_image_tokens: u32 = 16,
    glmv_max_image_tokens: u32 = 8000,
    glmv_max_video_tokens: u32 = 240000,
    // MiMo-ViT (src/mimo_vision.zig). Shares the qv_* geometry; attention is
    // GQA, and every block is full, or a ±window band over row-major or
    // column-major merge-unit order, with a per-head sink on the band blocks.
    mimo_vision: bool = false,
    mvit_kv_heads: u32 = 0,
    mvit_window: u32 = 0,
    mvit_sinks: bool = false,
    mvit_attn: [MAX_VISION_LAYERS]MimoVitAttn = @splat(.full),
    // Interleaved M-RoPE sections [t, h, w]; sum = rotary_dim/2 (e.g. [11,11,10]).
    mrope_section: [3]u32 = .{ 0, 0, 0 },
    mrope_interleaved: bool = false,
    // Qwen vision token ids (top-level config.json). image_token_id reuses the
    // shared field above (parsed generically at the image_token_id block).
    video_token_id: u32 = 0,
    vision_start_token_id: u32 = 0,
    vision_end_token_id: u32 = 0,

    // Gemma 4: dual head dimensions and KV sharing
    global_head_dim: u32 = 0, // 0 = same as head_dim
    num_global_key_value_heads: u32 = 0, // 0 = same as num_key_value_heads


    // Hybrid layers (LFM2, Nemotron-H): per-layer type dispatch
    has_hybrid_layers: bool = false,
    attn_fused_qkv: bool = false, // checkpoint ships self_attn.q_k_v_proj = [q | k | v] rows
    has_final_norm: bool = true, // false when a hyper-connection mixer replaces model.norm
    /// How many keys ONE query reads during prefill at this prompt length: every
    /// served arch attends the whole prompt (dense causal).
    pub fn prefillAttnKeys(self: *const ModelConfig, seq: u64) u64 {
        _ = self;
        return seq;
    }

    pub fn isGlobalLayer(self: ModelConfig, layer_idx: u32) bool {
        if (!self.has_sliding_window) return true;
        if (self.has_explicit_layer_types and layer_idx < 128) {
            return self.layer_is_global[layer_idx];
        }
        // HF/mlx-lm convention (Gemma 3): the GLOBAL layer closes each group —
        // global when `(idx + 1) % pattern == 0` (layers 5, 11, … for pattern 6).
        return (layer_idx % self.sliding_window_pattern) == self.sliding_window_pattern - 1;
    }

    /// Get effective head_dim for a layer (global layers may use global_head_dim).
    pub fn layerHeadDim(self: ModelConfig, layer_idx: u32) u32 {
        if (self.global_head_dim > 0 and self.isGlobalLayer(layer_idx)) {
            return self.global_head_dim;
        }
        return self.head_dim;
    }

    pub fn layerVHeadDim(self: ModelConfig, layer_idx: u32) u32 {
        if (self.global_v_head_dim > 0 and self.isGlobalLayer(layer_idx)) return self.global_v_head_dim;
        return if (self.v_head_dim > 0) self.v_head_dim else self.layerHeadDim(layer_idx);
    }

    pub fn layerHasAttnSinks(self: ModelConfig, layer_idx: u32) bool {
        return self.has_attn_sinks and
            (if (self.isGlobalLayer(layer_idx)) self.attn_sinks_global else self.attn_sinks_sliding);
    }

    /// Per-layer Q-head count (Laguna: 48 on full-attention layers, 72 on
    /// sliding). Every other arch has uniform heads, so this falls back to
    /// num_attention_heads. KV heads stay uniform (layerKVHeads).
    pub fn layerNumHeads(self: ModelConfig, layer_idx: u32) u32 {
        if (self.has_per_layer_heads and layer_idx < 128 and self.num_attention_heads_per_layer[layer_idx] > 0) {
            return self.num_attention_heads_per_layer[layer_idx];
        }
        return self.num_attention_heads;
    }

    /// Get effective num_kv_heads for a layer.
    pub fn layerKVHeads(self: ModelConfig, layer_idx: u32) u32 {
        if (self.num_global_key_value_heads > 0 and self.isGlobalLayer(layer_idx)) {
            return self.num_global_key_value_heads;
        }
        return self.num_key_value_heads;
    }

    pub fn isLinearLayer(self: ModelConfig, layer_idx: u32) bool {
        if (self.full_attention_interval == 0) return false;
        if (self.linear_attn_tail_from != 0 and layer_idx >= self.linear_attn_tail_from) return false;
        return ((layer_idx + 1) % self.full_attention_interval) != 0;
    }

    /// The `partial_rotary_factor` the YaRN table covers (qwen4_exp has a single
    /// rope for the whole trunk).
    pub fn yarnPartial(self: *const ModelConfig) f32 {
        return self.partial_rotary_factor;
    }

    /// `int(head_dim × yarnPartial())` — the rotating slice of a head, i.e. the
    /// dims the YaRN frequency table covers (qwen4_exp: 256 × 0.25 = 64, whose
    /// 32 frequencies are what `mrope_section` [11,11,10] sums to).
    pub fn yarnRotaryDim(self: *const ModelConfig) u32 {
        return @intFromFloat(@as(f32, @floatFromInt(self.head_dim)) * self.yarnPartial());
    }

    /// The longest sequence the rope can actually resolve. Plain:
    /// `max_position_embeddings`. YaRN: `original_max_position_embeddings ×
    /// factor` — the window HF and vLLM both derive `max_model_len` from — since
    /// a position past it aliases back inside the ramp. 0 = no rope-derived cap.
    pub fn contextCap(self: *const ModelConfig) u32 {
        const declared = self.max_position_embeddings;
        if (!self.rope_yarn) return declared;
        const orig: f64 = @floatFromInt(self.yarn_orig_max_pos);
        const factor: f64 = @floatCast(self.yarn_factor);
        const scaled: f64 = @floor(orig * factor);
        const max_u32: f64 = @floatFromInt(std.math.maxInt(u32));
        const window: u32 = if (scaled >= max_u32) std.math.maxInt(u32) else @intFromFloat(scaled);
        return if (declared == 0) window else @min(window, declared);
    }

    /// How many layers hold an attention KV cache. A hybrid arch interleaves
    /// linear-attention layers, which carry a FIXED-SIZE recurrent state
    /// instead of a per-token cache — billing them as attention layers made
    /// the memory model charge a uniform arch's footprint for a model
    /// carrying a fraction of it (bailing_hybrid: 6 of 24).
    pub fn attnCacheLayerCount(self: *const ModelConfig) u32 {
        if (self.full_attention_interval == 0) return self.num_hidden_layers;
        var n: u32 = 0;
        var i: u32 = 0;
        while (i < self.num_hidden_layers) : (i += 1) {
            if (!self.isLinearLayer(i)) n += 1;
        }
        return n;
    }

    /// Dense (bf16) KV-cache bytes ONE token occupies across the whole model.
    /// The uniform `layers × 2 × kv_heads × head_dim` formula is wrong on a
    /// hybrid MLA arch in both terms: only `attnCacheLayerCount` layers cache
    /// at all, and MLA's key (nope+rope) is WIDER than its value. Every
    /// memory estimate that sizes a KV cache reads this one helper so the
    /// auto-context sizer and the prefill admission guard cannot disagree.
    /// Whether the prefill chunk is resolved per request (by the admission bill) instead of
    /// once at load: a long session's load-time reserve (and, ungated, the hot-cache ask) pins
    /// every ordinary prompt to a narrow rung.
    pub fn perRequestPrefillChunk(self: *const ModelConfig) bool {
        return self.longCtxGated() or self.swaRingTokens() > 0 or self.isGlm5();
    }

    /// The width a per-request arch's prefill starts at; only a chunk that no longer fits beside its
    /// KV steps down. GLM's native prefill paths are built for 2048 rows.
    pub fn prefillStartWidth(self: *const ModelConfig) u32 {
        if (self.isGlm5()) return @import("glm5_forward.zig").prefill_chunk;
        if (self.longCtxGated()) return 4096;
        if (self.swaRingTokens() > 0) return 2048;
        return std.math.maxInt(u32);
    }

    /// Dense bf16 bytes ONE token of layer `li`'s K and V occupy. Only correct
    /// to bill per layer on an arch whose layers really differ (mimo_v2's
    /// global/sliding split); `kvBytesPerToken` keeps the uniform formula
    /// everywhere else so no arch's number moves without its bytes moving.
    pub fn layerKvBytes(self: *const ModelConfig, li: u32) u64 {
        if (self.isGlm5()) return if (self.isKvPerTokenLayer(li)) @as(u64, self.mla_kv_lora_rank) * 2 else 0;
        return @as(u64, self.layerKVHeads(li)) *
            (@as(u64, self.layerHeadDim(li)) + @as(u64, self.layerVHeadDim(li))) * 2;
    }

    /// Rows a ringed sliding layer holds past its window before it compacts.
    /// The compaction is a real copy of the retained window, so the slack is
    /// what amortizes it over decode steps.
    pub const SWA_RING_SLACK: u64 = 512;

    /// Tokens a sliding layer's KV buffer retains, 0 when every layer stores
    /// the full sequence. Non-zero only where dropping the rows below the
    /// window is provably invisible: the layer's whole attention is the window,
    /// every mask builder is handed the TRIMMED length (`slidingViewFor`), and
    /// no in-kernel band reads absolute positions (the fused hd-256 kernel
    /// does, so an arch at that width never rings).
    pub fn swaRingTokens(self: *const ModelConfig) u64 {
        if (!std.mem.eql(u8, self.model_type, "mimo_v2")) return 0;
        if (!self.has_sliding_window or self.sliding_window == 0) return 0;
        if (self.head_dim == 256) return 0;
        return @as(u64, self.sliding_window) + SWA_RING_SLACK;
    }

    /// Dense bytes of ringed sliding-layer storage ONE slot holds, whatever the
    /// context: the twin of `qsaRingBytes`, billed once per slot rather than
    /// per token. Kv-quantized like any other cache row, so callers scale it
    /// through `server.kvBytesPerTokenAtBits`.
    pub fn swaRingBytes(self: *const ModelConfig) u64 {
        const rows = self.swaRingTokens();
        if (rows == 0) return 0;
        return rows * self.slidingLayerKvBytesPerToken(self.num_hidden_layers);
    }

    /// Rows below the prompt end a ring checkpoint keeps past its window: the
    /// next turn's match lands a few tokens short of the prompt when the
    /// template re-renders the generation suffix (`generate.SSM_SNAPSHOT_BACKOFF`).
    pub const SWA_RING_CHECKPOINT_BACKOFF: u64 = 30;

    /// Rows per sliding layer of the prompt-end restore point
    /// (`KVCache.ringCheckpoint`), 0 on an arch that does not ring.
    pub fn swaRingCheckpointTokens(self: *const ModelConfig) u64 {
        if (self.swaRingTokens() == 0) return 0;
        return @as(u64, self.sliding_window) + SWA_RING_CHECKPOINT_BACKOFF;
    }

    /// Dense bytes of one ring checkpoint (`prefix_cache.SLOT_RING_CHECKPOINTS` per slot,
    /// `RING_CHECKPOINT_MAX` per hot entry).
    pub fn swaRingCheckpointBytes(self: *const ModelConfig) u64 {
        return self.swaRingCheckpointTokens() * self.slidingLayerKvBytesPerToken(self.num_hidden_layers);
    }

    /// Dense bytes one CHUNK token stages in the ringed layers: a prefill chunk
    /// is written whole before the ring compacts down to its window, so the
    /// rows exist for the width of the forward and nothing else bills them.
    /// `max_layers` is how many of them coexist — `ringCompact` runs after the
    /// layer's view is built and the pre-compaction buffer lives until that
    /// view evaluates, so the prefill loop's eval cadence is the bound, the
    /// same one the linear-attention stream term applies to itself.
    pub fn swaStreamBytesPerToken(self: *const ModelConfig, max_layers: u64) u64 {
        if (self.swaRingTokens() == 0) return 0;
        return self.slidingLayerKvBytesPerToken(max_layers);
    }

    /// Dense per-token KV of the sliding layers, at most `max_layers` of them.
    fn slidingLayerKvBytesPerToken(self: *const ModelConfig, max_layers: u64) u64 {
        var total: u64 = 0;
        var seen: u64 = 0;
        var li: u32 = 0;
        while (li < self.num_hidden_layers and seen < max_layers) : (li += 1) {
            if (self.isGlobalLayer(li)) continue;
            total += self.layerKvBytes(li);
            seen += 1;
        }
        return total;
    }

    pub fn kvBytesPerToken(self: *const ModelConfig) u64 {
        if (self.isGlm5()) return @as(u64, self.attnCacheLayerCount()) * self.mla_kv_lora_rank * 2;
        // A ringed arch pays per token only on its global layers; the sliding
        // half is `swaRingBytes`, a constant. Both halves land in the same
        // commit — billing the ring before the storage rings is an under-bill,
        // which ends in an uncatchable Metal OOM rather than a 400.
        if (self.swaRingTokens() > 0) {
            var total: u64 = 0;
            var li: u32 = 0;
            while (li < self.num_hidden_layers) : (li += 1) {
                if (self.isGlobalLayer(li)) total += self.layerKvBytes(li);
            }
            return total;
        }
        const widths: u64 = if (self.isMla())
            @as(u64, self.mlaQkHeadDim()) + @as(u64, self.mla_v_head_dim)
        else
            2 * @as(u64, self.head_dim);
        // MLA decompresses its latent to EVERY attention head before the write
        // (`mlaAttnWith` broadcasts the MQA rope key to `num_attention_heads`
        // and caches `[B, num_attention_heads, S, qk_dim]`), so its cache has
        // no grouping to save on — `num_key_value_heads` is the GQA question
        // and this arch never asks it. Equal on Ling 3.0 (16/16), so the
        // spelling is invisible today and would UNDER-bill the first MLA
        // checkpoint that groups — the direction that ends in an uncatchable
        // Metal OOM rather than a 400.
        const heads: u64 = if (self.isMla())
            @as(u64, self.num_attention_heads)
        else
            @as(u64, self.num_key_value_heads);
        return @as(u64, self.attnCacheLayerCount()) * heads * widths * 2;
    }

    /// Dense bf16 bytes of QSA indexer history ONE token occupies: the pooled
    /// blocks `[kv/ratio, idx_hd]` per full-attn layer. The raw keys are a fixed
    /// ring (`qsaRingBytes`, billed once per slot), not per token. Not
    /// kv-quantized. Zero on archs without an indexer. ONE copy; the billed width
    /// (copies + score bank) is `server.statePerTokenBilled`.
    pub fn qsaHistoryBytesPerToken(self: *const ModelConfig) u64 {
        if (self.indexer_budget == 0 or self.indexer_head_dim == 0) return 0;
        const n = @as(u64, self.attnCacheLayerCount());
        const hd = @as(u64, self.indexer_head_dim);
        const ratio = @max(@as(u64, self.indexer_compress_ratio), 1);
        return n * hd * 2 / ratio;
    }

    /// The raw indexer keys every live slot holds: `QSA_RING_ROWS` rows per
    /// full-attn layer, context-independent, billed once per slot.
    pub fn qsaRingBytes(self: *const ModelConfig) u64 {
        if (self.indexer_budget == 0 or self.indexer_head_dim == 0) return 0;
        const n = @as(u64, self.attnCacheLayerCount());
        const hd = @as(u64, self.indexer_head_dim);
        // Native GLM retains only the incomplete pool's key/gate pairs.
        if (self.isGlm5()) return n * (@as(u64, self.indexer_compress_ratio) -| 1) * hd * 2 * 2;
        const rows = @as(u64, @intCast(@import("transformer.zig").QSA_RING_ROWS));
        return n * rows * hd * 2 * @as(u64, if (self.isGlm5()) 2 else 1);
    }

    /// f32 bytes per token of the QSA block-score operand a live slot holds
    /// (`SSMCacheEntry.qsa_score_bank`). Never in an entry. Zero without an indexer.
    pub fn qsaScoreBankBytesPerToken(self: *const ModelConfig) u64 {
        if (self.isGlm5()) return 0;
        if (self.indexer_budget == 0 or self.indexer_head_dim == 0) return 0;
        if (@import("transformer.zig").qsaScoreFusedActiveFor(1, @intCast(self.indexer_n_heads), @intCast(self.indexer_head_dim))) return 0;
        const n = @as(u64, self.attnCacheLayerCount());
        const hd = @as(u64, self.indexer_head_dim);
        const ratio = @max(@as(u64, self.indexer_compress_ratio), 1);
        return n * hd * 4 / ratio;
    }

    /// Bytes one SSM checkpoint holds: recurrent state + conv window of every linear layer.
    /// The QSA key history is not here (it lands on the newest checkpoint only).
    pub fn ssmCheckpointBytes(self: *const ModelConfig) u64 {
        if (self.linear_num_value_heads == 0) return 0;
        const linear_layers: u64 = @as(u64, self.num_hidden_layers) -| self.attnCacheLayerCount();
        if (linear_layers == 0) return 0;
        const state: u64 = @as(u64, self.linear_num_value_heads) *
            @as(u64, self.linear_value_head_dim) * @as(u64, self.linear_key_head_dim) * @as(u64, if (self.isGlm5() or self.gdn_state_f32) 4 else 2);
        const conv_dim: u64 = 2 * @as(u64, self.linear_num_key_heads) * self.linear_key_head_dim +
            @as(u64, self.linear_num_value_heads) * self.linear_value_head_dim;
        const conv: u64 = @as(u64, self.linear_conv_kernel_dim) -| 1;
        return linear_layers * (state + conv * conv_dim * 2);
    }

    /// The GatedDeltaNet state's storage dtype; the recurrence computes in f32 either way.
    pub fn gdnStateDtype(self: *const ModelConfig) mlx.mlx_dtype {
        return if (self.gdn_state_f32) .float32 else .bfloat16;
    }

    pub fn isMoe(self: *const ModelConfig) bool {
        return self.num_experts > 0;
    }

    pub fn expertLayerCount(self: *const ModelConfig) u32 {
        return self.num_hidden_layers -| self.first_k_dense_replace;
    }

    /// True when the full-attention layers are Multi-head Latent Attention
    /// (compressed KV latent + low-rank Q), not plain GQA projections.
    pub fn isMla(self: *const ModelConfig) bool {
        return self.mla_kv_lora_rank > 0;
    }

    /// MLA query/key head dim = the non-positional part plus the rope part.
    /// This — not head_dim — is what the attention scale and the cached K's
    /// last dim are measured in.
    pub fn mlaQkHeadDim(self: *const ModelConfig) u32 {
        return self.mla_qk_nope_head_dim + self.mla_qk_rope_head_dim;
    }

    /// Which of fla's two KDA gate arms this checkpoint declares. A non-zero
    /// `kda_lower_bound` REPLACES the softplus form with the bounded sigmoid;
    /// absent (0) means the softplus form, which the shared GatedDeltaNet chain
    /// already computes elementwise and therefore serves a per-channel gate
    /// unchanged. Feeding bound 0 to the bounded chain yields exp(0) = 1 — a
    /// gate that never forgets — so the arm must be chosen, never defaulted.
    pub fn kdaUsesBoundedGate(self: *const ModelConfig) bool {
        return self.kda_vector_gate and self.kda_gate_lower_bound != 0.0;
    }

    /// The pooling op /v1/embeddings runs: the explicit signal, else masked
    /// mean (the historical default — correct for MiniLM and EmbeddingGemma).
    pub fn effectivePooling(self: *const ModelConfig) PoolingMode {
        return self.pooling_mode orelse .mean;
    }

    /// Whether this model serves /v1/embeddings meaningfully: encoder-only
    /// (BERT, EmbeddingGemma) or a decoder with a declared pooling contract
    /// (Qwen3-Embedding). Drives capability advertising, never dispatch.
    pub fn hasEmbeddingCapability(self: *const ModelConfig) bool {
        return self.is_encoder_only or self.pooling_mode != null;
    }

    /// Qwen3.8-Flash-Next (`qwen4_exp`): the qwen3_5 GDN + MoE trunk wrapped
    /// in hyper-connection residual streams, with the n-gram PLE and QSA.
    pub fn isQwen4(self: *const ModelConfig) bool {
        return std.mem.eql(u8, self.model_type, "qwen4_exp");
    }

    pub fn isGlm5(self: *const ModelConfig) bool {
        return std.mem.eql(u8, self.model_type, "glm5_next");
    }

    /// Streaming is a CAPABILITY of the checkpoint's routed-expert banks, not of
    /// its precision: both the dense HF layout and an MLX pack's per-projection
    /// banks stream. The disk-side half of the answer is the discovery layout probe.
    pub fn supportsExpertStreaming(self: *const ModelConfig) bool {
        return isExpertStreamingArch(self.model_type) and self.expertLayerCount() > 0 and self.num_experts > 0 and
            self.num_experts_per_tok > 0 and self.hidden_size > 0 and self.moe_intermediate_size > 0;
    }

    /// The routed-expert geometry every streamed store, cache and bill reads.
    pub fn expertGeometry(self: *const ModelConfig) expert_quant.Geometry {
        return .{
            .layers = @intCast(self.num_hidden_layers),
            .experts = @intCast(self.num_experts),
            .hidden = self.hidden_size,
            .intermediate = self.moe_intermediate_size,
            .first_moe_layer = @intCast(self.first_k_dense_replace),
            .exl3_n = self.expert_quant_rate.n,
        };
    }

    /// Dense banks and raw individual experts require the streaming loader.
    /// An EXL3 bank is a self-describing quantized weight the resident kernels
    /// read as they are, whatever the trunk's own width says.
    pub fn expertStreamingRequired(self: *const ModelConfig) bool {
        if (self.expert_layout == .exl3_k4) return false;
        return self.supportsExpertStreaming() and
            (self.quant_bits == 0 or self.expert_layout == .mxfp4_individual);
    }

    pub fn isMimo(self: *const ModelConfig) bool {
        return std.mem.eql(u8, self.model_type, "mimo_v2");
    }

    /// A MiMo checkpoint keeps its trunk in the source FP8 layout beside source
    /// MXFP4 or EXL3 experts, so both take the source loader.
    pub fn usesMimoSourceTrunk(self: *const ModelConfig) bool {
        return std.mem.eql(u8, self.model_type, "mimo_v2") and
            (self.expert_layout == .mxfp4_individual or self.expert_layout == .exl3_k4);
    }

    /// The long-context blast-radius predicate: every long-context mechanism (KV
    /// reservation, pad-waste cap, checkpoint thinning, admission terms, chunk bar) was
    /// measured on qwen4_exp only, so they are opt-in by arch. Never hand-roll it at a site.
    pub fn longCtxGated(self: *const ModelConfig) bool {
        return self.isQwen4();
    }

    /// Does admission credit the hot cache and evict it to admit a prefill? One predicate for the
    /// connection thread's credits and the inference thread's eviction pass. A ringed arch joins:
    /// its warm credit is the global layers' rows alone, the ring is billed whole. GLM joins: its
    /// context is sized with no cache reserve.
    pub fn admissionEvictsHotCache(self: *const ModelConfig) bool {
        return self.longCtxGated() or self.swaRingTokens() > 0 or self.isGlm5();
    }

    /// Does a request reserve its whole cache capacity up front instead of
    /// growing +25% at a time? Narrower than `longCtxGated`: a ringed arch
    /// joins because its per-token KV is nine layers of a 48-layer trunk, so a
    /// mid-prefill grow duplicates gigabytes the bill never modelled — but
    /// none of the other long-context mechanisms come with it.
    pub fn reservesKvCapacity(self: *const ModelConfig) bool {
        return self.longCtxGated() or self.swaRingTokens() > 0;
    }

    /// Layers `kvBytesPerToken` is the sum over. Every caching layer normally;
    /// on a ringed arch only the GLOBAL ones, because the sliding half stores a
    /// window rather than the sequence. Dividing the per-token bill by every
    /// caching layer under-bills a ringed arch by 48/9.
    pub fn kvPerTokenLayerCount(self: *const ModelConfig) u32 {
        if (self.swaRingTokens() == 0) return self.attnCacheLayerCount();
        var n: u32 = 0;
        var li: u32 = 0;
        while (li < self.num_hidden_layers) : (li += 1) {
            if (self.isGlobalLayer(li)) n += 1;
        }
        return n;
    }

    /// Does layer `li` carry a share of `kvBytesPerToken`? The per-layer twin
    /// of `kvPerTokenLayerCount`, so a window count and a total cannot drift.
    pub fn isKvPerTokenLayer(self: *const ModelConfig, li: u32) bool {
        if (self.swaRingTokens() > 0) return self.isGlobalLayer(li);
        return !self.isLinearLayer(li);
    }

    pub fn batchedEffectiveKvLen(self: *const ModelConfig, kv: u32, gather_on: bool, gather_min_kv: u32) u32 {
        if (!self.isQwen4() or !gather_on) return kv;
        if (kv <= gather_min_kv) return kv;
        const cap = self.indexer_budget + self.indexer_compress_ratio;
        if (cap == 0) return kv;
        return @min(kv, cap);
    }

    /// SSD-first prefix cache arch predicate; delegates to `longCtxGated`.
    pub fn ssdFirstCapable(self: *const ModelConfig) bool {
        return self.longCtxGated();
    }

    /// True when per-request SSM/conv cache entries must exist: hybrid
    /// recurrence (LFM2/Nemotron/GDN) or Inkling's four per-layer short
    /// convolutions. Shared by Transformer.init and the scheduler's per-slot
    /// allocation — the two predicates MUST agree or slots crash on a null
    /// `ctx.ssm_entries` (the Qwen3.5-MoE class).
    pub fn needsSsmEntries(self: *const ModelConfig) bool {
        if (self.isGlm5()) return false; // state belongs to glm5_forward.Request
        return self.has_hybrid_layers or self.full_attention_interval > 0;
    }

    /// Block-diffusion checkpoint (DiffusionGemma): generation is the canvas
    /// denoising loop, not autoregressive decode.
    pub fn isDiffusion(self: *const ModelConfig) bool {
        return self.canvas_length > 0;
    }

    /// Pure-config half of "can this arch ride the batched GatedDeltaNet
    /// decode kernel?" (`Transformer.forwardMoeBatchedDecode`) — a dense
    /// GDN trunk with periodic full attention, i.e. the qwen3_5 family.
    ///
    /// This exists because the answer is needed in TWO places that see
    /// different things: `server.zig` decides whether `--max-concurrent`
    /// clamps to 1 with only a ModelConfig in hand, while
    /// `Transformer.supportsBatchedGdnDecode` also checks the built layer
    /// set. Both MUST read this predicate — when they were hand-rolled
    /// separately, the server kept clamping qwen3_5 to serial decode while
    /// the scheduler was happily batching it, so `--max-concurrent 4` (the
    /// obvious serving config) silently DISABLED the batched path.
    ///
    /// Says nothing about MoE/hybrid archs that merely share the same
    /// forward — those stay serial, by name, in both callers.
    /// MiMo decodes concurrent slots as rows of one forward (`forwardMimoBatchedDecode`);
    /// a streamed load stays serial.
    pub fn supportsBatchedMimoDecode(self: *const ModelConfig) bool {
        return self.isMimo() and !self.expert_streaming;
    }

    /// GLM decodes concurrent slots as single-row groups of one verifier forward
    /// (`glm5_dflash_model.verifyGroups`); a streamed load stays serial.
    pub fn supportsBatchedGlmRows(self: *const ModelConfig) bool {
        return self.isGlm5() and !self.expert_streaming;
    }

    /// Qwen3.8-Flash-Next decodes concurrent slots as per-row decode ticks of one forward
    /// (`forwardQwen4DecodeRows`); a streamed load keeps the padded batch.
    pub fn supportsBatchedQwen4Rows(self: *const ModelConfig) bool {
        return self.isQwen4() and !self.expert_streaming;
    }

    pub fn supportsBatchedGdnDecode(self: *const ModelConfig) bool {
        return self.isQwen4() and self.full_attention_interval > 0;
    }

    /// True when the trunk uses the Gemma 4 layer structure (dual FFN with
    /// shared-expert branch, sigma-MoE router, 7 norms, layer_scalar, v_norm,
    /// proportional RoPE on full layers). DiffusionGemma reuses the Gemma 4
    /// 26B-A4B decoder verbatim, so transformer.zig's gemma4 forward/binding
    /// paths key on this rather than on the model_type string.
    pub fn isGemma4Layers(self: *const ModelConfig) bool {
        return std.mem.eql(u8, self.model_type, "gemma4") or
            std.mem.eql(u8, self.model_type, "diffusion_gemma");
    }

    /// Additive + dedup-guarded, like every terminator merge here.
    pub const NgramTableSource = enum { quantized, bf16_override, bf16_streamed };

    /// The override outranks the layout; only the HF bf16 checkpoint carries the table
    /// as bf16 shards — a streamed quantized pack still reads its own `ngram_table.bin`.
    pub fn ngramTableSource(self: *const ModelConfig) NgramTableSource {
        if (self.ngram_bf16_dir != null) return .bf16_override;
        if (self.expert_streaming and self.expert_layout == .bf16_fused) return .bf16_streamed;
        return .quantized;
    }

    pub fn mergeEosTokens(self: *ModelConfig, ids: []const u32) void {
        for (ids) |id| if (!self.isEosToken(id)) self.addEosToken(id);
    }

    pub fn addEosToken(self: *ModelConfig, id: u32) void {
        if (self.num_eos_tokens < self.eos_token_ids.len) {
            self.eos_token_ids[self.num_eos_tokens] = id;
            self.num_eos_tokens += 1;
        }
    }

    /// Gemma's chat template always ends turns with `<end_of_turn>` (id 106)
    /// and emits `<eos>` (id 1) at sequence end, so both must be stop tokens
    /// for EVERY Gemma family (gemma3 / gemma4 / diffusion_gemma). Some
    /// checkpoints declare only a SCALAR `eos_token_id: 1` (e.g. the
    /// abliterated text-only `-lm-` builds) — gating the 106 add on
    /// `num_eos_tokens == 0` then leaves it out and it leaks into output as
    /// repeated `<end_of_turn>`. Merge both ADDITIVELY + dedup-guarded (never
    /// removes a config-declared stop). Same leak class as the Qwen2.5-Coder
    /// `<|im_end|>` merge performed at load time (main.zig / scheduler doLoad).
    pub fn ensureGemmaTerminators(self: *ModelConfig) void {
        if (!self.isEosToken(1)) self.addEosToken(1);
        if (!self.isEosToken(106)) self.addEosToken(106);
    }

    /// NoPE layers (muse_glimmer: layer_rope_theta[i] == 0). The released
    /// checkpoint's NoPE layers are exactly its full-attention layers, but the
    /// two facts stay independently parsed — layer_types drives masking,
    /// layer_rope_theta drives rotation.
    pub fn layerSkipsRope(self: *const ModelConfig, layer_idx: u32) bool {
        return layer_idx < 128 and self.layer_no_rope[layer_idx];
    }

    /// SDPA softmax scale: 1/sqrt(query_pre_attn_scalar).
    pub fn attnScale(self: *const ModelConfig) f32 {
        return 1.0 / @sqrt(@as(f32, @floatFromInt(self.query_pre_attn_scalar)));
    }

    /// The head width the PREFILL SCORE tensor is actually built at. Normally
    /// `head_dim`, but an arch can score at a different width than it stores
    /// values at (an MLA q.k can contract over nope+rope widths while
    /// `head_dim` stays the value width) — reading `head_dim` there puts such
    /// an arch under the `<= 128` "fused SDPA covers it" early-out, so the
    /// score budget that exists for exactly this materializing path never
    /// applies. A new arch scoring wider than it stores adds its arm here.
    pub fn prefillScoreHeadDim(self: *const ModelConfig) u32 {
        if (self.isMla()) return self.mlaQkHeadDim();
        return self.head_dim;
    }

    /// Whether a chat request that names NO thinking preference should render
    /// with thinking on. Our server always passes `enable_thinking` explicitly,
    /// so a template whose own default is 'on' is silently overridden to off
    /// for every client that omits the field — the vendor's default mode
    /// becomes unreachable without a vendor-specific flag. An EXPLICIT request
    /// value always outranks this (see `server.resolveEnableThinking`); it
    /// only fills a silent request.
    ///
    /// First the checkpoint's OWN declaration
    /// (`generation_config.json` -> `default_chat_template_kwargs.enable_thinking`),
    /// then the per-arch allowlist below, which stays opt-in and only where the
    /// vendor documents thinking-on AND the shipped template agrees — never
    /// inferred from "the template mentions enable_thinking".
    pub fn defaultEnableThinking(self: *const ModelConfig) bool {
        if (thinkFlagArm(self.model_type)) |arm| return arm.effort != .off;
        // The checkpoint's own declared default outranks the arch allowlist:
        // it is the model author speaking, not our guess about the family.
        if (self.gen_enable_thinking) |v| return v;
        if (self.isGlm5()) return true;
        // mimo_v2: the vendor template's own default is on (only an explicit
        // `enable_thinking is false` closes the think block).
        if (std.mem.eql(u8, self.model_type, "mimo_v2")) return true;

        return false;
    }

    /// Fill still-null sampling recommendations with the FAMILY's documented
    /// upstream defaults. Community re-quants/distills routinely ship no
    /// generation_config.json (live 2026-07-13: a Qwen3.6-35B distill served
    /// to pi resolved omitted fields to the hardcoded 1.0/1.0/off — full
    /// untruncated tail sampling on a 4-bit MoE — and a 16K-token agent turn
    /// degenerated into word salad). Same pattern as the gemma3 head-count
    /// gotcha: when resolution relies on per-arch defaults a minimal
    /// checkpoint may omit, fill them explicitly.
    ///
    /// Deliberately fills ONLY the truncation knobs (top_k/top_p — what keeps
    /// the tail out of the sample space), never temperature: a null temp stays
    /// the neutral 1.0, and explicit request/flag/file values always win
    /// (this runs AFTER generation_config.json parse, nulls only).
    pub fn applyFamilySamplingDefaults(self: *ModelConfig) void {
        const t = self.model_type;
        // Qwen 3.x family only — Qwen2.5's upstream defaults differ (top_p
        // 0.8); never guess numbers the family didn't document.
        const is_qwen = std.mem.eql(u8, t, "qwen3") or
            std.mem.eql(u8, t, "qwen3_moe") or
            std.mem.eql(u8, t, "qwen3_5_moe") or
            std.mem.eql(u8, t, "qwen4_exp") or
            std.mem.eql(u8, t, "qwen3_next");
        if (is_qwen) {
            if (self.gen_top_k == null) self.gen_top_k = 20;
            if (self.gen_top_p == null) self.gen_top_p = 0.95;
        }
    }

    pub fn isEosToken(self: *const ModelConfig, id: u32) bool {
        for (self.eos_token_ids[0..self.num_eos_tokens]) |eos| {
            if (id == eos) return true;
        }
        return false;
    }

    pub fn eosTokenSlice(self: *const ModelConfig) []const u32 {
        return self.eos_token_ids[0..self.num_eos_tokens];
    }

    /// Free the allocator-owned fields (`ngram_table_path`, allocPrint'd by
    /// `parseConfig`; `expert_source_dir` of a streamed checkpoint; `ngram_bf16_dir`
    /// from the env override); everything else is plain data or a borrowed slice.
    /// Every `destroy` of a parsed config pairs with this, or a qwen4 load leaks the
    /// path. Idempotent.
    pub fn deinit(self: *ModelConfig, allocator: std.mem.Allocator) void {
        if (self.ngram_table_path) |p| allocator.free(p);
        self.ngram_table_path = null;
        if (self.expert_source_dir) |p| allocator.free(p);
        self.expert_source_dir = null;
        if (self.ngram_bf16_dir) |p| allocator.free(p);
        self.ngram_bf16_dir = null;
    }
};

pub fn parseConfig(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !ModelConfig {
    const path = try std.fmt.allocPrint(allocator, "{s}/config.json", .{model_dir});
    defer allocator.free(path);

    const file = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer file.close(io);

    var read_buf: [4096]u8 = undefined;
    var reader_state = file.reader(io, &read_buf);
    const content = try reader_state.interface.allocRemaining(allocator, .limited(10 * 1024 * 1024));
    defer allocator.free(content);

    var config = try parseConfigFromJson(allocator, content);
    errdefer config.deinit(allocator);
    config.gdn_state_f32 = @import("transformer.zig").diagEnvOn("SUSHI_GDN_STATE_F32");
    if (config.isQwen4()) {
        config.ngram_table_path = try std.fmt.allocPrint(allocator, "{s}/ngram_table.bin", .{model_dir});
        if (std.c.getenv("SUSHI_NGRAM_BF16_DIR")) |raw| {
            const dir = std.mem.span(raw);
            if (dir.len > 0) config.ngram_bf16_dir = try allocator.dupe(u8, dir);
        }
    }
    if (config.supportsExpertStreaming()) {
        const layers: u16 = std.math.cast(u16, config.num_hidden_layers) orelse return error.InvalidQwen4ConfigField;
        const first_moe: u16 = @intCast(config.first_k_dense_replace);
        if (expert_quant.layoutOfDirWithFirstMoe(allocator, io, config.model_type, model_dir, layers, first_moe)) |layout| {
            config.expert_layout = layout;
            // The MiMo binder reads the source QKV itself; the generic
            // fused-QKV names never apply to it.
            if (config.usesMimoSourceTrunk()) config.attn_fused_qkv = false;
            if (layout == .mxfp4_individual) {
                config.quant_mode = .mxfp4;
                config.quant_bits = 4;
                config.quant_group_size = 32;
            }
            if (layout == .exl3_k4) {
                const parsed = std.json.parseFromSlice(std.json.Value, allocator, content, .{}) catch return error.ExpertLayoutUnsupported;
                defer parsed.deinit();
                if (parsed.value != .object) return error.ExpertLayoutUnsupported;
                const spec = try sushi_exl3.parseExpertQuant(parsed.value.object);
                try sushi_exl3.admitTopK(config.num_experts_per_tok);
                config.expert_quant_rate = spec.rate;
                config.expert_quant_codebook = spec.codebook;
                config.expert_quant_window = spec.window;
                var k_buf: [8]u8 = undefined;
                log.info("[expert-exl3] engaged K={s} codebook={s} window={d}\n", .{ spec.rate.kText(&k_buf), @tagName(spec.codebook), spec.window.bits() });
            }
        } else if (expert_quant.hasGroupedExl3Index(allocator, io, model_dir)) {
            return error.ExpertLayoutUnsupported;
        }
    }

    // Model-author sampling recommendations ride in a sibling file. Optional —
    // any failure (missing file, bad JSON) leaves the fields null.
    const gen_path = try std.fmt.allocPrint(allocator, "{s}/generation_config.json", .{model_dir});
    defer allocator.free(gen_path);
    if (std.Io.Dir.openFileAbsolute(io, gen_path, .{})) |gen_file| {
        defer gen_file.close(io);
        var gen_buf: [4096]u8 = undefined;
        var gen_reader = gen_file.reader(io, &gen_buf);
        if (gen_reader.interface.allocRemaining(allocator, .limited(1024 * 1024))) |gen_content| {
            defer allocator.free(gen_content);
            const gd = parseGenerationDefaultsFromJson(gen_content);
            config.gen_temperature = gd.temperature;
            config.gen_top_p = gd.top_p;
            config.gen_top_k = gd.top_k;
            config.gen_enable_thinking = gd.enable_thinking;
            config.mergeEosTokens(gd.eos_token_ids[0..gd.num_eos]);
        } else |_| {}
    } else |_| {}
    // Pooling (issue #116), priority: explicit config.json `pooling_mode`
    // (already parsed) > the ST `1_Pooling/config.json` sidecar > the
    // known-family name fallback. A sidecar declaring only unsupported modes
    // fails the load here — explicitly, never a silent mean-pool.
    if (config.pooling_mode == null) {
        const pool_path = try std.fmt.allocPrint(allocator, "{s}/1_Pooling/config.json", .{model_dir});
        defer allocator.free(pool_path);
        if (std.Io.Dir.openFileAbsolute(io, pool_path, .{})) |pool_file| {
            defer pool_file.close(io);
            var pool_buf: [4096]u8 = undefined;
            var pool_reader = pool_file.reader(io, &pool_buf);
            if (pool_reader.interface.allocRemaining(allocator, .limited(1024 * 1024))) |pool_content| {
                defer allocator.free(pool_content);
                config.pooling_mode = try parsePoolingSidecar(pool_content);
                if (config.pooling_mode) |m|
                    log.info("[embed] pooling from 1_Pooling/config.json: {s}\n", .{@tagName(m)});
            } else |_| {}
        } else |_| {}
    }
    if (config.pooling_mode == null) {
        if (poolingFromDirName(std.fs.path.basename(model_dir), config.model_type)) |m| {
            config.pooling_mode = m;
            log.info("[embed] pooling inferred from checkpoint name: {s}\n", .{@tagName(m)});
        }
    }

    // Community re-quants often ship NO generation_config.json; fill the
    // still-null truncation knobs with the family's documented defaults so
    // omitted-field resolution never bottoms out at untruncated sampling.
    config.applyFamilySamplingDefaults();

    // Qwen image sizing is processor metadata rather than an architecture
    // constant. Prefer processor_config.json and fill any missing field from
    // the older preprocessor_config.json layout.
    if (config.qwen_vision or config.glm5_vision) {
        var vision_defaults = VisionProcessorDefaults{};
        const processor_files = [_][]const u8{
            "processor_config.json",
            "preprocessor_config.json",
        };
        for (processor_files) |name| {
            const processor_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ model_dir, name });
            defer allocator.free(processor_path);
            if (std.Io.Dir.openFileAbsolute(io, processor_path, .{})) |processor_file| {
                defer processor_file.close(io);
                var processor_buf: [4096]u8 = undefined;
                var processor_reader = processor_file.reader(io, &processor_buf);
                if (processor_reader.interface.allocRemaining(allocator, .limited(1024 * 1024))) |processor_content| {
                    defer allocator.free(processor_content);
                    const parsed_defaults = parseVisionProcessorDefaultsFromJson(processor_content);
                    if (vision_defaults.min_pixels == null)
                        vision_defaults.min_pixels = parsed_defaults.min_pixels;
                    if (vision_defaults.max_pixels == null)
                        vision_defaults.max_pixels = parsed_defaults.max_pixels;
                    if (vision_defaults.min_image_tokens == null)
                        vision_defaults.min_image_tokens = parsed_defaults.min_image_tokens;
                    if (vision_defaults.max_video_tokens == null)
                        vision_defaults.max_video_tokens = parsed_defaults.max_video_tokens;
                    if (vision_defaults.max_image_tokens == null)
                        vision_defaults.max_image_tokens = parsed_defaults.max_image_tokens;
                } else |_| {}
            } else |_| {}
        }
        if (vision_defaults.min_pixels != null and
            vision_defaults.max_pixels != null and
            vision_defaults.min_pixels.? > vision_defaults.max_pixels.?)
        {
            vision_defaults = .{};
        }
        config.qv_min_pixels = vision_defaults.min_pixels orelse 0;
        config.qv_max_pixels = vision_defaults.max_pixels orelse 0;
        if (config.glm5_vision) {
            config.glmv_min_image_tokens = vision_defaults.min_image_tokens orelse 16;
            config.glmv_max_image_tokens = vision_defaults.max_image_tokens orelse 8000;
            config.glmv_max_video_tokens = vision_defaults.max_video_tokens orelse 240000;
            if (config.glmv_min_image_tokens > config.glmv_max_image_tokens) return error.InvalidGlmVisionConfig;
        }
    }

    return config;
}

/// Sampling recommendations parsed out of a model's generation_config.json.
pub const GenerationDefaults = struct {
    temperature: ?f32 = null,
    top_p: ?f32 = null,
    top_k: ?u32 = null,
    /// `default_chat_template_kwargs.enable_thinking` — the checkpoint's own
    /// thinking default. null when absent or not a bool.
    enable_thinking: ?bool = null,
    /// `eos_token_id` (scalar or list): HF stops generation on these, and a
    /// checkpoint may name the chat terminator ONLY here (K2-Horizon's
    /// `<|ifm|im_end|>` rides beside config.json's `<|ifm|endoftext|>`).
    eos_token_ids: [8]u32 = @splat(0),
    num_eos: usize = 0,
};

/// Image-area limits parsed from a Qwen processor configuration.
pub const VisionProcessorDefaults = struct {
    min_pixels: ?u32 = null,
    max_pixels: ?u32 = null,
    /// Muse: the cap is on MERGED tokens, not pixels.
    max_image_tokens: ?u32 = null,
    min_image_tokens: ?u32 = null,
    max_video_tokens: ?u32 = null,
};

fn positiveJsonU32(value: ?std.json.Value) ?u32 {
    const actual = value orelse return null;
    return switch (actual) {
        .integer => |i| if (i > 0 and i <= std.math.maxInt(u32)) @intCast(i) else null,
        else => null,
    };
}

/// Parse both processor layouts used by Qwen checkpoints:
/// `image_processor.{min_pixels,max_pixels}` and
/// `size.{shortest_edge,longest_edge}`.
pub fn parseVisionProcessorDefaultsFromJson(content: []const u8) VisionProcessorDefaults {
    var buf: [16 * 1024]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&buf);
    const parsed = std.json.parseFromSlice(std.json.Value, fba.allocator(), content, .{}) catch return .{};
    defer parsed.deinit();
    if (parsed.value != .object) return .{};

    const root = parsed.value.object;
    const processor = if (root.get("image_processor")) |value|
        if (value == .object) value.object else root
    else
        root;

    var defaults = VisionProcessorDefaults{
        .min_pixels = positiveJsonU32(processor.get("min_pixels")),
        .max_pixels = positiveJsonU32(processor.get("max_pixels")),
        .max_image_tokens = positiveJsonU32(processor.get("max_image_tokens")),
        .min_image_tokens = positiveJsonU32(processor.get("min_image_tokens")),
        .max_video_tokens = if (root.get("video_processor")) |v| if (v == .object) positiveJsonU32(v.object.get("max_image_tokens")) else null else null,
    };
    if (processor.get("size")) |value| {
        if (value == .object) {
            if (defaults.min_pixels == null)
                defaults.min_pixels = positiveJsonU32(value.object.get("shortest_edge"));
            if (defaults.max_pixels == null)
                defaults.max_pixels = positiveJsonU32(value.object.get("longest_edge"));
        }
    }
    if (defaults.min_pixels != null and
        defaults.max_pixels != null and
        defaults.min_pixels.? > defaults.max_pixels.?)
    {
        return .{};
    }
    return defaults;
}

/// Pure parser for generation_config.json content. Total: malformed JSON or
/// out-of-range values yield nulls — a corrupt config must never pin
/// sampling to an extreme. (`do_sample` is deliberately ignored: HF uses it
/// for greedy-vs-sample mode selection, which the request's own temperature
/// already expresses.)
pub fn parseGenerationDefaultsFromJson(content: []const u8) GenerationDefaults {
    var buf: [16 * 1024]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&buf);
    const parsed = std.json.parseFromSlice(std.json.Value, fba.allocator(), content, .{}) catch return .{};
    defer parsed.deinit();
    if (parsed.value != .object) return .{};
    const root = parsed.value.object;

    var gd = GenerationDefaults{};
    if (root.get("temperature")) |v| {
        const t: ?f32 = switch (v) {
            .float => |f| @floatCast(f),
            .integer => |i| @floatFromInt(i),
            else => null,
        };
        if (t) |tv| {
            if (tv >= 0.0 and tv <= 2.0) gd.temperature = tv;
        }
    }
    if (root.get("top_p")) |v| {
        const p: ?f32 = switch (v) {
            .float => |f| @floatCast(f),
            .integer => |i| @floatFromInt(i),
            else => null,
        };
        if (p) |pv| {
            if (pv > 0.0 and pv <= 1.0) gd.top_p = pv;
        }
    }
    if (root.get("top_k")) |v| {
        switch (v) {
            .integer => |i| if (i > 0 and i <= 1000) {
                gd.top_k = @intCast(i);
            },
            else => {},
        }
    }
    if (root.get("eos_token_id")) |v| {
        switch (v) {
            .integer => |i| if (i >= 0) {
                gd.eos_token_ids[0] = @intCast(i);
                gd.num_eos = 1;
            },
            .array => |arr| for (arr.items) |item| {
                if (item == .integer and item.integer >= 0 and gd.num_eos < gd.eos_token_ids.len) {
                    gd.eos_token_ids[gd.num_eos] = @intCast(item.integer);
                    gd.num_eos += 1;
                }
            },
            else => {},
        }
    }
    // The checkpoint's own chat-template kwargs. Only a real bool counts —
    // anything else leaves the field null and the arch default in charge.
    if (root.get("default_chat_template_kwargs")) |v| {
        if (v == .object) {
            if (v.object.get("enable_thinking")) |et| {
                if (et == .bool) gd.enable_thinking = et.bool;
            }
        }
    }
    return gd;
}

/// One qwen4_exp integer bound, read strictly: wrong-typed or negative refuses.
fn qwen4ConfigU64(cfg_obj: std.json.ObjectMap, key: []const u8) !?u64 {
    const v = cfg_obj.get(key) orelse return null;
    if (v != .integer or v.integer < 0) return error.InvalidQwen4ConfigField;
    return @intCast(v.integer);
}

fn qwen4ConfigU32(cfg_obj: std.json.ObjectMap, key: []const u8) !?u32 {
    const v = try qwen4ConfigU64(cfg_obj, key) orelse return null;
    if (v > std.math.maxInt(u32)) return error.InvalidQwen4ConfigField;
    return @intCast(v);
}

/// Range-check every qwen4_exp bound the forward indexes a fixed array with or divides by.
/// Names travel to the client as "Model load failed: <name>".
fn validateQwen4Config(config: *const ModelConfig) !void {
    if (config.hidden_size == 0 or config.head_dim == 0 or config.full_attention_interval == 0 or
        config.num_attention_heads == 0 or config.num_key_value_heads == 0 or
        config.num_attention_heads % config.num_key_value_heads != 0 or
        config.num_experts_per_tok == 0 or config.num_experts_per_tok > config.num_experts)
    {
        return error.InvalidQwen4Geometry;
    }
    // `NgramHash.multipliers` is [MAX_NGRAM_SIZE]i64; `ple_prev` is written ngram_size-1 deep.
    if (config.ngram_size < 2 or config.ngram_size > qwen4_exp.MAX_NGRAM_SIZE) {
        return error.InvalidQwen4NgramSize;
    }
    // `vocab`/`offsets` are [MAX_HEADS]i64, written n_heads deep.
    if (config.heads_per_ngram == 0) return error.InvalidQwen4NgramHeads;
    if (config.heads_per_ngram > qwen4_exp.MAX_HEADS / (config.ngram_size - 1)) {
        return error.InvalidQwen4NgramHeads;
    }
    if (config.ngram_vocab_divisor == 0 or config.ngram_vocab_base < 2) {
        return error.InvalidQwen4NgramVocab;
    }
    // The forward divides kv by the ratio and selects `budget / ratio` blocks.
    if (config.indexer_n_heads > 0) {
        if (config.indexer_head_dim == 0) return error.InvalidQwen4Indexer;
        if (config.indexer_compress_ratio == 0) return error.InvalidQwen4Indexer;
        if (config.indexer_budget < config.indexer_compress_ratio) return error.InvalidQwen4Indexer;
    }
    if (config.ple_layer_idx < 0 or config.ple_layer_idx >= @as(i32, @intCast(config.num_hidden_layers))) {
        return error.InvalidQwen4PleLayer;
    }
}

/// True when the layer loop installed the PLE on exactly the layer the config names. A negative
/// index asks for no PLE (the MTP head's own layer) and is satisfied by a loop that installed none.
pub fn qwen4PleInstalledAt(has_ple: []const bool, ple_layer_idx: i32) bool {
    if (ple_layer_idx < 0) return std.mem.indexOfScalar(bool, has_ple, true) == null;
    if (ple_layer_idx >= has_ple.len) return false;
    const want: usize = @intCast(ple_layer_idx);
    for (has_ple, 0..) |p, i| if (p != (i == want)) return false;
    return true;
}

/// I/O-free variant for unit tests and for callers that already have the
/// config.json bytes in memory. The full I/O-bound `parseConfig` delegates here.
/// Qwen3-VL-family vision + M-RoPE fields, shared by the qwen3_5 and
/// qwen4_exp arms (same `vision_config` keys, `rope_parameters.mrope_*`,
/// vision token ids). The generic vision_config block already set
/// `has_vision`; this reads Qwen's own keys into `qv_*`.
fn parseQwenVisionFields(config: *ModelConfig, root: std.json.ObjectMap, cfg_obj: std.json.ObjectMap) !void {
    if (root.get("vision_config")) |vc_val| {
        if (vc_val == .object) {
            const vc = vc_val.object;
            config.has_vision = true;
            config.qwen_vision = true;
            if (vc.get("depth")) |v| {
                if (v == .integer) config.qv_depth = try cfgInt(u32, v);
            }
            if (vc.get("hidden_size")) |v| {
                if (v == .integer) config.qv_hidden = try cfgInt(u32, v);
            }
            if (vc.get("num_heads")) |v| {
                if (v == .integer) config.qv_heads = try cfgInt(u32, v);
            }
            if (vc.get("intermediate_size")) |v| {
                if (v == .integer) config.qv_intermediate = try cfgInt(u32, v);
            }
            if (vc.get("patch_size")) |v| {
                if (v == .integer) config.qv_patch = try cfgInt(u32, v);
            }
            if (vc.get("temporal_patch_size")) |v| {
                if (v == .integer) config.qv_temporal_patch = try cfgInt(u32, v);
            }
            if (vc.get("spatial_merge_size")) |v| {
                if (v == .integer) config.qv_merge = try cfgInt(u32, v);
            }
            if (vc.get("num_position_embeddings")) |v| {
                if (v == .integer) config.qv_num_pos_emb = try cfgInt(u32, v);
            }
            if (vc.get("out_hidden_size")) |v| {
                if (v == .integer) config.qv_out_hidden = try cfgInt(u32, v);
            }
            if (config.qv_heads != 0) config.qv_head_dim = config.qv_hidden / config.qv_heads;
            if (config.qv_out_hidden == 0) config.qv_out_hidden = config.hidden_size;
        }
    }
    // Interleaved M-RoPE sections (text_config.rope_parameters). rope_theta /
    // partial_rotary_factor already parsed in the generic rope block above.
    if (cfg_obj.get("rope_parameters")) |rp| {
        if (rp == .object) {
            if (rp.object.get("mrope_interleaved")) |v| {
                if (v == .bool) config.mrope_interleaved = v.bool;
            }
            if (rp.object.get("mrope_section")) |v| {
                if (v == .array) {
                    for (v.array.items, 0..) |item, i| {
                        if (i >= 3) break;
                        if (item == .integer) config.mrope_section[i] = try cfgInt(u32, item);
                    }
                }
            }
        }
    }
    // Qwen vision token ids (top-level).
    if (root.get("video_token_id")) |v| {
        if (v == .integer) config.video_token_id = try cfgInt(u32, v);
    }
    if (root.get("vision_start_token_id")) |v| {
        if (v == .integer) config.vision_start_token_id = try cfgInt(u32, v);
    }
    if (root.get("vision_end_token_id")) |v| {
        if (v == .integer) config.vision_end_token_id = try cfgInt(u32, v);
    }
}

/// Flat HF `rope_parameters` carrying `rope_type: "yarn"` — the YaRN context
/// extension, i.e. exactly what vLLM's `--hf-overrides` recipe for Qwen3.5
/// writes:
///
///   {"text_config": {"rope_parameters": {"rope_type": "yarn", "factor": 4.0,
///     "original_max_position_embeddings": 262144, "rope_theta": 10000000,
///     "partial_rotary_factor": 0.25, "mrope_interleaved": true,
///     "mrope_section": [11,11,10]}}}
///
/// `factor` may be omitted, in which case HF derives it from
/// `max_position_embeddings / original_max_position_embeddings` (as vLLM's
/// `_get_and_verify_max_len` does). `attention_factor` is HF's key and
/// REPLACES the computed mscale; `attn_factor` is vLLM's and MULTIPLIES
/// `yarnMscale(factor)`. Neither is present in a vendor config, and per HF's
/// default the mscale is then COMPUTED as 0.1·ln(factor)+1 — the value the
/// scaling was calibrated with. Nested per-layer-type `rope_parameters`
/// (laguna/gemma4) never reach here: they have no top-level `rope_type`.
fn parseYarnRopeParameters(config: *ModelConfig, cfg_obj: std.json.ObjectMap) !void {
    const rp_val = cfg_obj.get("rope_parameters") orelse return;
    if (rp_val != .object) return;
    const rp = rp_val.object;
    const rt = rp.get("rope_type") orelse return;
    if (!(rt == .string and std.mem.eql(u8, rt.string, "yarn"))) return;

    if (rp.get("original_max_position_embeddings")) |v| {
        if (v == .integer) config.yarn_orig_max_pos = try cfgInt(u32, v);
    }
    // A YaRN block with no window to scale FROM is not a scaling we can
    // reproduce: the ramp bounds (and so every mid-band frequency) come from
    // it. Refuse the load rather than serve a silently-wrong rotation.
    if (config.yarn_orig_max_pos == 0) return error.YarnRopeNeedsOriginalMaxPos;
    if (cfgField(rp, "factor")) |v| config.yarn_factor = try cfgF32(v);
    if (cfgField(rp, "beta_fast")) |v| config.yarn_beta_fast = try cfgF32(v);
    if (cfgField(rp, "beta_slow")) |v| config.yarn_beta_slow = try cfgF32(v);
    if (rp.get("truncate")) |v| {
        if (v == .bool) config.yarn_truncate = v.bool;
    }
    if (config.yarn_factor <= 0.0) return error.InvalidRopeScalingFactor;
    // HF: `factor = max_position_embeddings / original_max_position_embeddings`
    // when the block leaves it out (the config then only states the window).
    if (cfgField(rp, "factor") == null and config.max_position_embeddings > config.yarn_orig_max_pos) {
        config.yarn_factor = @as(f32, @floatFromInt(config.max_position_embeddings)) /
            @as(f32, @floatFromInt(config.yarn_orig_max_pos));
    }
    // HF `attention_factor` replaces; vLLM `attn_factor` multiplies the
    // computed 0.1·ln(factor)+1. Both present → HF wins.
    if (cfgField(rp, "attention_factor")) |v| {
        config.yarn_attention_factor = try cfgF32(v);
    } else if (cfgField(rp, "attn_factor")) |v| {
        config.yarn_attention_factor = yarnMscale(config.yarn_factor) * try cfgF32(v);
    } else {
        config.yarn_attention_factor = yarnMscale(config.yarn_factor);
    }
    config.rope_yarn = true;
}

/// HF's default YaRN mscale (`attention_factor`) for a scaling `factor`.
fn yarnMscale(factor: f32) f32 {
    if (factor <= 1.0) return 1.0;
    return 0.1 * @log(@as(f32, factor)) + 1.0;
}

/// Launch-time JSON deep-merged into every `config.json` before it is parsed,
/// set once from `--config-overrides`. vLLM's `--hf-overrides` analogue: the
/// only way to re-shape a checkpoint's declared geometry — most often to scale
/// its rope and widen the context — without editing the model directory, and
/// therefore the way to A/B a scaling experiment on identical weights.
var config_overrides: ?[]const u8 = null;

pub fn setConfigOverrides(raw: ?[]const u8) void {
    config_overrides = raw;
}

pub fn getConfigOverrides() ?[]const u8 {
    return config_overrides;
}

/// Deep-merge `overrides` into a config.json document: objects merge key by key
/// — so `{"text_config":{"rope_parameters":{"rope_type":"yarn","factor":4}}}`
/// keeps every sibling it passes through, exactly like vLLM's
/// `_apply_dict_overrides` — and anything else replaces. The whole merge lives
/// in an arena that dies before this returns; only the re-serialized bytes (in
/// `allocator`) escape.
fn mergeConfigJson(allocator: std.mem.Allocator, base: []const u8, overrides: []const u8) ![]const u8 {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var dst = try std.json.parseFromSliceLeaky(std.json.Value, a, base, .{});
    const src = try std.json.parseFromSliceLeaky(std.json.Value, a, overrides, .{});
    if (dst != .object or src != .object) return error.ConfigOverridesMustBeObject;
    try mergeObjects(a, &dst.object, src.object);
    var out: std.Io.Writer.Allocating = .init(a);
    var jws: std.json.Stringify = .{ .writer = &out.writer, .options = .{} };
    try dst.jsonStringify(&jws);
    return allocator.dupe(u8, out.written());
}

fn mergeObjects(a: std.mem.Allocator, dst: *std.json.ObjectMap, src: std.json.ObjectMap) !void {
    var it = src.iterator();
    while (it.next()) |e| {
        if (dst.getPtr(e.key_ptr.*)) |p| {
            if (p.* == .object and e.value_ptr.* == .object) {
                // The handle is copied, so a rehash inside the recursion would
                // be lost — merge through the copy and store it back. `p` stays
                // valid: `dst` itself is not written during the recursion.
                var child = p.object;
                try mergeObjects(a, &child, e.value_ptr.object);
                p.* = .{ .object = child };
                continue;
            }
        }
        // Keys and values are arena-owned by the override document, which
        // outlives this merge.
        try dst.put(a, e.key_ptr.*, e.value_ptr.*);
    }
}

fn glmU32(obj: std.json.ObjectMap, key: []const u8) !u32 {
    const value = try cfgInt(u32, obj.get(key) orelse return error.InvalidGlmConfig);
    if (value > std.math.maxInt(c_int)) return error.UnsupportedGlmConfig;
    return value;
}

fn glmF32(obj: std.json.ObjectMap, key: []const u8) !f32 {
    const value = try cfgF32(obj.get(key) orelse return error.InvalidGlmConfig);
    if (!std.math.isFinite(value)) return error.InvalidGlmConfig;
    return value;
}

fn glmRequireBool(obj: std.json.ObjectMap, key: []const u8, expected: bool) !void {
    const value = obj.get(key) orelse return error.InvalidGlmConfig;
    if (value != .bool) return error.InvalidGlmConfig;
    if (value.bool != expected) return error.UnsupportedGlmConfig;
}

fn glmRequireString(obj: std.json.ObjectMap, key: []const u8, expected: []const u8) !void {
    const value = obj.get(key) orelse return error.InvalidGlmConfig;
    if (value != .string) return error.InvalidGlmConfig;
    if (!std.mem.eql(u8, value.string, expected)) return error.UnsupportedGlmConfig;
}

fn glmLayerList(obj: std.json.ObjectMap, key: []const u8, layers: u32, linear: bool) !void {
    const value = obj.get(key) orelse return;
    if (value != .array) return error.InvalidGlmConfig;
    var item: usize = 0;
    for (0..layers) |li| {
        if (((li + 1) % 4 != 0) != linear) continue;
        if (item >= value.array.items.len or value.array.items[item] != .integer or
            value.array.items[item].integer != li) return error.UnsupportedGlmConfig;
        item += 1;
    }
    if (item != value.array.items.len) return error.UnsupportedGlmConfig;
}

fn parseGlm5VisionFields(c: *ModelConfig, root: std.json.ObjectMap) !void {
    const value = root.get("vision_config") orelse return;
    if (value != .object) return error.InvalidGlmVisionConfig;
    const v = value.object;
    try glmRequireString(v, "model_type", "glm5_next_vision");
    try glmRequireString(v, "hidden_act", "silu");
    try glmRequireBool(v, "attention_bias", true);
    c.qv_depth = try glmU32(v, "depth");
    c.qv_hidden = try glmU32(v, "hidden_size");
    c.qv_heads = try glmU32(v, "num_heads");
    c.qv_intermediate = try glmU32(v, "intermediate_size");
    c.qv_patch = try glmU32(v, "patch_size");
    c.qv_temporal_patch = try glmU32(v, "temporal_patch_size");
    c.qv_merge = try glmU32(v, "spatial_merge_size");
    c.qv_out_hidden = try glmU32(v, "out_hidden_size");
    c.glmv_projection_intermediate = try glmU32(v, "projection_intermediate_size");
    c.glmv_eps = try glmF32(v, "rms_norm_eps");
    c.glmv_swiglu_limit = try glmF32(v, "swiglu_limit");
    if (c.qv_heads == 0 or c.qv_hidden % c.qv_heads != 0 or c.qv_out_hidden != c.hidden_size or
        c.qv_depth == 0 or c.qv_hidden == 0 or c.qv_intermediate == 0 or c.qv_patch == 0 or c.qv_patch > 64 or
        c.qv_temporal_patch == 0 or c.qv_temporal_patch > 8 or c.qv_merge != 2 or
        c.glmv_projection_intermediate == 0 or !std.math.isFinite(c.glmv_eps) or c.glmv_eps <= 0 or
        !std.math.isFinite(c.glmv_swiglu_limit) or c.glmv_swiglu_limit <= 0) return error.InvalidGlmVisionConfig;
    c.qv_head_dim = c.qv_hidden / c.qv_heads;
    if (c.qv_head_dim % 4 != 0) return error.InvalidGlmVisionConfig;
    if (v.get("rope_parameters")) |rp| {
        if (rp == .object) {
            try glmRequireString(rp.object, "rope_type", "axial");
            c.glmv_rope_theta = try glmF32(rp.object, "rope_theta");
            if (!std.math.isFinite(c.glmv_rope_theta) or c.glmv_rope_theta <= 0) return error.InvalidGlmVisionConfig;
        }
    }
    c.image_token_id = try glmU32(root, "image_token_id");
    c.video_token_id = try glmU32(root, "video_token_id");
    c.vision_start_token_id = try glmU32(root, "image_start_token_id");
    c.vision_end_token_id = try glmU32(root, "image_end_token_id");
    for ([_]u32{ c.image_token_id, c.video_token_id, c.vision_start_token_id, c.vision_end_token_id }) |id| if (id >= c.vocab_size) return error.InvalidGlmVisionConfig;
    c.glm5_vision = true;
    c.has_vision = true;
}

fn parseGlm5Fields(c: *ModelConfig, obj: std.json.ObjectMap) !void {
    // These fields define the supported computation, not optional loader hints.
    try glmRequireBool(obj, "mhc", true);
    try glmRequireBool(obj, "attention_bias", false);
    try glmRequireBool(obj, "index_kpool_compress", true);
    try glmRequireBool(obj, "tie_word_embeddings", false);
    try glmRequireString(obj, "scoring_func", "sigmoid");
    try glmRequireString(obj, "topk_method", "noaux_tc");
    try glmRequireString(obj, "hidden_act", "silu");
    try glmRequireString(obj, "moe_router_dtype", "float32");
    c.hidden_size = try glmU32(obj, "hidden_size");
    c.vocab_size = try glmU32(obj, "vocab_size");
    c.intermediate_size = try glmU32(obj, "intermediate_size");
    c.num_hidden_layers = try glmU32(obj, "num_hidden_layers");
    c.num_attention_heads = try glmU32(obj, "num_attention_heads");
    c.num_experts_per_tok = try glmU32(obj, "num_experts_per_tok");
    c.moe_intermediate_size = try glmU32(obj, "moe_intermediate_size");
    c.max_position_embeddings = try glmU32(obj, "max_position_embeddings");
    c.rms_norm_eps = try glmF32(obj, "rms_norm_eps");
    c.has_vision = false;
    c.model_type = "glm5_next";
    c.weight_prefix = "model.language_model";
    c.norm_has_offset = false;
    c.scale_embeddings = false;
    c.has_pre_ff_norm = true;
    c.has_final_norm = true;
    c.has_qk_norm = false;
    c.hidden_act = .silu;
    c.has_sliding_window = false;
    c.attn_output_gate = false;
    c.partial_rotary_factor = 0;
    c.expert_layout = .bf16_individual;
    c.num_experts = try glmU32(obj, "n_routed_experts");
    c.first_k_dense_replace = try glmU32(obj, "first_k_dense_replace");
    c.shared_expert_intermediate_size = try std.math.mul(u32, c.moe_intermediate_size, try glmU32(obj, "n_shared_experts"));
    c.router_scaling_factor = try glmF32(obj, "routed_scaling_factor");
    c.moe_sigmoid_router = true;
    const norm = obj.get("norm_topk_prob") orelse return error.InvalidGlmConfig;
    if (norm != .bool) return error.InvalidGlmConfig;
    c.moe_route_norm = norm.bool;
    c.moe_n_group = try glmU32(obj, "n_group");
    c.moe_topk_group = try glmU32(obj, "topk_group");
    if (c.moe_n_group != 1 or c.moe_topk_group != 1) return error.UnsupportedGlmConfig;
    c.mla_q_lora_rank = try glmU32(obj, "q_lora_rank");
    c.mla_kv_lora_rank = try glmU32(obj, "kv_lora_rank");
    c.mla_qk_nope_head_dim = try glmU32(obj, "qk_nope_head_dim");
    c.mla_qk_rope_head_dim = try glmU32(obj, "qk_rope_head_dim");
    c.mla_v_head_dim = try glmU32(obj, "v_head_dim");
    const nope = obj.get("mla_use_nope") orelse return error.InvalidGlmConfig;
    if (nope != .bool or !nope.bool or c.mla_qk_rope_head_dim != 0) return error.UnsupportedGlmConfig;
    // Attention operates on one compressed latent, shared by every query head.
    c.head_dim = c.mla_kv_lora_rank;
    c.v_head_dim = c.mla_kv_lora_rank;
    c.num_key_value_heads = 1;
    c.query_pre_attn_scalar = c.mla_qk_nope_head_dim;
    const linear = try cfgObject(obj.get("linear_attn_config") orelse return error.InvalidGlmConfig);
    c.linear_num_key_heads = try glmU32(linear, "num_heads");
    c.linear_num_value_heads = c.linear_num_key_heads;
    c.linear_key_head_dim = try glmU32(linear, "head_dim");
    c.linear_value_head_dim = c.linear_key_head_dim;
    c.linear_conv_kernel_dim = try glmU32(linear, "short_conv_kernel_size");
    c.kda_vector_gate = true;
    c.kda_sigmoid_out_gate = true;
    c.kda_gate_lower_bound = try glmF32(linear, "gate_lower_bound");
    c.hc_count = try glmU32(obj, "hc_mult");
    c.glm_hc_sinkhorn_iters = try glmU32(obj, "hc_sinkhorn_iters");
    c.glm_hc_eps = try glmF32(obj, "hc_eps");
    c.glm_swiglu_limit = try glmF32(obj, "swiglu_limit");
    c.indexer_n_heads = try glmU32(obj, "index_n_heads");
    c.indexer_head_dim = try glmU32(obj, "index_head_dim");
    c.indexer_budget = try glmU32(obj, "index_topk");
    c.indexer_compress_ratio = try glmU32(obj, "index_kpool");
    const tail = obj.get("index_kpool_always_select_tail") orelse return error.InvalidGlmConfig;
    if (tail != .bool) return error.InvalidGlmConfig;
    c.glm_index_tail = tail.bool;
    if (obj.get("num_nextn_predict_layers")) |v| c.glm_mtp_layers = try cfgInt(u32, v);
    c.full_attention_interval = 4;
    if (c.hc_count != 4 or c.linear_key_head_dim != 128 or c.linear_conv_kernel_dim < 1 or
        c.num_experts == 0 or c.num_experts > 512 or c.num_experts_per_tok == 0 or c.num_experts_per_tok > 32 or
        c.num_experts_per_tok > c.num_experts or c.first_k_dense_replace >= c.num_hidden_layers or
        c.num_hidden_layers > 128 or c.hidden_size == 0 or c.hidden_size % 128 != 0 or
        c.moe_intermediate_size == 0 or c.moe_intermediate_size % 128 != 0 or
        c.mla_kv_lora_rank == 0 or c.mla_q_lora_rank == 0 or c.mla_qk_nope_head_dim == 0 or
        c.glm_hc_sinkhorn_iters == 0 or c.glm_hc_sinkhorn_iters > 100 or c.glm_hc_eps <= 0 or
        c.glm_swiglu_limit != 10 or c.indexer_compress_ratio != 4 or c.indexer_budget < 4 or
        c.indexer_budget % 4 != 0 or c.router_scaling_factor <= 0 or c.kda_gate_lower_bound >= 0)
        return error.UnsupportedGlmConfig;
    if (c.vocab_size == 0 or c.intermediate_size == 0 or c.num_attention_heads == 0 or
        c.linear_num_key_heads == 0 or c.indexer_n_heads == 0 or c.indexer_head_dim == 0 or
        c.mla_v_head_dim == 0 or c.max_position_embeddings == 0 or c.rms_norm_eps <= 0) return error.UnsupportedGlmConfig;
    // Array dimensions and kernel index arithmetic use signed 32-bit products.
    for ([_][2]u32{
        .{ c.linear_num_key_heads, c.linear_key_head_dim },
        .{ c.indexer_n_heads, c.indexer_head_dim },
        .{ c.num_attention_heads, c.mla_qk_nope_head_dim },
        .{ c.num_attention_heads, c.mla_v_head_dim },
        .{ c.hc_count, c.hidden_size },
    }) |dims| {
        const width = std.math.mul(u32, dims[0], dims[1]) catch return error.UnsupportedGlmConfig;
        if (width > std.math.maxInt(c_int)) return error.UnsupportedGlmConfig;
    }
    if (obj.get("qk_head_dim")) |value| {
        if (try cfgInt(u32, value) != c.mla_qk_nope_head_dim) return error.UnsupportedGlmConfig;
    }
    try glmLayerList(linear, "kda_layers", c.num_hidden_layers, true);
    try glmLayerList(linear, "full_attn_layers", c.num_hidden_layers, false);
    const mlps = obj.get("mlp_layer_types") orelse return error.InvalidGlmConfig;
    if (mlps != .array or mlps.array.items.len != c.num_hidden_layers) return error.InvalidGlmConfig;
    for (mlps.array.items, 0..) |value, i| {
        const expected = if (i < c.first_k_dense_replace) "dense" else "sparse";
        if (value != .string or !std.mem.eql(u8, value.string, expected)) return error.UnsupportedGlmConfig;
    }
    if (obj.get("indexer_types")) |types| {
        if (types != .array or types.array.items.len != c.num_hidden_layers) return error.InvalidGlmConfig;
        for (types.array.items) |value| {
            if (value != .string or !std.mem.eql(u8, value.string, "full")) return error.UnsupportedGlmConfig;
        }
    }
    const kinds = obj.get("layer_types") orelse return error.InvalidGlmConfig;
    if (kinds != .array or kinds.array.items.len != c.num_hidden_layers) return error.InvalidGlmConfig;
    for (kinds.array.items, 0..) |value, i| {
        const expected = if ((i + 1) % 4 == 0) "deepseek_sparse_attention" else "linear_attention";
        if (value != .string or !std.mem.eql(u8, value.string, expected)) return error.UnsupportedGlmConfig;
    }
}

pub fn parseConfigFromJson(allocator: std.mem.Allocator, content: []const u8) !ModelConfig {
    // The launch-time overrides apply to EVERY parse (primary load, on-demand
    // load, discovery stubs), so the advertised context and the loaded model
    // can never disagree about what window the checkpoint has.
    const merged: ?[]const u8 = if (config_overrides) |ov| blk: {
        break :blk try mergeConfigJson(allocator, content, ov);
    } else null;
    defer if (merged) |m| allocator.free(m);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, merged orelse content, .{});
    defer parsed.deinit();

    const root = try cfgObject(parsed.value);
    var config = ModelConfig{};

    // Detect model_type from top-level (always present)
    const model_type = if (cfgField(root, "model_type")) |v| try cfgString(v) else "gemma3";

    // Determine which object to read config from: text_config (nested) or root (flat)
    const cfg_obj = if (cfgField(root, "text_config")) |tc_val| try cfgObject(tc_val) else root;

    // Parse common fields
    if (cfgField(cfg_obj, "vocab_size")) |v| config.vocab_size = try cfgInt(u32, v);
    if (cfgField(cfg_obj, "hidden_size")) |v| config.hidden_size = try cfgInt(u32, v);
    if (cfgField(cfg_obj, "intermediate_size")) |v| {
        config.intermediate_size = try cfgInt(u32, v);
        config.intermediate_size_declared = true;
    }
    if (cfgField(cfg_obj, "num_hidden_layers")) |v| config.num_hidden_layers = try cfgInt(u32, v);
    if (cfgField(cfg_obj, "num_attention_heads")) |v| config.num_attention_heads = try cfgInt(u32, v);
    if (cfgField(cfg_obj, "num_key_value_heads")) |v| config.num_key_value_heads = try cfgInt(u32, v);
    if (cfgField(cfg_obj, "head_dim")) |v| config.head_dim = try cfgInt(u32, v);
    if (cfgField(cfg_obj, "max_position_embeddings")) |v| config.max_position_embeddings = try cfgInt(u32, v);
    if (cfgField(cfg_obj, "rms_norm_eps")) |v| config.rms_norm_eps = try cfgF32(v);
    if (cfgField(cfg_obj, "rope_theta")) |v| config.rope_theta = try cfgF32(v);
    if (cfgField(cfg_obj, "query_pre_attn_scalar")) |v| config.query_pre_attn_scalar = try cfgInt(u32, v);

    // MoE fields (guard against JSON null values)
    if (cfg_obj.get("num_experts")) |v| {
        if (v == .integer) config.num_experts = try cfgInt(u32, v);
    }
    if (cfg_obj.get("num_experts_per_tok")) |v| {
        if (v == .integer) config.num_experts_per_tok = try cfgInt(u32, v);
    }
    if (cfg_obj.get("top_k_experts")) |v| {
        if (v == .integer) config.num_experts_per_tok = try cfgInt(u32, v);
    }
    if (cfg_obj.get("moe_intermediate_size")) |v| {
        if (v == .integer) config.moe_intermediate_size = try cfgInt(u32, v);
    }
    if (cfg_obj.get("shared_expert_intermediate_size")) |v| {
        if (v == .integer) config.shared_expert_intermediate_size = try cfgInt(u32, v);
    }

    // Linear attention (GatedDeltaNet) fields
    if (cfgField(cfg_obj, "linear_num_key_heads")) |v| config.linear_num_key_heads = try cfgInt(u32, v);
    if (cfgField(cfg_obj, "linear_num_value_heads")) |v| config.linear_num_value_heads = try cfgInt(u32, v);
    if (cfgField(cfg_obj, "linear_key_head_dim")) |v| config.linear_key_head_dim = try cfgInt(u32, v);
    if (cfgField(cfg_obj, "linear_value_head_dim")) |v| config.linear_value_head_dim = try cfgInt(u32, v);
    if (cfgField(cfg_obj, "linear_conv_kernel_dim")) |v| config.linear_conv_kernel_dim = try cfgInt(u32, v);

    // Hybrid attention
    if (cfgField(cfg_obj, "full_attention_interval")) |v| config.full_attention_interval = try cfgInt(u32, v);
    if (cfg_obj.get("attn_output_gate")) |v| {
        if (v == .bool) config.attn_output_gate = v.bool;
    }

    // Explicit pooling contract (issue #116): "mean" | "cls" | "last_token" in
    // config.json marks a checkpoint as an embedding model and picks the pool
    // op. An unknown value is a parse error, never a silent mean-pool —
    // wrong-semantics vectors are harder to detect than a refused load.
    if (root.get("pooling_mode")) |v| {
        if (v == .string) {
            config.pooling_mode = PoolingMode.fromString(v.string) orelse
                return error.UnsupportedPoolingMode;
        }
    }
    if (cfg_obj.get("bos_token_id")) |v| {
        if (v == .integer and v.integer >= 0) config.bos_token_id = try cfgInt(u32, v);
    }

    // Rope parameters (nested for Qwen3.5)
    if (cfg_obj.get("rope_parameters")) |rp_val| {
        if (rp_val == .object) {
            if (cfgField(rp_val.object, "rope_theta")) |v| config.rope_theta = try cfgF32(v);
            if (cfgField(rp_val.object, "partial_rotary_factor")) |v| config.partial_rotary_factor = try cfgF32(v);
        }
    }

    // Sliding window
    if (cfg_obj.get("sliding_window")) |v| {
        if (v == .null) {
            config.has_sliding_window = false;
        } else {
            config.sliding_window = try cfgInt(u32, v);
            config.has_sliding_window = true;
        }
    }
    if (cfgField(cfg_obj, "sliding_window_pattern")) |v| config.sliding_window_pattern = try cfgInt(u32, v);

    // Gemma-specific: dual RoPE bases
    if (cfg_obj.get("rope_local_base_freq")) |v| config.rope_local_base_freq = jsonFloat(v);
    if (cfg_obj.get("rope_scaling")) |rs_val| {
        if (rs_val == .object) {
            if (rs_val.object.get("factor")) |v| config.rope_scaling_factor = jsonFloat(v);
        }
    }

    // Gemma 4: explicit layer_types array
    if (cfg_obj.get("layer_types")) |lt_val| {
        if (lt_val == .array) {
            config.has_explicit_layer_types = true;
            for (lt_val.array.items, 0..) |item, i| {
                if (i >= 128) break;
                if (item == .string) {
                    config.layer_is_global[i] = std.mem.eql(u8, item.string, "full_attention");
                }
            }
        }
    }

    // Tie word embeddings
    if (root.get("tie_word_embeddings")) |v| {
        if (v == .bool) config.tie_word_embeddings = v.bool;
    }
    if (cfg_obj.get("tie_word_embeddings")) |v| {
        if (v == .bool) config.tie_word_embeddings = v.bool;
    }

    // Check root level for max_position_embeddings (may not be in text_config)
    if (config.max_position_embeddings == 0) {
        if (root.get("max_position_embeddings")) |v| {
            if (v == .integer) config.max_position_embeddings = try cfgInt(u32, v);
        }
    }

    // Parse quantization from top level
    if (cfgField(root, "quantization")) |q_val| {
        const q = try cfgObject(q_val);
        if (cfgField(q, "bits")) |v| config.quant_bits = try cfgInt(u32, v);
        if (cfgField(q, "group_size")) |v| config.quant_group_size = try cfgInt(u32, v);
        if (q.get("mode")) |v| {
            if (v == .string) {
                config.quant_mode = QuantMode.fromString(v.string) orelse {
                    log.err("unsupported quantization mode '{s}' (supported: affine, nvfp4, mxfp4, mxfp8)\n", .{v.string});
                    return error.UnsupportedQuantMode;
                };
            }
        }
        // MLX ships affine kernels only for bits {2,3,4,5,6,8} (ops.cpp
        // rejects the rest at quantize() time, but an ALREADY-quantized
        // checkpoint skips that check and dies at Metal kernel load during
        // warmup — an uncatchable process kill). Reject at parse instead.
        if (config.quant_mode == .affine and config.quant_bits != 0) {
            switch (config.quant_bits) {
                2, 3, 4, 5, 6, 8 => {},
                else => {
                    log.err("unsupported affine quantization: {d}-bit (this MLX runtime supports 2, 3, 4, 5, 6, 8)\n", .{config.quant_bits});
                    return error.UnsupportedQuantBits;
                },
            }
        }
    }

    // EOS tokens
    if (root.get("eos_token_id")) |v| {
        switch (v) {
            .integer => config.addEosToken(try cfgInt(u32, v)),
            .array => |arr| {
                for (arr.items) |item| {
                    if (item == .integer) config.addEosToken(try cfgInt(u32, item));
                }
            },
            else => {},
        }
    }

    // Image token ID (top-level or in mm_tokens_per_image config)
    if (root.get("image_token_id")) |v| {
        if (v == .integer) config.image_token_id = try cfgInt(u32, v);
    }
    if (root.get("image_token_index")) |v| {
        if (v == .integer and config.image_token_id == 0) config.image_token_id = try cfgInt(u32, v);
    }

    // Set model-family defaults based on model_type
    if (std.mem.eql(u8, model_type, "glm5_next") or std.mem.eql(u8, model_type, "glm5_next_text")) {
        try parseGlm5Fields(&config, cfg_obj);
        if (root.get("sushi_pack")) |pack| {
            if (pack == .object) if (pack.object.get("trunk_storage")) |storage| {
                if (storage == .string) config.glm_fp8_trunk = std.mem.eql(u8, storage.string, "source-fp8-e4m3fn-block128");
            };
        }
        if (root.get("quantization_config")) |q| if (q == .object) if (q.object.get("quant_method")) |method| {
            if (method == .string and std.mem.eql(u8, method.string, "fp8")) {
                const block = q.object.get("weight_block_size") orelse return error.UnsupportedGlmConfig;
                if (block != .array or block.array.items.len != 2) return error.UnsupportedGlmConfig;
                for (block.array.items) |side| if (side != .integer or side.integer != 128) return error.UnsupportedGlmConfig;
                config.glm_fp8_trunk = true;
            }
        };
        try parseGlm5VisionFields(&config, root);
    } else if (std.mem.eql(u8, model_type, "qwen4_exp") or
        std.mem.eql(u8, model_type, "qwen4_exp_text"))
    {
        config.model_type = "qwen4_exp";
        config.weight_prefix = "language_model.model";
        config.norm_has_offset = false; // the converter folds every (1 + w) norm
        config.has_final_norm = false; // hyper_connection_mixer replaces model.norm
        config.scale_embeddings = false;
        config.has_pre_ff_norm = false;
        config.has_qk_norm = true;
        config.hidden_act = .silu;
        config.has_sliding_window = false;
        config.attn_output_gate = true;
        config.kda_sigmoid_out_gate = true; // output_gate_type "sigmoid"
        config.rope_scaling_factor = 1.0;
        config.rope_local_base_freq = config.rope_theta;
        if (cfg_obj.get("query_pre_attn_scalar") == null) {
            config.query_pre_attn_scalar = config.head_dim;
        }
        config.hc_count = 4;
        config.hc_lowrank = 320;
        config.ple_embed_dim = config.hidden_size;
        try parseQwenVisionFields(&config, root, cfg_obj);
        // YaRN (262144 → e.g. 1048576) rides ONE rotary table for the whole
        // trunk: attention, the QSA indexer and the MTP head all read it, so
        // a scaled rotation cannot desync the block selector from attention.
        try parseYarnRopeParameters(&config, cfg_obj);
        // Read strictly; range-checked in `validateQwen4Config` once every field is in.
        if (try qwen4ConfigU32(cfg_obj, "hc_count")) |v| config.hc_count = v;
        if (try qwen4ConfigU32(cfg_obj, "hc_lowrank")) |v| config.hc_lowrank = v;
        {
            const v = cfg_obj.get("ple_layer_ids") orelse return error.InvalidQwen4PleLayer;
            if (v != .array or v.array.items.len != 1 or v.array.items[0] != .integer) {
                return error.InvalidQwen4PleLayer;
            }
            const id = v.array.items[0].integer;
            if (id < 1 or id > @as(i64, config.num_hidden_layers)) return error.InvalidQwen4PleLayer;
            config.ple_layer_idx = @intCast(id - 1);
        }
        if (try qwen4ConfigU32(cfg_obj, "ple_embed_dim")) |v| config.ple_embed_dim = v;
        if (try qwen4ConfigU32(cfg_obj, "ple_conv_kernel_size")) |v| config.ple_conv_kernel = v;
        if (try qwen4ConfigU32(cfg_obj, "ngram_size")) |v| config.ngram_size = v;
        if (try qwen4ConfigU32(cfg_obj, "heads_per_ngram")) |v| config.heads_per_ngram = v;
        if (try qwen4ConfigU64(cfg_obj, "ngram_vocab_size_base")) |v| config.ngram_vocab_base = v;
        if (try qwen4ConfigU32(cfg_obj, "make_ngram_vocab_size_divisible_by")) |v| config.ngram_vocab_divisor = v;
        if (try qwen4ConfigU64(cfg_obj, "seed")) |v| config.ngram_seed = v;
        if (try qwen4ConfigU32(cfg_obj, "indexer_n_heads")) |v| config.indexer_n_heads = v;
        if (try qwen4ConfigU32(cfg_obj, "indexer_head_dim")) |v| config.indexer_head_dim = v;
        if (try qwen4ConfigU32(cfg_obj, "indexer_budget")) |v| config.indexer_budget = v;
        if (try qwen4ConfigU32(cfg_obj, "indexer_compress_ratio")) |v| config.indexer_compress_ratio = v;
        if (cfg_obj.get("eos_token_id")) |v| {
            switch (v) {
                .integer => config.ngram_eos = try cfgInt(u32, v),
                .array => |arr| if (arr.items.len > 0 and arr.items[0] == .integer) {
                    config.ngram_eos = try cfgInt(u32, arr.items[0]);
                },
                else => {},
            }
            if (config.num_eos_tokens == 0) config.addEosToken(config.ngram_eos);
        }
        try validateQwen4Config(&config);
    } else if (std.mem.eql(u8, model_type, "mimo_v2")) {
        try parseMimoConfig(&config, cfg_obj);
    } else {
        // Not served: the weight loader refuses it by name (`ArchitectureUnsupported`).
        config.model_type = "unsupported";
    }

    return config;
}

fn mimoUint(obj: std.json.ObjectMap, key: []const u8, fallback: u32) !u32 {
    const v = obj.get(key) orelse return fallback;
    if (v != .integer or v.integer < 0 or v.integer > std.math.maxInt(u32))
        return error.UnsupportedMimoV2Config;
    return @intCast(v.integer);
}

fn mimoFloat(obj: std.json.ObjectMap, key: []const u8, fallback: f32) !f32 {
    const v = obj.get(key) orelse return fallback;
    if (v != .integer and v != .float) return error.UnsupportedMimoV2Config;
    const f = jsonFloat(v);
    if (!std.math.isFinite(f)) return error.UnsupportedMimoV2Config;
    return f;
}

fn mimoBool(obj: std.json.ObjectMap, key: []const u8, fallback: bool) !bool {
    const v = obj.get(key) orelse return fallback;
    if (v != .bool) return error.UnsupportedMimoV2Config;
    return v.bool;
}

/// MiMo-ViT geometry, the image token ids, and the pixel bounds the vendor
/// processors read from config.json's own `processor_config` (they ignore
/// preprocessor_config.json).
fn parseMimoVision(c: *ModelConfig, obj: std.json.ObjectMap) !void {
    const vc_val = obj.get("vision_config") orelse return;
    if (vc_val != .object) return error.UnsupportedMimoV2Config;
    const vc = vc_val.object;
    c.qv_depth = try mimoUint(vc, "depth", 0);
    c.qv_hidden = try mimoUint(vc, "hidden_size", 0);
    c.qv_heads = try mimoUint(vc, "num_heads", 0);
    c.qv_head_dim = try mimoUint(vc, "qk_channels", 64);
    c.mvit_kv_heads = try mimoUint(vc, "num_key_value_heads", c.qv_heads);
    c.qv_intermediate = try mimoUint(vc, "intermediate_size", 0);
    c.qv_out_hidden = try mimoUint(vc, "out_hidden_size", c.hidden_size);
    c.qv_patch = try mimoUint(vc, "patch_size", 16);
    c.qv_merge = try mimoUint(vc, "spatial_merge_size", 2);
    c.qv_temporal_patch = try mimoUint(vc, "temporal_patch_size", 2);
    c.mvit_window = try mimoUint(vc, "visual_token_window_size", 0);
    c.mvit_sinks = try mimoBool(vc, "use_sink", false);
    if (c.qv_depth == 0 or c.qv_depth > MAX_VISION_LAYERS or c.qv_hidden == 0 or c.qv_heads == 0 or
        c.qv_head_dim == 0 or c.mvit_kv_heads == 0 or c.qv_heads % c.mvit_kv_heads != 0 or
        c.qv_intermediate == 0 or c.qv_out_hidden != c.hidden_size or c.qv_patch == 0 or c.qv_merge == 0 or
        c.qv_temporal_patch == 0 or c.mvit_window == 0)
        return error.UnsupportedMimoV2Config;

    var full: [MAX_VISION_LAYERS]bool = @splat(false);
    if (vc.get("fullatt_block_indexes")) |v| {
        if (v != .array) return error.UnsupportedMimoV2Config;
        for (v.array.items) |item| {
            if (item != .integer or item.integer < 0 or item.integer >= c.qv_depth) return error.UnsupportedMimoV2Config;
            full[@intCast(item.integer)] = true;
        }
    }
    const types = vc.get("vit_window_attn_types") orelse return error.UnsupportedMimoV2Config;
    if (types != .array or types.array.items.len != c.qv_depth) return error.UnsupportedMimoV2Config;
    for (types.array.items, 0..) |item, i| {
        if (item != .integer) return error.UnsupportedMimoV2Config;
        c.mvit_attn[i] = if (full[i]) .full else switch (item.integer) {
            -1, 0 => .row,
            1 => .col,
            else => return error.UnsupportedMimoV2Config,
        };
    }

    c.image_token_id = try mimoUint(obj, "image_token_id", 0);
    c.vision_start_token_id = try mimoUint(obj, "vision_start_token_id", 0);
    c.vision_end_token_id = try mimoUint(obj, "vision_end_token_id", 0);
    if (c.image_token_id == 0 or c.vision_start_token_id == 0 or c.vision_end_token_id == 0)
        return error.UnsupportedMimoV2Config;
    if (obj.get("processor_config")) |pc| {
        if (pc != .object) return error.UnsupportedMimoV2Config;
        c.qv_min_pixels = try mimoUint(pc.object, "image_min_pixels", 0);
        c.qv_max_pixels = try mimoUint(pc.object, "image_max_pixels", 0);
    }
    c.mimo_vision = true;
}

fn parseMimoConfig(c: *ModelConfig, obj: std.json.ObjectMap) !void {
    c.model_type = "mimo_v2";
    c.weight_prefix = "model";
    c.norm_has_offset = false;
    c.scale_embeddings = false;
    c.has_pre_ff_norm = false;
    c.has_qk_norm = false;
    c.hidden_act = .silu;
    try parseMimoVision(c, obj);
    c.has_vision = c.mimo_vision;
    // Video is not served yet; its pads must never join the splice.
    c.video_token_id = 0;
    c.has_sliding_window = true;
    c.has_explicit_layer_types = true;
    c.rope_scaling_factor = 1;
    c.rms_norm_eps = try mimoFloat(obj, "layernorm_epsilon", c.rms_norm_eps);
    c.partial_rotary_factor = try mimoFloat(obj, "partial_rotary_factor", c.partial_rotary_factor);
    c.rope_local_base_freq = try mimoFloat(obj, "swa_rope_theta", 10000);
    c.attention_value_scale = try mimoFloat(obj, "attention_value_scale", 1);
    c.sliding_window = try mimoUint(obj, "sliding_window", try mimoUint(obj, "sliding_window_size", 128));
    if (c.num_hidden_layers == 0 or c.num_hidden_layers > c.layer_is_global.len or
        c.sliding_window == 0 or c.hidden_size == 0 or c.rms_norm_eps <= 0 or
        c.rope_theta <= 0 or c.rope_local_base_freq <= 0 or
        c.partial_rotary_factor <= 0 or c.partial_rotary_factor > 1)
        return error.UnsupportedMimoV2Config;

    const pattern = obj.get("hybrid_layer_pattern") orelse return error.UnsupportedMimoV2Config;
    if (pattern != .array or pattern.array.items.len != c.num_hidden_layers)
        return error.UnsupportedMimoV2Config;
    for (pattern.array.items, 0..) |v, i| {
        if (v != .integer or (v.integer != 0 and v.integer != 1)) return error.UnsupportedMimoV2Config;
        c.layer_is_global[i] = v.integer == 0;
    }

    // The config's plain geometry is global; our layer helpers use sliding defaults.
    c.global_head_dim = c.head_dim;
    c.global_v_head_dim = try mimoUint(obj, "v_head_dim", c.head_dim);
    c.num_global_key_value_heads = c.num_key_value_heads;
    c.head_dim = try mimoUint(obj, "swa_head_dim", c.global_head_dim);
    c.v_head_dim = try mimoUint(obj, "swa_v_head_dim", c.global_v_head_dim);
    c.num_key_value_heads = try mimoUint(obj, "swa_num_key_value_heads", c.num_global_key_value_heads);
    const swa_heads = try mimoUint(obj, "swa_num_attention_heads", c.num_attention_heads);
    if (c.global_head_dim == 0 or c.global_v_head_dim == 0 or c.num_global_key_value_heads == 0 or
        c.head_dim == 0 or c.v_head_dim == 0 or c.num_key_value_heads == 0 or
        c.num_attention_heads == 0 or swa_heads == 0)
        return error.UnsupportedMimoV2Config;
    c.has_per_layer_heads = true;
    for (0..c.num_hidden_layers) |i| {
        c.num_attention_heads_per_layer[i] = if (c.layer_is_global[i]) c.num_attention_heads else swa_heads;
        const li: u32 = @intCast(i);
        const heads = c.layerNumHeads(li);
        const kv = c.layerKVHeads(li);
        const hd = c.layerHeadDim(li);
        if (heads == 0 or kv == 0 or heads % kv != 0 or hd == 0 or c.layerVHeadDim(li) == 0)
            return error.UnsupportedMimoV2Config;
        const rd: u32 = @intFromFloat(@as(f64, @floatFromInt(hd)) * c.partial_rotary_factor);
        if (rd == 0 or rd % 2 != 0) return error.UnsupportedMimoV2Config;
    }
    c.query_pre_attn_scalar = c.global_head_dim;
    c.attn_sinks_sliding = try mimoBool(obj, "add_swa_attention_sink_bias", true);
    c.attn_sinks_global = try mimoBool(obj, "add_full_attention_sink_bias", false);
    c.has_attn_sinks = c.attn_sinks_sliding or c.attn_sinks_global;
    if (obj.get("attention_projection_layout")) |v| {
        if (v != .string) return error.UnsupportedMimoV2Config;
        c.attn_fused_qkv = std.mem.eql(u8, v.string, "fused_qkv");
        if (!c.attn_fused_qkv and !std.mem.eql(u8, v.string, "split_qkv") and
            !std.mem.eql(u8, v.string, "split"))
            return error.UnsupportedMimoV2Config;
    }

    c.num_experts = try mimoUint(obj, "n_routed_experts", c.num_experts);
    c.moe_sigmoid_router = true;
    c.moe_n_group = try mimoUint(obj, "n_group", 1);
    c.moe_topk_group = try mimoUint(obj, "topk_group", 1);
    c.moe_route_norm = try mimoBool(obj, "norm_topk_prob", true);
    if (obj.get("routed_scaling_factor")) |v| {
        if (v != .null) c.router_scaling_factor = try mimoFloat(obj, "routed_scaling_factor", 1);
    }
    for ([_][]const u8{ "scoring_func", "topk_method", "hidden_act" }, [_][]const u8{ "sigmoid", "noaux_tc", "silu" }) |key, expected| {
        if (obj.get(key)) |v| {
            if (v != .string or !std.mem.eql(u8, v.string, expected)) return error.UnsupportedMimoV2Config;
        }
    }
    if (obj.get("n_shared_experts")) |v| {
        if (v != .null and (v != .integer or v.integer != 0)) return error.UnsupportedMimoV2Config;
    }
    if (c.num_experts == 0 or c.num_experts_per_tok == 0 or c.num_experts_per_tok > c.num_experts or
        c.moe_intermediate_size == 0 or c.moe_n_group == 0 or c.num_experts % c.moe_n_group != 0 or
        c.moe_topk_group == 0 or c.moe_topk_group > c.moe_n_group or
        c.num_experts_per_tok > c.num_experts / c.moe_n_group * c.moe_topk_group or
        (c.moe_n_group > 1 and c.num_experts / c.moe_n_group < 2))
        return error.UnsupportedMimoV2Config;
    const freq = obj.get("moe_layer_freq") orelse return error.UnsupportedMimoV2Config;
    c.first_k_dense_replace = try model_discovery.denseMoePrefix(freq, c.num_hidden_layers);
    // A pack stores its packed trunk linears (docs/pack-format.md); the engine
    // quantizes nothing at load, so a config asking it to is refused.
    if (obj.get("trunk_quant") != null) return error.UnsupportedMimoV2Config;
}

fn jsonFloat(v: std.json.Value) f32 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| @floatCast(f),
        else => 0.0,
    };
}

// Config JSON is untrusted: a field of the wrong JSON type or out of its range is
// `error.InvalidConfigField`, never a bare union read or `@intCast` (illegal in ReleaseFast).

/// A field, with JSON null read as absent: an optional left at its default.
fn cfgField(obj: std.json.ObjectMap, key: []const u8) ?std.json.Value {
    const v = obj.get(key) orelse return null;
    return if (v == .null) null else v;
}

fn cfgInt(comptime T: type, v: std.json.Value) !T {
    if (v != .integer) return error.InvalidConfigField;
    return std.math.cast(T, v.integer) orelse error.InvalidConfigField;
}

fn cfgF32(v: std.json.Value) !f32 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| @floatCast(f),
        else => error.InvalidConfigField,
    };
}

fn cfgObject(v: std.json.Value) !std.json.ObjectMap {
    return if (v == .object) v.object else error.InvalidConfigField;
}

fn cfgString(v: std.json.Value) ![]const u8 {
    return if (v == .string) v.string else error.InvalidConfigField;
}

/// Holds all loaded weights as mlx arrays, keyed by name.
pub const Weights = struct {
    map: std.StringHashMap(mlx.mlx_array),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) Weights {
        return .{
            .map = std.StringHashMap(mlx.mlx_array).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Weights) void {
        var it = self.map.iterator();
        while (it.next()) |entry| {
            _ = mlx.mlx_array_free(entry.value_ptr.*);
            self.allocator.free(entry.key_ptr.*);
        }
        self.map.deinit();
    }

    pub fn get(self: *const Weights, name: []const u8) ?mlx.mlx_array {
        return self.map.get(name);
    }

    /// Remove lazy tensors before the transformer binds or evaluates them.
    pub fn dropPrefix(self: *Weights, prefix: []const u8) !void {
        var keys: std.ArrayList([]const u8) = .empty;
        defer keys.deinit(self.allocator);
        var it = self.map.keyIterator();
        while (it.next()) |key| if (std.mem.startsWith(u8, key.*, prefix)) {
            try keys.append(self.allocator, key.*);
        };
        for (keys.items) |key| {
            const removed = self.map.fetchRemove(key).?;
            _ = mlx.mlx_array_free(removed.value);
            self.allocator.free(removed.key);
        }
    }

    pub fn count(self: *const Weights) u32 {
        return @intCast(self.map.count());
    }
};

/// The generic nestings a text trunk ships under: flat, mlx-community's
/// re-nest, and meta's VL original (Muse-Glimmer). `parseConfigFromJson`
/// picks from config KEYS; this probe corrects it from the checkpoint.
const FLAT_PREFIX = "model";
const NESTED_PREFIX = "language_model.model";
const VL_NESTED_PREFIX = "model.language_model";

fn hasWeightsUnder(weights: *const Weights, prefix: []const u8) bool {
    var it = weights.map.keyIterator();
    while (it.next()) |k| {
        const key = k.*;
        if (key.len > prefix.len and key[prefix.len] == '.' and std.mem.startsWith(u8, key, prefix)) return true;
    }
    return false;
}

/// Re-point `config.weight_prefix` at the nesting the CHECKPOINT actually uses.
///
/// Which of the two a converter emits is not reliably declared in config.json,
/// so `parseConfigFromJson` guesses from `text_config` presence — wrong for any
/// checkpoint that nests without declaring one (mlx-community LFM2.5-2.6B:
/// `Lfm2ForCausalLM`, an EMPTY `vision_config`, every weight under
/// `language_model.model.*`; the guess picked `model` and the load died on
/// `MISSING WEIGHT: model.embed_tokens.weight`). The class has now shipped in
/// both directions, so the weights get the last word.
///
/// Conservative by construction: only the generic spellings participate
/// (never an arch with its own — `backbone`, `model.llm`, `""`), and a swap
/// happens only when the configured one holds NOTHING, so every checkpoint
/// that already loaded binds byte-identically. Scan order puts the most
/// specific spelling first: a `model.language_model.*` checkpoint also
/// satisfies the bare "model" probe.
pub fn resolveWeightPrefix(config: *ModelConfig, weights: *const Weights) void {
    const candidates = [_][]const u8{ NESTED_PREFIX, VL_NESTED_PREFIX, FLAT_PREFIX };
    var known = false;
    for (candidates) |p| {
        if (std.mem.eql(u8, config.weight_prefix, p)) known = true;
    }
    if (!known) return;

    if (hasWeightsUnder(weights, config.weight_prefix)) return;
    for (candidates) |p| {
        if (std.mem.eql(u8, config.weight_prefix, p)) continue;
        if (!hasWeightsUnder(weights, p)) continue;
        log.info("weight prefix: config implies \"{s}\", checkpoint uses \"{s}\" — using the checkpoint's\n", .{ config.weight_prefix, p });
        config.weight_prefix = p;
        return;
    }
}

pub fn streamingDropsWeightKey(key: []const u8) bool {
    return std.mem.startsWith(u8, key, "language_model.mtp.");
}

pub const StreamingLoad = struct { layout: expert_quant.Layout, keep_mtp: bool = false };

/// The resident key a streamed load keeps, or null. A kept MTP head loads its own routed
/// experts: the stream engine serves only the trunk's MoE layers.
pub fn streamedResidentKey(load: StreamingLoad, buf: []u8, key: []const u8) ?[]const u8 {
    if (load.keep_mtp and std.mem.startsWith(u8, key, "language_model.mtp.") and expert_quant.isRoutedExpertKey(load.layout, key)) return key;
    return qwen4StreamingWeightKey(load.layout, buf, key);
}

pub fn qwen4StreamingWeightKey(layout: expert_quant.Layout, buf: []u8, key: []const u8) ?[]const u8 {
    if (expert_quant.isRoutedExpertKey(layout, key)) return null;
    if (layout == .mxfp4_split or layout == .mxfp4_individual) {
        if (std.mem.startsWith(u8, key, "mtp.") or std.mem.startsWith(u8, key, "model.mtp.")) return null;
        return key;
    }
    if (layout == .quantized_split or layout == .exl3_k4) return key;
    const trunk_prefix = "model.language_model.";
    if (std.mem.indexOf(u8, key, ".ple.ple_embedding.ngram_embedding.shard_") != null) return null;
    if (std.mem.startsWith(u8, key, trunk_prefix)) {
        return std.fmt.bufPrint(buf, "language_model.model.{s}", .{key[trunk_prefix.len..]}) catch null;
    }
    if (std.mem.startsWith(u8, key, "mtp.")) {
        return std.fmt.bufPrint(buf, "language_model.mtp.{s}", .{key[4..]}) catch null;
    }
    if (std.mem.eql(u8, key, "lm_head.weight")) return "language_model.lm_head.weight";
    return key;
}

/// `vision`: the tower's bytes, kept apart from `trunk` where a streamed load decides on it.
pub const ResidentSplit = struct { trunk: u64, mtp: u64, vision: u64 = 0 };

/// Resident Sushi EXL3 packs have a measured load envelope and exact component bills.
pub fn usesSushiQuantMemoryBill(config: *const ModelConfig) bool {
    return config.expert_layout == .exl3_k4 and !config.expert_streaming and
        (std.mem.eql(u8, config.model_type, "qwen4_exp") or config.isMimo());
}

/// Flash-Next's resident tensor payloads, with the loader's vision filter and optional native head.
pub fn qwenResidentWeightBytes(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, vision: bool, mtp_on: bool) !ResidentSplit {
    const split = try indexedResidentSplit(io, allocator, model_dir, null, mtp_on);
    return .{ .trunk = split.trunk +| (if (vision) split.vision else 0), .mtp = split.mtp };
}

/// What a streamed load keeps resident under `config.expert_layout`: everything but the routed banks,
/// with the tower apart.
pub fn streamingResidentSplit(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, config: *const ModelConfig) !ResidentSplit {
    if (config.isGlm5())
        return @import("glm5_diagnostic.zig").streamedSplit(io, allocator, model_dir, config.num_hidden_layers);
    if (config.usesMimoSourceTrunk()) {
        var streamed = config.*;
        streamed.expert_streaming = true;
        const mimo_source = @import("mimo_source.zig");
        return .{
            .trunk = try mimo_source.residentBytesWithConfig(io, allocator, model_dir, &streamed),
            .mtp = 0,
            .vision = try mimo_source.visionResidentBytes(io, allocator, model_dir),
        };
    }
    return indexedResidentSplit(io, allocator, model_dir, config.expert_layout, true);
}

fn indexedResidentSplit(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, streaming: ?expert_quant.Layout, mtp_on: bool) !ResidentSplit {
    var dir = try std.Io.Dir.openDirAbsolute(io, model_dir, .{ .iterate = true });
    defer dir.close(io);
    var referenced = model_discovery.indexShardSet(io, dir) orelse return error.InvalidSafetensorsIndex;
    defer model_discovery.freeShardSet(&referenced);
    var owners = indexWeightMap(io, allocator, dir);
    defer if (owners) |*o| o.deinit();
    var total: u64 = 0;
    var mtp: u64 = 0;
    var tower: u64 = 0;
    var found: usize = 0;
    var iterator = dir.iterate();
    while (try iterator.next(io)) |entry| {
        if (entry.kind != .file and entry.kind != .sym_link) continue;
        if (!std.mem.endsWith(u8, entry.name, ".safetensors") or !referenced.contains(entry.name)) continue;
        const file = try dir.openFile(io, entry.name, .{});
        defer file.close(io);
        const stat = try file.stat(io);
        var read_buffer: [8192]u8 = undefined;
        var reader = file.reader(io, &read_buffer);
        const header_len = try reader.interface.takeInt(u64, .little);
        if (header_len == 0 or header_len > 128 * 1024 * 1024 or header_len > stat.size -| 8) return error.InvalidSafetensorsHeader;
        const header = try allocator.alloc(u8, @intCast(header_len));
        defer allocator.free(header);
        try reader.interface.readSliceAll(header);
        const parsed = std.json.parseFromSlice(std.json.Value, allocator, header, .{}) catch return error.InvalidSafetensorsHeader;
        defer parsed.deinit();
        if (parsed.value != .object) return error.InvalidSafetensorsHeader;
        var tensor_iterator = parsed.value.object.iterator();
        while (tensor_iterator.next()) |tensor| {
            if (std.mem.eql(u8, tensor.key_ptr.*, "__metadata__")) continue;
            var key_buf: [512]u8 = undefined;
            if (owners) |o| if (o.value.object.get("weight_map").?.object.get(tensor.key_ptr.*)) |owner| {
                if (owner == .string and !std.mem.eql(u8, owner.string, entry.name) and referenced.contains(owner.string)) continue;
            };
            const canonical = if (streaming) |layout|
                streamedResidentKey(.{ .layout = layout, .keep_mtp = true }, &key_buf, tensor.key_ptr.*) orelse continue
            else
                tensor.key_ptr.*;
            if (!shouldKeepWeightKey(canonical, true)) continue;
            const is_vision = !shouldKeepWeightKey(canonical, false);
            const is_mtp = std.mem.startsWith(u8, canonical, "language_model.mtp.");
            if (is_mtp and !mtp_on) continue;
            if (tensor.value_ptr.* != .object) return error.InvalidSafetensorsHeader;
            const offsets = tensor.value_ptr.object.get("data_offsets") orelse return error.InvalidSafetensorsHeader;
            if (offsets != .array or offsets.array.items.len != 2 or offsets.array.items[0] != .integer or offsets.array.items[1] != .integer) return error.InvalidSafetensorsHeader;
            const start = offsets.array.items[0].integer;
            const end = offsets.array.items[1].integer;
            if (start < 0 or end < start or @as(u64, @intCast(end)) > stat.size -| header_len -| 8) return error.InvalidSafetensorsHeader;
            const size: u64 = @intCast(end - start);
            const bucket = if (is_mtp) &mtp else if (is_vision) &tower else &total;
            bucket.* = std.math.add(u64, bucket.*, size) catch return error.InvalidSafetensorsHeader;
            found += 1;
        }
    }
    if (found == 0) return error.NoWeightFiles;
    return .{ .trunk = total, .mtp = mtp, .vision = tower };
}

/// Load all safetensors files from model_dir.
/// When `load_vision` is true, vision_tower and multi_modal_projector weights are included.
pub fn loadWeights(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !Weights {
    return loadWeightsOpt(io, allocator, model_dir, false);
}

fn logMimoSourceLoad(config: *const ModelConfig, vision: bool) void {
    log.info("[mimo-source] loading original shards: {s} experts, FP8 trunk in source bytes{s}\n", .{
        if (config.expert_streaming) "SSD-streamed" else if (config.expert_layout == .exl3_k4) "resident EXL3" else "native MXFP4",
        if (vision) ", bf16 vision tower" else "",
    });
}

/// MiMo's trunk is FP8 on disk under either routed-expert layout, so both take
/// the source loader; an EXL3 pack's routed banks come resident beside it.
pub fn loadWeightsMimoSource(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, vision: bool) !Weights {
    const mimo_source = @import("mimo_source.zig");
    var config = try parseConfig(io, allocator, model_dir);
    defer config.deinit(allocator);
    logMimoSourceLoad(&config, vision);
    var weights = try mimo_source.loadWeights(io, allocator, model_dir, &config);
    errdefer weights.deinit();
    if (vision) try mimo_source.loadVisionWeightsInto(&weights, io, allocator, model_dir);
    return weights;
}

/// Resident bytes of a MiMo pack the source loader prepares: the trunk as
/// served (FP8 codes + scale grids, bf16 rest) plus any resident routed banks,
/// plus the vision tower as stored when it is loaded.
pub fn mimoSourceResidentBytes(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, vision: bool) !u64 {
    const mimo_source = @import("mimo_source.zig");
    const trunk = try mimo_source.residentBytes(io, allocator, model_dir);
    return if (vision) trunk + try mimo_source.visionResidentBytes(io, allocator, model_dir) else trunk;
}

/// Resident bytes of a MiMo checkpoint's MTP heads as `mimo_source` uploads them.
pub fn mimoMtpResidentBytes(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !u64 {
    return @import("mimo_source.zig").mtpResidentBytes(io, allocator, model_dir);
}

/// The architectures this build serves. Every other `model_type` is refused
/// by name at the loader, so the inherited forwards behind it are unreachable.
pub const served_model_types = [_][]const u8{ "qwen4_exp", "mimo_v2", "glm5_next" };

/// The engine's thinking-effort vocabulary. Each served arch accepts a subset
/// (`effortArms`); a word outside it is refused, never rounded (`selectEffort`
/// has the one exception).
pub const Effort = enum { off, on, minimal, low, medium, high, xhigh, max };

/// Explicit --think default for serving, CLI and later model loads.
pub var think_effort_flag: ?Effort = null;

/// The arm `--think` selects on this arch; null leaves the model on its own default.
fn thinkFlagArm(model_type: []const u8) ?EffortArm {
    const e = think_effort_flag orelse return null;
    return armForWord(effortArms(model_type) orelse return null, @tagName(e));
}

/// The budget `--think` selects on this arch; fallback leaves the server on its own default.
pub fn thinkFlagBudget(model_type: []const u8, fallback: i32) i32 {
    if (thinkFlagArm(model_type)) |arm| {
        if (arm.budget) |b| return b;
    }
    return fallback;
}

pub fn defaultEffortWord(config: *const ModelConfig) ?[]const u8 {
    if (thinkFlagArm(config.model_type)) |arm| return @tagName(arm.effort);
    return if (config.isGlm5()) "high" else null;
}

/// The word a request naming no effort runs at (`/v1/models` `default_reasoning_effort`).
pub fn defaultReasoningEffort(config: *const ModelConfig) ?[]const u8 {
    const arms = effortArms(config.model_type) orelse return null;
    if (defaultEffortWord(config)) |w| return w;
    if (!config.defaultEnableThinking()) return "off";
    // Silence renders the cheapest thinking level (`chat.qwen38EffortFor`).
    for (arms) |a| if (a.effort != .off) return @tagName(a.effort);
    return null;
}

/// One accepted effort word on one arch. `budget` is the decode-time thinking
/// cap in tokens; null = `--reasoning-budget` (unlimited by default). The word
/// itself reaches a template that reads it (qwen4_exp: low|medium|xhigh).
pub const EffortArm = struct { effort: Effort, budget: ?i32 = null };

const qwen4_exp_efforts = [_]EffortArm{
    .{ .effort = .off },
    .{ .effort = .low, .budget = 2048 },
    .{ .effort = .medium, .budget = 8192 },
    .{ .effort = .xhigh },
};

// MiMo's template has only on/off; do not advertise artificial effort levels.
const mimo_v2_efforts = [_]EffortArm{
    .{ .effort = .off },
    .{ .effort = .on },
};

const glm5_efforts = [_]EffortArm{
    .{ .effort = .low },
    .{ .effort = .high },
    .{ .effort = .max },
};

/// null = an inherited arch: its effort words keep `responses.effortBudget`.
pub fn effortArms(model_type: []const u8) ?[]const EffortArm {
    if (std.mem.eql(u8, model_type, "glm5_next")) return &glm5_efforts;
    if (std.mem.eql(u8, model_type, "qwen4_exp")) return &qwen4_exp_efforts;
    if (std.mem.eql(u8, model_type, "mimo_v2")) return &mimo_v2_efforts;
    return null;
}

/// `none` is an alias of off. No served table lists `minimal`; inherited arches
/// keep its legacy budget (`responses.effortBudget`).
pub fn parseEffort(word: []const u8) ?Effort {
    if (std.mem.eql(u8, word, "none")) return .off;
    return std.meta.stringToEnum(Effort, word);
}

pub fn findEffortArm(arms: []const EffortArm, effort: Effort) ?EffortArm {
    for (arms) |a| if (a.effort == effort) return a;
    return null;
}

/// The effort `e` selects on a model offering `offered`; null = refused. On an
/// on/off model (mimo_v2) every thinking word selects `on`.
pub fn selectEffort(offered: []const Effort, e: Effort) ?Effort {
    if (std.mem.indexOfScalar(Effort, offered, e) != null) return e;
    return if (e != .off and std.mem.indexOfScalar(Effort, offered, .on) != null) .on else null;
}

/// The arm a client's effort word selects (`selectEffort`); null = refused.
pub fn armForWord(arms: []const EffortArm, word: []const u8) ?EffortArm {
    var offered: [std.enums.values(Effort).len]Effort = undefined;
    for (arms, 0..) |a, i| offered[i] = a.effort;
    const e = selectEffort(offered[0..arms.len], parseEffort(word) orelse return null) orelse return null;
    return findEffortArm(arms, e);
}

pub fn isServedArch(model_type: []const u8) bool {
    for (served_model_types) |t| {
        if (std.mem.eql(u8, model_type, t)) return true;
    }
    return false;
}

/// The ONE weight-loader decision. A second construction site is how a
/// subcommand ends up forwarding through a model the server never serves —
/// a MiMo pack read without its source trunk binds the raw FP8 fused QKV.
pub fn loadWeightsForConfig(
    io: std.Io,
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    config: *const ModelConfig,
    load_vision: bool,
) !Weights {
    if (!isServedArch(config.model_type)) {
        log.err("model_type \"{s}\" is not served by this build (qwen4_exp, mimo_v2, glm5_next only)\n", .{config.model_type});
        return error.ArchitectureUnsupported;
    }
    if (config.expert_layout == .exl3_k4) try @import("mimo_source.zig").validateExl3Pack(io, allocator, model_dir, config);
    if (config.isGlm5()) {
        const glm = @import("glm5_diagnostic.zig");
        return glm.loadWeightsBoundedWithVision(io, allocator, model_dir, mlx.gpuStream(), config.expert_streaming, std.math.maxInt(u64), load_vision and config.glm5_vision);
    }
    if (config.expert_streaming and config.usesMimoSourceTrunk()) {
        const vision = load_vision and config.mimo_vision;
        logMimoSourceLoad(config, vision);
        var weights = try @import("mimo_source.zig").loadWeights(io, allocator, model_dir, config);
        errdefer weights.deinit();
        if (vision) try @import("mimo_source.zig").loadVisionWeightsInto(&weights, io, allocator, model_dir);
        return weights;
    }
    if (config.expert_streaming) return loadWeightsStreaming(io, allocator, model_dir, config.expert_layout, load_vision, config.stream_mtp_head);
    if (config.usesMimoSourceTrunk()) return loadWeightsMimoSource(io, allocator, model_dir, load_vision and config.mimo_vision);
    var weights = if (load_vision)
        try loadWeightsWithVision(io, allocator, model_dir)
    else
        try loadWeights(io, allocator, model_dir);
    errdefer weights.deinit();
    if (std.mem.eql(u8, config.model_type, "qwen4_exp") and config.mtp_override == false)
        try weights.dropPrefix("language_model.mtp.");
    return weights;
}

pub fn loadWeightsStreaming(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, layout: expert_quant.Layout, vision: bool, keep_mtp: bool) !Weights {
    if (layout == .mxfp4_individual) return loadWeightsMimoSource(io, allocator, model_dir, vision);
    var dir = try std.Io.Dir.openDirAbsolute(io, model_dir, .{ .iterate = true });
    defer dir.close(io);
    return loadWeightsFromOpenDirMode(io, allocator, dir, model_dir, vision, .{ .layout = layout, .keep_mtp = keep_mtp });
}

/// Load ONE safetensors file (absolute path) into a Weights map — for
/// sidecar files that live beside the trunk shards (e.g. a root-level
/// `mtp.safetensors`), where a directory scan would sweep in the trunk.
pub fn loadWeightsSingleFile(allocator: std.mem.Allocator, abs_path: []const u8) !Weights {
    var weights = Weights.init(allocator);
    errdefer weights.deinit();

    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);

    const pathz = try allocator.dupeSentinel(u8, abs_path, 0);
    defer allocator.free(pathz);
    try loadSafetensorsFile(allocator, &weights, pathz, s, false);

    if (weights.count() == 0) {
        log.err("no usable weights loaded from {s} — corrupt or empty safetensors file?\n", .{abs_path});
        return error.NoWeightFiles;
    }
    return weights;
}

pub fn loadWeightsWithVision(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !Weights {
    return loadWeightsOpt(io, allocator, model_dir, true);
}

fn loadWeightsOpt(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, load_vision: bool) !Weights {
    var dir = try std.Io.Dir.openDirAbsolute(io, model_dir, .{ .iterate = true });
    defer dir.close(io);
    return loadWeightsFromOpenDir(io, allocator, dir, model_dir, load_vision);
}

/// Load every `*.safetensors` in an already-open `dir` into a Weights map.
/// `model_dir` is the on-disk path string, used both to build the per-file
/// absolute path for `loadSafetensorsFile` and to phrase the error message.
/// Split out of `loadWeightsOpt` so the incomplete-checkpoint guard below is
/// unit-testable against a `tmpDir` (mirrors `model_discovery.discoverModelsInDir`).
fn loadWeightsFromOpenDir(io: std.Io, allocator: std.mem.Allocator, dir: std.Io.Dir, model_dir: []const u8, load_vision: bool) !Weights {
    return loadWeightsFromOpenDirMode(io, allocator, dir, model_dir, load_vision, null);
}

fn loadWeightsFromOpenDirMode(io: std.Io, allocator: std.mem.Allocator, dir: std.Io.Dir, model_dir: []const u8, load_vision: bool, streaming: ?StreamingLoad) !Weights {
    var weights = Weights.init(allocator);
    errdefer weights.deinit();

    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);

    // The index names the shards; anything else is dead weight (issue #274:
    // a pack shipped two shards no weight_map entry names — RAM for nothing)
    // or a foreign file whose parse failure would be an uncatchable MLX abort.
    var referenced = model_discovery.indexShardSet(io, dir);
    defer if (referenced) |*r| model_discovery.freeShardSet(r);
    // A live index also names each tensor's shard: a shard may still carry a tensor the index
    // assigns elsewhere (a MiMo pack's source shard keeps the bf16 o_proj beside the affine one).
    const owners: ?std.json.Parsed(std.json.Value) = if (referenced != null) indexWeightMap(io, allocator, dir) else null;
    defer if (owners) |o| o.deinit();
    // Only an owner that will load can claim its tensor: a partly stale index names shards that are gone.
    var present: std.StringHashMapUnmanaged(void) = .empty;
    defer present.deinit(allocator);
    if (owners != null) {
        var names = referenced.?.keyIterator();
        while (names.next()) |n| {
            _ = dir.statFile(io, n.*, .{}) catch continue;
            try present.put(allocator, n.*, {});
        }
    }

    var reader = expert_io.ParallelReader.init(allocator, expert_io.ParallelReader.default_chunk, expert_io.ParallelReader.default_workers);
    defer reader.deinit();
    var file_count: u32 = 0;
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        // Accept regular files AND symlinks: HuggingFace cache snapshots store
        // every weight file as a symlink into ../../blobs/<hash>. open(2)
        // resolves the link at the OS level, so a symlinked *.safetensors loads fine.
        if (entry.kind != .file and entry.kind != .sym_link) continue;
        if (!std.mem.endsWith(u8, entry.name, ".safetensors")) continue;
        if (referenced) |r| if (!r.contains(entry.name)) {
            log.warn("skipping {s}: not named by model.safetensors.index.json\n", .{entry.name});
            continue;
        };

        const path_slice = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ model_dir, entry.name });
        defer allocator.free(path_slice);
        const path = try allocator.dupeSentinel(u8, path_slice, 0);
        defer allocator.free(path);

        log.info("Loading {s}...\n", .{entry.name});
        const shard: ?ShardOwners = if (owners) |o| .{ .map = o.value.object.get("weight_map").?.object, .present = &present, .file = entry.name } else null;
        try loadSafetensorsFileMode(allocator, &reader, &weights, path, s, load_vision, streaming, shard);
        file_count += 1;
    }

    // Incomplete-checkpoint guard. A dir with config/tokenizer but no (or no
    // usable) *.safetensors is the classic interrupted-download shape: the
    // small files land first, the multi-GB weight shards never finalize. Before
    // this guard the loader returned an empty map and the caller crashed with a
    // misleading `MISSING WEIGHT: <prefix>.embed_tokens.weight` (the first
    // weight looked up) + `unreachable`, pointing at the model arch instead of
    // the download. Fail here with an actionable message, mirroring the
    // tokenizer path's "incomplete download?" hint (see main.zig).
    if (weights.count() == 0) {
        log.err("no usable weights loaded from {s} ({d} *.safetensors file(s) found) — the checkpoint looks like an incomplete download (config/tokenizer present, weight shards missing). Re-download the model (e.g. `sushi pull <model>`) or delete the dir and re-fetch.\n", .{ model_dir, file_count });
        return error.NoWeightFiles;
    }

    log.info("Loaded {d} weights from {d} file(s)\n", .{ weights.count(), file_count });
    reportF16Narrowing();
    return weights;
}

/// Whether a just-loaded f16 tensor must be narrowed to the engine's bf16
/// activation dtype.
///
/// Two shapes qualify, for the same underlying reason — an f16 value that
/// meets a bf16 activation promotes the RESULT to f32:
///
///   - Quant SIDE tensors (scales/biases), which can be 2-D so they are keyed
///     on the suffix. f16 side tensors force gather_qmm/qmatmul onto a ~4x
///     slower mixed-dtype path (hy_v3 2-bit live, 2026-07-14: 0.70 vs 0.18 ms
///     per 8-expert gather — 1.2 tok/s on the 295B instead of ~15+).
///   - ANY 1-D f16 tensor: a per-channel table (norm weight, bias, A_log,
///     dt_bias) that is multiplied or added straight into the residual. Leave
///     one f16 and the residual turns f32 at the first layer and STAYS f32,
///     so every later weight read is upcast — the Laguna YaRN-mscale class,
///     one level up. Measured on prism-ml/Ternary-Bonsai-27B-mlx-2bit (the
///     only f16 checkpoint on hand, qwen3_5 GDN hybrid): 27.99 -> 23.88
///     ms/forward, 14.7%, three paired boots with cooldown.
///
/// Plain multi-dimensional WEIGHTS keep their dtype. They are matmul
/// OPERANDS, and MLX selects its kernel off that dtype, so narrowing one is a
/// kernel-selection change rather than a promotion fix — measured as a wash
/// here (23.15 vs 23.47 ms, inside boot-to-boot drift), so the minimal rule
/// is the one that ships.
///
/// The cast node stays lazy, so the load-time batch eval materializes bf16
/// directly. Delta from the 3 dropped mantissa bits: cos 0.99999994 — far
/// below any quant noise floor.
pub fn narrowsLoadedF16(key: []const u8, ndim: usize, dtype: mlx.mlx_dtype) bool {
    if (dtype != .float16) return false;
    if (std.mem.endsWith(u8, key, ".scales") or std.mem.endsWith(u8, key, ".biases")) return true;
    return ndim == 1;
}

/// Kill switch for the 1-D arm (`SUSHI_F16_NARROW_1D=0`). A load-time
/// dtype normalization is invisible once the model is up, so a one-boot A/B
/// switch is the only way to attribute a future f16-checkpoint regression to
/// it. The side-tensor arm predates this and is not switchable.
var narrow_1d_env: ?bool = null;
fn narrow1dEnabled() bool {
    if (narrow_1d_env) |v| return v;
    const on = blk: {
        const raw = std.c.getenv("SUSHI_F16_NARROW_1D") orelse break :blk true;
        break :blk !std.mem.eql(u8, std.mem.sliceTo(raw, 0), "0");
    };
    narrow_1d_env = on;
    return on;
}

/// Count of 1-D f16 tables narrowed this load — reported once per model so a
/// declined normalization is nameable from the log instead of silently
/// reading as "this checkpoint just isn't f16".
var narrowed_1d: usize = 0;

pub fn reportF16Narrowing() void {
    if (narrowed_1d == 0) return;
    log.info("[dtype] narrowed {d} 1-D f16 tables to bf16 (SUSHI_F16_NARROW_1D=0 disables)\n", .{narrowed_1d});
    narrowed_1d = 0;
}

pub fn loadSafetensorsFile(
    allocator: std.mem.Allocator,
    weights: *Weights,
    path: [*:0]const u8,
    s: mlx.mlx_stream,
    load_vision: bool,
) !void {
    var reader = expert_io.ParallelReader.init(allocator, expert_io.ParallelReader.default_chunk, expert_io.ParallelReader.default_workers);
    defer reader.deinit();
    return loadSafetensorsFileMode(allocator, &reader, weights, path, s, load_vision, null, null);
}

fn qwen4NormFold(key: []const u8) bool {
    const suffixes = [_][]const u8{
        "hc_norm.weight",
        "q_norm.weight",
        "k_norm.weight",
        "q_layernorm.weight",
        "k_layernorm.weight",
        "ple.norm_key.weight",
        "ple.norm_query.weight",
        "ple.norm_conv.weight",
        "pre_fc_norm_embedding.weight",
        "pre_fc_norm_hidden.weight",
    };
    for (suffixes) |suffix| if (std.mem.endsWith(u8, key, suffix)) return true;
    return false;
}

fn qwen4FoldNorm(value: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const one_f32 = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one_f32);
    var one = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(one);
    try mlx.check(mlx.mlx_astype(&one, one_f32, mlx.mlx_array_dtype(value), s));
    var result = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&result, value, one, s));
    return result;
}

fn qwen4ConvShape(shape: []const c_int) !void {
    if (shape.len != 3 or shape[1] != 1) return error.InvalidQwen4ConvShape;
}

fn qwen4TransposeConv(value: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    try qwen4ConvShape(mlx.getShape(value));
    const axes = [_]c_int{ 0, 2, 1 };
    var view = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(view);
    try mlx.check(mlx.mlx_transpose_axes(&view, value, &axes, 3, s));
    var result = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_contiguous(&result, view, false, s));
    return result;
}

test "qwen4 convolution transform rejects a non-unit channel axis" {
    try std.testing.expectError(error.InvalidQwen4ConvShape, qwen4ConvShape(&.{ 4, 2, 8 }));
    try qwen4ConvShape(&.{ 4, 1, 8 });
}

test "streaming config owns and releases its expert source path" {
    var config = ModelConfig{};
    config.expert_source_dir = try std.testing.allocator.dupe(u8, "/tmp/qwen-stream-source");
    config.deinit(std.testing.allocator);
    try std.testing.expect(config.expert_source_dir == null);
}

fn qwen4SplitGateUp(value: mlx.mlx_array, s: mlx.mlx_stream) ![2]mlx.mlx_array {
    const shape = mlx.getShape(value);
    if (shape.len != 3 or shape[1] == 0 or @mod(shape[1], 2) != 0) return error.BadPackedGateUpShape;
    const half = @divExact(shape[1], 2);
    const strides = [_]c_int{ 1, 1, 1 };
    var result = [2]mlx.mlx_array{ mlx.mlx_array_new(), mlx.mlx_array_new() };
    errdefer {
        for (result) |arr| _ = mlx.mlx_array_free(arr);
    }
    for (0..2) |i| {
        const start = [_]c_int{ 0, @intCast(i * @as(usize, @intCast(half))), 0 };
        const stop = [_]c_int{ shape[0], @intCast((i + 1) * @as(usize, @intCast(half))), shape[2] };
        var view = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(view);
        try mlx.check(mlx.mlx_slice(&view, value, &start, 3, &stop, 3, &strides, 3, s));
        try mlx.check(mlx.mlx_contiguous(&result[i], view, false, s));
    }
    return result;
}

fn putLoadedWeight(allocator: std.mem.Allocator, weights: *Weights, key: []const u8, value: mlx.mlx_array) !void {
    const gop = try weights.map.getOrPut(key);
    if (gop.found_existing) {
        // A tensor two unindexed files carry: the later one stands, the earlier is released.
        _ = mlx.mlx_array_free(gop.value_ptr.*);
        gop.value_ptr.* = value;
        return;
    }
    gop.key_ptr.* = allocator.dupe(u8, key) catch |err| {
        weights.map.removeByPtr(gop.key_ptr);
        return err;
    };
    gop.value_ptr.* = value;
}

/// The shard being read, the index's tensor-to-shard map and the indexed shards on disk.
const ShardOwners = struct { map: std.json.ObjectMap, present: *const std.StringHashMapUnmanaged(void), file: []const u8 };

/// `model.safetensors.index.json` parsed with its own copies of every string, or null when it
/// has no `weight_map` object.
fn indexWeightMap(io: std.Io, allocator: std.mem.Allocator, dir: std.Io.Dir) ?std.json.Parsed(std.json.Value) {
    const raw = dir.readFileAlloc(io, "model.safetensors.index.json", allocator, .limited(16 * 1024 * 1024)) catch return null;
    defer allocator.free(raw);
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, raw, .{ .allocate = .alloc_always }) catch return null;
    if (parsed.value == .object) if (parsed.value.object.get("weight_map")) |wm| if (wm == .object) return parsed;
    parsed.deinit();
    return null;
}

/// MLX's safetensors dtype table (`dtype_from_safetensor_str`): FP8 codes load as `uint8`.
fn safetensorsDtype(name: []const u8) ?struct { mlx.mlx_dtype, u64 } {
    const table = [_]struct { []const u8, mlx.mlx_dtype, u64 }{
        .{ "F32", .float32, 4 },  .{ "F16", .float16, 2 }, .{ "BF16", .bfloat16, 2 }, .{ "I64", .int64, 8 },
        .{ "I32", .int32, 4 },    .{ "I16", .int16, 2 },   .{ "I8", .int8, 1 },       .{ "U64", .uint64, 8 },
        .{ "U32", .uint32, 4 },   .{ "U16", .uint16, 2 },  .{ "U8", .uint8, 1 },      .{ "BOOL", .bool_, 1 },
        .{ "C64", .complex64, 8 }, .{ "F8_E4M3", .uint8, 1 }, .{ "F8_E8M0", .uint8, 1 },
    };
    for (table) |row| if (std.mem.eql(u8, row[0], name)) return .{ row[1], row[2] };
    return null;
}

pub const TensorArray = struct {
    array: mlx.mlx_array,
    /// The array's own bytes; valid as long as `array`.
    bytes: []const u8,
    /// The mapping MLX imported as the array's buffer; null when it copied instead.
    mapped: ?[*]const u8 = null,
};

/// From this size a tensor is read into its own mapping that MLX imports without a copy;
/// below it, the scratch copy into a fresh MLX buffer costs less than a page per tensor.
const import_min_bytes: usize = 1 << 20;

/// `len` bytes at `offset` as a `shape`/`dtype` array, read around the file cache with the read
/// and copy of its chunks spread over several threads (`expert_io.ParallelReader`). A large tensor
/// lands at offset 0 of an anonymous mapping MLX imports as its buffer: a view at an offset
/// inside a buffer is not safe in this engine. Caller frees `array`.
pub fn readTensorArray(reader: *expert_io.ParallelReader, fd: std.c.fd_t, offset: u64, len: usize, shape: []const c_int, dtype: mlx.mlx_dtype) !TensorArray {
    if (len < import_min_bytes) {
        const host = try reader.allocator.alloc(u8, @max(len, 1));
        defer reader.allocator.free(host);
        try reader.fill(fd, host[0..len], offset);
        const array = mlx.mlx_array_new_data(host.ptr, shape.ptr, @intCast(shape.len), dtype);
        if (array.ctx == null) return error.OutOfMemory;
        errdefer _ = mlx.mlx_array_free(array);
        try mlx.check(mlx.mlx_array_eval(array));
        return .{ .array = array, .bytes = if (len == 0) &.{} else mlx.mlx_array_data_uint8(array).?[0..len] };
    }
    const size: usize = @intCast(expert_io.pageRoundUp(len));
    const mapped = std.c.mmap(null, size, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    if (mapped == std.c.MAP_FAILED) return error.OutOfMemory;
    const region = std.heap.c_allocator.create(MappedRegion) catch {
        _ = std.c.munmap(@alignCast(mapped), size);
        return error.OutOfMemory;
    };
    region.* = .{ .ptr = @ptrCast(@alignCast(mapped)), .len = size };
    reader.fill(fd, region.ptr[0..len], offset) catch |err| {
        unmapRegion(region);
        return err;
    };
    // When Metal refuses the mapping MLX copies it and runs `unmapRegion` before returning:
    // `region` is not touched after this call.
    const ptr = region.ptr;
    const array = mlx.mlx_array_new_data_managed_payload(ptr, shape.ptr, @intCast(shape.len), dtype, region, unmapRegion);
    if (array.ctx == null) {
        // MLX throws from this constructor only before adopting the mapping (its copy
        // branch runs the deleter last), so this is the region's only release.
        unmapRegion(region);
        return error.OutOfMemory;
    }
    errdefer _ = mlx.mlx_array_free(array);
    try mlx.check(mlx.mlx_array_eval(array));
    const data = mlx.mlx_array_data_uint8(array).?;
    return .{ .array = array, .bytes = data[0..len], .mapped = if (data == ptr) ptr else null };
}

const MappedRegion = struct { ptr: [*]align(std.heap.page_size_min) u8, len: usize };

fn unmapRegion(payload: ?*anyopaque) callconv(.c) void {
    const region: *MappedRegion = @ptrCast(@alignCast(payload.?));
    _ = std.c.munmap(region.ptr, region.len);
    std.heap.c_allocator.destroy(region);
}

/// One header entry's payload as an MLX array. Caller frees.
fn readSafetensor(reader: *expert_io.ParallelReader, fd: std.c.fd_t, header: *const expert_io.SafetensorsHeader, meta: std.json.Value) !mlx.mlx_array {
    if (meta != .object) return error.InvalidSafetensorsTensor;
    const dtype_value = meta.object.get("dtype") orelse return error.InvalidSafetensorsTensor;
    const dims = meta.object.get("shape") orelse return error.InvalidSafetensorsTensor;
    const offsets = meta.object.get("data_offsets") orelse return error.InvalidSafetensorsTensor;
    if (dtype_value != .string or dims != .array or dims.array.items.len > 16 or offsets != .array or offsets.array.items.len != 2) return error.InvalidSafetensorsTensor;
    const dtype, const itemsize = safetensorsDtype(dtype_value.string) orelse return error.UnsupportedSafetensorsDtype;
    var shape: [16]c_int = undefined;
    var bytes: u64 = itemsize;
    for (dims.array.items, 0..) |dim, i| {
        if (dim != .integer or dim.integer < 0 or dim.integer > std.math.maxInt(c_int)) return error.InvalidSafetensorsTensor;
        shape[i] = @intCast(dim.integer);
        bytes = try std.math.mul(u64, bytes, @intCast(dim.integer));
    }
    const lo = offsets.array.items[0];
    const hi = offsets.array.items[1];
    if (lo != .integer or hi != .integer or lo.integer < 0 or hi.integer < lo.integer or @as(u64, @intCast(hi.integer - lo.integer)) != bytes) return error.InvalidSafetensorsTensor;
    return (try readTensorArray(reader, fd, header.data_start + @as(u64, @intCast(lo.integer)), @intCast(bytes), shape[0..dims.array.items.len], dtype)).array;
}

fn loadSafetensorsFileMode(
    allocator: std.mem.Allocator,
    reader: *expert_io.ParallelReader,
    weights: *Weights,
    path: [*:0]const u8,
    s: mlx.mlx_stream,
    load_vision: bool,
    streaming: ?StreamingLoad,
    shard: ?ShardOwners,
) !void {
    // Only the dense HF layout needs the converter's work at load time: the
    // fused bank split, the delta norms and the conv transpose. An MLX pack
    // ships every resident tensor in its serving layout already.
    const fused_streaming = streaming != null and streaming.?.layout == .bf16_fused;
    // Read here rather than through MLX's loader, whose descriptor cannot take F_NOCACHE:
    // filtered tensors are never read, kept ones bypass the file cache (`readTensorArray`).
    const fd = try expert_io.openHinted(std.mem.span(path), .{ .readahead_off = false });
    defer _ = std.c.close(fd);
    var header = try expert_io.SafetensorsHeader.read(allocator, fd);
    defer header.deinit();
    var tensors = header.parsed.value.object.iterator();
    while (tensors.next()) |entry| {
        const key_str_raw = entry.key_ptr.*;
        if (std.mem.eql(u8, key_str_raw, "__metadata__")) continue;
        if (shard) |sh| if (sh.map.get(key_str_raw)) |owner| {
            if (owner == .string and !std.mem.eql(u8, owner.string, sh.file) and sh.present.contains(owner.string)) continue;
        };
        var key_buf: [512]u8 = undefined;
        const key_str = if (streaming) |load|
            streamedResidentKey(load, &key_buf, key_str_raw) orelse continue
        else
            key_str_raw;

        if (!shouldKeepWeightKey(key_str, load_vision) or (streaming != null and !streaming.?.keep_mtp and streamingDropsWeightKey(key_str))) continue;
        const value = try readSafetensor(reader, fd, &header, entry.value_ptr.*);

        // Read the shape BEFORE the cast frees `value` — a freed handle's
        // ndim is a use-after-free, not a zero.
        const ndim = mlx.mlx_array_ndim(value);
        var final_value = value;
        errdefer if (final_value.ctx != null) {
            _ = mlx.mlx_array_free(final_value);
        };
        if (narrowsLoadedF16(key_str, ndim, mlx.mlx_array_dtype(value)) and
            (ndim != 1 or narrow1dEnabled()))
        {
            var cast = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(cast);
            try mlx.check(mlx.mlx_astype(&cast, value, .bfloat16, s));
            _ = mlx.mlx_array_free(value);
            final_value = cast;
            if (ndim == 1) narrowed_1d += 1;
        }

        if (fused_streaming and std.mem.endsWith(u8, key_str, ".mlp.experts.gate_up_proj")) {
            var pair = try qwen4SplitGateUp(final_value, s);
            errdefer {
                if (pair[0].ctx != null) _ = mlx.mlx_array_free(pair[0]);
                if (pair[1].ctx != null) _ = mlx.mlx_array_free(pair[1]);
            }
            _ = mlx.mlx_array_free(final_value);
            final_value = .{ .ctx = null };
            var gate_key_buf: [512]u8 = undefined;
            var up_key_buf: [512]u8 = undefined;
            const prefix_len = key_str.len - "experts.gate_up_proj".len;
            const gate_key = std.fmt.bufPrint(&gate_key_buf, "{s}switch_mlp.gate_proj.weight", .{key_str[0..prefix_len]}) catch return error.NameTooLong;
            const up_key = std.fmt.bufPrint(&up_key_buf, "{s}switch_mlp.up_proj.weight", .{key_str[0..prefix_len]}) catch return error.NameTooLong;
            try putLoadedWeight(allocator, weights, gate_key, pair[0]);
            pair[0] = .{ .ctx = null };
            try putLoadedWeight(allocator, weights, up_key, pair[1]);
            pair[1] = .{ .ctx = null };
            continue;
        }
        if (fused_streaming and std.mem.endsWith(u8, key_str, ".mlp.experts.down_proj")) {
            var down_key_buf: [512]u8 = undefined;
            const prefix_len = key_str.len - "experts.down_proj".len;
            const down_key = std.fmt.bufPrint(&down_key_buf, "{s}switch_mlp.down_proj.weight", .{key_str[0..prefix_len]}) catch return error.NameTooLong;
            try putLoadedWeight(allocator, weights, down_key, final_value);
            continue;
        }
        if (fused_streaming and qwen4NormFold(key_str)) {
            const folded = try qwen4FoldNorm(final_value, s);
            _ = mlx.mlx_array_free(final_value);
            final_value = folded;
        }
        if (fused_streaming and std.mem.endsWith(u8, key_str, "conv1d.weight") and mlx.mlx_array_ndim(final_value) == 3) {
            const transposed = try qwen4TransposeConv(final_value, s);
            _ = mlx.mlx_array_free(final_value);
            final_value = transposed;
        }
        try putLoadedWeight(allocator, weights, key_str, final_value);
    }
}

/// True if the safetensors weight `key` should be retained for the text
/// forward pass. Audio is always dropped; vision is dropped unless
/// `load_vision` is set. MTP-style head tensors (`*.mtp.*`) on Qwen3.5/3.6
/// checkpoints are kept (the binder ignores them, but the loader doesn't
/// need to know that).
pub fn shouldKeepWeightKey(key: []const u8, load_vision: bool) bool {
    // Gemma 4 12B `gemma4_unified` is encoder-free: it ships a tiny vision
    // patch embedder (`vision_embedder.*` + `embed_vision.*`) and a raw-waveform
    // audio projection (`embed_audio.*`) instead of the SigLIP vision tower and
    // conformer audio tower of earlier Gemma 4 variants. Those embedders are
    // wired in src/vision.zig (UnifiedEmbedder), so keep them under the same
    // `load_vision` gate as the SigLIP weights (`--no-vision` → text only).
    const is_vision = std.mem.startsWith(u8, key, "vision_tower.") or
        std.mem.startsWith(u8, key, "embed_vision.") or
        std.mem.startsWith(u8, key, "vision_embedder.") or
        std.mem.startsWith(u8, key, "embed_audio.") or
        std.mem.startsWith(u8, key, "multi_modal_projector.") or
        std.mem.startsWith(u8, key, "language_model.multi_modal_projector.");
    // The heavy SigLIP-era conformer audio tower is still not wired — drop it.
    const is_audio_tower = std.mem.startsWith(u8, key, "audio_tower.") or
        std.mem.startsWith(u8, key, "language_model.audio_multi_modal_projector.");
    if (is_audio_tower) return false;
    // DiffusionGemma nests its (not-yet-wired) vision tower under
    // model.encoder.* — always drop it so a 26B text load doesn't carry
    // ~1 GB of dead tower weights. The encoder LAYER SCALARS
    // (model.encoder.language_model.layers.N.layer_scalar) must survive:
    // they're the only untied encoder text params and the causal encoder
    // pass multiplies by them instead of the decoder's layer_scalar.
    if (std.mem.startsWith(u8, key, "model.encoder.vision_tower.") or
        std.mem.startsWith(u8, key, "model.encoder.embed_vision.")) return false;
    // Muse-Glimmer nests its tower/adapter/projection under "model."; the
    // mlx-community re-nest drops that prefix (its bare "vision_tower." already
    // rides the is_vision gate above). Both follow --no-vision.
    if (!load_vision and (std.mem.startsWith(u8, key, "model.vision_tower.") or
        std.mem.startsWith(u8, key, "model.vision_adapter.") or
        std.mem.startsWith(u8, key, "model.vision_projection.") or
        std.mem.startsWith(u8, key, "vision_adapter.") or
        std.mem.startsWith(u8, key, "vision_projection.") or
        // avlp12's Qwen3.8 "Alis" packs spell the Qwen3-VL tower
        // `model.visual.` (pure rename of `vision_tower.`).
        std.mem.startsWith(u8, key, "model.visual."))) return false;
    if (is_vision and !load_vision) return false;
    return true;
}

// ── Tests ──

const testing = std.testing;
const expectError = @import("test_expect.zig").expectError;

test "ModelConfig defaults" {
    const config = ModelConfig{};
    try testing.expectEqual(@as(u32, 0), config.num_eos_tokens);
    try testing.expectEqual(@as(u32, 0), config.max_position_embeddings);
    try testing.expectEqual(@as(u32, 0), config.quant_bits); // 0 = dense bf16 (no "quantization" key)
    try testing.expectEqual(@as(u32, 64), config.quant_group_size);
    try testing.expect(!config.tie_word_embeddings);
}

test "loadWeights casts f16 quant scales/biases to bf16 (mixed-dtype qmm slow-path class)" {
    // hy_v3 2-bit (ox-ox) ships F16 scales/biases beside bf16 activations —
    // MLX's gather_qmm/qmatmul take a ~4x slower mixed-dtype path (measured
    // 2026-07-14: 0.70 vs 0.18 ms per 8-expert gather; 1.2 tok/s on the 295B
    // instead of ~15+). The loader must cast quant SIDE tensors to bf16 once;
    // weights and non-quant tensors keep their dtype. Dequant delta from the
    // 3 dropped mantissa bits: cos 0.99999994 — under the 2-bit noise floor.
    const allocator = testing.allocator;
    const s = mlx.gpuStream();

    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();
    var path_buf: [512]u8 = undefined;
    const root_len = try tmp_dir.dir.realPath(io, &path_buf);
    const dir_path = path_buf[0..root_len];
    const st_path = try std.fmt.allocPrintSentinel(allocator, "{s}/model.safetensors", .{dir_path}, 0);
    defer allocator.free(st_path);

    // Build a tiny map: an f16 "scales", an f16 "biases", an f16 plain weight
    // (must NOT be cast), and a bf16 scales (no-op).
    {
        const map = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(map);
        const meta = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(meta);

        const shape = [_]c_int{ 4, 4 };
        const data: [16]f32 = @splat(0.5);
        const f32_arr = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
        defer _ = mlx.mlx_array_free(f32_arr);
        var f16_arr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(f16_arr);
        try mlx.check(mlx.mlx_astype(&f16_arr, f32_arr, .float16, s));
        var bf16_arr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bf16_arr);
        try mlx.check(mlx.mlx_astype(&bf16_arr, f32_arr, .bfloat16, s));
        try mlx.check(mlx.mlx_array_eval(f16_arr));
        try mlx.check(mlx.mlx_array_eval(bf16_arr));

        _ = mlx.mlx_map_string_to_array_insert(map, "model.layers.0.mlp.gate_proj.scales", f16_arr);
        _ = mlx.mlx_map_string_to_array_insert(map, "model.layers.0.mlp.gate_proj.biases", f16_arr);
        _ = mlx.mlx_map_string_to_array_insert(map, "model.layers.0.mlp.up_proj.weight", f16_arr);
        _ = mlx.mlx_map_string_to_array_insert(map, "model.layers.0.mlp.down_proj.scales", bf16_arr);
        try mlx.check(mlx.mlx_save_safetensors(st_path.ptr, map, meta));
    }

    var weights = try loadWeights(io, allocator, dir_path);
    defer weights.deinit();

    try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(weights.get("model.layers.0.mlp.gate_proj.scales").?));
    try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(weights.get("model.layers.0.mlp.gate_proj.biases").?));
    // A plain WEIGHT stays f16 (dense-f16 tables are legitimate — only the
    // quant side tensors force the mixed-dtype qmm path).
    try testing.expectEqual(mlx.mlx_dtype.float16, mlx.mlx_array_dtype(weights.get("model.layers.0.mlp.up_proj.weight").?));
    try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(weights.get("model.layers.0.mlp.down_proj.scales").?));
}

test "loadWeights on a weightless dir (incomplete download) errors clearly, not empty map" {
    // Reproduces the live misdiagnosis: an interrupted `hf download`/`sushi
    // pull` lands config + tokenizer but never finalizes the *.safetensors
    // weight shards. Before the guard, loadWeights returned an empty map and
    // the caller crashed with a misleading "MISSING WEIGHT:
    // model.embed_tokens.weight" (the first weight looked up) + `unreachable`,
    // pointing at the model arch instead of the incomplete checkpoint.
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();

    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"mistral\"}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "tokenizer.json", .data = "{}" });
    // The index file names the shards but is NOT itself a weight file — it must
    // not be mistaken for one (it ends in .json, not .safetensors).
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{}" });

    try std.testing.expectError(
        error.NoWeightFiles,
        loadWeightsFromOpenDir(io, allocator, tmp.dir, "/incomplete-model", false),
    );
}

test "loadWeights reads only the shards the index names (issue #274)" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    // One real (hand-built) shard + one garbage file that mlx would abort on.
    const hdr = "{\"w\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}";
    var st: [8 + hdr.len + 4]u8 = undefined;
    std.mem.writeInt(u64, st[0..8], hdr.len, .little);
    @memcpy(st[8 .. 8 + hdr.len], hdr);
    @memset(st[8 + hdr.len ..], 0);
    try tmp.dir.writeFile(io, .{ .sub_path = "model-00001.safetensors", .data = &st });
    try tmp.dir.writeFile(io, .{ .sub_path = "stray.safetensors", .data = "not a safetensors file" });
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"w\":\"model-00001.safetensors\"}}" });

    var cwd_buf: [std.fs.max_path_bytes]u8 = undefined;
    const cwd_ptr = std.c.getcwd(&cwd_buf, cwd_buf.len) orelse return error.NoCwd;
    const cwd = std.mem.span(@as([*:0]const u8, @ptrCast(cwd_ptr)));
    const dir = try std.fmt.allocPrint(allocator, "{s}/.zig-cache/tmp/{s}", .{ cwd, tmp.sub_path });
    defer allocator.free(dir);
    var w = try loadWeightsFromOpenDir(io, allocator, tmp.dir, dir, false);
    defer w.deinit();
    try std.testing.expectEqual(@as(u32, 1), w.count());
}

test "loadWeights takes a tensor two shards carry from the shard the index names, and frees the other" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    // A MiMo pack's source shard keeps its bf16 o_proj beside the affine one the index names.
    for ([_]struct { name: []const u8, hdr: []const u8 }{
        .{ .name = "model-a.safetensors", .hdr = "{\"w\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[4,8]}}" },
        .{ .name = "model-b.safetensors", .hdr = "{\"w\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}" },
    }) |f| {
        const st = try allocator.alloc(u8, 8 + f.hdr.len + 8);
        defer allocator.free(st);
        std.mem.writeInt(u64, st[0..8], f.hdr.len, .little);
        @memcpy(st[8 .. 8 + f.hdr.len], f.hdr);
        @memset(st[8 + f.hdr.len ..], 0);
        try tmp.dir.writeFile(io, .{ .sub_path = f.name, .data = st });
    }
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"w\":\"model-b.safetensors\",\"x\":\"model-a.safetensors\"}}" });

    var cwd_buf: [std.fs.max_path_bytes]u8 = undefined;
    const cwd_ptr = std.c.getcwd(&cwd_buf, cwd_buf.len) orelse return error.NoCwd;
    const cwd = std.mem.span(@as([*:0]const u8, @ptrCast(cwd_ptr)));
    const dir = try std.fmt.allocPrint(allocator, "{s}/.zig-cache/tmp/{s}", .{ cwd, tmp.sub_path });
    defer allocator.free(dir);
    var w = try loadWeightsFromOpenDir(io, allocator, tmp.dir, dir, false);
    defer w.deinit();
    try std.testing.expectEqual(@as(u32, 2), w.count());
    try std.testing.expectEqualSlices(c_int, &.{2}, mlx.getShape(w.get("w").?));
}

test "loadWeights keeps a tensor whose index owner is not on disk (a partly stale index)" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const hdr = "{\"w\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[4,8]}}";
    var st: [8 + hdr.len + 8]u8 = undefined;
    std.mem.writeInt(u64, st[0..8], hdr.len, .little);
    @memcpy(st[8 .. 8 + hdr.len], hdr);
    @memset(st[8 + hdr.len ..], 0);
    try tmp.dir.writeFile(io, .{ .sub_path = "model-a.safetensors", .data = &st });
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"w\":\"model-gone.safetensors\",\"x\":\"model-a.safetensors\"}}" });

    var cwd_buf: [std.fs.max_path_bytes]u8 = undefined;
    const cwd_ptr = std.c.getcwd(&cwd_buf, cwd_buf.len) orelse return error.NoCwd;
    const cwd = std.mem.span(@as([*:0]const u8, @ptrCast(cwd_ptr)));
    const dir = try std.fmt.allocPrint(allocator, "{s}/.zig-cache/tmp/{s}", .{ cwd, tmp.sub_path });
    defer allocator.free(dir);
    var w = try loadWeightsFromOpenDir(io, allocator, tmp.dir, dir, false);
    defer w.deinit();
    try std.testing.expectEqual(@as(u32, 2), w.count());
}

test "loadWeights ignores an index that names no shard on disk (re-sharded upload, stale index)" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const hdr = "{\"w\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}";
    var st: [8 + hdr.len + 4]u8 = undefined;
    std.mem.writeInt(u64, st[0..8], hdr.len, .little);
    @memcpy(st[8 .. 8 + hdr.len], hdr);
    @memset(st[8 + hdr.len ..], 0);
    try tmp.dir.writeFile(io, .{ .sub_path = "model-00001-of-00002.safetensors", .data = &st });
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"w\":\"model-00001-of-00005.safetensors\"}}" });

    var cwd_buf: [std.fs.max_path_bytes]u8 = undefined;
    const cwd_ptr = std.c.getcwd(&cwd_buf, cwd_buf.len) orelse return error.NoCwd;
    const cwd = std.mem.span(@as([*:0]const u8, @ptrCast(cwd_ptr)));
    const dir = try std.fmt.allocPrint(allocator, "{s}/.zig-cache/tmp/{s}", .{ cwd, tmp.sub_path });
    defer allocator.free(dir);
    var w = try loadWeightsFromOpenDir(io, allocator, tmp.dir, dir, false);
    defer w.deinit();
    try std.testing.expectEqual(@as(u32, 1), w.count());
}

test "resolveWeightPrefix: the CHECKPOINT decides the nesting, not the config keys" {
    // mlx-community/LFM2.5-2.6B-{8bit,nvfp4} declare `Lfm2ForCausalLM` with NO
    // text_config (just an empty `vision_config`), yet ship every weight under
    // `language_model.model.*`. The config-key guess picked "model" and the
    // load died on `MISSING WEIGHT: model.embed_tokens.weight` (live
    // 2026-08-04). The same class shipped in the opposite direction before, so
    // the probe corrects either way.
    const allocator = testing.allocator;
    const put = struct {
        fn add(w: *Weights, alloc: std.mem.Allocator, key: []const u8) !void {
            const k = try alloc.dupe(u8, key);
            try w.map.put(k, mlx.mlx_array_new());
        }
    }.add;

    // Nested checkpoint, flat guess → re-pointed (the LFM2.5 crash).
    {
        var w = Weights.init(allocator);
        defer w.deinit();
        try put(&w, allocator, "language_model.model.embed_tokens.weight");
        try put(&w, allocator, "language_model.model.layers.0.self_attn.q_proj.weight");
        var config = ModelConfig{ .model_type = "lfm2", .weight_prefix = "model" };
        resolveWeightPrefix(&config, &w);
        try testing.expectEqualStrings("language_model.model", config.weight_prefix);
    }
    // Flat checkpoint, nested guess → re-pointed the other way.
    {
        var w = Weights.init(allocator);
        defer w.deinit();
        try put(&w, allocator, "model.embed_tokens.weight");
        var config = ModelConfig{ .model_type = "lfm2", .weight_prefix = "language_model.model" };
        resolveWeightPrefix(&config, &w);
        try testing.expectEqualStrings("model", config.weight_prefix);
    }
    // Both present (a real VL checkpoint) → the configured prefix stands, so
    // nothing that loads today can be re-pointed by this probe.
    {
        var w = Weights.init(allocator);
        defer w.deinit();
        try put(&w, allocator, "model.embed_tokens.weight");
        try put(&w, allocator, "language_model.model.embed_tokens.weight");
        var config = ModelConfig{ .model_type = "lfm2", .weight_prefix = "language_model.model" };
        resolveWeightPrefix(&config, &w);
        try testing.expectEqualStrings("language_model.model", config.weight_prefix);
    }
    // An arch with its OWN prefix is never touched, even when it holds nothing
    // (a genuinely broken checkpoint must stay a clear MISSING WEIGHT).
    {
        var w = Weights.init(allocator);
        defer w.deinit();
        try put(&w, allocator, "language_model.model.embed_tokens.weight");
        var config = ModelConfig{ .model_type = "nemotron_h", .weight_prefix = "backbone" };
        resolveWeightPrefix(&config, &w);
        try testing.expectEqualStrings("backbone", config.weight_prefix);
    }
    // A prefix that is a strict PREFIX of the key's first segment must not
    // count as a hit ("model" vs "model_extra.*").
    {
        var w = Weights.init(allocator);
        defer w.deinit();
        try put(&w, allocator, "model_extra.embed_tokens.weight");
        try put(&w, allocator, "language_model.model.embed_tokens.weight");
        var config = ModelConfig{ .model_type = "lfm2", .weight_prefix = "model" };
        resolveWeightPrefix(&config, &w);
        try testing.expectEqualStrings("language_model.model", config.weight_prefix);
    }
    // mlx-community/Muse-Glimmer-30B-4bit (live 2026-08-11): meta's config
    // keeps text_config, so the guess is the VL-original "model.language_model"
    // — but mlx_lm convert re-nests every text weight under
    // "language_model.model.*". The third spelling joins the probe.
    {
        var w = Weights.init(allocator);
        defer w.deinit();
        try put(&w, allocator, "language_model.model.embed_tokens.weight");
        try put(&w, allocator, "language_model.lm_head.weight");
        try put(&w, allocator, "vision_tower.layers.0.norm1.weight");
        var config = ModelConfig{ .model_type = "muse_glimmer", .weight_prefix = "model.language_model" };
        resolveWeightPrefix(&config, &w);
        try testing.expectEqualStrings("language_model.model", config.weight_prefix);
    }
    // Our own mirror layout (meta-original nesting) stays put.
    {
        var w = Weights.init(allocator);
        defer w.deinit();
        try put(&w, allocator, "model.language_model.embed_tokens.weight");
        var config = ModelConfig{ .model_type = "muse_glimmer", .weight_prefix = "model.language_model" };
        resolveWeightPrefix(&config, &w);
        try testing.expectEqualStrings("model.language_model", config.weight_prefix);
    }
    // Ordering: a "model.language_model.*" checkpoint ALSO matches the bare
    // "model" probe (the '.' check passes at "model.language_model"), so the
    // most specific spelling must win the scan.
    {
        var w = Weights.init(allocator);
        defer w.deinit();
        try put(&w, allocator, "model.language_model.embed_tokens.weight");
        var config = ModelConfig{ .model_type = "muse_glimmer", .weight_prefix = "language_model.model" };
        resolveWeightPrefix(&config, &w);
        try testing.expectEqualStrings("model.language_model", config.weight_prefix);
    }
}

test "applyFamilySamplingDefaults: qwen family gets top_k 20 / top_p 0.95 when the checkpoint ships no generation_config" {
    // Live soak capture 2026-07-13 (stamsam Qwen3.6-35B distill, served to pi):
    // the community re-quant ships NO generation_config.json, so omitted-field
    // sampling resolution bottomed out at the hardcoded 1.0/1.0/off — full
    // untruncated tail sampling on a 4-bit MoE. A 16K-token turn degenerated
    // into word salad (zero tool calls) and burned the client's whole output
    // budget. Qwen's own recommendation for the family is top_k 20/top_p 0.95;
    // fill exactly the truncation knobs, never temperature.
    var qwen = ModelConfig{ .model_type = "qwen3_5_moe" };
    qwen.applyFamilySamplingDefaults();
    try testing.expectEqual(@as(?u32, 20), qwen.gen_top_k);
    try testing.expectEqual(@as(?f32, 0.95), qwen.gen_top_p);
    try testing.expectEqual(@as(?f32, null), qwen.gen_temperature);

    // Families without a documented upstream recommendation stay null.
    var llama = ModelConfig{ .model_type = "llama" };
    llama.applyFamilySamplingDefaults();
    try testing.expectEqual(@as(?u32, null), llama.gen_top_k);
    try testing.expectEqual(@as(?f32, null), llama.gen_top_p);

}

test "applyFamilySamplingDefaults never overrides explicit generation_config values" {
    // The checkpoint's own generation_config.json (parsed before this runs)
    // always wins — the family fallback fills NULLS only.
    var config = ModelConfig{ .model_type = "qwen3_5_moe" };
    config.gen_top_k = 40;
    config.gen_top_p = 0.8;
    config.gen_temperature = 0.6;
    config.applyFamilySamplingDefaults();
    try testing.expectEqual(@as(?u32, 40), config.gen_top_k);
    try testing.expectEqual(@as(?f32, 0.8), config.gen_top_p);
    try testing.expectEqual(@as(?f32, 0.6), config.gen_temperature);
    // Partial file: only the missing knob is filled.
    var partial = ModelConfig{ .model_type = "qwen3" };
    partial.gen_top_p = 0.8;
    partial.applyFamilySamplingDefaults();
    try testing.expectEqual(@as(?u32, 20), partial.gen_top_k);
    try testing.expectEqual(@as(?f32, 0.8), partial.gen_top_p);
}

test "defaultEnableThinking: opt-in per arch, and every existing arch stays off" {
    // No prior arch opts in — including the families whose templates merely
    // MENTION enable_thinking, which is not evidence of a thinking-on default.
    for ([_][]const u8{ "qwen3", "qwen3_5_moe", "llama" }) |t| {
        const c = ModelConfig{ .model_type = t };
        try testing.expect(!c.defaultEnableThinking());
    }
}

test "defaultEnableThinking: mimo_v2 thinks by default, with and without tools" {
    const mimo = ModelConfig{ .model_type = "mimo_v2" };
    try testing.expect(mimo.defaultEnableThinking());
}

test "effortArms: every engine word on each served arch" {
    const Budget = struct { on: bool, cap: ?i32 };
    const Want = struct { word: []const u8, qwen: ?Budget, mimo: ?Budget };
    const off: Budget = .{ .on = false, .cap = null };
    const cases = [_]Want{
        .{ .word = "off", .qwen = off, .mimo = off },
        .{ .word = "on", .qwen = null, .mimo = .{ .on = true, .cap = null } },
        .{ .word = "none", .qwen = off, .mimo = off },
        .{ .word = "low", .qwen = .{ .on = true, .cap = 2048 }, .mimo = null },
        .{ .word = "medium", .qwen = .{ .on = true, .cap = 8192 }, .mimo = null },
        .{ .word = "high", .qwen = null, .mimo = null },
        .{ .word = "xhigh", .qwen = .{ .on = true, .cap = null }, .mimo = null },
        .{ .word = "max", .qwen = null, .mimo = null },
    };
    for (cases) |c| {
        const e = parseEffort(c.word).?;
        for ([_]struct { arch: []const u8, want: ?Budget }{ .{ .arch = "qwen4_exp", .want = c.qwen }, .{ .arch = "mimo_v2", .want = c.mimo } }) |a| {
            const got = findEffortArm(effortArms(a.arch).?, e);
            if (a.want) |w| {
                try testing.expectEqual(w.on, got.?.effort != .off);
                try testing.expectEqual(w.cap, got.?.budget);
            } else try testing.expect(got == null);
        }
    }
    // `minimal` parses but no served table lists it.
    for (served_model_types) |t| try testing.expect(findEffortArm(effortArms(t).?, parseEffort("minimal").?) == null);
    try testing.expect(parseEffort("ultra") == null);
    // Inherited arches keep the legacy ladder.
    try testing.expect(effortArms("qwen3_5_moe") == null);
}

test "defaultEnableThinking: the checkpoint's own generation_config default outranks the arch allowlist" {
    // A thinking model whose arch is not on the allowlist still thinks when
    // its own generation_config declares it — this is the case a silent
    // request used to lose (3 tokens and no reasoning where the same weights
    // reason for ~1000 tokens under a runner that obeys the template).
    var on = ModelConfig{ .model_type = "qwen3" };
    on.gen_enable_thinking = true;
    try testing.expect(on.defaultEnableThinking());
    // And a checkpoint that declares thinking OFF turns an opted-in arch off.
    var off = ModelConfig{ .model_type = "bailing_hybrid" };
    off.gen_enable_thinking = false;
    try testing.expect(!off.defaultEnableThinking());
}

test "parseGenerationDefaultsFromJson: reads default_chat_template_kwargs.enable_thinking" {
    const on = parseGenerationDefaultsFromJson(
        "{\"default_chat_template_kwargs\": {\"enable_thinking\": true}}",
    );
    try testing.expectEqual(@as(?bool, true), on.enable_thinking);
    const off = parseGenerationDefaultsFromJson(
        "{\"default_chat_template_kwargs\": {\"enable_thinking\": false}}",
    );
    try testing.expectEqual(@as(?bool, false), off.enable_thinking);
    // Absent, wrong shape, or a non-bool value: null, arch default stays.
    try testing.expectEqual(@as(?bool, null), parseGenerationDefaultsFromJson("{\"top_k\": 20}").enable_thinking);
    try testing.expectEqual(@as(?bool, null), parseGenerationDefaultsFromJson(
        "{\"default_chat_template_kwargs\": \"on\"}",
    ).enable_thinking);
    try testing.expectEqual(@as(?bool, null), parseGenerationDefaultsFromJson(
        "{\"default_chat_template_kwargs\": {\"enable_thinking\": \"yes\"}}",
    ).enable_thinking);
}

test "parseGenerationDefaultsFromJson: eos_token_id list merges additively into the stop set" {
    const gd = parseGenerationDefaultsFromJson("{\"eos_token_id\": [1, 250019]}");
    try testing.expectEqual(@as(usize, 2), gd.num_eos);
    var config = ModelConfig{};
    config.addEosToken(1);
    config.mergeEosTokens(gd.eos_token_ids[0..gd.num_eos]);
    try testing.expectEqual(@as(u32, 2), config.num_eos_tokens);
    try testing.expect(config.isEosToken(250019));
    const scalar = parseGenerationDefaultsFromJson("{\"eos_token_id\": 7}");
    try testing.expectEqual(@as(u32, 7), scalar.eos_token_ids[0]);
    try testing.expectEqual(@as(usize, 0), parseGenerationDefaultsFromJson("{\"eos_token_id\": \"x\"}").num_eos);
}

test "ModelConfig addEosToken" {
    var config = ModelConfig{};
    config.addEosToken(1);
    config.addEosToken(106);
    try testing.expectEqual(@as(u32, 2), config.num_eos_tokens);
    try testing.expect(config.isEosToken(1));
    try testing.expect(config.isEosToken(106));
    try testing.expect(!config.isEosToken(42));
}

test "ModelConfig addEosToken max capacity" {
    var config = ModelConfig{};
    // Fill all 8 slots
    for (0..8) |i| {
        config.addEosToken(@intCast(i + 100));
    }
    try testing.expectEqual(@as(u32, 8), config.num_eos_tokens);
    // 9th should be silently dropped
    config.addEosToken(999);
    try testing.expectEqual(@as(u32, 8), config.num_eos_tokens);
    try testing.expect(!config.isEosToken(999));
}

test "EOS merge: chat-terminator added even when config already provided an eos" {
    // Regression for the Qwen2.5-Coder-7B leak: its config.json sets
    // eos_token_id=<|endoftext|> (151643), but its chat template ends turns
    // with <|im_end|> (151645). The load path (main.zig / scheduler doLoad)
    // must ALWAYS merge the tokenizer's chat-terminator EOS — additively and
    // dedup-guarded — not only when config provided none; otherwise <|im_end|>
    // is never a stop token and leaks into the output (broke structured JSON /
    // tool calling). This pins the merge invariant those call sites implement.
    var config = ModelConfig{};
    config.addEosToken(151643); // from config.json eos_token_id
    try testing.expectEqual(@as(u32, 1), config.num_eos_tokens);

    // Merge step the fix performs: add the chat terminator if absent.
    const chat_eos: u32 = 151645; // <|im_end|>, from tokenizer_config eos_token
    if (!config.isEosToken(chat_eos)) config.addEosToken(chat_eos);

    try testing.expect(config.isEosToken(151645)); // now stops on <|im_end|>
    try testing.expect(config.isEosToken(151643)); // original preserved
    try testing.expectEqual(@as(u32, 2), config.num_eos_tokens);

    // Idempotent: re-running the merge must not duplicate.
    if (!config.isEosToken(chat_eos)) config.addEosToken(chat_eos);
    try testing.expectEqual(@as(u32, 2), config.num_eos_tokens);
}

test "ModelConfig eosTokenSlice" {
    var config = ModelConfig{};
    config.addEosToken(10);
    config.addEosToken(20);
    const slice = config.eosTokenSlice();
    try testing.expectEqual(@as(usize, 2), slice.len);
    try testing.expectEqual(@as(u32, 10), slice[0]);
    try testing.expectEqual(@as(u32, 20), slice[1]);
}

test "ModelConfig isGlobalLayer with sliding window" {
    // Gemma 3 convention (HF + mlx-lm): every Nth layer is global, with the
    // pattern anchored at the END of each group — global when
    // `(idx + 1) % pattern == 0`, i.e. layers 5, 11, 17… for pattern 6.
    // The old `% pattern == 0` phase made layer 0 global and layer 5 local —
    // every layer got the wrong RoPE base/scale and attention scope, which
    // surfaced as fluent-but-wrong output (spaced digits, broken arithmetic)
    // on gemma-3-12b. Gemma 4 ships explicit layer_types and never hits this
    // fallback.
    var config = ModelConfig{};
    config.has_sliding_window = true;
    config.sliding_window_pattern = 6;
    try testing.expect(!config.isGlobalLayer(0));
    try testing.expect(!config.isGlobalLayer(1));
    try testing.expect(config.isGlobalLayer(5));
    try testing.expect(!config.isGlobalLayer(6));
    try testing.expect(config.isGlobalLayer(11));
    try testing.expect(!config.isGlobalLayer(12));
}

test "ModelConfig isGlobalLayer without sliding window" {
    var config = ModelConfig{};
    config.has_sliding_window = false;
    // All layers should be global
    try testing.expect(config.isGlobalLayer(0));
    try testing.expect(config.isGlobalLayer(1));
    try testing.expect(config.isGlobalLayer(5));
}

test "ModelConfig isLinearLayer" {
    var config = ModelConfig{};
    config.full_attention_interval = 4;
    // Layer 0: (0+1) % 4 == 1 != 0 → linear
    try testing.expect(config.isLinearLayer(0));
    // Layer 3: (3+1) % 4 == 0 → NOT linear (full attention)
    try testing.expect(!config.isLinearLayer(3));
    // Layer 7: (7+1) % 4 == 0 → NOT linear
    try testing.expect(!config.isLinearLayer(7));
    // Layer 4: (4+1) % 4 == 1 → linear
    try testing.expect(config.isLinearLayer(4));
}

test "linear_attn_tail_from forces full attention past the last whole group" {
    // A layer count that is NOT a multiple of the group size is where the
    // reference's second clause bites: with 40 layers, 40//6*6 = 36, so layers
    // 36..39 are ALL full attention even though (idx+1) % 6 != 0. Dropping the
    // clause would run four layers through the wrong attention type silently.
    var config = ModelConfig{};
    config.num_hidden_layers = 40;
    config.full_attention_interval = 6;
    config.linear_attn_tail_from = 40 / 6 * 6; // 36
    try testing.expect(config.isLinearLayer(34));
    try testing.expect(!config.isLinearLayer(35)); // (35+1) % 6 == 0
    try testing.expect(!config.isLinearLayer(36)); // tail clause
    try testing.expect(!config.isLinearLayer(37));
    try testing.expect(!config.isLinearLayer(38));
    try testing.expect(!config.isLinearLayer(39));
}

test "linear_attn_tail_from is off by default so no existing arch moves" {
    // qwen3_next/lfm2 set full_attention_interval without a tail bound.
    var config = ModelConfig{};
    config.full_attention_interval = 4;
    try testing.expectEqual(@as(u32, 0), config.linear_attn_tail_from);
    try testing.expect(config.isLinearLayer(100));
    try testing.expect(config.isLinearLayer(1000));
}

test "ModelConfig isLinearLayer disabled" {
    var config = ModelConfig{};
    config.full_attention_interval = 0;
    try testing.expect(!config.isLinearLayer(0));
    try testing.expect(!config.isLinearLayer(5));
}

test "ModelConfig isMoe" {
    var config = ModelConfig{};
    try testing.expect(!config.isMoe());
    config.num_experts = 8;
    try testing.expect(config.isMoe());
}

test "jsonFloat converts integer" {
    const val = std.json.Value{ .integer = 42 };
    try testing.expectApproxEqAbs(@as(f32, 42.0), jsonFloat(val), 0.001);
}

test "jsonFloat converts float" {
    const val = std.json.Value{ .float = 3.14 };
    try testing.expectApproxEqAbs(@as(f32, 3.14), jsonFloat(val), 0.01);
}

test "ModelConfig isGlobalLayer with explicit layer_types" {
    var config = ModelConfig{};
    config.has_sliding_window = true;
    config.has_explicit_layer_types = true;
    // Set layer 4 and 9 as global (like Gemma 4 E2B pattern)
    config.layer_is_global[4] = true;
    config.layer_is_global[9] = true;
    try testing.expect(!config.isGlobalLayer(0));
    try testing.expect(!config.isGlobalLayer(3));
    try testing.expect(config.isGlobalLayer(4));
    try testing.expect(!config.isGlobalLayer(5));
    try testing.expect(config.isGlobalLayer(9));
}

test "parseConfigFromJson: qwen4_exp YaRN rope_parameters extends 262144 to 1048576" {
    const c = try parseConfigFromJson(testing.allocator, QWEN4_YARN);
    try testing.expect(c.isQwen4());
    // The scaling is recognised and lands where the engine reads it —
    // transformer.yarnSpec() consumes exactly these fields.
    try testing.expect(c.rope_yarn);
    try testing.expectApproxEqAbs(@as(f32, 4.0), c.yarn_factor, 1e-9);
    try testing.expectEqual(@as(u32, 262_144), c.yarn_orig_max_pos);
    try testing.expectApproxEqAbs(@as(f32, 32.0), c.yarn_beta_fast, 1e-9);
    try testing.expectApproxEqAbs(@as(f32, 1.0), c.yarn_beta_slow, 1e-9);
    try testing.expect(c.yarn_truncate); // HF's default, absent from the block
    // The mscale is COMPUTED (no `attention_factor` in the block, so HF's
    // default applies): 0.1·ln 4 + 1 — the value the extension is calibrated to.
    try testing.expectApproxEqAbs(@as(f32, 1.138629436111989), c.yarn_attention_factor, 1e-6);
    // qwen4_exp has ONE rope for the trunk, so the YaRN table spans exactly the
    // 64 dims attention rotates: `partial_rotary_factor`, NOT laguna's
    // dims and rotate the pass-through slice).
    try testing.expectApproxEqAbs(@as(f32, 0.25), c.yarnPartial(), 1e-9);
    try testing.expectEqual(@as(u32, 64), c.yarnRotaryDim());
    try testing.expectApproxEqAbs(@as(f32, 10_000_000.0), c.rope_theta, 1.0);
    // The 32 frequencies of the scaled table are the 32 halves the interleaved
    // M-RoPE selector splits [11,11,10] across. If they disagreed, half the
    // table would rotate against an axis the position table doesn't have.
    try testing.expectEqual(
        c.mrope_section[0] + c.mrope_section[1] + c.mrope_section[2],
        c.yarnRotaryDim() / 2,
    );
    // The window the server may advertise: original × factor, and the config's
    // own declaration agrees.
    try testing.expectEqual(@as(u32, 1_048_576), c.max_position_embeddings);
    try testing.expectEqual(@as(u32, 1_048_576), c.contextCap());
    // Cost of that window: only the 12 interval-full layers bill KV, so
    // 12 layers × 2 kv heads × (K+V) × 256 dims × 2 bytes = 24 KiB per token.
    try testing.expectEqual(@as(u32, 12), c.attnCacheLayerCount());
    try testing.expectEqual(@as(u64, 24_576), c.kvBytesPerToken());
}

test "ModelConfig layerHeadDim" {
    var config = ModelConfig{};
    config.head_dim = 256;
    config.global_head_dim = 512;
    config.has_sliding_window = true;
    config.has_explicit_layer_types = true;
    config.layer_is_global[4] = true;
    try testing.expectEqual(@as(u32, 256), config.layerHeadDim(0));
    try testing.expectEqual(@as(u32, 512), config.layerHeadDim(4));
}

test "ModelConfig BERT has no sliding window" {
    var config = ModelConfig{};
    config.is_encoder_only = true;
    config.has_sliding_window = false;
    try testing.expect(config.isGlobalLayer(0));
    try testing.expect(config.isGlobalLayer(5));
}

test "shouldKeepWeightKey accepts orphan MTP head weights on Qwen3.5/3.6 checkpoints" {
    // Some Qwen3.5/3.6 checkpoints embed `*.mtp.*` tensors in the MAIN
    // shards (the sidecar-based MTP head in src/mtp.zig loads separately).
    // The safetensors iterator must let them through (they're neither vision
    // nor audio) so the model loads cleanly; the trunk binder ignores them.
    try testing.expect(shouldKeepWeightKey("language_model.model.mtp.0.eh_proj.weight", true));
    try testing.expect(shouldKeepWeightKey("language_model.model.mtp.0.eh_proj.weight", false));
    try testing.expect(shouldKeepWeightKey("model.mtp.0.shared_head.head.weight", false));
}

test "shouldKeepWeightKey filters audio and gated vision weights" {
    // Regression: the existing filter should still reject audio and reject
    // vision when load_vision is false.
    try testing.expect(!shouldKeepWeightKey("audio_tower.encoder.layer.0.weight", true));
    try testing.expect(!shouldKeepWeightKey("vision_tower.encoder.layer.0.weight", false));
    // qwen4_exp / Alis packs spell the Qwen3-VL tower `model.visual.` — --no-vision drops it too.
    try testing.expect(!shouldKeepWeightKey("model.visual.blocks.0.attn.qkv.weight", false));
    try testing.expect(shouldKeepWeightKey("model.visual.blocks.0.attn.qkv.weight", true));
    try testing.expect(shouldKeepWeightKey("vision_tower.encoder.layer.0.weight", true));
    try testing.expect(shouldKeepWeightKey("language_model.model.layers.0.self_attn.q_proj.weight", false));
}

test "shouldKeepWeightKey keeps Gemma 4 12B unified embedder weights when vision enabled" {
    // gemma4_unified is encoder-free: vision_embedder.* (patch embedder),
    // embed_vision.* and embed_audio.* (raw projections) ARE wired in
    // src/vision.zig (UnifiedEmbedder), so they must be kept under load_vision.
    // The heavy SigLIP-era conformer audio_tower.* stays dropped.
    try testing.expect(shouldKeepWeightKey("vision_embedder.patch_dense.weight", true));
    try testing.expect(shouldKeepWeightKey("embed_vision.embedding_projection.weight", true));
    try testing.expect(shouldKeepWeightKey("embed_audio.embedding_projection.weight", true));
    // Gated off by --no-vision.
    try testing.expect(!shouldKeepWeightKey("vision_embedder.patch_dense.weight", false));
    try testing.expect(!shouldKeepWeightKey("embed_audio.embedding_projection.weight", false));
    // The conformer audio tower is never wired — always dropped.
    try testing.expect(!shouldKeepWeightKey("audio_tower.encoder.layer.0.weight", true));
}

test "shouldKeepWeightKey gates Muse-Glimmer vision on load_vision in both nestings" {
    // Ours nests the tower under `model.`; mlx-community re-nests it bare.
    // Both spellings ride the same gate --no-vision flips.
    try testing.expect(!shouldKeepWeightKey("model.vision_tower.layers.0.norm1.weight", false));
    try testing.expect(!shouldKeepWeightKey("model.vision_adapter.fc1.weight", false));
    try testing.expect(!shouldKeepWeightKey("model.vision_projection.weight", false));
    try testing.expect(!shouldKeepWeightKey("vision_adapter.fc1.weight", false));
    try testing.expect(!shouldKeepWeightKey("vision_tower.layers.0.norm1.weight", false));
    try testing.expect(shouldKeepWeightKey("model.vision_tower.layers.0.norm1.weight", true));
    try testing.expect(shouldKeepWeightKey("model.vision_adapter.fc1.weight", true));
    try testing.expect(shouldKeepWeightKey("model.vision_projection.weight", true));
    try testing.expect(shouldKeepWeightKey("vision_adapter.fc1.weight", true));
    try testing.expect(shouldKeepWeightKey("vision_tower.layers.0.norm1.weight", true));
    // avlp12 Alis spells the Qwen3-VL tower `model.visual.` — same gate, or
    // --no-vision cannot drop it and we hold ~0.9 GB we never read.
    try testing.expect(!shouldKeepWeightKey("model.visual.blocks.0.norm1.weight", false));
    try testing.expect(shouldKeepWeightKey("model.visual.blocks.0.norm1.weight", true));
    // Text weights are never touched either way.
    try testing.expect(shouldKeepWeightKey("model.language_model.embed_tokens.weight", false));
    try testing.expect(shouldKeepWeightKey("language_model.model.embed_tokens.weight", false));
    try testing.expect(shouldKeepWeightKey("language_model.lm_head.weight", false));
}

test "parseVisionProcessorDefaultsFromJson supports current and legacy Qwen layouts" {
    const current = parseVisionProcessorDefaultsFromJson(
        \\{"image_processor":{"min_pixels":65536,"max_pixels":16777216}}
    );
    try testing.expectEqual(@as(?u32, 65536), current.min_pixels);
    try testing.expectEqual(@as(?u32, 16777216), current.max_pixels);

    const legacy = parseVisionProcessorDefaultsFromJson(
        \\{"size":{"shortest_edge":3136,"longest_edge":1003520}}
    );
    try testing.expectEqual(@as(?u32, 3136), legacy.min_pixels);
    try testing.expectEqual(@as(?u32, 1003520), legacy.max_pixels);
}

test "parseVisionProcessorDefaultsFromJson rejects invalid values and ranges" {
    const reversed = parseVisionProcessorDefaultsFromJson(
        \\{"image_processor":{"min_pixels":4096,"max_pixels":1024}}
    );
    try testing.expectEqual(@as(?u32, null), reversed.min_pixels);
    try testing.expectEqual(@as(?u32, null), reversed.max_pixels);

    const invalid = parseVisionProcessorDefaultsFromJson(
        \\{"image_processor":{"min_pixels":0,"max_pixels":4294967296}}
    );
    try testing.expectEqual(@as(?u32, null), invalid.min_pixels);
    try testing.expectEqual(@as(?u32, null), invalid.max_pixels);

    const malformed = parseVisionProcessorDefaultsFromJson("not json");
    try testing.expectEqual(@as(?u32, null), malformed.min_pixels);
    try testing.expectEqual(@as(?u32, null), malformed.max_pixels);
}

test "parseConfig prefers processor_config and fills missing Qwen bounds from preprocessor_config" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    const config_json = @embedFile("fixtures/model-configs/qwen4_exp.json");
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = config_json });
    try tmp.dir.writeFile(io, .{
        .sub_path = "processor_config.json",
        .data = "{\"image_processor\":{\"min_pixels\":65536}}",
    });
    try tmp.dir.writeFile(io, .{
        .sub_path = "preprocessor_config.json",
        .data = "{\"size\":{\"shortest_edge\":3136,\"longest_edge\":16777216}}",
    });

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try tmp.dir.realPath(io, &path_buf);
    var config = try parseConfig(io, testing.allocator, path_buf[0..path_len]);
    defer config.deinit(testing.allocator);
    try testing.expect(config.qwen_vision);
    try testing.expectEqual(@as(u32, 65536), config.qv_min_pixels);
    try testing.expectEqual(@as(u32, 16777216), config.qv_max_pixels);
}

test "ModelConfig text-only qwen3_5 has no qwen_vision" {
    const json =
        \\{
        \\  "model_type": "qwen3_5_text",
        \\  "hidden_size": 1024,
        \\  "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.25}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expect(!config.qwen_vision);
    try testing.expect(!config.has_vision);
}

test "ModelConfig keeps explicit gemma3 head counts (12b)" {
    // Regression guard for the fix above: a gemma3 text_config that DOES ship
    // head counts must keep them, never get clobbered by the HF-default fill.
    const json =
        \\{
        \\  "model_type": "gemma3",
        \\  "text_config": {
        \\    "model_type": "gemma3_text",
        \\    "hidden_size": 3840,
        \\    "num_hidden_layers": 48,
        \\    "num_attention_heads": 16,
        \\    "num_key_value_heads": 8,
        \\    "head_dim": 256,
        \\    "sliding_window": 1024
        \\  },
        \\  "quantization": {"bits": 4, "group_size": 32}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expectEqual(@as(u32, 16), config.num_attention_heads);
    try testing.expectEqual(@as(u32, 8), config.num_key_value_heads);
    try testing.expectEqual(@as(u32, 256), config.head_dim);
}

test "ensureGemmaTerminators is additive and dedup-guarded" {
    var c = ModelConfig{};
    c.addEosToken(1); // config-provided scalar eos
    c.ensureGemmaTerminators();
    try testing.expect(c.isEosToken(1));
    try testing.expect(c.isEosToken(106));
    try testing.expectEqual(@as(u32, 2), c.num_eos_tokens);
    // Idempotent: re-running adds nothing.
    c.ensureGemmaTerminators();
    try testing.expectEqual(@as(u32, 2), c.num_eos_tokens);
}

test "bailing_hybrid gate arm is SELECTED by the bound, never defaulted" {
    // `kdaGateChain` with bound 0 computes exp(0) = 1: a decay that never
    // forgets, on a checkpoint that merely omitted the key. The arms are fla's
    // two, and the absent case belongs to the softplus chain (which is
    // elementwise, so it serves a per-channel gate unchanged).
    var bounded = ModelConfig{ .model_type = "bailing_hybrid" };
    bounded.kda_vector_gate = true;
    bounded.kda_gate_lower_bound = -5;
    try testing.expect(bounded.kdaUsesBoundedGate());

    var unbounded = ModelConfig{ .model_type = "bailing_hybrid" };
    unbounded.kda_vector_gate = true; // per-channel gate, softplus form
    try testing.expect(!unbounded.kdaUsesBoundedGate());

    // A per-HEAD gate is never the bounded arm regardless of the field.
    var per_head = ModelConfig{ .model_type = "qwen3_5_moe" };
    per_head.kda_gate_lower_bound = -5;
    try testing.expect(!per_head.kdaUsesBoundedGate());
}

test "parseConfigFromJson quantized qwen3_5_moe → quant_bits from key" {
    // Same arch but with a "quantization" block: quant_bits must reflect it so
    // the mandatory scale/bias fetches still fire (a missing scale is a clear
    // MISSING WEIGHT error, not a silent dense fallback). Guards the default flip.
    const json =
        \\{
        \\  "model_type": "qwen3_5_moe",
        \\  "text_config": {"hidden_size": 2048, "num_experts": 256},
        \\  "quantization": {"bits": 4, "group_size": 64}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expectEqual(@as(u32, 4), config.quant_bits);
    try testing.expectEqual(@as(u32, 64), config.quant_group_size);
    try testing.expectEqual(QuantMode.affine, config.quant_mode);
}

test "parseConfigFromJson rejects affine bits MLX has no kernels for" {
    // A checkpoint declaring an affine bit-width outside MLX's kernel set
    // ({2,3,4,5,6,8}) must fail at PARSE, not at warmup: mlx only validates
    // bits inside quantize(), so an already-quantized 1-bit checkpoint sails
    // through load and dies with an uncatchable Metal kernel-load error
    // ("Unable to load kernel affine_dequantize_..._b_1") that kills the
    // whole server. Live bite: prism-ml/Bonsai-27B-mlx-1bit.
    const json_1bit =
        \\{
        \\  "model_type": "qwen3_5",
        \\  "text_config": {"hidden_size": 5120},
        \\  "quantization": {"bits": 1, "group_size": 128}
        \\}
    ;
    try expectError(error.UnsupportedQuantBits, parseConfigFromJson(testing.allocator, json_1bit));

    const json_7bit =
        \\{
        \\  "model_type": "qwen3",
        \\  "hidden_size": 1024,
        \\  "quantization": {"bits": 7, "group_size": 64}
        \\}
    ;
    try expectError(error.UnsupportedQuantBits, parseConfigFromJson(testing.allocator, json_7bit));
}

test "parseConfigFromJson nvfp4 quantization mode" {
    // NVFP4 checkpoints (issue #24): {"group_size": 16, "bits": 4, "mode": "nvfp4"}.
    // The mode must land on config.quant_mode so the loader skips the .biases
    // fetches (nvfp4 stores no biases tensors) and the matmul call sites pass
    // "nvfp4" to mlx instead of "affine".
    const json =
        \\{
        \\  "model_type": "qwen3",
        \\  "hidden_size": 1024,
        \\  "quantization": {"group_size": 16, "bits": 4, "mode": "nvfp4"}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expectEqual(QuantMode.nvfp4, config.quant_mode);
    try testing.expectEqual(@as(u32, 4), config.quant_bits);
    try testing.expectEqual(@as(u32, 16), config.quant_group_size);
    try testing.expect(!config.quant_mode.hasBiases());
}

test "parseConfigFromJson explicit affine mode keeps biases" {
    const json =
        \\{
        \\  "model_type": "qwen3",
        \\  "hidden_size": 1024,
        \\  "quantization": {"group_size": 64, "bits": 4, "mode": "affine"}
        \\}
    ;
    const config = try parseConfigFromJson(testing.allocator, json);
    try testing.expectEqual(QuantMode.affine, config.quant_mode);
    try testing.expect(config.quant_mode.hasBiases());
}

test "parseConfigFromJson unknown quantization mode → error" {
    // An unrecognized mode must fail loudly at config parse — not crash later
    // in the weight loader with a misleading MISSING WEIGHT error.
    const json =
        \\{
        \\  "model_type": "qwen3",
        \\  "hidden_size": 1024,
        \\  "quantization": {"group_size": 32, "bits": 4, "mode": "fp99"}
        \\}
    ;
    try expectError(error.UnsupportedQuantMode, parseConfigFromJson(testing.allocator, json));
}

test "shouldKeepWeightKey drops DiffusionGemma encoder vision tower (text-only v1)" {
    // DiffusionGemma nests its vision tower under model.encoder.* — distinct
    // from the bare vision_tower.* prefixes of earlier checkpoints. Until the
    // tower is wired, those tensors must be dropped even with load_vision on,
    // and ALWAYS dropped when vision is off.
    try testing.expect(!shouldKeepWeightKey("model.encoder.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight", false));
    try testing.expect(!shouldKeepWeightKey("model.encoder.embed_vision.embedding_projection.weight", false));
    // Trunk + diffusion weights always survive.
    try testing.expect(shouldKeepWeightKey("model.decoder.layers.0.experts.gate_up_proj.weight", false));
    try testing.expect(shouldKeepWeightKey("model.decoder.self_conditioning.gate_proj.weight", false));
    try testing.expect(shouldKeepWeightKey("model.encoder.language_model.layers.0.layer_scalar", false));
}

test "narrowsLoadedF16 catches per-channel tables, not matmul operands" {
    // Quant side tensors: the pre-existing rule, keyed on the suffix because
    // they can be 2-D.
    try testing.expect(narrowsLoadedF16("model.layers.0.mlp.down_proj.scales", 2, .float16));
    try testing.expect(narrowsLoadedF16("model.layers.0.mlp.down_proj.biases", 2, .float16));

    // Any 1-D f16 tensor is a PER-CHANNEL table — a norm weight, a bias, a
    // gate table. It gets multiplied or added straight into the activation
    // stream, so leaving it f16 beside a bf16 residual promotes the residual
    // (and therefore every later weight read) to f32.
    try testing.expect(narrowsLoadedF16("language_model.model.layers.0.input_layernorm.weight", 1, .float16));
    try testing.expect(narrowsLoadedF16("language_model.model.layers.0.linear_attn.A_log", 1, .float16));
    try testing.expect(narrowsLoadedF16("language_model.model.layers.0.linear_attn.dt_bias", 1, .float16));
    try testing.expect(narrowsLoadedF16("language_model.model.norm.weight", 1, .float16));

    // A 2-D dense f16 weight is a MATMUL OPERAND, not a table. MLX picks its
    // kernel off that dtype, so narrowing it is a kernel-selection change and
    // not this rule's business — it stays per-site.
    try testing.expect(!narrowsLoadedF16("vision_tower.blocks.0.attn.qkv.weight", 2, .float16));
    try testing.expect(!narrowsLoadedF16("language_model.model.layers.0.linear_attn.conv1d.weight", 3, .float16));

    // Everything already in the engine's dtype, and packed weights, are left
    // alone.
    try testing.expect(!narrowsLoadedF16("model.layers.0.input_layernorm.weight", 1, .bfloat16));
    try testing.expect(!narrowsLoadedF16("model.layers.0.mlp.down_proj.weight", 2, .uint32));
    try testing.expect(!narrowsLoadedF16("model.layers.0.mlp.down_proj.scales", 2, .bfloat16));
}

test "parseGenerationDefaultsFromJson: reads model sampling recommendations" {
    // Verbatim shape of Qwen3.6 / Gemma 4 checkpoints' generation_config.json.
    const json =
        \\{"bos_token_id": 248044, "do_sample": true, "temperature": 1.0, "top_k": 20, "top_p": 0.95}
    ;
    const gd = parseGenerationDefaultsFromJson(json);
    try testing.expectEqual(@as(?f32, 1.0), gd.temperature);
    try testing.expectEqual(@as(?f32, 0.95), gd.top_p);
    try testing.expectEqual(@as(?u32, 20), gd.top_k);
}

test "pooling: config.json pooling_mode key parses; unknown value rejected at parse" {
    // Explicit converter/operator contract for checkpoints whose config alone
    // can't reveal pooling (Qwen3-Embedding declares plain `qwen3`).
    const base = "{{\"model_type\":\"qwen3\",\"hidden_size\":64,\"num_attention_heads\":8,\"num_hidden_layers\":2,\"pooling_mode\":\"{s}\"}}";
    inline for (.{ .{ "last_token", PoolingMode.last_token }, .{ "cls", PoolingMode.cls }, .{ "mean", PoolingMode.mean } }) |case| {
        const json = try std.fmt.allocPrint(testing.allocator, base, .{case[0]});
        defer testing.allocator.free(json);
        const config = try parseConfigFromJson(testing.allocator, json);
        try testing.expectEqual(@as(?PoolingMode, case[1]), config.pooling_mode);
        try testing.expect(config.hasEmbeddingCapability());
        try testing.expect(!config.is_encoder_only); // pooling never flips the arch
    }
    // An unknown mode is a parse error, not a silent mean-pool: wrong-semantics
    // vectors are harder to detect than a refused load.
    const bad = try std.fmt.allocPrint(testing.allocator, base, .{"weighted_mean"});
    defer testing.allocator.free(bad);
    try expectError(error.UnsupportedPoolingMode, parseConfigFromJson(testing.allocator, bad));
}

test "pooling: sentence-transformers 1_Pooling sidecar parses all three modes" {
    // Verbatim shape of ST `1_Pooling/config.json` (Qwen3-Embedding sets
    // lasttoken, bge/mxbai set cls_token, MiniLM sets mean_tokens).
    const last =
        \\{"word_embedding_dimension": 2560, "pooling_mode_cls_token": false,
        \\ "pooling_mode_mean_tokens": false, "pooling_mode_max_tokens": false,
        \\ "pooling_mode_mean_sqrt_len_tokens": false, "pooling_mode_lasttoken": true}
    ;
    try testing.expectEqual(@as(?PoolingMode, .last_token), try parsePoolingSidecar(last));
    const cls =
        \\{"pooling_mode_cls_token": true, "pooling_mode_mean_tokens": false, "pooling_mode_lasttoken": false}
    ;
    try testing.expectEqual(@as(?PoolingMode, .cls), try parsePoolingSidecar(cls));
    const mean =
        \\{"pooling_mode_cls_token": false, "pooling_mode_mean_tokens": true}
    ;
    try testing.expectEqual(@as(?PoolingMode, .mean), try parsePoolingSidecar(mean));
}

test "pooling: sidecar demanding an unsupported mode errors; non-pooling JSON is ignored" {
    // A sidecar that DOES declare pooling but none we implement (weighted-mean,
    // max) must refuse the load — mean-pooling it anyway is silent corruption.
    const unsupported =
        \\{"pooling_mode_cls_token": false, "pooling_mode_mean_tokens": false,
        \\ "pooling_mode_max_tokens": true, "pooling_mode_lasttoken": false}
    ;
    try testing.expectError(error.UnsupportedPoolingMode, parsePoolingSidecar(unsupported));
    // Malformed / unrelated JSON: best-effort null, like generation_config.json.
    try testing.expectEqual(@as(?PoolingMode, null), try parsePoolingSidecar("not json"));
    try testing.expectEqual(@as(?PoolingMode, null), try parsePoolingSidecar("{\"dimension\": 384}"));
}

test "pooling: known-family directory-name fallback" {
    // The mlx-community conversions ship NO sidecar and a plain chat
    // model_type, so a metadata-less checkpoint falls back to the family
    // table — gated on the arch so a name can never flip an unrelated model.
    try testing.expectEqual(@as(?PoolingMode, .last_token), poolingFromDirName("Qwen3-Embedding-4B-4bit-DWQ", "qwen3"));
    try testing.expectEqual(@as(?PoolingMode, .last_token), poolingFromDirName("qwen3-embedding-0.6b", "qwen3"));
    try testing.expectEqual(@as(?PoolingMode, null), poolingFromDirName("Qwen3-8B-4bit", "qwen3"));
    try testing.expectEqual(@as(?PoolingMode, null), poolingFromDirName("Qwen3-Embedding-4B", "llama"));
    // bge / mxbai are CLS-pooling BERTs (their cards say so); MiniLM stays mean.
    try testing.expectEqual(@as(?PoolingMode, .cls), poolingFromDirName("bge-small-en-v1.5-8bit", "bert"));
    try testing.expectEqual(@as(?PoolingMode, .cls), poolingFromDirName("mxbai-embed-large-v1", "bert"));
    try testing.expectEqual(@as(?PoolingMode, null), poolingFromDirName("all-MiniLM-L6-v2", "bert"));
    // EmbeddingGemma is mean-pooled via its own bidirectional path — the name
    // fallback must not touch non-qwen3 archs on the "embedding" substring.
    try testing.expectEqual(@as(?PoolingMode, null), poolingFromDirName("embeddinggemma-300m-8bit", "gemma3_text"));
}

test "pooling: effectivePooling defaults to mean; encoder capability unions" {
    var config = ModelConfig{};
    try testing.expectEqual(PoolingMode.mean, config.effectivePooling());
    try testing.expect(!config.hasEmbeddingCapability());
    config.is_encoder_only = true;
    try testing.expect(config.hasEmbeddingCapability());
    config.is_encoder_only = false;
    config.pooling_mode = .last_token;
    try testing.expectEqual(PoolingMode.last_token, config.effectivePooling());
    try testing.expect(config.hasEmbeddingCapability());
}

test "parseGenerationDefaultsFromJson: missing keys and malformed input give nulls" {
    const partial = parseGenerationDefaultsFromJson("{\"eos_token_id\": [1, 2]}");
    try testing.expectEqual(@as(?f32, null), partial.temperature);
    try testing.expectEqual(@as(?f32, null), partial.top_p);
    try testing.expectEqual(@as(?u32, null), partial.top_k);

    const broken = parseGenerationDefaultsFromJson("not json at all");
    try testing.expectEqual(@as(?f32, null), broken.temperature);

    // Out-of-range values are dropped, not clamped — a corrupt config must
    // not silently pin sampling to an extreme.
    const insane = parseGenerationDefaultsFromJson("{\"temperature\": 99.0, \"top_p\": 7.0, \"top_k\": -5}");
    try testing.expectEqual(@as(?f32, null), insane.temperature);
    try testing.expectEqual(@as(?f32, null), insane.top_p);
    try testing.expectEqual(@as(?u32, null), insane.top_k);
}

test "parseConfigFromJson: qwen4_exp (Qwen3.8-Flash-Next) reads the hyper-connection, PLE, QSA and text-config eos fields" {
    const json =
        \\{"architectures":["Qwen4ExpForConditionalGeneration"],"model_type":"qwen4_exp",
        \\ "text_config":{"model_type":"qwen4_exp_text","hidden_size":2560,"num_hidden_layers":48,
        \\ "full_attention_interval":4,"num_attention_heads":24,"num_key_value_heads":2,"head_dim":256,
        \\ "hc_count":4,"hc_lowrank":320,"ple_layer_ids":[2],"ple_embed_dim":2560,"ple_conv_kernel_size":4,
        \\ "ngram_size":3,"heads_per_ngram":8,"ngram_vocab_size_base":20000000,"make_ngram_vocab_size_divisible_by":128,
        \\ "indexer_n_heads":4,"indexer_kv_heads":1,"indexer_head_dim":128,"indexer_budget":2048,"indexer_compress_ratio":4,
        \\ "linear_num_key_heads":16,"linear_num_value_heads":48,"linear_key_head_dim":128,"linear_value_head_dim":128,
        \\ "num_experts":512,"num_experts_per_tok":10,"moe_intermediate_size":640,"shared_expert_intermediate_size":640,
        \\ "eos_token_id":248044,"vocab_size":248320,"rms_norm_eps":1e-6,"output_gate_type":"sigmoid",
        \\ "rope_parameters":{"rope_theta":10000000,"partial_rotary_factor":0.25,"mrope_section":[11,11,10],"mrope_interleaved":true}},
        \\ "quantization":{"group_size":64,"bits":4,"mode":"affine"}}
    ;
    const c = try parseConfigFromJson(testing.allocator, json);
    try testing.expect(c.isQwen4());
    try testing.expectEqualStrings("language_model.model", c.weight_prefix);
    try testing.expectEqual(@as(u32, 4), c.hc_count);
    try testing.expectEqual(@as(u32, 320), c.hc_lowrank);
    try testing.expectEqual(@as(i32, 1), c.ple_layer_idx); // 1-based [2] → layer 1
    try testing.expectEqual(@as(u32, 2560), c.ple_embed_dim);
    try testing.expectEqual(@as(u32, 4), c.indexer_n_heads);
    try testing.expectEqual(@as(u32, 2048), c.indexer_budget);
    try testing.expectEqual(@as(u32, 4), c.indexer_compress_ratio);
    try testing.expectEqual(@as(u32, 248044), c.ngram_eos);
    try testing.expectEqual(@as(u32, 4), c.full_attention_interval);
    try testing.expect(c.isLinearLayer(0) and !c.isLinearLayer(3));
    try testing.expectEqual(@as(u32, 12), c.attnCacheLayerCount());
    try testing.expectEqual(@as(u64, 12 * 128 * 2 / 4), c.qsaHistoryBytesPerToken());
    try testing.expectEqual(@as(u64, 12 * @as(u64, @intCast(@import("transformer.zig").QSA_RING_ROWS)) * 128 * 2), c.qsaRingBytes());
    try testing.expect(c.attn_output_gate and c.kda_sigmoid_out_gate and !c.has_final_norm and !c.norm_has_offset);
    try testing.expect(c.isMoe() and c.supportsBatchedGdnDecode()); // per-slot state on the SSMCacheEntry: batches
    try testing.expectEqual(@as(f32, 0.25), c.partial_rotary_factor);
    try testing.expectEqual(@as(f32, 10000000.0), c.rope_theta);
    try testing.expect(!c.qwen_vision and !c.has_vision);
}

test "parseConfigFromJson accepts dense qwen4 as an expert streaming architecture" {
    const json =
        \\{"model_type":"qwen4_exp","text_config":{"model_type":"qwen4_exp_text","hidden_size":2560,"num_hidden_layers":48,"num_attention_heads":24,"num_key_value_heads":2,"head_dim":256,"full_attention_interval":4,"num_experts":512,"num_experts_per_tok":10,"moe_intermediate_size":640,"shared_expert_intermediate_size":640,"ple_layer_ids":[2],"ngram_size":3,"heads_per_ngram":8,"ngram_vocab_size_base":20000000,"make_ngram_vocab_size_divisible_by":128,"indexer_n_heads":4,"indexer_head_dim":128,"indexer_budget":2048,"indexer_compress_ratio":4}}
    ;
    const config = try parseConfigFromJson(std.testing.allocator, json);
    try std.testing.expectEqual(@as(u32, 0), config.quant_bits);
    try std.testing.expect(config.supportsExpertStreaming());
}

test "expert streaming is a capability of every qwen4 pack and required only by the dense one" {
    const t = std.testing;
    const dense =
        \\{"model_type":"qwen4_exp","text_config":{"model_type":"qwen4_exp_text","hidden_size":2560,"num_hidden_layers":48,"num_attention_heads":24,"num_key_value_heads":2,"head_dim":256,"full_attention_interval":4,"num_experts":512,"num_experts_per_tok":10,"moe_intermediate_size":640,"shared_expert_intermediate_size":640,"ple_layer_ids":[2],"ngram_size":3,"heads_per_ngram":8,"ngram_vocab_size_base":20000000,"make_ngram_vocab_size_divisible_by":128,"indexer_n_heads":4,"indexer_head_dim":128,"indexer_budget":2048,"indexer_compress_ratio":4}}
    ;
    const quantized =
        \\{"model_type":"qwen4_exp","quantization":{"group_size":64,"bits":4,"mode":"affine"},"text_config":{"model_type":"qwen4_exp_text","hidden_size":2560,"num_hidden_layers":48,"num_attention_heads":24,"num_key_value_heads":2,"head_dim":256,"full_attention_interval":4,"num_experts":512,"num_experts_per_tok":10,"moe_intermediate_size":640,"shared_expert_intermediate_size":640,"ple_layer_ids":[2],"ngram_size":3,"heads_per_ngram":8,"ngram_vocab_size_base":20000000,"make_ngram_vocab_size_divisible_by":128,"indexer_n_heads":4,"indexer_head_dim":128,"indexer_budget":2048,"indexer_compress_ratio":4}}
    ;
    var dense_config = try parseConfigFromJson(t.allocator, dense);
    defer dense_config.deinit(t.allocator);
    try t.expect(dense_config.supportsExpertStreaming() and dense_config.expertStreamingRequired());
    var quant_config = try parseConfigFromJson(t.allocator, quantized);
    defer quant_config.deinit(t.allocator);
    try t.expectEqual(@as(u32, 4), quant_config.quant_bits);
    try t.expect(quant_config.supportsExpertStreaming() and !quant_config.expertStreamingRequired());
    var other = ModelConfig{ .model_type = "qwen3_5_moe", .num_hidden_layers = 48, .num_experts = 512, .num_experts_per_tok = 10, .hidden_size = 2560, .moe_intermediate_size = 640 };
    try t.expect(!other.supportsExpertStreaming() and !other.expertStreamingRequired());
}

test "qwen4 streaming loader canonicalizes resident keys and excludes disk tensors" {
    var buf: [256]u8 = undefined;
    try std.testing.expectEqualStrings("language_model.model.layers.7.mlp.gate.weight", qwen4StreamingWeightKey(.bf16_fused, &buf, "model.language_model.layers.7.mlp.gate.weight").?);
    try std.testing.expectEqualStrings("language_model.mtp.fc_hidden.weight", qwen4StreamingWeightKey(.bf16_fused, &buf, "mtp.fc_hidden.weight").?);
    try std.testing.expectEqualStrings("language_model.lm_head.weight", qwen4StreamingWeightKey(.bf16_fused, &buf, "lm_head.weight").?);
    try std.testing.expect(qwen4StreamingWeightKey(.bf16_fused, &buf, "model.language_model.layers.7.mlp.experts.gate_up_proj") == null);
    try std.testing.expect(qwen4StreamingWeightKey(.bf16_fused, &buf, "model.language_model.layers.7.mlp.experts.down_proj") == null);
    try std.testing.expect(qwen4StreamingWeightKey(.bf16_fused, &buf, "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_12.weight") == null);
}

test "the quantized streaming loader drops the nine routed banks and keeps everything else" {
    var buf: [256]u8 = undefined;
    for ([_][]const u8{ "weight", "scales", "biases" }) |part| {
        for ([_][]const u8{ "gate", "up", "down" }) |proj| {
            var key_buf: [192]u8 = undefined;
            const key = try std.fmt.bufPrint(&key_buf, "language_model.model.layers.7.mlp.switch_mlp.{s}_proj.{s}", .{ proj, part });
            try std.testing.expect(qwen4StreamingWeightKey(.quantized_split, &buf, key) == null);
        }
    }
    try std.testing.expectEqualStrings(
        "language_model.model.layers.7.mlp.shared_expert.gate_proj.scales",
        qwen4StreamingWeightKey(.quantized_split, &buf, "language_model.model.layers.7.mlp.shared_expert.gate_proj.scales").?,
    );
    try std.testing.expectEqualStrings(
        "language_model.model.layers.7.mlp.gate.weight",
        qwen4StreamingWeightKey(.quantized_split, &buf, "language_model.model.layers.7.mlp.gate.weight").?,
    );
    try std.testing.expectEqualStrings(
        "language_model.lm_head.weight",
        qwen4StreamingWeightKey(.quantized_split, &buf, "language_model.lm_head.weight").?,
    );
}

test "qwen4 streaming loader materializes only transformed resident tensors" {
    const t = std.testing;
    if (std.c.getenv("CODEX_SANDBOX") != null) return error.SkipZigTest;
    const io = t.io;
    const allocator = t.allocator;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try tmp.dir.realPath(io, &path_buf);
    const model_dir = path_buf[0..path_len];
    const header = "{\"model.language_model.layers.0.mlp.experts.gate_up_proj\":{\"dtype\":\"BF16\",\"shape\":[1,2,1],\"data_offsets\":[0,4]},\"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight\":{\"dtype\":\"BF16\",\"shape\":[1,2,1],\"data_offsets\":[4,8]},\"model.language_model.layers.0.mlp.gate.weight\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[8,10]},\"model.language_model.layers.0.self_attn.q_norm.weight\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[10,12]},\"model.language_model.layers.0.linear_attn.conv1d.weight\":{\"dtype\":\"BF16\",\"shape\":[1,1,2],\"data_offsets\":[12,16]},\"mtp.layers.0.mlp.experts.gate_up_proj\":{\"dtype\":\"BF16\",\"shape\":[1,2,1],\"data_offsets\":[16,20]},\"mtp.layers.0.mlp.experts.down_proj\":{\"dtype\":\"BF16\",\"shape\":[1,1,1],\"data_offsets\":[20,22]}}";
    const padded_header_len = std.mem.alignForward(usize, header.len, 8);
    const file_bytes = try allocator.alloc(u8, 8 + padded_header_len + 22);
    defer allocator.free(file_bytes);
    std.mem.writeInt(u64, file_bytes[0..8], padded_header_len, .little);
    @memset(file_bytes[8 .. 8 + padded_header_len], ' ');
    @memcpy(file_bytes[8 .. 8 + header.len], header);
    const tensor_data = [_]u16{ 0x3f80, 0x4000, 0x3f80, 0x4000, 0x3f80, 0, 0x3f80, 0x4000, 0x3f80, 0x4000, 0x3f80 };
    @memcpy(file_bytes[8 + padded_header_len ..], std.mem.sliceAsBytes(&tensor_data));
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = file_bytes });

    var weights = try loadWeightsStreaming(io, allocator, model_dir, .bf16_fused, false, false);
    defer weights.deinit();
    try t.expect(weights.get("model.language_model.layers.0.mlp.experts.gate_up_proj") == null);
    try t.expect(weights.get("language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight") == null);
    try t.expect(weights.get("language_model.model.layers.0.mlp.gate.weight") != null);
    const norm = weights.get("language_model.model.layers.0.self_attn.q_norm.weight").?;
    try t.expectEqualSlices(c_int, &.{1}, mlx.getShape(norm));
    try t.expectEqualSlices(c_int, &.{ 1, 2, 1 }, mlx.getShape(weights.get("language_model.model.layers.0.linear_attn.conv1d.weight").?));
    try t.expect(weights.get("language_model.mtp.layers.0.mlp.switch_mlp.gate_proj.weight") == null);
    try t.expect(weights.get("language_model.mtp.layers.0.mlp.switch_mlp.up_proj.weight") == null);
    try t.expect(weights.get("language_model.mtp.layers.0.mlp.switch_mlp.down_proj.weight") == null);
}

test "a streamed load that fails mid-transform frees the tensor it was holding" {
    const t = std.testing;
    if (std.c.getenv("CODEX_SANDBOX") != null) return error.SkipZigTest;
    const io = t.io;
    const allocator = t.allocator;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try tmp.dir.realPath(io, &path_buf);
    const payload: usize = 2048 * 2 * 2048 * 2;
    const header = "{\"model.language_model.layers.0.linear_attn.conv1d.weight\":{\"dtype\":\"BF16\",\"shape\":[2048,2,2048],\"data_offsets\":[0,16777216]}}";
    const padded_header_len = std.mem.alignForward(usize, header.len, 8);
    const file_bytes = try allocator.alloc(u8, 8 + padded_header_len + payload);
    defer allocator.free(file_bytes);
    std.mem.writeInt(u64, file_bytes[0..8], padded_header_len, .little);
    @memset(file_bytes[8 .. 8 + padded_header_len], ' ');
    @memcpy(file_bytes[8 .. 8 + header.len], header);
    @memset(file_bytes[8 + padded_header_len ..], 0x3c);
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = file_bytes });

    const model_dir = path_buf[0..path_len];
    const FdProbe = struct {
        fn count(iox: std.Io) usize {
            var dir = std.Io.Dir.openDirAbsolute(iox, "/dev/fd", .{ .iterate = true }) catch return 0;
            defer dir.close(iox);
            var n: usize = 0;
            var walker = dir.iterate();
            while (walker.next(iox) catch null) |_| n += 1;
            return n;
        }
    };
    try t.expectError(error.InvalidQwen4ConvShape, loadWeightsStreaming(io, allocator, model_dir, .bf16_fused, false, false));
    const before = FdProbe.count(io);
    for (0..8) |_| {
        try t.expectError(error.InvalidQwen4ConvShape, loadWeightsStreaming(io, allocator, model_dir, .bf16_fused, false, false));
    }
    const after = FdProbe.count(io);
    try t.expect(after <= before + 1);
}

test "the streamed load drops the MTP head the ledger bills at zero" {
    const t = std.testing;
    if (std.c.getenv("CODEX_SANDBOX") != null) return error.SkipZigTest;
    const io = t.io;
    const allocator = t.allocator;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try tmp.dir.realPath(io, &path_buf);
    const header = "{\"model.language_model.layers.0.mlp.gate.weight\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]},\"mtp.fc_hidden.weight\":{\"dtype\":\"BF16\",\"shape\":[1,1],\"data_offsets\":[2,4]},\"mtp.layers.0.mlp.experts.down_proj\":{\"dtype\":\"BF16\",\"shape\":[1,1,1],\"data_offsets\":[4,6]}}";
    const padded_header_len = std.mem.alignForward(usize, header.len, 8);
    const file_bytes = try allocator.alloc(u8, 8 + padded_header_len + 6);
    defer allocator.free(file_bytes);
    std.mem.writeInt(u64, file_bytes[0..8], padded_header_len, .little);
    @memset(file_bytes[8 .. 8 + padded_header_len], ' ');
    @memcpy(file_bytes[8 .. 8 + header.len], header);
    const tensor_data = [_]u16{ 0x3f80, 0x4000, 0x3f80 };
    @memcpy(file_bytes[8 + padded_header_len ..], std.mem.sliceAsBytes(&tensor_data));
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = file_bytes });

    var weights = try loadWeightsStreaming(io, allocator, path_buf[0..path_len], .bf16_fused, false, false);
    defer weights.deinit();
    try t.expect(weights.get("language_model.model.layers.0.mlp.gate.weight") != null);
    var it = weights.map.iterator();
    while (it.next()) |entry| {
        try t.expect(!std.mem.startsWith(u8, entry.key_ptr.*, "language_model.mtp."));
    }
}

test "qwen4 streaming resident byte estimate excludes experts PLE and vision" {
    const t = std.testing;
    const io = t.io;
    var tmp = t.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const header = "{\"model.language_model.layers.0.mlp.experts.gate_up_proj\":{\"dtype\":\"BF16\",\"shape\":[1,2,2],\"data_offsets\":[0,8]},\"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight\":{\"dtype\":\"BF16\",\"shape\":[1,2],\"data_offsets\":[8,12]},\"model.visual.x\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[12,14]},\"model.language_model.layers.0.mlp.gate.weight\":{\"dtype\":\"BF16\",\"shape\":[1,3],\"data_offsets\":[14,20]},\"mtp.layers.0.mlp.experts.down_proj\":{\"dtype\":\"BF16\",\"shape\":[1,3,2],\"data_offsets\":[20,32]}}";
    const bytes = try t.allocator.alloc(u8, 8 + header.len + 32);
    defer t.allocator.free(bytes);
    std.mem.writeInt(u64, bytes[0..8], header.len, .little);
    @memcpy(bytes[8 .. 8 + header.len], header);
    @memset(bytes[8 + header.len ..], 0);
    try tmp.dir.writeFile(io, .{ .sub_path = "s.safetensors", .data = bytes });
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"model.language_model.layers.0.mlp.experts.gate_up_proj\":\"s.safetensors\",\"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight\":\"s.safetensors\",\"model.visual.x\":\"s.safetensors\",\"model.language_model.layers.0.mlp.gate.weight\":\"s.safetensors\",\"mtp.layers.0.mlp.experts.down_proj\":\"s.safetensors\"}}" });
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try tmp.dir.realPath(io, &path_buf);
    const split = try streamingResidentSplit(io, t.allocator, path_buf[0..path_len], &.{ .expert_layout = .bf16_fused });
    try t.expectEqual(@as(u64, 6), split.trunk);
    try t.expectEqual(@as(u64, 12), split.mtp);
}

test "real qwen streaming resident estimate is trunk plus MTP only" {
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try @import("test_models.zig").packPath(&path_buf, "Qwen/Qwen3.8-Flash-Next");
    var dir = std.Io.Dir.openDirAbsolute(std.testing.io, path, .{}) catch return error.SkipZigTest;
    dir.close(std.testing.io);
    const split = try streamingResidentSplit(std.testing.io, std.testing.allocator, path, &.{ .expert_layout = .bf16_fused });
    const bytes = split.trunk +| split.mtp;
    try std.testing.expect(bytes > 14_000_000_000 and bytes < 16_000_000_000);
}

test "parseConfigFromJson: qwen4_exp with vision_config reads the Qwen3-VL tower, M-RoPE and vision token ids" {
    const json =
        \\{"architectures":["Qwen4ExpForConditionalGeneration"],"model_type":"qwen4_exp",
        \\ "image_token_id":248056,"video_token_id":248057,"vision_start_token_id":248053,"vision_end_token_id":248054,
        \\ "vision_config":{"depth":27,"hidden_size":1152,"num_heads":16,"intermediate_size":4304,"patch_size":16,
        \\   "temporal_patch_size":2,"spatial_merge_size":2,"num_position_embeddings":2304,"out_hidden_size":2560,"model_type":"qwen4_exp_vision"},
        \\ "text_config":{"model_type":"qwen4_exp_text","hidden_size":2560,"num_hidden_layers":48,
        \\ "full_attention_interval":4,"num_attention_heads":24,"num_key_value_heads":2,"head_dim":256,
        \\ "ple_layer_ids":[2],"indexer_n_heads":4,"indexer_head_dim":128,"indexer_budget":2048,"indexer_compress_ratio":4,
        \\ "num_experts":512,"num_experts_per_tok":10,"moe_intermediate_size":640,
        \\ "eos_token_id":248044,"vocab_size":248320,"rms_norm_eps":1e-6,
        \\ "rope_parameters":{"rope_theta":10000000,"partial_rotary_factor":0.25,"mrope_section":[11,11,10],"mrope_interleaved":true}},
        \\ "quantization":{"group_size":64,"bits":4,"mode":"affine"}}
    ;
    const c = try parseConfigFromJson(testing.allocator, json);
    try testing.expect(c.isQwen4() and c.has_vision and c.qwen_vision);
    try testing.expectEqual(@as(u32, 27), c.qv_depth);
    try testing.expectEqual(@as(u32, 1152), c.qv_hidden);
    try testing.expectEqual(@as(u32, 16), c.qv_heads);
    try testing.expectEqual(@as(u32, 72), c.qv_head_dim);
    try testing.expectEqual(@as(u32, 4304), c.qv_intermediate);
    try testing.expectEqual(@as(u32, 16), c.qv_patch);
    try testing.expectEqual(@as(u32, 2), c.qv_temporal_patch);
    try testing.expectEqual(@as(u32, 2), c.qv_merge);
    try testing.expectEqual(@as(u32, 2304), c.qv_num_pos_emb);
    try testing.expectEqual(@as(u32, 2560), c.qv_out_hidden);
    try testing.expect(c.mrope_interleaved);
    try testing.expectEqual([3]u32{ 11, 11, 10 }, c.mrope_section);
    try testing.expectEqual(@as(u32, 248056), c.image_token_id);
    try testing.expectEqual(@as(u32, 248057), c.video_token_id);
    try testing.expectEqual(@as(u32, 248053), c.vision_start_token_id);
    try testing.expectEqual(@as(u32, 248054), c.vision_end_token_id);
}

// ── qwen4_exp YaRN context extension (262144 → 1048576) ──────────────────
//
// Both documents below are the SHIPPED checkpoint's text config (Qwen3.8-Flash-
// Next, `model_type: qwen4_exp`) — one as it ships (plain rope, 262144) and one
// with the YaRN block vLLM's `--hf-overrides` recipe writes. Keeping them as
// literals means the parser is tested against the real file shape, braces and
// all, rather than a synthesized one.

/// The checkpoint as it ships: `rope_type: "default"`, a 262144 window.
const QWEN4_SHIPPED =
    \\{
    \\  "architectures": ["Qwen4ExpForConditionalGeneration"],
    \\  "model_type": "qwen4_exp",
    \\  "text_config": {
    \\    "model_type": "qwen4_exp_text",
    \\    "hidden_size": 2560, "num_hidden_layers": 48, "full_attention_interval": 4,
    \\    "num_attention_heads": 24, "num_key_value_heads": 2, "head_dim": 256,
    \\    "num_experts": 512, "num_experts_per_tok": 10, "moe_intermediate_size": 640,
    \\    "ple_layer_ids": [2],
    \\    "vocab_size": 248320, "eos_token_id": 248044, "max_position_embeddings": 262144,
    \\    "rope_parameters": {
    \\      "rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25,
    \\      "mrope_section": [11, 11, 10], "mrope_interleaved": true
    \\    }
    \\  }
    \\}
;

/// The same checkpoint with its rope scaled 4× and the window widened — exactly
/// `vllm serve ... --hf-overrides '{"text_config": {"rope_parameters": {...}}}'
/// --max-model-len 1010000` expressed as config instead of a flag.
const QWEN4_YARN =
    \\{
    \\  "architectures": ["Qwen4ExpForConditionalGeneration"],
    \\  "model_type": "qwen4_exp",
    \\  "text_config": {
    \\    "model_type": "qwen4_exp_text",
    \\    "hidden_size": 2560, "num_hidden_layers": 48, "full_attention_interval": 4,
    \\    "num_attention_heads": 24, "num_key_value_heads": 2, "head_dim": 256,
    \\    "num_experts": 512, "num_experts_per_tok": 10, "moe_intermediate_size": 640,
    \\    "ple_layer_ids": [2],
    \\    "vocab_size": 248320, "eos_token_id": 248044, "max_position_embeddings": 1048576,
    \\    "rope_parameters": {
    \\      "rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 262144,
    \\      "rope_theta": 10000000, "partial_rotary_factor": 0.25,
    \\      "mrope_section": [11, 11, 10], "mrope_interleaved": true
    \\    }
    \\  }
    \\}
;

test "parseConfigFromJson: the shipped (unscaled) qwen4_exp config is untouched" {
    // The regression guard for every checkpoint that predates the extension:
    // no YaRN, and `contextCap` is just max_position_embeddings, so no server
    // sizing path can shift for a model that did not ask to be scaled.
    const c = try parseConfigFromJson(testing.allocator, QWEN4_SHIPPED);
    try testing.expect(c.isQwen4());
    try testing.expect(!c.rope_yarn);
    try testing.expectEqual(@as(f32, 1.0), c.yarn_factor);
    try testing.expectEqual(@as(u32, 262_144), c.max_position_embeddings);
    try testing.expectEqual(c.max_position_embeddings, c.contextCap());
    try testing.expectApproxEqAbs(@as(f32, 1.0), c.yarn_attention_factor, 1e-9);
    // Same geometry otherwise — YaRN is a rotation, not an architecture change.
    const y = try parseConfigFromJson(testing.allocator, QWEN4_YARN);
    try testing.expectEqual(c.kvBytesPerToken(), y.kvBytesPerToken());
    try testing.expectEqual(c.num_hidden_layers, y.num_hidden_layers);
    try testing.expectEqual(c.head_dim, y.head_dim);
    try testing.expectEqual(c.mrope_section, y.mrope_section);
}

test "parseConfigFromJson: YaRN reads beta_fast/beta_slow/truncate and honours a pinned mscale" {
    defer setConfigOverrides(null);
    // HF's `attention_factor` REPLACES the computed 0.1·ln(factor)+1.
    setConfigOverrides(
        \\{"text_config":{"rope_parameters":{"rope_type":"yarn","factor":4.0,
        \\  "original_max_position_embeddings":262144,"attention_factor":1.25}}}
    );
    const pinned = try parseConfigFromJson(testing.allocator, QWEN4_SHIPPED);
    try testing.expect(pinned.rope_yarn);
    try testing.expectApproxEqAbs(@as(f32, 1.25), pinned.yarn_attention_factor, 1e-9);
    // Still reads theta/partial from the merged block (the base config's values).
    try testing.expectApproxEqAbs(@as(f32, 0.25), pinned.yarnPartial(), 1e-9);
    try testing.expectApproxEqAbs(@as(f32, 10_000_000.0), pinned.rope_theta, 1.0);

    // vLLM's `attn_factor` MULTIPLIES the computed 0.1·ln(factor)+1.
    setConfigOverrides(
        \\{"text_config":{"rope_parameters":{"rope_type":"yarn","factor":4.0,
        \\  "original_max_position_embeddings":262144,"attn_factor":0.5}}}
    );
    const vl = try parseConfigFromJson(testing.allocator, QWEN4_SHIPPED);
    try testing.expectApproxEqAbs(@as(f32, 0.5 * 1.138629436111989), vl.yarn_attention_factor, 1e-6);

    // Both keys present: HF's attention_factor wins (replace, not multiply).
    setConfigOverrides(
        \\{"text_config":{"rope_parameters":{"rope_type":"yarn","factor":4.0,
        \\  "original_max_position_embeddings":262144,
        \\  "attention_factor":1.25,"attn_factor":0.5}}}
    );
    const both = try parseConfigFromJson(testing.allocator, QWEN4_SHIPPED);
    try testing.expectApproxEqAbs(@as(f32, 1.25), both.yarn_attention_factor, 1e-9);

    // The ramp knobs are read too — they move the blend, and so every frequency
    // between the bands.
    setConfigOverrides(
        \\{"text_config":{"rope_parameters":{"rope_type":"yarn","factor":4.0,
        \\  "original_max_position_embeddings":262144,
        \\  "beta_fast":16,"beta_slow":2,"truncate":false}}}
    );
    const tuned = try parseConfigFromJson(testing.allocator, QWEN4_SHIPPED);
    try testing.expectApproxEqAbs(@as(f32, 16.0), tuned.yarn_beta_fast, 1e-9);
    try testing.expectApproxEqAbs(@as(f32, 2.0), tuned.yarn_beta_slow, 1e-9);
    try testing.expect(!tuned.yarn_truncate);
    // With no pinned mscale, the computed default returns.
    try testing.expectApproxEqAbs(@as(f32, 1.138629436111989), tuned.yarn_attention_factor, 1e-6);
}

test "parseConfigFromJson: YaRN derives factor from the window when the block omits it (HF)" {
    defer setConfigOverrides(null);
    // HF: `factor = max_position_embeddings / original_max_position_embeddings`
    // when the block names only the window. Here the override widens the
    // declared window to 2M out of 262144 → factor 8, mscale 0.1·ln 8 + 1.
    setConfigOverrides(
        \\{"text_config":{"max_position_embeddings":2097152,
        \\  "rope_parameters":{"rope_type":"yarn","original_max_position_embeddings":262144}}}
    );
    const c = try parseConfigFromJson(testing.allocator, QWEN4_SHIPPED);
    try testing.expect(c.rope_yarn);
    try testing.expectApproxEqAbs(@as(f32, 8.0), c.yarn_factor, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.2079441541679836), c.yarn_attention_factor, 1e-6);
    try testing.expectEqual(@as(u32, 2_097_152), c.contextCap());
}

test "parseConfigFromJson: YaRN with no pre-trained window, or a zero factor, fails the load" {
    defer setConfigOverrides(null);
    // The ramp bounds come from `original_max_position_embeddings`. Without it
    // every blended frequency is a guess — refuse the load rather than serve a
    // rope that looks fine at short contexts and decays beyond the window.
    setConfigOverrides(
        \\{"text_config":{"rope_parameters":{"rope_type":"yarn","factor":4.0}}}
    );
    try expectError(
        error.YarnRopeNeedsOriginalMaxPos,
        parseConfigFromJson(testing.allocator, QWEN4_SHIPPED),
    );
    // A zero factor is not "no scaling", it is a divide-by-zero waiting to run.
    setConfigOverrides(
        \\{"text_config":{"rope_parameters":{"rope_type":"yarn","factor":0.0,
        \\  "original_max_position_embeddings":262144}}}
    );
    try expectError(
        error.InvalidRopeScalingFactor,
        parseConfigFromJson(testing.allocator, QWEN4_SHIPPED),
    );
}

test "ModelConfig.contextCap: the rope-derived window binds what the server advertises" {
    const scaled = try parseConfigFromJson(testing.allocator, QWEN4_YARN);
    // Over-advertised: a config claiming 2M tokens on a factor-4 ramp out of
    // 262144 still cannot resolve past 1048576 — past there positions alias back
    // inside the window, which is the failure this clamps against.
    var c = scaled;
    c.max_position_embeddings = 2_000_000;
    try testing.expectEqual(@as(u32, 1_048_576), c.contextCap());
    // Under-advertised: serving LESS than the scaled window is legal — the ramp
    // is fixed by the pre-trained length, not by what you choose to run.
    c.max_position_embeddings = 400_000;
    try testing.expectEqual(@as(u32, 400_000), c.contextCap());
    // Declaring nothing: the ramp still says how far the rope reaches.
    c.max_position_embeddings = 0;
    try testing.expectEqual(@as(u32, 1_048_576), c.contextCap());
    // A fractional factor floors (vLLM's `int()` of the same product).
    c.yarn_factor = 3.5;
    try testing.expectEqual(@as(u32, 917_504), c.contextCap()); // floor(262144*3.5)
}

test "parseConfigFromJson: --config-overrides deep-merges a nested block without clobbering siblings" {
    // The merge is what makes the flag usable for rope at all: `rope_parameters`
    // is written as a whole object, and a REPLACE would drop the
    // `partial_rotary_factor` / `mrope_section` keys beside it — silently
    // rotating 256 dims instead of 64, or the wrong axes. vLLM has the same
    // rule (`_update_nested` merges, `_apply_dict_overrides` only replaces
    // non-config values), and the same trap is documented in its source.
    defer setConfigOverrides(null);
    // Pre-override: the shipped config really does have no scaling.
    try testing.expect(!(try parseConfigFromJson(testing.allocator, QWEN4_SHIPPED)).rope_yarn);
    setConfigOverrides(
        \\{"text_config":{"max_position_embeddings":1048576,
        \\  "rope_parameters":{"rope_type":"yarn","factor":4.0,
        \\    "original_max_position_embeddings":262144}}}
    );
    const c = try parseConfigFromJson(testing.allocator, QWEN4_SHIPPED);
    try testing.expect(c.rope_yarn);
    try testing.expectApproxEqAbs(@as(f32, 4.0), c.yarn_factor, 1e-9);
    try testing.expectEqual(@as(u32, 1_048_576), c.contextCap());
    // Keys the override never mentioned survived at BOTH levels of the merge.
    try testing.expectApproxEqAbs(@as(f32, 0.25), c.partial_rotary_factor, 1e-9);
    try testing.expectApproxEqAbs(@as(f32, 10_000_000.0), c.rope_theta, 1.0);
    try testing.expect(c.mrope_interleaved);
    try testing.expectEqual([3]u32{ 11, 11, 10 }, c.mrope_section);
    try testing.expectEqual(@as(u32, 262_144), c.yarn_orig_max_pos);
    // The result is indistinguishable from the hand-written extended config.
    const written = try parseConfigFromJson(testing.allocator, QWEN4_YARN);
    try testing.expectApproxEqAbs(written.yarn_factor, c.yarn_factor, 1e-9);
    try testing.expectEqual(written.contextCap(), c.contextCap());
    try testing.expectApproxEqAbs(written.yarn_attention_factor, c.yarn_attention_factor, 1e-9);
}

test "parseConfigFromJson: --config-overrides replaces scalars and arrays, creates new keys, rejects junk" {
    defer setConfigOverrides(null);
    // Scalars and arrays replace wholesale (vLLM's base case); an array nested
    // in an object that is otherwise merged still replaces the array it meets.
    setConfigOverrides(
        \\{"text_config":{"num_hidden_layers":8,"head_dim":128,
        \\  "rope_parameters":{"mrope_section":[9,9,9]}}}
    );
    const c = try parseConfigFromJson(testing.allocator, QWEN4_YARN);
    try testing.expectEqual(@as(u32, 8), c.num_hidden_layers);
    try testing.expectEqual(@as(u32, 128), c.head_dim);
    try testing.expectEqual([3]u32{ 9, 9, 9 }, c.mrope_section);
    try testing.expect(c.rope_yarn); // the block's other keys survived
    try testing.expectApproxEqAbs(@as(f32, 4.0), c.yarn_factor, 1e-9);

    // A key the document never had is created at the level the parser reads
    // (qwen4's fields come from `text_config`, so that's where it must land).
    setConfigOverrides(
        \\{"text_config":{"ngram_size":5}}
    );
    const n = try parseConfigFromJson(testing.allocator, QWEN4_SHIPPED);
    try testing.expectEqual(@as(u32, 5), n.ngram_size);

    // Only an object is a document; a bare array must not half-apply.
    setConfigOverrides(
        \\[1,2,3]
    );
    try expectError(
        error.ConfigOverridesMustBeObject,
        parseConfigFromJson(testing.allocator, QWEN4_SHIPPED),
    );
    // Clearing the seam restores the shipped document exactly.
    setConfigOverrides(null);
    const clean = try parseConfigFromJson(testing.allocator, QWEN4_SHIPPED);
    try testing.expectEqual(@as(u32, 48), clean.num_hidden_layers);
    try testing.expectEqual(@as(u32, 256), clean.head_dim);
    try testing.expectEqual(@as(u32, 3), clean.ngram_size);
    try testing.expectEqual([3]u32{ 11, 11, 10 }, clean.mrope_section);
    try testing.expect(!clean.rope_yarn);
}

test "ModelConfig.longCtxGated: the long-context blast radius is ONE predicate, qwen4_exp only" {
    const t = std.testing;
    var qwen4 = ModelConfig{ .model_type = "qwen4_exp" };
    try t.expect(qwen4.longCtxGated());
    try t.expect(qwen4.ssdFirstCapable());

    for ([_][]const u8{
        "qwen3_5",
        "qwen3_5_moe",
        "qwen3_next",
        "lfm2",
        "nemotron_h",
        "bailing_hybrid",
        "llama",
        "mistral",
        "gemma3",
        "gemma4",
        "deepseek_v4",
        "muse_glimmer",
    }) |mt| {
        var cfg = ModelConfig{ .model_type = mt };
        try t.expect(!cfg.longCtxGated());
        try t.expect(!cfg.ssdFirstCapable());
    }
}

/// One qwen4_exp config document with `extra` fields spliced in.
fn qwen4CaseJson(comptime extra: []const u8) []const u8 {
    return "{\"model_type\":\"qwen4_exp\",\"hidden_size\":2560,\"num_hidden_layers\":48," ++
        "\"full_attention_interval\":4,\"num_attention_heads\":24,\"num_key_value_heads\":2,\"head_dim\":256," ++
        "\"hc_count\":4,\"hc_lowrank\":320,\"ple_embed_dim\":2560,\"ple_conv_kernel_size\":4," ++
        "\"num_experts\":512,\"num_experts_per_tok\":10,\"moe_intermediate_size\":640," ++
        "\"eos_token_id\":248044,\"vocab_size\":248320,\"rms_norm_eps\":1e-6," ++
        extra ++ "}";
}

const QWEN4_GOOD_FIELDS =
    "\"ple_layer_ids\":[2],\"ngram_size\":3,\"heads_per_ngram\":8," ++
    "\"ngram_vocab_size_base\":20000000,\"make_ngram_vocab_size_divisible_by\":128," ++
    "\"indexer_n_heads\":4,\"indexer_head_dim\":128,\"indexer_budget\":2048,\"indexer_compress_ratio\":4";

test "qwen4_exp config: an n-gram bound past the fixed arrays is a named load error" {
    const good = try parseConfigFromJson(testing.allocator, qwen4CaseJson(QWEN4_GOOD_FIELDS));
    try testing.expectEqual(@as(u32, 3), good.ngram_size);
    try testing.expectEqual(@as(u32, 8), good.heads_per_ngram);

    try expectError(error.InvalidQwen4NgramSize, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[2],\"ngram_size\":9,\"heads_per_ngram\":8"),
    ));
    try expectError(error.InvalidQwen4NgramSize, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[2],\"ngram_size\":1,\"heads_per_ngram\":8"),
    ));
    try expectError(error.InvalidQwen4NgramHeads, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[2],\"ngram_size\":3,\"heads_per_ngram\":0"),
    ));
    try expectError(error.InvalidQwen4NgramHeads, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[2],\"ngram_size\":5,\"heads_per_ngram\":16"),
    ));
    try expectError(error.InvalidQwen4NgramVocab, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[2],\"make_ngram_vocab_size_divisible_by\":0"),
    ));
}

test "n-gram head count overflow is refused by the config" {
    try expectError(error.InvalidQwen4NgramHeads, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[2],\"ngram_size\":3,\"heads_per_ngram\":2147483656"),
    ));
}

test "qwen4_exp config: a wrong-typed or negative bound is a refusal, never a silent default" {
    try expectError(error.InvalidQwen4ConfigField, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[2],\"ngram_size\":-1"),
    ));
    try expectError(error.InvalidQwen4ConfigField, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[2],\"ngram_size\":\"3\""),
    ));
    try expectError(error.InvalidQwen4ConfigField, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[2],\"heads_per_ngram\":3.5"),
    ));
    try expectError(error.InvalidQwen4ConfigField, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[2],\"indexer_compress_ratio\":-4"),
    ));
}

test "qwen4_exp config: an armed QSA indexer must carry a usable budget and ratio" {
    try expectError(error.InvalidQwen4Indexer, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[2],\"indexer_n_heads\":4,\"indexer_head_dim\":128,\"indexer_budget\":2048"),
    ));
    try expectError(error.InvalidQwen4Indexer, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[2],\"indexer_n_heads\":4,\"indexer_head_dim\":128,\"indexer_budget\":2,\"indexer_compress_ratio\":4"),
    ));
    try expectError(error.InvalidQwen4Indexer, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[2],\"indexer_n_heads\":4,\"indexer_budget\":2048,\"indexer_compress_ratio\":4"),
    ));
    const dense = try parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[2],\"ngram_size\":3,\"heads_per_ngram\":8"),
    );
    try testing.expectEqual(@as(u32, 0), dense.indexer_n_heads);
    try testing.expectEqual(@as(u32, 0), dense.indexer_compress_ratio);
}

test "qwen4_exp config: the PLE layer id must name exactly one layer that exists" {
    try expectError(error.InvalidQwen4PleLayer, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ngram_size\":3"),
    ));
    try expectError(error.InvalidQwen4PleLayer, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[0]"),
    ));
    try expectError(error.InvalidQwen4PleLayer, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[49]"),
    ));
    try expectError(error.InvalidQwen4PleLayer, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[2,5]"),
    ));
    try expectError(error.InvalidQwen4PleLayer, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":[]"),
    ));
    try expectError(error.InvalidQwen4PleLayer, parseConfigFromJson(
        testing.allocator,
        qwen4CaseJson("\"ple_layer_ids\":2"),
    ));
    const c = try parseConfigFromJson(testing.allocator, qwen4CaseJson(QWEN4_GOOD_FIELDS));
    try testing.expectEqual(@as(i32, 1), c.ple_layer_idx);
}

test "qwen4 PLE placement: the layer loop must install exactly one PLE, at the configured layer" {
    try testing.expect(qwen4PleInstalledAt(&.{ false, true, false, false }, 1));
    try testing.expect(!qwen4PleInstalledAt(&.{ false, false, false, false }, 1));
    try testing.expect(!qwen4PleInstalledAt(&.{ true, true, false, false }, 1));
    try testing.expect(!qwen4PleInstalledAt(&.{ false, true, false, false }, 2));
    try testing.expect(!qwen4PleInstalledAt(&.{ false, true, false, false }, 4));
    // A negative index is a build that asks for no PLE (`loadQwen4Mtp` sets -1 for the head's layer).
    try testing.expect(qwen4PleInstalledAt(&.{false}, -1));
    try testing.expect(!qwen4PleInstalledAt(&.{true}, -1));
    try testing.expect(!qwen4PleInstalledAt(&.{ false, true, false, false }, -1));
    try testing.expect(qwen4PleInstalledAt(&.{ false, false }, -1));
    // ...while a config that DOES name a layer is unchanged.
    try testing.expect(!qwen4PleInstalledAt(&.{false}, 0));
    try testing.expect(qwen4PleInstalledAt(&.{true}, 0));
}

test "isExpertStreamingArch admits only implemented streaming families" {
    const t = std.testing;
    try t.expect(isExpertStreamingArch("qwen4_exp"));
    try t.expect(isExpertStreamingArch("mimo_v2"));
    for ([_][]const u8{ "qwen4_exp_text", "qwen3_5_moe", "qwen3_5_moe_text", "qwen3_next", "hy_v3", "laguna", "llama", "deepseek_v4", "gguf", "" }) |mt| {
        try t.expect(!isExpertStreamingArch(mt));
        var c = ModelConfig{
            .model_type = mt,
            .num_hidden_layers = 48,
            .num_experts = 512,
            .num_experts_per_tok = 10,
            .hidden_size = 2560,
            .moe_intermediate_size = 640,
        };
        try t.expect(!c.supportsExpertStreaming());
        try t.expect(!c.expertStreamingRequired());
        c.quant_bits = 0;
        try t.expect(!c.expertStreamingRequired());
    }
    var q4 = ModelConfig{
        .model_type = "qwen4_exp",
        .num_hidden_layers = 48,
        .num_experts = 512,
        .num_experts_per_tok = 10,
        .hidden_size = 2560,
        .moe_intermediate_size = 640,
    };
    try t.expect(q4.supportsExpertStreaming() and q4.expertStreamingRequired());
    q4.model_type = "mimo_v2";
    q4.first_k_dense_replace = 1;
    q4.quant_bits = 4;
    try t.expect(q4.supportsExpertStreaming() and !q4.expertStreamingRequired());
    q4.first_k_dense_replace = q4.num_hidden_layers;
    try t.expect(!q4.supportsExpertStreaming());
}

test "the n-gram table source: the bf16 override outranks streaming, streaming outranks the pack table" {
    var c = ModelConfig{ .model_type = "qwen4_exp", .weight_prefix = "language_model.model" };
    try std.testing.expectEqual(ModelConfig.NgramTableSource.quantized, c.ngramTableSource());
    c.expert_streaming = true;
    try std.testing.expectEqual(ModelConfig.NgramTableSource.bf16_streamed, c.ngramTableSource());
    c.expert_layout = .quantized_split;
    try std.testing.expectEqual(ModelConfig.NgramTableSource.quantized, c.ngramTableSource());
    c.expert_layout = .bf16_fused;
    var dir = [_]u8{ '/', 'x' };
    c.ngram_bf16_dir = dir[0..];
    try std.testing.expectEqual(ModelConfig.NgramTableSource.bf16_override, c.ngramTableSource());
    c.expert_streaming = false;
    try std.testing.expectEqual(ModelConfig.NgramTableSource.bf16_override, c.ngramTableSource());
}

test "ModelConfig parses mimo_v2 hybrid geometry and sigmoid routing" {
    const json =
        \\{
        \\  "model_type": "mimo_v2", "hidden_size": 384, "vocab_size": 128,
        \\  "num_hidden_layers": 4, "intermediate_size": 1536,
        \\  "num_attention_heads": 4, "num_key_value_heads": 2,
        \\  "head_dim": 192, "v_head_dim": 128,
        \\  "swa_num_attention_heads": 6, "swa_num_key_value_heads": 3,
        \\  "swa_head_dim": 192, "swa_v_head_dim": 128,
        \\  "hybrid_layer_pattern": [0,1,1,0], "sliding_window": 128,
        \\  "rope_theta": 10000000, "swa_rope_theta": 10000,
        \\  "partial_rotary_factor": 0.334, "attention_value_scale": 0.707,
        \\  "add_swa_attention_sink_bias": true, "add_full_attention_sink_bias": false,
        \\  "attention_projection_layout": "split_qkv", "layernorm_epsilon": 0.00001,
        \\  "n_routed_experts": 16, "num_experts_per_tok": 4,
        \\  "moe_intermediate_size": 192, "moe_layer_freq": [0,1,1,1],
        \\  "scoring_func": "sigmoid", "topk_method": "noaux_tc",
        \\  "n_group": 1, "topk_group": 1, "norm_topk_prob": true,
        \\  "routed_scaling_factor": null, "n_shared_experts": null,
        \\  "eos_token_id": 17, "tie_word_embeddings": false,
        \\  "quantization": {"bits": 4, "group_size": 32, "mode": "mxfp4"}
        \\}
    ;
    const c = try parseConfigFromJson(testing.allocator, json);
    try testing.expectEqualStrings("mimo_v2", c.model_type);
    try testing.expectEqualStrings("model", c.weight_prefix);
    try testing.expect(c.has_explicit_layer_types);
    try testing.expect(c.isGlobalLayer(0) and c.isGlobalLayer(3));
    try testing.expect(!c.isGlobalLayer(1) and !c.isGlobalLayer(2));
    try testing.expectEqual(@as(u32, 4), c.layerNumHeads(0));
    try testing.expectEqual(@as(u32, 6), c.layerNumHeads(1));
    try testing.expectEqual(@as(u32, 2), c.layerKVHeads(0));
    try testing.expectEqual(@as(u32, 3), c.layerKVHeads(1));
    try testing.expectEqual(@as(u32, 192), c.layerHeadDim(0));
    try testing.expectEqual(@as(u32, 128), c.layerVHeadDim(0));
    try testing.expectEqual(@as(u32, 128), c.layerVHeadDim(1));
    try testing.expect(!c.layerHasAttnSinks(0) and c.layerHasAttnSinks(1));
    try testing.expectEqual(@as(f32, 0.707), c.attention_value_scale);
    try testing.expect(!c.attn_fused_qkv);
    try testing.expectEqual(@as(f32, 0.334), c.partial_rotary_factor);
    try testing.expectEqual(@as(f32, 1e7), c.rope_theta);
    try testing.expectEqual(@as(f32, 1e4), c.rope_local_base_freq);
    try testing.expectEqual(@as(f32, 1e-5), c.rms_norm_eps);
    try testing.expectEqual(@as(u32, 16), c.num_experts);
    try testing.expectEqual(@as(u32, 1), c.first_k_dense_replace);
    try testing.expect(c.moe_sigmoid_router and c.moe_route_norm);
    try testing.expectEqual(@as(f32, 1), c.router_scaling_factor);
    try testing.expectEqual(QuantMode.mxfp4, c.quant_mode);
    try testing.expect(!c.norm_has_offset and !c.scale_embeddings and !c.has_qk_norm);
    try testing.expect(!c.has_pre_ff_norm and !c.has_vision);
    try testing.expect(c.isEosToken(17));
}

const MIMO_V2_VISION_JSON =
    \\{
    \\  "model_type": "mimo_v2", "hidden_size": 384, "vocab_size": 128,
    \\  "num_hidden_layers": 2, "intermediate_size": 1536,
    \\  "num_attention_heads": 4, "num_key_value_heads": 2,
    \\  "head_dim": 192, "v_head_dim": 128,
    \\  "hybrid_layer_pattern": [0,1], "sliding_window": 128,
    \\  "partial_rotary_factor": 0.334, "moe_layer_freq": [0,1],
    \\  "n_routed_experts": 16, "num_experts_per_tok": 4, "moe_intermediate_size": 192,
    \\  "image_token_id": 101, "video_token_id": 102, "audio_token_id": 103,
    \\  "vision_start_token_id": 104, "vision_end_token_id": 105,
    \\  "vision_config": {
    \\    "depth": 4, "hidden_size": 64, "num_heads": 4, "num_key_value_heads": 2,
    \\    "intermediate_size": 96, "out_hidden_size": 384, "patch_size": 16,
    \\    "spatial_merge_size": 2, "temporal_patch_size": 2, "use_sink": true,
    \\    "fullatt_block_indexes": [0, 3], "vit_window_attn_types": [-1, 0, 1, -1],
    \\    "visual_token_window_size": 64, "window_size": 128
    \\  },
    \\  "processor_config": {"image_min_pixels": 8192, "image_max_pixels": 8388608}
    \\}
;

test "mimo_v2 config reads the MiMo-ViT geometry and the processor's own pixel bounds" {
    const c = try parseConfigFromJson(testing.allocator, MIMO_V2_VISION_JSON);
    try testing.expect(c.has_vision and c.mimo_vision and !c.qwen_vision);
    try testing.expectEqual(@as(u32, 4), c.qv_depth);
    try testing.expectEqual(@as(u32, 64), c.qv_hidden);
    try testing.expectEqual(@as(u32, 4), c.qv_heads);
    // The reference's `qk_channels` default, not hidden / heads.
    try testing.expectEqual(@as(u32, 64), c.qv_head_dim);
    try testing.expectEqual(@as(u32, 2), c.mvit_kv_heads);
    try testing.expectEqual(@as(u32, 96), c.qv_intermediate);
    try testing.expectEqual(@as(u32, 384), c.qv_out_hidden);
    try testing.expectEqual(@as(u32, 16), c.qv_patch);
    try testing.expectEqual(@as(u32, 2), c.qv_merge);
    try testing.expectEqual(@as(u32, 2), c.qv_temporal_patch);
    try testing.expectEqual(@as(u32, 64), c.mvit_window);
    try testing.expect(c.mvit_sinks);
    try testing.expectEqualSlices(MimoVitAttn, &.{ .full, .row, .col, .full }, c.mvit_attn[0..4]);
    try testing.expectEqual(@as(u32, 8192), c.qv_min_pixels);
    try testing.expectEqual(@as(u32, 8388608), c.qv_max_pixels);
    try testing.expectEqual(@as(u32, 101), c.image_token_id);
    try testing.expectEqual(@as(u32, 104), c.vision_start_token_id);
    try testing.expectEqual(@as(u32, 105), c.vision_end_token_id);
    // Video and audio are not served yet: their placeholders must never join the splice.
    try testing.expectEqual(@as(u32, 0), c.video_token_id);
}

test "mimo_v2 config refuses a MiMo-ViT whose block tables disagree with its depth" {
    const cases = [_][]const u8{
        "\"fullatt_block_indexes\": [0, 4], \"vit_window_attn_types\": [-1, 0, 1, -1]",
        "\"fullatt_block_indexes\": [0, 3], \"vit_window_attn_types\": [-1, 0, 1]",
        "\"fullatt_block_indexes\": [0, 3], \"vit_window_attn_types\": [-1, 0, 2, -1]",
    };
    for (cases) |tables| {
        const json = try std.mem.replaceOwned(u8, testing.allocator, MIMO_V2_VISION_JSON, "\"fullatt_block_indexes\": [0, 3], \"vit_window_attn_types\": [-1, 0, 1, -1]", tables);
        defer testing.allocator.free(json);
        try expectError(error.UnsupportedMimoV2Config, parseConfigFromJson(testing.allocator, json));
    }
}

test "mimo_v2 config rejects unsupported routing and malformed layer geometry" {
    const base =
        \\{"model_type":"mimo_v2", "num_hidden_layers":2, "hidden_size":384,
        \\ "num_attention_heads":4, "num_key_value_heads":2, "head_dim":192,
        \\ "v_head_dim":128, "partial_rotary_factor":0.334,
        \\ "hybrid_layer_pattern":[0,1], "moe_layer_freq":[0,1],
        \\ "n_routed_experts":16, "num_experts_per_tok":4, "moe_intermediate_size":192}
    ;
    const good = try parseConfigFromJson(testing.allocator, base);
    try testing.expectEqualStrings("mimo_v2", good.model_type);
    for ([_][]const u8{
        \\{"scoring_func":"softmax"}
        ,
        \\{"topk_method":"greedy"}
        ,
        \\{"hidden_act":"gelu"}
        ,
        \\{"n_shared_experts":1}
        ,
        \\{"hybrid_layer_pattern":[0]}
        ,
        \\{"hybrid_layer_pattern":[0,2]}
        ,
        \\{"moe_layer_freq":[1,0]}
        ,
        \\{"moe_layer_freq":2}
        ,
        \\{"swa_num_attention_heads":3}
        ,
        \\{"swa_v_head_dim":0}
        ,
        \\{"partial_rotary_factor":0.34}
        ,
        \\{"partial_rotary_factor":2}
        ,
        \\{"attention_projection_layout":"interleaved"}
        ,
        \\{"sliding_window":0}
        ,
        \\{"n_group":0}
        ,
        \\{"n_group":3}
        ,
        \\{"n_group":4,"topk_group":5}
        ,
        \\{"n_group":16,"topk_group":4}
        ,
        \\{"layernorm_epsilon":0}
        ,
    }) |override| {
        const json = try mergeConfigJson(testing.allocator, base, override);
        defer testing.allocator.free(json);
        try expectError(error.UnsupportedMimoV2Config, parseConfigFromJson(testing.allocator, json));
    }
    const fused_json = try mergeConfigJson(testing.allocator, base,
        \\{"attention_projection_layout":"fused_qkv","routed_scaling_factor":2.5,
        \\ "norm_topk_prob":false,"add_swa_attention_sink_bias":false,
        \\ "add_full_attention_sink_bias":true}
    );
    defer testing.allocator.free(fused_json);
    const fused = try parseConfigFromJson(testing.allocator, fused_json);
    try testing.expect(fused.attn_fused_qkv and !fused.moe_route_norm);
    try testing.expectEqual(@as(f32, 2.5), fused.router_scaling_factor);
    try testing.expect(fused.layerHasAttnSinks(0) and !fused.layerHasAttnSinks(1));
}

test "mimo_v2 refuses a config asking to quantize trunk linears at load" {
    const base =
        \\{"model_type":"mimo_v2", "num_hidden_layers":2, "hidden_size":384,
        \\ "num_attention_heads":4, "num_key_value_heads":2, "head_dim":192,
        \\ "v_head_dim":128, "partial_rotary_factor":0.334,
        \\ "hybrid_layer_pattern":[0,1], "moe_layer_freq":[0,1],
        \\ "n_routed_experts":16, "num_experts_per_tok":4, "moe_intermediate_size":192}
    ;
    _ = try parseConfigFromJson(testing.allocator, base);
    const json = try mergeConfigJson(testing.allocator, base,
        \\{"trunk_quant":{"o_proj":{"mode":"affine","bits":8,"group_size":64}}}
    );
    defer testing.allocator.free(json);
    try expectError(error.UnsupportedMimoV2Config, parseConfigFromJson(testing.allocator, json));
}

test "layer value width and sink placement preserve existing defaults" {
    var c = ModelConfig{ .has_attn_sinks = true };
    try testing.expectEqual(c.layerHeadDim(0), c.layerVHeadDim(0));
    try testing.expect(c.layerHasAttnSinks(0));
    c.has_sliding_window = false;
    c.global_head_dim = 128;
    try testing.expectEqual(@as(u32, 128), c.layerVHeadDim(0));
    try testing.expect(c.layerHasAttnSinks(0));
    c.has_attn_sinks = false;
    try testing.expect(!c.layerHasAttnSinks(0));
}

test "real mimo_v2 Flash config agrees with source geometry" {
    const raw = std.c.getenv("MIMO_V2_SOURCE") orelse return error.SkipZigTest;
    var c = try parseConfig(testing.io, testing.allocator, std.mem.span(raw));
    defer c.deinit(testing.allocator);
    try testing.expectEqualStrings("mimo_v2", c.model_type);
    try testing.expectEqual(@as(u32, 48), c.num_hidden_layers);
    try testing.expectEqual(@as(u32, 256), c.num_experts);
    try testing.expectEqual(@as(u32, 8), c.num_experts_per_tok);
    try testing.expectEqual(@as(u32, 1), c.first_k_dense_replace);
    var global: u32 = 0;
    for (0..c.num_hidden_layers) |i| {
        const li: u32 = @intCast(i);
        global += @intFromBool(c.isGlobalLayer(li));
        try testing.expectEqual(@as(u32, 64), c.layerNumHeads(li));
        try testing.expectEqual(@as(u32, if (c.isGlobalLayer(li)) 4 else 8), c.layerKVHeads(li));
        try testing.expectEqual(@as(u32, 192), c.layerHeadDim(li));
        try testing.expectEqual(@as(u32, 128), c.layerVHeadDim(li));
        try testing.expectEqual(!c.isGlobalLayer(li), c.layerHasAttnSinks(li));
    }
    try testing.expectEqual(@as(u32, 9), global);
    // What the session costs: 9 global layers per token, the 39 sliding ones a
    // ring held once per slot.
    try testing.expectEqual(@as(u64, 9 * 4 * (192 + 128) * 2), c.kvBytesPerToken());
    try testing.expectEqual(@as(u64, 128) + ModelConfig.SWA_RING_SLACK, c.swaRingTokens());
    try testing.expectEqual(c.swaRingTokens() * 39 * 8 * (192 + 128) * 2, c.swaRingBytes());
    try testing.expectEqual(@as(f32, 0.707), c.attention_value_scale);
    try testing.expect(c.isEosToken(151643) and c.isEosToken(151645) and c.isEosToken(151672));
    try testing.expect(!c.attn_fused_qkv);
    try testing.expectEqual(QuantMode.mxfp4, c.quant_mode);
    try testing.expectEqual(@as(u32, 4), c.quant_bits);
    try testing.expectEqual(@as(u32, 32), c.quant_group_size);
    try testing.expect(c.expertStreamingRequired());

    try testing.expect(c.mimo_vision);
    try testing.expectEqual(@as(u32, 28), c.qv_depth);
    try testing.expectEqual(@as(u32, 1280), c.qv_hidden);
    try testing.expectEqual(@as(u32, 32), c.qv_heads);
    try testing.expectEqual(@as(u32, 8), c.mvit_kv_heads);
    try testing.expectEqual(@as(u32, 64), c.qv_head_dim);
    try testing.expectEqual(@as(u32, 4608), c.qv_intermediate);
    try testing.expectEqual(@as(u32, 64), c.mvit_window);
    var full: u32 = 0;
    for (c.mvit_attn[0..c.qv_depth]) |kind| full += @intFromBool(kind == .full);
    try testing.expectEqual(@as(u32, 4), full);
    try testing.expect(c.mvit_attn[0] == .full and c.mvit_attn[27] == .full and c.mvit_attn[1] == .row and c.mvit_attn[5] == .col);
    try testing.expectEqual(@as(u32, 8192), c.qv_min_pixels);
    try testing.expectEqual(@as(u32, 8388608), c.qv_max_pixels);
    try testing.expectEqual(@as(u32, 151655), c.image_token_id);
}

/// The mimo_v2 geometry of "ModelConfig parses mimo_v2 hybrid geometry", kept
/// beside the bill assertions so the expected bytes read against one spelling.
const MIMO_V2_BILL_JSON =
    \\{
    \\  "model_type": "mimo_v2", "hidden_size": 384, "vocab_size": 128,
    \\  "num_hidden_layers": 4, "intermediate_size": 1536,
    \\  "num_attention_heads": 4, "num_key_value_heads": 2,
    \\  "head_dim": 192, "v_head_dim": 128,
    \\  "swa_num_attention_heads": 6, "swa_num_key_value_heads": 3,
    \\  "swa_head_dim": 192, "swa_v_head_dim": 128,
    \\  "hybrid_layer_pattern": [0,1,1,0], "sliding_window": 128,
    \\  "rope_theta": 10000000, "swa_rope_theta": 10000,
    \\  "partial_rotary_factor": 0.334, "attention_value_scale": 0.707,
    \\  "add_swa_attention_sink_bias": true, "add_full_attention_sink_bias": false,
    \\  "attention_projection_layout": "split_qkv", "layernorm_epsilon": 0.00001,
    \\  "n_routed_experts": 16, "num_experts_per_tok": 4,
    \\  "moe_intermediate_size": 192, "moe_layer_freq": [0,1,1,1],
    \\  "scoring_func": "sigmoid", "topk_method": "noaux_tc",
    \\  "n_group": 1, "topk_group": 1, "norm_topk_prob": true,
    \\  "routed_scaling_factor": null, "n_shared_experts": null,
    \\  "eos_token_id": 17, "tie_word_embeddings": false,
    \\  "quantization": {"bits": 4, "group_size": 32, "mode": "mxfp4"}
    \\}
;

test "mimo_v2 bills per-layer KV geometry and the sliding window once per slot" {
    const c = try parseConfigFromJson(testing.allocator, MIMO_V2_BILL_JSON);
    // Global layers 0 and 3 at 2 KV heads x (qk 192 + v 128) x 2 bytes. The
    // sliding pair contributes NOTHING per token — its storage is a ring.
    try testing.expectEqual(@as(u64, 2 * 2 * (192 + 128) * 2), c.kvBytesPerToken());
    // The ring: two sliding layers at 3 KV heads x 320 x 2 bytes, held for
    // `swaRingTokens` rows however long the session runs.
    try testing.expectEqual(@as(u64, 128) + ModelConfig.SWA_RING_SLACK, c.swaRingTokens());
    try testing.expectEqual(c.swaRingTokens() * 2 * 3 * (192 + 128) * 2, c.swaRingBytes());
    // A restore point is the window plus the backoff, per sliding layer, at the same width.
    try testing.expectEqual(@as(u64, 128) + ModelConfig.SWA_RING_CHECKPOINT_BACKOFF, c.swaRingCheckpointTokens());
    try testing.expectEqual(c.swaRingCheckpointTokens() * 2 * 3 * (192 + 128) * 2, c.swaRingCheckpointBytes());
    // What one chunk token stages in those layers before the ring compacts,
    // for as many of them as one eval-cadence window lets coexist.
    try testing.expectEqual(@as(u64, 2 * 3 * (192 + 128) * 2), c.swaStreamBytesPerToken(5));
    try testing.expectEqual(@as(u64, 1 * 3 * (192 + 128) * 2), c.swaStreamBytesPerToken(1));
    try testing.expectEqual(@as(u64, 0), c.swaStreamBytesPerToken(0));
}

test "a non-ringing sliding arch keeps the uniform KV bill" {
    // gemma4 slides too, but stores every layer full-length, so its bill must
    // stay the uniform `layers x kv_heads x 2*head_dim x 2`: per-layer geometry
    // would read `global_head_dim` 512 and move a number whose bytes are still
    // there.
    var g = ModelConfig{};
    g.model_type = "gemma4";
    g.num_hidden_layers = 48;
    g.num_key_value_heads = 8;
    g.head_dim = 256;
    g.global_head_dim = 512;
    g.has_sliding_window = true;
    g.has_explicit_layer_types = true;
    g.layer_is_global[2] = true;
    try testing.expectEqual(@as(u64, 0), g.swaRingTokens());
    try testing.expectEqual(@as(u64, 0), g.swaRingBytes());
    try testing.expectEqual(@as(u64, 0), g.swaRingCheckpointBytes());
    try testing.expectEqual(@as(u64, 0), g.swaStreamBytesPerToken(5));
    try testing.expectEqual(@as(u64, 48 * 8 * 2 * 256 * 2), g.kvBytesPerToken());

    // qwen4_exp: 12 caching layers of 48, uniform geometry, no ring.
    const q = try parseConfigFromJson(testing.allocator, QWEN4_YARN);
    try testing.expectEqual(@as(u64, 0), q.swaRingTokens());
    try testing.expectEqual(@as(u64, 24_576), q.kvBytesPerToken());
}

test "real mimo_v2 original and converted packs bill the same resident trunk" {
    const source = std.c.getenv("MIMO_V2_SOURCE") orelse return error.SkipZigTest;
    const reference = std.c.getenv("MIMO_PACK_REFERENCE") orelse return error.SkipZigTest;
    var config = try parseConfig(testing.io, testing.allocator, std.mem.span(source));
    defer config.deinit(testing.allocator);
    try testing.expectEqual(expert_quant.Layout.mxfp4_individual, config.expert_layout);
    const original = try streamingResidentSplit(testing.io, testing.allocator, std.mem.span(source), &config);
    const converted = try streamingResidentSplit(testing.io, testing.allocator, std.mem.span(reference), &.{ .expert_layout = .mxfp4_split });
    try testing.expect(original.trunk > 0);
    try testing.expectEqual(converted, original);
}

test "real mimo_v2 bill carries the bf16 vision tower exactly when it is loaded" {
    const source = std.c.getenv("MIMO_V2_SOURCE") orelse return error.SkipZigTest;
    const dir = std.mem.span(source);
    const text = try mimoSourceResidentBytes(testing.io, testing.allocator, dir, false);
    const with_tower = try mimoSourceResidentBytes(testing.io, testing.allocator, dir, true);
    // 364 `visual.*` tensors, all bf16, as stored.
    try testing.expectEqual(@as(u64, 1_457_188_864), with_tower - text);
}

test "mimo_v2 original config selects split QKV and native expert quantization" {
    const a = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const raw = try expert_quant.writeTinyMxfp4IndividualCheckpoint(a, tmp.dir, 4, 128, 128);
    defer a.free(raw);
    try tmp.dir.writeFile(testing.io, .{
        .sub_path = "config.json",
        .data =
        \\{"model_type":"mimo_v2","num_hidden_layers":2,"hidden_size":128,
        \\ "num_attention_heads":4,"num_key_value_heads":2,"head_dim":32,
        \\ "v_head_dim":32,"swa_head_dim":32,"swa_v_head_dim":32,
        \\ "swa_num_attention_heads":4,"swa_num_key_value_heads":2,
        \\ "hybrid_layer_pattern":[0,1],"moe_layer_freq":[0,1],
        \\ "n_routed_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":128,
        \\ "attention_projection_layout":"fused_qkv",
        \\ "quantization_config":{"quant_method":"fp8","store_dtype":"mxfp4"}}
        ,
    });
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(testing.io, &path_buf);
    var c = try parseConfig(testing.io, a, path_buf[0..n]);
    defer c.deinit(a);
    try testing.expectEqual(expert_quant.Layout.mxfp4_individual, c.expert_layout);
    try testing.expect(!c.attn_fused_qkv);
    try testing.expectEqual(QuantMode.mxfp4, c.quant_mode);
    try testing.expectEqual(@as(u32, 4), c.quant_bits);
    try testing.expectEqual(@as(u32, 32), c.quant_group_size);
    try testing.expect(c.expertStreamingRequired());
}

test "mimo_v2 EXL3 routed banks stream on request and take the source trunk loader" {
    var c = ModelConfig{
        .model_type = "mimo_v2",
        .num_hidden_layers = 2,
        .first_k_dense_replace = 1,
        .num_experts = 4,
        .num_experts_per_tok = 2,
        .hidden_size = 128,
        .moe_intermediate_size = 128,
        .quant_bits = 0,
        .expert_layout = .exl3_k4,
    };
    try testing.expect(c.supportsExpertStreaming());
    try testing.expect(!c.expertStreamingRequired());
    try testing.expect(c.usesMimoSourceTrunk());
    c.expert_layout = .mxfp4_individual;
    try testing.expect(c.expertStreamingRequired());
    try testing.expect(c.usesMimoSourceTrunk());
    c.expert_layout = .mxfp4_split;
    try testing.expect(!c.usesMimoSourceTrunk());
    var q = ModelConfig{
        .model_type = "qwen4_exp",
        .num_hidden_layers = 2,
        .num_experts = 4,
        .num_experts_per_tok = 2,
        .hidden_size = 128,
        .moe_intermediate_size = 128,
        .quant_bits = 0,
        .expert_layout = .exl3_k4,
    };
    try testing.expect(!q.usesMimoSourceTrunk());
    try testing.expect(!q.expertStreamingRequired());
}

test "mimo_v2 original MXFP4 experts require streaming despite their quantized width" {
    var c = ModelConfig{
        .model_type = "mimo_v2",
        .num_hidden_layers = 2,
        .first_k_dense_replace = 1,
        .num_experts = 4,
        .num_experts_per_tok = 2,
        .hidden_size = 128,
        .moe_intermediate_size = 128,
        .quant_bits = 4,
        .quant_group_size = 32,
        .quant_mode = .mxfp4,
        .expert_layout = .mxfp4_individual,
    };
    try testing.expect(c.expertStreamingRequired());
    c.expert_layout = .mxfp4_split;
    try testing.expect(!c.expertStreamingRequired());
}

test "mimo_v2 original streaming retains resident names and excludes experts and MTP" {
    var buf: [256]u8 = undefined;
    const key = "model.layers.0.self_attn.qkv_proj.weight";
    try testing.expectEqualStrings(key, qwen4StreamingWeightKey(.mxfp4_individual, &buf, key).?);
    try testing.expect(qwen4StreamingWeightKey(.mxfp4_individual, &buf, "model.layers.1.mlp.experts.0.gate_proj.weight") == null);
    try testing.expect(qwen4StreamingWeightKey(.mxfp4_individual, &buf, "model.layers.1.mlp.experts.0.gate_proj.weight_scale") == null);
    try testing.expect(qwen4StreamingWeightKey(.mxfp4_individual, &buf, "model.mtp.layers.0.self_attn.qkv_proj.weight") == null);
}

test "mimo_v2 streaming leaves trunk keys intact and excludes only routed banks" {
    var buf: [256]u8 = undefined;
    for ([_][]const u8{
        "lm_head.weight",
        "model.embed_tokens.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.1.mlp.gate.e_score_correction_bias",
    }) |key| {
        try testing.expectEqualStrings(key, qwen4StreamingWeightKey(.mxfp4_split, &buf, key).?);
    }
    try testing.expect(qwen4StreamingWeightKey(.mxfp4_split, &buf, "model.layers.1.mlp.switch_mlp.gate_proj.weight") == null);
    try testing.expect(qwen4StreamingWeightKey(.mxfp4_split, &buf, "model.layers.1.mlp.switch_mlp.down_proj.scales") == null);
}

test "the loader refuses a model_type this build does not serve, by name, before reading the checkpoint" {
    const missing = "/nonexistent/sushi-arch-gate";
    const llama = ModelConfig{ .model_type = "llama" };
    try std.testing.expectError(error.ArchitectureUnsupported, loadWeightsForConfig(std.testing.io, std.testing.allocator, missing, &llama, false));
    // The served archs pass the gate and fail on the missing directory instead.
    for ([_][]const u8{ "qwen4_exp", "mimo_v2" }) |mt| {
        const cfg = ModelConfig{ .model_type = mt };
        if (loadWeightsForConfig(std.testing.io, std.testing.allocator, missing, &cfg, false)) |w| {
            var owned = w;
            owned.deinit();
            return error.TestUnexpectedResult;
        } else |err| try std.testing.expect(err != error.ArchitectureUnsupported);
    }
}

test "parseConfig releases owned paths when an EXL3 pack is refused" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(testing.io, .{
        .sub_path = "config.json",
        .data = qwen4CaseJson(QWEN4_GOOD_FIELDS),
    });
    var index: std.ArrayList(u8) = .empty;
    defer index.deinit(testing.allocator);
    try index.appendSlice(testing.allocator, "{\"weight_map\":{");
    for (0..48) |layer| {
        for ([_][]const u8{ "gate", "up", "down" }) |projection| {
            for ([_][]const u8{ "trellis", "suh", "svh" }) |part| {
                if (index.items[index.items.len - 1] != '{') try index.append(testing.allocator, ',');
                const item = try std.fmt.allocPrint(testing.allocator, "\"language_model.model.layers.{d}.mlp.switch_mlp.{s}_proj.{s}\":\"experts.safetensors\"", .{ layer, projection, part });
                defer testing.allocator.free(item);
                try index.appendSlice(testing.allocator, item);
            }
        }
    }
    try index.appendSlice(testing.allocator, "}}");
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "model.safetensors.index.json", .data = index.items });
    var path: [std.fs.max_path_bytes]u8 = undefined;
    const len = try tmp.dir.realPath(testing.io, &path);
    try expectError(error.ExpertLayoutUnsupported, parseConfig(testing.io, testing.allocator, path[0..len]));
}

test "MiMo EXL3 streaming CPU accepts budgets and preserves the resident default" {
    const stream = @import("expert_stream.zig");
    const c = ModelConfig{ .model_type = "mimo_v2", .num_hidden_layers = 48, .first_k_dense_replace = 1, .num_experts = 256, .num_experts_per_tok = 8, .hidden_size = 4096, .moe_intermediate_size = 2048, .expert_layout = .exl3_k4 };
    try testing.expect(c.supportsExpertStreaming());
    try testing.expectEqual(@as(u32, 47), c.expertLayerCount());
    try testing.expect(!stream.expertStreamingEngaged(c.supportsExpertStreaming(), c.expertStreamingRequired(), 0, 0));
    try testing.expect(stream.expertStreamingEngaged(c.supportsExpertStreaming(), c.expertStreamingRequired(), 0, 20 << 30));
    try testing.expectEqual(stream.MtpUnderStreaming.refuse, stream.mtpUnderStreaming(true, false, false, false));
    try testing.expectEqual(stream.MtpUnderStreaming.drop_settings, stream.mtpUnderStreaming(true, true, false, false));
    try testing.expectEqual(stream.MtpUnderStreaming.drop_default, stream.mtpUnderStreaming(true, false, true, false));
}

test "parseConfigFromJson: a wrong-typed or out-of-range field is a named error, never a bare read" {
    const t = testing;
    for ([_][]const u8{
        "[]",
        "17",
        "\"qwen4_exp\"",
        "null",
        "{\"model_type\":5}",
        "{\"model_type\":\"qwen4_exp\",\"text_config\":7}",
        "{\"model_type\":\"llama\",\"hidden_size\":\"4096\"}",
        "{\"model_type\":\"llama\",\"vocab_size\":-1}",
        "{\"model_type\":\"llama\",\"num_hidden_layers\":4294967296}",
        "{\"model_type\":\"llama\",\"rms_norm_eps\":\"1e-6\"}",
        "{\"model_type\":\"llama\",\"quantization\":[]}",
        "{\"model_type\":\"llama\",\"quantization\":{\"bits\":\"4\"}}",
        "{\"model_type\":\"llama\",\"eos_token_id\":-2}",
        "{\"model_type\":\"llama\",\"eos_token_id\":[1,-2]}",
        "{\"model_type\":\"llama\",\"num_experts\":-1}",
        "{\"model_type\":\"llama\",\"bos_token_id\":4294967296}",
        "{\"model_type\":\"llama\",\"image_token_id\":-7}",
        qwen4CaseJson(QWEN4_GOOD_FIELDS ++ ",\"vision_config\":{\"depth\":-1}"),
        qwen4CaseJson(QWEN4_GOOD_FIELDS ++ ",\"rope_parameters\":{\"rope_type\":\"yarn\",\"original_max_position_embeddings\":-1}"),
        qwen4CaseJson(QWEN4_GOOD_FIELDS ++ ",\"rope_parameters\":{\"rope_type\":\"yarn\",\"original_max_position_embeddings\":262144,\"factor\":\"4\"}"),
    }) |doc| {
        try expectError(error.InvalidConfigField, parseConfigFromJson(t.allocator, doc));
    }
}

test "qwen4_exp config: a geometry the forward divides by or indexes with is a named error" {
    const t = testing;
    const base = "\"model_type\":\"qwen4_exp\",\"hidden_size\":2560,\"num_hidden_layers\":48,\"head_dim\":256," ++
        "\"hc_count\":4,\"hc_lowrank\":320,\"ple_embed_dim\":2560,\"moe_intermediate_size\":640," ++
        "\"vocab_size\":248320," ++ QWEN4_GOOD_FIELDS;
    for ([_][]const u8{
        "{" ++ base ++ ",\"full_attention_interval\":0,\"num_attention_heads\":24,\"num_key_value_heads\":2,\"num_experts\":512,\"num_experts_per_tok\":10}",
        "{" ++ base ++ ",\"full_attention_interval\":4,\"num_attention_heads\":24,\"num_key_value_heads\":0,\"num_experts\":512,\"num_experts_per_tok\":10}",
        "{" ++ base ++ ",\"full_attention_interval\":4,\"num_attention_heads\":24,\"num_key_value_heads\":5,\"num_experts\":512,\"num_experts_per_tok\":10}",
        "{" ++ base ++ ",\"full_attention_interval\":4,\"num_attention_heads\":24,\"num_key_value_heads\":2,\"num_experts\":8,\"num_experts_per_tok\":10}",
    }) |doc| {
        try expectError(error.InvalidQwen4Geometry, parseConfigFromJson(t.allocator, doc));
    }
}

test "parseConfigFromJson: an optional field set to null keeps its default; sliding_window null still disables" {
    const t = testing;
    const c = try parseConfigFromJson(t.allocator, qwen4CaseJson(QWEN4_GOOD_FIELDS ++
        ",\"max_position_embeddings\":null,\"bos_token_id\":null,\"sliding_window\":null,\"image_token_id\":null"));
    const d = ModelConfig{};
    try t.expectEqual(d.max_position_embeddings, c.max_position_embeddings);
    try t.expectEqual(@as(?u32, null), c.bos_token_id);
    try t.expect(!c.has_sliding_window);
    try t.expectEqual(@as(u32, 0), c.image_token_id);
}

test "the shipped packs' configs parse to the geometry they serve (src/fixtures/model-configs)" {
    // Cut from Qwen3.8-Flash-Next-Sushi-3bpw and MiMo-V2.6-Flash-Sushi-2.3bpw. MiMo's `expert_quant.k` 4 is the
    // widest rate a layer packs, the one the engine bills: its last MoE layer is K4, the rest K2.25.
    const t = testing;
    const q = try parseConfigFromJson(t.allocator, @embedFile("fixtures/model-configs/qwen4_exp.json"));
    try t.expectEqualStrings("qwen4_exp", q.model_type);
    try t.expectEqual(@as(u32, 248320), q.vocab_size);
    try t.expectEqual(@as(u32, 2560), q.hidden_size);
    try t.expectEqual(@as(u32, 48), q.num_hidden_layers);
    try t.expectEqual(@as(u32, 24), q.num_attention_heads);
    try t.expectEqual(@as(u32, 2), q.num_key_value_heads);
    try t.expectEqual(@as(u32, 256), q.head_dim);
    try t.expectEqual(@as(u32, 1048576), q.max_position_embeddings);
    try t.expectEqual(@as(f32, 1e-6), q.rms_norm_eps);
    try t.expectEqual(@as(f32, 10_000_000), q.rope_theta);
    try t.expectEqual(@as(f32, 0.25), q.partial_rotary_factor);
    try t.expectEqual(@as(u32, 512), q.num_experts);
    try t.expectEqual(@as(u32, 10), q.num_experts_per_tok);
    try t.expectEqual(@as(u32, 640), q.moe_intermediate_size);
    try t.expectEqual(@as(u32, 640), q.shared_expert_intermediate_size);
    try t.expectEqual(@as(u32, 16), q.linear_num_key_heads);
    try t.expectEqual(@as(u32, 48), q.linear_num_value_heads);
    try t.expectEqual(@as(u32, 4), q.full_attention_interval);
    try t.expectEqual(@as(u32, 8), q.quant_bits);
    try t.expectEqual(@as(u32, 64), q.quant_group_size);
    try t.expectEqualSlices(u32, &.{248044}, q.eosTokenSlice());
    try t.expectEqual(@as(?u32, 248044), q.bos_token_id);
    try t.expectEqual(@as(u32, 248056), q.image_token_id);
    try t.expectEqual(@as(u32, 248057), q.video_token_id);
    try t.expectEqual(@as(u32, 27), q.qv_depth);
    try t.expectEqual(@as(u32, 1152), q.qv_hidden);
    try t.expectEqual([3]u32{ 11, 11, 10 }, q.mrope_section);
    try t.expect(q.rope_yarn);
    try t.expectEqual(@as(f32, 4), q.yarn_factor);
    try t.expectEqual(@as(u32, 262144), q.yarn_orig_max_pos);
    try t.expectEqual(@as(u32, 3), q.ngram_size);
    try t.expectEqual(@as(u32, 8), q.heads_per_ngram);
    try t.expectEqual(@as(u64, 20_000_000), q.ngram_vocab_base);
    try t.expectEqual(@as(u32, 2048), q.indexer_budget);
    try t.expectEqual(@as(i32, 1), q.ple_layer_idx);
    try t.expectEqual(@as(u32, 248044), q.ngram_eos);

    const m = try parseConfigFromJson(t.allocator, @embedFile("fixtures/model-configs/mimo_v2.json"));
    try t.expectEqualStrings("mimo_v2", m.model_type);
    try t.expectEqual(@as(u32, 152576), m.vocab_size);
    try t.expectEqual(@as(u32, 4096), m.hidden_size);
    try t.expectEqual(@as(u32, 48), m.num_hidden_layers);
    try t.expectEqual(@as(u32, 64), m.num_attention_heads);
    try t.expectEqual(@as(u32, 192), m.head_dim);
    try t.expectEqual(@as(u32, 128), m.v_head_dim);
    try t.expectEqual(@as(u32, 1048576), m.max_position_embeddings);
    try t.expectEqual(@as(u32, 128), m.sliding_window);
    try t.expectEqual(@as(u32, 256), m.num_experts);
    try t.expectEqual(@as(u32, 8), m.num_experts_per_tok);
    try t.expectEqual(@as(u32, 2048), m.moe_intermediate_size);
    try t.expectEqual(@as(u32, 1), m.first_k_dense_replace);
    try t.expectEqual(@as(f32, 0.334), m.partial_rotary_factor);
    try t.expectEqual(@as(?u32, null), m.bos_token_id);
    try t.expectEqualSlices(u32, &.{151645}, m.eosTokenSlice());

    for ([_]struct { doc: []const u8, rate_n: u32 }{
        .{ .doc = @embedFile("fixtures/model-configs/qwen4_exp.json"), .rate_n = 3 * 16 },
        .{ .doc = @embedFile("fixtures/model-configs/mimo_v2.json"), .rate_n = 4 * 16 },
    }) |c| {
        const meta = model_discovery.parseStubMeta(t.allocator, c.doc, true);
        try t.expect(meta.found and meta.quantized_experts);
        try t.expectEqual(c.rate_n, meta.expert_quant_rate.?.n);
        try t.expectEqual(@as(u32, 48), meta.num_hidden_layers);
    }
}

test "an f32 GDN state stores f32 and bills its checkpoint at four bytes per state element" {
    var c = ModelConfig{ .num_hidden_layers = 48, .full_attention_interval = 4, .linear_num_key_heads = 16, .linear_num_value_heads = 48 };
    const conv: u64 = 3 * (2 * 16 * 128 + 48 * 128) * 2;
    try testing.expectEqual(@as(mlx.mlx_dtype, .bfloat16), c.gdnStateDtype());
    try testing.expectEqual(@as(u64, 36 * (48 * 128 * 128 * 2 + conv)), c.ssmCheckpointBytes());
    c.gdn_state_f32 = true;
    try testing.expectEqual(@as(mlx.mlx_dtype, .float32), c.gdnStateDtype());
    try testing.expectEqual(@as(u64, 36 * (48 * 128 * 128 * 4 + conv)), c.ssmCheckpointBytes());
}

test "GLM config maps compressed MLA and FP32 recurrent state without Qwen assumptions" {
    const c = try parseConfigFromJson(testing.allocator, @embedFile("fixtures/glm5_config.json"));
    try testing.expect(c.isGlm5());
    try testing.expect(c.has_vision and c.glm5_vision);
    try testing.expectEqual(@as(u32, 14), c.qv_patch);
    try testing.expectEqual(@as(u32, 154854), c.image_token_id);
    try testing.expectEqual(@as(f32, 0.0625), c.attnScale());
    try testing.expectEqualStrings("model.language_model", c.weight_prefix);
    try testing.expectEqual(@as(u32, 45), c.num_hidden_layers);
    try testing.expectEqual(@as(u32, 288), c.num_experts);
    try testing.expectEqual(@as(u32, 8), c.num_experts_per_tok);
    try testing.expectEqual(@as(u32, 3), c.first_k_dense_replace);
    try testing.expectEqual(@as(u32, 64), c.linear_num_value_heads);
    try testing.expectEqual(@as(u32, 128), c.linear_key_head_dim);
    try testing.expectEqual(@as(u32, 11), c.attnCacheLayerCount());
    try testing.expectEqual(@as(u32, 512), c.mla_kv_lora_rank);
    try testing.expectEqual(@as(u32, 0), c.mla_qk_rope_head_dim);
    try testing.expectEqual(@as(u32, 20), c.glm_hc_sinkhorn_iters);
    try testing.expectEqual(@as(u64, 11 * 512 * 2), c.kvBytesPerToken());
    try testing.expectEqual(@as(u64, 34 * (64 * 128 * 128 * 4 + 3 * 3 * 64 * 128 * 2)), c.ssmCheckpointBytes());
    try testing.expectEqual(@as(u64, 11 * 128 * 2 / 4), c.qsaHistoryBytesPerToken());
    try testing.expectEqual(@as(u64, 0), c.qsaScoreBankBytesPerToken());
    try testing.expectEqual(@as(u64, 11 * 3 * 128 * 2 * 2), c.qsaRingBytes());
    try testing.expectApproxEqAbs(@as(f32, -5), c.kda_gate_lower_bound, 1e-6);
    try testing.expect(c.moe_sigmoid_router and c.moe_route_norm and !c.norm_has_offset);
}

test "GLM config refuses unsupported semantics and invalid geometry" {
    const source = @embedFile("fixtures/glm5_config.json");
    const cases = [_][2][]const u8{
        .{ "\"mhc\": true", "\"mhc\": false" },
        .{ "\"scoring_func\": \"sigmoid\"", "\"scoring_func\": \"softmax\"" },
        .{ "\"topk_method\": \"noaux_tc\"", "\"topk_method\": \"greedy\"" },
        .{ "\"index_kpool_compress\": true", "\"index_kpool_compress\": false" },
        .{ "\"attention_bias\": false", "\"attention_bias\": true" },
        .{ "\"hidden_act\": \"silu\"", "\"hidden_act\": \"relu\"" },
        .{ "\"moe_router_dtype\": \"float32\"", "\"moe_router_dtype\": \"bfloat16\"" },
        .{ "\"dense\"", "\"sparse\"" },
        .{ "\"full\"", "\"other\"" },
        .{ "\"num_attention_heads\": 64", "\"num_attention_heads\": 0" },
        .{ "\"num_heads\": 64", "\"num_heads\": 0" },
        .{ "\"index_n_heads\": 32", "\"index_n_heads\": 0" },
        .{ "\"index_head_dim\": 128", "\"index_head_dim\": 0" },
        .{ "\"v_head_dim\": 256", "\"v_head_dim\": 0" },
        .{ "\"intermediate_size\": 12288", "\"intermediate_size\": 0" },
        .{ "\"vocab_size\": 154880", "\"vocab_size\": 0" },
        .{ "\"rms_norm_eps\": 1e-05", "\"rms_norm_eps\": 0" },
        .{ "\"max_position_embeddings\": 1048576", "\"max_position_embeddings\": 0" },
        .{ "\"qk_head_dim\": 256", "\"qk_head_dim\": 128" },
        .{ "\"hidden_size\": 4096,", "" },
        .{ "\"num_attention_heads\": 64,", "" },
        .{ "\"full_attn_layers\": [", "\"full_attn_layers\": [0," },
        .{ "\"kda_layers\": [", "\"kda_layers\": [99," },
        .{ "\"index_head_dim\": 128", "\"index_head_dim\": 2147483648" },
        .{ "\"index_n_heads\": 32", "\"index_n_heads\": 2147483647" },
    };
    for (cases) |change| {
        const raw = try std.mem.replaceOwned(u8, testing.allocator, source, change[0], change[1]);
        defer testing.allocator.free(raw);
        if (parseConfigFromJson(testing.allocator, raw)) |_| return error.ExpectedGlmConfigRejection else |err| {
            try testing.expect(err == error.InvalidGlmConfig or err == error.UnsupportedGlmConfig);
        }
    }
}

test "GLM config preserves optional shared expert counts" {
    for ([_]u32{ 0, 2 }) |count| {
        const replacement = try std.fmt.allocPrint(testing.allocator, "\"n_shared_experts\": {d}", .{count});
        defer testing.allocator.free(replacement);
        const raw = try std.mem.replaceOwned(u8, testing.allocator, @embedFile("fixtures/glm5_config.json"), "\"n_shared_experts\": 1", replacement);
        defer testing.allocator.free(raw);
        const config = try parseConfigFromJson(testing.allocator, raw);
        try testing.expectEqual(count * 2048, config.shared_expert_intermediate_size);
    }
}

test "GLM serving accepts native effort levels and thinks by default" {
    const cfg = ModelConfig{ .model_type = "glm5_next" };
    try testing.expect(isServedArch(cfg.model_type));
    try testing.expect(cfg.defaultEnableThinking());
    try testing.expect(!cfg.needsSsmEntries());
    for ([_]Effort{ .low, .high, .max }) |effort| {
        try testing.expectEqual(@as(?i32, null), findEffortArm(effortArms(cfg.model_type).?, effort).?.budget);
    }
    try testing.expect(findEffortArm(effortArms(cfg.model_type).?, .medium) == null);
    try testing.expect(findEffortArm(effortArms(cfg.model_type).?, .xhigh) == null);
    try testing.expect(findEffortArm(effortArms(cfg.model_type).?, .off) == null);
    try testing.expectEqualStrings("high", defaultEffortWord(&cfg).?);
}

test "thinking policy launch defaults are model-specific and preserve GLM high" {
    const saved = think_effort_flag;
    defer think_effort_flag = saved;
    const glm = ModelConfig{ .model_type = "glm5_next" };
    const mimo = ModelConfig{ .model_type = "mimo_v2" };
    const qwen = ModelConfig{ .model_type = "qwen4_exp" };
    think_effort_flag = null;
    try testing.expectEqualStrings("high", defaultEffortWord(&glm).?);
    think_effort_flag = .low;
    try testing.expectEqualStrings("low", defaultEffortWord(&glm).?);
    try testing.expectEqualStrings("low", defaultEffortWord(&qwen).?);
    try testing.expectEqualStrings("on", defaultEffortWord(&mimo).?);
    think_effort_flag = .off;
    try testing.expect(!qwen.defaultEnableThinking());
    try testing.expect(!mimo.defaultEnableThinking());
    try testing.expect(glm.defaultEnableThinking());
}

test "thinking policy: --think binds the models that take it, the rest keep their own default" {
    const saved = think_effort_flag;
    defer think_effort_flag = saved;
    const glm = ModelConfig{ .model_type = "glm5_next" };
    const mimo = ModelConfig{ .model_type = "mimo_v2" };
    const qwen = ModelConfig{ .model_type = "qwen4_exp" };
    think_effort_flag = .on;
    try testing.expect(!qwen.defaultEnableThinking());
    try testing.expect(defaultEffortWord(&qwen) == null);
    try testing.expectEqualStrings("high", defaultEffortWord(&glm).?);
    for ([_]Effort{ .low, .medium, .high, .xhigh, .max }) |e| {
        think_effort_flag = e;
        try testing.expectEqualStrings("on", defaultEffortWord(&mimo).?);
        try testing.expect(mimo.defaultEnableThinking());
    }
    think_effort_flag = .minimal;
    try testing.expectEqualStrings("on", defaultEffortWord(&mimo).?);
    try testing.expect(defaultEffortWord(&qwen) == null);
    think_effort_flag = .off;
    try testing.expectEqualStrings("off", defaultEffortWord(&mimo).?);
    try testing.expectEqualStrings("high", defaultEffortWord(&glm).?);
    try testing.expectEqual(@as(?Effort, .on), selectEffort(&.{ .off, .on }, .max));
    try testing.expectEqual(@as(?Effort, .off), selectEffort(&.{ .off, .on }, .off));
    try testing.expectEqual(@as(?Effort, null), selectEffort(&.{ .low, .high, .max }, .off));
    try testing.expectEqual(@as(?Effort, null), selectEffort(&.{ .off, .low, .medium, .xhigh }, .minimal));
    // A model loaded on demand never fails its load over the flag.
    think_effort_flag = .on;
    if (loadWeightsForConfig(testing.io, testing.allocator, "/nonexistent/sushi-think-gate", &qwen, false)) |w| {
        var owned = w;
        owned.deinit();
        return error.TestUnexpectedResult;
    } else |err| try testing.expect(err != error.ThinkingUnsupported);
}

test "thinking policy: default_reasoning_effort is the word a request naming none runs at" {
    const saved = think_effort_flag;
    defer think_effort_flag = saved;
    const glm = ModelConfig{ .model_type = "glm5_next" };
    const mimo = ModelConfig{ .model_type = "mimo_v2" };
    const mimo_off = ModelConfig{ .model_type = "mimo_v2", .gen_enable_thinking = false };
    const qwen = ModelConfig{ .model_type = "qwen4_exp" };
    const qwen_on = ModelConfig{ .model_type = "qwen4_exp", .gen_enable_thinking = true };
    think_effort_flag = null;
    try testing.expectEqualStrings("high", defaultReasoningEffort(&glm).?);
    try testing.expectEqualStrings("on", defaultReasoningEffort(&mimo).?);
    try testing.expectEqualStrings("off", defaultReasoningEffort(&mimo_off).?);
    try testing.expectEqualStrings("off", defaultReasoningEffort(&qwen).?);
    try testing.expectEqualStrings("low", defaultReasoningEffort(&qwen_on).?);
    try testing.expect(defaultReasoningEffort(&ModelConfig{ .model_type = "llama" }) == null);
    think_effort_flag = .xhigh;
    try testing.expectEqualStrings("high", defaultReasoningEffort(&glm).?);
    try testing.expectEqualStrings("on", defaultReasoningEffort(&mimo_off).?);
    try testing.expectEqualStrings("xhigh", defaultReasoningEffort(&qwen).?);
}

test "thinking policy: thinkFlagBudget uses current arch caps and preserves uncapped fallbacks" {
    const saved = think_effort_flag;
    defer think_effort_flag = saved;
    think_effort_flag = null;
    try testing.expectEqual(@as(i32, 321), thinkFlagBudget("qwen4_exp", 321));
    think_effort_flag = .medium;
    try testing.expectEqual(@as(i32, 8192), thinkFlagBudget("qwen4_exp", -1));
    think_effort_flag = .low;
    try testing.expectEqual(@as(i32, 2048), thinkFlagBudget("qwen4_exp", -1));
    for ([_]Effort{ .off, .xhigh }) |effort| {
        think_effort_flag = effort;
        try testing.expectEqual(@as(i32, -1), thinkFlagBudget("qwen4_exp", -1));
        try testing.expectEqual(@as(i32, 321), thinkFlagBudget("qwen4_exp", 321));
    }
    for ([_]Effort{ .low, .high, .max }) |effort| {
        think_effort_flag = effort;
        try testing.expectEqual(@as(i32, 123), thinkFlagBudget("glm5_next", 123));
    }
    think_effort_flag = .on;
    try testing.expectEqual(@as(i32, 456), thinkFlagBudget("mimo_v2", 456));
    try testing.expectEqual(@as(i32, 789), thinkFlagBudget("unknown", 789));
}

test "GLM vision config rejects unsupported tower geometry and accepts a text-only checkpoint" {
    const a = testing.allocator;
    const source = @embedFile("fixtures/glm5_config.json");
    const parsed = try std.json.parseFromSlice(std.json.Value, a, source, .{});
    defer parsed.deinit();
    var root = parsed.value.object;
    const vision = root.get("vision_config").?.object;
    var cfg = try parseConfigFromJson(a, source);
    try testing.expect(cfg.has_vision and cfg.glm5_vision and !cfg.qwen_vision);
    try testing.expectEqual(@as(u32, 64), cfg.qv_head_dim);
    try testing.expectEqual(@as(f32, 10000), cfg.glmv_rope_theta);
    try testing.expectEqual(@as(u32, 10240), cfg.glmv_projection_intermediate);
    var invalid = vision;
    invalid.getPtr("num_heads").?.* = .{ .integer = 15 };
    cfg.glm5_vision = false;
    try testing.expectError(error.InvalidGlmVisionConfig, parseGlm5VisionFields(&cfg, root));
    _ = root.swapRemove("vision_config");
    cfg.glm5_vision = false;
    cfg.has_vision = false;
    try parseGlm5VisionFields(&cfg, root);
    try testing.expect(!cfg.has_vision and !cfg.glm5_vision);
    const processor = parseVisionProcessorDefaultsFromJson("{\"image_processor\":{\"min_image_tokens\":16,\"max_image_tokens\":8000},\"video_processor\":{\"max_image_tokens\":240000}}");
    try testing.expectEqual(@as(?u32, 16), processor.min_image_tokens);
    try testing.expectEqual(@as(?u32, 8000), processor.max_image_tokens);
    try testing.expectEqual(@as(?u32, 240000), processor.max_video_tokens);
}

test "GLM raw FP8 config identifies source storage without changing native KV" {
    const source = @embedFile("fixtures/glm5_config.json");
    const raw = try std.fmt.allocPrint(std.testing.allocator, "{{\"sushi_pack\":{{\"trunk_storage\":\"source-fp8-e4m3fn-block128\"}},{s}", .{source[1..]});
    defer std.testing.allocator.free(raw);
    const cfg = try parseConfigFromJson(std.testing.allocator, raw);
    try std.testing.expect(cfg.glm_fp8_trunk);
    try std.testing.expectEqual(@as(u64, 11 * 512 * 2), cfg.kvBytesPerToken());
    const affine = try parseConfigFromJson(std.testing.allocator, source);
    try std.testing.expect(!affine.glm_fp8_trunk);
    const release = try std.fmt.allocPrint(std.testing.allocator, "{{\"quantization_config\":{{\"quant_method\":\"fp8\",\"fmt\":\"e4m3\",\"activation_scheme\":\"dynamic\",\"weight_block_size\":[128,128]}},{s}", .{source[1..]});
    defer std.testing.allocator.free(release);
    try std.testing.expect((try parseConfigFromJson(std.testing.allocator, release)).glm_fp8_trunk);
    const other_block = try std.fmt.allocPrint(std.testing.allocator, "{{\"quantization_config\":{{\"quant_method\":\"fp8\",\"fmt\":\"e4m3\",\"weight_block_size\":[64,64]}},{s}", .{source[1..]});
    defer std.testing.allocator.free(other_block);
    try expectError(error.UnsupportedGlmConfig, parseConfigFromJson(std.testing.allocator, other_block));
}

test "loadSafetensorsFile returns what MLX's own safetensors loader returns, dtype, shape and bytes" {
    const t = std.testing;
    const a = t.allocator;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    // Odd payload sizes leave every later tensor at an unaligned file offset.
    const Spec = struct { name: []const u8, dtype: []const u8, shape: []const usize, item: usize };
    const specs = [_]Spec{
        .{ .name = "model.layers.0.a.weight", .dtype = "BF16", .shape = &.{ 3, 5 }, .item = 2 },
        .{ .name = "model.layers.0.b.weight", .dtype = "F32", .shape = &.{7}, .item = 4 },
        .{ .name = "model.layers.0.c.weight", .dtype = "U32", .shape = &.{ 2, 3 }, .item = 4 },
        .{ .name = "model.layers.0.e.weight", .dtype = "F16", .shape = &.{ 2, 3 }, .item = 2 },
        .{ .name = "model.layers.0.f.weight", .dtype = "U8", .shape = &.{5}, .item = 1 },
        .{ .name = "model.layers.0.g.weight", .dtype = "I32", .shape = &.{3}, .item = 4 },
        .{ .name = "vision_tower.dropped.weight", .dtype = "F32", .shape = &.{2}, .item = 4 },
    };
    var header = std.ArrayList(u8).empty;
    defer header.deinit(a);
    var data = std.ArrayList(u8).empty;
    defer data.deinit(a);
    try header.appendSlice(a, "{\"__metadata__\":{\"format\":\"mlx\"}");
    for (specs, 0..) |spec, n| {
        var count: usize = 1;
        for (spec.shape) |d| count *= d;
        const begin = data.items.len;
        // Finite bit patterns for every dtype: small integers in each element's low byte.
        for (0..count) |i| for (0..spec.item) |b| try data.append(a, if (b == spec.item - 1 and spec.item > 1) 0x3c else @truncate(n * 31 + i * 7 + b));
        try header.print(a, ",\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ spec.name, spec.dtype });
        for (spec.shape, 0..) |d, i| try header.print(a, "{s}{d}", .{ if (i == 0) "" else ",", d });
        try header.print(a, "],\"data_offsets\":[{d},{d}]}}", .{ begin, data.items.len });
    }
    try header.append(a, '}');
    var file = std.ArrayList(u8).empty;
    defer file.deinit(a);
    var len_bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_bytes, header.items.len, .little);
    try file.appendSlice(a, &len_bytes);
    try file.appendSlice(a, header.items);
    try file.appendSlice(a, data.items);
    try tmp.dir.writeFile(t.io, .{ .sub_path = "model.safetensors", .data = file.items });
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try tmp.dir.realPath(t.io, &path_buf);
    const path = try std.fmt.allocPrintSentinel(a, "{s}/model.safetensors", .{path_buf[0..path_len]}, 0);
    defer a.free(path);

    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    var weights = Weights.init(a);
    defer weights.deinit();
    try loadSafetensorsFile(a, &weights, path, cpu, false);

    var want = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(want);
    var meta = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta);
    try mlx.check(mlx.mlx_load_safetensors(&want, &meta, path, cpu));
    try t.expectEqual(specs.len - 1, weights.count());
    try t.expect(weights.get("vision_tower.dropped.weight") == null);
    for (specs[0 .. specs.len - 1]) |spec| {
        const name = try a.dupeSentinel(u8, spec.name, 0);
        defer a.free(name);
        var ref = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ref);
        try mlx.check(mlx.mlx_map_string_to_array_get(&ref, want, name));
        const got = weights.get(spec.name).?;
        try t.expectEqual(mlx.mlx_array_dtype(ref), mlx.mlx_array_dtype(got));
        try t.expectEqualSlices(c_int, mlx.getShape(ref), mlx.getShape(got));
        var eq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(eq);
        try mlx.check(mlx.mlx_array_equal(&eq, ref, got, true, cpu));
        try mlx.check(mlx.mlx_array_eval(eq));
        var same = false;
        try mlx.check(mlx.mlx_array_item_bool(&same, eq));
        try t.expect(same);
    }
}

test "readTensorArray returns the file's tensor, imported without a copy when large" {
    const t = std.testing;
    const a = t.allocator;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const page = expert_io.pageSize();
    const size = (2 << 20) + 3 * page + 77;
    const bytes = try a.alloc(u8, size);
    defer a.free(bytes);
    for (bytes, 0..) |*b, i| b.* = @truncate(i *% 13 +% i / 509);
    for (bytes, 0..) |*b, i| if (i % 2 == 1) {
        b.* &= 0x3f; // finite BF16/F32 elements
    };
    try tmp.dir.writeFile(t.io, .{ .sub_path = "t.safetensors", .data = bytes });
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try tmp.dir.realPath(t.io, &path_buf);
    const path = try std.fmt.allocPrintSentinel(a, "{s}/t.safetensors", .{path_buf[0..path_len]}, 0);
    defer a.free(path);
    const fd = try expert_io.openHinted(path, .{ .readahead_off = false });
    defer _ = std.c.close(fd);
    var reader = expert_io.ParallelReader.init(a, page, 3);
    defer reader.deinit();
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    const Case = struct { offset: usize, shape: []const c_int, dtype: mlx.mlx_dtype, item: usize, imported: bool };
    for ([_]Case{
        .{ .offset = 100, .shape = &.{ 3, 5 }, .dtype = .bfloat16, .item = 2, .imported = false },
        .{ .offset = page - 8, .shape = &.{ 64, 64 }, .dtype = .uint32, .item = 4, .imported = false },
        .{ .offset = 777, .shape = &.{ 512, 1024 }, .dtype = .float32, .item = 4, .imported = true },
        .{ .offset = size - (1 << 20) - 1, .shape = &.{ 1 << 20, 1 }, .dtype = .uint8, .item = 1, .imported = true },
        // Large but not a page multiple: Metal may import it or refuse it (MLX then copies and frees the mapping).
        .{ .offset = 5, .shape = &.{ (1 << 20) + 3, 1 }, .dtype = .uint8, .item = 1, .imported = false },
    }) |case| {
        var count: usize = 1;
        for (case.shape) |d| count *= @intCast(d);
        const len = count * case.item;
        const got = try readTensorArray(&reader, fd, case.offset, len, case.shape, case.dtype);
        defer _ = mlx.mlx_array_free(got.array);
        try t.expectEqualSlices(u8, bytes[case.offset..][0..len], got.bytes);
        try t.expectEqual(case.dtype, mlx.mlx_array_dtype(got.array));
        try t.expectEqualSlices(c_int, case.shape, mlx.getShape(got.array));
        const want = mlx.mlx_array_new_data(bytes[case.offset..].ptr, case.shape.ptr, @intCast(case.shape.len), case.dtype);
        defer _ = mlx.mlx_array_free(want);
        for ([_]mlx.mlx_stream{ cpu, mlx.gpuStream() }) |s| {
            var copy = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(copy);
            try mlx.check(mlx.mlx_copy(&copy, got.array, s));
            var eq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(eq);
            try mlx.check(mlx.mlx_array_equal(&eq, want, copy, true, cpu));
            try mlx.check(mlx.mlx_array_eval(eq));
            var same = false;
            try mlx.check(mlx.mlx_array_item_bool(&same, eq));
            try t.expect(same);
        }
        if (case.imported) try t.expect(got.mapped != null);
        if (got.mapped) |m| try t.expectEqual(@intFromPtr(got.bytes.ptr), @intFromPtr(m));
    }
}
