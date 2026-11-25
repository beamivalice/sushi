//! Hy4-Flash (`hy_v4_flash`): the released `config.json`, read into the geometry the forward, the loader and the bills
//! take. MLX-free. Every key the forward reads is checked against the one form the release takes; anything else is
//! refused by name, so a checkpoint we never served cannot load with a quietly wrong reading.
const std = @import("std");
const census = @import("hy4_census.zig");
const log = @import("../log.zig");

pub const max_layers = 64;
pub const max_engram = 4;
/// Primes per Engram module: one per head per n-gram order above one.
pub const max_engram_primes = 16;
pub const max_dfly_targets = 8;
/// The widest forward the model runs, in rows: the ceiling of a load's prefill chunk (`--prefill-chunk-max`).
pub const widest_chunk: u32 = 8192;

pub const LayerKind = enum(u1) { sliding, sparse };

pub const Rope = struct { theta: f64 = 0 };

pub const Indexer = struct {
    heads: u32 = 0,
    head_dim: u32 = 0,
    /// Tokens pooled into one group key.
    group: u32 = 0,
    /// Groups a query picks, its own group included.
    group_topk: u32 = 0,
};

pub const Engram = struct {
    n: u32 = 0,
    /// 0-based backbone layers the modules sit at.
    layers: [max_engram]u32 = @splat(0),
    /// The 1-based config ids the multipliers are drawn with.
    hash_ids: [max_engram]u32 = @splat(0),
    seed: u32 = 0,
    max_ngram: u32 = 0,
    heads: u32 = 0,
    head_dim: u32 = 0,
    vocab_size: u64 = 0,
    pad_id: u32 = 0,
    primes: [max_engram][max_engram_primes]u64 = @splat(@splat(0)),
    /// Table rows per module: the sum of its primes.
    rows: [max_engram]u64 = @splat(0),

    pub fn primeCount(self: Engram) u32 {
        return (self.max_ngram - 1) * self.heads;
    }

    pub fn totalRows(self: Engram) u64 {
        var t: u64 = 0;
        for (self.rows[0..self.n]) |r| t += r;
        return t;
    }
};

pub const Dfly = struct {
    layers: u32 = 0,
    targets: [max_dfly_targets]u32 = @splat(0),
    n_targets: u32 = 0,
    mask_id: u32 = 0,
    /// Tokens drafted a round.
    draft: u32 = 0,
};

pub const Vision = struct {
    layers: u32 = 0,
    hidden: u32 = 0,
    heads: u32 = 0,
    inter: u32 = 0,
    patch: u32 = 0,
    merge: u32 = 0,
    out_hidden: u32 = 0,
    /// The pack's `vision_quant`: the tower's matrices stored as MLX affine codes; null when stored BF16.
    quant: ?VisionQuant = null,
};

pub const VisionQuant = struct { bits: u32, group: u32 };

/// The group of a tower's affine matrices: the widest of 128, 64 and 32 dividing its width, which every block and
/// merger matrix but the MLP's down projection reads; null for a width none divides.
pub fn visionGroup(hidden: u32) ?u32 {
    for ([_]u32{ 128, 64, 32 }) |g| if (hidden % g == 0) return g;
    return null;
}

pub const Config = struct {
    /// A load with no flag and no setting takes its ssd budget from the Mac's RAM tier, and a request that omits
    /// `max_tokens` gets 65,536, the completion cap Tencent serves the release at; the context stays the engine's.
    pub fn serveDefaults(comptime D: type) D {
        return .{ .ram_budget = true, .max_tokens = 65_536 };
    }

    vocab: u32 = 0,
    dim: u32 = 0,
    layers: u32 = 0,
    kinds: [max_layers]LayerKind = @splat(.sliding),
    /// Layers below it run the dense FFN; the rest the MoE.
    dense_layers: u32 = 0,
    dense_inter: u32 = 0,
    sparse_heads: u32 = 0,
    sliding_heads: u32 = 0,
    q_lora: u32 = 0,
    kv_lora: u32 = 0,
    nope_dim: u32 = 0,
    rope_dim: u32 = 0,
    v_dim: u32 = 0,
    window: u32 = 0,
    eps: f32 = 0,
    max_pos: u32 = 0,
    sparse_rope: Rope = .{},
    sliding_rope: Rope = .{},
    indexer: Indexer = .{},
    experts: u32 = 0,
    top_k: u32 = 0,
    moe_inter: u32 = 0,
    route_scale: f32 = 0,
    swiglu_limit: f32 = 0,
    hc_mult: u32 = 0,
    hc_eps: f32 = 0,
    hc_magnitude: f32 = 0,
    engram: Engram = .{},
    dfly: Dfly = .{},
    vision: ?Vision = null,
    bos: u32 = 0,
    eos: u32 = 0,
    pad: u32 = 0,
    image_id: u32 = 0,
    image_start_id: u32 = 0,
    image_end_id: u32 = 0,
    image_newline_id: u32 = 0,

    pub fn qkDim(self: Config) u32 {
        return self.nope_dim + self.rope_dim;
    }

    pub fn heads(self: Config, layer: usize) u32 {
        return if (self.kinds[layer] == .sparse) self.sparse_heads else self.sliding_heads;
    }

    pub fn sparseCount(self: Config) u32 {
        var n: u32 = 0;
        for (self.kinds[0..self.layers]) |k| n += @intFromBool(k == .sparse);
        return n;
    }
};

const Error = error{UnsupportedHy4Config};

pub fn parse(text: std.json.ObjectMap, root: std.json.ObjectMap) Error!Config {
    var c: Config = .{};
    c.vocab = try uint(u32, text, "vocab_size");
    c.dim = try uint(u32, text, "hidden_size");
    c.layers = try uint(u32, text, "num_hidden_layers");
    if (c.layers == 0 or c.layers > max_layers) return refuse("{d} layers", .{c.layers});
    c.eps = @floatCast(try num(text, "rms_norm_eps"));
    c.max_pos = try uint(u32, text, "max_position_embeddings");
    c.bos = try uint(u32, text, "bos_token_id");
    c.eos = try uint(u32, text, "eos_token_id");
    c.pad = try uint(u32, text, "pad_token_id");
    if (!std.mem.eql(u8, try str(text, "hidden_act"), "silu")) return refuse("hidden_act is not silu", .{});
    if (flag(text, "attention_bias")) return refuse("attention_bias is on", .{});
    if (flag(text, "tie_word_embeddings")) return refuse("tie_word_embeddings is on", .{});
    inline for (.{ "use_mla", "gated_mla", "learnable_sink", "sandwich_norm", "qk_norm", "enable_ihc", "enable_lm_head_fp32", "use_dsa", "use_compressed_indexer", "moe_router_use_sigmoid", "moe_router_enable_expert_bias", "norm_topk_prob", "compressed_indexer_skip_fine_select" }) |key|
        if (!flag(text, key)) return refuse(key ++ " is not on", .{});
    if (!std.mem.eql(u8, try str(text, "gating_type"), "elementwise")) return refuse("gating_type is not elementwise", .{});
    if ((try uint(u32, text, "num_nextn_predict_layers")) != 0) return refuse("num_nextn_predict_layers is not 0 (the release drafts with DFly2)", .{});

    // Layer kinds: `layer_types` (any of the three spellings) must name the same sliding layers as `swa_pattern`.
    const types = try list(text, "layer_types");
    const pattern = try list(text, "swa_pattern");
    const indexers = try list(text, "indexer_types");
    if (types.len != c.layers or pattern.len != c.layers or indexers.len != c.layers) return refuse("layer_types, swa_pattern and indexer_types do not name {d} layers", .{c.layers});
    for (types, pattern, indexers, 0..) |t, p, ix, l| {
        if (t != .string) return refuse("layer_types[{d}] is not a string", .{l});
        c.kinds[l] = if (std.mem.eql(u8, t.string, "sliding_attention"))
            .sliding
        else if (isSparseSpelling(t.string))
            .sparse
        else
            return refuse("layer_types[{d}] is {s}", .{ l, t.string });
        if (p != .integer or p.integer != @intFromBool(c.kinds[l] == .sliding)) return refuse("swa_pattern[{d}] disagrees with layer_types", .{l});
        if (ix != .string or !std.mem.eql(u8, ix.string, "full")) return refuse("indexer_types[{d}] is not full", .{l});
    }

    // FFN: the first `first_k_dense_replace` layers are dense, the rest MoE.
    c.dense_layers = try uint(u32, text, "first_k_dense_replace");
    c.dense_inter = try uint(u32, text, "intermediate_size");
    const mlp = try list(text, "mlp_layer_types");
    if (mlp.len != c.layers) return refuse("mlp_layer_types does not name {d} layers", .{c.layers});
    for (mlp, 0..) |m, l| {
        const want: []const u8 = if (l < c.dense_layers) "dense" else "sparse";
        if (m != .string or !std.mem.eql(u8, m.string, want)) return refuse("mlp_layer_types[{d}] is not {s}", .{ l, want });
    }

    // MLA, both kinds: the sliding layers' head geometry is the sparse layers' with more heads.
    c.sparse_heads = try uint(u32, text, "num_attention_heads");
    if ((try uint(u32, text, "num_key_value_heads")) != c.sparse_heads) return refuse("num_key_value_heads is not num_attention_heads", .{});
    c.sliding_heads = try uint(u32, text, "swa_num_attention_heads");
    c.q_lora = try uint(u32, text, "q_lora_rank");
    c.kv_lora = try uint(u32, text, "kv_lora_rank");
    c.nope_dim = try uint(u32, text, "qk_nope_head_dim");
    c.rope_dim = try uint(u32, text, "qk_rope_head_dim");
    c.v_dim = try uint(u32, text, "v_head_dim");
    if ((try uint(u32, text, "qk_head_dim")) != c.qkDim()) return refuse("qk_head_dim is not nope + rope", .{});
    inline for (.{ .{ "swa_qk_nope_head_dim", "nope_dim" }, .{ "swa_qk_rope_head_dim", "rope_dim" }, .{ "swa_v_head_dim", "v_dim" } }) |kv|
        if ((try uint(u32, text, kv[0])) != @field(c, kv[1])) return refuse(kv[0] ++ " differs from the sparse layers'", .{});
    if ((try uint(u32, text, "swa_qk_head_dim")) != c.qkDim()) return refuse("swa_qk_head_dim is not nope + rope", .{});
    c.window = try uint(u32, text, "sliding_window");
    const ws = try list(text, "window_size");
    if (ws.len != 2 or ws[0] != .integer or ws[0].integer != c.window - 1 or ws[1] != .integer or ws[1].integer != 0) return refuse("window_size is not [sliding_window - 1, 0]", .{});
    if (c.sparse_heads == 0 or c.sliding_heads == 0 or c.rope_dim == 0 or c.rope_dim % 2 != 0 or c.kv_lora == 0 or c.window == 0 or c.dim % 32 != 0)
        return refuse("attention geometry (heads {d}/{d}, rope {d}, latent {d}, window {d})", .{ c.sparse_heads, c.sliding_heads, c.rope_dim, c.kv_lora, c.window });
    const rope = try obj(text, "rope_parameters");
    c.sliding_rope = try parseRope(rope, "sliding_attention");
    c.sparse_rope = try parseRope(rope, "deepseek_sparse_attention");

    // The compressed indexer: gated group pooling, no fine stage.
    c.indexer = .{
        .heads = try uint(u32, text, "compressed_indexer_num_attention_heads"),
        .head_dim = try uint(u32, text, "compressed_indexer_head_dim"),
        .group = try uint(u32, text, "compressed_indexer_compress_ratio"),
        .group_topk = try uint(u32, text, "compressed_indexer_group_topk"),
    };
    if (!std.mem.eql(u8, try str(text, "compressed_indexer_key_pooling"), "gated")) return refuse("compressed_indexer_key_pooling is not gated", .{});
    if ((try uint(u32, text, "compressed_indexer_candidate_topk")) != c.indexer.group * c.indexer.group_topk) return refuse("compressed_indexer_candidate_topk is not group x group_topk", .{});
    if (c.indexer.heads == 0 or c.indexer.head_dim < c.rope_dim or c.indexer.group == 0 or c.indexer.group_topk == 0) return refuse("indexer geometry", .{});

    // MoE: sigmoid scores, the bias picks only, one group, experts in a latent as wide as their inner width.
    c.experts = try uint(u32, text, "n_routed_experts");
    c.top_k = try uint(u32, text, "num_experts_per_tok");
    c.moe_inter = try uint(u32, text, "moe_intermediate_size");
    c.route_scale = @floatCast(try num(text, "routed_scaling_factor"));
    c.swiglu_limit = @floatCast(try num(text, "swiglu_limit"));
    if ((try uint(u32, text, "routed_expert_hidden_size")) != c.moe_inter) return refuse("routed_expert_hidden_size is not moe_intermediate_size", .{});
    if ((try uint(u32, text, "n_shared_experts")) != 1) return refuse("n_shared_experts is not 1", .{});
    if ((try uint(u32, text, "n_group")) != 1 or (try uint(u32, text, "topk_group")) != 1) return refuse("n_group and topk_group are not 1", .{});
    if (c.experts == 0 or c.experts > 1024 or c.top_k == 0 or c.top_k > c.experts or c.swiglu_limit != @trunc(c.swiglu_limit))
        return refuse("MoE geometry (experts {d}, top-k {d}, swiglu_limit {d})", .{ c.experts, c.top_k, c.swiglu_limit });

    // iHC: the hyper-connection streams with no comb.
    c.hc_mult = try uint(u32, text, "hc_mult");
    c.hc_eps = @floatCast(try num(text, "hc_eps"));
    c.hc_magnitude = @floatCast(try num(text, "hc_magnitude"));
    if (c.hc_mult == 0 or c.hc_mult > 8) return refuse("hc_mult {d}", .{c.hc_mult});

    c.engram = try parseEngram(text, c);
    c.dfly = try parseDfly(root, c);
    c.vision = try parseVision(root, c);
    if (c.vision == null and field(root, "vision_quant") != null) return refuse("vision_quant without a vision_config", .{});
    c.image_id = try uint(u32, root, "image_token_id");
    c.image_start_id = try uint(u32, root, "image_start_token_id");
    c.image_end_id = try uint(u32, root, "image_end_token_id");
    c.image_newline_id = try uint(u32, root, "image_newline_token_id");
    return c;
}

/// What the shared `ModelConfig` takes from the parse; the forward reads `config.hy4flash`.
pub fn parseFamily(config: anytype, root: std.json.ObjectMap, cfg_obj: std.json.ObjectMap) !void {
    const c = try parse(cfg_obj, root);
    config.model_type = "hy_v4_flash";
    config.weight_prefix = "";
    config.text_prompt_bos = config.bos_token_id;
    config.hidden_act = .silu;
    config.first_k_dense_replace = c.dense_layers;
    // The routed experts as the expert stream reads them: fused BF16 banks [E, 2I, L] and [E, L, I] under the
    // release's names, run in the latent width.
    config.num_experts = c.experts;
    config.num_experts_per_tok = c.top_k;
    config.moe_intermediate_size = c.moe_inter;
    config.expert_hidden = c.moe_inter;
    config.expert_layout = .bf16_fused;
    config.expert_key_prefix = "model.layers.";
    // The MoE hands the stream its picks a group of at most the cache's slots at a time: no union workspace (a BF16
    // layer's 512 experts would take 7.25 GB).
    config.expert_union_within_cache = true;
    config.has_vision = c.vision != null;
    if (c.vision) |v| {
        // The shared ViT geometry the media bills read.
        config.qv_depth = v.layers;
        config.qv_hidden = v.hidden;
        config.qv_heads = v.heads;
        config.qv_head_dim = v.hidden / v.heads;
        config.qv_intermediate = v.inter;
        config.qv_out_hidden = c.dim;
        config.qv_patch = v.patch;
        config.qv_merge = v.merge;
        config.qv_temporal_patch = 1;
    }
    config.hy4flash = c;
    config.serve_defaults = Config.serveDefaults(@TypeOf(config.serve_defaults));
}

/// The sparse-layer spellings: the release's, llama.cpp's and transformers' remapped name.
fn isSparseSpelling(s: []const u8) bool {
    for ([_][]const u8{ "deepseek_sparse_attention", "sparse_attention", "indexed_attention" }) |n| if (std.mem.eql(u8, s, n)) return true;
    return false;
}

fn parseRope(rope: std.json.ObjectMap, kind: []const u8) Error!Rope {
    const r = try obj(rope, kind);
    if (!std.mem.eql(u8, try str(r, "rope_type"), "default")) return refuse("rope_parameters.{s}.rope_type is not default", .{kind});
    const theta = try num(r, "rope_theta");
    if (!(theta > 0)) return refuse("rope_parameters.{s}.rope_theta", .{kind});
    return .{ .theta = theta };
}

fn parseEngram(text: std.json.ObjectMap, c: Config) Error!Engram {
    var e: Engram = .{};
    const ids = try list(text, "engram_layer_ids");
    if (ids.len == 0 or ids.len > max_engram) return refuse("{d} Engram modules", .{ids.len});
    e.n = @intCast(ids.len);
    e.max_ngram = try uint(u32, text, "engram_max_ngram_size");
    e.heads = try uint(u32, text, "engram_n_head_per_ngram");
    const per_ngram = try uint(u32, text, "engram_n_embed_per_ngram");
    e.seed = try uint(u32, text, "engram_seed");
    e.pad_id = try uint(u32, text, "engram_pad_id");
    if (e.max_ngram < 2 or e.heads == 0 or e.primeCount() > max_engram_primes or per_ngram % e.heads != 0) return refuse("Engram geometry (n-gram {d}, heads {d})", .{ e.max_ngram, e.heads });
    e.head_dim = per_ngram / e.heads;
    const vocab = try list(text, "engram_vocab_size");
    if (vocab.len != e.max_ngram - 1) return refuse("engram_vocab_size does not name {d} orders", .{e.max_ngram - 1});
    for (vocab) |v| {
        if (v != .integer or v.integer <= 0) return refuse("engram_vocab_size is not positive", .{});
        if (e.vocab_size != 0 and v.integer != e.vocab_size) return refuse("engram_vocab_size differs between orders", .{});
        e.vocab_size = @intCast(v.integer);
    }
    const sizes = try obj(text, "engram_layer_head_vocab_sizes");
    for (ids, 0..) |v, i| {
        // 1-based ids: the module at id k sits at layer k - 1 and draws its multipliers with k.
        if (v != .integer or v.integer < 1 or v.integer > c.layers) return refuse("engram_layer_ids[{d}] is not a 1-based backbone layer", .{i});
        e.hash_ids[i] = @intCast(v.integer);
        e.layers[i] = e.hash_ids[i] - 1;
        if (i > 0 and e.hash_ids[i] <= e.hash_ids[i - 1]) return refuse("engram_layer_ids are not ascending", .{});
        var key_buf: [8]u8 = undefined;
        const key = std.fmt.bufPrint(&key_buf, "{d}", .{e.hash_ids[i]}) catch unreachable;
        const orders = try list(sizes, key);
        if (orders.len != e.max_ngram - 1) return refuse("engram_layer_head_vocab_sizes.{s} does not name {d} orders", .{ key, e.max_ngram - 1 });
        var at: usize = 0;
        for (orders) |o| {
            if (o != .array or o.array.items.len != e.heads) return refuse("engram_layer_head_vocab_sizes.{s} does not hold {d} primes an order", .{ key, e.heads });
            for (o.array.items) |p| {
                if (p != .integer or p.integer <= 0) return refuse("engram_layer_head_vocab_sizes.{s} holds a non-positive size", .{key});
                e.primes[i][at] = @intCast(p.integer);
                e.rows[i] += e.primes[i][at];
                at += 1;
            }
        }
    }
    return e;
}

fn parseDfly(root: std.json.ObjectMap, c: Config) Error!Dfly {
    var d: Dfly = .{};
    d.layers = try uint(u32, root, "dfly_num_hidden_layers");
    d.mask_id = try uint(u32, root, "dfly_mask_token_id");
    const kv = try list(root, "dfly_target_kv_layers");
    const index = try list(root, "dfly_target_index_layer");
    if (kv.len == 0 or kv.len > max_dfly_targets or index.len != kv.len) return refuse("{d} DFly target layers", .{kv.len});
    for (kv, index, 0..) |k, ix, i| {
        if (k != .integer or k.integer < 0 or k.integer >= c.layers or c.kinds[@intCast(k.integer)] != .sparse) return refuse("dfly_target_kv_layers[{d}] is not a sparse backbone layer", .{i});
        if (ix != .integer or ix.integer != k.integer) return refuse("dfly_target_index_layer[{d}] differs from its KV layer", .{i});
        d.targets[i] = @intCast(k.integer);
    }
    d.n_targets = @intCast(kv.len);
    const spec = try obj(root, "speculative_config");
    if (!std.mem.eql(u8, try str(spec, "method"), "dfly")) return refuse("speculative_config.method is not dfly", .{});
    d.draft = try uint(u32, spec, "num_speculative_tokens");
    if (d.layers == 0 or d.draft == 0 or d.mask_id >= c.vocab) return refuse("DFly geometry (layers {d}, draft {d})", .{ d.layers, d.draft });
    // A round holds at most this many drafts (its buffers, the verify block's rows).
    if (d.draft > @import("../drafter_rows.zig").max_drafts) return refuse("num_speculative_tokens {d} past the {d} a round drafts", .{ d.draft, @import("../drafter_rows.zig").max_drafts });
    return d;
}

fn parseVision(root: std.json.ObjectMap, c: Config) Error!?Vision {
    const vc = field(root, "vision_config") orelse return null;
    if (vc != .object) return refuse("vision_config is not an object", .{});
    const v = vc.object;
    const out: Vision = .{
        .layers = try uint(u32, v, "num_hidden_layers"),
        .hidden = try uint(u32, v, "hidden_size"),
        .heads = try uint(u32, v, "num_attention_heads"),
        .inter = try uint(u32, v, "intermediate_size"),
        .patch = try uint(u32, v, "patch_size"),
        .merge = try uint(u32, v, "spatial_merge_size"),
        .out_hidden = try uint(u32, v, "out_hidden_size"),
    };
    if (out.out_hidden != c.dim) return refuse("vision_config.out_hidden_size is not hidden_size", .{});
    if ((try uint(u32, v, "temporal_patch_size")) != 1) return refuse("vision_config.temporal_patch_size is not 1", .{});
    if (out.heads == 0 or out.hidden % out.heads != 0 or out.merge == 0) return refuse("vision geometry", .{});
    var with = out;
    with.quant = try parseVisionQuant(root, out);
    return with;
}

/// `vision_quant` (docs/pack-format.md#hy4-stq): `{"format": "affine", "bits": 8, "group_size": <visionGroup>,
/// "bf16_kept": [{"tensor": <name>, "reason": <why>}, ...]}`, the list naming exactly the matrices no group fits
/// (`hy4_census.visionKept`).
fn parseVisionQuant(root: std.json.ObjectMap, v: Vision) Error!?VisionQuant {
    const q = field(root, "vision_quant") orelse return null;
    if (q != .object) return refuse("vision_quant is not an object", .{});
    if (!std.mem.eql(u8, try str(q.object, "format"), "affine")) return refuse("vision_quant.format is not affine", .{});
    const bits = try uint(u32, q.object, "bits");
    if (bits != 8) return refuse("vision_quant.bits {d} is not 8", .{bits});
    const group = try uint(u32, q.object, "group_size");
    if (visionGroup(v.hidden) != group) return refuse("vision_quant.group_size {d} is not the tower's group", .{group});
    const kept = try list(q.object, "bf16_kept");
    for (kept, 0..) |k, i| {
        if (k != .object) return refuse("vision_quant.bf16_kept holds a non-object", .{});
        const name = try str(k.object, "tensor");
        if ((try str(k.object, "reason")).len == 0) return refuse("vision_quant.bf16_kept {s} gives no reason", .{name});
        if (!census.visionKept(v, name)) return refuse("vision_quant.bf16_kept names {s}, a tensor the tower stores affine or not a matrix", .{name});
        for (kept[0..i]) |prev| if (std.mem.eql(u8, (try str(prev.object, "tensor")), name)) return refuse("vision_quant.bf16_kept names {s} twice", .{name});
    }
    // Only the MLP's down projection reads a width other than the group's multiples (hidden, hidden x merge²).
    const want: usize = if (v.inter % group != 0) v.layers else 0;
    if (kept.len != want) return refuse("vision_quant.bf16_kept names {d} tensors, the tower keeps {d} BF16", .{ kept.len, want });
    return .{ .bits = bits, .group = group };
}

fn refuse(comptime fmt: []const u8, args: anytype) Error {
    if (!@import("builtin").is_test) log.err("hy_v4_flash: " ++ fmt ++ "\n", args);
    return error.UnsupportedHy4Config;
}

fn field(o: std.json.ObjectMap, key: []const u8) ?std.json.Value {
    const v = o.get(key) orelse return null;
    return if (v == .null) null else v;
}

fn flag(o: std.json.ObjectMap, key: []const u8) bool {
    const v = field(o, key) orelse return false;
    return v == .bool and v.bool;
}

fn uint(comptime T: type, o: std.json.ObjectMap, key: []const u8) Error!T {
    const v = field(o, key) orelse return refuse("missing {s}", .{key});
    if (v != .integer) return refuse("{s} is not an integer", .{key});
    return std.math.cast(T, v.integer) orelse refuse("{s} is out of range", .{key});
}

fn num(o: std.json.ObjectMap, key: []const u8) Error!f64 {
    const v = field(o, key) orelse return refuse("missing {s}", .{key});
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => refuse("{s} is not a number", .{key}),
    };
}

fn str(o: std.json.ObjectMap, key: []const u8) Error![]const u8 {
    const v = field(o, key) orelse return refuse("missing {s}", .{key});
    if (v != .string) return refuse("{s} is not a string", .{key});
    return v.string;
}

fn list(o: std.json.ObjectMap, key: []const u8) Error![]const std.json.Value {
    const v = field(o, key) orelse return refuse("missing {s}", .{key});
    if (v != .array) return refuse("{s} is not a list", .{key});
    return v.array.items;
}

fn obj(o: std.json.ObjectMap, key: []const u8) Error!std.json.ObjectMap {
    const v = field(o, key) orelse return refuse("missing {s}", .{key});
    if (v != .object) return refuse("{s} is not an object", .{key});
    return v.object;
}

const testing = std.testing;

/// The release's `config.json`, unmodified.
/// The release's sampling (its generation_config.json: temperature 0.9, top_p 1, no top_k), for a pack that ships none.
pub fn samplingDefaults(config: *@import("../model.zig").ModelConfig) void {
    if (config.gen_temperature == null) config.gen_temperature = 0.9;
    if (config.gen_top_p == null) config.gen_top_p = 1.0;
}

pub const release_file = @embedFile("fixtures/hy_v4_flash.json");

/// The keys of `release_file` that `parse` reads, one form a test can edit by a string.
pub const release_config =
    \\{"architectures": ["HYV4FlashForConditionalGeneration"], "model_type": "hy_v4_flash", "dtype": "bfloat16",
    \\ "image_token_id": 120120, "image_start_token_id": 120118, "image_end_token_id": 120119, "image_newline_token_id": 120121,
    \\ "spatial_merge_size": 2,
    \\ "dfly_num_hidden_layers": 5, "dfly_target_kv_layers": [25, 29, 33, 37, 41], "dfly_target_index_layer": [25, 29, 33, 37, 41],
    \\ "dfly_mask_token_id": 120009, "speculative_config": {"method": "dfly", "num_speculative_tokens": 5},
    \\ "vision_config": {"hidden_size": 1152, "intermediate_size": 4304, "num_attention_heads": 16, "num_hidden_layers": 27,
    \\  "patch_size": 16, "spatial_merge_size": 2, "out_hidden_size": 3072, "temporal_patch_size": 1, "add_patchemb_bias": true},
    \\ "text_config": {"model_type": "hy_v4_flash", "vocab_size": 120832, "hidden_size": 3072, "num_hidden_layers": 42,
    \\  "bos_token_id": 120000, "eos_token_id": 120025, "pad_token_id": 120002, "rms_norm_eps": 1e-05,
    \\  "max_position_embeddings": 1048576, "hidden_act": "silu", "attention_bias": false, "tie_word_embeddings": false,
    \\  "layer_types": ["deepseek_sparse_attention", "deepseek_sparse_attention", "sliding_attention", "sliding_attention", "sliding_attention", "deepseek_sparse_attention", "sliding_attention", "sliding_attention", "sliding_attention", "deepseek_sparse_attention", "sliding_attention", "sliding_attention", "sliding_attention", "deepseek_sparse_attention", "sliding_attention", "sliding_attention", "sliding_attention", "deepseek_sparse_attention", "sliding_attention", "sliding_attention", "sliding_attention", "deepseek_sparse_attention", "sliding_attention", "sliding_attention", "sliding_attention", "deepseek_sparse_attention", "sliding_attention", "sliding_attention", "sliding_attention", "deepseek_sparse_attention", "sliding_attention", "sliding_attention", "sliding_attention", "deepseek_sparse_attention", "sliding_attention", "sliding_attention", "sliding_attention", "deepseek_sparse_attention", "sliding_attention", "sliding_attention", "sliding_attention", "deepseek_sparse_attention"],
    \\  "swa_pattern": [0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
    \\  "indexer_types": ["full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full", "full"],
    \\  "mlp_layer_types": ["dense", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse"],
    \\  "first_k_dense_replace": 1, "intermediate_size": 13824,
    \\  "use_mla": true, "gated_mla": true, "gating_type": "elementwise", "learnable_sink": true, "sandwich_norm": true, "qk_norm": true,
    \\  "num_attention_heads": 32, "num_key_value_heads": 32, "swa_num_attention_heads": 64,
    \\  "q_lora_rank": 1024, "kv_lora_rank": 512, "qk_nope_head_dim": 192, "qk_rope_head_dim": 64, "qk_head_dim": 256, "v_head_dim": 256,
    \\  "swa_qk_nope_head_dim": 192, "swa_qk_rope_head_dim": 64, "swa_qk_head_dim": 256, "swa_v_head_dim": 256,
    \\  "sliding_window": 512, "window_size": [511, 0],
    \\  "rope_parameters": {"sliding_attention": {"rope_type": "default", "rope_theta": 10000, "mrope_interleaved": true, "mrope_section": [12, 10, 10]},
    \\   "deepseek_sparse_attention": {"rope_type": "default", "rope_theta": 10000000, "mrope_interleaved": true, "mrope_section": [12, 10, 10]}},
    \\  "use_dsa": true, "use_compressed_indexer": true, "compressed_indexer_num_attention_heads": 16, "compressed_indexer_head_dim": 128,
    \\  "compressed_indexer_compress_ratio": 4, "compressed_indexer_group_topk": 512, "compressed_indexer_candidate_topk": 2048,
    \\  "compressed_indexer_key_pooling": "gated", "compressed_indexer_skip_fine_select": true, "index_n_heads": 16, "index_head_dim": 128, "index_topk": 2048,
    \\  "n_routed_experts": 512, "num_experts_per_tok": 16, "n_shared_experts": 1, "moe_intermediate_size": 1536, "routed_expert_hidden_size": 1536,
    \\  "moe_router_use_sigmoid": true, "moe_router_enable_expert_bias": true, "norm_topk_prob": true, "n_group": 1, "topk_group": 1,
    \\  "routed_scaling_factor": 3.998, "swiglu_limit": 10.0,
    \\  "enable_ihc": true, "hc_mult": 4, "hc_eps": 1e-06, "hc_magnitude": 2.0, "enable_lm_head_fp32": true, "num_nextn_predict_layers": 0,
    \\  "engram_layer_ids": [2, 12, 22, 32], "engram_max_ngram_size": 3, "engram_n_head_per_ngram": 8, "engram_n_embed_per_ngram": 768,
    \\  "engram_pad_id": 120002, "engram_seed": 0, "engram_vocab_size": [10312000, 10312000],
    \\  "engram_layer_head_vocab_sizes": {
    \\   "2": [[10312007, 10312021, 10312039, 10312069, 10312091, 10312121, 10312123, 10312147], [10312177, 10312187, 10312207, 10312213, 10312229, 10312241, 10312249, 10312259]],
    \\   "12": [[10312277, 10312283, 10312301, 10312343, 10312373, 10312397, 10312409, 10312453], [10312457, 10312483, 10312507, 10312513, 10312531, 10312543, 10312567, 10312571]],
    \\   "22": [[10312613, 10312619, 10312633, 10312657, 10312681, 10312699, 10312723, 10312747], [10312759, 10312789, 10312811, 10312823, 10312867, 10312873, 10312889, 10312903]],
    \\   "32": [[10312909, 10312919, 10312927, 10312957, 10312979, 10313011, 10313021, 10313057], [10313063, 10313117, 10313123, 10313129, 10313161, 10313179, 10313189, 10313201]]}}}
;

/// Parses `json` (a whole `config.json`) with the testing allocator.
pub fn parseText(json: []const u8) !Config {
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, json, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return refuse("config.json is not an object", .{});
    return parse(try obj(parsed.value.object, "text_config"), parsed.value.object);
}

/// The release config with `old` replaced by `new` once, parsed.
fn parseEdited(old: []const u8, new: []const u8) !Config {
    const at = std.mem.indexOf(u8, release_config, old) orelse return error.TestEditNotFound;
    const text = try std.mem.concat(testing.allocator, u8, &.{ release_config[0..at], new, release_config[at + old.len ..] });
    defer testing.allocator.free(text);
    return parseText(text);
}

test "hy4 config: the release parses to its geometry" {
    const c = try parseText(release_config);
    try testing.expectEqual(@as(u32, 42), c.layers);
    try testing.expectEqual(@as(u32, 3072), c.dim);
    try testing.expectEqual(@as(u32, 120832), c.vocab);
    try testing.expectEqual(@as(u32, 12), c.sparseCount());
    for ([_]usize{ 0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41 }) |l| try testing.expectEqual(LayerKind.sparse, c.kinds[l]);
    for ([_]usize{ 2, 3, 4, 6, 40 }) |l| try testing.expectEqual(LayerKind.sliding, c.kinds[l]);
    try testing.expectEqual(@as(u32, 32), c.heads(0));
    try testing.expectEqual(@as(u32, 64), c.heads(2));
    try testing.expectEqual(@as(u32, 256), c.qkDim());
    try testing.expectEqual(@as(u32, 512), c.kv_lora);
    try testing.expectEqual(@as(u32, 256), c.v_dim);
    try testing.expectEqual(@as(u32, 512), c.window);
    try testing.expectEqual(@as(f64, 1e7), c.sparse_rope.theta);
    try testing.expectEqual(@as(f64, 1e4), c.sliding_rope.theta);
    try testing.expectEqual(@as(u32, 1), c.dense_layers);
    try testing.expectEqual(@as(u32, 13824), c.dense_inter);
    try testing.expectEqual(@as(u32, 512), c.experts);
    try testing.expectEqual(@as(u32, 16), c.top_k);
    try testing.expectEqual(@as(u32, 1536), c.moe_inter);
    try testing.expectEqual(@as(f32, 3.998), c.route_scale);
    try testing.expectEqual(Indexer{ .heads = 16, .head_dim = 128, .group = 4, .group_topk = 512 }, c.indexer);
    try testing.expectEqual(@as(u32, 4), c.hc_mult);
    try testing.expectEqual(@as(f32, 2.0), c.hc_magnitude);
    try testing.expectEqual(@as(u32, 120025), c.eos);
    try testing.expectEqual(@as(u32, 120120), c.image_id);
    try testing.expectEqual(@as(u32, 5), c.dfly.layers);
    try testing.expectEqual(@as(u32, 5), c.dfly.draft);
    try testing.expectEqual(@as(u32, 41), c.dfly.targets[4]);
    try testing.expectEqual(@as(u32, 1152), c.vision.?.hidden);
    try testing.expectEqual(@as(u32, 3072), c.vision.?.out_hidden);
}

test "hy4 config: a pack with no generation_config.json samples at the release's 0.9 and 1.0" {
    // The release's generation_config.json says temperature 0.9, top_p 1, no top_k; a converted pack may ship none.
    const model_mod = @import("../model.zig");
    var config = try model_mod.parseConfigFromJson(testing.allocator, release_file);
    defer config.deinit(testing.allocator);
    config.applyFamilySamplingDefaults();
    try testing.expectEqual(@as(?f32, 0.9), config.gen_temperature);
    try testing.expectEqual(@as(?f32, 1.0), config.gen_top_p);
    try testing.expectEqual(@as(?u32, null), config.gen_top_k);
}

test "hy4 config: the edit base reads as the release's own config.json" {
    try testing.expectEqualDeep(try parseText(release_file), try parseText(release_config));
}

test "hy4 config: Engram sits at 0-based layers 1, 11, 21, 31, hashes with the 1-based ids, and its primes sum to the table" {
    const c = try parseText(release_config);
    const e = c.engram;
    try testing.expectEqual(@as(u32, 4), e.n);
    try testing.expectEqualSlices(u32, &.{ 1, 11, 21, 31 }, e.layers[0..4]);
    try testing.expectEqualSlices(u32, &.{ 2, 12, 22, 32 }, e.hash_ids[0..4]);
    try testing.expectEqual(@as(u32, 16), e.primeCount());
    try testing.expectEqual(@as(u32, 96), e.head_dim);
    try testing.expectEqual(@as(u64, 10312007), e.primes[0][0]);
    try testing.expectEqual(@as(u64, 10313201), e.primes[3][15]);
    try testing.expectEqual(@as(u64, 660_006_416), e.totalRows());
    try testing.expectEqual(@as(u32, 120002), e.pad_id);
}

test "hy4 config: every sparse-layer spelling the references use reads as sparse" {
    for ([_][]const u8{ "sparse_attention", "indexed_attention" }) |spelling| {
        const quoted = try std.fmt.allocPrint(testing.allocator, "[\"{s}\"", .{spelling});
        defer testing.allocator.free(quoted);
        const c = try parseEdited("[\"deepseek_sparse_attention\"", quoted);
        try testing.expectEqual(LayerKind.sparse, c.kinds[0]);
        try testing.expectEqual(@as(u32, 12), c.sparseCount());
    }
}

test "hy4 config: a form the release does not take is refused by name" {
    const edits = [_][2][]const u8{
        .{ "\"sliding_attention\", \"sliding_attention\", \"sliding_attention\", \"deepseek_sparse_attention\"]", "\"sliding_attention\", \"sliding_attention\", \"sliding_attention\", \"full_attention\"]" },
        .{ "\"swa_pattern\": [0, 0, 1", "\"swa_pattern\": [1, 0, 1" },
        .{ "\"compressed_indexer_skip_fine_select\": true", "\"compressed_indexer_skip_fine_select\": false" },
        .{ "\"compressed_indexer_key_pooling\": \"gated\"", "\"compressed_indexer_key_pooling\": \"max\"" },
        .{ "\"moe_router_use_sigmoid\": true", "\"moe_router_use_sigmoid\": false" },
        .{ "\"n_group\": 1", "\"n_group\": 2" },
        .{ "\"routed_expert_hidden_size\": 1536", "\"routed_expert_hidden_size\": 3072" },
        .{ "\"engram_vocab_size\": [10312000, 10312000]", "\"engram_vocab_size\": [10312000, 10313000]" },
        .{ "\"2\": [[10312007, ", "\"2\": [[" },
        .{ "\"mlp_layer_types\": [\"dense\", \"sparse\"", "\"mlp_layer_types\": [\"dense\", \"dense\"" },
        .{ "\"rope_type\": \"default\", \"rope_theta\": 10000,", "\"rope_type\": \"yarn\", \"rope_theta\": 10000," },
        .{ "\"enable_ihc\": true", "\"enable_ihc\": false" },
        .{ "\"num_nextn_predict_layers\": 0", "\"num_nextn_predict_layers\": 1" },
        .{ "\"qk_head_dim\": 256", "\"qk_head_dim\": 192" },
        .{ "\"dfly_target_kv_layers\": [25, 29, 33, 37, 41]", "\"dfly_target_kv_layers\": [25, 29, 33, 37, 42]" },
        .{ "\"num_speculative_tokens\": 5", "\"num_speculative_tokens\": 17" },
    };
    for (edits) |e| try testing.expectError(error.UnsupportedHy4Config, parseEdited(e[0], e[1]));
}

test "hy4 config: a pack's vision_quant reads as affine 8 at the tower's group, naming exactly the matrices it keeps BF16, and any other form is refused by name" {
    const a = testing.allocator;
    try testing.expectEqual(@as(?VisionQuant, null), (try parseText(release_config)).vision.?.quant);
    const at = "\"dtype\": \"bfloat16\",";
    // The release keeps each block's 4304-wide down projection BF16.
    var kept: std.Io.Writer.Allocating = .init(a);
    defer kept.deinit();
    for (0..27) |l| try kept.writer.print("{s}{{\"tensor\": \"vit.layers.{d}.mlp.dense_4h_to_h.weight\", \"reason\": \"4304 inputs\"}}", .{ if (l == 0) "" else ", ", l });
    const all = kept.written();
    const first_end = std.mem.indexOf(u8, all, "}").? + 1;
    const stamp = struct {
        fn of(group: []const u8, list_text: []const u8) ![]u8 {
            return std.mem.concat(testing.allocator, u8, &.{ "{\"format\": \"affine\", \"bits\": 8, \"group_size\": ", group, ", \"bf16_kept\": [", list_text, "]}" });
        }
        fn parse(q: []const u8) !Config {
            const new = try std.mem.concat(testing.allocator, u8, &.{ "\"dtype\": \"bfloat16\", \"vision_quant\": ", q, "," });
            defer testing.allocator.free(new);
            return parseEdited("\"dtype\": \"bfloat16\",", new);
        }
    };
    {
        const good = try stamp.of("128", all);
        defer a.free(good);
        try testing.expectEqual(@as(?VisionQuant, .{ .bits = 8, .group = 128 }), (try stamp.parse(good)).vision.?.quant);
    }
    const one_missing = all[first_end + 2 ..];
    const but_last = all[0..std.mem.lastIndexOf(u8, all, ", {").?];
    const twice = try std.mem.concat(a, u8, &.{ all[0..first_end], ", ", but_last });
    defer a.free(twice);
    const affine_named = try std.mem.concat(a, u8, &.{ but_last, ", {\"tensor\": \"vit.layers.0.mlp.dense_h_to_4h.weight\", \"reason\": \"x\"}" });
    defer a.free(affine_named);
    const no_reason = try std.mem.replaceOwned(u8, a, all, "\"4304 inputs\"", "\"\"");
    defer a.free(no_reason);
    for ([_][2][]const u8{
        .{ "64", all },            .{ "128", one_missing }, .{ "128", twice }, .{ "128", affine_named },
        .{ "128", no_reason },
    }) |bad| {
        const q = try stamp.of(bad[0], bad[1]);
        defer a.free(q);
        try testing.expectError(error.UnsupportedHy4Config, stamp.parse(q));
    }
    for ([_][]const u8{
        "{\"format\": \"affine\", \"bits\": 4, \"group_size\": 128, \"bf16_kept\": []}",
        "{\"format\": \"stq\", \"bits\": 8, \"group_size\": 128, \"bf16_kept\": []}",
        "{\"format\": \"affine\", \"bits\": 8, \"group_size\": 128}",
        "\"affine8\"",
    }) |bad| try testing.expectError(error.UnsupportedHy4Config, stamp.parse(bad));
    const no_tower = try std.mem.concat(a, u8, &.{ release_config[0..std.mem.indexOf(u8, release_config, " \"vision_config\"").?], release_config[std.mem.indexOf(u8, release_config, " \"text_config\"").?..] });
    defer a.free(no_tower);
    try testing.expectEqual(@as(?Vision, null), (try parseText(no_tower)).vision);
    const at_no = std.mem.indexOf(u8, no_tower, at).? + at.len;
    const quant_no_tower = try std.mem.concat(a, u8, &.{ no_tower[0..at_no], " \"vision_quant\": {\"format\": \"affine\", \"bits\": 8, \"group_size\": 128, \"bf16_kept\": []},", no_tower[at_no..] });
    defer a.free(quant_no_tower);
    try testing.expectError(error.UnsupportedHy4Config, parseText(quant_no_tower));
    // The release's 1152 takes 128; the tiny's 32 takes 32.
    for ([_][2]u32{ .{ 1152, 128 }, .{ 32, 32 }, .{ 64, 64 }, .{ 96, 32 } }) |w| try testing.expectEqual(@as(?u32, w[1]), visionGroup(w[0]));
    try testing.expectEqual(@as(?u32, null), visionGroup(100));
}
