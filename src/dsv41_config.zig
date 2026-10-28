//! DeepSeek-V4.1-Flash (`deepseek_v41`) geometry, read from `config.json`'s `text_config`.
//! Anything the forward does not implement is refused here by name.
const std = @import("std");
const log = @import("log.zig");
const sushi_exl3 = @import("sushi_exl3");

pub const max_layers = 64;
pub const max_engram = 4;
pub const max_ngram = 8;
pub const max_heads = 16;
pub const max_targets = 8;
pub const max_block = 16;
/// Rows a backbone window ring keeps past the window while DSpark is on, so a verify chunk writes its rows and
/// still rolls back to any accepted prefix.
pub const ring_slack: u32 = 16;

pub const Engram = struct {
    n: u32 = 0,
    layers: [max_engram]u32 = @splat(0),
    rows: [max_engram]u64 = @splat(0),
    max_ngram: u32 = 0,
    vocab_size: u64 = 0,
    heads: u32 = 0,
    head_dim: u32 = 0,
    pad_id: u32 = 0,
    compressed_vocab: u32 = 0,

    pub fn cols(self: Engram) u32 {
        return (self.max_ngram - 1) * self.heads;
    }

    pub fn layerAt(self: Engram, layer: usize) ?usize {
        for (self.layers[0..self.n], 0..) |l, i| if (l == layer) return i;
        return null;
    }
};

/// The DSpark drafter the checkpoint ships in its `mtp.*` stages: a block `[t, noise...]` drafted in one pass over
/// `n_mtp` sliding-window stages that read the main stream of the target layers.
pub const Dspark = struct {
    block: u32 = 0,
    noise_id: u32 = 0,
    targets: [max_targets]u32 = @splat(0),
    n_targets: u32 = 0,
    rank: u32 = 0,
    experts: u32 = 0,
    top_k: u32 = 0,
};

/// The release's vision tower (`vision_config`): a ViT over 14 px patches, then an aligner that folds
/// `downsample x downsample` patches into one text-width token.
pub const Vision = struct {
    layers: u32,
    hidden: u32,
    heads: u32,
    inter: u32,
    patch: u32,
    rope_theta: f64,
    downsample: u32,
    max_tokens: u32,
    min_pixels: u32,
    /// An image wider than this many heights is stretched into the capped canvas; 0 = no cap.
    max_wh_ratio: u32,

    pub fn headDim(self: Vision) u32 {
        return self.hidden / self.heads;
    }

    /// The most ViT patches one image can hold: its span has `max_tokens - 2` aligner rows at most.
    pub fn maxPatches(self: Vision) u64 {
        return @as(u64, self.downsample) * self.downsample * (self.max_tokens - 2);
    }
};

pub const Config = struct {
    vocab: u32 = 0,
    dim: u32 = 0,
    layers: u32 = 0,
    heads: u32 = 0,
    head_dim: u32 = 0,
    rope_dim: u32 = 0,
    q_lora: u32 = 0,
    o_lora: u32 = 0,
    o_groups: u32 = 0,
    window: u32 = 0,
    eps: f32 = 0,
    max_pos: u32 = 0,

    experts: u32 = 0,
    top_k: u32 = 0,
    moe_inter: u32 = 0,
    swiglu_limit: f32 = 0,
    route_scale: f32 = 1,
    norm_topk: bool = true,

    rope_theta: f64 = 10000,
    compress_rope_theta: f64 = 10000,
    yarn_factor: f64 = 1,
    yarn_orig: u32 = 0,
    beta_fast: f64 = 32,
    beta_slow: f64 = 1,

    /// 0 = window only, r = compressed r-to-1; one entry per backbone layer.
    ratios: [max_layers]u8 = @splat(0),
    /// Bit l = layer l makes the shared compressed rows / the shared index selection.
    kv_sources: u64 = 0,
    index_sources: u64 = 0,
    index_heads: u32 = 0,
    index_head_dim: u32 = 0,
    index_topk: u32 = 0,
    cand_source: i32 = -1,
    cand_blocks: u32 = 0,
    cand_block_size: u32 = 0,

    hc_mult: u32 = 0,
    hc_iters: u32 = 0,
    hc_eps: f32 = 0,

    engram: Engram = .{},

    /// The DSpark stages (`n_mtp` of them; absent when `dspark.block` is 0).
    n_mtp: u32 = 0,
    dspark: Dspark = .{},
    /// Null when the checkpoint ships no tower or names no image token.
    vision: ?Vision = null,
    image_token_id: u32 = 0,

    /// An STQ pack's routed experts (`expert_quant`) and trunk banks (`trunk_quant`); null for the release's own
    /// FP4 experts and FP8 trunk.
    expert_dec: ?sushi_exl3.format.Decode = null,
    trunk_dec: ?sushi_exl3.format.Decode = null,

    pub fn isDsparkTarget(self: *const Config, l: usize) bool {
        for (self.dspark.targets[0..self.dspark.n_targets]) |t| if (t == l) return true;
        return false;
    }

    pub fn isKvSource(self: *const Config, l: usize) bool {
        return l < self.layers and (self.kv_sources >> @intCast(l)) & 1 != 0;
    }

    pub fn isIndexSource(self: *const Config, l: usize) bool {
        return l < self.layers and (self.index_sources >> @intCast(l)) & 1 != 0;
    }

    /// A layer that makes index keys: it compresses its own KV and is an index source.
    pub fn isOwner(self: *const Config, l: usize) bool {
        return self.isKvSource(l) and self.isIndexSource(l);
    }

    /// The latest layer at or below `l` that `pred` accepts; config parse proved one exists
    /// wherever a compressed layer asks.
    pub fn latest(self: *const Config, l: usize, comptime pred: fn (*const Config, usize) bool) usize {
        var j = l + 1;
        while (j > 0) : (j -= 1) if (pred(self, j - 1)) return j - 1;
        unreachable;
    }

    /// CED (the tech report's causal encoder-decoder): the decoder's first layer, when the layout has one. Every layer
    /// from the last KV source on reads its ratio-1 rows and picks of its own, so a wide prefill chunk can run them over
    /// its last `window` rows. Engram layers must sit in the encoder; the DSpark targets and the candidate source in
    /// the decoder, where the rows they read exist.
    pub fn cedSplit(self: *const Config) ?usize {
        if (self.window == 0 or self.kv_sources == 0) return null;
        const split: usize = 63 - @clz(self.kv_sources);
        if (split == 0 or split >= self.layers or !self.isIndexSource(split)) return null;
        for (split..self.layers) |l| if (self.ratios[l] != 1) return null;
        for (self.engram.layers[0..self.engram.n]) |l| if (l >= split) return null;
        for (self.dspark.targets[0..self.dspark.n_targets]) |l| if (l < split) return null;
        if (self.cand_source >= 0 and self.cand_source < split) return null;
        return split;
    }

    pub fn qkvOutRows(self: *const Config) u32 {
        return self.heads * self.head_dim;
    }

    /// Bytes ONE token adds to a request's state: the compressed KV rows of each KV source (FP4 with an E4M3 scale per
    /// 16) and the index keys of each owner (MXFP4), at their `ratio`-to-1 compression, stored as the round trips
    /// left them. Growth slack and the copy a growth makes are transient (`forwardCap`).
    pub fn stateBytesPerToken(self: *const Config) u64 {
        var raw: u64 = 0;
        for (0..self.layers) |l| {
            const r: u64 = @max(self.ratios[l], 1);
            if (self.isKvSource(l)) raw += compRowBytes(self.head_dim) / r;
            if (self.isOwner(l)) raw += indexRowBytes(self.index_head_dim) / r;
        }
        return raw;
    }

    /// The largest one layer's state per token: a growth holds its old rows beside the new buffer.
    fn worstLayerBytesPerToken(self: *const Config) u64 {
        var worst: u64 = 0;
        for (0..self.layers) |l| {
            const r: u64 = @max(self.ratios[l], 1);
            var b: u64 = 0;
            if (self.isKvSource(l)) b += compRowBytes(self.head_dim) / r;
            if (self.isOwner(l)) b += indexRowBytes(self.index_head_dim) / r;
            worst = @max(worst, b);
        }
        return worst;
    }

    /// What a request holds whatever its length: every layer's window ring and open pooling group, and with DSpark
    /// on its stages' windows and the backbone rings' verify slack.
    pub fn stateFixedBytes(self: *const Config, dspark: bool) u64 {
        var pending: u64 = 0;
        for (0..self.layers) |l| {
            // An open pooling group: up to ratio - 1 rows of kv and of gate, f32.
            if (self.isKvSource(l) and self.ratios[l] > 1) pending += (@as(u64, self.ratios[l]) - 1) * self.head_dim * 4 * 2;
        }
        const base = @as(u64, self.layers) * self.window * self.head_dim * act_bytes + pending;
        if (!dspark) return base;
        return base + (@as(u64, self.n_mtp) * self.window + @as(u64, self.layers) * ring_slack) * self.head_dim * act_bytes;
    }

    /// The most one forward of `rows` ending at position `seq` may allocate through MLX beyond what is live when it
    /// starts: its transients and the state rows it appends. The transient cap installs it (`dsv41_borrow`) and
    /// admission bills it. A prefill-width forward costs `cap_row_bytes_per_dim` a row and a channel; a decode or
    /// verify block scores every key (`narrow_rows` rows, `score_copies` copies) and its rows fit in the slack. Both
    /// carry the rows' growth: its slack plus one layer's old rows beside the new buffer during the copy.
    pub fn forwardCap(self: *const Config, rows: u64, seq: u64) u64 {
        const ctx = @max(seq, 1);
        const by_rows = if (rows > verify_rows) rows *| self.dim *| cap_row_bytes_per_dim else narrow_rows * ctx * 4 * score_copies;
        const growth = ctx * (self.stateBytesPerToken() / rows_grow_div + self.worstLayerBytesPerToken());
        return (by_rows +| growth) *| 5 / 4 +| cap_slack_bytes;
    }

    /// `forwardCap` of a prefill chunk of `chunk` rows ending at `seq`: the last chunk of a prompt absorbs a tail of
    /// up to `tail_merge_rows`, so the widest forward is `chunk + tail_merge_rows` rows.
    pub fn chunkCap(self: *const Config, chunk: u64, seq: u64) u64 {
        return self.forwardCap(@min(chunk +| tail_merge_rows, @max(seq, 1)), seq);
    }
};

const act_bytes: u64 = 2;

/// A compressed KV row: FP4 codes and one E4M3 scale per 16.
fn compRowBytes(width: u64) u64 {
    return width / 2 + width / 16;
}

/// An index key: FP4 codes and one E8M0 scale per 32.
fn indexRowBytes(width: u64) u64 {
    return width / 2 + width / 32;
}
/// The prefill chunk a load runs at when its bill fits (an explicit `--prefill-chunk` outranks it): with CED the bulk of a
/// chunk is the encoder's 20 layers, whose experts a wider chunk reads once for more rows.
pub const prefill_width: u32 = 2048;
/// Added to every cap: twice the 0.12 GiB of small tensors and tables a decode forward was measured to hold.
const cap_slack_bytes: u64 = 256 << 20;
const score_copies: u64 = 3;
/// `generate.TAIL_MERGE_MAX - 1`: the longest tail the last chunk of a prompt absorbs.
pub const tail_merge_rows: u64 = 511;
/// Peak bytes a prefill forward row holds per hidden channel: measured on the 5120-wide release as 2.3 MiB a row from
/// 256 to 2048 rows (2.8 MiB for one 363-row chunk, its state rows included), rounded up.
const cap_row_bytes_per_dim: u64 = 480;
/// Keys one pass of a wide span's streamed index top-k scores, as `dsv41_model` runs them.
pub const index_stream_tile: u64 = 32768;
/// Rows up to which the indexer scores a block with the scalar kernel, whole rows at a time.
pub const narrow_rows: u64 = 16;
/// The widest verify block, `t1` and 16 drafts: up to it a forward is a decode forward.
pub const verify_rows: u64 = 17;
/// A row store grows by its length over this: slack is that fraction of the state, and each growth copies the store.
pub const rows_grow_div: u64 = 16;

fn refuse(comptime fmt: []const u8, args: anytype) error{UnsupportedDsv41Config} {
    log.err("deepseek_v41: " ++ fmt ++ "\n", args);
    return error.UnsupportedDsv41Config;
}

fn field(obj: std.json.ObjectMap, key: []const u8) ?std.json.Value {
    const v = obj.get(key) orelse return null;
    return if (v == .null) null else v;
}

fn uint(comptime T: type, obj: std.json.ObjectMap, key: []const u8) !T {
    const v = field(obj, key) orelse return refuse("missing {s}", .{key});
    if (v != .integer) return refuse("{s} is not an integer", .{key});
    return std.math.cast(T, v.integer) orelse refuse("{s} is out of range", .{key});
}

fn num(obj: std.json.ObjectMap, key: []const u8) !f64 {
    const v = field(obj, key) orelse return refuse("missing {s}", .{key});
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => refuse("{s} is not a number", .{key}),
    };
}

fn str(obj: std.json.ObjectMap, key: []const u8) ![]const u8 {
    const v = field(obj, key) orelse return refuse("missing {s}", .{key});
    if (v != .string) return refuse("{s} is not a string", .{key});
    return v.string;
}

fn intList(obj: std.json.ObjectMap, key: []const u8) ![]const std.json.Value {
    const v = field(obj, key) orelse return refuse("missing {s}", .{key});
    if (v != .array) return refuse("{s} is not a list", .{key});
    return v.array.items;
}

fn layerMask(obj: std.json.ObjectMap, key: []const u8, layers: u32) !u64 {
    var mask: u64 = 0;
    for (try intList(obj, key)) |v| {
        if (v != .integer or v.integer < 0 or v.integer >= layers) return refuse("{s} names a layer outside the {d} backbone layers", .{ key, layers });
        mask |= @as(u64, 1) << @intCast(v.integer);
    }
    return mask;
}

fn parseDspark(text: std.json.ObjectMap, c: Config) !Dspark {
    var d: Dspark = .{};
    d.block = try uint(u32, text, "dspark_block_size");
    d.noise_id = try uint(u32, text, "dspark_noise_token_id");
    d.rank = try uint(u32, text, "dspark_markov_rank");
    d.experts = try uint(u32, text, "dspark_n_routed_experts");
    d.top_k = try uint(u32, text, "dspark_num_experts_per_tok");
    const targets = try intList(text, "dspark_target_layer_ids");
    if (targets.len == 0 or targets.len > max_targets) return refuse("{d} DSpark target layers", .{targets.len});
    for (targets, 0..) |v, i| {
        if (v != .integer or v.integer < 0 or v.integer >= c.layers) return refuse("dspark_target_layer_ids[{d}] is not a backbone layer", .{i});
        d.targets[i] = @intCast(v.integer);
    }
    d.n_targets = @intCast(targets.len);
    if (c.n_mtp == 0 or d.block > max_block or d.rank == 0 or d.experts == 0 or d.experts > 1024 or d.top_k == 0 or d.top_k > d.experts or d.noise_id >= c.vocab)
        return refuse("DSpark geometry (block {d}, {d} stages, rank {d}, experts {d}, top-k {d})", .{ d.block, c.n_mtp, d.rank, d.experts, d.top_k });
    return d;
}

fn parseVision(obj: std.json.ObjectMap) !Vision {
    const v: Vision = .{
        .layers = try uint(u32, obj, "num_hidden_layers"),
        .hidden = try uint(u32, obj, "hidden_size"),
        .heads = try uint(u32, obj, "num_attention_heads"),
        .inter = try uint(u32, obj, "intermediate_size"),
        .patch = try uint(u32, obj, "patch_size"),
        .rope_theta = if (field(obj, "rope_theta") != null) try num(obj, "rope_theta") else 10000,
        .downsample = try uint(u32, obj, "downsample_ratio"),
        .max_tokens = try uint(u32, obj, "max_image_tokens"),
        .min_pixels = try uint(u32, obj, "min_pixels"),
        .max_wh_ratio = if (field(obj, "max_wh_ratio") != null) try uint(u32, obj, "max_wh_ratio") else 0,
    };
    // The 2-D rotary splits a head into row and column halves of rotate-half pairs.
    if (v.layers == 0 or v.heads == 0 or v.hidden % v.heads != 0 or v.headDim() % 4 != 0 or v.inter == 0 or v.patch == 0 or
        v.downsample == 0 or v.max_tokens < 4)
        return refuse("vision geometry ({d} layers, hidden {d}, {d} heads, patch {d}, downsample {d})", .{ v.layers, v.hidden, v.heads, v.patch, v.downsample });
    return v;
}

/// `text` is `text_config` (or the root of a flat config), `root` the file's top level.
pub fn parse(text: std.json.ObjectMap, root: std.json.ObjectMap) !Config {
    var c: Config = .{};
    c.vocab = try uint(u32, text, "vocab_size");
    c.dim = try uint(u32, text, "hidden_size");
    c.layers = try uint(u32, text, "num_hidden_layers");
    c.heads = try uint(u32, text, "num_attention_heads");
    c.head_dim = try uint(u32, text, "head_dim");
    c.rope_dim = try uint(u32, text, "qk_rope_head_dim");
    c.q_lora = try uint(u32, text, "q_lora_rank");
    c.o_lora = try uint(u32, text, "o_lora_rank");
    c.o_groups = try uint(u32, text, "o_groups");
    c.window = try uint(u32, text, "sliding_window");
    c.eps = @floatCast(try num(text, "rms_norm_eps"));
    c.max_pos = try uint(u32, text, "max_position_embeddings");
    c.experts = try uint(u32, text, "n_routed_experts");
    c.top_k = try uint(u32, text, "num_experts_per_tok");
    c.moe_inter = try uint(u32, text, "moe_intermediate_size");
    c.swiglu_limit = @floatCast(try num(text, "swiglu_limit"));
    c.route_scale = @floatCast(try num(text, "routed_scaling_factor"));
    c.norm_topk = if (field(text, "norm_topk_prob")) |v| (v == .bool and v.bool) else true;
    c.rope_theta = try num(text, "rope_theta");
    c.compress_rope_theta = try num(text, "compress_rope_theta");
    const rs_val = field(text, "rope_scaling") orelse return refuse("missing rope_scaling", .{});
    if (rs_val != .object) return refuse("rope_scaling is not an object", .{});
    const rs = rs_val.object;
    if (!std.mem.eql(u8, try str(rs, "rope_type"), "yarn")) return refuse("rope_scaling.rope_type is not yarn", .{});
    c.yarn_factor = try num(rs, "factor");
    c.beta_fast = try num(rs, "beta_fast");
    c.beta_slow = try num(rs, "beta_slow");
    c.yarn_orig = try uint(u32, rs, "original_max_position_embeddings");

    if (!std.mem.eql(u8, try str(text, "scoring_func"), "sqrtsoftplus")) return refuse("scoring_func is not sqrtsoftplus", .{});
    if (!std.mem.eql(u8, try str(text, "topk_method"), "noaux_tc")) return refuse("topk_method is not noaux_tc", .{});
    if (!std.mem.eql(u8, try str(text, "hidden_act"), "silu")) return refuse("hidden_act is not silu", .{});
    if ((try uint(u32, text, "n_shared_experts")) != 1) return refuse("n_shared_experts is not 1", .{});
    if ((try uint(u32, text, "num_key_value_heads")) != 1) return refuse("num_key_value_heads is not 1", .{});
    if (field(text, "attention_bias")) |v| if (v == .bool and v.bool) return refuse("attention_bias is on", .{});

    if (c.layers == 0 or c.layers > max_layers or c.dim == 0 or c.heads == 0 or c.head_dim == 0 or
        c.rope_dim == 0 or c.rope_dim > c.head_dim or c.rope_dim % 2 != 0 or c.window == 0 or
        c.experts == 0 or c.top_k == 0 or c.top_k > c.experts or c.o_groups == 0 or c.heads % c.o_groups != 0 or
        c.eps <= 0 or c.dim % 32 != 0 or c.head_dim % 32 != 0 or c.q_lora % 32 != 0)
        return refuse("geometry (layers {d}, dim {d}, heads {d}, head_dim {d}, rope {d}, experts {d}, top-k {d})", .{ c.layers, c.dim, c.heads, c.head_dim, c.rope_dim, c.experts, c.top_k });

    const ratios = try intList(text, "compress_ratios");
    if (ratios.len < c.layers) return refuse("compress_ratios lists {d} layers for {d}", .{ ratios.len, c.layers });
    for (ratios[0..c.layers], 0..) |v, l| {
        if (v != .integer or v.integer < 0 or v.integer > 128) return refuse("compress_ratios[{d}] is not a ratio", .{l});
        c.ratios[l] = @intCast(v.integer);
    }
    c.kv_sources = try layerMask(text, "kv_source_layer_ids", c.layers);
    c.index_sources = try layerMask(text, "index_source_layer_ids", c.layers);
    c.index_heads = try uint(u32, text, "index_n_heads");
    c.index_head_dim = try uint(u32, text, "index_head_dim");
    c.index_topk = try uint(u32, text, "index_topk");
    if (field(text, "candidate_source_layer_id")) |v| {
        if (v != .integer) return refuse("candidate_source_layer_id is not an integer", .{});
        c.cand_source = std.math.cast(i32, v.integer) orelse return refuse("candidate_source_layer_id is out of range", .{});
        c.cand_blocks = try uint(u32, text, "candidate_topk_blocks");
        c.cand_block_size = try uint(u32, text, "candidate_block_size");
    }

    // A compressed layer reads the latest KV source at or below it and, when it has its own indexer
    // or borrows one, the latest index source and index-key owner: each must exist.
    for (0..c.layers) |l| {
        const compressed = c.ratios[l] > 0;
        const kv = c.isKvSource(l);
        const ix = c.isIndexSource(l);
        if ((kv or ix) and !compressed) return refuse("layer {d} is a source but has no compression ratio", .{l});
        if (kv and c.ratios[l] > 1 and c.index_head_dim % 32 != 0) return refuse("index_head_dim {d} is not a multiple of 32", .{c.index_head_dim});
        if (!compressed) continue;
        if (c.kv_sources & ((@as(u64, 2) << @intCast(l)) -% 1) == 0) return refuse("layer {d} compresses with no KV source below it", .{l});
        if (c.index_sources & ((@as(u64, 2) << @intCast(l)) -% 1) == 0) return refuse("layer {d} compresses with no index source below it", .{l});
        if (ix) {
            var owner = false;
            for (0..l + 1) |j| owner = owner or c.isOwner(j);
            if (!owner) return refuse("index source {d} has no index-key owner below it", .{l});
        }
    }
    if (c.cand_source >= 0) {
        const cs: usize = @intCast(c.cand_source);
        if (cs >= c.layers or !c.isIndexSource(cs)) return refuse("candidate source {d} is not an index source", .{c.cand_source});
        if (c.cand_blocks == 0 or c.cand_block_size == 0) return refuse("candidate source without blocks", .{});
    }
    if (c.index_heads == 0 or c.index_head_dim == 0 or c.index_topk == 0) {
        if (c.index_sources != 0) return refuse("an index source without index geometry", .{});
    }

    c.hc_mult = try uint(u32, text, "hc_mult");
    c.hc_iters = try uint(u32, text, "hc_sinkhorn_iters");
    c.hc_eps = @floatCast(try num(text, "hc_eps"));
    if (c.hc_mult != 4 or c.hc_iters == 0 or c.hc_iters > 100 or c.hc_eps <= 0) return refuse("hyper-connection geometry (mult {d}, {d} iterations)", .{ c.hc_mult, c.hc_iters });

    if (field(text, "engram_layer_ids")) |ids| {
        if (ids != .array) return refuse("engram_layer_ids is not a list", .{});
        if (ids.array.items.len > max_engram) return refuse("{d} engram layers", .{ids.array.items.len});
        const rows = try intList(text, "engram_num_embeddings");
        if (rows.len != ids.array.items.len) return refuse("{d} engram tables for {d} layers", .{ rows.len, ids.array.items.len });
        var e: Engram = .{ .n = @intCast(ids.array.items.len) };
        for (ids.array.items, rows, 0..) |l, r, i| {
            if (l != .integer or l.integer < 0 or l.integer >= c.layers or r != .integer or r.integer <= 0) return refuse("engram layer {d}", .{i});
            e.layers[i] = @intCast(l.integer);
            e.rows[i] = @intCast(r.integer);
        }
        if (e.n > 0) {
            e.max_ngram = try uint(u32, text, "engram_max_ngram_size");
            e.vocab_size = try uint(u64, text, "engram_vocab_size");
            e.heads = try uint(u32, text, "engram_n_heads");
            e.head_dim = try uint(u32, text, "engram_head_dim");
            e.pad_id = try uint(u32, text, "engram_pad_token_id");
            e.compressed_vocab = try uint(u32, text, "engram_compressed_vocab_size");
            if (e.max_ngram < 2 or e.max_ngram > max_ngram or e.heads == 0 or e.heads > max_heads or
                e.head_dim == 0 or e.head_dim % 32 != 0 or e.compressed_vocab == 0 or e.vocab_size == 0)
                return refuse("engram geometry ({d}-grams, {d} heads x {d})", .{ e.max_ngram, e.heads, e.head_dim });
        }
        c.engram = e;
    }

    if (field(text, "num_nextn_predict_layers")) |v| if (v == .integer and v.integer > 0) {
        c.n_mtp = @intCast(v.integer);
    };
    if (field(text, "dspark_block_size")) |v| {
        if (v == .integer and v.integer > 0) c.dspark = try parseDspark(text, c);
    }
    if (field(root, "image_token_id")) |v| if (v == .integer and v.integer > 0) {
        c.image_token_id = @intCast(v.integer);
    };
    if (field(root, "vision_config")) |v| if (v == .object and c.image_token_id != 0) {
        c.vision = try parseVision(v.object);
    };
    if (field(root, "expert_quant")) |b| {
        c.expert_dec = try stqDecode(b);
        // The trellis chain clamps at a whole limit.
        if (c.swiglu_limit != @round(c.swiglu_limit) or c.swiglu_limit < 1 or c.swiglu_limit > 128)
            return refuse("swiglu_limit {d} is not a whole number in 1..128, which STQ experts need", .{c.swiglu_limit});
    }
    if (field(root, "trunk_quant")) |b| c.trunk_dec = try stqDecode(b);
    return c;
}

fn stqDecode(block: std.json.Value) !sushi_exl3.format.Decode {
    const spec = try sushi_exl3.parseQuantBlock(block);
    if (!spec.codebook.hasTable()) return refuse("a quant block names {s}, not an STQ codebook", .{@tagName(spec.codebook)});
    return .{ .codebook = spec.codebook, .window = spec.window, .table = spec.table };
}

/// The config of the checkpoint in `dir_path`.
pub fn readFile(io: std.Io, gpa: std.mem.Allocator, dir_path: []const u8) !Config {
    var dir = try std.Io.Dir.openDirAbsolute(io, dir_path, .{});
    defer dir.close(io);
    const raw = try dir.readFileAlloc(io, "config.json", gpa, .limited(1 << 20));
    defer gpa.free(raw);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, raw, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return refuse("config.json is not an object", .{});
    const root = parsed.value.object;
    const text = if (root.get("text_config")) |t| (if (t == .object) t.object else return refuse("text_config is not an object", .{})) else root;
    return parse(text, root);
}

const testing = std.testing;

test "dsv41 bills: a token costs its compressed rows" {
    const c = try parseText(release_config);
    // KV sources and index owners 2, 8, 14 compress 2:1 and 20 keeps a row per token: main rows 288 B, index keys
    // 68 B, the release's 890 B per token.
    try testing.expectEqual(@as(u64, 890), c.stateBytesPerToken());
    // 40 bf16 window rings of 128 rows and the open pooling group of the three 2:1 sources.
    try testing.expectEqual(@as(u64, 40 * 128 * 512 * 2 + 3 * 512 * 4 * 2), c.stateFixedBytes(false));
}

const gib: u64 = 1 << 30;

fn gibs(x: f64) u64 {
    return @intFromFloat(x * @as(f64, @floatFromInt(gib)));
}

test "dsv41 forward cap: covers a quarter over what every measured peak added to the state it started from" {
    const c = try parseText(release_config);
    // Peaks of MLX's own bytes (imported weights left out, state rows in), STQ pack, budget 108, binary 26a28040. The
    // cap is over the level a forward starts at, which holds the state of `ctx` tokens at least.
    const peaks = [_]struct { rows: u64, ctx: u64, peak: f64 }{
        .{ .rows = 1, .ctx = 9_000, .peak = 0.116 },
        .{ .rows = 1, .ctx = 225_000, .peak = 0.336 },
        .{ .rows = verify_rows, .ctx = 225_000, .peak = 0.336 },
        .{ .rows = 2048, .ctx = 2048, .peak = 4.57 },
        .{ .rows = 2048, .ctx = 225_000, .peak = 4.772 },
        .{ .rows = 1183, .ctx = 1183, .peak = 2.479 },
        .{ .rows = 1000, .ctx = 1000, .peak = 2.253 },
        .{ .rows = 363, .ctx = 363, .peak = 1.02 },
        .{ .rows = 256, .ctx = 256, .peak = 0.572 },
        .{ .rows = 42, .ctx = 42, .peak = 0.121 },
        .{ .rows = 31, .ctx = 225_000, .peak = 0.336 },
    };
    for (peaks) |m| {
        const added = gibs(m.peak) -| m.ctx * c.stateBytesPerToken();
        try testing.expect(c.forwardCap(m.rows, m.ctx) >= added * 5 / 4);
    }
}

test "dsv41 forward cap: grows with the rows and the context, a chunk's tail merge is priced, and 1M fits" {
    const c = try parseText(release_config);
    const m: u64 = 1 << 20;
    try testing.expect(c.forwardCap(2048, 9_000) > c.forwardCap(1024, 9_000));
    try testing.expect(c.forwardCap(2048, m) > c.forwardCap(2048, 9_000));
    try testing.expect(c.forwardCap(verify_rows, m) > c.forwardCap(1, 9_000));
    try testing.expect(c.forwardCap(verify_rows, m) * 4 < c.forwardCap(2048, m));
    try testing.expectEqual(c.forwardCap(2048 + tail_merge_rows, m), c.chunkCap(2048, m));
    // A prompt shorter than the chunk cannot run more rows than it has.
    try testing.expectEqual(c.forwardCap(100, 100), c.chunkCap(2048, 100));
    // The measured growth beyond the state rows (1094 B a token less 890) is a fraction of the cap's per-token slope.
    const measured: u64 = (gibs(0.336) - gibs(0.116)) / (225_000 - 9_000) - c.stateBytesPerToken();
    const slope = (c.forwardCap(1, m) - c.forwardCap(1, 9_000)) / (m - 9_000);
    try testing.expect(slope >= measured * 5 / 4);
    // The extrapolation to 1M context: a decode forward under 1.1 GB, a tail-merged 2048-row chunk under 8.8 GB.
    try testing.expect(c.forwardCap(verify_rows, m) < 1_100_000_000);
    try testing.expect(c.chunkCap(2048, m) < 8_800_000_000);
}

/// The V4.1-Flash release's `config.json`.
pub const release_config =
    \\{"architectures": ["DeepseekV41ForCausalLM"], "model_type": "deepseek_v41", "dtype": "bfloat16",
    \\ "bos_token_id": 0, "eos_token_id": 1, "pad_token_id": 2, "image_token_id": 129264,
    \\ "quantization_config": {"quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [32, 32], "scale_fmt": "ue8m0", "expert_dtype": "fp4"},
    \\ "text_config": {"model_type": "deepseek_v41_text", "vocab_size": 129280, "hidden_size": 5120,
    \\  "moe_intermediate_size": 2304, "num_hidden_layers": 40, "num_attention_heads": 64,
    \\  "num_key_value_heads": 1, "head_dim": 512, "qk_rope_head_dim": 64, "q_lora_rank": 1280,
    \\  "o_lora_rank": 1024, "o_groups": 8, "hidden_act": "silu", "swiglu_limit": 10.0,
    \\  "rms_norm_eps": 1e-20, "attention_bias": false, "tie_word_embeddings": false,
    \\  "max_position_embeddings": 1048576, "rope_theta": 10000,
    \\  "rope_scaling": {"rope_type": "yarn", "factor": 16, "beta_fast": 32, "beta_slow": 1, "original_max_position_embeddings": 65536},
    \\  "n_routed_experts": 384, "n_shared_experts": 1, "num_experts_per_tok": 6,
    \\  "scoring_func": "sqrtsoftplus", "topk_method": "noaux_tc", "norm_topk_prob": true,
    \\  "routed_scaling_factor": 1.5, "sliding_window": 128,
    \\  "compress_ratios": [0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    \\  "compress_rope_theta": 160000, "kv_source_layer_ids": [2, 8, 14, 20],
    \\  "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36], "index_n_heads": 32, "index_head_dim": 128,
    \\  "index_topk": 512, "candidate_source_layer_id": 20, "candidate_topk_blocks": 2048, "candidate_block_size": 8,
    \\  "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-06, "engram_layer_ids": [1, 14],
    \\  "engram_num_embeddings": [384006168, 384016682], "engram_max_ngram_size": 4, "engram_vocab_size": 16000000,
    \\  "engram_n_heads": 8, "engram_head_dim": 256, "engram_pad_token_id": 2, "engram_compressed_vocab_size": 99092,
    \\  "num_nextn_predict_layers": 3, "dspark_block_size": 5, "dspark_noise_token_id": 128799,
    \\  "dspark_target_layer_ids": [37, 38, 39], "dspark_markov_rank": 256, "dspark_n_routed_experts": 128,
    \\  "dspark_num_experts_per_tok": 3},
    \\ "vision_config": {"model_type": "deepseek_v41_vision", "num_hidden_layers": 32, "hidden_size": 1024,
    \\  "num_attention_heads": 16, "intermediate_size": 2816, "patch_size": 14, "rope_theta": 10000, "downsample_ratio": 3,
    \\  "max_image_tokens": 1024, "min_pixels": 295936, "max_wh_ratio": null}}
;

/// Test fixtures elsewhere (the server's bills) parse the release config through it.
pub fn parseText(json: []const u8) !Config {
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, json, .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    return parse(root.get("text_config").?.object, root);
}

test "dsv41 config: the V4.1-Flash release parses to its geometry" {
    const c = try parseText(release_config);
    try testing.expectEqual(@as(u32, 40), c.layers);
    try testing.expectEqual(@as(u32, 5120), c.dim);
    try testing.expectEqual(@as(u32, 512), c.head_dim);
    try testing.expectEqual(@as(u32, 64), c.rope_dim);
    try testing.expectEqual(@as(u32, 384), c.experts);
    try testing.expectEqual(@as(u32, 6), c.top_k);
    try testing.expectEqual(@as(u8, 2), c.ratios[2]);
    try testing.expectEqual(@as(u8, 1), c.ratios[39]);
    try testing.expectEqual(@as(u64, (1 << 2) | (1 << 8) | (1 << 14) | (1 << 20)), c.kv_sources);
    try testing.expectEqual(@as(u64, (1 << 2) | (1 << 8) | (1 << 14) | (1 << 20) | (1 << 24) | (1 << 28) | (1 << 32) | (1 << 36)), c.index_sources);
    try testing.expect(c.isOwner(20) and !c.isOwner(24) and c.isIndexSource(24));
    try testing.expectEqual(@as(usize, 20), c.latest(25, Config.isOwner));
    try testing.expectEqual(@as(i32, 20), c.cand_source);
    try testing.expectEqual(@as(u32, 2048), c.cand_blocks);
    try testing.expectEqual(@as(u32, 2), c.engram.n);
    try testing.expectEqual(@as(u32, 14), c.engram.layers[1]);
    try testing.expectEqual(@as(u64, 384016682), c.engram.rows[1]);
    try testing.expectEqual(@as(u32, 24), c.engram.cols());
    try testing.expectEqual(@as(?usize, 1), c.engram.layerAt(14));
    try testing.expectEqual(@as(u32, 3), c.n_mtp);
    try testing.expectEqual(@as(u32, 5), c.dspark.block);
    try testing.expectEqual(@as(u32, 128799), c.dspark.noise_id);
    try testing.expectEqualSlices(u32, &.{ 37, 38, 39 }, c.dspark.targets[0..c.dspark.n_targets]);
    try testing.expectEqual(@as(u32, 256), c.dspark.rank);
    try testing.expectEqual(@as(u32, 128), c.dspark.experts);
    try testing.expectEqual(@as(u32, 3), c.dspark.top_k);
    const v = c.vision.?;
    try testing.expectEqual(@as(u32, 32), v.layers);
    try testing.expectEqual(@as(u32, 64), v.headDim());
    try testing.expectEqual(@as(u32, 2816), v.inter);
    try testing.expectEqual(@as(u32, 14), v.patch);
    try testing.expectEqual(@as(u32, 3), v.downsample);
    try testing.expectEqual(@as(u32, 1024), v.max_tokens);
    try testing.expectEqual(@as(u32, 295936), v.min_pixels);
    try testing.expectEqual(@as(u32, 0), v.max_wh_ratio);
    try testing.expectEqual(@as(u64, 9 * 1022), v.maxPatches());
    try testing.expectEqual(@as(u32, 129264), c.image_token_id);
    try testing.expectEqual(@as(f64, 160000), c.compress_rope_theta);
    try testing.expectEqual(@as(f64, 16), c.yarn_factor);
}

test "dsv41 config: a tower needs its image token, and a geometry the rotary cannot split is refused by name" {
    const no_token = try std.mem.replaceOwned(u8, testing.allocator, release_config, "\"image_token_id\": 129264", "\"image_token_id\": 0");
    defer testing.allocator.free(no_token);
    try testing.expect((try parseText(no_token)).vision == null);
    const odd = try std.mem.replaceOwned(u8, testing.allocator, release_config, "\"num_attention_heads\": 16", "\"num_attention_heads\": 24");
    defer testing.allocator.free(odd);
    try testing.expectError(error.UnsupportedDsv41Config, parseText(odd));
}

test "dsv41 config: the CED decoder starts at the last KV source when every layer after it reads its ratio-1 rows" {
    const c = try parseText(release_config);
    try testing.expectEqual(@as(?usize, 20), c.cedSplit());
    var t = c;
    t.ratios[30] = 2;
    try testing.expectEqual(@as(?usize, null), t.cedSplit());
    t = c;
    t.engram.layers[1] = 25;
    try testing.expectEqual(@as(?usize, null), t.cedSplit());
    t = c;
    t.dspark.targets[0] = 19;
    try testing.expectEqual(@as(?usize, null), t.cedSplit());
    t = c;
    t.cand_source = 14;
    try testing.expectEqual(@as(?usize, null), t.cedSplit());
    t = c;
    t.index_sources &= ~@as(u64, 1 << 20);
    try testing.expectEqual(@as(?usize, null), t.cedSplit());
    t = c;
    t.window = 0;
    try testing.expectEqual(@as(?usize, null), t.cedSplit());
}

test "dsv41 config: an STQ pack names its expert and trunk codebooks, and a fractional clamp is refused by name" {
    const stq = "\"expert_quant\": {\"format\": \"stq\", \"k\": 4, \"codebook\": \"pair1\", \"window\": 16, \"codebook_table\": \"pair1-w16-v1\", \"codebook_digest\": \"c089b4eb\"}, " ++
        "\"trunk_quant\": {\"format\": \"stq\", \"k\": 6, \"codebook\": \"spec1\", \"window\": 15, \"codebook_table\": \"spec1-w15\", \"codebook_digest\": \"37f72431\"}, ";
    const json = try std.mem.replaceOwned(u8, testing.allocator, release_config, "\"dtype\": \"bfloat16\",", stq ++ "\"dtype\": \"bfloat16\",");
    defer testing.allocator.free(json);
    const c = try parseText(json);
    try testing.expectEqual(sushi_exl3.format.Codebook.pair1, c.expert_dec.?.codebook);
    try testing.expectEqual(sushi_exl3.format.Window.w16, c.expert_dec.?.window);
    try testing.expectEqual(sushi_exl3.format.Codebook.spec1, c.trunk_dec.?.codebook);
    try testing.expectEqual(sushi_exl3.format.Window.w15, c.trunk_dec.?.window);
    try testing.expect(c.trunk_dec.?.table != null);
    try testing.expect((try parseText(release_config)).trunk_dec == null);
    const frac = try std.mem.replaceOwned(u8, testing.allocator, json, "\"swiglu_limit\": 10.0", "\"swiglu_limit\": 7.5");
    defer testing.allocator.free(frac);
    try testing.expectError(error.UnsupportedDsv41Config, parseText(frac));
    const mcg = try std.mem.replaceOwned(u8, testing.allocator, json, "\"codebook\": \"spec1\", \"window\": 15, \"codebook_table\": \"spec1-w15\", \"codebook_digest\": \"37f72431\"", "\"codebook\": \"mcg\"");
    defer testing.allocator.free(mcg);
    try testing.expectError(error.ExpertLayoutUnsupported, parseText(mcg));
}

test "dsv41 config: shapes the forward does not implement are refused by name" {
    const cases = [_]struct { from: []const u8, to: []const u8 }{
        .{ .from = "\"scoring_func\": \"sqrtsoftplus\"", .to = "\"scoring_func\": \"sigmoid\"" },
        .{ .from = "\"n_shared_experts\": 1", .to = "\"n_shared_experts\": 2" },
        .{ .from = "\"num_key_value_heads\": 1", .to = "\"num_key_value_heads\": 8" },
        .{ .from = "\"hc_mult\": 4", .to = "\"hc_mult\": 2" },
        .{ .from = "\"kv_source_layer_ids\": [2, 8, 14, 20]", .to = "\"kv_source_layer_ids\": [3]" },
        .{ .from = "\"candidate_source_layer_id\": 20", .to = "\"candidate_source_layer_id\": 21" },
        .{ .from = "\"engram_n_heads\": 8", .to = "\"engram_n_heads\": 0" },
        .{ .from = "\"rope_type\": \"yarn\"", .to = "\"rope_type\": \"linear\"" },
    };
    for (cases) |case| {
        const json = try std.mem.replaceOwned(u8, testing.allocator, release_config, case.from, case.to);
        defer testing.allocator.free(json);
        try testing.expect(!std.mem.eql(u8, json, release_config));
        try testing.expectError(error.UnsupportedDsv41Config, parseText(json));
    }
}

test "dsv41 config: DSpark geometry the forward cannot run is refused by name" {
    const cases = [_]struct { from: []const u8, to: []const u8 }{
        .{ .from = "\"dspark_target_layer_ids\": [37, 38, 39]", .to = "\"dspark_target_layer_ids\": [37, 38, 40]" },
        .{ .from = "\"dspark_target_layer_ids\": [37, 38, 39]", .to = "\"dspark_target_layer_ids\": []" },
        .{ .from = "\"dspark_num_experts_per_tok\": 3", .to = "\"dspark_num_experts_per_tok\": 129" },
        .{ .from = "\"dspark_block_size\": 5", .to = "\"dspark_block_size\": 17" },
        .{ .from = "\"num_nextn_predict_layers\": 3", .to = "\"num_nextn_predict_layers\": 0" },
    };
    for (cases) |case| {
        const json = try std.mem.replaceOwned(u8, testing.allocator, release_config, case.from, case.to);
        defer testing.allocator.free(json);
        try testing.expect(!std.mem.eql(u8, json, release_config));
        try testing.expectError(error.UnsupportedDsv41Config, parseText(json));
    }
}

test "dsv41 bills: DSpark's stage windows and the backbone's verify slack are billed only when it is on" {
    const c = try parseText(release_config);
    const rings: u64 = 3 * 128 * 512 * 2;
    const slack: u64 = 40 * ring_slack * 512 * 2;
    try testing.expectEqual(c.stateFixedBytes(false) + rings + slack, c.stateFixedBytes(true));
}
