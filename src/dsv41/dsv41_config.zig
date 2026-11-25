//! DeepSeek-V4.1-Flash (`deepseek_v41`) geometry, read from `config.json`'s `text_config`.
//! Anything the forward does not implement is refused here by name.
const std = @import("std");
const log = @import("../log.zig");
const sushi_exl3 = @import("sushi_exl3");
const pool_class = @import("../tensor/pool_class.zig");
const engram = @import("../engram.zig");

pub const max_layers = 64;
pub const max_engram = engram.max_modules;
pub const max_ngram = engram.max_ngram;
pub const max_heads = engram.max_heads;
pub const max_targets = 8;
pub const max_block = 16;
/// Rows a backbone window ring keeps past the window while DSpark is on, so a verify chunk writes its rows and
/// still rolls back to any accepted prefix.
pub const ring_slack: u32 = 16;

/// The widest decode pass that queues its routed chains before their picks are read (`dsv41_model.specOn`). The queue
/// saves a wait a layer and loses a layer's work when a pick is outside the cache map; each row brings `top_k` picks.
pub const route_spec_rows: u32 = 8;

pub const Engram = engram.Geometry;

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

/// A wide forward's hyper-connection expand (the fused kernel, or `hcExpandOps`'s chain), Engram gate (the kernel, or
/// `engramGateOps`'s chain) and attention heads (every span's epilogue writing one grouped buffer, or a buffer a span
/// joined after): the chains hold planes the kernels do not.
pub const WideArms = struct {
    expand_fused: bool,
    engram_fused: bool,
    grouped_heads: bool,

    /// The served kernels' arms: a matrix-unit GPU with its fused kernels on.
    pub const kernels: WideArms = .{ .expand_fused = true, .engram_fused = true, .grouped_heads = true };

    /// A load's arms: `fused`, its kernels on; `expand_probed`, `hcExpandFused` reproduces this GPU's chain.
    pub fn of(fused: bool, expand_probed: bool) WideArms {
        return .{ .expand_fused = fused and expand_probed, .engram_fused = fused, .grouped_heads = fused };
    }
};

/// Who asked for decode on MLX instead of the engine of our own.

pub const Config = struct {
    /// DeepSeek's recommended serving for V4.1-Flash: a 1M window and a generation budget of at least 256K.
    pub fn serveDefaults(comptime D: type) D {
        return .{ .context = 1_000_000, .max_tokens = 256_000, .ram_budget = true };
    }

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
    /// The most rate groups one layer's routed experts span in an STQ pack (`ModelConfig.expert_rate_groups`, set
    /// once the pack's layout is read); 1 for an ungrouped pack.
    rate_groups: u8 = 1,
    /// Whether a routed chain can read a spill pool beside each rate group's cache pool (`Ragged.initSplit`): a streamed
    /// plan's route does when an expert misses the cache; a resident plan's cache holds every expert, so its route makes
    /// none. The shape sum bills the pools the route makes. True until the plan says resident (`adoptPlan`, and the load's
    /// own plan, `dsv41_serving.initDsv41`).
    spill_pools: bool = true,
    /// The wide forward's arms the shape sum bills (`dsv41_wide_bytes`): what the parse predicts for this GPU until
    /// the load records its own (`dsv41_model.setFused`). Null only for a config no parse made.
    wide_arms: ?WideArms = null,

    pub fn isDsparkTarget(self: *const Config, l: usize) bool {
        for (self.dspark.targets[0..self.dspark.n_targets]) |t| if (t == l) return true;
        return false;
    }

    /// The pack's layout gives a row one arm at every width: trellis experts and trunk (`Model.widthExact` checks the
    /// loaded linears). The release's FP8 trunk and FP4 experts take a product plan by row count.
    pub fn widthExactLayout(self: *const Config) bool {
        return self.expert_dec != null and self.trunk_dec != null;
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
    /// left them. A request's stores are made at its last row (`stateReservedBytes`), so they never grow.
    pub fn stateBytesPerToken(self: *const Config) u64 {
        var raw: u64 = 0;
        for (0..self.layers) |l| {
            const r: u64 = @max(self.ratios[l], 1);
            if (self.isKvSource(l)) raw += compRowBytes(self.head_dim) / r;
            if (self.isOwner(l)) raw += indexRowBytes(self.index_head_dim) / r;
        }
        return raw;
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

    /// The most one forward of `rows` ending at position `seq` may allocate beyond what is live when it starts: its
    /// transients and the state rows it appends. The transient cap installs it (`dsv41_borrow`) and admission bills it.
    /// A prefill-width forward holds the tensors `dsv41_wide_bytes` counts from its shapes, the state rows among them; a
    /// decode or verify block scores every key (`narrow_rows` rows, `score_copies` copies) beside `forward_held_bytes`.
    /// Both carry the bill's one margin; the row stores are made at the request's last row, so neither grows one.
    pub fn forwardCap(self: *const Config, rows: u64, seq: u64) u64 {
        const ctx = @max(seq, 1);
        if (rows > verify_rows) return wide_bytes.bytes(self, rows, ctx) +| BILL_MARGIN_BYTES;
        return narrow_rows * ctx * 4 * score_copies +| cap_fixed_bytes;
    }

    /// `forwardCap` of a load on MLX's stream: MLX's graph holds what its evaluation order keeps, which no shape sum
    /// of ours counts, so a prefill-width forward there keeps the per-row rate measured on it.
    pub fn mlxForwardCap(self: *const Config, rows: u64, seq: u64) u64 {
        if (rows <= verify_rows) return self.forwardCap(rows, seq) +| self.rowsGridBytes();
        return rows *| self.dim *| mlx_row_bytes_per_dim +| self.rowsGridBytes() +| cap_fixed_bytes;
    }

    /// The most MLX's page grid adds to the row stores (`Rows.pagedCap`): under one grid step of each packed buffer.
    pub fn rowsGridBytes(self: *const Config) u64 {
        const pages = @import("../mlx_pages.zig");
        var sum: u64 = 0;
        for (0..self.layers) |l| {
            if (self.isKvSource(l)) sum +|= pages.gridStepBytes(&.{ self.head_dim / 2, self.head_dim / 16 });
            if (self.isOwner(l)) sum +|= pages.gridStepBytes(&.{ self.index_head_dim / 2, self.index_head_dim / 32 });
        }
        return sum;
    }

    /// `forwardCap` on the side a load runs on: the engine of our own (`engine`) or MLX.
    pub fn forwardCapOn(self: *const Config, engine: bool, rows: u64, seq: u64) u64 {
        return if (engine) self.forwardCap(rows, seq) else self.mlxForwardCap(rows, seq);
    }

    /// The rows layer `l`'s row stores hold for a request that ends at token `end`, which they are made at up front
    /// (`dsv41_model.bindRequest`): the rows of `end` and of a verify block written past it.
    pub fn storeRows(self: *const Config, l: usize, end: u64) u64 {
        return (end +| verify_rows) / @max(self.ratios[l], 1);
    }

    /// What a request that ends at token `end` holds in its row stores, made at those rows up front: each store's codes
    /// and scales a pool buffer of their own on the engine, the rows' exact bytes on MLX (whose page grid `rowsGridBytes`
    /// adds to the caps).
    pub fn stateReservedBytes(self: *const Config, engine: bool, end: u64) u64 {
        var bytes: u64 = 0;
        for (0..self.layers) |l| {
            if (!self.isKvSource(l)) continue;
            const rows = self.storeRows(l, end);
            bytes += storeBytes(engine, rows, self.head_dim, 16);
            if (self.isOwner(l)) bytes += storeBytes(engine, rows, self.index_head_dim, 32);
        }
        return bytes;
    }

    /// What the row stores of a state at token `n` keep once their request has ended (`dsv41_model.Request.reset`): the
    /// rows they hold and a sixteenth over (`reservedRows`), each buffer at its pool size class.
    pub fn keptStoreBytes(self: *const Config, n: u64) u64 {
        var bytes: u64 = 0;
        for (0..self.layers) |l| {
            if (!self.isKvSource(l)) continue;
            const rows = n / @max(self.ratios[l], 1);
            bytes += if (rows == 0) 0 else packedStoreBytes(reservedRows(rows), self.head_dim, 16);
            if (self.isOwner(l) and rows > 0) bytes += packedStoreBytes(reservedRows(rows), self.index_head_dim, 32);
        }
        return bytes;
    }

    /// What a forward of `rows` rows from position `kv_len` makes of the row stores, which hold `caps` rows a layer: each
    /// store its rows overflow is regrown by `grownRows` (to the request's `end` when one is bound), its codes and its
    /// scales a buffer each at their pool size class. Returns those buffers' bytes; `caps` becomes what the stores hold after.
    pub fn growRowStores(self: *const Config, caps: *[max_layers]u32, kv_len: u64, rows: u64, end: ?u64) u64 {
        var bytes: u64 = 0;
        for (0..self.layers) |l| {
            if (!self.isKvSource(l)) continue;
            const r: u64 = @max(self.ratios[l], 1);
            const used = kv_len / r;
            const n = (kv_len +| rows) / r - used;
            if (used + n <= caps[l]) continue;
            const grown = grownRows(caps[l], used, n, if (end) |e| self.storeRows(l, e) else 0);
            caps[l] = @intCast(grown);
            bytes += packedStoreBytes(grown, self.head_dim, 16);
            if (self.isOwner(l)) bytes += packedStoreBytes(grown, self.index_head_dim, 32);
        }
        return bytes;
    }

    /// `forwardCap` of a joined decode forward, a row per request at `seqs`: each request's scores are its own,
    /// `cap_fixed_bytes` the forward's.
    pub fn joinedCap(self: *const Config, seqs: []const u64) u64 {
        return self.joinedCapOn(true, seqs);
    }

    /// `joinedCap` on the side a load runs on (`forwardCapOn`): on MLX each request's row stores take the grid's step.
    pub fn joinedCapOn(self: *const Config, engine: bool, seqs: []const u64) u64 {
        var sum: u64 = 0;
        for (seqs) |seq| sum +|= self.forwardCapOn(engine, 1, seq) - cap_fixed_bytes;
        return sum +| cap_fixed_bytes;
    }

    /// `forwardCap` of the widest forward a prefill of `seq` tokens runs at `chunk`-row chunks with the flat tail merge
    /// (`prefill_plan.widestForward`): a served request's is `bills.requestForwardRows`.
    pub fn chunkCap(self: *const Config, chunk: u64, seq: u64) u64 {
        const prefill_plan = @import("../prefill_plan.zig");
        const widest = prefill_plan.widestForward(@intCast(seq -| 1), @intCast(@max(chunk, 1)), false);
        return self.forwardCapOn(true, @max(@min(chunk, @max(seq, 1)), widest), seq);
    }
};

const act_bytes: u64 = 2;

/// A compressed KV row: FP4 codes and one E4M3 scale per 16.
fn compRowBytes(width: u64) u64 {
    return width / 2 + width / 16;
}

/// A packed store of `rows` rows of `width` values: its codes and its scales, each a pool buffer.
fn packedStoreBytes(rows: u64, width: u64, group: u64) u64 {
    return pool_class.poolBytes(@intCast(rows * (width / 2))) + pool_class.poolBytes(@intCast(rows * (width / group)));
}

/// `packedStoreBytes` on the engine, else the rows' bytes as they are.
fn storeBytes(engine: bool, rows: u64, width: u64, group: u64) u64 {
    return if (engine) packedStoreBytes(rows, width, group) else rows * (width / 2 + width / group);
}

/// An index key: FP4 codes and one E8M0 scale per 32.
fn indexRowBytes(width: u64) u64 {
    return width / 2 + width / 32;
}
/// The prefill chunk a request starts at on the GPU named `chip` (`chip.chipBrand()`), stepping down where its bill does
/// not fit (a named `--prefill-chunk-max` replaces it): with CED the bulk of a chunk is the encoder's 20 layers, whose
/// experts a wider chunk reads once for more rows. One width for every chip until a chip measures another.
pub fn prefillWidthFor(chip: []const u8) u32 {
    _ = chip;
    return 8192;
}

/// The widest chunk the load's own kernel warm runs (`dsv41_serving.warmAll`); the served warm (`cooler_warm`) runs a
/// request's widest rung.
pub const warm_width: u32 = 2048;

/// Rows the router scores in one dense product on a prefill chunk (`dsv41_model.gateLogitsWide`): a divisor of every
/// ladder rung from 2048 up, so those chunks pad nothing. A narrower forward pads one f32 tile a layer at a time (42 MB
/// at the release width), inside the cap's margin: the measured chunk peaks the cap is fitted to are taken with it.
pub const gate_tile_rows: u64 = 2048;

/// The MLX buffer pool in decode, which holds under 1 GB; the ledger bills it.
pub const dsv41_pool_bytes: u64 = 1 << 30;
/// The small tensors and tables a forward holds whatever its rows: the 0.12 GiB a decode forward was measured to hold.
pub const forward_held_bytes: u64 = 123 << 20;
/// The bill's one safety margin, by the name and value `bills.zig` gives the shared one, which replaces it.
pub const BILL_MARGIN_BYTES: u64 = 256 << 20;
/// What a decode or verify forward's cap carries beyond its scores and its context.
pub const cap_fixed_bytes: u64 = forward_held_bytes + BILL_MARGIN_BYTES;
const score_copies: u64 = 3;
/// Rows of a wide chunk one attention span takes: a span's scores and gathered rows are freed before the next.
pub const attn_span: u64 = 256;
/// Layers a prefill pass keeps in flight (`Model.pipe_depth`): their output streams stay in use until it waits on them.
pub const prefill_pipe_depth: usize = 2;
/// Peak bytes a prefill forward row on MLX's stream holds per hidden channel: measured on the 5120-wide release as 2.3
/// MiB a row from 256 to 2048 rows (2.8 MiB for one 363-row chunk, its state rows included), rounded up.
const mlx_row_bytes_per_dim: u64 = 480;
const wide_bytes = @import("dsv41_wide_bytes.zig");
/// Keys one pass of a wide span's streamed index top-k scores, as `dsv41_model` runs them.
pub const index_stream_tile: u64 = 32768;
/// Rows up to which the indexer scores a block with the scalar kernel, whole rows at a time.
pub const narrow_rows: u64 = 16;
/// The widest verify block, `t1` and 16 drafts: up to it a forward is a decode forward.
pub const verify_rows: u64 = 17;
/// A row store grows by its length over this: slack is that fraction of the state, and each growth copies the store.
pub const rows_grow_div: u64 = 16;

/// The rows a row store of `cap` rows that holds `used` has once it appends `n` (`dsv41_model.Rows.append`): as it is
/// while they fit, else the rows its request ends at (`goal`, the larger when the rows asked pass it) or, for a store no
/// request has bound (`goal` 0), the rows it needs or its capacity a sixteenth over, the larger, and at least 64.
pub fn grownRows(cap: u64, used: u64, n: u64, goal: u64) u64 {
    if (used + n <= cap) return cap;
    if (goal > 0) return @max(used + n, goal);
    return @max(used + n, @max(64, cap + cap / rows_grow_div));
}

/// The rows `dsv41_model.Rows.reserve` makes a store holding `used` rows: the slack its next append would have taken.
pub fn reservedRows(used: u64) u64 {
    return @max(64, used + @max(1, used / rows_grow_div));
}

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

/// The DeepSeek part of the parse, onto the shared `ModelConfig` (taken as `anytype`: this file stays free of `model.zig`).
pub fn parseFamily(config: anytype, root: std.json.ObjectMap, cfg_obj: std.json.ObjectMap) !void {
    var c = try parse(cfg_obj, root);
    // What a planner bills before the load probes its arms: the fused expand serves on the matrix-unit (NAX) GPUs.
    c.wide_arms = .of(true, sushi_exl3.kernels.gemmNaxOn());
    config.model_type = "deepseek_v41";
    config.weight_prefix = "";
    config.text_prompt_bos = config.bos_token_id;
    config.hidden_act = .silu;
    config.num_experts = c.experts;
    config.num_experts_per_tok = c.top_k;
    config.moe_intermediate_size = c.moe_inter;
    config.first_k_dense_replace = 0;
    config.rms_norm_eps = c.eps;
    config.has_vision = c.vision != null;
    if (c.vision) |v| {
        // The shared ViT geometry the media bills read.
        config.qv_depth = v.layers;
        config.qv_hidden = v.hidden;
        config.qv_heads = v.heads;
        config.qv_head_dim = v.headDim();
        config.qv_intermediate = v.inter;
        config.qv_out_hidden = c.dim;
        config.qv_patch = v.patch;
        config.qv_merge = v.downsample;
        config.qv_temporal_patch = 1;
    }
    if (root.get("trunk_quant")) |b| config.trunk_quant_rate = (try sushi_exl3.parseQuantBlock(b)).rate;
    config.dsv41 = c;
    config.serve_defaults = Config.serveDefaults(@TypeOf(config.serve_defaults));
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
            // DeepSeek's engram.py draws a module's multipliers from its layer index, with no seed.
            e.hash_ids[i] = @intCast(l.integer);
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

test "dsv41 forward cap: on MLX, covers what every measured peak added to the state it started from, with no multiplier" {
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
        try testing.expect(c.mlxForwardCap(m.rows, m.ctx) >= added);
    }
}

test "dsv41 forward cap: a wide forward's row term is the shape sum of its widest layer's tensors, for the tiny fixture's geometry and the release's" {
    var tiny = try parseText(stq_tiny_config);
    tiny.rate_groups = 3;
    const rel_stq = try releaseStqForTest();
    const rel = try parseText(release_config);
    const Case = struct { c: *const Config, rows: u64, shapes: u64 };
    // At `parseText`'s arms, the served kernels'. The tiny: layer 2 (Engram, KV source and index owner) at its routed chain over six pools; its trace at three
    // pools reads these less the three spill pools. The release: layer 14 (the same roles) over ten pools (the STQ
    // pack's combined chain) or the FP4 experts' sorted gather.
    for ([_]Case{
        .{ .c = &tiny, .rows = 512, .shapes = 22_872_064 },
        .{ .c = &tiny, .rows = 2048, .shapes = 89_899_008 },
        .{ .c = &tiny, .rows = 8192, .shapes = 358_449_152 },
        .{ .c = &rel_stq, .rows = 363, .shapes = 499_957_760 },
        .{ .c = &rel_stq, .rows = 2048, .shapes = 2_696_380_416 },
        .{ .c = &rel_stq, .rows = 4096, .shapes = 5_391_433_728 },
        .{ .c = &rel_stq, .rows = 8192, .shapes = 10_775_347_200 },
        .{ .c = &rel, .rows = 2048, .shapes = 1_333_067_776 },
    }) |k| {
        const got = k.c.forwardCap(k.rows, k.rows) - BILL_MARGIN_BYTES;
        errdefer std.debug.print("{d} rows: shape sum {d}\n", .{ k.rows, got });
        try testing.expectEqual(k.shapes, got);
    }
}

test "dsv41 forward cap: a resident plan's route makes no spill pool, so its wide forward bills the cache pools alone, never past the streamed worst case" {
    var tiny = try parseText(stq_tiny_config);
    tiny.rate_groups = 3;
    var rel_stq = try releaseStqForTest();
    const Case = struct { c: *Config, rows: u64, shapes: u64 };
    for ([_]Case{
        .{ .c = &tiny, .rows = 512, .shapes = 20_267_008 },
        .{ .c = &tiny, .rows = 8192, .shapes = 282_918_912 },
        .{ .c = &rel_stq, .rows = 363, .shapes = 433_356_800 },
        .{ .c = &rel_stq, .rows = 2048, .shapes = 2_327_232_512 },
        .{ .c = &rel_stq, .rows = 4096, .shapes = 4_653_219_840 },
        .{ .c = &rel_stq, .rows = 8192, .shapes = 9_298_886_656 },
    }) |k| {
        k.c.spill_pools = true;
        const streamed = k.c.forwardCap(k.rows, k.rows) - BILL_MARGIN_BYTES;
        k.c.spill_pools = false;
        const got = k.c.forwardCap(k.rows, k.rows) - BILL_MARGIN_BYTES;
        errdefer std.debug.print("{d} rows: resident shape sum {d}, streamed {d}\n", .{ k.rows, got, streamed });
        try testing.expectEqual(k.shapes, got);
        try testing.expectEqual(wide_bytes.bytesAt(k.c, k.rows, k.rows, k.c.rate_groups), got);
        try testing.expect(got < streamed);
    }
}

test "dsv41 forward cap: each config bills its own wide arms, whichever was priced or loaded before it" {
    const rows: u64 = 8192;
    var chain = try parseText(stq_tiny_config);
    chain.rate_groups = 3;
    chain.wide_arms = .of(false, true);
    var fused = chain;
    fused.wide_arms = .of(true, true);
    const chain_cap = chain.forwardCap(rows, rows);
    const fused_cap = fused.forwardCap(rows, rows);
    errdefer std.debug.print("chain arms {d}, kernel arms {d}\n", .{ chain_cap, fused_cap });
    // The chains' planes: four f32 `[L,hc,D]` and an f32 `[L,D]` per expand, eight f32 `[L,hc,D]` in the Engram write.
    try testing.expect(chain_cap > fused_cap);
    try testing.expectEqual(chain_cap, chain.forwardCap(rows, rows));
    try testing.expectEqual(fused_cap, fused.forwardCap(rows, rows));
}

test "dsv41 prefill width: a request starts at 8192 rows on every chip, the router's tile a whole number of times" {
    for ([_][]const u8{ "Apple M5 Ultra", "Apple M5 Max", "Apple M1 Max", "" }) |chip| {
        try testing.expectEqual(@as(u32, 8192), prefillWidthFor(chip));
        try testing.expectEqual(@as(u64, 0), prefillWidthFor(chip) % gate_tile_rows);
    }
}

test "dsv41 forward cap: a joined decode forward bills each request's one-row cap, the fixed bytes once" {
    const c = try parseText(release_config);
    for ([_]u64{ 1, 4096, 1 << 20 }) |seq| try testing.expectEqual(c.forwardCap(1, seq), c.joinedCap(&.{seq}));
    try testing.expectEqual(c.forwardCap(1, 4096) + c.forwardCap(1, 90_000) - cap_fixed_bytes, c.joinedCap(&.{ 4096, 90_000 }));
    try testing.expectEqual(3 * c.forwardCap(1, 9_000) - 2 * cap_fixed_bytes, c.joinedCap(&.{ 9_000, 9_000, 9_000 }));
}

test "dsv41 forward cap: grows with the rows, a decode block's with the context, a chunk's tail merge is priced, and 1M fits" {
    const c = try releaseStqForTest();
    const m: u64 = 1 << 20;
    try testing.expect(c.forwardCap(2048, 9_000) > c.forwardCap(1024, 9_000));
    try testing.expect(c.forwardCap(verify_rows, m) > c.forwardCap(1, 9_000));
    try testing.expect(c.forwardCap(verify_rows, m) * 3 < c.forwardCap(2048, m));
    // A prompt's widest forward is a chunk with the longest tail it absorbs; one 300 rows past a chunk runs them whole.
    try testing.expectEqual(c.forwardCap(2048 + 511, m), c.chunkCap(2048, m));
    try testing.expectEqual(c.forwardCap(2348, 2349), c.chunkCap(2048, 2349));
    // A prompt shorter than the chunk cannot run more rows than it has.
    try testing.expectEqual(c.forwardCap(100, 100), c.chunkCap(2048, 100));
    // A request's row stores are made whole at its last row, so a cap's slope in the context is the decode block's scores
    // alone, and a prefill-width cap has none.
    const slope = (c.forwardCap(1, m) - c.forwardCap(1, 9_000)) / (m - 9_000);
    try testing.expectEqual(narrow_rows * 4 * score_copies, slope);
    try testing.expectEqual(c.forwardCap(2048, 9_000), c.forwardCap(2048, m));
    // The extrapolation to 1M context: a decode forward under 1.05 GB, a tail-merged 2048-row chunk under 7.2 GB.
    try testing.expect(c.forwardCap(verify_rows, m) < 1_050_000_000);
    try testing.expect(c.chunkCap(2048, m) < 7_200_000_000);
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

/// The release as its STQ pack serves it (`DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw`): trellis experts and trunk, and
/// five rate groups at its widest layer.
pub fn releaseStqForTest() !Config {
    var c = try parseText(release_config);
    const quant =
        \\{"expert_quant": {"format": "stq", "k": 2.5, "codebook": "pair1", "window": 16, "codebook_table": "pair1-w16-v1", "codebook_digest": "c089b4eb"},
        \\ "trunk_quant": {"format": "stq", "k": 6, "codebook": "spec1", "window": 15, "codebook_table": "spec1-w15", "codebook_digest": "37f72431"}}
    ;
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, quant, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.TestUnexpectedResult;
    const blocks = parsed.value.object;
    c.expert_dec = try stqDecode(blocks.get("expert_quant").?);
    c.trunk_dec = try stqDecode(blocks.get("trunk_quant").?);
    c.rate_groups = 5;
    return c;
}

/// `DSV41_STQ_TINY`'s config: the release's layout of layers at a tiny width, three rate groups a layer.
const stq_tiny_config =
    \\{"model_type": "deepseek_v41", "image_token_id": 129264,
    \\ "text_config": {"vocab_size": 256, "hidden_size": 256, "moe_intermediate_size": 128, "num_hidden_layers": 8,
    \\  "num_attention_heads": 2, "num_key_value_heads": 1, "hidden_act": "silu", "head_dim": 128, "qk_rope_head_dim": 16,
    \\  "q_lora_rank": 128, "o_lora_rank": 128, "o_groups": 2, "swiglu_limit": 1.0, "rms_norm_eps": 1e-20,
    \\  "max_position_embeddings": 128, "rope_theta": 10000.0,
    \\  "rope_scaling": {"rope_type": "yarn", "factor": 4, "beta_fast": 32, "beta_slow": 1, "original_max_position_embeddings": 64},
    \\  "n_routed_experts": 8, "n_shared_experts": 1, "num_experts_per_tok": 3, "scoring_func": "sqrtsoftplus",
    \\  "topk_method": "noaux_tc", "norm_topk_prob": true, "routed_scaling_factor": 1.5, "sliding_window": 8,
    \\  "compress_ratios": [0, 0, 2, 2, 2, 1, 1, 1, 0, 0, 0], "compress_rope_theta": 40000.0, "kv_source_layer_ids": [2, 5],
    \\  "index_source_layer_ids": [2, 5, 7], "index_n_heads": 8, "index_head_dim": 128, "index_topk": 4,
    \\  "candidate_source_layer_id": 5, "candidate_topk_blocks": 3, "candidate_block_size": 2, "hc_mult": 4,
    \\  "hc_sinkhorn_iters": 20, "hc_eps": 1e-06, "engram_layer_ids": [1, 2], "engram_max_ngram_size": 4,
    \\  "engram_vocab_size": 50, "engram_n_heads": 2, "engram_head_dim": 64, "engram_pad_token_id": 2,
    \\  "engram_compressed_vocab_size": 256, "engram_num_embeddings": [384, 552], "num_nextn_predict_layers": 3,
    \\  "dspark_block_size": 4, "dspark_noise_token_id": 255, "dspark_target_layer_ids": [5, 6, 7], "dspark_markov_rank": 32,
    \\  "dspark_n_routed_experts": 4, "dspark_num_experts_per_tok": 2},
    \\ "expert_quant": {"format": "stq", "k": 4.0, "codebook": "pair1", "window": 16, "codebook_table": "pair1-w16-v1", "codebook_digest": "c089b4eb"},
    \\ "trunk_quant": {"format": "stq", "k": 6.0, "codebook": "spec1", "window": 15, "codebook_table": "spec1-w15", "codebook_digest": "37f72431"}}
;

/// Test fixtures elsewhere (the server's bills) parse the release config through it.
pub fn parseText(json: []const u8) !Config {
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, json, .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    var c = try parse(root.get("text_config").?.object, root);
    // A fixture prices the served kernels' arms on every box; a test of another arm sets its own.
    c.wide_arms = .kernels;
    return c;
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
