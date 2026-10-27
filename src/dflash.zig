//! DFlash block-drafter — generic block-parallel speculative decoding.
//!
//! A small assistant transformer drafts `block_size - 1` tokens in ONE
//! forward, conditioned on the trunk's intermediate hidden states. Mirrors
//! transformers' `DFlashTokenCandidateGenerator` + `DFlashCache` and the
//! `muse_glimmer_assistant` reference model. DFlash is a METHOD, not a
//! model-family feature: detection keys on the CONFIG CONTRACT
//! (`block_size` + `mask_token_id` + `target_layer_ids`), never on the
//! `model_type` string.
//!
//! Protocol per round (assistant borrows the trunk's embed table + lm_head):
//!   1. Context delta = trunk hiddens at the OUTPUT of layers
//!      `target_layer_ids` for the tokens the trunk just accepted,
//!      concatenated on features → `encoder.fc` → RMS norm → per-layer K/V
//!      appended to the assistant cache at those tokens' absolute positions.
//!   2. `noise_embeds` = RAW trunk embedding lookup (no norm, no scale) of
//!      `[anchor(t1), mask_token × (block_size-1)]`.
//!   3. Assistant forward: Q from the block only; K/V = cached context +
//!      fresh block K/V. Block attends bidirectionally among itself,
//!      sliding-windowed/full to context.
//!   4. Draft logits = trunk `lm_head(hidden)[:, 1:]` — the anchor position
//!      is DROPPED → block_size-1 drafts.
//!   5. Trunk verify over `[t1, drafts...]` (standard spec verify invariant);
//!      the verify forward doubles as the next round's context producer.
//!   6. Block K/V are NEVER cached (reference evicts them via
//!      `crop(-block_size)`; we append them into spare capacity and
//!      truncate back — the committed context is unchanged).

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const model_mod = @import("model.zig");
const transformer_mod = @import("transformer.zig");
// The chunked row requantizer + its packed-triple handle are shared with the
// MTP head — both sidecars shrink the SAME trunk lm_head for drafts only, and
// one requantizer with one chunking discipline is the point.
const mtp_mod = @import("mtp.zig");

const Weights = model_mod.Weights;
const ModelConfig = model_mod.ModelConfig;
const Transformer = transformer_mod.Transformer;
const KVCache = transformer_mod.KVCache;

pub const LayerType = enum {
    sliding_attention,
    full_attention,

    pub fn fromString(s: []const u8) !LayerType {
        if (std.mem.eql(u8, s, "sliding_attention")) return .sliding_attention;
        if (std.mem.eql(u8, s, "full_attention")) return .full_attention;
        return error.UnknownLayerType;
    }
};

pub const DflashConfig = struct {
    hidden_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    intermediate_size: u32,
    rms_norm_eps: f32,
    rope_theta: f32,
    sliding_window: u32,
    layer_types: []LayerType, // owned
    // ── The DFlash contract fields ──
    block_size: u32,
    mask_token_id: u32,
    target_layer_ids: []u32, // owned, ascending
    // ── DFlash2 extension (all 0/absent on a v1 assistant) ──
    selector_rank: u32 = 0,
    selector_top_k: u32 = 0,
    conv_kernel_size: u32 = 0,
    conv_group_size: u32 = 0,
    /// Reference `compute_logits`: `logits * output_multiplier`, then
    /// `tanh(l/cap)*cap`. The trunk head we borrow is the BARE Linear — an
    /// argmax draft is invariant to these (monotone), but the selector SUMS
    /// unary logits with codebook edges, so the muse sidecar's declared
    /// scale is load-bearing there. 0 cap / 1.0 multiplier = no transform.
    logit_softcap: f32 = 0,
    output_multiplier: f32 = 1.0,
    // ── DSpark extension (0/false on a v1/v2 assistant) ──
    /// Rank of the vanilla Markov head's low-rank bigram bias. > 0 means the
    /// block is drafted SEMI-autoregressively: each step's logits get
    /// `markov_w2(markov_w1[prev_token])` added before its own draft is
    /// picked, chaining the block without a second assistant forward.
    markov_rank: u32 = 0,
    /// Draft row mapping. SpecForge DFlash exports (muse, z-lab DFlash2) DROP
    /// the anchor row — mask row j predicts draft j. DSpark exports read ALL
    /// rows and the ANCHOR row emits draft 0, so their `block_size` counts
    /// DRAFTS, not verify width. Reading the wrong mapping is a silent
    /// 0%-acceptance drafter, not an error.
    anchor_row_drafts: bool = false,
    /// GPT-J interleaved rope (`rope_is_neox_style: false`). Every DFlash
    /// v1/v2 sidecar so far is neox (half-split); DSpark's is not, and the
    /// wrong half rotates silently.
    rope_traditional: bool = false,

    pub fn isDflash2(self: *const DflashConfig) bool {
        return self.selector_rank > 0 or self.conv_kernel_size > 0;
    }

    pub fn isDspark(self: *const DflashConfig) bool {
        return self.markov_rank > 0;
    }

    pub fn deinit(self: *DflashConfig, allocator: std.mem.Allocator) void {
        allocator.free(self.layer_types);
        allocator.free(self.target_layer_ids);
    }
};

/// Does this config.json declare the DFlash contract? ALL THREE fields must
/// be present — `model_type` (`*_assistant`) is only the discovery-level
/// "this is a drafter" signal, never the DFlash detection.
pub fn isDflashConfigJson(root: std.json.ObjectMap) bool {
    return dflashContractObject(root) != null;
}

/// The object holding the DFlash contract triple: DFlash2 checkpoints nest it
/// under `dflash_config`, v1 assistants declare it at the root. Null when
/// neither shape declares all three fields.
/// The contract fields, looked up NESTED-FIRST then at the root. DSpark
/// splits the triple across both (`block_size` at the root, the rest under
/// `dflash_config`), so neither object alone answers the question and a
/// nested-only reader silently classifies the sidecar as "not a drafter".
const Contract = struct {
    nested: ?std.json.ObjectMap,
    root: std.json.ObjectMap,

    fn get(self: Contract, key: []const u8) ?std.json.Value {
        if (self.nested) |n| {
            if (n.get(key)) |v| return v;
        }
        return self.root.get(key);
    }
};

fn dflashContractObject(root: std.json.ObjectMap) ?Contract {
    const nested: ?std.json.ObjectMap = blk: {
        if (root.get("dflash_config")) |dc| {
            if (dc == .object) break :blk dc.object;
        }
        break :blk null;
    };
    const c = Contract{ .nested = nested, .root = root };
    if (c.get("block_size") == null) return null;
    if (c.get("mask_token_id") == null) return null;
    if (c.get("target_layer_ids") == null) return null;
    return c;
}

/// Read `<dir>/config.json` and answer whether it declares DFlash. Any
/// read/parse failure is a quiet false — the caller falls through to the
/// gemma drafter loader, whose own errors are the user-facing ones.
pub fn probeIsDflash(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) bool {
    const content = readConfigFile(io, allocator, model_dir) catch return false;
    defer allocator.free(content);
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, content, .{}) catch return false;
    defer parsed.deinit();
    if (parsed.value != .object) return false;
    return isDflashConfigJson(parsed.value.object);
}

fn readConfigFile(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) ![]u8 {
    const path = try std.fmt.allocPrint(allocator, "{s}/config.json", .{model_dir});
    defer allocator.free(path);
    const file = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer file.close(io);
    var read_buf: [4096]u8 = undefined;
    var reader_state = file.reader(io, &read_buf);
    return try reader_state.interface.allocRemaining(allocator, .limited(1 << 20));
}

/// Conventional in-checkpoint home for a DFlash assistant: a subdirectory of
/// the model's own directory declaring the config contract. It is what
/// turns the drafter from a LAUNCH-flag dependency into a LOAD-time one — a
/// hot model switch brings its own drafter, no pairing table decides which
/// sidecar goes with which checkpoint, and a mismatched pair is unbuildable.
pub const IN_DIR_SUBDIR = "drafter";
pub const DFLASH2_IN_DIR_SUBDIR = "dflash2";
pub const SHIPPED_GLM_SUBDIR = "GLM-5.3-Flash-DFlash2";

/// Prefer `<model_dir>/dflash2`, then the legacy `<model_dir>/drafter`,
/// when the folder declares the DFlash contract; otherwise return null.
/// Caller owns the returned path. Only ever consulted when no explicit
/// `--drafter` was given — an explicit flag always wins, so an external
/// sidecar can still be pointed at a merged checkpoint.
pub fn resolveInDirDrafter(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) ?[]u8 {
    if (model_dir.len == 0 or !std.fs.path.isAbsolute(model_dir)) return null;
    for ([_][]const u8{ DFLASH2_IN_DIR_SUBDIR, IN_DIR_SUBDIR, SHIPPED_GLM_SUBDIR }) |subdir| {
        const path = std.fs.path.join(allocator, &.{ model_dir, subdir }) catch return null;
        if (probeIsDflash(io, allocator, path) and (!std.mem.eql(u8, subdir, DFLASH2_IN_DIR_SUBDIR) or @import("glm5_dflash_cache.zig").cacheValid(io, allocator, model_dir, path))) return path;
        allocator.free(path);
    }
    return null;
}

pub fn parseConfig(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !DflashConfig {
    const content = try readConfigFile(io, allocator, model_dir);
    defer allocator.free(content);
    return parseConfigFromJson(allocator, content);
}

pub fn parseConfigFromJson(allocator: std.mem.Allocator, content: []const u8) !DflashConfig {
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, content, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.NotDflashConfig;
    const root = parsed.value.object;

    const contract = dflashContractObject(root) orelse return error.NotDflashConfig;

    // DSpark export markers. `markov_rank` is the load-bearing one (the head
    // IS DSpark); the architecture string and `projector_type` cover exports
    // that ship the row convention without one.
    var anchor_row_drafts = root.get("markov_rank") != null;
    if (contract.get("projector_type")) |pt| if (pt == .string and std.mem.eql(u8, pt.string, "dspark")) {
        anchor_row_drafts = true;
    };
    if (root.get("architectures")) |archs| if (archs == .array) {
        for (archs.array.items) |a| {
            if (a == .string and std.mem.indexOf(u8, a.string, "DSpark") != null) anchor_row_drafts = true;
        }
    };

    const declared_block: u32 = try jsonU32(contract.get("block_size").?);
    // Normalize to ENGINE semantics: block_size is the verify width
    // (1 anchor + drafts) everywhere below this line.
    const block_size: u32 = if (anchor_row_drafts) declared_block + 1 else declared_block;
    if (block_size < 2) return error.InvalidDflashBlockSize;
    const mask_token_id: u32 = try jsonU32(contract.get("mask_token_id").?);

    // Our block forward is bidirectional-only (v1 parity-pinned); the z-lab
    // reference defaults sliding layers CAUSAL inside the block. DFlash2
    // ships an explicit false — a true must refuse, not silently mis-attend.
    if (root.get("is_causal")) |ic| {
        if (ic == .bool and ic.bool) return error.DflashCausalBlockUnsupported;
    }

    const selector_rank: u32 = if (contract.get("selector_rank")) |v| try jsonU32(v) else 0;
    const selector_top_k: u32 = if (contract.get("selector_top_k")) |v| try jsonU32(v) else 0;
    const conv_kernel_size: u32 = if (contract.get("conv_kernel_size")) |v| try jsonU32(v) else 0;
    const conv_group_size: u32 = if (contract.get("conv_group_size")) |v| try jsonU32(v) else 0;
    // The pairs travel together: a selector needs a candidate width, a conv
    // needs a group width. Half-declared is a converter bug worth naming.
    if ((selector_rank > 0) != (selector_top_k > 1)) return error.InvalidDflashConfigValue;
    if ((conv_kernel_size > 0) != (conv_group_size > 0)) return error.InvalidDflashConfigValue;

    // Reference load_draft reads the softcap from the dflash section, falling
    // back to the root; the multiplier lives in the dflash section only.
    var logit_softcap: f32 = 0;
    if (contract.get("final_logit_softcapping") orelse root.get("final_logit_softcapping")) |v| logit_softcap = jsonFloat(v);
    var output_multiplier: f32 = 1.0;
    if (contract.get("output_multiplier")) |v| {
        output_multiplier = jsonFloat(v);
        if (output_multiplier == 0) return error.InvalidDflashConfigValue;
    }

    const tl_val = contract.get("target_layer_ids").?;
    if (tl_val != .array or tl_val.array.items.len == 0) return error.InvalidDflashTargetLayers;
    const target_layer_ids = try allocator.alloc(u32, tl_val.array.items.len);
    errdefer allocator.free(target_layer_ids);
    for (tl_val.array.items, 0..) |elem, i| {
        target_layer_ids[i] = try jsonU32(elem);
        // The encoder concatenates in list order and the fc weight is trained
        // against it; a non-ascending list is a converter bug worth naming.
        if (i > 0 and target_layer_ids[i] <= target_layer_ids[i - 1]) return error.InvalidDflashTargetLayers;
    }

    const hidden_size: u32 = try jsonU32(root.get("hidden_size") orelse return error.IncompleteDflashConfig);
    if (conv_group_size > 0 and hidden_size % conv_group_size != 0) return error.InvalidDflashConfigValue;
    const num_layers: u32 = try jsonU32(root.get("num_hidden_layers") orelse return error.IncompleteDflashConfig);
    const n_heads: u32 = try jsonU32(root.get("num_attention_heads") orelse return error.IncompleteDflashConfig);
    const kv_heads: u32 = if (root.get("num_key_value_heads")) |v| try jsonU32(v) else n_heads;
    const head_dim: u32 = try jsonU32(root.get("head_dim") orelse return error.IncompleteDflashConfig);
    const intermediate: u32 = try jsonU32(root.get("intermediate_size") orelse return error.IncompleteDflashConfig);
    const eps: f32 = jsonFloat(root.get("rms_norm_eps") orelse return error.IncompleteDflashConfig);
    const sliding: u32 = if (root.get("sliding_window")) |v| try jsonU32(v) else 0;

    // `rope_parameters.rope_theta` is the transformers-5 spelling (muse,
    // DFlash2); DSpark ships a flat root `rope_theta`. Reading only the
    // nested one leaves theta at 10000 and every drafted position mis-rotated.
    var rope_theta: f32 = 10000.0;
    if (root.get("rope_theta")) |t| rope_theta = jsonFloat(t);
    if (root.get("rope_parameters")) |rp| if (rp == .object) {
        if (rp.object.get("rope_theta")) |t| rope_theta = jsonFloat(t);
    };
    var rope_traditional = false;
    if (root.get("rope_is_neox_style")) |v| {
        if (v == .bool) rope_traditional = !v.bool;
    }

    // DSpark: only the `vanilla` Markov head is ported. `gated` and `rnn`
    // carry extra trained modules (gate_proj / joint_proj) our block forward
    // has no arm for — refuse by name rather than draft from the base logits
    // and read as a bad drafter.
    const markov_rank: u32 = if (root.get("markov_rank")) |v| try jsonU32(v) else 0;
    if (markov_rank > 0) {
        const kind = if (root.get("markov_head_type")) |v| (if (v == .string) v.string else "vanilla") else "vanilla";
        if (!std.mem.eql(u8, kind, "vanilla")) return error.UnsupportedMarkovHeadType;
    }

    const lt_val = root.get("layer_types") orelse return error.IncompleteDflashConfig;
    if (lt_val != .array or lt_val.array.items.len != num_layers) return error.LayerTypesLengthMismatch;
    const layer_types = try allocator.alloc(LayerType, num_layers);
    errdefer allocator.free(layer_types);
    for (lt_val.array.items, 0..) |elem, i| {
        if (elem != .string) return error.InvalidLayerTypeEntry;
        layer_types[i] = try LayerType.fromString(elem.string);
        if (layer_types[i] == .sliding_attention and sliding == 0) return error.IncompleteDflashConfig;
    }

    return DflashConfig{
        .hidden_size = hidden_size,
        .num_hidden_layers = num_layers,
        .num_attention_heads = n_heads,
        .num_key_value_heads = kv_heads,
        .head_dim = head_dim,
        .intermediate_size = intermediate,
        .rms_norm_eps = eps,
        .rope_theta = rope_theta,
        .sliding_window = sliding,
        .layer_types = layer_types,
        .block_size = block_size,
        .mask_token_id = mask_token_id,
        .target_layer_ids = target_layer_ids,
        .selector_rank = selector_rank,
        .selector_top_k = selector_top_k,
        .conv_kernel_size = conv_kernel_size,
        .conv_group_size = conv_group_size,
        .logit_softcap = logit_softcap,
        .output_multiplier = output_multiplier,
        .markov_rank = markov_rank,
        .rope_traditional = rope_traditional,
        .anchor_row_drafts = anchor_row_drafts,
    };
}

fn jsonU32(v: std.json.Value) !u32 {
    return switch (v) {
        .integer => |i| if (i >= 0 and i <= std.math.maxInt(u32)) @intCast(i) else error.InvalidDflashConfigValue,
        else => error.InvalidDflashConfigValue,
    };
}

fn jsonFloat(v: std.json.Value) f32 {
    return switch (v) {
        .float => |f| @floatCast(f),
        .integer => |i| @floatFromInt(i),
        else => 0.0,
    };
}

/// Every `target_layer_ids` entry must name an existing trunk layer — the
/// capture seam retains `hidden_states[i+1]`, so `i` must be < trunk depth.
/// Named error at load, not warmup death.
pub fn validateTargetLayers(ids: []const u32, trunk_num_layers: u32) !void {
    for (ids) |id| {
        if (id >= trunk_num_layers) return error.DflashTargetLayerOutOfRange;
    }
}

/// Widest verify a machine without the NAX m16 tile serves well: the block
/// IS the verify width, and `transformer.vqmmLaneFor`'s split-K lane stops at
/// M=7. Past it MLX's own 32x32-tiled `qmm_splitk` takes over and a trunk
/// forward jumps from ~1.3x a serial step to ~4.2x, for barely more accepted
/// tokens. Measured on Muse 4-bit / applegpu_g16s, 160-token generations,
/// serial reference in the same boot (28.2 tok/s):
///   block 16 → 0.92x   8 → 1.16x   7 → 1.43x   6 → 1.79x   5 → 1.97x
///   4 → 1.81x   3 → 1.77x
/// The block is a REQUEST cost, not a training constant — the assistant
/// drafts against the same mask token at any width. NAX-capable machines
/// (M5-class) have a real M 8..16 lane and keep the checkpoint's block.
pub const NO_WIDE_LANE_BLOCK_CAP: u32 = 5;

/// A no-wide-lane block cap with the machine row it came from, for the
/// `DFlash drafter ready` line — a capped block must say WHY in tester logs.
pub const BlockCap = struct {
    cap: u32,
    label: []const u8,
    /// True when a HUMAN measured this chip. A measured row wins over the
    /// probe: the probe times a forward, while these rows were measured as
    /// realized throughput (acceptance included), which the forward ladder
    /// cannot see.
    measured: bool = false,
};

/// Per-silicon cap table for machines WITHOUT the NAX m16 verify lane. The
/// cap is a MACHINE measurement, so each row is one: the M4 row is
/// NO_WIDE_LANE_BLOCK_CAP (see above); the M3 Ultra row is 8 on oMLX PR
/// #2850's evidence — block-8 DFlash2 measured 1.33-1.43x over serial at
/// T=0.7 on the same Qwen3.8-27B + incoai drafter pairing there. New
/// silicon rows are one-liners (an M1 row lands when the user measures it).
/// `chip` is sysctl machdep.cpu.brand_string ("Apple M3 Ultra"); the GPU
/// arch string cannot tell Ultra from Max, hence the CPU brand.
pub fn blockCapForMachine(chip: []const u8) BlockCap {
    if (std.mem.indexOf(u8, chip, "M3 Ultra") != null) return .{ .cap = 8, .label = "m3-ultra", .measured = true };
    // The DEFAULT VALUE and the M4 ROW are the same number doing two
    // different jobs: on an M4 it is the measured sweep at the top of this
    // file (block 5 -> 1.97x serial, 7 -> 1.43x), on an unknown chip it is a
    // conservative guess. Only the first is evidence, so only the first
    // outranks a probe.
    if (std.mem.indexOf(u8, chip, "M4") != null) return .{ .cap = NO_WIDE_LANE_BLOCK_CAP, .label = "m4", .measured = true };
    return .{ .cap = NO_WIDE_LANE_BLOCK_CAP, .label = "no wide verify lane" };
}

/// `--drafter-min-depth` / `--drafter-max-depth`: the verifier rows (t1 plus drafts) a drafter round may
/// carry; null = the model's default.
pub var drafter_min_depth: ?u32 = null;
pub var drafter_max_depth: ?u32 = null;
pub const DRAFTER_DEPTH_MAX: u32 = 16;
/// Fewest rows a Flash-Next round verifies: a round with no draft is not a DFlash round.
pub const QWEN_MIN_ROWS: u32 = 2;
/// The widest verify whose rows are decode-tick identical on qwen4 (`mtp.MAX_DEPTH` drafts + t1).
pub const QWEN_MAX_ROWS: u32 = 9;
pub const GLM_MIN_ROWS: u32 = 2;
/// The widest tree a GLM round verifies (`glm5_attention_decode_batch.max_tail`).
pub const GLM_MAX_ROWS: u32 = 8;

pub fn qwenMinRows() u32 {
    return drafter_min_depth orelse QWEN_MIN_ROWS;
}

pub fn drafterDepthExplicit() bool {
    return drafter_min_depth != null or drafter_max_depth != null;
}

pub fn parseDrafterDepth(text: []const u8) error{InvalidDrafterDepth}!u32 {
    const n = std.fmt.parseInt(u32, text, 10) catch return error.InvalidDrafterDepth;
    if (n < QWEN_MIN_ROWS or n > DRAFTER_DEPTH_MAX) return error.InvalidDrafterDepth;
    return n;
}

pub const RowBounds = struct { min: u32, max: u32 };

/// Qwen3.8-Flash-Next: the rows a round may verify, from the flags or the defaults; the drafter's own
/// block and the verify path bound them.
pub fn qwenRowBounds(min: ?u32, max: ?u32, config_block: u32) error{DrafterDepthUnsupported}!RowBounds {
    const hi = max orelse @min(config_block, QWEN_MAX_ROWS);
    const lo = min orelse @min(QWEN_MIN_ROWS, hi);
    if (hi > @min(config_block, QWEN_MAX_ROWS) or lo > hi or lo < QWEN_MIN_ROWS) return error.DrafterDepthUnsupported;
    return .{ .min = lo, .max = hi };
}

/// GLM-5.3-Flash: the rows a round may verify, from the flags or the defaults; the verify path bounds them.
pub fn glmRowBounds(min: ?u32, max: ?u32) error{DrafterDepthUnsupported}!RowBounds {
    const hi = max orelse GLM_MAX_ROWS;
    const lo = min orelse GLM_MIN_ROWS;
    if (hi > GLM_MAX_ROWS or lo < GLM_MIN_ROWS or lo > hi) return error.DrafterDepthUnsupported;
    return .{ .min = lo, .max = hi };
}

test "dflash: a drafter depth parses an integer in 2..16 and names everything else" {
    try testing.expectEqual(@as(u32, 4), try parseDrafterDepth("4"));
    try testing.expectEqual(@as(u32, 2), try parseDrafterDepth("2"));
    try testing.expectEqual(@as(u32, 16), try parseDrafterDepth("16"));
    for ([_][]const u8{ "", "1", "0", "17", "-3", "four", "4.5", " 4" }) |bad| {
        try testing.expectError(error.InvalidDrafterDepth, parseDrafterDepth(bad));
    }
}

test "dflash: Qwen rows default to 2..block and are bounded by the drafter block and the verify path" {
    try testing.expectEqual(RowBounds{ .min = 2, .max = 8 }, try qwenRowBounds(null, null, 8));
    try testing.expectEqual(RowBounds{ .min = 2, .max = 3 }, try qwenRowBounds(null, 3, 8));
    try testing.expectEqual(RowBounds{ .min = 4, .max = 8 }, try qwenRowBounds(4, null, 8));
    try testing.expectEqual(RowBounds{ .min = 4, .max = 4 }, try qwenRowBounds(4, 4, 8));
    try testing.expectEqual(RowBounds{ .min = 2, .max = 3 }, try qwenRowBounds(null, null, 3));
    try testing.expectError(error.DrafterDepthUnsupported, qwenRowBounds(null, 10, 16));
    try testing.expectError(error.DrafterDepthUnsupported, qwenRowBounds(null, 6, 5));
    try testing.expectError(error.DrafterDepthUnsupported, qwenRowBounds(5, 3, 8));
    try testing.expectError(error.DrafterDepthUnsupported, qwenRowBounds(9, null, 8));
}

test "dflash: GLM rows default to 2..8 and the flags move either end inside them" {
    try testing.expectEqual(RowBounds{ .min = 2, .max = 8 }, try glmRowBounds(null, null));
    try testing.expectEqual(RowBounds{ .min = 2, .max = 4 }, try glmRowBounds(null, 4));
    try testing.expectEqual(RowBounds{ .min = 3, .max = 8 }, try glmRowBounds(3, null));
    try testing.expectEqual(RowBounds{ .min = 5, .max = 5 }, try glmRowBounds(5, 5));
    try testing.expectError(error.DrafterDepthUnsupported, glmRowBounds(null, 9));
    try testing.expectError(error.DrafterDepthUnsupported, glmRowBounds(4, 3));
    try testing.expectError(error.DrafterDepthUnsupported, glmRowBounds(1, null));
}

/// Resolve the effective block size: the config's value, capped by what the
/// machine's verify lanes serve (`no_lane_cap` from `blockCapForMachine`
/// when there is no wide lane). Floor 2 (1 draft + t1).
pub fn resolveBlockSize(config_block: u32, wide_verify_lane: bool, no_lane_cap: u32) u32 {
    const base = if (wide_verify_lane) config_block else @min(config_block, no_lane_cap);
    return @max(base, 2);
}

/// True when this machine has a verify lane for widths past the split-K
/// ceiling (the NAX m16 tile, M5-class + macOS >= 26.2).
pub fn wideVerifyLaneAvailable() bool {
    return transformer_mod.verifyQmmNaxAvailable();
}

// ── Weight precision ──

/// Default affine width the dense bf16 checkpoint is quantized to at load.
/// The assistant weight read is the per-round cost the block forward is
/// bound by (5.11 GB bf16 on the Muse sidecar); 8-bit halves it. Only DRAFT
/// quality rides on it — the trunk verify is exact either way, so the worst
/// a bad width can do is cost acceptance.
pub const DEFAULT_QUANT_BITS: u32 = 8;

/// Preferred affine group size, narrowed to 32 when 64 does not divide the
/// contraction dim, dense when neither does.
pub const QUANT_GROUP: u32 = 64;

/// `SUSHI_DFLASH_QUANT_BITS`: absent → `DEFAULT_QUANT_BITS`, a supported
/// affine width → that, anything else ("0", "off") → dense bf16.
pub fn quantBitsFromEnv() u32 {
    const p = std.c.getenv("SUSHI_DFLASH_QUANT_BITS") orelse return DEFAULT_QUANT_BITS;
    const v = std.fmt.parseInt(u32, std.mem.span(p), 10) catch return 0;
    return switch (v) {
        2, 3, 4, 5, 6, 8 => v,
        else => 0,
    };
}

/// Widest supported group that divides the contraction dim, or null when the
/// weight cannot be affine-quantized at all (that weight stays dense).
pub fn quantGroupFor(in_features: u32) ?u32 {
    if (in_features == 0) return null;
    if (in_features % QUANT_GROUP == 0) return QUANT_GROUP;
    if (in_features % 32 == 0) return 32;
    return null;
}

/// Bytes an assistant holds, and what its load holds above them while it runs.
pub const AssistantBytes = struct {
    /// Resident once loaded: the linears at `bits`, every other tensor as stored.
    resident: u64,
    /// The stored-precision copies of the quantized linears, which the weight map keeps until the load ends.
    transient: u64,
};

/// The linears `loadLinear` quantizes (and, for the codebooks, the tensors it leaves dense).
fn isAssistantLinear(key: []const u8) bool {
    const suffixes = [_][]const u8{ ".q_proj.weight", ".k_proj.weight", ".v_proj.weight", ".o_proj.weight", ".gate_proj.weight", ".up_proj.weight", ".down_proj.weight", ".kernel_projection.weight", ".hidden_projection.weight", ".markov_w2.weight" };
    if (std.mem.eql(u8, key, "fc.weight") or std.mem.eql(u8, key, "encoder.fc.weight")) return true;
    for (suffixes) |suffix| if (std.mem.endsWith(u8, key, suffix)) return true;
    return false;
}

fn allocationBytes(n: u64) u64 {
    return (n + 16383) / 16384 * 16384;
}

/// What loading the assistant in `dir` at `bits` (0 = dense) holds, from the safetensors headers alone.
/// Each array is billed at its allocation granule so the bill is never under what the allocator holds.
pub fn assistantLoadBytes(io: std.Io, allocator: std.mem.Allocator, dir_path: []const u8, bits: u32) !AssistantBytes {
    const discovery = @import("model_discovery.zig");
    var dir = try std.Io.Dir.openDirAbsolute(io, dir_path, .{ .iterate = true });
    defer dir.close(io);
    var referenced = discovery.indexShardSet(io, dir);
    defer if (referenced) |*r| discovery.freeShardSet(r);
    var out = AssistantBytes{ .resident = 0, .transient = 0 };
    var files = dir.iterate();
    while (try files.next(io)) |entry| {
        if ((entry.kind != .file and entry.kind != .sym_link) or !std.mem.endsWith(u8, entry.name, ".safetensors")) continue;
        if (referenced) |r| if (!r.contains(entry.name)) continue;
        const file = try dir.openFile(io, entry.name, .{});
        defer file.close(io);
        const stat = try file.stat(io);
        var buffer: [8192]u8 = undefined;
        var reader = file.reader(io, &buffer);
        const header_len = try reader.interface.takeInt(u64, .little);
        if (header_len == 0 or header_len > 128 * 1024 * 1024 or header_len > stat.size -| 8) return error.InvalidSafetensorsHeader;
        const header = try allocator.alloc(u8, @intCast(header_len));
        defer allocator.free(header);
        try reader.interface.readSliceAll(header);
        const parsed = std.json.parseFromSlice(std.json.Value, allocator, header, .{}) catch return error.InvalidSafetensorsHeader;
        defer parsed.deinit();
        if (parsed.value != .object) return error.InvalidSafetensorsHeader;
        var it = parsed.value.object.iterator();
        while (it.next()) |tensor| {
            const key = tensor.key_ptr.*;
            if (std.mem.eql(u8, key, "__metadata__")) continue;
            if (tensor.value_ptr.* != .object) return error.InvalidSafetensorsHeader;
            const offsets = tensor.value_ptr.object.get("data_offsets") orelse return error.InvalidSafetensorsHeader;
            if (offsets != .array or offsets.array.items.len != 2 or offsets.array.items[0] != .integer or offsets.array.items[1] != .integer) return error.InvalidSafetensorsHeader;
            const start = offsets.array.items[0].integer;
            const end = offsets.array.items[1].integer;
            if (start < 0 or end < start or @as(u64, @intCast(end)) > stat.size -| header_len -| 8) return error.InvalidSafetensorsHeader;
            const stored: u64 = @intCast(end - start);
            const shape = tensor.value_ptr.object.get("shape");
            // A weight that ships packed (`.scales` beside it) is served as stored.
            var packed_buf: [512]u8 = undefined;
            const shipped_packed = key.len > ".weight".len and std.mem.endsWith(u8, key, ".weight") and
                parsed.value.object.contains(std.fmt.bufPrint(&packed_buf, "{s}.scales", .{key[0 .. key.len - ".weight".len]}) catch "");
            if (bits != 0 and !shipped_packed and isAssistantLinear(key) and shape != null and shape.? == .array and shape.?.array.items.len == 2 and
                shape.?.array.items[0] == .integer and shape.?.array.items[1] == .integer)
            {
                const rows: u64 = @intCast(shape.?.array.items[0].integer);
                const cols: u64 = @intCast(shape.?.array.items[1].integer);
                if (quantGroupFor(@intCast(cols))) |group| {
                    out.resident += allocationBytes(rows * cols * bits / 8) + 2 * allocationBytes(rows * (cols / group) * 2);
                    out.transient += stored;
                    continue;
                }
            }
            out.resident += allocationBytes(stored);
        }
    }
    if (out.resident == 0) return error.MissingAssistantWeights;
    return out;
}

/// One assistant linear. Dense weights are pre-transposed to `[in, out]` and
/// contracted with a plain matmul; quantized ones
/// keep the checkpoint's `[out, in]` packing and ride `mlx_quantized_matmul`
/// with transpose=true, exactly like the trunk. `bits == 0` IS the dense
/// discriminator — never probe the scales handle.
pub const DflashLinear = struct {
    w: mlx.mlx_array,
    scales: mlx.mlx_array,
    biases: mlx.mlx_array,
    bits: u32 = 0,
    group_size: u32 = 0,

    pub fn isQuantized(self: *const DflashLinear) bool {
        return self.bits != 0;
    }

    pub fn deinit(self: *DflashLinear) void {
        _ = mlx.mlx_array_free(self.w);
        _ = mlx.mlx_array_free(self.scales);
        _ = mlx.mlx_array_free(self.biases);
    }

    pub fn apply(self: *const DflashLinear, x: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
        var out = mlx.mlx_array_new();
        if (!self.isQuantized()) {
            try mlx.check(mlx.mlx_matmul(&out, x, self.w, s));
            return out;
        }
        if (self.bits == 4 and self.group_size == 64) {
            if (try @import("dflash_qmv.zig").matmul(s, x, self.w, self.scales, self.biases)) |fused| {
                _ = mlx.mlx_array_free(out);
                return fused;
            }
        }
        try mlx.check(mlx.mlx_quantized_matmul(
            &out,
            x,
            self.w,
            self.scales,
            self.biases,
            true,
            mlx.mlx_optional_int.some(@intCast(self.group_size)),
            mlx.mlx_optional_int.some(@intCast(self.bits)),
            "affine",
            s,
        ));
        return out;
    }

    fn appendEval(self: *const DflashLinear, vec: mlx.mlx_vector_array) void {
        _ = mlx.mlx_vector_array_append_value(vec, self.w);
        if (!self.isQuantized()) return;
        _ = mlx.mlx_vector_array_append_value(vec, self.scales);
        _ = mlx.mlx_vector_array_append_value(vec, self.biases);
    }
};

// ── Model ──

/// DFlash2 grouped dynamic causal conv (one per sublayer): two-tap depthwise
/// conv whose kernels are `base + dynamic`, the dynamic part projected
/// per-position from the sublayer's normed input. `base_kernel` axes are
/// `[prepare|finish, tap, channel]` — BOTH leading dims are 2 at ksize 2, so
/// the order is pinned by the parity fixture, never by shape.
pub const DynConv = struct {
    base_kernel: mlx.mlx_array, // [2, ksize, hidden] bf16
    kernel_projection: DflashLinear, // [2*ksize*groups ← hidden]

    fn deinit(self: *DynConv) void {
        _ = mlx.mlx_array_free(self.base_kernel);
        self.kernel_projection.deinit();
    }

    fn appendEval(self: *const DynConv, vec: mlx.mlx_vector_array) void {
        _ = mlx.mlx_vector_array_append_value(vec, self.base_kernel);
        self.kernel_projection.appendEval(vec);
    }
};

/// DFlash2 path selector: per-position top-k candidates scored pairwise
/// through two per-token codebooks + a hidden→rank projection. Codebooks are
/// GATHER-read tables — bf16, never quantized (the NEVER_QUANTIZE class).
pub const Selector = struct {
    pred_codebook: mlx.mlx_array, // [vocab, rank] bf16
    succ_codebook: mlx.mlx_array, // [vocab, rank] bf16
    hidden_projection: DflashLinear, // [rank ← hidden]

    fn deinit(self: *Selector) void {
        _ = mlx.mlx_array_free(self.pred_codebook);
        _ = mlx.mlx_array_free(self.succ_codebook);
        self.hidden_projection.deinit();
    }

    fn appendEval(self: *const Selector, vec: mlx.mlx_vector_array) void {
        _ = mlx.mlx_vector_array_append_value(vec, self.pred_codebook);
        _ = mlx.mlx_vector_array_append_value(vec, self.succ_codebook);
        self.hidden_projection.appendEval(vec);
    }
};

/// Per-layer assistant weights.
/// DSpark's vanilla Markov head: a rank-`markov_rank` bigram bias added to
/// each block position's logits from the token drafted at the PREVIOUS
/// position. It is what makes the block semi-autoregressive off ONE assistant
/// forward — the base logits are position-parallel, the bias chains them.
///
/// `markov_w1` is a gather table (`[vocab, rank]`, one row per previous
/// token) and stays DENSE: a packed table read by `mlx_take_axis` gathers
/// uint32 words, not rows. `markov_w2` is an ordinary `rank → vocab` linear
/// and rides the same load-time quantization as the rest of the sidecar —
/// only the DRAFT sees it, so a lossier bias costs acceptance, never a token.
pub const MarkovHead = struct {
    w1: mlx.mlx_array,
    w2: DflashLinear,

    pub fn deinit(self: *MarkovHead) void {
        _ = mlx.mlx_array_free(self.w1);
        self.w2.deinit();
    }

    pub fn appendEval(self: *const MarkovHead, vec: mlx.mlx_vector_array) void {
        _ = mlx.mlx_vector_array_append_value(vec, self.w1);
        self.w2.appendEval(vec);
    }

    /// `logits + w2(w1[prev_token])` for ONE block position. `base_row` is
    /// `[1, 1, vocab]`; the result is a fresh array the caller owns.
    pub fn stepLogits(
        self: *const MarkovHead,
        base_row: mlx.mlx_array,
        prev_token: u32,
        s: mlx.mlx_stream,
    ) !mlx.mlx_array {
        const idx_i32: i32 = @intCast(prev_token);
        const idx_shape = [_]c_int{1};
        const idx = mlx.mlx_array_new_data(&idx_i32, &idx_shape, 1, .int32);
        defer _ = mlx.mlx_array_free(idx);

        var row = mlx.mlx_array_new(); // [1, rank]
        defer _ = mlx.mlx_array_free(row);
        try mlx.check(mlx.mlx_take_axis(&row, self.w1, idx, 0, s));

        const rsh = mlx.getShape(row);
        const shaped = [_]c_int{ 1, 1, rsh[1] };
        var row3 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(row3);
        try mlx.check(mlx.mlx_reshape(&row3, row, &shaped, 3, s));

        const bias = try self.w2.apply(row3, s);
        defer _ = mlx.mlx_array_free(bias);

        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_add(&out, base_row, bias, s));
        return out;
    }
};

pub const DflashLayer = struct {
    layer_type: LayerType,

    input_norm: mlx.mlx_array,
    post_attn_norm: mlx.mlx_array,

    q: DflashLinear, // [hidden → n_heads*head_dim]
    q_norm: mlx.mlx_array, // [head_dim]
    k: DflashLinear, // [hidden → kv_heads*head_dim]
    k_norm: mlx.mlx_array, // [head_dim]
    v: DflashLinear, // [hidden → kv_heads*head_dim]
    o: DflashLinear, // [n_heads*head_dim → hidden]

    gate: DflashLinear, // [hidden → intermediate]
    up: DflashLinear, // [hidden → intermediate]
    down: DflashLinear, // [intermediate → hidden]

    // DFlash2 only — null on a v1 assistant.
    attention_conv: ?DynConv = null,
    mlp_conv: ?DynConv = null,

    fn deinit(self: *DflashLayer) void {
        if (self.attention_conv) |*c| c.deinit();
        if (self.mlp_conv) |*c| c.deinit();
        _ = mlx.mlx_array_free(self.input_norm);
        _ = mlx.mlx_array_free(self.post_attn_norm);
        _ = mlx.mlx_array_free(self.q_norm);
        _ = mlx.mlx_array_free(self.k_norm);
        self.q.deinit();
        self.k.deinit();
        self.v.deinit();
        self.o.deinit();
        self.gate.deinit();
        self.up.deinit();
        self.down.deinit();
    }

    fn appendEval(self: *const DflashLayer, vec: mlx.mlx_vector_array) void {
        _ = mlx.mlx_vector_array_append_value(vec, self.input_norm);
        _ = mlx.mlx_vector_array_append_value(vec, self.post_attn_norm);
        _ = mlx.mlx_vector_array_append_value(vec, self.q_norm);
        _ = mlx.mlx_vector_array_append_value(vec, self.k_norm);
        for ([_]*const DflashLinear{ &self.q, &self.k, &self.v, &self.o, &self.gate, &self.up, &self.down }) |lin| {
            lin.appendEval(vec);
        }
        if (self.attention_conv) |*c| c.appendEval(vec);
        if (self.mlp_conv) |*c| c.appendEval(vec);
    }
};

pub const DflashModel = struct {
    /// All-sliding DFlash2 assistant (GLM, Qwen3.8-Flash-Next) served with a physical window trim,
    /// a bounded block tail and a cached mask.
    sliding_serving: bool = false,
    config: DflashConfig,
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,

    fc: DflashLinear, // encoder.fc [n_targets*hidden → hidden]
    enc_norm: mlx.mlx_array, // encoder.output_norm_enc [hidden]
    final_norm: mlx.mlx_array, // norm [hidden]
    layers: []DflashLayer,

    /// DFlash2 path selector — null on a v1 assistant.
    selector: ?Selector = null,

    /// DSpark Markov head — null unless the config declares `markov_rank`.
    markov: ?MarkovHead = null,

    pub fn deinit(self: *DflashModel) void {
        const allocator = self.allocator;
        if (self.selector) |*sel| sel.deinit();
        if (self.markov) |*mh| mh.deinit();
        self.fc.deinit();
        _ = mlx.mlx_array_free(self.enc_norm);
        _ = mlx.mlx_array_free(self.final_norm);
        for (self.layers) |*lw| lw.deinit();
        allocator.free(self.layers);
        self.config.deinit(allocator);
    }

    /// Validate compatibility with the target trunk. The assistant borrows
    /// the trunk's embedding table and lm_head, so hidden size and token-id
    /// space must line up; the capture seam lives in the STANDARD forward
    /// path only (v1), so module-owned / MoE / hybrid trunks are refused by
    /// name at load rather than silently drafting from empty captures.
    pub fn bind(self: *DflashModel, target: *Transformer) !void {
        if (self.config.hidden_size != target.config.hidden_size) {
            log.err("[dflash] hidden_size mismatch: assistant={d}, target={d}\n", .{
                self.config.hidden_size, target.config.hidden_size,
            });
            return error.DflashTargetMismatch;
        }
        if (self.config.mask_token_id >= target.config.vocab_size) {
            log.err("[dflash] mask_token_id {d} outside target vocab {d}\n", .{
                self.config.mask_token_id, target.config.vocab_size,
            });
            return error.DflashTargetMismatch;
        }
        validateTargetLayers(self.config.target_layer_ids, target.config.num_hidden_layers) catch |err| {
            log.err("[dflash] target_layer_ids {any} out of range for {d}-layer target\n", .{
                self.config.target_layer_ids, target.config.num_hidden_layers,
            });
            return err;
        };
        if (self.selector) |*sel| {
            // Candidate ids come from the TRUNK head's logits and gather
            // codebook rows — a trunk vocab wider than the tables reads OOB.
            const rows: u32 = @intCast(mlx.getShape(sel.pred_codebook)[0]);
            if (target.config.vocab_size > rows) {
                log.err("[dflash] selector codebook rows {d} < target vocab {d}\n", .{
                    rows, target.config.vocab_size,
                });
                return error.DflashTargetMismatch;
            }
        }
    }

    /// Draft logits for the block hidden: the trunk's own head, the bare
    /// Linear; when the sidecar DECLARES a softcap / output_multiplier (muse DFlash2) they are
    /// applied here — argmax drafts are invariant (monotone), but the
    /// selector sums these logits with codebook edges and the sampled arm
    /// softmaxes them, neither of which survives a scale change.
    pub fn draftLogits(self: *const DflashModel, target: *const Transformer, x: mlx.mlx_array) !mlx.mlx_array {
        const out = try target.lmHeadForDraft(x);
        if (self.config.output_multiplier == 1.0 and self.config.logit_softcap <= 0) return out;
        defer _ = mlx.mlx_array_free(out);
        return applyLogitTransforms(out, self.config.output_multiplier, self.config.logit_softcap, self.s);
    }
};

/// Reference `compute_logits`: `l * multiplier`, then `tanh(l/cap) * cap`.
/// Scalars are cast to the logits' own dtype — a f32 scalar would silently
/// promote a bf16 stream (the activation-dtype rule).
pub fn applyLogitTransforms(logits: mlx.mlx_array, multiplier: f32, softcap: f32, s: mlx.mlx_stream) !mlx.mlx_array {
    const dtype = mlx.mlx_array_dtype(logits);
    var cur = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(cur);
    try mlx.check(mlx.mlx_array_set(&cur, logits));
    if (multiplier != 1.0) {
        const m = try scalarAs(multiplier, dtype, s);
        defer _ = mlx.mlx_array_free(m);
        var scaled = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&scaled, cur, m, s));
        _ = mlx.mlx_array_free(cur);
        cur = scaled;
    }
    if (softcap > 0) {
        const inv = try scalarAs(1.0 / softcap, dtype, s);
        defer _ = mlx.mlx_array_free(inv);
        const cap = try scalarAs(softcap, dtype, s);
        defer _ = mlx.mlx_array_free(cap);
        var normed = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(normed);
        try mlx.check(mlx.mlx_multiply(&normed, cur, inv, s));
        var squashed = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(squashed);
        try mlx.check(mlx.mlx_tanh(&squashed, normed, s));
        var capped = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&capped, squashed, cap, s));
        _ = mlx.mlx_array_free(cur);
        cur = capped;
    }
    return cur;
}

/// 0-dim scalar in the given dtype.
fn scalarAs(v: f32, dtype: mlx.mlx_dtype, s: mlx.mlx_stream) !mlx.mlx_array {
    const f = mlx.mlx_array_new_float(v);
    defer _ = mlx.mlx_array_free(f);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, f, dtype, s));
    return out;
}

// ── Weight loading ──

fn ownWeight(w: *const Weights, key: []const u8) !mlx.mlx_array {
    const arr = w.get(key) orelse {
        log.err("[dflash] MISSING WEIGHT: {s}\n", .{key});
        return error.MissingDflashWeight;
    };
    var owned = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&owned, arr));
    return owned;
}

/// `key`, falling back to `alt` — the DFlash2 codebooks ship WITHOUT a
/// `.weight` suffix (the reference loader renames them before load_weights);
/// a re-export through transformers would put the suffix back.
fn ownWeightEither(w: *const Weights, key: []const u8, alt: []const u8) !mlx.mlx_array {
    if (w.get(key) != null) return ownWeight(w, key);
    return ownWeight(w, alt);
}

/// Load `<prefix>.weight` as an assistant linear. A checkpoint that already
/// ships `<prefix>.scales` is served packed as-is (affine only — its true
/// params are solved from the packed geometry, never assumed); a dense bf16
/// weight is quantized to `bits` when the contraction dim allows it and
/// pre-transposed for a plain matmul otherwise.
fn loadLinear(
    w: *const Weights,
    prefix: []const u8,
    in_features: u32,
    bits: u32,
    s: mlx.mlx_stream,
) !DflashLinear {
    var key_buf: [256]u8 = undefined;
    const scales_key = try std.fmt.bufPrint(&key_buf, "{s}.scales", .{prefix});
    if (w.get(scales_key)) |_| {
        var kb: [256]u8 = undefined;
        var out = DflashLinear{
            .w = try ownWeight(w, try std.fmt.bufPrint(&kb, "{s}.weight", .{prefix})),
            .scales = try ownWeight(w, try std.fmt.bufPrint(&kb, "{s}.scales", .{prefix})),
            .biases = try ownWeight(w, try std.fmt.bufPrint(&kb, "{s}.biases", .{prefix})),
        };
        errdefer out.deinit();
        const qp = transformer_mod.affineParamsFromGeometry(out.w, out.scales, in_features) orelse {
            log.err("[dflash] {s}: packed geometry is not affine-servable (in={d})\n", .{ prefix, in_features });
            return error.UnsupportedDflashQuant;
        };
        out.bits = qp.bits;
        out.group_size = qp.group_size;
        return out;
    }

    var kb: [256]u8 = undefined;
    const raw = try ownWeight(w, try std.fmt.bufPrint(&kb, "{s}.weight", .{prefix}));
    defer _ = mlx.mlx_array_free(raw);

    if (bits != 0) {
        if (quantGroupFor(in_features)) |group| return quantizeDense(raw, bits, group, s);
    }
    var transposed = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(transposed);
    const perm = [_]c_int{ 1, 0 };
    try mlx.check(mlx.mlx_transpose_axes(&transposed, raw, &perm, 2, s));
    return .{ .w = transposed, .scales = mlx.mlx_array_new(), .biases = mlx.mlx_array_new() };
}

/// Affine-quantize a dense `[out, in]` weight in place of a transpose — the
/// packed layout is exactly what `mlx_quantized_matmul(transpose=true)` reads.
pub fn quantizeDense(raw: mlx.mlx_array, bits: u32, group: u32, s: mlx.mlx_stream) !DflashLinear {
    var triple = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple);
    try mlx.check(mlx.mlx_quantize(
        &triple,
        raw,
        mlx.mlx_optional_int.some(@intCast(group)),
        mlx.mlx_optional_int.some(@intCast(bits)),
        "affine",
        .{}, // global_scale
        s,
    ));
    if (mlx.mlx_vector_array_size(triple) != 3) return error.UnexpectedQuantizeOutput;
    var out = DflashLinear{
        .w = mlx.mlx_array_new(),
        .scales = mlx.mlx_array_new(),
        .biases = mlx.mlx_array_new(),
        .bits = bits,
        .group_size = group,
    };
    errdefer out.deinit();
    try mlx.check(mlx.mlx_vector_array_get(&out.w, triple, 0));
    try mlx.check(mlx.mlx_vector_array_get(&out.scales, triple, 1));
    try mlx.check(mlx.mlx_vector_array_get(&out.biases, triple, 2));
    return out;
}

/// Load a DFlash assistant from `model_dir`. After loading, call
/// `bind(target)` before serving.
pub fn loadDflash(
    io: std.Io,
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,
    model_dir: []const u8,
) !DflashModel {
    return loadDflashQuant(io, allocator, s, model_dir, quantBitsFromEnv());
}

/// `loadDflash` with the load-time quantization width passed explicitly
/// (0 = dense bf16). Tests drive both arms from here rather than through the
/// environment — a skipped arm reads as a pass.
pub fn loadDflashQuant(
    io: std.Io,
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,
    model_dir: []const u8,
    bits: u32,
) !DflashModel {
    var cfg = try parseConfig(io, allocator, model_dir);
    errdefer cfg.deinit(allocator);

    var weights = try model_mod.loadWeights(io, allocator, model_dir);
    defer weights.deinit();

    const hidden = cfg.hidden_size;
    const q_out = cfg.num_attention_heads * cfg.head_dim;
    const fc_in: u32 = @intCast(cfg.target_layer_ids.len * hidden);

    // Two encoder spellings in the wild: transformers assistants (muse) ship
    // `encoder.fc` + `encoder.output_norm_enc`, z-lab DFlash2 ships root
    // `fc` + `hidden_norm`. Same tensors, keyed on which one the file has.
    const zlab_names = weights.get("fc.weight") != null;
    var fc = try loadLinear(&weights, if (zlab_names) "fc" else "encoder.fc", fc_in, bits, s);
    errdefer fc.deinit();
    const enc_norm = try ownWeight(&weights, if (zlab_names) "hidden_norm.weight" else "encoder.output_norm_enc.weight");
    errdefer _ = mlx.mlx_array_free(enc_norm);
    const final_norm = try ownWeight(&weights, "norm.weight");
    errdefer _ = mlx.mlx_array_free(final_norm);

    const layers = try allocator.alloc(DflashLayer, cfg.num_hidden_layers);
    errdefer allocator.free(layers);
    var layers_inited: u32 = 0;
    errdefer {
        var i: u32 = 0;
        while (i < layers_inited) : (i += 1) layers[i].deinit();
    }

    var key_buf: [256]u8 = undefined;
    var li: u32 = 0;
    while (li < cfg.num_hidden_layers) : (li += 1) {
        layers[li] = .{
            .layer_type = cfg.layer_types[li],
            .input_norm = try ownWeight(&weights, try std.fmt.bufPrint(&key_buf, "layers.{d}.input_layernorm.weight", .{li})),
            .post_attn_norm = try ownWeight(&weights, try std.fmt.bufPrint(&key_buf, "layers.{d}.post_attention_layernorm.weight", .{li})),
            .q = try loadLinear(&weights, try std.fmt.bufPrint(&key_buf, "layers.{d}.self_attn.q_proj", .{li}), hidden, bits, s),
            .q_norm = try ownWeight(&weights, try std.fmt.bufPrint(&key_buf, "layers.{d}.self_attn.q_norm.weight", .{li})),
            .k = try loadLinear(&weights, try std.fmt.bufPrint(&key_buf, "layers.{d}.self_attn.k_proj", .{li}), hidden, bits, s),
            .k_norm = try ownWeight(&weights, try std.fmt.bufPrint(&key_buf, "layers.{d}.self_attn.k_norm.weight", .{li})),
            .v = try loadLinear(&weights, try std.fmt.bufPrint(&key_buf, "layers.{d}.self_attn.v_proj", .{li}), hidden, bits, s),
            .o = try loadLinear(&weights, try std.fmt.bufPrint(&key_buf, "layers.{d}.self_attn.o_proj", .{li}), q_out, bits, s),
            .gate = try loadLinear(&weights, try std.fmt.bufPrint(&key_buf, "layers.{d}.mlp.gate_proj", .{li}), hidden, bits, s),
            .up = try loadLinear(&weights, try std.fmt.bufPrint(&key_buf, "layers.{d}.mlp.up_proj", .{li}), hidden, bits, s),
            .down = try loadLinear(&weights, try std.fmt.bufPrint(&key_buf, "layers.{d}.mlp.down_proj", .{li}), cfg.intermediate_size, bits, s),
        };
        layers_inited += 1;
        if (cfg.conv_kernel_size > 0) {
            layers[li].attention_conv = try loadDynConv(&weights, li, "attention_conv", hidden, bits, s);
            layers[li].mlp_conv = try loadDynConv(&weights, li, "mlp_conv", hidden, bits, s);
        }
    }

    var selector: ?Selector = null;
    errdefer if (selector) |*sel| sel.deinit();
    if (cfg.selector_rank > 0) {
        const pred = try ownWeightEither(&weights, "candidate_selector.predecessor_codebook", "candidate_selector.predecessor_codebook.weight");
        errdefer _ = mlx.mlx_array_free(pred);
        const succ = try ownWeightEither(&weights, "candidate_selector.successor_codebook", "candidate_selector.successor_codebook.weight");
        errdefer _ = mlx.mlx_array_free(succ);
        const hp = try loadLinear(&weights, "candidate_selector.hidden_projection", hidden, bits, s);
        selector = .{ .pred_codebook = pred, .succ_codebook = succ, .hidden_projection = hp };
    }

    var markov: ?MarkovHead = null;
    errdefer if (markov) |*mh| mh.deinit();
    if (cfg.markov_rank > 0) {
        const w1 = try ownWeight(&weights, "markov_head.markov_w1.weight");
        errdefer _ = mlx.mlx_array_free(w1);
        const w2 = try loadLinear(&weights, "markov_head.markov_w2", cfg.markov_rank, bits, s);
        markov = .{ .w1 = w1, .w2 = w2 };
        // The confidence head trims the drafted block per request in the
        // reference's ragged-verify mode; we serve a STATIC block, so its
        // weights are deliberately unread. Say so rather than let a silently
        // ignored trained module read as a port that covers it.
        if (weights.get("confidence_head.proj.weight") != null) {
            log.info("[dflash] dspark: confidence head present but unused (static verify width)\n", .{});
        }
    }

    // Force-eval all weights so serve time never faults a lazy transpose or
    // an un-materialized load-time quantization.
    {
        const eval_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(eval_vec);
        fc.appendEval(eval_vec);
        _ = mlx.mlx_vector_array_append_value(eval_vec, enc_norm);
        _ = mlx.mlx_vector_array_append_value(eval_vec, final_norm);
        for (layers) |*lw| lw.appendEval(eval_vec);
        if (selector) |*sel| sel.appendEval(eval_vec);
        if (markov) |*mh| mh.appendEval(eval_vec);
        _ = mlx.mlx_eval(eval_vec);
    }

    if (fc.isQuantized()) {
        log.info("[dflash] loaded {d} layers, hidden={d}, block_size={d}, targets={any}, weights={d}-bit/gs{d}\n", .{
            cfg.num_hidden_layers, cfg.hidden_size, cfg.block_size, cfg.target_layer_ids, fc.bits, fc.group_size,
        });
    } else {
        log.info("[dflash] loaded {d} layers, hidden={d}, block_size={d}, targets={any}, weights=dense bf16\n", .{
            cfg.num_hidden_layers, cfg.hidden_size, cfg.block_size, cfg.target_layer_ids,
        });
    }

    if (cfg.isDflash2()) {
        log.info("[dflash] dflash2: selector rank={d} top_k={d}, dyn-convs ksize={d} group={d}\n", .{
            cfg.selector_rank, cfg.selector_top_k, cfg.conv_kernel_size, cfg.conv_group_size,
        });
    }
    if (cfg.isDspark()) {
        log.info("[dflash] dspark: markov head rank={d}, rope theta={d:.0} ({s})\n", .{
            cfg.markov_rank, cfg.rope_theta, if (cfg.rope_traditional) "interleaved" else "neox",
        });
    }

    return DflashModel{
        .config = cfg,
        .allocator = allocator,
        .s = s,
        .fc = fc,
        .enc_norm = enc_norm,
        .final_norm = final_norm,
        .layers = layers,
        .selector = selector,
        .markov = markov,
    };
}

/// Load one DFlash2 dynamic conv pair (`base_kernel` + `kernel_projection`)
/// for layer `li`. Both weights are REQUIRED once the config declares
/// `conv_kernel_size` — a DFlash2 pack missing them is a broken download.
fn loadDynConv(
    w: *const Weights,
    li: u32,
    comptime which: []const u8,
    hidden: u32,
    bits: u32,
    s: mlx.mlx_stream,
) !DynConv {
    var key_buf: [256]u8 = undefined;
    const base = try ownWeight(w, try std.fmt.bufPrint(&key_buf, "layers.{d}." ++ which ++ ".base_kernel", .{li}));
    errdefer _ = mlx.mlx_array_free(base);
    const proj = try loadLinear(w, try std.fmt.bufPrint(&key_buf, "layers.{d}." ++ which ++ ".kernel_projection", .{li}), hidden, bits, s);
    return .{ .base_kernel = base, .kernel_projection = proj };
}

// ── Per-request context cache ──

/// The assistant's context K/V, one entry per ASSISTANT layer, dense.
/// Grows with committed trunk tokens; block K/V transit through spare
/// capacity and are truncated straight back out (never cached). Invariant
/// at round start: `base_pos + cache.step == trunk cache.step`.
pub const DflashCtx = struct {
    cache: KVCache,
    /// Absolute trunk position of cache index 0. Nonzero on hot-prefix-cache
    /// reuse, where only the freshly forwarded tail produced captures — a
    /// late-starting context is self-consistent (sliding-window semantics,
    /// same rule as the MTP history).
    base_pos: usize,

    pub fn init(allocator: std.mem.Allocator, model: *const DflashModel, base_pos: usize) !DflashCtx {
        return .{
            .cache = try KVCache.init(allocator, model.config.num_hidden_layers),
            .base_pos = base_pos,
        };
    }

    pub fn deinit(self: *DflashCtx) void {
        self.cache.deinit();
    }

    /// Absolute trunk position one past the last cached context token.
    pub fn absLen(self: *const DflashCtx) usize {
        return self.base_pos + self.cache.step;
    }

    /// Collect cache buffers for a batched eval (prefill-chunk discipline:
    /// materialize appended context so the chunk's activation graph frees).
    pub fn appendEvalArrays(self: *DflashCtx, vec: mlx.mlx_vector_array) void {
        for (self.cache.entries) |*entry| {
            if (!entry.initialized) continue;
            _ = mlx.mlx_vector_array_append_value(vec, entry.keys);
            _ = mlx.mlx_vector_array_append_value(vec, entry.values);
        }
    }
};

// ── Forward helpers ──

inline fn rmsNormFn(x: mlx.mlx_array, w: mlx.mlx_array, eps: f32, s: mlx.mlx_stream) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rms_norm(&out, x, w, eps, s));
    return out;
}

/// silu(gate) * up.
fn swiglu(gate: mlx.mlx_array, up: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sig);
    try mlx.check(mlx.mlx_sigmoid(&sig, gate, s));
    var silu_out = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(silu_out);
    try mlx.check(mlx.mlx_multiply(&silu_out, gate, sig, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&out, silu_out, up, s));
    return out;
}

/// Project x through `w_t`, reshape to heads, apply optional per-head RMS
/// norm, transpose to `[B, heads, L, hd]`, RoPE at `offset`.
fn projectHeads(
    x: mlx.mlx_array,
    lin: *const DflashLinear,
    norm_w: mlx.mlx_array,
    n_heads: u32,
    head_dim: u32,
    eps: f32,
    theta: f32,
    rope_offset: usize,
    apply_rope: bool,
    rope_traditional: bool,
    s: mlx.mlx_stream,
) !mlx.mlx_array {
    const proj = try lin.apply(x, s);
    defer _ = mlx.mlx_array_free(proj);
    const xsh = mlx.getShape(x);
    const hs = [_]c_int{ xsh[0], xsh[1], @intCast(n_heads), @intCast(head_dim) };
    var reshaped = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(reshaped);
    try mlx.check(mlx.mlx_reshape(&reshaped, proj, &hs, 4, s));

    const normed = try rmsNormFn(reshaped, norm_w, eps, s);
    defer _ = mlx.mlx_array_free(normed);

    const perm = [_]c_int{ 0, 2, 1, 3 };
    var transposed = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_transpose_axes(&transposed, normed, &perm, 4, s));
    if (!apply_rope) return transposed;
    defer _ = mlx.mlx_array_free(transposed);

    var roped = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rope(
        &roped,
        transposed,
        @intCast(head_dim),
        rope_traditional,
        .{ .value = theta, .has_value = true },
        1.0,
        @intCast(rope_offset),
        .{ .ctx = null },
        s,
    ));
    return roped;
}

/// Additive attention bias `[1, 1, q_len, kv_len]` for one assistant layer,
/// or null when nothing is masked. KV layout: context at absolute positions
/// `[base_pos, base_pos + ctx_len)` followed by the block at
/// `[anchor_pos, anchor_pos + q_len)`. Sliding layers: query q sees key k iff
/// `|q - k| < window` (block queries sit at/after every context position, so
/// "bidirectional sliding" reduces to a back-window over context plus full
/// visibility inside the block whenever `q_len <= window`). Full layers see
/// everything → null.
fn buildBlockMask(
    layer_type: LayerType,
    base_pos: usize,
    ctx_len: usize,
    anchor_pos: usize,
    q_len: u32,
    window: u32,
    s: mlx.mlx_stream,
) !?mlx.mlx_array {
    if (layer_type == .full_attention) return null;
    std.debug.assert(q_len <= window);
    // No context row falls outside the LAST query's back-window → nothing masked.
    const max_dist = anchor_pos + q_len - 1 - base_pos;
    if (max_dist < window) return null;

    // Absolute key positions: [ctx…, block…].
    const kv_len = ctx_len + q_len;
    var k_abs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_abs);
    {
        var ctx_pos = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ctx_pos);
        try mlx.check(mlx.mlx_arange(&ctx_pos, @floatFromInt(base_pos), @floatFromInt(base_pos + ctx_len), 1, .float32, s));
        var blk_pos = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(blk_pos);
        try mlx.check(mlx.mlx_arange(&blk_pos, @floatFromInt(anchor_pos), @floatFromInt(anchor_pos + q_len), 1, .float32, s));
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        _ = mlx.mlx_vector_array_append_value(vec, ctx_pos);
        _ = mlx.mlx_vector_array_append_value(vec, blk_pos);
        try mlx.check(mlx.mlx_concatenate_axis(&k_abs, vec, 0, s));
    }
    var q_abs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_abs);
    try mlx.check(mlx.mlx_arange(&q_abs, @floatFromInt(anchor_pos), @floatFromInt(anchor_pos + q_len), 1, .float32, s));
    var q_col = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_col);
    const q_shape = [_]c_int{ @intCast(q_len), 1 };
    try mlx.check(mlx.mlx_reshape(&q_col, q_abs, &q_shape, 2, s));

    // diff = q - k; allowed iff |diff| < window.
    var diff = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(diff);
    try mlx.check(mlx.mlx_subtract(&diff, q_col, k_abs, s));
    var abs_diff = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(abs_diff);
    try mlx.check(mlx.mlx_abs(&abs_diff, diff, s));
    const win_arr = mlx.mlx_array_new_float(@floatFromInt(window));
    defer _ = mlx.mlx_array_free(win_arr);
    var allowed = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(allowed);
    try mlx.check(mlx.mlx_less(&allowed, abs_diff, win_arr, s));

    const zero = mlx.mlx_array_new_float(0.0);
    defer _ = mlx.mlx_array_free(zero);
    const ninf = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer _ = mlx.mlx_array_free(ninf);
    var bias_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bias_f32);
    try mlx.check(mlx.mlx_where(&bias_f32, allowed, zero, ninf, s));

    var bias_bf16 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bias_bf16);
    try mlx.check(mlx.mlx_astype(&bias_bf16, bias_f32, .bfloat16, s));

    var bias_4d = mlx.mlx_array_new();
    const out_shape = [_]c_int{ 1, 1, @intCast(q_len), @intCast(kv_len) };
    try mlx.check(mlx.mlx_reshape(&bias_4d, bias_bf16, &out_shape, 4, s));
    return bias_4d;
}

// ── DFlash2 grouped dynamic causal conv (forward) ──

/// The reference's `_grouped_dynamic_convolve`, transcribed verbatim:
/// `out_t = Σ_tap (base[tap] + dyn_t[tap]) ⊙ x_{t-tap}`, with positions
/// before the block start reading ZERO — the block is self-contained, so the
/// anchor's predecessor tap is the reference's zero pad, never the previous
/// block. `base` is per-CHANNEL `[ksize, H]`; `dynamic` is per-GROUP
/// `[1, L, ksize, groups]`, each coefficient broadcasting over `group_size`
/// channels. Two separate multiply-adds per tap keep the reference's bf16
/// rounding order.
pub fn groupedDynConv(hidden: mlx.mlx_array, dynamic: mlx.mlx_array, base: mlx.mlx_array, group_size: u32, s: mlx.mlx_stream) !mlx.mlx_array {
    if (try @import("dflash_conv.zig").apply(s, hidden, dynamic, base, group_size)) |out| return out;
    return groupedDynConvReference(hidden, dynamic, base, group_size, s);
}

pub fn groupedDynConvReference(
    hidden: mlx.mlx_array, // [1, L, H]
    dynamic: mlx.mlx_array, // [1, L, ksize, groups]
    base: mlx.mlx_array, // [ksize, H]
    group_size: u32,
    s: mlx.mlx_stream,
) !mlx.mlx_array {
    const hsh = mlx.getShape(hidden);
    const len = hsh[1];
    const h = hsh[2];
    const groups = @divExact(h, @as(c_int, @intCast(group_size)));
    const ksize: usize = @intCast(mlx.getShape(base)[0]);
    const dtype = mlx.mlx_array_dtype(hidden);

    const blk_shape = [_]c_int{ 1, len, groups, @intCast(group_size) };
    var blocks = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(blocks);
    try mlx.check(mlx.mlx_reshape(&blocks, hidden, &blk_shape, 4, s));

    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_zeros(&out, &blk_shape, 4, dtype, s));

    var tap: usize = 0;
    while (tap < ksize) : (tap += 1) {
        // values = hidden shifted right by `tap`, zero-padded at the front.
        var values = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(values);
        if (tap == 0) {
            try mlx.check(mlx.mlx_array_set(&values, blocks));
        } else {
            const pad_shape = [_]c_int{ 1, @intCast(tap), groups, @intCast(group_size) };
            var pad = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(pad);
            try mlx.check(mlx.mlx_zeros(&pad, &pad_shape, 4, dtype, s));
            var head = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(head);
            const start = [_]c_int{ 0, 0, 0, 0 };
            const stop = [_]c_int{ 1, len - @as(c_int, @intCast(tap)), groups, @intCast(group_size) };
            const strides = [_]c_int{ 1, 1, 1, 1 };
            try mlx.check(mlx.mlx_slice(&head, blocks, &start, 4, &stop, 4, &strides, 4, s));
            const vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(vec);
            _ = mlx.mlx_vector_array_append_value(vec, pad);
            _ = mlx.mlx_vector_array_append_value(vec, head);
            try mlx.check(mlx.mlx_concatenate_axis(&values, vec, 1, s));
        }

        // base[tap] as [1, 1, groups, group_size] (per-channel).
        var kernel = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(kernel);
        {
            var row = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(row);
            const start = [_]c_int{ @intCast(tap), 0 };
            const stop = [_]c_int{ @as(c_int, @intCast(tap)) + 1, h };
            const strides = [_]c_int{ 1, 1 };
            try mlx.check(mlx.mlx_slice(&row, base, &start, 2, &stop, 2, &strides, 2, s));
            const k_shape = [_]c_int{ 1, 1, groups, @intCast(group_size) };
            try mlx.check(mlx.mlx_reshape(&kernel, row, &k_shape, 4, s));
        }
        // dynamic[:, :, tap, :] as [1, L, groups, 1] (per-group).
        var dslice = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dslice);
        {
            var cut = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(cut);
            const start = [_]c_int{ 0, 0, @intCast(tap), 0 };
            const stop = [_]c_int{ 1, len, @as(c_int, @intCast(tap)) + 1, groups };
            const strides = [_]c_int{ 1, 1, 1, 1 };
            try mlx.check(mlx.mlx_slice(&cut, dynamic, &start, 4, &stop, 4, &strides, 4, s));
            const d_shape = [_]c_int{ 1, len, groups, 1 };
            try mlx.check(mlx.mlx_reshape(&dslice, cut, &d_shape, 4, s));
        }

        inline for (.{ kernel, dslice }) |coeff| {
            var term = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(term);
            try mlx.check(mlx.mlx_multiply(&term, coeff, values, s));
            var next = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&next, out, term, s));
            _ = mlx.mlx_array_free(out);
            out = next;
        }
    }

    var flat = mlx.mlx_array_new();
    const out_shape = [_]c_int{ 1, len, h };
    try mlx.check(mlx.mlx_reshape(&flat, out, &out_shape, 3, s));
    _ = mlx.mlx_array_free(out);
    return flat;
}

const ConvPrep = struct {
    hidden: mlx.mlx_array, // conv'd sublayer input
    finish_dyn: mlx.mlx_array, // [1, L, ksize, groups] — kernels for finish()
};

/// Reference `GroupedDynamicCausalConv.prepare`: project BOTH tap sets from
/// the normed sublayer input (the finish kernels come from the INPUT, not the
/// sublayer output), convolve with `base_kernel[0]`, hand the finish set back.
fn convPrepare(
    conv: *const DynConv,
    normed: mlx.mlx_array, // [1, L, H]
    ksize: u32,
    group_size: u32,
    s: mlx.mlx_stream,
) !ConvPrep {
    const nsh = mlx.getShape(normed);
    const len = nsh[1];
    const h = nsh[2];
    const groups = @divExact(h, @as(c_int, @intCast(group_size)));

    const dyn_flat = try conv.kernel_projection.apply(normed, s);
    defer _ = mlx.mlx_array_free(dyn_flat);
    var dyn = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dyn);
    const dyn_shape = [_]c_int{ 1, len, 2, @intCast(ksize), groups };
    try mlx.check(mlx.mlx_reshape(&dyn, dyn_flat, &dyn_shape, 5, s));

    var prep_dyn = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(prep_dyn);
    var finish_dyn = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(finish_dyn);
    const set_shape = [_]c_int{ 1, len, @intCast(ksize), groups };
    inline for (.{ .{ 0, &prep_dyn }, .{ 1, &finish_dyn } }) |sel| {
        var cut = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cut);
        const start = [_]c_int{ 0, 0, sel[0], 0, 0 };
        const stop = [_]c_int{ 1, len, sel[0] + 1, @intCast(ksize), groups };
        const strides = [_]c_int{ 1, 1, 1, 1, 1 };
        try mlx.check(mlx.mlx_slice(&cut, dyn, &start, 5, &stop, 5, &strides, 5, s));
        try mlx.check(mlx.mlx_reshape(sel[1], cut, &set_shape, 4, s));
    }

    const base0 = try baseKernelHalf(conv.base_kernel, 0, s);
    defer _ = mlx.mlx_array_free(base0);
    const hidden_out = try groupedDynConv(normed, prep_dyn, base0, group_size, s);
    return .{ .hidden = hidden_out, .finish_dyn = finish_dyn };
}

/// Reference `GroupedDynamicCausalConv.finish`: convolve the sublayer OUTPUT
/// with `base_kernel[1]` + the finish kernels captured at prepare time.
fn convFinish(
    conv: *const DynConv,
    sub_out: mlx.mlx_array,
    finish_dyn: mlx.mlx_array,
    group_size: u32,
    s: mlx.mlx_stream,
) !mlx.mlx_array {
    const base1 = try baseKernelHalf(conv.base_kernel, 1, s);
    defer _ = mlx.mlx_array_free(base1);
    return groupedDynConv(sub_out, finish_dyn, base1, group_size, s);
}

/// `out = residual + convFinish(...)`, in one dispatch where the fused convolution serves the shape.
fn convFinishAdd(
    out: *mlx.mlx_array,
    conv: *const DynConv,
    sub_out: mlx.mlx_array,
    finish_dyn: mlx.mlx_array,
    group_size: u32,
    residual: mlx.mlx_array,
    s: mlx.mlx_stream,
) !void {
    const base1 = try baseKernelHalf(conv.base_kernel, 1, s);
    defer _ = mlx.mlx_array_free(base1);
    if (try @import("dflash_conv.zig").applyAdd(s, sub_out, finish_dyn, base1, group_size, residual)) |fused| {
        defer _ = mlx.mlx_array_free(fused);
        return mlx.check(mlx.mlx_array_set(out, fused));
    }
    const fin = try groupedDynConv(sub_out, finish_dyn, base1, group_size, s);
    defer _ = mlx.mlx_array_free(fin);
    try mlx.check(mlx.mlx_add(out, residual, fin, s));
}

/// `base_kernel[half]` → `[ksize, H]`.
fn baseKernelHalf(base_kernel: mlx.mlx_array, half: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const bsh = mlx.getShape(base_kernel); // [2, ksize, H]
    var cut = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cut);
    const start = [_]c_int{ half, 0, 0 };
    const stop = [_]c_int{ half + 1, bsh[1], bsh[2] };
    const strides = [_]c_int{ 1, 1, 1 };
    try mlx.check(mlx.mlx_slice(&cut, base_kernel, &start, 3, &stop, 3, &strides, 3, s));
    var out = mlx.mlx_array_new();
    const out_shape = [_]c_int{ bsh[1], bsh[2] };
    try mlx.check(mlx.mlx_reshape(&out, cut, &out_shape, 2, s));
    return out;
}

// ── DFlash2 path selector (forward + host trace) ──

pub const SelectedPath = struct {
    ids: []u32, // [m] chosen draft token ids
    chosen_idx: []u32, // [m] index of the choice within its candidate row
    cand_ids: []i32, // [m * top_k] candidate token ids per position
    /// [m * top_k] softmax(scores / temperature) per step — the proposal
    /// density q over the candidate set (zero everywhere else). Null on the
    /// greedy trace.
    q: ?[]f32,
    /// [m] the selector's confidence in each step: the chosen candidate's softmax share
    /// (the sampled arm's own q for it), what the round policy calibrates acceptance on.
    conf: []f32,

    pub fn deinit(self: *SelectedPath, allocator: std.mem.Allocator) void {
        allocator.free(self.ids);
        allocator.free(self.chosen_idx);
        allocator.free(self.cand_ids);
        allocator.free(self.conf);
        if (self.q) |qv| allocator.free(qv);
    }
};

/// Reference `CandidateSelector.select`: top-k candidates per position by
/// draft logit; score adjacent pairs `S_t(a,b) = U_t(b) + <pred(a) ⊙ H(h_t),
/// succ(b)>`; trace the best (or sampled) path from the anchor. All pairwise
/// edge scores are precomputed in ONE batched GPU dispatch ([m-1, k, k] +
/// the anchor row) and the 16-wide trace runs on host — same math as the
/// reference's sequential loop, chosen path identical, no per-step sync.
///
/// `blk_hidden` is the POST-final-norm block hidden `[1, bs, H]` (row 0 =
/// anchor, dropped here — the reference's `logits_start=1`); `draft_logits`
/// already has the anchor row dropped (`[1, m, V]`).
pub fn selectPath(
    allocator: std.mem.Allocator,
    sel: *const Selector,
    top_k: u32,
    blk_hidden: mlx.mlx_array,
    draft_logits: mlx.mlx_array,
    anchor_id: u32,
    temperature: f32,
    rand: std.Random,
    s: mlx.mlx_stream,
) !SelectedPath {
    const dl_shape = mlx.getShape(draft_logits);
    const m: usize = @intCast(dl_shape[1]);
    const vocab: c_int = dl_shape[2];
    const k: usize = @min(@as(usize, top_k), @as(usize, @intCast(vocab)));
    const hsh = mlx.getShape(blk_hidden);
    std.debug.assert(hsh[1] == dl_shape[1] + 1); // anchor row present on hidden

    // ── Candidates + unary logits ──
    var cands_i32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cands_i32);
    var unary_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(unary_f32);
    {
        var part = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(part);
        try mlx.check(mlx.mlx_argpartition_axis(&part, draft_logits, vocab - @as(c_int, @intCast(k)), 2, s));
        var cands_raw = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cands_raw);
        const start = [_]c_int{ 0, 0, vocab - @as(c_int, @intCast(k)) };
        const stop = [_]c_int{ 1, @intCast(m), vocab };
        const strides = [_]c_int{ 1, 1, 1 };
        try mlx.check(mlx.mlx_slice(&cands_raw, part, &start, 3, &stop, 3, &strides, 3, s));
        try mlx.check(mlx.mlx_astype(&cands_i32, cands_raw, .int32, s));
        var unary = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(unary);
        try mlx.check(mlx.mlx_take_along_axis(&unary, draft_logits, cands_i32, 2, s));
        try mlx.check(mlx.mlx_astype(&unary_f32, unary, .float32, s));
    }

    // ── Hidden projection over the draft rows ──
    var hidden_rows = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hidden_rows);
    {
        const start = [_]c_int{ 0, 1, 0 };
        const stop = [_]c_int{ 1, hsh[1], hsh[2] };
        const strides = [_]c_int{ 1, 1, 1 };
        try mlx.check(mlx.mlx_slice(&hidden_rows, blk_hidden, &start, 3, &stop, 3, &strides, 3, s));
    }
    const hp = try sel.hidden_projection.apply(hidden_rows, s);
    defer _ = mlx.mlx_array_free(hp);
    const rank: c_int = mlx.getShape(hp)[2];

    // ── Codebook rows for every candidate ──
    var cands_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cands_flat);
    {
        const flat_shape = [_]c_int{@intCast(m * k)};
        try mlx.check(mlx.mlx_reshape(&cands_flat, cands_i32, &flat_shape, 1, s));
    }
    var succ_rows = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(succ_rows);
    var pred_rows = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pred_rows);
    {
        var succ_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(succ_flat);
        try mlx.check(mlx.mlx_take_axis(&succ_flat, sel.succ_codebook, cands_flat, 0, s));
        var pred_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pred_flat);
        try mlx.check(mlx.mlx_take_axis(&pred_flat, sel.pred_codebook, cands_flat, 0, s));
        const rows_shape = [_]c_int{ @intCast(m), @intCast(k), rank };
        try mlx.check(mlx.mlx_reshape(&succ_rows, succ_flat, &rows_shape, 3, s));
        try mlx.check(mlx.mlx_reshape(&pred_rows, pred_flat, &rows_shape, 3, s));
    }

    // ── Edge scores: anchor row [k] + pairwise [m-1, k, k] ──
    var e0_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(e0_f32);
    {
        const aid: i32 = @intCast(anchor_id);
        const a_shape = [_]c_int{1};
        const aid_arr = mlx.mlx_array_new_data(&aid, &a_shape, 1, .int32);
        defer _ = mlx.mlx_array_free(aid_arr);
        var anchor_row = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(anchor_row);
        try mlx.check(mlx.mlx_take_axis(&anchor_row, sel.pred_codebook, aid_arr, 0, s)); // [1, rank]
        var h0 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(h0);
        {
            var cut = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(cut);
            const start = [_]c_int{ 0, 0, 0 };
            const stop = [_]c_int{ 1, 1, rank };
            const strides = [_]c_int{ 1, 1, 1 };
            try mlx.check(mlx.mlx_slice(&cut, hp, &start, 3, &stop, 3, &strides, 3, s));
            const h0_shape = [_]c_int{ 1, rank };
            try mlx.check(mlx.mlx_reshape(&h0, cut, &h0_shape, 2, s));
        }
        var ah = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ah);
        try mlx.check(mlx.mlx_multiply(&ah, anchor_row, h0, s)); // [1, rank]
        var succ0_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(succ0_t);
        {
            var succ0 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(succ0);
            const start = [_]c_int{ 0, 0, 0 };
            const stop = [_]c_int{ 1, @intCast(k), rank };
            const strides = [_]c_int{ 1, 1, 1 };
            try mlx.check(mlx.mlx_slice(&succ0, succ_rows, &start, 3, &stop, 3, &strides, 3, s));
            var succ0_2d = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(succ0_2d);
            const s2 = [_]c_int{ @intCast(k), rank };
            try mlx.check(mlx.mlx_reshape(&succ0_2d, succ0, &s2, 2, s));
            const perm = [_]c_int{ 1, 0 };
            try mlx.check(mlx.mlx_transpose_axes(&succ0_t, succ0_2d, &perm, 2, s));
        }
        var e0 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(e0);
        try mlx.check(mlx.mlx_matmul(&e0, ah, succ0_t, s)); // [1, k]
        try mlx.check(mlx.mlx_astype(&e0_f32, e0, .float32, s));
    }
    var e_f32: mlx.mlx_array = .{ .ctx = null };
    defer if (e_f32.ctx != null) {
        _ = mlx.mlx_array_free(e_f32);
    };
    if (m > 1) {
        // A = pred_rows[0..m-1] ⊙ H[1..m]  → [m-1, k, rank]
        var pred_head = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pred_head);
        {
            const start = [_]c_int{ 0, 0, 0 };
            const stop = [_]c_int{ @intCast(m - 1), @intCast(k), rank };
            const strides = [_]c_int{ 1, 1, 1 };
            try mlx.check(mlx.mlx_slice(&pred_head, pred_rows, &start, 3, &stop, 3, &strides, 3, s));
        }
        var h_tail = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(h_tail);
        {
            var cut = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(cut);
            const start = [_]c_int{ 0, 1, 0 };
            const stop = [_]c_int{ 1, @intCast(m), rank };
            const strides = [_]c_int{ 1, 1, 1 };
            try mlx.check(mlx.mlx_slice(&cut, hp, &start, 3, &stop, 3, &strides, 3, s));
            const t_shape = [_]c_int{ @intCast(m - 1), 1, rank };
            try mlx.check(mlx.mlx_reshape(&h_tail, cut, &t_shape, 3, s));
        }
        var a_mat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(a_mat);
        try mlx.check(mlx.mlx_multiply(&a_mat, pred_head, h_tail, s));
        var succ_tail_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(succ_tail_t);
        {
            var succ_tail = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(succ_tail);
            const start = [_]c_int{ 1, 0, 0 };
            const stop = [_]c_int{ @intCast(m), @intCast(k), rank };
            const strides = [_]c_int{ 1, 1, 1 };
            try mlx.check(mlx.mlx_slice(&succ_tail, succ_rows, &start, 3, &stop, 3, &strides, 3, s));
            const perm = [_]c_int{ 0, 2, 1 };
            try mlx.check(mlx.mlx_transpose_axes(&succ_tail_t, succ_tail, &perm, 3, s));
        }
        var e = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(e);
        try mlx.check(mlx.mlx_matmul(&e, a_mat, succ_tail_t, s)); // [m-1, k, k]
        e_f32 = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&e_f32, e, .float32, s));
    }

    // ── ONE batched eval, then the host trace ──
    {
        const eval_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(eval_vec);
        _ = mlx.mlx_vector_array_append_value(eval_vec, cands_i32);
        _ = mlx.mlx_vector_array_append_value(eval_vec, unary_f32);
        _ = mlx.mlx_vector_array_append_value(eval_vec, e0_f32);
        if (e_f32.ctx != null) _ = mlx.mlx_vector_array_append_value(eval_vec, e_f32);
        try mlx.check(mlx.mlx_eval(eval_vec));
    }
    const cand_data = mlx.mlx_array_data_int32(cands_i32) orelse return error.MlxArrayDataNull;
    const unary_data = mlx.mlx_array_data_float32(unary_f32) orelse return error.MlxArrayDataNull;
    const e0_data = mlx.mlx_array_data_float32(e0_f32) orelse return error.MlxArrayDataNull;
    const e_data: ?[*]const f32 = if (e_f32.ctx != null)
        (mlx.mlx_array_data_float32(e_f32) orelse return error.MlxArrayDataNull)
    else
        null;

    const stochastic = temperature > 0;
    var out = SelectedPath{
        .ids = try allocator.alloc(u32, m),
        .chosen_idx = undefined,
        .cand_ids = undefined,
        .q = null,
        .conf = undefined,
    };
    errdefer allocator.free(out.ids);
    out.chosen_idx = try allocator.alloc(u32, m);
    errdefer allocator.free(out.chosen_idx);
    out.cand_ids = try allocator.alloc(i32, m * k);
    errdefer allocator.free(out.cand_ids);
    out.conf = try allocator.alloc(f32, m);
    errdefer allocator.free(out.conf);
    @memcpy(out.cand_ids, cand_data[0 .. m * k]);
    if (stochastic) out.q = try allocator.alloc(f32, m * k);

    var scores_buf: [64]f32 = undefined; // top_k is 16 on the real checkpoint
    std.debug.assert(k <= scores_buf.len);
    var prev_idx: usize = 0;
    var t: usize = 0;
    while (t < m) : (t += 1) {
        const scores = scores_buf[0..k];
        for (scores, 0..) |*sc, j| {
            const edge = if (t == 0) e0_data[j] else e_data.?[(t - 1) * k * k + prev_idx * k + j];
            sc.* = unary_data[t * k + j] + edge;
        }
        var choice: usize = 0;
        if (!stochastic) {
            for (scores, 0..) |sc, j| {
                if (sc > scores[choice]) choice = j;
            }
        } else {
            // Reference `_sampling_probs(scores, temperature)`: f32 softmax
            // over the candidate set — no top-p/top-k inside the selector.
            var mx: f32 = -std.math.inf(f32);
            for (scores) |sc| mx = @max(mx, sc);
            var total: f32 = 0;
            const q_row = out.q.?[t * k .. (t + 1) * k];
            for (scores, q_row) |sc, *qv| {
                qv.* = @exp((sc - mx) / temperature);
                total += qv.*;
            }
            for (q_row) |*qv| qv.* /= total;
            const u = rand.float(f32);
            var acc: f32 = 0;
            choice = k - 1;
            for (q_row, 0..) |qv, j| {
                acc += qv;
                if (u < acc) {
                    choice = j;
                    break;
                }
            }
        }
        out.ids[t] = @intCast(cand_data[t * k + choice]);
        out.chosen_idx[t] = @intCast(choice);
        out.conf[t] = if (stochastic) out.q.?[t * k + choice] else softmaxShare(scores, choice);
        prev_idx = choice;
    }
    return out;
}

/// Softmax of `scores` at the chosen index.
fn softmaxShare(scores: []const f32, choice: usize) f32 {
    var mx: f32 = -std.math.inf(f32);
    for (scores) |sc| mx = @max(mx, sc);
    var total: f32 = 0;
    for (scores) |sc| total += @exp(sc - mx);
    return @exp(scores[choice] - mx) / total;
}

// ── Context append + block forward ──

/// Project trunk captures through the encoder and append per-layer K/V for
/// `n` new context tokens at absolute positions `[first_pos, first_pos+n)`.
/// `captures` are the trunk layer OUTPUTS at `target_layer_ids`, in config
/// order, each `[1, n, hidden]`. Purely lazy — caller decides when to eval
/// (`ctx.appendEvalArrays`).
pub fn appendContext(
    model: *const DflashModel,
    ctx: *DflashCtx,
    captures: []const mlx.mlx_array,
    first_pos: usize,
) !void {
    const s = model.s;
    const cfg = &model.config;
    std.debug.assert(first_pos == ctx.absLen()); // contiguity — no holes

    const enc = try encodeContext(model, captures);
    defer _ = mlx.mlx_array_free(enc);

    for (model.layers, 0..) |*lw, li| {
        const k = try projectHeads(enc, &lw.k, lw.k_norm, cfg.num_key_value_heads, cfg.head_dim, cfg.rms_norm_eps, cfg.rope_theta, first_pos, true, cfg.rope_traditional, s);
        defer _ = mlx.mlx_array_free(k);
        const v = try projectHeadsNoNorm(enc, &lw.v, cfg.num_key_value_heads, cfg.head_dim, s);
        defer _ = mlx.mlx_array_free(v);
        _ = try ctx.cache.update(@intCast(li), k, v, s, 0);
    }
    if (model.sliding_serving) try trimGlmServingContext(model, ctx);
}

/// A sliding-serving assistant is entirely sliding attention. Materialize the
/// retained window after every prefill chunk, so an old view cannot pin the
/// complete prompt allocation. Absolute RoPE positions live in base_pos.
pub fn trimGlmServingContext(model: *const DflashModel, ctx: *DflashCtx) !void {
    if (model.config.sliding_window < 2) return error.GlmDflashWindowUnsupported;
    for (model.layers) |layer| if (layer.layer_type != .sliding_attention) return error.GlmDflashWindowUnsupported;
    const keep = @as(usize, model.config.sliding_window) - 1;
    if (ctx.cache.step <= keep) return;
    const drop = ctx.cache.step - keep;
    var ops = @import("glm5_model.zig").Ops{ .s = model.s };
    defer ops.deinit();
    const arrays = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(arrays);
    for (ctx.cache.entries) |*entry| {
        const keys = try ops.contiguous(try ops.slice(entry.keys, 2, @intCast(drop), @intCast(ctx.cache.step)));
        const values = try ops.contiguous(try ops.slice(entry.values, 2, @intCast(drop), @intCast(ctx.cache.step)));
        try mlx.check(mlx.mlx_array_set(&entry.keys, keys));
        try mlx.check(mlx.mlx_array_set(&entry.values, values));
        entry.offset = keep;
        try mlx.check(mlx.mlx_vector_array_append_value(arrays, entry.keys));
        try mlx.check(mlx.mlx_vector_array_append_value(arrays, entry.values));
    }
    try mlx.check(mlx.mlx_eval(arrays));
    ctx.cache.step = keep;
    ctx.base_pos += drop;
}

/// Encoder projection: concatenate the trunk captures on features →
/// `encoder.fc` → RMS norm. Returns `[1, n, hidden]`, caller frees.
pub fn encodeContext(model: *const DflashModel, captures: []const mlx.mlx_array) !mlx.mlx_array {
    const s = model.s;
    std.debug.assert(captures.len == model.config.target_layer_ids.len);
    var cat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cat);
    {
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        for (captures) |c| _ = mlx.mlx_vector_array_append_value(vec, c);
        try mlx.check(mlx.mlx_concatenate_axis(&cat, vec, 2, s));
    }
    const projected = try model.fc.apply(cat, s);
    defer _ = mlx.mlx_array_free(projected);
    return rmsNormFn(projected, model.enc_norm, model.config.rms_norm_eps, s);
}

/// V path: project + reshape + transpose, no norm, no RoPE.
fn projectHeadsNoNorm(
    x: mlx.mlx_array,
    lin: *const DflashLinear,
    n_heads: u32,
    head_dim: u32,
    s: mlx.mlx_stream,
) !mlx.mlx_array {
    const proj = try lin.apply(x, s);
    defer _ = mlx.mlx_array_free(proj);
    const xsh = mlx.getShape(x);
    const hs = [_]c_int{ xsh[0], xsh[1], @intCast(n_heads), @intCast(head_dim) };
    var reshaped = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(reshaped);
    try mlx.check(mlx.mlx_reshape(&reshaped, proj, &hs, 4, s));
    const perm = [_]c_int{ 0, 2, 1, 3 };
    var transposed = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_transpose_axes(&transposed, reshaped, &perm, 4, s));
    return transposed;
}

/// One assistant forward over the noise block. `noise_embeds` is the RAW
/// trunk embedding lookup `[1, q_len, hidden]` of `[anchor, mask…]`;
/// `anchor_pos` is the anchor's absolute position (== trunk `cache.step`).
/// Returns the post-final-norm hidden `[1, q_len, hidden]` (caller frees;
/// the caller projects through the trunk lm_head and DROPS row 0). Block
/// K/V transit through the cache's spare capacity and are truncated back
/// out before returning — the context cache is unchanged.
pub fn forwardBlock(
    model: *const DflashModel,
    ctx: *DflashCtx,
    noise_embeds: mlx.mlx_array,
    anchor_pos: usize,
) !mlx.mlx_array {
    return forwardBlockMode(model, ctx, noise_embeds, anchor_pos, blockTailEligible(model, ctx, noise_embeds), @intCast(mlx.getShape(noise_embeds)[1]));
}

/// Return only the requested leading outputs. Every layer still attends to
/// the entire noise block; only the final layer's queries and row-local MLP
/// are shortened. Causal convolutions cannot read discarded later rows.
pub fn forwardBlockPrefix(model: *const DflashModel, ctx: *DflashCtx, noise_embeds: mlx.mlx_array, anchor_pos: usize, rows: u32) !mlx.mlx_array {
    if (rows == 0 or rows > mlx.getShape(noise_embeds)[1]) return error.InvalidDflashPrefix;
    return forwardBlockMode(model, ctx, noise_embeds, anchor_pos, blockTailEligible(model, ctx, noise_embeds), rows);
}

var tail_mask: ?mlx.mlx_array = null;
var tail_mask_shape: [4]usize = .{ 0, 0, 0, 0 };

/// The sliding mask depends only on positions relative to the first context key (`relative` is
/// the anchor's), so one evaluated copy serves every draft of that geometry; the caller owns the
/// returned reference.
fn tailMask(q_len: u32, ctx_len: usize, relative: usize, window: u32, s: mlx.mlx_stream) !?mlx.mlx_array {
    if (relative + q_len - 1 < window) return null;
    const shape = [4]usize{ q_len, ctx_len, relative, window };
    if (tail_mask == null or !std.mem.eql(usize, &shape, &tail_mask_shape)) {
        const built = (try buildBlockMask(.sliding_attention, 0, ctx_len, relative, q_len, window, s)) orelse return null;
        errdefer _ = mlx.mlx_array_free(built);
        try mlx.check(mlx.mlx_array_eval(built));
        if (tail_mask) |old| _ = mlx.mlx_array_free(old);
        tail_mask = built;
        tail_mask_shape = shape;
    }
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&out, tail_mask.?));
    return out;
}

fn blockTailEligible(model: *const DflashModel, ctx: *const DflashCtx, noise: mlx.mlx_array) bool {
    if (!model.sliding_serving or model.layers.len != 5 or model.config.block_size != 8 or model.config.sliding_window != 2048 or ctx.cache.config.scheme != .off or ctx.cache.step < 2048 or mlx.getShape(noise)[1] != 8) return false;
    for (model.layers, ctx.cache.entries) |layer, entry| {
        if (layer.layer_type != .sliding_attention or !entry.initialized or entry.base != 0 or entry.offset != ctx.cache.step or mlx.mlx_array_dtype(entry.keys) != .bfloat16 or mlx.mlx_array_dtype(entry.values) != .bfloat16) return false;
    }
    return true;
}

fn forwardBlockMode(model: *const DflashModel, ctx: *DflashCtx, noise_embeds: mlx.mlx_array, anchor_pos: usize, block_tail: bool, output_rows: u32) !mlx.mlx_array {
    const s = model.s;
    const cfg = &model.config;
    const q_len_c = mlx.getShape(noise_embeds)[1];
    const q_len: u32 = @intCast(q_len_c);
    std.debug.assert(anchor_pos == ctx.absLen()); // block starts one past context
    const ctx_len = ctx.cache.step;
    const tail_len = if (block_tail) @min(ctx_len, cfg.sliding_window - 1) else ctx_len;
    const dropped = ctx_len - tail_len;
    var sliding_mask: ?mlx.mlx_array = null;
    defer if (sliding_mask) |mask| {
        _ = mlx.mlx_array_free(mask);
    };
    for (model.layers) |layer| if (layer.layer_type == .sliding_attention) {
        sliding_mask = if (model.sliding_serving)
            try tailMask(q_len, tail_len, anchor_pos - (ctx.base_pos + dropped), cfg.sliding_window, s)
        else
            try buildBlockMask(.sliding_attention, ctx.base_pos + dropped, tail_len, anchor_pos, q_len, cfg.sliding_window, s);
        break;
    };
    const attn_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));
    const perm_back = [_]c_int{ 0, 2, 1, 3 };
    const none_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(none_mask);

    var x = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&x, noise_embeds));
    errdefer _ = mlx.mlx_array_free(x);

    for (model.layers, 0..) |*lw, li| {
        const rows: c_int = if (li + 1 == model.layers.len) @intCast(output_rows) else q_len_c;
        var tail_ops = @import("glm5_model.zig").Ops{ .s = s };
        defer tail_ops.deinit();
        const normed = try rmsNormFn(x, lw.input_norm, cfg.rms_norm_eps, s);
        defer _ = mlx.mlx_array_free(normed);

        // DFlash2: conv the normed input; attention runs on the conv'd
        // hidden and its output is conv'd again with kernels projected from
        // the SAME normed input. v1 layers (`attention_conv == null`) take
        // the exact original path.
        var attn_prep: ?ConvPrep = if (lw.attention_conv) |*cv|
            try convPrepare(cv, normed, cfg.conv_kernel_size, cfg.conv_group_size, s)
        else
            null;
        defer if (attn_prep) |*cp| {
            _ = mlx.mlx_array_free(cp.hidden);
            _ = mlx.mlx_array_free(cp.finish_dyn);
        };
        const attn_in = if (attn_prep) |*cp| cp.hidden else normed;

        const query_in = if (rows < q_len_c) try tail_ops.slice(attn_in, 1, 0, rows) else attn_in;
        const q = try projectHeads(query_in, &lw.q, lw.q_norm, cfg.num_attention_heads, cfg.head_dim, cfg.rms_norm_eps, cfg.rope_theta, anchor_pos, true, cfg.rope_traditional, s);
        defer _ = mlx.mlx_array_free(q);
        const bk = try projectHeads(attn_in, &lw.k, lw.k_norm, cfg.num_key_value_heads, cfg.head_dim, cfg.rms_norm_eps, cfg.rope_theta, anchor_pos, true, cfg.rope_traditional, s);
        defer _ = mlx.mlx_array_free(bk);
        const bv = try projectHeadsNoNorm(attn_in, &lw.v, cfg.num_key_value_heads, cfg.head_dim, s);
        defer _ = mlx.mlx_array_free(bv);

        const view = if (block_tail) blk: {
            const entry = &ctx.cache.entries[li];
            const k = try tail_ops.slice(entry.keys, 2, @intCast(dropped), @intCast(ctx_len));
            const v = try tail_ops.slice(entry.values, 2, @intCast(dropped), @intCast(ctx_len));
            break :blk transformer_mod.DenseKVView{ .k = try tail_ops.concat(&.{ k, bk }, 2), .v = try tail_ops.concat(&.{ v, bv }, 2), .owned = false };
        } else try ctx.cache.update(@intCast(li), bk, bv, s, 0);

        const full_mask = if (lw.layer_type == .sliding_attention) sliding_mask else null;
        const mask = if (full_mask) |m| (if (rows < q_len_c) try tail_ops.slice(m, 2, 0, rows) else m) else null;

        var attn_out = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn_out);
        if (mask) |m| {
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q, view.k, view.v, attn_scale, "array", m, .{ .ctx = null }, false, s));
        } else {
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q, view.k, view.v, attn_scale, "", none_mask, .{ .ctx = null }, false, s));
        }

        var attn_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn_t);
        try mlx.check(mlx.mlx_transpose_axes(&attn_t, attn_out, &perm_back, 4, s));
        var attn_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn_flat);
        const flat_shape = [_]c_int{ 1, rows, @intCast(cfg.num_attention_heads * cfg.head_dim) };
        try mlx.check(mlx.mlx_reshape(&attn_flat, attn_t, &flat_shape, 3, s));
        const o_out = try lw.o.apply(attn_flat, s);
        defer _ = mlx.mlx_array_free(o_out);
        var h_new = mlx.mlx_array_new();
        const residual = if (rows < q_len_c) try tail_ops.slice(x, 1, 0, rows) else x;
        if (lw.attention_conv) |*cv| {
            const finish_dyn = if (rows < q_len_c) try tail_ops.slice(attn_prep.?.finish_dyn, 1, 0, rows) else attn_prep.?.finish_dyn;
            try convFinishAdd(&h_new, cv, o_out, finish_dyn, cfg.conv_group_size, residual, s);
        } else try mlx.check(mlx.mlx_add(&h_new, residual, o_out, s));
        _ = mlx.mlx_array_free(x);
        x = h_new;

        const ff_normed = try rmsNormFn(x, lw.post_attn_norm, cfg.rms_norm_eps, s);
        defer _ = mlx.mlx_array_free(ff_normed);
        var mlp_prep: ?ConvPrep = if (lw.mlp_conv) |*cv|
            try convPrepare(cv, ff_normed, cfg.conv_kernel_size, cfg.conv_group_size, s)
        else
            null;
        defer if (mlp_prep) |*cp| {
            _ = mlx.mlx_array_free(cp.hidden);
            _ = mlx.mlx_array_free(cp.finish_dyn);
        };
        const mlp_in = if (mlp_prep) |*cp| cp.hidden else ff_normed;
        const act = if (try @import("dflash_qmv.zig").gateUpAct(s, mlp_in, .{ lw.gate.w, lw.gate.scales, lw.gate.biases }, .{ lw.up.w, lw.up.scales, lw.up.biases })) |fused| fused else blk: {
            const gate = try lw.gate.apply(mlp_in, s);
            defer _ = mlx.mlx_array_free(gate);
            const up = try lw.up.apply(mlp_in, s);
            defer _ = mlx.mlx_array_free(up);
            break :blk try swiglu(gate, up, s);
        };
        defer _ = mlx.mlx_array_free(act);
        const down = try lw.down.apply(act, s);
        defer _ = mlx.mlx_array_free(down);
        var h_next = mlx.mlx_array_new();
        if (lw.mlp_conv) |*cv| {
            try convFinishAdd(&h_next, cv, down, mlp_prep.?.finish_dyn, cfg.conv_group_size, x, s);
        } else try mlx.check(mlx.mlx_add(&h_next, x, down, s));
        _ = mlx.mlx_array_free(x);
        x = h_next;
        // The GPU runs each finished layer while the host builds the next one.
        if (model.sliding_serving and li + 1 < model.layers.len) {
            const pending = mlx.mlx_vector_array_new_value(x);
            defer _ = mlx.mlx_vector_array_free(pending);
            try mlx.check(mlx.mlx_async_eval(pending));
        }
    }

    // Evict the block K/V — the context cache must be exactly as it was.
    if (!block_tail) try ctx.cache.truncate(ctx_len, s);

    const out = try rmsNormFn(x, model.final_norm, cfg.rms_norm_eps, s);
    _ = mlx.mlx_array_free(x);
    return out;
}

// ── Tests ──

const testing = std.testing;

const MUSE_ASSISTANT_CONFIG_JSON =
    \\{
    \\  "architectures": ["MuseGlimmerAssistantModel"],
    \\  "block_size": 16,
    \\  "head_dim": 128,
    \\  "hidden_act": "silu",
    \\  "hidden_size": 6656,
    \\  "intermediate_size": 19968,
    \\  "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention"],
    \\  "mask_token_id": 201818,
    \\  "max_position_embeddings": 131072,
    \\  "model_type": "muse_glimmer_assistant",
    \\  "num_attention_heads": 32,
    \\  "num_hidden_layers": 5,
    \\  "num_key_value_heads": 8,
    \\  "rms_norm_eps": 1e-05,
    \\  "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
    \\  "sliding_window": 2048,
    \\  "target_layer_ids": [1, 13, 25, 37, 49]
    \\}
;

// Gemma 4 assistant drafter shape (cross-attention drafter, NOT DFlash):
// no block_size / mask_token_id / target_layer_ids.
const GEMMA_ASSISTANT_CONFIG_JSON =
    \\{
    \\  "model_type": "gemma4_assistant",
    \\  "backbone_hidden_size": 2560,
    \\  "num_centroids": 512,
    \\  "centroid_intermediate_top_k": 16,
    \\  "use_ordered_embeddings": true,
    \\  "text_config": {
    \\    "hidden_size": 256,
    \\    "num_hidden_layers": 4,
    \\    "num_attention_heads": 4,
    \\    "head_dim": 256,
    \\    "intermediate_size": 1024,
    \\    "sliding_window": 512,
    \\    "vocab_size": 262144,
    \\    "rms_norm_eps": 1e-06,
    \\    "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"]
    \\  }
    \\}
;

test "dflash: muse assistant config parses with the full DFlash contract" {
    const allocator = testing.allocator;
    var cfg = try parseConfigFromJson(allocator, MUSE_ASSISTANT_CONFIG_JSON);
    defer cfg.deinit(allocator);

    try testing.expectEqual(@as(u32, 16), cfg.block_size);
    try testing.expectEqual(@as(u32, 201818), cfg.mask_token_id);
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 13, 25, 37, 49 }, cfg.target_layer_ids);
    try testing.expectEqual(@as(u32, 6656), cfg.hidden_size);
    try testing.expectEqual(@as(u32, 5), cfg.num_hidden_layers);
    try testing.expectEqual(@as(u32, 32), cfg.num_attention_heads);
    try testing.expectEqual(@as(u32, 8), cfg.num_key_value_heads);
    try testing.expectEqual(@as(u32, 128), cfg.head_dim);
    try testing.expectEqual(@as(u32, 19968), cfg.intermediate_size);
    try testing.expectEqual(@as(u32, 2048), cfg.sliding_window);
    try testing.expectApproxEqAbs(@as(f32, 1e-5), cfg.rms_norm_eps, 1e-9);
    try testing.expectApproxEqAbs(@as(f32, 500000.0), cfg.rope_theta, 1.0);
    try testing.expectEqual(@as(usize, 5), cfg.layer_types.len);
    for (cfg.layer_types) |lt| try testing.expectEqual(LayerType.sliding_attention, lt);
}

// The real incoai/Qwen3.8-27B-DFlash2 config shape (fetched 2026-08-18):
// the contract nests under `dflash_config`, model_type is a bare "qwen3".
const DFLASH2_CONFIG_JSON =
    \\{
    \\  "architectures": ["DFlash2DraftModel"],
    \\  "is_causal": false,
    \\  "dflash_config": {
    \\    "block_size": 8,
    \\    "conv_group_size": 16,
    \\    "conv_kernel_size": 2,
    \\    "mask_token_id": 248070,
    \\    "selector_rank": 256,
    \\    "selector_top_k": 16,
    \\    "target_layer_ids": [5, 19, 33, 47, 61]
    \\  },
    \\  "head_dim": 128,
    \\  "hidden_size": 5120,
    \\  "intermediate_size": 17408,
    \\  "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention"],
    \\  "max_position_embeddings": 262144,
    \\  "model_type": "qwen3",
    \\  "num_attention_heads": 32,
    \\  "num_hidden_layers": 5,
    \\  "num_key_value_heads": 8,
    \\  "rms_norm_eps": 1e-06,
    \\  "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
    \\  "sliding_window": 2048,
    \\  "vocab_size": 248320
    \\}
;

// LiquidAI/LFM2.5-2.6B-DSpark, fetched 2026-08-21. The contract is SPLIT:
// `block_size` sits at the root while `mask_token_id` / `target_layer_ids`
// nest under `dflash_config`, rope theta is a ROOT `rope_theta`, and the
// drafter's own rope is GPT-J interleaved (`rope_is_neox_style: false`).
const DSPARK_LFM2_CONFIG_JSON =
    \\{
    \\  "architectures": ["Lfm2DSparkDraftModel"],
    \\  "model_type": "qwen3",
    \\  "hidden_size": 2048,
    \\  "num_hidden_layers": 5,
    \\  "num_attention_heads": 32,
    \\  "num_key_value_heads": 8,
    \\  "head_dim": 64,
    \\  "intermediate_size": 6144,
    \\  "rms_norm_eps": 1e-05,
    \\  "vocab_size": 128000,
    \\  "rope_theta": 10000000.0,
    \\  "layer_types": ["full_attention", "full_attention", "full_attention", "full_attention", "full_attention"],
    \\  "block_size": 9,
    \\  "dflash_config": {"mask_token_id": 125017, "target_layer_ids": [2, 9, 17, 21, 27], "num_target_layers": 30},
    \\  "markov_rank": 256,
    \\  "rope_is_neox_style": false,
    \\  "enable_confidence_head": true,
    \\  "markov_head_type": "vanilla"
    \\}
;

test "dflash: DSpark contract splits across root and dflash_config, and both halves are read" {
    const allocator = testing.allocator;

    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, DSPARK_LFM2_CONFIG_JSON, .{});
    defer parsed.deinit();
    try testing.expect(isDflashConfigJson(parsed.value.object));

    var cfg = try parseConfigFromJson(allocator, DSPARK_LFM2_CONFIG_JSON);
    defer cfg.deinit(allocator);
    // Declared 9 counts DRAFTS (DSpark reads the anchor row); the engine
    // speaks verify width, so it normalizes to 10.
    try testing.expectEqual(@as(u32, 10), cfg.block_size);
    try testing.expect(cfg.anchor_row_drafts);
    try testing.expectEqual(@as(u32, 125017), cfg.mask_token_id); // nested
    try testing.expectEqualSlices(u32, &[_]u32{ 2, 9, 17, 21, 27 }, cfg.target_layer_ids);
    // Root `rope_theta` — the muse/DFlash2 spelling nests it under
    // `rope_parameters`, and reading only that silently drafts at theta 10000.
    try testing.expectApproxEqAbs(@as(f32, 10000000.0), cfg.rope_theta, 1.0);
    try testing.expect(cfg.rope_traditional);
    try testing.expectEqual(@as(u32, 256), cfg.markov_rank);
    try testing.expect(cfg.isDspark());
    try testing.expect(!cfg.isDflash2());
}

test "dflash: a DSpark config with an unported markov head type is refused by name" {
    const allocator = testing.allocator;
    const gated = try std.mem.replaceOwned(u8, allocator, DSPARK_LFM2_CONFIG_JSON, "\"vanilla\"", "\"gated\"");
    defer allocator.free(gated);
    try testing.expectError(error.UnsupportedMarkovHeadType, parseConfigFromJson(allocator, gated));
}

test "dflash: DFlash2 nested dflash_config contract parses with selector + conv fields" {
    const allocator = testing.allocator;

    // Detection accepts the nested shape…
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, DFLASH2_CONFIG_JSON, .{});
    defer parsed.deinit();
    try testing.expect(isDflashConfigJson(parsed.value.object));

    // …and the parser reads the triple + the four DFlash2 fields from it.
    var cfg = try parseConfigFromJson(allocator, DFLASH2_CONFIG_JSON);
    defer cfg.deinit(allocator);
    try testing.expectEqual(@as(u32, 8), cfg.block_size);
    try testing.expectEqual(@as(u32, 248070), cfg.mask_token_id);
    try testing.expectEqualSlices(u32, &[_]u32{ 5, 19, 33, 47, 61 }, cfg.target_layer_ids);
    try testing.expectEqual(@as(u32, 16), cfg.conv_group_size);
    try testing.expectEqual(@as(u32, 2), cfg.conv_kernel_size);
    try testing.expectEqual(@as(u32, 256), cfg.selector_rank);
    try testing.expectEqual(@as(u32, 16), cfg.selector_top_k);
    try testing.expectEqual(@as(u32, 5120), cfg.hidden_size);
    try testing.expectEqual(@as(u32, 5), cfg.num_hidden_layers);
    try testing.expectApproxEqAbs(@as(f32, 1e7), cfg.rope_theta, 1.0);

    // A v1 config carries none of the new fields → all zero / neutral.
    var v1 = try parseConfigFromJson(allocator, MUSE_ASSISTANT_CONFIG_JSON);
    defer v1.deinit(allocator);
    try testing.expectEqual(@as(u32, 0), v1.selector_rank);
    try testing.expectEqual(@as(u32, 0), v1.conv_kernel_size);
    try testing.expectEqual(@as(f32, 0), v1.logit_softcap);
    try testing.expectEqual(@as(f32, 1.0), v1.output_multiplier);
}

test "dflash2: muse sidecar's softcap + output_multiplier parse and transform the draft logits" {
    // incoai/Muse-Glimmer-30B-DFlash2 nests final_logit_softcapping 20 +
    // output_multiplier 0.196… under dflash_config — the trunk head we borrow
    // is the BARE Linear, so without these the selector's unary term is on
    // the wrong scale against the codebook edges (argmax drafts are invariant
    // to monotone transforms; pairwise SUMS are not).
    const allocator = testing.allocator;
    const muse2_json =
        \\{"model_type":"qwen3","is_causal":false,
        \\ "dflash_config":{"block_size":16,"mask_token_id":201818,"target_layer_ids":[1,13],
        \\   "conv_kernel_size":2,"conv_group_size":16,"selector_rank":256,"selector_top_k":16,
        \\   "final_logit_softcapping":20.0,"output_multiplier":0.19611613513818404},
        \\ "hidden_size":6656,"num_hidden_layers":1,"num_attention_heads":32,"head_dim":128,
        \\ "intermediate_size":19968,"rms_norm_eps":1e-5,"sliding_window":2048,
        \\ "rope_parameters":{"rope_theta":500000.0,"rope_type":"default"},
        \\ "layer_types":["sliding_attention"]}
    ;
    var cfg = try parseConfigFromJson(allocator, muse2_json);
    defer cfg.deinit(allocator);
    try testing.expectApproxEqAbs(@as(f32, 20.0), cfg.logit_softcap, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.19611613), cfg.output_multiplier, 1e-6);

    if (mlx.noGpuBackend()) return;
    const s = mlx.gpuStream();
    // Reference compute_logits: l*mult, then tanh(l/cap)*cap. Hand-computed.
    const vals = [_]f32{ 0.0, 51.0, -102.0, 300.0 };
    const shape = [_]c_int{ 1, 1, 4 };
    const raw = mlx.mlx_array_new_data(&vals, &shape, 3, .float32);
    defer _ = mlx.mlx_array_free(raw);
    const out = try applyLogitTransforms(raw, 0.19611613513818404, 20.0, s);
    defer _ = mlx.mlx_array_free(out);
    const got = try TinyFix.readF32(out, allocator, s);
    defer testing.allocator.free(got);
    for (vals, got) |x, g| {
        const want = 20.0 * std.math.tanh(x * 0.19611613513818404 / 20.0);
        try testing.expect(@abs(g - want) < 1e-4);
    }
}

test "dflash: a config declaring is_causal true is refused by name" {
    // Our block forward is bidirectional-only (v1 parity-pinned); the z-lab
    // reference defaults sliding layers CAUSAL inside the block, so a future
    // checkpoint shipping is_causal:true must refuse rather than silently
    // draft against the wrong attention pattern. DFlash2 ships false.
    const allocator = testing.allocator;
    const causal_json =
        \\{"model_type":"qwen3","is_causal":true,
        \\ "dflash_config":{"block_size":4,"mask_token_id":7,"target_layer_ids":[1,3]},
        \\ "hidden_size":64,"num_hidden_layers":1,"num_attention_heads":4,"head_dim":16,
        \\ "intermediate_size":128,"rms_norm_eps":1e-5,
        \\ "layer_types":["full_attention"]}
    ;
    try testing.expectError(error.DflashCausalBlockUnsupported, parseConfigFromJson(allocator, causal_json));
}

test "dflash: gemma assistant config is NOT detected as DFlash" {
    const allocator = testing.allocator;
    // Detection helper says no…
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, GEMMA_ASSISTANT_CONFIG_JSON, .{});
    defer parsed.deinit();
    try testing.expect(!isDflashConfigJson(parsed.value.object));
    // …and the parser rejects it by name.
    try testing.expectError(error.NotDflashConfig, parseConfigFromJson(allocator, GEMMA_ASSISTANT_CONFIG_JSON));
}

test "dflash: muse assistant config IS detected" {
    const allocator = testing.allocator;
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, MUSE_ASSISTANT_CONFIG_JSON, .{});
    defer parsed.deinit();
    try testing.expect(isDflashConfigJson(parsed.value.object));
}

test "dflash: out-of-range target_layer_ids rejected by name" {
    // Muse trunk has 52 layers; id 49 is fine, id 52 is not.
    try validateTargetLayers(&[_]u32{ 1, 13, 25, 37, 49 }, 52);
    try testing.expectError(error.DflashTargetLayerOutOfRange, validateTargetLayers(&[_]u32{ 1, 52 }, 52));
    try testing.expectError(error.DflashTargetLayerOutOfRange, validateTargetLayers(&[_]u32{60}, 52));
}

test "dflash: block size resolves from config" {
    // A machine WITH a wide verify lane keeps the checkpoint's own block.
    try testing.expectEqual(@as(u32, 16), resolveBlockSize(16, true, NO_WIDE_LANE_BLOCK_CAP));
    // Floor 2.
    try testing.expectEqual(@as(u32, 2), resolveBlockSize(1, true, NO_WIDE_LANE_BLOCK_CAP));
}

test "dflash: an assistant merged into the checkpoint is found without a flag" {
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &path_buf);
    const model_dir = path_buf[0..root_len];

    // A model dir with nothing in it resolves to nothing.
    try testing.expectEqual(@as(?[]u8, null), resolveInDirDrafter(io, allocator, model_dir));

    // A `drafter/` subdir declaring the CONTRACT is the sidecar — the same
    // three fields the flag path probes, so detection cannot drift between
    // "pointed at" and "shipped with".
    try tmp.dir.createDirPath(io, IN_DIR_SUBDIR);
    var sub = try tmp.dir.openDir(io, IN_DIR_SUBDIR, .{});
    defer sub.close(io);
    {
        var f = try sub.createFile(io, "config.json", .{});
        defer f.close(io);
        var wbuf: [512]u8 = undefined;
        var w = f.writer(io, &wbuf);
        try w.interface.writeAll(
            \\{"model_type":"muse_glimmer_assistant","block_size":16,"mask_token_id":7,
            \\ "target_layer_ids":[1,3],"hidden_size":64,"num_hidden_layers":2,
            \\ "num_attention_heads":4,"head_dim":16,"intermediate_size":128}
        );
        try w.interface.flush();
    }
    const found = resolveInDirDrafter(io, allocator, model_dir) orelse
        return error.TestExpectedInDirDrafter;
    defer allocator.free(found);
    try testing.expect(std.mem.endsWith(u8, found, IN_DIR_SUBDIR));
    try testing.expect(probeIsDflash(io, allocator, found));

    // A relative or empty model dir is refused rather than joined blindly —
    // `openDirAbsolute` on a non-absolute path is ReleaseFast UB.
    try testing.expectEqual(@as(?[]u8, null), resolveInDirDrafter(io, allocator, ""));
    try testing.expectEqual(@as(?[]u8, null), resolveInDirDrafter(io, allocator, "relative/dir"));
}

test "dflash: dflash2 folder is preferred with valid legacy fallback" {
    const a = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &path_buf);
    const model_dir = path_buf[0..root_len];
    const config = "{\"architectures\":[\"DFlash2DraftModel\"],\"dflash_config\":{\"block_size\":8,\"mask_token_id\":7,\"target_layer_ids\":[1,3]}}";
    for ([_][]const u8{ IN_DIR_SUBDIR, DFLASH2_IN_DIR_SUBDIR }) |name| {
        try tmp.dir.createDirPath(io, name);
        var sub = try tmp.dir.openDir(io, name, .{});
        defer sub.close(io);
        try sub.writeFile(io, .{ .sub_path = "config.json", .data = config });
    }
    const preferred = resolveInDirDrafter(io, a, model_dir) orelse return error.TestExpectedInDirDrafter;
    defer a.free(preferred);
    try testing.expect(std.mem.endsWith(u8, preferred, "/dflash2"));
    var dflash2 = try tmp.dir.openDir(io, DFLASH2_IN_DIR_SUBDIR, .{});
    defer dflash2.close(io);
    try dflash2.writeFile(io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"other\"}" });
    const fallback = resolveInDirDrafter(io, a, model_dir) orelse return error.TestExpectedInDirDrafter;
    defer a.free(fallback);
    try testing.expect(std.mem.endsWith(u8, fallback, "/drafter"));
}

test "dflash: no wide verify lane caps the block at the split-K width" {
    // Without an M 8..16 verify lane the trunk forward falls off a cliff at
    // width 8, so the block is capped even though the checkpoint asks for 16.
    try testing.expectEqual(NO_WIDE_LANE_BLOCK_CAP, resolveBlockSize(16, false, NO_WIDE_LANE_BLOCK_CAP));
    // The cap is a CEILING, never a floor: a checkpoint whose own block is
    // already narrow is left alone.
    try testing.expectEqual(@as(u32, 4), resolveBlockSize(4, false, NO_WIDE_LANE_BLOCK_CAP));
}

test "dflash: per-silicon cap table — M3 Ultra rides oMLX's block-8 evidence" {
    // The cap is a MACHINE measurement, keyed on the CPU brand string (the
    // GPU arch cannot tell Ultra from Max). M3 Ultra -> 8 (oMLX PR #2850:
    // 1.33-1.43x at block 8 on the same pairing); everything else without a
    // wide lane keeps the M4-measured default.
    const ultra = blockCapForMachine("Apple M3 Ultra");
    try testing.expectEqual(@as(u32, 8), ultra.cap);
    try testing.expectEqualStrings("m3-ultra", ultra.label);
    try testing.expectEqual(NO_WIDE_LANE_BLOCK_CAP, blockCapForMachine("Apple M4 Max").cap);
    try testing.expectEqual(NO_WIDE_LANE_BLOCK_CAP, blockCapForMachine("Apple M3 Max").cap);
    try testing.expectEqual(NO_WIDE_LANE_BLOCK_CAP, blockCapForMachine("").cap);
    // Resolution with the M3 Ultra row: a block-16 checkpoint caps at 8, a
    // block-8 one is left alone.
    try testing.expectEqual(@as(u32, 8), resolveBlockSize(16, false, ultra.cap));
    try testing.expectEqual(@as(u32, 8), resolveBlockSize(8, false, ultra.cap));
}

// ── Hermetic fixtures: tiny llama trunk + tiny DFlash assistant ──
// pub: generate.zig's nextDflash equivalence test builds the same pair.

pub const TinyFix = struct {
    // Dims are small but affine-quantizable: every contraction dim is a
    // multiple of 32, so the load-time quantization and the draft-only head
    // are exercisable hermetically (MLX affine groups are 32/64/128).
    pub const HIDDEN: usize = 64;
    pub const VOCAB: usize = 128;
    pub const HEAD_DIM: usize = 16;
    pub const N_HEADS: usize = 4;
    pub const N_KV: usize = 2;
    pub const INTER: usize = 128;
    pub const TRUNK_LAYERS: usize = 4;
    pub const ASSISTANT_LAYERS: usize = 2;
    pub const N_TARGETS: usize = 2;
    pub const BLOCK: usize = 4;

    /// Deterministic small weight values; `seed` de-correlates tensors.
    pub fn val(i: usize, seed: usize) f32 {
        return @as(f32, @floatFromInt((i * 7 + seed * 13) % 23)) * 0.02 - 0.2;
    }

    pub fn bf16Arr(rows: usize, cols: usize, seed: usize, s: mlx.mlx_stream) !mlx.mlx_array {
        const total = rows * @max(cols, 1);
        const data = try testing.allocator.alloc(f32, total);
        defer testing.allocator.free(data);
        for (data, 0..) |*x, i| x.* = val(i, seed);
        const shape2 = [_]c_int{ @intCast(rows), @intCast(cols) };
        const shape1 = [_]c_int{@intCast(rows)};
        const f32_arr = if (cols > 0)
            mlx.mlx_array_new_data(data.ptr, &shape2, 2, .float32)
        else
            mlx.mlx_array_new_data(data.ptr, &shape1, 1, .float32);
        defer _ = mlx.mlx_array_free(f32_arr);
        var bf = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(bf);
        try mlx.check(mlx.mlx_astype(&bf, f32_arr, .bfloat16, s));
        try mlx.check(mlx.mlx_array_eval(bf));
        return bf;
    }

    /// Norm weights near 1.0 so norms stay neutral-ish.
    pub fn normArr(n: usize, s: mlx.mlx_stream) !mlx.mlx_array {
        const data = try testing.allocator.alloc(f32, n);
        defer testing.allocator.free(data);
        for (data, 0..) |*x, i| x.* = 1.0 + val(i, 5) * 0.1;
        const shape = [_]c_int{@intCast(n)};
        const f32_arr = mlx.mlx_array_new_data(data.ptr, &shape, 1, .float32);
        defer _ = mlx.mlx_array_free(f32_arr);
        var bf = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(bf);
        try mlx.check(mlx.mlx_astype(&bf, f32_arr, .bfloat16, s));
        try mlx.check(mlx.mlx_array_eval(bf));
        return bf;
    }

    pub fn put(map: mlx.mlx_map_string_to_array, key: []const u8, arr: mlx.mlx_array) !void {
        const key_z = try std.fmt.allocPrintSentinel(testing.allocator, "{s}", .{key}, 0);
        defer testing.allocator.free(key_z);
        _ = mlx.mlx_map_string_to_array_insert(map, key_z.ptr, arr);
    }

    pub fn putW(map: mlx.mlx_map_string_to_array, key: []const u8, rows: usize, cols: usize, seed: usize, s: mlx.mlx_stream) !void {
        const arr = try bf16Arr(rows, cols, seed, s);
        defer _ = mlx.mlx_array_free(arr);
        try put(map, key, arr);
    }

    pub fn putNorm(map: mlx.mlx_map_string_to_array, key: []const u8, n: usize, s: mlx.mlx_stream) !void {
        const arr = try normArr(n, s);
        defer _ = mlx.mlx_array_free(arr);
        try put(map, key, arr);
    }

    // Tiny llama trunk: vocab 128, hidden 64, 4 layers, 4 q / 2 kv heads,
    // hd 16, inter 128, tied embeddings.
    // The trunk SLIDES at 8 on purpose: the greedy-equivalence test runs a
    // 12-token prompt out to 16 tokens, so every decode step and every verify
    // block sits past the window. That makes serial (width 1, view trimmed to
    // the window) vs dflash (width 4, trimmed to window + 3) the pairing guard
    // for `slidingTailSpan` and the relative mask offsets — a wrong trim or a
    // mask built against the untrimmed length diverges the two streams.
    pub const TRUNK_CONFIG =
        \\{
        \\  "model_type": "llama",
        \\  "hidden_size": 64,
        \\  "intermediate_size": 128,
        \\  "num_hidden_layers": 4,
        \\  "num_attention_heads": 4,
        \\  "num_key_value_heads": 2,
        \\  "head_dim": 16,
        \\  "vocab_size": 128,
        \\  "rms_norm_eps": 1e-5,
        \\  "rope_theta": 10000.0,
        \\  "tie_word_embeddings": true,
        \\  "sliding_window": 8,
        \\  "max_position_embeddings": 2048,
        \\  "torch_dtype": "bfloat16"
        \\}
    ;

    pub fn writeTrunk(io: std.Io, dir: std.Io.Dir, dir_path: []const u8, s: mlx.mlx_stream) !void {
        try dir.writeFile(io, .{ .sub_path = "config.json", .data = TRUNK_CONFIG });
        const st_path = try std.fmt.allocPrintSentinel(testing.allocator, "{s}/model.safetensors", .{dir_path}, 0);
        defer testing.allocator.free(st_path);
        const map = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(map);
        const meta = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(meta);

        const q_out = N_HEADS * HEAD_DIM;
        const kv_out = N_KV * HEAD_DIM;
        try putW(map, "model.embed_tokens.weight", VOCAB, HIDDEN, 1, s);
        try putNorm(map, "model.norm.weight", HIDDEN, s);
        var key_buf: [128]u8 = undefined;
        var li: usize = 0;
        while (li < TRUNK_LAYERS) : (li += 1) {
            try putNorm(map, try std.fmt.bufPrint(&key_buf, "model.layers.{d}.input_layernorm.weight", .{li}), HIDDEN, s);
            try putNorm(map, try std.fmt.bufPrint(&key_buf, "model.layers.{d}.post_attention_layernorm.weight", .{li}), HIDDEN, s);
            try putW(map, try std.fmt.bufPrint(&key_buf, "model.layers.{d}.self_attn.q_proj.weight", .{li}), q_out, HIDDEN, 10 + li, s);
            try putW(map, try std.fmt.bufPrint(&key_buf, "model.layers.{d}.self_attn.k_proj.weight", .{li}), kv_out, HIDDEN, 20 + li, s);
            try putW(map, try std.fmt.bufPrint(&key_buf, "model.layers.{d}.self_attn.v_proj.weight", .{li}), kv_out, HIDDEN, 30 + li, s);
            try putW(map, try std.fmt.bufPrint(&key_buf, "model.layers.{d}.self_attn.o_proj.weight", .{li}), HIDDEN, q_out, 40 + li, s);
            try putW(map, try std.fmt.bufPrint(&key_buf, "model.layers.{d}.mlp.gate_proj.weight", .{li}), INTER, HIDDEN, 50 + li, s);
            try putW(map, try std.fmt.bufPrint(&key_buf, "model.layers.{d}.mlp.up_proj.weight", .{li}), INTER, HIDDEN, 60 + li, s);
            try putW(map, try std.fmt.bufPrint(&key_buf, "model.layers.{d}.mlp.down_proj.weight", .{li}), HIDDEN, INTER, 70 + li, s);
        }
        try mlx.check(mlx.mlx_save_safetensors(st_path.ptr, map, meta));
    }

    // Tiny assistant: hidden 64 (== trunk), 2 layers (sliding + full),
    // 4 q / 2 kv heads, hd 16, inter 128, window 8, block 4, mask id 127,
    // targets [0, 2].
    pub const ASSISTANT_CONFIG =
        \\{
        \\  "model_type": "tiny_assistant",
        \\  "block_size": 4,
        \\  "mask_token_id": 127,
        \\  "target_layer_ids": [0, 2],
        \\  "hidden_size": 64,
        \\  "intermediate_size": 128,
        \\  "num_hidden_layers": 2,
        \\  "num_attention_heads": 4,
        \\  "num_key_value_heads": 2,
        \\  "head_dim": 16,
        \\  "rms_norm_eps": 1e-5,
        \\  "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
        \\  "sliding_window": 8,
        \\  "layer_types": ["sliding_attention", "full_attention"]
        \\}
    ;

    /// `quantize`: also emit `.scales`/`.biases` for every matmul weight, the
    /// shape a sidecar published pre-quantized would ship.
    pub fn writeAssistant(io: std.Io, dir: std.Io.Dir, dir_path: []const u8, s: mlx.mlx_stream) !void {
        return writeAssistantOpts(io, dir, dir_path, s, false);
    }

    pub fn writeAssistantOpts(io: std.Io, dir: std.Io.Dir, dir_path: []const u8, s: mlx.mlx_stream, quantize: bool) !void {
        try dir.writeFile(io, .{ .sub_path = "config.json", .data = ASSISTANT_CONFIG });
        const st_path = try std.fmt.allocPrintSentinel(testing.allocator, "{s}/model.safetensors", .{dir_path}, 0);
        defer testing.allocator.free(st_path);
        const map = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(map);
        const meta = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(meta);
        try putV1AssistantWeights(map, s, quantize);
        try mlx.check(mlx.mlx_save_safetensors(st_path.ptr, map, meta));
    }

    fn putV1AssistantWeights(map: mlx.mlx_map_string_to_array, s: mlx.mlx_stream, quantize: bool) !void {
        const q_out = N_HEADS * HEAD_DIM;
        const kv_out = N_KV * HEAD_DIM;
        const putLin = struct {
            fn f(m: mlx.mlx_map_string_to_array, key: []const u8, rows: usize, cols: usize, seed: usize, st: mlx.mlx_stream, q: bool) !void {
                if (q) return putQuantW(m, key, rows, cols, seed, st);
                return putW(m, key, rows, cols, seed, st);
            }
        }.f;

        try putLin(map, "encoder.fc.weight", HIDDEN, N_TARGETS * HIDDEN, 100, s, quantize);
        try putNorm(map, "encoder.output_norm_enc.weight", HIDDEN, s);
        try putNorm(map, "norm.weight", HIDDEN, s);
        var key_buf: [128]u8 = undefined;
        var li: usize = 0;
        while (li < ASSISTANT_LAYERS) : (li += 1) {
            try putNorm(map, try std.fmt.bufPrint(&key_buf, "layers.{d}.input_layernorm.weight", .{li}), HIDDEN, s);
            try putNorm(map, try std.fmt.bufPrint(&key_buf, "layers.{d}.post_attention_layernorm.weight", .{li}), HIDDEN, s);
            try putLin(map, try std.fmt.bufPrint(&key_buf, "layers.{d}.self_attn.q_proj.weight", .{li}), q_out, HIDDEN, 110 + li, s, quantize);
            try putNorm(map, try std.fmt.bufPrint(&key_buf, "layers.{d}.self_attn.q_norm.weight", .{li}), HEAD_DIM, s);
            try putLin(map, try std.fmt.bufPrint(&key_buf, "layers.{d}.self_attn.k_proj.weight", .{li}), kv_out, HIDDEN, 120 + li, s, quantize);
            try putNorm(map, try std.fmt.bufPrint(&key_buf, "layers.{d}.self_attn.k_norm.weight", .{li}), HEAD_DIM, s);
            try putLin(map, try std.fmt.bufPrint(&key_buf, "layers.{d}.self_attn.v_proj.weight", .{li}), kv_out, HIDDEN, 130 + li, s, quantize);
            try putLin(map, try std.fmt.bufPrint(&key_buf, "layers.{d}.self_attn.o_proj.weight", .{li}), HIDDEN, q_out, 140 + li, s, quantize);
            try putLin(map, try std.fmt.bufPrint(&key_buf, "layers.{d}.mlp.gate_proj.weight", .{li}), INTER, HIDDEN, 150 + li, s, quantize);
            try putLin(map, try std.fmt.bufPrint(&key_buf, "layers.{d}.mlp.up_proj.weight", .{li}), INTER, HIDDEN, 160 + li, s, quantize);
            try putLin(map, try std.fmt.bufPrint(&key_buf, "layers.{d}.mlp.down_proj.weight", .{li}), HIDDEN, INTER, 170 + li, s, quantize);
        }
    }

    // DFlash2 tiny fixture: v1 dims + nested contract, selector rank 16 /
    // top_k 4 and 2-tap convs at group 8 (hidden 64 → 8 groups). model_type
    // is a bare "qwen3" — exactly the real checkpoint's shape.
    pub const SEL_RANK: usize = 16;
    pub const SEL_TOPK: usize = 4;
    pub const CONV_K: usize = 2;
    pub const CONV_GS: usize = 8;

    pub const ASSISTANT2_CONFIG =
        \\{
        \\  "architectures": ["DFlash2DraftModel"],
        \\  "model_type": "qwen3",
        \\  "is_causal": false,
        \\  "dflash_config": {
        \\    "block_size": 4,
        \\    "mask_token_id": 127,
        \\    "target_layer_ids": [0, 2],
        \\    "conv_kernel_size": 2,
        \\    "conv_group_size": 8,
        \\    "selector_rank": 16,
        \\    "selector_top_k": 4
        \\  },
        \\  "hidden_size": 64,
        \\  "intermediate_size": 128,
        \\  "num_hidden_layers": 2,
        \\  "num_attention_heads": 4,
        \\  "num_key_value_heads": 2,
        \\  "head_dim": 16,
        \\  "rms_norm_eps": 1e-5,
        \\  "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
        \\  "sliding_window": 8,
        \\  "vocab_size": 128,
        \\  "layer_types": ["sliding_attention", "full_attention"]
        \\}
    ;

    /// bf16 array with an arbitrary shape (base_kernel is 3-D).
    pub fn bf16ArrShaped(shape: []const c_int, seed: usize, s: mlx.mlx_stream) !mlx.mlx_array {
        var total: usize = 1;
        for (shape) |d| total *= @intCast(d);
        const data = try testing.allocator.alloc(f32, total);
        defer testing.allocator.free(data);
        for (data, 0..) |*x, i| x.* = val(i, seed);
        const f32_arr = mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), .float32);
        defer _ = mlx.mlx_array_free(f32_arr);
        var bf = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(bf);
        try mlx.check(mlx.mlx_astype(&bf, f32_arr, .bfloat16, s));
        try mlx.check(mlx.mlx_array_eval(bf));
        return bf;
    }

    pub fn writeAssistant2(io: std.Io, dir: std.Io.Dir, dir_path: []const u8, s: mlx.mlx_stream) !void {
        try dir.writeFile(io, .{ .sub_path = "config.json", .data = ASSISTANT2_CONFIG });
        const st_path = try std.fmt.allocPrintSentinel(testing.allocator, "{s}/model.safetensors", .{dir_path}, 0);
        defer testing.allocator.free(st_path);
        const map = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(map);
        const meta = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(meta);
        try putV1AssistantWeights(map, s, false);

        // Selector: codebooks ship SUFFIX-LESS, like the real checkpoint.
        try putW(map, "candidate_selector.predecessor_codebook", VOCAB, SEL_RANK, 300, s);
        try putW(map, "candidate_selector.successor_codebook", VOCAB, SEL_RANK, 301, s);
        try putW(map, "candidate_selector.hidden_projection.weight", SEL_RANK, HIDDEN, 302, s);

        // Dynamic convs per layer: base [2, ksize, H] + projection
        // [2*ksize*groups, H].
        const groups = HIDDEN / CONV_GS;
        var key_buf: [128]u8 = undefined;
        var li: usize = 0;
        while (li < ASSISTANT_LAYERS) : (li += 1) {
            inline for (.{ "attention_conv", "mlp_conv" }) |which| {
                const base_shape = [_]c_int{ 2, CONV_K, HIDDEN };
                const base = try bf16ArrShaped(&base_shape, 310 + li * 7 + which.len, s);
                defer _ = mlx.mlx_array_free(base);
                try put(map, try std.fmt.bufPrint(&key_buf, "layers.{d}." ++ which ++ ".base_kernel", .{li}), base);
                try putW(map, try std.fmt.bufPrint(&key_buf, "layers.{d}." ++ which ++ ".kernel_projection.weight", .{li}), 2 * CONV_K * groups, HIDDEN, 330 + li * 7 + which.len, s);
            }
        }
        try mlx.check(mlx.mlx_save_safetensors(st_path.ptr, map, meta));
    }

    /// A `<prefix>.weight` written PACKED, with its `.scales`/`.biases`
    /// siblings — the pre-quantized-sidecar shape.
    pub fn putQuantW(map: mlx.mlx_map_string_to_array, key: []const u8, rows: usize, cols: usize, seed: usize, s: mlx.mlx_stream) !void {
        const dense = try bf16Arr(rows, cols, seed, s);
        defer _ = mlx.mlx_array_free(dense);
        var lin = try quantizeDense(dense, 8, @intCast(@min(cols, 64)), s);
        defer lin.deinit();
        try put(map, key, lin.w);
        const base = key[0 .. key.len - ".weight".len];
        var buf: [160]u8 = undefined;
        try put(map, try std.fmt.bufPrint(&buf, "{s}.scales", .{base}), lin.scales);
        try put(map, try std.fmt.bufPrint(&buf, "{s}.biases", .{base}), lin.biases);
    }

    pub fn readF32(arr: mlx.mlx_array, allocator: std.mem.Allocator, s: mlx.mlx_stream) ![]f32 {
        var f = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(f);
        try mlx.check(mlx.mlx_astype(&f, arr, .float32, s));
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        const sh = mlx.getShape(f);
        var total: c_int = 1;
        for (sh) |d| total *= d;
        const flat_shape = [_]c_int{total};
        try mlx.check(mlx.mlx_reshape(&flat, f, &flat_shape, 1, s));
        try mlx.check(mlx.mlx_array_eval(flat));
        const ptr = mlx.mlx_array_data_float32(flat) orelse return error.MlxArrayDataNull;
        const out = try allocator.alloc(f32, @intCast(total));
        @memcpy(out, ptr[0..@intCast(total)]);
        return out;
    }

    /// Synthetic trunk-capture array `[1, n, HIDDEN]`.
    pub fn capArr(n: usize, seed: usize, s: mlx.mlx_stream) !mlx.mlx_array {
        const data = try testing.allocator.alloc(f32, n * HIDDEN);
        defer testing.allocator.free(data);
        for (data, 0..) |*x, i| x.* = val(i, seed);
        const shape = [_]c_int{ 1, @intCast(n), HIDDEN };
        const f32_arr = mlx.mlx_array_new_data(data.ptr, &shape, 3, .float32);
        defer _ = mlx.mlx_array_free(f32_arr);
        var bf = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(bf);
        try mlx.check(mlx.mlx_astype(&bf, f32_arr, .bfloat16, s));
        try mlx.check(mlx.mlx_array_eval(bf));
        return bf;
    }
};

test "dflash: loadDflash keeps a dense bf16 assistant dense + pre-transposed when quantization is off" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();
    var path_buf: [512]u8 = undefined;
    const root_len = try tmp_dir.dir.realPath(io, &path_buf);
    const dir_path = path_buf[0..root_len];
    try TinyFix.writeAssistant(io, tmp_dir.dir, dir_path, s);

    var m = try loadDflashQuant(io, allocator, s, dir_path, 0);
    defer m.deinit();

    try testing.expectEqual(@as(u32, 4), m.config.block_size);
    try testing.expectEqual(TinyFix.ASSISTANT_LAYERS, m.layers.len);
    // fc pre-transposed: checkpoint [H, NT*H] → [n_targets*hidden, hidden].
    try testing.expect(!m.fc.isQuantized());
    const fc_shape = mlx.getShape(m.fc.w);
    try testing.expectEqual(@as(c_int, TinyFix.N_TARGETS * TinyFix.HIDDEN), fc_shape[0]);
    try testing.expectEqual(@as(c_int, TinyFix.HIDDEN), fc_shape[1]);
    try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(m.fc.w));
    // k pre-transposed: [kv_out, H] → [H, kv_out].
    const k_shape = mlx.getShape(m.layers[0].k.w);
    try testing.expectEqual(@as(c_int, TinyFix.HIDDEN), k_shape[0]);
    try testing.expectEqual(@as(c_int, TinyFix.N_KV * TinyFix.HEAD_DIM), k_shape[1]);
    try testing.expectEqual(LayerType.sliding_attention, m.layers[0].layer_type);
    try testing.expectEqual(LayerType.full_attention, m.layers[1].layer_type);
}

test "dflash: buildBlockMask sliding/full arms and window edges" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();

    // Full layer: never a mask.
    try testing.expectEqual(@as(?mlx.mlx_array, null), try buildBlockMask(.full_attention, 0, 100, 100, 4, 8, s));
    // Sliding, everything inside the window: no mask needed.
    try testing.expectEqual(@as(?mlx.mlx_array, null), try buildBlockMask(.sliding_attention, 0, 4, 4, 4, 8, s));

    // Sliding, ctx_len 10, anchor 10, block 4, window 8: rows are queries at
    // abs 10..13, cols are ctx abs 0..9 then block abs 10..13.
    const mask = (try buildBlockMask(.sliding_attention, 0, 10, 10, 4, 8, s)).?;
    defer _ = mlx.mlx_array_free(mask);
    const msh = mlx.getShape(mask);
    try testing.expectEqualSlices(c_int, &[_]c_int{ 1, 1, 4, 14 }, msh);
    const vals = try TinyFix.readF32(mask, allocator, s);
    defer allocator.free(vals);
    const at = struct {
        fn f(v: []const f32, row: usize, col: usize) f32 {
            return v[row * 14 + col];
        }
    }.f;
    // q=10: k=0,1,2 masked (dist ≥ 8), k=3.. allowed.
    try testing.expect(std.math.isNegativeInf(at(vals, 0, 0)));
    try testing.expect(std.math.isNegativeInf(at(vals, 0, 2)));
    try testing.expectEqual(@as(f32, 0.0), at(vals, 0, 3));
    // q=13: k=5 masked (dist 8), k=6 allowed (dist 7).
    try testing.expect(std.math.isNegativeInf(at(vals, 3, 5)));
    try testing.expectEqual(@as(f32, 0.0), at(vals, 3, 6));
    // Block ↔ block always visible (|q-k| ≤ 3 < 8).
    try testing.expectEqual(@as(f32, 0.0), at(vals, 0, 13));
    try testing.expectEqual(@as(f32, 0.0), at(vals, 3, 10));
}

test "dflash: appendContext grows the cache; forwardBlock evicts its block K/V" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();
    var path_buf: [512]u8 = undefined;
    const root_len = try tmp_dir.dir.realPath(io, &path_buf);
    const dir_path = path_buf[0..root_len];
    try TinyFix.writeAssistant(io, tmp_dir.dir, dir_path, s);
    var m = try loadDflash(io, allocator, s, dir_path);
    defer m.deinit();

    var ctx = try DflashCtx.init(allocator, &m, 0);
    defer ctx.deinit();

    // Append 6 context tokens (2 capture streams, one per target layer id).
    {
        const c0 = try TinyFix.capArr(6, 200, s);
        defer _ = mlx.mlx_array_free(c0);
        const c1 = try TinyFix.capArr(6, 201, s);
        defer _ = mlx.mlx_array_free(c1);
        try appendContext(&m, &ctx, &[_]mlx.mlx_array{ c0, c1 }, 0);
    }
    try testing.expectEqual(@as(usize, 6), ctx.cache.step);
    try testing.expectEqual(@as(usize, 6), ctx.absLen());
    // Context K/V shape per layer: [1, N_KV, 6, HEAD_DIM].
    const ksh = mlx.getShape(ctx.cache.entries[0].key_view);
    try testing.expectEqual(@as(c_int, TinyFix.N_KV), ksh[1]);
    try testing.expectEqual(@as(c_int, TinyFix.HEAD_DIM), ksh[3]);

    // Block forward at anchor 6 → [1, BLOCK, HIDDEN]; cache unchanged after.
    const noise = try TinyFix.capArr(4, 300, s);
    defer _ = mlx.mlx_array_free(noise);
    const hidden = try forwardBlock(&m, &ctx, noise, 6);
    defer _ = mlx.mlx_array_free(hidden);
    try mlx.check(mlx.mlx_array_eval(hidden));
    const hsh = mlx.getShape(hidden);
    try testing.expectEqualSlices(c_int, &[_]c_int{ 1, TinyFix.BLOCK, TinyFix.HIDDEN }, hsh);
    try testing.expectEqual(@as(usize, 6), ctx.cache.step); // block evicted

    // Continue appending (next round's accepted tokens).
    {
        const c0 = try TinyFix.capArr(3, 210, s);
        defer _ = mlx.mlx_array_free(c0);
        const c1 = try TinyFix.capArr(3, 211, s);
        defer _ = mlx.mlx_array_free(c1);
        try appendContext(&m, &ctx, &[_]mlx.mlx_array{ c0, c1 }, 6);
    }
    try testing.expectEqual(@as(usize, 9), ctx.cache.step);
}

/// Run the full context-append + block forward on `m` and return the block
/// hidden as f32 — the comparison surface for weight-precision arms.
fn tinyBlockHidden(m: *DflashModel, allocator: std.mem.Allocator, s: mlx.mlx_stream) ![]f32 {
    var ctx = try DflashCtx.init(allocator, m, 0);
    defer ctx.deinit();
    const c0 = try TinyFix.capArr(10, 600, s);
    defer _ = mlx.mlx_array_free(c0);
    const c1 = try TinyFix.capArr(10, 601, s);
    defer _ = mlx.mlx_array_free(c1);
    try appendContext(m, &ctx, &[_]mlx.mlx_array{ c0, c1 }, 0);
    const noise = try TinyFix.capArr(TinyFix.BLOCK, 700, s);
    defer _ = mlx.mlx_array_free(noise);
    const hidden = try forwardBlock(m, &ctx, noise, 10);
    defer _ = mlx.mlx_array_free(hidden);
    return TinyFix.readF32(hidden, allocator, s);
}

test "dflash: load-time quantization packs every matmul weight and tracks the dense forward" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();
    if (mlx.noGpuBackend()) return;

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();
    var path_buf: [512]u8 = undefined;
    const root_len = try tmp_dir.dir.realPath(io, &path_buf);
    const dir_path = path_buf[0..root_len];
    try TinyFix.writeAssistant(io, tmp_dir.dir, dir_path, s);

    var dense = try loadDflashQuant(io, allocator, s, dir_path, 0);
    defer dense.deinit();
    var quant = try loadDflashQuant(io, allocator, s, dir_path, 8);
    defer quant.deinit();

    // Every contracted weight is packed at the requested width, and the
    // packing is the CHECKPOINT's [out, in] layout (qmm transpose=true) —
    // a pre-transposed packed weight would contract the wrong axis.
    try testing.expect(quant.fc.isQuantized());
    try testing.expectEqual(@as(u32, 8), quant.fc.bits);
    try testing.expectEqual(QUANT_GROUP, quant.fc.group_size);
    try testing.expectEqual(@as(c_int, TinyFix.HIDDEN), mlx.getShape(quant.fc.w)[0]);
    for (quant.layers) |*lw| {
        for ([_]*const DflashLinear{ &lw.q, &lw.k, &lw.v, &lw.o, &lw.gate, &lw.up, &lw.down }) |lin| {
            try testing.expect(lin.isQuantized());
            try testing.expectEqual(mlx.mlx_dtype.uint32, mlx.mlx_array_dtype(lin.w));
        }
    }

    const a = try tinyBlockHidden(&dense, allocator, s);
    defer allocator.free(a);
    const b = try tinyBlockHidden(&quant, allocator, s);
    defer allocator.free(b);
    const p = try paritySlices(a, b);
    try testing.expect(p.cos > 0.99);
    try testing.expect(@abs(p.rms_ratio - 1.0) < 0.05);
}

test "dflash: a sidecar shipping packed weights loads at its own declared width" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();
    if (mlx.noGpuBackend()) return;

    var tmp_dense = std.testing.tmpDir(.{});
    defer tmp_dense.cleanup();
    var dense_buf: [512]u8 = undefined;
    const dense_path = dense_buf[0..try tmp_dense.dir.realPath(io, &dense_buf)];
    try TinyFix.writeAssistant(io, tmp_dense.dir, dense_path, s);

    var tmp_packed = std.testing.tmpDir(.{});
    defer tmp_packed.cleanup();
    var packed_buf: [512]u8 = undefined;
    const packed_path = packed_buf[0..try tmp_packed.dir.realPath(io, &packed_buf)];
    try TinyFix.writeAssistantOpts(io, tmp_packed.dir, packed_path, s, true);

    var dense = try loadDflashQuant(io, allocator, s, dense_path, 0);
    defer dense.deinit();
    // The request for dense bf16 does NOT unpack a packed checkpoint — the
    // shipped weights are served as they are, at the width the geometry says.
    var shipped = try loadDflashQuant(io, allocator, s, packed_path, 0);
    defer shipped.deinit();

    try testing.expect(shipped.fc.isQuantized());
    try testing.expectEqual(@as(u32, 8), shipped.fc.bits);
    try testing.expectEqual(@as(u32, 64), shipped.layers[0].down.group_size);

    const a = try tinyBlockHidden(&dense, allocator, s);
    defer allocator.free(a);
    const b = try tinyBlockHidden(&shipped, allocator, s);
    defer allocator.free(b);
    const p = try paritySlices(a, b);
    try testing.expect(p.cos > 0.99);
    try testing.expect(@abs(p.rms_ratio - 1.0) < 0.05);
}

test "dflash2: loader picks up selector + dyn convs; v1 assistant loads with neither" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [512]u8 = undefined;
    const dir_path = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    try TinyFix.writeAssistant2(io, tmp.dir, dir_path, s);

    var m = try loadDflashQuant(io, allocator, s, dir_path, 0);
    defer m.deinit();

    try testing.expect(m.config.isDflash2());
    try testing.expectEqual(@as(u32, TinyFix.SEL_RANK), m.config.selector_rank);
    try testing.expectEqual(@as(u32, TinyFix.SEL_TOPK), m.config.selector_top_k);
    const sel = m.selector orelse return error.TestExpectedSelector;
    // Codebooks are gather tables: bf16, NEVER quantized, [vocab, rank].
    try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(sel.pred_codebook));
    try testing.expectEqualSlices(c_int, &[_]c_int{ TinyFix.VOCAB, TinyFix.SEL_RANK }, mlx.getShape(sel.succ_codebook));
    for (m.layers) |*lw| {
        const ac = lw.attention_conv orelse return error.TestExpectedConv;
        const mc = lw.mlp_conv orelse return error.TestExpectedConv;
        try testing.expectEqualSlices(c_int, &[_]c_int{ 2, TinyFix.CONV_K, TinyFix.HIDDEN }, mlx.getShape(ac.base_kernel));
        try testing.expectEqualSlices(c_int, &[_]c_int{ 2, TinyFix.CONV_K, TinyFix.HIDDEN }, mlx.getShape(mc.base_kernel));
    }

    // A v1 assistant never even looks for the DFlash2 weights.
    var tmp_v1 = std.testing.tmpDir(.{});
    defer tmp_v1.cleanup();
    var v1_buf: [512]u8 = undefined;
    const v1_path = v1_buf[0..try tmp_v1.dir.realPath(io, &v1_buf)];
    try TinyFix.writeAssistant(io, tmp_v1.dir, v1_path, s);
    var v1 = try loadDflashQuant(io, allocator, s, v1_path, 0);
    defer v1.deinit();
    try testing.expect(v1.selector == null);
    try testing.expect(v1.layers[0].attention_conv == null);
}

test "dflash2: groupedDynConv matches the closed form on a hand-computed case" {
    if (mlx.noGpuBackend()) return;
    const allocator = testing.allocator;
    const s = mlx.gpuStream();

    // L=2 positions, H=4 channels, group_size=2 → 2 groups, ksize=2 taps.
    const x_data = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const x_shape = [_]c_int{ 1, 2, 4 };
    const x = mlx.mlx_array_new_data(&x_data, &x_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(x);

    // base[tap][channel]
    const base_data = [_]f32{ 0.5, 0.5, 1.0, 1.0, 0.25, 0.25, 0.5, 0.5 };
    const base_shape = [_]c_int{ 2, 4 };
    const base = mlx.mlx_array_new_data(&base_data, &base_shape, 2, .float32);
    defer _ = mlx.mlx_array_free(base);

    // dynamic[pos][tap][group]
    const dyn_data = [_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 };
    const dyn_shape = [_]c_int{ 1, 2, 2, 2 };
    const dyn = mlx.mlx_array_new_data(&dyn_data, &dyn_shape, 4, .float32);
    defer _ = mlx.mlx_array_free(dyn);

    const out = try groupedDynConv(x, dyn, base, 2, s);
    defer _ = mlx.mlx_array_free(out);
    const got = try TinyFix.readF32(out, allocator, s);
    defer allocator.free(got);

    // out[t][c] = (base[0][c] + dyn[t][0][c/2]) * x[t][c]
    //           + (base[1][c] + dyn[t][1][c/2]) * x[t-1][c], x[-1] == 0.
    var want: [8]f32 = undefined;
    for (0..2) |t| for (0..4) |c| {
        const g = c / 2;
        const tap0 = (base_data[c] + dyn_data[t * 4 + g]) * x_data[t * 4 + c];
        const prev: f32 = if (t == 0) 0 else x_data[(t - 1) * 4 + c];
        const tap1 = (base_data[4 + c] + dyn_data[t * 4 + 2 + g]) * prev;
        want[t * 4 + c] = tap0 + tap1;
    };
    for (got, want) |a, b| try testing.expect(@abs(a - b) < 1e-5);
}

test "dflash2: convPrepare taps base_kernel[0], convFinish taps [1], kernels from the INPUT" {
    if (mlx.noGpuBackend()) return;
    const allocator = testing.allocator;
    const s = mlx.gpuStream();

    // H=4, group_size=2, ksize=2. Zero kernel_projection isolates the base
    // halves: prepare must convolve with base[0] only, finish with base[1]
    // only — with both leading dims equal to 2 a transposed reshape is
    // silent, so the halves get DISTINCT values.
    const H: usize = 4;
    const groups: usize = 2;
    const base_data = [_]f32{
        // half 0 (prepare): tap0, tap1
        2, 2, 2, 2, 0, 0, 0, 0,
        // half 1 (finish): tap0, tap1
        3, 3, 3, 3, 1, 1, 1, 1,
    };
    const base_shape = [_]c_int{ 2, 2, @intCast(H) };
    const base = mlx.mlx_array_new_data(&base_data, &base_shape, 3, .float32);
    var zeros_w = mlx.mlx_array_new();
    const zw_shape = [_]c_int{ @intCast(H), @intCast(2 * 2 * groups) }; // dense pre-transposed [in, out]
    try mlx.check(mlx.mlx_zeros(&zeros_w, &zw_shape, 2, .float32, s));
    var conv = DynConv{
        .base_kernel = base,
        .kernel_projection = .{ .w = zeros_w, .scales = mlx.mlx_array_new(), .biases = mlx.mlx_array_new() },
    };
    defer conv.deinit();

    const x_data = [_]f32{ 1, 1, 1, 1, 2, 2, 2, 2 };
    const x_shape = [_]c_int{ 1, 2, @intCast(H) };
    const x = mlx.mlx_array_new_data(&x_data, &x_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(x);

    const cp = try convPrepare(&conv, x, 2, 2, s);
    defer {
        _ = mlx.mlx_array_free(cp.hidden);
        _ = mlx.mlx_array_free(cp.finish_dyn);
    }
    const prep = try TinyFix.readF32(cp.hidden, allocator, s);
    defer allocator.free(prep);
    // prepare: out[t] = 2*x[t] + 0*x[t-1] → [2,2,2,2, 4,4,4,4]
    for (prep[0..4]) |v| try testing.expectApproxEqAbs(@as(f32, 2), v, 1e-6);
    for (prep[4..8]) |v| try testing.expectApproxEqAbs(@as(f32, 4), v, 1e-6);

    const fin = try convFinish(&conv, x, cp.finish_dyn, 2, s);
    defer _ = mlx.mlx_array_free(fin);
    const finv = try TinyFix.readF32(fin, allocator, s);
    defer allocator.free(finv);
    // finish: out[t] = 3*x[t] + 1*x[t-1] → [3,3,3,3, 7,7,7,7]
    for (finv[0..4]) |v| try testing.expectApproxEqAbs(@as(f32, 3), v, 1e-6);
    for (finv[4..8]) |v| try testing.expectApproxEqAbs(@as(f32, 7), v, 1e-6);
}

test "dflash2: selectPath traces the pairwise-scored path, edges outvote unary logits" {
    if (mlx.noGpuBackend()) return;
    const allocator = testing.allocator;
    const s = mlx.gpuStream();

    // vocab 8, rank 2, k 2, m 2, H 2. Codebooks: pred[v] = [v, 1],
    // succ[v] = [1, v]; hidden_projection = identity.
    var pred_data: [16]f32 = undefined;
    var succ_data: [16]f32 = undefined;
    for (0..8) |v| {
        pred_data[v * 2] = @floatFromInt(v);
        pred_data[v * 2 + 1] = 1;
        succ_data[v * 2] = 1;
        succ_data[v * 2 + 1] = @floatFromInt(v);
    }
    const cb_shape = [_]c_int{ 8, 2 };
    const eye = [_]f32{ 1, 0, 0, 1 };
    const eye_shape = [_]c_int{ 2, 2 };
    var sel = Selector{
        .pred_codebook = mlx.mlx_array_new_data(&pred_data, &cb_shape, 2, .float32),
        .succ_codebook = mlx.mlx_array_new_data(&succ_data, &cb_shape, 2, .float32),
        .hidden_projection = .{ .w = mlx.mlx_array_new_data(&eye, &eye_shape, 2, .float32), .scales = mlx.mlx_array_new(), .biases = mlx.mlx_array_new() },
    };
    defer sel.deinit();

    // blk_hidden rows: anchor (ignored), H1=[1,0], H2=[0,1].
    const hid_data = [_]f32{ 9, 9, 1, 0, 0, 1 };
    const hid_shape = [_]c_int{ 1, 3, 2 };
    const hid = mlx.mlx_array_new_data(&hid_data, &hid_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(hid);

    // draft logits: pos 0 favors {3:10, 5:9}; pos 1 favors {1:5, 6:4}.
    var lg_data: [16]f32 = @splat(0);
    lg_data[3] = 10;
    lg_data[5] = 9;
    lg_data[8 + 1] = 5;
    lg_data[8 + 6] = 4;
    const lg_shape = [_]c_int{ 1, 2, 8 };
    const lg = mlx.mlx_array_new_data(&lg_data, &lg_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(lg);

    var prng = std.Random.DefaultPrng.init(7);
    var path = try selectPath(allocator, &sel, 2, hid, lg, 2, 0.0, prng.random(), s);
    defer path.deinit(allocator);

    // Pos 0: pred=anchor(2) → pred_row [2,1]; H1 ⊙ → [2,0]; edge = 2 for
    // both candidates → unary decides: token 3 (12 vs 11).
    try testing.expectEqual(@as(u32, 3), path.ids[0]);
    // Pos 1: pred=3 → [3,1]; H2 ⊙ → [0,1]; edge = candidate id. Scores:
    // token 1 → 5+1=6, token 6 → 4+6=10 — the EDGE flips the unary order.
    try testing.expectEqual(@as(u32, 6), path.ids[1]);
    try testing.expect(path.q == null);
    // The selector's confidence in each step is the chosen candidate's softmax share:
    // scores (12, 11) pick 12, scores (6, 10) pick 10.
    try testing.expectApproxEqAbs(@as(f32, 0.7311), path.conf[0], 1e-3);
    try testing.expectApproxEqAbs(@as(f32, 0.9820), path.conf[1], 1e-3);

    // Sampled arm: q is a proper distribution over each candidate row and
    // the drawn token's q is positive (the acceptance ratio's denominator).
    var path_s = try selectPath(allocator, &sel, 2, hid, lg, 2, 1.0, prng.random(), s);
    defer path_s.deinit(allocator);
    const q = path_s.q orelse return error.TestExpectedQ;
    for (0..2) |t| {
        var total: f32 = 0;
        for (q[t * 2 .. t * 2 + 2]) |qv| {
            try testing.expect(qv >= 0);
            total += qv;
        }
        try testing.expectApproxEqAbs(@as(f32, 1.0), total, 1e-5);
        try testing.expect(q[t * 2 + path_s.chosen_idx[t]] > 0);
    }
}

test "dflash: quantGroupFor picks the widest divisor, declines what affine cannot pack" {
    try testing.expectEqual(@as(?u32, 64), quantGroupFor(6656));
    try testing.expectEqual(@as(?u32, 64), quantGroupFor(19968));
    try testing.expectEqual(@as(?u32, 64), quantGroupFor(33280));
    try testing.expectEqual(@as(?u32, 32), quantGroupFor(96));
    try testing.expectEqual(@as(?u32, null), quantGroupFor(48));
    try testing.expectEqual(@as(?u32, null), quantGroupFor(0));
}

test "dflash: sliding window hides out-of-window context from the block, in-window reaches it" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();
    var path_buf: [512]u8 = undefined;
    const root_len = try tmp_dir.dir.realPath(io, &path_buf);
    const dir_path = path_buf[0..root_len];
    try TinyFix.writeAssistant(io, tmp_dir.dir, dir_path, s);
    var m = try loadDflash(io, allocator, s, dir_path);
    defer m.deinit();
    // Both layers sliding for this test — layer 1's full attention would
    // legitimately see the far row and mask nothing.
    m.layers[1].layer_type = .sliding_attention;

    // ctx_len 12, window 8, block 4 at anchor 12: queries at 12..15.
    // ctx row 0 (abs 0) is ≥ 8 away from every query → invisible.
    // ctx row 11 (abs 11) is ≤ 4 away → visible to all queries.
    const Run = struct {
        fn f(mm: *DflashModel, alloc: std.mem.Allocator, st: mlx.mlx_stream, perturb_row: ?usize) ![]f32 {
            var ctx = try DflashCtx.init(alloc, mm, 0);
            defer ctx.deinit();
            const c0 = try TinyFix.capArr(12, 400, st);
            defer _ = mlx.mlx_array_free(c0);
            var c1 = try TinyFix.capArr(12, 401, st);
            defer _ = mlx.mlx_array_free(c1);
            if (perturb_row) |row| {
                // Overwrite one context row with a big value.
                const big: [TinyFix.HIDDEN]f32 = @splat(100.0);
                const bshape = [_]c_int{ 1, 1, TinyFix.HIDDEN };
                const big_arr = mlx.mlx_array_new_data(&big, &bshape, 3, .float32);
                defer _ = mlx.mlx_array_free(big_arr);
                var big_bf = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(big_bf);
                try mlx.check(mlx.mlx_astype(&big_bf, big_arr, .bfloat16, st));
                const start = [_]c_int{ 0, @intCast(row), 0 };
                const stop = [_]c_int{ 1, @as(c_int, @intCast(row)) + 1, TinyFix.HIDDEN };
                const strides = [_]c_int{ 1, 1, 1 };
                var updated = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_slice_update(&updated, c1, big_bf, &start, 3, &stop, 3, &strides, 3, st));
                _ = mlx.mlx_array_free(c1);
                c1 = updated;
            }
            try appendContext(mm, &ctx, &[_]mlx.mlx_array{ c0, c1 }, 0);
            const noise = try TinyFix.capArr(4, 500, st);
            defer _ = mlx.mlx_array_free(noise);
            const hidden = try forwardBlock(mm, &ctx, noise, 12);
            defer _ = mlx.mlx_array_free(hidden);
            return TinyFix.readF32(hidden, alloc, st);
        }
    }.f;

    const base = try Run(&m, allocator, s, null);
    defer allocator.free(base);
    const far = try Run(&m, allocator, s, 0); // out-of-window row
    defer allocator.free(far);
    const near = try Run(&m, allocator, s, 11); // in-window row
    defer allocator.free(near);

    var far_diff: f32 = 0;
    var near_diff: f32 = 0;
    for (base, far) |b, f| far_diff = @max(far_diff, @abs(b - f));
    for (base, near) |b, n| near_diff = @max(near_diff, @abs(b - n));
    // The masked-out row must not leak into the block at all…
    try testing.expect(far_diff == 0.0);
    // …while an in-window row must move it.
    try testing.expect(near_diff > 0.001);
}

// ── Env-gated real-checkpoint parity vs the executable reference ──
// Fixtures: tests/dump_dflash_fixtures.py (transformers 5.15 CPU fp32).

fn fixtureF32Slice(allocator: std.mem.Allocator, v: std.json.Value) ![]f32 {
    const arr = v.array;
    const out = try allocator.alloc(f32, arr.items.len);
    for (arr.items, 0..) |elem, i| {
        out[i] = switch (elem) {
            .float => |f| @floatCast(f),
            .integer => |n| @floatFromInt(n),
            else => return error.BadFixture,
        };
    }
    return out;
}

/// [1, n, H] bf16 mlx array from a flat f32 slice.
fn fixtureArr(vals: []const f32, n: usize, h: usize, s: mlx.mlx_stream) !mlx.mlx_array {
    const shape = [_]c_int{ 1, @intCast(n), @intCast(h) };
    const f32_arr = mlx.mlx_array_new_data(vals.ptr, &shape, 3, .float32);
    defer _ = mlx.mlx_array_free(f32_arr);
    var bf = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&bf, f32_arr, .bfloat16, s));
    return bf;
}

/// Split a flat [1, n, NT*H] context stream into NT per-target [1, n, H]
/// capture arrays (feature-dim chunks, target order — the encoder's concat
/// reassembles them).
fn fixtureCaptures(allocator: std.mem.Allocator, flat: []const f32, n: usize, nt: usize, h: usize, s: mlx.mlx_stream) ![]mlx.mlx_array {
    const caps = try allocator.alloc(mlx.mlx_array, nt);
    const buf = try allocator.alloc(f32, n * h);
    defer allocator.free(buf);
    for (0..nt) |t| {
        for (0..n) |row| {
            @memcpy(buf[row * h .. (row + 1) * h], flat[row * nt * h + t * h .. row * nt * h + (t + 1) * h]);
        }
        caps[t] = try fixtureArr(buf, n, h, s);
    }
    return caps;
}

const ParityStats = struct { cos: f64, rms_ratio: f64 };

/// Cosine AND rms ratio — a cosine alone cannot see a scale error.
fn paritySlices(got: []const f32, want: []const f32) !ParityStats {
    try testing.expectEqual(want.len, got.len);
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (got, want) |a, b| {
        dot += @as(f64, a) * @as(f64, b);
        na += @as(f64, a) * @as(f64, a);
        nb += @as(f64, b) * @as(f64, b);
    }
    // Finiteness before the diff — an all-NaN stream scores cos 0/0.
    try testing.expect(std.math.isFinite(dot) and na > 0 and nb > 0);
    return .{ .cos = dot / (@sqrt(na) * @sqrt(nb)), .rms_ratio = @sqrt(na / nb) };
}

fn parityVs(got: mlx.mlx_array, want: []const f32, allocator: std.mem.Allocator, s: mlx.mlx_stream) !ParityStats {
    const g = try TinyFix.readF32(got, allocator, s);
    defer allocator.free(g);
    return paritySlices(g, want);
}

test "dflash fixture parity: encoder + two draft rounds vs transformers reference" {
    const fixtures_path = std.c.getenv("DFLASH_FIXTURES") orelse return;
    const assistant_dir = std.c.getenv("DFLASH_ASSISTANT_DIR") orelse return;
    if (mlx.noGpuBackend()) return;
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    const file = try std.Io.Dir.openFileAbsolute(io, std.mem.span(fixtures_path), .{});
    var rb: [65536]u8 = undefined;
    var rs = file.reader(io, &rb);
    const content = try rs.interface.allocRemaining(allocator, .limited(1 << 30));
    file.close(io);
    defer allocator.free(content);
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, content, .{});
    defer parsed.deinit();
    const root = parsed.value.object;

    const h: usize = @intCast(root.get("hidden_size").?.integer);
    const nt: usize = @intCast(root.get("n_targets").?.integer);
    const bs: usize = @intCast(root.get("block_size").?.integer);
    const n_ctx: usize = @intCast(root.get("n_ctx").?.integer);
    const n_delta: usize = @intCast(root.get("n_delta").?.integer);

    const ctx_stream = try fixtureF32Slice(allocator, root.get("ctx_stream").?);
    defer allocator.free(ctx_stream);
    const ctx_delta = try fixtureF32Slice(allocator, root.get("ctx_delta").?);
    defer allocator.free(ctx_delta);
    const noise1 = try fixtureF32Slice(allocator, root.get("noise1").?);
    defer allocator.free(noise1);
    const noise2 = try fixtureF32Slice(allocator, root.get("noise2").?);
    defer allocator.free(noise2);
    const encoder_out = try fixtureF32Slice(allocator, root.get("encoder_out").?);
    defer allocator.free(encoder_out);
    const round1_hidden = try fixtureF32Slice(allocator, root.get("round1_hidden").?);
    defer allocator.free(round1_hidden);
    const round2_hidden = try fixtureF32Slice(allocator, root.get("round2_hidden").?);
    defer allocator.free(round2_hidden);

    var m = try loadDflash(io, allocator, s, std.mem.span(assistant_dir));
    defer m.deinit();
    try testing.expectEqual(bs, m.config.block_size);
    try testing.expectEqual(nt, m.config.target_layer_ids.len);

    const caps1 = try fixtureCaptures(allocator, ctx_stream, n_ctx, nt, h, s);
    defer {
        for (caps1) |a| _ = mlx.mlx_array_free(a);
        allocator.free(caps1);
    }

    // Encoder projection parity (bf16 engine vs fp32 oracle — cos AND rms,
    // the concat-stream rule: a cosine alone cannot see a scale error).
    {
        const enc = try encodeContext(&m, caps1);
        defer _ = mlx.mlx_array_free(enc);
        const p = try parityVs(enc, encoder_out, allocator, s);
        std.debug.print("[dflash-fixture] encoder cos={d:.6} rms_ratio={d:.4}\n", .{ p.cos, p.rms_ratio });
        try testing.expect(p.cos > 0.99);
        try testing.expect(@abs(p.rms_ratio - 1.0) < 0.05);
    }

    // Round 1: context [0, n_ctx), block anchored at n_ctx.
    var dctx = try DflashCtx.init(allocator, &m, 0);
    defer dctx.deinit();
    try appendContext(&m, &dctx, caps1, 0);
    {
        const noise = try fixtureArr(noise1, bs, h, s);
        defer _ = mlx.mlx_array_free(noise);
        const hidden = try forwardBlock(&m, &dctx, noise, n_ctx);
        defer _ = mlx.mlx_array_free(hidden);
        const p = try parityVs(hidden, round1_hidden, allocator, s);
        std.debug.print("[dflash-fixture] round1 cos={d:.6} rms_ratio={d:.4}\n", .{ p.cos, p.rms_ratio });
        try testing.expect(p.cos > 0.99);
        try testing.expect(@abs(p.rms_ratio - 1.0) < 0.05);
    }

    // Round 2: n_delta more context, new anchor — pins absolute-position
    // RoPE and the block-K/V eviction across rounds.
    const caps2 = try fixtureCaptures(allocator, ctx_delta, n_delta, nt, h, s);
    defer {
        for (caps2) |a| _ = mlx.mlx_array_free(a);
        allocator.free(caps2);
    }
    try appendContext(&m, &dctx, caps2, n_ctx);
    {
        const noise = try fixtureArr(noise2, bs, h, s);
        defer _ = mlx.mlx_array_free(noise);
        const hidden = try forwardBlock(&m, &dctx, noise, n_ctx + n_delta);
        defer _ = mlx.mlx_array_free(hidden);
        const p = try parityVs(hidden, round2_hidden, allocator, s);
        std.debug.print("[dflash-fixture] round2 cos={d:.6} rms_ratio={d:.4}\n", .{ p.cos, p.rms_ratio });
        try testing.expect(p.cos > 0.99);
        try testing.expect(@abs(p.rms_ratio - 1.0) < 0.05);
    }
}

test "dflash2 fixture parity: conv block forward + greedy selector path vs z-lab model_mlx" {
    // Fixtures: tests/dump_dflash2_fixtures.py on the real
    // incoai/Qwen3.8-27B-DFlash2. Bars: block hidden cos + rms_ratio (the
    // scale rule), and the selector path ids EXACTLY — both sides read the
    // identical sparse logits and the identical bf16 hidden, so the trace is
    // a pure-math question.
    const fixtures_path = std.c.getenv("DFLASH2_FIXTURES") orelse return;
    const assistant_dir = std.c.getenv("DFLASH2_ASSISTANT_DIR") orelse return;
    if (mlx.noGpuBackend()) return;
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    const file = try std.Io.Dir.openFileAbsolute(io, std.mem.span(fixtures_path), .{});
    var rb: [65536]u8 = undefined;
    var rs = file.reader(io, &rb);
    const content = try rs.interface.allocRemaining(allocator, .limited(1 << 30));
    file.close(io);
    defer allocator.free(content);
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, content, .{});
    defer parsed.deinit();
    const root = parsed.value.object;

    const h: usize = @intCast(root.get("hidden_size").?.integer);
    const nt: usize = @intCast(root.get("n_targets").?.integer);
    const bs: usize = @intCast(root.get("block_size").?.integer);
    const n_ctx: usize = @intCast(root.get("n_ctx").?.integer);
    const anchor_id: u32 = @intCast(root.get("anchor_id").?.integer);
    const vocab: usize = @intCast(root.get("vocab_size").?.integer);
    const n_active: usize = @intCast(root.get("n_active").?.integer);
    const m = bs - 1;

    const ctx_stream = try fixtureF32Slice(allocator, root.get("ctx_stream").?);
    defer allocator.free(ctx_stream);
    const noise1 = try fixtureF32Slice(allocator, root.get("noise1").?);
    defer allocator.free(noise1);
    const round1_hidden = try fixtureF32Slice(allocator, root.get("round1_hidden").?);
    defer allocator.free(round1_hidden);
    const active_vals = try fixtureF32Slice(allocator, root.get("active_vals").?);
    defer allocator.free(active_vals);

    // Dense bf16 load — parity isolates the conv/selector math from the
    // 8-bit serving default.
    var model = try loadDflashQuant(io, allocator, s, std.mem.span(assistant_dir), 0);
    defer model.deinit();
    try testing.expect(model.config.isDflash2());
    try testing.expect(model.selector != null);
    try testing.expectEqual(bs, model.config.block_size);

    // ── Conv block forward parity ──
    var dctx = try DflashCtx.init(allocator, &model, 0);
    defer dctx.deinit();
    const caps = try fixtureCaptures(allocator, ctx_stream, n_ctx, nt, h, s);
    defer {
        for (caps) |a| _ = mlx.mlx_array_free(a);
        allocator.free(caps);
    }
    try appendContext(&model, &dctx, caps, 0);
    const noise = try fixtureArr(noise1, bs, h, s);
    defer _ = mlx.mlx_array_free(noise);
    const hidden = try forwardBlock(&model, &dctx, noise, n_ctx);
    defer _ = mlx.mlx_array_free(hidden);
    {
        const p = try parityVs(hidden, round1_hidden, allocator, s);
        std.debug.print("[dflash2-fixture] block hidden cos={d:.6} rms_ratio={d:.4}\n", .{ p.cos, p.rms_ratio });
        try testing.expect(p.cos > 0.99);
        try testing.expect(@abs(p.rms_ratio - 1.0) < 0.05);
    }

    // ── Selector path parity, on the REFERENCE's hidden (decoupled) ──
    const ref_hidden = try fixtureArr(round1_hidden, bs, h, s);
    defer _ = mlx.mlx_array_free(ref_hidden);
    var logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(logits);
    {
        const buf = try allocator.alloc(f32, m * vocab);
        defer allocator.free(buf);
        @memset(buf, -10.0);
        const ids_val = root.get("active_ids").?.array;
        for (0..m) |t| {
            for (0..n_active) |j| {
                const id: usize = @intCast(ids_val.items[t * n_active + j].integer);
                buf[t * vocab + id] = active_vals[t * n_active + j];
            }
        }
        const l_shape = [_]c_int{ 1, @intCast(m), @intCast(vocab) };
        const f32_arr = mlx.mlx_array_new_data(buf.ptr, &l_shape, 3, .float32);
        defer _ = mlx.mlx_array_free(f32_arr);
        try mlx.check(mlx.mlx_array_set(&logits, f32_arr));
    }
    var prng = std.Random.DefaultPrng.init(1);
    var path = try selectPath(allocator, &model.selector.?, model.config.selector_top_k, ref_hidden, logits, anchor_id, 0.0, prng.random(), s);
    defer path.deinit(allocator);
    const want_path = root.get("path_ids").?.array;
    std.debug.print("[dflash2-fixture] path got={any}\n", .{path.ids});
    try testing.expectEqual(@as(usize, m), want_path.items.len);
    for (path.ids, want_path.items) |got, want| {
        try testing.expectEqual(@as(u32, @intCast(want.integer)), got);
    }
}

test "GLM assistant bounded block tail preserves visible keys and masks" {
    const s = mlx.gpuStream();
    const window = 2048;
    const q_len = 8;
    for ([_]usize{ 2048, 32768 }) |ctx_len| {
        const dropped = ctx_len - (window - 1);
        const full = (try buildBlockMask(.sliding_attention, 37, ctx_len, 37 + ctx_len, q_len, window, s)).?;
        defer _ = mlx.mlx_array_free(full);
        const tail = (try buildBlockMask(.sliding_attention, 37 + dropped, window - 1, 37 + ctx_len, q_len, window, s)).?;
        defer _ = mlx.mlx_array_free(tail);
        try mlx.check(mlx.mlx_array_eval(full));
        try mlx.check(mlx.mlx_array_eval(tail));
        const a = mlx.mlx_array_data_bfloat16(full) orelse return error.MlxArrayDataNull;
        const b = mlx.mlx_array_data_bfloat16(tail) orelse return error.MlxArrayDataNull;
        for (0..q_len) |q| {
            for (0..dropped) |k| try testing.expectEqual(@as(u16, 0xff80), a[q * (ctx_len + q_len) + k]);
            try testing.expectEqualSlices(u16, a[q * (ctx_len + q_len) + dropped .. (q + 1) * (ctx_len + q_len)], b[q * (window - 1 + q_len) .. (q + 1) * (window - 1 + q_len)]);
        }
    }
}

test "GLM assistant cached sliding mask equals the mask built at any absolute position" {
    const s = mlx.gpuStream();
    const window = 2048;
    for ([_]u32{ 8, 3 }) |q_len| for ([_]usize{ 0, 6145, 1_040_000 }) |base| for ([_]usize{ window - 1, 2040 }) |ctx_len| {
        // `ctx_len` rows ending one before the anchor, as the cropped serving context and the tail view hold them.
        const built = (try buildBlockMask(.sliding_attention, base, ctx_len, base + ctx_len, q_len, window, s)) orelse {
            try testing.expect((try tailMask(q_len, ctx_len, ctx_len, window, s)) == null);
            continue;
        };
        defer _ = mlx.mlx_array_free(built);
        const cached = (try tailMask(q_len, ctx_len, ctx_len, window, s)).?;
        defer _ = mlx.mlx_array_free(cached);
        try mlx.check(mlx.mlx_array_eval(built));
        try mlx.check(mlx.mlx_array_eval(cached));
        try testing.expectEqualSlices(c_int, mlx.getShape(built), mlx.getShape(cached));
        const n = mlx.mlx_array_size(built);
        try testing.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(built).?[0..n], mlx.mlx_array_data_bfloat16(cached).?[0..n]);
    };
}

test "GLM serving DFlash2 trims physical window and preserves absolute append positions" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = std.testing.allocator;
    const s = mlx.gpuStream();
    var ops = @import("glm5_model.zig").Ops{ .s = s };
    defer ops.deinit();
    var layer: DflashLayer = undefined;
    layer.layer_type = .sliding_attention;
    var types = [_]LayerType{.sliding_attention};
    var model: DflashModel = undefined;
    model.s = s;
    model.layers = @as(*[1]DflashLayer, @ptrCast(&layer));
    model.config = .{ .hidden_size = 8, .num_hidden_layers = 1, .num_attention_heads = 2, .num_key_value_heads = 2, .head_dim = 4, .intermediate_size = 16, .rms_norm_eps = 1e-5, .rope_theta = 10000, .sliding_window = 8, .layer_types = &types, .block_size = 4, .mask_token_id = 3, .target_layer_ids = &.{} };
    var ctx = try DflashCtx.init(a, &model, 0);
    defer ctx.deinit();
    var values: [80]f32 = undefined;
    for (&values, 0..) |*v, i| v.* = @floatFromInt(i);
    const raw = try ops.own(mlx.mlx_array_new_data(&values, &[_]c_int{ 1, 2, 10, 4 }, 4, .float32));
    const keys = try ops.cast(raw, .bfloat16);
    _ = try ctx.cache.update(0, keys, keys, s, 0);
    try trimGlmServingContext(&model, &ctx);
    try std.testing.expectEqual(@as(usize, 10), ctx.absLen());
    try std.testing.expectEqual(@as(usize, 3), ctx.base_pos);
    try std.testing.expectEqual(@as(u64, 224), ctx.cache.residentBytes());
    try std.testing.expectEqual(@as(usize, 28), mlx.mlx_array_strides(ctx.cache.entries[0].keys)[1]);
    const view = try ops.cast(ctx.cache.entries[0].keys, .float32);
    try mlx.check(mlx.mlx_array_eval(view));
    try std.testing.expectEqual(@as(f32, 12), mlx.mlx_array_data_float32(view).?[0]);
    try std.testing.expectEqual(@as(f32, 52), mlx.mlx_array_data_float32(view).?[28]);
    const next = try ops.slice(keys, 2, 0, 1);
    _ = try ctx.cache.update(0, next, next, s, 0);
    try trimGlmServingContext(&model, &ctx);
    try std.testing.expectEqual(@as(usize, 11), ctx.absLen());
    try std.testing.expectEqual(@as(usize, 4), ctx.base_pos);
    try std.testing.expectEqual(@as(u64, 224), ctx.cache.residentBytes());
}

test "DFlash block tail engages only for the native GLM drafter" {
    const nil = mlx.mlx_array{ .ctx = null };
    const lin = DflashLinear{ .w = nil, .scales = nil, .biases = nil };
    var layers: [5]DflashLayer = @splat(.{ .layer_type = .sliding_attention, .input_norm = nil, .post_attn_norm = nil, .q = lin, .q_norm = nil, .k = lin, .k_norm = nil, .v = lin, .o = lin, .gate = lin, .up = lin, .down = lin });
    var types: [5]LayerType = @splat(.sliding_attention);
    var taps = [_]u32{0};
    var model = DflashModel{
        .allocator = testing.allocator,
        .s = mlx.gpuStream(),
        .config = .{ .hidden_size = 128, .num_hidden_layers = 5, .num_attention_heads = 1, .num_key_value_heads = 1, .head_dim = 128, .intermediate_size = 128, .rms_norm_eps = 1e-5, .rope_theta = 10000, .sliding_window = 2048, .layer_types = &types, .block_size = 8, .mask_token_id = 3, .target_layer_ids = &taps },
        .fc = lin,
        .enc_norm = nil,
        .final_norm = nil,
        .layers = &layers,
    };
    var ctx = try DflashCtx.init(testing.allocator, &model, 0);
    defer ctx.deinit();
    const bits = [_]u16{0};
    const kv = mlx.mlx_array_new_data(&bits, &.{ 1, 1, 1, 1 }, 4, .bfloat16);
    defer _ = mlx.mlx_array_free(kv);
    for (ctx.cache.entries) |*entry| {
        try mlx.check(mlx.mlx_array_set(&entry.keys, kv));
        try mlx.check(mlx.mlx_array_set(&entry.values, kv));
        entry.initialized = true;
        entry.offset = 2048;
    }
    ctx.cache.step = 2048;
    var noise = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(noise);
    try mlx.check(mlx.mlx_zeros(&noise, &.{ 1, 8, 128 }, 3, .bfloat16, mlx.gpuStream()));
    try testing.expect(!blockTailEligible(&model, &ctx, noise));
    model.sliding_serving = true;
    try testing.expect(blockTailEligible(&model, &ctx, noise));
}

test "dflash2: prefix outputs preserve full block attention and causal convolution" {
    if (mlx.noGpuBackend()) return;
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [512]u8 = undefined;
    const path = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    try TinyFix.writeAssistant2(io, tmp.dir, path, s);
    for ([_]u32{ 0, 4 }) |bits| {
        var m = try loadDflashQuant(io, allocator, s, path, bits);
        defer m.deinit();
        var ctx = try DflashCtx.init(allocator, &m, 0);
        defer ctx.deinit();
        const c0 = try TinyFix.capArr(10, 600, s);
        defer _ = mlx.mlx_array_free(c0);
        const c1 = try TinyFix.capArr(10, 601, s);
        defer _ = mlx.mlx_array_free(c1);
        try appendContext(&m, &ctx, &.{ c0, c1 }, 0);
        const noise = try TinyFix.capArr(TinyFix.BLOCK, 700, s);
        defer _ = mlx.mlx_array_free(noise);
        const full = try forwardBlock(&m, &ctx, noise, 10);
        defer _ = mlx.mlx_array_free(full);
        const expected = try TinyFix.readF32(full, allocator, s);
        defer allocator.free(expected);
        for ([_]u32{ 1, 2, 3, TinyFix.BLOCK }) |rows| {
            const prefix = try forwardBlockPrefix(&m, &ctx, noise, 10, rows);
            defer _ = mlx.mlx_array_free(prefix);
            try testing.expectEqualSlices(c_int, &.{ 1, @intCast(rows), TinyFix.HIDDEN }, mlx.getShape(prefix));
            const actual = try TinyFix.readF32(prefix, allocator, s);
            defer allocator.free(actual);
            try testing.expectEqualSlices(f32, expected[0 .. rows * TinyFix.HIDDEN], actual);
            try testing.expectEqual(@as(usize, 10), ctx.cache.step);
        }
        try testing.expectError(error.InvalidDflashPrefix, forwardBlockPrefix(&m, &ctx, noise, 10, 0));
        try testing.expectError(error.InvalidDflashPrefix, forwardBlockPrefix(&m, &ctx, noise, 10, TinyFix.BLOCK + 1));
    }
}

test "assistantLoadBytes: a quantizing load bills the packed linears, the dense rest, and the stored copies as its transient" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    // fc [1024, 2048] bf16 quantizes (gs 64); a codebook and a norm stay dense; a [8, 30] linear has no group.
    const header =
        \\{"fc.weight":{"dtype":"BF16","shape":[1024,2048],"data_offsets":[0,4194304]},
        \\"candidate_selector.predecessor_codebook":{"dtype":"BF16","shape":[256,128],"data_offsets":[4194304,4259840]},
        \\"layers.0.self_attn.q_proj.weight":{"dtype":"BF16","shape":[8,30],"data_offsets":[4259840,4260320]},
        \\"norm.weight":{"dtype":"BF16","shape":[2048],"data_offsets":[4260320,4264416]}}
    ;
    const body = try testing.allocator.alloc(u8, 8 + header.len + 4264416);
    defer testing.allocator.free(body);
    @memset(body, 0);
    std.mem.writeInt(u64, body[0..8], header.len, .little);
    @memcpy(body[8 .. 8 + header.len], header);
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = body });
    var path_buf: [512]u8 = undefined;
    const dir_path = path_buf[0..try tmp.dir.realPath(io, &path_buf)];

    const g: u64 = 16384;
    const q = try assistantLoadBytes(io, testing.allocator, dir_path, 8);
    const packed_fc: u64 = 1024 * 2048 + 2 * 65536; // codes, then scales and biases of [1024, 32] bf16
    try testing.expectEqual(packed_fc + 65536 + g + g, q.resident);
    try testing.expectEqual(@as(u64, 4194304), q.transient);
    const dense = try assistantLoadBytes(io, testing.allocator, dir_path, 0);
    try testing.expectEqual(@as(u64, 4194304 + 65536 + g + g), dense.resident);
    try testing.expectEqual(@as(u64, 0), dense.transient);
}

test "assistantLoadBytes: a tiny assistant's bill is never under what its load keeps or peaks at" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [512]u8 = undefined;
    const dir_path = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    try TinyFix.writeAssistant2(io, tmp.dir, dir_path, s);
    const bill = try assistantLoadBytes(io, allocator, dir_path, 8);
    _ = mlx.mlx_clear_cache();
    var before: usize = 0;
    _ = mlx.mlx_get_active_memory(&before);
    _ = mlx.mlx_reset_peak_memory();
    var m = try loadDflashQuant(io, allocator, s, dir_path, 8);
    var after: usize = 0;
    var peak: usize = 0;
    _ = mlx.mlx_get_active_memory(&after);
    _ = mlx.mlx_get_peak_memory(&peak);
    m.deinit();
    try testing.expect(bill.resident >= after - before);
    try testing.expect(bill.resident + bill.transient >= peak - before);
}
