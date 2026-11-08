//! What Flash-Next's load checks beside the generic loaders. Reached through `arch/load.zig`.
const std = @import("std");
const log = @import("../log.zig");
const model_mod = @import("../model.zig");
const dflash_mod = @import("../dflash.zig");
const model_load = @import("../model_load.zig");
const tokenizer_mod = @import("../tokenizer.zig");
const qwen4_config = @import("qwen4_config.zig");
const qwen4_forward = @import("qwen4_forward.zig");
const qwen4_state = @import("qwen4_state.zig");
const mlx = @import("../mlx.zig");
const trunk = @import("../trunk.zig");
const io_util_mod = @import("../io_util.zig");
const forward_diag = @import("../forward_diag.zig");
const arch_state_mod = @import("../arch/arch_state.zig");
const testing = std.testing;
const Trunk = trunk.Trunk;
const ForwardCtx = trunk.ForwardCtx;
const KVQuantConfig = trunk.KVQuantConfig;
const MAX_BATCH_ROWS = trunk.MAX_BATCH_ROWS;
const SpecWarmSlot = Trunk.SpecWarmSlot;
const specWarmIds = Trunk.specWarmIds;
const specWarmEvalFree = Trunk.specWarmEvalFree;
const verifyQmmNaxAvailable = trunk.verifyQmmNaxAvailable;

/// The resident key of a streamed tensor of the native fused BF16 checkpoint; null drops it.
pub fn fusedStreamKey(buf: []u8, key: []const u8) ?[]const u8 {
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

/// The norms the native checkpoint stores as a delta from one.
pub fn fusedStreamFoldsNorm(key: []const u8) bool {
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

pub fn drafterChosen(io: std.Io, allocator: std.mem.Allocator, config: *model_mod.ModelConfig, drafter_dir: []const u8, no_drafter: bool, mtp_on: bool) !void {
    if (drafter_dir.len > 0) return model_load.stampDflashRequestTerms(io, allocator, config, drafter_dir);
    const why: []const u8 = if (no_drafter) "--no-drafter" else "DFlash2: --drafter <path>";
    log.info("[spec] Qwen3.8-Flash-Next decodes with {s} ({s})\n", .{ if (mtp_on) "MTP" else "plain ticks", why });
}

/// The widest verify whose rows are decode-tick identical (`mtp.MAX_DEPTH` drafts + t1).
const DFLASH_MAX_ROWS: u32 = 9;

/// The rows a DFlash2 round may verify, from the flags or the defaults; the drafter's own block and the verify path
/// bound them.
pub fn dflashRowBoundsOf(min: ?u32, max: ?u32, config_block: u32) error{DrafterDepthUnsupported}!dflash_mod.RowBounds {
    const hi = max orelse @min(config_block, DFLASH_MAX_ROWS);
    const lo = min orelse @min(dflash_mod.DRAFTER_DEPTH_MIN, hi);
    if (hi > @min(config_block, DFLASH_MAX_ROWS) or lo > hi or lo < dflash_mod.DRAFTER_DEPTH_MIN) return error.DrafterDepthUnsupported;
    return .{ .min = lo, .max = hi };
}

pub fn dflashRowBounds(min: ?u32, max: ?u32, config_block: u32) !?u32 {
    const bounds = dflashRowBoundsOf(min, max, config_block) catch |err| {
        log.err("--drafter-min-depth/--drafter-max-depth: this drafter serves {d}..{d} rows per round on Qwen3.8-Flash-Next\n", .{ dflash_mod.DRAFTER_DEPTH_MIN, @min(config_block, DFLASH_MAX_ROWS) });
        return err;
    };
    return bounds.max;
}

pub fn bindDflash(_: std.mem.Allocator, xfm: *Trunk, d: *dflash_mod.DflashModel, _: KVQuantConfig, _: *tokenizer_mod.Tokenizer) !bool {
    d.sliding_serving = true;
    try d.bind(xfm);
    return true;
}

pub fn logDflashReady(block_size: u32) void {
    log.info("[spec] Qwen3.8-Flash-Next decodes with DFlash2 ({d}..{d} rows per round; --drafter outranks MTP, --no-drafter selects it)\n", .{ dflash_mod.minRows(), block_size });
}

test "Flash-Next rows default to 2..block and are bounded by the drafter block and the verify path" {
    const RowBounds = dflash_mod.RowBounds;
    try testing.expectEqual(RowBounds{ .min = 2, .max = 8 }, try dflashRowBoundsOf(null, null, 8));
    try testing.expectEqual(RowBounds{ .min = 2, .max = 3 }, try dflashRowBoundsOf(null, 3, 8));
    try testing.expectEqual(RowBounds{ .min = 4, .max = 8 }, try dflashRowBoundsOf(4, null, 8));
    try testing.expectEqual(RowBounds{ .min = 4, .max = 4 }, try dflashRowBoundsOf(4, 4, 8));
    try testing.expectEqual(RowBounds{ .min = 2, .max = 3 }, try dflashRowBoundsOf(null, null, 3));
    try testing.expectError(error.DrafterDepthUnsupported, dflashRowBoundsOf(null, 10, 16));
    try testing.expectError(error.DrafterDepthUnsupported, dflashRowBoundsOf(null, 6, 5));
    try testing.expectError(error.DrafterDepthUnsupported, dflashRowBoundsOf(5, 3, 8));
    try testing.expectError(error.DrafterDepthUnsupported, dflashRowBoundsOf(9, null, 8));
}

/// The binder places the PLE by exact equality and nothing downstream notices its absence: a PLE-less trunk emits
/// plausible text.
pub fn checkPlePlacement(config: *const model_mod.ModelConfig, has_ple: []const bool) !void {
    if (qwen4_config.qwen4PleInstalledAt(has_ple, config.ple_layer_idx)) return;
    log.err("[qwen4] the PLE was not installed at layer {d} ({d} layers carry PLE weights)\n", .{ config.ple_layer_idx, std.mem.count(bool, has_ple, &.{true}) });
    return error.Qwen4PleNotInstalled;
}

/// The head's verify at every draft width up to `max_width`, before the first request; a load without the head skips.
pub fn warmVerify(config: *const model_mod.ModelConfig, xfm: *Trunk, max_width: u32, kv_config: KVQuantConfig) !void {
    _ = config;
    if (qwen4_state.mtpOf(xfm) == null) return;
    return warmupSpecVerify(xfm, max_width, kv_config);
}

fn specWarmVerify(self: *Trunk, slots: []const *SpecWarmSlot, seq: usize) !void {
    var ids: [MAX_BATCH_ROWS][MAX_WARM_SEQ]i32 = undefined;
    var rows: [MAX_BATCH_ROWS]mlx.mlx_array = undefined;
    var ctxs: [MAX_BATCH_ROWS]*ForwardCtx = undefined;
    defer for (rows[0..slots.len]) |r| {
        _ = mlx.mlx_array_free(r);
    };
    for (slots, 0..) |sl, i| {
        rows[i] = specWarmIds(ids[i][0..seq]);
        ctxs[i] = &sl.ctx;
    }
    var logits: [MAX_BATCH_ROWS]mlx.mlx_array = undefined;
    var last: [MAX_BATCH_ROWS]mlx.mlx_array = undefined;
    var all: [MAX_BATCH_ROWS]mlx.mlx_array = undefined;
    try self.forwardRowAxisVerify(rows[0..slots.len], ctxs[0..slots.len], logits[0..slots.len], last[0..slots.len], all[0..slots.len], null, null);
    specWarmEvalFree(logits[0..slots.len]);
    specWarmEvalFree(last[0..slots.len]);
    specWarmEvalFree(all[0..slots.len]);
}

const MAX_WARM_SEQ: usize = MAX_WIDTH_WARM + 1;
const MAX_WIDTH_WARM: usize = 8;
/// Wider than the expert decode chain (16 rows) and the multi-row trunk kernel (8): a prefill forward.
const SPEC_WARM_PREFILL_ROWS: usize = 64;

// The verify at each draft width, the grouped verify at each row total and the head's own step
// each JIT their Metal pipelines on first dispatch; left to the first live request that compile
// lands inside a measured round and the table folds it as that width's price.
fn warmupSpecVerify(self: *Trunk, max_width: u32, kv_config: KVQuantConfig) !void {
    forward_diag.moe_dump_warmup_depth += 1;
    defer forward_diag.moe_dump_warmup_depth -= 1;
    var lap = io_util_mod.Stopwatch.init(std.Io.Threaded.global_single_threaded.io());
    const width_cap: usize = @min(@max(max_width, 1), MAX_WIDTH_WARM);
    const group_cap: usize = if (verifyQmmNaxAvailable()) 16 else 7;
    const max_rows: usize = @min(specWarmGroupRows(group_cap), MAX_BATCH_ROWS);

    var slots: [MAX_BATCH_ROWS]*SpecWarmSlot = undefined;
    var n_slots: usize = 0;
    defer {
        for (slots[0..n_slots]) |sl| {
            arch_state_mod.releaseGroup(self, &sl.ctx) catch {};
            sl.deinit(self.allocator);
        }
        _ = mlx.mlx_clear_cache();
    }
    var head_in = mlx.mlx_array_new();
    defer if (head_in.ctx != null) {
        _ = mlx.mlx_array_free(head_in);
    };
    while (n_slots < max_rows) : (n_slots += 1) {
        slots[n_slots] = try SpecWarmSlot.init(self, kv_config);
        var prompt_ids: [SPEC_WARM_PREFILL_ROWS]i32 = undefined;
        // The first slot's prompt is prefill-shaped, so the first request does not pay the prefill kernels' first use.
        const prompt = specWarmIds(prompt_ids[0..if (n_slots == 0) SPEC_WARM_PREFILL_ROWS else 8]);
        defer _ = mlx.mlx_array_free(prompt);
        var stream = mlx.mlx_array_new();
        qwen4_state.ctxOf(&slots[n_slots].ctx).capture_stream_all = if (n_slots == 0) &stream else null;
        const logits = try self.forwardWith(&slots[n_slots].ctx, prompt);
        qwen4_state.ctxOf(&slots[n_slots].ctx).capture_stream_all = null;
        if (n_slots == 0 and stream.ctx != null) {
            const sh = mlx.getShape(stream);
            try mlx.check(mlx.mlx_slice(&head_in, stream, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ sh[0], 1, sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, self.s));
        }
        specWarmEvalFree(&[_]mlx.mlx_array{ logits, stream });
    }

    var w: usize = 1;
    while (w <= width_cap) : (w += 1) try specWarmVerify(self, slots[0..1], w + 1);

    var n: usize = 2;
    var group_seq_max: usize = 0;
    while (n <= n_slots) : (n += 1) {
        const per_row_cap = specWarmGroupSeq(group_cap, n, width_cap) - 1;
        group_seq_max = @max(group_seq_max, per_row_cap + 1);
        var seq: usize = 2;
        while (seq <= per_row_cap + 1) : (seq += 1) try specWarmVerify(self, slots[0..n], seq);
    }
    if (head_in.ctx != null) try specWarmHead(self, head_in);
    log.info("[spec-warmup] verify widths 1..{d} solo, group rows 2..{d} at up to {d} tokens each, head width 1 ({d} ms).\n", .{ width_cap, n_slots, group_seq_max, lap.read() / std.time.ns_per_ms });
}



// Mirror of `mtp_tick.mtpSubGroupSize`: the slots one row-axis tick can carry at this row
// budget (NAX wide lane 16, stock lane 7) and the drafts each slot gets.
fn specWarmGroupRows(group_cap: usize) usize {
    return if (group_cap >= 16) 4 else 3;
}

fn specWarmGroupSeq(group_cap: usize, rows: usize, width_cap: usize) usize {
    return @min(group_cap / rows - 1, width_cap) + 1;
}

fn specWarmHead(self: *Trunk, stream_row: mlx.mlx_array) !void {
    var st = try qwen4_forward.qwen4MtpStateNew(self);
    defer st.deinit();
    var id_buf: [1]i32 = undefined;
    const id_arr = specWarmIds(&id_buf);
    defer _ = mlx.mlx_array_free(id_arr);
    for ([_]qwen4_forward.Qwen4MtpProject{ .mixed_last_row, .none }, 0..) |project, i| {
        const out = try qwen4_forward.qwen4MtpForwardOn(self, &st, stream_row, id_arr, @intCast(1 + i), null, project);
        specWarmEvalFree(&[_]mlx.mlx_array{ out.logits, out.stream, out.mixed });
    }
}

test "spec warm-up covers exactly the group shapes the scheduler can build" {
    try testing.expectEqual(@as(usize, 4), specWarmGroupRows(16));
    try testing.expectEqual(@as(usize, 3), specWarmGroupRows(7));
    try testing.expectEqual(@as(usize, 7), specWarmGroupSeq(16, 2, 6));
    try testing.expectEqual(@as(usize, 5), specWarmGroupSeq(16, 3, 6));
    try testing.expectEqual(@as(usize, 4), specWarmGroupSeq(16, 4, 6));
    try testing.expectEqual(@as(usize, 3), specWarmGroupSeq(7, 2, 6));
    try testing.expectEqual(@as(usize, 2), specWarmGroupSeq(7, 3, 6));
    try testing.expectEqual(@as(usize, 3), specWarmGroupSeq(16, 2, 2));
}
