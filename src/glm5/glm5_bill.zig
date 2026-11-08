//! GLM-5.3's own terms of the bill: the latent cache, the KDA checkpoints, the DFlash2 window and the native prefill's scratch.
const std = @import("std");
const model_mod = @import("../model.zig");
const fixtures = @import("../bills_fixtures_test.zig");
const bills = @import("../bills.zig");
const PinCfg = model_mod.ModelConfig;
const fp8_block = @import("../fp8_block.zig");
const glm5_generate = @import("glm5_generate.zig");
const glm5_prefix = @import("glm5_prefix.zig");
const prefix_cache_mod = @import("../prefix_cache.zig");
const testing = std.testing;

/// Most MLA layers among any `pending` consecutive layers: MLA-only prefill scratch is held
/// once per MLA layer in flight, not per pending layer.
fn glmMlaLayersPending(config: *const model_mod.ModelConfig, pending: u32) u32 {
    var most: u32 = 0;
    var start: u32 = 0;
    while (start + pending <= config.num_hidden_layers) : (start += 1) {
        var count: u32 = 0;
        for (start..start + pending) |i| count += @intFromBool(!config.isLinearLayer(@intCast(i)));
        most = @max(most, count);
    }
    return most;
}

/// Native MLA fuses attention and bounds index scores internally. It stores no
/// per-head K/V, affine rebuild, QSA score bank or generic SSM checkpoints.
pub fn glm5TransientBytes(config: *const model_mod.ModelConfig, seq: u64, chunk: u64, kv_bits: u64) u64 {
    const rows = @min(chunk, @max(seq, 1));
    const width = @as(u64, config.linear_num_value_heads) * config.linear_value_head_dim;
    const per_row = @as(u64, config.hidden_size) * 64 + width * 64 +
        @as(u64, config.num_experts_per_tok) * (@as(u64, config.moe_intermediate_size) * 8 + @as(u64, config.hidden_size) * 4);
    // A growing latent/pool layer retains its old allocation through evaluation.
    const latent_row = bills.kvBytesPerTokenAtBits(@as(u64, config.mla_kv_lora_rank) * 2, kv_bits);
    const grow = seq * (latent_row + @as(u64, config.indexer_head_dim) * 2 / @max(config.indexer_compress_ratio, 1));
    const pending = 2; // glm5_forward.Request.prefill_sync_layers
    const mla_pending = glmMlaLayersPending(config, pending);
    const n = std.math.cast(usize, rows) orelse return std.math.maxInt(u64);
    const quant_scratch: u64 = if (kv_bits < 16) @import("glm5_latent.zig").scratchBytes(config.mla_kv_lora_rank, n, mla_pending, @intCast(kv_bits)) else 0;
    const native = (@import("glm5_a6_dense_once.zig").transientBudget(n, pending) catch return std.math.maxInt(u64)) +|
        (@import("glm5_mla_prefill_batch.zig").transientBudget(n, mla_pending) catch return std.math.maxInt(u64)) +|
        (@import("glm5_attention_nax_packed.zig").transientBudget(n, mla_pending) catch return std.math.maxInt(u64)) +|
        (@import("glm5_attention.zig").packedCadenceTransientBudget(n, mla_pending) catch return std.math.maxInt(u64)) +|
        (@import("glm5_indexpool_nax.zig").transientBudget(n, mla_pending) catch return std.math.maxInt(u64)) +|
        (@import("glm5_kda_prefill_cluster.zig").transientBudget(n, pending) catch return std.math.maxInt(u64)) +|
        (@import("glm5_attention_decode_batch.zig").transientBudget(4) catch return std.math.maxInt(u64));
    const dflash = if (config.dflash_loaded) glmDflashRequestBytes(config, rows) else 0;
    const attention = @import("glm5_attention.zig");
    const history = std.math.cast(usize, seq) orelse return std.math.maxInt(u64);
    const selection = @max(attention.score_scratch_bytes, attention.selectionScratchBytes(history) catch return std.math.maxInt(u64));
    return rows *| per_row +| grow +| quant_scratch +| bills.slotRingBytes(config, 16) +| native +| dflash +| glmFp8DequantScratchBytes(config, rows, pending) +|
        selection +| attention.attention_scratch_bytes +| bills.PREFILL_RUNTIME_FLOOR_BYTES;
}

/// MiMo's raw-FP8 prefill route retains BF16 scratch for each projection until
/// the pending layer graph is evaluated. Decode uses its direct GEMV, with none.
pub fn glmFp8DequantScratchBytes(config: *const model_mod.ModelConfig, rows: u64, pending_layers: u64) u64 {
    if (!config.glm_fp8_trunk or rows <= @as(u64, @intCast(fp8_block.gemv_max_rows))) return 0;
    const hidden: u64 = config.hidden_size;
    const dense = 3 *| hidden *| config.intermediate_size;
    const shared = 3 *| hidden *| config.moe_intermediate_size;
    const mla = @as(u64, config.mla_q_lora_rank) *| (hidden +| @as(u64, config.num_attention_heads) *| config.mla_qk_nope_head_dim) +|
        @as(u64, config.mla_kv_lora_rank) *| hidden +| hidden *| config.num_attention_heads *| config.mla_v_head_dim +| shared;
    return @max(dense, mla) *| 2 *| pending_layers;
}

/// Dense sliding assistant window, capture/encoder/projection rows, target
/// replay states and the native verifier's bounded MLA work. Model weights
/// are already resident and are never charged again per request.
pub fn glmDflashRequestBytes(config: *const model_mod.ModelConfig, chunk: u64) u64 {
    if (!config.dflash_loaded) return 0;
    return config.dflash_window_bytes *| 4 +|
        config.dflash_capture_bytes_per_token *| chunk +|
        config.ssmCheckpointBytes() *| 3 +|
        @import("glm5_dflash_memory.zig").limit_bytes +| 64 * 1024 * 1024;
}

/// The checkpoints a GLM prefill of `seq` tokens takes after restoring `matched`: the one schedule the generator runs.
pub fn glmPrefillSchedule(seq: u64, matched: u64, cap: u32) glm5_generate.GlmCaptureSchedule {
    const offset: usize = @intCast(@min(matched, seq));
    return glm5_generate.glmCaptureSchedule(glm5_generate.glm_capture_stride, cap, offset, @as(usize, @intCast(seq)) - offset);
}

/// GLM's twin: held from the closer to the commit beside the checkpoints the commit lists, only for a request that keeps any.
pub fn glmThoughtPointBytes(config: *const model_mod.ModelConfig, warm: bills.WarmPrefix) u64 {
    if (!warm.thought or bills.effectiveSsmCheckpointStride(bills.launch_state.ssm_checkpoint_stride, bills.launch_state.prefix_cache_capacity, bills.launch_state.prefix_cache_ram_enabled, bills.diskBytesFor(config)) == 0 or warm.checkpoints == 0) return 0;
    return config.ssmCheckpointBytes();
}

/// What a GLM request holds for the prefix cache while it runs: its prefill's KDA checkpoints (at
/// most `warm.checkpoints`, plus the copy taken before each thin; a released prefill holds
/// `glm5_prefix.checkpoint_ram_held`) and the assistant window it keeps from prefill end.
pub fn glmPrefixStateBytes(config: *const model_mod.ModelConfig, seq: u64, warm: bills.WarmPrefix) u64 {
    if (bills.effectiveSsmCheckpointStride(bills.launch_state.ssm_checkpoint_stride, bills.launch_state.prefix_cache_capacity, bills.launch_state.prefix_cache_ram_enabled, bills.diskBytesFor(config)) == 0) return 0;
    if (warm.checkpoints == 0) return 0;
    const window: u64 = if (config.dflash_loaded) config.dflash_window_bytes else 0;
    const peak = glmPrefillSchedule(seq, warm.matched_tokens, warm.checkpoints).peak;
    const held = @as(u64, if (warm.released) @min(peak, glm5_prefix.checkpoint_ram_held) else peak) * config.ssmCheckpointBytes();
    return held +| window;
}

/// What the finished request holds while its commit runs, beside its live cache: the MLA rows copied, the
/// checkpoints, the window and the SSD writer's staged flush. Only the RAM tier copies rows, at most its
/// budget (`HotPrefixCache.glmCommitLen`); the SSD flush reads the request's own (`glmDiskLen`).
pub fn glmCommitStateBytes(config: *const model_mod.ModelConfig, seq: u64, warm: bills.WarmPrefix, kv_bits: u64) u64 {
    if (bills.effectiveSsmCheckpointStride(bills.launch_state.ssm_checkpoint_stride, bills.launch_state.prefix_cache_capacity, bills.launch_state.prefix_cache_ram_enabled, bills.diskBytesFor(config)) == 0 or warm.checkpoints == 0) return 0;
    const sched = glmPrefillSchedule(seq, warm.matched_tokens, warm.checkpoints);
    const rows = @as(u64, sched.newest) * bills.sessionBytesPerToken(config, kv_bits);
    const bound = bills.resolvedPrefixCacheMem();
    const copied: u64 = if (!bills.launch_state.prefix_cache_ram_enabled) 0 else if (bound == 0) rows else @min(rows, bound);
    const window: u64 = if (config.dflash_loaded) config.dflash_window_bytes else 0;
    const listed = @min(sched.events, warm.checkpoints);
    const kept = @as(u64, if (warm.released) @min(listed, glm5_prefix.checkpoint_ram_held) else listed) * config.ssmCheckpointBytes();
    return copied +| kept +| window;
}

/// A native GLM request's latent and pooled-index capacity at its eventual length.
pub fn glmCacheBytes(config: *const model_mod.ModelConfig, prompt_len: u64, max_tokens: u32, kv_bits: u64) u64 {
    const ctx = bills.getEffectiveContextLength(config);
    const reserved = @max(prompt_len, @min(prompt_len +| max_tokens, ctx));
    // Native appendRows rounds latent and pooled capacities separately to 256 rows.
    const rows = (reserved +| 1023) / 1024 * 1024;
    return bills.sessionBytesPerToken(config, kv_bits) *| rows;
}

// Hooks: bills.zig reaches these through the arch's `bill` declaration; a hook returns null where it has no say.

pub fn requestBill(config: *const model_mod.ModelConfig, seq: u64, max_tokens: u32, kv_bits: u64, chunk: u64, warm: bills.WarmPrefix) ?u64 {
    return (glmCacheBytes(config, seq, max_tokens, kv_bits) +| @max(
        glm5TransientBytes(config, seq, chunk, kv_bits) +| glmPrefixStateBytes(config, seq, warm),
        glmCommitStateBytes(config, seq, warm, kv_bits) +| glmThoughtPointBytes(config, warm),
    )) *| 5 / 4 +|
        (if (config.expert_streaming) config.expert_fill_peak_bytes else 0) +|
        bills.cadenceInflightBytes(bills.mlx_mb_per_buffer);
}

pub fn transientReserve(config: *const model_mod.ModelConfig, kv_bits: u64, chunk: u64, seq: u64) ?u64 {
    return glm5TransientBytes(config, seq, chunk, kv_bits);
}

pub fn sessionBytes(config: *const model_mod.ModelConfig, kv_bits: u64) ?u64 {
    return bills.kvBytesPerTokenAtBits(config.kvBytesPerToken(), kv_bits) + config.qsaHistoryBytesPerToken();
}

pub fn slotRing(config: *const model_mod.ModelConfig, kv_bits: u64) ?u64 {
    _ = kv_bits;
    return config.qsaRingBytes() + config.ssmCheckpointBytes();
}

pub fn stateBilled(config: *const model_mod.ModelConfig) ?u64 {
    return config.qsaHistoryBytesPerToken();
}

pub fn retainedCheckpoints(config: *const model_mod.ModelConfig, seq: u64, matched: u64) ?u64 {
    return glmPrefillSchedule(seq, matched, glm5_prefix.checkpointMax(bills.launch_state.ssm_checkpoint_max)).peak * config.ssmCheckpointBytes();
}

pub fn commitBytes(config: *const model_mod.ModelConfig, prompt_len: usize, max_tokens: u32, kv_bits: u64) ?u64 {
    return glmCacheBytes(config, prompt_len, max_tokens, kv_bits) +| bills.slotRingBytes(config, kv_bits);
}

pub fn neutralChunkWidth(config: *const model_mod.ModelConfig) ?u32 {
    _ = config;
    return @import("glm5_forward.zig").prefill_chunk;
}

/// A load that starts at the width GLM's kernels are built for says so, instead of the memory-sized-down line.
pub fn prefillChunkLoadLine(config: *const model_mod.ModelConfig, buf: []u8, pinned: u32, launch: usize) ?[]const u8 {
    _ = config;
    if (pinned >= launch or pinned != @import("glm5_forward.zig").prefill_chunk) return null;
    return std.fmt.bufPrint(buf, "Prefill chunk: {d} tokens (GLM's numerics width; --prefill-chunk overrides)\n", .{pinned}) catch null;
}

/// GLM warms T=1 and T=8, then resets; its KDA state, rings and first latent rows exist from then on.
pub fn loadWarmup(config: *const model_mod.ModelConfig, mtp_on: bool) ?u64 {
    _ = mtp_on;
    const gib: u64 = 1 << 30;
    return config.ssmCheckpointBytes() +| config.qsaRingBytes() +|
        (config.kvBytesPerToken() +| config.qsaHistoryBytesPerToken()) *| 1024 +| gib * 3 / 8;
}

/// GLM's admission evicts its RAM tier and prices each request's checkpoints instead.
pub fn cacheReserve(config: *const model_mod.ModelConfig) ?u64 {
    _ = config;
    return 0;
}

pub fn unnamedCacheAsk(config: *const model_mod.ModelConfig, requested: u64, explicit: bool) ?u64 {
    _ = config;
    if (!explicit and requested == bills.launch_state.PREFIX_CACHE_MEM_DEFAULT) return bills.GLM_PREFIX_CACHE_MEM_DEFAULT;
    return null;
}


test "a model whose SSD tier did not come up bills and budgets as with --prefix-cache-disk off" {
    const t = std.testing;
    var glm = try model_mod.parseConfigFromJson(t.allocator, @embedFile("../fixtures/glm5_config.json"));
    glm.dflash_loaded = true;
    glm.dflash_window_bytes = 5 * 8 * 128 * 4 * 2304;
    const saved = .{ bills.launch_state.prefix_cache_capacity, bills.launch_state.ssm_checkpoint_stride, bills.launch_state.ssm_checkpoint_max };
    defer {
        bills.launch_state.prefix_cache_capacity = saved[0];
        bills.launch_state.ssm_checkpoint_stride = saved[1];
        bills.launch_state.ssm_checkpoint_max = saved[2];
    }
    bills.launch_state.prefix_cache_capacity = 32;
    bills.launch_state.ssm_checkpoint_stride = 256;
    bills.launch_state.ssm_checkpoint_max = 16;
    const warm: bills.WarmPrefix = .{ .checkpoints = 4 };

    // RAM off: the SSD tier is the only retention, so with it GLM keeps checkpoints and without it none.
    const tiers = bills.TierGuard.pin(false, 20 << 30);
    defer tiers.restore();
    try t.expect(bills.diskTierOn(&glm));
    const with_tier = glmCommitStateBytes(&glm, 65536, warm, 8);
    try t.expect(with_tier > 0);
    // Declined at load (no room, a failed init): the same answers as an explicit `--prefix-cache-disk off`.
    glm.prefix_cache_disk_declined = true;
    try t.expect(!bills.diskTierOn(&glm));
    bills.launch_state.prefix_cache_disk_bytes = 0;
    glm.prefix_cache_disk_declined = false;
    const off_flag = .{ glmCommitStateBytes(&glm, 65536, warm, 8), glmPrefixStateBytes(&glm, 65536, warm) };
    bills.launch_state.prefix_cache_disk_bytes = 20 << 30;
    glm.prefix_cache_disk_declined = true;
    try t.expectEqual(off_flag[0], glmCommitStateBytes(&glm, 65536, warm, 8));
    try t.expectEqual(off_flag[1], glmPrefixStateBytes(&glm, 65536, warm));
    try t.expectEqual(@as(u64, 0), off_flag[0]);
    // The RAM sizing no longer takes the SSD-first arm for this model either; the probe config is Qwen's.
    if (!bills.archBuilt("qwen4")) return;
    var cfg = fixtures.qwen4DeployedTestConfig();
    const saved_over = prefix_cache_mod.ssd_first_override;
    defer prefix_cache_mod.ssd_first_override = saved_over;
    prefix_cache_mod.ssd_first_override = true;
    bills.launch_state.prefix_cache_ram_enabled = true;
    var idle: u64 = 0;
    try t.expect(bills.ssdFirstBudgetForLoad(&cfg, 1 << 30, 100 << 30, 40 << 30, 4 << 30, 1 << 30, &idle, true) != null);
    cfg.prefix_cache_disk_declined = true;
    try t.expect(bills.ssdFirstBudgetForLoad(&cfg, 1 << 30, 100 << 30, 40 << 30, 4 << 30, 1 << 30, &idle, true) == null);
}

test "GLM MLA prefill scratch is held by the MLA layers a pending window can contain" {
    const cfg = try fixtures.glmAssistantFixture();
    // Layers 3, 7, ..., 43: two consecutive layers never hold two of them.
    try std.testing.expectEqual(@as(u32, 1), glmMlaLayersPending(&cfg, 2));
    try std.testing.expectEqual(@as(u32, 2), glmMlaLayersPending(&cfg, 5));
    var adjacent = cfg;
    adjacent.full_attention_interval = 1;
    try std.testing.expectEqual(@as(u32, 2), glmMlaLayersPending(&adjacent, 2));
}

test "GLM serving DFlash2 bill includes the bounded window captures replay and scratch" {
    var cfg = try model_mod.parseConfigFromJson(std.testing.allocator, @embedFile("../fixtures/glm5_config.json"));
    try std.testing.expectEqual(@as(u64, 0), glmDflashRequestBytes(&cfg, 2048));
    cfg.dflash_loaded = true;
    cfg.dflash_window_bytes = 5 * 8 * 128 * 4 * 2304;
    cfg.dflash_capture_bytes_per_token = (5 * 4096 + 2 * 4096) * 2 + 5 * 8 * 128 * 4;
    const needed = glmDflashRequestBytes(&cfg, 2048);
    try std.testing.expect(needed >= cfg.dflash_window_bytes * 4 + 256 * 1024 * 1024);
    try std.testing.expect(needed >= cfg.ssmCheckpointBytes() * 3);
    try std.testing.expect(needed < 2 * 1024 * 1024 * 1024);
    try std.testing.expect(glmDflashRequestBytes(&cfg, 1024) < needed);
}

test "GLM prefill transient bill follows the latent width: kv6 sheds 128 bytes per latent row against kv8" {
    const cfg = try model_mod.parseConfigFromJson(std.testing.allocator, @embedFile("../fixtures/glm5_config.json"));
    const seq: u64 = 4096;
    const chunk: u64 = 2048;
    const kv8 = glm5TransientBytes(&cfg, seq, chunk, 8);
    const kv6 = glm5TransientBytes(&cfg, seq, chunk, 6);
    const pending: u64 = glmMlaLayersPending(&cfg, 2);
    try std.testing.expectEqual(seq * 128 + pending * chunk * 128, kv8 - kv6);
}

test "GLM raw FP8 scratch bills all pending projections only beyond direct decode" {
    var cfg = try model_mod.parseConfigFromJson(std.testing.allocator, @embedFile("../fixtures/glm5_config.json"));
    try std.testing.expectEqual(@as(u64, 0), glmFp8DequantScratchBytes(&cfg, 2048, 2));
    cfg.glm_fp8_trunk = true;
    try std.testing.expectEqual(@as(u64, 0), glmFp8DequantScratchBytes(&cfg, @intCast(fp8_block.gemv_max_rows), 2));
    try std.testing.expectEqual(@as(u64, 576 * 1024 * 1024), glmFp8DequantScratchBytes(&cfg, 2048, 2));
    try std.testing.expectEqual(glmFp8DequantScratchBytes(&cfg, 17, 1) * 2, glmFp8DequantScratchBytes(&cfg, 2048, 2));
}

test "GLM billed checkpoint peak equals the checkpoints its prefill captures, whatever stride is configured" {
    var cfg = try model_mod.parseConfigFromJson(std.testing.allocator, @embedFile("../fixtures/glm5_config.json"));
    cfg.dflash_loaded = false;
    const saved = .{ bills.launch_state.prefix_cache_capacity, bills.launch_state.ssm_checkpoint_stride, bills.launch_state.prefix_cache_disk_bytes, bills.launch_state.prefix_cache_ram_enabled };
    defer {
        bills.launch_state.prefix_cache_capacity = saved[0];
        bills.launch_state.ssm_checkpoint_stride = saved[1];
        bills.launch_state.prefix_cache_disk_bytes = saved[2];
        bills.launch_state.prefix_cache_ram_enabled = saved[3];
    }
    bills.launch_state.prefix_cache_capacity = 32;
    bills.launch_state.prefix_cache_disk_bytes = 0;
    bills.launch_state.prefix_cache_ram_enabled = true;
    const per_cp = cfg.ssmCheckpointBytes();
    const Case = struct { seq: u64, matched: u64 };
    const cases = [_]Case{
        .{ .seq = 16, .matched = 0 },       .{ .seq = 31, .matched = 0 },          .{ .seq = 40, .matched = 0 },
        .{ .seq = 2049, .matched = 0 },     .{ .seq = 5000, .matched = 0 },        .{ .seq = 18432, .matched = 0 },
        .{ .seq = 70000, .matched = 0 },    .{ .seq = 65536, .matched = 61440 },   .{ .seq = 18432, .matched = 16384 },
        .{ .seq = 9000, .matched = 8188 },  .{ .seq = 9100, .matched = 8188 },
    };
    for ([_]u32{ 256, 2048, 8192, 65536 }) |stride| {
        bills.launch_state.ssm_checkpoint_stride = stride;
        for (cases) |c| for (0..9) |cap_usize| {
            const cap: u32 = @intCast(cap_usize);
            const oracle = glm5_generate.glmCaptureOracle(glm5_generate.glm_checkpoint_stride, glm5_generate.glm_capture_stride, cap, c.matched, c.seq - c.matched, 2048);
            const billed = glmPrefixStateBytes(&cfg, c.seq, .{ .matched_tokens = c.matched, .checkpoints = cap });
            const want: u64 = if (cap == 0) 0 else @as(u64, oracle.peak) * per_cp;
            std.testing.expectEqual(want, billed) catch |e| {
                std.debug.print("stride {d} seq {d} matched {d} cap {d}: billed {d} cps, captured {d}\n", .{ stride, c.seq, c.matched, cap, billed / per_cp, want / per_cp });
                return e;
            };
            // A prefill that releases landed states holds the newest two, or all it captures if fewer.
            const released = glmPrefixStateBytes(&cfg, c.seq, .{ .matched_tokens = c.matched, .checkpoints = cap, .released = true });
            try std.testing.expectEqual(@min(want, 2 * per_cp), released);
        };
    }
}

test "GLM SSD-only bills its checkpoints and window, no writer term (the shared admission headroom has it), and sizes no cache" {
    var cfg = try model_mod.parseConfigFromJson(std.testing.allocator, @embedFile("../fixtures/glm5_config.json"));
    cfg.dflash_loaded = true;
    cfg.dflash_window_bytes = 5 * 8 * 128 * 4 * 2304;
    const saved = .{ bills.launch_state.prefix_cache_capacity, bills.launch_state.ssm_checkpoint_stride, bills.launch_state.ssm_checkpoint_max, bills.launch_state.prefix_cache_disk_bytes, bills.launch_state.prefix_cache_ram_enabled };
    defer {
        bills.launch_state.prefix_cache_capacity = saved[0];
        bills.launch_state.ssm_checkpoint_stride = saved[1];
        bills.launch_state.ssm_checkpoint_max = saved[2];
        bills.launch_state.prefix_cache_disk_bytes = saved[3];
        bills.launch_state.prefix_cache_ram_enabled = saved[4];
    }
    bills.launch_state.ssm_checkpoint_stride = 256;
    bills.launch_state.ssm_checkpoint_max = 16;
    bills.launch_state.prefix_cache_capacity = 0;
    const sizer_off = bills.prefillTransientReserve(&cfg, 8, 2048);
    bills.launch_state.prefix_cache_capacity = 32;
    bills.launch_state.prefix_cache_disk_bytes = 12 << 30;
    bills.launch_state.prefix_cache_ram_enabled = false;
    const held = @as(u64, glmPrefillSchedule(65536, 0, glm5_prefix.checkpointMax(16)).peak) * cfg.ssmCheckpointBytes() + cfg.dflash_window_bytes;
    try std.testing.expectEqual(held, glmPrefixStateBytes(&cfg, 65536, .{ .checkpoints = glm5_prefix.checkpointMax(16) }));
    try std.testing.expectEqual(@as(u64, 0), glmPrefixStateBytes(&cfg, 65536, .{}));
    try std.testing.expectEqual(@as(u64, 0), bills.ctxSizingCacheReserve(&cfg));
    try std.testing.expectEqual(@as(u64, 0), bills.resolvedPrefixCacheMem());
    try std.testing.expectEqual(sizer_off, bills.prefillTransientReserve(&cfg, 8, 2048));
}

test "bill pin: the checkpoint and thought-point terms" {
    try bills.pinConfigs("checkpointTerms", &.{
        115458048, 0, 0, 0, 0, 0, 0, 0,
        115458048, 115458048, 115458048, 0, 0, 0, 0, 0,
        0, 0, 115458048, 115458048, 346374144, 0, 0, 0,
        0, 0, 0, 0, 230916096, 230916096, 346374144, 0,
        0, 0, 0, 0, 0, 0, 230916096, 230916096,
        0, 0, 0, 0, 0, 0, 115458048, 115458048,
        3, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 147619840, 0, 0, 0, 0, 0,
        0, 0, 147619840, 147619840, 147619840, 0, 0, 0,
        0, 0, 0, 0, 147619840, 147619840, 1328578560, 0,
        0, 0, 0, 0, 0, 0, 295239680, 295239680,
        1328578560, 0, 0, 0, 0, 0, 0, 0,
        295239680, 295239680, 0, 0, 0, 0, 0, 0,
        0, 147619840, 0, 0, 67487440, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 67487440, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        169837440, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 169837440, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0,
    }, struct {
        fn run(c: *const PinCfg, g: *bills.PinGot) void {
            for ([_]u64{ 5000, 120_000 }) |seq| for ([_]u64{ 0, 3000 }) |matched| {
                g.add(bills.retainedSsmCheckpointBytes(c, seq, matched, 2048));
                g.add(bills.checkpointReserve(c, seq, 2048));
                for (bills.pinWarms()) |w| {
                    g.add(glmPrefixStateBytes(c, seq, w));
                    g.add(glmCommitStateBytes(c, seq, w, 8));
                }
            };
            for (bills.pinWarms()) |w| {
                g.add(bills.thoughtPointBytes(c, w));
                g.add(glmThoughtPointBytes(c, w));
            }
            g.add(bills.ssmResidentCap(c));
        }
    }.run);
}

test "bill pin: the GLM terms" {
    try bills.pinConfigsRagged("glmTerms", &.{
        1, 0, 0, 0, 0, 0, 0, 2,
        0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 3251384576, 61636608, 1120247784,
        3250570240, 110297088, 1181447424, 2, 0, 0, 0, 0,
        0, 0,
    }, .{ 7, 7, 13, 7 }, struct {
        fn run(c: *const PinCfg, g: *bills.PinGot) void {
            g.add(glmMlaLayersPending(c, 2));
            for ([_]u64{ 1, 5000 }) |rows| {
                g.add(glmFp8DequantScratchBytes(c, rows, 2));
                g.add(glmDflashRequestBytes(c, rows));
                g.add(bills.dflashRequestBytes(c, rows));
            }
            if (c.isGlm5()) {
                for (bills.pin_bits) |b| {
                    g.add(glm5TransientBytes(c, 5000, 2048, b));
                    g.add(glmCacheBytes(c, 5000, 4096, b));
                    g.add(bills.glmDflashReserveBudget(c, 5000, 4096, b, 1 << 20));
                }
            }
        }
    }.run);
}
