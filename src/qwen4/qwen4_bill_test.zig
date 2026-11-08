//! Flash-Next bills: the QSA mask, sheet and history, the MTP head and the long-context admission.
const bills = @import("../bills.zig");
const CTX_SIZING_CACHE_RESERVE = bills.CTX_SIZING_CACHE_RESERVE;
const CheckoutCase = bills.CheckoutCase;
const PREFILL_CHUNK_LADDER = bills.PREFILL_CHUNK_LADDER;
const PREFILL_RUNTIME_FLOOR_BYTES = bills.PREFILL_RUNTIME_FLOOR_BYTES;
const PrefillRequestTerms = bills.PrefillRequestTerms;
const TierGuard = bills.TierGuard;
const WarmPrefix = bills.WarmPrefix;
const adaptCapFor = bills.adaptCapFor;
const adaptivePrefillChunkEnabled = bills.adaptivePrefillChunkEnabled;
const adaptivePrefillWidenStillFits = bills.adaptivePrefillWidenStillFits;
const adaptivePrefillWidth = bills.adaptivePrefillWidth;
const adaptivePrefillWidthNow = bills.adaptivePrefillWidthNow;
const admissionAvailable = bills.admissionAvailable;
const archBuilt = bills.archBuilt;
const attnLayersPerEvalWindow = bills.attnLayersPerEvalWindow;
const autoContextFrom = bills.autoContextFrom;
const clampReserveWidth = bills.clampReserveWidth;
const defaultKvBits = bills.defaultKvBits;
const defaultPrefixCacheAsk = bills.defaultPrefixCacheAsk;
const diskBytesPerToken = bills.diskBytesPerToken;
const expertStreamingLoadFits = bills.expertStreamingLoadFits;
const expertStreamingPlannedKvBytes = bills.expertStreamingPlannedKvBytes;
const expertStreamingPlannedSeq = bills.expertStreamingPlannedSeq;
const expertStreamingServingBytes = bills.expertStreamingServingBytes;
const explicitPrefillChunk = bills.explicitPrefillChunk;
const fixtures = @import("../bills_fixtures_test.zig");
const forceMtpFor = bills.forceMtpFor;
const getEffectiveContextLength = bills.getEffectiveContextLength;
const kvBytesPerTokenAtBits = bills.kvBytesPerTokenAtBits;
const kvDequantScratchBytes = bills.kvDequantScratchBytes;
const kv_disk_cache = @import("../kv_disk_cache.zig");
const launch_state = @import("../server_state.zig");
const memoryContextAt = bills.memoryContextAt;
const memoryRefusalMessage = bills.memoryRefusalMessage;
const model_load = @import("../model_load.zig");
const model_mod = @import("../model.zig");
const mtpHeadKvBytesPerToken = bills.mtpHeadKvBytesPerToken;
const mtp_plan = @import("../mtp_plan.zig");
const nax_gate = @import("../nax_gate.zig");
const oneSessionEntryBytes = bills.oneSessionEntryBytes;
const oneSessionFor = bills.oneSessionFor;
const oomBillFor = bills.oomBillFor;
const perRequestPrefillChunkEnabled = bills.perRequestPrefillChunkEnabled;
const physicalMemoryCeiling = bills.physicalMemoryCeiling;
const planHotCache = bills.planHotCache;
const prefillChunkCost = bills.prefillChunkCost;
const prefillDequantWeightBytes = bills.prefillDequantWeightBytes;
const prefillFfnWidth = bills.prefillFfnWidth;
const prefillMemoryNeeded = bills.prefillMemoryNeeded;
const prefillNeededAtChunk = bills.prefillNeededAtChunk;
const prefillRequestTerms = bills.prefillRequestTerms;
const prefillStreamBytesPerToken = bills.prefillStreamBytesPerToken;
const prefillTransientReserve = bills.prefillTransientReserve;
const prefillTransientReserveAtKv = bills.prefillTransientReserveAtKv;
const prefill_plan = @import("../prefill_plan.zig");
const qsaMaskBytes = bills.qsaMaskBytes;
const qwen4_qsa_attn = @import("qwen4_qsa_attn.zig");
const qwen4_qsa_gather = @import("qwen4_qsa_gather.zig");
const qwen4_qsa_select = @import("qwen4_qsa_select.zig");
const request_policy = @import("../request_policy.zig");
const reservedCacheTokens = bills.reservedCacheTokens;
const resolvePrefillChunk = bills.resolvePrefillChunk;
const resolvedContextForLoad = bills.resolvedContextForLoad;
const retainedSsmCheckpointBytes = bills.retainedSsmCheckpointBytes;
const safeContextForBudget = bills.safeContextForBudget;
const scheduler_mod = @import("../scheduler.zig");
const sessionBytesPerToken = bills.sessionBytesPerToken;
const settings_resolve = @import("../settings_resolve.zig");
const sizerCtxKvBytes = bills.sizerCtxKvBytes;
const slotRingBytes = bills.slotRingBytes;
const slot_admission = @import("../slot_admission.zig");
const ssdFirstPrefixCacheMem = bills.ssdFirstPrefixCacheMem;
const ssdFirstSessionKvBytes = bills.ssdFirstSessionKvBytes;
const ssdFirstSessionTokensNow = bills.ssdFirstSessionTokensNow;
const statePerTokenBilled = bills.statePerTokenBilled;
const std = @import("std");
const trunk = @import("../trunk.zig");

/// The deployed long-context shape at the live per-token widths: 12 caching layers, hd 256,
/// 2 KV heads (13,056 B/tok at 8-bit); a 128-wide QSA indexer at ratio 4.
pub fn qsaScoreFusedOffGuard() struct {
    pub fn deinit(_: @This()) void {
        qwen4_qsa_select.qsa_score_fused_override = null;
    }
} {
    qwen4_qsa_select.qsa_score_fused_override = false;
    return .{};
}

/// The incidents these bills reproduce ran an engine that rebuilt the whole kv8 cache per layer.
pub fn qsaWholeRebuildGuard() struct {
    pub fn deinit(_: @This()) void {
        qwen4_qsa_attn.qsa_attn_kernel_override = null;
    }
} {
    qwen4_qsa_attn.qsa_attn_kernel_override = false;
    return .{};
}

test "diskBytesPerToken: the serialized geometry, whatever the QSA execution switches say" {
    if (!archBuilt("qwen4") or !archBuilt("mimo2") or !archBuilt("glm5")) return error.SkipZigTest;
    const t = std.testing;
    const saved_mtp = launch_state.configured_mtp;
    defer launch_state.configured_mtp = saved_mtp;
    const saved_fused = qwen4_qsa_select.qsa_score_fused_override;
    defer qwen4_qsa_select.qsa_score_fused_override = saved_fused;
    var q = fixtures.qwen4DeployedTestConfig();
    // kv8 rows plus the one pooled history; the head is off.
    launch_state.configured_mtp = false;
    // Plus its share of the recurrent state kept on every grid point.
    const points = (q.ssmCheckpointBytes() + kv_disk_cache.POINT_GRID - 1) / kv_disk_cache.POINT_GRID;
    try t.expect(points > 0);
    try t.expectEqual(@as(u64, 13_824) + points, diskBytesPerToken(&q, 8));
    const head = mtpHeadKvBytesPerToken(&q) + q.qsaHistoryBytesPerToken() / q.attnCacheLayerCount();
    try t.expect(head > 0);
    launch_state.configured_mtp = true;
    try t.expectEqual(@as(u64, 13_824) + points + head, diskBytesPerToken(&q, 8));
    try t.expect(diskBytesPerToken(&q, 16) > diskBytesPerToken(&q, 8));
    // The score bank is execution state, never in an entry.
    for ([_]?bool{ true, false }) |fused| {
        qwen4_qsa_select.qsa_score_fused_override = fused;
        for ([_]bool{ false, true }) |mtp| {
            launch_state.configured_mtp = mtp;
            try t.expectEqual(@as(u64, 13_824) + points + (if (mtp) head else 0), diskBytesPerToken(&q, 8));
        }
    }
    qwen4_qsa_select.qsa_score_fused_override = false;
    launch_state.configured_mtp = false;
    try t.expect(sessionBytesPerToken(&q, 8) > diskBytesPerToken(&q, 8) - points);

    // A ringed arch pays per token only on its global layers; its head is a fixed window.
    const mimo = fixtures.mimoV2FlashBillConfig();
    for ([_]bool{ false, true }) |mtp| {
        launch_state.configured_mtp = mtp;
        try t.expectEqual(kvBytesPerTokenAtBits(mimo.kvBytesPerToken(), 8), diskBytesPerToken(&mimo, 8));
        try t.expectEqual(kvBytesPerTokenAtBits(mimo.kvBytesPerToken(), 16), diskBytesPerToken(&mimo, 16));
    }

    // GLM: the latent rows (6,688 at kv8, 11,968 BF16) and the pooled index; DFlash2's window is fixed per entry.
    var glm = try model_mod.parseConfigFromJson(t.allocator, @embedFile("../fixtures/glm5_config.json"));
    glm.dflash_loaded = false;
    const glm_points = (glm.ssmCheckpointBytes() + kv_disk_cache.POINT_GRID - 1) / kv_disk_cache.POINT_GRID;
    try t.expect(glm_points > 0);
    try t.expectEqual(@as(u64, 6_688) + glm_points, diskBytesPerToken(&glm, 8));
    try t.expectEqual(@as(u64, 5_280) + glm_points, diskBytesPerToken(&glm, 6));
    try t.expectEqual(@as(u64, 11_968) + glm_points, diskBytesPerToken(&glm, 16));
    glm.dflash_loaded = true;
    glm.dflash_window_bytes = 5 * 8 * 128 * 4 * 2304;
    try t.expectEqual(@as(u64, 6_688) + glm_points, diskBytesPerToken(&glm, 8));
}

test "oneSessionEntryBytes: a cached session is billed with its SSM checkpoints and the capacity it reserved" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const t = std.testing;
    const guard = qsaScoreFusedOffGuard();
    defer guard.deinit();
    var cfg = fixtures.qwen4DeployedTestConfig();
    const ctx: u64 = 262_144;
    const chunk: u64 = 8192;
    cfg.pinned_context = @intCast(ctx);
    const entry = oneSessionEntryBytes(&cfg, 16, ctx, chunk);
    try t.expect(cfg.ssmCheckpointBytes() > 0);
    const fixed = cfg.qsaRingBytes() + retainedSsmCheckpointBytes(&cfg, ctx, 0, chunk);
    // A request at the working context reserves past it (`reservedCacheTokens`), and the entry
    // shares that capacity: no session the context admits outgrows the default.
    for ([_]u64{ ctx - 4000, ctx }) |seq| {
        const reserved = reservedCacheTokens(seq, 8192, PREFILL_CHUNK_LADDER[0], ctx);
        try t.expect(reserved > ctx);
        try t.expect(sessionBytesPerToken(&cfg, 16) * reserved + fixed <= entry);
    }
    try t.expect(defaultPrefixCacheAsk(launch_state.PREFIX_CACHE_MEM_DEFAULT, false, entry) >= entry);
}

test "oneSessionFor: an unnamed budget grows only into room the n-gram table and a full-context prompt leave" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const t = std.testing;
    const guard = qsaScoreFusedOffGuard();
    defer guard.deinit();
    const GiB: u64 = 1 << 30;
    var cfg = fixtures.qwen4DeployedTestConfig();
    const ctx: u64 = 1_048_576;
    cfg.pinned_context = @intCast(ctx);
    const chunk: u64 = 512;
    const weights: u64 = 51 * GiB;
    const table: u64 = 30 * GiB;

    const roomy = oneSessionFor(&cfg, 8, 400 * GiB, weights, table, ctx, chunk);
    try t.expectEqual(roomy.session, roomy.ask());

    // This box: the session fits beside the weights, not beside the table as well.
    const ceiling: u64 = 111_808 * (1 << 20);
    const bare = oneSessionFor(&cfg, 8, ceiling, weights, 0, ctx, chunk);
    try t.expectEqual(bare.session, bare.ask());
    const with_table = oneSessionFor(&cfg, 8, ceiling, weights, table, ctx, chunk);
    const ask = with_table.ask();
    try t.expect(ask > launch_state.PREFIX_CACHE_MEM_DEFAULT and ask < with_table.session);
    try t.expect(weights + table + ask + with_table.bill <= ceiling);

    // A room past half the prompt's bill stops at the half: the trim a session makes when it
    // outgrows the cache copies up to the budget beside the entry it replaces and the live KV.
    const wide = oneSessionFor(&cfg, 8, weights + roomy.bill + roomy.session - 1, weights, 0, ctx, chunk);
    try t.expect(wide.bill / 2 < wide.room);
    try t.expectEqual(wide.bill / 2, wide.ask());

    // No room: nothing grows and the 2 GB default stands.
    const none = oneSessionFor(&cfg, 8, weights + table, weights, table, ctx, chunk);
    try t.expectEqual(@as(u64, 0), none.ask());
    try t.expectEqual(launch_state.PREFIX_CACHE_MEM_DEFAULT, defaultPrefixCacheAsk(launch_state.PREFIX_CACHE_MEM_DEFAULT, false, none.ask()));
}

test "a qwen4_exp SSD restore is credited: its first grow bills one eval window of the restored rows" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    // The SSD tier fills buffers the slot owns at exactly the restored length (`LookupResult.slot_owned`),
    // so the first append grows every layer beside its old buffer and frees that buffer at the layer's
    // eval window. A share is billed the whole copy instead.
    const t = std.testing;
    const guard = qsaScoreFusedOffGuard();
    defer guard.deinit();
    var cfg = fixtures.qwen4DeployedTestConfig();
    cfg.pinned_context = 1 << 20;
    const seq: u64 = 400_000;
    const restored: u64 = seq - 2000;
    const disk = WarmPrefix{ .matched_tokens = restored, .capacity_tokens = restored, .will_donate = true };
    const shared = WarmPrefix{ .matched_tokens = restored, .capacity_tokens = restored };
    const d = prefillRequestTerms(&cfg, seq, 2048, 8, 1024, disk);
    try t.expectEqual(restored * kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), 8), d.shared_resident_bytes);
    // Twelve caching layers, one inside each four-layer eval window.
    try t.expectEqual(d.shared_resident_bytes / 12, d.grow_coexist_bytes);
    try t.expectEqual(
        prefillNeededAtChunk(&cfg, seq, 2048, 8, 1024, shared) - (d.shared_resident_bytes - d.grow_coexist_bytes) * 5 / 4,
        prefillNeededAtChunk(&cfg, seq, 2048, 8, 1024, disk),
    );
}

test "bf16 expert load admission includes 65k KV and QSA serving terms" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const fused_guard = qsaScoreFusedOffGuard();
    defer fused_guard.deinit();
    var config = fixtures.qwen4RequestTestConfig();
    config.expert_streaming = true;
    config.expert_fill_peak_bytes = 11_852_000_000;
    const saved_ctx = launch_state.server_config.max_context_size;
    defer launch_state.server_config.max_context_size = saved_ctx;
    launch_state.server_config.max_context_size = 65_536;
    const saved_kv = launch_state.configured_kv_quant;
    defer launch_state.configured_kv_quant = saved_kv;
    launch_state.configured_kv_quant = trunk.KVQuantConfig.affine(8);
    const serving = expertStreamingServingBytes(&config);
    try std.testing.expect(serving >= kvBytesPerTokenAtBits(config.kvBytesPerToken(), 8) * 65_536);
    const fixed: u64 = 10_000_000_000 + 16_000_000_000 + 60_000_000_000 + 536_870_912 + config.expert_fill_peak_bytes;
    try std.testing.expect(expertStreamingLoadFits(fixed + serving, 10_000_000_000, 16_000_000_000, 60_000_000_000, 536_870_912, config.expert_fill_peak_bytes, serving));
    try std.testing.expect(!expertStreamingLoadFits(fixed + serving - 1, 10_000_000_000, 16_000_000_000, 60_000_000_000, 536_870_912, config.expert_fill_peak_bytes, serving));
}

test "a streamed load bills the narrowest rung the per-request ladder can take" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const fused_guard = qsaScoreFusedOffGuard();
    defer fused_guard.deinit();
    var config = fixtures.qwen4RequestTestConfig();
    config.expert_streaming = true;
    // The fit check runs before the post-load sizer pins a chunk.
    config.pinned_prefill_chunk = 0;
    const saved_ctx = launch_state.server_config.max_context_size;
    defer launch_state.server_config.max_context_size = saved_ctx;
    launch_state.server_config.max_context_size = 32_768;
    const saved_kv = launch_state.configured_kv_quant;
    defer launch_state.configured_kv_quant = saved_kv;
    launch_state.configured_kv_quant = trunk.KVQuantConfig.affine(8);
    const saved_source = settings_resolve.prefill_chunk.source;
    const saved_chunk = settings_resolve.prefill_chunk.value;
    defer {
        settings_resolve.prefill_chunk.source = saved_source;
        settings_resolve.prefill_chunk.value = saved_chunk;
    }
    const saved_per_request = bills.per_request_chunk_override;
    defer bills.per_request_chunk_override = saved_per_request;
    settings_resolve.prefill_chunk.source = .default;
    bills.per_request_chunk_override = true;
    const floor: u64 = PREFILL_CHUNK_LADDER[PREFILL_CHUNK_LADDER.len - 1];
    const at = struct {
        fn bill(c: *const model_mod.ModelConfig, chunk: u64) u64 {
            return prefillNeededAtChunk(c, @max(32_768, chunk), 2048, 8, chunk, .{}) -| c.expert_fill_peak_bytes;
        }
    }.bill;
    try std.testing.expectEqual(at(&config, floor), expertStreamingServingBytes(&config));
    try std.testing.expect(at(&config, floor) < at(&config, 8192));
    // An explicit width caps the ladder, so the load proves the floor, or a narrower flag.
    settings_resolve.prefill_chunk.source = .flag;
    settings_resolve.prefill_chunk.value = 4096;
    try std.testing.expectEqual(at(&config, floor), expertStreamingServingBytes(&config));
    settings_resolve.prefill_chunk.value = 256;
    try std.testing.expectEqual(at(&config, 256), expertStreamingServingBytes(&config));
    // With the ladder off the flag is a pin again.
    settings_resolve.prefill_chunk.value = 4096;
    bills.per_request_chunk_override = false;
    try std.testing.expectEqual(at(&config, 4096), expertStreamingServingBytes(&config));
    bills.per_request_chunk_override = true;
    settings_resolve.prefill_chunk.source = .default;
    config.pinned_prefill_chunk = 1024;
    try std.testing.expectEqual(at(&config, 1024), expertStreamingServingBytes(&config));
    config.pinned_prefill_chunk = 0;
    bills.per_request_chunk_override = false;
    try std.testing.expectEqual(at(&config, floor), expertStreamingServingBytes(&config));
}

test "qwen4RequestTestConfig does not leak qsa_score_fused_override" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const saved = qwen4_qsa_select.qsa_score_fused_override;
    defer qwen4_qsa_select.qsa_score_fused_override = saved;
    qwen4_qsa_select.qsa_score_fused_override = null;
    _ = fixtures.qwen4RequestTestConfig();
    try std.testing.expectEqual(@as(?bool, null), qwen4_qsa_select.qsa_score_fused_override);
}

test "Flash-Next's request bill brackets the peaks measured on the M1 Max, with no padding beyond a small net" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const saved_ctx = launch_state.server_config.max_context_size;
    defer launch_state.server_config.max_context_size = saved_ctx;
    launch_state.server_config.max_context_size = 100000;
    const saved_width = settings_resolve.prefill_chunk.value;
    defer settings_resolve.prefill_chunk.value = saved_width;
    settings_resolve.prefill_chunk.value = 512;
    const cfg = fixtures.qwen4MeasuredTestConfig();
    // Live `peak_bytes - active_bytes` after one prompt per boot (512 rung, kv8, no MTP, SSD tier on, 16 new
    // tokens), with each landed checkpoint already moved to the SSD.
    const measured = [_]struct { tokens: u64, peak_mib: u64 }{
        .{ .tokens = 2857, .peak_mib = 935 },
        .{ .tokens = 8505, .peak_mib = 984 },
        .{ .tokens = 16874, .peak_mib = 1196 },
        .{ .tokens = 33958, .peak_mib = 1381 },
        .{ .tokens = 59314, .peak_mib = 1976 },
    };
    for (measured) |m| {
        const bill = prefillNeededAtChunk(&cfg, m.tokens, 16, 8, 512, .{});
        const peak = m.peak_mib * (1 << 20);
        // Under-billing is a Metal OOM: at least a tenth over the peak. The upper bar keeps the net small.
        try std.testing.expect(bill >= peak + peak / 10);
        try std.testing.expect(bill <= peak + peak * 2 / 5);
    }
}

test "Flash-Next's streamed request bill brackets the streamed peaks measured on the M1 Max" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const saved_ctx = launch_state.server_config.max_context_size;
    defer launch_state.server_config.max_context_size = saved_ctx;
    launch_state.server_config.max_context_size = 49152;
    var cfg = fixtures.qwen4MeasuredTestConfig();
    cfg.expert_streaming = true;
    cfg.expert_fill_peak_bytes = model_load.billedFillPeakBytes(&cfg, 1_848_000_000);
    // Live `peak_bytes - active_bytes` after one cold prompt per width, `--ssd-budget-gb 23`, kv8, no MTP, SSD tier on.
    const measured = [_]struct { tokens: u64, chunk: u64, peak_mib: u64 }{
        .{ .tokens = 7987, .chunk = 1024, .peak_mib = 1224 },
        .{ .tokens = 45106, .chunk = 512, .peak_mib = 1683 },
    };
    for (measured) |m| {
        const bill = prefillNeededAtChunk(&cfg, m.tokens, 16, 8, m.chunk, .{});
        const peak = m.peak_mib << 20;
        try std.testing.expect(bill >= peak + peak / 10);
        try std.testing.expect(bill <= peak + peak * 2 / 5);
    }
}

test "the advertised Flash-Next context is the largest one the request bill admits" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const saved_ctx = launch_state.server_config.max_context_size;
    defer launch_state.server_config.max_context_size = saved_ctx;
    launch_state.server_config.max_context_size = 0;
    const saved_disk = launch_state.prefix_cache_disk_bytes;
    defer launch_state.prefix_cache_disk_bytes = saved_disk;
    launch_state.prefix_cache_disk_bytes = 20 << 30;
    const saved_mtp = launch_state.configured_mtp;
    defer launch_state.configured_mtp = saved_mtp;
    launch_state.configured_mtp = false;
    var cfg = fixtures.qwen4MeasuredTestConfig();
    cfg.pinned_prefill_chunk = 512;
    const MiB: u64 = 1 << 20;
    const ceiling: u64 = 28976 * MiB; // 30000 MiB wired limit less the 1 GiB margin
    const active: u64 = 27_378_205_274; // the measured lean pack's resident bytes
    // The SSD writer's host bytes sit outside the wired limit: the 1 GiB margin under it is the only reserve.
    const budget = ceiling - active;
    const raw = memoryContextAt(&cfg, ceiling, active);
    const fits = struct {
        fn at(c: *const model_mod.ModelConfig, ctx: u64, avail: u64) bool {
            var sized = c.*;
            sized.pinned_context = @intCast(ctx);
            return prefillNeededAtChunk(&sized, ctx, 0, 8, 512, .{}) <= avail;
        }
    };
    // What admission accepts is what the sizer advertises, to within the search step.
    try std.testing.expect(raw > 0);
    try std.testing.expect(fits.at(&cfg, raw, budget));
    try std.testing.expect(!fits.at(&cfg, @as(u64, raw) + 1024, budget));
}

test "the load-time budget is reproducible: free RAM at load does not move it" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // Two boots 11 minutes apart resolved the same 10 GB ask to 1076 and 9757 MB off the live ceiling.
    const t = std.testing;
    const cfg = fixtures.qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const GB: u64 = 1000 * 1000 * 1000;
    const weights: u64 = 70 * GB;
    const ctx_tokens: u64 = 1_048_576; // the deployed --ctx-size
    const static_ceiling: u64 = 120_000 * 1024 * 1024; // the wired limit, both boots
    // `requested == 0` is the arm where the clamp alone decides: the budget IS the headroom.
    const uncapped: u64 = 0;

    const busy = physicalMemoryCeiling(static_ceiling, weights, 108_700 * 1000 * 1000 -| weights);
    const quiet = physicalMemoryCeiling(static_ceiling, weights, 116_600 * 1000 * 1000 -| weights);
    try t.expect(busy < quiet);

    const on_busy = planHotCache(&cfg, kv_bits, busy, weights, ctx_tokens, 0, uncapped, 0);
    const on_quiet = planHotCache(&cfg, kv_bits, quiet, weights, ctx_tokens, 0, uncapped, 0);
    try t.expect(on_busy.budget != on_quiet.budget);
    try t.expect(on_quiet.budget > on_busy.budget);

    // Billed against the static term, both boots agree.
    const a = planHotCache(&cfg, kv_bits, static_ceiling, weights, ctx_tokens, 0, uncapped, 0);
    const b = planHotCache(&cfg, kv_bits, static_ceiling, weights, ctx_tokens, 0, uncapped, 0);
    try t.expectEqual(a.budget, b.budget);
    try t.expect(a.budget >= on_quiet.budget);

    const capped_a = planHotCache(&cfg, kv_bits, static_ceiling, weights, ctx_tokens, 0, 24 * GB, 0);
    const capped_b = planHotCache(&cfg, kv_bits, static_ceiling, weights, ctx_tokens, 0, 24 * GB, 0);
    try t.expectEqual(capped_a.budget, capped_b.budget);
    try t.expect(capped_a.budget <= a.budget);

    try t.expectEqual(static_ceiling, physicalMemoryCeiling(static_ceiling, weights, 400 * GB));
}

test "an auto boot sizes the SAME context whatever the cache ask (live check #6)" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // `--prefix-cache-mem 60GB` collapsed the advertised context to 870 tokens (ctx KV billed at 26 MB).
    const t = std.testing;
    const cfg = fixtures.qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const MiB: u64 = 1 << 20;
    const active: u64 = 69_827 * MiB; // backed out of the live clamp line
    const live_ceiling: u64 = 109_395 * MiB;
    const per_tok: u64 = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits) +| statePerTokenBilled(&cfg);
    const transient: u64 = prefillTransientReserve(&cfg, kv_bits, 4096);
    const cap: u32 = cfg.contextCap();

    // The sizing reserve is a constant, so the answer cannot move with the ask.
    const ctx_default = resolvedContextForLoad(0, 0, live_ceiling, active, CTX_SIZING_CACHE_RESERVE, transient, per_tok, cap, 0);
    const ctx_big_ask = resolvedContextForLoad(0, 0, live_ceiling, active, CTX_SIZING_CACHE_RESERVE, transient, per_tok, cap, 0);
    try t.expectEqual(ctx_default, ctx_big_ask);
    try t.expect(ctx_default > 900_000); // a real context, not the floor

    // The defect, both spellings: sizing against the granted budget...
    const vs_granted = resolvedContextForLoad(0, 0, live_ceiling, active, 48_673 * MiB, transient, per_tok, cap, 0);
    try t.expect(vs_granted <= 1024);
    // ...and against the raw ask, which saturates usable to zero.
    const vs_raw_ask = resolvedContextForLoad(0, 0, live_ceiling, active, 60 * 1024 * MiB, transient, per_tok, cap, 0);
    try t.expect(vs_raw_ask <= 1024);
    try t.expect(ctx_default > vs_granted * 500);
}

test "the clamp bills the context that will be SERVED, not the placeholder" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    // With a placeholder context the clamp hands back nearly the whole ask.
    const t = std.testing;
    qwen4_qsa_select.qsa_score_fused_override = false;
    defer qwen4_qsa_select.qsa_score_fused_override = null;
    var cfg = fixtures.qwen4RequestTestConfig();
    cfg.mtp_override = false; // the MTP head's own term: `the load-time bill prices the MTP head ...`
    const kv_bits: u64 = 8;
    const MiB: u64 = 1 << 20;
    const active: u64 = 69_827 * MiB;
    const static_ceiling: u64 = 120_000 * MiB;
    const ask: u64 = 60 * 1024 * MiB;

    const placeholder = planHotCache(&cfg, kv_bits, static_ceiling, active, 1024, 0, ask, 4096);
    const real = planHotCache(&cfg, kv_bits, static_ceiling, active, 1_048_576, 0, ask, 4096);

    try t.expect(placeholder.budget > 45_000 * MiB); // reproduces the live 48,673
    // 13,056 KV + 5,376 state (one history copy + the f32 score bank).
    try t.expectEqual(@as(u64, 1_048_576) * (13_056 + 2_304) + cfg.qsaRingBytes(), real.ctx_kv);

    // Both budgets are their headroom, so the whole difference is the session each billed.
    try t.expectEqual(placeholder.budget - real.budget, real.ctx_kv - placeholder.ctx_kv);
    try t.expect(real.budget < placeholder.budget);
    try t.expect(placeholder.ctx_kv < 30 * MiB);
    try t.expect(real.ctx_kv > 15_000 * MiB);
    try t.expect(real.budget > 0);
}

test "with no --kv-quant, the per-token KV bill and the auto-context follow the kv8 engine default" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const t = std.testing;
    var cfg = fixtures.qwen4RequestTestConfig();
    const saved_kv = launch_state.configured_kv_quant;
    defer launch_state.configured_kv_quant = saved_kv;

    // One fixed memory reading for every arm: live free RAM moves between two reads.
    const ceiling: u64 = 96 << 30;
    const active: u64 = 90 << 30; // tight enough that the checkpoint's own cap does not bind
    launch_state.configured_kv_quant = null;
    try t.expectEqual(@as(u64, 8), defaultKvBits(&cfg));
    try t.expectEqual(sessionBytesPerToken(&cfg, 8), sessionBytesPerToken(&cfg, defaultKvBits(&cfg)));
    const ctx_default = memoryContextAt(&cfg, ceiling, active);

    launch_state.configured_kv_quant = trunk.KVQuantConfig.affine(8);
    try t.expectEqual(ctx_default, memoryContextAt(&cfg, ceiling, active));
    launch_state.configured_kv_quant = trunk.KVQuantConfig.dense;
    try t.expect(memoryContextAt(&cfg, ceiling, active) < ctx_default);

    // A per-model setting still outranks the default.
    launch_state.configured_kv_quant = null;
    cfg.kv_quant_override = trunk.KVQuantConfig.dense;
    try t.expectEqual(@as(u64, 16), defaultKvBits(&cfg));
}

test "clampReserveWidth: the load-time reserve is a promise to the FIRST request" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // The ceiling inversion: same ask, `iogpu.wired_limit_mb 120000` got a 3.6x smaller cache
    // because the wider rung's reserve grew faster than the headroom. Billing the ladder floor
    // makes the budget monotone in the ceiling.
    const t = std.testing;
    const cfg = fixtures.qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const MiB: u64 = 1 << 20;
    const GiB: u64 = 1 << 30;
    const weights: u64 = 70 * GiB;
    const ctx_tokens: u64 = 1_048_576;
    const floor_rung = PREFILL_CHUNK_LADDER[PREFILL_CHUNK_LADDER.len - 1];

    const low = planHotCache(&cfg, kv_bits, weights + 28_909 * MiB, weights, ctx_tokens, 0, 10 * GiB, 0);
    const high = planHotCache(&cfg, kv_bits, weights + 43_000 * MiB, weights, ctx_tokens, 0, 10 * GiB, 0);
    try t.expectEqual(floor_rung, low.reserve_chunk);
    try t.expectEqual(floor_rung, high.reserve_chunk);
    try t.expectEqual(prefillTransientReserve(&cfg, kv_bits, floor_rung), low.reserve);
    // A bigger ceiling never buys a smaller cache.
    try t.expect(high.budget >= low.budget);
    try t.expect(low.chunk >= low.reserve_chunk);

    var prev: u64 = 0;
    var mb: u64 = 24_000;
    while (mb <= 60_000) : (mb += 1_000) {
        const p = planHotCache(&cfg, kv_bits, weights + mb * MiB, weights, ctx_tokens, 0, 10 * GiB, 0);
        try t.expect(p.budget >= prev);
        prev = p.budget;
    }

    // An explicit --prefill-chunk caps the ladder, so the clamp still reserves the floor.
    settings_resolve.prefill_chunk.value = 4096;
    settings_resolve.prefill_chunk.source = .flag;
    const capped = planHotCache(&cfg, kv_bits, weights + 43_000 * MiB, weights, ctx_tokens, 0, 10 * GiB, explicitPrefillChunk());
    try t.expectEqual(@as(u32, 4096), capped.chunk);
    try t.expectEqual(floor_rung, capped.reserve_chunk);
    try t.expectEqual(prefillTransientReserve(&cfg, kv_bits, floor_rung), capped.reserve);
    // With the ladder off it is a pin, billed as-is.
    bills.per_request_chunk_override = false;
    const pinned = planHotCache(&cfg, kv_bits, weights + 43_000 * MiB, weights, ctx_tokens, 0, 10 * GiB, explicitPrefillChunk());
    bills.per_request_chunk_override = null;
    settings_resolve.prefill_chunk.value = 8192;
    settings_resolve.prefill_chunk.source = .default;
    try t.expectEqual(@as(u32, 4096), pinned.reserve_chunk);
    try t.expectEqual(prefillTransientReserve(&cfg, kv_bits, 4096), pinned.reserve);

    // Gate off: the clamp reserves at the sizer's rung.
    bills.per_request_chunk_override = false;
    defer bills.per_request_chunk_override = null;
    const off = planHotCache(&cfg, kv_bits, weights + 28_909 * MiB, weights, ctx_tokens, 0, 10 * GiB, 0);
    try t.expectEqual(off.chunk, off.reserve_chunk);
    try t.expectEqual(prefillTransientReserve(&cfg, kv_bits, off.chunk), off.reserve);
    bills.per_request_chunk_override = null;

    var other = fixtures.qwen4RequestTestConfig();
    other.model_type = "not_a_served_arch";
    const neutral = planHotCache(&other, kv_bits, weights + 28_909 * MiB, weights, ctx_tokens, 0, 10 * GiB, 0);
    try t.expectEqual(neutral.chunk, neutral.reserve_chunk);
}

test "the per-token widths the request chooser is built on" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    // Pin the two constants every number in the story is derived from.
    const t = std.testing;
    const cfg = fixtures.qwen4RequestTestConfig();
    try t.expectEqual(@as(u32, 12), cfg.attnCacheLayerCount());
    try t.expectEqual(@as(u64, 13_056), kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), 8));
    qwen4_qsa_select.qsa_score_fused_override = false;
    defer qwen4_qsa_select.qsa_score_fused_override = null;
    try t.expectEqual(@as(u64, 768), cfg.qsaHistoryBytesPerToken());
    try t.expectEqual(@as(u64, 1_536), cfg.qsaScoreBankBytesPerToken());
    try t.expectEqual(@as(u64, 2_304), statePerTokenBilled(&cfg));
    try t.expectEqual(@as(u64, 15_360 * (1 << 20)), (13_056 + 2_304) * @as(u64, 1_048_576));
}

test "qsaMaskBytes: a qwen4_exp twin bills the QSA mask and steps a rung the same trunk without qwen4_exp keeps" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const t = std.testing;
    var twin = model_mod.ModelConfig{};
    twin.num_hidden_layers = 40;
    twin.num_attention_heads = 24;
    twin.num_key_value_heads = 2;
    twin.head_dim = 256;
    twin.hidden_size = 2560;
    twin.intermediate_size = 9216;
    twin.intermediate_size_declared = true;
    twin.quant_bits = 4;
    twin.max_position_embeddings = 262144;
    var q4 = twin;
    q4.indexer_budget = 2048;
    q4.indexer_n_heads = 4;
    q4.indexer_head_dim = 128;
    q4.indexer_compress_ratio = 4;
    try t.expectEqual(@as(u64, 0), qsaMaskBytes(&twin, 4096, 25000));
    // Decode/verify widths build the dense mask; prefill widths gather by
    // block and bill the bounded score sheet instead of 4 B x rows x keys.
    try t.expectEqual(@as(u64, 4 * 8 * 25000 * 5 / 4), qsaMaskBytes(&q4, 8, 25000));
    // Only the arch that gathers bills the sheet; the same config unclaimed bills the dense mask.
    var gathering = q4;
    gathering.model_type = "qwen4_exp";
    qwen4_qsa_gather.qsa_gather_override = true;
    defer qwen4_qsa_gather.qsa_gather_override = null;
    const saved_fused = qwen4_qsa_select.qsa_score_fused_override;
    const saved_nax = nax_gate.vqmm_nax_probe_override;
    defer {
        qwen4_qsa_select.qsa_score_fused_override = saved_fused;
        nax_gate.vqmm_nax_probe_override = saved_nax;
    }
    nax_gate.vqmm_nax_probe_override = false;
    qwen4_qsa_select.qsa_score_fused_override = true;
    const gathered_composed = qsaMaskBytes(&gathering, 4096, 250_000);
    try t.expect(gathered_composed > 0);
    try t.expect(gathered_composed < 4 * 4096 * 250_000);
    try t.expectEqual(qwen4_qsa_attn.qsaPrefillTransientBytes(4, 4096, 250_000, 4, 128), gathered_composed);
    nax_gate.vqmm_nax_probe_override = null;
    qwen4_qsa_select.qsa_score_fused_override = true;
    const gathered_fused = qsaMaskBytes(&gathering, 4096, 250_000);
    try t.expectEqual(qwen4_qsa_attn.qsaPrefillTransientBytes(4, 4096, 250_000, 4, 128), gathered_fused);
    if (qwen4_qsa_select.qsaScoreFusedActive()) {
        try t.expect(gathered_fused < gathered_composed);
    }
    qwen4_qsa_select.qsa_score_fused_override = false;
    nax_gate.vqmm_nax_probe_override = false;
    const gathered = qsaMaskBytes(&gathering, 4096, 250_000);
    try t.expectEqual(gathered_composed, gathered);
    const twin_bill = prefillTransientReserve(&twin, 16, 8192);
    const q4_bill = prefillTransientReserve(&q4, 16, 8192);
    try t.expectEqual(twin_bill + qsaMaskBytes(&q4, 8192, 8192) + q4.qsaRingBytes() * 5 / 4, q4_bill);
    // A ceiling whose quarter-share sits between the two bills at 8192.
    const weights: u64 = 70_000_000_000;
    const ceiling = weights + (twin_bill + q4_bill) / 2 * 4;
    try t.expectEqual(@as(u32, 8192), resolvePrefillChunk(&twin, 16, ceiling, weights, 0, 0));
    try t.expectEqual(@as(u32, 4096), resolvePrefillChunk(&q4, 16, ceiling, weights, 0, 0));
}

test "qwen4ExpOomConfig does not leak qsa_score_fused_override" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const saved = qwen4_qsa_select.qsa_score_fused_override;
    defer qwen4_qsa_select.qsa_score_fused_override = saved;
    qwen4_qsa_select.qsa_score_fused_override = null;
    _ = fixtures.qwen4ExpOomConfig();
    try std.testing.expectEqual(@as(?bool, null), qwen4_qsa_select.qsa_score_fused_override);
}

test "the 458k prefill's two unbilled terms are billed: retained checkpoints and reserved capacity" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    // These pins price a cache with no SSD tier behind it: every checkpoint stays in RAM.
    const no_tier = TierGuard.pin(false, 0);
    defer no_tier.restore();
    const whole_rebuild = qsaWholeRebuildGuard();
    defer whole_rebuild.deinit();
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const t = std.testing;
    // A 458,832-token cold prefill was admitted against 26.6 GB of headroom and died mid-prefill.
    const cfg = fixtures.qwen4ExpOomConfig();
    const seq: u64 = 458_832;
    const chunk: u64 = 4096;

    // 1. SSM checkpoints: 36 GDN layers x (state + conv window), up to --ssm-checkpoint-max.
    const per_cp = cfg.ssmCheckpointBytes();
    // 36 x (3,145,728 + 61,440) B = 115.46 MB: the state is f32.
    try t.expect(per_cp > 115_000_000 and per_cp < 116_000_000);
    const saved_cp_max = launch_state.ssm_checkpoint_max;
    launch_state.ssm_checkpoint_max = 32;
    defer launch_state.ssm_checkpoint_max = saved_cp_max;
    const held = retainedSsmCheckpointBytes(&cfg, seq, 0, chunk);
    try t.expectEqual(@as(u64, launch_state.ssm_checkpoint_max) * per_cp, held);
    try t.expect(held > 1_800_000_000);
    // It saturates: 393k holds the same 32 as 458k.
    try t.expectEqual(held, retainedSsmCheckpointBytes(&cfg, 393_216, 0, chunk));
    // An arch with no linear layers is billed nothing.
    var dense = model_mod.ModelConfig{};
    dense.num_hidden_layers = 32;
    dense.num_attention_heads = 32;
    try t.expectEqual(@as(u64, 0), dense.ssmCheckpointBytes());
    try t.expectEqual(@as(u64, 0), retainedSsmCheckpointBytes(&dense, seq, 0, chunk));

    // 2. Reserved capacity: prompt + max_tokens + one chunk; the guard bills the headroom beyond
    // the prompt's own rows, not a second copy of the cache.
    const terms = prefillRequestTerms(&cfg, seq, 2048, 8, chunk, .{});
    const kv_per_tok = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), 8);
    try t.expectEqual((2048 + chunk) * kv_per_tok, terms.reserved_kv_bytes);
    try t.expect(terms.reserved_kv_bytes * 4 < seq * kv_per_tok); // headroom, not a copy
    // The indexer history is billed at one copy plus the score bank (`statePerTokenBilled`).
    try t.expectEqual(cfg.qsaHistoryBytesPerToken() + cfg.qsaScoreBankBytesPerToken(), statePerTokenBilled(&cfg));
    try t.expectEqual((seq + 2048 + chunk) * statePerTokenBilled(&cfg), terms.state_bytes);
    try t.expectEqual(held, terms.checkpoint_bytes);

    // 3. The bill moved by more than 3 GB, toward a 503 instead of a corpse.
    const with_terms = prefillMemoryNeeded(seq, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.kvBytesPerToken(), cfg.head_dim, cfg.prefillScoreHeadDim(), cfg.hidden_size, prefillFfnWidth(&cfg), 8, chunk, cfg.prefillAttnKeys(seq), prefillStreamBytesPerToken(&cfg), prefillDequantWeightBytes(&cfg), terms);
    const without = prefillMemoryNeeded(seq, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.kvBytesPerToken(), cfg.head_dim, cfg.prefillScoreHeadDim(), cfg.hidden_size, prefillFfnWidth(&cfg), 8, chunk, cfg.prefillAttnKeys(seq), prefillStreamBytesPerToken(&cfg), prefillDequantWeightBytes(&cfg), .{});
    try t.expect(with_terms > without + 3_000_000_000);
    // The sentinel runs through the same path: an omitted max_tokens must not move the bill past the context.
    const sentinel: u64 = request_policy.omittedMaxTokensDefault(&cfg);
    const with_sentinel = prefillRequestTerms(&cfg, seq, sentinel, 8, chunk, .{});
    const with_explicit = prefillRequestTerms(&cfg, seq, 2048, 8, chunk, .{});
    try t.expect(with_sentinel.reserved_kv_bytes < 8 * 1024 * 1024 * 1024);
    try t.expect(with_sentinel.state_bytes < 4 * with_explicit.state_bytes);
    try t.expectEqual(with_explicit.checkpoint_bytes, with_sentinel.checkpoint_bytes);

    // A short prompt is untouched.
    const short = prefillRequestTerms(&cfg, 4096, 2048, 8, chunk, .{});
    try t.expectEqual(@as(u64, 0), short.reserved_kv_bytes);
    try t.expectEqual(4096 * statePerTokenBilled(&cfg), short.state_bytes);
}

test "the QSA history is billed at the one copy a slot HOLDS, plus the score bank" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const t = std.testing;
    const cfg = fixtures.qwen4ExpOomConfig();
    qwen4_qsa_select.qsa_score_fused_override = false;
    defer qwen4_qsa_select.qsa_score_fused_override = null;
    const one = cfg.qsaHistoryBytesPerToken();
    const bank = cfg.qsaScoreBankBytesPerToken();
    try t.expect(one > 0);
    try t.expect(bank > 0);
    try t.expectEqual(one + bank, statePerTokenBilled(&cfg));

    const seq: u64 = 458_832;
    const terms = prefillRequestTerms(&cfg, seq, 2048, 8, 4096, .{});
    try t.expectEqual((seq + 2048 + 4096) * (one + bank), terms.state_bytes);

    var dense = model_mod.ModelConfig{};
    dense.num_hidden_layers = 32;
    dense.num_attention_heads = 32;
    try t.expectEqual(@as(u64, 0), statePerTokenBilled(&dense));
}

test "adaptivePrefillWidth: a step-down is immediate, by as many rungs as it takes, and never 0" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // The down direction does not wait for a trend or a margin.
    const t = std.testing;
    const cfg = fixtures.qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const seq: u64 = 1_048_576;
    const kv: u64 = seq;
    const cap = adaptCapFor(&cfg, seq);

    {
        var st: mtp_plan.AdaptiveWidthState = .{};
        const h = prefillChunkCost(&cfg, kv_bits, 2048, kv);
        try t.expect(h < prefillChunkCost(&cfg, kv_bits, cap, kv));
        try t.expectEqual(@as(u32, 2048), adaptivePrefillWidth(&cfg, kv_bits, kv, h, cap, cap, &st));
        try t.expect(st.ratcheted);
    }
    // Several rungs in one call.
    {
        var st: mtp_plan.AdaptiveWidthState = .{};
        const h = prefillChunkCost(&cfg, kv_bits, 512, kv);
        try t.expectEqual(@as(u32, 512), adaptivePrefillWidth(&cfg, kv_bits, kv, h, cap, cap, &st));
    }
    // No headroom at all: the ladder floor, never 0 (the loop reads 0 as "no change").
    {
        var st: mtp_plan.AdaptiveWidthState = .{};
        const got = adaptivePrefillWidth(&cfg, kv_bits, kv, 0, cap, cap, &st);
        try t.expectEqual(@as(u32, @intCast(prefill_plan.PREFILL_CHUNK_FLOOR)), got);
        try t.expect(got != 0);
    }
    {
        var st: mtp_plan.AdaptiveWidthState = .{};
        const floor: u32 = @intCast(prefill_plan.PREFILL_CHUNK_FLOOR);
        try t.expectEqual(floor, adaptivePrefillWidth(&cfg, kv_bits, kv, 0, floor, cap, &st));
    }
}

test "adaptivePrefillWidth: HOLD is margin 1.0 — the admitted width is not second-guessed" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // `prefillMemoryNeeded` already carries its own 5/4; a second 1.25 on the hold stepped the
    // 768k prompt down from the width admission admitted.
    const t = std.testing;
    const cfg = fixtures.qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const kv: u64 = 786_432;
    const cap = adaptCapFor(&cfg, 786_432);
    const cost_2k = prefillChunkCost(&cfg, kv_bits, 2048, kv);

    var st: mtp_plan.AdaptiveWidthState = .{};
    try t.expectEqual(@as(u32, 2048), adaptivePrefillWidth(&cfg, kv_bits, kv, cost_2k, 2048, cap, &st));
    try t.expect(!st.ratcheted);
    try t.expect(cost_2k * 5 / 4 > cost_2k);
    var st2: mtp_plan.AdaptiveWidthState = .{};
    try t.expectEqual(@as(u32, 2048), adaptivePrefillWidth(&cfg, kv_bits, kv, cost_2k * 5 / 4 - 1, 2048, cap, &st2));
    var st3: mtp_plan.AdaptiveWidthState = .{};
    try t.expect(adaptivePrefillWidth(&cfg, kv_bits, kv, cost_2k - 1, 2048, cap, &st3) < 2048);
}

test "adaptivePrefillWidth: a widen costs 1.25x AND two consecutive supporting probes" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // Holding is free, growing is a bet: two consecutive probes.
    const t = std.testing;
    const cfg = fixtures.qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const kv: u64 = 524_288;
    const cap = adaptCapFor(&cfg, 524_288);
    // Starts at half the cap so one widen step lands exactly on the cap.
    const from: u32 = cap / 2;
    const cost_up = prefillChunkCost(&cfg, kv_bits, cap, kv);

    var st: mtp_plan.AdaptiveWidthState = .{};
    try t.expectEqual(from, adaptivePrefillWidth(&cfg, kv_bits, kv, cost_up * 4, from, cap, &st));
    try t.expectEqual(@as(u8, 1), st.supporting);
    try t.expectEqual(cap, adaptivePrefillWidth(&cfg, kv_bits, kv, cost_up * 4, from, cap, &st));
    try t.expectEqual(@as(u8, 0), st.supporting); // and the count restarts at the new width

    // Room for the wider chunk's bill but not its margin: never.
    var st2: mtp_plan.AdaptiveWidthState = .{};
    const between = cost_up * 5 / 4 - 1;
    try t.expect(between >= cost_up);
    var i: usize = 0;
    while (i < 8) : (i += 1) {
        try t.expectEqual(from, adaptivePrefillWidth(&cfg, kv_bits, kv, between, from, cap, &st2));
    }

    // A single unsupporting probe resets the run.
    var st3: mtp_plan.AdaptiveWidthState = .{};
    try t.expectEqual(from, adaptivePrefillWidth(&cfg, kv_bits, kv, cost_up * 4, from, cap, &st3));
    try t.expectEqual(from, adaptivePrefillWidth(&cfg, kv_bits, kv, between, from, cap, &st3));
    try t.expectEqual(@as(u8, 0), st3.supporting);
    try t.expectEqual(from, adaptivePrefillWidth(&cfg, kv_bits, kv, cost_up * 4, from, cap, &st3));
}

test "the tail-merge bound scales ONLY where the per-chunk adaptive width is live" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // Every arch but an adaptive-width qwen4_exp keeps the flat `TAIL_MERGE_MAX` at every chunk width.
    const t = std.testing;

    var other = fixtures.qwen4RequestTestConfig();
    other.model_type = "not_a_served_arch";
    try t.expect(!other.perRequestPrefillChunk());
    try t.expect(!adaptivePrefillChunkEnabled(&other));
    const off = adaptivePrefillChunkEnabled(&other);
    try t.expectEqual(prefill_plan.TAIL_MERGE_MAX, prefill_plan.tailMergeMaxFor(512, off));
    try t.expectEqual(prefill_plan.TAIL_MERGE_MAX, prefill_plan.tailMergeMaxFor(2048, off));
    try t.expectEqual(@as(usize, 812), prefill_plan.nextChunkEnd(0, 812, 512, false, 0, 0, off));

    const cfg = fixtures.qwen4RequestTestConfig();
    try t.expect(cfg.perRequestPrefillChunk());
    bills.per_request_chunk_override = true;
    defer bills.per_request_chunk_override = null;
    bills.adaptive_chunk_override = true;
    defer bills.adaptive_chunk_override = null;
    const unpinned = explicitPrefillChunk() == 0;
    try t.expectEqual(unpinned, adaptivePrefillChunkEnabled(&cfg));
    try t.expectEqual(@as(usize, 64), prefill_plan.tailMergeMaxFor(512, true));
    try t.expectEqual(@as(usize, 512), prefill_plan.nextChunkEnd(0, 812, 512, false, 0, 0, true));
    // An operator pin turns the scaling back off.
    bills.adaptive_chunk_override = false;
    try t.expect(!adaptivePrefillChunkEnabled(&cfg));
    try t.expectEqual(prefill_plan.TAIL_MERGE_MAX, prefill_plan.tailMergeMaxFor(512, adaptivePrefillChunkEnabled(&cfg)));
}

test "adaptivePrefillWidth: the ratchet is one-way, and the cap is a ceiling" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const t = std.testing;
    const cfg = fixtures.qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const kv: u64 = 1_048_576;
    const cap = adaptCapFor(&cfg, 1_048_576);

    // Step down once, then hand it the whole machine: it stays (a re-widen would be a metronome).
    var st: mtp_plan.AdaptiveWidthState = .{};
    const narrowed = adaptivePrefillWidth(&cfg, kv_bits, kv, prefillChunkCost(&cfg, kv_bits, 1024, kv), cap, cap, &st);
    try t.expect(narrowed < cap);
    try t.expect(st.ratcheted);
    var i: usize = 0;
    while (i < 8) : (i += 1) {
        try t.expectEqual(narrowed, adaptivePrefillWidth(&cfg, kv_bits, kv, std.math.maxInt(u64) / 2, narrowed, cap, &st));
    }

    // Nothing above the arch cap, ever.
    var st2: mtp_plan.AdaptiveWidthState = .{};
    i = 0;
    while (i < 8) : (i += 1) {
        try t.expectEqual(cap, adaptivePrefillWidth(&cfg, kv_bits, kv, std.math.maxInt(u64) / 2, cap, cap, &st2));
    }
    // A cap of 0 (the unchunked vision arm) is "do not touch the width".
    var st3: mtp_plan.AdaptiveWidthState = .{};
    try t.expectEqual(@as(u32, 2048), adaptivePrefillWidth(&cfg, kv_bits, kv, 0, 2048, 0, &st3));
}

test "adaptivePrefillWidth: a 1M prompt walks DOWN as the box fills, and a small one never moves" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // Sweep: the width is the widest rung that fits, only narrows within one prefill, and is always forwardable.
    const t = std.testing;
    const cfg = fixtures.qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const MiB: u64 = 1 << 20;
    const kv: u64 = 1_048_576;
    const cap = adaptCapFor(&cfg, 1_048_576);

    var st: mtp_plan.AdaptiveWidthState = .{};
    var width: u32 = cap;
    var seen_narrower: usize = 0;
    var headroom: u64 = 50 * 1024 * MiB; // the wired-limit box, quiet
    var step: usize = 0;
    while (step < 64) : (step += 1) {
        const next = adaptivePrefillWidth(&cfg, kv_bits, kv, headroom, width, cap, &st);
        try t.expect(next <= width); // never widens after the first step-down
        try t.expect(next >= @as(u32, @intCast(prefill_plan.PREFILL_CHUNK_FLOOR)));
        try t.expect(next <= cap);
        try t.expect(prefillChunkCost(&cfg, kv_bits, next, kv) <= headroom or
            next == @as(u32, @intCast(prefill_plan.PREFILL_CHUNK_FLOOR)));
        if (next < width) seen_narrower += 1;
        width = next;
        headroom -|= 1024 * MiB; // a co-tenant, or the accreting state
    }
    try t.expect(seen_narrower >= 2);
    try t.expectEqual(@as(u32, @intCast(prefill_plan.PREFILL_CHUNK_FLOOR)), width);

    // A 32k prompt on the same box: the width never moves.
    var st2: mtp_plan.AdaptiveWidthState = .{};
    const cap32 = adaptCapFor(&cfg, 32_768);
    var w32: u32 = cap32;
    step = 0;
    while (step < 16) : (step += 1) {
        w32 = adaptivePrefillWidth(&cfg, kv_bits, 32_768, 50 * 1024 * MiB, w32, cap32, &st2);
        try t.expectEqual(cap32, w32);
    }
    try t.expect(!st2.ratcheted);
}

test "adaptivePrefillWidth: the widen prices the QSA sheet at the LIVE KV, not at one chunk" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const whole_rebuild = qsaWholeRebuildGuard();
    defer whole_rebuild.deinit();
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // `prefillTransientReserve` bills the QSA sheet at `kv = chunk`; past the indexer budget the
    // sheet costs its whole budget, so a widen decided on that number is unguarded.
    const t = std.testing;
    const cfg = fixtures.qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const cap = adaptCapFor(&cfg, 786_432);
    const short_kv: u64 = 4096;
    const long_kv: u64 = 400_000;

    // The bill is strictly larger at the live KV.
    const at_short = prefillChunkCost(&cfg, kv_bits, cap, short_kv);
    const at_long = prefillChunkCost(&cfg, kv_bits, cap, long_kv);
    try t.expect(at_long > at_short);
    // `prefillTransientReserve` is exactly the `kv = chunk` case.
    try t.expectEqual(
        prefillTransientReserve(&cfg, kv_bits, cap),
        prefillTransientReserveAtKv(&cfg, kv_bits, cap, cap),
    );

    // Headroom that clears the short bill's 1.25 bar but not the long one: widen at 4k, refuse at 400k.
    const headroom: u64 = at_short * 5 / 4 + 1;
    try t.expect(headroom < at_long * 5 / 4);

    // Starts at half the cap so one widen step lands exactly on the cap.
    const from: u32 = cap / 2;
    var st_short: mtp_plan.AdaptiveWidthState = .{};
    try t.expectEqual(from, adaptivePrefillWidth(&cfg, kv_bits, short_kv, headroom, from, cap, &st_short));
    try t.expectEqual(cap, adaptivePrefillWidth(&cfg, kv_bits, short_kv, headroom, from, cap, &st_short));

    var st_long: mtp_plan.AdaptiveWidthState = .{};
    var i: usize = 0;
    while (i < 8) : (i += 1) {
        try t.expectEqual(from, adaptivePrefillWidth(&cfg, kv_bits, long_kv, headroom, from, cap, &st_long));
    }
    try t.expectEqual(@as(u8, 0), st_long.supporting);

    // The hold keeps its margin-1.0 semantics at the live KV.
    const hold = prefillChunkCost(&cfg, kv_bits, 2048, long_kv);
    var st_hold: mtp_plan.AdaptiveWidthState = .{};
    try t.expectEqual(@as(u32, 2048), adaptivePrefillWidth(&cfg, kv_bits, long_kv, hold, 2048, cap, &st_hold));
    var st_down: mtp_plan.AdaptiveWidthState = .{};
    try t.expect(adaptivePrefillWidth(&cfg, kv_bits, long_kv, hold - 1, 2048, cap, &st_down) < 2048);
}

test "adaptivePrefillChunkEnabled: the arch, both kill switches, and any pinned width" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const t = std.testing;
    const cfg = fixtures.qwen4RequestTestConfig();
    var other = fixtures.qwen4RequestTestConfig();
    other.model_type = "not_a_served_arch";

    try t.expect(adaptivePrefillChunkEnabled(&cfg));
    // Another arch never adapts; one switch disables both.
    try t.expect(!adaptivePrefillChunkEnabled(&other));

    bills.per_request_chunk_override = false;
    try t.expect(!adaptivePrefillChunkEnabled(&cfg));
    bills.per_request_chunk_override = null;
    try t.expect(adaptivePrefillChunkEnabled(&cfg));

    bills.adaptive_chunk_override = false;
    try t.expect(!adaptivePrefillChunkEnabled(&cfg));
    bills.adaptive_chunk_override = null;

    // An operator who pinned a width pinned every forward.
    settings_resolve.prefill_chunk.value = 4096;
    settings_resolve.prefill_chunk.source = .flag;
    defer {
        settings_resolve.prefill_chunk.value = 8192;
        settings_resolve.prefill_chunk.source = .default;
    }
    try t.expect(!adaptivePrefillChunkEnabled(&cfg));
}

test "the advertised context does not move with the cache ask" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // Sizing the context against the ask collapsed it (60GB ask -> 870 tokens); sizing against the
    // resolved budget was a one-step loop. A constant reserve closes both.
    const t = std.testing;
    const cfg = fixtures.qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const MiB: u64 = 1 << 20;
    const active: u64 = 69_827 * MiB;
    const live_ceiling: u64 = 109_395 * MiB;
    const per_tok: u64 = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits) +| statePerTokenBilled(&cfg);
    const transient: u64 = prefillTransientReserve(&cfg, kv_bits, 4096);

    // `max_pos` is 0: the checkpoint cap is applied by the caller after the margin.
    const memory_ctx = safeContextForBudget(live_ceiling, active, CTX_SIZING_CACHE_RESERVE +| transient, per_tok, 0);
    try t.expect(memory_ctx > 900_000);

    // The two defect spellings.
    const vs_ask = safeContextForBudget(live_ceiling, active, 60 * 1024 * MiB +| transient, per_tok, 0);
    const vs_resolved = safeContextForBudget(live_ceiling, active, 48_673 * MiB +| transient, per_tok, 0);
    try t.expect(vs_ask <= 1024);
    try t.expect(vs_resolved <= 1024);

    // The pair agrees at the advertised number: both sides go through `autoContextFrom`.
    const advertised = autoContextFrom(memory_ctx, cfg.contextCap(), cfg.defaultContext());
    const clamp_ctx = resolvedContextForLoad(0, 0, live_ceiling, active, CTX_SIZING_CACHE_RESERVE, transient, per_tok, cfg.contextCap(), cfg.defaultContext());
    try t.expectEqual(advertised, clamp_ctx);
}

test "prefillRequestTerms: the admission bill's new terms are qwen4_exp-only" {
    if (!archBuilt("qwen4") or !archBuilt("glm5")) return error.SkipZigTest;
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // The allocator side of the reservation is gated (`prefill_plan.reservedPrefillTokens`); the
    // guard side was not, so off qwen4_exp the bill charged headroom never reserved.
    const t = std.testing;
    const seq: u64 = 200_000; // well past RESERVE_MIN_TOKENS
    const chunk: u64 = 4096;
    const max_tokens: u64 = 8192;
    const kv_bits: u64 = 8;

    // A 27B-class GDN pack: `ssmCheckpointBytes()` is non-zero here.
    var q35 = fixtures.qwen4ExpOomConfig();
    q35.model_type = "not_a_served_arch";
    try t.expect(q35.ssmCheckpointBytes() > 0);
    try t.expect(retainedSsmCheckpointBytes(&q35, seq, 0, chunk) > 0);

    // Supplies the donating case so the qwen4_exp assertion below is falsifiable; the ungated loop reads 0 through `.{}`.
    const warm = WarmPrefix{ .matched_tokens = 100_000, .capacity_tokens = 400_000, .will_donate = true };
    for ([_][]const u8{ "glm5_next", "not_a_served_arch" }) |mt| {
        var cfg = fixtures.qwen4ExpOomConfig();
        cfg.model_type = mt;
        const terms = prefillRequestTerms(&cfg, seq, max_tokens, kv_bits, chunk, warm);
        try t.expectEqual(@as(u64, 0), terms.reserved_kv_bytes);
        try t.expectEqual(@as(u64, 0), terms.checkpoint_bytes);
        try t.expectEqual(@as(u64, 0), terms.state_bytes);
        try t.expectEqual(@as(u64, 0), terms.shared_resident_bytes);
        // `.{}` is the identity: the whole bill is the previous expression.
        try t.expectEqual(
            prefillMemoryNeeded(seq, 24, 2, cfg.kvBytesPerToken(), 256, 256, 2560, 9216, kv_bits, chunk, cfg.prefillAttnKeys(seq), prefillStreamBytesPerToken(&cfg), prefillDequantWeightBytes(&cfg), .{}),
            prefillMemoryNeeded(seq, 24, 2, cfg.kvBytesPerToken(), 256, 256, 2560, 9216, kv_bits, chunk, cfg.prefillAttnKeys(seq), prefillStreamBytesPerToken(&cfg), prefillDequantWeightBytes(&cfg), terms),
        );
    }

    const q4 = fixtures.qwen4ExpOomConfig();
    const gated = prefillRequestTerms(&q4, seq, max_tokens, kv_bits, chunk, warm);
    try t.expect(gated.reserved_kv_bytes > 0);
    try t.expect(gated.checkpoint_bytes > 0);
    try t.expect(gated.state_bytes > 0);
    try t.expect(gated.shared_resident_bytes > 0);
}

test "memoryRefusalMessage: the 400 names the hot cache only where the bill carries it" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // A non-qwen4 refusal read "the hot prefix cache holds ~0MB ..." on a box with a multi-GB resident cache.
    const t = std.testing;
    const MB: u64 = 1024 * 1024;

    // Ungated: the previous sentence, byte for byte.
    const bare = try memoryRefusalMessage(t.allocator, 12_000, 4096, 2048, .{ .needed = 30 * MB, .available = 20 * MB }, false);
    defer t.allocator.free(bare);
    try t.expectEqualStrings(
        "Prompt (12000 tokens) requires ~4096MB GPU memory but only ~2048MB available. Reduce prompt size or use a smaller model.",
        bare,
    );

    // Gated: the message quotes what the guard compared.
    const rich = try memoryRefusalMessage(t.allocator, 12_000, 4096, 2048, .{
        .needed = 30 * MB,
        .available = 20 * MB,
        .evictable = 1500 * MB,
        .reclaimable = 900 * MB,
    }, true);
    defer t.allocator.free(rich);
    try t.expect(std.mem.indexOf(u8, rich, "holds ~1500MB more") != null);
    try t.expect(std.mem.indexOf(u8, rich, "(~900MB") != null);

    // The gated arch reaches this arm with an empty cache too and keeps its bytes, zeroes and all.
    const q4_cold = try memoryRefusalMessage(t.allocator, 12_000, 4096, 2048, .{ .needed = 30 * MB, .available = 20 * MB }, true);
    defer t.allocator.free(q4_cold);
    try t.expectEqualStrings(
        "Prompt (12000 tokens) requires ~4096MB GPU memory but only ~2048MB is available and the hot prefix cache holds ~0MB more, all of which can be reclaimed (~0MB — the shortfall is elsewhere), which is not enough. Reduce prompt size, lower --ctx-size, or use a smaller model.",
        q4_cold,
    );
    const ungated_rich = try memoryRefusalMessage(t.allocator, 12_000, 4096, 2048, .{
        .needed = 30 * MB,
        .available = 20 * MB,
        .evictable = 1500 * MB,
        .reclaimable = 900 * MB,
    }, false);
    defer t.allocator.free(ungated_rich);
    try t.expectEqualStrings(bare, ungated_rich);
}

test "the live 364k session's admission numbers reproduce, to the megabyte" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    // These pins price a cache with no SSD tier behind it: every checkpoint stays in RAM.
    const no_tier = TierGuard.pin(false, 0);
    defer no_tier.restore();
    const whole_rebuild = qsaWholeRebuildGuard();
    defer whole_rebuild.deinit();
    // The anchor: fed the deployed config and the incident's prompt lengths, the estimator prints
    // the incident's own numbers (Turn A admitted at 16,233 MB; Turn B refused at 13,664 MB).
    const t = std.testing;
    qwen4_qsa_select.qsa_score_fused_override = false;
    defer qwen4_qsa_select.qsa_score_fused_override = null;
    const cfg = fixtures.qwen4ExpLive364kConfig();
    const mb: u64 = 1024 * 1024;
    const kv_bits: u64 = 8;
    // Turn A, width 2048, prompt 364,509, max_gen 421,923.
    try t.expectEqual(
        @as(u64, 13_269),
        prefillNeededAtChunk(&cfg, 364_509, 421_923, kv_bits, 2048, .{}) / mb,
    );
    try t.expectEqual(
        @as(u64, 11_854),
        prefillNeededAtChunk(&cfg, 368_208, 418_224, kv_bits, 512, .{}) / mb,
    );
}

test "a QSA arch's kv-quant dequant scratch follows the widest forward that still gathers from a rebuild" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const t = std.testing;
    const cfg = fixtures.qwen4RequestTestConfig();
    const seq: u64 = 1024 * 1024;
    const layer_bytes: u64 = 2 * (256 + 256) * 2;
    const span: u64 = 2048 + 4 - 1 + trunk.FUSED256_MIN_Q_LEN - 1;
    const split_k: u64 = @intCast(qwen4_qsa_attn.QSA_ATTN_PACKED_MAX_S);
    // Split-K widths read the packed cache at every kv: only the dense span rebuilds.
    try t.expectEqual(span * layer_bytes, kvDequantScratchBytes(&cfg, seq, split_k));
    try t.expectEqual(span * layer_bytes, kvDequantScratchBytes(&cfg, 2 * seq, 16));
    // A wider forward rebuilds only while its rows would stage each packed row over 4 times.
    try t.expectEqual(512 * 2048 / 4 * layer_bytes, kvDequantScratchBytes(&cfg, seq, 512));
    try t.expectEqual(seq * layer_bytes, kvDequantScratchBytes(&cfg, seq, 8192));
    // A short prompt never rebuilds more than it stores.
    try t.expectEqual(@as(u64, 1000) * layer_bytes, kvDequantScratchBytes(&cfg, 1000, 16));
    // Either kill switch brings the whole-cache rebuild (the generic term) back.
    qwen4_qsa_attn.qsa_attn_kernel_override = false;
    try t.expectEqual(@as(u64, 0), kvDequantScratchBytes(&cfg, seq, 16));
    qwen4_qsa_attn.qsa_attn_kernel_override = null;
    qwen4_qsa_gather.qsa_gather_override = false;
    defer qwen4_qsa_gather.qsa_gather_override = null;
    try t.expectEqual(@as(u64, 0), kvDequantScratchBytes(&cfg, seq, 16));
}

test "a streamed Qwen plans its QSA history beside its KV for every planned token" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const fused_guard = qsaScoreFusedOffGuard();
    defer fused_guard.deinit();
    var cfg = fixtures.qwen4RequestTestConfig();
    cfg.expert_streaming = true;
    const saved = launch_state.configured_kv_quant;
    defer launch_state.configured_kv_quant = saved;
    launch_state.configured_kv_quant = trunk.KVQuantConfig.affine(8);
    const state = statePerTokenBilled(&cfg);
    try std.testing.expect(state > 0);
    const per_token = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), 8) + state;
    try std.testing.expectEqual(per_token * expertStreamingPlannedSeq(&cfg) + slotRingBytes(&cfg, 8), expertStreamingPlannedKvBytes(&cfg));
}

test "a 32 GB Mac's lean Flash-Next budget advertises and admits an 80k prompt" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const saved_ctx = launch_state.server_config.max_context_size;
    defer launch_state.server_config.max_context_size = saved_ctx;
    launch_state.server_config.max_context_size = 0;
    const saved_disk = launch_state.prefix_cache_disk_bytes;
    defer launch_state.prefix_cache_disk_bytes = saved_disk;
    launch_state.prefix_cache_disk_bytes = 20 << 30;
    const saved_mtp = launch_state.configured_mtp;
    defer launch_state.configured_mtp = saved_mtp;
    launch_state.configured_mtp = false;
    var cfg = fixtures.qwen4MeasuredTestConfig();
    cfg.pinned_prefill_chunk = 512;
    const ceiling: u64 = 28976 << 20; // 30000 MiB wired limit less the 1 GiB margin
    const active: u64 = 27_378_205_274; // the measured lean pack's resident bytes
    const prompt: u64 = 80_000;
    const new_tokens: u32 = 16;
    // What the boot advertises: within the measured range the bill is the one reserve, no margin on top.
    var sized = cfg;
    sized.pinned_context = ssdFirstSessionTokensNow(&cfg, 8, ceiling, active, 512);
    try std.testing.expect(sized.pinned_context >= prompt + new_tokens);
    const budget = admissionAvailable(ceiling, active);
    try std.testing.expect(prefillNeededAtChunk(&sized, prompt, new_tokens, 8, 512, .{}) <= budget);
}

test "the SSD-first budget bills the FLOOR reserve, at the deployed pack's live numbers" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // `ssdFirstBudgetForLoad` short-circuits above `planHotCache`, so the floor-width reserve never ran on qwen4_exp.
    const t = std.testing;
    var cfg = fixtures.qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const MiB: u64 = 1 << 20;
    const active: u64 = 69_827 * MiB;
    const ceiling: u64 = 109_395 * MiB;

    const orig_ctx = launch_state.server_config.max_context_size;
    defer launch_state.server_config.max_context_size = orig_ctx;
    launch_state.server_config.max_context_size = 0;
    cfg.pinned_context = 0;
    bills.per_request_chunk_override = true;
    defer bills.per_request_chunk_override = null;
    const orig_source = settings_resolve.prefill_chunk.source;
    defer settings_resolve.prefill_chunk.source = orig_source;
    settings_resolve.prefill_chunk.source = .default;

    // The pure chunk resolver: `pinPrefillChunk` reads live memory.
    const chunk: u32 = @intCast(prefill_plan.effectivePrefillChunk(
        cfg.prefillScoreHeadDim(),
        cfg.num_attention_heads,
        0,
        cfg.has_sliding_window,
        cfg.isMoe(),
        cfg.longCtxGated(),
        cfg.pinned_prefill_chunk,
    ));
    try t.expect(perRequestPrefillChunkEnabled(&cfg));
    const floor_w = clampReserveWidth(&cfg, chunk);
    try t.expectEqual(PREFILL_CHUNK_LADDER[PREFILL_CHUNK_LADDER.len - 1], floor_w);
    try t.expect(floor_w < chunk);

    const reserve_pinned = prefillTransientReserve(&cfg, kv_bits, chunk);
    const reserve_floor = prefillTransientReserve(&cfg, kv_bits, floor_w);
    try t.expect(reserve_floor < reserve_pinned);

    const ctx_kv: u64 = (kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits) +| statePerTokenBilled(&cfg)) *|
        ssdFirstSessionTokensNow(&cfg, kv_bits, ceiling, active, chunk);
    const ask: u64 = 10 * 1024 * MiB;

    // The idle allowance is the residual after the reserve, so the whole difference lands there.
    const now = ssdFirstPrefixCacheMem(ask, ceiling, active, ctx_kv, reserve_floor);
    const before = ssdFirstPrefixCacheMem(ask, ceiling, active, ctx_kv, reserve_pinned);
    try t.expect(now >= before);

    // The RAM arm reserves the same floor for the same pack.
    const plan = planHotCache(&cfg, kv_bits, ceiling, active, 1_048_576, sizerCtxKvBytes(&cfg, kv_bits), ask, chunk);
    try t.expectEqual(floor_w, plan.reserve_chunk);
    try t.expectEqual(reserve_floor, plan.reserve);

    // The session the floor holds is the one the server advertises.
    const advertised = autoContextFrom(memoryContextAt(&cfg, ceiling, active), cfg.contextCap(), cfg.defaultContext());
    try t.expectEqual(advertised, ssdFirstSessionTokensNow(&cfg, kv_bits, ceiling, active, chunk));
}

test "an auto boot advertises the session the SSD-first budget floor was billed for" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // The load-time bill and the pinned context must be one session: the SSD arm billed with
    // `cache_reserve = 0` while `pinAutoContext` sized against `CTX_SIZING_CACHE_RESERVE`, so
    // the floor held RAM for a larger session than the one advertised.
    const t = std.testing;
    var cfg = fixtures.qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const MiB: u64 = 1 << 20;
    const active: u64 = 69_827 * MiB;
    const ceiling: u64 = 109_395 * MiB;

    const original = launch_state.server_config.max_context_size;
    defer launch_state.server_config.max_context_size = original;
    launch_state.server_config.max_context_size = 0;
    cfg.pinned_context = 0;

    const chunk: u32 = @intCast(prefill_plan.effectivePrefillChunk(
        cfg.prefillScoreHeadDim(),
        cfg.num_attention_heads,
        0,
        cfg.has_sliding_window,
        cfg.isMoe(),
        cfg.longCtxGated(),
        cfg.pinned_prefill_chunk,
    ));
    const per_tok: u64 = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits) +| statePerTokenBilled(&cfg);
    const transient: u64 = prefillTransientReserve(&cfg, kv_bits, chunk);

    const advertised = autoContextFrom(memoryContextAt(&cfg, ceiling, active), cfg.contextCap(), cfg.defaultContext());
    const billed = ssdFirstSessionTokensNow(&cfg, kv_bits, ceiling, active, chunk);
    try t.expectEqual(advertised, billed);

    // And on a tighter box where the checkpoint cap does not bind.
    const tight: u64 = active + 24_000 * MiB;
    const billed_t = ssdFirstSessionTokensNow(&cfg, kv_bits, tight, active, chunk);
    try t.expectEqual(autoContextFrom(memoryContextAt(&cfg, tight, active), cfg.contextCap(), cfg.defaultContext()), billed_t);
    try t.expect(billed > 500_000);
    try t.expect(billed_t > 300_000);
    try t.expect(billed_t < billed);

    // The floor holds RAM for exactly the advertised session.
    const idle: u64 = 4 * 1024 * MiB;
    const floor_t = ssdFirstPrefixCacheMem(idle, tight, active, per_tok *| @as(u64, billed_t), transient);
    const floor_more = ssdFirstPrefixCacheMem(idle, tight, active, per_tok *| @as(u64, billed_t + 1024), transient);
    try t.expectEqual(per_tok * 1024, floor_more - floor_t);
}

test "the load-time session bill is billed at the boot's --kv-quant, not bf16" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    // Live: the SSD-first budget billed the session at 22,464 MB with `--kv-quant 8` and with
    // `off` alike (dense 29,952 B/tok): `defaultKvBits` asked `global_scheduler`, which `serve`
    // assigns only after the load has run its preflight.
    const t = std.testing;
    var cfg = fixtures.qwen4RequestTestConfig();
    cfg.mtp_override = false; // the MTP head's own term: `the load-time bill prices the MTP head ...`
    const MiB: u64 = 1 << 20;
    qwen4_qsa_select.qsa_score_fused_override = false;
    defer qwen4_qsa_select.qsa_score_fused_override = null;
    try t.expectEqual(@as(u64, 2_304), statePerTokenBilled(&cfg));
    try t.expectEqual(@as(u64, 13_056), kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), 8));
    try t.expectEqual(@as(u64, 24_576), kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), 16));

    // An explicit `--ctx-size`, so the two numbers differ in the width and nothing else.
    const saved_ctx = launch_state.server_config.max_context_size;
    defer launch_state.server_config.max_context_size = saved_ctx;
    launch_state.server_config.max_context_size = 786_432;
    const ceiling: u64 = 109_395 * MiB;
    const active: u64 = 69_827 * MiB;
    const chunk: u32 = cfg.pinned_prefill_chunk;

    const saved_kv = launch_state.configured_kv_quant;
    defer launch_state.configured_kv_quant = saved_kv;

    // `--kv-quant 8`, no scheduler yet: the state the load runs in.
    launch_state.configured_kv_quant = trunk.KVQuantConfig.affine(8);
    try t.expectEqual(@as(u64, 8), defaultKvBits(&cfg));
    try t.expectEqual(
        (13_056 + 2_304) * 786_432 + cfg.qsaRingBytes(),
        ssdFirstSessionKvBytes(&cfg, defaultKvBits(&cfg), ceiling, active, chunk),
    );

    launch_state.configured_kv_quant = trunk.KVQuantConfig.dense;
    try t.expectEqual(@as(u64, 16), defaultKvBits(&cfg));
    try t.expectEqual(
        (24_576 + 2_304) * 786_432 + cfg.qsaRingBytes(),
        ssdFirstSessionKvBytes(&cfg, defaultKvBits(&cfg), ceiling, active, chunk),
    );
    launch_state.configured_kv_quant = null;
    try t.expectEqual(@as(u64, 8), defaultKvBits(&cfg));

    const idle: u64 = 10 * 1024 * MiB;
    const transient: u64 = prefillTransientReserve(&cfg, 8, chunk);
    const at8 = ssdFirstPrefixCacheMem(idle, ceiling, active, 13_824 * MiB, transient);
    const at16 = ssdFirstPrefixCacheMem(idle, ceiling, active, 22_464 * MiB, transient);
    try t.expectEqual(@as(u64, 8_640 * MiB), at16 -| at8);
}

test "an explicit --ctx-size boot never consults the session reserve" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // `--ctx-size` wins in the resolver's first branch; asserted as invariance in the reserve.
    const t = std.testing;
    var cfg = fixtures.qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const MiB: u64 = 1 << 20;
    const per_tok: u64 = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits) +| statePerTokenBilled(&cfg);

    const original = launch_state.server_config.max_context_size;
    defer launch_state.server_config.max_context_size = original;

    launch_state.server_config.max_context_size = 262_144;
    cfg.pinned_context = 0;
    for ([_]u64{ 0, 40_000 * MiB, 109_395 * MiB, 900_000 * MiB }) |ceiling| {
        for ([_]u64{ 0, 69_827 * MiB }) |active| {
            for ([_]u32{ 512, 1024, 4096, 8192 }) |chunk| {
                try t.expectEqual(@as(u32, 262_144), ssdFirstSessionTokensNow(&cfg, kv_bits, ceiling, active, chunk));
            }
        }
    }
    for ([_]u64{ 0, CTX_SIZING_CACHE_RESERVE, 99_000 * MiB }) |reserve| {
        try t.expectEqual(
            @as(u32, 262_144),
            resolvedContextForLoad(262_144, 0, 109_395 * MiB, 69_827 * MiB, reserve, 3 * 1024 * MiB, per_tok, cfg.contextCap(), cfg.defaultContext()),
        );
    }

    launch_state.server_config.max_context_size = 0;
    cfg.pinned_context = 131_072;
    try t.expectEqual(@as(u32, 131_072), ssdFirstSessionTokensNow(&cfg, kv_bits, 109_395 * MiB, 69_827 * MiB, 4096));
}

// Bar: the ring is billed once per slot, independent of prompt length.

test "a 512k QSA session drops the raw-key overbill and bills the ring once" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    // These pins price a cache with no SSD tier behind it: every checkpoint stays in RAM.
    const no_tier = TierGuard.pin(false, 0);
    defer no_tier.restore();
    const whole_rebuild = qsaWholeRebuildGuard();
    defer whole_rebuild.deinit();
    const t = std.testing;
    qwen4_qsa_select.qsa_score_fused_override = false;
    defer qwen4_qsa_select.qsa_score_fused_override = null;
    const cfg = fixtures.qwen4ExpOomConfig();
    const seq: u64 = 512 * 1024;
    const short = prefillRequestTerms(&cfg, 1024, 2048, 8, 4096, .{});
    const long = prefillRequestTerms(&cfg, seq, 2048, 8, 4096, .{});
    try t.expectEqual(cfg.qsaRingBytes(), short.qsa_ring_bytes);
    try t.expectEqual(cfg.qsaRingBytes(), long.qsa_ring_bytes);
    const reserved = @max(reservedCacheTokens(seq, 2048, 4096, getEffectiveContextLength(&cfg)), seq);
    const new_needed = prefillNeededAtChunk(&cfg, seq, 2048, 8, 4096, .{});
    const old_needed = new_needed +| (reserved * 3_072 * 5 / 4);
    const ceiling: u64 = 26_000 * 1024 * 1024;
    try t.expect(old_needed > ceiling);
    try t.expect(new_needed <= ceiling);
}

test "a warm append is billed the rows it ALLOCATES, not the rows it already holds" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const whole_rebuild = qsaWholeRebuildGuard();
    defer whole_rebuild.deinit();
    // Bar: a checked-out restore credits the rows it holds even when the append outgrows the
    // buffer; the grow bills the new capacity plus one eval window of old buffers.
    const t = std.testing;
    qwen4_qsa_select.qsa_score_fused_override = false;
    defer qwen4_qsa_select.qsa_score_fused_override = null;
    const mb: u64 = 1024 * 1024;
    const cfg = fixtures.qwen4ExpLive364kConfig();
    const kv_bits: u64 = 8;
    const seq: u64 = 448_604;
    const matched: u64 = 448_531;
    const chunk: u64 = 512;
    const max_tokens: u32 = @intCast(getEffectiveContextLength(&cfg) - seq);
    const kv_per_tok = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits);
    const reserved = @max(reservedCacheTokens(seq, max_tokens, chunk, getEffectiveContextLength(&cfg)), seq);

    // Pessimal reading of the entry: capacity exactly the rows it matched, so the append DOES
    // grow every attention layer.
    const grows = WarmPrefix{ .matched_tokens = matched, .capacity_tokens = matched, .will_donate = true };
    const g = prefillRequestTerms(&cfg, seq, max_tokens, kv_bits, chunk, grows);
    try t.expectEqual(matched *| kv_per_tok, g.shared_resident_bytes);
    try t.expectEqual((reserved - matched) *| kv_per_tok, g.reserved_kv_bytes +| (seq - matched) *| kv_per_tok);
    // ...plus one eval window of old buffers: qwen4_exp caches 12 of 48 layers, one per window.
    try t.expectEqual(@as(u32, 1), attnLayersPerEvalWindow(&cfg, trunk.Trunk.MOE_EVAL_EVERY_N_LAYERS));
    try t.expectEqual(@as(u64, 465), g.grow_coexist_bytes / mb);
    // The bill at the narrowest and a wider width.
    try t.expectEqual(@as(u64, 5_057), prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, grows) / mb);
    try t.expectEqual(@as(u64, 6_548), prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, 2048, grows) / mb);

    // A buffer the previous turn sized above this prompt but below the reservation is grown to
    // it before the prefill writes, one attention layer per eval: one window of old rows.
    const fits = WarmPrefix{ .matched_tokens = matched, .capacity_tokens = 450_048, .will_donate = true };
    const f = prefillRequestTerms(&cfg, seq, max_tokens, kv_bits, chunk, fits);
    try t.expect(reserved > 450_048);
    try t.expectEqual(450_048 *| kv_per_tok / 12, f.grow_coexist_bytes);
    try t.expectEqual(@as(u64, 5_059), prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, fits) / mb);

    // Byte-identical arms: a SHARED restore is copied whole by the first append, so it credits
    // nothing and bills no window either; a COLD prompt has nothing to credit.
    const shared = WarmPrefix{ .matched_tokens = matched, .capacity_tokens = matched, .will_donate = false };
    const sh = prefillRequestTerms(&cfg, seq, max_tokens, kv_bits, chunk, shared);
    try t.expectEqual(@as(u64, 0), sh.shared_resident_bytes);
    try t.expectEqual(@as(u64, 0), sh.grow_coexist_bytes);
    const cold = prefillRequestTerms(&cfg, seq, max_tokens, kv_bits, chunk, .{});
    try t.expectEqual(@as(u64, 0), cold.shared_resident_bytes);
    try t.expectEqual(@as(u64, 0), cold.grow_coexist_bytes);
}

test "an omitted max_tokens reserves a generation HEADROOM, not the rest of the context" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const t = std.testing;
    qwen4_qsa_select.qsa_score_fused_override = false;
    defer qwen4_qsa_select.qsa_score_fused_override = null;
    // A 383,067-token prompt with no `max_tokens` was admitted by eviction and then refused on
    // the inference thread: `clampMaxTokens` made the reservation cover the whole context
    // (~13 GB for a generation that would never run). The reservation exists to stop buffer
    // coexistence during prefill, not to pre-buy a generation.
    const KVCache = trunk.KVCache;
    const MB: u64 = 1024 * 1024;
    const cfg = fixtures.qwen4ExpOomConfig();
    const ctx: u64 = getEffectiveContextLength(&cfg);
    try t.expectEqual(@as(u64, 1_048_576), ctx);
    const seq: u64 = 383_067;
    const chunk: u64 = 4096;
    const clamped: u64 = request_policy.clampMaxTokens(@intCast(request_policy.omittedMaxTokensDefault(&cfg)), @intCast(seq), @intCast(ctx));
    try t.expectEqual(ctx - seq, clamped); // 665,509 — the live number

    // The reservation is bounded by the headroom, not by the context.
    const reserved = KVCache.reservedTokens(seq, clamped, chunk, ctx);
    try t.expectEqual(seq + KVCache.RESERVE_GEN_HEADROOM + chunk, reserved);
    try t.expect(reserved < seq + ctx / 8);
    try t.expectEqual(seq + 512 + chunk, KVCache.reservedTokens(seq, 512, chunk, ctx));
    try t.expectEqual(ctx + chunk, KVCache.reservedTokens(ctx - 16, clamped, chunk, ctx));

    const kv_per_tok = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), 8);
    const terms = prefillRequestTerms(&cfg, seq, clamped, 8, chunk, .{});
    try t.expectEqual((KVCache.RESERVE_GEN_HEADROOM + chunk) * kv_per_tok, terms.reserved_kv_bytes);
    try t.expectEqual(reserved * statePerTokenBilled(&cfg), terms.state_bytes);

    // ...and the whole bill drops by more than 10 GB, from above the live headroom to under it.
    const ctx_bounded: PrefillRequestTerms = .{
        .reserved_kv_bytes = (ctx + chunk - seq) * kv_per_tok,
        .state_bytes = (ctx + chunk) * statePerTokenBilled(&cfg),
        .checkpoint_bytes = terms.checkpoint_bytes,
    };
    const bill = oomBillFor(&cfg, seq, chunk, terms);
    const old_bill = oomBillFor(&cfg, seq, chunk, ctx_bounded);
    try t.expect(old_bill > bill + 10 * 1024 * MB);
    try t.expect(old_bill > 28_000 * MB);
    try t.expect(bill < 30_000 * MB); // under the live "~37130MB available"
}

test "the tail-merge gate reads the ARCH, not the installed hook (serve installs it for everyone)" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // `serve` installs `prefill_chunk_adapt` unconditionally, so `chunk_width_hook != null` is
    // true on every arch and reading it as the gate put the scaled bound back everywhere. This
    // builds the slot state the way `serve` does and asserts the two answers differ.
    const t = std.testing;
    scheduler_mod.prefill_chunk_adapt = &adaptivePrefillWidthNow;
    defer scheduler_mod.prefill_chunk_adapt = null;
    scheduler_mod.prefill_chunk_widen_ok = &adaptivePrefillWidenStillFits;
    defer scheduler_mod.prefill_chunk_widen_ok = null;
    scheduler_mod.prefill_chunk_adaptive_enabled = &adaptivePrefillChunkEnabled;
    defer scheduler_mod.prefill_chunk_adaptive_enabled = null;

    const hook_installed = scheduler_mod.prefill_chunk_adapt != null;
    try t.expect(hook_installed);

    var other = fixtures.qwen4RequestTestConfig();
    other.model_type = "not_a_served_arch";
    const adaptive = slot_admission.adaptiveChunkWidthFor(&other);
    try t.expect(!adaptive);
    try t.expect(hook_installed != adaptive);

    // A slot without the adaptive hook keeps the flat bound.
    try t.expectEqual(prefill_plan.TAIL_MERGE_MAX, prefill_plan.tailMergeMaxFor(512, adaptive));
    try t.expectEqual(@as(usize, 812), prefill_plan.nextChunkEnd(0, 812, 512, false, 0, 0, adaptive));
    try t.expectEqual(@as(usize, 1), prefill_plan.prefillChunkCount(812, 512, false, 0, 0, adaptive));

    const cfg = fixtures.qwen4RequestTestConfig();
    bills.per_request_chunk_override = true;
    defer bills.per_request_chunk_override = null;
    bills.adaptive_chunk_override = true;
    defer bills.adaptive_chunk_override = null;
    const unpinned = explicitPrefillChunk() == 0;
    try t.expectEqual(unpinned, slot_admission.adaptiveChunkWidthFor(&cfg));
    if (unpinned) {
        try t.expectEqual(@as(usize, 64), prefill_plan.tailMergeMaxFor(512, slot_admission.adaptiveChunkWidthFor(&cfg)));
        try t.expectEqual(@as(usize, 512), prefill_plan.nextChunkEnd(0, 812, 512, false, 0, 0, slot_admission.adaptiveChunkWidthFor(&cfg)));
    }
    try t.expect(!slot_admission.adaptiveChunkWidthFor(null));
    scheduler_mod.prefill_chunk_adaptive_enabled = null;
    try t.expect(!slot_admission.adaptiveChunkWidthFor(&cfg));
    try t.expect(scheduler_mod.prefill_chunk_adapt != null); // ...with the hook still installed
}

test "a WARM turn is not billed for the prefix it is about to SHARE (786,707 @ 19,032 MB)" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const whole_rebuild = qsaWholeRebuildGuard();
    defer whole_rebuild.deinit();
    const t = std.testing;
    // Live: a 786,707-token warm turn restored its own 786,676-token entry and was then refused
    // by name; the restore hands over the entry's buffers without a second copy, so those
    // ~14.5 GB were already inside `mlx_get_active_memory` and `needed` billed them again.
    const cfg = fixtures.qwen4ExpOomConfig();
    qwen4_qsa_select.qsa_score_fused_override = false;
    defer qwen4_qsa_select.qsa_score_fused_override = null;
    const seq: u64 = 786_707;
    const chunk: u64 = 512; // the width the ladder came down to that evening
    const kv_bits: u64 = 8;
    const max_tokens: u32 = 200;
    const mb: u64 = 1024 * 1024;
    const available: u64 = 19_032 * mb;

    // 13,056 B of 8-bit KV and 5,376 B of state per token (one history copy + the score bank).
    // The credit prices KV only: the history is privatised on restore.
    const kv_per_tok = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits);
    try t.expectEqual(@as(u64, 13_056), kv_per_tok);
    try t.expectEqual(@as(u64, 768), cfg.qsaHistoryBytesPerToken());
    try t.expectEqual(@as(u64, 2_304), statePerTokenBilled(&cfg));

    // Cold: the refusal, reproduced.
    const cold = prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, .{});
    try t.expect(cold / mb > 18_000 and cold / mb < 23_000);
    try t.expect(cold > available);

    // Warm: the entry's buffers hold the capacity the previous turn reserved, which still covers
    // this turn's reservation, so nothing grows.
    const matched: u64 = 786_676;
    const capacity: u64 = 787_456;
    const reserved = reservedCacheTokens(seq, max_tokens, chunk, getEffectiveContextLength(&cfg));
    try t.expectEqual(@as(u64, 787_419), reserved);
    try t.expect(reserved <= capacity);

    // `will_donate`: this restore checked out its entry, so the buffers are the slot's own.
    const warm_prefix = WarmPrefix{ .matched_tokens = matched, .capacity_tokens = capacity, .will_donate = true };
    try t.expectEqual(matched, warm_prefix.creditedRows(reserved));

    // The credit is the handed-over rows at this arch's KV width and nothing else, inside the 5/4.
    const credit = matched * kv_per_tok;
    try t.expectEqual(@as(u64, 786_676 * 13_056), credit);
    try t.expect(credit < matched * (kv_per_tok + cfg.qsaHistoryBytesPerToken()));
    const warm = prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, warm_prefix);
    // Isolated against a shared restore of the same span: `checkpoint_bytes` follows `matched` too.
    const shared_same_span = prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, .{ .matched_tokens = matched, .capacity_tokens = capacity });
    try t.expect(shared_same_span < cold);
    try t.expectEqual(shared_same_span - credit * 5 / 4, warm);

    // The verdict flips: admitted outright, no eviction pass. Under-crediting is the safe direction.
    try t.expect(warm < available);
    try t.expect(warm / mb > 6_500 and warm / mb < 8_500);

    // A cold prompt on the same box keeps the old refusal.
    try t.expect(prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, .{}) > available);
    try t.expect(prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, .{ .matched_tokens = 0, .capacity_tokens = capacity, .will_donate = true }) > available);
}

test "a chain extension that OUTGROWS the resident entry is credited the rows it holds" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const t = std.testing;
    // Past the restored capacity the cache grows per layer: the resident rows stay credited and
    // the grow bills the new capacity beyond the old plus one eval window of old buffers.
    const cfg = fixtures.qwen4ExpOomConfig();
    const seq: u64 = 1_047_556;
    const chunk: u64 = 512;
    const kv_bits: u64 = 8;
    const max_tokens: u32 = 200;
    const capacity: u64 = 787_456; // what the 786k turn reserved

    const reserved = reservedCacheTokens(seq, max_tokens, chunk, getEffectiveContextLength(&cfg));
    try t.expectEqual(@as(u64, 1_048_268), reserved);
    const extend = WarmPrefix{ .matched_tokens = 786_707, .capacity_tokens = capacity, .will_donate = true };
    try t.expect(reserved > capacity);
    try t.expect(extend.grows(seq));
    try t.expectEqual(@as(u64, 786_707), extend.creditedRows(reserved));
    const per_tok = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits);
    const ext_terms = prefillRequestTerms(&cfg, seq, max_tokens, kv_bits, chunk, extend);
    const window = attnLayersPerEvalWindow(&cfg, trunk.Trunk.MOE_EVAL_EVERY_N_LAYERS);
    try t.expectEqual(capacity * per_tok / cfg.attnCacheLayerCount() * window, ext_terms.grow_coexist_bytes);
    try t.expectEqual(
        prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, .{}) -
            (786_707 * per_tok - ext_terms.grow_coexist_bytes) * 5 / 4,
        prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, extend),
    );

    // A buffer that already covers the reservation credits the full shared prefix: the gate is capacity, not prompt length.
    const roomy = WarmPrefix{ .matched_tokens = 786_707, .capacity_tokens = 1_048_576, .will_donate = true };
    try t.expectEqual(@as(u64, 786_707), roomy.creditedRows(reserved));
    const credit = 786_707 * kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits);
    try t.expectEqual(
        prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, .{}) - credit * 5 / 4,
        prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, roomy),
    );
}

test "the warm KV credit fires ONLY where the restore checked its entry out" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const whole_rebuild = qsaWholeRebuildGuard();
    defer whole_rebuild.deinit();
    const t = std.testing;
    qwen4_qsa_select.qsa_score_fused_override = false;
    defer qwen4_qsa_select.qsa_score_fused_override = null;
    // `KVCache.restore` binds by refcount and mlx donates only at `use_count() == 1`, so on a
    // shared restore the first `writeAtOffset` copies the whole prefix: the credited bytes are
    // allocated. Only the checkout makes the credit true; it used to fire on `matched > 0` alone.
    const cfg = fixtures.qwen4ExpOomConfig();
    const seq: u64 = 786_707;
    const chunk: u64 = 512;
    const kv_bits: u64 = 8;
    const max_tokens: u32 = 200;
    const mb: u64 = 1024 * 1024;
    const matched: u64 = 786_676;
    const capacity: u64 = 787_456;
    const reserved = reservedCacheTokens(seq, max_tokens, chunk, getEffectiveContextLength(&cfg));
    const cold = prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, .{});

    // (1) qwen4_exp, SSD-first, full-prefix hit, switch on: the credit is real.
    const donating = (CheckoutCase{}).warm(matched, capacity);
    try t.expect(donating.will_donate);
    try t.expectEqual(matched, donating.creditedRows(reserved));
    const warm = prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, donating);
    // Baseline = a shared restore of the same span, so the two credits are not folded into one comparison.
    const shared_same_span = prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, .{ .matched_tokens = matched, .capacity_tokens = capacity });
    try t.expectEqual(shared_same_span - matched * kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits) * 5 / 4, warm);
    try t.expect(warm / mb > 6_500 and warm / mb < 8_500);

    // Every arm below is the same restore minus one conjunct of the checkout, and every one shares.
    const shared_arms = [_]WarmPrefix{
        // (2) a partial hit: the entry describes a prefix this prompt left.
        (CheckoutCase{ .shared = 700_000 }).warm(matched, capacity),
        // (3) a non-qwen4 arch: a checkout can never be taken there.
        (CheckoutCase{ .ssd_first = false }).warm(matched, capacity),
        // (4) restore by move off: the refcount share.
        (CheckoutCase{ .move_enabled = false }).warm(matched, capacity),
        // (5) a pending disk record shares the same buffers.
        (CheckoutCase{ .pending_disk = true }).warm(matched, capacity),
        // (6) no slot to hand the buffers to (the connection thread's shape).
        (CheckoutCase{ .has_slot = false }).warm(matched, capacity),
    };
    for (shared_arms, 0..) |arm, i| {
        errdefer std.debug.print("shared arm {d} credited\n", .{i});
        try t.expect(!arm.will_donate);
        try t.expectEqual(@as(u64, 0), arm.creditedRows(reserved));
        try t.expectEqual(shared_same_span, prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, arm));
    }
    // ...and `cold` is still the number a prompt with NO resident prefix pays,
    // which is what the `.{}` arm above is for.
    try t.expect(cold > shared_same_span);

    // Capacity is a CEILING on the credit, not a switch: a buffer holding 700,000 rows donates
    // them and the request is billed the growth on top.
    const outgrown = (CheckoutCase{}).warm(matched, 700_000);
    try t.expect(outgrown.will_donate);
    try t.expectEqual(@as(u64, 700_000), outgrown.creditedRows(reserved));
    try t.expect(outgrown.grows(seq));
    const kv_per_tok = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits);
    const og_terms = prefillRequestTerms(&cfg, seq, max_tokens, kv_bits, chunk, outgrown);
    // 31 tokens wide, BELOW `prefillEvalCadenceApplies`: one eval for the whole layer loop, so
    // every old buffer is alive at once and the coexistence term cancels the credit exactly.
    try t.expect(seq - matched < 32);
    try t.expectEqual(700_000 * kv_per_tok, og_terms.grow_coexist_bytes);
    try t.expectEqual(shared_same_span, prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, outgrown));
    // An append wide enough to take the cadence pays ONE eval window instead, and that is the
    // difference between the old bill and the new one.
    const wide = WarmPrefix{ .matched_tokens = 700_000, .capacity_tokens = 700_000, .will_donate = true };
    const wide_terms = prefillRequestTerms(&cfg, seq, max_tokens, kv_bits, chunk, wide);
    const window = attnLayersPerEvalWindow(&cfg, trunk.Trunk.MOE_EVAL_EVERY_N_LAYERS);
    try t.expect(window < cfg.attnCacheLayerCount());
    try t.expectEqual(700_000 * kv_per_tok / cfg.attnCacheLayerCount() * window, wide_terms.grow_coexist_bytes);
    try t.expect(prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, wide) < shared_same_span);
}

test "prefillRequestTerms: qwen4 MTP head KV is billed when MTP is on and zero when off" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const t = std.testing;
    const saved_force = launch_state.server_config.default_force_mtp;
    defer launch_state.server_config.default_force_mtp = saved_force;
    launch_state.server_config.default_force_mtp = false;
    const saved_ov = trunk.Trunk.mtp_head_kv_quant_override;
    defer trunk.Trunk.mtp_head_kv_quant_override = saved_ov;
    trunk.Trunk.mtp_head_kv_quant_override = false;
    const saved_boot = launch_state.configured_kv_quant;
    defer launch_state.configured_kv_quant = saved_boot;
    launch_state.configured_kv_quant = trunk.KVQuantConfig.affine(8);
    var cfg = fixtures.qwen4ExpOomConfig();
    cfg.mtp_override = null;
    try t.expect(!forceMtpFor(&cfg));
    const seq: u64 = 200_000;
    const chunk: u64 = 4096;
    const max_tokens: u64 = 8192;
    const kv_bits: u64 = 8;
    try t.expectEqual(@as(u64, 2048), mtpHeadKvBytesPerToken(&cfg));
    trunk.Trunk.mtp_head_kv_quant_override = true;
    try t.expectEqual(@as(u64, 1088), mtpHeadKvBytesPerToken(&cfg));
    trunk.Trunk.mtp_head_kv_quant_override = false;
    const off = prefillRequestTerms(&cfg, seq, max_tokens, kv_bits, chunk, .{});
    try t.expectEqual(@as(u64, 0), off.mtp_head_kv_bytes);
    const on = prefillRequestTerms(&cfg, seq, max_tokens, kv_bits, chunk, .{ .mtp_on = true });
    const reserved = @max(reservedCacheTokens(seq, max_tokens, chunk, getEffectiveContextLength(&cfg)), seq);
    try t.expectEqual(reserved * 2048, on.mtp_head_kv_bytes);
    const n: u64 = cfg.attnCacheLayerCount();
    try t.expectEqual(cfg.qsaRingBytes() + cfg.qsaRingBytes() / n, on.qsa_ring_bytes);
    try t.expectEqual(reserved * (statePerTokenBilled(&cfg) + statePerTokenBilled(&cfg) / n), on.state_bytes);
    try t.expectEqual(cfg.qsaRingBytes(), off.qsa_ring_bytes);
    try t.expectEqual(reserved * statePerTokenBilled(&cfg), off.state_bytes);
    var other = cfg;
    other.model_type = "not_a_served_arch";
    const other_on = prefillRequestTerms(&other, seq, max_tokens, kv_bits, chunk, .{ .mtp_on = true });
    try t.expectEqual(@as(u64, 0), other_on.mtp_head_kv_bytes);
}

test "prefillRequestTerms: MTP head KV bills the boot scheme, not a request kv_quant override" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const t = std.testing;
    const saved_ov = trunk.Trunk.mtp_head_kv_quant_override;
    defer trunk.Trunk.mtp_head_kv_quant_override = saved_ov;
    trunk.Trunk.mtp_head_kv_quant_override = true;
    const saved_boot = launch_state.configured_kv_quant;
    defer launch_state.configured_kv_quant = saved_boot;
    launch_state.configured_kv_quant = trunk.KVQuantConfig.affine(8);
    var cfg = fixtures.qwen4ExpOomConfig();
    cfg.mtp_override = null;
    const seq: u64 = 200_000;
    const chunk: u64 = 4096;
    const max_tokens: u64 = 8192;
    const request_bits: u64 = 4;
    const n: u64 = cfg.attnCacheLayerCount();
    const request_head = kvBytesPerTokenAtBits(cfg.kvBytesPerToken() / n, request_bits);
    try t.expectEqual(@as(u64, 576), request_head);
    try t.expectEqual(@as(u64, 1088), mtpHeadKvBytesPerToken(&cfg));
    const on = prefillRequestTerms(&cfg, seq, max_tokens, request_bits, chunk, .{ .mtp_on = true });
    const reserved = @max(reservedCacheTokens(seq, max_tokens, chunk, getEffectiveContextLength(&cfg)), seq);
    try t.expectEqual(reserved * 1088, on.mtp_head_kv_bytes);
}

test "a WARM append bills the checkpoints its prefill CAPTURES, not the prompt's" {
    if (!archBuilt("qwen4")) return error.SkipZigTest;
    // These pins price a cache with no SSD tier behind it: every checkpoint stays in RAM.
    const no_tier = TierGuard.pin(false, 0);
    defer no_tier.restore();
    const whole_rebuild = qsaWholeRebuildGuard();
    defer whole_rebuild.deinit();
    // Turn B: prompt 368,208, restored 364,478 by move, refused by 290 MB at width 512. 1,795 MB
    // of the bill was 32 SSM checkpoints; a warm prefill forwards a 3,730-token tail and captures
    // at most one, and the entry's own checkpoints are already resident and merged at commit.
    const t = std.testing;
    qwen4_qsa_select.qsa_score_fused_override = false;
    defer qwen4_qsa_select.qsa_score_fused_override = null;
    const cfg = fixtures.qwen4ExpLive364kConfig();
    const mb: u64 = 1024 * 1024;
    const per_cp = cfg.ssmCheckpointBytes();
    const seq: u64 = 368_208;
    const matched: u64 = 364_478;
    const chunk: u64 = 512;
    // The stride follows the uncapped base chunk (`settings_resolve.prefill_chunk`), not the adaptive width.
    const saved_override = settings_resolve.prefill_chunk.value;
    defer settings_resolve.prefill_chunk.value = saved_override;
    settings_resolve.prefill_chunk.value = 8192;
    const stride: u64 = 8192;
    try t.expectEqual(stride, prefill_plan.effectiveSsmCheckpointStride(launch_state.ssm_checkpoint_stride, 8192));

    // Cold: `seq / stride` saturates at --ssm-checkpoint-max as before.
    try t.expectEqual(@as(u64, launch_state.ssm_checkpoint_max) * per_cp, retainedSsmCheckpointBytes(&cfg, seq, 0, chunk));
    // Warm: a 3,730-token append at stride 8192 crosses no boundary, so the end snapshot is the whole of it.
    try t.expectEqual(@as(u64, 1) * per_cp, retainedSsmCheckpointBytes(&cfg, seq, matched, chunk));
    // A chain extension long enough to fill the ring still bills the cap: the
    // credit is proportional to the SPAN, not a blanket warm discount.
    try t.expectEqual(@as(u64, launch_state.ssm_checkpoint_max) * per_cp, retainedSsmCheckpointBytes(&cfg, seq, 60_000, chunk));
    // A span shorter than the backoff window snapshots nothing.
    try t.expectEqual(@as(u64, 0), retainedSsmCheckpointBytes(&cfg, seq, seq - 4, chunk));

    // The bill term by term, from the same inputs `prefillMemoryNeeded` reads.
    const kv_bits: u64 = 8;
    const max_tokens: u32 = 418_224;
    // The restored buffer is smaller than the reservation, so it grows and the rows it holds
    // are still credited. This block keeps the original turn-B decomposition by pricing the
    // SHARED restore of the same span, whose credit is zero for the other reason.
    const turn_b_donating = WarmPrefix{ .matched_tokens = matched, .capacity_tokens = 365_568, .will_donate = true };
    const turn_b = WarmPrefix{ .matched_tokens = matched, .capacity_tokens = 365_568, .will_donate = false };
    const reserve_rows = reservedCacheTokens(seq, max_tokens, chunk, getEffectiveContextLength(&cfg));
    try t.expectEqual(seq + trunk.KVCache.RESERVE_GEN_HEADROOM + chunk, reserve_rows);
    try t.expect(reserve_rows > turn_b.capacity_tokens);
    try t.expectEqual(@as(u64, 0), turn_b.creditedRows(reserve_rows));
    try t.expectEqual(@as(u64, 364_478), turn_b_donating.creditedRows(reserve_rows));

    const kv_per_tok = kvBytesPerTokenAtBits(cfg.kvBytesPerToken(), kv_bits);
    const terms = prefillRequestTerms(&cfg, seq, max_tokens, kv_bits, chunk, turn_b);
    // On a GDN MoE the envelope is the stream arm, not 3 x MLP.
    const mlp: u64 = 8 * chunk * @max(cfg.hidden_size, prefillFfnWidth(&cfg)) * 2;
    const envelope: u64 = @max(3 * mlp, mlp + chunk * prefillStreamBytesPerToken(&cfg));
    try t.expect(envelope > 3 * mlp);
    const gross =
        seq * kv_per_tok +
        terms.reserved_kv_bytes +
        terms.state_bytes +
        terms.checkpoint_bytes +
        terms.qsa_ring_bytes +
        2 * seq * cfg.num_key_value_heads * cfg.head_dim * 2 +
        envelope +
        PREFILL_RUNTIME_FLOOR_BYTES;
    // No score scratch (hd 256 is fused) and no dequant-GEMM weights (fwd 512
    // is below PREFILL_DQ_GEMM_MIN_M) at this width.
    try t.expect(trunk.prefillHeadDimFused(@intCast(cfg.prefillScoreHeadDim())));
    try t.expect(chunk < trunk.PREFILL_DQ_GEMM_MIN_M);
    const needed = prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, chunk, turn_b);
    try t.expectEqual(gross * 5 / 4 + qsaMaskBytes(&cfg, chunk, seq), needed);

    // AND THE VERDICT MOVES. 11,854 MB refused against 10,000 MB available;
    // the honest checkpoint term brings it to 9,789 MB, so the turn is served.
    const available: u64 = 10_000 * mb;
    try t.expect(needed < available);
    try t.expectEqual(@as(u64, 9_789), needed / mb);
    const before = gross - terms.checkpoint_bytes + @as(u64, launch_state.ssm_checkpoint_max) * per_cp;
    try t.expectEqual(@as(u64, 11_854), (before * 5 / 4 + qsaMaskBytes(&cfg, chunk, seq)) / mb);
    try t.expect((before * 5 / 4 + qsaMaskBytes(&cfg, chunk, seq)) > available);
}

test "an arch without QSA bills its requests and transients free of the QSA kernel arms" {
    if (!archBuilt("qwen4") or !archBuilt("glm5")) return error.SkipZigTest;
    const t = std.testing;
    var glm = try model_mod.parseConfigFromJson(t.allocator, @embedFile("../fixtures/glm5_config.json"));
    defer glm.deinit(t.allocator);
    glm.pinned_context = 262_144;
    glm.pinned_prefill_chunk = 2048;
    const saved_gather = qwen4_qsa_gather.qsa_gather_override;
    const saved_fused = qwen4_qsa_select.qsa_score_fused_override;
    defer {
        qwen4_qsa_gather.qsa_gather_override = saved_gather;
        qwen4_qsa_select.qsa_score_fused_override = saved_fused;
    }
    var first: ?[2]u64 = null;
    for ([_]bool{ true, false }) |gather| for ([_]bool{ true, false }) |fused| {
        qwen4_qsa_gather.qsa_gather_override = gather;
        qwen4_qsa_select.qsa_score_fused_override = fused;
        const got = [2]u64{ prefillTransientReserve(&glm, 8, 4096), prefillNeededAtChunk(&glm, 70_000, 4096, 8, 4096, .{}) };
        if (first) |want| try t.expectEqualSlices(u64, &want, &got) else first = got;
    };
}
