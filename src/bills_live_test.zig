//! The bill tests that read the live GPU or need MLX. They run in the MLX tier; the rest of bills.zig runs in test-host.
const Trunk = trunk.Trunk;
const fixtures = @import("bills_fixtures_test.zig");
const bills = @import("bills.zig");
const generate_mod = @import("generate.zig");
const glm5_generate = @import("glm5/glm5_generate.zig");
const glm5_prefix = @import("glm5/glm5_prefix.zig");
const launch_state = @import("server_state.zig");
const mlx = @import("mlx.zig");
const model_mod = @import("model.zig");
const mtp_plan = @import("mtp_plan.zig");
const prefill_plan = @import("prefill_plan.zig");
const prefix_cache_mod = @import("prefix_cache.zig");
const qwen4_qsa_select = @import("qwen4/qwen4_qsa_select.zig");
const request_policy = @import("request_policy.zig");
const scheduler_mod = @import("scheduler.zig");
const slot_admission = @import("slot_admission.zig");
const std = @import("std");
const settings_resolve = @import("settings_resolve.zig");
const testing = std.testing;
const trunk = @import("trunk.zig");

test "chooseRequestPrefillChunk: an ordinary prompt buys the wide chunk a 1M session cannot" {
    // `--ctx-size 1048576` with no chunk or cache flags must still prefill ordinary prompts at
    // the wide chunk; load-time sizing reserved for the whole session (rung 1024 for the boot).
    const t = std.testing;
    qwen4_qsa_select.qsa_score_fused_override = false;
    defer qwen4_qsa_select.qsa_score_fused_override = null;
    const cfg = fixtures.qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const MiB: u64 = 1 << 20;
    const available: u64 = 28_909 * MiB;
    const pin = cfg.pinned_prefill_chunk;

    try t.expectEqual(@as(u32, 4096), bills.chooseRequestPrefillChunk(&cfg, 4096, 2048, kv_bits, available, pin, 0, .{}));
    try t.expectEqual(@as(u32, 4096), bills.chooseRequestPrefillChunk(&cfg, 300_000, 2048, kv_bits, available, pin, 0, .{}));
    try t.expectEqual(@as(u32, 4096), bills.chooseRequestPrefillChunk(&cfg, 384_000, 2048, kv_bits, available, pin, 0, .{}));
    try t.expect(4096 > pin);

    // The full 1M context fits at 4096 on this box under the one-copy history bill.
    const full = bills.chooseRequestPrefillChunk(&cfg, 1_048_576, 2048, kv_bits, available, pin, 0, .{});
    try t.expectEqual(@as(u32, 4096), full);
    try t.expect(bills.prefillNeededAtChunk(&cfg, 1_048_576, 2048, kv_bits, 2048, .{}) <= available);
    try t.expect(bills.prefillNeededAtChunk(&cfg, 1_048_576, 2048, kv_bits, 4096, .{}) <= available);

    // A shorter prompt can afford at least as wide a forward.
    var prev: u32 = 0;
    var i: usize = 0;
    const seqs = [_]u64{ 1_048_576, 384_000, 300_000, 60_000, 4096 };
    while (i < seqs.len) : (i += 1) {
        const w = bills.chooseRequestPrefillChunk(&cfg, seqs[i], 2048, kv_bits, available, pin, 0, .{});
        try t.expect(w >= prev);
        prev = w;
    }
}

test "chooseRequestPrefillChunk: WIDEST that fits, at the boundary" {
    const qsa_fused_off = fixtures.qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // "Widest that fits": priced at the exact bill of a width that width comes out; one byte
    // under, a strictly narrower one (except at the floor). Walks the config's own ladder.
    const t = std.testing;
    const cfg = fixtures.qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const seq: u64 = 300_000;
    const pin = cfg.pinned_prefill_chunk;
    const floor_width: u32 = bills.widthForRung(&cfg, seq, bills.PREFILL_CHUNK_LADDER[bills.PREFILL_CHUNK_LADDER.len - 1]);

    var prev_width: u32 = 0;
    var distinct: usize = 0;
    for (bills.PREFILL_CHUNK_LADDER) |rung| {
        if (rung > bills.ladderTop(&cfg, 0)) continue; // above the arch's start width
        const width = bills.widthForRung(&cfg, seq, rung);
        if (width == prev_width) continue; // a rung this arch narrows onto a width already walked
        prev_width = width;
        distinct += 1;
        const bill = bills.prefillNeededAtChunk(&cfg, seq, 2048, kv_bits, width, .{});
        // A rung the long-context gate widened is taken only past its margin.
        const needed = if (width > bills.rungWidth(&cfg, seq, rung, false))
            bill * (100 + bills.PREFILL_WIDE_RUNG_MARGIN_PCT) / 100
        else
            bill;
        try t.expectEqual(width, bills.chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, needed, pin, 0, .{}));
        const under = bills.chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, needed - 1, pin, 0, .{});
        if (width == floor_width) {
            try t.expectEqual(floor_width, under);
        } else {
            try t.expect(under < width);
        }
    }
    try t.expect(distinct >= 4);

    // Nothing fits: the ladder floor, never 0 and never the load-time pin.
    try t.expectEqual(floor_width, bills.chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, 0, pin, 0, .{}));
}

test "chooseRequestPrefillChunk: the rung the gate widened clears its bill by the estimator's margin" {
    const qsa_fused_off = fixtures.qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const t = std.testing;
    const cfg = fixtures.qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const seq: u64 = 300_000;
    const pin = cfg.pinned_prefill_chunk;

    const wide_bill = bills.prefillNeededAtChunk(&cfg, seq, 2048, kv_bits, 8192, .{});
    const narrow_bill = bills.prefillNeededAtChunk(&cfg, seq, 2048, kv_bits, 4096, .{});
    try t.expect(wide_bill > narrow_bill);
    // 8192 is a width only the gate offers: ungated, this rung forwards at 4096.
    try t.expectEqual(@as(u32, 8192), bills.widthForRung(&cfg, seq, 8192));
    try t.expectEqual(@as(u64, 4096), bills.rungWidth(&cfg, seq, 8192, false));

    // Only an explicit `--prefill-chunk 8192` offers the rung past the 4096 start width.
    const with_margin = wide_bill * (100 + bills.PREFILL_WIDE_RUNG_MARGIN_PCT) / 100;
    try t.expectEqual(@as(u32, 4096), bills.chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, with_margin, pin, 0, .{}));
    try t.expectEqual(@as(u32, 4096), bills.chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, wide_bill, pin, 8192, .{}));
    try t.expectEqual(@as(u32, 4096), bills.chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, with_margin - 1, pin, 8192, .{}));
    try t.expectEqual(@as(u32, 8192), bills.chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, with_margin, pin, 8192, .{}));
    // The rungs the gate did not widen are still taken at their plain bill.
    try t.expectEqual(@as(u32, 4096), bills.chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, narrow_bill, pin, 8192, .{}));
}

test "prefillNeededAtChunk: the deployed pack's 8192 step is at least the measured peak step" {
    const qsa_fused_off = fixtures.qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const t = std.testing;
    const cfg = fixtures.qwen4DeployedTestConfig();
    const kv_bits: u64 = 8;

    for ([_]u64{ 300_000, 326_351 }) |seq| {
        for ([_]bool{ false, true }) |mtp_on| {
            const narrow = bills.prefillNeededAtChunk(&cfg, seq, 2048, kv_bits, 4096, .{ .mtp_on = mtp_on });
            const wide = bills.prefillNeededAtChunk(&cfg, seq, 2048, kv_bits, 8192, .{ .mtp_on = mtp_on });
            // The widest active+cache step the 4096 -> 8192 trade measured on this pack was
            // +3.3 GB; the bill for the step must not sit under what the box pays for it.
            try t.expect(wide - narrow >= 3_300_000_000);
        }
    }
}

test "chooseRequestPrefillChunk: the explicit flag caps it, the gate outranks it" {
    const qsa_fused_off = fixtures.qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const t = std.testing;
    const cfg = fixtures.qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const roomy: u64 = 200 * (@as(u64, 1) << 30);
    const pin = cfg.pinned_prefill_chunk;

    // An explicit `--prefill-chunk` sets the ladder's top: never wider however roomy, the floor when nothing fits.
    try t.expectEqual(@as(u32, 4096), bills.chooseRequestPrefillChunk(&cfg, 300_000, 2048, kv_bits, roomy, pin, 0, .{}));
    try t.expectEqual(@as(u32, 8192), bills.chooseRequestPrefillChunk(&cfg, 300_000, 2048, kv_bits, roomy, pin, 8192, .{}));
    try t.expectEqual(@as(u32, 4096), bills.chooseRequestPrefillChunk(&cfg, 300_000, 2048, kv_bits, roomy, pin, 4096, .{}));
    try t.expectEqual(@as(u32, 2048), bills.chooseRequestPrefillChunk(&cfg, 300_000, 2048, kv_bits, roomy, pin, 2048, .{}));
    try t.expectEqual(bills.widthForRung(&cfg, 300_000, 512), bills.chooseRequestPrefillChunk(&cfg, 300_000, 2048, kv_bits, 0, pin, 2048, .{}));

    // Kill switch: the load-time pin for every request.
    bills.per_request_chunk_override = false;
    defer bills.per_request_chunk_override = null;
    try t.expectEqual(pin, bills.chooseRequestPrefillChunk(&cfg, 300_000, 2048, kv_bits, roomy, pin, 0, .{}));
    bills.per_request_chunk_override = null;

    // Another arch keeps the load-time pin exactly.
    var other = fixtures.qwen4RequestTestConfig();
    other.model_type = "not_a_served_arch";
    try t.expect(!other.perRequestPrefillChunk());
    try t.expectEqual(pin, bills.chooseRequestPrefillChunk(&other, 300_000, 2048, kv_bits, roomy, pin, 0, .{}));
    try t.expect(cfg.perRequestPrefillChunk());
}

test "admission tries the ladder DOWN before refusing: a prompt that fits only at 512" {
    const qsa_fused_off = fixtures.qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // The probe used to price the load-time pin while the scheduler chose a per-request width,
    // so a prompt that fit only at the floor was refused for a forward that never ran.
    const t = std.testing;
    const cfg = fixtures.qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const seq: u64 = 384_000;
    const pin = cfg.pinned_prefill_chunk;
    const floor_width = bills.widthForRung(&cfg, seq, bills.PREFILL_CHUNK_LADDER[bills.PREFILL_CHUNK_LADDER.len - 1]);
    const pin_width = bills.widthForRung(&cfg, seq, pin);

    try t.expect(pin_width > floor_width);
    const at_floor = bills.prefillNeededAtChunk(&cfg, seq, 2048, kv_bits, floor_width, .{});
    const at_pin = bills.prefillNeededAtChunk(&cfg, seq, 2048, kv_bits, pin_width, .{});
    try t.expect(at_floor < at_pin);

    // Exactly enough for the floor width and not a byte more.
    const available = at_floor;

    try t.expectEqual(floor_width, bills.chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, available, pin, 0, .{}));
    try t.expect(bills.prefillNeededAtChunk(&cfg, seq, 2048, kv_bits, floor_width, .{}) <= available);

    // Gate off: the load-time pin does not fit.
    bills.per_request_chunk_override = false;
    defer bills.per_request_chunk_override = null;
    try t.expectEqual(pin, bills.chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, available, pin, 0, .{}));
    try t.expect(bills.prefillNeededAtChunk(&cfg, seq, 2048, kv_bits, pin_width, .{}) > available);
    bills.per_request_chunk_override = null;

    // An explicit `--prefill-chunk` caps the ladder, it does not pin it: the same step down.
    settings_resolve.prefill_chunk.value = 4096;
    settings_resolve.prefill_chunk.source = .flag;
    defer {
        settings_resolve.prefill_chunk.value = 8192;
        settings_resolve.prefill_chunk.source = .default;
    }
    const capped = bills.chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, available, pin, bills.explicitPrefillChunk(), .{});
    try t.expectEqual(floor_width, capped);
    try t.expect(bills.prefillNeededAtChunk(&cfg, seq, 2048, kv_bits, capped, .{}) <= available);
}

test "the post-eviction re-ask never exceeds what live memory affords, and never runs on an arch that has no per-request width" {
    const qsa_fused_off = fixtures.qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // The composition the scheduler runs: the chooser is asked before and after the eviction
    // pass, and the second reading (live memory) runs.
    const t = std.testing;
    const cfg = fixtures.qwen4RequestTestConfig();
    const kv_bits: u64 = 8;
    const seq: u64 = 383_000;
    const pin: u32 = 512;
    const floor_width = bills.widthForRung(&cfg, seq, bills.PREFILL_CHUNK_LADDER[bills.PREFILL_CHUNK_LADDER.len - 1]);

    // Pre-eviction: nothing on the ladder fits, the chooser hands back the floor.
    const admitted = bills.chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, 0, pin, 0, .{});
    try t.expectEqual(floor_width, admitted);

    // Post-eviction, with the cache's bytes returned: a wider rung fits.
    const roomy: u64 = bills.prefillNeededAtChunk(&cfg, seq, 2048, kv_bits, bills.widthForRung(&cfg, seq, 4096), .{});
    const reasked = bills.chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, roomy, pin, 0, .{});
    try t.expect(reasked > admitted);
    try t.expectEqual(reasked, slot_admission.postEvictionPrefillChunk(admitted, reasked).width);

    try t.expectEqual(admitted, slot_admission.postEvictionPrefillChunk(0, admitted).width);

    // The reverse reading (memory moved between the asks) runs at the floor.
    try t.expectEqual(admitted, slot_admission.postEvictionPrefillChunk(reasked, admitted).width);

    // A non-qwen4 arch is untouched end to end.
    var other = fixtures.qwen4RequestTestConfig();
    other.model_type = "not_a_served_arch";
    const other_pre = bills.chooseRequestPrefillChunk(&other, seq, 2048, kv_bits, 0, pin, 0, .{});
    const other_post = bills.chooseRequestPrefillChunk(&other, seq, 2048, kv_bits, roomy, pin, 0, .{});
    try t.expectEqual(pin, other_pre);
    try t.expectEqual(pin, other_post);
    try t.expectEqual(pin, slot_admission.postEvictionPrefillChunk(other_pre, other_post).width);

    bills.per_request_chunk_override = false;
    defer bills.per_request_chunk_override = null;
    const off_pre = bills.chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, 0, pin, 0, .{});
    const off_post = bills.chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, roomy, pin, 0, .{});
    try t.expectEqual(pin, slot_admission.postEvictionPrefillChunk(off_pre, off_post).width);
}

test "GLM serving memory bills one latent at the request's KV width and pooled index per token" {
    const cfg = try model_mod.parseConfigFromJson(std.testing.allocator, @embedFile("fixtures/glm5_config.json"));
    const saved = launch_state.configured_kv_quant;
    defer launch_state.configured_kv_quant = saved;
    launch_state.configured_kv_quant = null;
    try std.testing.expectEqual(@as(u64, 8), bills.defaultKvBits(&cfg));
    try std.testing.expectEqual(@as(u64, 11968), bills.sessionBytesPerToken(&cfg, 16));
    // kv8: 11 layers x (512 codes + 8 BF16 scale/bias pairs) beside the BF16 pooled index.
    try std.testing.expectEqual(@as(u64, 11 * 544 + 704), bills.sessionBytesPerToken(&cfg, 8));
    launch_state.configured_kv_quant = trunk.KVQuantConfig.dense;
    try std.testing.expectEqual(@as(u64, 16), bills.defaultKvBits(&cfg));
    launch_state.configured_kv_quant = null;
    // kv8 retains 480 fewer growth bytes per token and adds the pending MLA layer's dequant and quantizer scratch.
    try std.testing.expectEqual(bills.prefillTransientReserveAtKv(&cfg, 16, 2048, 65536) + (2051 * 1024 + 2048 * 544), bills.prefillTransientReserveAtKv(&cfg, 8, 2048, 65536) + 65536 * 480);
    try std.testing.expect(bills.prefillNeededAtChunk(&cfg, 65536, 1024, 8, 2048, .{}) < bills.prefillNeededAtChunk(&cfg, 65536, 1024, 16, 2048, .{}));
    try std.testing.expectEqual(cfg.ssmCheckpointBytes() + cfg.qsaRingBytes(), bills.slotRingBytes(&cfg, 16));
    try std.testing.expectEqual(@as(u64, 0), bills.ctxSizingCacheReserve(&cfg));
    try std.testing.expectEqual(@as(u64, 704), bills.statePerTokenBilled(&cfg));
    try std.testing.expectEqual(@as(u64, 0), cfg.layerKvBytes(0));
    try std.testing.expectEqual(@as(u64, 1024), cfg.layerKvBytes(3));
    try std.testing.expect(bills.prefillTransientReserveAtKv(&cfg, 16, 2048, 500000) < 6 * 1024 * 1024 * 1024);
}

test "a streamed GLM bills its fill peak per request and its latent at the KV width plus the pooled index per planned token" {
    var cfg = try model_mod.parseConfigFromJson(std.testing.allocator, @embedFile("fixtures/glm5_config.json"));
    const saved = launch_state.configured_kv_quant;
    defer launch_state.configured_kv_quant = saved;
    var planned: [2]u64 = undefined;
    for ([_]trunk.KVQuantConfig{ .dense, trunk.KVQuantConfig.affine(8) }, [_]u64{ 16, 8 }, &planned) |kv, bits, *bytes| {
        launch_state.configured_kv_quant = kv;
        try std.testing.expectEqual(bits, bills.defaultKvBits(&cfg));
        cfg.expert_streaming = false;
        const resident = bills.prefillNeededAtChunk(&cfg, 4096, 2048, bits, 2048, .{});
        cfg.expert_streaming = true;
        cfg.expert_fill_peak_bytes = 3 << 30;
        try std.testing.expectEqual(resident + (3 << 30), bills.prefillNeededAtChunk(&cfg, 4096, 2048, bits, 2048, .{}));
        bytes.* = bills.expertStreamingPlannedKvBytes(&cfg);
        try std.testing.expectEqual(bills.sessionBytesPerToken(&cfg, bits) * bills.expertStreamingPlannedSeq(&cfg) + bills.slotRingBytes(&cfg, bits), bytes.*);
    }
    try std.testing.expect(planned[1] < planned[0]);
}

test "GLM requests prefill at 2048 and step down a rung only where a chunk stops fitting" {
    const t = std.testing;
    const cfg = try fixtures.glmAssistantFixture();
    const kv_bits: u64 = 8;
    try t.expect(cfg.perRequestPrefillChunk());
    try t.expectEqual(@as(u32, 2048), bills.chooseRequestPrefillChunk(&cfg, 4096, 256, kv_bits, 1 << 40, 2048, 0, .{}));
    const unpinned = bills.explicitPrefillChunk() == 0;
    try t.expectEqual(unpinned, bills.adaptivePrefillChunkEnabled(&cfg));

    // Release defaults: a 1M request is admitted with a 2048-row tail, and 2048 still fits there.
    const available: u64 = 115_904 * 1024 * 1024 - 104_350_000_000;
    const top: u64 = 1_048_576 - 256;
    try t.expectEqual(@as(u32, 2048), bills.chooseRequestPrefillChunk(&cfg, top, 256, kv_bits, available, 2048, 0, .{}));
    var st: mtp_plan.AdaptiveWidthState = .{};
    const kv_top = bills.sessionBytesPerToken(&cfg, kv_bits) * top;
    try t.expectEqual(@as(u32, 2048), bills.adaptivePrefillWidth(&cfg, kv_bits, top, available - kv_top, 2048, 2048, &st));

    // A tighter box: the top only fits at 1024 rows, and that tail's bill is what admits it.
    const at_2048 = bills.prefillNeededAtChunk(&cfg, top, 256, kv_bits, 2048, .{});
    try t.expectEqual(@as(u32, 1024), bills.chooseRequestPrefillChunk(&cfg, top, 256, kv_bits, at_2048 - 1, 2048, 0, .{}));
    try t.expect(bills.prefillNeededAtChunk(&cfg, top, 256, kv_bits, 1024, .{}) <= at_2048 - 1);
    // Early chunks keep 2048; the chunk that no longer fits beside the KV steps down one rung.
    st = .{};
    try t.expectEqual(@as(u32, 2048), bills.adaptivePrefillWidth(&cfg, kv_bits, 4096, bills.prefillChunkCost(&cfg, kv_bits, 2048, 4096), 2048, 2048, &st));
    try t.expectEqual(@as(u32, 1024), bills.adaptivePrefillWidth(&cfg, kv_bits, top, bills.prefillChunkCost(&cfg, kv_bits, 2048, top) - 1, 2048, 2048, &st));
    // Once stepped down, a request never widens again.
    try t.expectEqual(@as(u32, 1024), bills.adaptivePrefillWidth(&cfg, kv_bits, top, available, 1024, 2048, &st));

    // A small box narrows from the first chunk.
    st = .{};
    try t.expectEqual(@as(u32, 512), bills.adaptivePrefillWidth(&cfg, kv_bits, 0, bills.prefillChunkCost(&cfg, kv_bits, 512, 0), 2048, 2048, &st));
}

test "GLM near the top of its context keeps fewer checkpoints at the start width instead of being refused" {
    var cfg = try model_mod.parseConfigFromJson(std.testing.allocator, @embedFile("fixtures/glm5_config.json"));
    cfg.dflash_loaded = true;
    cfg.dflash_window_bytes = 5 * 8 * 128 * 4 * 2304;
    const saved = .{ launch_state.prefix_cache_capacity, launch_state.ssm_checkpoint_stride, launch_state.ssm_checkpoint_max };
    defer {
        launch_state.prefix_cache_capacity = saved[0];
        launch_state.ssm_checkpoint_stride = saved[1];
        launch_state.ssm_checkpoint_max = saved[2];
    }
    launch_state.ssm_checkpoint_stride = 256;
    launch_state.ssm_checkpoint_max = 16;
    launch_state.prefix_cache_capacity = 32;
    const seq: u64 = 1_040_000;
    const width: u64 = bills.rungWidth(&cfg, seq, cfg.prefillStartWidth(), false);
    // Free memory admits the prompt with its prompt-end checkpoint and one grid copy, no more.
    const Fits = struct {
        cfg: *const model_mod.ModelConfig,
        seq: u64,
        width: u64,
        available: u64,
        fn at(ctx: ?*anyopaque, n: u32) bool {
            const self: *@This() = @ptrCast(@alignCast(ctx.?));
            return bills.prefillNeededAtChunk(self.cfg, self.seq, 2048, 8, self.width, .{ .checkpoints = n }) <= self.available;
        }
    };
    var probe = Fits{ .cfg = &cfg, .seq = seq, .width = width, .available = bills.prefillNeededAtChunk(&cfg, seq, 2048, 8, width, .{ .checkpoints = 1 }) };
    const cap = glm5_prefix.checkpointMax(16);
    try std.testing.expect(bills.prefillNeededAtChunk(&cfg, seq, 2048, 8, width, .{ .checkpoints = cap }) > probe.available);
    try std.testing.expectEqual(@as(?u32, 1), slot_admission.fewerCheckpointsToAdmit(cap, &probe, Fits.at));
    // The checkpoints yield to the width: the prefill still starts at the start width.
    try std.testing.expectEqual(@as(u32, @intCast(width)), bills.chooseRequestPrefillChunk(&cfg, seq, 2048, 8, probe.available, cfg.pinned_prefill_chunk, 0, .{ .checkpoints = cap }));
    // Room for the prompt alone keeps none; less than that is refused for the prompt, not the hold.
    probe.available = bills.prefillNeededAtChunk(&cfg, seq, 2048, 8, width, .{ .checkpoints = 0 });
    try std.testing.expectEqual(@as(?u32, 0), slot_admission.fewerCheckpointsToAdmit(cap, &probe, Fits.at));
    probe.available -= 1;
    try std.testing.expectEqual(@as(?u32, null), slot_admission.fewerCheckpointsToAdmit(cap, &probe, Fits.at));
}

test "MiMo honors the declared wired limit streamed or resident, without enabling long-context paths" {
    const prior = bills.wired_limit_mb_override;
    bills.wired_limit_mb_override = 120000;
    defer bills.wired_limit_mb_override = prior;
    const floor = bills.wiredLimitFloor(120000 * 1024 * 1024, 128 << 30, launch_state.wired_limit_margin_bytes);
    var c = model_mod.ModelConfig{ .model_type = "mimo_v2", .expert_streaming = true };
    try std.testing.expect(!c.longCtxGated());
    try std.testing.expectEqual(floor, bills.wiredCeilingFloorForRam(&c, 128 << 30));
    c.expert_streaming = false;
    try std.testing.expectEqual(floor, bills.wiredCeilingFloorForRam(&c, 128 << 30));
    // An arch the engine does not serve keeps the free-RAM ceiling.
    const other = model_mod.ModelConfig{ .model_type = "not_a_served_arch" };
    try std.testing.expectEqual(@as(u64, 0), bills.wiredCeilingFloorForRam(&other, 128 << 30));
}

test "diskContextForLoad pins the served context, so the Allocating line names what /props and /v1/models name" {
    const t = std.testing;
    var cfg = fixtures.qwen4DeployedTestConfig();
    cfg.pinned_context = 0;
    const saved_ctx = launch_state.server_config.max_context_size;
    defer launch_state.server_config.max_context_size = saved_ctx;
    launch_state.server_config.max_context_size = 0;
    const ctx = bills.diskContextForLoad(&cfg);
    try t.expect(cfg.pinned_context != 0);
    try t.expectEqual(@as(u64, cfg.pinned_context), ctx);
    try t.expectEqual(@as(u64, bills.getEffectiveContextLength(&cfg)), ctx);
}

test "SSD-first is gated on a DISK TIER: with --prefix-cache-disk off, qwen4_exp takes the RAM arm" {
    const qsa_fused_off = fixtures.qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // `ssdFirstEnabled()` never checked that a tier exists, and `--prefix-cache-disk` is off by default.
    const t = std.testing;
    var cfg = fixtures.qwen4RequestTestConfig();

    const orig_disk = launch_state.prefix_cache_disk_bytes;
    defer launch_state.prefix_cache_disk_bytes = orig_disk;
    // `prefixCacheMemForLoad` publishes into a process global; put it back or a later test reads it.
    defer bills.clearResolvedPrefixCacheMem();
    const orig_over = prefix_cache_mod.ssd_first_override;
    defer prefix_cache_mod.ssd_first_override = orig_over;
    prefix_cache_mod.ssd_first_override = true; // the env switch is not what is on trial
    // The budget is a property of the machine; the test's machine is a 128 GB one, not the runner.
    const orig_ceiling = bills.static_ceiling_override;
    defer bills.static_ceiling_override = orig_ceiling;
    bills.static_ceiling_override = 109_395 * (1 << 20);

    try t.expect(!prefix_cache_mod.ssdFirstActive(&cfg, false, true));
    try t.expect(prefix_cache_mod.ssdFirstActive(&cfg, true, true));
    var dense = model_mod.ModelConfig{};
    dense.model_type = "not_a_served_arch";
    try t.expect(!prefix_cache_mod.ssdFirstActive(&dense, true, true));
    try t.expect(prefix_cache_mod.ssdFirstActive(&dense, true, false));
    prefix_cache_mod.ssd_first_override = false;
    try t.expect(!prefix_cache_mod.ssdFirstActive(&cfg, true, true));
    prefix_cache_mod.ssd_first_override = true;

    // The SSD arm floors at one session's KV, the RAM arm is a residual bounded by the ask.
    const orig_ctx = launch_state.server_config.max_context_size;
    defer launch_state.server_config.max_context_size = orig_ctx;
    launch_state.server_config.max_context_size = 1_048_576; // explicit: no auto-sizing in the way
    cfg.pinned_context = 0;
    const ask: u64 = 4 * 1024 * 1024 * 1024;

    launch_state.prefix_cache_disk_bytes = 0;
    var idle_off: u64 = 12345;
    const ram_arm = bills.prefixCacheMemForLoad(&cfg, ask, .{}, &idle_off);
    try t.expectEqual(@as(u64, 0), idle_off);
    try t.expect(ram_arm <= ask);

    launch_state.prefix_cache_disk_bytes = 64 * 1024 * 1024 * 1024;
    var idle_on: u64 = 0;
    const ssd_arm = bills.prefixCacheMemForLoad(&cfg, ask, .{}, &idle_on);
    try t.expect(ssd_arm > ask);
    try t.expect(ssd_arm > ram_arm);
    try t.expect(idle_on > 0 and idle_on <= ask);
    try t.expect(ssd_arm -| idle_on > ask); // the one-session floor
}

test "autoContextFor: the safety margin applies to MEMORY, never to the model's own max" {
    const original = launch_state.server_config.max_context_size;
    defer launch_state.server_config.max_context_size = original;
    launch_state.server_config.max_context_size = 0;

    // A tiny model whose `max_position_embeddings` is far below anything memory
    // could constrain: it must get its FULL declared context, un-margined.
    // (Regression: applying the margin AFTER the model-max clamp shaved context off a
    // 131,072-token checkpoint that fits in RAM with room to spare.)
    var small = model_mod.ModelConfig{};
    small.max_position_embeddings = 4096;
    small.num_attention_heads = 8;
    small.num_hidden_layers = 4;
    small.num_key_value_heads = 2;
    small.head_dim = 64;
    small.hidden_size = 512;
    small.intermediate_size = 1024;
    try testing.expectEqual(@as(u32, 4096), bills.autoContextFor(&small));

    // No declared max: fall back to the margined memory ceiling.
    var unbounded = small;
    unbounded.max_position_embeddings = 0;
    try testing.expect(bills.autoContextFor(&unbounded) > 0);
    const memory_ctx = bills.memoryContextAt(&unbounded, 96 << 30, 40 << 30);
    try testing.expectEqual(bills.safeAutoContext(memory_ctx), bills.autoContextFrom(memory_ctx, unbounded.contextCap(), 0));
}

test "pinAutoContext is idempotent and a no-op under --ctx-size" {
    const original = launch_state.server_config.max_context_size;
    defer launch_state.server_config.max_context_size = original;

    var config = model_mod.ModelConfig{};
    config.max_position_embeddings = 131072;
    config.num_attention_heads = 8;
    config.num_hidden_layers = 42;
    config.num_key_value_heads = 2;
    config.head_dim = 256;

    launch_state.server_config.max_context_size = 0;
    const first = bills.pinAutoContext(&config);
    try testing.expect(first > 0);
    try testing.expectEqual(first, config.pinned_context);
    // Second call must not re-read memory and shift the value.
    try testing.expectEqual(first, bills.pinAutoContext(&config));

    // Explicit --ctx-size: never overwrite, always report the override.
    var manual = model_mod.ModelConfig{};
    manual.max_position_embeddings = 131072;
    launch_state.server_config.max_context_size = 8192;
    try testing.expectEqual(@as(u32, 8192), bills.pinAutoContext(&manual));
    try testing.expectEqual(@as(u32, 0), manual.pinned_context);
    try testing.expect(manual.pinned_prefill_chunk > 0);
}

test "getEffectiveContextLength computes safe default from GPU memory" {
    const original = launch_state.server_config.max_context_size;
    defer launch_state.server_config.max_context_size = original;

    launch_state.server_config.max_context_size = 0;
    var config = model_mod.ModelConfig{};
    config.max_position_embeddings = 131072;
    config.num_attention_heads = 8;
    config.num_hidden_layers = 42;
    config.num_key_value_heads = 2;
    config.head_dim = 256;
    // Should compute a value based on GPU memory, not hardcoded 16K
    const computed = bills.getEffectiveContextLength(&config);
    try testing.expect(computed > 0);
    try testing.expect(computed <= config.max_position_embeddings);

    // Explicit --ctx-size overrides the computed default
    launch_state.server_config.max_context_size = 32768;
    try testing.expectEqual(@as(u32, 32768), bills.getEffectiveContextLength(&config));
}

test "an explicitly raised iogpu.wired_limit_mb is a FLOOR under the ceiling" {
    const whole_rebuild = fixtures.qsaWholeRebuildGuard();
    defer whole_rebuild.deinit();
    // Bar: a raised sysctl lifts the free-RAM term and nothing else; every other arch and the
    // default sysctl keep the old expression byte for byte.
    const t = std.testing;
    qwen4_qsa_select.qsa_score_fused_override = false;
    defer qwen4_qsa_select.qsa_score_fused_override = null;
    const mb: u64 = 1024 * 1024;
    const total_ram: u64 = 131_072 * mb;
    const working_set: u64 = 120_000 * mb;
    const saved = bills.wired_limit_mb_override;
    defer bills.wired_limit_mb_override = saved;
    bills.wired_limit_mb_override = 120_000;

    try t.expectEqual(@as(u64, 1 << 30), launch_state.WIRED_LIMIT_MARGIN_BYTES);
    try t.expectEqual(@as(u64, 118_976), bills.wiredLimitFloor(120_000 * mb, total_ram, launch_state.WIRED_LIMIT_MARGIN_BYTES) / mb);
    // The macOS default (75% of RAM) and anything under it declares nothing; and however
    // absurd the sysctl, never plan within the margin of physical RAM.
    try t.expectEqual(@as(u64, 0), bills.wiredLimitFloor(98_304 * mb, total_ram, launch_state.WIRED_LIMIT_MARGIN_BYTES));
    try t.expectEqual(@as(u64, 0), bills.wiredLimitFloor(0, total_ram, launch_state.WIRED_LIMIT_MARGIN_BYTES));
    try t.expectEqual(@as(u64, 0), bills.wiredLimitFloor(120_000 * mb, 0, launch_state.WIRED_LIMIT_MARGIN_BYTES));
    try t.expectEqual(total_ram - launch_state.WIRED_LIMIT_MARGIN_BYTES, bills.wiredLimitFloor(400 * (1 << 30), total_ram, launch_state.WIRED_LIMIT_MARGIN_BYTES));

    const box96: u64 = 96_000 * mb;
    const six: u64 = 6 << 30;
    try t.expectEqual(@as(u64, 90_000) * mb - six, bills.wiredLimitFloor(90_000 * mb, box96, six));
    try t.expectEqual(box96 - six, bills.wiredLimitFloor(96_000 * mb, box96, six));
    try t.expectEqual(@as(u64, 90_000) * mb - launch_state.WIRED_LIMIT_MARGIN_BYTES, bills.wiredLimitFloor(90_000 * mb, box96, launch_state.WIRED_LIMIT_MARGIN_BYTES));

    // Gated: every other arch takes floor 0, which is `physicalMemoryCeiling` exactly.
    var cfg = fixtures.qwen4ExpLive364kConfig();
    try t.expect(bills.wiredCeilingFloorForRam(&cfg, total_ram) > 0);
    try t.expectEqual(@as(u64, 0), bills.wiredCeilingFloorForRam(null, total_ram));
    for ([_][]const u8{"not_a_served_arch"}) |mt| {
        var other = cfg;
        other.model_type = mt;
        try t.expect(!other.longCtxGated());
        try t.expectEqual(@as(u64, 0), bills.wiredCeilingFloorForRam(&other, total_ram));
    }
    const footprint: u64 = 87_722 * mb;
    for ([_]u64{ 0, 12_934 * mb, 400 * (1 << 30) }) |free| {
        try t.expectEqual(
            bills.physicalMemoryCeiling(working_set, footprint, free),
            bills.gpuCeilingWithWiredFloor(working_set, footprint, free, 0),
        );
    }

    // A 448k warm append that the free-RAM term refused fits under the floor.
    const kv_bits: u64 = 8;
    const seq: u64 = 448_604;
    const max_tokens: u32 = @intCast(bills.getEffectiveContextLength(&cfg) - seq);
    const shared = bills.WarmPrefix{ .matched_tokens = 448_531, .capacity_tokens = 448_531, .will_donate = false };
    const needed = bills.prefillNeededAtChunk(&cfg, seq, max_tokens, kv_bits, 512, shared);
    try t.expectEqual(@as(u64, 11_456), needed / mb);
    try t.expect(needed > bills.gpuCeilingWithWiredFloor(working_set, footprint, 10_000 * mb, 0) -| footprint);
    const lifted = bills.gpuCeilingWithWiredFloor(working_set, footprint, 12_934 * mb, bills.wiredCeilingFloorForRam(&cfg, total_ram));
    try t.expectEqual(@as(u64, 31_254), (lifted -| footprint) / mb);
    try t.expect(needed <= lifted -| footprint);

    // Under a declared limit the margin holds however much RAM is free: a load on a quiet box saw the
    // working set, the margin gone, and proved a context its own requests then refused.
    const floor = bills.wiredCeilingFloorForRam(&cfg, total_ram);
    try t.expectEqual(floor, bills.gpuCeilingWithWiredFloor(working_set, 0, working_set, floor));
    try t.expectEqual(floor, bills.gpuCeilingWithWiredFloor(working_set, footprint, 400 * (1 << 30), floor));
}

test "the admission probe bills the request's OWN kv-quant and chunking, not the process defaults" {
    const qsa_fused_off = fixtures.qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    // The probe used to hardcode `kv_override = null`, pricing a `kv_quant: 4` request at fp16 on the inference thread.
    const t = std.testing;
    const cfg = fixtures.qwen4ExpOomConfig();
    const seq: usize = 200_000;
    const q4 = trunk.KVQuantConfig.affine(4);
    const q8 = trunk.KVQuantConfig.affine(8);
    const bill4 = bills.prefillAdmissionBill(&cfg, seq, 2048, q4, false, null, .{});
    const bill8 = bills.prefillAdmissionBill(&cfg, seq, 2048, q8, false, null, .{});
    const bill16 = bills.prefillAdmissionBill(&cfg, seq, 2048, trunk.KVQuantConfig.dense, false, null, .{});
    try t.expect(bill4.needed < bill8.needed);
    try t.expect(bill8.needed < bill16.needed);
}

test "a second GLM admission leaves free the cache growth the first was admitted to allocate" {
    var cfg = try model_mod.parseConfigFromJson(std.testing.allocator, @embedFile("fixtures/glm5_config.json"));
    cfg.pinned_context = 1_048_576;
    const gib: u64 = 1 << 30;
    const prompt: usize = 16;
    const max_tokens: u32 = 1_048_560;
    for ([_]u64{ 16, 8 }) |kv_bits| {
        const bill = bills.prefillAdmissionBill(&cfg, prompt, max_tokens, if (kv_bits == 16) trunk.KVQuantConfig.dense else trunk.KVQuantConfig.affine(8), false, null, .{});
        const nums = bills.prefillBillNumbersNow(&cfg, prompt, max_tokens, if (kv_bits == 16) trunk.KVQuantConfig.dense else trunk.KVQuantConfig.affine(8), false, false);
        const commit = bills.admissionCommitBytes(&cfg, prompt, max_tokens, kv_bits);
        try std.testing.expectEqual(commit, nums.commit);
        // What the request retains is most of its bill, and it is not resident at admission.
        try std.testing.expect(commit > bill.needed / 2 and commit <= bill.needed);
        // Headroom that admits A alone and, a tick later with A's growth unallocated, B.
        const headroom = bill.needed + gib;
        const b = slot_admission.MemoryBill{ .needed = bill.needed, .available = headroom };
        try std.testing.expectEqual(@as(usize, 1), slot_admission.admitsWithinMemory(&.{b}, true, 0));
        try std.testing.expectEqual(@as(usize, 0), slot_admission.admitsWithinMemory(&.{b}, true, commit));
    }
    // kv8 audit shape: one request's eventual state is 6.669 GiB.
    const kv8 = bills.admissionCommitBytes(&cfg, prompt, max_tokens, 8);
    try std.testing.expect(kv8 > 6 * gib + gib * 2 / 3 and kv8 < 6 * gib + gib * 7 / 10);
    try std.testing.expect(bills.admissionCommitBytes(&cfg, prompt, max_tokens, 16) > kv8);
}

test "GLM raw FP8 honors the declared wired limit with the same reserve as MiMo" {
    const prior = bills.wired_limit_mb_override;
    bills.wired_limit_mb_override = 120000;
    defer bills.wired_limit_mb_override = prior;
    const cfg = model_mod.ModelConfig{ .model_type = "glm5_next", .glm_fp8_trunk = true };
    try std.testing.expectEqual(bills.wiredLimitFloor(120000 * 1024 * 1024, 128 << 30, launch_state.wired_limit_margin_bytes), bills.wiredCeilingFloorForRam(&cfg, 128 << 30));
    bills.wired_limit_mb_override = 98304;
    try std.testing.expectEqual(@as(u64, 0), bills.wiredCeilingFloorForRam(&cfg, 128 << 30));
}

