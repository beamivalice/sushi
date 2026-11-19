//! Flash-Next bills against live memory: the request chunk chooser, SSD-first and the reservation clamp.
const bills = @import("../bills.zig");
const fixtures = @import("../bills_fixtures_test.zig");
const launch_state = @import("../server_state.zig");
const model_mod = @import("../model.zig");
const model_load = @import("../model_load.zig");
const http_conn = @import("../http_conn.zig");
const prefix_cache_mod = @import("../prefix_cache.zig");
const qwen4_qsa_select = @import("qwen4_qsa_select.zig");
const settings_resolve = @import("../settings_resolve.zig");
const slot_admission = @import("../slot_admission.zig");
const std = @import("std");
const trunk = @import("../trunk.zig");
const request_policy = @import("../request_policy.zig");
const clampMaxTokens = request_policy.clampMaxTokens;
const server = @import("../server.zig");
const prefillRequestTerms = server.prefillRequestTerms;
const qsaScoreFusedOffGuard = @import("qwen4_bill_test.zig").qsaScoreFusedOffGuard;
const qsaWholeRebuildGuard = @import("qwen4_bill_test.zig").qsaWholeRebuildGuard;
const qwen4ExpOomConfig = fixtures.qwen4ExpOomConfig;

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
    const qsa_fused_off = qsaScoreFusedOffGuard();
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
    const qsa_fused_off = qsaScoreFusedOffGuard();
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

test "chooseRequestPrefillChunk: on Sushi Cooler the widened rung clears its exact bill alone, no estimator margin" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const t = std.testing;
    var cfg = fixtures.qwen4RequestTestConfig();
    cfg.cooler_for_test = true;
    cfg.expert_layout = .exl3_k4;
    cfg.expert_quant_codebook = .spec1;
    cfg.trunk_quant = null;
    cfg.quant_bits = 0;
    try t.expect(bills.coolerPrefillRowBytes(&cfg) > 0);
    const kv_bits: u64 = 8;
    const seq: u64 = 300_000;
    const pin = cfg.pinned_prefill_chunk;
    try t.expectEqual(@as(u64, 4096), bills.rungWidth(&cfg, seq, 8192, false));
    const wide_bill = bills.prefillNeededAtChunk(&cfg, seq, 2048, kv_bits, 8192, .{});
    try t.expectEqual(@as(u32, 8192), bills.chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, wide_bill, pin, 8192, .{}));
    try t.expectEqual(@as(u32, 4096), bills.chooseRequestPrefillChunk(&cfg, seq, 2048, kv_bits, wide_bill - 1, pin, 8192, .{}));
}

test "prefillNeededAtChunk: the deployed pack's 8192 step is at least the measured peak step" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
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
    const qsa_fused_off = qsaScoreFusedOffGuard();
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
    const qsa_fused_off = qsaScoreFusedOffGuard();
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
    const qsa_fused_off = qsaScoreFusedOffGuard();
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

test "SSD-first is gated on a DISK TIER: with --prefix-cache-disk off, qwen4_exp takes the RAM arm" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
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

test "an explicitly raised iogpu.wired_limit_mb is a FLOOR under the ceiling" {
    const whole_rebuild = qsaWholeRebuildGuard();
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
    try t.expectEqual(@as(u64, 17_501), needed / mb);
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
    const qsa_fused_off = qsaScoreFusedOffGuard();
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

test "the reservation is bounded by the CONTEXT: an omitted max_tokens cannot bill 26 TB" {
    const qsa_fused_off = qsaScoreFusedOffGuard();
    defer qsa_fused_off.deinit();
    const t = std.testing;
    // An omitted `max_tokens` was maxInt(u32)/4 and the memory guard ran before `clampMaxTokens`,
    // so the reservation billed ~26 TB of KV and refused every long prompt. A budget past the
    // window is still a client's to send.
    const KVCache = trunk.KVCache;
    const ctx: u64 = 262_144;
    const seq: u64 = 50_000;
    const huge: u64 = std.math.maxInt(u32) / 4;

    // (1) The cache clamps: the reservation cannot exceed the context plus a chunk.
    const reserved = KVCache.reservedTokens(seq, huge, 4096, ctx);
    // Bounded by the context: the prompt, the room past it, and a chunk.
    try t.expectEqual(ctx + 4096, reserved);
    // (2) The four surfaces clamp first and hand the guard the clamped value.
    const clamped: u64 = clampMaxTokens(@intCast(huge), true, @intCast(seq), @intCast(ctx));
    try t.expectEqual(ctx - seq, clamped);
    try t.expectEqual(reserved, KVCache.reservedTokens(seq, clamped, 4096, ctx));

    try t.expectEqual(seq + 2048 + 4096, KVCache.reservedTokens(seq, 2048, 4096, ctx));
    // A context of 0 (none known; a served request always has one) leaves no room to generate into.
    try t.expectEqual(seq + 4096, KVCache.reservedTokens(seq, huge, 4096, 0));
    try t.expectEqual(ctx + 4096, KVCache.reservedTokens(ctx, huge, 4096, ctx));

    const cfg = qwen4ExpOomConfig();
    const terms = prefillRequestTerms(&cfg, seq, clamped, 8, 4096, .{});
    try t.expect(terms.reserved_kv_bytes < 4 * 1024 * 1024 * 1024);
}

test "Flash-Next streamed: a request refused at admission for its max_tokens names the max_tokens that fits, and no empty prefix-cache clause (QWEN4_STQ_TINY)" {
    const t = std.testing;
    const a = t.allocator;
    const dir = std.mem.span(std.c.getenv("QWEN4_STQ_TINY") orelse return error.SkipZigTest);
    const saved_ctx = launch_state.server_config.max_context_size;
    defer launch_state.server_config.max_context_size = saved_ctx;
    const saved_mtp = launch_state.configured_mtp;
    defer launch_state.configured_mtp = saved_mtp;
    launch_state.configured_mtp = null;
    var config = try model_mod.parseConfig(t.io, a, dir);
    defer config.deinit(a);
    model_load.applyModelSettings(&config, .{});
    launch_state.server_config.max_context_size = config.contextCap();
    // The smallest budget the tiny pack plans: a streamed load, its experts read from the SSD.
    var budget: u64 = 1 << 30;
    const plan = while (budget <= 64 << 30) : (budget += 1 << 30) {
        break model_load.planExpertStreamingWith(t.io, a, &config, dir, budget, false, 0) catch continue;
    } else return error.TestUnexpectedResult;
    model_load.applyStreamingPlan(&config, plan, budget);
    try t.expect(config.expert_streaming and !config.expert_resident);
    // A declared GPU limit whose floor is `room` past what the process holds: the streamed load's ceiling.
    const ram = @import("../status.zig").getTotalMemBytes();
    const saved_margin = launch_state.wired_margin_flag_bytes;
    defer launch_state.wired_margin_flag_bytes = saved_margin;
    bills.wired_limit_mb_override = ram >> 20;
    defer bills.wired_limit_mb_override = null;
    var prompt: [15]u32 = undefined;
    for (&prompt, 0..) |*p, i| p.* = @intCast(i + 10);
    const asked: u32 = @intCast(config.contextCap() - prompt.len);
    const Ask = struct {
        fn at(max_tokens: u32, cfg: *const model_mod.ModelConfig, ids: []const u32, out: []u8) !struct { admitted: bool, body: []const u8 } {
            var sv: [2]std.posix.fd_t = undefined;
            try t.expect(std.c.socketpair(1, 1, 0, &sv) == 0);
            defer _ = std.c.close(sv[1]);
            var conn: http_conn.Conn = undefined;
            http_conn.Conn.init(&conn, .{ .socket = .{ .handle = sv[0], .address = undefined } }, t.io);
            const admitted = try server.checkAttentionMemory(t.allocator, &conn, ids, max_tokens, cfg, "", false, null, false, false);
            _ = std.c.close(sv[0]);
            const raw = server.readAllFromPair(sv[1], out);
            return .{ .admitted = admitted, .body = raw[(std.mem.indexOf(u8, raw, "\r\n\r\n") orelse 0)..] };
        }
    };
    const active = @import("../gpu_mem.zig").activeBytes();
    var buf: [8192]u8 = undefined;
    // The least room that admits a one-token reply: there the prompt fits and the whole context does not.
    var lo: u64 = 1 << 20;
    var hi: u64 = 4 << 30;
    while (hi - lo > 1 << 20) {
        const mid = lo + (hi - lo) / 2;
        launch_state.wired_margin_flag_bytes = (ram -| mid) -| active;
        if ((try Ask.at(1, &config, &prompt, &buf)).admitted) hi = mid else lo = mid;
    }
    launch_state.wired_margin_flag_bytes = (ram -| hi) -| active;
    const got = try Ask.at(asked, &config, &prompt, &buf);
    errdefer std.debug.print("the streamed load's refusal at {d} MB of room: {s}\n", .{ hi >> 20, got.body });
    try t.expect(!got.admitted);
    try t.expect(std.mem.indexOf(u8, got.body, "Reduce prompt size") == null);
    try t.expect(std.mem.indexOf(u8, got.body, "prefix cache can give back") == null);
    try t.expect(std.mem.indexOf(u8, got.body, "--ctx-size") == null);
    const key = " fits: lower max_tokens to ";
    const at = std.mem.indexOf(u8, got.body, key) orelse return error.TestUnexpectedResult;
    const rest = got.body[at + key.len ..];
    const fits = try std.fmt.parseInt(u32, rest[0..std.mem.indexOfScalar(u8, rest, ' ').?], 10);
    // The max_tokens it names is admitted on the same memory.
    var again: [8192]u8 = undefined;
    try t.expect((try Ask.at(fits, &config, &prompt, &again)).admitted);
}

test "Flash-Next resident: the ssd budget a refused resident load offers loads it streamed on the memory that refused it (QWEN4_STQ_TINY)" {
    const t = std.testing;
    const a = t.allocator;
    const dir = std.mem.span(std.c.getenv("QWEN4_STQ_TINY") orelse return error.SkipZigTest);
    var config = try model_mod.parseConfig(t.io, a, dir);
    defer config.deinit(a);
    model_load.applyModelSettings(&config, .{});
    try t.expect(!config.expert_streaming and config.supportsExpertStreaming());
    const planner = model_load.BudgetPlanner{ .io = t.io, .allocator = a, .config = &config, .model_dir = dir, .vision = false, .assistant = 0, .beside = 0, .mtp_on = false };
    // Room for the smallest streamed plan and one GiB more, under what the whole pack streamed at 512 GiB needs.
    var floor: u64 = 1;
    while (planner.needAt(floor << 30) == null) floor += 1;
    const avail = planner.needAt(floor << 30).? + (1 << 30);
    const fits = bills.streamedBudgetThatFits(512 << 30, avail, planner);
    try t.expect(fits >= floor);
    try t.expect(planner.needAt(fits << 30).? <= avail);
}

test "Flash-Next streamed: the ssd budget a refused budget names is admitted under the same declared limit (QWEN4_STQ_TINY)" {
    const t = std.testing;
    const a = t.allocator;
    const dir = std.mem.span(std.c.getenv("QWEN4_STQ_TINY") orelse return error.SkipZigTest);
    var config = try model_mod.parseConfig(t.io, a, dir);
    defer config.deinit(a);
    model_load.applyModelSettings(&config, .{});
    const gib: u64 = 1 << 30;
    // A 32 GB Mac with iogpu.wired_limit_mb 28000: its floor under the 1 GiB margin is the ceiling every check reads.
    const readings = bills.FitReadings{ .wired_limit = 28000 << 20, .wired_floor = 26976 << 20, .budget_working_set = 21 * gib, .working_set = 21 * gib, .free = 20 * gib, .total_ram = 32 * gib };
    const asked: u64 = 30 * gib;
    var planned = config;
    model_load.applyStreamingPlan(&planned, try model_load.planExpertStreamingWith(t.io, a, &config, dir, asked, false, 0), asked);
    const refused = bills.expertCacheRefusal(&planned, 0, readings) orelse return error.TestExpectedRefusal;
    try t.expectEqual(error.SsdBudgetExceedsWiredLimit, refused.err);
    const plan_at = model_load.BudgetPlanner{ .io = t.io, .allocator = a, .config = &config, .model_dir = dir, .vision = false, .assistant = 0, .beside = 0, .mtp_on = false };
    const planner = model_load.LoadPlanner{ .plan = plan_at, .readings = readings };
    const fits = bills.streamedBudgetThatFits(asked, 0, planner);
    try t.expect(fits > 0 and fits < 30);
    try t.expectEqual(@as(?u64, 0), planner.needAt(fits << 30));
    try t.expect(planner.needAt((fits + 1) << 30).? > 0);
}

test "Flash-Next streamed: an ssd budget refusal for a pack with no smaller published pack names none (QWEN4_STQ_TINY)" {
    const t = std.testing;
    const a = t.allocator;
    const dir = std.mem.span(std.c.getenv("QWEN4_STQ_TINY") orelse return error.SkipZigTest);
    var config = try model_mod.parseConfig(t.io, a, dir);
    defer config.deinit(a);
    model_load.applyModelSettings(&config, .{});
    var buf: [512]u8 = undefined;
    // A RAM default the GPU limit left no room for, the one text that suggested a pack.
    const text = model_load.ssdBudgetBelowRefusal(&buf, t.io, a, &config, dir, .{ .bytes = 0, .origin = .ram_default, .setting_ignored = false }, false, 0);
    errdefer std.debug.print("{s}\n", .{text});
    try t.expect(std.mem.indexOf(u8, text, "is unset") != null);
    try t.expect(std.mem.indexOf(u8, text, "smaller pack") == null);
}

test "Flash-Next streamed: the ssd budget a fit refusal names passes the preflight on the readings that refused it (QWEN4_TEST_MODEL)" {
    const t = std.testing;
    const a = t.allocator;
    const dir = std.mem.span(std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest);
    var config = try model_mod.parseConfig(t.io, a, dir);
    defer config.deinit(a);
    model_load.applyModelSettings(&config, .{});
    const gib: u64 = 1 << 30;
    const asked: u64 = 30 * gib;
    const plan = try model_load.planExpertStreamingWith(t.io, a, &config, dir, asked, false, 0);
    var planned = config;
    model_load.applyStreamingPlan(&planned, plan, asked);
    const resident = plan.split.trunk +| plan.split.mtp +| plan.split.vision;
    // A 32 GB Mac at iogpu.wired_limit_mb 28000 (floor 26976 MB under the 1 GiB margin), its free RAM below the floor.
    for ([_]u64{ 6, 8 }) |free_gib| {
        const readings = bills.FitReadings{ .wired_limit = 28000 << 20, .wired_floor = 26976 << 20, .budget_working_set = 28000 << 20, .working_set = 28000 << 20, .free = free_gib * gib, .total_ram = 32 * gib, .unplanned = gib };
        _ = bills.expertCacheRefusal(&planned, resident, readings) orelse return error.TestExpectedRefusal;
        const plan_at = model_load.BudgetPlanner{ .io = t.io, .allocator = a, .config = &config, .model_dir = dir, .vision = false, .assistant = 0, .beside = 0, .mtp_on = false };
        const fits = bills.streamedBudgetThatFits(asked, 0, model_load.LoadPlanner{ .plan = plan_at, .readings = readings });
        try t.expect(fits > 0);
        const need = plan_at.needAt(fits << 30).?;
        const avail = readings.preflightAvailable(true).bytes.?;
        errdefer std.debug.print("free {d} GiB: the fit refusal names --ssd-budget-gb {d}, whose preflight needs {d} MB of {d} MB\n", .{ free_gib, fits, need >> 20, avail >> 20 });
        try t.expect(need <= avail);
    }
}

test "Flash-Next: the ssd budget a resident or streamed preflight refusal names passes the fit checks on the same readings (QWEN4_TEST_MODEL)" {
    const t = std.testing;
    const a = t.allocator;
    const dir = std.mem.span(std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest);
    var config = try model_mod.parseConfig(t.io, a, dir);
    defer config.deinit(a);
    model_load.applyModelSettings(&config, .{});
    try t.expect(config.supportsExpertStreaming());
    const gib: u64 = 1 << 30;
    const mib: u64 = 1 << 20;
    const saved_ctx = launch_state.server_config.max_context_size;
    defer launch_state.server_config.max_context_size = saved_ctx;
    const plan_at = model_load.BudgetPlanner{ .io = t.io, .allocator = a, .config = &config, .model_dir = dir, .vision = false, .assistant = 0, .beside = 0, .mtp_on = false };
    var named: u32 = 0;
    // A 32 GB Mac at macOS's default and at iogpu.wired_limit_mb 28000, the planned context automatic and the window's.
    for ([_]u32{ 0, 262_144 }) |ctx| for ([_]u64{ 0, 28000 }) |wired_mb| for ([_]u64{ 4, 8, 12, 16 }) |free_gib| {
        launch_state.server_config.max_context_size = @min(ctx, config.contextCap());
        const ws: u64 = if (wired_mb > 0) wired_mb * mib else 21 * gib;
        const readings = bills.FitReadings{ .wired_limit = wired_mb * mib, .wired_floor = bills.wiredLimitFloor(wired_mb * mib, 32 * gib, gib), .budget_working_set = ws, .working_set = ws, .free = free_gib * gib, .total_ram = 32 * gib, .unplanned = gib };
        const fits = bills.streamedBudgetThatFits(512 << 30, 0, model_load.LoadPlanner{ .plan = plan_at, .readings = readings });
        if (fits == 0) continue;
        named += 1;
        const at = plan_at.planAt(fits << 30).?;
        if (bills.expertCacheRefusal(&at.config, at.resident, readings)) |refused| {
            var buf: [2048]u8 = undefined;
            std.debug.print("--ctx-size {d}, wired {d} MB, free {d} GiB: the named --ssd-budget-gb {d} is refused by the fit check:\n{s}\n", .{ ctx, wired_mb, free_gib, fits, bills.loadMemoryRefusalText(&buf, refused.refusal) });
            return error.TestUnexpectedResult;
        }
        try t.expect(plan_at.needAt(fits << 30).? <= readings.preflightAvailable(true).bytes.?);
    };
    try t.expect(named > 0);
}

test "Flash-Next streamed: the 32 GB recipe's verdicts at --ssd-budget-gb 17 and 18 and the budget a refusal names hold (QWEN4_TEST_MODEL)" {
    const t = std.testing;
    const a = t.allocator;
    const dir = std.mem.span(std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest);
    var config = try model_mod.parseConfig(t.io, a, dir);
    defer config.deinit(a);
    model_load.applyModelSettings(&config, .{});
    const gib: u64 = 1 << 30;
    const mib: u64 = 1 << 20;
    const plan_at = model_load.BudgetPlanner{ .io = t.io, .allocator = a, .config = &config, .model_dir = dir, .vision = false, .assistant = 0, .beside = 0, .mtp_on = false };
    var got = bills.PinGot{};
    for ([_]u64{ 17, 18 }) |budget| got.add(plan_at.needAt(budget * gib).?);
    // A 32 GB M1 at iogpu.wired_limit_mb 28000 (floor 26976 MB under the 1 GiB margin): about 21 GB free, then free RAM
    // 200 MB over and under the cut's need.
    for ([_]u64{ 21 * gib, 8_818_350_426 + 200 * mib, 8_818_350_426 - 200 * mib }) |free| {
        const readings = bills.FitReadings{ .wired_limit = 28000 * mib, .wired_floor = 26976 * mib, .budget_working_set = 28000 * mib, .working_set = 28000 * mib, .free = free, .total_ram = 32 * gib, .unplanned = gib };
        const planner = model_load.LoadPlanner{ .plan = plan_at, .readings = readings };
        for ([_]u64{ 17, 18 }) |budget| got.add(planner.needAt(budget * gib).? == 0);
        got.add(bills.streamedBudgetThatFits(512 * gib, 0, planner));
    }
    try bills.pinEq("the 32 GB recipe", &.{ 8818350426, 8818350426, 1, 1, 26, 1, 1, 26, 0, 0, 4 }, got.slice());
}
