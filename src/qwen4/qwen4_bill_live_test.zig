//! Flash-Next bills against live memory: the request chunk chooser, SSD-first and the reservation clamp.
const bills = @import("../bills.zig");
const fixtures = @import("../bills_fixtures_test.zig");
const launch_state = @import("../server_state.zig");
const model_mod = @import("../model.zig");
const prefix_cache_mod = @import("../prefix_cache.zig");
const qwen4_qsa_select = @import("qwen4_qsa_select.zig");
const settings_resolve = @import("../settings_resolve.zig");
const slot_admission = @import("../slot_admission.zig");
const std = @import("std");
const trunk = @import("../trunk.zig");
const request_policy = @import("../request_policy.zig");
const clampMaxTokens = request_policy.clampMaxTokens;
const omittedMaxTokensDefaultWith = request_policy.omittedMaxTokensDefaultWith;
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
    // An omitted `max_tokens` is maxInt(u32)/4; the memory guard used to run before
    // `clampMaxTokens`, so the reservation billed ~26 TB of KV and refused every long prompt.
    const KVCache = trunk.KVCache;
    const ctx: u64 = 262_144;
    const seq: u64 = 50_000;
    const sentinel: u64 = omittedMaxTokensDefaultWith(@intCast(ctx), 0, 0);
    try t.expectEqual(@as(u64, std.math.maxInt(u32) / 4), sentinel);

    // (1) The cache clamps: the reservation cannot exceed the context plus a chunk.
    const reserved = KVCache.reservedTokens(seq, sentinel, 4096, ctx);
    // Bounded twice: by the generation headroom first, by the context second.
    try t.expectEqual(seq + KVCache.RESERVE_GEN_HEADROOM + 4096, reserved);
    try t.expect(reserved < ctx);
    // (2) The four surfaces clamp first and hand the guard the clamped value.
    const clamped: u64 = clampMaxTokens(@intCast(sentinel), @intCast(seq), @intCast(ctx));
    try t.expectEqual(ctx - seq, clamped);
    try t.expectEqual(reserved, KVCache.reservedTokens(seq, clamped, 4096, ctx));

    try t.expectEqual(seq + 2048 + 4096, KVCache.reservedTokens(seq, 2048, 4096, ctx));
    // A caller that names no context is not context-clamped, but the headroom bound still applies.
    try t.expectEqual(seq + KVCache.RESERVE_GEN_HEADROOM + 4096, KVCache.reservedTokens(seq, sentinel, 4096, 0));
    try t.expectEqual(ctx + 4096, KVCache.reservedTokens(ctx, sentinel, 4096, ctx));

    const cfg = qwen4ExpOomConfig();
    const terms = prefillRequestTerms(&cfg, seq, clamped, 8, 4096, .{});
    try t.expect(terms.reserved_kv_bytes < 4 * 1024 * 1024 * 1024);
}
