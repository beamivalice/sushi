//! DeepSeek-V4.1's bill at its release geometry: the request bill, the prefill ladder, the advertised context and the
//! decoder pools a prefill may lend. MLX-free, so the host tier runs it.
const std = @import("std");
const bills = @import("../bills.zig");
const dsv41_bill = @import("dsv41_bill.zig");
const dsv41_config = @import("dsv41_config.zig");
const model_load = @import("../model_load.zig");
const model_mod = @import("../model.zig");
const mtp_plan = @import("../mtp_plan.zig");
const prefill_plan = @import("../prefill_plan.zig");

test "DeepSeek: the streamed request bill brackets the prefill peaks measured on the M5 Max" {
    var config = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .stream_mtp_head = true };
    config.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    config.serve_defaults = dsv41_config.Config.serveDefaults(model_mod.ServeDefaults);
    config.max_position_embeddings = 1_048_576;
    config.expert_fill_peak_bytes = model_load.billedFillPeakBytes(&config, 9_911 << 20);
    // No decoder pool to lend: the bill is the working set's, as measured.
    config.expert_borrow_bytes = 0;
    try std.testing.expectEqual(prefill_plan.TAIL_MERGE_MAX - 1, dsv41_config.tail_merge_rows);
    // Live per-chunk `peak_bytes - active_bytes` (`SUSHI_PREFILL_CHUNK_PEAK`), CED prefill, `--ssd-budget-gb 94`, the STQ d4
    // pack, DSpark on: `chunk` is the configured width, `seq` where the chunk ended (a last chunk absorbs its tail).
    const measured = [_]struct { seq: u64, chunk: u64, peak_mb: u64 }{
        .{ .seq = 7168, .chunk = 1024, .peak_mb = 3_560 },
        .{ .seq = 8675, .chunk = 1024, .peak_mb = 4_049 },
        .{ .seq = 18432, .chunk = 512, .peak_mb = 2_635 },
        .{ .seq = 34625, .chunk = 512, .peak_mb = 2_997 },
        .{ .seq = 108032, .chunk = 512, .peak_mb = 2_600 },
        .{ .seq = 108675, .chunk = 512, .peak_mb = 2_740 },
        // Transient peaks of MLX's own bytes, state included, binary 26a28040, budget 108.
        .{ .seq = 2048, .chunk = 2048, .peak_mb = 4_907 },
        .{ .seq = 225_000, .chunk = 2048, .peak_mb = 5_124 },
        // The two-layer prefill pipeline holds 0.25 GB more of a 2048-row chunk's transient.
        .{ .seq = 2048, .chunk = 2048, .peak_mb = 4_907 + 250 },
        .{ .seq = 225_000, .chunk = 2048, .peak_mb = 5_124 + 250 },
    };
    for (measured) |m| {
        const bill = bills.prefillNeededAtChunk(&config, m.seq, 16, 16, m.chunk, .{});
        const peak = m.peak_mb * 1_000_000;
        // Under-billing is a Metal OOM: at least a tenth over the peak. The upper bar keeps the net small.
        try std.testing.expect(bill >= peak + peak / 10);
        try std.testing.expect(bill <= peak * 4);
    }
}

test "DeepSeek's live headroom under a raised wired limit is what the machine can still give, not the declared limit" {
    const t = std.testing;
    var config = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .stream_mtp_head = true };
    config.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    config.serve_defaults = dsv41_config.Config.serveDefaults(model_mod.ServeDefaults);
    config.max_position_embeddings = 1_048_576;
    config.cooler_for_test = true;
    config.expert_borrow_bytes = 0;
    const MiB: u64 = 1 << 20;
    const GB: u64 = 1_000_000_000;
    // r-65, the M5 at --ssd-budget-gb 108: iogpu.wired_limit_mb=125000 less the 1 GiB margin, every expert cached
    // (112.83 GB active), and the owner's apps holding the rest of the 128 GB, so 3 GB reclaimable, the live reserve of it kept.
    const declared: u64 = 123_976 * MiB;
    const active: u64 = 112_828_234_064;
    const footprint: u64 = active + 400 * MiB;
    const reclaimable: u64 = 3 * GB;
    const live = bills.liveCeilingFrom(declared, footprint, reclaimable);
    try t.expectEqual(footprint + reclaimable - bills.LIVE_RESERVE_BYTES, live);
    // The served warm's widest chunk is what that headroom holds: never the 8192 rows the declared limit admitted,
    // whose ~13 GB took the box down.
    const at_declared = bills.widestChunkAdmitted(&config, 16, 8192, bills.prefillHeadroomFrom(declared, active, 0));
    const at_live = bills.widestChunkAdmitted(&config, 16, 8192, bills.prefillHeadroomFrom(live, active, 0));
    try t.expectEqual(@as(u32, 8192), at_declared);
    try t.expect(at_live < 8192);
    try t.expect(at_live == 0 or bills.prefillChunkCost(&config, 16, at_live, 0) <= live - active);
    // With the memory free, the declared limit stands.
    try t.expectEqual(declared, bills.liveCeilingFrom(declared, footprint, 40 * GB));
}

test "issue 105: a raised --wired-margin-gib lowers the declared ceiling only; the live headroom keeps its own reserve, so the served warm still finds DeepSeek's 2048-row chunk" {
    const t = std.testing;
    var config = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .stream_mtp_head = true };
    config.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    config.serve_defaults = dsv41_config.Config.serveDefaults(model_mod.ServeDefaults);
    config.max_position_embeddings = 1_048_576;
    config.cooler_for_test = true;
    config.expert_borrow_bytes = 0;
    const MiB: u64 = 1 << 20;
    const GiB: u64 = 1 << 30;
    // The report: an M5 at iogpu.wired_limit_mb=124518 with --wired-margin-gib 16 for other apps' models, DeepSeek STQ
    // 1.75 at --ssd-budget-gb 100 loaded (~99.4 GB active), and ~10.4 GiB still reclaimable.
    const margin: u64 = 16 * GiB;
    const declared: u64 = bills.wiredLimitFloor(124_518 * MiB, 128 * GiB, margin);
    try t.expectEqual(@as(u64, 108_134 * MiB), declared);
    const active: u64 = 99_400 * MiB;
    const footprint: u64 = active + 400 * MiB;
    const reclaimable: u64 = 10_600 * MiB;
    const live = bills.liveCeilingFrom(declared, footprint, reclaimable);
    const width = bills.widestChunkAdmitted(&config, 16, 8192, bills.prefillHeadroomFrom(live, active, 0));
    try t.expect(width >= 2048);
}

test "DeepSeek prices its chunk per request: 2048 while the bill fits, then 1024, then 512, never wider" {
    const t = std.testing;
    var config = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .stream_mtp_head = true };
    config.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    config.serve_defaults = dsv41_config.Config.serveDefaults(model_mod.ServeDefaults);
    config.max_position_embeddings = 1_048_576;
    config.num_attention_heads = 64;
    config.expert_fill_peak_bytes = model_load.billedFillPeakBytes(&config, 9_911 << 20);
    // No decoder pool to lend: the bill is the working set's, as measured.
    config.expert_borrow_bytes = 0;
    const MB: u64 = 1 << 20;
    // What `--ssd-budget-gb 96` on the 128 GB M5 Max leaves to serve, as its pin measured it.
    const at96: u64 = 110_100 * MB - 97_080 * MB;
    try t.expectEqual(@as(u32, 2048), bills.chooseRequestPrefillChunk(&config, 8192, 2048, 16, at96, 512, 0, .{}));
    // Its attention scores only the picked keys, so a long prompt keeps the width while the bill fits.
    try t.expectEqual(@as(u32, 2048), bills.chooseRequestPrefillChunk(&config, 400_000, 2048, 16, at96, 512, 0, .{}));
    const long: u64 = 524_288;
    const at1024 = bills.prefillNeededAtChunk(&config, long, 2048, 16, 1024, .{});
    const only_narrow = bills.prefillNeededAtChunk(&config, long, 2048, 16, 512, .{});
    try t.expect(bills.prefillNeededAtChunk(&config, long, 2048, 16, 2048, .{}) > at1024 and at1024 > only_narrow);
    try t.expectEqual(@as(u32, 1024), bills.chooseRequestPrefillChunk(&config, long, 2048, 16, at1024, 512, 0, .{}));
    try t.expectEqual(@as(u32, 512), bills.chooseRequestPrefillChunk(&config, long, 2048, 16, only_narrow, 512, 0, .{}));
    try t.expectEqual(@as(u32, 512), bills.chooseRequestPrefillChunk(&config, 8192, 2048, 16, at96, 512, 512, .{}));
    var buf: [256]u8 = undefined;
    try t.expectEqualStrings("Prefill chunk: per request, up to 2048 at a short prompt (the widest rung each request's bill admits); load-time fallback 512\n", bills.prefillChunkLoadLine(&buf, &config, 512, 8192).?);
    // The ladder never narrows the advertised context: it is priced at the 512 floor.
    try t.expectEqual(@as(u32, 1_000_000), bills.advertisedContextAt(&config, 16, 110_100 * MB, 97_080 * MB, 512));
}

test "a DeepSeek load's served warm forwards at the widest width a served request takes, 2048, not the global 8192" {
    const t = std.testing;
    var config = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .stream_mtp_head = true };
    config.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    config.serve_defaults = dsv41_config.Config.serveDefaults(model_mod.ServeDefaults);
    config.max_position_embeddings = 1_048_576;
    config.num_attention_heads = 64;
    try t.expectEqual(@as(u32, 2048), bills.widestServedRung(&config));
    const generic = prefill_plan.requestPrefillChunk(config.prefillScoreHeadDim(), config.num_attention_heads, 1, config.has_sliding_window, config.isMoe(), config.longCtxGated(), 0);
    try t.expect(generic > bills.widestServedRung(&config));
    // An operator's --prefill-chunk is what served requests take, so the warm follows it.
    const sr = @import("../settings_resolve.zig");
    const saved_width = sr.prefill_chunk.value;
    defer sr.prefill_chunk.value = saved_width;
    const saved_source = sr.prefill_chunk.source;
    defer sr.prefill_chunk.source = saved_source;
    sr.prefill_chunk.value = 4096;
    sr.prefill_chunk.source = .flag;
    try t.expectEqual(@as(u32, 4096), bills.widestServedRung(&config));
}

test "DeepSeek advertises the longest context its own request bill admits" {
    const t = std.testing;
    var config = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .stream_mtp_head = true };
    config.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    config.serve_defaults = dsv41_config.Config.serveDefaults(model_mod.ServeDefaults);
    config.max_position_embeddings = 1_048_576;
    config.num_attention_heads = 64;
    config.expert_fill_peak_bytes = model_load.billedFillPeakBytes(&config, 9_911 << 20);
    // No decoder pool to lend: the bill is the working set's, as measured.
    config.expert_borrow_bytes = 0;
    const GB: u64 = 1_000_000_000;
    // The request bill grows with the prompt (the state and the rows' growth), so a serving budget of exactly one
    // prompt's bill advertises that prompt and not one token less.
    for ([_]u64{ 8192, 65_536, 131_072, 262_144, 524_288, 1_048_576 }) |seq| {
        for ([_]u64{ 512, 1024 }) |chunk| {
            const bill = bills.prefillNeededAtChunk(&config, seq, 0, 16, chunk, .{});
            try t.expect(bills.memoryContextAtChunk(&config, 16, 100 * GB + bill, 100 * GB, chunk) >= seq);
            try t.expect(bills.memoryContextAtChunk(&config, 16, 100 * GB + bill - 1, 100 * GB, chunk) < seq);
        }
    }
    // At 96 the whole 1M context fits the fallback width's bill, with the memory the margined default needs to spare.
    const MB: u64 = 1 << 20;
    try t.expect(bills.memoryContextAtChunk(&config, 16, 110_100 * MB, 97_080 * MB, 512) >= bills.memoryContextForMargined(1_000_000));
}

test "a streamed DeepSeek bills its workspace at load and no fill peak at load or per request" {
    var config = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .stream_mtp_head = true };
    config.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    config.serve_defaults = dsv41_config.Config.serveDefaults(model_mod.ServeDefaults);
    config.max_position_embeddings = 1_048_576;
    config.expert_cache_bytes = 93_993 << 20;
    config.expert_bounce_bytes = 512 << 20;
    config.expert_workspace_bytes = 3_525 << 20;
    config.expert_fill_peak_bytes = model_load.billedFillPeakBytes(&config, 9_911 << 20);
    const mib: u64 = 1 << 20;
    // No decoder pool to lend: the bill is the working set's, as measured.
    config.expert_borrow_bytes = 0;
    // A 100-token request pays its own transient, not the cache layer and a second workspace.
    try std.testing.expect(bills.prefillNeededAtChunk(&config, 100, 512, 16, 512, .{}) < 2048 * mib);
    const resident: u64 = 8_295 * mib;
    const held = resident + config.expert_cache_bytes + config.expert_bounce_bytes + config.expert_workspace_bytes + config.expertPoolBytes();
    const serving = bills.expertStreamingServingBytes(&config);
    try std.testing.expect(bills.expertStreamingConfigFits(&config, held + serving, 0, resident));
    try std.testing.expect(!bills.expertStreamingConfigFits(&config, held + serving - 1, 0, resident));
}

test "DeepSeek's live headroom and its load fit each count the decode pool the ledger parks once" {
    const t = std.testing;
    var config = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .stream_mtp_head = true };
    config.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    config.serve_defaults = dsv41_config.Config.serveDefaults(model_mod.ServeDefaults);
    config.max_position_embeddings = 1_048_576;
    // The parked pool is in MLX's cache, not its active bytes, so `ceiling - active` does not see it.
    try t.expectEqual(@as(u64, 10 << 30) - config.expertPoolBytes(), dsv41_bill.dsv41HeadroomFrom(10 << 30, 0, 0, &config));
    try t.expectEqual(@as(u64, 0), dsv41_bill.dsv41HeadroomFrom(1 << 20, 0, 0, &config));
    // The load fit holds the pool in its staging term, so the serving term it adds is the bill without it.
    config.expert_borrow_bytes = 60_000_000_000;
    const bill = bills.prefillNeededAtChunk(&config, bills.expertStreamingPlannedSeq(&config), 2048, bills.defaultKvBits(&config), 512, .{});
    try t.expectEqual(bill - config.expertPoolBytes(), bills.expertStreamingServingBytes(&config));
}

test "DeepSeek's live headroom is the ceiling less the active bytes and the engine's free buffers, each once, and zero when they pass it" {
    const t = std.testing;
    var config = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .stream_mtp_head = true };
    config.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    config.serve_defaults = dsv41_config.Config.serveDefaults(model_mod.ServeDefaults);
    const gb: u64 = 1 << 30;
    const without = dsv41_bill.dsv41HeadroomFrom(100 * gb, 60 * gb, 0, &config);
    try t.expectEqual(40 * gb - config.expertPoolBytes(), without);
    try t.expectEqual(without - 3 * gb, dsv41_bill.dsv41HeadroomFrom(100 * gb, 63 * gb, 0, &config));
    try t.expectEqual(without - 3 * gb, dsv41_bill.dsv41HeadroomFrom(100 * gb, 60 * gb, 3 * gb, &config));
    try t.expectEqual(@as(u64, 0), dsv41_bill.dsv41HeadroomFrom(100 * gb, 110 * gb, 0, &config));
    try t.expectEqual(@as(u64, 0), dsv41_bill.dsv41HeadroomFrom(100 * gb, 60 * gb, 50 * gb, &config));
}

test "on a Cooler load DeepSeek's live headroom counts the parked pool once: it is the engine's free buffers" {
    const t = std.testing;
    var config = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .stream_mtp_head = true };
    config.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    config.serve_defaults = dsv41_config.Config.serveDefaults(model_mod.ServeDefaults);
    @import("../cooler_side.zig").forTest(&config, true);
    defer @import("../cooler_side.zig").forTest(&config, false);
    const gb: u64 = 1 << 30;
    const pool = config.expertPoolBytes();
    try t.expect(pool > 0);
    // The pool parked, held as the engine's free buffers: taken off once.
    try t.expectEqual(40 * gb - pool, dsv41_bill.dsv41HeadroomFrom(100 * gb, 60 * gb, pool, &config));
    // Not grown to its cap yet: the cap is still reserved.
    try t.expectEqual(40 * gb - pool, dsv41_bill.dsv41HeadroomFrom(100 * gb, 60 * gb, 0, &config));
    // Free buffers past the cap come off too.
    try t.expectEqual(40 * gb - pool - 2 * gb, dsv41_bill.dsv41HeadroomFrom(100 * gb, 60 * gb, pool + 2 * gb, &config));
}

test "DeepSeek's request bill takes its prefill working set out of the decoder pools a prefill may release" {
    const t = std.testing;
    var config = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .stream_mtp_head = true };
    config.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    config.serve_defaults = dsv41_config.Config.serveDefaults(model_mod.ServeDefaults);
    config.max_position_embeddings = 1_048_576;
    const c = config.dsv41.?;
    const seq: u64 = 524_288;
    const state = ((seq + 2048) * c.stateBytesPerToken() + c.stateFixedBytes(true) + bills.checkpointReserve(&config, seq, 2048)) * 5 / 4;
    // A load on MLX: its chunk bills MLX's per-row rate.
    const work = c.chunkCapOn(false, 2048, seq) + config.expertPoolBytes();
    const decode = c.forwardCap(dsv41_config.verify_rows, seq);
    // No pool to lend: the whole working set is billed, as before.
    try t.expectEqual(state + work, bills.prefillNeededAtChunk(&config, seq, 2048, 16, 2048, .{}));
    // Pools that cover part of it: admission pays only the rest.
    config.expert_borrow_bytes = work / 2;
    try t.expectEqual(state + work - work / 2, bills.prefillNeededAtChunk(&config, seq, 2048, 16, 2048, .{}));
    // Pools that cover all of it: the state, a decode forward and the decode pool it leaves parked, at any chunk.
    config.expert_borrow_bytes = 60_000_000_000;
    for ([_]u64{ 512, 1024, 2048 }) |chunk| {
        const ck = ((seq + 2048) * c.stateBytesPerToken() + c.stateFixedBytes(true) + bills.checkpointReserve(&config, seq, chunk)) * 5 / 4;
        try t.expectEqual(ck + decode + config.expertPoolBytes(), bills.prefillNeededAtChunk(&config, seq, 2048, 16, chunk, .{}));
    }
    // The widest rung is then always affordable, and a chunk boundary never steps down for the working set.
    const MB: u64 = 1 << 20;
    try t.expectEqual(@as(u32, 2048), bills.chooseRequestPrefillChunk(&config, seq, 2048, 16, state + decode + config.expertPoolBytes(), 512, 0, .{}));
    var st: mtp_plan.AdaptiveWidthState = .{};
    try t.expectEqual(@as(u32, 2048), bills.adaptivePrefillWidth(&config, 16, seq, 64 * MB, 2048, 2048, &st));
    config.expert_borrow_bytes = 0;
    var st0: mtp_plan.AdaptiveWidthState = .{};
    try t.expect(bills.adaptivePrefillWidth(&config, 16, seq, 64 * MB, 2048, 2048, &st0) < 2048);
}

test "the default request bill at 1M tokens is the borrowed bill plus the decode pool" {
    const t = std.testing;
    var config = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .stream_mtp_head = true };
    config.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    config.serve_defaults = dsv41_config.Config.serveDefaults(model_mod.ServeDefaults);
    config.max_position_embeddings = 1_048_576;
    config.num_attention_heads = 64;
    const saved = .{ bills.launch_state.prefix_cache_ram_enabled, bills.launch_state.prefix_cache_disk_bytes };
    defer {
        bills.launch_state.prefix_cache_ram_enabled = saved[0];
        bills.launch_state.prefix_cache_disk_bytes = saved[1];
    }
    bills.launch_state.prefix_cache_ram_enabled = false;
    bills.launch_state.prefix_cache_disk_bytes = 20 << 30;
    const bill = bills.defaultRequestBill(&config);
    // dsv41-memory: 3.4 GB at 1M with the prefill borrowed (state, writer in flight, rings, x5/4, the decode cap), plus the
    // 1 GiB decode pool the ledger parks beside it.
    try t.expectApproxEqAbs(@as(f64, 4_400_000_000), @as(f64, @floatFromInt(bill)), 300_000_000);
    // The bill leaves the config's own streaming fields as they were.
    try t.expect(config.expert_streaming);
    try t.expectEqual(@as(u64, 0), config.expert_borrow_bytes);
}

test "with its decoder pools lendable, DeepSeek at --ssd-budget-gb 117 with DSpark on wired 124000 advertises its whole context" {
    const t = std.testing;
    var config = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .stream_mtp_head = true };
    config.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    config.serve_defaults = dsv41_config.Config.serveDefaults(model_mod.ServeDefaults);
    config.max_position_embeddings = 1_048_576;
    config.num_attention_heads = 64;
    const saved = .{ bills.launch_state.prefix_cache_ram_enabled, bills.launch_state.prefix_cache_disk_bytes };
    defer {
        bills.launch_state.prefix_cache_ram_enabled = saved[0];
        bills.launch_state.prefix_cache_disk_bytes = saved[1];
    }
    // The served default: no RAM tier, the SSD tier on, so restore points go to SSD as a prefill takes them.
    bills.launch_state.prefix_cache_ram_enabled = false;
    bills.launch_state.prefix_cache_disk_bytes = 20 << 30;
    const MB: u64 = 1 << 20;
    // The budget-100 boot on wired 124000 read a 122,170 MiB ceiling and 100,328 MiB active; 17 GiB more at 117.
    const ceiling: u64 = 122_170 * MB;
    // The budget now holds the decode pool, parked in MLX's cache: the same budget leaves active that much lower.
    const active: u64 = (100_328 + 17 * 1024) * MB - (1 << 30);
    // Its expert cache, 13.57 GB under the budget (94% of the experts), split evenly: the decoder half is lendable.
    config.expert_borrow_bytes = (117 * (1 << 30) - 13_570_000_000) / 2;
    try t.expectEqual(@as(u32, 1_000_000), bills.advertisedContextAt(&config, 16, ceiling, active, 2048));
    // Billing the working set to the request, with nothing to lend, advertises well under the whole context.
    config.expert_borrow_bytes = 0;
    try t.expectEqual(@as(u32, 581_632), bills.advertisedContextAt(&config, 16, ceiling, active, 512));
}

test "a default context never exceeds the checkpoint's window, and --ctx-size or model-settings still win" {
    const t = std.testing;
    const original = bills.launch_state.server_config.max_context_size;
    defer bills.launch_state.server_config.max_context_size = original;
    var config = model_mod.ModelConfig{ .model_type = "deepseek_v41", .max_position_embeddings = 262_144, .serve_defaults = dsv41_config.Config.serveDefaults(model_mod.ServeDefaults) };
    try t.expectEqual(@as(u32, 262_144), bills.autoContextFrom(1_048_576, config.contextCap(), config.defaultContext()));

    config.max_position_embeddings = 1_048_576;
    config.pinned_context = 1_000_000;
    bills.launch_state.server_config.max_context_size = 0;
    try t.expectEqual(@as(u32, 1_000_000), bills.getEffectiveContextLength(&config));
    config.ctx_override = 300_000;
    try t.expectEqual(@as(u32, 300_000), bills.getEffectiveContextLength(&config));
    bills.launch_state.server_config.max_context_size = 250_000;
    try t.expectEqual(@as(u32, 250_000), bills.getEffectiveContextLength(&config));
}

test "DeepSeek's unlent bill is what a Cooler request's forwards take from the pool themselves, whatever the pools may lend" {
    const t = std.testing;
    var config = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .stream_mtp_head = true };
    config.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    config.serve_defaults = dsv41_config.Config.serveDefaults(model_mod.ServeDefaults);
    config.max_position_embeddings = 1_048_576;
    config.cooler_for_test = true;
    const c = &config.dsv41.?;
    // The served warm's widest request on the M5 at --ssd-budget-gb 95 (r-62): 8,193 tokens at 8192-row chunks.
    const seq: u64 = 8193;
    const chunk: u64 = 8192;
    const unlent = bills.prefillNeededAtChunk(&config, seq, 96, 16, chunk, .{});
    // Decoder pools that cover the chunk's cap: admission bills the state and a decode forward, since a forward the
    // headroom cannot hold releases pools first (`dsv41_borrow.beforeForward`)...
    config.expert_borrow_bytes = 60_000_000_000;
    const lent = bills.prefillNeededAtChunk(&config, seq, 96, 16, chunk, .{});
    try t.expectEqual(unlent - (c.chunkCap(chunk, seq) - c.forwardCap(dsv41_config.verify_rows, seq)), lent);
    // ...but a forward the headroom holds releases none and takes its whole cap from the pool: what it asks is measured
    // against the bill with nothing lent, the same with or without pools to lend.
    try t.expectEqual(unlent, bills.prefillNeededAtChunkUnlent(&config, seq, 96, 16, chunk, .{}));
    config.expert_borrow_bytes = 0;
    try t.expectEqual(unlent, bills.prefillNeededAtChunkUnlent(&config, seq, 96, 16, chunk, .{}));
}

test "DeepSeek's request bill on a Cooler load is its exact state, one forward's cap and the margin: MLX's parked pool, the MLX net and MLX's row rate are not billed" {
    const t = std.testing;
    var config = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .stream_mtp_head = true };
    config.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    config.serve_defaults = dsv41_config.Config.serveDefaults(model_mod.ServeDefaults);
    config.max_position_embeddings = 1_048_576;
    const c = &config.dsv41.?;
    for ([_]u64{ 9_000, 225_000, 1_000_000 }) |seq| {
        // What the request stores: its state rows and fixed state, and the checkpoints its prefill keeps.
        const stored = (seq + 2048) * c.stateBytesPerToken() + c.stateFixedBytes(config.dsparkOn()) + dsv41_bill.checkpointReserve(&config, seq, 2048);
        config.cooler_for_test = false;
        const off = bills.prefillNeededAtChunk(&config, seq, 2048, 16, 2048, .{});
        config.cooler_for_test = true;
        const net = stored * 5 / 4 - stored;
        const engine_chunk = c.chunkCap(2048, seq);
        try t.expectEqual(off - dsv41_config.dsv41_pool_bytes - net - c.chunkCapOn(false, 2048, seq) + engine_chunk + bills.BILL_MARGIN_BYTES, bills.prefillNeededAtChunk(&config, seq, 2048, 16, 2048, .{}));
    }
}


test "DeepSeek: a request's decode state is billed as growth after admission only on a fully resident load, which joins rows" {
    var config = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .stream_mtp_head = true };
    config.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    // A streamed load serves one row: the model's parked state is resident before the request is admitted.
    try std.testing.expectEqual(@as(u64, 0), bills.admissionCommitBytes(&config, 4096, 512, 16, false));
    // A fully resident load joins rows: the request makes its own state at its first forward, all of it after admission.
    config.expert_resident = true;
    // Rows are Sushi Cooler's: a resident load on MLX keeps the one-row decode and its parked state.
    try std.testing.expectEqual(@as(u64, 0), bills.admissionCommitBytes(&config, 4096, 512, 16, false));
    config.cooler_for_test = true;
    const c = &config.dsv41.?;
    const state = (4096 + 512) * c.stateBytesPerToken() + c.stateFixedBytes(config.dsparkOn());
    try std.testing.expectEqual(state, bills.admissionCommitBytes(&config, 4096, 512, 16, false));
}

/// What the served warm reads as its headroom right after a streamed load took what it holds out of `avail` free bytes:
/// the live headroom, whose footprint is the load's bytes, the decode pool among them parked as the engine's.
fn warmHeadroomAfterLoad(config: *const model_mod.ModelConfig, ceiling: u64, avail: u64, weights: u64) u64 {
    const held = bills.expertStreamingGateBytes(weights, config.expert_cache_bytes, model_load.loadStagingBytes(config, config.expert_fill_peak_bytes, config.expert_workspace_bytes), config.expert_bounce_bytes);
    const pool = config.expertPoolBytes();
    return bills.liveHeadroomFrom(config, bills.liveCeilingFrom(ceiling, held, avail -| held), held - pool, 0, pool);
}

/// The 32 GB ledger's planner at a smaller budget: the GiBs it gives up come out of the cache, which keeps whole slots of
/// about 0.385 GB, and a cache under two slots is refused.
const SlotPlanner = struct {
    config: *const model_mod.ModelConfig,
    weights: u64,
    pub fn needAt(p: SlotPlanner, budget: u64) ?u64 {
        const slot: u64 = 385_000_000;
        const given_up = p.config.expert_ssd_budget_bytes -| budget;
        if (p.config.expert_cache_bytes < given_up +| 2 * slot) return null;
        var planned = p.config.*;
        planned.expert_cache_bytes = (p.config.expert_cache_bytes - given_up) / slot * slot;
        return bills.loadRequirementBytes(&planned, p.weights, true, 0);
    }
};

test "DeepSeek: a streamed load the preflight and the fit admit on a 32 GB Mac leaves its served warm a prefill chunk, and a refused budget names one that fits" {
    const t = std.testing;
    var config = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .stream_mtp_head = true };
    config.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    config.serve_defaults = dsv41_config.Config.serveDefaults(model_mod.ServeDefaults);
    config.max_position_embeddings = 1_048_576;
    config.cooler_for_test = true;
    // dsv41-memory, the 32 GB Mac at --ssd-budget-gb 18: trunk, MTP and vision 9.55 GB, workspace 3.70, bounce 0.54,
    // cache 4.23 GB; decoder pools about 2.1 GB, which the warm cannot borrow.
    const MiB: u64 = 1 << 20;
    const weights: u64 = 9_550_000_000;
    config.expert_cache_bytes = 4_230_000_000;
    config.expert_workspace_bytes = 3_700_000_000;
    config.expert_bounce_bytes = 540_000_000;
    config.expert_borrow_bytes = 2_100_000_000;
    config.expert_ssd_budget_bytes = 18 << 30;
    const bits = bills.defaultKvBits(&config);
    const top = bills.widestServedRung(&config);

    // Free memory: every reading the preflight admits leaves the warm a chunk.
    const ceiling = bills.wiredLimitFloor(28_000 * MiB, 32 << 30, 1 << 30);
    const need = bills.loadRequirementBytes(&config, weights, true, 0);
    var avail = need -| (2 << 30);
    while (avail < need + (2 << 30)) : (avail += 50 * MiB) {
        if (bills.memInsufficientForLoad(weights, avail, need)) {
            // The refusal's budget is one the ledger's planner fits in what is free.
            const fits = bills.streamedBudgetThatFits(config.expert_ssd_budget_bytes, avail, SlotPlanner{ .config = &config, .weights = weights });
            if (fits > 0) try t.expect(SlotPlanner.needAt(.{ .config = &config, .weights = weights }, fits << 30).? <= avail);
            continue;
        }
        try t.expect(bills.warmChunkAdmitted(&config, bits, top, warmHeadroomAfterLoad(&config, ceiling, avail, weights)) > 0);
    }

    // The declared ceiling: every one the load's fit admits leaves the warm a chunk with memory to spare.
    var c: u64 = 18_000 * MiB;
    while (c < 24_000 * MiB) : (c += 50 * MiB) {
        if (!bills.expertStreamingConfigFits(&config, c, 0, weights)) continue;
        try t.expect(bills.warmChunkAdmitted(&config, bits, top, warmHeadroomAfterLoad(&config, c, 1 << 40, weights)) > 0);
    }

    // The RAM default's budget is bounded by a request at the default context beside it, which leaves the warm a chunk.
    try t.expect(bills.defaultRequestBill(&config) >= bills.warmFloorChunkBytes(&config));
}

/// A DeepSeek prompt of `seq` tokens on a machine with `avail` bytes free beside the load, as serve runs it: admission
/// picks and bills its width, the per-chunk hook narrows the adaptive start, and before each forward the borrow covers
/// its need from the headroom and the decoder pools. On a Cooler load the engine keeps `free_pct`% of the last forward's
/// cap as free buffers at a chunk boundary (its free cap), trimmed to the next forward's cap as it begins, and starts
/// with the decode pool parked as free buffers. The forward the borrow cannot cover, if one is, and by how much.
fn borrowRefusal(config: *const model_mod.ModelConfig, seq: u64, avail: u64, explicit: u32, max_tokens: u32, free_pct: u64) ?struct { rows: u64, end: u64, short: u64 } {
    const c = &config.dsv41.?;
    const cooler = @import("../cooler_side.zig").onEngine(config);
    const bits = bills.defaultKvBits(config);
    const pool = config.expertPoolBytes();
    var free: u64 = if (cooler) pool else 0;
    const available = bills.admissionAvailableFor(config, avail, 0, free);
    const billed = bills.chooseRequestPrefillChunk(config, seq, max_tokens, bits, available, 0, explicit, .{});
    if (bills.prefillNeededAtChunk(config, seq, max_tokens, bits, billed, .{}) > available) return null;
    const adaptive = explicit == 0;
    const cap: u32 = @min(prefill_plan.effectivePrefillChunk(config.prefillScoreHeadDim(), config.num_attention_heads, seq, config.has_sliding_window, config.isMoe(), config.longCtxGated(), 0), config.prefillStartWidth());
    var cur: u32 = @intCast(prefill_plan.firstChunkWidth(config, billed, cap, adaptive));
    var st: mtp_plan.AdaptiveWidthState = .{ .width_min = @intCast(billed), .width_max = @intCast(billed) };
    const loop_end = seq - 1;
    var pos: u64 = 0;
    while (pos < loop_end) {
        const grown = c.stateFixedBytes(config.dsparkOn()) + pos * c.stateBytesPerToken();
        if (adaptive) {
            const next = bills.adaptivePrefillWidth(config, bits, pos, bills.liveHeadroomFrom(config, avail, grown, 0, free), cur, cap, &st);
            if (next != 0 and next < cur) cur = next;
        }
        var end = @min(pos + cur, loop_end);
        if (end < loop_end and loop_end - end < prefill_plan.tailMergeMaxFor(cur, adaptive)) end = loop_end;
        const need = c.forwardCapOn(cooler, end - pos, end);
        const kept = @min(free, need);
        const cover = dsv41_bill.borrowHeadroomFrom(avail, grown, kept, config) +| config.expert_borrow_bytes;
        if (cover < need) return .{ .rows = end - pos, .end = end, .short = need - cover };
        if (cooler) free = @max(pool, need * free_pct / 100);
        pos = end;
    }
    return null;
}

test "DeepSeek: an admitted prompt is never refused by a forward's borrow, at the default width or --prefill-chunk 512, whatever the decoder pools hold and the engine keeps free" {
    const MiB: u64 = 1 << 20;
    for ([_]bool{ true, false }) |cooler| {
        var config = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .stream_mtp_head = true };
        config.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
        config.serve_defaults = dsv41_config.Config.serveDefaults(model_mod.ServeDefaults);
        config.max_position_embeddings = 1_048_576;
        config.pinned_context = 1_000_000;
        config.cooler_for_test = cooler;
        // dsv41-memory, the 32 GB Mac at --ssd-budget-gb 18 holds about 2.1 GB of decoder pools; smaller budgets less.
        for ([_]u64{ 300_000_000, 1_000_000_000, 2_100_000_000 }) |borrowable| {
            config.expert_borrow_bytes = borrowable;
            for ([_]u64{ 0, 50, 100 }) |free_pct| {
                if (!cooler and free_pct > 0) continue;
                for ([_]u32{ 16, 8192 }) |max_tokens| {
                    for ([_]u64{ 4096, 4097, 4300, 4600, 5100, 6300, 8400, 20_000 }) |seq| {
                        var avail: u64 = 256 * MiB;
                        while (avail <= 8192 * MiB) : (avail += 16 * MiB) {
                            for ([_]u32{ 0, 512 }) |explicit| {
                                const r = borrowRefusal(&config, seq, avail, explicit, max_tokens, free_pct) orelse continue;
                                std.debug.print("{s}, {d} MB of decoder pools, the engine keeping {d}% free: a {d}-token prompt (max_tokens {d}) admitted with {d} MB free at --prefill-chunk {d} has its {d}-row forward ending at {d} refused {d} MB short\n", .{
                                    if (cooler) "Cooler" else "MLX", borrowable / 1_000_000, free_pct, seq, max_tokens, avail / MiB, explicit, r.rows, r.end, r.short / MiB,
                                });
                                return error.TestUnexpectedResult;
                            }
                        }
                    }
                }
            }
        }
    }
}
