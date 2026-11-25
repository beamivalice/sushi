//! DeepSeek-V4.1's bill at its release geometry: the request bill, the prefill ladder, the advertised context and the
//! decoder pools a prefill may lend. MLX-free, so the host tier runs it.
const std = @import("std");
const bills = @import("../bills.zig");
const dsv41_bill = @import("dsv41_bill.zig");
const dsv41_borrow = @import("dsv41_borrow.zig");
const dsv41_wide_bytes = @import("dsv41_wide_bytes.zig");
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
    // Each ran at `--prefill-chunk`, whose loop absorbs a tail under 512 rows.
    const sr = @import("../settings_resolve.zig");
    const saved_chunk = sr.prefill_chunk;
    defer sr.prefill_chunk = saved_chunk;
    for (measured) |m| {
        sr.prefill_chunk = .{ .value = m.chunk, .source = .flag };
        const bill = bills.prefillNeededAtChunk(&config, m.seq, 16, 16, m.chunk, .{});
        const peak = m.peak_mb * 1_000_000;
        // Under-billing is a Metal OOM: at least a tenth over the peak. The upper bar keeps the net small.
        try std.testing.expect(bill >= peak + peak / 10);
        try std.testing.expect(bill <= peak * 4);
    }
}

test "issue 105: a raised --wired-margin-gib lowers the declared ceiling only, so the served warm still finds DeepSeek's 2048-row chunk" {
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
    // 1.75 at --ssd-budget-gb 100 loaded (~99.4 GB active).
    const margin: u64 = 16 * GiB;
    const declared: u64 = bills.wiredLimitFloor(124_518 * MiB, 128 * GiB, margin).?;
    try t.expectEqual(@as(u64, 108_134 * MiB), declared);
    const active: u64 = 99_400 * MiB;
    const width = bills.widestChunkAdmitted(&config, 16, 8192, bills.prefillHeadroomFrom(declared, active, 0));
    try t.expect(width >= 2048);
}

test "DeepSeek prices its chunk per request: 8192 while the bill fits, then each narrower rung, never wider" {
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
    // There, a short prompt's 8192 rows do not fit beside the load and 4096 do.
    try t.expectEqual(@as(u32, 4096), bills.chooseRequestPrefillChunk(&config, 8192, 2048, 16, at96, 512, 0, .{}));
    // Its attention scores only the picked keys, so a long prompt keeps each rung while that rung's bill fits.
    const long: u64 = 524_288;
    var wider: u64 = std.math.maxInt(u64);
    for (bills.PREFILL_CHUNK_LADDER) |rung| {
        const need = bills.prefillNeededAtChunk(&config, long, 2048, 16, rung, .{});
        try t.expect(need < wider);
        wider = need;
        try t.expectEqual(rung, bills.chooseRequestPrefillChunk(&config, long, 2048, 16, need, 512, 0, .{}));
    }
    try t.expectEqual(@as(u32, 512), bills.chooseRequestPrefillChunk(&config, 8192, 2048, 16, at96, 512, 512, .{}));
    var buf: [256]u8 = undefined;
    try t.expectEqualStrings("Prefill chunk: per request, up to 8192 at a short prompt (the widest rung each request's bill admits); load-time fallback 512\n", bills.prefillChunkLoadLine(&buf, &config, 512, 8192).?);
    // The ladder never narrows the advertised context: it is priced at the 512 floor.
    try t.expectEqual(@as(u32, 1_000_000), bills.advertisedContextAt(&config, 16, 110_100 * MB, 97_080 * MB, 512));
}

test "a DeepSeek load's served warm forwards at the widest width a served request takes, its start width" {
    const t = std.testing;
    var config = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .stream_mtp_head = true };
    config.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    config.serve_defaults = dsv41_config.Config.serveDefaults(model_mod.ServeDefaults);
    config.max_position_embeddings = 1_048_576;
    config.num_attention_heads = 64;
    try t.expectEqual(@as(u32, 8192), bills.widestServedRung(&config));
    try t.expectEqual(config.prefillStartWidth(), bills.widestServedRung(&config));
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
    const state = (c.stateReservedBytes(false, seq + 2048) + c.stateFixedBytes(true) + bills.checkpointReserve(&config, seq, 2048)) * 5 / 4;
    // A load on MLX: its chunk bills MLX's per-row rate.
    const work = c.forwardCapOn(false, bills.requestForwardRows(&config, seq, 2048, 0), seq) + config.expertPoolBytes();
    const decode = c.forwardCap(dsv41_config.verify_rows, seq + 2048 + dsv41_config.verify_rows);
    // No pool to lend: the whole working set is billed, as before.
    try t.expectEqual(state + work, bills.prefillNeededAtChunk(&config, seq, 2048, 16, 2048, .{}));
    // Pools that cover part of it: admission pays only the rest.
    config.expert_borrow_bytes = work / 2;
    try t.expectEqual(state + work - work / 2, bills.prefillNeededAtChunk(&config, seq, 2048, 16, 2048, .{}));
    // Pools that cover all of it: the state, a decode forward and the decode pool it leaves parked, at any chunk.
    config.expert_borrow_bytes = 60_000_000_000;
    for ([_]u64{ 512, 1024, 2048 }) |chunk| {
        const ck = (c.stateReservedBytes(false, seq + 2048) + c.stateFixedBytes(true) + bills.checkpointReserve(&config, seq, chunk)) * 5 / 4;
        try t.expectEqual(ck + decode + config.expertPoolBytes(), bills.prefillNeededAtChunk(&config, seq, 2048, 16, chunk, .{}));
    }
    // The widest rung is then always affordable, and a chunk boundary never steps down for the working set.
    const MB: u64 = 1 << 20;
    try t.expectEqual(@as(u32, 8192), bills.chooseRequestPrefillChunk(&config, seq, 2048, 16, state + decode + config.expertPoolBytes(), 512, 0, .{}));
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
    // dsv41-memory: 2.9 GB at 1M with the prefill borrowed (state, writer in flight, rings, x5/4, the decode cap), plus the
    // 1 GiB decode pool the ledger parks beside it.
    try t.expectApproxEqAbs(@as(f64, 4_000_000_000), @as(f64, @floatFromInt(bill)), 300_000_000);
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
    // With nothing to lend the request is billed the working set: it fits the whole context here, and less in a tighter room.
    config.expert_borrow_bytes = 0;
    try t.expectEqual(@as(u32, 1_000_000), bills.advertisedContextAt(&config, 16, ceiling, active, 512));
    // The same room a GiB tighter: the request's row stores, made whole at its last row, take no growth slack.
    try t.expectEqual(@as(u32, 700_416), bills.advertisedContextAt(&config, 16, ceiling, active + (1 << 30), 512));
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
    try t.expectEqual(unlent - (c.chunkCap(chunk, seq) - c.forwardCap(dsv41_config.verify_rows, seq + 96 + dsv41_config.verify_rows)), lent);
    // ...but a forward the headroom holds releases none and takes its whole cap from the pool: what it asks is measured
    // against the bill with nothing lent, the same with or without pools to lend.
    try t.expectEqual(unlent, bills.prefillNeededAtChunkUnlent(&config, seq, 96, 16, chunk, .{}));
    config.expert_borrow_bytes = 0;
    try t.expectEqual(unlent, bills.prefillNeededAtChunkUnlent(&config, seq, 96, 16, chunk, .{}));
}

test "DeepSeek's request bill on a Cooler load is its exact state and one forward's cap, whose margin is the bill's one: MLX's parked pool, the MLX net and MLX's row rate are not billed" {
    const t = std.testing;
    var config = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .stream_mtp_head = true };
    config.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    config.serve_defaults = dsv41_config.Config.serveDefaults(model_mod.ServeDefaults);
    config.max_position_embeddings = 1_048_576;
    const c = &config.dsv41.?;
    for ([_]u64{ 9_000, 225_000, 1_000_000 }) |seq| {
        // What the request stores: its state rows and fixed state, and the checkpoints its prefill keeps.
        // The request's row stores, made at its last row: MLX's at their rows, the engine's at pool size classes.
        const kept = c.stateFixedBytes(config.drafterOn()) + dsv41_bill.checkpointReserve(&config, seq, 2048);
        const stored_off = c.stateReservedBytes(false, seq + 2048) + kept;
        config.cooler_for_test = false;
        const off = bills.prefillNeededAtChunk(&config, seq, 2048, 16, 2048, .{});
        config.cooler_for_test = true;
        const net = stored_off * 5 / 4 - (c.stateReservedBytes(true, seq + 2048) + kept);
        const rows = bills.requestForwardRows(&config, seq, 2048, 0);
        const engine_chunk = c.forwardCapOn(true, rows, seq);
        try t.expectEqual(off - dsv41_config.dsv41_pool_bytes - net - c.forwardCapOn(false, rows, seq) + engine_chunk, bills.prefillNeededAtChunk(&config, seq, 2048, 16, 2048, .{}));
    }
}


test "DeepSeek: a request's decode state is billed as growth after admission only on a fully resident load, which joins rows" {
    var config = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .stream_mtp_head = true };
    config.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    // A streamed load serves one row: the model's parked state is resident before the request is admitted.
    try std.testing.expectEqual(@as(u64, 0), bills.admissionCommitBytes(&config, 4096, 512, 16, 2048, .{}));
    // A fully resident load joins rows: the request makes its own state at its first forward, all of it after admission.
    config.expert_resident = true;
    // Rows are Sushi Cooler's: a resident load on MLX keeps the one-row decode and its parked state.
    try std.testing.expectEqual(@as(u64, 0), bills.admissionCommitBytes(&config, 4096, 512, 16, 2048, .{}));
    config.cooler_for_test = true;
    const c = &config.dsv41.?;
    const state = c.stateReservedBytes(true, 4096 + 512) + c.stateFixedBytes(config.drafterOn());
    try std.testing.expectEqual(state, bills.admissionCommitBytes(&config, 4096, 512, 16, 2048, .{}));
}

/// What the served warm reads as its headroom right after a streamed load took what it holds under `ceiling`, the
/// decode pool among it parked as the engine's.
fn warmHeadroomAfterLoad(config: *const model_mod.ModelConfig, ceiling: u64, weights: u64) u64 {
    const held = bills.expertStreamingGateBytes(weights, config.expert_cache_bytes, model_load.loadStagingBytes(config, config.expert_fill_peak_bytes, config.expert_workspace_bytes), config.expert_bounce_bytes);
    const pool = config.expertPoolBytes();
    return bills.liveHeadroomFrom(config, ceiling, held - pool, 0, pool);
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
    const ceiling = bills.wiredLimitFloor(28_000 * MiB, 32 << 30, 1 << 30).?;
    const need = bills.loadRequirementBytes(&config, weights, true, 0);
    var avail = need -| (2 << 30);
    while (avail < need + (2 << 30)) : (avail += 50 * MiB) {
        if (bills.memInsufficientForLoad(weights, avail, need)) {
            // The refusal's budget is one the ledger's planner fits in what is free.
            const fits = bills.streamedBudgetThatFits(config.expert_ssd_budget_bytes, avail, SlotPlanner{ .config = &config, .weights = weights });
            if (fits > 0) try t.expect(SlotPlanner.needAt(.{ .config = &config, .weights = weights }, fits << 30).? <= avail);
            continue;
        }
        try t.expect(bills.warmChunkAdmitted(&config, bits, top, warmHeadroomAfterLoad(&config, ceiling, weights)) > 0);
    }

    // The declared ceiling: every one the load's fit admits leaves the warm a chunk with memory to spare.
    var c: u64 = 18_000 * MiB;
    while (c < 24_000 * MiB) : (c += 50 * MiB) {
        if (!bills.expertStreamingConfigFits(&config, c, 0, weights)) continue;
        try t.expect(bills.warmChunkAdmitted(&config, bits, top, warmHeadroomAfterLoad(&config, c, weights)) > 0);
    }

    // The RAM default's budget is bounded by a request at the default context beside it, which leaves the warm a chunk.
    try t.expect(bills.defaultRequestBill(&config) >= bills.warmFloorChunkBytes(&config));
}

test "DeepSeek: the iogpu.wired_limit_mb a refused ssd budget names admits that budget, the unplanned margin under it included" {
    const t = std.testing;
    var config = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .stream_mtp_head = true };
    config.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    config.serve_defaults = dsv41_config.Config.serveDefaults(model_mod.ServeDefaults);
    config.max_position_embeddings = 1_048_576;
    const MiB: u64 = 1 << 20;
    const GiB: u64 = 1 << 30;
    const ram: u64 = 128 * GiB;
    // macOS's default limit and two declared ones, at the default margin and a raised one; a budget just past each.
    for ([_]u64{ 1, 4 }) |margin_gib| for ([_]u64{ 0, 100_000, 110_000 }) |wired_mb| {
        const unplanned = margin_gib * GiB;
        const floor = bills.wiredLimitFloor(if (wired_mb > 0) wired_mb * MiB else null, ram, unplanned);
        const r = bills.FitReadings{ .ceiling = .{ .declared = if (wired_mb > 0) wired_mb * MiB else null, .working_set = 96 * GiB, .total_ram = ram, .margin = unplanned, .unplanned = unplanned }, .free = ram };
        const limit = floor orelse r.ceiling.working_set;
        config.expert_ssd_budget_bytes = (limit -| bills.expertStreamingBesideBudgetBytes(&config)) / GiB * GiB + GiB;
        const refused = bills.expertCacheRefusal(&config, 0, r) orelse return error.TestExpectedRefusal;
        try t.expectEqual(error.SsdBudgetExceedsWiredLimit, refused.err);
        const mb = refused.refusal.wiredLimitThatFits() orelse return error.TestExpectedWiredLimit;
        var raised = r;
        raised.ceiling.declared = mb * MiB;
        if (bills.expertCacheRefusal(&config, 0, raised)) |again| {
            var buf: [2048]u8 = undefined;
            std.debug.print("margin {d} GiB, iogpu.wired_limit_mb {d}, --ssd-budget-gb {d}: the refusal names {d}, under which the same check refuses again:\n{s}\n", .{ margin_gib, wired_mb, config.expert_ssd_budget_bytes / GiB, mb, bills.loadMemoryRefusalText(&buf, again.refusal) });
            return error.TestUnexpectedResult;
        }
    };
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
    // `--prefill-chunk`, as admission reads it.
    const sr = @import("../settings_resolve.zig");
    const saved_chunk = sr.prefill_chunk;
    defer sr.prefill_chunk = saved_chunk;
    if (explicit > 0) sr.prefill_chunk = .{ .value = explicit, .source = .flag };
    var free: u64 = if (cooler) pool else 0;
    const available = bills.admissionAvailableFor(config, avail, 0, free);
    const billed = bills.chooseRequestPrefillChunk(config, seq, max_tokens, bits, available, 0, explicit, .{});
    if (bills.prefillNeededAtChunk(config, seq, max_tokens, bits, billed, .{}) > available) return null;
    const adaptive = explicit == 0;
    const cap: u32 = @min(prefill_plan.effectivePrefillChunk(config.prefillScoreHeadDim(), config.num_attention_heads, seq, config.has_sliding_window, config.isMoe(), config.longCtxGated(), 0), config.prefillStartWidth());
    var cur: u32 = @intCast(prefill_plan.firstChunkWidth(config, billed, cap, adaptive));
    const loop_end = seq - 1;
    var st: mtp_plan.AdaptiveWidthState = .{ .width_min = @intCast(billed), .width_max = @intCast(billed), .loop_end = loop_end };
    var pos: u64 = 0;
    // The request's row stores are made at its last row before its first chunk, so they are all resident from the start.
    const grown = c.stateFixedBytes(config.drafterOn()) + c.stateReservedBytes(cooler, seq + max_tokens);
    while (pos < loop_end) {
        if (adaptive) {
            const next = bills.adaptivePrefillWidth(config, bits, pos, bills.headroomFrom(config, avail, grown, 0, free).bound(), cur, cap, &st);
            if (next != 0 and next < cur) cur = next;
        }
        const end = prefill_plan.nextChunkEnd(pos, loop_end, cur, false, 0, 0, adaptive);
        const need = c.forwardCapOn(cooler, end - pos, end);
        const cover = bills.headroomFrom(config, avail, grown, 0, free).forNeed(need) +| config.expert_borrow_bytes;
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

// Contract (engine-memory-admission#bill-net): a Cooler request bill is its exact bytes plus BILL_MARGIN_BYTES, once.
// `Config.forwardCap` carries that margin (dsv41-memory#transient-caps), so the bill takes the forward's cost under it.
test "DeepSeek's Cooler request bill carries its one margin once, not once in the forward's cap and again on the bill" {
    const t = std.testing;
    var config = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .stream_mtp_head = true };
    config.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    config.serve_defaults = dsv41_config.Config.serveDefaults(model_mod.ServeDefaults);
    config.max_position_embeddings = 1_048_576;
    config.cooler_for_test = true;
    const c = &config.dsv41.?;
    for ([_]u64{ 100, 9_000, 225_000, 1_000_000 }) |seq| for ([_]u64{ 512, 2048 }) |chunk| {
        const stored = c.stateReservedBytes(true, seq + 2048) + c.stateFixedBytes(config.drafterOn()) + dsv41_bill.checkpointReserve(&config, seq, chunk);
        const cap = @max(c.forwardCap(dsv41_config.verify_rows, seq + 2048 + dsv41_config.verify_rows), c.forwardCapOn(true, bills.requestForwardRows(&config, seq, chunk, 0), seq));
        const exact = stored + cap - bills.BILL_MARGIN_BYTES;
        const bill = bills.prefillNeededAtChunk(&config, seq, 2048, 16, chunk, .{});
        errdefer std.debug.print("DeepSeek seq {d} chunk {d}: bill {d} MB = exact {d} MB + {d} MB of margin\n", .{ seq, chunk, bill >> 20, exact >> 20, (bill -% exact) >> 20 });
        try t.expectEqual(exact + bills.BILL_MARGIN_BYTES, bill);
    };
}

// The decode and verify forwards a request runs last are at its reply's end, where their scores and the rows' growth
// span the whole context: a bill priced at the prompt's position under-billed a short prompt's long reply.
test "DeepSeek's request bill prices its decode forward at the reply's end, not at the prompt's" {
    const t = std.testing;
    var config = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .stream_mtp_head = true };
    config.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    config.serve_defaults = dsv41_config.Config.serveDefaults(model_mod.ServeDefaults);
    config.max_position_embeddings = 1_048_576;
    config.cooler_for_test = true;
    const c = &config.dsv41.?;
    const seq: u64 = 64;
    const reply: u32 = 1_000_000;
    const stored = c.stateReservedBytes(true, seq + reply) + c.stateFixedBytes(config.drafterOn());
    const decode = c.forwardCap(dsv41_config.verify_rows, seq + reply + dsv41_config.verify_rows);
    try t.expect(bills.prefillNeededAtChunk(&config, seq, reply, 16, 2048, .{}) >= stored + decode);
}

// Contract (engine-memory-admission, Admission): an explicit max_tokens that does not fit is refused by name, with the
// largest max_tokens that fits; any value a refusal names is admitted when applied.
test "DeepSeek: a refusal caused by max_tokens names the max_tokens and the context that fit, and both are admitted when applied" {
    const t = std.testing;
    var config = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .stream_mtp_head = true };
    config.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    config.serve_defaults = dsv41_config.Config.serveDefaults(model_mod.ServeDefaults);
    config.max_position_embeddings = 1_048_576;
    config.num_attention_heads = 64;
    config.pinned_context = 1_048_576;
    config.expert_fill_peak_bytes = model_load.billedFillPeakBytes(&config, 9_911 << 20);
    config.expert_borrow_bytes = 0;
    for ([_]bool{ false, true }) |cooler| {
        config.cooler_for_test = cooler;
        const seq: u64 = 64;
        const asked: u32 = 1_000_000;
        const needed = bills.prefillNeededAtChunk(&config, seq, asked, 16, 512, .{});
        const prompt_alone = bills.prefillNeededAtChunk(&config, seq, 1, 16, 512, .{});
        const available = prompt_alone + (needed - prompt_alone) / 2;
        const bill = bills.creditedAdmissionBill(&config, needed, available, 0, 0, 512);
        try t.expect(!bill.fits());
        const r = bills.requestContextRefusal(&config, seq, asked, @import("../kv_quant.zig").KVQuantConfig.dense, bill, .{}) orelse return error.TestExpectedRefusalFigures;
        try t.expect(r.fitting_max_tokens > 0 and r.fitting_max_tokens < asked);
        try t.expect(bills.prefillNeededAtChunk(&config, seq, r.fitting_max_tokens, 16, 512, .{}) <= available);
        try t.expect(r.maximum > seq);
        try t.expect(bills.prefillNeededAtChunk(&config, seq, @intCast(r.maximum - seq), 16, 512, .{}) <= available);
    }
}

// Contract (a1fcae5c): admission, the chunk width, the warm, image encodes and DeepSeek's borrow read one headroom. An
// encode on the engine runs under a cap of its bill (`arch_serve.enterVision`), so it reuses the free buffers a forward
// under its cap reuses, which the live headroom counts out as parked.
test "DeepSeek: an image encode on a Cooler load reads the headroom a forward of its size reads: the free buffers it reuses count" {
    const gb: u64 = 1 << 30;
    var config = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .stream_mtp_head = true };
    config.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    config.serve_defaults = dsv41_config.Config.serveDefaults(model_mod.ServeDefaults);
    config.cooler_for_test = true;
    const ceiling: u64 = 40 * gb;
    const active: u64 = 30 * gb;
    // The engine keeps a prefill's scratch as free buffers past the decode pool.
    const free = config.expertPoolBytes() + 2 * gb;
    const h = bills.headroomFrom(&config, ceiling, active, 0, free);
    const bill = h.live + gb;
    // What the borrow lets a forward of `bill` bytes take without releasing a pool, and what an encode of it may take.
    try std.testing.expect(h.forNeed(bill) >= bill);
    try std.testing.expect(h.bound() >= bill);
    // The live part alone, which an encode read before, is short of it.
    try std.testing.expect(h.live < bill);
}

/// A DeepSeek release load's plan at a budget, as the 32 GB rows of dsv41-memory size it: the trunk, MTP and vision
/// resident, a fixed workspace and bounce, and the expert cache the rest of the budget.
const ReleasePlan = struct {
    config: *const model_mod.ModelConfig,
    beside: u64 = 0,
    mtp_on: bool = false,
    const resident: u64 = 9_550_000_000;
    const workspace: u64 = 3_700_000_000;
    const bounce: u64 = 540_000_000;

    pub fn planAt(p: ReleasePlan, budget: u64) ?struct { config: model_mod.ModelConfig, resident: u64 } {
        if (budget < resident + workspace + bounce + (1 << 30)) return null;
        var c = p.config.*;
        c.expert_ssd_budget_bytes = budget;
        c.expert_cache_bytes = budget - resident - workspace - bounce;
        c.expert_workspace_bytes = workspace;
        c.expert_bounce_bytes = bounce;
        return .{ .config = c, .resident = resident };
    }

    pub fn needAt(p: ReleasePlan, budget: u64) ?u64 {
        const at = p.planAt(budget) orelse return null;
        return bills.loadRequirementBytes(&at.config, at.resident, false, 0);
    }
};

test "DeepSeek: the RAM default a load takes passes every check the load runs on the readings at the load, another model's buffers active" {
    const t = std.testing;
    var config = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .stream_mtp_head = true };
    config.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    config.serve_defaults = dsv41_config.Config.serveDefaults(model_mod.ServeDefaults);
    config.max_position_embeddings = 1_048_576;
    config.expert_fill_peak_bytes = model_load.billedFillPeakBytes(&config, 9_911 << 20);
    config.expert_borrow_bytes = 2_100_000_000;
    const plan = ReleasePlan{ .config = &config };
    const GiB: u64 = 1 << 30;
    const ram: u64 = 128 * GiB;
    const wired: u64 = 125_000 << 20;
    const limits = model_load.BudgetLimits{ .wired_limit = wired, .ceiling = wired, .ceiling_name = "iogpu.wired_limit_mb", .margin = GiB, .user_wire = 0, .user_wire_name = "", .user_wire_margin = GiB };
    // A 128 GB Mac at iogpu.wired_limit_mb 125000 with plenty of RAM free, and another model holding 20 GiB on the GPU.
    const readings = bills.FitReadings{ .ceiling = .{ .declared = wired, .working_set = wired, .total_ram = ram, .margin = GiB, .unplanned = GiB }, .active = 20 * GiB, .free = 118 * GiB };
    const d = model_load.ramDefaultAtLoad(.gb128, 0, 5_000_000_000, limits, plan, readings);
    // It steps down from 108 rather than leave the load to refuse it.
    try t.expect(d.bytes > 0 and d.bytes < 108 * GiB);
    try t.expectEqualStrings("the memory free at the load", d.bound_by);
    const at = plan.planAt(d.bytes).?;
    if (bills.expertCacheRefusal(&at.config, at.resident, readings)) |refused| {
        var buf: [2048]u8 = undefined;
        std.debug.print("the RAM default {d} GiB is refused at the load:\n{s}\n", .{ d.bytes >> 30, bills.loadMemoryRefusalText(&buf, refused.refusal) });
        return error.TestUnexpectedResult;
    }
    try t.expect(!bills.memInsufficientForLoad(at.resident, readings.preflightAvailable().bytes, plan.needAt(d.bytes).?));
    // With nothing else on the GPU the same Mac keeps 108.
    var idle = readings;
    idle.active = 0;
    try t.expectEqual(108 * GiB, model_load.ramDefaultAtLoad(.gb128, 0, 5_000_000_000, limits, plan, idle).bytes);
}

// Contract (engine-memory-admission "Every runtime ceiling..."): the advertised context is admitted when asked, at the readings
// the load saw and after an earlier request left the engine's free buffers (up to its free cap, RAM/16: 2 GiB on 32 GB),
// which admission takes off as parked (`dsv41_bill.parkedBytes`) and a forward reuses under its cap.
test "DeepSeek: a request at the advertised context is admitted at the readings the load saw, and after a request leaves the engine's free buffers" {
    var config = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .stream_mtp_head = true };
    config.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    config.serve_defaults = dsv41_config.Config.serveDefaults(model_mod.ServeDefaults);
    config.max_position_embeddings = 1_048_576;
    config.num_attention_heads = 64;
    config.expert_fill_peak_bytes = model_load.billedFillPeakBytes(&config, 9_911 << 20);
    const saved = .{ bills.launch_state.prefix_cache_ram_enabled, bills.launch_state.prefix_cache_disk_bytes };
    defer {
        bills.launch_state.prefix_cache_ram_enabled = saved[0];
        bills.launch_state.prefix_cache_disk_bytes = saved[1];
    }
    bills.launch_state.prefix_cache_ram_enabled = false;
    bills.launch_state.prefix_cache_disk_bytes = 20 << 30;
    const MiB: u64 = 1 << 20;
    const GiB: u64 = 1 << 30;
    var bad: u32 = 0;
    var tried: u32 = 0;
    for ([_]bool{ true, false }) |cooler| for ([_]u64{ 0, 2_100_000_000 }) |borrow| {
        config.cooler_for_test = cooler;
        config.expert_borrow_bytes = borrow;
        const bits = bills.defaultKvBits(&config);
        const active: u64 = 20 * GiB;
        var room: u64 = 1 * GiB;
        while (room <= 24 * GiB) : (room += 256 * MiB) for ([_]u64{ 512, 2048 }) |pin| {
            const ceiling = active + room;
            const adv = bills.advertisedContextAt(&config, bits, ceiling, active, pin);
            if (adv < 64) continue;
            var probe = config;
            probe.pinned_context = adv;
            probe.pinned_prefill_chunk = @intCast(pin);
            // The engine keeps free buffers only on Cooler.
            for ([_]u64{ 0, 512 * MiB, 1 * GiB, 2 * GiB }) |engine_free| {
                if (!cooler and engine_free > 0) continue;
                const available = bills.admissionAvailableFor(&probe, ceiling, active, engine_free);
                for ([_][2]u64{ .{ adv - 1, 1 }, .{ 15, adv - 15 } }) |req| {
                    tried += 1;
                    const mt: u32 = @intCast(req[1]);
                    const pick = bills.chooseRequestPrefillChunk(&probe, req[0], mt, bits, available, @intCast(pin), 0, .{});
                    const need = bills.prefillNeededAtChunk(&probe, req[0], mt, bits, pick, .{});
                    if (need <= available) continue;
                    if (bad < 6) std.debug.print("{s}, borrowable {d} MB, room {d} MB, pin {d}, engine free {d} MB: advertised {d}; prompt {d} + max_tokens {d} needs {d} MB at width {d}, admission has {d} MB\n", .{ if (cooler) "Cooler" else "MLX", borrow / MiB, room / MiB, pin, engine_free / MiB, adv, req[0], mt, need / MiB, pick, available / MiB });
                    bad += 1;
                }
            }
        };
    };
    try std.testing.expect(tried > 0);
    try std.testing.expectEqual(@as(u32, 0), bad);
}

/// The STQ-1.75bpw pack served fully resident (dsv41-memory#resident): every expert cached, no union workspace, no
/// bounce, and the decoder pools a prefill may release less the union workspace its first release makes.
fn residentPack(lend: bool) !model_mod.ModelConfig {
    var config = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .expert_resident = true, .stream_mtp_head = true };
    config.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    config.serve_defaults = dsv41_config.Config.serveDefaults(model_mod.ServeDefaults);
    config.max_position_embeddings = 1_048_576;
    config.cooler_for_test = true;
    config.expert_cache_bytes = 111_400_000_000;
    config.expert_borrow_bytes = if (lend) dsv41_borrow.plannedBytes(&config.dsv41.?, true, config.expert_cache_bytes) - 3_540_000_000 else 0;
    return config;
}

const resident_weights: u64 = 9_000_000_000;

test "DeepSeek: a fully resident load whose ceiling fits it and its floor warm serves every width as a rung, a borrow or a named refusal, never a forward past the headroom" {
    const t = std.testing;
    const MiB: u64 = 1 << 20;
    for ([_]bool{ true, false }) |lend| {
        const config = try residentPack(lend);
        const owned = resident_weights + config.expert_cache_bytes;
        const bits = bills.defaultKvBits(&config);
        const top = bills.widestServedRung(&config);
        var narrowed: u32 = 0;
        var borrowed: u32 = 0;
        // From the least ceiling the load's fit admits to one that holds the widest forward beside it.
        var ceiling: u64 = owned;
        while (ceiling < owned + 24 * (1 << 30)) : (ceiling += 32 * MiB) {
            if (!bills.expertStreamingConfigFits(&config, ceiling, 0, resident_weights)) continue;
            // The served warm takes the widest rung its headroom holds, and the fit leaves it one.
            const head = warmHeadroomAfterLoad(&config, ceiling, resident_weights);
            const rung = bills.warmChunkAdmitted(&config, bits, top, head);
            try t.expect(rung > 0);
            try t.expect(bills.warmChunkPeakBytes(&config, bits, rung) <= head);
            if (rung < top) {
                narrowed += 1;
                for (bills.PREFILL_CHUNK_LADDER) |wider| {
                    if (wider > rung and wider <= top) try t.expect(bills.warmChunkPeakBytes(&config, bits, wider) > head);
                }
            }
            // A request at that ceiling is admitted, then every forward of it is covered by the headroom and the pools it
            // may release, or it is refused at admission with its figures.
            for ([_]u64{ 4096, 131_072 }) |seq| {
                if (borrowRefusal(&config, seq, ceiling - owned, 0, 256, 0)) |r| {
                    std.debug.print("{s} resident, ceiling owned + {d} MiB: a {d}-token prompt admitted has its {d}-row forward ending at {d} refused {d} MB short\n", .{
                        if (lend) "lending" else "non-lending", (ceiling - owned) / MiB, seq, r.rows, r.end, r.short / MiB,
                    });
                    return error.TestUnexpectedResult;
                }
                const need = config.dsv41.?.chunkCap(top, seq);
                if (ceiling - owned < need and bills.prefillNeededAtChunk(&config, seq, 256, bits, top, .{}) <= ceiling - owned) borrowed += 1;
            }
        }
        try t.expect(narrowed > 0);
        // Pools to lend are what let the widest rung through a ceiling that cannot hold it.
        try t.expectEqual(lend, borrowed > 0);
    }
}

test "DeepSeek: a resident load with no decoder pool to lend bills its widest forward whole" {
    const t = std.testing;
    const config = try residentPack(false);
    const c = &config.dsv41.?;
    const top = bills.widestServedRung(&config);
    try t.expectEqual(@as(u64, 0), dsv41_borrow.plannedBytes(c, false, config.expert_cache_bytes));
    try t.expectEqual(@as(u64, 0), bills.lendableBytes(&config));
    const seq: u64 = 4096;
    const wide = c.chunkCap(top, seq) - dsv41_config.BILL_MARGIN_BYTES;
    try t.expect(wide >= dsv41_wide_bytes.bytes(c, bills.requestForwardRows(&config, seq, top, 0), seq));
    try t.expect(bills.prefillNeededAtChunk(&config, seq, 256, bills.defaultKvBits(&config), top, .{}) >= wide);
}

test "DeepSeek: a request's bill never falls as its prompt grows within a context, on Cooler and MLX, its checkpoints in RAM or SSD-first" {
    const t = std.testing;
    var config = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .stream_mtp_head = true };
    config.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    config.serve_defaults = dsv41_config.Config.serveDefaults(model_mod.ServeDefaults);
    config.max_position_embeddings = 1_048_576;
    const launch = bills.launch_state;
    const saved = .{ launch.prefix_cache_ram_enabled, launch.ssm_checkpoint_stride };
    defer {
        launch.prefix_cache_ram_enabled = saved[0];
        launch.ssm_checkpoint_stride = saved[1];
    }
    launch.ssm_checkpoint_stride = 2048;
    var falls: u32 = 0;
    for ([_]bool{ false, true }) |engine| for ([_]bool{ false, true }) |ram| for ([_]u64{ 2048, 8192 }) |chunk| for ([_]u64{ 9_000, 131_072, 1_000_000 }) |ctx| {
        config.cooler_for_test = engine;
        launch.prefix_cache_ram_enabled = ram;
        config.pinned_context = @intCast(ctx);
        var prev: u64 = 0;
        var p: u64 = 1;
        while (p <= ctx) : (p += @as(u64, if (ctx - p < 300 or p < 1100 or p % 2048 < 3 or p % 2048 > 2045) 1 else 97)) {
            const b = bills.prefillNeededAtChunk(&config, p, @intCast(ctx - p), 16, chunk, .{});
            if (b < prev) {
                falls += 1;
                if (falls < 10) std.debug.print("engine {} ram {} chunk {d} ctx {d}: prompt {d} bills {d} B under the prompt before it\n", .{ engine, ram, chunk, ctx, p, prev - b });
            }
            prev = b;
        }
    };
    try t.expectEqual(@as(u32, 0), falls);
}

test "DeepSeek's forward caps carry no growth for the row stores: a request makes them at its last row" {
    const t = std.testing;
    const c = try dsv41_config.parseText(dsv41_config.release_config);
    for ([_]u64{ 8192, 1 << 20 }) |seq| {
        try t.expectEqual(dsv41_config.narrow_rows * seq * 4 * 3 + dsv41_config.cap_fixed_bytes, c.forwardCap(dsv41_config.verify_rows, seq));
        try t.expectEqual(dsv41_wide_bytes.bytes(&c, 2048, seq) + dsv41_config.BILL_MARGIN_BYTES, c.forwardCap(2048, seq));
    }
}

test "DeepSeek: a request bound to its last row bills its stores whole at the first chunk that appends and nothing after" {
    const t = std.testing;
    var config = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .stream_mtp_head = true };
    config.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    config.serve_defaults = dsv41_config.Config.serveDefaults(model_mod.ServeDefaults);
    config.cooler_for_test = true;
    const c = &config.dsv41.?;
    const end: u32 = 9000;
    var room: @import("../kv_cache.zig").Room = .{};
    room.hold = .{ .end = end };
    // A chunk wide enough that every store appends: the stores are made at the rows the request ends at.
    const first = dsv41_bill.chunkGrowthBytes(&config, 8, 2048, 0, room).?;
    try t.expectEqual(c.stateReservedBytes(true, end), first);
    const after = dsv41_bill.roomAfter(&config, 2048, 0, room).?;
    try t.expectEqual(@as(u64, 0), dsv41_bill.chunkGrowthBytes(&config, 8, 2048, 2048, after).?);
    // On MLX the generic term stands: its stores are not the engine's.
    config.cooler_for_test = false;
    try t.expectEqual(@as(?u64, null), dsv41_bill.chunkGrowthBytes(&config, 8, 2048, 0, room));
}

test "DeepSeek: warmNeedBytes covers warmRequestBill with DSpark on Cooler" {
    const t = std.testing;
    var config = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .stream_mtp_head = true };
    config.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    config.serve_defaults = dsv41_config.Config.serveDefaults(model_mod.ServeDefaults);
    config.cooler_for_test = true;
    const floor: u64 = bills.PREFILL_CHUNK_LADDER[bills.PREFILL_CHUNK_LADDER.len - 1];
    const req = bills.warmRequestBill(&config, bills.defaultKvBits(&config), floor, true);
    const need = bills.warmNeedBytes(&config);
    errdefer std.debug.print("dsv41 floor={d}: warmNeedBytes={d} MB, warmRequestBill={d} MB\n", .{ floor, need >> 20, req >> 20 });
    try t.expect(need >= req);
}
