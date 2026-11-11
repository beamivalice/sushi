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
    try t.expectEqual(@as(u64, 10 << 30) - config.expertPoolBytes(), dsv41_bill.dsv41HeadroomOf(10 << 30, &config));
    try t.expectEqual(@as(u64, 0), dsv41_bill.dsv41HeadroomOf(1 << 20, &config));
    // The load fit holds the pool in its staging term, so the serving term it adds is the bill without it.
    config.expert_borrow_bytes = 60_000_000_000;
    const bill = bills.prefillNeededAtChunk(&config, bills.expertStreamingPlannedSeq(&config), 2048, bills.defaultKvBits(&config), 512, .{});
    try t.expectEqual(bill - config.expertPoolBytes(), bills.expertStreamingServingBytes(&config));
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
    const work = c.chunkCap(2048, seq) + config.expertPoolBytes();
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
    try t.expectEqual(@as(u32, 390_144), bills.advertisedContextAt(&config, 16, ceiling, active, 512));
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

test "DeepSeek's request bill adds a decode forward's cap for the pool of the engine of our own, only when decode runs on it" {
    const t = std.testing;
    const dsv41_borrow = @import("dsv41_borrow.zig");
    var config = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .stream_mtp_head = true };
    config.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    config.serve_defaults = dsv41_config.Config.serveDefaults(model_mod.ServeDefaults);
    config.max_position_embeddings = 1_048_576;
    const c = config.dsv41.?;
    const was = dsv41_borrow.own_override;
    defer dsv41_borrow.own_override = was;
    for ([_]u64{ 9_000, 225_000, 1_000_000 }) |seq| {
        dsv41_borrow.own_override = false;
        const off = bills.prefillNeededAtChunk(&config, seq, 2048, 16, 2048, .{});
        dsv41_borrow.own_override = true;
        try t.expectEqual(off + c.forwardCap(dsv41_config.verify_rows, seq), bills.prefillNeededAtChunk(&config, seq, 2048, 16, 2048, .{}));
    }
}

