//! GLM-5.3's bill tests that read the live GPU: they run in the MLX tier, beside glm5_bill.zig's tests.
const std = @import("std");
const model_mod = @import("../model.zig");
const bills = @import("../bills.zig");
const glm5_generate = @import("glm5_generate.zig");
const glm5_prefix = @import("glm5_prefix.zig");
const glm_bill = @import("glm5_bill.zig");
const prefix_cache_mod = @import("../prefix_cache.zig");
const testing = std.testing;

test "GLM admission bills the checkpoints and assistant window a request keeps; the context sizer reserves none of them" {
    var cfg = try model_mod.parseConfigFromJson(std.testing.allocator, @embedFile("../fixtures/glm5_config.json"));
    cfg.dflash_loaded = true;
    cfg.dflash_window_bytes = 5 * 8 * 128 * 4 * 2304;
    cfg.dflash_capture_bytes_per_token = (5 * 4096 + 2 * 4096) * 2 + 5 * 8 * 128 * 4;
    const saved = .{ bills.launch_state.prefix_cache_capacity, bills.launch_state.ssm_checkpoint_stride, bills.launch_state.ssm_checkpoint_max };
    const tiers = bills.TierGuard.pin(true, 0);
    defer tiers.restore();
    defer {
        bills.launch_state.prefix_cache_capacity = saved[0];
        bills.launch_state.ssm_checkpoint_stride = saved[1];
        bills.launch_state.ssm_checkpoint_max = saved[2];
    }
    bills.launch_state.ssm_checkpoint_stride = 256;
    bills.launch_state.ssm_checkpoint_max = 16;
    const per_cp = cfg.ssmCheckpointBytes();
    try std.testing.expectEqual(@as(u64, 147_619_840), per_cp);
    bills.launch_state.prefix_cache_capacity = 0;
    const off = bills.prefillNeededAtChunk(&cfg, 65536, 1024, 8, 2048, .{ .checkpoints = 8 });
    const sizer_off = bills.prefillTransientReserve(&cfg, 8, 2048);
    const ctx_off = bills.memoryContextAtChunk(&cfg, 8, 120 << 30, 100 << 30, 2048);
    const cap = glm5_prefix.checkpointMax(16);
    try std.testing.expectEqual(@as(u64, 0), glm_bill.glmPrefixStateBytes(&cfg, 65536, .{ .checkpoints = cap }));
    bills.launch_state.prefix_cache_capacity = 32;
    // Every 8192-token grid point and the prompt end, thinned to the GLM cap; the newest
    // copy exists before each thin.
    const held = @as(u64, glm_bill.glmPrefillSchedule(65536, 0, cap).peak) * per_cp + cfg.dflash_window_bytes;
    try std.testing.expectEqual(held, glm_bill.glmPrefixStateBytes(&cfg, 65536, .{ .checkpoints = cap }));
    const on = bills.prefillNeededAtChunk(&cfg, 65536, 1024, 8, 2048, .{ .checkpoints = cap });
    try std.testing.expect(on - off >= held * 5 / 4 - 1 and on - off <= held * 5 / 4 + 1);
    // Fewer checkpoints hold fewer copies; none hold no window either.
    try std.testing.expectEqual(2 * per_cp + cfg.dflash_window_bytes, glm_bill.glmPrefixStateBytes(&cfg, 65536, .{ .checkpoints = 1 }));
    try std.testing.expectEqual(off, bills.prefillNeededAtChunk(&cfg, 65536, 1024, 8, 2048, .{}));
    // The advertised context and the RAM tier's clamp are sized as with the cache off.
    try std.testing.expectEqual(sizer_off, bills.prefillTransientReserve(&cfg, 8, 2048));
    try std.testing.expectEqual(@as(u64, 0), bills.ctxSizingCacheReserve(&cfg));
    try std.testing.expectEqual(ctx_off, bills.memoryContextAtChunk(&cfg, 8, 120 << 30, 100 << 30, 2048));
    // A warm restore at 61440 captures only over its own span: the prompt-end checkpoint (the grid point 65536 is past the loop's end).
    try std.testing.expectEqual(per_cp, bills.retainedSsmCheckpointBytes(&cfg, 65536, 61440, 2048));
    try std.testing.expect(bills.prefillNeededAtChunk(&cfg, 65536, 1024, 8, 2048, .{ .matched_tokens = 61440, .checkpoints = cap }) < on);
}

test "GLM bills two KDA states, not eight, while its prefill moves each to SSD and frees the RAM copy" {
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
    bills.launch_state.prefix_cache_capacity = 32;
    bills.launch_state.prefix_cache_disk_bytes = 12 << 30;
    bills.launch_state.prefix_cache_ram_enabled = false;
    const per_cp = cfg.ssmCheckpointBytes();
    const window = cfg.dflash_window_bytes;
    const cap = glm5_prefix.checkpointMax(16);
    const keeps = bills.WarmPrefix{ .checkpoints = cap };
    const releases = bills.WarmPrefix{ .checkpoints = cap, .released = true };
    // 65,536 tokens capture 7 grid points and the prompt end: eight held before, two now.
    try std.testing.expectEqual(8 * per_cp + window, glm_bill.glmPrefixStateBytes(&cfg, 65536, keeps));
    try std.testing.expectEqual(2 * per_cp + window, glm_bill.glmPrefixStateBytes(&cfg, 65536, releases));
    // A prompt that captures one checkpoint holds one either way.
    try std.testing.expectEqual(per_cp + window, glm_bill.glmPrefixStateBytes(&cfg, 100, releases));
    // The commit after the response: the checkpoints it lists, its two newest still in RAM.
    try std.testing.expectEqual(8 * per_cp + window, glm_bill.glmCommitStateBytes(&cfg, 65536, keeps, 8));
    try std.testing.expectEqual(2 * per_cp + window, glm_bill.glmCommitStateBytes(&cfg, 65536, releases, 8));
    // Admission follows the storage: the difference is the six states that are no longer held (a 5/4 margin rides the checkpoints).
    const kept_bill = bills.prefillNeededAtChunk(&cfg, 65536, 1024, 8, 2048, keeps);
    const released_bill = bills.prefillNeededAtChunk(&cfg, 65536, 1024, 8, 2048, releases);
    try std.testing.expect(kept_bill - released_bill >= 6 * per_cp * 5 / 4 - 1 and kept_bill - released_bill <= 6 * per_cp * 5 / 4 + 1);
}

test "a thought point is one more checkpoint state in the bill: every Qwen request, a GLM request that keeps checkpoints" {
    const t = std.testing;
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
    bills.launch_state.prefix_cache_capacity = 32;
    bills.launch_state.prefix_cache_disk_bytes = 12 << 30;
    bills.launch_state.prefix_cache_ram_enabled = false;

    var qwen = try model_mod.parseConfigFromJson(t.allocator, @embedFile("../fixtures/model-configs/qwen4_exp.json"));
    defer qwen.deinit(t.allocator);
    qwen.expert_layout = .exl3_k4;
    const q_per = qwen.ssmCheckpointBytes();
    try t.expect(q_per > 0);
    try t.expectEqual(q_per, bills.thoughtPointBytes(&qwen, .{ .thought = true }));
    try t.expectEqual(@as(u64, 0), bills.thoughtPointBytes(&qwen, .{}));
    const q_terms = bills.prefillRequestTerms(&qwen, 4096, 1024, 8, 512, .{ .thought = true });
    try t.expectEqual(bills.retainedSsmCheckpointBytes(&qwen, 4096, 0, 512) + q_per, q_terms.checkpoint_bytes);
    try t.expectEqual(bills.retainedSsmCheckpointBytes(&qwen, 4096, 0, 512), bills.prefillRequestTerms(&qwen, 4096, 1024, 8, 512, .{}).checkpoint_bytes);
    bills.launch_state.ssm_checkpoint_stride = 0;
    try t.expectEqual(@as(u64, 0), bills.thoughtPointBytes(&qwen, .{ .thought = true }));
    bills.launch_state.ssm_checkpoint_stride = 256;

    var glm = try model_mod.parseConfigFromJson(t.allocator, @embedFile("../fixtures/glm5_config.json"));
    defer glm.deinit(t.allocator);
    glm.dflash_loaded = true;
    const g_per = glm.ssmCheckpointBytes();
    try t.expectEqual(@as(u64, 147_619_840), g_per);
    try t.expectEqual(g_per, glm_bill.glmThoughtPointBytes(&glm, .{ .checkpoints = 8, .thought = true }));
    try t.expectEqual(g_per, glm_bill.glmThoughtPointBytes(&glm, .{ .checkpoints = 1, .released = true, .thought = true }));
    // A request that takes no thought point, keeps no checkpoints, or has no tier to take one bills nothing.
    try t.expectEqual(@as(u64, 0), glm_bill.glmThoughtPointBytes(&glm, .{ .checkpoints = 8 }));
    try t.expectEqual(@as(u64, 0), glm_bill.glmThoughtPointBytes(&glm, .{ .thought = true }));
    bills.launch_state.prefix_cache_capacity = 0;
    bills.launch_state.prefix_cache_disk_bytes = 0;
    try t.expectEqual(@as(u64, 0), glm_bill.glmThoughtPointBytes(&glm, .{ .checkpoints = 8, .thought = true }));
}

// ── Bill pins ──
// Each test pins one bill function's output on the committed fixture config of every served pack.

test "GLM admission bills the commit's row copy, so a long generation cannot commit above the ceiling, and the copy never costs a long prompt its checkpoints" {
    var cfg = try model_mod.parseConfigFromJson(std.testing.allocator, @embedFile("../fixtures/glm5_config.json"));
    cfg.dflash_loaded = true;
    cfg.dflash_window_bytes = 5 * 8 * 128 * 4 * 2304;
    const saved = .{ bills.launch_state.prefix_cache_capacity, bills.launch_state.ssm_checkpoint_stride, bills.launch_state.prefix_cache_disk_bytes, bills.launch_state.prefix_cache_ram_enabled, bills.launch_state.prefix_cache_mem_bytes };
    defer {
        bills.launch_state.prefix_cache_capacity = saved[0];
        bills.launch_state.ssm_checkpoint_stride = saved[1];
        bills.launch_state.prefix_cache_disk_bytes = saved[2];
        bills.launch_state.prefix_cache_ram_enabled = saved[3];
        bills.launch_state.prefix_cache_mem_bytes = saved[4];
    }
    bills.launch_state.ssm_checkpoint_stride = 256;
    bills.launch_state.prefix_cache_capacity = 32;
    bills.launch_state.prefix_cache_mem_bytes = bills.GLM_PREFIX_CACHE_MEM_DEFAULT;
    const per_cp = cfg.ssmCheckpointBytes();
    const row_bytes = bills.sessionBytesPerToken(&cfg, 8);
    const cap = glm5_prefix.checkpointMax(16);
    const Arm = struct { ram: bool, disk: u64 };
    for ([_]Arm{ .{ .ram = true, .disk = 0 }, .{ .ram = false, .disk = 12 << 30 }, .{ .ram = true, .disk = 12 << 30 } }) |arm| {
        const ram = arm.ram;
        bills.launch_state.prefix_cache_ram_enabled = ram;
        bills.launch_state.prefix_cache_disk_bytes = arm.disk;
        const staged: u64 = 0;
        for ([_]u64{ 64, 131_072, 900_000, 1_040_000 }) |seq| {
            const max_tokens: u32 = @intCast(bills.getEffectiveContextLength(&cfg) - seq);
            const oracle = glm5_generate.glmCaptureOracle(glm5_generate.glm_checkpoint_stride, glm5_generate.glm_capture_stride, cap, 0, seq, 2048);
            // Eight spread checkpoints below the newest, each at its real size.
            var positions: [8]usize = undefined;
            const bytes: [8]u64 = @splat(per_cp);
            for (&positions, 1..) |*p, i| p.* = if (i == 8) oracle.newest else @max(oracle.newest * i / 8 / 2048 * 2048, 4);
            const kept: usize = @min(@as(usize, oracle.events), 8);
            // What the commit copies: the RAM tier's rows within its budget, chosen before the copy.
            // The SSD tier copies none: its flush reads the request's own rows.
            const copied_len = if (!ram) 0 else glm5_prefix.glmCommitLenFor(
                .min_span_recency,
                bills.resolvedPrefixCacheMem(),
                false,
                oracle.newest,
                row_bytes,
                cfg.dflash_window_bytes,
                positions[8 - kept ..],
                bytes[8 - kept ..],
            ) orelse 0;
            const commit_peak = glm_bill.glmCacheBytes(&cfg, seq, max_tokens, 8) + copied_len * row_bytes + @as(u64, @intCast(kept)) * per_cp + cfg.dflash_window_bytes + staged;
            const needed = bills.prefillNeededAtChunk(&cfg, seq, max_tokens, 8, 2048, .{ .checkpoints = cap });
            std.testing.expect(needed >= commit_peak) catch |e| {
                std.debug.print("ram={any} seq {d}: admitted {d} bytes, commit peaks at {d}\n", .{ ram, seq, needed, commit_peak });
                return e;
            };
            // A long prompt's checkpoints are priced by its prefill alone: the copy is smaller than its transient.
            const warm = bills.WarmPrefix{ .checkpoints = cap };
            try std.testing.expect(glm_bill.glmCommitStateBytes(&cfg, seq, warm, 8) <= glm_bill.glm5TransientBytes(&cfg, seq, 2048, 8) +| glm_bill.glmPrefixStateBytes(&cfg, seq, warm));
            // SSD-only: checkpoints, window and the writer's staged bytes, no rows at any length.
            if (!ram) try std.testing.expectEqual(@as(u64, @min(oracle.events, cap)) * per_cp + cfg.dflash_window_bytes + staged, glm_bill.glmCommitStateBytes(&cfg, seq, warm, 8));
            // Both tiers: the RAM tier's copy; the disk tier adds no term of its own here.
            if (ram and arm.disk > 0) {
                bills.launch_state.prefix_cache_disk_bytes = 0;
                const ram_only = glm_bill.glmCommitStateBytes(&cfg, seq, warm, 8);
                bills.launch_state.prefix_cache_disk_bytes = arm.disk;
                try std.testing.expectEqual(ram_only, glm_bill.glmCommitStateBytes(&cfg, seq, warm, 8));
            }
        }
    }
}

test "GLM prefill bills the packed selection planes of its history" {
    const cfg = try model_mod.parseConfigFromJson(std.testing.allocator, @embedFile("../fixtures/glm5_config.json"));
    const attention = @import("glm5_attention.zig");
    const short: u64 = 1 << 19;
    const long: u64 = 1 << 20;
    // Beyond selection planes, only the growing kv8 latent (544) and pooled index (64) scale with history.
    const growth = (long - short) * (544 + 64);
    const planes = try attention.selectionScratchBytes(long) - try attention.selectionScratchBytes(short);
    try std.testing.expectEqual(growth + planes, bills.prefillTransientReserveAtKv(&cfg, 8, 2048, long) - bills.prefillTransientReserveAtKv(&cfg, 8, 2048, short));
}

test "GLM DFlash2 reserve at prefill end fits the admission bill at every request shape" {
    const reserve = @import("glm5_dflash_reserve.zig");
    var cfg = try model_mod.parseConfigFromJson(std.testing.allocator, @embedFile("../fixtures/glm5_config.json"));
    cfg.dflash_loaded = true;
    const mla_layers: usize = cfg.num_hidden_layers / cfg.full_attention_interval;
    const pool_row: usize = @as(usize, cfg.indexer_head_dim) * 2;
    const shapes = [_][2]u64{ .{ 16, std.math.maxInt(u32) }, .{ 18, 32 }, .{ 100, 920 }, .{ 1000, 24 }, .{ 1022, 2 }, .{ 2008, 2400 }, .{ 32768, 32768 }, .{ 131072, 4096 }, .{ 2008, std.math.maxInt(u32) }, .{ 500000, std.math.maxInt(u32) } };
    for ([_]u32{ 1024, 8192, 65536, 1048576 }) |ctx| {
        cfg.pinned_context = ctx;
        for ([_]u64{ 16, 8, 6 }) |kv_bits| {
            const latent_row: usize = if (kv_bits == 16) @as(usize, cfg.mla_kv_lora_rank) * 2 else @import("glm5_latent.zig").rowBytes(cfg.mla_kv_lora_rank, @intCast(kv_bits));
            for (shapes) |shape| {
                const seq: usize = @intCast(shape[0]);
                if (seq >= ctx) continue;
                const max_tokens: u32 = @intCast(@min(shape[1], ctx - seq));
                const lc = try reserve.capacity(seq);
                const pc = try reserve.capacity(seq / 4);
                const p = try reserve.plan(seq, lc, pc, latent_row, cfg.indexer_head_dim, 2, seq + max_tokens + 3);
                const held: u64 = mla_layers * (lc * latent_row + pc * pool_row) + cfg.ssmCheckpointBytes();
                try std.testing.expect(try reserve.statesPeak(mla_layers, p) <= bills.glmDflashReserveBudget(&cfg, seq, max_tokens, kv_bits, held));
            }
        }
    }
}

test "GLM DFlash2 reserve fits the admission bill when max_tokens fills the advertised context" {
    const reserve = @import("glm5_dflash_reserve.zig");
    var cfg = try model_mod.parseConfigFromJson(std.testing.allocator, @embedFile("../fixtures/glm5_config.json"));
    cfg.dflash_loaded = true;
    var mla_layers: usize = 0;
    for (0..cfg.num_hidden_layers) |l| mla_layers += @intFromBool(!cfg.isLinearLayer(@intCast(l)));
    const pool_row: usize = @as(usize, cfg.indexer_head_dim) * 2;
    for ([_]struct { bits: u64, ctx: u32 }{ .{ .bits = 8, .ctx = 1048576 }, .{ .bits = 6, .ctx = 1048576 }, .{ .bits = 16, .ctx = 524288 } }) |arm| {
        cfg.pinned_context = arm.ctx;
        const latent_row: usize = if (arm.bits == 16) @as(usize, cfg.mla_kv_lora_rank) * 2 else @import("glm5_latent.zig").rowBytes(cfg.mla_kv_lora_rank, @intCast(arm.bits));
        for ([_]usize{ 16, 2008, 100000 }) |seq| {
            const max_tokens: u32 = @intCast(arm.ctx - seq);
            const lc = try reserve.capacity(seq);
            const pc = try reserve.capacity(seq / 4);
            const p = try reserve.plan(seq, lc, pc, latent_row, cfg.indexer_head_dim, 2, seq + max_tokens + 3);
            const held: u64 = mla_layers * (lc * latent_row + pc * pool_row) + cfg.ssmCheckpointBytes();
            const peak = try reserve.statesPeak(mla_layers, p);
            // The generator's config copy is never pinned: it resolves its own, smaller context.
            var unpinned = cfg;
            unpinned.pinned_context = arm.ctx / 4;
            try std.testing.expect(peak <= bills.glmDflashReserveBudget(&unpinned, seq, max_tokens, arm.bits, held));
        }
    }
}
