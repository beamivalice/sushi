//! The bill tests that read the live GPU or need MLX. They run in the MLX tier; the rest of bills.zig runs in test-host.
const fixtures = @import("bills_fixtures_test.zig");
const arch_prefix = @import("arch/prefix.zig");
const bills = @import("bills.zig");
const launch_state = @import("server_state.zig");
const model_mod = @import("model.zig");
const mtp_plan = @import("mtp_plan.zig");
const slot_admission = @import("slot_admission.zig");
const std = @import("std");
const testing = std.testing;
const trunk = @import("trunk.zig");

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
    const cap = arch_prefix.checkpointMax(&cfg, 16);
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
    const cfg = model_mod.ModelConfig{ .model_type = "glm5_next", .fp8_block128_trunk = true };
    try std.testing.expectEqual(bills.wiredLimitFloor(120000 * 1024 * 1024, 128 << 30, launch_state.wired_limit_margin_bytes), bills.wiredCeilingFloorForRam(&cfg, 128 << 30));
    bills.wired_limit_mb_override = 98304;
    try std.testing.expectEqual(@as(u64, 0), bills.wiredCeilingFloorForRam(&cfg, 128 << 30));
}


test "GPU memory readings are MLX's plus the engine's pool: tensors in use as active, free buffers as cache" {
    const core = @import("tensor/core.zig");
    const mlx = @import("mlx.zig");
    const gpu_mem = @import("gpu_mem.zig");
    _ = @import("metal.zig").device() catch return error.SkipZigTest;
    try core.sync();
    const mb = 1 << 20;
    const held = try core.fresh(.uint8, &.{24 * mb});
    const idle = try core.fresh(.uint8, &.{16 * mb});
    core.release(idle);
    try core.sync();
    defer core.release(held);
    var active: usize = 0;
    var cache: usize = 0;
    _ = mlx.mlx_get_active_memory(&active);
    _ = mlx.mlx_get_cache_memory(&cache);
    const pool = core.stats();
    try testing.expect(pool.pool_free >= 16 * mb);
    try testing.expect(pool.pool_bytes - pool.pool_free >= 24 * mb);
    try testing.expectEqual(active + (pool.pool_bytes - pool.pool_free), gpu_mem.activeBytes());
    try testing.expectEqual(cache + pool.pool_free, gpu_mem.cacheBytes());
}

test "GPU memory readings count a mapping the engine wraps as active, and a pass keeps its queue in use only while one is wrapped" {
    const core = @import("tensor/core.zig");
    const metal = @import("metal.zig");
    const gpu_mem = @import("gpu_mem.zig");
    const mlx = @import("mlx.zig");
    _ = metal.device() catch return error.SkipZigTest;
    const Region = struct {
        fn release(payload: ?*anyopaque) callconv(.c) void {
            const span: *const [2]usize = @ptrCast(@alignCast(payload.?));
            _ = std.c.munmap(@ptrFromInt(span[0]), span[1]);
        }
    };
    const len = 4 * std.heap.pageSize();
    const mapped = std.c.mmap(null, len, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    try testing.expect(mapped != std.c.MAP_FAILED);
    var span = [2]usize{ @intFromPtr(mapped), len };
    try core.sync();
    if (core.stats().mapped_bytes != 0) return error.SkipZigTest;
    const idle = metal.touches;
    core.touch();
    try testing.expectEqual(idle, metal.touches);
    const before = gpu_mem.activeBytes();
    const t = try core.wrapMapped(@ptrCast(mapped), .uint8, &.{@intCast(len)}, &span, Region.release);
    try testing.expectEqual(before + len, gpu_mem.activeBytes());
    core.touch();
    try testing.expectEqual(idle + 1, metal.touches);
    // The engine wires the mapping itself, so MLX's own wired limit gives those bytes up while it stands.
    const limit = mlx.maxRecommendedWorkingSet();
    if (limit > 0) try testing.expectEqual(@as(?usize, limit - len), mlx.applyWiredPolicy());
    core.release(t);
    try core.sync();
    try testing.expectEqual(before, gpu_mem.activeBytes());
    if (limit > 0) try testing.expectEqual(@as(?usize, limit), mlx.applyWiredPolicy());
}

test "a wrapped mapping MLX takes for an op is MLX's only for that op: MLX's active memory gives its bytes back with the result" {
    const core = @import("tensor/core.zig");
    const metal = @import("metal.zig");
    const mlx = @import("mlx.zig");
    _ = metal.device() catch return error.SkipZigTest;
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer mlx.engine.enabled = was;
    const Region = struct {
        fn release(payload: ?*anyopaque) callconv(.c) void {
            const span: *const [2]usize = @ptrCast(@alignCast(payload.?));
            _ = std.c.munmap(@ptrFromInt(span[0]), span[1]);
        }
    };
    const len = 64 * std.heap.pageSize();
    const mapped = std.c.mmap(null, len, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    try testing.expect(mapped != std.c.MAP_FAILED);
    var span = [2]usize{ @intFromPtr(mapped), len };
    const t = try core.wrapMapped(@ptrCast(mapped), .float32, &.{@intCast(len / 4)}, &span, Region.release);
    const x = core.handle(t);
    defer _ = mlx.mlx_array_free(x);
    const gpu = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(gpu);
    var before: usize = 0;
    _ = mlx.mlx_get_active_memory(&before);
    const lends = core.stats().mapped_lends;
    {
        var sum = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sum);
        try mlx.check(mlx.mlx_add(&sum, x, x, gpu));
        try mlx.check(mlx.mlx_array_eval(sum));
    }
    // One lend an operand: the same mapping twice is two.
    try testing.expectEqual(lends + 2, core.stats().mapped_lends);
    // MLX lets a buffer go when the command buffer that read it has finished.
    _ = mlx.mlx_synchronize(gpu);
    var after: usize = 0;
    for (0..400) |_| {
        _ = mlx.mlx_get_active_memory(&after);
        if (after == before) break;
        std.Io.sleep(std.testing.io, .fromMilliseconds(5), .awake) catch {};
    }
    try testing.expectEqual(before, after);
}

test "GPU memory readings are MLX's alone while the engine has made no pool buffer" {
    const mlx = @import("mlx.zig");
    const gpu_mem = @import("gpu_mem.zig");
    if (mlx.engine.stats().pool_bytes != 0) return error.SkipZigTest;
    var active: usize = 0;
    var cache: usize = 0;
    _ = mlx.mlx_get_active_memory(&active);
    _ = mlx.mlx_get_cache_memory(&cache);
    try testing.expectEqual(@as(u64, 0), mlx.engine.poolTotal());
    try testing.expectEqual(@as(u64, 0), mlx.engine.poolFree());
    try testing.expectEqual(active, gpu_mem.activeBytes());
    try testing.expectEqual(cache, gpu_mem.cacheBytes());
}

test "the peak reading never sits under the active reading, right after a reset too" {
    const mlx = @import("mlx.zig");
    const gpu_mem = @import("gpu_mem.zig");
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var held = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(held);
    try mlx.check(mlx.mlx_zeros(&held, &[_]c_int{1 << 20}, 1, .float32, s));
    try mlx.check(mlx.mlx_array_eval(held));
    gpu_mem.resetPeak();
    try testing.expect(gpu_mem.peakBytes() >= gpu_mem.activeBytes());
}
