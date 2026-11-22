//! The bill tests that read the live GPU or need MLX. They run in the MLX tier; the rest of bills.zig runs in test-host.
const fixtures = @import("bills_fixtures_test.zig");
const arch_prefix = @import("arch/prefix.zig");
const arch_media = @import("arch/media.zig");
const bills = @import("bills.zig");
const chat_mod = @import("chat.zig");
const launch_state = @import("server_state.zig");
const model_mod = @import("model.zig");
const request_media = @import("request_media.zig");
const server = @import("server.zig");
const mtp_plan = @import("mtp_plan.zig");
const slot_admission = @import("slot_admission.zig");
const std = @import("std");
const test_note = @import("test_note.zig");
const testing = std.testing;
const trunk = @import("trunk.zig");

test "GLM serving memory bills one latent at the request's KV width and pooled index per token" {

    try test_note.needArchs(&.{"glm5"});
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

    try test_note.needArchs(&.{"glm5"});
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

    try test_note.needArchs(&.{"glm5"});
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
    try t.expectEqual(@as(u32, 2048), bills.adaptivePrefillWidth(&cfg, kv_bits, 4096, bills.prefillChunkCost(&cfg, kv_bits, 2048, 4096, .{}), 2048, 2048, &st));
    try t.expectEqual(@as(u32, 1024), bills.adaptivePrefillWidth(&cfg, kv_bits, top, bills.prefillChunkCost(&cfg, kv_bits, 2048, top, .{}) - 1, 2048, 2048, &st));
    // Once stepped down, a request never widens again.
    try t.expectEqual(@as(u32, 1024), bills.adaptivePrefillWidth(&cfg, kv_bits, top, available, 1024, 2048, &st));

    // A small box narrows from the first chunk.
    st = .{};
    try t.expectEqual(@as(u32, 512), bills.adaptivePrefillWidth(&cfg, kv_bits, 0, bills.prefillChunkCost(&cfg, kv_bits, 512, 0, .{}), 2048, 2048, &st));
}

test "GLM near the top of its context keeps fewer checkpoints at the start width instead of being refused" {

    try test_note.needArchs(&.{"glm5"});
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

test "an unpinned model with no --ctx-size still has a concrete effective context" {
    const original = launch_state.server_config.max_context_size;
    defer launch_state.server_config.max_context_size = original;
    launch_state.server_config.max_context_size = 0;
    try testing.expect(bills.getEffectiveContextLength(&model_mod.ModelConfig{}) > 0);
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

    try test_note.needArchs(&.{"glm5"});
    var cfg = try model_mod.parseConfigFromJson(std.testing.allocator, @embedFile("fixtures/glm5_config.json"));
    cfg.pinned_context = 1_048_576;
    const gib: u64 = 1 << 30;
    const prompt: usize = 16;
    const max_tokens: u32 = 1_048_560;
    for ([_]u64{ 16, 8 }) |kv_bits| {
        const bill = bills.prefillAdmissionBill(&cfg, null, prompt, max_tokens, if (kv_bits == 16) trunk.KVQuantConfig.dense else trunk.KVQuantConfig.affine(8), false, null, .{});
        const nums = bills.prefillBillNumbersNow(&cfg, prompt, max_tokens, if (kv_bits == 16) trunk.KVQuantConfig.dense else trunk.KVQuantConfig.affine(8), false, false);
        const commit = bills.admissionCommitBytes(&cfg, prompt, max_tokens, kv_bits, bill.chunk, .{});
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
    const kv8 = bills.admissionCommitBytes(&cfg, prompt, max_tokens, 8, 2048, .{});
    try std.testing.expect(kv8 > 6 * gib + gib * 2 / 3 and kv8 < 6 * gib + gib * 7 / 10);
    try std.testing.expect(bills.admissionCommitBytes(&cfg, prompt, max_tokens, 16, 2048, .{}) > kv8);
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
        // The engine may call `release` once the test has returned: the span it is handed outlives the test.
        var span: [2]usize = undefined;
        fn release(payload: ?*anyopaque) callconv(.c) void {
            const at: *const [2]usize = @ptrCast(@alignCast(payload.?));
            _ = std.c.munmap(@ptrFromInt(at[0]), at[1]);
        }
    };
    const len = 4 * std.heap.pageSize();
    const mapped = std.c.mmap(null, len, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    try testing.expect(mapped != std.c.MAP_FAILED);
    Region.span = .{ @intFromPtr(mapped), len };
    try core.sync();
    if (core.stats().mapped_bytes != 0) return error.SkipZigTest;
    const idle = metal.touches;
    core.touch();
    try testing.expectEqual(idle, metal.touches);
    const before = gpu_mem.activeBytes();
    const t = try core.wrapMapped(@ptrCast(mapped), .uint8, &.{@intCast(len)}, &Region.span, Region.release);
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
        // The engine may call `release` once the test has returned: the span it is handed outlives the test.
        var span: [2]usize = undefined;
        fn release(payload: ?*anyopaque) callconv(.c) void {
            const at: *const [2]usize = @ptrCast(@alignCast(payload.?));
            _ = std.c.munmap(@ptrFromInt(at[0]), at[1]);
        }
    };
    const len = 64 * std.heap.pageSize();
    const mapped = std.c.mmap(null, len, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    try testing.expect(mapped != std.c.MAP_FAILED);
    Region.span = .{ @intFromPtr(mapped), len };
    const t = try core.wrapMapped(@ptrCast(mapped), .float32, &.{@intCast(len / 4)}, &Region.span, Region.release);
    // Runs once `x` is freed: takes back the lends MLX returned, so the mapping goes before the test returns.
    defer core.trim();
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

test "the width the scheduler runs a request at is the one its admission bill was taken at, the engine holding free buffers" {
    const core = @import("tensor/core.zig");
    _ = @import("metal.zig").device() catch return error.SkipZigTest;
    var cfg = fixtures.qwen4DeployedTestConfig();
    @import("cooler_side.zig").forTest(&cfg, true);
    // One free-RAM reading for the bill and the width: the machine's moves between them.
    const status = @import("status.zig");
    status.available_mem_override = status.getAvailableMemBytes();
    defer status.available_mem_override = null;
    // Free buffers the engine keeps: the live headroom leaves them out, and so must the width the request runs at.
    try core.sync();
    const idle = try core.fresh(.uint8, &.{256 << 20});
    core.release(idle);
    try core.sync();
    for ([_]usize{ 4096, 65_536, 262_144 }) |prompt| {
        const bill = bills.prefillAdmissionBill(&cfg, null, prompt, 4096, trunk.KVQuantConfig.affine(8), false, null, .{});
        const width = bills.requestPrefillChunkNow(&cfg, prompt, 4096, trunk.KVQuantConfig.affine(8), false, 0, 0, false, false);
        try testing.expectEqual(bills.admissionAvailableNow(&cfg), bill.available);
        try testing.expectEqual(bills.requestPrefillPick(&cfg, prompt, 4096, 8, bill.available, false, .{}), width);
    }
}

test "a model loaded while decoder pools are lent pins its auto-context and prefill chunk with them in use, as admission reads them" {
    const status = @import("status.zig");
    const avail_was = status.available_mem_override;
    status.available_mem_override = 1 << 40;
    defer status.available_mem_override = avail_was;
    var qwen = fixtures.qwen4ExpOomConfig();
    qwen.max_position_embeddings = 0;
    const Pin = struct { context: u32, chunk: u32, admission: u64 };
    const pin = struct {
        fn at(c: *const model_mod.ModelConfig) Pin {
            var fresh = c.*;
            fresh.pinned_prefill_chunk = 0;
            return .{ .chunk = bills.pinPrefillChunk(&fresh), .context = bills.computeMemoryContext(c), .admission = bills.admissionAvailableNow(c) };
        }
    }.at;
    const rest = pin(&qwen);
    // A lease that publishes all but 512 MiB of what admission holds free (`Engine.publishLent`).
    const lent = rest.admission -| (512 << 20);
    const lent_total = &@import("expert_stream.zig").lent_total;
    _ = lent_total.fetchAdd(lent, .monotonic);
    defer _ = lent_total.fetchSub(lent, .monotonic);
    const now = pin(&qwen);
    errdefer std.debug.print("{d} B lent: at rest {any}, during the lease {any}\n", .{ lent, rest, now });
    try testing.expect(rest.admission - now.admission >= lent);
    try testing.expect(now.context < rest.context);
    try testing.expect(now.chunk < rest.chunk);
}

const ArchCfg = struct { name: []const u8, cfg: model_mod.ModelConfig };

/// Every served arch's bill config, on each engine an arch runs on: Qwen on MLX and Cooler, MiMo, MiMo-Flash's band
/// geometry, GLM with and without its DFlash2 assistant, DeepSeek on MLX and Cooler.
fn everyArchConfig(fx: *const fixtures.PinFx, glm_dflash: *const model_mod.ModelConfig, out: *[9]ArchCfg) []ArchCfg {
    var n: usize = 0;
    if (bills.archBuilt("qwen4")) {
        out[n] = .{ .name = "qwen-mlx", .cfg = fixtures.qwen4MeasuredTestConfig() };
        out[n + 1] = .{ .name = "qwen-cooler", .cfg = @import("qwen4/qwen4_bill_test.zig").coolerFlashNext() };
        out[n + 2] = .{ .name = "qwen-pack", .cfg = fx.qwen };
        n += 3;
    }
    if (bills.archBuilt("mimo2")) {
        out[n] = .{ .name = "mimo", .cfg = fx.mimo };
        out[n + 1] = .{ .name = "mimo-flash", .cfg = fixtures.mimoV2FlashBillConfig() };
        n += 2;
    }
    if (bills.archBuilt("glm5")) {
        out[n] = .{ .name = "glm", .cfg = fx.glm };
        out[n + 1] = .{ .name = "glm-dflash", .cfg = glm_dflash.* };
        n += 2;
    }
    if (bills.archBuilt("dsv41")) {
        out[n] = .{ .name = "dsv-mlx", .cfg = fx.dsv };
        out[n].cfg.cooler_for_test = false;
        out[n + 1] = .{ .name = "dsv-cooler", .cfg = fx.dsv };
        out[n + 1].cfg.cooler_for_test = true;
        n += 2;
    }
    return out[0..n];
}

test "a request's prefill phase and its decode phase less the decode transient never fall as its prompt grows within a context, and the regrow alone falls only to nothing, every served arch on each engine, MTP on and off, the SSD tier on and off" {
    var fx = try fixtures.PinFx.init();
    defer fx.deinit();
    var glm_dflash = try fixtures.glmAssistantFixture();
    defer glm_dflash.deinit(testing.allocator);
    var buf: [9]ArchCfg = undefined;
    const cfgs = everyArchConfig(&fx, &glm_dflash, &buf);
    const edge = trunk.KVCache.RESERVE_MIN_TOKENS;
    var falls: u32 = 0;
    for ([_]bool{ false, true }) |tier| {
        const guard = bills.TierGuard.pin(false, if (tier) launch_state.prefix_cache_disk_bytes else 0);
        defer guard.restore();
        for (cfgs) |*c| for ([_]bool{ false, true }) |mtp| for ([_]u64{ 512, 8192 }) |chunk| for ([_]u64{ 16_383, 32_768, 70_001, 262_144 }) |ctx| {
            const warm: bills.WarmPrefix = .{ .mtp_on = mtp };
            var probe = c.cfg;
            probe.pinned_context = @intCast(ctx);
            var prev_prefill: u64 = 0;
            var prev_common: u64 = 0;
            var prev_regrow: u64 = 0;
            var p: u64 = 1;
            // Every prompt near a step a term may take (the forward widths, 256 rows, the reservation, the end).
            while (p <= ctx) : (p += @as(u64, if (ctx - p < 600 or p < 1100 or (p + 300 > edge and p < edge + 300) or p % 256 < 3 or p % 256 > 252) 1 else 67)) {
                const terms = bills.prefillRequestTerms(&probe, p, ctx - p, 8, chunk, warm);
                // A ringed arch bills its rings' compaction at every prompt, which no prompt's capacity holds: that floor is not the regrow.
                const regrow = (terms.decode_regrow_bytes -| bills.ringCompactionBytes(&probe, 8)) * terms.uplift_16ths / 16;
                // An arch that prices its own request has one bill, and no regrow.
                var prefill: u64 = bills.prefillNeededAtChunk(&probe, p, @intCast(ctx - p), 8, chunk, warm);
                var common = prefill;
                if (!bills.archBillsRequest(&probe)) {
                    const phases = bills.requestPhasesFor(&probe, p, @intCast(ctx - p), 8, chunk, warm);
                    prefill = phases.prefill;
                    // What the decode phase holds past the terms both phases hold: the larger of the regrow and a round.
                    common = phases.decode - @max(terms.decode_regrow_bytes, terms.round_bytes) * terms.uplift_16ths / 16;
                }
                if (prefill < prev_prefill or common < prev_common or (regrow < prev_regrow and regrow != 0)) {
                    falls += 1;
                    if (falls < 10) std.debug.print("{s} tier {} mtp {} chunk {d} ctx {d}: prompt {d} bills {d} B in the prefill and {d} B beside the decode transient, {d} and {d} B at the prompt before it, or its regrow falls from {d} to {d} B\n", .{ c.name, tier, mtp, chunk, ctx, p, prefill, common, prev_prefill, prev_common, prev_regrow, regrow });
                }
                prev_prefill = prefill;
                prev_common = common;
                prev_regrow = regrow;
            }
        };
    }
    try testing.expectEqual(@as(u32, 0), falls);
}

test "the width the max-context search prices never narrows as the context grows, every served arch on each engine" {
    var fx = try fixtures.PinFx.init();
    defer fx.deinit();
    var glm_dflash = try fixtures.glmAssistantFixture();
    defer glm_dflash.deinit(testing.allocator);
    var buf: [9]ArchCfg = undefined;
    const cfgs = everyArchConfig(&fx, &glm_dflash, &buf);
    var narrows: u32 = 0;
    for (cfgs) |*c| for ([_]u32{ 0, 512, 2048 }) |pin| {
        var probe = c.cfg;
        probe.pinned_prefill_chunk = pin;
        var prev: u64 = 0;
        var ctx: u64 = 1;
        while (ctx <= 1 << 20) : (ctx += @max(1, ctx / 64)) {
            probe.pinned_context = @intCast(ctx);
            const w = bills.sushiContextWidth(&probe, @intCast(ctx));
            if (w < prev) {
                narrows += 1;
                if (narrows < 10) std.debug.print("{s} pin {d}: context {d} priced at width {d}, under {d} a shorter context priced\n", .{ c.name, pin, ctx, w, prev });
            }
            prev = w;
        }
    };
    try testing.expectEqual(@as(u32, 0), narrows);
}

const live_gpu = @import("gpu_mem.zig");

/// One machine's readings for a live test: its RAM, declared wired limit and margin, the pages macOS can reclaim, and
/// the pool MLX and the engine hold beside the active bytes (none unless a test names one).
const Machine = struct {
    total: ?u64,
    wired_mb: ?u64,
    margin: ?u64,
    avail: ?u64,
    gpu_floor: u64,
    cache: ?usize = 0,

    fn pin(m: Machine) Machine {
        const status = @import("status.zig");
        const was: Machine = .{ .total = status.total_mem_override, .wired_mb = bills.wired_limit_mb_override, .margin = launch_state.wired_margin_flag_bytes, .avail = status.available_mem_override, .gpu_floor = bills.gpu_floor_bytes.load(.monotonic), .cache = live_gpu.cache_bytes_override };
        status.total_mem_override = m.total;
        bills.wired_limit_mb_override = m.wired_mb;
        launch_state.wired_margin_flag_bytes = m.margin;
        status.available_mem_override = m.avail;
        bills.gpu_floor_bytes.store(m.gpu_floor, .monotonic);
        live_gpu.cache_bytes_override = m.cache;
        return was;
    }
};

test "a pinned machine's room is the one it names, whatever pool MLX holds beside its active bytes" {
    const mlx = @import("mlx.zig");
    try mlx.requireGpu();
    var cfg = try model_mod.parseConfigFromJson(testing.allocator, @embedFile("fixtures/model-configs/qwen4_exp.json"));
    defer cfg.deinit(testing.allocator);
    // A freed buffer goes to MLX's pool when its command buffer completes, on a thread no test orders.
    var z: mlx.mlx_array = .{};
    try mlx.check(mlx.mlx_zeros(&z, &[_]c_int{(2 << 30) / 4}, 1, .float32, mlx.gpuStream()));
    try mlx.check(mlx.mlx_array_eval(z));
    _ = mlx.mlx_array_free(z);
    defer _ = mlx.mlx_clear_cache();
    var spins: usize = 0;
    while (live_gpu.cacheBytes() < 2 << 30 and spins < 10_000_000) : (spins += 1) std.Thread.yield() catch {};
    try testing.expect(live_gpu.cacheBytes() >= 2 << 30);
    const saved_static = bills.static_ceiling_override;
    defer bills.static_ceiling_override = saved_static;
    bills.static_ceiling_override = 48 << 30;
    const was = (Machine{ .total = 64 << 30, .wired_mb = 0, .margin = 1 << 30, .avail = 4 << 30, .gpu_floor = 0 }).pin();
    defer _ = was.pin();
    try testing.expectEqual(@as(u64, 4 << 30), bills.admissionAvailableNow(&cfg));
}

test "a declared wired limit is the ceiling a request is admitted under on MiMo and GLM too, streamed or resident, and on an arch the engine does not serve" {
    const MiB: u64 = 1 << 20;
    const was = (Machine{ .total = 128 << 30, .wired_mb = 120_000, .margin = 1 << 30, .avail = 2 << 30, .gpu_floor = 0 }).pin();
    defer _ = was.pin();
    const floor = bills.wiredLimitFloor(120_000 * MiB, 128 << 30, 1 << 30).?;
    const active: u64 = live_gpu.activeBytes();
    var mimo = model_mod.ModelConfig{ .model_type = "mimo_v2", .expert_streaming = true };
    var glm = model_mod.ModelConfig{ .model_type = "glm5_next", .fp8_block128_trunk = true };
    const other = model_mod.ModelConfig{ .model_type = "not_a_served_arch" };
    try testing.expect(!mimo.longCtxGated());
    const cases = [_]*const model_mod.ModelConfig{ &mimo, &glm, &other };
    for (cases) |c| try testing.expectEqual(floor -| active, bills.admissionAvailableNow(c));
    mimo.expert_streaming = false;
    try testing.expectEqual(floor -| active, bills.admissionAvailableNow(&mimo));
    // Any declared limit is the ceiling, whatever its size against the RAM.
    bills.wired_limit_mb_override = 98_304;
    glm.expert_streaming = false;
    try testing.expectEqual(bills.wiredLimitFloor(98_304 * MiB, 128 << 30, 1 << 30).? -| active, bills.admissionAvailableNow(&glm));
}

test "#110: under a declared wired limit a Flash-Next request on MLX is admitted up to the limit, however little free RAM other apps leave" {
    var cfg = try model_mod.parseConfigFromJson(testing.allocator, @embedFile("fixtures/model-configs/qwen4_exp.json"));
    defer cfg.deinit(testing.allocator);
    cfg.pinned_context = 262_144;
    const MiB: u64 = 1 << 20;
    // The reporter's M2 Ultra 64 GB: `iogpu.wired_limit_mb=59392 --wired-margin-gib 4`, 3,501 MB that macOS could still
    // reclaim beside the 42.8 GB pack (1.7.0's reading at margin 2: 1,453 MB available + 2,048).
    const was = (Machine{ .total = 64 << 30, .wired_mb = 59_392, .margin = 4 << 30, .avail = 3_501 * MiB, .gpu_floor = 340 * MiB }).pin();
    defer _ = was.pin();
    const kv4 = trunk.KVQuantConfig.affine(4);
    const active: u64 = live_gpu.activeBytes();
    try testing.expectEqual((55_296 * MiB) -| active, bills.admissionAvailableNow(&cfg));
    // The whole 262,144-token context, a 32,000-token reply in it, as asked at `--kv-quant 4`.
    const bill = bills.prefillAdmissionBill(&cfg, null, 262_144 - 32_000, 32_000, kv4, false, null, .{});
    errdefer std.debug.print("needed {d} MB, available {d} MB\n", .{ bill.needed >> 20, bill.available >> 20 });
    try testing.expect(bill.fits());
}

test "#110: with no declared wired limit a request is bounded by the working set and the RAM macOS can reclaim, as on 1.6.0" {
    var cfg = try model_mod.parseConfigFromJson(testing.allocator, @embedFile("fixtures/model-configs/qwen4_exp.json"));
    defer cfg.deinit(testing.allocator);
    cfg.pinned_context = 262_144;
    const free: u64 = 2 << 30;
    const was = (Machine{ .total = null, .wired_mb = 0, .margin = 1 << 30, .avail = free, .gpu_floor = 0 }).pin();
    defer _ = was.pin();
    const active: u64 = live_gpu.activeBytes();
    const footprint = active + live_gpu.cacheBytes();
    try testing.expectEqual(@min(live_gpu.getGpuWorkingSetLimit(), footprint + free) -| active, bills.admissionAvailableNow(&cfg));
}

test "#109: admission under a declared wired limit reads the same room whatever free RAM moves to between requests" {
    var cfg = try model_mod.parseConfigFromJson(testing.allocator, @embedFile("fixtures/model-configs/qwen4_exp.json"));
    defer cfg.deinit(testing.allocator);
    cfg.pinned_context = 208_896;
    const MiB: u64 = 1 << 20;
    const kv4 = trunk.KVQuantConfig.affine(4);
    // #109's readings between requests with nothing beside them (4.5, 3.2, 3.1, 2.1 and 1.7 GB free), on #110's
    // declared limit; its prompts of 156k-183k tokens with max_gen 52,769 at KV4, MTP on.
    var room: ?u64 = null;
    for ([_]u64{ 4_608, 3_277, 3_174, 2_150, 1_741 }) |free_mb| {
        const was = (Machine{ .total = 64 << 30, .wired_mb = 59_392, .margin = 4 << 30, .avail = free_mb * MiB, .gpu_floor = 340 * MiB }).pin();
        defer _ = was.pin();
        const available = bills.admissionAvailableNow(&cfg);
        try testing.expectEqual(room orelse available, available);
        room = available;
        for ([_]usize{ 156_000, 183_000 }) |prompt| {
            const bill = bills.prefillAdmissionBill(&cfg, null, prompt, 52_769, kv4, false, null, .{ .mtp_on = true });
            errdefer std.debug.print("free {d} MB: a {d}-token prompt needs {d} MB, available {d} MB\n", .{ free_mb, prompt, bill.needed >> 20, bill.available >> 20 });
            try testing.expect(bill.fits());
        }
    }
}

test "a 64 GB Mac at a declared 59,000 MB serves the 2.75 pack at kv8 to 262,144 with a 124K prompt, and an image at 131,072 and 262,144, whatever free RAM other apps leave" {
    if (!bills.archBuilt("qwen4")) return error.SkipZigTest;
    var cfg = try model_mod.parseConfigFromJson(testing.allocator, @embedFile("fixtures/model-configs/qwen4_exp.json"));
    defer cfg.deinit(testing.allocator);
    cfg.expert_layout = .exl3_k4;
    cfg.gdn_state_f32 = false; // --qwen-gdn 16
    const MiB: u64 = 1 << 20;
    const saved = .{ trunk.Trunk.mtp_head_kv_quant_flag, launch_state.server_config.max_context_size, launch_state.configured_kv_quant, launch_state.configured_mtp, live_gpu.active_bytes_override };
    defer {
        trunk.Trunk.mtp_head_kv_quant_flag = saved[0];
        launch_state.server_config.max_context_size = saved[1];
        launch_state.configured_kv_quant = saved[2];
        launch_state.configured_mtp = saved[3];
        live_gpu.active_bytes_override = saved[4];
    }
    trunk.Trunk.mtp_head_kv_quant_flag = true;
    launch_state.server_config.max_context_size = 0; // no --ctx-size: the model's own
    const kv8 = trunk.KVQuantConfig.affine(8);
    launch_state.configured_kv_quant = kv8;
    launch_state.configured_mtp = true;
    // The booted pack: its resident bytes (trunk, vision tower, MTP head and its rerank copy) and the 1 GiB hot cache.
    live_gpu.active_bytes_override = 49_784_531_962 + (1 << 30);

    const vp = request_media.visionPreprocFromConfig(&cfg);
    const plan = arch_media.planImage(vp, 1536, 1536).?;
    const shot = [_]chat_mod.ImageData{.{ .pixels = &.{}, .width = plan.width, .height = plan.height, .grid_h = plan.grid_h, .grid_w = plan.grid_w }};
    const shot_rows = arch_media.imageRows(&cfg, vp.merge, plan.grid_h, plan.grid_w);

    var first: ?struct { bill: bills.AdmissionBill, image: u64, advertised: u32 } = null;
    // #109's readings (4.5, 3.2, 3.1, 2.1 and 1.7 GB free), the 64 GB user's 5.8 GB, and almost none.
    for ([_]u64{ 5_939, 4_608, 3_277, 3_174, 2_150, 1_741, 256 }) |free_mb| {
        const was = (Machine{ .total = 64 << 30, .wired_mb = 59_000, .margin = 1 << 30, .avail = free_mb * MiB, .gpu_floor = 340 * MiB }).pin();
        defer _ = was.pin();
        cfg.pinned_context = 0;
        const advertised = bills.getEffectiveContextLength(&cfg);
        const bill = bills.prefillAdmissionBill(&cfg, null, 124_000, 32_000, kv8, false, null, .{ .mtp_on = true });
        const image_bill = request_media.visionEncodeBill(&cfg, &shot, &.{}, shot_rows).bytes;
        errdefer std.debug.print("free {d} MB: advertised {d}; the 124K request needs {d} MB of {d} MB; the image encode needs {d} MB of {d} MB\n", .{ free_mb, advertised, bill.needed >> 20, bill.available >> 20, image_bill >> 20, request_media.visionEncodeAvailable(&cfg) >> 20 });
        try testing.expectEqual(bills.AdmissionVerdict.admit, bills.admissionVerdict(bill));
        try testing.expect(advertised >= 262_144);
        // The first chunk's width, asked with the headroom that admitted the request, on the pack's Cooler load: the
        // reservation's rows are its commit's, so the cache's first growth never steps the width below the admitted one.
        var cooler = cfg;
        cooler.cooler_for_test = true;
        cooler.expert_quant_codebook = .spec1;
        cooler.trunk_quant = null;
        cooler.quant_bits = 0;
        const chunk: u32 = @intCast(bill.chunk);
        const hold: trunk.Hold = .{
            .reserve = @import("prefill_plan.zig").reservedPrefillTokens(&cooler, 124_000, 32_000, chunk, 0),
            .end = trunk.KVCache.requestEnd(124_000, 32_000, advertised, 0),
        };
        var st_ungrown: mtp_plan.AdaptiveWidthState = .{};
        var st_reserved: mtp_plan.AdaptiveWidthState = .{ .room = .{ .hold = hold } };
        const ungrown = bills.adaptivePrefillWidth(&cooler, 8, 0, bill.available, chunk, chunk, &st_ungrown);
        const reserved = bills.adaptivePrefillWidth(&cooler, 8, 0, bill.available, chunk, chunk, &st_reserved);
        errdefer std.debug.print("free {d} MB: admitted at {d} rows, first chunk {d} without the reservation's growth, {d} with it\n", .{ free_mb, chunk, ungrown, reserved });
        try testing.expectEqual(@as(u64, 0), bills.chunkGrowthBytes(&cooler, 8, chunk, 0, .{ .hold = hold }));
        try testing.expectEqual(chunk, reserved);
        try testing.expectEqual(ungrown, reserved);
        // The context is the model's own, so the request is the whole window, as the user's client sent it.
        for ([_]u32{ 131_072, 262_144 }) |context| {
            cfg.pinned_context = context;
            const whole = bills.prefillAdmissionBill(&cfg, null, context - 32_000, 32_000, kv8, false, null, .{ .mtp_on = true });
            try testing.expect(whole.fits());
            try testing.expectEqual(@as(?request_media.MediaFault, null), server.towerFitFault(&cfg, &shot, &.{}, shot_rows, request_media.visionEncodeAvailable(&cfg)));
        }
        // The same verdict and the same figures at every reading.
        if (first) |f| {
            try testing.expectEqual(f.bill.needed, bill.needed);
            try testing.expectEqual(f.bill.available, bill.available);
            try testing.expectEqual(f.image, image_bill);
            try testing.expectEqual(f.advertised, advertised);
        } else first = .{ .bill = bill, .image = image_bill, .advertised = advertised };
    }
}

test "DeepSeek streamed under a declared wired limit reads the limit's floor as its headroom, however little free RAM other apps leave (#110)" {

    try test_note.needArchs(&.{"dsv41"});
    const dsv41_config = @import("dsv41/dsv41_config.zig");
    var cfg = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .stream_mtp_head = true };
    // A build without DeepSeek (`-Darchs`) has no field of its config to set.
    if (comptime bills.archBuilt("dsv41")) cfg.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    cfg.serve_defaults = dsv41_config.Config.serveDefaults(model_mod.ServeDefaults);
    cfg.max_position_embeddings = 1_048_576;
    cfg.cooler_for_test = true;
    cfg.expert_borrow_bytes = 0;
    const MiB: u64 = 1 << 20;
    // r-65's box, the owner's decision: a 128 GB M5 at iogpu.wired_limit_mb=125000 with the default 1 GiB margin and
    // 3 GB that macOS could still reclaim beside other apps. The declared limit is the user's contract for their room.
    const was = (Machine{ .total = 128 << 30, .wired_mb = 125_000, .margin = 1 << 30, .avail = 3_000 * MiB, .gpu_floor = 0 }).pin();
    defer _ = was.pin();
    const active: u64 = live_gpu.activeBytes() + live_gpu.lentBytes();
    try testing.expectEqual(bills.liveHeadroomFrom(&cfg, 123_976 * MiB, active, 0, live_gpu.engineFree()), bills.headroomNow(&cfg, 0).live);
}

test "a Qwen request commits the cache its admitted width reserves, not the default chunk's" {

    try test_note.needArchs(&.{"qwen4"});
    const cfg = fixtures.qwen4ExpOomConfig();
    const status = @import("status.zig");
    const avail_was = status.available_mem_override;
    status.available_mem_override = 1 << 40;
    defer status.available_mem_override = avail_was;
    const kv8 = trunk.KVQuantConfig.affine(8);
    // A long prompt reserves its prompt, its reply and one chunk of the width it is admitted at.
    const seq: usize = 40_000;
    const bill = bills.prefillAdmissionBill(&cfg, null, seq, 4096, kv8, false, null, .{});
    try testing.expect(bill.chunk < @import("settings_resolve.zig").prefill_chunk.value);
    const nums = bills.prefillBillNumbersNow(&cfg, seq, 4096, kv8, false, false);
    try testing.expectEqual(bills.cacheCommitBytes(&cfg, seq, 4096, 8, bill.chunk, false), nums.commit);
}

test "owner ruling: #109 at a declared limit equal to the macOS default and under it: admission reads the same room whatever free RAM moves to" {
    var cfg = try model_mod.parseConfigFromJson(testing.allocator, @embedFile("fixtures/model-configs/qwen4_exp.json"));
    defer cfg.deinit(testing.allocator);
    cfg.pinned_context = 208_896;
    const MiB: u64 = 1 << 20;
    // 64 GiB: 75% is 49,152 MiB; 40,000 is a limit lowered to keep room for other apps, as the #110 CHANGELOG advises.
    for ([_]u64{ 49_152, 40_000, 49_153 }) |wired_mb| {
        var room: ?u64 = null;
        for ([_]u64{ 4_608, 1_741 }) |free_mb| {
            const was = (Machine{ .total = 64 << 30, .wired_mb = wired_mb, .margin = 4 << 30, .avail = free_mb * MiB, .gpu_floor = 340 * MiB }).pin();
            defer _ = was.pin();
            const available = bills.admissionAvailableNow(&cfg);
            errdefer std.debug.print("iogpu.wired_limit_mb={d}, free {d} MB: admission reads {d} MB (first reading {?d} MB)\n", .{ wired_mb, free_mb, available >> 20, if (room) |r| r >> 20 else null });
            try testing.expectEqual(room orelse available, available);
            room = available;
        }
    }
}

/// What every load's runtime reads under a declared wired limit (owner ruling, #110): the limit's floor less its active
/// bytes, whatever free RAM other apps leave.
fn readsDeclaredFloor(cfg: *const model_mod.ModelConfig, machine: Machine, floor: u64) !void {
    const was = machine.pin();
    defer _ = was.pin();
    const active: u64 = live_gpu.activeBytes() + live_gpu.lentBytes();
    errdefer std.debug.print("{s} at iogpu.wired_limit_mb={?d}: headroom {d} MB, the floor less active {d} MB\n", .{ cfg.model_type, machine.wired_mb, bills.headroomNow(cfg, 0).live >> 20, (floor -| active) >> 20 });
    try testing.expectEqual(bills.liveHeadroomFrom(cfg, floor, active, 0, live_gpu.engineFree()), bills.headroomNow(cfg, 0).live);
    // The load preflight reads the same Ceiling, whichever engine loads the model.
    try testing.expect(bills.FitReadings.now().preflightAvailable().bytes.? <= floor -| active);
}

test "owner ruling: a declared iogpu.wired_limit_mb is the ceiling of every load, GLM and MiMo resident on MLX and DeepSeek resident included, at any size" {

    try test_note.needArchs(&.{"dsv41"});
    const MiB: u64 = 1 << 20;
    var glm = try model_mod.parseConfigFromJson(testing.allocator, @embedFile("fixtures/glm5_config.json"));
    defer glm.deinit(testing.allocator);
    const mimo = fixtures.mimoV2FlashBillConfig();
    const dsv41_config = @import("dsv41/dsv41_config.zig");
    var dsv = model_mod.ModelConfig{ .model_type = "deepseek_v41", .expert_resident = true };
    if (comptime bills.archBuilt("dsv41")) dsv.dsv41 = try dsv41_config.parseText(dsv41_config.release_config);
    dsv.serve_defaults = dsv41_config.Config.serveDefaults(model_mod.ServeDefaults);
    dsv.max_position_embeddings = 1_048_576;
    // A 128 GB Mac whose owner lowered the limit under macOS's default 75% (98,304 MiB) to keep room for other apps,
    // 2 GB reclaimable; the default 1 GiB margin.
    const machine: Machine = .{ .total = 128 << 30, .wired_mb = 90_000, .margin = 1 << 30, .avail = 2_000 * MiB, .gpu_floor = 0 };
    for ([_]*const model_mod.ModelConfig{ &glm, &mimo, &dsv }) |cfg| try readsDeclaredFloor(cfg, machine, 88_976 * MiB);
}

// engine-memory-admission "One bills.Ceiling per reading, the same for every load and arch" + "Under a declared
// iogpu.wired_limit_mb the ceiling IS the limit less --wired-margin-gib, however much RAM is free": a load's preflight
// is a reader of that Ceiling, so on a quiet box it reads no more than the floor, whichever engine loads the model.
test "under a declared wired limit every load's preflight reads at most the Ceiling's floor, an MLX load's included" {
    const MiB: u64 = 1 << 20;
    // The working set tracks the sysctl (`[mem] ceiling 26976 MB from iogpu.wired_limit_mb=28000 (working set 28000 MB`):
    // declare the limit this box's Metal reports, 40 GB of RAM free (a quiet box), the default 1 GiB margin.
    const ws = live_gpu.getGpuWorkingSetLimit();
    const was = (Machine{ .total = 64 << 30, .wired_mb = ws / MiB, .margin = 1 << 30, .avail = 40 << 30, .gpu_floor = 0 }).pin();
    defer _ = was.pin();
    const r = bills.FitReadings.now();
    const floor = r.ceiling.floor().?;
    const got = r.preflightAvailable().bytes.?;
    errdefer std.debug.print("preflight: limit {d} MiB, floor {d} MiB, active {d} MiB; preflight {d} MiB\n", .{ ws / MiB, floor / MiB, r.active / MiB, got / MiB });
    try testing.expect(got <= floor);
}

// defaultBudgetLimits' doc: "the residency limit ... and the lower user-wire sysctl, each less what stays unplanned under
// it"; engine-memory-admission: "The ssd budget is admitted against ... a declared iogpu.wired_limit_mb's floor (less the
// unplanned margin) ...; never the raw declared limit". The RAM default's room is the Ceiling's floor at most.
test "a default ssd budget's limits plan against the Ceiling's floor: the GPU floor's margin and a limit past RAM" {
    const MiB: u64 = 1 << 20;
    const GiB: u64 = 1 << 30;
    const model_load = @import("model_load.zig");
    const NoPreflight = struct {
        pub fn needAt(_: @This(), _: u64) ?u64 {
            return null;
        }
    };
    const cases = [_]Machine{
        // This box: 28,000 MiB declared, default 1 GiB flag, a 3 GiB GPU floor (other apps' Metal): unplanned 3.5 GiB.
        .{ .total = 32 * GiB, .wired_mb = 28_000, .margin = GiB, .avail = 20 * GiB, .gpu_floor = 3 * GiB },
        // A limit declared past the RAM: the floor clamps at the RAM.
        .{ .total = 32 * GiB, .wired_mb = 40_000, .margin = GiB, .avail = 20 * GiB, .gpu_floor = 0 },
    };
    var bad: usize = 0;
    for (cases) |m| {
        const was = m.pin();
        defer _ = was.pin();
        const c = bills.ceilingNow();
        const floor = c.floor().?;
        const limits = bills.defaultBudgetLimits();
        const d = model_load.ramDefaultBudget(.gb96, 0, 0, limits, NoPreflight{});
        const by_ceiling = limits.ceiling -| limits.margin;
        if (by_ceiling > floor or d.bytes > floor) {
            bad += 1;
            std.debug.print("budget: limit {d} MiB, RAM {d} MiB, flag margin {d} MiB, unplanned {d} MiB: floor {d} MiB, but the default's limits plan {d} MiB ({s} {d} MiB less margin {d} MiB) and the default is {d} MiB\n", .{ m.wired_mb.?, m.total.? / MiB, c.margin / MiB, c.unplanned / MiB, floor / MiB, by_ceiling / MiB, limits.ceiling_name, limits.ceiling / MiB, limits.margin / MiB, d.bytes / MiB });
        }
    }
    try testing.expectEqual(@as(usize, 0), bad);
}

// The commit's contract ("a margin at or past the declared limit refused at boot by name") and the brief's definition of
// the margin (the max of --wired-margin-gib and the GPU floor + 512 MB, `wiredUnplannedBytes`): a Ceiling whose floor is
// 0 is one the launch refuses, and one the launch admits has a floor above 0.
test "a Ceiling the unplanned margin leaves no floor is refused at boot, whatever made the margin" {
    const MiB: u64 = 1 << 20;
    const GiB: u64 = 1 << 30;
    const cases = [_]Machine{
        // The flag alone past the limit: refused (the commit's case).
        .{ .total = 32 * GiB, .wired_mb = 28_000, .margin = 28 * GiB, .avail = 20 * GiB, .gpu_floor = 0 },
        // A lowered limit (any value declares, owner ruling) under the GPU floor + 512 MB, the flag at its default.
        .{ .total = 32 * GiB, .wired_mb = 2_048, .margin = GiB, .avail = 20 * GiB, .gpu_floor = 1_600 * MiB },
        // A limit declared past the RAM, the flag at the RAM: under the limit, past what the floor clamps to.
        .{ .total = 32 * GiB, .wired_mb = 40_000, .margin = 32 * GiB, .avail = 20 * GiB, .gpu_floor = 0 },
    };
    var bad: usize = 0;
    for (cases) |m| {
        const was = m.pin();
        defer _ = was.pin();
        const c = bills.ceilingNow();
        var buf: [192]u8 = undefined;
        // As main.zig and `sushi kld` call it.
        const refused = bills.wiredMarginRefusal(&buf, c) != null;
        const runtime = c.runtime(0, 20 * GiB);
        if ((c.floor().? == 0) != refused) {
            bad += 1;
            std.debug.print("boot: limit {d} MiB, RAM {d} MiB, flag {d} MiB, unplanned {d} MiB: floor {d} MiB, runtime ceiling {d} MiB, refused at boot: {}\n", .{ m.wired_mb.?, m.total.? / MiB, c.margin / MiB, c.unplanned / MiB, c.floor().? / MiB, runtime / MiB, refused });
        }
    }
    try testing.expectEqual(@as(usize, 0), bad);
}

// h20-1: one FitReadings reading takes one margin, its floor and its unplanned term on the same value, while a write of
// the margin races it.
test "a margin written while a load reads takes one margin per reading" {
    const t = std.testing;
    // The saved margin moves under no launch flag.
    const was = (Machine{ .total = 128 << 30, .wired_mb = 120_000, .margin = null, .avail = 20 << 30, .gpu_floor = 0 }).pin();
    defer _ = was.pin();
    defer launch_state.setWiredMarginFile(null);
    var stop = std.atomic.Value(bool).init(false);
    const Saver = struct {
        fn run(s: *std.atomic.Value(bool)) void {
            var g: u32 = 1;
            while (!s.load(.monotonic)) {
                launch_state.setWiredMarginFile(g);
                g = if (g == 1) 16 else 1;
            }
        }
    };
    const th = try std.Thread.spawn(.{}, Saver.run, .{&stop});
    var torn: usize = 0;
    var i: usize = 0;
    while (i < 200_000) : (i += 1) {
        const r = bills.FitReadings.now();
        const c = r.ceiling;
        if (c.unplanned != bills.wiredUnplannedBytes(c.margin, c.gpu_floor) or c.floor().? != bills.wiredLimitFloor(c.declared, c.total_ram, bills.wiredUnplannedBytes(c.margin, c.gpu_floor)).?) torn += 1;
    }
    stop.store(true, .monotonic);
    th.join();
        try t.expectEqual(@as(usize, 0), torn);
}

// The same reader beside a model already loaded (pure readings, as the bills tests build them): a 128 GB Mac at
// iogpu.wired_limit_mb=120000, 60 GiB of a first model active, 60 GiB of RAM free. The declared ceiling leaves the second
// load its floor less the active buffers (engine-memory-admission's owner rule: "the lesser of the limit's floor, less
// --wired-margin-gib or the GPU floor and the resident buffers, and the free RAM").
test "beside a loaded model an MLX load's preflight reads at most the declared floor less the active buffers" {
    const MiB: u64 = 1 << 20;
    const GiB: u64 = 1 << 30;
    const limit: u64 = 120_000 * MiB;
    const floor = bills.wiredLimitFloor(limit, 128 * GiB, GiB).?;
    const r = bills.FitReadings{ .ceiling = .{ .declared = limit, .working_set = limit, .total_ram = 128 * GiB, .margin = GiB, .unplanned = GiB }, .active = 60 * GiB, .free = 60 * GiB };
    const got = r.preflightAvailable().bytes.?;
    errdefer std.debug.print("beside: floor {d} MiB less active {d} MiB = {d} MiB; preflight {d} MiB\n", .{ floor / MiB, r.active / MiB, (floor - r.active) / MiB, got / MiB });
    try testing.expect(got <= floor - r.active);
}

// The growth a request commits covers the cache the engine reserves at the width it runs at, which the scheduler re-asks
// after the eviction pass. Readings: no declared limit, a 64 GB Mac; admission reads `before` free, the eviction pass
// returns 8 GiB of hot cache and the re-ask reads that much more.
test "a Qwen request's growth commit covers the cache it reserves at the width the post-eviction re-ask runs it at" {

    try test_note.needArchs(&.{"qwen4"});
    const cfg = fixtures.qwen4ExpOomConfig();
    const MiB: u64 = 1 << 20;
    const saved_static = bills.static_ceiling_override;
    defer bills.static_ceiling_override = saved_static;
    bills.static_ceiling_override = 48 << 30;
    const kv8 = trunk.KVQuantConfig.affine(8);
    const seq: usize = 40_000;
    var widened: usize = 0;
    var bad: usize = 0;
    var before: u64 = 2 << 30;
    while (before <= 24 << 30) : (before += 1 << 30) {
        const at_admit = (Machine{ .total = 64 << 30, .wired_mb = 0, .margin = 1 << 30, .avail = before, .gpu_floor = 0 }).pin();
        const nums = bills.prefillBillNumbersNow(&cfg, seq, 4096, kv8, false, false);
        const admitted = bills.requestPrefillChunkNow(&cfg, seq, 4096, kv8, false, 0, 0, false, false);
        _ = at_admit.pin();
        const after_pin = (Machine{ .total = 64 << 30, .wired_mb = 0, .margin = 1 << 30, .avail = before + (8 << 30), .gpu_floor = 0 }).pin();
        const reasked = bills.requestPrefillChunkNow(&cfg, seq, 4096, kv8, false, 0, 0, false, false);
        _ = after_pin.pin();
        if (reasked <= admitted) continue;
        widened += 1;
        const reserved = bills.cacheCommitBytes(&cfg, seq, 4096, 8, reasked, false);
        const commit = slot_admission.growthCommitAt(&cfg, seq, 4096, 8, nums.commit, slot_admission.postEvictionPrefillChunk(admitted, reasked), .{});
        if (commit < reserved) {
            bad += 1;
            std.debug.print("{d} MiB free at admission: commit {d} MiB at width {d}; the re-ask after 8 GiB evicted runs {d} rows and reserves {d} MiB (+{d} MiB uncommitted)\n", .{ before / MiB, commit / MiB, admitted, reasked, reserved / MiB, (reserved - commit) / MiB });
        }
    }
    errdefer std.debug.print("{d} readings widened, {d} under-committed\n", .{ widened, bad });
    try testing.expect(widened > 0);
    try testing.expectEqual(@as(usize, 0), bad);
}
