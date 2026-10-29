const std = @import("std");
const transformer_mod = @import("../transformer.zig");
const qwen4_qsa_gather = @import("qwen4_qsa_gather.zig");
const qwen4_qsa_attn = @import("qwen4_qsa_attn.zig");
const model_mod = @import("../model.zig");
const model_registry_mod = @import("../model_registry.zig");
const Transformer = transformer_mod.Transformer;
const KVCache = transformer_mod.KVCache;
const ModelConfig = model_mod.ModelConfig;
const expert_stream_mod = @import("../expert_stream.zig");
const LoadedModel = model_registry_mod.LoadedModel;
const slot_admission = @import("../slot_admission.zig");
const MAX_BATCH_GROUP = slot_admission.MAX_BATCH_GROUP;
const configBatchesDecode = slot_admission.configBatchesDecode;
const batchGroupCap = slot_admission.batchGroupCap;
const batchedKvKeepCount = slot_admission.batchedKvKeepCount;
const groupKeepCount = slot_admission.groupKeepCount;
const groupCtxMax = slot_admission.groupCtxMax;
const batchKvLenOf = slot_admission.batchKvLenOf;
const batchKvLenOfWith = slot_admission.batchKvLenOfWith;
const fillGroupPadWasteKvLens = slot_admission.fillGroupPadWasteKvLens;
const AdmitCand = slot_admission.AdmitCand;
const admitPendingTick = slot_admission.admitPendingTick;
const MemoryBill = slot_admission.MemoryBill;
const memoryAdmitCount = slot_admission.memoryAdmitCount;
const model_load = @import("../model_load.zig");
const applyModelSettings = model_load.applyModelSettings;
const resolveSsdBudget = model_load.resolveSsdBudget;
const LoadDrafterDir = model_load.LoadDrafterDir;
const testing = std.testing;
const scheduler_mod = @import("../scheduler.zig");
const Scheduler = scheduler_mod.Scheduler;
const Slot = scheduler_mod.Slot;

test "LoadDrafterDir: Flash-Next loads a drafter only when --drafter names it, every other model also finds the pack's own" {
    const a = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &path_buf);
    const model_dir = path_buf[0..root_len];
    try tmp.dir.createDirPath(io, "dflash2");
    var beside = try tmp.dir.openDir(io, "dflash2", .{});
    defer beside.close(io);
    try beside.writeFile(io, .{ .sub_path = "config.json", .data = "{\"architectures\":[\"DFlash2DraftModel\"],\"dflash_config\":{\"block_size\":8,\"mask_token_id\":7,\"target_layer_ids\":[1,3]}}" });
    const qwen = ModelConfig{ .model_type = "qwen4_exp" };
    const other = ModelConfig{ .model_type = "not_a_served_arch" };
    const beside_qwen = LoadDrafterDir.resolve(io, a, &qwen, false, "", model_dir);
    defer beside_qwen.deinit(a);
    try testing.expectEqualStrings("", beside_qwen.dir);
    const beside_other = LoadDrafterDir.resolve(io, a, &other, false, "", model_dir);
    defer beside_other.deinit(a);
    try testing.expect(std.mem.endsWith(u8, beside_other.dir, "/dflash2"));
    const named = LoadDrafterDir.resolve(io, a, &qwen, false, "/elsewhere/drafter", model_dir);
    try testing.expectEqualStrings("/elsewhere/drafter", named.dir);
    const refused = LoadDrafterDir.resolve(io, a, &qwen, true, "/elsewhere/drafter", model_dir);
    try testing.expectEqualStrings("", refused.dir);
}

test "a non-qwen4 checkpoint never engages streaming: the flag is dropped, the per-model setting warns" {
    const t = testing;
    const GiB: u64 = 1 << 30;
    var moe = model_mod.ModelConfig{
        .model_type = "not_a_served_arch",
        .num_hidden_layers = 48,
        .num_experts = 512,
        .num_experts_per_tok = 10,
        .hidden_size = 2560,
        .moe_intermediate_size = 640,
    };
    applyModelSettings(&moe, .{ .ssd_budget_gb = 60 });
    try t.expectEqual(@as(u32, 60), moe.ssd_budget_gb_override);
    try t.expect(!moe.supportsExpertStreaming());

    const from_setting = resolveSsdBudget(0, moe.ssd_budget_gb_override, moe.supportsExpertStreaming());
    try t.expectEqual(@as(u64, 0), from_setting.bytes);
    try t.expect(from_setting.setting_ignored);
    const from_flag = resolveSsdBudget(60 * GiB, 0, moe.supportsExpertStreaming());
    try t.expectEqual(@as(u64, 0), from_flag.bytes);
    try t.expect(!from_flag.setting_ignored);

    try t.expect(!expert_stream_mod.expertStreamingEngaged(
        moe.supportsExpertStreaming(),
        moe.expertStreamingRequired(),
        from_flag.bytes,
    ));
    try t.expect(!moe.expert_streaming);

    var q4 = moe;
    q4.model_type = "qwen4_exp";
    try t.expectEqual(60 * GiB, resolveSsdBudget(60 * GiB, 0, q4.supportsExpertStreaming()).bytes);
    try t.expect(expert_stream_mod.expertStreamingEngaged(
        q4.supportsExpertStreaming(),
        q4.expertStreamingRequired(),
        resolveSsdBudget(0, q4.ssd_budget_gb_override, q4.supportsExpertStreaming()).bytes,
    ));
}

test "a resident qwen4_exp decodes rows of one forward without padding; a streamed one keeps the padded batch" {
    var cfg = std.mem.zeroes(model_mod.ModelConfig);
    cfg.model_type = "qwen4_exp";
    try testing.expect(cfg.supportsBatchedQwen4Rows());
    try testing.expectEqual(@as(usize, 8), batchGroupCap(&cfg));
    cfg.expert_streaming = true;
    try testing.expect(!cfg.supportsBatchedQwen4Rows());
    try testing.expectEqual(MAX_BATCH_GROUP, batchGroupCap(&cfg));
}

test "a 137k stream beside two fresh sub-agents stays one group on the gather bill" {
    const prev_b = qwen4_qsa_gather.qsa_batched_gather_override;
    const prev_g = qwen4_qsa_gather.qsa_gather_override;
    const prev_d = qwen4_qsa_gather.qsa_decode_gather_override;
    defer {
        qwen4_qsa_gather.qsa_batched_gather_override = prev_b;
        qwen4_qsa_gather.qsa_gather_override = prev_g;
        qwen4_qsa_gather.qsa_decode_gather_override = prev_d;
    }
    qwen4_qsa_gather.qsa_batched_gather_override = true;
    qwen4_qsa_gather.qsa_gather_override = true;
    qwen4_qsa_gather.qsa_decode_gather_override = true;
    var q4 = model_mod.ModelConfig{ .model_type = "qwen4_exp", .indexer_budget = 2048, .indexer_compress_ratio = 4 };
    const lens = [_]usize{ 10, 12, 137_000 };
    var caches: [3]KVCache = undefined;
    var built: usize = 0;
    defer for (caches[0..built]) |*c| c.deinit();
    var ptrs: [3]*const KVCache = undefined;
    for (lens, 0..) |len, i| {
        caches[i] = try KVCache.init(testing.allocator, 32);
        built += 1;
        caches[i].entries[3].initialized = true;
        caches[i].entries[3].offset = len;
        ptrs[i] = &caches[i];
    }
    var billed: [3]u32 = undefined;
    fillGroupPadWasteKvLens(&ptrs, &q4, 1, &billed);
    try testing.expectEqual(@as(u32, 2052), billed[2]);
    try testing.expectEqual(@as(usize, 2), batchedKvKeepCount(&billed));
    try testing.expectEqual(@as(u32, 137_000), groupCtxMax(&ptrs));
    try testing.expectEqual(@as(usize, 3), groupKeepCount(&billed, groupCtxMax(&ptrs)));
    // Below the context floor the serial forward is cheap and the plain cap splits the stream off.
    caches[2].entries[3].offset = 100_000;
    try testing.expectEqual(@as(usize, 2), groupKeepCount(&billed, groupCtxMax(&ptrs)));
}

test "batchKvLenOf bills qwen4 selected length when the gather arm is on" {
    const prev = qwen4_qsa_gather.qsa_batched_gather_override;
    defer qwen4_qsa_gather.qsa_batched_gather_override = prev;
    qwen4_qsa_gather.qsa_batched_gather_override = true;
    var q4 = model_mod.ModelConfig{
        .model_type = "qwen4_exp",
        .indexer_budget = 2048,
        .indexer_compress_ratio = 4,
    };
    var cache = try KVCache.init(testing.allocator, 32);
    defer cache.deinit();
    cache.entries[3].initialized = true;
    cache.entries[3].offset = 162_000;
    try testing.expectEqual(@as(u32, 2052), batchKvLenOf(&cache, &q4));
    qwen4_qsa_gather.qsa_batched_gather_override = false;
    try testing.expectEqual(@as(u32, 162_000), batchKvLenOf(&cache, &q4));
    var unserved = model_mod.ModelConfig{ .model_type = "not_a_served_arch" };
    cache.step = 162_000;
    cache.entries[0].initialized = true;
    cache.entries[0].offset = 162_000;
    qwen4_qsa_gather.qsa_batched_gather_override = true;
    try testing.expectEqual(@as(u32, 162_000), batchKvLenOf(&cache, &unserved));
}

test "batchKvLenOf bills raw when the gather switch for that width is off" {
    const prev_b = qwen4_qsa_gather.qsa_batched_gather_override;
    const prev_g = qwen4_qsa_gather.qsa_gather_override;
    const prev_d = qwen4_qsa_gather.qsa_decode_gather_override;
    const prev_v = qwen4_qsa_attn.qsa_verify_gather_override;
    defer {
        qwen4_qsa_gather.qsa_batched_gather_override = prev_b;
        qwen4_qsa_gather.qsa_gather_override = prev_g;
        qwen4_qsa_gather.qsa_decode_gather_override = prev_d;
        qwen4_qsa_attn.qsa_verify_gather_override = prev_v;
    }
    qwen4_qsa_gather.qsa_batched_gather_override = true;
    qwen4_qsa_gather.qsa_gather_override = true;
    qwen4_qsa_gather.qsa_decode_gather_override = true;
    qwen4_qsa_attn.qsa_verify_gather_override = true;
    var q4 = model_mod.ModelConfig{
        .model_type = "qwen4_exp",
        .indexer_budget = 2048,
        .indexer_compress_ratio = 4,
    };
    var cache = try KVCache.init(testing.allocator, 32);
    defer cache.deinit();
    cache.entries[3].initialized = true;
    cache.entries[3].offset = 162_000;
    try testing.expectEqual(@as(u32, 2052), batchKvLenOfWith(&cache, &q4, 1));
    try testing.expectEqual(@as(u32, 2052), batchKvLenOfWith(&cache, &q4, 4));
    qwen4_qsa_gather.qsa_batched_gather_override = false;
    try testing.expectEqual(@as(u32, 162_000), batchKvLenOfWith(&cache, &q4, 1));
    qwen4_qsa_gather.qsa_batched_gather_override = true;
    qwen4_qsa_gather.qsa_gather_override = false;
    try testing.expectEqual(@as(u32, 162_000), batchKvLenOfWith(&cache, &q4, 1));
    qwen4_qsa_gather.qsa_gather_override = true;
    qwen4_qsa_gather.qsa_decode_gather_override = false;
    try testing.expectEqual(@as(u32, 162_000), batchKvLenOfWith(&cache, &q4, 1));
    try testing.expectEqual(@as(u32, 2052), batchKvLenOfWith(&cache, &q4, 4));
    qwen4_qsa_gather.qsa_decode_gather_override = true;
    qwen4_qsa_attn.qsa_verify_gather_override = false;
    try testing.expectEqual(@as(u32, 2052), batchKvLenOfWith(&cache, &q4, 1));
    try testing.expectEqual(@as(u32, 162_000), batchKvLenOfWith(&cache, &q4, 4));
}

test "a 300k slot beside a 1k slot groups on the sparse bill" {
    const prev_b = qwen4_qsa_gather.qsa_batched_gather_override;
    const prev_g = qwen4_qsa_gather.qsa_gather_override;
    const prev_d = qwen4_qsa_gather.qsa_decode_gather_override;
    defer {
        qwen4_qsa_gather.qsa_batched_gather_override = prev_b;
        qwen4_qsa_gather.qsa_gather_override = prev_g;
        qwen4_qsa_gather.qsa_decode_gather_override = prev_d;
    }
    qwen4_qsa_gather.qsa_batched_gather_override = true;
    qwen4_qsa_gather.qsa_gather_override = true;
    qwen4_qsa_gather.qsa_decode_gather_override = true;
    var q4 = model_mod.ModelConfig{
        .model_type = "qwen4_exp",
        .indexer_budget = 2048,
        .indexer_compress_ratio = 4,
    };
    var caches: [2]KVCache = undefined;
    var built: usize = 0;
    defer for (caches[0..built]) |*c| c.deinit();
    caches[0] = try KVCache.init(testing.allocator, 32);
    built = 1;
    caches[0].entries[3].initialized = true;
    caches[0].entries[3].offset = 300_000;
    caches[1] = try KVCache.init(testing.allocator, 32);
    built = 2;
    caches[1].entries[3].initialized = true;
    caches[1].entries[3].offset = 1_000;
    const ptrs = [_]*const KVCache{ &caches[0], &caches[1] };
    var billed: [2]u32 = undefined;
    fillGroupPadWasteKvLens(&ptrs, &q4, 1, &billed);
    try testing.expectEqual(@as(u32, 2052), billed[0]);
    try testing.expectEqual(@as(u32, 1_000), billed[1]);
    var billed_asc = billed;
    std.mem.sort(u32, &billed_asc, {}, std.sort.asc(u32));
    try testing.expectEqual(@as(usize, 2), batchedKvKeepCount(&billed_asc));
}

test "S>=2 pad-waste floor is max of gather and verify mins" {
    const prev_b = qwen4_qsa_gather.qsa_batched_gather_override;
    const prev_g = qwen4_qsa_gather.qsa_gather_override;
    const prev_v = qwen4_qsa_attn.qsa_verify_gather_override;
    const prev_gm = qwen4_qsa_gather.qsa_gather_min_kv_override;
    const prev_vm = qwen4_qsa_attn.qsa_verify_gather_min_kv_override;
    defer {
        qwen4_qsa_gather.qsa_batched_gather_override = prev_b;
        qwen4_qsa_gather.qsa_gather_override = prev_g;
        qwen4_qsa_attn.qsa_verify_gather_override = prev_v;
        qwen4_qsa_gather.qsa_gather_min_kv_override = prev_gm;
        qwen4_qsa_attn.qsa_verify_gather_min_kv_override = prev_vm;
    }
    qwen4_qsa_gather.qsa_batched_gather_override = true;
    qwen4_qsa_gather.qsa_gather_override = true;
    qwen4_qsa_attn.qsa_verify_gather_override = true;
    qwen4_qsa_gather.qsa_gather_min_kv_override = 20_000;
    qwen4_qsa_attn.qsa_verify_gather_min_kv_override = 16_384;
    var q4 = model_mod.ModelConfig{
        .model_type = "qwen4_exp",
        .indexer_budget = 2048,
        .indexer_compress_ratio = 4,
    };
    var cache = try KVCache.init(testing.allocator, 32);
    defer cache.deinit();
    cache.entries[3].initialized = true;
    cache.entries[3].offset = 18_000;
    try testing.expectEqual(@as(u32, 18_000), batchKvLenOfWith(&cache, &q4, 4));
    cache.entries[3].offset = 162_000;
    try testing.expectEqual(@as(u32, 2052), batchKvLenOfWith(&cache, &q4, 4));
}

test "batchedEffectiveKvLen: qwen4 bills selected length, other archs keep raw kv" {
    var q4 = model_mod.ModelConfig{
        .model_type = "qwen4_exp",
        .indexer_budget = 2048,
        .indexer_compress_ratio = 4,
    };
    const min_kv: u32 = 8192;
    try testing.expectEqual(@as(u32, 2052), q4.batchedEffectiveKvLen(162_000, true, min_kv));
    try testing.expectEqual(@as(u32, 2052), q4.batchedEffectiveKvLen(64_000, true, min_kv));
    try testing.expectEqual(@as(u32, 8000), q4.batchedEffectiveKvLen(8000, true, min_kv));
    try testing.expectEqual(@as(u32, 162_000), q4.batchedEffectiveKvLen(162_000, false, min_kv));
    var unserved = model_mod.ModelConfig{ .model_type = "not_a_served_arch" };
    try testing.expectEqual(@as(u32, 162_000), unserved.batchedEffectiveKvLen(162_000, true, min_kv));

    const billed = [_]u32{
        q4.batchedEffectiveKvLen(16_384, true, min_kv),
        q4.batchedEffectiveKvLen(162_000, true, min_kv),
    };
    try testing.expectEqual(@as(usize, 2), batchedKvKeepCount(&billed));
    const raw_pair = [_]u32{ 16_384, 162_000 };
    try testing.expectEqual(@as(usize, 0), batchedKvKeepCount(&raw_pair));
    const other = [_]u32{ 1_000, 162_000 };
    try testing.expectEqual(@as(usize, 0), batchedKvKeepCount(&other));
}

test "a GDN trunk is batchable AND is not clamped by the server's concurrency gate" {
    // The two sites that decide "does this model batch?" must agree. They
    // disagreed once: the scheduler batched qwen4_exp while server.zig still
    // clamped --max-concurrent to 1 for anything with
    // full_attention_interval > 0, so asking for concurrency turned the
    // batched path OFF. Both now read ModelConfig.supportsBatchedGdnDecode.
    var cfg = std.mem.zeroes(model_mod.ModelConfig);
    cfg.model_type = "qwen4_exp";
    cfg.full_attention_interval = 4;
    try testing.expect(cfg.supportsBatchedGdnDecode());

    // The server's startup line, /props and /v1/models read this one predicate.
    try testing.expect(configBatchesDecode(&cfg));
}

test "supportsBatchedGdnDecode refuses every arch the batched GDN path does not model" {
    // A new arch on the shared moe forward must default to SERIAL, not ride
    // a kernel that never modelled its state.
    {
        var kda = std.mem.zeroes(model_mod.ModelConfig);
        kda.model_type = "glm5_next";
        kda.full_attention_interval = 4;
        kda.kda_vector_gate = true;
        try testing.expect(!kda.supportsBatchedGdnDecode());
    }
    {
        // Pure attention: not a GDN trunk at all, rides the standard kernel.
        var dense = std.mem.zeroes(model_mod.ModelConfig);
        dense.model_type = "not_a_served_arch";
        try testing.expect(!dense.supportsBatchedGdnDecode());
    }
    {
        // qwen4_exp: MoE, but every per-slot piece is on the SSMCacheEntry.
        var q4 = std.mem.zeroes(model_mod.ModelConfig);
        q4.model_type = "qwen4_exp";
        q4.full_attention_interval = 4;
        q4.num_experts = 256;
        q4.num_experts_per_tok = 8;
        try testing.expect(q4.supportsBatchedGdnDecode());
    }
}

test "admitPendingTick: per-slot exclusivity (qwen4 MTP slot) blocks only its own class" {
    const A: usize = 0xA0;
    var out: [16]usize = undefined;
    // An MTP candidate beside a live PLAIN slot on the same model admits.
    {
        const cands = [_]AdmitCand{.{ .model = A, .exclusive = true }};
        const active = [_]AdmitCand{.{ .model = A, .exclusive = false }};
        try testing.expectEqual(@as(usize, 1), admitPendingTick(&cands, &active, &out));
    }
    // An MTP candidate beside a live MTP slot holds.
    {
        const cands = [_]AdmitCand{.{ .model = A, .exclusive = true }};
        const active = [_]AdmitCand{.{ .model = A, .exclusive = true }};
        try testing.expectEqual(@as(usize, 0), admitPendingTick(&cands, &active, &out));
    }
    // A plain candidate beside a live MTP slot admits.
    {
        const cands = [_]AdmitCand{.{ .model = A, .exclusive = false }};
        const active = [_]AdmitCand{.{ .model = A, .exclusive = true }};
        try testing.expectEqual(@as(usize, 1), admitPendingTick(&cands, &active, &out));
    }
}

test "admission combines Qwen and MiMo reservations in either order" {
    const Probe = struct {
        fn numbers(cfg: *const ModelConfig, _: usize, _: u32, _: transformer_mod.KVQuantConfig, _: bool, _: bool) MemoryBill {
            return .{ .needed = if (cfg.longCtxGated()) 8 else 7, .available = 10 };
        }
    };
    const saved = scheduler_mod.prefill_admission_numbers;
    scheduler_mod.prefill_admission_numbers = Probe.numbers;
    defer scheduler_mod.prefill_admission_numbers = saved;
    var qwen_cfg = ModelConfig{ .model_type = "qwen4_exp" };
    var mimo_cfg = ModelConfig{ .model_type = "mimo_v2" };
    var xfm: Transformer = undefined;
    var qwen: LoadedModel = undefined;
    qwen.config = &qwen_cfg;
    qwen.transformer = &xfm;
    var mimo: LoadedModel = undefined;
    mimo.config = &mimo_cfg;
    mimo.transformer = &xfm;
    var a: Slot = undefined;
    a.model = &qwen;
    a.full_prompt = &.{};
    a.max_tokens = 1;
    a.cache.config = .dense;
    a.vision_embeddings = null;
    a.enable_mtp = false;
    a.memory_hold_logged = true;
    var b = a;
    b.model = &mimo;
    var sch: Scheduler = undefined;
    sch.pending = .empty;
    sch.decoding = .empty;
    defer sch.pending.deinit(testing.allocator);
    try sch.pending.appendSlice(testing.allocator, &.{ &a, &b });
    try testing.expectEqual(@as(usize, 1), memoryAdmitCount(&sch, &.{ 0, 1 }));
    try testing.expectEqual(@as(usize, 1), memoryAdmitCount(&sch, &.{ 1, 0 }));
    try testing.expectEqual(@as(usize, 1), memoryAdmitCount(&sch, &.{0}));
    try testing.expectEqual(@as(usize, 1), memoryAdmitCount(&sch, &.{1}));
}
