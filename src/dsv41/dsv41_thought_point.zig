const std = @import("std");
const mlx = @import("../mlx.zig");
const transformer_mod = @import("../transformer.zig");
const generate_mod = @import("../generate.zig");
const model_mod = @import("../model.zig");
const prefix_cache_mod = @import("../prefix_cache.zig");
const dsv41_prefix = @import("dsv41_prefix.zig");
const kv_disk_cache = @import("../kv_disk_cache.zig");
const model_registry_mod = @import("../model_registry.zig");
const model_settings = @import("../model_settings.zig");
const log = @import("../log.zig");
const Transformer = transformer_mod.Transformer;
const ModelConfig = model_mod.ModelConfig;
const Generator = generate_mod.Generator;
const mtp_plan = @import("../mtp_plan.zig");
const testing = std.testing;
const scheduler_mod = @import("../scheduler.zig");
const Scheduler = scheduler_mod.Scheduler;
const Slot = scheduler_mod.Slot;
const SpecTickMode = scheduler_mod.SpecTickMode;
const billedFillPeakBytes = scheduler_mod.billedFillPeakBytes;
const coldLoadVision = scheduler_mod.coldLoadVision;
const commitCancelledAtCleanup = scheduler_mod.commitCancelledAtCleanup;
const cullDecodingWith = scheduler_mod.cullDecodingWith;
const loadStagingBytes = scheduler_mod.loadStagingBytes;
const planExpertStreaming = scheduler_mod.planExpertStreaming;
const resolveExpertCache = scheduler_mod.resolveExpertCache;
const resolveStreamedVision = scheduler_mod.resolveStreamedVision;
const slotStateBytes = scheduler_mod.slotStateBytes;
const specTickMode = scheduler_mod.specTickMode;
const thinkBoundForcesPlain = scheduler_mod.thinkBoundForcesPlain;
const visionChoiceFor = scheduler_mod.visionChoiceFor;

test "a DeepSeek ledger carries the MLX pool and gives that many bytes up from the expert cache" {
    const GiB: u64 = 1 << 30;
    var cfg = model_mod.ModelConfig{ .model_type = "deepseek_v41", .num_hidden_layers = 40, .num_experts = 384, .num_experts_per_tok = 6 };
    const split = model_mod.ResidentSplit{ .trunk = 5_350_000_000, .mtp = 3_920_000_000 };
    const before = try resolveExpertCache(108 * GiB, &cfg, split, true, 8_100_000);
    cfg.expert_shape.pool_bytes = cfg.expertPoolBytes();
    const after = try resolveExpertCache(108 * GiB, &cfg, split, true, 8_100_000);
    try testing.expectEqual(cfg.expertPoolBytes(), after.ledger.?.pool_bytes);
    try testing.expectEqual(@as(u64, 0), before.ledger.?.pool_bytes);
    try testing.expect(after.cache_bytes < before.cache_bytes);
    const l = after.ledger.?;
    try testing.expect(l.trunk_bytes + l.mtp_bytes + l.workspace_bytes + l.selected_bytes + l.bounce_bytes + l.pool_bytes + l.cache_bytes <= 108 * GiB);
}

test "a default-on DeepSeek-V4.1 tower is billed in the ssd budget exactly as a --vision load bills it" {
    const t = testing;
    const saved = model_settings.vision_flag;
    defer model_settings.vision_flag = saved;
    model_settings.vision_flag = null;
    const cfg = model_mod.ModelConfig{
        .model_type = "deepseek_v41",
        .num_hidden_layers = 48,
        .num_experts = 512,
        .num_experts_per_tok = 10,
        .hidden_size = 2560,
        .moe_intermediate_size = 640,
        .has_vision = true,
    };
    const split = model_mod.ResidentSplit{ .trunk = 9_900_000_000, .mtp = 0, .vision = 970_000_000 };
    const GiB: u64 = 1 << 30;
    const by_default = try resolveStreamedVision(60 * GiB, &cfg, split, 9_830_400, coldLoadVision(&cfg, true));
    model_settings.vision_flag = true;
    const by_flag = try resolveStreamedVision(60 * GiB, &cfg, split, 9_830_400, coldLoadVision(&cfg, true));
    try t.expect(by_default.cost.on);
    try t.expectEqual(split.vision, by_default.resolved.ledger.?.vision_bytes);
    try t.expectEqual(by_flag.resolved.ledger.?, by_default.resolved.ledger.?);
    try t.expectEqual(by_flag.cost, by_default.cost);
}

test "visionChoiceFor: DeepSeek-V4.1 loads its tower by default streamed or resident; --no-vision and the per-model vision:false skip it; other archs keep the streamed default off" {
    const saved = model_settings.vision_flag;
    defer model_settings.vision_flag = saved;
    var ds = ModelConfig{ .has_vision = true, .model_type = "deepseek_v41" };
    var qwen = ModelConfig{ .has_vision = true, .model_type = "qwen4_exp" };
    model_settings.vision_flag = null;
    try std.testing.expect(visionChoiceFor(&ds, true).on);
    try std.testing.expectEqualStrings("default", visionChoiceFor(&ds, true).sourceName());
    try std.testing.expect(coldLoadVision(&ds, true));
    try std.testing.expect(visionChoiceFor(&ds, false).on);
    try std.testing.expect(!visionChoiceFor(&qwen, true).on);
    try std.testing.expect(visionChoiceFor(&qwen, false).on);
    model_settings.vision_flag = false;
    try std.testing.expect(!visionChoiceFor(&ds, true).on);
    try std.testing.expect(!coldLoadVision(&ds, true));
    model_settings.vision_flag = null;
    ds.vision_override = false;
    try std.testing.expect(!visionChoiceFor(&ds, true).on);
    try std.testing.expectEqualStrings("model-settings.json", visionChoiceFor(&ds, true).sourceName());
    model_settings.vision_flag = true;
    try std.testing.expect(visionChoiceFor(&ds, true).on);
    try std.testing.expectEqualStrings("--vision", visionChoiceFor(&ds, true).sourceName());
    qwen.vision_override = false;
    try std.testing.expect(visionChoiceFor(&qwen, true).on);
}

test "a streamed DeepSeek load bills its union workspace and MLX pool and no fill peak" {
    const fill: u64 = 9_911 << 20;
    const workspace: u64 = 3_525 << 20;
    const ds = ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true };
    const qwen = ModelConfig{ .model_type = "qwen4_exp", .expert_streaming = true };
    const mimo = ModelConfig{ .model_type = "mimo_v2", .expert_streaming = true };
    try testing.expectEqual(@as(u64, 0), billedFillPeakBytes(&ds, fill));
    try testing.expect(ds.expertPoolBytes() > 0);
    try testing.expectEqual(workspace + ds.expertPoolBytes(), loadStagingBytes(&ds, billedFillPeakBytes(&ds, fill), workspace));
    try testing.expectEqual(@as(u64, 0), loadStagingBytes(&qwen, billedFillPeakBytes(&qwen, fill), workspace));
    try testing.expectEqual(fill, loadStagingBytes(&mimo, billedFillPeakBytes(&mimo, fill), workspace));
}

/// DeepSeek's request state lives in the model, not in `slot.cache` (a one-byte-a-token row marker): its window
/// rings, compressed and index rows, DSpark windows, and the live restore points the request keeps.
pub fn dsv41StateBytes(s: *const Slot) u64 {
    const dsv = (s.model.transformer orelse return 0).dsv41 orelse return 0;
    var bytes: u64 = if (dsv.dec_state) |*st| @import("dsv41_model.zig").stateBytes(st) else 0;
    if (s.model.prefix_cache) |*hc| bytes += hc.live_dsv41.bytes();
    return bytes;
}

/// DeepSeek: a thinking request with tools is re-rendered with its thought, so the state right behind the generated
/// `</think>` is a restore point the next turn's prompt shares (docs/dsv41-arch.md#prefix-cache). Only a DSpark round
/// leaves the state on the emitted tokens, and only an SSD-first prefill has a tier to hand the point to.
pub fn armDsv41ThoughtPoint(slot: *Slot, gen: *Generator, tier_takes_points: bool) void {
    if (!tier_takes_points or !gen.dspark or !slot.has_tools or !slot.enable_thinking) return;
    if (slot.full_prompt.len < kv_disk_cache.MIN_PERSIST_TOKENS) return;
    const tok = slot.model.tokenizer orelse return;
    const open = tok.specialTokenId("<think>") orelse return;
    const close = tok.specialTokenId("</think>") orelse return;
    if (slot.full_prompt[slot.full_prompt.len - 1] != open) return;
    gen.dspark_round_ends_at = close;
}

/// The round that committed `</think>` ended right behind it: its state goes to the SSD tier as one restore point.
pub fn writeDsv41ThoughtPoint(slot: *Slot, gen: *Generator, block: []const u32) void {
    // The round's own finish committed and published the live record; a point written now would clear it.
    if (slot.finished or slot.error_code != null) return;
    const close = gen.dspark_round_ends_at orelse return;
    if (block.len == 0 or block[block.len - 1] != close) return;
    gen.dspark_round_ends_at = null;
    const generated = gen.generated_ids.items;
    const thought = generated.len - 1;
    const hc: *prefix_cache_mod.HotPrefixCache = if (slot.model.prefix_cache) |*p| p else return;
    const xfm = slot.model.transformer orelse return;
    const tok = slot.model.tokenizer orelse return;
    const dsv = xfm.dsv41 orelse return;
    if (thought < dsv41_prefix.min_thought_tokens) {
        log.debug("[dsv41-prefix] no decode restore point: a {d}-token thought is cheaper to prefill than to keep\n", .{thought});
        return;
    }
    const first = tok.decode(slot.allocator, generated[0..1]) catch return;
    defer slot.allocator.free(first);
    const last = tok.decode(slot.allocator, generated[thought - 1 .. thought]) catch return;
    defer slot.allocator.free(last);
    if (!dsv41_prefix.thoughtEdgesSurviveTrim(first, last)) {
        log.info("[dsv41-prefix] no decode restore point: the thought starts or ends in whitespace the next prompt trims\n", .{});
        return;
    }
    const pos = slot.full_prompt.len + generated.len;
    if (slot.cache.step != pos) return;
    const tokens = slot.allocator.alloc(u32, pos) catch return;
    defer slot.allocator.free(tokens);
    @memcpy(tokens[0..slot.full_prompt.len], slot.full_prompt);
    @memcpy(tokens[slot.full_prompt.len..], generated);
    const media = prefix_cache_mod.truncatedMediaKey(slot.media_chain, slot.vision_key, slot.media_start, pos);
    hc.writeDecodePointDsv41(&slot.cache, dsv, tokens, slot.has_tools, media, xfm.s) catch |err| {
        log.warn("[dsv41-prefix] decode restore point not written: {s}\n", .{@errorName(err)});
    };
}

test "a fired thinking bound keeps a DSpark slot in DSpark ticks, unless the close left it disabled" {
    var forced = [_]u32{7};
    var tb = mtp_plan.ThinkBound{ .budget = 4, .opener_id = null, .closer_id = 7, .forced = &forced, .in_think = true, .fired = true };
    var gen: Generator = undefined;
    gen.sampling = .{ .think_bound = &tb };
    gen.think_closed_by_loop = false;
    gen.mtp = null;
    gen.dspark = true;
    // What commitThoughtClose leaves behind after `dsparkResumeAfterForced` drained the commit.
    gen.spec_disabled_runtime = false;
    gen.spec_disable_reason = .none;
    try testing.expect(!thinkBoundForcesPlain(&gen));
    try testing.expectEqual(SpecTickMode.mtp, specTickMode(true, gen.mtp != null or gen.dspark, false, false));
    // A refused resume (a grammar, `--max-mtp-ctx`) disables the slot at the close: plain.
    gen.spec_disabled_runtime = true;
    gen.spec_disable_reason = .think_bound;
    try testing.expect(thinkBoundForcesPlain(&gen));
}

test "a DeepSeek STQ pack's ssd budget bills its tower as the requantized 8-bit codes" {
    const t = std.testing;
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try @import("../test_models.zig").packPath(&path_buf, "DeepSeek-V4.1-Flash-1017-Sushi-STQ-1.75bpw");
    var config = model_mod.parseConfig(t.io, t.allocator, path) catch return error.SkipZigTest;
    defer config.deinit(t.allocator);
    if (!config.has_vision) return error.SkipZigTest;
    const stored_tower: u64 = 970_536_960 + 40 * 1536;
    const report = std.c.getenv("SUSHI_VISION_Q8_REPORT") != null;
    for ([_]u64{ 96, 108 }) |gb| {
        const plan = try planExpertStreaming(t.io, t.allocator, &config, path, gb << 30, true);
        try t.expect(plan.vision.on);
        // 483M of the tower's 485M parameters at a byte and 4/128 byte each: the codes, scales and biases.
        try t.expect(plan.split.vision > 490_000_000 and plan.split.vision < 520_000_000);
        const l = plan.resolved.ledger.?;
        try t.expectEqual(plan.split.vision, l.vision_bytes);
        const per_slot = l.cache_bytes / l.slots_per_layer;
        const fixed = l.trunk_bytes + l.mtp_bytes + l.workspace_bytes + l.selected_bytes + l.bounce_bytes + l.pool_bytes;
        const slots_stored: u64 = (l.budget_bytes - fixed - stored_tower) / per_slot;
        try t.expect(plan.cache.slots_per_layer >= slots_stored);
        if (report) std.debug.print("[vision-q8] ssd budget {d}: tower {d} B (stored {d} B), slots/layer without {d}, with {d} (stored tower: {d}), per slot {d} B\n", .{ gb, plan.split.vision, stored_tower, plan.vision.slots_without, plan.cache.slots_per_layer, slots_stored, per_slot });
    }
}

fn dsv41CullScaffold(sch: *Scheduler, model: *model_registry_mod.LoadedModel, xfm: *Transformer, slot: *Slot) void {
    sch.io = testing.io;
    sch.allocator = testing.allocator;
    sch.queue_mu = .init;
    sch.metrics = null;
    sch.live_session_count = 0;
    sch.decoding = .empty;
    model.id = "org/dsv41-cull";
    model.prefix_cache = null;
    model.transformer = xfm;
    slot.model = model;
    slot.cache = .{ .entries = &.{}, .step = 0, .allocator = testing.allocator, .config = .dense };
    slot.ssm_entries = null;
    slot.ring_cps = .{};
    slot.restored_entry = 0;
    slot.full_prompt = &.{};
    slot.prompt_tokens = 4;
    slot.completion_tokens = 0;
    slot.cached_tokens = 0;
    slot.request_id = 0;
    slot.max_tokens = 8;
    slot.legacy_gen = null;
    slot.glm5_request = null;
    slot.finished = false;
    slot.error_code = null;
    slot.cancelled = .init(false);
    slot.committed_at_cull = false;
}

test "a cancelled DeepSeek slot commits when it is culled and not again at the cleanup drain" {
    const Hook = struct {
        var commits: usize = 0;
        fn commit(_: *Scheduler, _: *Slot) void {
            commits += 1;
        }
    };
    var sch: Scheduler = undefined;
    var model: model_registry_mod.LoadedModel = undefined;
    var xfm: Transformer = undefined;
    xfm.dsv41 = @ptrFromInt(@alignOf(@import("dsv41_model.zig").Model));
    var slot: Slot = undefined;
    dsv41CullScaffold(&sch, &model, &xfm, &slot);
    defer sch.decoding.deinit(testing.allocator);
    try sch.decoding.append(testing.allocator, &slot);
    Hook.commits = 0;
    slot.cancelled.store(true, .release);

    // The slot leaves `decoding` here, and the next exclusive request is admitted over the model's state at once:
    // its live record has to be published before that, not at the drain behind `complete()`.
    cullDecodingWith(&sch, Hook.commit);
    try testing.expectEqual(@as(usize, 0), sch.decoding.items.len);
    try testing.expectEqual(@as(usize, 1), Hook.commits);
    commitCancelledAtCleanup(&sch, &slot, Hook.commit);
    try testing.expectEqual(@as(usize, 1), Hook.commits);
}

test "a finished DeepSeek slot is not committed again by the cull" {
    const Hook = struct {
        var commits: usize = 0;
        fn commit(_: *Scheduler, _: *Slot) void {
            commits += 1;
        }
    };
    var sch: Scheduler = undefined;
    var model: model_registry_mod.LoadedModel = undefined;
    var xfm: Transformer = undefined;
    xfm.dsv41 = @ptrFromInt(@alignOf(@import("dsv41_model.zig").Model));
    var slot: Slot = undefined;
    dsv41CullScaffold(&sch, &model, &xfm, &slot);
    defer sch.decoding.deinit(testing.allocator);
    try sch.decoding.append(testing.allocator, &slot);
    Hook.commits = 0;
    slot.finished = true;
    slot.cancelled.store(true, .release);
    cullDecodingWith(&sch, Hook.commit);
    try testing.expectEqual(@as(usize, 0), Hook.commits);
}

test "a DSpark round that finished the slot leaves the thought point to the commit" {
    var slot: Slot = undefined;
    var model: model_registry_mod.LoadedModel = undefined;
    model.prefix_cache = null;
    slot.model = &model;
    slot.error_code = null;
    var gen: Generator = undefined;
    gen.generated_ids = .empty;
    defer gen.generated_ids.deinit(testing.allocator);
    try gen.generated_ids.appendSlice(testing.allocator, &.{ 5, 6, 7, 9 });
    const block = [_]u32{ 7, 9 };

    // The round's finish already committed and published the live record, which a point written now would clear.
    slot.finished = true;
    gen.dspark_round_ends_at = 9;
    writeDsv41ThoughtPoint(&slot, &gen, &block);
    try testing.expectEqual(@as(?u32, 9), gen.dspark_round_ends_at);

    slot.finished = false;
    writeDsv41ThoughtPoint(&slot, &gen, &block);
    try testing.expectEqual(@as(?u32, null), gen.dspark_round_ends_at);
}

test "a DeepSeek slot's state bytes hold the model's rows and windows and the live points (DSV41_TINY)" {
    try mlx.requireGpu();
    const dsv41_model = @import("dsv41_model.zig");
    const a = testing.allocator;
    const dir = try dsv41_model.fixtureDir();
    var fx = try dsv41_model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    var sc = @import("dsv41_scope.zig").Scope.init(a, mlx.gpuStream());
    defer sc.deinit();
    const ids = try dsv41_model.tokens(&sc, &fx);
    defer a.free(ids);
    const m = try dsv41_model.tinyModel(dir, .{ .act = .bfloat16, .dspark = true }, mlx.gpuStream());
    defer m.deinit();
    _ = mlx.mlx_array_free(try dsv41_model.extend(m, ids[0..40], .last));

    var xfm: Transformer = undefined;
    xfm.dsv41 = m;
    var model: model_registry_mod.LoadedModel = undefined;
    model.prefix_cache = prefix_cache_mod.HotPrefixCache.init(a, 2);
    defer model.prefix_cache.?.deinit();
    model.transformer = &xfm;
    const hc = &model.prefix_cache.?;
    var cp = try dsv41_prefix.capture(a, m, 40);
    defer cp.deinit(a);
    try hc.live_dsv41.keep(a, m, &cp);
    var slot: Slot = undefined;
    slot.model = &model;
    slot.cache = .{ .entries = &.{}, .step = 0, .allocator = a, .config = .dense };
    slot.ssm_entries = null;
    slot.ring_cps = .{};
    slot.glm5_request = null;
    slot.legacy_gen = null;

    const rows_and_windows = dsv41_model.stateBytes(&m.dec_state.?);
    try testing.expect(rows_and_windows > 0);
    try testing.expect(hc.live_dsv41.bytes() > 0);
    try testing.expectEqual(rows_and_windows + hc.live_dsv41.bytes(), slotStateBytes(&slot));
}
