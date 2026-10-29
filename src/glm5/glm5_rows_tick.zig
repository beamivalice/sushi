const std = @import("std");
const mlx = @import("../mlx.zig");
const transformer_mod = @import("../transformer.zig");
const tokenizer_mod = @import("../tokenizer.zig");
const generate_mod = @import("../generate.zig");
const model_mod = @import("../model.zig");
const prefix_cache_mod = @import("../prefix_cache.zig");
const glm5_prefix = @import("glm5_prefix.zig");
const metrics_mod = @import("../metrics.zig");
const model_registry_mod = @import("../model_registry.zig");
const log = @import("../log.zig");
const io_util = @import("../io_util.zig");
const Transformer = transformer_mod.Transformer;
const ModelConfig = model_mod.ModelConfig;
const Tokenizer = tokenizer_mod.Tokenizer;
const SamplingParams = sampler.SamplingParams;
const dflash_mod = @import("../dflash.zig");
const glm5_forward_mod = @import("glm5_forward.zig");
const DflashModel = dflash_mod.DflashModel;
const LoadedModel = model_registry_mod.LoadedModel;
const mtp_plan = @import("../mtp_plan.zig");
const sampler = @import("../sampler.zig");
const testing = std.testing;
const scheduler_mod = @import("../scheduler.zig");
const Scheduler = scheduler_mod.Scheduler;
const Slot = scheduler_mod.Slot;
const batchGroupCap = scheduler_mod.batchGroupCap;
const configBatchesDecode = scheduler_mod.configBatchesDecode;
const cullDecoding = scheduler_mod.cullDecoding;
const cullDecodingWith = scheduler_mod.cullDecodingWith;
const expectError = scheduler_mod.expectError;
const expectOutcomes = scheduler_mod.expectOutcomes;
const glmStreamedDrafterOff = scheduler_mod.glmStreamedDrafterOff;
const postPrefillTerminal = scheduler_mod.postPrefillTerminal;
const publishSpeculativeBlock = scheduler_mod.publishSpeculativeBlock;
const recordSlotCleanup = scheduler_mod.recordSlotCleanup;
const specTickMode = scheduler_mod.specTickMode;

test "a streamed GLM load leaves a found assistant unloaded and refuses an asked-for one" {
    var glm = ModelConfig{ .model_type = "glm5_next" };
    try testing.expect(!try glmStreamedDrafterOff(&glm, "/assistant"));
    glm.expert_streaming = true;
    try testing.expect(try glmStreamedDrafterOff(&glm, ""));
    try testing.expectError(error.GlmStreamingSpecUnsupported, glmStreamedDrafterOff(&glm, "/assistant"));
    const qwen = ModelConfig{ .model_type = "qwen4_exp", .expert_streaming = true };
    try testing.expect(!try glmStreamedDrafterOff(&qwen, ""));
}

/// GLM: the request's MLA rows through the newest checkpoint a tier keeps serve every KDA checkpoint
/// at or below it; without one nothing restores. The RAM tier copies the rows its budget keeps; the
/// SSD tier shares every row through the newest checkpoint, read by the flush after the response.
/// Ownership of `cps` passes to the cache.
pub fn commitGlmSlot(hc: *prefix_cache_mod.HotPrefixCache, slot: *Slot, request: *const glm5_forward_mod.Request, tokens: []const u32, cps_opt: ?[]transformer_mod.SSMCheckpoint, dflash: ?prefix_cache_mod.DflashCommit, prompt_len: usize) void {
    const cps_all = cps_opt orelse return;
    const s = slot.model.transformer.?.s;
    // The rows are chosen before the copy: only what the RAM tier keeps, never a full copy followed by a trim.
    const window: u64 = if (dflash) |d| prefix_cache_mod.HotPrefixCache.liveCacheBytes(d.cache) else 0;
    const row_bytes = glm5_prefix.rowBytesOf(request);
    const ram_len = hc.glmCommitLen(cps_all, request.offset, row_bytes, window);
    const disk_len = hc.glmDiskLen(cps_all, request.offset, row_bytes, window);
    const len = @max(ram_len orelse 0, disk_len orelse 0);
    if (len == 0) {
        log.debug("[hot-cache] GLM commit keeps no rows within either tier's budget; not committed\n", .{});
        for (cps_all) |*cp| cp.deinit(hc.allocator);
        hc.allocator.free(cps_all);
        return;
    }
    const cps = glm5_prefix.keepThrough(hc.allocator, cps_all, len);
    const ram_rows: ?glm5_prefix.MlaRows = if (ram_len) |n| glm5_prefix.MlaRows.capture(hc.allocator, request, n, s) catch |err| blk: {
        log.warn("[hot-cache] GLM MLA rows not captured: {s}; the RAM tier keeps nothing\n", .{@errorName(err)});
        break :blk null;
    } else null;
    const disk_rows: ?glm5_prefix.MlaRows = if (disk_len) |n| glm5_prefix.MlaRows.shareLive(hc.allocator, request, n, s) catch |err| blk: {
        log.warn("[disk-cache] GLM MLA rows not shared: {s}; the SSD tier gets the RAM tier's rows\n", .{@errorName(err)});
        break :blk null;
    } else null;
    if (ram_rows == null and disk_rows == null) {
        for (cps) |*cp| cp.deinit(hc.allocator);
        hc.allocator.free(cps);
        return;
    }
    const n_cps = cps.len;
    const st = hc.commitGlmTiers(&slot.cache, tokens, slot.has_tools, slot.vision_key, slot.cache_key, slot.media_start, cps, ram_rows, disk_rows, dflash, prompt_len) catch |err| {
        log.warn("[hot-cache] commit failed: {s}\n", .{@errorName(err)});
        return;
    };
    switch (st) {
        .ok => |n| log.info("[hot-cache] committed {d}/{d} GLM tokens ({d} KDA checkpoints)\n", .{ n, tokens.len, n_cps }),
        .disk_only => log.info("[disk-cache] captured {d}/{d} GLM tokens for the SSD tier ({d} KDA checkpoints)\n", .{ disk_len orelse 0, tokens.len, n_cps }),
        else => {},
    }
}

/// DIAGNOSTIC (SUSHI_GLM_ROWS_UBENCH=N): `glm5_rows_ubench.run` per `_CTX=<tokens>[,<tokens>...]`
/// (default 1024) over `_TEXT=<abs path>`.
pub fn glmRowsUbench(alloc: std.mem.Allocator, target: *@import("glm5_forward.zig").Model, latent_bits: u8, tok: *Tokenizer) void {
    const tio = std.Io.Threaded.global_single_threaded.io();
    const rounds = @max(1, std.fmt.parseInt(usize, std.mem.sliceTo(std.c.getenv("SUSHI_GLM_ROWS_UBENCH").?, 0), 10) catch 8);
    const ctxs: []const u8 = if (std.c.getenv("SUSHI_GLM_ROWS_UBENCH_CTX")) |r| std.mem.sliceTo(r, 0) else "1024";
    var ids: []u32 = &.{};
    defer alloc.free(ids);
    if (std.c.getenv("SUSHI_GLM_ROWS_UBENCH_TEXT")) |path| blk: {
        const text = std.Io.Dir.cwd().readFileAlloc(tio, std.mem.sliceTo(path, 0), alloc, .limited(64 << 20)) catch break :blk;
        defer alloc.free(text);
        ids = tok.encode(alloc, text) catch break :blk;
    }
    var it = std.mem.tokenizeScalar(u8, ctxs, ',');
    while (it.next()) |field| {
        const ctx = std.fmt.parseInt(usize, field, 10) catch continue;
        @import("glm5_rows_ubench.zig").run(alloc, target, latent_bits, ids, ctx, rounds) catch |err| log.warn("[glm-rows-ubench] failed: {s}\n", .{@errorName(err)});
    }
}

/// DIAGNOSTIC (SUSHI_GLM_ROUND_UBENCH=N): `glm5_round_ubench.run` per `_CTX=<tokens>[,<tokens>...]`
/// (default 8192) over `_TEXT=<abs path>`.
pub fn glmRoundUbench(alloc: std.mem.Allocator, target: *@import("glm5_forward.zig").Model, assistant: *DflashModel, latent_bits: u8, tok: *Tokenizer) void {
    const tio = std.Io.Threaded.global_single_threaded.io();
    const rounds = @max(1, std.fmt.parseInt(usize, std.mem.sliceTo(std.c.getenv("SUSHI_GLM_ROUND_UBENCH").?, 0), 10) catch 64);
    const ctxs: []const u8 = if (std.c.getenv("SUSHI_GLM_ROUND_UBENCH_CTX")) |r| std.mem.sliceTo(r, 0) else "8192";
    var ids: []u32 = &.{};
    defer alloc.free(ids);
    if (std.c.getenv("SUSHI_GLM_ROUND_UBENCH_TEXT")) |path| blk: {
        const text = std.Io.Dir.cwd().readFileAlloc(tio, std.mem.sliceTo(path, 0), alloc, .limited(64 << 20)) catch break :blk;
        defer alloc.free(text);
        ids = tok.encode(alloc, text) catch break :blk;
    }
    var it = std.mem.tokenizeScalar(u8, ctxs, ',');
    while (it.next()) |field| {
        const ctx = std.fmt.parseInt(usize, field, 10) catch continue;
        @import("glm5_round_ubench.zig").run(alloc, target, assistant, latent_bits, ids, ctx, rounds) catch |err| log.warn("[glm-round-ubench] failed: {s}\n", .{@errorName(err)});
    }
}

pub fn glmNativeDflashArmed(
    is_glm5: bool,
    enable_drafter: bool,
    has_dflash: bool,
    sampling: SamplingParams,
    logprobs_n: u32,
) bool {
    return is_glm5 and enable_drafter and has_dflash and generate_mod.glmDflashEligible(sampling, logprobs_n);
}

/// Step costs of a grouped GLM forward in ms (rows ubench, glm5-arch#concurrency): fixed, per
/// row, and one assistant draft.
pub const GlmRowCost = struct { fixed: f32 = 25, row: f32 = 14.5, draft: f32 = 5 };

/// Rows one grouped GLM tick carries: the verify kernels' four-row tile.
const glm_tick_rows = 4;

/// Draft nodes (0, 1 or 2) for the best drafter among `slots` single rows under the four-row cap:
/// a tree pays while its expected accepted tokens, priced at the group's ms per token, beat its
/// rows and draft. `rate` is that request's accepted share of drafts (`Generator.glm_draft_rate`).
pub fn glmTreeNodes(slots: usize, rate: f32, cost: GlmRowCost) usize {
    if (slots == 0) return 0;
    const n: f32 = @floatFromInt(slots);
    const per_token = (cost.fixed + n * cost.row) / n;
    var best: usize = 0;
    var best_gain: f32 = 0;
    var accepted: f32 = 0;
    var power: f32 = 1;
    for (1..@min(2, glm_tick_rows -| slots) + 1) |k| {
        power *= rate;
        accepted += power;
        const gain = accepted * per_token - (@as(f32, @floatFromInt(k)) * cost.row + cost.draft);
        if (gain > best_gain) {
            best = k;
            best_gain = gain;
        }
    }
    return best;
}

/// Does another slot of this one's model decode this tick (a drafting GLM slot or a plain batchable one)?
pub fn glmRowsInCompany(slot: *const Slot, drafting: []const *Slot, plain: []const *Slot) bool {
    var company: usize = 0;
    for (drafting) |other| company += @intFromBool(other.model == slot.model);
    for (plain) |other| company += @intFromBool(other.model == slot.model);
    return company >= 2;
}

/// A DFlash2 GLM slot that can ride a plain grouped tick: its rounds only need the row committed
/// with the assistant's taps (`Generator.glmRowCommit`).
pub fn slotGlmRowable(slot: *const Slot) bool {
    const gen = if (slot.legacy_gen) |*g| g else return false;
    if (!gen.glm_dflash_native or gen.dflash_ctx == null or gen.done) return false;
    if (slot.sampling.constraint != null or slot.logprobs_n > 0 or sampler.penaltyActive(slot.sampling)) return false;
    const t = slot.model.transformer orelse return false;
    return t.supportsBatchedGlmRows();
}

pub fn slotMtpGroupable(slot: *const Slot) bool {
    const gen = if (slot.legacy_gen) |*g| g else return false;
    if (!slot.enable_mtp or gen.mtp == null or gen.mtp_cache == null or !gen.has_last_hidden) return false;
    if (gen.mtp.?.moduleOwned()) return false;
    if (gen.spec_disabled_runtime or gen.mtp_serial_left > 0 or gen.mtp_serial_exit != .none) return false;
    if (gen.ctx.ssm_entries == null) return false;
    if (slot.sampling.constraint != null or slot.logprobs_n > 0) return false;
    const t = slot.model.transformer orelse return false;
    if (!t.supportsBatchedGdnDecode()) return false;
    return specTickMode(slot.enable_mtp, true, slot.enable_drafter, gen.dflash != null) == .mtp;
}

var glm_rows_logged = false;

var glm_tree_logged = false;

/// A batched GLM tick: one group per slot through `verifyGroups`, at most four rows. Every slot
/// decodes one plain row, committed by its own generator, unless `glmTreeNodes` hands the spare
/// rows to the best drafter's DFlash2 tree: that slot's round is published here and it leaves
/// `batch`. Returns the plain rows' logits in the order of the shortened `batch`.
pub fn glmRowsForward(sch: *Scheduler, allocator: std.mem.Allocator, xfm: *Transformer, batch: *[]*Slot, tokens: []const u32) ![]mlx.mlx_array {
    const verifier = @import("glm5_dflash_model.zig");
    const target = xfm.glm5.?;
    target.s = xfm.s;
    target.suppress_mask = xfm.suppress_mask;
    const schedule = verifier.bindDefaultSchedule();
    defer schedule.restore();
    var timer = io_util.Stopwatch.init(sch.io);
    // The drafter with the best landing rate; its tree takes the rows the plain slots leave.
    var tree_at: ?usize = null;
    var proposal: @import("glm5_dflash.zig").Proposal = undefined;
    for (batch.*, 0..) |slot, i| {
        const gen = &slot.legacy_gen.?;
        if (!gen.glm_dflash_native or gen.dflash_ctx == null) continue;
        if (tree_at) |best| if (batch.*[best].legacy_gen.?.glm_draft_rate >= gen.glm_draft_rate) continue;
        tree_at = i;
    }
    if (tree_at) |at| tree: {
        const gen = &batch.*[at].legacy_gen.?;
        const nodes = glmTreeNodes(batch.len, gen.glm_draft_rate, .{});
        if (nodes == 0 or gen.done or gen.completion_tokens + 2 > gen.max_tokens) {
            tree_at = null;
            break :tree;
        }
        gen.dflash.?.s = xfm.s;
        proposal = try gen.glmPropose(nodes);
    }
    const draft_ns = timer.read();
    timer.reset();
    var groups: [verifier.max_rows]verifier.Group = undefined;
    // A DFlash2 row keeps its assistant context; every row then carries the assistant's taps.
    var taps: []const u32 = &.{};
    for (batch.*, 0..) |slot, i| {
        const gen = &slot.legacy_gen.?;
        if (gen.glm_dflash_native) taps = gen.dflash.?.config.target_layer_ids;
        const request = gen.ctx.glm5_request orelse return error.GlmRequestMissing;
        // Plain rows take the policy when their slot samples them (`thinkShifted`).
        groups[i] = if (tree_at == i)
            .{ .request = request, .tokens = proposal.tokens[0..proposal.count], .parents = proposal.parents[0..proposal.count], .penalty = gen.glmPenalty() }
        else
            .{ .request = request, .tokens = tokens[i .. i + 1], .parents = &.{-1} };
    }
    if (!glm_rows_logged) {
        glm_rows_logged = true;
        log.info("[glm-rows] grouped plain tick engaged: {d} rows\n", .{batch.len});
    }
    if (tree_at != null and !glm_tree_logged) {
        glm_tree_logged = true;
        log.info("[glm-rows] a {d}-row draft tree joined {d} plain rows\n", .{ proposal.count, batch.len - 1 });
    }
    var rows = GlmRows{ .sch = sch, .target = target, .taps = taps, .proposal = &proposal, .draft_ns = draft_ns, .timer = timer };
    return glmRowsSettle(&rows, allocator, batch, groups[0..batch.len], tree_at);
}

/// The grouped tick's production side of `glmRowsSettle`.
const GlmRows = struct {
    const verifier = @import("glm5_dflash_model.zig");
    sch: *Scheduler,
    target: *const @import("glm5_forward.zig").Model,
    taps: []const u32,
    proposal: *const @import("glm5_dflash.zig").Proposal,
    draft_ns: u64,
    timer: io_util.Stopwatch,
    verify_ns: u64 = 0,

    fn verify(self: *GlmRows, groups: []const verifier.Group, out: []verifier.Verified) !void {
        try verifier.verifyGroups(self.target, groups, self.taps, .affine_rows_ffn, out);
        self.verify_ns = self.timer.read();
    }

    fn commitRow(_: *GlmRows, slot: *Slot, v: *const verifier.Verified) !mlx.mlx_array {
        return slot.legacy_gen.?.glmRowCommit(v);
    }

    fn endTree(self: *GlmRows, slot: *Slot, v: *verifier.Verified) void {
        const gen = &slot.legacy_gen.?;
        const result = gen.glmRoundEnd(slot.allocator, self.proposal, v, self.draft_ns, self.verify_ns, false) catch |err| {
            slot.markError(@errorName(err));
            return;
        };
        defer slot.allocator.free(result.tokens);
        publishSpeculativeBlock(self.sch, slot, gen, result.tokens);
    }
};

/// Verifies `groups` (one per `batch` slot), commits the plain rows and ends the tree slot's round
/// through `ops`; the plain slots stay in `batch`, in order, and their logits are returned.
fn glmRowsSettle(ops: anytype, allocator: std.mem.Allocator, batch: *[]*Slot, groups: []const GlmRows.verifier.Group, tree_at: ?usize) ![]mlx.mlx_array {
    const verifier = GlmRows.verifier;
    var out: [verifier.max_rows]verifier.Verified = undefined;
    // `batch` shrinks below; the verified rows are destroyed by the count that built them.
    const verified = out[0..batch.len];
    try ops.verify(groups, verified);
    defer for (verified) |*v| v.deinit();
    const logits = try allocator.alloc(mlx.mlx_array, batch.len - @intFromBool(tree_at != null));
    var made: usize = 0;
    errdefer {
        for (logits[0..made]) |a| _ = mlx.mlx_array_free(a);
        allocator.free(logits);
    }
    var kept: usize = 0;
    for (batch.*, verified, 0..) |slot, *v, i| {
        if (tree_at == i) {
            ops.endTree(slot, v);
            continue;
        }
        logits[made] = try ops.commitRow(slot, v);
        made += 1;
        batch.*[kept] = slot;
        kept += 1;
    }
    batch.* = batch.*[0..kept];
    return logits;
}

test "outcome: a real Slot's fields satisfy the recorder, and a cancelled GLM slot counts once" {
    var m = metrics_mod.Metrics.init();
    var slot: Slot = undefined;
    slot.metrics_recorded = false;
    slot.finished = false;
    slot.error_code = null;
    slot.stop_hit = .init(false);
    slot.first_token_ns = 1;
    slot.prefill_ns = 1;
    slot.decode_ns = 1;
    slot.prompt_tokens = 4;
    slot.completion_tokens = 2;
    slot.cached_tokens = 0;
    slot.finish_reason = "";
    recordSlotCleanup(&m, &slot);
    recordSlotCleanup(&m, &slot);
    try expectOutcomes(&m, 0, 1, 0, 0);
}

test "glmNativeDflashArmed: eligible greedy/sampled requests arm it, constrained/penalised/logprobs/budgeted-thinking/disabled ones do not" {
    const t = testing;
    try t.expect(glmNativeDflashArmed(true, true, true, .{ .temperature = 0 }, 0));
    try t.expect(glmNativeDflashArmed(true, true, true, .{ .temperature = 1 }, 0));
    // tool_choice "none" bans the call opener; the verifier honours the bias.
    const none_ban = [_]@import("../logit_bias.zig").Bias{.{ .id = 5, .delta = -100 }};
    try t.expect(glmNativeDflashArmed(true, true, true, .{ .temperature = 0, .think_penalty = .{ .biases = &none_ban } }, 0));

    try t.expect(!glmNativeDflashArmed(false, true, true, .{ .temperature = 0 }, 0));
    try t.expect(!glmNativeDflashArmed(true, false, true, .{ .temperature = 0 }, 0));
    try t.expect(!glmNativeDflashArmed(true, true, false, .{ .temperature = 0 }, 0));

    var constraint: generate_mod.Constraint = undefined;
    try t.expect(!glmNativeDflashArmed(true, true, true, .{ .temperature = 0, .constraint = &constraint }, 0));
    try t.expect(!glmNativeDflashArmed(true, true, true, .{ .temperature = 0, .presence_penalty = 1 }, 0));
    try t.expect(!glmNativeDflashArmed(true, true, true, .{ .temperature = 0 }, 1));
    var think_bound: mtp_plan.ThinkBound = undefined;
    try t.expect(!glmNativeDflashArmed(true, true, true, .{ .temperature = 0, .think_bound = &think_bound }, 0));
}

test "the four-row planner gives spare rows to drafts only while they beat the group's rows" {
    const cost = GlmRowCost{};
    try testing.expectEqual(@as(usize, 2), glmTreeNodes(2, 0.98, cost));
    try testing.expectEqual(@as(usize, 0), glmTreeNodes(2, 0.5, cost));
    try testing.expectEqual(@as(usize, 1), glmTreeNodes(3, 0.95, cost));
    try testing.expectEqual(@as(usize, 0), glmTreeNodes(3, 0.7, cost));
    try testing.expectEqual(@as(usize, 0), glmTreeNodes(4, 1.0, cost));
}

test "a grouped GLM tick destroys every verified row whichever slot holds the tree" {
    const verifier = GlmRows.verifier;
    const Fake = struct {
        allocator: std.mem.Allocator,
        built: usize = 0,
        ended: usize = 0,
        committed: [4]*Slot = undefined,
        committed_n: usize = 0,

        fn verify(self: *@This(), _: []const verifier.Group, out: []verifier.Verified) !void {
            const Tape = @typeInfo(@FieldType(verifier.Verified, "layers")).pointer.child;
            for (out) |*v| {
                const layers = try self.allocator.alloc(Tape, 1);
                layers[0] = null;
                const arrays = try self.allocator.alloc(mlx.mlx_array, 0);
                v.* = .{ .allocator = self.allocator, .layers = layers, .captures = .{ .allocator = self.allocator, .hook = .{ .ids = &.{}, .out = arrays } }, .count = 1, .offset = 0 };
                self.built += 1;
            }
        }
        fn commitRow(self: *@This(), slot: *Slot, _: *const verifier.Verified) !mlx.mlx_array {
            self.committed[self.committed_n] = slot;
            self.committed_n += 1;
            return mlx.mlx_array_new();
        }
        fn endTree(self: *@This(), _: *Slot, _: *verifier.Verified) void {
            self.ended += 1;
        }
    };
    var slots: [4]*Slot = undefined;
    for (&slots, 1..) |*s, i| s.* = @ptrFromInt(@alignOf(Slot) * i);
    var groups: [4]verifier.Group = undefined;
    for (2..5) |width| {
        for (0..width) |tree_at| {
            var fake = Fake{ .allocator = testing.allocator };
            var live = slots;
            var batch: []*Slot = live[0..width];
            const logits = try glmRowsSettle(&fake, testing.allocator, &batch, groups[0..width], tree_at);
            defer testing.allocator.free(logits);
            for (logits) |a| _ = mlx.mlx_array_free(a);
            try testing.expectEqual(width, fake.built);
            try testing.expectEqual(@as(usize, 1), fake.ended);
            try testing.expectEqual(width - 1, batch.len);
            try testing.expectEqual(width - 1, logits.len);
            for (batch, fake.committed[0..fake.committed_n]) |kept, done| try testing.expectEqual(kept, done);
            for (batch) |kept| try testing.expect(kept != slots[tree_at]);
        }
    }
}

test "a drafting GLM slot rides plain rows only in company of its own model" {
    var glm_a: LoadedModel = undefined;
    var glm_b: LoadedModel = undefined;
    var slots: [4]Slot = undefined;
    for (&slots, 0..) |*s, i| s.model = if (i == 3) &glm_b else &glm_a;
    try testing.expect(!glmRowsInCompany(&slots[0], &.{&slots[0]}, &.{}));
    try testing.expect(!glmRowsInCompany(&slots[0], &.{&slots[0]}, &.{&slots[3]}));
    try testing.expect(glmRowsInCompany(&slots[0], &.{&slots[0]}, &.{&slots[1]}));
    try testing.expect(glmRowsInCompany(&slots[0], &.{ &slots[0], &slots[2] }, &.{}));
}

test "a resident GLM batches plain rows four at a time, a streamed one stays serial" {
    var cfg = try model_mod.parseConfigFromJson(testing.allocator, @embedFile("../fixtures/glm5_config.json"));
    try testing.expect(configBatchesDecode(&cfg));
    try testing.expectEqual(@as(usize, 4), batchGroupCap(&cfg));
    cfg.expert_streaming = true;
    try testing.expect(!configBatchesDecode(&cfg));
}

test "every GLM slot owns its native request and hands it to its forward context" {
    const allocator = testing.allocator;
    var sch: Scheduler = undefined;
    sch.allocator = allocator;
    sch.io = testing.io;
    sch.kv_quant_config = .dense;
    sch.kv_quant_explicit = false;
    sch.queue_mu = .init;
    sch.queue_cond = .init;
    sch.submit_cond = .init;
    sch.shutdown = .init(false);
    sch.queue_cap = 2;
    sch.in_flight = 0;
    sch.pending = .empty;
    sch.decoding = .empty;
    sch.cleanup_queue = .empty;
    sch.prefilling = .empty;
    defer sch.pending.deinit(allocator);
    defer sch.decoding.deinit(allocator);
    defer sch.cleanup_queue.deinit(allocator);
    defer sch.prefilling.deinit(allocator);
    var cfg = try model_mod.parseConfigFromJson(allocator, @embedFile("../fixtures/glm5_config.json"));
    var model: LoadedModel = undefined;
    model.config = &cfg;
    model.transformer = null;
    model.prefix_cache = null;
    var slots: [2]*Slot = undefined;
    for (&slots) |*slot| slot.* = try sch.submit(.{ .model = &model, .prompt_ids = &.{1}, .sampling = .{}, .eos_token_ids = &.{}, .max_tokens = 1 });
    for (slots) |slot| {
        const request = &slot.glm5_request.?;
        try testing.expectEqual(request, slot.ctx.glm5_request.?);
        try testing.expectEqual(@as(usize, cfg.num_hidden_layers), request.layers.len);
        try testing.expect(request.dense_prefill and request.prefill_async);
    }
    try testing.expect(slots[0].ctx.glm5_request.? != slots[1].ctx.glm5_request.?);
    for (slots) |slot| sch.complete(slot);
    while (sch.cleanup_queue.items.len > 0) sch.cleanup_queue.orderedRemove(0).deinit();
}

test "a GLM slot submitted with a kv4 override is refused by the same name admission uses elsewhere" {
    const allocator = testing.allocator;
    var sch: Scheduler = undefined;
    sch.allocator = allocator;
    sch.io = testing.io;
    sch.kv_quant_config = .dense;
    sch.kv_quant_explicit = false;
    sch.queue_mu = .init;
    sch.queue_cond = .init;
    sch.submit_cond = .init;
    sch.shutdown = .init(false);
    sch.queue_cap = 2;
    sch.in_flight = 0;
    sch.pending = .empty;
    sch.decoding = .empty;
    sch.cleanup_queue = .empty;
    sch.prefilling = .empty;
    defer sch.pending.deinit(allocator);
    defer sch.decoding.deinit(allocator);
    defer sch.cleanup_queue.deinit(allocator);
    defer sch.prefilling.deinit(allocator);
    var cfg = try model_mod.parseConfigFromJson(allocator, @embedFile("../fixtures/glm5_config.json"));
    var model: LoadedModel = undefined;
    model.config = &cfg;
    model.transformer = null;
    model.prefix_cache = null;
    const result = sch.submit(.{ .model = &model, .prompt_ids = &.{1}, .sampling = .{}, .eos_token_ids = &.{}, .max_tokens = 1, .kv_quant_config = transformer_mod.KVQuantConfig.affine(4) });
    try expectError(error.GlmKvQuantUnsupported, result);
}

test "culling an errored or cancelled GLM slot releases its native state, a live one keeps it" {
    var mm = metrics_mod.Metrics.init();
    var sch: Scheduler = undefined;
    sch.io = testing.io;
    sch.allocator = testing.allocator;
    sch.queue_mu = .init;
    sch.metrics = &mm;
    sch.live_sessions = undefined;
    sch.live_session_count = 0;
    sch.decoding = .empty;
    defer sch.decoding.deinit(testing.allocator);
    var model: model_registry_mod.LoadedModel = undefined;
    model.id = "org/glm-cull";
    model.prefix_cache = null;
    model.transformer = null;
    var slots: [3]Slot = undefined;
    for (&slots, 0..) |*slot, i| {
        slot.model = &model;
        slot.cache = .{ .entries = &.{}, .step = 0, .allocator = testing.allocator, .config = .dense };
        slot.ssm_entries = null;
        slot.ring_cps = .{};
        slot.restored_entry = 0;
        slot.full_prompt = &.{};
        slot.prompt_tokens = 4;
        slot.completion_tokens = 0;
        slot.cached_tokens = 0;
        slot.request_id = i;
        slot.max_tokens = 8;
        slot.request_start_ts = std.Io.Timestamp.now(testing.io, .boot);
        slot.legacy_gen = null;
        slot.finished = false;
        slot.error_code = null;
        slot.cancelled = .init(false);
        slot.glm5_request = try glm5_forward_mod.Request.init(testing.allocator, 2);
        slot.glm5_request.?.offset = 4;
        try sch.decoding.append(testing.allocator, slot);
    }
    defer for (&slots) |*slot| slot.glm5_request.?.deinit();
    slots[0].error_code = "GlmReserveMemoryLimit";
    slots[1].cancelled.store(true, .release);
    cullDecoding(&sch);
    try testing.expectEqual(@as(usize, 1), sch.decoding.items.len);
    try testing.expectEqual(&slots[2], sch.decoding.items[0]);
    try testing.expectEqual(@as(usize, 0), slots[0].glm5_request.?.offset);
    try testing.expectEqual(@as(usize, 0), slots[1].glm5_request.?.offset);
    try testing.expectEqual(@as(usize, 4), slots[2].glm5_request.?.offset);
}

test "a GLM cancel landing between the cull passes is committed once before its native reset" {
    const Hook = struct {
        var commits: [2]usize = .{ 0, 0 };
        var offset_at_commit: [2]usize = .{ 0, 0 };
        var slots: [2]*Slot = undefined;
        fn commit(_: *Scheduler, s: *Slot) void {
            const i: usize = if (s == slots[0]) 0 else 1;
            commits[i] += 1;
            offset_at_commit[i] = s.glm5_request.?.offset;
            slots[1].cancelled.store(true, .release);
        }
    };
    var sch: Scheduler = undefined;
    sch.io = testing.io;
    sch.allocator = testing.allocator;
    sch.queue_mu = .init;
    sch.metrics = null;
    sch.live_session_count = 0;
    sch.decoding = .empty;
    defer sch.decoding.deinit(testing.allocator);
    var model: model_registry_mod.LoadedModel = undefined;
    model.id = "org/glm-cull-race";
    model.prefix_cache = null;
    model.transformer = null;
    var slots: [2]Slot = undefined;
    for (&slots, 0..) |*slot, i| {
        slot.model = &model;
        slot.cache = .{ .entries = &.{}, .step = 0, .allocator = testing.allocator, .config = .dense };
        slot.ssm_entries = null;
        slot.ring_cps = .{};
        slot.restored_entry = 0;
        slot.full_prompt = &.{};
        slot.prompt_tokens = 4;
        slot.completion_tokens = 0;
        slot.cached_tokens = 0;
        slot.request_id = i;
        slot.max_tokens = 8;
        slot.legacy_gen = null;
        slot.finished = false;
        slot.error_code = null;
        slot.cancelled = .init(false);
        slot.glm5_request = try glm5_forward_mod.Request.init(testing.allocator, 2);
        slot.glm5_request.?.offset = 4;
        Hook.slots[i] = slot;
        try sch.decoding.append(testing.allocator, slot);
    }
    defer for (&slots) |*slot| slot.glm5_request.?.deinit();
    Hook.commits = .{ 0, 0 };
    slots[0].cancelled.store(true, .release);

    cullDecodingWith(&sch, Hook.commit);
    // The late cancel stays in `decoding` with its state, and the next cull commits it.
    try testing.expectEqual(@as(usize, 1), sch.decoding.items.len);
    try testing.expectEqual(@as(usize, 4), slots[1].glm5_request.?.offset);
    cullDecodingWith(&sch, Hook.commit);
    try testing.expectEqual(@as(usize, 0), sch.decoding.items.len);
    try testing.expectEqual([2]usize{ 1, 1 }, Hook.commits);
    try testing.expectEqual([2]usize{ 4, 4 }, Hook.offset_at_commit);
    try testing.expectEqual(@as(usize, 0), slots[1].glm5_request.?.offset);
}

test "postPrefillTerminal releases native GLM state for a cancelled slot but not a live one" {
    var cancelled_slot: Slot = undefined;
    cancelled_slot.state = .decoding;
    cancelled_slot.cancelled = .init(true);
    cancelled_slot.glm5_request = try glm5_forward_mod.Request.init(testing.allocator, 2);
    cancelled_slot.glm5_request.?.offset = 4;
    defer cancelled_slot.glm5_request.?.deinit();

    var live_slot: Slot = undefined;
    live_slot.state = .decoding;
    live_slot.cancelled = .init(false);
    live_slot.glm5_request = try glm5_forward_mod.Request.init(testing.allocator, 2);
    live_slot.glm5_request.?.offset = 4;
    defer live_slot.glm5_request.?.deinit();

    var errored_slot: Slot = undefined;
    errored_slot.state = .errored;
    errored_slot.cancelled = .init(false);
    errored_slot.glm5_request = null;

    try testing.expect(postPrefillTerminal(&cancelled_slot));
    try testing.expectEqual(@as(usize, 0), cancelled_slot.glm5_request.?.offset);

    try testing.expect(!postPrefillTerminal(&live_slot));
    try testing.expectEqual(@as(usize, 4), live_slot.glm5_request.?.offset);

    try testing.expect(postPrefillTerminal(&errored_slot));
}
