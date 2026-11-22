//! DeepSeek-V4.1 as the scheduler sees it: a streamed load runs one request at a time over a decode state its slot
//! request holds while it runs and the model parks between requests; a fully resident load joins its slots' requests as
//! rows of one decode forward, each over a state of its own. Restore points always go to SSD, and a thought point is
//! written behind a DSpark round.
const std = @import("std");
const testTrunk = @import("../test_support.zig").testTrunk;
const arch_handle = @import("../arch/handle.zig");
const mlx = @import("../mlx.zig");
const model = @import("../model.zig");
const transformer = @import("../trunk.zig");
const prefix_cache = @import("../prefix_cache.zig");
const prefix_arch = @import("../arch/prefix.zig");
const model_registry = @import("../model_registry.zig");
const scheduler = @import("../scheduler.zig");
const generate = @import("../generate.zig");
const sched = @import("../arch/sched.zig");
const dsv41_prefix = @import("dsv41_prefix.zig");
const dsv41_borrow = @import("dsv41_borrow.zig");
const thought_point = @import("dsv41_thought_point.zig");
const dsv41_model = @import("dsv41_model.zig");
const dsv41_state = @import("dsv41_state.zig");
const dsv41_serving = @import("dsv41_serving.zig");
const log = @import("../log.zig");

const Slot = scheduler.Slot;
const Scheduler = scheduler.Scheduler;
const Trunk = transformer.Trunk;
const ForwardCtx = transformer.ForwardCtx;
const Generator = generate.Generator;
const ModelConfig = model.ModelConfig;

pub fn facts(config: *const ModelConfig) sched.Facts {
    _ = config;
    return .{ .always_write_through = true };
}

/// A fully resident DeepSeek Cooler load joins requests as rows of one decode forward, each over a decode state of its
/// own, as many as a joined forward carries on this GPU (`joinedRowsMax`); a streamed one serves one row. Every Cooler
/// load runs its experts through the stream engine (`expert_streaming`), resident ones too: the plan's answer is
/// `expert_resident`, and the load refuses a config that claims it over a streamed plan (`dsv41_serving.applyRows`).
pub fn coolerRows(config: *const ModelConfig) sched.CoolerRows {
    if (!config.expert_resident) return .{ .rows = 1, .why = "its routed experts stream from SSD (the expert cache is one request's working set)" };
    // Rows are Sushi Cooler's: the user's opt-out to MLX keeps the one-request decode it had.
    if (!@import("../cooler_side.zig").onEngine(config)) return .{ .rows = 1, .why = "its decode runs on MLX" };
    if (config.dsv41) |c| if (!c.widthExactLayout()) return .{ .rows = 1, .why = "its FP8 trunk or FP4 experts give a row other bytes beside other rows" };
    return .{ .rows = joinedRowsMax(), .note = "steps run the exact arm (+~0.6 ms a step on NAX); a next turn restores from the SSD tier" };
}

/// The rows a joined forward carries with each row its one-row step's bytes (`Model.maxJoinedRows`): 16 with matrix
/// units, 8 without.
fn joinedRowsMax() usize {
    return @intCast(@import("sushi_exl3").project_rows.rowsMax(.{ .exact = true, .wide = true }));
}

/// The rows the load joins; 1 for a streamed load.
fn joinedRows(config: *const ModelConfig) usize {
    return coolerRows(config).rows;
}

pub fn batches(config: *const ModelConfig) bool {
    return joinedRows(config) > 1;
}

pub fn rowCap(config: *const ModelConfig) ?usize {
    const n = joinedRows(config);
    return if (n > 1) n else null;
}

pub fn modelBatches(xfm: *const Trunk) bool {
    const m = xfm.arch_state.dsv41.model orelse return false;
    return m.serves_rows;
}

/// Each row's attention reads its own request's state: nothing pads.
pub fn padsBatch(config: *const ModelConfig) bool {
    return joinedRows(config) <= 1;
}

/// A DSpark slot with this many of its model's slots decoding (itself included) rides the joined tick as a plain row:
/// from three, joined plain rows out-run DSpark rounds taken one slot at a time (dsv41-arch#concurrency).
pub const rows_company_min: usize = 3;

/// A DSpark slot on a load that joins rows decodes as a plain row of the joined tick when in company; alone it drafts.
pub fn tickClass(slot: *const Slot) sched.TickClass {
    const gen = if (slot.legacy_gen) |*g| g else return .none;
    if (!gen.dspark or gen.done) return .none;
    if (slot.sampling.constraint != null or @import("../sampler.zig").penaltyActive(slot.sampling)) return .none;
    return if (modelBatches(arch_handle.trunk(slot.model.handle orelse return .none))) .rows_company else .none;
}

/// A slot joins once its prefill made its request's state.
pub fn readyToBatch(xfm: *const Trunk, ctx: *ForwardCtx) bool {
    const m = xfm.arch_state.dsv41.model orelse return true;
    if (!m.serves_rows) return true;
    const r = dsv41_state.requestOf(ctx) orelse return false;
    return r.state != null;
}

/// Rows are positioned by each request's state; the tick moves the slot's offset as a solo step would leave it.
pub fn batchArm(xfm: *Trunk, ctxs: []const *ForwardCtx) ?sched.BatchArm {
    const m = xfm.arch_state.dsv41.model orelse return null;
    if (!m.serves_rows) return null;
    for (ctxs) |c| if (!readyToBatch(xfm, c)) return null;
    return .{ .moe_positions = false, .advances_offset = true };
}

/// One decode forward over the group's requests (`dsv41_serving.forwardRows`): each row is its request's one-row step,
/// and each slot's cache position follows its state as a solo step leaves it.
pub fn batchedForward(
    sch: *Scheduler,
    allocator: std.mem.Allocator,
    xfm: *Trunk,
    batch: *[]*Slot,
    next_tokens: []const u32,
    ctxs: []const *ForwardCtx,
    rope_offsets: []const u32,
    arm: sched.BatchArm,
    hidden_rows: ?*?[]mlx.mlx_array,
) ![]mlx.mlx_array {
    _ = .{ sch, batch, rope_offsets, arm, hidden_rows };
    var segs: [max_rows]dsv41_model.Seg = undefined;
    var ids: [max_rows][1]u32 = undefined;
    if (ctxs.len > max_rows) return error.Dsv41RowsWidth;
    for (ctxs, next_tokens, 0..) |ctx, t, i| {
        ids[i] = .{t};
        segs[i] = .{ .request = dsv41_state.requestOf(ctx) orelse return error.Dsv41NoRequest, .ids = &ids[i] };
    }
    if (!rows_logged) {
        rows_logged = true;
        log.info("[dsv41-rows] joined decode tick engaged: {d} rows\n", .{ctxs.len});
    }
    const joined = try dsv41_serving.forwardRows(xfm, segs[0..ctxs.len]);
    defer _ = mlx.mlx_array_free(joined);
    for (ctxs, segs[0..ctxs.len]) |ctx, sg| ctx.cache.step = sg.request.state.?.n;
    return rowsOf(allocator, joined, ctxs.len, xfm.s);
}

const max_rows = 16;
var rows_logged = false;

/// Row `i` of `[n, vocab]` logits as the `[1, 1, vocab]` a one-row step returns, for each row.
fn rowsOf(allocator: std.mem.Allocator, joined: mlx.mlx_array, n: usize, s: mlx.mlx_stream) ![]mlx.mlx_array {
    const out = try allocator.alloc(mlx.mlx_array, n);
    var made: usize = 0;
    errdefer {
        for (out[0..made]) |o| _ = mlx.mlx_array_free(o);
        allocator.free(out);
    }
    const vocab: c_int = mlx.getShape(joined)[1];
    for (0..n) |i| {
        const start = [2]c_int{ @intCast(i), 0 };
        const stop = [2]c_int{ @intCast(i + 1), vocab };
        const step = [2]c_int{ 1, 1 };
        var row = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(row);
        try mlx.check(mlx.mlx_slice(&row, joined, &start, 2, &stop, 2, &step, 2, s));
        out[i] = mlx.mlx_array_new();
        made = i + 1;
        const shape = [3]c_int{ 1, 1, vocab };
        try mlx.check(mlx.mlx_reshape(&out[i], row, &shape, 3, s));
    }
    return out;
}

/// A slot's request holds the decode state while it runs; the model parks it between requests.
pub fn initRequest(allocator: std.mem.Allocator, config: *const ModelConfig) !?dsv41_model.Request {
    _ = config;
    return try dsv41_model.Request.init(allocator, 0);
}

pub fn modelStateBytes(slot: *const Slot) u64 {
    return thought_point.dsv41StateBytes(slot);
}

pub fn parkedStateBytes(entry: *model_registry.LoadedModel) u64 {
    return thought_point.dsv41ModelStateBytes(entry);
}

/// The next exclusive request is admitted as soon as the slot leaves `decoding` and takes the record over, so a
/// commit at the cleanup drain would publish onto its state.
pub fn commitsAtCull(slot: *const Slot) bool {
    const xfm = arch_handle.trunk(slot.model.handle orelse return false);
    return xfm.arch_state.dsv41.model != null;
}

pub const prepare_commit_label = "DeepSeek row marker";
pub const cancelled_entry_label = "a DeepSeek entry";

/// The cache holds a one-byte-a-token row marker for the model's rows.
pub fn prepareCommit(slot: *Slot) !void {
    const xfm = arch_handle.trunk(slot.model.handle orelse return);
    const dsv = xfm.arch_state.dsv41.model orelse return;
    try dsv41_prefix.markRows(&slot.cache, @import("dsv41_model.zig").stateStream(dsv));
}

pub fn publishLive(sch: *Scheduler, slot: *Slot, hc: *prefix_cache.HotPrefixCache, tokens: []const u32) void {
    thought_point.publishLive(sch, slot, hc, tokens);
}

/// A restore point is the whole state, so a cancelled prefill with none restores as a miss.
pub fn cancelledNeedsCheckpoints() bool {
    return true;
}

/// With joined rows the restore installs into the slot's own request, lent until `restoreEnd`.
pub fn restoreTarget(slot: *Slot, xfm: *Trunk) ?sched.RestoreTarget {
    const dsv = xfm.arch_state.dsv41.model orelse return null;
    if (dsv.serves_rows) {
        // Nothing is lent at a lookup, so `restoreEnd` ends this lend by clearing it.
        std.debug.assert(dsv.bound == null);
        _ = dsv41_model.lendFor(dsv, slotRequest(slot));
    }
    return .{ .rec = prefix_arch.recurrent("dsv41", dsv), .log_tag = "[dsv41-prefix]" };
}

/// The lookup runs off any forward, where nothing is lent: the lend `restoreTarget` opened is the only one to end.
pub fn restoreEnd(slot: *Slot, xfm: *Trunk) void {
    const dsv = xfm.arch_state.dsv41.model orelse return;
    if (!dsv.serves_rows) return;
    std.debug.assert(dsv.bound == slotRequest(slot));
    dsv41_model.unbindRows(dsv, null);
}

/// The slot's DeepSeek request; null when it holds none.
fn slotRequest(slot: *Slot) ?*dsv41_model.Request {
    return if (slot.request) |*r| r.as(.dsv41) else null;
}

/// With joined rows a hook reads the slot's own state, lent for its span (`dsv41_model.lendFor`).
fn lendSlot(slot: *Slot) ?dsv41_model.Lent {
    const xfm = arch_handle.trunk(slot.model.handle orelse return null);
    return dsv41_model.lendFor(xfm.arch_state.dsv41.model orelse return null, slotRequest(slot));
}

pub fn writeThrough(
    hc: *prefix_cache.HotPrefixCache,
    slot: *Slot,
    entry_id: *?u64,
    prompt: []const u32,
    media: prefix_cache.MediaStamp,
    cps: []transformer.SSMCheckpoint,
    _: mlx.mlx_stream,
) bool {
    _ = entry_id;
    const dsv = arch_handle.trunk(slot.model.handle orelse return false).arch_state.dsv41.model orelse return false;
    const lent = lendSlot(slot);
    defer if (lent) |l| l.end();
    dsv41_prefix.writeThroughDsv41(hc, dsv, &slot.cache, prompt, slot.has_tools, media, cps, "prefill");
    return true;
}

pub fn armThoughtPoint(slot: *Slot, gen: *Generator, tier_takes_points: bool) void {
    thought_point.armDsv41ThoughtPoint(slot, gen, tier_takes_points);
}

pub fn afterSpecBlock(slot: *Slot, gen: *Generator, block: []const u32) void {
    const lent = lendSlot(slot);
    defer if (lent) |l| l.end();
    thought_point.writeDsv41ThoughtPoint(slot, gen, block);
}

pub fn endRequest(config: *const ModelConfig) void {
    dsv41_borrow.endRequestMlx(config);
}

pub fn flushResident(entry: *model_registry.LoadedModel, why: []const u8) void {
    dsv41_prefix.flushResident(entry, why);
}

test "an arch whose request state lives in the model restores into it under its own log tag and needs checkpoints from a cancelled prefill" {
    var xfm = testTrunk("deepseek_v41");
    var lm: model_registry.LoadedModel = undefined;
    var slot = sched.testSlot(&xfm, &lm, .{ .model_type = "deepseek_v41" });
    // A model the hooks only name, and read only as a one-row load.
    const m = try std.testing.allocator.create(@import("dsv41_model.zig").Model);
    defer std.testing.allocator.destroy(m);
    m.serves_rows = false;
    xfm.arch_state.dsv41.model = m;
    const target = sched.restoreTarget(&slot, &xfm).?;
    try std.testing.expectEqualStrings("[dsv41-prefix]", target.log_tag);
    try std.testing.expect(!target.takes_dflash and !target.takes_mtp);
    try std.testing.expect(sched.cancelledNeedsCheckpoints(&slot));
    try std.testing.expect(sched.commitsAtCull(&slot));
    xfm.arch_state.dsv41.model = null;
    try std.testing.expect(sched.restoreTarget(&slot, &xfm) == null);
    try std.testing.expect(!sched.commitsAtCull(&slot));
}

test "a DeepSeek slot with no DSpark round ticks alone or joins as plain, and a drafting one needs three of its model's slots for company" {
    const sched_policy = @import("../sched_policy.zig");
    var xfm = testTrunk("deepseek_v41");
    var lm: model_registry.LoadedModel = undefined;
    var other: model_registry.LoadedModel = undefined;
    var slots = [_]Slot{ sched.testSlot(&xfm, &lm, .{ .model_type = "deepseek_v41" }), sched.testSlot(&xfm, &lm, .{ .model_type = "deepseek_v41" }), sched.testSlot(&xfm, &lm, .{ .model_type = "deepseek_v41" }), sched.testSlot(&xfm, &other, .{ .model_type = "deepseek_v41" }) };
    // A slot with no generator drafting: no rows class.
    try std.testing.expectEqual(sched.TickClass.none, sched.tickClass(&slots[0]));
    try std.testing.expectEqual(sched_policy.TickBucket.serial, sched_policy.bucketOf(.spec_active, .none, .{}));
    try std.testing.expectEqual(sched_policy.TickBucket.batch, sched_policy.bucketOf(.ok, .none, .{}));
    // Two of the model's slots are no company for a drafting one; three are, a slot of another model never counts.
    try std.testing.expect(!sched.rowsInCompany(&slots[0], &.{&slots[0]}, &.{&slots[1]}));
    try std.testing.expect(!sched.rowsInCompany(&slots[0], &.{ &slots[0], &slots[1] }, &.{&slots[3]}));
    try std.testing.expect(sched.rowsInCompany(&slots[0], &.{ &slots[0], &slots[1] }, &.{&slots[2]}));
}
