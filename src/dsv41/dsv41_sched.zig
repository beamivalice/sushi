//! DeepSeek-V4.1 as the scheduler sees it: one request at a time over state the model holds, restore points that
//! always go to SSD, and a thought point written behind a DSpark round.
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

const Slot = scheduler.Slot;
const Scheduler = scheduler.Scheduler;
const Trunk = transformer.Trunk;
const Generator = generate.Generator;
const ModelConfig = model.ModelConfig;

pub fn facts(config: *const ModelConfig) sched.Facts {
    _ = config;
    return .{ .always_write_through = true };
}

pub fn modelStateBytes(slot: *const Slot) u64 {
    return thought_point.dsv41StateBytes(slot);
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

pub fn restoreTarget(slot: *Slot, xfm: *Trunk) ?sched.RestoreTarget {
    _ = slot;
    const dsv = xfm.arch_state.dsv41.model orelse return null;
    return .{ .rec = prefix_arch.recurrent("dsv41", dsv), .log_tag = "[dsv41-prefix]" };
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
    dsv41_prefix.writeThroughDsv41(hc, dsv, &slot.cache, prompt, slot.has_tools, media, cps, "prefill");
    return true;
}

pub fn armThoughtPoint(slot: *Slot, gen: *Generator, tier_takes_points: bool) void {
    thought_point.armDsv41ThoughtPoint(slot, gen, tier_takes_points);
}

pub fn afterSpecBlock(slot: *Slot, gen: *Generator, block: []const u32) void {
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
    xfm.arch_state.dsv41.model = @ptrFromInt(@alignOf(@import("dsv41_model.zig").Model));
    const target = sched.restoreTarget(&slot, &xfm).?;
    try std.testing.expectEqualStrings("[dsv41-prefix]", target.log_tag);
    try std.testing.expect(!target.takes_dflash and !target.takes_mtp);
    try std.testing.expect(sched.cancelledNeedsCheckpoints(&slot));
    try std.testing.expect(sched.commitsAtCull(&slot));
    xfm.arch_state.dsv41.model = null;
    try std.testing.expect(sched.restoreTarget(&slot, &xfm) == null);
    try std.testing.expect(!sched.commitsAtCull(&slot));
}
