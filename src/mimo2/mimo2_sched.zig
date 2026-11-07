//! MiMo as the scheduler sees it: it batches plain slots as rows of one forward, and a crowded MTP slot may drop to
//! plain ticks.
const std = @import("std");
const mimo_forward = @import("mimo_forward.zig");
const mlx = @import("../mlx.zig");
const model = @import("../model.zig");
const sampler = @import("../sampler.zig");
const trunk_mod = @import("../trunk.zig");
const scheduler = @import("../scheduler.zig");
const sched = @import("../arch/sched.zig");
const model_load = @import("../model_load.zig");

const Slot = scheduler.Slot;
const Scheduler = scheduler.Scheduler;
const Transformer = trunk_mod.Transformer;
const ForwardCtx = trunk_mod.ForwardCtx;
const ModelConfig = model.ModelConfig;

pub fn batches(config: *const ModelConfig) bool {
    return config.supportsBatchedMimoDecode();
}

pub fn rowCap(config: *const ModelConfig) ?usize {
    return if (config.supportsBatchedMimoDecode()) 4 else null;
}

pub fn modelBatches(xfm: *const Transformer) bool {
    return mimo_forward.supportsBatchedMimoDecode(xfm);
}

/// Per-row attention reads each slot's own cache: nothing pads.
pub fn padsBatch(config: *const ModelConfig) bool {
    return !config.supportsBatchedMimoDecode();
}

pub fn batchArm(xfm: *Transformer, ctxs: []const *ForwardCtx) ?sched.BatchArm {
    _ = ctxs;
    if (!mimo_forward.supportsBatchedMimoDecode(xfm)) return null;
    return .{ .moe_positions = true, .advances_offset = true };
}

pub fn batchedForward(
    sch: *Scheduler,
    allocator: std.mem.Allocator,
    xfm: *Transformer,
    batch: *[]*Slot,
    next_tokens: []const u32,
    ctxs: []const *ForwardCtx,
    rope_offsets: []const u32,
    arm: sched.BatchArm,
    hidden_rows: ?*?[]mlx.mlx_array,
) ![]mlx.mlx_array {
    _ = .{ sch, allocator, batch, arm };
    return mimo_forward.forwardMimoBatchedDecode(xfm, next_tokens, ctxs, rope_offsets, hidden_rows);
}

/// A MiMo MTP slot that may drop to plain batched ticks when crowded (`mtpCrowdThresholdFor`); its rounds otherwise
/// stay solo, and no grouped verify ever takes it.
fn mtpCrowdable(slot: *const Slot) bool {
    const gen = if (slot.legacy_gen) |*g| g else return false;
    if (!slot.enable_mtp or gen.mtp == null or gen.mtp_cache == null or !gen.has_last_hidden) return false;
    if (gen.spec_disabled_runtime or gen.mtp_serial_left > 0 or gen.mtp_serial_exit != .none) return false;
    if (slot.sampling.constraint != null or slot.logprobs_n > 0 or sampler.penaltyActive(slot.sampling)) return false;
    const t = slot.model.transformer orelse return false;
    if (!mimo_forward.supportsBatchedMimoDecode(t)) return false;
    return scheduler.specTickMode(slot.enable_mtp, true, slot.enable_drafter, gen.dflash != null) == .mtp;
}

pub fn tickClass(slot: *const Slot) sched.TickClass {
    return if (mtpCrowdable(slot)) .mtp_group else .none;
}

pub fn coldLoadBill(load: sched.ColdLoad) !?u64 {
    if (!load.config.usesMimoSourceTrunk()) return null;
    return try model_load.residentColdLoadBillBytes(load.io, load.allocator, load.config, load.path, load.load_vision, load.mtp_on, load.no_drafter, load.drafter_dir);
}
