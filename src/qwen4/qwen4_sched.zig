//! Flash-Next as the scheduler sees it: what batches, which rounds group, the head it commits beside the KV.
const std = @import("std");
const arch_handle = @import("../arch/handle.zig");
const sched_policy = @import("../sched_policy.zig");
const qwen4_forward = @import("qwen4_forward.zig");
const mlx = @import("../mlx.zig");
const model = @import("../model.zig");
const log = @import("../log.zig");
const trunk_mod = @import("../trunk.zig");
const scheduler = @import("../scheduler.zig");
const sched = @import("../arch/sched.zig");
const qsa_state = @import("qwen4_qsa_state.zig");

const Slot = scheduler.Slot;
const Scheduler = scheduler.Scheduler;
const Trunk = trunk_mod.Trunk;
const ForwardCtx = trunk_mod.ForwardCtx;
const ModelConfig = model.ModelConfig;

pub fn facts(config: *const ModelConfig) sched.Facts {
    return .{
        .long_ctx = config.longCtxGated(),
        .caps_ssm_checkpoints = true,
        .thought_point = .with_mtp,
    };
}

pub fn batches(config: *const ModelConfig) bool {
    return config.supportsBatchedGdnDecode();
}

pub fn rowCap(config: *const ModelConfig) ?usize {
    return if (config.supportsBatchedQwen4Rows()) 8 else null;
}

pub fn modelBatches(xfm: *const Trunk) bool {
    return qwen4_forward.supportsBatchedGdnDecode(xfm);
}

/// A GDN slot without recurrent state yet cannot merge.
pub fn readyToBatch(xfm: *const Trunk, ctx: *ForwardCtx) bool {
    return !qwen4_forward.supportsBatchedGdnDecode(xfm) or qwen4_forward.batchedGdnReady(xfm, &.{ctx});
}

pub fn padsBatch(config: *const ModelConfig) bool {
    return !config.supportsBatchedQwen4Rows();
}

pub fn plannerObservesPlainRows(xfm: *const Trunk) bool {
    return xfm.arch_state.qwen4.model != null;
}

const arm_padded: u8 = 0;
const arm_rows: u8 = 1;

pub fn batchArm(xfm: *Trunk, ctxs: []const *ForwardCtx) ?sched.BatchArm {
    if (!(qwen4_forward.supportsBatchedGdnDecode(xfm) and qwen4_forward.batchedGdnReady(xfm, ctxs))) return null;
    // A per-row tick moves each slot's `moe_seq_offset` itself; the padded batch leaves it to the scheduler.
    const rows = qwen4_forward.supportsBatchedQwen4Rows(xfm);
    return .{ .arm = if (rows) arm_rows else arm_padded, .moe_positions = true, .advances_offset = !rows };
}

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
    _ = .{ sch, allocator, batch };
    if (arm.arm == arm_rows) return qwen4_forward.forwardQwen4DecodeRows(xfm, next_tokens, ctxs, hidden_rows);
    return qwen4_forward.forwardMoeBatchedDecode(xfm, next_tokens, ctxs, rope_offsets, hidden_rows);
}

/// A head the model owns (never a per-request one) cannot ride a grouped verify.
pub fn mtpGroupable(slot: *const Slot) bool {
    const gen = if (slot.legacy_gen) |*g| g else return false;
    if (!slot.enable_mtp or gen.mtp == null or gen.mtp_cache == null or !gen.has_last_hidden) return false;
    if (gen.mtp.?.moduleOwned()) return false;
    if (gen.spec_disabled_runtime or gen.mtp_serial_left > 0 or gen.mtp_serial_exit != .none) return false;
    if (gen.ctx.ssm_entries == null) return false;
    if (slot.sampling.constraint != null or slot.logprobs_n > 0) return false;
    const t = arch_handle.trunk(slot.model.handle orelse return false);
    if (!qwen4_forward.supportsBatchedGdnDecode(t)) return false;
    return sched_policy.specTickMode(slot.enable_mtp, true, slot.enable_drafter, gen.dflash != null) == .mtp;
}

pub fn tickClass(slot: *const Slot) sched.TickClass {
    return if (mtpGroupable(slot)) .mtp_group else .none;
}

/// The newest checkpoint takes the slot's live QSA indexer history as a view of the capacity buffer (the slot is
/// torn down right after). A failure commits the entry history-less, which a QSA arch treats as a miss.
pub fn handoffHistory(slot: *Slot, cps: []trunk_mod.SSMCheckpoint) void {
    if (qsa_state.checkpointHasQsaPooled(&cps[cps.len - 1])) return;
    const ents = slot.ssm_entries orelse return;
    const xf = arch_handle.trunk(slot.model.handle orelse return);
    qsa_state.handoffQsaHistoryToLatest(cps, ents, xf.s) catch |err| {
        log.warn("[hot-cache] QSA history handoff failed: {s} — committing without it\n", .{@errorName(err)});
    };
}

/// The in-checkpoint head commits its QSA half beside the KV. Its row count IS its cache's step; a snapshot whose
/// step disagrees is refused.
pub fn mtpHeadCommit(head: *const Trunk) ?sched.HeadParts {
    const hm = &head.arch_state.qwen4.mtp.?;
    if (hm.cache.step != hm.seq_offset) {
        log.warn("[hot-cache] mtp head step gap (cache.step={d}, history={d} rows) — head history not committed\n", .{ hm.cache.step, hm.seq_offset });
        return null;
    }
    return .{ .entry = &hm.entry, .pos_base = hm.pos_base, .marks = qwen4_forward.qwen4MtpMarks(head) };
}

pub fn resetForRetry(slot: *const Slot, xfm: *Trunk) void {
    qwen4_forward.qwen4MtpResetOwned(xfm, slot.enable_mtp);
}
