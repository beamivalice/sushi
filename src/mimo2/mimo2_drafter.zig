//! MiMo's drafter: the checkpoint's trained MTP heads behind `arch/drafter.zig`.
const std = @import("std");
const mlx = @import("../mlx.zig");
const trunk_mod = @import("../trunk.zig");
const mtp_mod = @import("../mtp.zig");
const spec_types = @import("../spec_types.zig");
const mimo_mtp = @import("mimo_mtp.zig");
const Transformer = trunk_mod.Transformer;
const KVCache = trunk_mod.KVCache;

pub const kinds = [_]spec_types.Kind{ .mtp, .dflash2 };
pub const head_kind: spec_types.Kind = .mtp;
pub const nextDflash = @import("../dflash_round.zig").nextDflash;

/// The trained heads; per-request rows live in `mimo_mtp.State`.
pub const Head = *mimo_mtp.Head;

/// The heads loaded beside the trunk travel as `aux`.
pub fn headOf(_: *Transformer, _: bool, aux: ?*anyopaque) ?Head {
    return if (aux) |p| @ptrCast(@alignCast(p)) else null;
}

/// A request's head rows, and the draft step its next forward runs.
pub const Cache = struct {
    st: *mimo_mtp.State,
    next_step: usize = 0,

    pub fn activate(_: *const Cache) void {}

    pub fn step(self: *const Cache) usize {
        return self.st.step();
    }

    /// Head rows are rebuilt from the prompt's last window; a snapshot would carry nothing more.
    pub fn kv(_: *const Cache) ?*KVCache {
        return null;
    }

    /// The prompt's last window rebuilds the head rows, so the final prefill chunk's storage is not evaluated.
    pub const rebuilt_from_tail = true;

    pub fn truncate(self: *Cache, len: usize, s: mlx.mlx_stream) !void {
        try self.st.truncate(s, len);
        self.next_step = 0;
    }

    pub fn deinit(self: *const Cache) void {
        const allocator = self.st.allocator;
        self.st.deinit();
        allocator.destroy(self.st);
    }

    pub fn appendEvalArrays(self: *const Cache, vec: mlx.mlx_vector_array) void {
        self.st.appendEvalArrays(vec);
    }
};

pub fn moduleOwned(_: Head) bool {
    return false;
}

/// MiMo's heads are a separately allocated object; an in-trunk head is owned by the Transformer.
pub fn destroyHead(h: Head, allocator: std.mem.Allocator) void {
    h.deinit();
    allocator.destroy(h);
}

pub fn makeCache(h: Head, allocator: std.mem.Allocator) !Cache {
    return .{ .st = try h.newState(allocator) };
}

fn freeArray(a: mlx.mlx_array) void {
    _ = mlx.mlx_array_free(a);
}

/// Draft steps chain no hidden: step 0 carries the round's committed rows (`hidden` the target's), each later step
/// only its draft. The step index travels in `hidden_next`, a scalar the next call recognises by its rank.
pub fn forward(h: Head, target: *Transformer, ref: *Cache, id_arr: mlx.mlx_array, hidden: mlx.mlx_array, rope_offset_in: c_int, want: mtp_mod.StepWant, _: ?mtp_mod.MropeContext, host_ids: ?[]const u32) !mtp_mod.StepOut {
    const rope_offset: usize = @intCast(rope_offset_in);
    const chained = mlx.mlx_array_ndim(hidden) != 3;
    if (want == .none) {
        try h.appendHistory(target, ref.st, host_ids orelse return error.MimoMtpNeedsHostIds, hidden, rope_offset);
        return .{ .logits = .{ .ctx = null }, .hidden_next = mlx.mlx_array_new_int(0) };
    }
    const step_i: usize = if (chained) ref.next_step else 0;
    const last = if (chained)
        try h.draftStep(target, ref.st, step_i, id_arr, &.{}, null, rope_offset)
    else
        try h.draftStep(target, ref.st, 0, null, host_ids orelse return error.MimoMtpNeedsHostIds, hidden, rope_offset);
    ref.next_step = step_i + 1;
    return stepOut(target, freeArray, last, step_i + 1, want);
}

/// Hands a draft step's last row over: through the trunk's lm_head (then freed) under `.logits`, else as
/// `rerank_x`. Takes `last`; on error nothing leaks.
pub fn stepOut(target: anytype, comptime free: fn (mlx.mlx_array) void, last: mlx.mlx_array, step_next: usize, want: mtp_mod.StepWant) !mtp_mod.StepOut {
    errdefer free(last);
    const hidden_next = mlx.mlx_array_new_int(@intCast(step_next));
    errdefer free(hidden_next);
    if (want != .logits) return .{ .logits = .{ .ctx = null }, .hidden_next = hidden_next, .rerank_x = last };
    const logits = try target.lmHeadLogits(last);
    free(last);
    return .{ .logits = logits, .hidden_next = hidden_next };
}

pub fn appendHistory(h: Head, target: *Transformer, cache: *Cache, token_ids: []const u32, hidden: mlx.mlx_array, rope_offset: c_int, _: ?mtp_mod.MropeContext, _: std.mem.Allocator) !void {
    try h.appendHistory(target, cache.st, token_ids, hidden, @intCast(rope_offset));
}

pub fn canRerankDrafts(h: Head) bool {
    return h.canRerankDrafts();
}

pub fn draftSelect(h: Head, target: *Transformer, x: mlx.mlx_array, suppress_mask: ?mlx.mlx_array) !mlx.mlx_array {
    return h.draftSelect(target, x, suppress_mask);
}

pub fn draftShortlist(h: Head, target: *Transformer, x: mlx.mlx_array, suppress_mask: ?mlx.mlx_array) !?mtp_mod.Shortlist {
    return h.draftShortlist(target, x, suppress_mask);
}

/// Every G17/NAX cost surface is calibrated against the qwen4_exp in-checkpoint head, so every other head plans
/// under `generic` unless it has a surface of its own.
pub fn costProfile(_: Head, target: *const Transformer, _: trunk_mod.KVQuantConfig) mtp_mod.MtpCostProfile {
    return if (target.config.expert_layout == .exl3_k4) .mimo_exl3 else .generic;
}

pub fn evSeed(h: Head) ?spec_types.EvSeed {
    const a = h.ev_seed_accept orelse return null;
    return .{ .accept = a, .m_lo = h.ev_seed_m_lo };
}

pub fn setEvSeed(h: Head, accept: [mtp_mod.MAX_DEPTH]f32, m_lo: u32) void {
    h.ev_seed_accept = accept;
    h.ev_seed_m_lo = m_lo;
}
