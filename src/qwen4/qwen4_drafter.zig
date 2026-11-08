//! Flash-Next's drafters: the in-checkpoint MTP head (`Head`, `Cache`) behind `arch/drafter.zig`.
const std = @import("std");
const qwen4_forward = @import("qwen4_forward.zig");
const mlx = @import("../mlx.zig");
const trunk_mod = @import("../trunk.zig");
const mtp_mod = @import("../mtp.zig");
const spec_types = @import("../spec_types.zig");
const Trunk = trunk_mod.Trunk;
const KVCache = trunk_mod.KVCache;
/// DFlash2 is opt-in on a Qwen pack that ships a block drafter.
pub const kinds = [_]spec_types.Kind{ .mtp, .dflash2 };
pub const head_kind: spec_types.Kind = .mtp;

/// Flash-Next verifies a DFlash2 block by rows that cost expert bytes, so the round picks its rows.
pub const dflash_adaptive_rows = true;

/// Chains of several requests build in one head forward.
pub const batches_head_rows = true;
pub const nextDflash = @import("../dflash_round.zig").nextDflash;

/// The head and its history live on the Trunk (`qwen4_mtp`, module-owned, so single-flight); row r of the
/// history is (pre-mixer stream at position r, token r+1), query position r+1.
pub const Head = *Trunk;

/// This arch's head for a loaded model: the in-checkpoint head, when MTP is on and the pack ships one.
pub fn headOf(xfm: *Trunk, mtp_enabled: bool, _: ?*anyopaque) ?Head {
    return if (mtp_enabled and xfm.arch_state.qwen4.mtp != null) xfm else null;
}

pub const buildBatchedChains = @import("qwen4_mtp_batch.zig").build;

/// The in-checkpoint head plus this request's own half of it.
pub const Cache = struct {
    t: *Trunk,
    st: *qwen4_forward.Qwen4MtpState,
    allocator: std.mem.Allocator,

    /// Installs this request's state on the head.
    pub fn activate(self: *const Cache) void {
        qwen4_forward.qwen4MtpActivate(self.t, self.st);
    }

    pub fn batchState(self: *const Cache) *qwen4_forward.Qwen4MtpState {
        qwen4_forward.qwen4MtpRelease(self.t, self.st);
        return self.st;
    }

    pub fn step(self: *const Cache) usize {
        self.activate();
        return self.t.arch_state.qwen4.mtp.?.seq_offset;
    }

    /// The KVCache the prefix cache snapshots; the head's state is not KV-only, so a caller must also carry `head()`.
    /// Null when head persistence is off.
    pub fn kv(self: *const Cache) ?*KVCache {
        self.activate();
        return if (persistEnabled()) &self.t.arch_state.qwen4.mtp.?.cache else null;
    }

    /// The Trunk owning the head; null whenever `kv()` is null.
    pub fn head(self: *const Cache) ?*Trunk {
        self.activate();
        return if (persistEnabled()) self.t else null;
    }

    pub fn truncate(self: *const Cache, len: usize, _: mlx.mlx_stream) !void {
        self.activate();
        try qwen4_forward.qwen4MtpTruncate(self.t, len);
    }

    /// How far the head's position has drifted from where the next round would read it: a serial block advances the
    /// trunk while the head's `seq_offset` stays put. `off0` is the round's first head row.
    pub fn positionDrift(self: *const Cache, off0: usize) i64 {
        self.activate();
        const m = &(self.t.arch_state.qwen4.mtp orelse return 0);
        return spec_types.headPositionDrift(@intCast(m.pos_base), m.seq_offset, @intCast(off0 + 1));
    }

    /// A forced commit ran plain through the trunk, leaving a hole in the history; the head's positions refuse a
    /// gap, so it starts over.
    pub fn reseedAfterForcedCommit(self: *const Cache) !void {
        self.activate();
        try qwen4_forward.qwen4MtpReset(self.t);
    }

    pub fn deinit(self: *const Cache) void {
        qwen4_forward.qwen4MtpRelease(self.t, self.st);
        self.st.deinit();
        self.allocator.destroy(self.st);
    }

    /// Adds this cache's storage to a prefill eval batch, so the chunk's graph is freed with the chunk.
    pub fn appendEvalArrays(self: *const Cache, vec: mlx.mlx_vector_array) void {
        self.activate();
        self.t.arch_state.qwen4.mtp.?.cache.appendEvalArrays(vec);
    }
};

var persist_env: ?bool = null;

/// Only a literal "0" turns head-history persistence off.
pub fn persistFromEnv(raw: ?[]const u8) bool {
    const v = raw orelse return true;
    return !std.mem.eql(u8, v, "0");
}

/// `SUSHI_MTP_HEAD_PERSIST=0`: the head's committed history is neither snapshotted into a prefix-cache entry nor
/// restored from one.
pub fn persistEnabled() bool {
    if (persist_env) |v| return v;
    const raw = std.c.getenv("SUSHI_MTP_HEAD_PERSIST");
    const on = persistFromEnv(if (raw) |r| std.mem.sliceTo(r, 0) else null);
    persist_env = on;
    return on;
}

/// Per-request state swaps onto the module (`Qwen4MtpState`); nothing is shared.
pub fn moduleOwned(_: Head) bool {
    return false;
}

pub fn makeCache(t: Head, allocator: std.mem.Allocator) !Cache {
    const st = try allocator.create(qwen4_forward.Qwen4MtpState);
    errdefer allocator.destroy(st);
    st.* = try qwen4_forward.qwen4MtpStateNew(t);
    qwen4_forward.qwen4MtpActivate(t, st);
    return .{ .t = t, .st = st, .allocator = allocator };
}

/// The head projects only what the caller consumes: this step wants the LAST row (`want_logits`) or no logits at
/// all, never the S-row block.
pub fn forward(t: Head, _: *Trunk, cache: *Cache, id_arr: mlx.mlx_array, hidden: mlx.mlx_array, rope_offset: c_int, want: mtp_mod.StepWant, mrope_ctx: ?mtp_mod.MropeContext, _: ?[]const u32) !mtp_mod.StepOut {
    cache.activate();
    return step(t, id_arr, hidden, rope_offset, want, mrope_ctx);
}

fn step(t: Head, id_arr: mlx.mlx_array, hidden: mlx.mlx_array, rope_offset: c_int, want: mtp_mod.StepWant, mrope_ctx: ?mtp_mod.MropeContext) !mtp_mod.StepOut {
    // `.mixed` is NOT `.none`: a rerank draft still needs the mixer output, which is the vector the lm_head
    // consumes. `hidden_next` is the PRE-mixer stream here and would be a silent shape error there.
    const project: qwen4_forward.Qwen4MtpProject = switch (want) {
        .logits => .last_row,
        .mixed => .mixed_last_row,
        .none => .none,
    };
    const out = try qwen4_forward.qwen4MtpForward(t, hidden, id_arr, rope_offset + 1, mrope_ctx, project);
    return .{ .logits = out.logits, .hidden_next = out.stream, .rerank_x = out.mixed };
}

pub fn appendHistory(t: Head, _: *Trunk, cache: *Cache, token_ids: []const u32, hidden: mlx.mlx_array, rope_offset: c_int, mrope_ctx: ?mtp_mod.MropeContext, allocator: std.mem.Allocator) !void {
    var ids_buf: [mtp_mod.MAX_DEPTH + 1]i32 = undefined;
    const on_heap = token_ids.len > ids_buf.len;
    const ids_i32 = if (on_heap) try allocator.alloc(i32, token_ids.len) else ids_buf[0..token_ids.len];
    defer if (on_heap) allocator.free(ids_i32);
    for (token_ids, 0..) |tok, i| ids_i32[i] = @intCast(tok);
    const shape = [_]c_int{@intCast(token_ids.len)};
    const id_arr = mlx.mlx_array_new_data(ids_i32.ptr, &shape, 1, .int32);
    defer _ = mlx.mlx_array_free(id_arr);
    cache.activate();
    const out = try step(t, id_arr, hidden, rope_offset, .none, mrope_ctx);
    _ = mlx.mlx_array_free(out.hidden_next);
}

/// Built lazily on the first ask: `loadQwen4Mtp` runs while the Trunk is still under construction and its
/// lm_head, which the coarse head is a copy of, is not assigned yet.
pub fn canRerankDrafts(t: Head) bool {
    return qwen4_forward.qwen4DraftRerankReady(t);
}

/// `x` is the MIXER output the target's lm_head consumes, never the pre-mixer `[B,S,hc*H]` stream.
pub fn draftSelect(t: Head, _: *Trunk, x: mlx.mlx_array, suppress_mask: ?mlx.mlx_array) !mlx.mlx_array {
    return qwen4_forward.qwen4DraftSelect(t, x, suppress_mask);
}

pub fn draftShortlist(t: Head, _: *Trunk, x: mlx.mlx_array, suppress_mask: ?mlx.mlx_array) !?mtp_mod.Shortlist {
    return qwen4_forward.qwen4DraftShortlist(t, x, suppress_mask);
}

pub fn costProfile(t: Head, _: *const Trunk, kv: trunk_mod.KVQuantConfig) mtp_mod.MtpCostProfile {
    return mtp_mod.qwen4G17CostProfileForKv(t, kv);
}

pub fn evSeed(t: Head) ?spec_types.EvSeed {
    const m = if (t.arch_state.qwen4.mtp) |*m| m else return null;
    const a = m.ev_seed_accept orelse return null;
    return .{ .accept = a, .m_lo = m.ev_seed_m_lo };
}

/// A model served with `--no-mtp` has no head to seed.
pub fn setEvSeed(t: Head, accept: [mtp_mod.MAX_DEPTH]f32, m_lo: u32) void {
    if (t.arch_state.qwen4.mtp) |*m| {
        m.ev_seed_accept = accept;
        m.ev_seed_m_lo = m_lo;
    }
}

test "MTP head persistence kill switch: only a literal 0 turns it off" {
    try std.testing.expect(persistFromEnv(null));
    try std.testing.expect(persistFromEnv(""));
    try std.testing.expect(persistFromEnv("1"));
    try std.testing.expect(!persistFromEnv("0"));
}
