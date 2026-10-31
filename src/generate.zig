const std = @import("std");
const mlx = @import("mlx.zig");
const transformer_mod = @import("transformer.zig");
const qwen4_forward = @import("qwen4/qwen4_forward.zig");
const tokenizer_mod = @import("tokenizer.zig");
const model_mod = @import("model.zig");
const log = @import("log.zig");
const json_grammar = @import("json_grammar.zig");
const json_schema = @import("json_schema.zig");
const token_mask = @import("token_mask.zig");
const rp_mod = @import("reasoning_protocol.zig");
const io_util = @import("io_util.zig");
const mtp_lookup = @import("mtp_lookup.zig");
const mtp_mod = @import("mtp.zig");
const mimo_mtp = @import("mimo2/mimo_mtp.zig");
const mtp_acceptance = @import("mtp_acceptance.zig");
const model_settings = @import("model_settings.zig");
const round_cost = @import("round_cost.zig");
const spec_cost_gate = @import("spec_cost_gate.zig");
const group_planner = @import("mtp_group_planner.zig");
const depth_bounds = @import("mtp_depth_bounds.zig");
const scheduler_mod = @import("scheduler.zig");
const think_penalty = @import("think_penalty.zig");
const seen_penalty = @import("seen_penalty.zig");
const logit_bias = @import("logit_bias.zig");

const Transformer = transformer_mod.Transformer;
const Tokenizer = tokenizer_mod.Tokenizer;
const ForwardCtx = transformer_mod.ForwardCtx;
const SSMCacheEntrySnapshot = transformer_mod.SSMCacheEntrySnapshot;
const ssmSnapshot = transformer_mod.ssmSnapshot;
const ssmSnapshotDeinit = transformer_mod.ssmSnapshotDeinit;
const ssmRestore = transformer_mod.ssmRestore;
const SSMCheckpoint = transformer_mod.SSMCheckpoint;
const captureSsmCheckpoint = transformer_mod.captureSsmCheckpoint;
const glm5_forward = @import("glm5/glm5_forward.zig");
const glm5_prefix = @import("glm5/glm5_prefix.zig");
const dsv41_prefix = @import("dsv41/dsv41_prefix.zig");
const dflash_mod = @import("dflash.zig");
const dflash_policy = @import("dflash_policy.zig");
const dspark_rows = @import("dsv41/dspark_rows.zig");
const DflashModel = dflash_mod.DflashModel;
const KVCache = transformer_mod.KVCache;
const glm5_generate = @import("glm5/glm5_generate.zig");
const sampler = @import("sampler.zig");
const prefill_plan = @import("prefill_plan.zig");
const mtp_plan = @import("mtp_plan.zig");
const degenerate = @import("degenerate.zig");
const dsv41_dspark = @import("dsv41/dsv41_dspark.zig");
const mtp_policy = @import("mtp_policy.zig");
const mtp_round = @import("mtp_round.zig");

/// Module-level overrides for prefill behavior. Defaults match the original
/// hardcoded values; main.zig may overwrite these from CLI flags before
/// `serve()` runs. Per-request reads happen on the same thread that did the
/// CLI parse, so no atomicity needed.
pub var prefill_chunk_override: usize = 8192;

/// Set by `--prefill-chunk` in main.zig. An operator-chosen width outranks the
/// per-model chunk the memory sizer pins (`ModelConfig.pinned_prefill_chunk`);
/// without this flag the default 8192 is indistinguishable from a request.
/// Same set-once-at-CLI-parse contract as `prefill_chunk_override`.
pub var prefill_chunk_explicit: bool = false;

/// Set once by the serve CLI before request construction. Direct callers may
/// override it through InitOptions; exact remains the library default.
pub var mtp_acceptance_default: mtp_acceptance.Mode = .exact;
/// `--mtp-typical` / `--mtp-tokenv3` chose the default above.
pub var mtp_acceptance_explicit: bool = false;

/// `--mtp-greedy-tail` was given.
pub var mtp_greedy_tail_explicit: bool = false;

/// `--no-mtp-lookup`: no prompt-lookup drafts in MTP rounds or GLM DFlash2 chains .
pub var mtp_lookup_disabled: bool = false;

pub fn tickUbenchN(raw: ?[*:0]const u8) u32 {
    const v = raw orelse return 0;
    if (v[0] == '0') return 0;
    return std.fmt.parseInt(u32, std.mem.sliceTo(v, 0), 10) catch 0;
}

var tick_ubench_left: ?u32 = null;

pub fn tickUbenchArmed() bool {
    if (tick_ubench_left == null) {
        tick_ubench_left = tickUbenchN(std.c.getenv("SUSHI_DECODE_TICK_UBENCH"));
    }
    return tick_ubench_left.? > 0;
}

fn tickUbenchConsume() void {
    if (tick_ubench_left) |*n| {
        if (n.* > 0) n.* -= 1;
    }
}

const TickKind = enum(u2) { fwd, sample, resolve, eval };

const TickProf = struct {
    on: bool,
    wall: io_util.Stopwatch,
    ns: [4]u64 = @splat(0),

    fn begin() TickProf {
        if (!tickUbenchArmed()) return .{ .on = false, .wall = undefined };
        return .{ .on = true, .wall = io_util.Stopwatch.init(std.Io.Threaded.global_single_threaded.io()) };
    }

    fn mark(self: *const TickProf) u64 {
        if (!self.on) return 0;
        return self.wall.read();
    }

    fn add(self: *TickProf, kind: TickKind, t0: u64) void {
        if (!self.on) return;
        self.ns[@backingInt(kind)] += self.wall.read() - t0;
    }

    fn finish(self: *TickProf) void {
        if (!self.on) return;
        const wall = self.wall.read();
        const fwd = self.ns[@backingInt(TickKind.fwd)];
        const sample = self.ns[@backingInt(TickKind.sample)];
        const resolve = self.ns[@backingInt(TickKind.resolve)];
        const eval = self.ns[@backingInt(TickKind.eval)];
        const accounted = fwd + sample + resolve + eval;
        const rest = if (wall > accounted) wall - accounted else 0;
        const ms = struct {
            fn f(n: u64) f64 {
                return @as(f64, @floatFromInt(n)) / 1e6;
            }
        }.f;
        log.info("[tick-ubench] wall={d:.3} ms fwd={d:.3} sample={d:.3} resolve={d:.3} eval={d:.3} rest={d:.3}\n", .{
            ms(wall),
            ms(fwd),
            ms(sample),
            ms(resolve),
            ms(eval),
            ms(rest),
        });
        tickUbenchConsume();
    }
};

/// `--max-mtp-ctx N`: MTP stays off past N context tokens (0 = unlimited). Read by the HTTP
/// admission gate and by `nextMtp` per round (`mtp.mtpCtxWithinLimit`).
pub var max_mtp_ctx: u32 = 0;

/// Truthy if the env var is exactly "1". Anything else (unset, "0", "true",
/// "yes") is false — keep matching surface tight to avoid surprises.
pub fn readEnvBool(name: [:0]const u8) bool {
    const raw = std.c.getenv(name.ptr);
    if (raw == null) return false;
    const slice = std.mem.sliceTo(raw.?, 0);
    return std.mem.eql(u8, slice, "1");
}

fn freeArray(a: mlx.mlx_array) void {
    _ = mlx.mlx_array_free(a);
}

/// Grammar-constrained sampling state. The caller owns `grammar`, `token_bytes`,
/// and `mask_buf`; the generator only reads them. `mask_buf.len` must equal
/// `token_bytes.bytes.len` (the tokenizer's vocab size).
/// Which MTP head a generator drives.
///
/// The controller around it is head-agnostic — the EV planner, the acceptance
/// gate, `mtpBatchedAcceptGraph`, the pre-draft and the horizon valve all work
/// in tokens and probabilities — so the split is exactly these five operations
/// and nothing else.
pub const MtpHeadRef = union(enum) {
    /// qwen4_exp: the head and its history live on the Transformer
    /// (`qwen4_mtp`, module-owned ⇒ single-flight); row r of the history is
    /// (pre-mixer stream at position r, token r+1), query position r+1.
    qwen4: *Transformer,
    /// mimo_v2: the checkpoint's trained heads; per-request rows live in `mimo_mtp.State`.
    mimo: *mimo_mtp.Head,
    /// deepseek_v41: the DSpark stages and their windows live on the model's own state, so a request that
    /// drafts with them (`nextDspark`) holds the model exclusively.
    dsv41: *Transformer,

    /// Is this head's decode state module-owned (one per model, `Qwen4Mtp.cache`) rather
    /// than per-request? Both the scheduler and the sticky-serial arm ask this.
    pub fn moduleOwned(self: MtpHeadRef) bool {
        return switch (self) {
            // Per-request state swaps onto the module (`Qwen4MtpState`); nothing is shared.
            .qwen4, .mimo => false,
            // The model's own request state; its slot is exclusive through `ownsModuleDecodeState`.
            .dsv41 => false,
        };
    }

    pub fn makeCache(self: MtpHeadRef, allocator: std.mem.Allocator) !MtpCacheRef {
        return switch (self) {
            .qwen4 => |t| blk: {
                const st = try allocator.create(transformer_mod.Transformer.Qwen4MtpState);
                errdefer allocator.destroy(st);
                st.* = try t.qwen4MtpStateNew();
                t.qwen4MtpActivate(st);
                break :blk .{ .qwen4 = .{ .t = t, .st = st, .allocator = allocator } };
            },
            .mimo => |h| .{ .mimo = .{ .st = try h.newState(allocator) } },
            .dsv41 => error.Dsv41DsparkHasNoHeadCache,
        };
    }

    /// The head projects only what the caller consumes: this step wants the
    /// LAST row (`want_logits`) or no logits at all, never the S-row block —
    /// the merged history step after a partial accept is `1 + accepted` rows
    /// wide and a full-block mixer + 248320-wide lm_head over it is thrown
    /// away but for one row.
    fn qwen4Step(t: *Transformer, id_arr: mlx.mlx_array, hidden: mlx.mlx_array, rope_offset: c_int, want: mtp_mod.StepWant, mrope_ctx: ?mtp_mod.MropeContext) !mtp_mod.StepOut {
        // `.mixed` is NOT `.none`: a rerank draft still needs the mixer output,
        // which is the vector the lm_head consumes. `hidden_next` is the
        // PRE-mixer stream on this arm and would be a silent shape error there.
        const project: Transformer.Qwen4MtpProject = switch (want) {
            .logits => .last_row,
            .mixed => .mixed_last_row,
            .none => .none,
        };
        const out = try t.qwen4MtpForward(hidden, id_arr, rope_offset + 1, mrope_ctx, project);
        return .{ .logits = out.logits, .hidden_next = out.stream, .rerank_x = out.mixed };
    }

    /// One head forward over L positions. `.logits` returns the LAST row only.
    /// `.mixed` asks for the lm_head's INPUT vector instead: on
    /// qwen4_exp it is the mixer output, returned in `rerank_x`. `host_ids` are
    /// `id_arr`'s values when the caller holds them (a round's committed rows);
    /// the MiMo arm records them.
    pub fn forward(
        self: MtpHeadRef,
        target: *Transformer,
        cache: *MtpCacheRef,
        id_arr: mlx.mlx_array,
        hidden: mlx.mlx_array,
        rope_offset: c_int,
        want: mtp_mod.StepWant,
        mrope_ctx: ?mtp_mod.MropeContext,
        host_ids: ?[]const u32,
    ) !mtp_mod.StepOut {
        return switch (self) {
            .mimo => |h| mimoStep(h, target, &cache.mimo, id_arr, hidden, @intCast(rope_offset), want, host_ids),
            .qwen4 => |t| blk: {
                cache.activate();
                break :blk qwen4Step(t, id_arr, hidden, rope_offset, want, mrope_ctx);
            },
            .dsv41 => error.Dsv41DsparkHasNoHeadStep,
        };
    }

    /// MiMo draft steps chain no hidden: step 0 carries the round's committed rows
    /// (`hidden` the target's), each later step only its draft. The step index
    /// travels in `hidden_next`, a scalar the next call recognises by its rank.
    fn mimoStep(h: *mimo_mtp.Head, target: *Transformer, ref: *MtpCacheRef.MimoRef, id_arr: mlx.mlx_array, hidden: mlx.mlx_array, rope_offset: usize, want: mtp_mod.StepWant, host_ids: ?[]const u32) !mtp_mod.StepOut {
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
        return mimoStepOut(target, freeArray, last, step_i + 1, want);
    }

    /// Hands a draft step's last row over: through the trunk's lm_head (then freed)
    /// under `.logits`, else as `rerank_x`. Takes `last`; on error nothing leaks.
    pub fn mimoStepOut(target: anytype, comptime free: fn (mlx.mlx_array) void, last: mlx.mlx_array, step_next: usize, want: mtp_mod.StepWant) !mtp_mod.StepOut {
        errdefer free(last);
        const hidden_next = mlx.mlx_array_new_int(@intCast(step_next));
        errdefer free(hidden_next);
        if (want != .logits) return .{ .logits = .{ .ctx = null }, .hidden_next = hidden_next, .rerank_x = last };
        const logits = try target.lmHeadLogits(last);
        free(last);
        return .{ .logits = logits, .hidden_next = hidden_next };
    }

    /// Append committed history without projecting logits.
    pub fn appendHistory(
        self: MtpHeadRef,
        target: *Transformer,
        cache: *MtpCacheRef,
        token_ids: []const u32,
        hidden: mlx.mlx_array,
        rope_offset: c_int,
        mrope_ctx: ?mtp_mod.MropeContext,
        allocator: std.mem.Allocator,
    ) !void {
        switch (self) {
            .mimo => |h| try h.appendHistory(target, cache.mimo.st, token_ids, hidden, @intCast(rope_offset)),
            .qwen4 => |t| {
                const ids_i32 = try allocator.alloc(i32, token_ids.len);
                defer allocator.free(ids_i32);
                for (token_ids, 0..) |tok, i| ids_i32[i] = @intCast(tok);
                const shape = [_]c_int{@intCast(token_ids.len)};
                const id_arr = mlx.mlx_array_new_data(ids_i32.ptr, &shape, 1, .int32);
                defer _ = mlx.mlx_array_free(id_arr);
                cache.activate();
                const out = try qwen4Step(t, id_arr, hidden, rope_offset, .none, mrope_ctx);
                _ = mlx.mlx_array_free(out.hidden_next);
            },
            .dsv41 => return error.Dsv41DsparkHasNoHeadStep,
        }
    }

    /// Draft-rerank scheme (see mtp.zig): greedy drafts pick via a coarse
    /// 2-bit readout + trunk-head top-32 re-score instead of a full
    /// draft-head projection + argmax.
    pub fn canRerankDrafts(self: MtpHeadRef) bool {
        return switch (self) {
            // Built lazily on the first ask: `loadQwen4Mtp` runs while the
            // Transformer is still under construction and its lm_head — the
            // very weight the coarse head is a copy of — is not assigned yet.
            .qwen4 => |t| t.qwen4DraftRerankReady(),
            .mimo => |h| h.canRerankDrafts(),
            .dsv41 => false,
        };
    }

    /// `x` is whatever the TARGET's lm_head consumes for this arm — qwen4_exp's
    /// MIXER output. Never `hidden_next` on qwen4_exp: that is the pre-mixer
    /// `[B,S,hc*H]` stream.
    pub fn draftSelect(
        self: MtpHeadRef,
        target: *Transformer,
        x: mlx.mlx_array,
        suppress_mask: ?mlx.mlx_array,
    ) !mlx.mlx_array {
        return switch (self) {
            .qwen4 => |t| t.qwen4DraftSelect(x, suppress_mask),
            .mimo => |h| h.draftSelect(target, x, suppress_mask),
            .dsv41 => error.Dsv41DsparkHasNoHeadStep,
        };
    }

    /// The exact re-scored top-32 shortlist a SAMPLED draft proposes from, or
    /// null when the coarse rerank head is unavailable. `x` is the same vector
    /// `draftSelect` takes.
    pub fn draftShortlist(
        self: MtpHeadRef,
        target: *Transformer,
        x: mlx.mlx_array,
        suppress_mask: ?mlx.mlx_array,
    ) !?mtp_mod.Shortlist {
        return switch (self) {
            .qwen4 => |t| t.qwen4DraftShortlist(x, suppress_mask),
            .mimo => |h| h.draftShortlist(target, x, suppress_mask),
            .dsv41 => null,
        };
    }

    /// Every G17/NAX cost surface is calibrated against one exact runtime
    /// geometry — the qwen4_exp in-checkpoint head on its own arm — so every
    /// other head plans under `generic`: an off-profile head served by a
    /// calibrated surface would plan depths the measurement never covered.
    pub fn costProfile(self: MtpHeadRef, target: *const Transformer, kv: transformer_mod.KVQuantConfig) mtp_mod.MtpCostProfile {
        return switch (self) {
            .qwen4 => |t| mtp_mod.qwen4G17CostProfileForKv(t, kv),
            .mimo => if (target.config.expert_layout == .exl3_k4) .mimo_exl3 else .generic,
            .dsv41 => .generic,
        };
    }

    /// The head's last healthy acceptance surface, or null when it has never
    /// published one. Every arm stores it on the head object itself — the
    /// in-checkpoint `Qwen4Mtp` and MiMo's `Head` — so a seed dies
    /// with the head and can never cross models.
    pub fn evSeed(self: MtpHeadRef) ?struct { accept: [mtp_mod.MAX_DEPTH]f32, m_lo: u32 } {
        return switch (self) {
            .mimo => |h| if (h.ev_seed_accept) |a| .{ .accept = a, .m_lo = h.ev_seed_m_lo } else null,
            .qwen4 => |t| blk: {
                if (t.qwen4_mtp) |*m| {
                    if (m.ev_seed_accept) |a| break :blk .{ .accept = a, .m_lo = m.ev_seed_m_lo };
                }
                break :blk null;
            },
            .dsv41 => null,
        };
    }

    pub fn setEvSeed(self: MtpHeadRef, accept: [mtp_mod.MAX_DEPTH]f32, m_lo: u32) void {
        switch (self) {
            .mimo => |h| {
                h.ev_seed_accept = accept;
                h.ev_seed_m_lo = m_lo;
            },
            .qwen4 => |t| {
                // A model served with `--no-mtp` has no head to seed.
                if (t.qwen4_mtp) |*m| {
                    m.ev_seed_accept = accept;
                    m.ev_seed_m_lo = m_lo;
                }
            },
            .dsv41 => {},
        }
    }
};

/// `SUSHI_MTP_HEAD_PERSIST=0`: the qwen4_exp head's committed history is neither
/// snapshotted into a prefix-cache entry nor restored from one.
var mtp_head_persist_env: ?bool = null;
/// Only a literal "0" turns persistence off.
pub fn mtpHeadPersistFromEnv(raw: ?[]const u8) bool {
    const v = raw orelse return true;
    return !std.mem.eql(u8, v, "0");
}
pub fn mtpHeadPersistEnabled() bool {
    if (mtp_head_persist_env) |v| return v;
    const raw = std.c.getenv("SUSHI_MTP_HEAD_PERSIST");
    const on = mtpHeadPersistFromEnv(if (raw) |r| std.mem.sliceTo(r, 0) else null);
    mtp_head_persist_env = on;
    return on;
}

/// The head's committed-history cache.
pub const MtpCacheRef = union(enum) {
    qwen4: Qwen4Ref,
    mimo: MimoRef,

    /// A MiMo request's head rows, and the draft step its next forward runs.
    pub const MimoRef = struct { st: *mimo_mtp.State, next_step: usize = 0 };

    /// The in-checkpoint head plus this request's own half of it.
    pub const Qwen4Ref = struct { t: *Transformer, st: *transformer_mod.Transformer.Qwen4MtpState, allocator: std.mem.Allocator };

    /// Install this request's state on the qwen4 head; no-op on the other arms.
    pub fn activate(self: *const MtpCacheRef) void {
        switch (self.*) {
            .mimo => {},
            .qwen4 => |r| r.t.qwen4MtpActivate(r.st),
        }
    }

    pub fn qwen4StateForBatch(self: *const MtpCacheRef) *Transformer.Qwen4MtpState {
        const ref = self.qwen4;
        ref.t.qwen4MtpRelease(ref.st);
        return ref.st;
    }

    pub fn step(self: *const MtpCacheRef) usize {
        return switch (self.*) {
            .qwen4 => |r| blk: {
                r.t.qwen4MtpActivate(r.st);
                break :blk r.t.qwen4_mtp.?.seq_offset;
            },
            .mimo => |r| r.st.step(),
        };
    }

    /// The underlying KVCache — what the prefix cache's spec-snap machinery
    /// snapshots and restores. The qwen4 head's state is not KV-only, so a caller taking this
    /// pointer must also carry `head()`. Null when head persistence is off.
    pub fn kv(self: *MtpCacheRef) ?*KVCache {
        return switch (self.*) {
            .qwen4 => |r| blk: {
                r.t.qwen4MtpActivate(r.st);
                break :blk if (mtpHeadPersistEnabled()) &r.t.qwen4_mtp.?.cache else null;
            },
            // Head rows are rebuilt from the prompt's last window; a snapshot would carry nothing more.
            .mimo => null,
        };
    }

    /// The Transformer owning the in-checkpoint head; null on the other arms and whenever `kv()` is null.
    pub fn head(self: *MtpCacheRef) ?*Transformer {
        return switch (self.*) {
            .mimo => null,
            .qwen4 => |r| blk: {
                r.t.qwen4MtpActivate(r.st);
                break :blk if (mtpHeadPersistEnabled()) r.t else null;
            },
        };
    }

    pub fn truncate(self: *MtpCacheRef, len: usize, s: mlx.mlx_stream) !void {
        switch (self.*) {
            .qwen4 => |r| {
                r.t.qwen4MtpActivate(r.st);
                try r.t.qwen4MtpTruncate(len);
            },
            .mimo => |*r| {
                try r.st.truncate(s, len);
                r.next_step = 0;
            },
        }
    }

    pub fn deinit(self: *MtpCacheRef) void {
        switch (self.*) {
            .qwen4 => |r| {
                r.t.qwen4MtpRelease(r.st);
                r.st.deinit();
                r.allocator.destroy(r.st);
            },
            .mimo => |r| {
                const allocator = r.st.allocator;
                r.st.deinit();
                allocator.destroy(r.st);
            },
        }
    }

    /// Append this cache's storage to a prefill eval batch, so the chunk's
    /// activation graph can be freed with the chunk instead of accumulating
    /// across the whole prompt.
    pub fn appendEvalArrays(self: *const MtpCacheRef, vec: mlx.mlx_vector_array) void {
        switch (self.*) {
            .qwen4 => |r| {
                r.t.qwen4MtpActivate(r.st);
                r.t.qwen4_mtp.?.cache.appendEvalArrays(vec);
            },
            .mimo => |r| r.st.appendEvalArrays(vec),
        }
    }
};

/// An MTP committed-history cache restored from the prefix cache, plus the
/// absolute target position its index 0 represents.
pub const MtpRestored = struct { cache: MtpCacheRef, base: usize };

pub const Constraint = struct {
    grammar: *json_grammar.Grammar,
    token_bytes: *const token_mask.TokenBytes,
    mask_buf: []bool,
    /// Final-answer grammars normally apply at token 0 (thinking-off and all
    /// prompt-committed content channels). A reasoning protocol instead makes
    /// generation start in the model's channel machinery: either inside an
    /// already-open reasoning block (prompt-opened) or at the unresolved
    /// opener choice. The protocol state drives the phase switches below.
    proto: ?*const rp_mod.Protocol = null,
    pstate: rp_mod.State = .{},
    /// Where the constrained payload begins in the generated stream — the
    /// authoritative reasoning→answer boundary the response emitters route
    /// on. Set exactly once, at the transition; `null` until then.
    pending_span: ?rp_mod.ConstraintSpan = null,
    /// The loaded model's drafter verifies rows under per-row masks (`row_grammar`), so this request may speculate.
    row_masks: bool = false,

    /// Whether DSpark rounds can serve this constraint: the protocol never reaches a channel header or an opener choice.
    pub fn rowsDraftable(self: *const Constraint) bool {
        if (!self.row_masks) return false;
        const proto = self.proto orelse return true;
        return proto.header_len == 0 and (self.pstate.phase == .reasoning or self.pstate.phase == .json_body);
    }
};

/// A forced recovery must leave room for the whole remaining transition AND
/// at least one constrained answer token. At the ordinary completion cap the
/// caller keeps the existing length-stop behavior — no reserve is created.
pub fn forcedBoundaryCanContinue(completion_tokens: u32, max_tokens: u32, transition_tokens: usize) bool {
    return completion_tokens +| @as(u32, @intCast(transition_tokens)) + 1 <= max_tokens;
}

/// RAII bundle for grammar-constrained sampling. Owns the parsed schema,
/// grammar state machine, and per-step mask buffer. The embedded `Constraint`
/// holds pointers/slices into the surrounding struct, so this struct must NOT
/// be moved after `initFromValue`. Construct via `var sc: SchemaConstraint =
/// undefined; try sc.initFromValue(...);` and pass `&sc.constraint` to
/// `SamplingParams`.
pub const SchemaConstraint = struct {
    schema: json_schema.Schema,
    grammar: json_grammar.Grammar,
    mask_buf: []bool,
    constraint: Constraint,
    allocator: std.mem.Allocator,

    /// Initialize in-place from a JSON schema value. On failure, any partial
    /// allocations made during this call are freed and the struct is left
    /// undefined (do not call `deinit`).
    pub fn initFromValue(
        self: *SchemaConstraint,
        allocator: std.mem.Allocator,
        schema_value: std.json.Value,
        token_bytes: *const token_mask.TokenBytes,
    ) !void {
        self.allocator = allocator;
        self.schema = try json_schema.parse(allocator, schema_value);
        errdefer self.schema.deinit();

        self.grammar = try json_grammar.Grammar.init(allocator, &self.schema);
        errdefer self.grammar.deinit();

        self.mask_buf = try allocator.alloc(bool, token_bytes.bytes.len);
        errdefer allocator.free(self.mask_buf);

        self.constraint = .{
            .grammar = &self.grammar,
            .token_bytes = token_bytes,
            .mask_buf = self.mask_buf,
        };
    }

    pub fn deinit(self: *SchemaConstraint) void {
        self.allocator.free(self.mask_buf);
        self.grammar.deinit();
        self.schema.deinit();
    }

    /// Route this request through a resolved reasoning protocol. Must be
    /// called before generation starts; the protocol is a per-request value
    /// the caller keeps alive (the Constraint borrows it).
    pub fn deferWithProtocol(self: *SchemaConstraint, proto: *const rp_mod.Protocol) void {
        self.constraint.proto = proto;
        self.constraint.pstate = proto.startState();
        if (self.constraint.pstate.phase == .json_body) self.constraint.pending_span = .{ .token_index = 0, .byte_offset = 0 };
    }
};

test "decode tick ubench count is off unless a positive integer" {
    try std.testing.expectEqual(@as(u32, 0), tickUbenchN(null));
    try std.testing.expectEqual(@as(u32, 0), tickUbenchN("0"));
    try std.testing.expectEqual(@as(u32, 1), tickUbenchN("1"));
    try std.testing.expectEqual(@as(u32, 8), tickUbenchN("8"));
}

test "forced recovery leaves room for the whole transition and one answer token" {
    try std.testing.expect(forcedBoundaryCanContinue(0, 2, 1));
    try std.testing.expect(forcedBoundaryCanContinue(8, 10, 1));
    try std.testing.expect(!forcedBoundaryCanContinue(0, 1, 1));
    try std.testing.expect(!forcedBoundaryCanContinue(9, 10, 1));
    // A multi-token transition plans against the WHOLE sequence.
    try std.testing.expect(forcedBoundaryCanContinue(0, 4, 3));
    try std.testing.expect(!forcedBoundaryCanContinue(0, 3, 3));
    // An in-place activation needs no transition token.
    try std.testing.expect(forcedBoundaryCanContinue(0, 1, 0));
}

/// A forced tool call (`tool_choice` required or named) enforced while
/// decoding: the thought runs free, and the position after its closer (or
/// token 0 when the prompt already closed it) is `forced` instead of a sample.
/// For `required` the name follows as one of `names`, the model choosing among
/// the ids that continue one. Handler-owned like `ThinkBound`; every successor
/// decision consults `decide`, which reads the published tokens incrementally.
pub const CallForce = struct {
    /// Separator ids, then the call's; `call_at` indexes the call, which alone
    /// is committed when whitespace already follows the closer.
    forced: []const u32,
    call_at: usize,
    closer_id: ?u32,
    /// Committed at token 0 in `.open`, where the template leaves the think
    /// opener to the model.
    opener: [1]u32 = .{0},
    phase: Phase,
    /// The token before the next position: the last prompt id until one is published.
    last: u32,
    /// Each declared function's name and closing delimiter as ids; empty when
    /// `forced` already names the function.
    names: []const []const u32 = &.{},
    /// The storage `names` slice into, when allocated.
    names_flat: []const u32 = &.{},
    /// Scratch for a branch's choices, at least `names.len` long.
    choice_buf: []u32 = &.{},
    name_from: ?usize = null,
    cursor: usize = 0,
    queue: []const u32 = &.{},

    pub const Phase = enum { open, thinking, answer, name, done };
    pub const Decision = union(enum) { sample, force: u32, choose: []const u32 };

    pub fn pending(self: *const CallForce) bool {
        return self.phase != .done or self.queue.len > 0;
    }

    /// What the next position must be.
    pub fn decide(self: *CallForce, published: []const u32) Decision {
        while (self.cursor < published.len) : (self.cursor += 1) {
            const id = published[self.cursor];
            if (self.phase == .thinking and self.closer_id != null and id == self.closer_id.?) self.phase = .answer;
            self.last = id;
        }
        if (self.queue.len == 0) switch (self.phase) {
            .open => {
                self.queue = &self.opener;
                self.phase = .thinking;
            },
            .answer => {
                const after_closer = self.closer_id != null and self.last == self.closer_id.?;
                self.queue = if (after_closer) self.forced else self.forced[self.call_at..];
                self.phase = if (self.names.len > 0) .name else .done;
            },
            .name => return self.nextNameToken(published),
            .thinking, .done => return .sample,
        };
        if (self.queue.len == 0) return .sample;
        const id = self.queue[0];
        self.queue = self.queue[1..];
        return .{ .force = id };
    }

    fn nextNameToken(self: *CallForce, published: []const u32) Decision {
        const from = self.name_from orelse published.len;
        self.name_from = from;
        const typed = published[from..];
        var n: usize = 0;
        var complete = false;
        for (self.names) |name| {
            if (name.len < typed.len or !std.mem.eql(u32, name[0..typed.len], typed)) continue;
            if (name.len == typed.len) {
                complete = true;
                break;
            }
            if (std.mem.indexOfScalar(u32, self.choice_buf[0..n], name[typed.len]) == null) {
                self.choice_buf[n] = name[typed.len];
                n += 1;
            }
        }
        if (!complete and n == 1) return .{ .force = self.choice_buf[0] };
        if (!complete and n > 1) return .{ .choose = self.choice_buf[0..n] };
        self.phase = .done;
        return .sample;
    }

    /// Frees what `server.armCallForce` allocated.
    pub fn deinit(self: *CallForce, allocator: std.mem.Allocator) void {
        allocator.free(self.forced);
        allocator.free(self.names_flat);
        allocator.free(self.names);
        allocator.free(self.choice_buf);
    }
};

/// Map the overthinking markers onto this tokenizer once per load (`think_penalty.markerIds`).
/// Never fails a load: an error leaves the mask null and the penalty inert.
pub fn installThinkMarkers(xfm: *Transformer, tok: *const Tokenizer) void {
    const ids = think_penalty.markerIds(xfm.allocator, &think_penalty.WORDS, tok) catch |err| {
        log.warn("[think-penalty] marker mapping failed ({s}); penalty off\n", .{@errorName(err)});
        return;
    };
    defer xfm.allocator.free(ids);
    const logits_dim: usize = xfm.config.vocab_size;
    xfm.think_marker_mask = think_penalty.buildMask(xfm.allocator, ids, logits_dim) catch |err| {
        log.warn("[think-penalty] mask build failed ({s}); penalty off\n", .{@errorName(err)});
        return;
    };
    log.debug("[think-penalty] {d} marker tokens from {d} words (multi-token spellings skipped)\n", .{ ids.len, think_penalty.WORDS.len });
}

/// A scalar in the array's own dtype, so a masked bf16 block is not widened to
/// f32 by its own `-inf`.
fn scalarLike(v: f32, like: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const raw = mlx.mlx_array_new_float(v);
    defer _ = mlx.mlx_array_free(raw);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_astype(&out, raw, mlx.mlx_array_dtype(like), s));
    return out;
}

/// A greedy spec-verify argmax whose entries are COMMITTED as next tokens.
/// `verifyArgmax` is its only constructor and every commit site takes this type,
/// so a raw `mlx_argmax_axis` cannot reach a commit without the reserved-id mask.
pub const CommittedArgmax = struct {
    arr: mlx.mlx_array = .{ .ctx = null },

    pub fn deinit(self: *CommittedArgmax) void {
        if (self.arr.ctx != null) _ = mlx.mlx_array_free(self.arr);
        self.arr = .{ .ctx = null };
    }

    pub fn present(self: CommittedArgmax) bool {
        return self.arr.ctx != null;
    }

    /// The lazy handle, for an eval vector or a forced eval - never for a commit.
    pub fn lazy(self: CommittedArgmax) mlx.mlx_array {
        return self.arr;
    }

    /// The realized ids; the caller must have evaluated `lazy()` first.
    pub fn ids(self: CommittedArgmax, len: usize) ![]const i32 {
        const data = mlx.mlx_array_data_int32(self.arr) orelse return error.MlxArrayDataNull;
        return data[0..len];
    }
};

/// The greedy spec-verify argmax, over the reserved-id mask when the model has
/// one. The sampled verify filters through `applySuppressMask` and the rerank
/// draft select takes the mask, so without this the greedy verify was the one
/// path that could commit a flagged special or a padding row past the defined
/// vocabulary — ids the serial sampler can never draw.
pub fn verifyArgmax(logits: mlx.mlx_array, mask: ?mlx.mlx_array, s: mlx.mlx_stream) !CommittedArgmax {
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    if (mask) |m| {
        const neg_inf = try scalarLike(-std.math.inf(f32), logits, s);
        defer _ = mlx.mlx_array_free(neg_inf);
        var masked = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(masked);
        try mlx.check(mlx.mlx_where(&masked, m, neg_inf, logits, s));
        try mlx.check(mlx.mlx_argmax_axis(&out, masked, 2, false, s));
    } else {
        try mlx.check(mlx.mlx_argmax_axis(&out, logits, 2, false, s));
    }
    return .{ .arr = out };
}

/// Generation result (for non-streaming use).
pub const GenerationResult = struct {
    text: []u8,
    token_ids: []u32,
    prompt_tokens: u32,
    completion_tokens: u32,
    finish_reason: []const u8,
    prefill_tps: f64,
    decode_tps: f64,
    /// Wall-clock nanoseconds spent on prefill (prompt processing).
    prefill_ns: u64 = 0,
    /// Wall-clock nanoseconds spent on decode (token generation).
    decode_ns: u64 = 0,
    /// Prompt tokens served from a KV-cache prefix (the hot prefix cache).
    /// `prompt_tokens - cached_tokens`
    /// is what was actually run through the model this turn, so `prefill_tps`
    /// reflects real compute rather than an inflated full-prompt rate.
    cached_tokens: u32 = 0,
    logprobs: ?[]sampler.LogprobResult = null, // per-token logprobs (caller must free)
    /// Non-null only when the degenerate-tail guard cut this generation:
    /// the `finish_details.type` value emitted beside `finish_reason`
    /// ("stop" for loop cuts — see `slot_outcome.loopStopReason`).
    /// Static string; nothing to free.
    finish_details: ?[]const u8 = null,
    /// Absolute byte offset into `text` where the constrained JSON payload
    /// begins, when a reasoning protocol was active and the payload began.
    /// 0 = direct answer. Null when no protocol ran, the payload never began
    /// (completion cap mid-reasoning), or the boundary fell outside the
    /// emitted (loop-trimmed) tokens.
    constraint_payload_byte: ?usize = null,
    /// Time the prefix lookup and restore took inside `prefill_ns`; null when no lookup ran.
    restore_ns: ?u64 = null,
    spec_mode: Generator.SpecMode = .none,
};

/// Throughput in tokens/sec. Returns 0 when no time elapsed so unmeasured paths
/// report 0 rather than inf / NaN.
pub fn tokensPerSec(tokens: u64, elapsed_ns: u64) f64 {
    if (elapsed_ns == 0) return 0.0;
    const tok_f: f64 = @floatFromInt(tokens);
    const ns_f: f64 = @floatFromInt(elapsed_ns);
    return tok_f * @as(f64, @floatFromInt(std.time.ns_per_s)) / ns_f;
}

/// True prefill compute throughput: divides by the tokens actually pushed through
/// the model (prompt minus the prefix served from KV cache). A near-full cache
/// hit therefore reports the small suffix's real rate, not an inflated
/// full-prompt number. With `cached_tokens == 0` this is just the full-prompt
/// rate, matching the pre-instrumentation behavior.
pub fn prefillTokensPerSec(prompt_tokens: u32, cached_tokens: u32, prefill_ns: u64) f64 {
    const uncached: u32 = if (prompt_tokens > cached_tokens) prompt_tokens - cached_tokens else 0;
    return tokensPerSec(uncached, prefill_ns);
}

/// A prefill checkpoint at absolute `pos`: the slot's SSM entries, the GLM request's KDA state, or DeepSeek's whole request state.
/// `to_ssd`: a write-through takes the point to the SSD tier before the next forward, so DeepSeek's shares its rows.
fn capturePrefillCheckpoint(allocator: std.mem.Allocator, xfm: *Transformer, ctx: *const ForwardCtx, glm: ?*glm5_forward.Request, pos: usize, to_ssd: bool) !SSMCheckpoint {
    if (xfm.dsv41) |m| return if (to_ssd and pos >= @import("kv_disk_cache.zig").MIN_PERSIST_TOKENS) dsv41_prefix.captureShared(allocator, m, pos) else dsv41_prefix.capture(allocator, m, pos);
    if (glm) |request| {
        if (request.offset != pos) return error.GlmCachePositionMismatch;
        return glm5_prefix.captureKda(allocator, request, xfm.s);
    }
    return captureSsmCheckpoint(allocator, ctx.ssm_entries.?, pos, xfm.s);
}

/// Reserve this request's cache capacity, and grow the restored layers to it, before the
/// prefill's first chunk writes (`KVCache.growToReservation`). Returns the reserved tokens.
pub fn reserveRequestCapacity(cache: *KVCache, config: *const model_mod.ModelConfig, total_ctx: u64, max_tokens: u64, chunk: u64, s: mlx.mlx_stream) !u64 {
    const reserved = prefill_plan.reservedPrefillTokens(config, total_ctx, max_tokens, chunk);
    cache.reserve(@intCast(reserved));
    try cache.growToReservation(s);
    return reserved;
}

/// Step-based generator. Call `init` to prefill, then `next` per token.
/// Uses a fully-lazy async pipeline matching mlx-lm: sample + next forward are
/// built as a single lazy computation graph, async_eval'd together. The GPU
/// never idles between token generation steps.
pub const Generator = struct {

    xfm: *Transformer,
    /// Forward-pass context. Stores per-request KVCache pointer, moe_seq_offset
    /// pointer, ssm_entries slice, vision_embeddings handle, and capture_hidden
    /// override. The legacy single-slot path uses `xfm.defaultCtx()` (pointing at
    /// the Transformer's own fields). Phase 2 concurrent batching constructs a
    /// per-slot ForwardCtx pointing at the slot's own KVCache, etc., so multiple
    /// generators can share one Transformer's weights without colliding on
    /// per-request state. Stored by value; `&self.ctx` is what we pass to
    /// `xfm.forwardWith` / `lazyForward` / drafter step.
    ctx: ForwardCtx,
    tok: *const Tokenizer,
    next_token_id: u32,
    step: u32,
    /// `step` at the last `mlx_clear_cache()`. Advanced only by `advanceStep`,
    /// which is the ONE place `step` may move.
    last_cache_clear_step: u32 = 0,
    max_tokens: u32,
    sampling: sampler.SamplingParams,
    prompt_tokens: u32,
    completion_tokens: u32,
    finish_reason: []const u8,
    done: bool,
    /// The EOS (or pad) that ended generation, when one did; never part of `generated_ids`' accounting.
    stop_token: ?u32 = null,
    eos_token_ids: []const u32,
    generated_ids: std.ArrayList(u32),
    /// Degenerate-tail detection must not reconsider a reasoning loop after
    /// its boundary has activated the final-answer grammar. The scheduler
    /// scans only this suffix; a later loop in the answer is still handled by
    /// the ordinary repetition-stop policy.
    loop_guard_start: usize = 0,
    loop_guard: degenerate.TailGuard = .{},
    consecutive_pad: u32 = 0, // count of consecutive token-0 (pad) generations
    timeout_ns: u64, // 0 = no timeout; measures SILENCE, not total time (see StallClock)
    stall: degenerate.StallClock = .{},
    timer: io_util.Stopwatch,
    logprobs_n: u32 = 0, // 0 = disabled, >0 = number of top_logprobs to return
    last_logprob: ?sampler.LogprobResult = null, // logprob result for the most recently returned token
    /// Logprobs for the token the NEXT `next()` call will return.
    ///
    /// The decode loop returns `next_token_id` and, in the same call, forwards
    /// it to sample its successor — so the result `sampleToken` hands back
    /// describes the token AFTER the one being returned. Publishing it as
    /// `last_logprob` there shifted the whole array by one: the caller zips
    /// token_ids with logprobs by index, so at temp 0 every entry reported the
    /// distribution of the token that FOLLOWED it (a one-token "OK" reply came
    /// back with `<|role_end|>` at rank 1). Seeded at init with t1's own
    /// distribution — t1 is sampled from the prefill's final forward, which
    /// this loop never sees.
    pending_logprob: ?sampler.LogprobResult = null,
    // Async pipeline state: pre-computed forward pass logits for next decode step
    pending_logits: mlx.mlx_array = .{},
    has_pending_logits: bool = false,
    // Deferred token: lazy array from async pipeline, eval'd at start of next iteration
    pending_token: mlx.mlx_array = .{},
    has_pending_token: bool = false,

    // ── Spec-decode shared state ──
    // Post-final-norm hidden state at the last produced token's position.
    // Owned by the Generator (freed in `deinit`). Captured during the prefill
    // final-token forward and every verify forward; the MTP head's h_prev seed.
    last_hidden: mlx.mlx_array = .{},
    has_last_hidden: bool = false,
    /// PRNG for the DFlash stochastic-verify accept test (probability-
    /// ratio requires a uniform draw per draft step). Seeded from
    /// `sampling.seed` when set, otherwise from system time at init.
    prng: std.Random.DefaultPrng = std.Random.DefaultPrng.init(0),

    // Owned copy of the lookup prompt ids: the MTP prompt-lookup index extends from
    // `prompt_ids_owned ++ generated_ids`, and the caller-supplied `prompt_ids` slice is freed after
    // `init` returns. `generated_ids` (above) tracks post-prefill tokens; `prompt_ids_owned` is the
    // immutable prefix.
    prompt_ids_owned: []u32 = &.{},
    /// Allocator that owns `prompt_ids_owned`. Stored so `deinit` can free it
    /// without requiring callers to thread the allocator a second time. (Other
    /// owned slices are freed via the `allocator` argument to `deinit` for
    /// historical reasons; this one is set during `initWithOptions`.)
    prompt_ids_alloc: ?std.mem.Allocator = null,
    /// Tokens proposed per round (the DFlash-resolved block size).
    drafter_block_size: u32 = 4,

    // ── DFlash block-drafter state ──
    // External block-parallel assistant (src/dflash.zig). When
    // `dflash != null`, callers use `nextDflash` instead of `next`. Weights
    // are owned by the server (per-model); the per-request context cache
    // (`dflash_ctx`) is OWNED by the Generator — built during prefill from
    // the trunk's capture_layers hiddens, freed in `deinit`.
    dflash: ?*DflashModel = null,
    dflash_ctx: ?dflash_mod.DflashCtx = null,
    glm_dflash_native: bool = false,
    /// Accepted share of a DFlash2 round's drafts, smoothed; the grouped tick prices draft rows with it.
    glm_draft_rate: f32 = 0.8,
    /// Verbatim-copy rounds (`glmLookupChain`) and the drafts they landed, for `[spec-stats]`.
    glm_lookup: struct { rounds: u64 = 0, landed: u64 = 0 } = .{},
    /// The GLM assistant window as prefill left it, committed with the prefix: every GLM restore
    /// point lies at or below the prompt end, which a window cropped at the reply's end can miss.
    glm_prefill_window: ?glm5_generate.GlmSpecWindow = null,
    /// Cumulative GLM DFlash2 drafts proposed and round phases, for `[spec-stats]`.
    glm_round: struct { drafted: u64 = 0, draft_ns: u64 = 0, verify_ns: u64 = 0, replay_ns: u64 = 0, commit_ns: u64 = 0 } = .{},
    /// Whether a DFlash2 round pays against a plain one-row tick for this request, and the plain ticks it ran.
    glm_gate: spec_cost_gate.Gate = .{},
    glm_plain: struct { ticks: u64 = 0, verify_ns: u64 = 0, replay_ns: u64 = 0, commit_ns: u64 = 0 } = .{},
    /// Effective block size (assistant config, capped by the machine's verify lanes).
    /// Drafts per round = dflash_block_size - 1.
    dflash_block_size: u32 = 0,
    /// Wall ms of this request's DFlash rounds, for the cost gate.
    dflash_wall_ms: f64 = 0,
    dflash_gate_window: glm5_generate.DflashGateWindow = .{},
    /// Drafts the verifies of this request carried, and the selector-confidence calibration that picks rows.
    dflash_drafts_verified: u64 = 0,
    dflash_calib: dflash_policy.Calibration = .{},
    /// Drafts this round ran (0 = serial), read by the table feed.
    dflash_round_width: u32 = 0,
    /// Stats: count of nextDflash calls that ran a verify forward.
    dflash_attempted: u64 = 0,
    /// Stats: cumulative draft tokens accepted (excluding always-accepted t1).
    dflash_accepted_tokens: u64 = 0,
    /// deepseek_v41: this request drafts with DSpark (`nextDspark`) over the model's own state.
    dspark: bool = false,
    /// Forces the verify width (the `SUSHI_DSV41_DSPARK_ROWS` diagnostic, and the tests).
    dspark_diag: dsv41_dspark.DsparkDiag = .off,
    dspark_policy: dspark_rows.Policy = .{},
    /// A DSpark round commits through the first draft with this id and no further, so the request's state
    /// stands right behind it for the scheduler to checkpoint. The scheduler clears it once taken.
    dspark_round_ends_at: ?u32 = null,
    /// Armed by the scheduler for a hybrid thinking request with tools; the restore point it takes is the last
    /// of `ssm_checkpoints` while `state` is `.taken`.
    thought_point: ?mtp_plan.ThoughtPoint = null,
    /// A constrained DSpark request: `next_token_id` is sampled under the grammar but not yet emitted or in the cache,
    /// and `has_pending_logits` is false. Serial decode (`next`) emits it first.
    dspark_pending: bool = false,
    /// One verify mask per row (`row_grammar.plan`), reused across rounds.
    dspark_masks: std.ArrayList(bool) = .empty,
    dspark_serial_logged: bool = false,
    dspark_stats: struct { rounds: u64 = 0, plain: u64 = 0, drafted: u64 = 0, accepted: u64 = 0, draft_ns: u64 = 0, verify_ns: u64 = 0, rows_hist: [dsv41_dspark.dsv41_max_drafts + 2]u32 = @splat(0), mask_ns: u64 = 0, serial: u64 = 0 } = .{},
    /// Per-phase wall-time trace (SUSHI_DFLASH_TRACE=1; else untouched).
    /// Unlike the MTP trace this one INSERTS eval barriers to attribute a
    /// fully-lazy round — a traced round is slower than a real one by
    /// whatever overlap the barriers destroy. Diagnostic only.
    dflash_trace: DflashTrace = .{},
    /// Trace-only stopwatch across the scheduler gap (round return → next
    /// round entry); bills the `.gap` phase. Null when not tracing.
    dflash_gap_watch: ?io_util.Stopwatch = null,

    // ── Qwen native MTP head state ──
    // The model's own one-layer multi-token-prediction head (src/mtp.zig).
    // When `mtp != null`, callers use `nextMtp` instead of `next`. The head
    // is owned by the server (loaded with the model); the Generator only
    // holds a non-owning pointer. `mtp_cache` is the head's committed-history
    // KV cache — OWNED by the Generator (built during prefill, freed in
    // `deinit`).
    mtp: ?MtpHeadRef = null,
    /// Installed verifier route; selected once at init, not checked per token.
    mtp_accept_graph: mtp_round.MtpGraphFn = mtpBatchedExactGraph,
    mtp_accept_prefix: mtp_acceptance.PrefixFn = mtp_acceptance.exactPrefix,
    mtp_accept_param: f32 = 0,
    /// A sampled request drafts only depth 0 from the draft sampler, every later depth by argmax.
    mtp_greedy_tail: bool = false,
    /// Does the model have a usable MTP head? `--no-mtp` clears it; a per-request `enable_mtp:false` does not.
    model_has_mtp: bool = false,
    mtp_cache: ?MtpCacheRef = null,
    /// Absolute target position represented by MTP-cache position 0. Usually
    /// zero; nonzero when the head keeps only a suffix of a restored/long
    /// prompt. Used to map the sidecar's relative offsets into Qwen M-RoPE.
    mtp_position_base: usize = 0,
    /// CONFIGURED max tokens drafted per round (verify length = depth + 1); the
    /// floor is `depth_bounds.active.min`.
    mtp_depth: u32 = mtp_mod.DEFAULT_DEPTH,
    /// The cap WITHOUT the per-silicon row: an explicit --mtp-max-depth, else
    /// the adaptive default. The row is the cold-start cap; once the table
    /// has trusted widths it may plan up to this instead (the M4 base row
    /// of 4 measured -6% against what the table found at 6).
    mtp_depth_free: u32 = mtp_mod.DEFAULT_DEPTH,
    /// CURRENT adaptive depth (see updateMtpDepth). Starts at the floor,
    /// demoted/promoted per windowed acceptance, always inside floor..`mtp_depth`.
    mtp_depth_current: u32 = mtp_mod.DEFAULT_DEPTH,
    /// Stats: count of nextMtp calls that ran a verify forward.
    mtp_attempted: u64 = 0,
    /// Stats: cumulative draft tokens accepted (excluding always-accepted t1).
    mtp_accepted_tokens: u64 = 0,
    /// Adaptive-depth moving window: per-round drafted/accepted counts.
    mtp_window_drafted: [mtp_policy.MTP_DEPTH_WINDOW]u8 = @splat(0),
    mtp_window_accepted: [mtp_policy.MTP_DEPTH_WINDOW]u8 = @splat(0),
    mtp_window_idx: u32 = 0,
    mtp_rounds_since_switch: u32 = 0,
    /// Rounds remaining during which promotion is blocked (set after a
    /// demotion so a failed depth excursion isn't immediately retried).
    mtp_promote_cooldown: u32 = 0,
    /// Cumulative drafted tokens across rounds. The EV controller varies m
    /// per round, so `attempts x depth` no longer measures proposals — this
    /// is the honest per_draft_pct denominator.
    mtp_drafted_tokens: u64 = 0,
    /// Prompt-lookup rounds (`mtp_lookup`), kept apart from the MTP counters so the head's
    /// acceptance stats stay the head's.
    mtp_lookup_rounds: u64 = 0,
    mtp_lookup_drafted: u64 = 0,
    mtp_lookup_accepted: u64 = 0,
    /// Prompt + committed output, indexed up to `mtp_lookup_gen` generated tokens.
    mtp_lookup_idx: ?mtp_lookup.Index = null,
    mtp_lookup_gen: usize = 0,
    /// Landed drafts per round, smoothed, for lookup and MTP rounds (`mtp_lookup.gate`).
    mtp_lookup_ema: f32 = @floatFromInt(mtp_lookup.MAX_DRAFT),
    mtp_round_ema: f32 = 0,
    /// Drafts per MTP round, smoothed: the width `mtp_round_ema`'s rounds actually ran.
    mtp_round_drafted_ema: f32 = 0,
    /// The previous round was a lookup round.
    mtp_after_lookup: bool = false,
    /// The last lookup round landed every draft (`mtp_lookup.gate` streak).
    mtp_lookup_streak: bool = false,
    /// Rounds where the confidence gate extended into chunk B.
    mtp_ext_rounds: u64 = 0,
    /// This request's round walls, for the `stalls=` field.
    mtp_round_log: round_cost.RoundLog = .{},
    /// Speculative rounds that rolled recurrent state back on a partial accept.
    partial_rounds: u64 = 0,
    /// Extension dry-spell gate: consecutive extension-CONSIDERED rounds
    /// whose confidence gate did not clear, and the single-chunk cooldown
    /// that a full dry streak triggers (see mtpExtDryAllows).
    mtp_ext_dry_streak: u32 = 0,
    mtp_ext_cooldown: u32 = 0,
    /// EV controller: conditional acceptance EMA per draft index,
    /// a[i] = P(draft i accepted | drafts 0..i-1 accepted). Optimistic prior;
    /// warmup rounds pull the low indices to reality before it can matter.
    mtp_ev_accept: [mtp_mod.MAX_DEPTH]f32 = @splat(mtp_policy.MTP_EV_PRIOR),
    /// Rounds seen by the EV controller (drives the legacy-behavior warmup).
    mtp_ev_rounds: u32 = 0,
    /// Last round's planned m_lo (base-depth climb damping: +1/round max).
    mtp_ev_m_lo_prev: u32 = 1,
    /// Round-cost surface selected once for this target+MTP head. The M5/G17
    /// NAX surface requires the measured trunk, native sidecar, and 3-bit
    /// draft-only head; every other combination keeps the M1-M4 surface.
    mtp_ev_costs: mtp_policy.MtpEvCosts = mtp_policy.MTP_EV_DEFAULT_COSTS,
    /// Live-cost EMAs (ms), 0 = unseeded. `mtp_ev_sync_ms` is the measured
    /// chunk-A confidence-read sync — updated ONLY on extension-considered
    /// rounds, so it holds the true cost of a sync when one happens (not the
    /// suppressed-round average). `mtp_ev_round_ms` is the realized round
    /// wall-clock. Their ratio drives the cost-aware exploration throttle
    /// (mtpExtDryThresholdFor) so a dry sync tax self-limits to a small
    /// fraction of the round. Always-on; the trace is not required.
    mtp_ev_sync_ms: f32 = 0,
    mtp_ev_round_ms: f32 = 0,
    mtp_regime: mtp_policy.MtpRegime = .{},
    mtp_regime_verdict: ?bool = null,
    /// Width-trial schedule for the round-cost table (one 2-round block per
    /// period at m_lo+1, so the table learns the next cliff once).
    mtp_width_trial: round_cost.TrialSchedule = .{},
    /// Consecutive plans with the same m_lo (the base has stopped climbing).
    mtp_m_lo_streak: u32 = 0,
    /// Drafts of the previous TWO rounds (0 = serial), every round: a round
    /// is a transition unless both predecessors ran its width (the first
    /// round after a width change is the transition, the second is still
    /// elevated — live, every w4 sample taken beside a trial read 17% high
    /// on the M4 base and fed the churn that produced it).
    spec_round_prev_width: ?u32 = null,
    spec_round_prev_width2: ?u32 = null,
    /// Shape of the previous two MTP rounds: a shape change is a transition too.
    spec_round_prev_two_chunk: bool = false,
    spec_round_prev_two_chunk2: bool = false,
    /// Wall clock between round ends: the regime compares the quantity
    /// tok/s reports, so per-round work OUTSIDE the round stopwatch (token
    /// publish, stop checks, scheduler) must be in the denominator — fewer
    /// rounds is part of a wider shape's win.
    mtp_regime_clock: ?io_util.Stopwatch = null,
    /// True while this generator is the ONLY decoding stream. Contention is
    /// strictly one-sided — it only ADDS time — so a busy server simply
    /// stops feeding the kv-term learner rather than teaching it a lie.
    /// Set per tick by the scheduler.
    spec_cost_solo: bool = true,
    /// Draft-depth cap while this slot verifies inside a batched group (0 = none):
    /// the group's rows must stay on the split-K verify lane (`mtp_tick.mtpGroupRowCap`).
    mtp_group_cap: u32 = 0,
    mtp_planner_owned: bool = false,
    mtp_planner_pending: bool = false,
    mtp_planner_width: ?u8 = null,
    mtp_planner_probe: bool = false,
    mtp_planner_probes: u8 = 0,
    mtp_planner_recovery: group_planner.Recovery = .{},
    mtp_planner_recovering: bool = false,
    mtp_planner_history: group_planner.History = .{},
    mtp_hidden_stale: bool = false,
    mtp_planner_plain_ticks: u32 = 0,
    mtp_planner_prime_ticks: u32 = 0,
    mtp_planner_spec_rounds: u32 = 0,
    mtp_planner_max_gap_ms: f32 = 0,
    /// This slot's next `mtpRoundBegin` inits the chain and returns `.open` so the
    /// scheduler can run the head step for the group in one graph.
    mtp_batch_head: bool = false,
    /// Per-phase wall-time trace (SUSHI_MTP_TRACE=1; else untouched).
    mtp_trace: mtp_policy.MtpTrace = .{},
    /// Trace-only: stopwatch running across the scheduler gap (round return
    /// → next round entry); bills the `.gap` phase. Null when not tracing.
    mtp_gap_watch: ?io_util.Stopwatch = null,
    /// Deferred committed-history append: the round's (tokens, true verify
    /// hiddens) pair, folded into the NEXT round's first draft step as one
    /// multi-row head forward instead of a separate appendHistory forward.
    /// Rounds with no successor (EOS/length/runtime disable) never pay for
    /// the append; the stash is freed unconsumed in `deinit`.
    mtp_hist_stash: ?mtp_round.MtpHistStash = null,
    /// Cross-round pre-draft (round pipelining): the NEXT round's chunk-A
    /// draft chain, built and async-dispatched at the CURRENT round's tail
    /// so the head chain runs on the GPU while the CPU emits tokens. The
    /// build consumes `mtp_hist_stash`, so the two are mutually exclusive
    /// (asserted at consume). Freed unconsumed in `deinit`.
    mtp_pre_draft: ?mtp_round.MtpPreDraft = null,

    // ── Phase 1: SSM checkpoints captured during prefill ──
    /// Owned SSM-state snapshots taken at stride-aligned positions during
    /// chunked prefill. Drained by the scheduler in `commitSlotIfApplicable`
    /// via `takeSsmCheckpoints()`. Empty on non-hybrid models or when
    /// `ssm_checkpoint_stride == 0`. Allocator: the Generator's `allocator`
    /// (passed to `initWithOptions`); the same allocator must be passed to
    /// `deinit` for any checkpoint that wasn't taken.
    ssm_checkpoints: std.ArrayList(SSMCheckpoint) = std.ArrayList(SSMCheckpoint).empty,
    /// Allocator used for `ssm_checkpoints` storage and each checkpoint's
    /// per-layer slice. Set during `initWithOptions`. We track it separately
    /// from the `allocator` argument to `deinit` because `takeSsmCheckpoints`
    /// transfers ownership: the consumer (HotPrefixCache) must use the SAME
    /// allocator to free, since the layer-slice backing memory was allocated
    /// here.
    ssm_checkpoint_alloc: ?std.mem.Allocator = null,
    /// Most checkpoints the prefill held at once, counting the copy taken before each thin.
    ssm_cp_peak: u32 = 0,

    // ── Runtime acceptance gate ──
    // Set to true mid-request when the per-request acceptance rate
    // (`*_accepted_tokens / *_attempted`) falls below
    // the DFlash gate threshold after its warmup attempts. When set,
    // `nextDflash` and `nextMtp` short-circuit to `next()` for the
    // remainder of the request — the prompt-time gate could not foresee that
    // the workload's *runtime* draft acceptance rate wasn't paying for the
    // per-step verify overhead. The flag is sticky for the rest of the
    // generation; we never re-enable speculation within a single request.
    spec_disabled_runtime: bool = false,
    /// Why `spec_disabled_runtime` was set, for `[spec-stats]`.
    spec_disable_reason: mtp_plan.SpecDisableReason = .none,
    /// A loop convicted inside the thought already closed it: one recovery per request.
    think_closed_by_loop: bool = false,
    /// Adaptive serial controller: which arm this request is on and the KV bucket it was decided in.
    mtp_adaptive: mtp_policy.MtpAdaptive = .{},
    /// This request has released the module-owned MTP head (sticky serial arm reached a serial
    /// block boundary). Per request: one slow request must not turn MTP off process-wide.
    mtp_head_released: bool = false,
    /// This request's realized MTP price (ms per emitted token) over a trailing window.
    mtp_price: mtp_policy.MtpPriceWindow = .{},
    /// Ticks left in a bounded serial block (the serial probe); the slot stays an MTP slot.
    mtp_serial_left: u32 = 0,
    /// Where a serial block sits in its ramp back to `nextMtp`'s entry invariant.
    mtp_serial_exit: mtp_round.MtpSerialExit = .none,
    /// Ticks of the current serial block spent warming (the previous round's tail).
    mtp_serial_warm: u32 = 0,
    mtp_serial_giveup_logged: bool = false,
    /// Inter-tick wall clock feeding the table's serial cell. Null = the next read seeds.
    mtp_serial_clock: ?io_util.Stopwatch = null,
    pub const dflashBreakEvenAccepted = glm5_generate.dflashBreakEvenAccepted;
    pub const dflashGateShouldDisable = glm5_generate.dflashGateShouldDisable;

    /// Verifier rows a round of this request carried on average: the block's until a round has run.
    fn dflashMeanRows(self: *const Generator) f32 {
        if (self.dflash_attempted == 0 or self.dflash_drafts_verified == 0) return @floatFromInt(self.dflash_block_size);
        return 1 + @as(f32, @floatFromInt(self.dflash_drafts_verified)) / @as(f32, @floatFromInt(self.dflash_attempted));
    }

    pub const dflashSerialPrice = glm5_generate.dflashSerialPrice;

    /// Emit a stable, easy-to-grep one-line summary of spec-decode acceptance
    /// for this request. External tooling parses the `[spec-stats]` prefix;
    /// keep the format stable.
    ///
    /// No-op when this Generator never ran a speculative path.
    ///
    /// Field semantics:
    /// - `attempts` = number of speculative rounds (one verify forward each).
    /// - `accepts` = total drafted tokens accepted across all rounds (excludes
    ///   the always-committed t1 token at the start of each round).
    /// - `avg_per_round` = accepts/attempts. Bounded by `(block_size - 1)` for
    ///   the drafter.
    /// - `per_draft_pct` (drafter only) = accepts / (attempts × (block_size-1)),
    ///   the per-draft acceptance probability comparable to vLLM's reported
    ///   "62% acceptance rate" metric.
    /// Which QSA arm served this request's attention (qwen4_exp only; no-op
    /// elsewhere). The engaged/declined meters are one-shots for the whole
    /// PROCESS, so they cannot say what ran on THIS request at THIS context
    /// length — a gather that quietly hands every call back to the dense mask
    /// (a full kv8 dequant per layer per forward) otherwise reads like a
    /// healthy run. Emitted next to `[spec-stats]`; resets the tally.
    pub fn logQsaArms(self: *Generator) void {
        // The tally rides THIS request's ForwardCtx — the one every forward
        // of this slot was handed. A Transformer-level counter printed the
        // union of every slot that decoded or prefilled since the last
        // finisher reset it.
        const c = self.ctx.qsa_arms.take() orelse return;
        const arm = c.majority() orelse return;
        log.info(
            "  [qsa-arms] qsa={s} calls={d} mask={d} decode={d} verify={d} prefill={d}\n",
            .{ @tagName(arm), c.total(), c.mask, c.decode, c.verify, c.prefill },
        );
    }

    pub const SpecMode = enum { none, dspark, mtp, dflash };

    /// The mode `logSpecStats` reports: the first that drafted.
    pub fn specMode(self: *const Generator) SpecMode {
        if (self.dspark and self.dspark_stats.rounds > 0) return .dspark;
        if (self.mtp != null and (self.mtp_attempted > 0 or self.mtp_lookup_rounds > 0)) return .mtp;
        if (self.dflash != null and self.dflash_attempted > 0) return .dflash;
        return .none;
    }

    pub fn logSpecStats(self: *const Generator) void {
        var table_buf: [256]u8 = undefined;
        var lookup_buf: [256]u8 = undefined;
        const mode = self.specMode();
        if (mode == .dspark) {
            const st = self.dspark_stats;
            const rounds: f64 = @floatFromInt(st.rounds);
            const round_ms = @as(f64, @floatFromInt(st.draft_ns + st.verify_ns)) / rounds / 1e6;
            const per_round = @as(f64, @floatFromInt(st.accepted)) / rounds;
            var rows_buf: [160]u8 = undefined;
            var rows_w = std.Io.Writer.fixed(&rows_buf);
            for (st.rows_hist, 0..) |n, r| if (n > 0) rows_w.print("{s}{d}:{d}", .{ if (rows_w.end > 0) "," else "", r, n }) catch break;
            log.info(
                "  [spec-stats] mode=dspark attempts={d} accepts={d} avg_per_round={d:.2} per_draft_pct={d:.1}% drafted={d} draft_ms={d:.1} verify_ms={d:.1} round_ms={d:.1} tok_s={d:.2} plain={d} rows={s}\n",
                .{
                    st.rounds,
                    st.accepted,
                    per_round,
                    if (st.drafted > 0) 100.0 * @as(f64, @floatFromInt(st.accepted)) / @as(f64, @floatFromInt(st.drafted)) else 0.0,
                    st.drafted,
                    @as(f64, @floatFromInt(st.draft_ns)) / rounds / 1e6,
                    @as(f64, @floatFromInt(st.verify_ns)) / rounds / 1e6,
                    round_ms,
                    (per_round + 1.0) * 1000.0 / round_ms,
                    st.plain,
                    rows_w.buffered(),
                },
            );
            if (self.sampling.constraint != null) log.info("  [spec-stats] mode=dspark grammar mask_ms={d:.2} per round, serial_ticks={d}\n", .{ @as(f64, @floatFromInt(st.mask_ns)) / rounds / 1e6, st.serial });
            return;
        }
        const table_bucket = self.xfm.round_cost.bucketOf(self.mtpKvLen());
        if (mode == .mtp) {
            const avg_per_round: f64 = @as(f64, @floatFromInt(self.mtp_accepted_tokens)) /
                @as(f64, @floatFromInt(@max(self.mtp_attempted, 1)));
            // Depth varies per round under the EV controller — the honest
            // denominator is the DRAFTED count, not attempts x cap.
            const drafts_proposed: u64 = if (self.mtp_drafted_tokens > 0)
                self.mtp_drafted_tokens
            else
                self.mtp_attempted * @as(u64, self.mtp_depth);
            const per_draft_pct: f64 = if (drafts_proposed > 0)
                100.0 * @as(f64, @floatFromInt(self.mtp_accepted_tokens)) /
                    @as(f64, @floatFromInt(drafts_proposed))
            else
                0.0;
            const stalls = self.mtp_round_log.stalls();
            // The line is at the formatter's 32-argument limit.
            var stalls_buf: [32]u8 = undefined;
            const stalls_txt = std.fmt.bufPrint(&stalls_buf, "{d}/{d:.0}", .{ stalls.count, stalls.max_ms }) catch "?";
            log.info(
                "  [spec-stats] mode=mtp attempts={d} accepts={d} avg_per_round={d:.2} per_draft_pct={d:.1}% depth={d} drafted={d} ext_rounds={d} partial_rounds={d} runtime_disabled={s} reason={s} adaptive={s} serial_cell={d:.2} sync_ms={d:.2} round_ms={d:.2} two_ms_tok={d:.2} one_ms_tok={d:.2} verdict_round={d} trials={d} width_trials={d} table={s}:{s} table_drops=t{d}/c{d}/b{d}/i{d} serial_drops=t{d}/c{d}/b{d} lookup={d}/{d}/{d} stalls={s}\n",
                .{
                    self.mtp_attempted,
                    self.mtp_accepted_tokens,
                    avg_per_round,
                    per_draft_pct,
                    self.mtp_depth,
                    self.mtp_drafted_tokens,
                    self.mtp_ext_rounds,
                    self.partial_rounds,
                    if (self.spec_disabled_runtime) "true" else "false",
                    @tagName(self.spec_disable_reason),
                    @tagName(self.mtp_adaptive.arm),
                    self.xfm.round_cost.serialMsPerTok(table_bucket) orelse 0.0,
                    self.mtp_ev_sync_ms,
                    self.mtp_ev_round_ms,
                    if (self.mtp_regime.two_tok > 0) self.mtp_regime.two_ms / self.mtp_regime.two_tok else 0.0,
                    if (self.mtp_regime.one_tok > 0) self.mtp_regime.one_ms / self.mtp_regime.one_tok else 0.0,
                    self.mtp_regime.verdict_round,
                    self.mtp_regime.trials,
                    self.mtp_width_trial.trials,
                    round_cost.bucketName(self.xfm.round_cost.layout, table_bucket),
                    self.xfm.round_cost.formatBucket(table_bucket, &table_buf),
                    self.xfm.round_cost.dropped_transition,
                    self.xfm.round_cost.dropped_contended,
                    self.xfm.round_cost.dropped_bad,
                    self.xfm.round_cost.dropped_implausible,
                    self.xfm.round_cost.serial_dropped_transition,
                    self.xfm.round_cost.serial_dropped_contended,
                    self.xfm.round_cost.serial_dropped_bad,
                    self.mtp_lookup_rounds,
                    self.mtp_lookup_drafted,
                    self.mtp_lookup_accepted,
                    stalls_txt,
                },
            );
            if (self.mtp_lookup_rounds > 0) log.info("  [spec-stats] lookup_table={s}:{s}\n", .{
                round_cost.bucketName(self.xfm.round_cost.layout, table_bucket),
                self.xfm.round_cost.formatLookupBucket(table_bucket, &lookup_buf),
            });
            return;
        }
        if (mode == .dflash) {
            const avg_per_round: f64 = @as(f64, @floatFromInt(self.dflash_accepted_tokens)) /
                @as(f64, @floatFromInt(self.dflash_attempted));
            const drafts_per_round: u32 = if (self.dflash_block_size >= 1) self.dflash_block_size - 1 else 0;
            const drafts_proposed: u64 = if (self.glm_dflash_native) self.glm_round.drafted else (if (self.dflash_drafts_verified > 0) self.dflash_drafts_verified else self.dflash_attempted * @as(u64, drafts_per_round));
            const per_draft_pct: f64 = if (drafts_proposed > 0)
                100.0 * @as(f64, @floatFromInt(self.dflash_accepted_tokens)) /
                    @as(f64, @floatFromInt(drafts_proposed))
            else
                0.0;
            // GLM's native path keeps its historical line: it has no yield gate and prints the old generic bar.
            var gate_buf: [16]u8 = undefined;
            var serial_buf: [32]u8 = undefined;
            const gate_txt = std.fmt.bufPrint(&gate_buf, "{d:.2}", .{if (self.glm_dflash_native) 1.80 else dflashBreakEvenAccepted(self.dflashMeanRows())}) catch "";
            const serial_txt = if (self.glm_dflash_native) "" else std.fmt.bufPrint(&serial_buf, " serial_ms={d:.2}", .{dflashSerialPrice(&self.xfm.round_cost, table_bucket) orelse 0.0}) catch "";
            log.info(
                "  [spec-stats] mode=dflash attempts={d} accepts={d} avg_per_round={d:.2} gate_min={s}{s} per_draft_pct={d:.1}% block_size={d} partial_rounds={d} runtime_disabled={s} table={s}:{s} table_drops=t{d}/c{d}/b{d}/i{d}\n",
                .{
                    self.dflash_attempted,
                    self.dflash_accepted_tokens,
                    avg_per_round,
                    gate_txt,
                    serial_txt,
                    per_draft_pct,
                    self.dflash_block_size,
                    self.partial_rounds,
                    if (self.spec_disabled_runtime) "true" else "false",
                    round_cost.bucketName(self.xfm.round_cost.layout, table_bucket),
                    self.xfm.round_cost.formatBucket(table_bucket, &table_buf),
                    self.xfm.round_cost.dropped_transition,
                    self.xfm.round_cost.dropped_contended,
                    self.xfm.round_cost.dropped_bad,
                    self.xfm.round_cost.dropped_implausible,
                },
            );
            if (self.glm_dflash_native) {
                const rounds: f64 = @floatFromInt(self.dflash_attempted);
                const r = self.glm_round;
                const plain_ticks: f64 = @floatFromInt(@max(self.glm_plain.ticks, 1));
                log.info("  [spec-stats] glm_round_ms draft={d:.2} verify={d:.2} replay={d:.2} commit={d:.2} lookup={d}/{d} plain={d} serial_ms={d:.2} plain_ms verify={d:.2} replay={d:.2} commit={d:.2}\n", .{
                    @as(f64, @floatFromInt(r.draft_ns)) / rounds / 1e6,
                    @as(f64, @floatFromInt(r.verify_ns)) / rounds / 1e6,
                    @as(f64, @floatFromInt(r.replay_ns)) / rounds / 1e6,
                    @as(f64, @floatFromInt(r.commit_ns)) / rounds / 1e6,
                    self.glm_lookup.rounds,
                    self.glm_lookup.landed,
                    self.glm_plain.ticks,
                    self.glm_gate.serial_ms orelse 0.0,
                    @as(f64, @floatFromInt(self.glm_plain.verify_ns)) / plain_ticks / 1e6,
                    @as(f64, @floatFromInt(self.glm_plain.replay_ns)) / plain_ticks / 1e6,
                    @as(f64, @floatFromInt(self.glm_plain.commit_ns)) / plain_ticks / 1e6,
                });
            }
            return;
        }
    }

    /// Prefill the prompt and prepare for token-by-token generation.
    /// Backwards-compatible — prefer `initWithOptions` for new callers.
    pub fn init(
        io: std.Io,
        allocator: std.mem.Allocator,
        xfm: *Transformer,
        tok: *const Tokenizer,
        prompt_ids: []const u32,
        max_tokens: u32,
        sampling: sampler.SamplingParams,
        eos_token_ids: []const u32,
    ) !Generator {
        return initWithOptions(io, allocator, xfm, tok, prompt_ids, max_tokens, sampling, eos_token_ids, .{});
    }

    /// Receiver for SSM checkpoints salvaged out of a cancelled prefill.
    /// `initWithOptions` captures stride checkpoints into a local list that
    /// its errdefer frees — on `error.Cancelled` it instead moves them here
    /// so the scheduler can commit a partial-prefix cache entry. Ownership
    /// transfers either to the hot-cache entry (`commitWithSsm`) or back to
    /// the sink's `deinit`.
    pub const CancelledCheckpointSink = struct {
        /// Absolute prompt-token count forwarded into the KV when the chunk
        /// loop aborted (`ssm_checkpoint_pos_offset + pos`). This — NOT
        /// `cache.step`, which only advances when init completes — is the
        /// authoritative length for a cancelled-prefill commit.
        forwarded: usize = 0,
        checkpoints: []SSMCheckpoint = &.{},
        /// The allocator the checkpoints (and the slice itself) were
        /// allocated with — the same `sch.allocator` family the hot-cache
        /// entry later frees them with.
        alloc: ?std.mem.Allocator = null,

        pub fn deinit(self: *CancelledCheckpointSink) void {
            const a = self.alloc orelse return;
            for (self.checkpoints) |*cp| cp.deinit(a);
            if (self.checkpoints.len > 0) a.free(self.checkpoints);
            self.checkpoints = &.{};
            self.alloc = null;
        }
    };

    pub const InitOptions = struct {
        /// The machine-sized prefill chunk frozen at load
        /// (`ModelConfig.pinned_prefill_chunk`, resolved by
        /// `server.resolvePrefillChunk`). 0 = unpinned, keep the launch width.
        ///
        /// It arrives per REQUEST rather than off `xfm.config` because the
        /// Transformer holds a COPY of the config taken when it was built, and
        /// the pin is written to the registry's config after load — the copy
        /// never sees it. The scheduler reads `slot.model.config`, which IS the
        /// object the admission guard bills against, so bill and forward cannot
        /// disagree. (Live 2026-08-14: pinned 4096, prefilled at 8192.)
        pinned_prefill_chunk: usize = 0,
        /// Number of tokens per draft round. Default 4 (3 drafter steps +
        /// 1 t1 prepend → length-4 verify forward).
        drafter_block_size: u32 = 4,
        /// Enable the DFlash block-drafter. When set, `dflash` must be
        /// non-null and `bind()`-ed to `xfm`. Prefill captures the trunk's
        /// target_layer_ids outputs chunk-by-chunk into the per-request
        /// assistant context cache (`Generator.dflash_ctx`). Same
        /// lazy-pre-forward skip semantics as drafter/MTP.
        dflash_enabled: bool = false,
        /// Non-owning pointer to the loaded DFlash assistant.
        dflash: ?*DflashModel = null,
        /// Effective DFlash block size (loader-resolved: assistant config
        /// capped by the machine's verify lanes). 0 → the assistant config's value.
        dflash_block_size: u32 = 0,
        /// Enable the Qwen native MTP head. When set, `mtp` must be non-null
        /// and `bind()`-ed to `xfm`. Prefill builds the head's committed-
        /// history KV cache chunk-by-chunk (full-hidden capture) and the
        /// final-token forward captures `last_hidden`, exactly like the
        /// drafter path. Same lazy-pre-forward skip semantics as drafter.
        mtp_enabled: bool = false,
        /// Explicit verifier selected by the caller at construction.
        mtp_acceptance: mtp_acceptance.Mode = .exact,
        mtp_greedy_tail: bool = false,
        /// Non-owning pointer to the loaded MTP head.
        mtp: ?MtpHeadRef = null,
        /// The model's head before this request's opt-out (the qwen4 weights load with the trunk regardless).
        model_has_mtp: bool = false,
        /// Max tokens drafted per nextMtp round. 0 = auto (`--mtp-max-depth` not
        /// passed): resolved by `resolveMtpDepthCapForProfile` —
        /// MTP_ADAPTIVE_DEFAULT_CAP under the EV controller; DEFAULT_DEPTH in
        /// fixed mode. Explicit depths remain unchanged.
        mtp_depth: u32 = 0,
        /// When set, this slice (rather than `prompt_ids`) becomes the
        /// `prompt_ids_owned` source for the MTP prompt-lookup index. Used by the
        /// server's KV-cache-reuse path to forward only the trailing tokens
        /// while still giving the lookup the full prompt for matching.
        lookup_prompt: ?[]const u32 = null,
        /// Per-slot forward context (Phase 2 concurrent batching). When null,
        /// `initWithOptions` builds one from `xfm.defaultCtx()` so the legacy
        /// single-slot path is unchanged. Phase 2 callers pass a ForwardCtx
        /// whose `cache` / `moe_seq_offset` / `ssm_entries` / `vision_embeddings`
        /// point at the slot's own state. Stored by value on the Generator.
        ctx: ?ForwardCtx = null,
        /// Skip the lazy first-token pre-forward (regular path only). When set,
        /// init samples t1 synchronously and leaves `pending_logits` /
        /// `pending_token` empty — cache.step lands at exactly prompt_len with
        /// t1 NOT in cache. The first `next()` call's transition shim will
        /// sync-forward `[t1]` to seed pending_logits before the lazy chain.
        /// Used by the Phase 2 scheduler so a slot's cache state matches
        /// `forwardBatchedDecode`'s expectation (cache.step == prompt_len at
        /// the start of every decode tick). PLD / drafter paths already skip
        /// the lazy pre-forward unconditionally; this flag generalizes that
        /// behavior to the regular sampling path. Has no effect when
        /// `drafter_enabled` is true.
        skip_lazy_preforward: bool = false,
        /// deepseek_v41: decode with DSpark rounds (`dsparkEligible`); the pending token is
        /// then not in the cache, as after a skipped lazy pre-forward.
        dspark_enabled: bool = false,
        /// Phase 1 (performance-plan): during prefill, capture an SSM
        /// checkpoint every `ssm_checkpoint_stride` tokens. 0 = disabled.
        /// Snapshots land in `Generator.ssm_checkpoints` for the caller to
        /// drain into the hot prefix cache via `takeSsmCheckpoints()`. Only
        /// effective when the model has hybrid layers (otherwise the
        /// `ssm_entries` slice is empty and snapshots become no-op stubs).
        /// Chunked prefill aligns chunk ends to stride positions so each
        /// snapshot reflects a coherent state.
        ssm_checkpoint_stride: u32 = 0,
        /// Cap on retained checkpoints. Past it the list is thinned span-preservingly
        /// (`transformer.ssmCheckpointDropIndex`): lowest and newest always survive. 0 = unlimited.
        ssm_checkpoint_max: u32 = 16,
        /// Phase 1: absolute position of the FIRST token in `prompt_ids`.
        /// On a cold prefill this is 0. On the warm path (where the
        /// scheduler restored some prefix and now forwards only the tail),
        /// callers pass `hot_matched` so the captured checkpoints stamp
        /// absolute positions usable by future warm-path lookups against
        /// the full prompt.
        ssm_checkpoint_pos_offset: usize = 0,
        /// Placeholder rows inside a restored prefix (prefix-cache hit on an
        /// image prompt): the vision splice starts here, not at row 0.
        vision_rows_before: usize = 0,
        /// A DFlash assistant context restored from the prefix cache, whose
        /// `absLen()` already equals `ssm_checkpoint_pos_offset`. Ownership
        /// transfers to the Generator. Null = start the assistant blind at
        /// the offset (a restore forwards no trunk layers, so a cold context
        /// is what a reused prefix would otherwise get).
        dflash_ctx_restored: ?dflash_mod.DflashCtx = null,
        /// An MTP committed-history cache restored from the prefix cache,
        /// whose `base + cache.step()` already equals
        /// `ssm_checkpoint_pos_offset`. Ownership transfers to the
        /// Generator. Null = start the history blind at the offset (same
        /// contract as `dflash_ctx_restored`).
        mtp_cache_restored: ?MtpRestored = null,
        /// Cooperative abort for abandoned requests: checked between prefill
        /// chunks. The scheduler passes `&slot.cancelled`, set by the conn
        /// thread when the client disconnects mid-prefill. When it flips,
        /// `initWithOptions` returns `error.Cancelled` instead of grinding
        /// out the rest of a multi-minute ghost prefill.
        cancel_flag: ?*const std.atomic.Value(bool) = null,
        /// When the chunk loop aborts on `cancel_flag`, the SSM checkpoints
        /// captured so far move into this sink instead of dying with the
        /// failed construction — the scheduler salvages a partial-prefix
        /// cache commit from them. Null keeps the old drop-everything
        /// behaviour.
        cancelled_checkpoint_sink: ?*CancelledCheckpointSink = null,
        /// Per-token logprobs count for this request (0 = disabled). Callers
        /// that set `Generator.logprobs_n` after init must ALSO pass it here:
        /// init's split-prefill final-token forward is a single-row dispatch,
        /// and the certified lm_head prune may only engage when the request
        /// consumes nothing but the argmax — a logprobs request reads the
        /// full logit row, which the pruned head does not produce.
        logprobs_n: u32 = 0,
        /// LIVE prefill progress, in tokens actually forwarded so far by THIS
        /// prefill. Bumped once per chunk (not per token), read off-thread by
        /// the metrics gauge sampler.
        ///
        /// Without it the panel is blind during prefill: `prompt_tokens_total`
        /// and `prefill_time_seconds` only advance at request COMPLETION, and
        /// generated tokens only accrue during decode — so a multi-minute
        /// prefill saturates the GPU while both tiles read 0 / "—". The
        /// scheduler resets it to 0 when the prefill ends.
        prefill_progress: ?*std.atomic.Value(u64) = null,
        /// Total tokens this prefill will forward (post-cache tail, same scale
        /// as `prefill_progress`), published once; null on the bypass engines.
        prefill_expected: ?*std.atomic.Value(u64) = null,

        /// Called once per completed prefill chunk boundary (chunk state
        /// evaluated, allocator cache cleared), except the final one. The
        /// scheduler runs decode ticks for the streams already decoding at
        /// this seam, so a long cold prefill stalls them for at most one
        /// chunk-forward instead of its whole duration. Null = the prefill
        /// runs atomically (pre-interleave behavior, and the
        /// SUSHI_PREFILL_INTERLEAVE=0 kill switch).
        interleave_hook: ?InterleaveHook = null,
        /// SSD-first write-through (qwen4_exp and GLM; see `WriteThroughHook`).
        write_through_hook: ?WriteThroughHook = null,
        /// DeepSeek: a prefill chunk whose decoder tail nothing reads leaves state only. False runs every tail, the
        /// full passes the tests compare against.
        skip_unread_tails: bool = true,
        /// The per-chunk prefill width, asked at every chunk boundary after that chunk's
        /// `mlx_clear_cache` and before the interleave tick. Null keeps the admitted width.
        /// Presence is NOT the arch gate (`serve` installs it process-wide); read `adaptive_chunk_width`.
        chunk_width_hook: ?ChunkWidthHook = null,
        /// Is the per-chunk adaptive width live for this request (`server.adaptivePrefillChunkEnabled`)?
        /// The only thing that may widen `tailMergeMaxFor`'s bound.
        adaptive_chunk_width: bool = false,
    };

    pub const InterleaveHook = struct {
        ctx: *anyopaque,
        call: *const fn (ctx: *anyopaque) void,
    };

    /// SSD-first write-through: called at every completed prefill chunk with the absolute KV
    /// position and the SSM checkpoints so far, so a cancelled prefill leaves a restorable prefix.
    /// The hook may `release` a checkpoint whose state it has put on SSD; its position stays listed.
    pub const WriteThroughHook = struct {
        ctx: *anyopaque,
        call: *const fn (ctx: *anyopaque, abs_kv_pos: usize, cps: []SSMCheckpoint) void,
    };

    /// Returns the width the next chunk should run at (`cur` unchanged = no change).
    pub const ChunkWidthHook = struct {
        ctx: *anyopaque,
        call: *const fn (ctx: *anyopaque, pos: usize, cur: u32, cap: u32, st: *mtp_plan.AdaptiveWidthState) u32,
        /// Re-price a widen after the interleave tick has allocated for co-tenant slots. A
        /// step-down needs no confirmation: it is the safe direction.
        confirm: *const fn (ctx: *anyopaque, pos: usize, want: u32) bool,
    };

    /// Selects the source slice that `initWithOptions` will dupe into
    /// `prompt_ids_owned`. When `lookup_prompt` is non-null it wins (server
    /// cache-reuse path: full original prompt for the MTP lookup); otherwise the
    /// caller's `prompt_ids` is used (back-compat path).
    pub fn pickLookupPromptSource(prompt_ids: []const u32, lookup_prompt: ?[]const u32) []const u32 {
        return lookup_prompt orelse prompt_ids;
    }

    pub fn initWithOptions(
        io: std.Io,
        allocator: std.mem.Allocator,
        xfm: *Transformer,
        tok: *const Tokenizer,
        prompt_ids: []const u32,
        max_tokens: u32,
        sampling_in: sampler.SamplingParams,
        eos_token_ids: []const u32,
        options_in: InitOptions,
    ) !Generator {
        // The logits forward starts at the prompt's last token; there is none to index.
        if (prompt_ids.len == 0) return error.EmptyPrompt;
        // Reserved-token suppression rides the sampling params from HERE —
        // the one chokepoint every init site funnels through — so every
        // sampling path (serial, DFlash/MTP corrections, draft heads,
        // stochastic-verify filters, batched decode via `gen.sampling`)
        // inherits the model's mask without per-site wiring.
        var sampling = sampling_in;
        sampling.suppress_mask = xfm.suppress_mask;
        sampling.think_penalty.file_biases = xfm.logit_bias;
        const bias_vocab: usize = xfm.config.vocab_size;
        try sampling.think_penalty.prepare(allocator, bias_vocab, xfm.think_marker_mask, xfm.s);
        errdefer sampling.think_penalty.deinitPrepared();
        var options = options_in;
        const glm_dflash_native = xfm.glm5 != null and options.dflash_enabled and options.dflash != null and glm5_generate.glmDflashEligible(sampling, options.logprobs_n);
        if (xfm.glm5 != null) {
            options.mtp_enabled = false;
            options.dflash_enabled = glm_dflash_native;
        }
        const s = xfm.s;
        // Per-slot ForwardCtx (Phase 2). Stored by value on the Generator;
        // callers either supply one (scheduler) or fall through to
        // `xfm.defaultCtx()` for the legacy single-slot path. We pass
        // `&ctx` to every forward call below; the cache/moe/ssm fields
        // mutate in-place through their pointers.
        var ctx: ForwardCtx = options.ctx orelse xfm.defaultCtx();

        // lm_head gate: the certified prune and MiMo's coarse shortlist serve
        // the ARGMAX, not the tail distribution, so they engage only when this
        // request consumes nothing else (`argmaxOnlyRequest`).
        ctx.argmax_only = sampler.argmaxOnlyRequest(sampling, options.logprobs_n);

        const ids_i32 = try allocator.alloc(i32, prompt_ids.len);
        defer allocator.free(ids_i32);
        for (prompt_ids, 0..) |id, i| {
            ids_i32[i] = @intCast(id);
        }

        // Clone the lookup prompt for the lifetime of the Generator. The MTP
        // lookup index needs `prompt + generated`, and the caller-owned
        // slice can be freed after init. When `options.lookup_prompt`
        // is set (server cache-reuse path), it carries the full original prompt
        // so the match coverage isn't gutted when only a trailing tail was
        // forwarded into the KV cache. Defaults to `prompt_ids` otherwise.
        // Allocated up front so init's errdefer paths don't have to track
        // partial state.
        const owned_src = pickLookupPromptSource(prompt_ids, options.lookup_prompt);
        const prompt_owned = try allocator.dupe(u32, owned_src);
        errdefer allocator.free(prompt_owned);

        // Split prefill: process first N-1 tokens (cache-only, skip lm_head eval),
        // then the last token (produces logits for sampling). This mirrors mlx-lm's
        // generate_step which avoids the expensive lm_head projection over the full
        // sequence length. For vocab_size=262144, skipping lm_head on N-1 tokens
        // avoids a [N-1, hidden] @ [hidden, 262144] matmul.
        //
        // Chunked prefill: large prompts are processed in PREFILL_CHUNK-sized pieces
        // to bound peak activation memory. Each chunk fills KV cache entries for its
        // positions, gets eval'd, and intermediates are freed before the next chunk.
        // Without chunking, a MoE's stacked layers can spike to
        // ~20 GB of activations alone on a 50k-token prompt, causing Metal OOM.
        // Vision requests skip chunking since image token positions must be visible
        // in a single forward pass for spliceVisionEmbeddings to work correctly.
        // PREFILL_CHUNK comes from the module-level `prefill_chunk_override`
        // (set by --prefill-chunk in main.zig).
        //
        // Safety cap: on unfused head dims (>128)
        // the composed SDPA materializes [heads, chunk, total_kv] scores per
        // layer, so the chunk shrinks with the prompt's FINAL KV length to keep
        // that one tensor bounded (boundedPrefillChunk). Warm-path restores
        // start at ssm_checkpoint_pos_offset, so the final KV length is that
        // offset plus everything we're about to forward.
        const total_ctx_for_chunk = options.ssm_checkpoint_pos_offset + prompt_ids.len;
        const PREFILL_CHUNK: usize = prefill_plan.requestPrefillChunk(
            xfm.config.prefillScoreHeadDim(),
            xfm.config.num_attention_heads,
            total_ctx_for_chunk,
            xfm.config.has_sliding_window,
            xfm.config.isMoe(),
            xfm.config.longCtxGated(),
            options.pinned_prefill_chunk,
        );
        // Where cold prefill time goes (chunked-forward vs eval vs last-token-forward), at debug level.
        const trace_enabled = log.isDebug();
        const chunk_peak = trace_enabled and transformer_mod.diagEnvOn("SUSHI_PREFILL_CHUNK_PEAK");
        var prefill_sw = io_util.Stopwatch.init(io);
        var chunked_ns: u64 = 0;
        var eval_ns: u64 = 0;
        var n_chunks: usize = 0;

        // Phase 1: SSM checkpointing during prefill. When enabled, the chunked
        // prefill loop forces a chunk boundary at every multiple of
        // `ssm_checkpoint_stride`, then snapshots `ctx.ssm_entries` after that
        // chunk evaluates. Snapshots accumulate in `Generator.ssm_checkpoints`
        // for the scheduler to drain in `commitSlotIfApplicable`. Plain-attn
        // models have an empty `ssm_entries` slice, so this becomes a no-op
        // even at stride > 0 — but we still bail early so we never allocate
        // empty checkpoints.
        var ssm_checkpoints: std.ArrayList(SSMCheckpoint) = std.ArrayList(SSMCheckpoint).empty;
        var ssm_cp_peak: u32 = 0;
        errdefer {
            for (ssm_checkpoints.items) |*cp| cp.deinit(allocator);
            ssm_checkpoints.deinit(allocator);
        }
        const has_vision = ctx.vision_embeddings != null;
        const glm_request = ctx.glm5_request;
        const want_ssm_cp = prefill_plan.shouldCheckpointSsmPrefill(
            options.ssm_checkpoint_stride,
            (ctx.ssm_entries != null and ctx.ssm_entries.?.len > 0) or glm_request != null or xfm.dsv41 != null,
            has_vision,
        );
        // The snapshot backoff: a checkpoint AT the prompt end is
        // unreachable next turn.
        const want_state_cp = want_ssm_cp;
        // Coarsen the checkpoint stride so checkpointing never sub-divides the
        // prefill chunk, on ANY arch (see effectiveSsmCheckpointStride: fine
        // strides push every projection under prefillDqGemm's M>=2048 floor and
        // multiply per-chunk fixed costs — 17-25% cold-prefill tax measured on
        // dense AND MoE hybrids). Warm reuse keeps chunk-granularity snapshots
        // plus the always-on end-of-prompt snapshot (the append-growth case
        // llmprobe's cache-hit tests pin — verified green at coarse strides).
        // Coarsen against the UNCAPPED base chunk: the head_dim safety cap
        // above must not densify checkpoint spacing (16× more captures at
        // 255K ctx otherwise). nextChunkEnd already shortens a chunk to land
        // on stride boundaries, so a capped chunk stays compatible.
        const ssm_cp_stride: usize = if (!want_ssm_cp)
            0
        else if (glm_request != null)
            glm5_generate.glm_checkpoint_stride
        else
            prefill_plan.effectiveSsmCheckpointStride(@intCast(options.ssm_checkpoint_stride), @max(PREFILL_CHUNK, prefill_chunk_override));
        // Where a checkpoint is captured: GLM chunks on its chunk grid but keeps points only on the wider
        // restore-point grid.
        const cp_grid: usize = if (glm_request != null and want_ssm_cp) glm5_generate.glm_capture_stride else ssm_cp_stride;
        // Absolute KV position of `prompt_ids[0]`. Warm-path callers (the
        // scheduler after restoring a checkpoint) pass the matched prefix
        // length so the snapshots stamp positions valid in the full original
        // sequence, not relative offsets inside the tail-only prefill.
        const ssm_cp_offset: usize = options.ssm_checkpoint_pos_offset;

        // Qwen native MTP: build the head's committed-history KV cache during
        // prefill. Entry j pairs (trunk hidden at prompt position j, token at
        // j+1); the (hidden[last], t1) pair is appended by the first nextMtp
        // round. On KV-prefix reuse the history covers only the freshly
        // forwarded tail — RoPE offsets are cache-relative, so a late-starting
        // history is self-consistent (sliding-window history semantics).
        const mtp_active = options.mtp_enabled and options.mtp != null;
        const AcceptRoute = struct {
            graph: mtp_round.MtpGraphFn,
            prefix: mtp_acceptance.PrefixFn,
            param: f32,
        };
        const accept_route: AcceptRoute = switch (options.mtp_acceptance) {
            .exact => .{ .graph = mtpBatchedExactGraph, .prefix = mtp_acceptance.exactPrefix, .param = 0 },
            .typical => |cfg| .{ .graph = mtpBatchedTypicalGraph, .prefix = mtp_acceptance.typicalPrefix, .param = cfg.delta },
            .tokenv3 => |alpha| .{ .graph = mtpBatchedTokenV3Graph, .prefix = mtp_acceptance.tokenV3Prefix, .param = alpha },
        };
        if (mtp_active and std.meta.activeTag(options.mtp_acceptance) != .exact) {
            switch (options.mtp_acceptance) {
                .typical => |cfg| log.info("  [mtp-accept] typical delta={d:.3} eps={d:.3} NOT distribution-exact\n", .{ cfg.delta, cfg.eps }),
                .tokenv3 => |alpha| log.info("  [mtp-accept] TokenV3 alpha={d:.3} NOT distribution-exact\n", .{alpha}),
                .exact => unreachable,
            }
        }
        var mtp_cache: ?MtpCacheRef = null;
        var mtp_position_base: usize = ssm_cp_offset;
        var mtp_history_started = false;
        if (mtp_active) {
            if (options.mtp_cache_restored) |restored| {
                // A restored history already covers [base, ssm_cp_offset);
                // the tail prefill APPENDS to it (RoPE offsets are
                // cache-relative, so continuation is self-consistent).
                std.debug.assert(restored.base + restored.cache.step() == ssm_cp_offset);
                mtp_cache = restored.cache;
                mtp_position_base = restored.base;
                mtp_history_started = true;
            } else {
                mtp_cache = try options.mtp.?.makeCache(allocator);
            }
        }
        errdefer if (mtp_cache) |*mc| mc.deinit();

        // DFlash: build the assistant's context cache during prefill — every
        // chunk's target_layer_ids captures project through the encoder and
        // append immediately (the chunk's activation graph then frees with
        // the chunk eval). On KV-prefix reuse the context covers only the
        // freshly forwarded tail (`base_pos = ssm_cp_offset`) — sliding-
        // window semantics, same rule as the MTP history.
        const dflash_active = options.dflash_enabled and options.dflash != null;
        var dflash_ctx: ?dflash_mod.DflashCtx = if (!dflash_active)
            null
        else if (options.dflash_ctx_restored) |restored| blk: {
            std.debug.assert(restored.absLen() == ssm_cp_offset);
            break :blk restored;
        } else try dflash_mod.DflashCtx.init(allocator, options.dflash.?, ssm_cp_offset);
        errdefer if (dflash_ctx) |*dc| dc.deinit();
        var dfl_out_buf: []mlx.mlx_array = &.{};
        defer if (dfl_out_buf.len > 0) allocator.free(dfl_out_buf);
        var dfl_cl: transformer_mod.CaptureLayers = undefined;
        if (dflash_active) {
            dfl_out_buf = try allocator.alloc(mlx.mlx_array, options.dflash.?.config.target_layer_ids.len);
            dfl_cl = .{ .ids = options.dflash.?.config.target_layer_ids, .out = dfl_out_buf, .hc_mean = xfm.config.hc_count };
        }

        // Start of the final (logits) forward's token span. Without SSM
        // checkpointing this is the last prompt token (the classic 1-token
        // logits forward); with checkpointing the chunk loop stops
        // SSM_SNAPSHOT_BACKOFF tokens early so the always-on snapshot lands
        // where the next turn's prefix match can actually reach it (see
        // ssmSnapshotBackoff), and the final forward covers the held-back
        // tail + the last token in the same weight sweep.
        var final_start: usize = prompt_ids.len - 1;
        // Chunked vision prefill (issue #197): placeholder rows consumed by
        // completed chunks, fed to ctx.vision_splice_offset per forward
        // (chunk loop AND final-span forward). Stays 0 on the kill-switch arm
        // so that path is byte-identical to the old whole-prompt behavior.
        const vision_chunked = has_vision and prefill_plan.visionChunkedPrefillEnabled();
        var vision_rows_consumed: usize = options.vision_rows_before;
        var adapt_state: mtp_plan.AdaptiveWidthState = .{
            .width_min = @intCast(PREFILL_CHUNK),
            .width_max = @intCast(PREFILL_CHUNK),
        };

        if (prompt_ids.len > 1) {
            const prefix_len = prompt_ids.len - 1;
            const snapshot_backoff = if (glm_request != null and want_state_cp)
                glm5_generate.glmCaptureSchedule(glm5_generate.glm_capture_stride, options.ssm_checkpoint_max, ssm_cp_offset, prompt_ids.len).backoff
            else
                prefill_plan.ssmSnapshotBackoff(want_state_cp, prefix_len, ssm_cp_offset > 0);
            const loop_end = prefix_len - snapshot_backoff;
            final_start = loop_end;
            // Vision prompts chunk like text (issue #197) — the splice offset
            // below keeps the row scatter chunk-exact. Kill switch restores
            // the whole-prompt forward.
            const default_chunk = if (has_vision and !vision_chunked) loop_end else PREFILL_CHUNK;
            // Per-chunk adaptive width: the first chunk runs the arch's widest width, and every
            // boundary re-asks the same estimator. `cap_adapt` is the widest this arch forwards
            // for this prompt, never wider than `ssm_cp_stride`.
            const adapt_chunked = !(has_vision and !vision_chunked);
            // The scaled tail-merge bound reads the arch predicate, never `chunk_width_hook != null`
            // (installed process-wide).
            const width_is_adaptive = adapt_chunked and options.adaptive_chunk_width;
            const cap_adapt: u32 = if (!adapt_chunked) 0 else @intCast(@min(prefill_plan.effectivePrefillChunk(
                xfm.config.prefillScoreHeadDim(),
                xfm.config.num_attention_heads,
                total_ctx_for_chunk,
                xfm.config.has_sliding_window,
                xfm.config.isMoe(),
                xfm.config.longCtxGated(),
                0,
            ), @as(usize, xfm.config.prefillStartWidth())));
            adapt_state.width_min = @intCast(default_chunk);
            adapt_state.width_max = @intCast(default_chunk);
            var cur_chunk: usize = default_chunk;

            // Reserve the whole request's cache capacity before the first chunk writes: a grow
            // is not in place, so a long prefill's peak carried a second copy of everything so
            // far (~7.75 GB at 458k on qwen4_exp, #353). Same number the admission guard billed
            // (`KVCache.reservedTokens`), in ABSOLUTE positions: `prompt_ids` is the tail after
            // a prefix-cache hit. `cp_thin` names this site's previous retention policy.
            const cp_thin: transformer_mod.ThinPolicy =
                if (xfm.config.longCtxGated() or glm_request != null) .min_span_recency else .oldest;

            const reserved_tokens = try reserveRequestCapacity(ctx.cache, &xfm.config, total_ctx_for_chunk, max_tokens, default_chunk, s);
            // The arch's own per-request buffers reserve at the same length.
            if (mtp_cache) |*mc| mc.activate();
            transformer_mod.reserveQsaHistoryWithHead(
                ctx.ssm_entries,
                if (xfm.qwen4_mtp) |*m| &m.entry else null,
                @intCast(reserved_tokens),
            );
            // Admission billed the request's top at `default_chunk`; the chunks before it start at
            // the widest width and step down only where one no longer fits beside the reserved KV.
            const start_chunk = if (width_is_adaptive) cap_adapt else 0;
            if (start_chunk > cur_chunk) {
                cur_chunk = start_chunk;
                adapt_state.width_min = @intCast(start_chunk);
                adapt_state.width_max = @intCast(start_chunk);
                if (options.chunk_width_hook) |hk| {
                    const w = hk.call(hk.ctx, ssm_cp_offset, @intCast(cur_chunk), cap_adapt, &adapt_state);
                    if (w != 0 and w < cur_chunk) mtp_plan.commitAdaptiveWidth(&cur_chunk, &adapt_state, w);
                }
            }

            var pos: usize = 0;
            if (options.prefill_expected) |e| e.store(@intCast(loop_end), .monotonic);
            while (pos < loop_end) {
                // Abandoned-request abort: the client disconnected and the
                // conn thread flagged the slot. Bail before the next chunk —
                // the KV built so far is freed with the slot.
                if (options.cancel_flag) |cf| {
                    if (cf.load(.acquire)) {
                        // Salvage the chunk loop's progress: `forwarded` is
                        // the authoritative length (cache.step only advances
                        // when init completes), the stride checkpoints ride
                        // for hybrid restore (they die with the failed
                        // construction otherwise). toOwnedSlice empties the
                        // local list so the errdefer frees nothing (on OOM
                        // the items stay and the errdefer cleans up — the
                        // sink keeps the length, the commit declines the
                        // checkpoints only).
                        if (options.cancelled_checkpoint_sink) |sink| {
                            sink.forwarded = ssm_cp_offset + pos;
                            if (ssm_checkpoints.items.len > 0) {
                                if (ctx.ssm_entries) |ents| {
                                    transformer_mod.handoffQsaHistoryToLatest(ssm_checkpoints.items, ents, xfm.s) catch {};
                                }
                                if (ssm_checkpoints.toOwnedSlice(allocator)) |owned| {
                                    sink.checkpoints = owned;
                                    sink.alloc = allocator;
                                } else |_| {}
                            }
                        }
                        return error.Cancelled;
                    }
                }
                // An MLX failure in a previous chunk latched instead of killing the process (#353);
                // abandon this request, keep the server. No checkpoint salvage: the state failed.
                try mlx.checkError();
                // Pick this chunk's end. Normal path: hit the configured chunk
                // size. Phase 1 path: if a checkpoint stride boundary lands
                // inside the would-be chunk, shrink the chunk so it ends
                // exactly on that boundary. That gives us an snapshot-point
                // every `stride` tokens without changing the model's seen
                // input — the forward result is identical to the unchunked
                // version because attention is causal and SSM/conv update
                // chunk-locally. Boundary alignment is in ABSOLUTE position
                // (pos + offset), so the saved snapshot list is correct for
                // the full prompt, not the truncated tail.
                const end = if (glm_request != null)
                    glm5_generate.glmChunkEnd(pos, loop_end, cur_chunk, ssm_cp_stride, ssm_cp_offset, width_is_adaptive)
                else
                    prefill_plan.nextChunkEnd(pos, loop_end, cur_chunk, want_ssm_cp, ssm_cp_stride, ssm_cp_offset, width_is_adaptive);
                if (has_vision) ctx.vision_splice_offset = vision_rows_consumed;
                // The model reads the next forward's Engram rows while this chunk runs: the next chunk, or the final span.
                if (xfm.dsv41 != null) {
                    const ahead_end = if (end < loop_end)
                        prefill_plan.nextChunkEnd(end, loop_end, cur_chunk, want_ssm_cp, ssm_cp_stride, ssm_cp_offset, width_is_adaptive)
                    else
                        prompt_ids.len;
                    ctx.ahead_ids = ids_i32[end..ahead_end];
                    // A narrower width can follow the hook's step-down at this boundary, never below the ladder's floor.
                    const floor: usize = if (adapt_chunked and options.chunk_width_hook != null) @min(cap_adapt, prefill_plan.PREFILL_CHUNK_FLOOR) else std.math.maxInt(usize);
                    ctx.ahead_min_rows = if (options.skip_unread_tails and end < loop_end) @min(ahead_end - end, floor) else 0;
                    ctx.holds_point = want_ssm_cp and cp_grid > 0 and (end + ssm_cp_offset) % cp_grid == 0;
                }
                defer ctx.ahead_ids = &.{};
                defer ctx.ahead_min_rows = 0;
                defer ctx.holds_point = false;
                const chunk_len: c_int = @intCast(end - pos);
                const chunk_shape = [_]c_int{ 1, chunk_len };
                const chunk_input = mlx.mlx_array_new_data(@ptrCast(&ids_i32[pos]), &chunk_shape, 2, .int32);
                defer _ = mlx.mlx_array_free(chunk_input);

                const chunk_start_ns = if (trace_enabled) prefill_sw.read() else 0;
                var chunk_hidden_all = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(chunk_hidden_all);
                if (dflash_active) {
                    for (dfl_out_buf) |*a| a.* = mlx.mlx_array_new();
                    ctx.capture_layers = &dfl_cl;
                }
                errdefer if (dflash_active) {
                    for (dfl_out_buf) |a| _ = mlx.mlx_array_free(a);
                };
                const chunk_logits = if (mtp_active) blk: {
                    var last_unused = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(last_unused);
                    break :blk try xfm.forwardWithCaptureAll(&ctx, chunk_input, &last_unused, &chunk_hidden_all);
                } else try xfm.forwardWith(&ctx, chunk_input);
                _ = mlx.mlx_array_free(chunk_logits);
                if (dflash_active) {
                    ctx.capture_layers = null;
                    try dflash_mod.appendContext(options.dflash.?, &dflash_ctx.?, dfl_out_buf, ssm_cp_offset + pos);
                    for (dfl_out_buf) |*a| {
                        _ = mlx.mlx_array_free(a.*);
                        a.* = .{ .ctx = null };
                    }
                }
                if (trace_enabled) {
                    const dt = prefill_sw.read() - chunk_start_ns;
                    chunked_ns += dt;
                    log.debug(
                        "  [prefill-trace] chunk pos={d} end={d} width={d} ms={d}\n",
                        .{ pos, end, end - pos, dt / std.time.ns_per_ms },
                    );
                }

                // MTP history for this chunk: hiddens [pos, end) pair with
                // tokens [pos+1, end+1) — prompt_ids[end] always exists since
                // the chunk loop spans [0, prefix_len) and prompt_ids has
                // prefix_len + 1 entries.
                if (mtp_active) {
                    if (!mtp_history_started) {
                        std.debug.assert(mtp_cache.?.step() == 0);
                        mtp_position_base = ssm_cp_offset + pos;
                        mtp_history_started = true;
                    }
                    const mtp_mrope_ctx: ?mtp_mod.MropeContext = if (ctx.mrope_pos) |positions| .{
                        .pos = positions,
                        .total = ctx.mrope_total,
                        .delta = ctx.mrope_delta,
                        .base = mtp_position_base,
                    } else null;
                    try options.mtp.?.appendHistory(
                        xfm,
                        &mtp_cache.?,
                        prompt_ids[pos + 1 .. end + 1],
                        chunk_hidden_all,
                        @intCast(mtp_cache.?.step()),
                        mtp_mrope_ctx,
                        allocator,
                    );
                }

                // Eval KV cache — materializes this chunk's K/V, frees activation graph
                const eval_start_ns = if (trace_enabled) prefill_sw.read() else 0;
                {
                    const eval_vec = mlx.mlx_vector_array_new();
                    defer _ = mlx.mlx_vector_array_free(eval_vec);
                    for (ctx.cache.entries) |*entry| {
                        if (!entry.initialized) continue;
                        _ = mlx.mlx_vector_array_append_value(eval_vec, entry.keys);
                        _ = mlx.mlx_vector_array_append_value(eval_vec, entry.values);
                    }
                    if (chunk_hidden_all.ctx != null) {
                        _ = mlx.mlx_vector_array_append_value(eval_vec, chunk_hidden_all);
                    }
                    // Materialize this chunk's MTP history entries alongside
                    // the trunk KV so the chunk's activation graph (incl. the
                    // full-hidden capture) can be freed before the next chunk.
                    // MiMo's last chunk leaves its heads' catch-up to the first draft chain,
                    // which runs after the scheduler streams t1.
                    if (mtp_cache) |*mc| if (mc.* != .mimo or end < loop_end) mc.appendEvalArrays(eval_vec);
                    // Same discipline for the DFlash context appended above.
                    if (dflash_ctx) |*dc| dc.appendEvalArrays(eval_vec);
                    // Phase 1: also force SSM state to materialize so any
                    // snapshot below holds a concrete tensor, not a lazy node
                    // that would re-execute the prefill graph if anyone reads
                    // from it later. Unconditional on EVERY chunk (not just
                    // stride-aligned ones): a tail-merged final chunk ends
                    // off-boundary, and the always-on end-of-prompt snapshot
                    // then hits exactly that re-execution — capturing an
                    // un-evaluated state re-ran the whole last chunk's GDN
                    // scan (measured −4% on an 8K 27B prefill). The
                    // states are ~KB-to-MB scale; evaluating them alongside
                    // the KV costs nothing measurable.
                    if (want_ssm_cp) if (ctx.ssm_entries) |entries| {
                        for (entries) |*ssm| {
                            if (!ssm.initialized) continue;
                            if (ssm.conv_state.ctx != null) {
                                _ = mlx.mlx_vector_array_append_value(eval_vec, ssm.conv_state);
                            }
                            if (ssm.ssm_state.ctx != null) {
                                _ = mlx.mlx_vector_array_append_value(eval_vec, ssm.ssm_state);
                            }
                        }
                    };
                    _ = mlx.mlx_eval(eval_vec);
                }
                // Every chunk: handing the pool to the next chunk was slower and larger (glm5-perf).
                _ = mlx.mlx_clear_cache();
                if (trace_enabled) {
                    const eval_dt = prefill_sw.read() - eval_start_ns;
                    eval_ns += eval_dt;
                    var mem_buf: [256]u8 = undefined;
                    const line = formatPrefillMemLine(&mem_buf, pos, end, eval_dt / std.time.ns_per_ms, mlxPrefillMem());
                    log.debug("{s}", .{line});
                    // The next line's peak is then its own chunk's: what a transient bill is fitted to.
                    if (chunk_peak) _ = mlx.mlx_reset_peak_memory();
                }

                // This chunk's latch, read before anything persists it, snapshots it or yields
                // the thread: Metal returns zeros before it aborts, so a later read let the
                // write-through index garbage and a co-tenant decode eat the latch.
                try mlx.checkError();

                // Phase 1: snapshot SSM state at stride-aligned boundaries.
                // We snapshot AFTER the eval above so the underlying buffers
                // are realized; the snapshot is just a refcount-share of the
                // already-resident state.
                const abs_end_for_cp2 = end + ssm_cp_offset;
                if (want_ssm_cp and cp_grid > 0 and abs_end_for_cp2 % cp_grid == 0 and glm5_generate.restorableCheckpoint(glm_request, abs_end_for_cp2)) {
                    const cp = try capturePrefillCheckpoint(allocator, xfm, &ctx, glm_request, abs_end_for_cp2, options.write_through_hook != null);
                    ssm_checkpoints.append(allocator, cp) catch |e| {
                        var doomed = cp;
                        doomed.deinit(allocator);
                        return e;
                    };
                    ssm_cp_peak = @max(ssm_cp_peak, @as(u32, @intCast(ssm_checkpoints.items.len)));
                    // The head's own QSA leftover at this position: its ring is 32 rows and
                    // the clamp back to this checkpoint comes a whole generated tail later.
                    if (mtp_active) {
                        mtp_cache.?.activate();
                    }
                    if (mtp_active) xfm.qwen4MtpMarkQsaLeftover(abs_end_for_cp2 -| mtp_position_base) catch |e| {
                        log.debug("[qwen4] MTP head leftover mark failed: {s}\n", .{@errorName(e)});
                    };
                    // Thin the interior, never the oldest (#330): drop-oldest survivors covered only
                    // the last `max * stride` tokens and left no affordable trim point.
                    prefill_plan.thinHeldCheckpoints(allocator, &ssm_checkpoints, options.ssm_checkpoint_max, cp_thin);
                }

                if (vision_chunked) {
                    vision_rows_consumed += prefill_plan.countSpliceRows(
                        ids_i32[pos..end],
                        xfm.config.image_token_id,
                        xfm.config.video_token_id,
                    );
                }
                pos = end;
                n_chunks += 1;
                // Publish progress once per chunk — same cadence discipline as
                // `inflight_generated_tokens` (once per decode tick), never per token.
                if (options.prefill_progress) |p| p.store(@intCast(pos), .monotonic);
                // Persist this chunk now: the KV bytes for [0, pos) are final.
                if (options.write_through_hook) |wt| {
                    wt.call(wt.ctx, pos + ssm_cp_offset, ssm_checkpoints.items);
                }
                // The next chunk's width, re-priced at this boundary: after the `mlx_clear_cache`
                // (steady state) and before the interleave tick (a co-tenant's decode is not this
                // prefill's pressure). A step-down commits here; a widen is re-priced after the tick.
                var pending_widen: u32 = 0;
                var widen_confirm: ?*const fn (*anyopaque, usize, u32) bool = null;
                var widen_ctx: ?*anyopaque = null;
                if (adapt_chunked and pos < loop_end) {
                    if (options.chunk_width_hook) |hk| {
                        const next_w = hk.call(hk.ctx, ssm_cp_offset + pos, @intCast(cur_chunk), cap_adapt, &adapt_state);
                        if (next_w != 0 and next_w != cur_chunk) {
                            if (next_w < cur_chunk) {
                                mtp_plan.commitAdaptiveWidth(&cur_chunk, &adapt_state, next_w);
                            } else {
                                pending_widen = next_w;
                                widen_confirm = hk.confirm;
                                widen_ctx = hk.ctx;
                            }
                        }
                    }
                }
                // Yield to the scheduler between chunks — never after the
                // last (the post-prefill decode tick covers that boundary).
                if (pos < loop_end) {
                    if (options.interleave_hook) |hk| hk.call(hk.ctx);
                }
                // The tick allocated for co-tenant slots; re-price the widen against what is left.
                if (pending_widen != 0) {
                    if (widen_confirm) |cf| {
                        if (cf(widen_ctx.?, ssm_cp_offset + pos, pending_widen)) {
                            mtp_plan.commitAdaptiveWidth(&cur_chunk, &adapt_state, pending_widen);
                        } else {
                            adapt_state.supporting = 0;
                        }
                    }
                }
            }
            try mlx.checkError();
            // One line per request when the width moved.
            if (adapt_state.transitions > 0) {
                log.info("[prefill] adaptive: {d} chunks, width {d}..{d}, {d} change(s)\n", .{
                    n_chunks,
                    adapt_state.width_min,
                    adapt_state.width_max,
                    adapt_state.transitions,
                });
            }

            // Phase 1: always-on snapshot at the post-prefill position
            // (= prefix_len, i.e., prompt_ids.len - 1). The stride loop
            // captures snapshots at [stride, 2*stride, ...]; this final
            // capture covers the most common warm-path case where the next
            // turn's prompt fully matches turn-1's prompt and matched lands
            // at prompt_ids.len. Without this, a stride=256 setup with a
            // 750-token prompt could only restore at position 512 (losing
            // ~234 tokens of potential reuse to the next stride boundary).
            // With it, the cache restores to position 749 (~99% of the
            // prompt) and only the last token + new tail re-forwards.
            // Skipped on `prompt_ids.len == 1` (no prefill chunks ran).
            if (want_ssm_cp and loop_end > 0) {
                // The snapshot sits at the chunk loop's end — snapshot_backoff
                // tokens BEFORE the prompt end, where the next turn's prefix
                // match can actually reach it (the template's generation
                // suffix renders differently in history, so a match always
                // falls a few tokens short of the full prompt).
                const final_abs = loop_end + ssm_cp_offset;
                // Skip if we already captured at this exact position (the
                // chunked loop would have done so when loop_end happens
                // to be a stride multiple).
                const already_have = ssm_checkpoints.items.len > 0 and
                    ssm_checkpoints.items[ssm_checkpoints.items.len - 1].pos == final_abs;
                if (!already_have and glm5_generate.restorableCheckpoint(glm_request, final_abs)) {
                    // SSM state is already materialized — the chunked loop
                    // evaluated it at every chunk boundary. The final chunk
                    // may have been a stride-aligned one (already evaluated)
                    // or a partial tail (also evaluated). The snapshot is a
                    // cheap refcount-share.
                    const cp = try capturePrefillCheckpoint(allocator, xfm, &ctx, glm_request, final_abs, options.write_through_hook != null);
                    ssm_checkpoints.append(allocator, cp) catch |e| {
                        var doomed = cp;
                        doomed.deinit(allocator);
                        return e;
                    };
                    // DeepSeek's shared point leaves before the final span appends to the rows it views.
                    if (xfm.dsv41 != null) if (options.write_through_hook) |wt| wt.call(wt.ctx, final_abs, ssm_checkpoints.items);
                    ssm_cp_peak = @max(ssm_cp_peak, @as(u32, @intCast(ssm_checkpoints.items.len)));
                    // The head's own QSA leftover at this position: its ring is 32 rows and
                    // the clamp back to this checkpoint comes a whole generated tail later.
                    if (mtp_active) {
                        mtp_cache.?.activate();
                    }
                    if (mtp_active) xfm.qwen4MtpMarkQsaLeftover(final_abs -| mtp_position_base) catch |e| {
                        log.debug("[qwen4] MTP head leftover mark failed: {s}\n", .{@errorName(e)});
                    };
                    prefill_plan.thinHeldCheckpoints(allocator, &ssm_checkpoints, options.ssm_checkpoint_max, cp_thin);
                }
            }
        }

        // Chunked vision: the final-span forward continues the row scatter
        // where the chunk loop left it (an image ending in the final span
        // otherwise re-splices from row 0). One-shot engagement line — the
        // silent-no-op class.
        if (has_vision) {
            ctx.vision_splice_offset = vision_rows_consumed;
            if (vision_chunked and n_chunks > 1) log.debug("[vision] chunked prefill: {d} chunks, {d} placeholder rows consumed\n", .{ n_chunks, vision_rows_consumed });
        }

        // Process the final span — the last token, plus (under SSM
        // checkpointing) the snapshot-backoff tail held back from the chunk
        // loop. One forward, one weight sweep, logits sliced to the last
        // position so every consumer sees the classic [1, 1, V] shape.
        const tail_len: usize = prompt_ids.len - final_start;
        const last_shape = [_]c_int{ 1, @as(c_int, @intCast(tail_len)) };
        const last_input = mlx.mlx_array_new_data(@ptrCast(&ids_i32[final_start]), &last_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(last_input);

        // The MTP head needs the post-final-norm hidden as its first-step h_prev.
        // `forwardWithCapture` captures the LAST position regardless of span
        // length. When the span holds backed-off tokens the head's history must
        // also cover them (a hole right before the generation point is
        // acceptance-critical), so capture ALL positions and append.
        const need_capture = mtp_active;
        var captured_hidden: mlx.mlx_array = mlx.mlx_array_new();
        var has_captured_hidden = false;
        const last_start_ns = if (trace_enabled) prefill_sw.read() else 0;
        const tail_mtp_capture = mtp_active and tail_len > 1;
        var tail_hidden_all = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(tail_hidden_all);
        // DFlash context for the tail span (rides ctx into all three forward
        // arms below; the forwardWithCapture* wrappers only override
        // capture_hidden/_all).
        if (dflash_active) {
            for (dfl_out_buf) |*a| a.* = mlx.mlx_array_new();
            ctx.capture_layers = &dfl_cl;
        }
        errdefer if (dflash_active) {
            for (dfl_out_buf) |a| _ = mlx.mlx_array_free(a);
        };
        const raw_logits = if (tail_mtp_capture) blk: {
            has_captured_hidden = true;
            break :blk try xfm.forwardWithCaptureAll(&ctx, last_input, &captured_hidden, &tail_hidden_all);
        } else if (need_capture) blk: {
            has_captured_hidden = true;
            break :blk try xfm.forwardWithCapture(&ctx, last_input, &captured_hidden);
        } else try xfm.forwardWith(&ctx, last_input);
        // History entries for the held-back span: (hidden[j], token[j+1]) for
        // j in [final_start, prefix_len) — same pairing as the chunk loop.
        // The last row's pair (hidden[last], t1) is appended by the first
        // nextMtp round, exactly as before.
        if (tail_mtp_capture) {
            if (!mtp_history_started) {
                std.debug.assert(mtp_cache.?.step() == 0);
                mtp_position_base = ssm_cp_offset + final_start;
                mtp_history_started = true;
            }
            var tail_hist_hidden = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(tail_hist_hidden);
            const all_shape = mlx.getShape(tail_hidden_all);
            const start = [_]c_int{ 0, 0, 0 };
            const stop = [_]c_int{ all_shape[0], @intCast(tail_len - 1), all_shape[2] };
            const strides = [_]c_int{ 1, 1, 1 };
            try mlx.check(mlx.mlx_slice(&tail_hist_hidden, tail_hidden_all, &start, 3, &stop, 3, &strides, 3, xfm.s));
            const tail_mrope_ctx: ?mtp_mod.MropeContext = if (ctx.mrope_pos) |positions| .{
                .pos = positions,
                .total = ctx.mrope_total,
                .delta = ctx.mrope_delta,
                .base = mtp_position_base,
            } else null;
            try options.mtp.?.appendHistory(
                xfm,
                &mtp_cache.?,
                prompt_ids[final_start + 1 .. prompt_ids.len],
                tail_hist_hidden,
                @intCast(mtp_cache.?.step()),
                tail_mrope_ctx,
                allocator,
            );
        }
        if (dflash_active) {
            ctx.capture_layers = null;
            try dflash_mod.appendContext(options.dflash.?, &dflash_ctx.?, dfl_out_buf, ssm_cp_offset + final_start);
            for (dfl_out_buf) |*a| {
                _ = mlx.mlx_array_free(a.*);
                a.* = .{ .ctx = null };
            }
        }
        // Slice to the last position when the span is longer than one token,
        // so downstream sampling/grammar paths see the classic shape.
        const logits = if (tail_len == 1) raw_logits else blk: {
            defer _ = mlx.mlx_array_free(raw_logits);
            const lshape = mlx.getShape(raw_logits);
            var sliced = mlx.mlx_array_new();
            const start = [_]c_int{ 0, lshape[1] - 1, 0 };
            const stop = [_]c_int{ lshape[0], lshape[1], lshape[2] };
            const strides = [_]c_int{ 1, 1, 1 };
            try mlx.check(mlx.mlx_slice(&sliced, raw_logits, &start, 3, &stop, 3, &strides, 3, xfm.s));
            break :blk sliced;
        };
        if (trace_enabled) {
            const last_ns = prefill_sw.read() - last_start_ns;
            const total_ns = prefill_sw.read();
            const ms = std.time.ns_per_ms;
            log.debug(
                "  [prefill-trace] tokens={d} chunks={d} chunk_size={d} chunk_widths={d}..{d} chunked={d}ms eval={d}ms last_token={d}ms total={d}ms{s}\n",
                .{
                    prompt_ids.len,
                    n_chunks,
                    PREFILL_CHUNK,
                    adapt_state.width_min,
                    adapt_state.width_max,
                    chunked_ns / ms,
                    eval_ns / ms,
                    last_ns / ms,
                    total_ns / ms,
                    if (need_capture) " [capture-hidden]" else "",
                },
            );
        }
        errdefer if (has_captured_hidden) {
            _ = mlx.mlx_array_free(captured_hidden);
        };

        // Attach the SSM-checkpoint buffer to whichever Generator variant
        // we're about to return. Clears the local list so the errdefer above
        // doesn't double-free. All four init paths below call this once
        // before returning their Generator.
        const attachCp = struct {
            fn f(g: *Generator, list: *std.ArrayList(SSMCheckpoint), a: std.mem.Allocator, peak: u32) void {
                g.ssm_checkpoints = list.*;
                g.ssm_cp_peak = peak;
                g.ssm_checkpoint_alloc = a;
                list.* = std.ArrayList(SSMCheckpoint).empty;
            }
        }.f;

        // Constrained generation skips the lazy first-sample fast path: we cannot
        // sample the first token until we have applied the grammar mask, and we
        // cannot pipeline because grammar advancement depends on the realized id.
        if (sampling.constraint != null) {
            // Grammar-constrained requests never speculate; release the MTP
            // history cache / DFlash context if dispatch enabled them anyway.
            if (mtp_cache) |*mc| {
                mc.deinit();
                mtp_cache = null;
            }
            if (dflash_ctx) |*dc| {
                dc.deinit();
                dflash_ctx = null;
            }
            var gen = Generator{
                .xfm = xfm,
                .model_has_mtp = options.model_has_mtp,
                .ctx = ctx,
                .tok = tok,
                .next_token_id = 0,
                .step = 0,
                .max_tokens = max_tokens,
                .sampling = sampling,
                .prompt_tokens = @intCast(prompt_ids.len),
                .completion_tokens = 0,
                .finish_reason = "length",
                .done = false,
                .eos_token_ids = eos_token_ids,
                .generated_ids = std.ArrayList(u32).empty,
                .timeout_ns = 0,
                .timer = io_util.Stopwatch.init(io),
                .last_hidden = if (has_captured_hidden) captured_hidden else mlx.mlx_array_new(),
                .has_last_hidden = has_captured_hidden,
                .prompt_ids_owned = prompt_owned,
                .prompt_ids_alloc = allocator,
                .prng = std.Random.DefaultPrng.init(sampling.seed orelse @intCast(std.Io.Timestamp.now(io, .real).toMilliseconds())),
                .dspark = options.dspark_enabled and dsv41_dspark.dsparkEligible(sampling, options.logprobs_n),
                .dspark_diag = dsv41_dspark.dsparkDiag(),
                .dspark_policy = dspark_rows.Policy.init(depth_bounds.active, if (xfm.dsv41) |m| m.cfg.dspark.block else 5),
            };
            gen.pending_logits = logits;
            gen.has_pending_logits = true;
            attachCp(&gen, &ssm_checkpoints, allocator, ssm_cp_peak);
            return gen;
        }

        // A forced tool call decides each successor after its predecessor is
        // known, which no draft path does: drafts never start for it.
        if (sampling.call_force != null) {
            if (mtp_cache) |*mc| {
                mc.deinit();
                mtp_cache = null;
            }
            if (dflash_ctx) |*dc| {
                dc.deinit();
                dflash_ctx = null;
            }
        }

        // DFlash / MTP path: sample synchronously and DO NOT
        // pre-forward the sampled token. The first nextDflash /
        // nextMtp call needs the cache at exactly prompt_len (last prompt
        // token forwarded; first sampled token deferred). The lazy
        // pre-forward path below would over-advance the cache and corrupt
        // every verify forward.
        if (sampling.call_force == null and (mtp_active or dflash_active)) {
            const sample_lazy = try sampler.firstTokenSample(xfm, logits, sampling, s);
            _ = mlx.mlx_array_free(logits);
            try mlx.check(mlx.mlx_array_eval(sample_lazy));
            var first_val: i32 = 0;
            try mlx.check(mlx.mlx_array_item_int32(&first_val, sample_lazy));
            _ = mlx.mlx_array_free(sample_lazy);

            if (glm_dflash_native) try glm5_generate.reserveGlmDflash(&xfm.config, ctx, &dflash_ctx.?, max_tokens, s);
            var glm_window: ?glm5_generate.GlmSpecWindow = if (glm_dflash_native and want_ssm_cp) .{ .snapshot = try dflash_ctx.?.cache.snapshot(), .base_pos = dflash_ctx.?.base_pos } else null;
            errdefer if (glm_window) |*w| w.snapshot.deinit();
            const mtp_cost_profile: mtp_mod.MtpCostProfile = if (mtp_active)
                options.mtp.?.costProfile(xfm, ctx.cache.config)
            else
                .generic;
            const dflash_bs: u32 = if (dflash_active)
                (if (options.dflash_block_size > 0) options.dflash_block_size else options.dflash.?.config.block_size)
            else if (options.dflash_block_size > 0)
                options.dflash_block_size
            else
                0;
            const mtp_cap = resolveMtpDepthCapForProfile(options.mtp_depth, mtp_cost_profile);
            var gen = Generator{
                .xfm = xfm,
                .model_has_mtp = options.model_has_mtp,
                .ctx = ctx,
                .tok = tok,
                .next_token_id = @intCast(first_val),
                .step = 0,
                .max_tokens = max_tokens,
                .sampling = blk: {
                    var sp = sampling;
                    sp.draw = 1; // t1 above was draw 0
                    break :blk sp;
                },
                .prompt_tokens = @intCast(prompt_ids.len),
                .completion_tokens = 0,
                .finish_reason = "length",
                .done = false,
                .eos_token_ids = eos_token_ids,
                .generated_ids = std.ArrayList(u32).empty,
                .timeout_ns = 0,
                .timer = io_util.Stopwatch.init(io),
                .last_hidden = if (need_capture) captured_hidden else mlx.mlx_array_new(),
                .has_last_hidden = need_capture,
                .prng = std.Random.DefaultPrng.init(sampling.seed orelse @intCast(std.Io.Timestamp.now(io, .real).toMilliseconds())),
                .prompt_ids_owned = prompt_owned,
                .prompt_ids_alloc = allocator,
                .drafter_block_size = options.drafter_block_size,
                .dflash = if (dflash_active) options.dflash else null,
                .dflash_ctx = dflash_ctx,
                .glm_dflash_native = glm_dflash_native,
                .glm_prefill_window = glm_window,
                .dflash_block_size = dflash_bs,
                .mtp = if (mtp_active) options.mtp else null,
                .mtp_accept_graph = accept_route.graph,
                .mtp_accept_prefix = accept_route.prefix,
                .mtp_accept_param = accept_route.param,
                .mtp_greedy_tail = options.mtp_greedy_tail,
                .mtp_cache = mtp_cache,
                .mtp_position_base = mtp_position_base,
                .mtp_depth = mtp_cap,
                .mtp_depth_free = if (xfm.mtp_depth_free != 0) xfm.mtp_depth_free else mtpDepthCapFree(options.mtp_depth),
                .mtp_ev_costs = mtpEvCostsForProfile(mtp_cost_profile),
                // Start at the floor (depth 1 by default) and climb with evidence:
                // the cheap depth is the safe default (1.11x on cold/creative
                // content), and hot workloads promote within ~8 rounds.
                .mtp_depth_current = depth_bounds.floorFor(depth_bounds.active, mtp_cap),
            };
            mtp_cache = null; // ownership transferred to the Generator
            dflash_ctx = null; // ownership transferred to the Generator
            glm_window = null;
            // pending_logits/pending_token left empty — the lazy pipeline is
            // skipped under drafter / MTP. The speculative `next*` paths
            // drive every subsequent step with predictable cache offset.
            attachCp(&gen, &ssm_checkpoints, allocator, ssm_cp_peak);
            return gen;
        }

        // Phase 2: scheduler-managed slots ask init to sample t1 synchronously
        // and skip the lazy pre-forward. Cache lands at prompt_len with t1 NOT
        // in cache — matches `forwardBatchedDecode`'s expectation and
        // the drafter init path's invariant. Generator.next's transition
        // shim handles the bootstrap on the first decode tick.
        if (options.skip_lazy_preforward or options.dspark_enabled) {
            const sample_lazy = try sampler.firstTokenSample(xfm, logits, sampling, s);
            try mlx.check(mlx.mlx_array_eval(sample_lazy));
            var first_val: i32 = 0;
            try mlx.check(mlx.mlx_array_item_int32(&first_val, sample_lazy));
            _ = mlx.mlx_array_free(sample_lazy);
            // t1's own distribution comes from the prefill's final forward and
            // is invisible to the decode loop — read it here or the first
            // returned token has no logprobs and the array shifts by one.
            const first_lp = try sampler.firstTokenLogprobs(allocator, logits, @intCast(first_val), options.logprobs_n, s);
            _ = mlx.mlx_array_free(logits);

            var gen = Generator{
                .pending_logprob = first_lp,
                .xfm = xfm,
                .model_has_mtp = options.model_has_mtp,
                .ctx = ctx,
                .tok = tok,
                .next_token_id = @intCast(first_val),
                .step = 0,
                .max_tokens = max_tokens,
                .sampling = blk: {
                    var sp = sampling;
                    sp.draw = 1; // t1 above was draw 0
                    break :blk sp;
                },
                .prompt_tokens = @intCast(prompt_ids.len),
                .completion_tokens = 0,
                .finish_reason = "length",
                .done = false,
                .eos_token_ids = eos_token_ids,
                .generated_ids = std.ArrayList(u32).empty,
                .timeout_ns = 0,
                .timer = io_util.Stopwatch.init(io),
                .last_hidden = if (has_captured_hidden) captured_hidden else mlx.mlx_array_new(),
                .has_last_hidden = has_captured_hidden,
                .prng = std.Random.DefaultPrng.init(sampling.seed orelse @intCast(std.Io.Timestamp.now(io, .real).toMilliseconds())),
                .prompt_ids_owned = prompt_owned,
                .prompt_ids_alloc = allocator,
                .dspark = options.dspark_enabled and dsv41_dspark.dsparkEligible(sampling, options.logprobs_n),
                .dspark_diag = dsv41_dspark.dsparkDiag(),
                .dspark_policy = dspark_rows.Policy.init(depth_bounds.active, if (xfm.dsv41) |m| m.cfg.dspark.block else 5),
            };
            attachCp(&gen, &ssm_checkpoints, allocator, ssm_cp_peak);
            return gen;
        }

        // Regular path: sample first token lazily, then build the next forward pass
        const lazy_token = try sampler.firstTokenSample(xfm, logits, sampling, s);

        const next_logits = try lazyForward(xfm, &ctx, lazy_token, ctx.argmax_only);

        // Async-eval the decode pipeline (single-token graphs, much smaller)
        {
            const eval_vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(eval_vec);
            _ = mlx.mlx_vector_array_append_value(eval_vec, lazy_token);
            _ = mlx.mlx_vector_array_append_value(eval_vec, next_logits);
            _ = mlx.mlx_async_eval(eval_vec);
        }

        // Sync to get the first token value
        try mlx.check(mlx.mlx_array_eval(lazy_token));
        var val: i32 = 0;
        try mlx.check(mlx.mlx_array_item_int32(&val, lazy_token));
        _ = mlx.mlx_array_free(lazy_token);
        // See the skip_lazy_preforward branch: t1's distribution lives only in
        // the prefill's own output, so it is read here, against the id that was
        // actually drawn.
        const first_lp = try sampler.firstTokenLogprobs(allocator, logits, @intCast(val), options.logprobs_n, s);
        _ = mlx.mlx_array_free(logits);

        var gen = Generator{
            .pending_logprob = first_lp,
            .xfm = xfm,
            .model_has_mtp = options.model_has_mtp,
            .ctx = ctx,
            .tok = tok,
            .next_token_id = @intCast(val),
            .step = 0,
            .max_tokens = max_tokens,
            .sampling = blk: {
                var sp = sampling;
                sp.draw = 1; // t1 above was draw 0
                break :blk sp;
            },
            .prompt_tokens = @intCast(prompt_ids.len),
            .completion_tokens = 0,
            .finish_reason = "length",
            .done = false,
            .eos_token_ids = eos_token_ids,
            .generated_ids = std.ArrayList(u32).empty,
            .timeout_ns = 0,
            .timer = io_util.Stopwatch.init(io),
            .last_hidden = if (has_captured_hidden) captured_hidden else mlx.mlx_array_new(),
            .has_last_hidden = has_captured_hidden,
            .prng = std.Random.DefaultPrng.init(sampling.seed orelse @intCast(std.Io.Timestamp.now(io, .real).toMilliseconds())),
            .prompt_ids_owned = prompt_owned,
            .prompt_ids_alloc = allocator,
        };

        gen.pending_logits = next_logits;
        gen.has_pending_logits = true;

        attachCp(&gen, &ssm_checkpoints, allocator, ssm_cp_peak);
        return gen;
    }

    /// Return the prefill's parked transients once, at the handover to decode.
    /// Shares `advanceStep`'s clock: a clear here is a clear, so the decode
    /// cadence measures its next interval from this point.
    pub fn clearPoolBeforeDecode(self: *Generator) void {
        if (!prefill_plan.clearsPoolAtPrefillEnd(&self.xfm.config)) return;
        _ = mlx.mlx_clear_cache();
        self.last_cache_clear_step = self.step;
    }

    /// The ONE place `step` and `completion_tokens` advance — every decode path
    /// (plain, constrained, drafter, MTP, batched) routes through here.
    ///
    /// Advancing them by hand is how three paths — `nextDrafter`, `nextMtp` and
    /// `scheduler.runBatchedDecodeTick` — ended up never calling
    /// `mlx_clear_cache()` at all (issue #110). MLX parks freed buffers in a
    /// size-keyed pool instead of returning them to the OS, so a decode path
    /// with no clear ratchets the process footprint while `active_bytes` stays
    /// flat: the reporter's process reached 81 GB with the panel reading 19.6.
    /// A `-mtp` checkpoint on a dense trunk defaults straight onto one of them.
    /// The serial-decode test asserts the clock advances on the plain and logprob paths.
    pub fn advanceStep(self: *Generator, n: u32) void {
        self.completion_tokens += n;
        self.step += n;
        if (prefill_plan.shouldClearAllocatorCache(self.step, self.last_cache_clear_step, prefill_plan.CACHE_CLEAR_INTERVAL)) {
            _ = mlx.mlx_clear_cache();
            self.last_cache_clear_step = self.step;
        }
    }

    pub fn hasPrePayloadConstraint(self: *const Generator) bool {
        const constraint = self.sampling.constraint orelse return false;
        return constraint.proto != null and constraint.pstate.phase != .json_body;
    }

    /// The pending reasoning→payload boundary, consumed by the scheduler when
    /// it publishes the token that carries it (same delivery path as the
    /// token itself, so a streaming reader cannot see one without the other).
    pub fn takeConstraintSpan(self: *Generator) ?rp_mod.ConstraintSpan {
        const span = self.sampling.constraint orelse return null;
        const s = span.pending_span orelse return null;
        span.pending_span = null;
        return s;
    }

    pub const ConstraintRecovery = union(enum) {
        /// One forced token was committed through the live forward path; the
        /// caller publishes it like a sampled token.
        committed: u32,
        /// The choice state recovers by answering directly: nothing to
        /// commit, the next sample is JSON-masked.
        activated,
        /// The remaining transition does not fit the completion allowance.
        refused,
    };

    /// Force the reasoning→constrained transition when the protocol has not
    /// reached the JSON body: a reasoning loop, premature EOS, or terminal
    /// padding recovers into a valid final answer when the remaining
    /// completion allowance fits the WHOLE remaining transition plus one
    /// answer token. Commits at most one forced token per call (the next
    /// tick drains the rest), so cancellation, stop sequences, inference
    /// errors, and stalled forwards keep their usual precedence.
    pub fn forceConstraintTransition(self: *Generator, allocator: std.mem.Allocator) !ConstraintRecovery {
        const constraint = self.sampling.constraint orelse return .refused;
        const proto = constraint.proto orelse return .refused;
        std.debug.assert(!self.has_pending_token);

        switch (constraint.pstate.phase) {
            .json_body => return .refused,
            .choice => {
                if (constraint.pstate.open_cursor == 0) {
                    // The model never committed a channel and emitted no
                    // opener bytes: a direct constrained answer needs no
                    // transition token at all. The pending logits stay —
                    // they are the distribution the next JSON-masked sample
                    // reads.
                    constraint.pstate.phase = .json_body;
                    constraint.pstate.recovering = false;
                    constraint.pending_span = .{ .token_index = @intCast(self.generated_ids.items.len), .byte_offset = 0 };
                    self.loop_guard_start = self.generated_ids.items.len;
                    log.info("[grammar] reasoning loop at the channel choice; enforcing JSON schema immediately\n", .{});
                    return .activated;
                }
                // Opener bytes are already in the output: starting the JSON
                // here would strand them (the response split reads them as an
                // unclosed opener and files the answer as reasoning).
                // Complete the opener, close it, then constrain. Fall through
                // to the shared planner below.
                log.info("[grammar] completing the generated reasoning opener before the boundary\n", .{});
                if (self.has_pending_logits) {
                    _ = mlx.mlx_array_free(self.pending_logits);
                    self.has_pending_logits = false;
                }
                if (!constraint.pstate.recovering) {
                    var planned: [rp_mod.MAX_TRANSITION_TOKENS]u32 = undefined;
                    const planned_len = proto.planRecovery(&constraint.pstate, &planned) orelse return .refused;
                    if (!forcedBoundaryCanContinue(self.completion_tokens, self.max_tokens, planned_len)) return .refused;
                    constraint.pstate.recovering = true;
                    constraint.pstate.forced_cursor = 0;
                    constraint.pstate.pending_len = @intCast(planned_len);
                    @memcpy(constraint.pstate.pending[0..planned_len], planned[0..planned_len]);
                }
                return .{ .committed = try self.commitForcedConstraintToken(allocator, constraint) };
            },
            .reasoning, .header => {
                self.dspark_pending = false;
                if (self.has_pending_logits) {
                    _ = mlx.mlx_array_free(self.pending_logits);
                    self.has_pending_logits = false;
                }
                if (!constraint.pstate.recovering) {
                    var planned: [rp_mod.MAX_TRANSITION_TOKENS]u32 = undefined;
                    const planned_len = proto.planRecovery(&constraint.pstate, &planned) orelse return .refused;
                    if (!forcedBoundaryCanContinue(self.completion_tokens, self.max_tokens, planned_len)) return .refused;
                    constraint.pstate.recovering = true;
                    constraint.pstate.forced_cursor = 0;
                    constraint.pstate.pending_len = @intCast(planned_len);
                    @memcpy(constraint.pstate.pending[0..planned_len], planned[0..planned_len]);
                }
                return .{ .committed = try self.commitForcedConstraintToken(allocator, constraint) };
            },
        }
    }

    /// Commit ONE canonical transition token through the live model state,
    /// then leave its logits pending for the next sample. Called once per
    /// scheduler tick while `pstate.recovering` drains.
    fn commitForcedConstraintToken(self: *Generator, allocator: std.mem.Allocator, constraint: *Constraint) !u32 {
        if (self.has_pending_logits) {
            _ = mlx.mlx_array_free(self.pending_logits);
            self.has_pending_logits = false;
        }
        const seq = constraint.pstate.pending[0..constraint.pstate.pending_len];
        const tok = seq[constraint.pstate.forced_cursor];
        constraint.pstate.forced_cursor += 1;

        try self.generated_ids.append(allocator, tok);
        self.loop_guard_start = self.generated_ids.items.len;
        self.next_token_id = tok;
        self.advanceStep(1);
        self.consecutive_pad = 0;

        const tok_i32: i32 = @intCast(tok);
        const tok_shape = [_]c_int{ 1, 1 };
        const tok_input = mlx.mlx_array_new_data(&tok_i32, &tok_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(tok_input);
        const next_logits = try self.xfm.forwardWith(&self.ctx, tok_input);
        const arr = [_]mlx.mlx_array{next_logits};
        const vec = mlx.mlx_vector_array_new_data(&arr, 1);
        _ = mlx.mlx_async_eval(vec);
        _ = mlx.mlx_vector_array_free(vec);
        self.pending_logits = next_logits;
        self.has_pending_logits = true;

        if (constraint.pstate.forced_cursor >= seq.len) {
            // The transition drained: the next sample is the first constrained
            // payload token. Reset the repetition window at the final-body
            // transition so the drained sequence cannot re-trigger recovery.
            constraint.pstate.recovering = false;
            constraint.pstate.phase = .json_body;
            constraint.pending_span = .{ .token_index = @intCast(self.generated_ids.items.len), .byte_offset = 0 };
            self.loop_guard_start = self.generated_ids.items.len;
        }
        return tok;
    }

    pub fn loopGuardStart(self: *const Generator) usize {
        return @min(self.loop_guard_start, self.generated_ids.items.len);
    }

    pub fn deinit(self: *Generator, allocator: std.mem.Allocator) void {
        self.sampling.think_penalty.deinitPrepared();
        if (group_planner.enabled() and self.mtp_planner_owned) log.info("[mtp-planner-stats] tokens={d} plain={d} prime={d} spec={d} probes={d} max_gap_ms={d:.2} recovery={d}\n", .{ self.completion_tokens, self.mtp_planner_plain_ticks, self.mtp_planner_prime_ticks, self.mtp_planner_spec_rounds, self.mtp_planner_probes, self.mtp_planner_max_gap_ms, self.mtp_planner_recovery.rounds });
        if (self.last_logprob) |*lp| {
            allocator.free(lp.top_logprobs);
        }
        if (self.pending_logprob) |*lp| {
            allocator.free(lp.top_logprobs);
        }
        if (self.has_pending_logits) {
            _ = mlx.mlx_array_free(self.pending_logits);
            self.has_pending_logits = false;
        }
        if (self.has_pending_token) {
            _ = mlx.mlx_array_free(self.pending_token);
            self.has_pending_token = false;
        }
        self.dspark_masks.deinit(allocator);
        if (self.has_last_hidden) {
            _ = mlx.mlx_array_free(self.last_hidden);
            self.has_last_hidden = false;
        }
        if (self.prompt_ids_alloc) |a| {
            a.free(self.prompt_ids_owned);
            self.prompt_ids_owned = &.{};
            self.prompt_ids_alloc = null;
        }
        self.xfm.markQsaPooledRopeStale();
        if (self.mtp_cache) |*mc| {
            mc.deinit();
            self.mtp_cache = null;
        }
        if (self.dflash_ctx) |*dc| {
            dc.deinit();
            self.dflash_ctx = null;
        }
        if (self.glm_prefill_window) |*w| {
            w.snapshot.deinit();
            self.glm_prefill_window = null;
        }
        if (self.mtp_hist_stash) |*st| {
            st.deinit();
            self.mtp_hist_stash = null;
        }
        if (self.mtp_pre_draft) |*pd| {
            pd.deinit(allocator);
            self.mtp_pre_draft = null;
        }
        if (self.mtp_lookup_idx) |*idx| {
            idx.deinit();
            self.mtp_lookup_idx = null;
        }
        // Publish the EV surface for the next request's seed.
        // Only healthy runs qualify — a runtime-disabled or barely-sampled
        // run would poison the next request's plans (inference thread only,
        // same discipline as every other head-state write).
        if (self.mtp) |head| {
            // A pinned-depth run never consulted the EV plan, so its surface
            // is not one the controller chose — publishing it would hand a
            // later ordinary request a diagnostic's numbers.
            if (mtpAdaptiveEnabled() and mtpPinnedDepth() == null and
                !self.spec_disabled_runtime and self.mtp_attempted >= 8 and
                (!group_planner.enabled() or !self.mtp_planner_owned))
            {
                head.setEvSeed(self.mtp_ev_accept, self.mtp_ev_m_lo_prev);
            }
        }
        // Free any SSM checkpoints the caller didn't claim. Each layer-slice
        // was allocated by `ssm_checkpoint_alloc` (= the allocator passed to
        // `initWithOptions`), so we use that one. The ArrayList itself was
        // also created with that allocator.
        if (self.ssm_checkpoint_alloc) |a| {
            for (self.ssm_checkpoints.items) |*cp| cp.deinit(a);
            self.ssm_checkpoints.deinit(a);
            self.ssm_checkpoints = std.ArrayList(SSMCheckpoint).empty;
            self.ssm_checkpoint_alloc = null;
        } else {
            // Defensive: if init never set it, the list is empty — but the
            // backing ArrayList state may still need a deinit call. Use the
            // passed allocator as a fallback.
            self.ssm_checkpoints.deinit(allocator);
            self.ssm_checkpoints = std.ArrayList(SSMCheckpoint).empty;
        }
        self.generated_ids.deinit(allocator);
    }

    /// Transfer ownership of accumulated SSM checkpoints to the caller.
    /// Returns an owned slice; caller must free each `SSMCheckpoint` via
    /// `cp.deinit(allocator)` and the slice itself via `allocator.free`,
    /// where `allocator` is the same one passed to `initWithOptions`.
    /// After return, `ssm_checkpoints` is empty and the Generator owns
    /// nothing related to checkpoints.
    pub fn takeSsmCheckpoints(self: *Generator) []SSMCheckpoint {
        const a = self.ssm_checkpoint_alloc orelse return &[_]SSMCheckpoint{};
        const out = self.ssm_checkpoints.toOwnedSlice(a) catch return &[_]SSMCheckpoint{};
        return out;
    }

    /// Legacy→batched transition (scheduler.runBatchedDecodeTick): consume
    /// the lazy pipeline state so the slot can join a batched tick. The
    /// legacy pipelined decode keeps a lookahead token ALREADY FORWARDED
    /// into the KV cache (`pending_token` / `next_token_id`) plus
    /// `pending_logits` for the position after it. Dropping that state and
    /// re-forwarding `next_token_id` would append a duplicate position to
    /// the cache and re-emit an already-emitted token — corrupting every
    /// stream whose slot enters a batch mid-generation
    /// (tests/test_batched_transition.sh).
    ///
    /// Returns the token to emit this step (the pipelined lookahead), or
    /// null when generation stopped (`checkStop`: EOS / pad-run /
    /// max_tokens / timeout — `finish_reason` is set). On return:
    /// `next_token_id` is sampled but NOT in the cache and pending state is
    /// empty — exactly the batched-tick entry invariant.
    pub fn drainPipelineForBatch(self: *Generator, allocator: std.mem.Allocator) !?u32 {
        try self.resolvePendingToken();
        if (try self.checkStop()) {
            if (self.has_pending_logits) {
                _ = mlx.mlx_array_free(self.pending_logits);
                self.has_pending_logits = false;
            }
            return null;
        }
        // Both pipeline shapes (fresh-from-prefill and post-`next()` fast
        // path) carry pending_logits alongside the in-cache lookahead; a
        // lookahead without logits would force a re-forward of an in-cache
        // token, which is the corruption this method exists to prevent.
        if (!self.has_pending_logits) return error.MissingPendingLogits;

        const token = self.next_token_id;
        self.advanceStep(1);
        try self.generated_ids.append(allocator, token);

        const step_logits = self.pending_logits;
        self.has_pending_logits = false;
        const lazy = blk: {
            defer _ = mlx.mlx_array_free(step_logits);
            break :blk try self.sampleLazy(step_logits, .committed);
        };
        try mlx.check(mlx.mlx_array_eval(lazy));
        var val: i32 = 0;
        try mlx.check(mlx.mlx_array_item_int32(&val, lazy));
        _ = mlx.mlx_array_free(lazy);
        self.next_token_id = @intCast(val);
        return token;
    }

    /// Resolve the deferred pending token: eval the lazy array and extract the u32 value.
    /// This is called at the START of each iteration, giving the GPU maximum time
    /// to compute since the async_eval at the END of the previous iteration.
    /// The ONE lazy sampler for a slot's own draws: advances the seed draw index. `after` says
    /// what precedes the sampled position, which decides the think penalty there.
    pub fn sampleLazy(self: *Generator, logits: mlx.mlx_array, after: ThinkAfter) !mlx.mlx_array {
        const shifted = try self.thinkShifted(logits, after);
        defer if (shifted.ctx != logits.ctx) {
            _ = mlx.mlx_array_free(shifted);
        };
        defer self.sampling.draw +%= 1;
        // A penalised slot reaches here only from the constrained steps and a forced call's
        // name choice, whose logits follow `generated_ids` (`samplesSync`, `draftsRefused`).
        return sampler.penalizedSampleLazy(std.heap.page_allocator, shifted, self.sampling, self.generated_ids.items, self.xfm.s);
    }

    /// This slot samples on the synchronous serial path (`sampleToken`), never the lazy pipeline.
    pub fn samplesSync(self: *const Generator) bool {
        return self.logprobs_n > 0 or sampler.penaltyActive(self.sampling);
    }

    /// What precedes a sampled position beyond the committed `generated_ids`.
    pub const ThinkAfter = union(enum) {
        committed,
        /// Decided ids not committed yet (t1, a block decoder's accepted prefix).
        decided: []const u32,
        /// The pipelined token still in flight, `[1]` or `[1, 1]`.
        pending: mlx.mlx_array,
        /// No penalty here: a draft's proposal, or rows a verify already shifted.
        none,
    };

    /// `logits` under the think penalty where the span is open; the input itself when nothing
    /// shifts, else a new handle the caller frees.
    pub fn thinkShifted(self: *Generator, logits: mlx.mlx_array, after: ThinkAfter) !mlx.mlx_array {
        const tp = &self.sampling.think_penalty;
        if (!tp.active() or after == .none) return logits;
        tp.observe(self.generated_ids.items);
        var state = tp.*;
        switch (after) {
            .decided => |ids| state.phase = tp.phaseAfter(ids),
            else => {},
        }
        if (!tp.hasBias() and (self.xfm.think_marker_mask == null or state.phase == .after or (after != .pending and state.phase != .inside))) return logits;
        const s = self.xfm.s;
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        if (after == .pending) {
            var ids = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ids);
            try mlx.check(mlx.mlx_reshape(&ids, after.pending, &[_]c_int{ 1, 1 }, 2, s));
            try think_penalty.shiftScoped(&out, logits, self.xfm.think_marker_mask, state, ids, s);
        } else {
            try think_penalty.shiftScoped(&out, logits, self.xfm.think_marker_mask, state, null, s);
        }
        return out;
    }

    /// Shift a verify block in place: row j follows the committed ids, then `ids[0..j]`
    /// (`[1, L]`, the rows the forward ran). True = the block changed.
    pub fn thinkShiftRows(self: *Generator, logits: *mlx.mlx_array, ids: mlx.mlx_array) !bool {
        const tp = &self.sampling.think_penalty;
        if (!tp.active()) return false;
        tp.observe(self.generated_ids.items);
        if (!tp.hasBias() and (self.xfm.think_marker_mask == null or tp.phase == .after)) return false;
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        try think_penalty.shiftScoped(&out, logits.*, self.xfm.think_marker_mask, tp.*, ids, self.xfm.s);
        _ = mlx.mlx_array_free(logits.*);
        logits.* = out;
        return true;
    }

    /// Row j of a verify block under the request's repeat, presence and frequency penalties, over the committed
    /// ids and `ids[0..j]` (inclusive): the logits its serial tick would sample. Drafts stay raw.
    pub fn penaltyShiftRows(self: *Generator, logits: *mlx.mlx_array, ids: mlx.mlx_array) !bool {
        if (!sampler.penaltyActive(self.sampling)) return false;
        const shape = mlx.getShape(logits.*);
        const committed = try seen_penalty.buildCounts(std.heap.page_allocator, self.generated_ids.items, @intCast(shape[shape.len - 1]));
        defer _ = mlx.mlx_array_free(committed);
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        try seen_penalty.shiftRows(&out, logits.*, ids, committed, sampler.penaltiesOf(self.sampling), self.xfm.s);
        _ = mlx.mlx_array_free(logits.*);
        logits.* = out;
        return true;
    }

    /// The host-decided penalty for a sample after the committed ids, for `sampleToken`.
    fn markerShiftCommitted(self: *Generator) ?sampler.MarkerShift {
        const tp = &self.sampling.think_penalty;
        if (!tp.active()) return null;
        tp.observe(self.generated_ids.items);
        if (!tp.hasBias() and (self.xfm.think_marker_mask == null or tp.phase != .inside)) return null;
        return .{ .mask = self.xfm.think_marker_mask, .state = tp.* };
    }

    /// What a forced tool call makes of the next position.
    fn forcedNext(self: *Generator) CallForce.Decision {
        const cf = self.sampling.call_force orelse return .sample;
        return cf.decide(self.generated_ids.items);
    }

    /// A sample restricted to `ids`, the branch of a forced call's name.
    fn sampleAmong(self: *Generator, allocator: std.mem.Allocator, logits: mlx.mlx_array, ids: []const u32) !mlx.mlx_array {
        const shape = mlx.getShape(logits);
        const mask = try allocator.alloc(bool, @intCast(shape[shape.len - 1]));
        defer allocator.free(mask);
        @memset(mask, false);
        for (ids) |id| {
            if (id < mask.len) mask[id] = true;
        }
        var masked = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(masked);
        try sampler.applyGrammarMask(allocator, &masked, logits, mask, self.xfm.s);
        return self.sampleLazy(masked, .committed);
    }

    /// While a forced call is armed every successor is decided AFTER its
    /// predecessor is known, so the pipelined fast path stands aside.
    fn callForcePending(self: *const Generator) bool {
        const cf = self.sampling.call_force orelse return false;
        return cf.pending();
    }

    fn resolvePendingToken(self: *Generator) !void {
        if (!self.has_pending_token) return;
        try mlx.check(mlx.mlx_array_eval(self.pending_token));
        var val: i32 = 0;
        try mlx.check(mlx.mlx_array_item_int32(&val, self.pending_token));
        _ = mlx.mlx_array_free(self.pending_token);
        self.has_pending_token = false;
        self.next_token_id = @intCast(val);
    }

    const DrainResult = union(enum) {
        /// No pending pipeline state — the spec entry invariant already holds.
        already_clean,
        /// One token was emitted while draining; caller returns it this step.
        drained: u32,
        /// The drained token hit a stop condition — generation is over.
        stopped,
        /// Unexpected half-state; do not re-enable speculation.
        stay_disabled,
    };

    /// Transition from the pipelined `next()` state back to the spec-decode
    /// entry invariant (next_token_id known but NOT in cache, no pending
    /// state). The pipeline holds `pending_token` (lazy, its forward already
    /// in the cache) and `pending_logits` (logits for the position after it):
    /// resolving the token, emitting it, and sampling its successor from
    /// `pending_logits` WITHOUT forwarding lands exactly on the invariant.
    /// One sync. Also handles the shim-seeded state (`pending_logits` only).
    pub fn drainPipelineForSpec(self: *Generator, allocator: std.mem.Allocator) !DrainResult {
        if (!self.has_pending_logits) {
            if (!self.has_pending_token) return .already_clean;
            // pending_token without pending_logits never occurs in the
            // pipelined state machine; bail rather than risk the invariant.
            return .stay_disabled;
        }
        try self.resolvePendingToken();
        if (try self.checkStop()) return .stopped;
        const token = self.next_token_id;
        self.advanceStep(1);
        try self.generated_ids.append(allocator, token);

        const step_logits = self.pending_logits;
        self.has_pending_logits = false;
        // With the latch these checks return instead of ending the process, so the handle needs an owner.
        const val: i32 = blk: {
            const lazy = sampled: {
                defer _ = mlx.mlx_array_free(step_logits);
                break :sampled try self.sampleLazy(step_logits, .committed);
            };
            defer _ = mlx.mlx_array_free(lazy);
            try mlx.check(mlx.mlx_array_eval(lazy));
            var v: i32 = 0;
            try mlx.check(mlx.mlx_array_item_int32(&v, lazy));
            break :blk v;
        };
        self.next_token_id = @intCast(val);
        return .{ .drained = token };
    }

    pub const ForcedCommit = struct {
        /// Tokens emitted by this call: any pending token plus every forced
        /// token but the last, which stays as `next_token_id` for the next
        /// regular tick to publish.
        emitted: []const u32,
        stopped: bool,
    };

    /// Commit `forced` through the model as ONE multi-token forward, from any
    /// of the three inter-tick states (pipelined, spec exit, shim-seeded), and
    /// leave the generator in the shim-seeded state a regular `next()` reads.
    /// The caller routes every later tick through the regular path.
    pub fn commitForcedTokens(self: *Generator, allocator: std.mem.Allocator, forced: []const u32) !ForcedCommit {
        std.debug.assert(forced.len > 0);
        var emitted = std.ArrayList(u32).empty;
        errdefer emitted.deinit(allocator);
        var input = std.ArrayList(u32).empty;
        defer input.deinit(allocator);

        if (self.has_pending_token or self.has_pending_logits) {
            // Pipelined or shim-seeded: `next_token_id` is in the cache, unpublished.
            try self.resolvePendingToken();
            if (try self.checkStop()) return .{ .emitted = try emitted.toOwnedSlice(allocator), .stopped = true };
            try self.generated_ids.append(allocator, self.next_token_id);
            try emitted.append(allocator, self.next_token_id);
            self.advanceStep(1);
            if (self.has_pending_logits) {
                _ = mlx.mlx_array_free(self.pending_logits);
                self.has_pending_logits = false;
            }
        } else {
            // Spec exit: `next_token_id` is decided but not in the cache.
            try input.append(allocator, self.next_token_id);
            try self.generated_ids.append(allocator, self.next_token_id);
            try emitted.append(allocator, self.next_token_id);
            self.advanceStep(1);
        }
        try input.appendSlice(allocator, forced);
        for (forced[0 .. forced.len - 1]) |f| {
            try self.generated_ids.append(allocator, f);
            try emitted.append(allocator, f);
        }
        self.advanceStep(@intCast(forced.len - 1));
        self.consecutive_pad = 0;
        self.loop_guard_start = self.generated_ids.items.len;

        const ids_i32 = try allocator.alloc(i32, input.items.len);
        defer allocator.free(ids_i32);
        for (input.items, 0..) |t, i| ids_i32[i] = @intCast(t);
        const shape = [_]c_int{ 1, @intCast(input.items.len) };
        const tok_input = mlx.mlx_array_new_data(ids_i32.ptr, &shape, 2, .int32);
        defer _ = mlx.mlx_array_free(tok_input);
        const all_logits = try self.xfm.forwardWith(&self.ctx, tok_input);
        defer _ = mlx.mlx_array_free(all_logits);
        const shp = mlx.getShape(all_logits);
        var logits = mlx.mlx_array_new();
        const start = [_]c_int{ 0, shp[1] - 1, 0 };
        const stop = [_]c_int{ shp[0], shp[1], shp[2] };
        const strides = [_]c_int{ 1, 1, 1 };
        try mlx.check(mlx.mlx_slice(&logits, all_logits, &start, 3, &stop, 3, &strides, 3, self.xfm.s));
        const arr = [_]mlx.mlx_array{logits};
        const vec = mlx.mlx_vector_array_new_data(&arr, 1);
        _ = mlx.mlx_async_eval(vec);
        _ = mlx.mlx_vector_array_free(vec);
        self.pending_logits = logits;
        self.has_pending_logits = true;
        self.next_token_id = forced[forced.len - 1];
        self.mtp_hidden_stale = true;
        return .{ .emitted = try emitted.toOwnedSlice(allocator), .stopped = false };
    }

    /// The absolute position the request's state stands at: GLM's request offset, else the model's position counter.
    fn thoughtStatePos(self: *const Generator) usize {
        if (self.ctx.glm5_request) |request| return request.offset;
        return self.ctx.moe_seq_offset.*;
    }

    /// A batched tick keeps the slot's recurrent state in the group's merged arrays, not in its entries.
    fn thoughtStateInGroup(self: *const Generator) bool {
        const entries = self.ctx.ssm_entries orelse return false;
        for (entries) |*e| if (e.group) |g| if (g.live) return true;
        return false;
    }

    /// A speculative round's part in the thought point: how many of its emitted tokens `path` (the first at absolute
    /// position `anchor`) to keep so the state ends where the point sits; null runs the round whole.
    pub fn thoughtRoundCut(self: *Generator, path: []const u32, anchor: usize) ?usize {
        const tp = &(self.thought_point orelse return null);
        if (tp.state != .armed or self.thoughtStateInGroup()) return null;
        switch (mtp_plan.thoughtRound(tp.*, path, anchor)) {
            .whole => return null,
            .skip => {
                tp.state = .passed;
                return null;
            },
            .keep => |n| {
                tp.state = .cut;
                tp.at = anchor + n;
                return n;
            },
        }
    }

    /// After the cut round committed: the state it stands on becomes the newest checkpoint.
    pub fn thoughtPointTake(self: *Generator) void {
        const tp = &(self.thought_point orelse return);
        if (tp.state != .cut) return;
        tp.state = .passed;
        const a = self.ssm_checkpoint_alloc orelse return;
        const pos = self.thoughtStatePos();
        if (pos != tp.at) {
            log.debug("[prefix] no thought restore point: the state stands at {d}, not {d}\n", .{ pos, tp.at });
            return;
        }
        var cp = capturePrefillCheckpoint(a, self.xfm, &self.ctx, self.ctx.glm5_request, pos, false) catch |err| {
            log.warn("[prefix] thought restore point at {d} not taken: {s}\n", .{ pos, @errorName(err) });
            return;
        };
        if (self.mtp_cache != null) self.xfm.qwen4MtpMarkQsaLeftover(pos -| self.mtp_position_base) catch {};
        self.ssm_checkpoints.append(a, cp) catch {
            cp.deinit(a);
            return;
        };
        tp.state = .taken;
        log.info("[prefix] thought restore point at {d} ({d} tokens in)\n", .{ pos, pos - tp.base });
    }

    /// Frees the thought point the commit must not carry.
    pub fn dropThoughtPoint(self: *Generator) void {
        const tp = &(self.thought_point orelse return);
        if (tp.state != .taken) return;
        tp.state = .passed;
        const a = self.ssm_checkpoint_alloc orelse return;
        var cp = self.ssm_checkpoints.pop() orelse return;
        cp.deinit(a);
    }

    /// The ONE place a partial accept rolls recurrent state back from the verify capture:
    /// the loop and the `partial_rounds` count travel together, so a caller cannot take the
    /// rollback without being counted.
    pub fn rollbackSsmFromCapture(self: *Generator, entries: []transformer_mod.SSMCacheEntry, accepted: u32, verify_len: u32, s: mlx.mlx_stream) !void {
        self.partial_rounds += 1;
        for (entries) |*entry| {
            try transformer_mod.ssmRollbackFromCapture(entry, accepted, verify_len, s);
        }
    }

    /// Result of one speculative step: the tokens to emit and how many drafts landed.
    pub const DrafterStepResult = struct {
        /// Tokens to emit this step. On a full accept this is
        /// `[t1, ...all_drafts]` (length `block_size`); on partial accept j
        /// it is `[t1, draft[0..j]]` (length `1+j`). The corrected fallback
        /// becomes `next_token_id` for the next call.
        tokens: []const u32,
        /// Number of *drafted* tokens accepted (excludes always-accepted t1).
        accepted_tokens: u32,
    };

    pub const glmLookupChain = glm5_generate.glmLookupChain;
    pub const glmPenalty = glm5_generate.glmPenalty;
    pub const glmPlainProposal = glm5_generate.glmPlainProposal;
    pub const glmPropose = glm5_generate.glmPropose;
    pub const glmProposeRound = glm5_generate.glmProposeRound;
    pub const glmRoundEnd = glm5_generate.glmRoundEnd;
    pub const glmRoundReady = glm5_generate.glmRoundReady;
    pub const glmRowCommit = glm5_generate.glmRowCommit;
    pub const glm_lookup_drafts = glm5_generate.glm_lookup_drafts;
    pub const nextGlmDflash = glm5_generate.nextGlmDflash;
    pub const sampleGlmNode = glm5_generate.sampleGlmNode;
    pub const sampledGlmDecisions = glm5_generate.sampledGlmDecisions;

    pub const dsparkConstrainedEntry = dsv41_dspark.dsparkConstrainedEntry;
    pub const dsparkEmitPending = dsv41_dspark.dsparkEmitPending;
    pub const dsparkJudgePending = dsv41_dspark.dsparkJudgePending;
    pub const dsparkNoteSerial = dsv41_dspark.dsparkNoteSerial;
    pub const dsparkObserve = dsv41_dspark.dsparkObserve;
    pub const dsparkResumeAfterForced = dsv41_dspark.dsparkResumeAfterForced;
    pub const dsparkSamplePending = dsv41_dspark.dsparkSamplePending;
    pub const dsparkSampledVerdict = dsv41_dspark.dsparkSampledVerdict;
    pub const nextDspark = dsv41_dspark.nextDspark;
    pub const rowView = dsv41_dspark.rowView;

    // Cap accepted drafts with capAcceptedForTokenBudget before committing.
    pub fn nextDflash(self: *Generator, allocator: std.mem.Allocator) !?DrafterStepResult {
        if (self.glm_dflash_native) return self.nextGlmDflash(allocator);
        // The kv term is the same physics for either block decoder (one
        // forward, one KV read, shared across the block's rows), so a DFlash
        // round is an observation for it too — and on a DFlash-only server
        // it is the ONLY source.
        const dflash_kv_watch = io_util.Stopwatch.init(self.timer.io);
        // The round-cost table sees the same round: drafts = block - 1, or
        // 0 when the runtime gate fell back to serial.
        const dflash_gen_before = self.generated_ids.items.len;
        self.dflash_round_width = 0;
        const dflash_rounds_before: u64 = self.dflash_attempted;
        defer {
            const ms = @as(f32, @floatFromInt(dflash_kv_watch.read())) / @as(f32, std.time.ns_per_ms);
            const emitted = self.generated_ids.items.len - dflash_gen_before;
            const post_warmup = dflash_rounds_before >= glm5_generate.DFLASH_GATE_WARMUP;
            const wall = if (post_warmup) self.mtpRegimeWallMs(ms) else ms;
            self.specObserveRound(self.dflash_round_width, wall, @floatFromInt(emitted), post_warmup and emitted > 0, false);
            if (self.dflash_round_width > 0) self.dflash_wall_ms += wall;
        }
        if (self.done) return null;
        std.debug.assert(self.dflash != null);
        std.debug.assert(self.dflash_ctx != null or self.spec_disabled_runtime);
        if (mtp_plan.specDecodeUnsupported(self.sampling, self.logprobs_n)) return error.SpecDecodeUnsupported;

        if (self.completion_tokens >= self.max_tokens) {
            self.done = true;
            self.finish_reason = "length";
            return null;
        }

        if (self.spec_disabled_runtime) {
            const tok_opt = try self.next(allocator);
            if (tok_opt == null) return null;
            const tokens = try allocator.alloc(u32, 1);
            tokens[0] = tok_opt.?;
            return DrafterStepResult{ .tokens = tokens, .accepted_tokens = 0 };
        }

        const round_width: u32 = @max(self.dflash_block_size, 2) - 1;
        self.dflash_round_width = round_width;

        if (try self.checkStop()) return null; // t1 is this block's first emit: stop before drafting

        const xfm = self.xfm;
        const s = xfm.s;
        const model = self.dflash.?;
        const dctx = &self.dflash_ctx.?;
        const bs: u32 = round_width + 1;
        const m_full: u32 = bs - 1;
        // Drafts this round verifies: all of the block's, or the policy's rows - 1 once the selector has spoken.
        var m: u32 = m_full;
        const t1: u32 = self.next_token_id;
        // Positions come from moe_seq_offset (cache.step is a bookkeeping
        // counter the model never reads — same rule as nextMtp).
        const anchor_pos: usize = self.ctx.moe_seq_offset.*;
        const kv_step_snap = self.ctx.cache.step;
        std.debug.assert(dctx.absLen() == anchor_pos);

        const tracing = dflashTraceEnabled();
        var ph: io_util.Stopwatch = undefined;
        if (tracing) {
            ph = io_util.Stopwatch.init(self.timer.io);
            if (self.dflash_gap_watch) |*gw| self.dflash_trace.add(.gap, gw.read());
            self.dflash_gap_watch = null;
        }

        // ── Phase 1: one assistant forward drafts all m tokens ──
        // DFlash reads mask rows 1..bs-1 (anchor row dropped).
        const noise_rows: u32 = bs;
        const noise_ids = try allocator.alloc(i32, noise_rows);
        defer allocator.free(noise_ids);
        noise_ids[0] = @intCast(t1);
        for (noise_ids[1..]) |*v| v.* = @intCast(model.config.mask_token_id);
        const noise_shape = [_]c_int{ 1, @intCast(noise_rows) };
        const noise_input = mlx.mlx_array_new_data(noise_ids.ptr, &noise_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(noise_input);
        // RAW table rows — no embed norm, no scale (the DFlash contract).
        const noise_embeds = try xfm.rawEmbedding(noise_input);
        defer _ = mlx.mlx_array_free(noise_embeds);

        const blk_hidden = try dflash_mod.forwardBlock(model, dctx, noise_embeds, anchor_pos);
        defer _ = mlx.mlx_array_free(blk_hidden);
        if (tracing) {
            try mlx.check(mlx.mlx_array_eval(blk_hidden));
            self.dflash_trace.add(.assist, ph.read());
            ph.reset();
        }
        const draft_logits_all = try model.draftLogits(xfm, blk_hidden);
        defer _ = mlx.mlx_array_free(draft_logits_all);

        // Slice the m draft rows: DFlash drops the anchor row (1..bs-1).
        const dl_shape = mlx.getShape(draft_logits_all);
        var draft_logits = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(draft_logits);
        {
            const row0: c_int = 1;
            const start = [_]c_int{ 0, row0, 0 };
            const stop = [_]c_int{ 1, row0 + @as(c_int, @intCast(m_full)), dl_shape[2] };
            const strides = [_]c_int{ 1, 1, 1 };
            try mlx.check(mlx.mlx_slice(&draft_logits, draft_logits_all, &start, 3, &stop, 3, &strides, 3, s));
        }
        // Sampled requests accept through the full Leviathan ratio
        // min(1, p/q), so a GREEDY draft is a one-hot q — and at temperature
        // the target row is flat, which is exactly where min(1, p(argmax))
        // collapses. Drafting FROM the request's own distribution makes q
        // track p and keeps acceptance flat across temperature. Greedy
        // requests keep the argmax path untouched, so the byte-equality
        // guard is unaffected.
        const stochastic = !sampler.isGreedyTemperature(self.sampling.temperature);
        // DFlash2 path selector: when the sidecar ships one, drafts come from
        // the pairwise-scored path trace instead of per-position argmax /
        // block sampling. Greedy requests keep the byte-equality bar (a
        // selector draft only survives verify if it IS the trunk argmax);
        // stochastic requests sample the selector's own candidate softmax and
        // accept through min(1, p/q) with q read off the traced path —
        // exact by construction.
        const use_selector = model.selector != null;
        var sel_path: ?dflash_mod.SelectedPath = null;
        defer if (sel_path) |*sp| sp.deinit(allocator);
        var draft_ids = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(draft_ids);
        if (use_selector) {
            const sel_temp: f32 = if (stochastic) self.sampling.temperature else 0.0;
            sel_path = try dflash_mod.selectPath(
                allocator,
                &model.selector.?,
                model.config.selector_top_k,
                blk_hidden,
                draft_logits,
                t1,
                sel_temp,
                self.prng.random(),
                s,
            );
            if (self.dflashAdaptiveRows()) {
                m = dflash_policy.pickRows(&self.dflash_calib, sel_path.?.conf, dflash_mod.qwenMinRows(), self.dflash_block_size) - 1;
                self.dflash_round_width = m;
            }
            const ids_i32 = try allocator.alloc(i32, m);
            defer allocator.free(ids_i32);
            for (sel_path.?.ids[0..m], ids_i32) |v, *d| d.* = @intCast(v);
            const row_shape = [_]c_int{ 1, @intCast(m) };
            const host_arr = mlx.mlx_array_new_data(ids_i32.ptr, &row_shape, 2, .int32);
            defer _ = mlx.mlx_array_free(host_arr);
            try mlx.check(mlx.mlx_array_set(&draft_ids, host_arr));
        } else {
            var draft_amax = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(draft_amax);
            try mlx.check(mlx.mlx_argmax_axis(&draft_amax, draft_logits, 2, false, s));
            try mlx.check(mlx.mlx_astype(&draft_ids, draft_amax, .int32, s));
        }
        if (tracing) {
            try mlx.check(mlx.mlx_array_eval(draft_ids));
            self.dflash_trace.add(.head, ph.read());
            ph.reset();
        }

        // ── Phase 2: verify input [t1, drafts...] — [1, bs] int32, lazy ──
        const t1_i32: i32 = @intCast(t1);
        const t1_shape = [_]c_int{ 1, 1 };
        const t1_arr = mlx.mlx_array_new_data(&t1_i32, &t1_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(t1_arr);
        var verify_input = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(verify_input);
        {
            const vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(vec);
            _ = mlx.mlx_vector_array_append_value(vec, t1_arr);
            _ = mlx.mlx_vector_array_append_value(vec, draft_ids);
            try mlx.check(mlx.mlx_concatenate_axis(&verify_input, vec, 1, s));
        }

        // Verify forward with layer captures — this round's verify IS the
        // next round's context producer.
        const cap_out = try allocator.alloc(mlx.mlx_array, model.config.target_layer_ids.len);
        defer {
            for (cap_out) |a| _ = mlx.mlx_array_free(a);
            allocator.free(cap_out);
        }
        for (cap_out) |*a| a.* = mlx.mlx_array_new();
        var cl = transformer_mod.CaptureLayers{ .ids = model.config.target_layer_ids, .out = cap_out, .hc_mean = xfm.config.hc_count };
        self.ctx.capture_layers = &cl;
        defer self.ctx.capture_layers = null;
        // Per-position SSM capture on a GDN trunk so partial accept can roll
        // back the recurrent state without re-forwarding (mirrors nextMtp).
        self.ctx.capture_ssm_seq = self.ctx.ssm_entries != null;
        var verify_logits = try xfm.forwardWith(&self.ctx, verify_input);
        errdefer _ = mlx.mlx_array_free(verify_logits);
        _ = try self.thinkShiftRows(&verify_logits, verify_input);
        self.ctx.capture_ssm_seq = false;
        defer if (self.ctx.ssm_entries) |entries| {
            for (entries) |*entry| transformer_mod.ssmFreeSpecCapture(entry);
        };
        self.dflash_attempted += 1;
        if (tracing) {
            // Captures ride the same forward — eval them here or their cost
            // is billed to `append`.
            const vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(vec);
            _ = mlx.mlx_vector_array_append_value(vec, verify_logits);
            for (cap_out) |a| _ = mlx.mlx_vector_array_append_value(vec, a);
            try mlx.check(mlx.mlx_eval(vec));
            self.dflash_trace.add(.verify, ph.read());
            ph.reset();
        }

        // ── Phase 3: decide the longest accepted prefix ──
        const vl_shape = mlx.getShape(verify_logits);
        var per_pos_logits: ?[]mlx.mlx_array = null;
        var per_pos_logits_n: usize = 0;
        defer if (per_pos_logits) |slots| {
            for (slots[0..per_pos_logits_n]) |arr| _ = mlx.mlx_array_free(arr);
            allocator.free(slots);
        };
        if (stochastic) {
            const slots = try allocator.alloc(mlx.mlx_array, m + 1);
            per_pos_logits = slots;
            const slice_strides = [_]c_int{ 1, 1, 1 };
            for (slots, 0..) |*slot, idx| {
                slot.* = mlx.mlx_array_new();
                per_pos_logits_n = idx + 1;
                const start = [_]c_int{ 0, @intCast(idx), 0 };
                const stop = [_]c_int{ vl_shape[0], @as(c_int, @intCast(idx)) + 1, vl_shape[2] };
                try mlx.check(mlx.mlx_slice(slot, verify_logits, &start, 3, &stop, 3, &slice_strides, 3, s));
            }
        }
        // Committed entries: masked like every other block decoder's.
        var verify_argmax: CommittedArgmax = .{};
        defer verify_argmax.deinit();
        if (!stochastic) {
            verify_argmax = try verifyArgmax(verify_logits, self.sampling.suppress_mask, s);
        }
        _ = mlx.mlx_array_free(verify_logits);
        verify_logits = .{ .ctx = null };

        // One batched dispatch, then read (drafts branch + verify branch are
        // separate graphs — eval BOTH before reading either; the v26.5.6
        // 0%-acceptance class).
        {
            const eval_vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(eval_vec);
            _ = mlx.mlx_vector_array_append_value(eval_vec, draft_ids);
            if (!stochastic) _ = mlx.mlx_vector_array_append_value(eval_vec, verify_argmax.lazy());
            try mlx.check(mlx.mlx_async_eval(eval_vec));
        }
        var drafts = try allocator.alloc(u32, m);
        errdefer allocator.free(drafts);
        if (sel_path) |*sp| {
            @memcpy(drafts, sp.ids[0..m]);
        } else {
            try mlx.check(mlx.mlx_array_eval(draft_ids));
            const draft_data = mlx.mlx_array_data_int32(draft_ids) orelse return error.MlxArrayDataNull;
            for (drafts, 0..) |*d, idx| d.* = @intCast(draft_data[idx]);
        }
        if (!stochastic) try mlx.check(mlx.mlx_array_eval(verify_argmax.lazy()));

        var accepted: u32 = 0;
        if (stochastic) {
            var k: u32 = 0;
            while (k < m) : (k += 1) {
                const target_p = try sampler.probsAtLastPos(per_pos_logits.?[k], self.sampling, s);
                defer _ = mlx.mlx_array_free(target_p);
                const p_draft = try sampler.probAt(target_p, drafts[k], s);
                const accept_prob: f32 = if (sel_path) |*sp| blk: {
                    // Selector-sampled draft: q is the traced step's own
                    // candidate softmax — exact, no GPU read.
                    const kk = sp.cand_ids.len / @as(usize, m_full);
                    break :blk specAcceptProb(p_draft, sp.q.?[@as(usize, k) * kk + sp.chosen_idx[k]]);
                } else @min(1.0, p_draft);
                const u: f32 = self.prng.random().float(f32);
                if (u >= accept_prob) break;
                accepted += 1;
            }
        } else {
            const argmax_data = try verify_argmax.ids(1 + m);
            var k: u32 = 0;
            while (k < m) : (k += 1) {
                if (@as(u32, @intCast(argmax_data[k])) != drafts[k]) break;
                accepted += 1;
            }
        }

        self.dflash_drafts_verified += m;
        if (sel_path) |*sp| self.dflash_calib.observeRound(sp.conf, m, accepted);

        // A speculative verify may accept the whole block even when only a
        // few output positions remain. Cap the accepted prefix BEFORE cache
        // and assistant-context commit so every representation of the turn —
        // returned tokens, generated_ids, usage, KV and DFlash context — ends
        // at exactly max_tokens.
        accepted = mtp_plan.capAcceptedForTokenBudget(
            accepted,
            self.completion_tokens,
            self.max_tokens,
        );

        const next_pending: u32 = blk: {
            if (stochastic) {
                const correction_logits = per_pos_logits.?[accepted];
                const probs = try sampler.probsAtLastPos(correction_logits, self.sampling, s);
                defer _ = mlx.mlx_array_free(probs);
                if (accepted < m) {
                    // norm(max(0, p − q)) against the q the draft was drawn
                    // from — a one-hot here is the WRONG residual for a
                    // sampled draft, and wrong silently.
                    const q_row = if (sel_path) |*sp|
                        try sampler.selectorQRow(sp, accepted, m_full, vl_shape[2], s)
                    else
                        try sampler.pldOneHotRow(drafts[accepted], vl_shape[2], s);
                    defer _ = mlx.mlx_array_free(q_row);
                    break :blk try sampler.sampleResidual(probs, q_row, s);
                } else {
                    break :blk try sampler.sampleFromProbs(probs, s);
                }
            } else {
                const argmax_data = try verify_argmax.ids(1 + m);
                break :blk @intCast(argmax_data[accepted]);
            }
        };

        if (tracing) {
            self.dflash_trace.add(.accept, ph.read());
            ph.reset();
        }

        // ── Phase 4: commit — truncate on partial accept, then grow the
        // assistant context by exactly the committed positions ──
        const n_commit: usize = 1 + @as(usize, accepted);
        if (accepted < m) {
            try self.ctx.cache.truncate(anchor_pos + n_commit, s);
            // Same bookkeeping as nextMtp's GDN arm: preserve the
            // pre-verify cache.step (prefix-cache kv_step contract),
            // roll every linear layer's recurrent state back to the
            // accepted position from the verify pass's capture, and
            // re-point moe_seq_offset at the committed length.
            self.ctx.cache.step = kv_step_snap;
            if (self.ctx.ssm_entries) |entries| {
                const gdn_captured = entries.len > 0 and transformer_mod.ssmHasSpecCapture(&entries[0]);
                if (!gdn_captured) return error.SpecRollbackUnavailable;
                try self.rollbackSsmFromCapture(entries, accepted, 1 + m, s);
            }
            self.ctx.moe_seq_offset.* = anchor_pos + n_commit;
        }
        if (accepted == m) {
            try dflash_mod.appendContext(model, dctx, cap_out, anchor_pos);
        } else {
            const sliced = try allocator.alloc(mlx.mlx_array, cap_out.len);
            var sliced_n: usize = 0;
            defer {
                for (sliced[0..sliced_n]) |a| _ = mlx.mlx_array_free(a);
                allocator.free(sliced);
            }
            for (cap_out, sliced) |full, *out| {
                out.* = mlx.mlx_array_new();
                sliced_n += 1;
                const fsh = mlx.getShape(full);
                const start = [_]c_int{ 0, 0, 0 };
                const stop = [_]c_int{ 1, @intCast(n_commit), fsh[2] };
                const strides = [_]c_int{ 1, 1, 1 };
                try mlx.check(mlx.mlx_slice(out, full, &start, 3, &stop, 3, &strides, 3, s));
            }
            try dflash_mod.appendContext(model, dctx, sliced, anchor_pos);
        }
        // Materialize the appended context off the critical path so next
        // round's assistant forward doesn't drag this round's verify graph.
        {
            const eval_vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(eval_vec);
            dctx.appendEvalArrays(eval_vec);
            if (tracing) {
                try mlx.check(mlx.mlx_eval(eval_vec));
                self.dflash_trace.add(.append, ph.read());
                ph.reset();
            } else {
                try mlx.check(mlx.mlx_async_eval(eval_vec));
            }
        }

        const tokens = try allocator.alloc(u32, n_commit);
        tokens[0] = t1;
        for (drafts[0..accepted], 0..) |d, idx| tokens[1 + idx] = d;
        try self.generated_ids.append(allocator, t1);
        for (drafts[0..accepted]) |d| try self.generated_ids.append(allocator, d);
        allocator.free(drafts);

        self.dflash_accepted_tokens += accepted;
        self.next_token_id = next_pending;
        self.advanceStep(@intCast(n_commit));
        self.checkDflashRuntimeGate();
        if (self.completion_tokens >= self.max_tokens) {
            self.done = true;
            self.finish_reason = "length";
        }

        if (tracing) {
            self.dflashTraceRoundEnd(accepted);
            self.dflash_gap_watch = io_util.Stopwatch.init(self.timer.io);
        }

        return DrafterStepResult{
            .tokens = tokens,
            .accepted_tokens = accepted,
        };
    }

    /// Rows are chosen per round on Flash-Next, whose verify rows are bytes; every other target
    /// verifies the whole block.
    fn dflashAdaptiveRows(self: *const Generator) bool {
        return self.xfm.qwen4 != null and self.dflash_block_size > dflash_mod.qwenMinRows();
    }

    /// DFlash runtime economics gate. Sticky within the request: once a
    /// `DFLASH_GATE_WARMUP`-round sample proves the block-parallel path yields
    /// less than its width-normalized request-class threshold, subsequent
    /// ticks use the regular pipelined decoder through `nextDflash`'s entry
    /// fallback.
    fn checkDflashRuntimeGate(self: *Generator) void {
        if (self.spec_disabled_runtime) return;
        const serial = dflashSerialPrice(&self.xfm.round_cost, self.xfm.round_cost.bucketOf(self.mtpKvLen()));
        if (!self.dflash_gate_window.shouldDisable(self.dflash_attempted, self.dflash_accepted_tokens, self.dflash_wall_ms, self.dflashMeanRows(), serial)) return;
        const w = self.dflash_gate_window;
        const rounds = self.dflash_attempted - w.attempted;
        const accepted = self.dflash_accepted_tokens - w.accepted;
        log.info(
            "  dflash=disabled (runtime yield {d:.2} accepted/round, {d:.2} ms/token against {d:.2} serial, over {d} rounds after the warmup)\n",
            .{ @as(f32, @floatFromInt(accepted)) / @as(f32, @floatFromInt(rounds)), (self.dflash_wall_ms - w.wall_ms) / @as(f64, @floatFromInt(rounds + accepted)), serial orelse 0.0, rounds },
        );
        self.stopDflash();
    }

    /// Serial decoding for the rest of the request: nothing resumes the drafter, so its window goes now.
    pub fn stopDflash(self: *Generator) void {
        self.spec_disabled_runtime = true;
        if (self.dflash_ctx) |*dc| {
            dc.deinit();
            self.dflash_ctx = null;
        }
    }

    pub const acceptGraphFor = mtp_round.acceptGraphFor;
    pub const acceptPrefixFor = mtp_round.acceptPrefixFor;
    pub const mtpAcceptRowGreedy = mtp_round.mtpAcceptRowGreedy;
    pub const mtpApplyStash = mtp_round.mtpApplyStash;
    pub const mtpBatchHeadPosition = mtp_round.mtpBatchHeadPosition;
    pub const mtpBatchedAcceptGraph = mtp_round.mtpBatchedAcceptGraph;
    pub const mtpBatchedExactGraph = mtp_round.mtpBatchedExactGraph;
    pub const mtpBatchedLossyGraph = mtp_round.mtpBatchedLossyGraph;
    pub const mtpBatchedTokenV3Graph = mtp_round.mtpBatchedTokenV3Graph;
    pub const mtpBatchedTypicalGraph = mtp_round.mtpBatchedTypicalGraph;
    pub const mtpChainBuild = mtp_round.mtpChainBuild;
    pub const mtpChainBuildBatched = mtp_round.mtpChainBuildBatched;
    pub const mtpChainBuildBatchedActive = mtp_round.mtpChainBuildBatchedActive;
    pub const mtpChainDispatch = mtp_round.mtpChainDispatch;
    pub const mtpChainDispatchBatched = mtp_round.mtpChainDispatchBatched;
    pub const mtpChainInit = mtp_round.mtpChainInit;
    pub const mtpCommittedHistoryLen = mtp_round.mtpCommittedHistoryLen;
    pub const mtpCommittedLen = mtp_round.mtpCommittedLen;
    pub const mtpDetachHead = mtp_round.mtpDetachHead;
    pub const mtpDraftActiveRows = mtp_round.mtpDraftActiveRows;
    pub const mtpDraftGreedyAt = mtp_round.mtpDraftGreedyAt;
    pub const mtpDraftGreedyFor = mtp_round.mtpDraftGreedyFor;
    pub const mtpDraftSampling = mtp_round.mtpDraftSampling;
    pub const mtpDraftSamplingFor = mtp_round.mtpDraftSamplingFor;
    pub const mtpDraftStepPath = mtp_round.mtpDraftStepPath;
    pub const mtpDraftTempFor = mtp_round.mtpDraftTempFor;
    pub const mtpFullAccept = mtp_round.mtpFullAccept;
    pub const mtpGreedyVerdict = mtp_round.mtpGreedyVerdict;
    pub const mtpGroupAccept = mtp_round.mtpGroupAccept;
    pub const mtpGroupAcceptBatchable = mtp_round.mtpGroupAcceptBatchable;
    pub const mtpGroupSampledAccept = mtp_round.mtpGroupSampledAccept;
    pub const mtpGroupSampledDrafts = mtp_round.mtpGroupSampledDrafts;
    pub const mtpGroupVerify = mtp_round.mtpGroupVerify;
    pub const mtpKvLen = mtp_round.mtpKvLen;
    pub const mtpLookupAllowed = mtp_round.mtpLookupAllowed;
    pub const mtpLookupChain = mtp_round.mtpLookupChain;
    pub const mtpLookupCostsFor = mtp_round.mtpLookupCostsFor;
    pub const mtpLookupDraftCap = mtp_round.mtpLookupDraftCap;
    pub const mtpLookupIndex = mtp_round.mtpLookupIndex;
    pub const mtpLookupObserve = mtp_round.mtpLookupObserve;
    pub const mtpLookupPriceWidth = mtp_round.mtpLookupPriceWidth;
    pub const mtpMaybePreDraft = mtp_round.mtpMaybePreDraft;
    pub const mtpMropeContext = mtp_round.mtpMropeContext;
    pub const mtpPreDraftOpenAllows = mtp_round.mtpPreDraftOpenAllows;
    pub const mtpResumeAfterForced = mtp_round.mtpResumeAfterForced;
    pub const mtpRetained = mtp_round.mtpRetained;
    pub const mtpRoundAcceptObserve = mtp_round.mtpRoundAcceptObserve;
    pub const mtpRoundAfterChain = mtp_round.mtpRoundAfterChain;
    pub const mtpRoundBegin = mtp_round.mtpRoundBegin;
    pub const mtpRoundContinue = mtp_round.mtpRoundContinue;
    pub const mtpRoundFinish = mtp_round.mtpRoundFinish;
    pub const mtpRoundOff0 = mtp_round.mtpRoundOff0;
    pub const mtpRoundUntimed = mtp_round.mtpRoundUntimed;
    pub const mtpRoundVerify = mtp_round.mtpRoundVerify;
    pub const mtpRowRerankMode = mtp_round.mtpRowRerankMode;
    pub const mtpSamplingDraw = mtp_round.mtpSamplingDraw;
    pub const mtpSerialCaptureReady = mtp_round.mtpSerialCaptureReady;
    pub const mtpSerialCaptureTick = mtp_round.mtpSerialCaptureTick;
    pub const mtpSerialExitAfterForced = mtp_round.mtpSerialExitAfterForced;
    pub const mtpSerialGiveUp = mtp_round.mtpSerialGiveUp;
    pub const mtpSerialMayResume = mtp_round.mtpSerialMayResume;
    pub const mtpSerialOneToken = mtp_round.mtpSerialOneToken;
    pub const mtpSerialTick = mtp_round.mtpSerialTick;
    pub const mtpSerialWallMs = mtp_round.mtpSerialWallMs;
    pub const mtpStopPrefix = mtp_round.mtpStopPrefix;
    pub const nextMtp = mtp_round.nextMtp;
    pub const specAcceptProb = mtp_round.specAcceptProb;

    // Prompt lookup inside the MTP round, ported from mlx-serve #523/#533 (Samuel Reed).
    comptime {
        // A lookup round commits up to MAX_DRAFT_STRONG drafts into `MtpHistStash`.
        std.debug.assert(mtp_lookup.MAX_DRAFT_STRONG <= mtp_mod.MAX_DEPTH);
    }

    // Serial blocks inside nextMtp. Leaving MTP is cheap; coming back needs `t1 NOT in cache`
    // and an `h_prev` for that position: `drainPipelineForSpec` lands the first, the capture
    // tick the second. The head's committed history does not grow across a serial block
    // (a content gap that costs acceptance for a while); M-RoPE turns never come back.

    pub const invalidateRoundClock = mtp_policy.invalidateRoundClock;
    pub const invalidateSerialClock = mtp_policy.invalidateSerialClock;
    pub const mtpAdaptiveArchEligible = mtp_policy.mtpAdaptiveArchEligible;
    pub const mtpAdaptiveBucket = mtp_policy.mtpAdaptiveBucket;
    pub const mtpAdaptiveBucketOf = mtp_policy.mtpAdaptiveBucketOf;
    pub const mtpAdaptiveEnabled = mtp_policy.mtpAdaptiveEnabled;
    pub const mtpAdaptiveHeadEligible = mtp_policy.mtpAdaptiveHeadEligible;
    pub const mtpAdaptiveHeadMayResume = mtp_policy.mtpAdaptiveHeadMayResume;
    pub const mtpAdaptiveKvEligible = mtp_policy.mtpAdaptiveKvEligible;
    pub const mtpAdaptiveMinKvFor = mtp_policy.mtpAdaptiveMinKvFor;
    pub const mtpAdaptiveModelEligible = mtp_policy.mtpAdaptiveModelEligible;
    pub const mtpAdaptiveModelOk = mtp_policy.mtpAdaptiveModelOk;
    pub const mtpAdaptiveRegimeMoved = mtp_policy.mtpAdaptiveRegimeMoved;
    pub const mtpAdaptiveSerialEnabled = mtp_policy.mtpAdaptiveSerialEnabled;
    pub const mtpAdaptiveSerialEnabledFromEnv = mtp_policy.mtpAdaptiveSerialEnabledFromEnv;
    pub const mtpAdaptiveSerialStep = mtp_policy.mtpAdaptiveSerialStep;
    pub const mtpAdaptiveSerialTickFor = mtp_policy.mtpAdaptiveSerialTickFor;
    pub const mtpAdaptiveSyncWindow = mtp_policy.mtpAdaptiveSyncWindow;
    pub const mtpAdaptiveVoteFor = mtp_policy.mtpAdaptiveVoteFor;
    pub const mtpChainLogConf = mtp_policy.mtpChainLogConf;
    pub const mtpCostTableEnabled = mtp_policy.mtpCostTableEnabled;
    pub const mtpCostTableEnabledFromEnv = mtp_policy.mtpCostTableEnabledFromEnv;
    pub const mtpDepthCapForProfile = mtp_policy.mtpDepthCapForProfile;
    pub const mtpDepthCapForProfileChip = mtp_policy.mtpDepthCapForProfileChip;
    pub const mtpDepthCapFree = mtp_policy.mtpDepthCapFree;
    pub const mtpDepthCapResolved = mtp_policy.mtpDepthCapResolved;
    pub const mtpDepthDecision = mtp_policy.mtpDepthDecision;
    pub const mtpEmaMs = mtp_policy.mtpEmaMs;
    pub const mtpEvCostsForProfile = mtp_policy.mtpEvCostsForProfile;
    pub const mtpEvExpectedTokens = mtp_policy.mtpEvExpectedTokens;
    pub const mtpEvMarginalCost = mtp_policy.mtpEvMarginalCost;
    pub const mtpEvMarginalCostAt = mtp_policy.mtpEvMarginalCostAt;
    pub const mtpEvObserve = mtp_policy.mtpEvObserve;
    pub const mtpEvPlanFor = mtp_policy.mtpEvPlanFor;
    pub const mtpEvPlanForAt = mtp_policy.mtpEvPlanForAt;
    pub const mtpEvPlanSrc = mtp_policy.mtpEvPlanSrc;
    pub const mtpEvRoundCost = mtp_policy.mtpEvRoundCost;
    pub const mtpEvRoundCostAt = mtp_policy.mtpEvRoundCostAt;
    pub const mtpExtDryAllows = mtp_policy.mtpExtDryAllows;
    pub const mtpExtDryThresholdFor = mtp_policy.mtpExtDryThresholdFor;
    pub const mtpFloorDisableObserve = mtp_policy.mtpFloorDisableObserve;
    pub const mtpForcedWidth = mtp_policy.mtpForcedWidth;
    pub const mtpHeadDriftNow = mtp_policy.mtpHeadDriftNow;
    pub const mtpHeadPositionDrift = mtp_policy.mtpHeadPositionDrift;
    pub const mtpMaybeReleaseModuleHead = mtp_policy.mtpMaybeReleaseModuleHead;
    pub const mtpModuleHeadReleased = mtp_policy.mtpModuleHeadReleased;
    pub const mtpNextDepth = mtp_policy.mtpNextDepth;
    pub const mtpPinnedDepth = mtp_policy.mtpPinnedDepth;
    pub const mtpPlanInBounds = mtp_policy.mtpPlanInBounds;
    pub const mtpPlannerMode = mtp_policy.mtpPlannerMode;
    pub const mtpPlannerObserve = mtp_policy.mtpPlannerObserve;
    pub const mtpRegimeExplorePeriod = mtp_policy.mtpRegimeExplorePeriod;
    pub const mtpRegimeForce = mtp_policy.mtpRegimeForce;
    pub const mtpRegimeForceAt = mtp_policy.mtpRegimeForceAt;
    pub const mtpRegimeMeasured = mtp_policy.mtpRegimeMeasured;
    pub const mtpRegimeObserve = mtp_policy.mtpRegimeObserve;
    pub const mtpRegimeTwoChunkWorse = mtp_policy.mtpRegimeTwoChunkWorse;
    pub const mtpRegimeVerdict = mtp_policy.mtpRegimeVerdict;
    pub const mtpRegimeWallMs = mtp_policy.mtpRegimeWallMs;
    pub const mtpReleasePending = mtp_policy.mtpReleasePending;
    pub const mtpRoundEndObserve = mtp_policy.mtpRoundEndObserve;
    pub const mtpRoundPlan = mtp_policy.mtpRoundPlan;
    pub const mtpRoundPlanInner = mtp_policy.mtpRoundPlanInner;
    pub const mtpRoundPlanTraced = mtp_policy.mtpRoundPlanTraced;
    pub const mtpSerialProbeArm = mtp_policy.mtpSerialProbeArm;
    pub const mtpSerialProbeUseful = mtp_policy.mtpSerialProbeUseful;
    pub const mtpTraceEnabled = mtp_policy.mtpTraceEnabled;
    pub const mtpTraceOn = mtp_policy.mtpTraceOn;
    pub const mtpTraceRoundEnd = mtp_policy.mtpTraceRoundEnd;
    pub const mtpTraceSub = mtp_policy.mtpTraceSub;
    pub const mtpVerifyDraftsMax = mtp_policy.mtpVerifyDraftsMax;
    pub const mtpWidthTrialForce = mtp_policy.mtpWidthTrialForce;
    pub const mtpWidthTrialPeriod = mtp_policy.mtpWidthTrialPeriod;
    pub const mtpWidthTrialPlan = mtp_policy.mtpWidthTrialPlan;
    pub const mtpWidthTrialTarget = mtp_policy.mtpWidthTrialTarget;
    pub const observeSerialTick = mtp_policy.observeSerialTick;
    pub const persistRoundCost = mtp_policy.persistRoundCost;
    pub const regimeFold = mtp_policy.regimeFold;
    pub const resolveMtpDepthCapForProfile = mtp_policy.resolveMtpDepthCapForProfile;
    pub const serialCellWanted = mtp_policy.serialCellWanted;
    pub const specObserveRound = mtp_policy.specObserveRound;
    pub const stickyOnSerialSwitch = mtp_policy.stickyOnSerialSwitch;
    pub const stickyReleaseNow = mtp_policy.stickyReleaseNow;
    pub const updateMtpDepth = mtp_policy.updateMtpDepth;
    pub const updateMtpEvRound = mtp_policy.updateMtpEvRound;


    // ── MTP EV (expected-value) adaptive controller ──
    // Fixed-depth drafting is the warm-decode ceiling: at ~77% per-draft the
    // marginal chain decays with index, so one global depth wastes verify
    // width on hard stretches and leaves easy stretches (code boilerplate
    // where 8/8 accept) under-drafted. The EV controller tracks CONDITIONAL
    // per-index acceptance EMAs a[i] = P(draft i accepted | i-1 accepted) and
    // plans each round as two chunks: a base chunk `m_lo` (the static EV
    // optimum), then — when the head's own confidence on chunk A clears a
    // cost-derived threshold tau — an extension to `m_hi`. Only rounds that
    // CONSIDER extension pay the one bounded chunk-A sync; when the plan
    // collapses to m_lo == m_hi the round is byte-identical in shape to the
    // fixed-depth path (no confidence graph, no sync).
    // Disable via SUSHI_MTP_ADAPTIVE=0 (reverts to the windowed
    // fixed-depth controller above).

    /// Per-phase wall-time accumulator behind SUSHI_DFLASH_TRACE=1.
    /// `nextDflash` stamps phases with a Stopwatch and emits one summary line
    /// every LOG_EVERY rounds. Zero cost when the env is absent.
    pub const DflashTrace = struct {
        pub const LOG_EVERY: u32 = 16;
        pub const Phase = enum(u4) { assist, head, verify, accept, append, gap };
        pub const N_PHASES = @typeInfo(Phase).@"enum".field_names.len;

        rounds: u32 = 0,
        ns: [N_PHASES]u64 = @splat(0),
        accepted: u64 = 0,

        pub fn add(self: *DflashTrace, phase: Phase, dur_ns: u64) void {
            self.ns[@backingInt(phase)] += dur_ns;
        }

        /// Close one round; true when a summary line is due (caller logs,
        /// then calls reset()).
        pub fn endRound(self: *DflashTrace, accepted_n: u32) bool {
            self.rounds += 1;
            self.accepted += accepted_n;
            return self.rounds >= LOG_EVERY;
        }

        pub fn avgMs(self: *const DflashTrace, phase: Phase) f64 {
            if (self.rounds == 0) return 0.0;
            return @as(f64, @floatFromInt(self.ns[@backingInt(phase)])) /
                (@as(f64, @floatFromInt(self.rounds)) * 1e6);
        }

        pub fn totalAvgMs(self: *const DflashTrace) f64 {
            if (self.rounds == 0) return 0.0;
            var total: u64 = 0;
            for (self.ns) |v| total += v;
            return @as(f64, @floatFromInt(total)) / (@as(f64, @floatFromInt(self.rounds)) * 1e6);
        }

        pub fn reset(self: *DflashTrace) void {
            self.* = .{};
        }
    };

    var dflash_trace_cache: ?bool = null;
    fn dflashTraceEnabled() bool {
        if (dflash_trace_cache) |v| return v;
        const on = readEnvBool("SUSHI_DFLASH_TRACE");
        dflash_trace_cache = on;
        return on;
    }

    /// Close one traced dflash round; emits + resets at the cadence.
    fn dflashTraceRoundEnd(self: *Generator, accepted: u32) void {
        if (!self.dflash_trace.endRound(accepted)) return;
        const t = &self.dflash_trace;
        log.info(
            "  [dflash-trace] rounds={d} avg_ms assist={d:.2} head={d:.2} verify={d:.2} accept={d:.2} append={d:.2} gap={d:.2} total={d:.2} | acc_avg={d:.2}\n",
            .{
                t.rounds,
                t.avgMs(.assist),
                t.avgMs(.head),
                t.avgMs(.verify),
                t.avgMs(.accept),
                t.avgMs(.append),
                t.avgMs(.gap),
                t.totalAvgMs(),
                @as(f64, @floatFromInt(t.accepted)) / @as(f64, @floatFromInt(t.rounds)),
            },
        );
        t.reset();
    }

    // Adaptive serial: the EV controller picks the best depth but never compares a round with
    // the serial step it replaces, and the acceptance floor is a model fitted where a verify
    // row costs ~0.1 of a forward. On qwen4_exp a verify row is bytes (62.7k prose: 47-58 tok/s
    // speculative vs 55 serial). So the two measured prices are compared per KV bucket:
    //     planned ms/token = roundMs(m_lo) / E[tokens per round at m_lo]
    //     serial  ms/token = the bucket's measured plain-decode token
    // Any missing input is `.undecided`. The only cost a request may pay is a bounded serial
    // probe for a bucket that has never seen a serial token.

    /// Width-trial schedule: same shape as the regime's (explicit
    /// trial_end/next_trial, a 2-round block — transition then measurement),
    /// idempotent per round because `mtpRoundPlan` has two call sites.

    /// Returns the next token ID, or null when generation is finished.
    ///
    /// Pipeline architecture (matches mlx-lm's generator pattern):
    ///
    ///   The KEY to effective pipelining is the ORDER of operations:
    ///   1. Build next step's lazy graph (depends on pending lazy token)
    ///   2. async_eval the next graph — GPU computes pending token as a DEPENDENCY,
    ///      then continues with the forward pass
    ///   3. eval(pending_token) — returns INSTANTLY since GPU already computed it
    ///   4. Return the token while GPU continues computing the next forward pass
    ///
    ///   This mirrors mlx-lm's: _step(y) → async_eval(next_y) → yield y.item()
    ///   where y.item() is instant because async_eval forced y's computation.
    pub fn next(self: *Generator, allocator: std.mem.Allocator) !?u32 {
        var tick_prof = TickProf.begin();
        defer tick_prof.finish();
        if (self.done) return null;
        if (self.sampling.constraint != null) return if (self.dspark_pending) self.dsparkEmitPending(allocator) else self.nextConstrained(allocator);

        // Transition shim: speculative-decode paths may exit with
        // `next_token_id` set but `pending_logits` unset (drafter's exit
        // invariant is "t1 NOT in cache" — its hand-off to `next()` would
        // otherwise crash on the slow path which assumes pending_logits is
        // always lazily seeded). When we observe that state, synchronously
        // forward `[next_token_id]` to seed `pending_logits` so the fast
        // path picks up cleanly. the speculative exit state already matches `next()`'s
        // invariant, so this only fires for drafter→next runtime-gate
        // fallbacks (and any future spec methods that share drafter's shape).
        if (!self.has_pending_logits and !self.has_pending_token and
            self.step < self.max_tokens and !self.samplesSync())
        {
            const tok_i32: i32 = @intCast(self.next_token_id);
            const tok_shape = [_]c_int{ 1, 1 };
            const tok_input = mlx.mlx_array_new_data(&tok_i32, &tok_shape, 2, .int32);
            defer _ = mlx.mlx_array_free(tok_input);
            const t_fwd = tick_prof.mark();
            self.pending_logits = try self.xfm.forwardWith(&self.ctx, tok_input);
            tick_prof.add(.fwd, t_fwd);
            self.has_pending_logits = true;
        }

        // ── Phase 1: Build and submit the NEXT step FIRST ──
        // This forces the GPU to compute the pending token as a dependency,
        // so when we eval it in Phase 2, it's already ready.
        if (self.has_pending_logits and !self.samplesSync() and self.step + 1 < self.max_tokens and !self.callForcePending()) {
            const step_logits = self.pending_logits;
            self.has_pending_logits = false;

            const t_sample = tick_prof.mark();
            const lazy_token = blk: {
                defer _ = mlx.mlx_array_free(step_logits);
                break :blk try self.sampleLazy(step_logits, if (self.has_pending_token)
                    .{ .pending = self.pending_token }
                else
                    .{ .decided = &.{self.next_token_id} });
            };
            tick_prof.add(.sample, t_sample);

            var adopted = false;
            defer if (!adopted) {
                _ = mlx.mlx_array_free(lazy_token);
            };
            const t_fwd = tick_prof.mark();
            if (lazyForward(self.xfm, &self.ctx, lazy_token, self.ctx.argmax_only)) |next_logits| {
                tick_prof.add(.fwd, t_fwd);
                defer if (!adopted) {
                    _ = mlx.mlx_array_free(next_logits);
                };
                const arr = [_]mlx.mlx_array{ lazy_token, next_logits };
                const vec = mlx.mlx_vector_array_new_data(&arr, 2);
                const t_eval = tick_prof.mark();
                _ = mlx.mlx_async_eval(vec);
                tick_prof.add(.eval, t_eval);
                _ = mlx.mlx_vector_array_free(vec);

                // NOW resolve the pending token — GPU already computed it as a
                // dependency of the graph we just submitted. Should be instant.
                const t_res = tick_prof.mark();
                try self.resolvePendingToken();
                tick_prof.add(.resolve, t_res);

                // Check stop conditions on the resolved token
                if (try self.checkStop()) return null;

                const token = self.next_token_id;
                self.advanceStep(1);
                try self.generated_ids.append(allocator, token);

                // Store new pending state
                self.pending_token = lazy_token;
                self.has_pending_token = true;
                self.pending_logits = next_logits;
                self.has_pending_logits = true;
                adopted = true;

                return token;
            } else |_| {
                tick_prof.add(.fwd, t_fwd);
                // lazyForward failed — fall through to slow path
                try mlx.check(mlx.mlx_array_eval(lazy_token));
                var val: i32 = 0;
                try mlx.check(mlx.mlx_array_item_int32(&val, lazy_token));
                _ = mlx.mlx_array_free(lazy_token);
                adopted = true;
                self.next_token_id = @intCast(val);
                self.has_pending_token = false;
            }
        }

        // ── Phase 2: Slow path (first token, last token, logprobs, or pipeline miss) ──
        const t_res = tick_prof.mark();
        try self.resolvePendingToken();
        tick_prof.add(.resolve, t_res);

        if (try self.checkStop()) return null;

        const token = self.next_token_id;
        self.advanceStep(1);
        try self.generated_ids.append(allocator, token);

        const step_logits = if (self.has_pending_logits) blk: {
            const logits = self.pending_logits;
            self.has_pending_logits = false;
            break :blk logits;
        } else blk: {
            const tok_i32: i32 = @intCast(token);
            const tok_shape = [_]c_int{ 1, 1 };
            const tok_input = mlx.mlx_array_new_data(&tok_i32, &tok_shape, 2, .int32);
            defer _ = mlx.mlx_array_free(tok_input);
            const t_fwd = tick_prof.mark();
            const out = try self.xfm.forwardWith(&self.ctx, tok_input);
            tick_prof.add(.fwd, t_fwd);
            break :blk out;
        };

        // Logprobs and penalties: fully synchronous
        if (self.samplesSync()) {
            defer _ = mlx.mlx_array_free(step_logits);
            const t_sample = tick_prof.mark();
            const result: sampler.SampleResult = switch (self.forcedNext()) {
                .sample => try sampler.sampleToken(allocator, step_logits, self.sampling, self.generated_ids.items, self.logprobs_n, self.markerShiftCommitted(), self.xfm.s),
                .force => |id| .{ .token_id = id, .logprob_result = try sampler.firstTokenLogprobs(allocator, step_logits, id, self.logprobs_n, self.xfm.s) },
                .choose => |ids| blk: {
                    const lazy = try self.sampleAmong(allocator, step_logits, ids);
                    defer _ = mlx.mlx_array_free(lazy);
                    try mlx.check(mlx.mlx_array_eval(lazy));
                    var v: i32 = 0;
                    try mlx.check(mlx.mlx_array_item_int32(&v, lazy));
                    const id: u32 = @intCast(v);
                    break :blk .{ .token_id = id, .logprob_result = try sampler.firstTokenLogprobs(allocator, step_logits, id, self.logprobs_n, self.xfm.s) };
                },
            };
            tick_prof.add(.sample, t_sample);
            self.sampling.draw +%= 1;
            self.next_token_id = result.token_id;
            // `result` belongs to the token we just SAMPLED, which the next
            // call returns; `token` is carrying the previous round's result.
            if (self.last_logprob) |*lp| allocator.free(lp.top_logprobs);
            self.last_logprob = self.pending_logprob;
            self.pending_logprob = result.logprob_result;
            if (self.step < self.max_tokens) self.startAsyncForward(result.token_id);
            return token;
        }

        // Last token or pipeline bootstrap
        const t_sample = tick_prof.mark();
        const lazy_token = blk: {
            defer _ = mlx.mlx_array_free(step_logits);
            break :blk switch (self.forcedNext()) {
                .sample => try self.sampleLazy(step_logits, .committed),
                .force => |id| tokenArray(id),
                .choose => |ids| try self.sampleAmong(allocator, step_logits, ids),
            };
        };
        tick_prof.add(.sample, t_sample);

        if (self.step < self.max_tokens) {
            const t_fwd = tick_prof.mark();
            const next_logits = lazyForward(self.xfm, &self.ctx, lazy_token, self.ctx.argmax_only) catch {
                tick_prof.add(.fwd, t_fwd);
                try mlx.check(mlx.mlx_array_eval(lazy_token));
                var val: i32 = 0;
                try mlx.check(mlx.mlx_array_item_int32(&val, lazy_token));
                _ = mlx.mlx_array_free(lazy_token);
                self.next_token_id = @intCast(val);
                return token;
            };
            tick_prof.add(.fwd, t_fwd);

            const arr = [_]mlx.mlx_array{ lazy_token, next_logits };
            const vec = mlx.mlx_vector_array_new_data(&arr, 2);
            const t_eval = tick_prof.mark();
            _ = mlx.mlx_async_eval(vec);
            tick_prof.add(.eval, t_eval);
            _ = mlx.mlx_vector_array_free(vec);

            self.pending_token = lazy_token;
            self.has_pending_token = true;
            self.pending_logits = next_logits;
            self.has_pending_logits = true;
        } else {
            try mlx.check(mlx.mlx_array_eval(lazy_token));
            var val: i32 = 0;
            try mlx.check(mlx.mlx_array_item_int32(&val, lazy_token));
            _ = mlx.mlx_array_free(lazy_token);
            self.next_token_id = @intCast(val);
        }

        return token;
    }

    /// Sample one token while the reasoning protocol has not reached the JSON
    /// body: drain a forced recovery sequence, work the unresolved opener
    /// choice under the union mask, or sample unconstrained reasoning while
    /// the close-delimiter matcher watches. The active JSON phase reuses the
    /// existing grammar mask builder.
    fn nextReasoningStep(self: *Generator, allocator: std.mem.Allocator, constraint: *Constraint, proto: *const rp_mod.Protocol) !?u32 {
        // Recovery drain: at most one canonical transition token per tick,
        // through the same forward/accounting path as a sampled token.
        if (constraint.pstate.recovering) {
            return try self.commitForcedConstraintToken(allocator, constraint);
        }
        switch (constraint.pstate.phase) {
            .choice, .header => return self.nextChoiceStep(allocator, constraint, proto),
            .reasoning => {},
            .json_body => unreachable,
        }

        const step_logits = self.pending_logits;
        self.has_pending_logits = false;
        var owns_step_logits = true;
        defer {
            if (owns_step_logits) _ = mlx.mlx_array_free(step_logits);
        }

        // Boundary-crossing candidates (tokens carrying the close delimiter
        // plus a payload suffix) are validated BEFORE they can be sampled: a
        // suffix the schema rejects is masked out here, never sampled and
        // degraded afterwards.
        const invalid_crossers = try rp_mod.applyReasoningMask(proto, &constraint.pstate, constraint.grammar, constraint.token_bytes, constraint.mask_buf);
        const lazy = blk: {
            if (invalid_crossers == 0) break :blk try self.sampleLazy(step_logits, .committed);
            var masked_logits = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(masked_logits);
            try sampler.applyGrammarMask(allocator, &masked_logits, step_logits, constraint.mask_buf, self.xfm.s);
            break :blk try self.sampleLazy(masked_logits, .committed);
        };
        defer _ = mlx.mlx_array_free(lazy);
        try mlx.check(mlx.mlx_array_eval(lazy));
        var val: i32 = 0;
        try mlx.check(mlx.mlx_array_item_int32(&val, lazy));
        const token: u32 = @intCast(val);
        self.next_token_id = token;

        for (self.eos_token_ids) |eos_id| {
            if (token == eos_id and (proto.closer_atomic == null or token != proto.closer_atomic.?)) {
                _ = mlx.mlx_array_free(step_logits);
                owns_step_logits = false;
                switch (try self.forceConstraintTransition(allocator)) {
                    .committed => |forced| {
                        log.info("[grammar] reasoning boundary forced after premature EOS\n", .{});
                        return forced;
                    },
                    .activated => unreachable, // the reasoning phase always needs its close delimiter
                    .refused => {
                        self.done = true;
                        self.finish_reason = "stop";
                        self.stop_token = token;
                        return null;
                    },
                }
            }
        }
        if (token == 0) {
            self.consecutive_pad += 1;
            if (self.consecutive_pad >= 3) {
                _ = mlx.mlx_array_free(step_logits);
                owns_step_logits = false;
                switch (try self.forceConstraintTransition(allocator)) {
                    .committed => |forced| {
                        log.info("[grammar] reasoning boundary forced after terminal padding\n", .{});
                        return forced;
                    },
                    .activated => unreachable,
                    .refused => {
                        self.done = true;
                        self.finish_reason = "stop";
                        self.stop_token = token;
                        return null;
                    },
                }
            }
        } else {
            self.consecutive_pad = 0;
        }

        const token_bytes = if (token < constraint.token_bytes.bytes.len) constraint.token_bytes.bytes[token] else null;
        const payload_suffix = rp_mod.observeProtocolToken(proto, &constraint.pstate, token, token_bytes);
        if (constraint.pstate.invalid) return error.InvalidProtocolTransition;
        if (payload_suffix) |suffix| {
            // The close delimiter completed inside this token; any bytes after
            // it are the first payload bytes. The pre-sample mask validated
            // every crossing candidate's suffix. A mismatch is an internal
            // error and must never disable the constraint.
            for (suffix) |b| {
                const ok = try constraint.grammar.acceptByte(b);
                if (!ok) {
                    return error.InvalidProtocolPayload;
                }
            }
            constraint.pstate.phase = .json_body;
            // A byte-spelled close shares this token: the payload begins at
            // byte_offset inside it. An atomic close has no bytes at all —
            // the payload begins with the NEXT token (never unwrap the
            // null token bytes).
            constraint.pending_span = if (token_bytes == null)
                .{ .token_index = @intCast(self.generated_ids.items.len + 1), .byte_offset = 0 }
            else
                .{ .token_index = @intCast(self.generated_ids.items.len), .byte_offset = rp_mod.payloadByteOffset(token_bytes, suffix) };
            self.loop_guard_start = self.generated_ids.items.len;
            log.info("[grammar] reasoning boundary reached; enforcing final-answer schema\n", .{});
        }

        self.advanceStep(1);
        try self.generated_ids.append(allocator, token);

        if (self.step < self.max_tokens) {
            const tok_i32: i32 = @intCast(token);
            const tok_shape = [_]c_int{ 1, 1 };
            const tok_input = mlx.mlx_array_new_data(&tok_i32, &tok_shape, 2, .int32);
            defer _ = mlx.mlx_array_free(tok_input);
            const next_logits = try self.xfm.forwardWith(&self.ctx, tok_input);
            const arr = [_]mlx.mlx_array{next_logits};
            const vec = mlx.mlx_vector_array_new_data(&arr, 1);
            _ = mlx.mlx_async_eval(vec);
            _ = mlx.mlx_vector_array_free(vec);
            self.pending_logits = next_logits;
            self.has_pending_logits = true;
        } else {
            self.done = true;
            self.finish_reason = "length";
        }
        return token;
    }

    /// One constrained sample at the unresolved opener choice. The mask is
    /// the union of schema-legal tokens (the direct answer is constrained
    /// from its first byte) and opener-prefix candidates; once opener bytes
    /// are in flight only opener continuations remain. The JSON grammar never
    /// consumes protocol bytes.
    fn nextChoiceStep(self: *Generator, allocator: std.mem.Allocator, constraint: *Constraint, proto: *const rp_mod.Protocol) !?u32 {
        const s = self.xfm.s;

        const allowed = if (constraint.pstate.phase == .header)
            try rp_mod.applyHeaderMask(proto, &constraint.pstate, constraint.grammar, constraint.token_bytes, constraint.mask_buf)
        else
            try rp_mod.applyChoiceMask(proto, &constraint.pstate, constraint.grammar, constraint.token_bytes, constraint.mask_buf);
        if (allowed == 0) {
            return error.NoValidProtocolToken;
        }

        const step_logits = self.pending_logits;
        self.has_pending_logits = false;
        defer _ = mlx.mlx_array_free(step_logits);

        var masked_logits = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(masked_logits);
        try sampler.applyGrammarMask(allocator, &masked_logits, step_logits, constraint.mask_buf, s);

        const lazy = try self.sampleLazy(masked_logits, .committed);
        defer _ = mlx.mlx_array_free(lazy);
        try mlx.check(mlx.mlx_array_eval(lazy));
        var val: i32 = 0;
        try mlx.check(mlx.mlx_array_item_int32(&val, lazy));
        const token: u32 = @intCast(val);
        self.next_token_id = token;

        const token_bytes = if (token < constraint.token_bytes.bytes.len) constraint.token_bytes.bytes[token] else null;

        // The choice mask never admits EOS (the grammar is incomplete) and
        // pad tokens carry no admissible bytes; stop the way the constrained
        // path does if one is sampled anyway.
        for (self.eos_token_ids) |eos_id| {
            if (token == eos_id and constraint.pstate.phase != .header) {
                self.done = true;
                self.finish_reason = "stop";
                self.stop_token = token;
                return null;
            }
        }
        if (token == 0) {
            self.consecutive_pad += 1;
            if (self.consecutive_pad >= 3) {
                self.done = true;
                self.finish_reason = "stop";
                self.stop_token = token;
                return null;
            }
        } else {
            self.consecutive_pad = 0;
        }

        const outcome: rp_mod.ChoiceOutcome = if (constraint.pstate.phase == .header) blk: {
            const suffix = rp_mod.observeProtocolToken(proto, &constraint.pstate, token, token_bytes);
            if (constraint.pstate.invalid) return error.InvalidProtocolTransition;
            if (suffix) |bytes| break :blk .{ .payload = bytes };
            break :blk if (constraint.pstate.phase == .reasoning) .opened else .progress;
        } else rp_mod.observeChoice(proto, &constraint.pstate, token, token_bytes);
        switch (outcome) {
            .progress => {},
            .opened => {
                self.loop_guard_start = self.generated_ids.items.len;
                log.info("[grammar] reasoning opener generated; reasoning unconstrained\n", .{});
            },
            .json, .payload => {
                const payload = if (outcome == .payload) outcome.payload else token_bytes;
                if (payload) |bytes| {
                    for (bytes) |b| {
                        const ok = try constraint.grammar.acceptByte(b);
                        if (!ok) {
                            return error.InvalidProtocolPayload;
                        }
                    }
                }
                constraint.pstate.phase = .json_body;
                constraint.pending_span = .{
                    .token_index = @intCast(self.generated_ids.items.len + @as(usize, if (token_bytes == null) 1 else 0)),
                    .byte_offset = if (outcome == .payload) rp_mod.payloadByteOffset(token_bytes, outcome.payload) else 0,
                };
                self.loop_guard_start = self.generated_ids.items.len;
            },
        }

        self.advanceStep(1);
        try self.generated_ids.append(allocator, token);

        if (self.step < self.max_tokens) {
            const tok_i32: i32 = @intCast(token);
            const tok_shape = [_]c_int{ 1, 1 };
            const tok_input = mlx.mlx_array_new_data(&tok_i32, &tok_shape, 2, .int32);
            defer _ = mlx.mlx_array_free(tok_input);
            const next_logits = try self.xfm.forwardWith(&self.ctx, tok_input);
            const arr = [_]mlx.mlx_array{next_logits};
            const vec = mlx.mlx_vector_array_new_data(&arr, 1);
            _ = mlx.mlx_async_eval(vec);
            _ = mlx.mlx_vector_array_free(vec);
            self.pending_logits = next_logits;
            self.has_pending_logits = true;
        } else {
            self.done = true;
            self.finish_reason = "length";
        }
        return token;
    }

    /// Synchronous, grammar-constrained sampling step. Used whenever
    /// `sampling.constraint` is non-null. Builds a token mask from the grammar's
    /// current state, applies it to the pending logits, samples, advances the
    /// grammar by the sampled token's bytes, and pre-launches the next forward
    /// pass to overlap with the next mask build.
    pub fn nextConstrained(self: *Generator, allocator: std.mem.Allocator) !?u32 {
        if (self.logprobs_n == 0 or !self.has_pending_logits) return self.nextConstrainedToken(allocator);
        // A constrained token belongs to these raw logits, before the grammar mask.
        var logits = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(logits);
        try mlx.check(mlx.mlx_array_set(&logits, self.pending_logits));
        const token = (try self.nextConstrainedToken(allocator)) orelse return null;
        const lp = try sampler.computeLogprobs(allocator, logits, token, self.logprobs_n, self.xfm.s);
        if (self.last_logprob) |old| allocator.free(old.top_logprobs);
        self.pending_logprob = lp;
        self.last_logprob = self.pending_logprob;
        self.pending_logprob = null;
        return token;
    }

    fn nextConstrainedToken(self: *Generator, allocator: std.mem.Allocator) !?u32 {
        if (!self.has_pending_logits) {
            self.done = true;
            return null;
        }
        if (self.stall.expired(self.timer.read(), self.generated_ids.items.len, self.timeout_ns)) {
            self.done = true;
            self.finish_reason = "length";
            return null;
        }
        if (self.step >= self.max_tokens) {
            self.done = true;
            self.finish_reason = "length";
            return null;
        }
        const constraint = self.sampling.constraint.?;
        if (constraint.proto) |proto| {
            if (constraint.pstate.phase != .json_body) return self.nextReasoningStep(allocator, constraint, proto);
        }
        const s = self.xfm.s;

        var allowed = (try token_mask.buildMask(constraint.grammar, constraint.token_bytes, constraint.mask_buf)).allowed;
        // The tokenizer's EOS is not necessarily the model's only stop id.
        if (constraint.grammar.isComplete()) {
            for (self.eos_token_ids) |eos_id| {
                if (eos_id < constraint.mask_buf.len and !constraint.mask_buf[eos_id]) {
                    constraint.mask_buf[eos_id] = true;
                    allowed += 1;
                }
            }
        }
        if (allowed == 0) {
            // 0 legal bytes is a grammar dead end; otherwise the vocabulary lacks a token for them.
            const legal_bytes = blk: {
                const m = constraint.grammar.allowedBytes() catch break :blk 0;
                var n: u32 = 0;
                for (m.bits) |w| n += @popCount(w);
                break :blk n;
            };
            log.warn("[grammar] no token satisfies the schema at this position ({d} legal bytes) — disabling further mask enforcement\n", .{legal_bytes});
            constraint.grammar.dead = true;
            @memset(constraint.mask_buf, true);
        }

        const step_logits = self.pending_logits;
        self.has_pending_logits = false;
        defer _ = mlx.mlx_array_free(step_logits);

        var masked_logits = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(masked_logits);
        try sampler.applyGrammarMask(allocator, &masked_logits, step_logits, constraint.mask_buf, s);

        // The next forward is built on the still-lazy sample so the CPU graph
        // build overlaps this step's GPU work; the token is realized after
        // dispatch and only then advances the grammar (the mask for the next
        // logits is built on the next call, off the realized state).
        const lazy = try self.sampleLazy(masked_logits, .committed);
        defer _ = mlx.mlx_array_free(lazy);
        var next_logits: ?mlx.mlx_array = null;
        if (self.step + 1 < self.max_tokens) {
            if (lazyForward(self.xfm, &self.ctx, lazy, false)) |nl| {
                const arr = [_]mlx.mlx_array{ lazy, nl };
                const vec = mlx.mlx_vector_array_new_data(&arr, 2);
                _ = mlx.mlx_async_eval(vec);
                _ = mlx.mlx_vector_array_free(vec);
                next_logits = nl;
            } else |_| {}
        }
        errdefer if (next_logits) |nl| {
            _ = mlx.mlx_array_free(nl);
        };
        try mlx.check(mlx.mlx_array_eval(lazy));
        var val: i32 = 0;
        try mlx.check(mlx.mlx_array_item_int32(&val, lazy));
        const token: u32 = @intCast(val);
        self.next_token_id = token;

        // Stop on EOS — do not advance grammar or include in output. A
        // dispatched forward on the stop token is dropped, as on the lazy path.
        for (self.eos_token_ids) |eos_id| {
            if (token == eos_id) {
                if (next_logits) |nl| _ = mlx.mlx_array_free(nl);
                self.done = true;
                self.finish_reason = "stop";
                self.stop_token = token;
                return null;
            }
        }
        if (token == 0) {
            self.consecutive_pad += 1;
            if (self.consecutive_pad >= 3) {
                if (next_logits) |nl| _ = mlx.mlx_array_free(nl);
                self.done = true;
                self.finish_reason = "stop";
                self.stop_token = token;
                return null;
            }
        } else {
            self.consecutive_pad = 0;
        }

        // Advance the grammar by the sampled token's byte sequence. The mask
        // guarantees every byte is accepted (or the token has no byte form, e.g. a
        // special tag) — so a rejection here means a bug we want to surface.
        if (token < constraint.token_bytes.bytes.len) {
            if (constraint.token_bytes.bytes[token]) |bytes| {
                for (bytes) |b| {
                    const ok = try constraint.grammar.acceptByte(b);
                    if (!ok) {
                        log.warn("[grammar] sampled token {d} produced byte 0x{x} that was rejected — disabling further mask enforcement\n", .{ token, b });
                        constraint.grammar.dead = true;
                        break;
                    }
                }
            }
        }

        self.advanceStep(1);
        try self.generated_ids.append(allocator, token);

        if (next_logits) |nl| {
            self.pending_logits = nl;
            self.has_pending_logits = true;
        } else if (self.step < self.max_tokens) {
            // Pipeline miss: forward the realized token.
            const tok_i32: i32 = @intCast(token);
            const tok_shape = [_]c_int{ 1, 1 };
            const tok_input = mlx.mlx_array_new_data(&tok_i32, &tok_shape, 2, .int32);
            defer _ = mlx.mlx_array_free(tok_input);
            const nl = try self.xfm.forwardWith(&self.ctx, tok_input);
            const arr = [_]mlx.mlx_array{nl};
            const vec = mlx.mlx_vector_array_new_data(&arr, 1);
            _ = mlx.mlx_async_eval(vec);
            _ = mlx.mlx_vector_array_free(vec);
            self.pending_logits = nl;
            self.has_pending_logits = true;
        } else {
            self.done = true;
            self.finish_reason = "length";
        }

        return token;
    }

    /// The length and stall stops, which no token decides.
    pub fn checkBudgetStop(self: *Generator) bool {
        if (self.step >= self.max_tokens) {
            self.done = true;
            self.finish_reason = "length";
            return true;
        }
        if (self.stall.expired(self.timer.read(), self.generated_ids.items.len, self.timeout_ns)) {
            self.done = true;
            self.finish_reason = "length";
            return true;
        }
        return false;
    }

    /// Scheduler preflight; the selected decode path owns token accounting.
    pub fn checkDispatchStop(self: *Generator) !bool {
        if (self.checkBudgetStop()) return true;
        if (isEosId(self.next_token_id, self.eos_token_ids)) {
            self.done = true;
            self.finish_reason = "stop";
            self.stop_token = self.next_token_id;
            return true;
        }
        return false;
    }

    /// Check all stop conditions. Returns true if generation should stop.
    pub fn checkStop(self: *Generator) !bool {
        if (self.checkBudgetStop()) return true;
        if (tokenStops(self.next_token_id, self.eos_token_ids, &self.consecutive_pad)) {
            self.done = true;
            self.finish_reason = "stop";
            self.stop_token = self.next_token_id;
            return true;
        }
        return false;
    }

    /// Legacy sync forward for logprobs path.
    fn startAsyncForward(self: *Generator, token_id: u32) void {
        const tok_i32: i32 = @intCast(token_id);
        const tok_shape = [_]c_int{ 1, 1 };
        const tok_input = mlx.mlx_array_new_data(&tok_i32, &tok_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(tok_input);

        const logits = self.xfm.forwardWith(&self.ctx, tok_input) catch return;
        const arr = [_]mlx.mlx_array{logits};
        const vec = mlx.mlx_vector_array_new_data(&arr, 1);
        _ = mlx.mlx_async_eval(vec);
        _ = mlx.mlx_vector_array_free(vec);

        self.pending_logits = logits;
        self.has_pending_logits = true;
    }
};

/// Build forward pass from a lazy sampled token array.
/// Reshapes [1] -> [1, 1] and calls transformer forward. All lazy (no eval).
fn lazyForward(xfm: *Transformer, ctx: *ForwardCtx, lazy_token: mlx.mlx_array, lazy_logits: bool) !mlx.mlx_array {
    const tok_shape = [_]c_int{ 1, 1 };
    var reshaped = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(reshaped);
    try mlx.check(mlx.mlx_reshape(&reshaped, lazy_token, &tok_shape, 2, xfm.s));
    // The graph is built while the previous step still runs on the GPU; an
    // arch with a host-side token lookup (qwen4 n-gram PLE) fills its leaf
    // afterwards, so the one sync on the token comes AFTER the build.
    ctx.ple_defer = true;
    ctx.lazy_logits = lazy_logits;
    defer {
        ctx.ple_defer = false;
        ctx.lazy_logits = false;
    }
    const logits = xfm.forwardWith(ctx, reshaped) catch |e| {
        xfm.discardDeferredPle(ctx);
        return e;
    };
    xfm.flushDeferredPle(ctx) catch |e| {
        xfm.discardDeferredPle(ctx);
        _ = mlx.mlx_array_free(logits);
        return e;
    };
    return logits;
}

/// The `[1+m, V]` rows a round reads from its verify block: a batched group
/// right-pads every row to its widest draft, so the block may be longer.
pub fn verifyRows2d(probs_all: mlx.mlx_array, m: u32, s: mlx.mlx_stream) !mlx.mlx_array {
    const shape = mlx.getShape(probs_all);
    const rows: c_int = @intCast(1 + m);
    if (shape.len != 3 or shape[0] != 1 or shape[1] < rows) return error.MtpVerifyBlockShape;
    var head = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(head);
    try mlx.check(mlx.mlx_slice(&head, probs_all, &.{ 0, 0, 0 }, 3, &.{ 1, rows, shape[2] }, 3, &.{ 1, 1, 1 }, 3, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, head, &.{ rows, shape[2] }, 2, s));
    return out;
}

/// A committed token as the `[1]` int32 array a sample would be.
pub fn tokenArray(id: u32) mlx.mlx_array {
    const v: i32 = @intCast(id);
    const shape = [_]c_int{1};
    return mlx.mlx_array_new_data(&v, &shape, 1, .int32);
}

/// Convenience: generate all tokens at once (non-streaming).
pub fn generate(
    io: std.Io,
    allocator: std.mem.Allocator,
    xfm: *Transformer,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
    max_tokens: u32,
    sampling: sampler.SamplingParams,
    eos_token_ids: []const u32,
    timeout_ns: u64,
    logprobs_n: u32,
) !GenerationResult {
    var timer = io_util.Stopwatch.init(io);
    // logprobs_n rides InitOptions so init's argmax-only gate sees it (a
    // post-init field write would let the split-prefill final-token forward
    // engage the pruned lm_head on a logprobs request).
    var gen = try Generator.initWithOptions(io, allocator, xfm, tok, prompt_ids, max_tokens, sampling, eos_token_ids, .{ .logprobs_n = logprobs_n });
    gen.timeout_ns = timeout_ns;
    gen.logprobs_n = logprobs_n;
    defer gen.deinit(allocator);

    const prefill_ns = timer.read();
    const prefill_tps: f64 = if (prefill_ns > 0)
        @as(f64, @floatFromInt(prompt_ids.len)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(prefill_ns))
    else
        0.0;
    log.debug("Prefill: {d}ms ({d} tokens, {d:.3} tok/s)\n", .{
        prefill_ns / std.time.ns_per_ms,
        prompt_ids.len,
        prefill_tps,
    });

    var output_ids = std.ArrayList(u32).empty;
    defer output_ids.deinit(allocator);

    var logprob_results = std.ArrayList(sampler.LogprobResult).empty;
    defer {
        if (logprobs_n == 0) {
            for (logprob_results.items) |*lp| allocator.free(lp.top_logprobs);
            logprob_results.deinit(allocator);
        }
    }

    timer.reset();
    while (try gen.next(allocator)) |token_id| {
        try output_ids.append(allocator, token_id);
        if (logprobs_n > 0) {
            if (gen.last_logprob) |lp| {
                try logprob_results.append(allocator, lp);
                gen.last_logprob = null; // Transfer ownership
            }
        }
    }

    const decode_ns = timer.read();
    const num_decoded = output_ids.items.len;
    const decode_tps: f64 = if (decode_ns > 0)
        @as(f64, @floatFromInt(num_decoded)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(decode_ns))
    else
        0.0;
    log.debug("Decode: {d}ms ({d} tokens, {d:.3} tok/s)\n", .{
        decode_ns / std.time.ns_per_ms,
        num_decoded,
        decode_tps,
    });

    const text = try tok.decode(allocator, output_ids.items);
    const token_ids = try output_ids.toOwnedSlice(allocator);

    return .{
        .text = text,
        .token_ids = token_ids,
        .prompt_tokens = gen.prompt_tokens,
        .completion_tokens = gen.completion_tokens,
        .finish_reason = gen.finish_reason,
        .prefill_tps = prefill_tps,
        .decode_tps = decode_tps,
        .logprobs = if (logprobs_n > 0) try logprob_results.toOwnedSlice(allocator) else null,
    };
}

pub fn isEosId(id: u32, eos: []const u32) bool {
    for (eos) |e| if (id == e) return true;
    return false;
}

/// Token-only stop conditions of `checkStop`; `consecutive_pad` is carried state reset by any real token.
pub fn tokenStops(next_token_id: u32, eos_token_ids: []const u32, consecutive_pad: *u32) bool {
    if (isEosId(next_token_id, eos_token_ids)) return true;
    if (next_token_id != 0) {
        consecutive_pad.* = 0;
        return false;
    }
    consecutive_pad.* += 1;
    return consecutive_pad.* >= 3;
}

// ── Tests ──

const testing = std.testing;

test "GenerationResult fields" {
    // Just verifying the struct shape compiles correctly with all fields
    const result = GenerationResult{
        .text = @constCast("hello"),
        .token_ids = @constCast(&[_]u32{ 1, 2, 3 }),
        .prompt_tokens = 10,
        .completion_tokens = 3,
        .finish_reason = "stop",
        .prefill_tps = 100.0,
        .decode_tps = 35.0,
        .logprobs = null,
    };
    try testing.expectEqual(@as(u32, 10), result.prompt_tokens);
    try testing.expectEqual(@as(u32, 3), result.completion_tokens);
    try testing.expectEqualStrings("stop", result.finish_reason);
    try testing.expect(result.logprobs == null);
}

test "tokensPerSec basic and zero-time" {
    // 100 tokens in 1 second = 100 tok/s.
    try testing.expectApproxEqAbs(@as(f64, 100.0), tokensPerSec(100, std.time.ns_per_s), 1e-6);
    // 50 tokens in 0.5s = 100 tok/s.
    try testing.expectApproxEqAbs(@as(f64, 100.0), tokensPerSec(50, std.time.ns_per_s / 2), 1e-6);
    // Zero elapsed → 0, never inf/NaN.
    try testing.expectEqual(@as(f64, 0.0), tokensPerSec(100, 0));
}

const PrefillMem = struct { active: usize, cache: usize, peak: usize };

fn mlxPrefillMem() PrefillMem {
    var active: usize = 0;
    var cache: usize = 0;
    var peak: usize = 0;
    _ = mlx.mlx_get_active_memory(&active);
    _ = mlx.mlx_get_cache_memory(&cache);
    _ = mlx.mlx_get_peak_memory(&peak);
    return .{ .active = active, .cache = cache, .peak = peak };
}

fn formatPrefillMemLine(buf: []u8, pos: usize, end: usize, eval_ms: u64, mem: PrefillMem) []const u8 {
    return std.fmt.bufPrint(buf, "  [prefill-trace] mem pos={d} end={d} eval_ms={d} active_bytes={d} cache_bytes={d} peak_bytes={d}\n", .{
        pos,
        end,
        eval_ms,
        mem.active,
        mem.cache,
        mem.peak,
    }) catch "";
}

test "prefill-trace mem line names active_bytes and cache_bytes" {
    var buf: [256]u8 = undefined;
    const line = formatPrefillMemLine(&buf, 8192, 16384, 12, .{ .active = 11, .cache = 22, .peak = 33 });
    try std.testing.expect(std.mem.indexOf(u8, line, "active_bytes=11") != null);
    try std.testing.expect(std.mem.indexOf(u8, line, "cache_bytes=22") != null);
    try std.testing.expect(std.mem.indexOf(u8, line, "peak_bytes=33") != null);
    try std.testing.expect(std.mem.indexOf(u8, line, "eval_ms=12") != null);
}

test "prefillTokensPerSec divides by uncached tokens" {
    // Cold prefill: 754 tokens, none cached, 2s → 377 tok/s.
    try testing.expectApproxEqAbs(
        @as(f64, 377.0),
        prefillTokensPerSec(754, 0, 2 * std.time.ns_per_s),
        1e-6,
    );
    // Warm prefill: 754-token prompt, 700 cached, only 54 ran. A fast 54-token
    // suffix in 0.1s is 540 tok/s — NOT 7540 (the inflated full-prompt rate).
    try testing.expectApproxEqAbs(
        @as(f64, 540.0),
        prefillTokensPerSec(754, 700, std.time.ns_per_s / 10),
        1e-6,
    );
    // Full cache hit: 0 uncached → 0 tok/s (no compute happened).
    try testing.expectEqual(@as(f64, 0.0), prefillTokensPerSec(754, 754, std.time.ns_per_s));
    // Defensive: cached > prompt (shouldn't happen) clamps to 0, no underflow.
    try testing.expectEqual(@as(f64, 0.0), prefillTokensPerSec(10, 20, std.time.ns_per_s));
}

test "InitOptions.lookup_prompt overrides prompt_ids_owned source" {
    // When the server's cache-reuse path forwards only a trailing-tail
    // prompt slice but supplies the full original prompt via
    // `InitOptions.lookup_prompt`, the lookup buffer must be cloned from
    // the full slice — not the truncated tail.
    const tail = [_]u32{99};
    const full = [_]u32{ 10, 20, 30, 99 };
    const src = Generator.pickLookupPromptSource(&tail, &full);
    try testing.expectEqual(@as(usize, 4), src.len);
    try testing.expectEqualSlices(u32, &full, src);
}

test "InitOptions.lookup_prompt = null preserves existing behavior" {
    // Back-compat path: when callers don't set `lookup_prompt`, the source
    // is the unmodified `prompt_ids` slice — same buffer the Generator
    // received pre-fix.
    const prompt = [_]u32{ 1, 2, 3, 4, 5 };
    const src = Generator.pickLookupPromptSource(&prompt, null);
    try testing.expectEqual(prompt.len, src.len);
    try testing.expectEqualSlices(u32, &prompt, src);
    try testing.expectEqual(@as([*]const u32, prompt[0..].ptr), src.ptr);
}

test "CallForce: a required call names one of the declared functions, the model picking among them" {
    const CLOSE: u32 = 901;
    const t = std.testing;
    const head = [_]u32{ 10, 11 }; // the call up to the name
    // Each declared name as ids, its closing delimiter included.
    const get_time = [_]u32{ 20, 21, 30 };
    const get_weather = [_]u32{ 20, 22, 30 };
    const list_files = [_]u32{ 23, 30 };
    const names = [_][]const u32{ &get_time, &get_weather, &list_files };
    var choices: [names.len]u32 = undefined;
    var cf = CallForce{ .forced = &head, .call_at = 0, .closer_id = CLOSE, .phase = .answer, .last = CLOSE, .names = &names, .choice_buf = &choices };

    var gen = std.ArrayList(u32).empty;
    defer gen.deinit(t.allocator);
    for (head) |want| {
        try t.expectEqual(want, cf.decide(gen.items).force);
        try gen.append(t.allocator, want);
    }
    try t.expectEqualSlices(u32, &.{ 20, 23 }, cf.decide(gen.items).choose);
    try gen.append(t.allocator, 20); // the model's pick
    try t.expectEqualSlices(u32, &.{ 21, 22 }, cf.decide(gen.items).choose);
    try gen.append(t.allocator, 22);
    try t.expect(cf.pending());
    try t.expectEqual(@as(u32, 30), cf.decide(gen.items).force);
    try gen.append(t.allocator, 30);
    try t.expect(cf.decide(gen.items) == .sample);
    try t.expect(!cf.pending());
}

test "MtpTrace: per-phase accumulation, round averaging, log cadence, reset" {
    var t = mtp_policy.MtpTrace{};
    // 2 rounds: draft 2ms+4ms, eval 10ms+30ms.
    t.add(.draft, 2_000_000);
    t.add(.eval, 10_000_000);
    try testing.expect(!t.endRound(3, 2, false));
    t.add(.draft, 4_000_000);
    t.add(.eval, 30_000_000);
    try testing.expect(!t.endRound(7, 6, true));
    try testing.expectApproxEqAbs(@as(f64, 3.0), t.avgMs(.draft), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 20.0), t.avgMs(.eval), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 0.0), t.avgMs(.sync), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 23.0), t.totalAvgMs(), 1e-9);
    try testing.expectEqual(@as(u64, 10), t.drafted);
    try testing.expectEqual(@as(u64, 8), t.accepted);
    try testing.expectEqual(@as(u32, 1), t.extended);
    // Log line falls due exactly at LOG_EVERY rounds.
    var i: u32 = 2;
    while (i < mtp_policy.MtpTrace.LOG_EVERY - 1) : (i += 1) {
        try testing.expect(!t.endRound(1, 1, false));
    }
    try testing.expect(t.endRound(1, 1, false));
    t.reset();
    try testing.expectEqual(@as(u32, 0), t.rounds);
    try testing.expectApproxEqAbs(@as(f64, 0.0), t.avgMs(.draft), 1e-9);
}

test "the [mtp-trace] phase summary stays contiguous for its log consumers" {
    const line = try std.fmt.allocPrint(testing.allocator, mtp_policy.MtpTrace.LINE_FMT, .{
        @as(u32, 32), 0.45, 0.0, 0.0,         4.09, 56.86, 0.0,  0.0,   0.03, 0.0,  0.07, 61.57,
        2.0,          1.4,  0.0, "0.91/0.80", 4.09, 7.13,  6.40, 52.28, 0.94, 0.44,
    });
    defer testing.allocator.free(line);
    try testing.expect(std.mem.indexOf(u8, line, "total=61.57 | m_avg=") != null);
    const acc_at = std.mem.indexOf(u8, line, "acc_idx=").?;
    const sub_at = std.mem.indexOf(u8, line, "| sub_ms ").?;
    try testing.expect(acc_at < sub_at);
}

test "MtpTrace sub-laps attribute inside a phase without changing the round total" {
    var t = mtp_policy.MtpTrace{};
    t.add(.verify, 20_000_000);
    t.addSub(.build, 15_000_000);
    t.addSub(.ple, 3_000_000);
    t.addSub(.ple_sync, 2_000_000);
    t.add(.corr, 6_000_000);
    t.addSub(.dispatch, 2_000_000);
    t.add(.eval, 60_000_000);
    t.addSub(.wait, 58_000_000);
    t.add(.gap, 12_000_000);
    t.addSub(.chain, 2_000_000);
    try testing.expect(!t.endRound(2, 2, false));
    try testing.expectApproxEqAbs(@as(f64, 98.0), t.totalAvgMs(), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 20.0), t.avgMs(.verify), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 20.0), t.subAvgMs(.build) + t.subAvgMs(.ple) + t.subAvgMs(.ple_sync), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 58.0), t.subAvgMs(.wait), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 2.0), t.subAvgMs(.chain), 1e-9);
    t.reset();
    try testing.expectApproxEqAbs(@as(f64, 0.0), t.subAvgMs(.build), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 0.0), t.totalAvgMs(), 1e-9);
}

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

test "Tokenizer.definedVocabSize: highest defined id + 1, 0 when nothing is defined" {
    const allocator = testing.allocator;
    var tok = Tokenizer.initEmptyForTests(allocator);
    defer tok.deinit();
    try testing.expectEqual(@as(usize, 0), tok.definedVocabSize());
    // Added specials sit ABOVE the base vocab and are not contiguous with it,
    // so the answer is the maximum key, never the entry count.
    try tok.id_to_token.put(0, "a");
    try tok.id_to_token.put(1, "b");
    try tok.id_to_token.put(11, "<|special|>");
    try testing.expectEqual(@as(usize, 12), tok.definedVocabSize());
}

test "MtpTrace per-index acceptance: index i counts only rounds that drafted it" {
    var t: mtp_policy.MtpTrace = .{};
    _ = t.endRound(3, 3, false); // all three accepted
    _ = t.endRound(3, 1, false); // a0 accepted, a1 rejected, a2 never judged
    _ = t.endRound(1, 0, false); // depth-1 round: only index 0 drafted
    var buf: [64]u8 = undefined;
    try std.testing.expectEqualStrings("0.67/0.50/0.50", t.accIdxStr(&buf));
    t.reset();
    try std.testing.expectEqualStrings("", t.accIdxStr(&buf));
}

test "MtpPriceWindow: reads nothing until FULL, skips width trials, prices per TOKEN" {
    const N = mtp_policy.MTP_PRICE_WINDOW;
    var w = mtp_policy.MtpPriceWindow{};
    try testing.expect(w.msPerTok() == null);

    var i: u32 = 0;
    while (i < N - 1) : (i += 1) {
        w.observe(20.0, 2.0, false);
        try testing.expect(w.msPerTok() == null); // still not full
    }
    w.observe(20.0, 2.0, false);
    try testing.expectApproxEqAbs(@as(f32, 10.0), w.msPerTok().?, 1e-4);

    var t = mtp_policy.MtpPriceWindow{};
    i = 0;
    while (i < N * 2) : (i += 1) t.observe(999.0, 1.0, true);
    try testing.expect(t.msPerTok() == null);
    try testing.expectEqual(@as(u32, 0), t.filled);

    var d = mtp_policy.MtpPriceWindow{};
    d.observe(std.math.nan(f32), 2.0, false);
    d.observe(20.0, 0.0, false);
    d.observe(-1.0, 2.0, false);
    try testing.expectEqual(@as(u32, 0), d.filled);

    // Summed over the window: 8 rounds at 1 tok/30 ms + 8 at 4 tok/60 ms = 18.0 ms/tok, not 22.5.
    var m = mtp_policy.MtpPriceWindow{};
    i = 0;
    while (i < N / 2) : (i += 1) m.observe(30.0, 1.0, false);
    while (i < N) : (i += 1) m.observe(60.0, 4.0, false);
    try testing.expectApproxEqAbs(@as(f32, 18.0), m.msPerTok().?, 1e-4);

    i = 0;
    while (i < N) : (i += 1) m.observe(10.0, 2.0, false);
    try testing.expectApproxEqAbs(@as(f32, 5.0), m.msPerTok().?, 1e-4);
    m.reset();
    try testing.expect(m.msPerTok() == null);
}

test "MtpAdaptive: the switch needs CONFIRM consecutive rounds, counted once per round" {
    const need = mtp_policy.MTP_ADAPTIVE_CONFIRM;
    var a = mtp_policy.MtpAdaptive{};
    var r: u32 = 0;
    while (r < need - 1) : (r += 1) {
        try testing.expectEqual(mtp_policy.MtpAdaptiveAction.none, a.round(r, 3, .serial, need));
    }
    try testing.expectEqual(mtp_policy.MtpAdaptiveAction.to_serial, a.round(need - 1, 3, .serial, need));
    try testing.expectEqual(mtp_policy.MtpAdaptiveArm.serial, a.arm);
    try testing.expectEqual(@as(u32, 1), a.switches);

    // `mtpRoundPlan` has two call sites per round: the second call at the same index must not advance the streak.
    var b = mtp_policy.MtpAdaptive{};
    var i: u32 = 0;
    while (i < need - 1) : (i += 1) {
        try testing.expectEqual(mtp_policy.MtpAdaptiveAction.none, b.round(i, 3, .serial, need));
        try testing.expectEqual(mtp_policy.MtpAdaptiveAction.none, b.round(i, 3, .serial, need));
    }
    try testing.expectEqual(mtp_policy.MtpAdaptiveArm.undecided, b.arm);
    try testing.expectEqual(mtp_policy.MtpAdaptiveAction.to_serial, b.round(need - 1, 3, .serial, need));

    var c = mtp_policy.MtpAdaptive{};
    _ = c.round(0, 3, .serial, need);
    _ = c.round(1, 3, .serial, need);
    try testing.expectEqual(mtp_policy.MtpAdaptiveAction.none, c.round(2, 3, .mtp, need));
    try testing.expectEqual(mtp_policy.MtpAdaptiveArm.mtp, c.arm);
    try testing.expectEqual(mtp_policy.MtpAdaptiveAction.none, c.round(3, 3, .serial, need));
    try testing.expectEqual(mtp_policy.MtpAdaptiveAction.none, c.round(4, 3, .serial, need));
    try testing.expectEqual(mtp_policy.MtpAdaptiveAction.to_serial, c.round(5, 3, .serial, need));

    var d = mtp_policy.MtpAdaptive{};
    _ = d.round(0, 3, .serial, need);
    _ = d.round(1, 3, .serial, need);
    try testing.expectEqual(mtp_policy.MtpAdaptiveAction.none, d.round(2, 3, .undecided, need));
    try testing.expectEqual(mtp_policy.MtpAdaptiveAction.none, d.round(3, 3, .serial, need));
    try testing.expectEqual(mtp_policy.MtpAdaptiveArm.undecided, d.arm);

    var e = mtp_policy.MtpAdaptive{};
    _ = e.round(0, 3, .serial, need);
    _ = e.round(1, 3, .serial, need);
    try testing.expectEqual(mtp_policy.MtpAdaptiveAction.none, e.round(2, 4, .serial, need));
    try testing.expectEqual(@as(u32, 1), e.confirm);

    var f = mtp_policy.MtpAdaptive{};
    try testing.expectEqual(mtp_policy.MtpAdaptiveAction.to_serial, f.round(0, 3, .serial, 0));
}

test "MtpAdaptive: only a bucket crossing re-opens the serial decision" {
    var off = mtp_policy.MtpAdaptive{};
    _ = off.round(0, 3, .serial, 1);
    var i: u32 = 0;
    while (i < 1024) : (i += 1) {
        try testing.expectEqual(mtp_policy.MtpAdaptiveAction.none, off.serialTick(3));
    }
    try testing.expectEqual(mtp_policy.MtpAdaptiveArm.serial, off.arm);
    try testing.expectEqual(mtp_policy.MtpAdaptiveAction.to_mtp, off.serialTick(4));
}

test "MTP head persistence kill switch: only a literal 0 turns it off" {
    try testing.expect(mtpHeadPersistFromEnv(null));
    try testing.expect(mtpHeadPersistFromEnv(""));
    try testing.expect(mtpHeadPersistFromEnv("1"));
    try testing.expect(!mtpHeadPersistFromEnv("0"));
}

test "a warm prefill grows its restored layers to the reservation before the first chunk writes" {
    // seq <= C < R: left to the decode, the first step past C grows every layer inside one forward.
    try mlx.requireGpu();
    const t = std.testing;
    const s = mlx.gpuStream();
    const KVC = transformer_mod.KVCache;
    const seq: u64 = 40_000;
    const matched: c_int = 36_000;
    const capacity: usize = 40_960;
    const qwen4 = model_mod.ModelConfig{ .model_type = "qwen4_exp", .max_position_embeddings = 1_048_576 };
    const mimo = model_mod.ModelConfig{ .model_type = "mimo_v2", .has_sliding_window = true, .sliding_window = 128, .head_dim = 192, .max_position_embeddings = 1_048_576 };
    var rows = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rows);
    try mlx.check(mlx.mlx_zeros(&rows, &[_]c_int{ 1, 2, matched, 64 }, 4, .bfloat16, s));
    var tail = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tail);
    try mlx.check(mlx.mlx_zeros(&tail, &[_]c_int{ 1, 2, @intCast(seq - @as(u64, @intCast(matched))), 64 }, 4, .bfloat16, s));
    var one = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(one);
    try mlx.check(mlx.mlx_zeros(&one, &[_]c_int{ 1, 2, 1, 64 }, 4, .bfloat16, s));
    for ([_]*const model_mod.ModelConfig{ &qwen4, &mimo }) |cfg| {
        const ring: u32 = if (cfg.swaRingTokens() > 0) cfg.sliding_window else 0;
        for ([_]transformer_mod.KVQuantConfig{ .dense, .affine(8), .affine(6) }) |kvq| {
            for ([_]bool{ true, false }) |donate| {
                // The previous turn's cache: it reserved C and holds the matched rows. Layer 1 rings on MiMo.
                var donor = try KVC.initWithConfig(t.allocator, 2, kvq);
                defer donor.deinit();
                donor.setSwaRing(ring);
                donor.reserve(capacity);
                for (0..2) |li| {
                    var v = try donor.update(@intCast(li), rows, rows, s, if (li == 1) ring else 0);
                    v.deinit();
                }
                var snap = try donor.snapshotRetained(s);
                defer snap.deinit();
                var warm = try KVC.initWithConfig(t.allocator, 2, kvq);
                defer warm.deinit();
                warm.setSwaRing(ring);
                try warm.restore(&snap);
                if (donate) {
                    snap.releaseHandles();
                    warm.adoptRestored();
                }
                const ring_cap = mlx.getShape(warm.entries[1].keys)[2];

                const reserved = try reserveRequestCapacity(&warm, cfg, seq, 32_000, 2048, s);
                try t.expect(seq <= capacity and capacity < reserved);
                try t.expectEqual(@as(c_int, capacity), mlx.getShape(donor.entries[0].keys)[2]);
                try t.expect(mlx.getShape(warm.entries[0].keys)[2] >= reserved);
                try t.expect(mlx.getShape(warm.entries[0].values)[2] >= reserved);
                try t.expectEqual(@as(usize, @intCast(matched)), warm.entries[0].offset);
                if (ring > 0) try t.expectEqual(ring_cap, mlx.getShape(warm.entries[1].keys)[2]);

                // The prefill's rows and the first decode step write in place: nothing grows.
                const grows = KVC.kv_cap_buf_grows;
                var pv = try warm.update(0, tail, tail, s, 0);
                pv.deinit();
                var dv = try warm.update(0, one, one, s, 0);
                dv.deinit();
                try t.expectEqual(grows, KVC.kv_cap_buf_grows);
            }
        }
    }
}

test "a block decoder's entry token stops the round before it drafts" {
    // Bar: an EOS or third pad as the round's entry token ends the request before any draft.
    const eos = [_]u32{ 151645, 151643 };
    var pad: u32 = 0;
    try testing.expect(tokenStops(151645, &eos, &pad));
    try testing.expect(tokenStops(151643, &eos, &pad));
    try testing.expect(!tokenStops(17, &eos, &pad));

    // A pad run stops on the third; any real token resets it.
    pad = 0;
    try testing.expect(!tokenStops(0, &eos, &pad));
    try testing.expect(!tokenStops(0, &eos, &pad));
    try testing.expect(tokenStops(0, &eos, &pad));
    pad = 0;
    try testing.expect(!tokenStops(0, &eos, &pad));
    try testing.expect(!tokenStops(9, &eos, &pad));
    try testing.expectEqual(@as(u32, 0), pad);
}

test "batched MTP state includes the last row activated by round begin" {
    const allocator = testing.allocator;
    var xfm: Transformer = undefined;
    xfm.qwen4_mtp_owner = null;
    var head: transformer_mod.Qwen4Mtp = undefined;
    head.cache = try KVCache.init(allocator, 1);
    head.entry = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true };
    head.qsa_marks = .{};
    head.seq_offset = 0;
    head.pos_base = 0;
    xfm.qwen4_mtp = head;
    defer xfm.qwen4_mtp.?.cache.deinit();
    var a: Transformer.Qwen4MtpState = .{ .cache = try KVCache.init(allocator, 1), .entry = head.entry };
    defer a.deinit();
    var b: Transformer.Qwen4MtpState = .{ .cache = try KVCache.init(allocator, 1), .entry = head.entry };
    defer b.deinit();
    xfm.qwen4MtpActivate(&a);
    xfm.qwen4_mtp.?.seq_offset = 7;
    xfm.qwen4_mtp.?.cache.step = 7;
    xfm.qwen4_mtp.?.pos_base = 3;
    xfm.qwen4MtpActivate(&b);
    xfm.qwen4_mtp.?.seq_offset = 2;
    xfm.qwen4_mtp.?.cache.step = 2;
    xfm.qwen4_mtp.?.pos_base = 11;
    const caches = [_]MtpCacheRef{
        .{ .qwen4 = .{ .t = &xfm, .st = &a, .allocator = allocator } },
        .{ .qwen4 = .{ .t = &xfm, .st = &b, .allocator = allocator } },
    };
    const want = [_]Transformer.Qwen4MtpState.Meta{
        .{ .stash_len = 7, .origin = 3, .cache_step = 7 },
        .{ .stash_len = 2, .origin = 11, .cache_step = 2 },
    };
    for (caches, want) |cache, meta| {
        const state = cache.qwen4StateForBatch();
        try testing.expectEqual(meta, state.meta());
    }
    try testing.expect(xfm.qwen4_mtp_owner == null);
}

pub fn installLogitBias(io: std.Io, xfm: *Transformer, tok: *const Tokenizer) !void {
    const setting = if (xfm.config.logit_bias_file_override) |*p| p.slice() else null;
    const path = setting orelse return;
    const body = try std.Io.Dir.cwd().readFileAlloc(io, path, xfm.allocator, .limited(16 << 20));
    defer xfm.allocator.free(body);
    const logits_dim: usize = xfm.config.vocab_size;
    const loaded = logit_bias.parse(xfm.allocator, body, std.fs.path.extension(path), tok, @min(logits_dim, tok.definedVocabSize())) catch |err| {
        log.err("[logit-bias] {s}: {s}\n", .{ path, @errorName(err) });
        return err;
    };
    xfm.logit_bias = loaded.biases;
    log.info("[logit-bias] {s}: {d} entries, {d} expanded ids, {d} multi-token spellings skipped\n", .{ path, loaded.entries, loaded.biases.len, loaded.skipped });
}
