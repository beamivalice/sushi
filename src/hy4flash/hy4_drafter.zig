//! Hy4-Flash's drafter slot: DFly2 (`hy4_dfly`) in the shared drafter rounds (`drafter_round.Rounds`). A round drafts a
//! block `[t1, mask × draft]` from the model's own state, verifies `[t1, drafts]` through the trunk with a mark taken
//! before it (`Model.mark`), and keeps the first `accepted + 1` rows by rolling back to the mark and running them again
//! (`Model.rollbackTo`). There is no head to step: `arch/drafter.zig` answers every head operation with its default.
const mlx = @import("../mlx.zig");
const trunk_mod = @import("../trunk.zig");
const spec_types = @import("../spec_types.zig");
const sampler = @import("../sampler.zig");
const drafter_round = @import("../drafter_round.zig");
const hy4_dfly = @import("hy4_dfly.zig");
const Trunk = trunk_mod.Trunk;
const ForwardCtx = trunk_mod.ForwardCtx;
const Generator = @import("../generate.zig").Generator;

pub const kinds = [_]spec_types.Kind{.dfly2};
pub const head_kind: spec_types.Kind = .dfly2;

/// DFly2 drafts over the model's own state, so a request that drafts with it holds the model exclusively.
pub const Head = *Trunk;

pub fn headOf(xfm: *Trunk, _: bool, _: ?*anyopaque) ?Head {
    return if (xfm.arch_state.hy4flash.model != null and xfm.arch_state.hy4flash.drafter != null) xfm else null;
}

/// The widest verify a round runs: the pending token and its drafts, and room past them.
pub const max_rows: usize = 16;

pub fn groupEligible(gen: *const Generator) bool {
    if (!gen.drafter_on or gen.done or gen.spec_disabled_runtime) return false;
    if (gen.sampling.constraint != null or @import("../sampler.zig").penaltyActive(gen.sampling)) return false;
    if (gen.sampling.think_bound != null or gen.sampling.call_force != null) return false;
    return !gen.samplesSync();
}

pub fn draftBlock(xfm: *Trunk, ctx: *ForwardCtx, t1: u32) !Hooks.Draft {
    return Hooks.draft(xfm, ctx, t1);
}

pub fn draftBatch(
    xfm: *Trunk,
    ctxs: []const *ForwardCtx,
    t1s: []const u32,
    max_drafts: usize,
    out: []Hooks.Draft,
) !void {
    const st = &xfm.arch_state.hy4flash;
    const m = st.model orelse return error.Hy4ModelMissing;
    const d = &(st.drafter orelse return error.Hy4DflyMissing);
    const n = ctxs.len;
    if (n == 0 or n > hy4_dfly.Drafter.max_batch_slots or t1s.len != n or out.len != n) return error.Hy4RowsWidth;
    var reqs: [hy4_dfly.Drafter.max_batch_slots]*@import("hy4_model.zig").Request = undefined;
    for (ctxs, 0..) |ctx, i| {
        const req = @import("hy4_state.zig").requestOf(ctx) orelse return error.Hy4NoRequest;
        if (req.pos != ctx.cache.step) return error.Hy4CachePositionMismatch;
        reqs[i] = req;
    }
    return @import("hy4_plan.zig").runDraftBatch(m, d, reqs[0..n], t1s, max_drafts, out);
}

pub fn beginGroupVerify(xfm: *Trunk, ctx: *ForwardCtx, ids: []const u32) !void {
    const st = &xfm.arch_state.hy4flash;
    const m = st.model orelse return error.Hy4ModelMissing;
    const scope = m.enterForward(@import("hy4_state.zig").requestOf(ctx));
    defer scope.endForward();
    const r = m.req();
    if (r.pos != ctx.cache.step) return error.Hy4CachePositionMismatch;
    if (ids.len == 0 or ids.len > max_rows) return error.SpecDecodeUnsupported;
    if (r.verify != null) return error.Hy4VerifyOpen;
    var v: @import("hy4_state.zig").State.Verify = .{ .mark = if (ids.len > 1) try m.mark() else null, .ids = undefined, .n = ids.len };
    @memcpy(v.ids[0..ids.len], ids);
    r.verify = v;
}

pub fn finishGroupVerify(xfm: *Trunk, ctx: *ForwardCtx, accepted: usize) !void {
    return Hooks.finish(xfm, ctx, accepted);
}

const Hooks = struct {
    pub const Draft = hy4_dfly.Drafter.Draft;
    pub const row_grammar = @import("../row_grammar.zig");
    pub const max_verify_rows: u8 = max_rows;
    pub const diag_env = "SUSHI_HY4_DFLY2_ROWS";
    /// No copy rounds (`copyArm` is null).
    pub const copy_narrow: usize = 0;

    pub fn draft(xfm: *Trunk, ctx: *ForwardCtx, t1: u32) !Draft {
        const st = &xfm.arch_state.hy4flash;
        const m = st.model orelse return error.Hy4ModelMissing;
        const d = &(st.drafter orelse return error.Hy4DflyMissing);
        const scope = m.enterForward(@import("hy4_state.zig").requestOf(ctx));
        defer scope.endForward();
        if (m.reqConst().pos != ctx.cache.step) return error.Hy4CachePositionMismatch;
        return @import("hy4_plan.zig").runDraft(m, d, t1);
    }

    /// Advances the request by `ids` (`[t1, drafts...]`) with a mark before them, which `finish` returns to; the
    /// logits of every row, f32 `[1, L, vocab]`.
    pub fn verify(xfm: *Trunk, ctx: *ForwardCtx, ids: []const u32) !mlx.mlx_array {
        const st = &xfm.arch_state.hy4flash;
        const m = st.model orelse return error.Hy4ModelMissing;
        const scope = m.enterForward(@import("hy4_state.zig").requestOf(ctx));
        defer scope.endForward();
        const r = m.req();
        if (r.pos != ctx.cache.step) return error.Hy4CachePositionMismatch;
        if (ids.len == 0 or ids.len > max_rows) return error.SpecDecodeUnsupported;
        if (r.verify != null) return error.Hy4VerifyOpen;
        var v: @import("hy4_state.zig").State.Verify = .{ .mark = if (ids.len > 1) try m.mark() else null, .ids = undefined, .n = ids.len };
        errdefer if (v.mark) |*mk| m.dropMark(mk);
        @memcpy(v.ids[0..ids.len], ids);
        if (xfm.expert_stream) |engine| engine.beginForward();
        defer if (xfm.expert_stream) |engine| engine.finishForward(ids.len);
        const logits = try @import("hy4_plan.zig").run(m, ids, null, .all);
        defer _ = mlx.mlx_array_free(logits);
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_reshape(&out, logits, &[_]c_int{ 1, @intCast(ids.len), mlx.getShape(logits)[1] }, 3, xfm.s));
        // The mark is the open verify's from here, the last step that can fail behind: `finish` or `abort` drops it.
        r.verify = v;
        ctx.cache.step = r.pos;
        return out;
    }

    pub fn verifyPlain(xfm: *Trunk, ctx: *ForwardCtx, id: u32) !mlx.mlx_array {
        const st = &xfm.arch_state.hy4flash;
        const m = st.model orelse return error.Hy4ModelMissing;
        const scope = m.enterForward(@import("hy4_state.zig").requestOf(ctx));
        defer scope.endForward();
        const r = m.req();
        if (r.pos != ctx.cache.step) return error.Hy4CachePositionMismatch;
        if (r.verify != null) return error.Hy4VerifyOpen;
        var v: @import("hy4_state.zig").State.Verify = .{ .mark = null, .ids = undefined, .n = 1 };
        v.ids[0] = id;
        const was_keep = m.keep_hidden;
        m.keep_hidden = false;
        defer m.keep_hidden = was_keep;
        m.dropHidden();
        if (xfm.expert_stream) |engine| engine.beginForward();
        defer if (xfm.expert_stream) |engine| engine.finishForward(1);
        const logits = try @import("hy4_plan.zig").run(m, &.{id}, null, .last);
        defer _ = mlx.mlx_array_free(logits);
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_reshape(&out, logits, &[_]c_int{ 1, 1, mlx.getShape(logits)[1] }, 3, xfm.s));
        r.verify = v;
        ctx.cache.step = r.pos;
        return out;
    }

    pub fn setPlainKeepHidden(xfm: *Trunk, keep: bool) bool {
        const m = xfm.arch_state.hy4flash.model orelse return false;
        const was = m.keep_hidden;
        m.keep_hidden = keep;
        if (!keep and (!m.serves_requests or m.bound != null)) m.dropHidden();
        return was;
    }

    pub fn hasHandover(xfm: *Trunk, ctx: *ForwardCtx) bool {
        const m = xfm.arch_state.hy4flash.model orelse return false;
        const r = @import("hy4_state.zig").requestOf(ctx) orelse &m.own;
        return m.hasHandover(r);
    }

    /// The round failed between `verify` and `finish`: the open verify's mark is dropped and the model reset, as its
    /// state stands mid-round and no record names it.
    pub fn abort(xfm: *Trunk, ctx: *ForwardCtx) void {
        const m = xfm.arch_state.hy4flash.model orelse return;
        const scope = m.enterForward(@import("hy4_state.zig").requestOf(ctx));
        defer scope.endForward();
        @import("hy4_state.zig").dropVerifyReq(m, m.req(), true);
    }

    /// Keeps the first `accepted + 1` rows of the open verify: back to its mark, the kept rows run again.
    pub fn finish(xfm: *Trunk, ctx: *ForwardCtx, accepted: usize) !void {
        const st = &xfm.arch_state.hy4flash;
        const m = st.model orelse return error.Hy4ModelMissing;
        const scope = m.enterForward(@import("hy4_state.zig").requestOf(ctx));
        defer scope.endForward();
        const r = m.req();
        var v = r.verify orelse return error.Hy4VerifyNotOpen;
        r.verify = null;
        defer if (v.mark) |*mk| m.dropMark(mk);
        const keep = 1 + accepted;
        if (keep > v.n) return error.SpecDecodeUnsupported;
        if (keep < v.n) {
            if (Hooks.widthExact(xfm) and m.keepRowsApplies(&v.mark.?, keep)) {
                try m.keepRows(&v.mark.?, keep);
            } else {
                if (xfm.expert_stream) |engine| engine.beginForward();
                defer if (xfm.expert_stream) |engine| engine.finishForward(keep);
                try m.rollbackTo(&v.mark.?, v.ids[0..keep]);
            }
        }
        ctx.cache.step = r.pos;
    }

    /// Whether a verify row is its one-row step's bytes over widths 2..8, so greedy and seeded rounds may take any
    /// width: true on an STQ load (trellis trunk and experts) on Sushi Cooler's GPU stream without expert-cache bias
    /// when the drafter block fits 8 rows (`hy4_row`, `hy4_absorb`, f32 linears and attention products match one-row
    /// steps, and sliding attention reads keys in ring order). False on BF16 loads, off Cooler's GPU stream, under
    /// cache bias, or for blocks wider than 8 rows.
    pub fn widthExact(xfm: *Trunk) bool {
        const m = xfm.arch_state.hy4flash.model orelse return false;
        if (!mlx.engine.isStream(m.s) or !mlx.streamIsGpu(m.s)) return false;
        if (m.quant.trunk == null or m.quant.expert == null) return false;
        if (m.config.dfly.draft + 1 > @import("hy4_row.zig").max_rows) return false;
        if (m.experts) |e| if (e.cache_bias > 0) return false;
        return true;
    }

    pub fn verdictStream(_: *Trunk, s: mlx.mlx_stream) mlx.mlx_stream {
        return s;
    }

    /// The verify rows' own draws, row j keyed by the generated position it predicts (`sampler.engineDraws`): every
    /// Hy4 load runs on Sushi Cooler.
    pub fn sampledDraws(sampling: sampler.SamplingParams, logits: mlx.mlx_array, rows: usize, s: mlx.mlx_stream) !mlx.mlx_array {
        const host = mlx.engine.enter(s);
        defer host.end();
        if (rows > max_rows or !host.own) return error.SpecDecodeUnsupported;
        var masked = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(masked);
        if (sampling.suppress_mask) |mk| try sampler.applySuppressMask(&masked, logits, mk, s);
        const rows2d = try spec_types.verifyRows2d(if (sampling.suppress_mask != null) masked else logits, @intCast(rows - 1), s);
        defer _ = mlx.mlx_array_free(rows2d);
        return sampler.engineDraws(rows2d, sampling, s);
    }

    pub fn vocab(xfm: *Trunk) usize {
        return if (xfm.arch_state.hy4flash.model) |m| m.config.vocab else 0;
    }

    pub fn stageBlock(xfm: *Trunk) u32 {
        return if (xfm.arch_state.hy4flash.model) |m| m.config.dfly.draft else @import("../arch/drafter_rounds.zig").default_block;
    }

    pub fn copyArm(_: *Trunk) ?drafter_round.CopyArm {
        return null;
    }
};

pub const rounds = drafter_round.Rounds(Hooks);
pub const drafter_rounds = rounds;
pub const nextDrafterRound = rounds.nextDrafterRound;
