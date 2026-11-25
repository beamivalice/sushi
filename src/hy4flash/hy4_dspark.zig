//! Hy4-Flash's drafter slot: DSpark rounds (`dspark_round.Rounds`) over DFly2 (`hy4_dfly`). A round drafts a block
//! `[t1, mask × draft]` from the model's own state, verifies `[t1, drafts]` through the trunk with a mark taken before
//! it (`Model.mark`), and keeps the first `accepted + 1` rows by rolling back to the mark and running them again
//! (`Model.rollbackTo`). There is no head to step: `arch/drafter.zig` answers every head operation with its default.
const mlx = @import("../mlx.zig");
const trunk_mod = @import("../trunk.zig");
const spec_types = @import("../spec_types.zig");
const sampler = @import("../sampler.zig");
const dspark_round = @import("../dspark_round.zig");
const hy4_dfly = @import("hy4_dfly.zig");
const Trunk = trunk_mod.Trunk;
const ForwardCtx = trunk_mod.ForwardCtx;

pub const kinds = [_]spec_types.Kind{.dfly2};
pub const head_kind: spec_types.Kind = .dfly2;

/// DFly2 drafts over the model's own state, so a request that drafts with it holds the model exclusively.
pub const Head = *Trunk;

pub fn headOf(xfm: *Trunk, _: bool, _: ?*anyopaque) ?Head {
    return if (xfm.arch_state.hy4flash.model != null and xfm.arch_state.hy4flash.drafter != null) xfm else null;
}

/// The widest verify a round runs: the pending token and its drafts, and room past them.
const max_rows = 16;

const Hooks = struct {
    pub const Draft = hy4_dfly.Drafter.Draft;
    pub const row_grammar = @import("../row_grammar.zig");
    pub const max_verify_rows: u8 = max_rows;
    pub const diag_env = "SUSHI_HY4_DSPARK_ROWS";
    /// No copy rounds (`copyArm` is null).
    pub const copy_narrow: usize = 0;

    pub fn draft(xfm: *Trunk, ctx: *ForwardCtx, t1: u32) !Draft {
        const st = &xfm.arch_state.hy4flash;
        const m = st.model orelse return error.Hy4ModelMissing;
        const d = &(st.drafter orelse return error.Hy4DflyMissing);
        if (m.pos != ctx.cache.step) return error.Hy4CachePositionMismatch;
        return @import("hy4_plan.zig").runDraft(m, d, t1);
    }

    /// Advances the request by `ids` (`[t1, drafts...]`) with a mark before them, which `finish` returns to; the
    /// logits of every row, f32 `[1, L, vocab]`.
    pub fn verify(xfm: *Trunk, ctx: *ForwardCtx, ids: []const u32) !mlx.mlx_array {
        const st = &xfm.arch_state.hy4flash;
        const m = st.model orelse return error.Hy4ModelMissing;
        if (m.pos != ctx.cache.step) return error.Hy4CachePositionMismatch;
        if (ids.len == 0 or ids.len > max_rows) return error.SpecDecodeUnsupported;
        if (st.verify != null) return error.Hy4VerifyOpen;
        var v: @import("hy4_state.zig").State.Verify = .{ .mark = if (ids.len > 1) try m.mark() else null, .ids = undefined, .n = ids.len };
        errdefer if (v.mark) |*mk| m.dropMark(mk);
        @memcpy(v.ids[0..ids.len], ids);
        if (xfm.expert_stream) |engine| engine.beginForward();
        const logits = try @import("hy4_plan.zig").run(m, ids, null, .all);
        defer _ = mlx.mlx_array_free(logits);
        if (xfm.expert_stream) |engine| engine.finishForward(ids.len);
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_reshape(&out, logits, &[_]c_int{ 1, @intCast(ids.len), mlx.getShape(logits)[1] }, 3, xfm.s));
        // The mark is the open verify's from here, the last step that can fail behind: `finish` or `abort` drops it.
        st.verify = v;
        ctx.cache.step = m.pos;
        return out;
    }

    pub fn verifyPlain(xfm: *Trunk, ctx: *ForwardCtx, id: u32) !mlx.mlx_array {
        const st = &xfm.arch_state.hy4flash;
        const m = st.model orelse return error.Hy4ModelMissing;
        if (m.pos != ctx.cache.step) return error.Hy4CachePositionMismatch;
        if (st.verify != null) return error.Hy4VerifyOpen;
        var v: @import("hy4_state.zig").State.Verify = .{ .mark = null, .ids = undefined, .n = 1 };
        v.ids[0] = id;
        const was_keep = m.keep_hidden;
        m.keep_hidden = false;
        defer m.keep_hidden = was_keep;
        m.dropHidden();
        if (xfm.expert_stream) |engine| engine.beginForward();
        const logits = try @import("hy4_plan.zig").run(m, &.{id}, null, .last);
        defer _ = mlx.mlx_array_free(logits);
        if (xfm.expert_stream) |engine| engine.finishForward(1);
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_reshape(&out, logits, &[_]c_int{ 1, 1, mlx.getShape(logits)[1] }, 3, xfm.s));
        st.verify = v;
        ctx.cache.step = m.pos;
        return out;
    }

    pub fn setPlainKeepHidden(xfm: *Trunk, keep: bool) bool {
        const m = xfm.arch_state.hy4flash.model orelse return false;
        const was = m.keep_hidden;
        m.keep_hidden = keep;
        if (!keep) m.dropHidden();
        return was;
    }

    /// The round failed between `verify` and `finish`: the open verify's mark is dropped and the model reset, as its
    /// state stands mid-round and no record names it.
    pub fn abort(xfm: *Trunk, _: *ForwardCtx) void {
        @import("hy4_state.zig").dropVerify(&xfm.arch_state.hy4flash, true);
    }

    /// Keeps the first `accepted + 1` rows of the open verify: back to its mark, the kept rows run again.
    pub fn finish(xfm: *Trunk, ctx: *ForwardCtx, accepted: usize) !void {
        const st = &xfm.arch_state.hy4flash;
        const m = st.model orelse return error.Hy4ModelMissing;
        var v = st.verify orelse return error.Hy4VerifyNotOpen;
        st.verify = null;
        defer if (v.mark) |*mk| m.dropMark(mk);
        const keep = 1 + accepted;
        if (keep > v.n) return error.SpecDecodeUnsupported;
        if (keep < v.n) {
            if (Hooks.widthExact(xfm) and m.keepRowsApplies(&v.mark.?, keep)) {
                try m.keepRows(&v.mark.?, keep);
            } else {
                if (xfm.expert_stream) |engine| engine.beginForward();
                try m.rollbackTo(&v.mark.?, v.ids[0..keep]);
                if (xfm.expert_stream) |engine| engine.finishForward(keep);
            }
        }
        ctx.cache.step = m.pos;
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
        return if (xfm.arch_state.hy4flash.model) |m| m.config.dfly.draft else @import("../arch/dspark.zig").default_block;
    }

    pub fn copyArm(_: *Trunk) ?dspark_round.CopyArm {
        return null;
    }
};

pub const rounds = dspark_round.Rounds(Hooks);
pub const dspark_rounds = rounds;
pub const nextDspark = rounds.nextDspark;
