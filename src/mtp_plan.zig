const std = @import("std");
const transformer_mod = @import("transformer.zig");
const rp_mod = @import("reasoning_protocol.zig");
const io_util = @import("io_util.zig");
const mtp_acceptance = @import("mtp_acceptance.zig");
const model_settings = @import("model_settings.zig");
const round_cost = @import("round_cost.zig");
const depth_bounds = @import("mtp_depth_bounds.zig");
const Transformer = transformer_mod.Transformer;
const testing = std.testing;
const generate = @import("generate.zig");
const dsv41_dspark = @import("dsv41/dsv41_dspark.zig");
const sampler = @import("sampler.zig");
const CallForce = generate.CallForce;
const Constraint = generate.Constraint;
const Generator = generate.Generator;
const SamplingParams = sampler.SamplingParams;
const dsparkEligible = dsv41_dspark.dsparkEligible;
const penaltyActive = sampler.penaltyActive;

/// THIS model's acceptance mode: the launch flag > its `mtp_acceptance` > exact.
pub fn mtpAcceptanceFor(setting: ?mtp_acceptance.Mode) model_settings.Pick(mtp_acceptance.Mode) {
    return model_settings.pick(mtp_acceptance.Mode, model_settings.launchFlag(mtp_acceptance.Mode, generate.mtp_acceptance_default, generate.mtp_acceptance_explicit), setting, generate.mtp_acceptance_default);
}

/// THIS model's greedy tail: the launch flag > its `mtp_greedy_tail` > off.
pub fn mtpGreedyTailFor(setting: ?bool) model_settings.Pick(bool) {
    return model_settings.pick(bool, model_settings.launchFlag(bool, true, generate.mtp_greedy_tail_explicit), setting, false);
}

/// Per-prefill state for the adaptive width; lives on `runPrefill`'s stack. The policy is
/// `server.adaptivePrefillWidth`.
/// The one place a chosen width becomes the running width.
pub fn commitAdaptiveWidth(cur: *usize, st: *AdaptiveWidthState, w: u32) void {
    cur.* = w;
    st.transitions +|= 1;
    st.width_min = @min(st.width_min, w);
    st.width_max = @max(st.width_max, w);
}

pub const AdaptiveWidthState = struct {
    supporting: u8 = 0,
    /// One-way ratchet: a prefill that has stepped down never widens again.
    ratcheted: bool = false,
    transitions: u32 = 0,
    width_min: u32 = 0,
    width_max: u32 = 0,
};

/// Why a request left speculative decoding for the rest of its generation.
pub const SpecDisableReason = enum {
    none,
    acceptance,
    max_ctx,
    /// A thinking budget closed the thought through a plain forward; the spec
    /// state is not resynced, the answer decodes regular.
    think_bound,
    /// The measured round cost more per token than a measured serial token (`MtpAdaptive`).
    adaptive,
};

/// A thinking budget enforced while decoding: once `budget` tokens have been
/// generated inside the think block, `forced` (an early-stop sentence, the
/// pack's atomic closer, a separator) is committed through the model and the
/// answer follows. Markers are single token ids, so only packs whose think
/// tags are atomic tokens can be bound; `observe` walks the generated ids
/// incrementally and `due` asks whether the tail is an open block at budget.
pub const ThinkBound = struct {
    budget: u32,
    opener_id: ?u32,
    closer_id: u32,
    forced: []const u32,
    in_think: bool,
    count: u32 = 0,
    cursor: usize = 0,
    fired: bool = false,

    pub fn observe(self: *ThinkBound, ids: []const u32) void {
        while (self.cursor < ids.len) : (self.cursor += 1) {
            const id = ids[self.cursor];
            if (id == self.closer_id) {
                self.in_think = false;
            } else if (self.opener_id != null and id == self.opener_id.?) {
                self.in_think = true;
                self.count = 0;
            } else if (self.in_think) {
                self.count += 1;
            }
        }
    }

    pub fn due(self: *const ThinkBound) bool {
        return !self.fired and self.in_think and self.count >= self.budget;
    }

    /// Thought tokens left before the bound is due; null once it cannot fire. `observe` first.
    pub fn room(self: *const ThinkBound) ?u32 {
        if (self.fired or !self.in_think) return null;
        return self.budget -| self.count;
    }
};

/// Drafts an MTP round may carry so it emits at most `room` tokens (t1 plus accepted drafts) and the
/// bound fires at the token serial decode fires at. Null = no draft fits: the round runs serial.
pub fn mtpDraftCapForRoom(room: u32) ?u32 {
    return if (room > 1) room - 1 else null;
}

/// A hybrid request's restore point behind its generated thought's closer
/// ([engine-prefix-cache](../docs/engine-prefix-cache.md#thought-points)). The scheduler arms it after the prefill;
/// the speculative round that carries the closer is cut to end on it, and the state there joins the checkpoints.
pub const ThoughtPoint = struct {
    closer: u32,
    /// Positions a point may sit on: 1 where the recurrent state restores anywhere, the pool size on GLM.
    grid: u32,
    /// A thought shorter than this is cheaper to prefill than to keep.
    min_tokens: u32,
    /// Absolute position of the state the thought starts from.
    base: usize,
    state: State = .armed,
    /// Where the cut round leaves the state.
    at: usize = 0,

    pub const State = enum { armed, cut, taken, passed };
};

pub const ThoughtRound = union(enum) {
    /// The closer is not in the round: it runs whole.
    whole,
    /// The closer is in the round and no point can sit behind it.
    skip,
    /// Emit this many of the round's tokens, then take the point.
    keep: usize,
};

/// What a round whose emitted tokens `path` start at absolute position `anchor` does about the thought point: end on
/// the closer, or on the grid position at or below it.
pub fn thoughtRound(tp: ThoughtPoint, path: []const u32, anchor: usize) ThoughtRound {
    const at = std.mem.indexOfScalar(u32, path, tp.closer) orelse return .whole;
    const closer_pos = anchor + at;
    if (closer_pos -| tp.base < tp.min_tokens) return .skip;
    const end = (closer_pos + 1) / tp.grid * tp.grid;
    if (end <= anchor) return .skip;
    return .{ .keep = end - anchor };
}

/// Thoughts shorter than this are cheaper to prefill than to keep a state for.
pub const thought_point_min_tokens: u32 = 512;

/// How a template writes a thought's closer back: the Qwen template trims the thought and writes `\n</think>`, GLM's
/// writes the trimmed thought and `</think>`.
pub const CloserStyle = enum { newline_before_closer, bare_closer };

/// Does the next prompt repeat the generated thought? The server trims the reasoning it delivers and the template
/// trims it again, so a thought that starts in whitespace, or ends in any the style does not write, is re-rendered
/// differently. `first` is the thought's first token as text, `tail` its last two.
pub fn thoughtEdgesSurvive(style: CloserStyle, first: []const u8, tail: []const u8) bool {
    const white = struct {
        fn at(c: u8) bool {
            return c == ' ' or c == '\n' or c == '\t' or c == '\r';
        }
    }.at;
    if (first.len == 0 or white(first[0])) return false;
    return switch (style) {
        .bare_closer => tail.len > 0 and !white(tail[tail.len - 1]),
        .newline_before_closer => tail.len >= 2 and tail[tail.len - 1] == '\n' and !white(tail[tail.len - 2]),
    };
}

/// A round of `t1` and its `accepted` drafts, ending in `pending`, cut to emit `keep` tokens: the draft the round accepted
/// after the cut is the pending token, so the stream is the whole round's.
pub fn cutMtpRound(drafts: []const u32, accepted: u32, pending: u32, keep: usize) struct { accepted: u32, pending: u32 } {
    if (keep >= accepted + 1) return .{ .accepted = accepted, .pending = pending };
    return .{ .accepted = @intCast(keep - 1), .pending = drafts[keep - 1] };
}

/// A speculative decode step (DFlash / MTP) cannot honor a grammar
/// constraint — the drafts bypass the per-token grammar mask — nor per-token
/// logprobs, which the draft path never captures. The scheduler and server
/// already select a spec mode only when both are absent, but `nextDflash` /
/// `nextMtp` only *documented* that with `std.debug.assert`,
/// which compiles out in the ReleaseFast build users actually run (issue #97).
/// A future dispatch bug that let a constrained request reach a spec step would
/// then read the constrained branch's placeholder token as a real commit and
/// stream silently off-schema output as a normal 200. So the spec steps gate on
/// this real, release-enforced check and return `error.SpecDecodeUnsupported`
/// instead of asserting.
pub fn specDecodeUnsupported(sampling: SamplingParams, logprobs_n: u32) bool {
    return draftsRefused(sampling) or logprobs_n != 0;
}

/// MTP rounds shift every verify row for a repeat, presence or frequency penalty (`penaltyShiftRows`), so
/// only what a draft run cannot honour keeps them off.
pub fn mtpUnsupported(sampling: SamplingParams, logprobs_n: u32) bool {
    return mtpRefusal(sampling, logprobs_n) != null;
}

/// Why a request runs without MTP, for the log; null when MTP may run.
pub fn mtpRefusal(sampling: SamplingParams, logprobs_n: u32) ?[]const u8 {
    if (logprobs_n != 0) return "logprobs requested";
    if (sampling.constraint) |c| if (!c.rowsDraftable()) return "grammar-constrained output";
    if (sampling.call_force != null) return "forced tool call";
    return null;
}

/// A grammar mask or a forced tool call decides tokens a draft run would skip past, and a
/// repeat, presence or frequency penalty reads ids no verify row has committed yet; only MTP shifts the rows.
pub fn draftsRefused(sampling: SamplingParams) bool {
    return sampling.constraint != null or sampling.call_force != null or penaltyActive(sampling);
}

/// Verify rows that end a round at the first `id` among `ids`; the trunk's own token after it is still the pending one.
pub fn rowsThroughId(ids: []const u32, id: ?u32) usize {
    const want = id orelse return ids.len;
    return if (std.mem.indexOfScalar(u32, ids, want)) |at| at + 1 else ids.len;
}

test "rowsThroughId: a round ends on the id, or runs whole without one" {
    try testing.expectEqual(@as(usize, 5), rowsThroughId(&.{ 1, 2, 3, 4, 5 }, null));
    try testing.expectEqual(@as(usize, 5), rowsThroughId(&.{ 1, 2, 3, 4, 5 }, 9));
    try testing.expectEqual(@as(usize, 3), rowsThroughId(&.{ 1, 2, 3, 4, 3 }, 3));
    try testing.expectEqual(@as(usize, 1), rowsThroughId(&.{ 3, 2, 3 }, 3));
    try testing.expectEqual(@as(usize, 0), rowsThroughId(&.{}, 3));
}

/// Number of accepted draft tokens that may accompany the always-committed
/// anchor token without crossing the request's output-token ceiling.
pub fn capAcceptedForTokenBudget(accepted: u32, completion: u32, max_tokens: u32) u32 {
    const remaining = max_tokens -| completion;
    if (remaining <= 1) return 0;
    return @min(accepted, remaining - 1);
}

test "capAcceptedForTokenBudget keeps speculative commits inside max_tokens" {
    try std.testing.expectEqual(@as(u32, 15), capAcceptedForTokenBudget(15, 0, 400));
    try std.testing.expectEqual(@as(u32, 2), capAcceptedForTokenBudget(15, 397, 400));
    try std.testing.expectEqual(@as(u32, 0), capAcceptedForTokenBudget(15, 399, 400));
    try std.testing.expectEqual(@as(u32, 0), capAcceptedForTokenBudget(15, 400, 400));
    try std.testing.expectEqual(@as(u32, 0), capAcceptedForTokenBudget(15, 401, 400));
}

test "specDecodeUnsupported: release-enforced guard for spec + constraint/logprobs (issue #97)" {
    // Speculative decode (drafter/MTP) cannot honor a grammar constraint or
    // per-token logprobs. nextDflash/nextMtp only asserted this, which
    // compiles out in ReleaseFast (issue #97) — this is the real check they now
    // use to fail loud instead of streaming silently off-schema output.
    try testing.expect(!specDecodeUnsupported(.{}, 0)); // plain sampling → spec OK
    try testing.expect(specDecodeUnsupported(.{}, 1)); // any logprobs → not OK
    try testing.expect(specDecodeUnsupported(.{}, 5));
    // A grammar constraint present → not OK. The guard only null-checks the
    // pointer (never dereferences), so a dummy address is sufficient here.
    var dummy: Constraint = undefined;
    try testing.expect(specDecodeUnsupported(.{ .constraint = &dummy }, 0));
    try testing.expect(specDecodeUnsupported(.{ .constraint = &dummy }, 3));
}

test "specDecodeUnsupported: a forced tool call keeps every draft path off" {
    // Drafts are accepted in runs, so no draft path can hand the position after
    // the think closer to the forced call.
    var cf = CallForce{ .forced = &.{1}, .call_at = 0, .closer_id = null, .phase = .answer, .last = 0 };
    try testing.expect(specDecodeUnsupported(.{ .call_force = &cf }, 0));
}

test "ThinkBound: counts only tokens inside the think block and fires at the budget" {
    const OPEN: u32 = 10;
    const CLOSE: u32 = 11;
    const forced = [_]u32{ 30, 31, CLOSE, 32 };
    var tb = ThinkBound{ .budget = 3, .opener_id = OPEN, .closer_id = CLOSE, .forced = &forced, .in_think = false };

    // Content before any opener never counts.
    tb.observe(&[_]u32{ 1, 2, 3, 4, 5 });
    try testing.expect(!tb.due());

    // Opened by the model: the budget counts from the opener.
    tb.observe(&[_]u32{ 1, 2, 3, 4, 5, OPEN, 6, 7 });
    try testing.expect(!tb.due());
    tb.observe(&[_]u32{ 1, 2, 3, 4, 5, OPEN, 6, 7, 8 });
    try testing.expect(tb.due());

    // A model that closed on its own is never forced.
    var closed = ThinkBound{ .budget = 3, .opener_id = OPEN, .closer_id = CLOSE, .forced = &forced, .in_think = true };
    closed.observe(&[_]u32{ 6, 7, CLOSE, 8, 9, 10, 11 });
    try testing.expect(!closed.due());

    // Prompt-opened: the count starts at token 0 and the forced closer ends it.
    var po = ThinkBound{ .budget = 2, .opener_id = null, .closer_id = CLOSE, .forced = &forced, .in_think = true };
    po.observe(&[_]u32{ 6, 7 });
    try testing.expect(po.due());
    po.fired = true;
    po.observe(&[_]u32{ 6, 7, 30, 31, CLOSE, 32, 40 });
    try testing.expect(!po.in_think);
    try testing.expect(!po.due());
}

test "commitAdaptiveWidth: the summary follows the running width from both commit sites" {
    const t = testing;
    var cur: usize = 2048;
    var st: AdaptiveWidthState = .{ .width_min = 2048, .width_max = 2048 };
    commitAdaptiveWidth(&cur, &st, 1024);
    try t.expectEqual(@as(usize, 1024), cur);
    try t.expectEqual(@as(u32, 1), st.transitions);
    try t.expectEqual(@as(u32, 1024), st.width_min);
    try t.expectEqual(@as(u32, 2048), st.width_max);
    commitAdaptiveWidth(&cur, &st, 4096);
    try t.expectEqual(@as(usize, 4096), cur);
    try t.expectEqual(@as(u32, 2), st.transitions);
    try t.expectEqual(@as(u32, 1024), st.width_min);
    try t.expectEqual(@as(u32, 4096), st.width_max);
}

fn mtpEvTestGenerator() Generator {
    var g: Generator = undefined;
    g.mtp_depth = Generator.MTP_ADAPTIVE_DEFAULT_CAP;
    g.mtp_depth_current = 1;
    g.mtp_window_drafted = @splat(0);
    g.mtp_window_accepted = @splat(0);
    g.mtp_window_idx = 0;
    g.mtp_rounds_since_switch = 0;
    g.mtp_promote_cooldown = 0;
    g.mtp_ev_accept = @splat(Generator.MTP_EV_PRIOR);
    g.mtp_ev_rounds = Generator.MTP_EV_WARMUP_ROUNDS;
    g.mtp_ev_m_lo_prev = 1;
    g.mtp_ev_costs = Generator.MTP_EV_DEFAULT_COSTS;
    g.spec_disabled_runtime = false;
    return g;
}

test "mtpGreedyTailFor: the greedy tail flag outranks the per-model mtp_greedy_tail, which outranks off" {
    const saved = generate.mtp_greedy_tail_explicit;
    defer generate.mtp_greedy_tail_explicit = saved;
    generate.mtp_greedy_tail_explicit = false;
    try testing.expectEqual(model_settings.Pick(bool){ .value = false, .source = .default }, mtpGreedyTailFor(null));
    try testing.expectEqual(model_settings.Pick(bool){ .value = true, .source = .model_settings }, mtpGreedyTailFor(true));
    try testing.expectEqual(model_settings.Pick(bool){ .value = false, .source = .model_settings }, mtpGreedyTailFor(false));
    generate.mtp_greedy_tail_explicit = true;
    try testing.expectEqual(model_settings.Pick(bool){ .value = true, .source = .flag }, mtpGreedyTailFor(false));
    try testing.expectEqual(model_settings.Pick(bool){ .value = true, .source = .flag }, mtpGreedyTailFor(null));
}

test "updateMtpEvRound: sticky disable uses the first draft at base depth one" {
    var good = mtpEvTestGenerator();
    for (0..Generator.MTP_DEPTH_WINDOW * 2) |_| good.updateMtpEvRound(8, 1);
    try testing.expect(!good.spec_disabled_runtime);
    for (good.mtp_window_drafted) |sample| try testing.expectEqual(@as(u8, 1), sample);
    for (good.mtp_window_accepted) |sample| try testing.expectEqual(@as(u8, 1), sample);

    var bad = mtpEvTestGenerator();
    for (0..Generator.MTP_DEPTH_WINDOW - 1) |_| {
        bad.updateMtpEvRound(8, 0);
        try testing.expect(!bad.spec_disabled_runtime);
    }
    bad.updateMtpEvRound(8, 0);
    try testing.expect(bad.spec_disabled_runtime);
}

test "updateMtpEvRound: warmup and wider base rounds reset floor evidence" {
    var g = mtpEvTestGenerator();
    g.mtp_ev_rounds = Generator.MTP_EV_WARMUP_ROUNDS - 1;
    g.mtp_window_idx = 7;
    g.updateMtpEvRound(4, 0);
    try testing.expectEqual(Generator.MTP_EV_WARMUP_ROUNDS, g.mtp_ev_rounds);
    try testing.expectEqual(@as(u32, 0), g.mtp_window_idx);

    for (0..Generator.MTP_DEPTH_WINDOW - 1) |_| g.updateMtpEvRound(1, 0);
    try testing.expect(!g.spec_disabled_runtime);
    try testing.expectEqual(Generator.MTP_DEPTH_WINDOW - 1, g.mtp_window_idx);

    g.mtp_ev_m_lo_prev = 2;
    g.updateMtpEvRound(4, 0);
    try testing.expectEqual(@as(u32, 0), g.mtp_window_idx);
    try testing.expect(!g.spec_disabled_runtime);
}

fn mtpBoundsPlanFixture(xfm: *Transformer, prompt: []u32, depth: u32) Generator {
    xfm.config.model_type = "qwen4_exp";
    xfm.round_cost = .{ .layout = .long, .first_use_logged = true };
    for (0..round_cost.MIN_SAMPLES) |_| {
        _ = xfm.round_cost.observe(2, 1000, 29.0, 2.6, true, false);
        _ = xfm.round_cost.observe(3, 1000, 31.0, 2.9, true, false);
        _ = xfm.round_cost.observe(4, 1000, 36.0, 3.1, true, false);
        _ = xfm.round_cost.observe(5, 1000, 42.0, 3.5, true, false);
    }
    return Generator{
        .xfm = xfm,
        .ctx = undefined,
        .tok = undefined,
        .next_token_id = 0,
        .step = 0,
        .max_tokens = 1,
        .sampling = .{ .temperature = 0 },
        .prompt_tokens = 0,
        .completion_tokens = 0,
        .finish_reason = "length",
        .done = false,
        .eos_token_ids = &.{},
        .generated_ids = .empty,
        .timeout_ns = 0,
        .timer = io_util.Stopwatch.init(testing.io),
        .last_hidden = .{ .ctx = null },
        .has_last_hidden = false,
        .prompt_ids_owned = prompt,
        .mtp_depth = depth,
        .mtp_depth_free = depth,
        .mtp_ev_rounds = 64,
        .mtp_ev_accept = .{ 0.92, 0.9, 0.88, 0.86, 0.84, 0.8, 0.8, 0.8 },
        .mtp_ev_m_lo_prev = 4,
        .spec_cost_solo = false,
    };
}

test "MTP depth bounds: the planner reads min..max from its table-priced plan" {
    const before = depth_bounds.active;
    defer depth_bounds.active = before;
    var xfm: Transformer = undefined;
    var prompt: [1000]u32 = @splat(0);

    depth_bounds.active = .{};
    var gen = mtpBoundsPlanFixture(&xfm, &prompt, 6);
    try testing.expectEqual(@as(u32, 3), gen.mtpRoundPlan().m_lo);

    depth_bounds.active = .{ .min = 5 };
    gen = mtpBoundsPlanFixture(&xfm, &prompt, 6);
    var plan = gen.mtpRoundPlan();
    try testing.expect(plan.m_lo >= 5 and plan.m_hi >= plan.m_lo and plan.m_hi <= 6);

    depth_bounds.active = .{ .max = 2 };
    gen = mtpBoundsPlanFixture(&xfm, &prompt, 2);
    plan = gen.mtpRoundPlan();
    try testing.expect(plan.m_lo <= 2 and plan.m_hi <= 2);
}

test "MTP depth bounds: min == max drafts that depth every round" {
    const before = depth_bounds.active;
    defer depth_bounds.active = before;
    var xfm: Transformer = undefined;
    var prompt: [1000]u32 = @splat(0);
    for ([_]u32{ 2, 4, 5 }) |depth| {
        depth_bounds.active = .{ .min = depth, .max = depth };
        var gen = mtpBoundsPlanFixture(&xfm, &prompt, depth);
        for (0..4) |_| {
            const plan = gen.mtpRoundPlan();
            try testing.expectEqual(depth, plan.m_lo);
            try testing.expectEqual(depth, plan.m_hi);
        }
    }
}

test "MTP depth bounds: the windowed controller never leaves min..max and never disables above a floor of 1" {
    const before = depth_bounds.active;
    defer depth_bounds.active = before;
    depth_bounds.active = .{ .min = 3, .max = 5 };
    var hot = mtpEvTestGenerator();
    hot.mtp_depth = 5;
    hot.mtp_depth_current = 3;
    for (0..64) |_| {
        hot.updateMtpDepth(5, 5);
        try testing.expect(hot.mtp_depth_current >= 3 and hot.mtp_depth_current <= 5);
    }
    try testing.expectEqual(@as(u32, 5), hot.mtp_depth_current);

    var cold = mtpEvTestGenerator();
    cold.mtp_depth = 5;
    cold.mtp_depth_current = 5;
    for (0..64) |_| {
        cold.updateMtpDepth(5, 0);
        try testing.expect(cold.mtp_depth_current >= 3 and cold.mtp_depth_current <= 5);
    }
    try testing.expectEqual(@as(u32, 3), cold.mtp_depth_current);
    try testing.expect(!cold.spec_disabled_runtime);
}

test "mtpRefusal: a constraint speculates only where the drafter verifies rows under masks, and never through a channel header" {
    var c = Constraint{ .grammar = undefined, .token_bytes = undefined, .mask_buf = &.{} };
    try testing.expectEqualStrings("grammar-constrained output", mtpRefusal(.{ .constraint = &c }, 0).?);
    c.row_masks = true;
    try testing.expect(mtpRefusal(.{ .constraint = &c }, 0) == null);
    try testing.expect(dsparkEligible(.{ .constraint = &c }, 0));
    try testing.expectEqualStrings("logprobs requested", mtpRefusal(.{ .constraint = &c }, 2).?);
    var proto = rp_mod.Protocol{ .kind = .bare_think };
    c.proto = &proto;
    c.pstate = .{ .phase = .reasoning };
    try testing.expect(mtpRefusal(.{ .constraint = &c }, 0) == null);
    c.pstate = .{ .phase = .choice };
    try testing.expect(mtpRefusal(.{ .constraint = &c }, 0) != null);
    c.pstate = .{ .phase = .reasoning };
    proto.header_len = 1;
    try testing.expect(mtpRefusal(.{ .constraint = &c }, 0) != null);
}

test "thoughtRound: a round ends on the closer or the grid position at or below it, and a short thought or a passed grid takes no point" {
    const recurrent: ThoughtPoint = .{ .closer = 9, .grid = 1, .min_tokens = 4, .base = 100 };
    try testing.expectEqual(ThoughtRound.whole, thoughtRound(recurrent, &.{ 1, 2, 3 }, 110));
    try testing.expectEqual(ThoughtRound{ .keep = 3 }, thoughtRound(recurrent, &.{ 1, 2, 9, 4, 5 }, 110));
    try testing.expectEqual(ThoughtRound{ .keep = 1 }, thoughtRound(recurrent, &.{ 9, 4 }, 110));
    try testing.expectEqual(ThoughtRound{ .keep = 5 }, thoughtRound(recurrent, &.{ 1, 2, 3, 4, 9 }, 110));
    // The closer at 103 ends a thought of three tokens.
    try testing.expectEqual(ThoughtRound.skip, thoughtRound(recurrent, &.{ 9, 4 }, 103));

    const pooled: ThoughtPoint = .{ .closer = 9, .grid = 4, .min_tokens = 4, .base = 100 };
    // The closer at 112: the state behind it is 113, the pool boundary at or below is 112.
    try testing.expectEqual(ThoughtRound{ .keep = 2 }, thoughtRound(pooled, &.{ 1, 2, 9, 4, 5 }, 110));
    try testing.expectEqual(ThoughtRound{ .keep = 2 }, thoughtRound(pooled, &.{ 1, 9, 4, 5 }, 110));
    try testing.expectEqual(ThoughtRound{ .keep = 1 }, thoughtRound(pooled, &.{ 9, 4 }, 111));
    // The boundary below the closer is behind the round's own start: no state stands there.
    try testing.expectEqual(ThoughtRound.skip, thoughtRound(pooled, &.{ 9, 4 }, 110));
}

test "cutMtpRound: a round cut at any of its tokens streams what the whole round does" {
    const drafts = [_]u32{ 41, 42, 43, 44 };
    const stream = [_]u32{ 40, 41, 42, 43, 44, 45 };
    for (1..6) |keep| {
        const cut = cutMtpRound(&drafts, 4, 45, keep);
        try testing.expectEqual(keep, 1 + @as(usize, cut.accepted));
        try testing.expectEqual(stream[keep], cut.pending);
    }
    const whole = cutMtpRound(&drafts, 4, 45, 6);
    try testing.expectEqual(@as(u32, 4), whole.accepted);
    try testing.expectEqual(@as(u32, 45), whole.pending);
}

test "thoughtEdgesSurvive: the thought comes back only as the template writes it" {
    try testing.expect(thoughtEdgesSurvive(.newline_before_closer, "The", "t.\n"));
    try testing.expect(thoughtEdgesSurvive(.newline_before_closer, "The", ".\n"));
    try testing.expect(!thoughtEdgesSurvive(.newline_before_closer, "The", "it."));
    try testing.expect(!thoughtEdgesSurvive(.newline_before_closer, "The", ". \n"));
    try testing.expect(!thoughtEdgesSurvive(.newline_before_closer, "The", ".\n\n"));
    try testing.expect(!thoughtEdgesSurvive(.newline_before_closer, " The", ".\n"));
    try testing.expect(!thoughtEdgesSurvive(.newline_before_closer, "\n", ".\n"));
    try testing.expect(thoughtEdgesSurvive(.bare_closer, "The", "it."));
    try testing.expect(!thoughtEdgesSurvive(.bare_closer, "The", ".\n"));
    try testing.expect(!thoughtEdgesSurvive(.bare_closer, "", "it."));
}
