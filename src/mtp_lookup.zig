//! Prompt-lookup drafts for the MTP round, ported from mlx-serve #523 (Samuel Reed). When
//! the last committed tokens plus the round's first token occurred earlier in the prompt or
//! output, the tokens that followed make a draft with no head forward; `gate` decides when
//! that beats the MTP chain. Only committed tokens are indexed.

const std = @import("std");

pub const NGRAM: usize = 4;
/// Shorter context matches land too few drafts to pay for their verify.
pub const MIN_SUFFIX: u32 = 8;
pub const MAX_DRAFT: u32 = 7;
/// Nine verify rows: the widest verify whose rows match the decode tick and whose history
/// stash fits `MtpHistStash` (upstream drafts 14 here).
pub const MAX_DRAFT_STRONG: u32 = 8;
pub const STRONG_SUFFIX: u32 = 32;
const SUFFIX_CAP: u32 = 64;
/// Earlier occurrences `matchBest` compares.
const CANDIDATES: u32 = 8;

/// Round costs in one unit (the caller's EV cost units): an MTP round at the chain's
/// width, and a lookup round by draft count.
pub const Costs = struct {
    mtp: f32,
    lookup: [MAX_DRAFT_STRONG + 1]f32,
};

/// Accepted drafts per round, smoothed.
pub fn emaStep(ema: f32, accepted: u32) f32 {
    return 0.7 * ema + 0.3 * @as(f32, @floatFromInt(accepted));
}

/// Pulls the lookup EMA back toward its prior on MTP rounds, so lookups get retried.
pub fn driftStep(ema: f32) f32 {
    return 0.98 * ema + 0.02 * @as(f32, @floatFromInt(MAX_DRAFT));
}

pub const Match = struct {
    /// Borrowed from the index.
    draft: []const u32,
    /// Tokens the two sites agree on going back, n-gram included, capped.
    suffix: u32,
    /// The agreement runs past the start of a line: the output copies lines, not the tail of
    /// one line whose next line it will not follow (a diff's `-`/`+`/` ` prefix, a quote).
    crosses_line: bool = true,
};

pub const Index = struct {
    allocator: std.mem.Allocator,
    toks: std.ArrayList(u32) = .empty,
    /// Per position: the token carries a line break.
    breaks: std.ArrayList(bool) = .empty,
    /// n-gram -> position of the token after its latest occurrence.
    next: std.AutoHashMapUnmanaged(u128, u32) = .empty,
    /// By the position of an n-gram's token after it: the same position for the n-gram's previous occurrence, or `none`.
    prev: std.ArrayList(u32) = .empty,

    const none: u32 = std.math.maxInt(u32);

    pub fn init(allocator: std.mem.Allocator) Index {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Index) void {
        self.toks.deinit(self.allocator);
        self.breaks.deinit(self.allocator);
        self.next.deinit(self.allocator);
        self.prev.deinit(self.allocator);
    }

    /// `lines.hasNewline(id)` says which tokens carry a line break.
    pub fn extend(self: *Index, ids: []const u32, lines: anytype) !void {
        try self.toks.ensureUnusedCapacity(self.allocator, ids.len);
        try self.breaks.ensureUnusedCapacity(self.allocator, ids.len);
        try self.prev.ensureTotalCapacity(self.allocator, self.toks.items.len + ids.len);
        for (ids) |id| {
            self.toks.appendAssumeCapacity(id);
            self.breaks.appendAssumeCapacity(lines.hasNewline(id));
            const n = self.toks.items.len;
            self.prev.items.len = n;
            self.prev.items[n - 1] = none;
            if (n <= NGRAM) continue;
            const g = self.toks.items[n - 1 - NGRAM .. n - 1];
            const slot = try self.next.getOrPut(self.allocator, key(g[0], g[1], g[2], g[3]));
            if (slot.found_existing) self.prev.items[n - 1] = slot.value_ptr.*;
            slot.value_ptr.* = @intCast(n - 1);
        }
    }

    /// Continuation after the latest earlier occurrence of the last NGRAM-1
    /// committed tokens plus `t1` (not yet committed).
    pub fn match(self: *const Index, t1: u32, max_draft: u32) ?Match {
        const toks = self.toks.items;
        if (toks.len < NGRAM - 1 or max_draft == 0) return null;
        const t = toks[toks.len - (NGRAM - 1) ..];
        const p: usize = self.next.get(key(t[0], t[1], t[2], t1)) orelse return null;
        return self.matchAt(p, t1, max_draft);
    }

    /// `match`, but of the last `CANDIDATES` occurrences the one that agrees furthest back (the latest among equals): an
    /// edit that left the latest occurrence on the wrong copy still finds the right one.
    pub fn matchBest(self: *const Index, t1: u32, max_draft: u32) ?Match {
        const toks = self.toks.items;
        if (toks.len < NGRAM - 1 or max_draft == 0) return null;
        const t = toks[toks.len - (NGRAM - 1) ..];
        var p: usize = self.next.get(key(t[0], t[1], t[2], t1)) orelse return null;
        var best = self.matchAt(p, t1, max_draft);
        var seen: u32 = 1;
        while (seen < CANDIDATES and best.suffix < SUFFIX_CAP and self.prev.items[p] != none) : (seen += 1) {
            p = self.prev.items[p];
            const m = self.matchAt(p, t1, max_draft);
            if (m.suffix > best.suffix) best = m;
        }
        return best;
    }

    fn matchAt(self: *const Index, p: usize, t1: u32, max_draft: u32) Match {
        const toks = self.toks.items;
        var suffix: u32 = 0;
        var crosses = false;
        var a: usize = p;
        var b: usize = toks.len + 1;
        while (suffix < SUFFIX_CAP and a > 0) : (suffix += 1) {
            a -= 1;
            b -= 1;
            if (toks[a] != (if (b == toks.len) t1 else toks[b])) break;
            // A break counts only with agreement after it: a line's last token (a code line
            // ends in e.g. `):\n`) agrees whatever the next line starts with.
            crosses = crosses or (suffix > 0 and self.breaks.items[a]);
        }
        return .{ .draft = toks[p..@min(p + max_draft, toks.len)], .suffix = suffix, .crosses_line = crosses };
    }

    fn key(a: u32, b: u32, c: u32, d: u32) u128 {
        return (@as(u128, a) << 96) | (@as(u128, b) << 64) | (@as(u128, c) << 32) | d;
    }
};

/// DSpark copy rounds: agreement that marks a quoted span; a shorter one lands too few drafts to pay for the wider verify.
pub const COPY_SUFFIX: u32 = 16;

/// Drafts a DSpark copy round takes from a match agreeing `min_suffix` back (`COPY_SUFFIX`), or 0 for the stage draft: only a quoted span whose copy outruns the
/// stage `block` is worth the wider verify.
pub fn copyLen(found: ?Match, cap: usize, block: usize, min_suffix: u32) usize {
    const f = found orelse return 0;
    if (f.suffix < min_suffix) return 0;
    const n = @min(f.draft.len, cap);
    return if (n > block) n else 0;
}

/// When a request stops taking DSpark copies: each copy round is priced in ms per token against the stage rounds'; copies
/// that lose make the next matches be skipped, the stretch doubling to a cap.
pub const CopyBackoff = struct {
    owed: u32 = 0,
    period: u32 = 1,
    /// Decayed ms a copy round saved against stage rounds for its tokens; negative while copies lose.
    edge: f32 = 0,
    stage_ms_per_token: f32 = 12,
    /// Drafts kept per copy round, smoothed, and whether the last copy kept every draft.
    kept: f32 = 5,
    streak: bool = false,

    pub const max_period: u32 = 16;
    const decay: f32 = 0.8;

    /// Drafts the next copy round verifies, of `cap`: what copies have been landing plus 2, or the cap right after one
    /// landed whole.
    pub fn width(self: *const CopyBackoff, cap: usize) usize {
        if (self.streak) return cap;
        return @min(cap, @as(usize, @intFromFloat(@ceil(@max(self.kept, 0)))) + 2);
    }

    /// Whether to take a match worth copying.
    pub fn take(self: *CopyBackoff) bool {
        if (self.owed == 0) return true;
        self.owed -= 1;
        return false;
    }

    /// A stage or plain round that committed `tokens` in `ms`.
    pub fn observeStage(self: *CopyBackoff, tokens: usize, ms: f32) void {
        if (tokens == 0 or !std.math.isFinite(ms) or ms <= 0) return;
        self.stage_ms_per_token = decay * self.stage_ms_per_token + (1 - decay) * ms / @as(f32, @floatFromInt(tokens));
    }

    /// A copy round that verified `drafted` copies, kept `kept` of them and took `ms`.
    pub fn observeCopy(self: *CopyBackoff, drafted: usize, kept: usize, ms: f32) void {
        self.kept = emaStep(self.kept, @intCast(kept));
        self.streak = kept == drafted;
        if (!std.math.isFinite(ms) or ms <= 0) return;
        const saved = @as(f32, @floatFromInt(1 + kept)) * self.stage_ms_per_token - ms;
        self.edge = decay * self.edge + (1 - decay) * saved;
        if (self.edge < 0) {
            self.owed = self.period;
            self.period = @min(self.period * 2, max_period);
        } else self.period = 1;
    }
};

/// Draft length for a match, or 0 to run MTP: an ordinary match must agree across a line
/// break (a strong one need not); sized to what lookups have been landing, or
/// to the cap right after one landed every draft (`streak`), and taken only when it
/// promises more tokens per cost than the MTP chain it replaces. Never past `remaining` or
/// the draft itself.
pub fn gate(m: ?Match, remaining: u32, lookup_ema: f32, mtp_ema: f32, streak: bool, costs: Costs) u32 {
    const got = m orelse return 0;
    if (got.suffix < MIN_SUFFIX or (got.suffix < STRONG_SUFFIX and !got.crosses_line)) return 0;
    const cap = if (got.suffix >= STRONG_SUFFIX) MAX_DRAFT_STRONG else MAX_DRAFT;
    const sized: u32 = if (streak) cap else @as(u32, @intFromFloat(@ceil(std.math.clamp(lookup_ema, 0, @as(f32, @floatFromInt(cap)))))) + 2;
    const k = @min(@min(cap, sized), @min(@as(u32, @intCast(got.draft.len)), remaining));
    if (k == 0) return 0;
    const kf: f32 = @floatFromInt(k);
    const lookup_rate = ((if (streak) kf else @min(lookup_ema, kf)) + 1) / costs.lookup[k];
    return if (lookup_rate > (mtp_ema + 1) / costs.mtp) k else 0;
}

const testing = std.testing;

/// A verify row at `row`, a head draft step at `draft`, in units of one serial forward.
fn linearCosts(mtp_width: u32, row: f32, draft: f32) Costs {
    var c = Costs{ .mtp = 1 + @as(f32, @floatFromInt(mtp_width)) * (row + draft), .lookup = undefined };
    for (&c.lookup, 0..) |*x, k| x.* = 1 + @as(f32, @floatFromInt(k)) * row;
    return c;
}

/// Token 0 is the line break in the index tests.
const TestLines = struct {
    fn hasNewline(_: TestLines, id: u32) bool {
        return id == 0;
    }
};

test "lookup: latest earlier continuation, keyed on the tail plus t1, with how far back it agrees" {
    var idx = Index.init(testing.allocator);
    defer idx.deinit();
    // Two earlier "1 2 3 4" sites; the later one agrees back through "0 11 12 13".
    try idx.extend(&.{ 1, 2, 3, 4, 10, 50, 0, 11, 12, 13, 1, 2, 3, 4, 77, 5 }, TestLines{});
    try idx.extend(&.{ 60, 0, 11, 12, 13, 1, 2, 3 }, TestLines{});
    const m = idx.match(4, 3) orelse return error.NoMatch;
    try testing.expectEqualSlices(u32, &.{ 77, 5, 60 }, m.draft);
    try testing.expectEqual(@as(u32, 8), m.suffix);
    try testing.expect(m.crosses_line);
    try testing.expect(idx.match(9, 3) == null);
}

test "lookup: a match that agrees only within one line does not cross it" {
    var idx = Index.init(testing.allocator);
    defer idx.deinit();
    // The context line "0 21 22 23 24 25 26 27 28 29 30 31" is echoed behind a prefix token 9.
    try idx.extend(&.{ 0, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 0, 40 }, TestLines{});
    try idx.extend(&.{ 0, 9, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30 }, TestLines{});
    const m = idx.match(31, 2) orelse return error.NoMatch;
    try testing.expectEqual(@as(u32, 11), m.suffix);
    try testing.expect(!m.crosses_line);
    try testing.expectEqualSlices(u32, &.{ 0, 40 }, m.draft);
    // The line echoed to its end: t1 is the break, which agrees whatever comes next.
    try idx.extend(&.{31}, TestLines{});
    const end = idx.match(0, 2) orelse return error.NoMatch;
    try testing.expectEqual(@as(u32, 12), end.suffix);
    try testing.expect(!end.crosses_line);
}

test "lookup gate: short matches, weak lookups and exhausted budgets run MTP" {
    var d: [20]u32 = undefined;
    for (&d, 0..) |*x, i| x.* = @intCast(i);
    const strong = Match{ .draft = &d, .suffix = STRONG_SUFFIX };
    const ordinary = Match{ .draft = &d, .suffix = MIN_SUFFIX };
    const w6 = linearCosts(6, 0.23, 0.05);
    try testing.expectEqual(@as(u32, 0), gate(.{ .draft = &d, .suffix = MIN_SUFFIX - 1 }, 100, 8, 0, true, w6));
    // Within one line only an agreement past STRONG_SUFFIX drafts.
    try testing.expectEqual(@as(u32, 0), gate(.{ .draft = &d, .suffix = STRONG_SUFFIX - 1, .crosses_line = false }, 100, 8, 0, true, w6));
    try testing.expectEqual(MAX_DRAFT_STRONG, gate(.{ .draft = &d, .suffix = STRONG_SUFFIX, .crosses_line = false }, 100, 8, 0, true, w6));
    try testing.expectEqual(@as(u32, 0), gate(strong, 100, 3.5, 4.6, false, w6));
    try testing.expectEqual(@as(u32, 0), gate(ordinary, 100, 4.0, 5.3, false, w6));
    try testing.expectEqual(@as(u32, 0), gate(strong, 0, 8, 0, true, w6));
}

test "lookup gate: a landing streak drafts to the cap; otherwise the draft tracks the EMA" {
    var d: [20]u32 = undefined;
    for (&d, 0..) |*x, i| x.* = @intCast(i);
    const strong = Match{ .draft = &d, .suffix = STRONG_SUFFIX };
    const w6 = linearCosts(6, 0.23, 0.05);
    try testing.expectEqual(MAX_DRAFT_STRONG, gate(strong, 100, 6.0, 4.6, true, w6));
    try testing.expectEqual(MAX_DRAFT, gate(.{ .draft = &d, .suffix = MIN_SUFFIX }, 100, 6.0, 4.6, true, w6));
    try testing.expectEqual(@as(u32, 7), gate(strong, 100, 5.0, 1.0, false, linearCosts(3, 0.23, 0.05)));
    try testing.expectEqual(@as(u32, 5), gate(strong, 5, 6.0, 1.0, true, w6));
    try testing.expectEqual(@as(u32, 3), gate(.{ .draft = d[0..3], .suffix = STRONG_SUFFIX }, 100, 6.0, 1.0, true, w6));
}

test "lookup gate: the draft never passes the strong cap, whatever the EMA says" {
    var d: [20]u32 = undefined;
    for (&d, 0..) |*x, i| x.* = @intCast(i);
    const strong = Match{ .draft = &d, .suffix = SUFFIX_CAP };
    const costs = linearCosts(1, 0.01, 0.5);
    for ([_]f32{ 0, 3, 8, 50, std.math.inf(f32) }) |ema| {
        for ([_]bool{ false, true }) |streak| try testing.expect(gate(strong, 1000, ema, 0, streak, costs) <= MAX_DRAFT_STRONG);
    }
}

test "lookup gate: the prices decide, not the draft length" {
    var d: [20]u32 = undefined;
    for (&d, 0..) |*x, i| x.* = @intCast(i);
    const ordinary = Match{ .draft = &d, .suffix = MIN_SUFFIX };
    // At linear prices MTP wins (6.3 tokens per 2.68 against 5 per 2.38); a lookup measured
    // cheap on this machine wins.
    var cheap = linearCosts(6, 0.23, 0.05);
    try testing.expectEqual(@as(u32, 0), gate(ordinary, 100, 4.0, 5.3, false, cheap));
    cheap.lookup[6] = 1.5;
    try testing.expectEqual(@as(u32, 6), gate(ordinary, 100, 4.0, 5.3, false, cheap));
    // At linear prices the lookup wins at seven drafts; measured dear, MTP runs.
    const strong = Match{ .draft = &d, .suffix = STRONG_SUFFIX };
    var dear = linearCosts(3, 0.23, 0.05);
    try testing.expectEqual(@as(u32, 7), gate(strong, 100, 5.0, 1.0, false, dear));
    dear.lookup[7] = 20.0;
    try testing.expectEqual(@as(u32, 0), gate(strong, 100, 5.0, 1.0, false, dear));
}

test "lookup EMAs: landed drafts pull toward the round, MTP rounds drift back to the prior" {
    try testing.expectApproxEqAbs(@as(f32, 0.7 * 7.0 + 0.3 * 2.0), emaStep(7.0, 2), 1e-6);
    var ema: f32 = 0;
    for (0..400) |_| ema = driftStep(ema);
    try testing.expectApproxEqAbs(@as(f32, MAX_DRAFT), ema, 0.01);
}

test "lookup best: the earlier site that agrees furthest back wins over a later one an edit left on another copy" {
    var idx = Index.init(testing.allocator);
    defer idx.deinit();
    try idx.extend(&.{ 20, 21, 22, 23, 1, 2, 3, 4, 5, 6, 40, 41, 1, 2, 3, 4, 99 }, TestLines{});
    try idx.extend(&.{ 20, 21, 22, 23, 1, 2, 3 }, TestLines{});
    const m = idx.matchBest(4, 2) orelse return error.NoMatch;
    try testing.expectEqualSlices(u32, &.{ 5, 6 }, m.draft);
    // Equal agreement: the latest site.
    var tie = Index.init(testing.allocator);
    defer tie.deinit();
    try tie.extend(&.{ 1, 2, 3, 4, 7, 1, 2, 3, 4, 8, 1, 2, 3 }, TestLines{});
    try testing.expectEqualSlices(u32, &.{ 8, 1 }, (tie.matchBest(4, 2) orelse return error.NoMatch).draft);
    try testing.expect(tie.matchBest(9, 2) == null);
}

test "dspark copy: only a long agreement whose draft outruns the stage block is copied" {
    const draft: [15]u32 = @splat(0);
    try testing.expectEqual(@as(usize, 15), copyLen(.{ .draft = &draft, .suffix = COPY_SUFFIX }, 15, 5, COPY_SUFFIX));
    try testing.expectEqual(@as(usize, 9), copyLen(.{ .draft = &draft, .suffix = COPY_SUFFIX }, 9, 5, COPY_SUFFIX));
    try testing.expectEqual(@as(usize, 0), copyLen(null, 15, 5, COPY_SUFFIX));
    try testing.expectEqual(@as(usize, 0), copyLen(.{ .draft = &draft, .suffix = COPY_SUFFIX - 1 }, 15, 5, COPY_SUFFIX));
    try testing.expectEqual(@as(usize, 0), copyLen(.{ .draft = draft[0..5], .suffix = COPY_SUFFIX }, 15, 5, COPY_SUFFIX));
    try testing.expectEqual(@as(usize, 0), copyLen(.{ .draft = &draft, .suffix = COPY_SUFFIX }, 5, 5, COPY_SUFFIX));
}

test "dspark copy backoff: copies that lose to stage rounds are skipped for a doubling stretch; a winning copy clears it" {
    var b = CopyBackoff{};
    for (0..10) |_| b.observeStage(3, 30);
    try testing.expect(b.take());
    b.observeCopy(15, 2, 60);
    try testing.expectEqual(@as(u32, 1), b.owed);
    try testing.expect(!b.take());
    try testing.expect(b.take());
    b.observeCopy(15, 2, 60);
    try testing.expectEqual(@as(u32, 2), b.owed);
    for (0..8) |_| {
        b.owed = 0;
        b.observeCopy(15, 2, 60);
    }
    try testing.expectEqual(CopyBackoff.max_period, b.period);
    b.owed = 0;
    for (0..4) |_| b.observeCopy(15, 15, 60);
    try testing.expect(b.edge > 0);
    try testing.expectEqual(@as(u32, 1), b.period);
}

test "dspark copy backoff: a round verifies what copies have landed plus 2, the cap after one landed whole" {
    var b = CopyBackoff{};
    try testing.expectEqual(@as(usize, 7), b.width(15));
    b.observeCopy(7, 7, 50);
    try testing.expectEqual(@as(usize, 15), b.width(15));
    b.observeCopy(15, 1, 50);
    try testing.expectEqual(@as(usize, 7), b.width(15));
    for (0..6) |_| b.observeCopy(7, 1, 50);
    try testing.expect(b.width(15) <= 5);
    try testing.expectEqual(@as(usize, 4), b.width(4));
}
