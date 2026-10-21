//! DFlash round policy for a sparse target, where a verify row is not free: a round drafts once,
//! then verifies only the first R rows (t1 and R - 1 drafts), the R that maximizes emitted tokens
//! per ms under a per-position acceptance probability learned online from the drafter's own
//! confidence. Pure data, no MLX.
//!
//! The confidence calibration and the row choice follow mlx-serve's `dflash_policy.zig`
//! (https://github.com/ddalcu/mlx-serve, MIT); the cost ladder is Qwen3.8-Flash-Next's.
const std = @import("std");

/// t1 plus the drafts of the widest block a drafter serves.
pub const MAX_ROWS: u32 = 16;
pub const MAX_DRAFTS: u32 = MAX_ROWS - 1;

/// Round cost by verify rows relative to a one-row round; index = rows, 0 unused.
pub const Costs = [MAX_ROWS + 1]f32;

/// Qwen3.8-Flash-Next on an M5 Max, Sushi-2.6bpw: the verify costs 20.6, 26.3 and 41.7 ms at 2, 4 and 8
/// rows, a one-row round about 17.2 ms, so a row adds 0.2 of the first; accept and append add 4%.
const ROW_STEP: f32 = 0.20;
const FIXED: f32 = 0.04;

pub fn ladder() Costs {
    var out: Costs = undefined;
    out[0] = 0;
    for (1..MAX_ROWS + 1) |r| out[r] = 1 + FIXED + ROW_STEP * @as(f32, @floatFromInt(r - 1));
    return out;
}

pub const Choice = struct {
    /// Rows to verify, t1 included (1 = no draft is worth its row).
    rows: u32,
    /// Expected tokens the round emits.
    tokens: f32,
};

/// A row count within this fraction of the best rate loses to the cheaper one: the model is not that
/// exact, and a verify's cost is certain where its tokens are not.
const ROW_TIE: f32 = 0.02;

/// The row count with the best expected tokens per unit cost. `p[k]` is the probability that draft k is
/// accepted given drafts 0..k-1 were.
pub fn chooseRows(p: []const f32, costs: *const Costs) Choice {
    const n: usize = @min(p.len, MAX_DRAFTS);
    var tokens: [MAX_ROWS + 1]f32 = undefined;
    tokens[1] = 1;
    var alive: f32 = 1;
    for (p[0..n], 0..) |pk, k| {
        alive *= if (pk >= 0) @min(pk, 1) else 0; // a NaN is no acceptance
        tokens[k + 2] = tokens[k + 1] + alive;
    }
    var best: f32 = 0;
    for (1..n + 2) |r| best = @max(best, tokens[r] / costs[r]);
    for (1..n + 2) |r| {
        if (tokens[r] / costs[r] >= best * (1 - ROW_TIE)) return .{ .rows = @intCast(r), .tokens = tokens[r] };
    }
    return .{ .rows = 1, .tokens = 1 };
}

/// Probability that a draft is accepted given its predecessors were, by the drafter's confidence in it:
/// online bins over `-log10(1 - conf)`, each pulled toward a logistic prior by a few pseudo-counts and
/// forgetting its old rounds, so a change of content class re-calibrates in a few rounds.
pub const Calibration = struct {
    const BINS = 12;
    const PER_UNIT: f32 = 4;
    /// Pseudo-observations the prior is worth.
    const PRIOR_WEIGHT: f32 = 3;
    /// A bin forgets this share of its history per observation it takes.
    const KEEP: f32 = 0.97;

    hit: [BINS]f32 = @splat(0),
    seen: [BINS]f32 = @splat(0),

    /// 0 at a coin flip's confidence, 1 at 0.9, 2 at 0.99, capped at 3.
    fn axis(conf: f32) f32 {
        return -std.math.log10(@max(1 - std.math.clamp(conf, 0, 1), 1e-3));
    }

    fn bin(conf: f32) usize {
        return @min(BINS - 1, @as(usize, @intFromFloat(axis(conf) * PER_UNIT)));
    }

    fn prior(conf: f32) f32 {
        return 1 / (1 + @exp(-(-1.15 + 1.58 * axis(conf))));
    }

    pub fn probability(self: *const Calibration, conf: f32) f32 {
        const b = bin(conf);
        return (self.hit[b] + PRIOR_WEIGHT * prior(conf)) / (self.seen[b] + PRIOR_WEIGHT);
    }

    pub fn observe(self: *Calibration, conf: f32, accepted: bool) void {
        const b = bin(conf);
        self.hit[b] = self.hit[b] * KEEP + @as(f32, if (accepted) 1 else 0);
        self.seen[b] = self.seen[b] * KEEP + 1;
    }

    /// A round that verified `drafts` drafts with confidences `conf` and kept `accepted` of them: it reached
    /// every position up to the first rejection and no further.
    pub fn observeRound(self: *Calibration, conf: []const f32, drafts: u32, accepted: u32) void {
        const m: usize = @min(@min(drafts, conf.len), MAX_DRAFTS);
        for (0..m) |k| {
            if (k > accepted) break;
            self.observe(conf[k], k < accepted);
        }
    }
};

/// Rows (t1 included) for a round whose drafts carry confidences `conf`, kept inside `min..max`.
pub fn pickRows(calib: *const Calibration, conf: []const f32, min: u32, max: u32) u32 {
    var p: [MAX_DRAFTS]f32 = undefined;
    const n = @min(conf.len, p.len);
    for (conf[0..n], p[0..n]) |c, *out| out.* = calib.probability(c);
    const costs = ladder();
    return std.math.clamp(chooseRows(p[0..n], &costs).rows, min, max);
}

const testing = std.testing;

test "ladder: a row adds a fifth of the first, never cheaper for more rows" {
    const c = ladder();
    try testing.expectApproxEqAbs(@as(f32, 1.04), c[1], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 1.64), c[4], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 2.44), c[8], 1e-4);
    var r: u32 = 2;
    while (r <= MAX_ROWS) : (r += 1) try testing.expect(c[r] >= c[r - 1]);
}

test "chooseRows: sure drafts buy every row, a weak first draft buys none, a poor draft hides the ones behind it" {
    const costs = ladder();
    const sure: [7]f32 = @splat(0.97);
    try testing.expectEqual(@as(u32, 8), chooseRows(&sure, &costs).rows);
    try testing.expectEqual(@as(u32, 1), chooseRows(&[_]f32{ 0.1, 0.1, 0.1 }, &costs).rows);
    // Draft 1 almost never lands, so draft 2 is reached 5% of the time: two rows.
    try testing.expectEqual(@as(u32, 2), chooseRows(&[_]f32{ 0.95, 0.05, 0.99, 0.99 }, &costs).rows);
    // 70% a draft: the third and fourth rows still pay their fifth.
    try testing.expectEqual(@as(u32, 4), chooseRows(&[_]f32{ 0.8, 0.8, 0.8 }, &costs).rows);
    // 40% a draft: only the first row is worth its cost.
    try testing.expectEqual(@as(u32, 2), chooseRows(&[_]f32{ 0.4, 0.4, 0.4 }, &costs).rows);
}

test "chooseRows: a poisoned input still answers" {
    const costs = ladder();
    const bad: [3]f32 = .{ std.math.nan(f32), 0.9, 0.9 };
    try testing.expect(chooseRows(&bad, &costs).rows >= 1);
}

test "Calibration: no data answers the prior, then the data takes over bin by bin, and an old regime fades" {
    var c = Calibration{};
    try testing.expectApproxEqAbs(@as(f32, 0.606), c.probability(0.9), 0.02);
    for (0..40) |_| c.observe(0.9, false);
    try testing.expect(c.probability(0.9) < 0.3);
    try testing.expectApproxEqAbs(@as(f32, 0.269), c.probability(0.2), 0.02);
    var d = Calibration{};
    for (0..80) |_| d.observe(0.7, true);
    try testing.expect(d.probability(0.7) > 0.9);
    for (0..80) |_| d.observe(0.7, false);
    try testing.expect(d.probability(0.7) < 0.2);
}

test "Calibration.observeRound counts only the positions a round reached" {
    var c = Calibration{};
    // three drafts, the second rejected: positions 0 and 1 were reached, 2 never.
    c.observeRound(&[_]f32{ 0.9, 0.9, 0.99 }, 3, 1);
    var total: f32 = 0;
    for (c.seen) |v| total += v;
    try testing.expectApproxEqAbs(@as(f32, 2), total, 1e-4);
    var hits: f32 = 0;
    for (c.hit) |v| hits += v;
    try testing.expectApproxEqAbs(@as(f32, 1), hits, 1e-4);
}

test "pickRows: the calibrated choice, held inside min..max" {
    var c = Calibration{};
    const sure: [7]f32 = @splat(0.995);
    const unsure: [7]f32 = @splat(0.4);
    try testing.expect(pickRows(&c, &sure, 2, 8) >= 6);
    try testing.expectEqual(@as(u32, 4), pickRows(&c, &sure, 2, 4));
    try testing.expectEqual(@as(u32, 2), pickRows(&c, &unsure, 2, 8));
    try testing.expectEqual(@as(u32, 3), pickRows(&c, &unsure, 3, 8));
    // A regime where confident drafts keep failing pulls the choice down.
    for (0..200) |_| c.observeRound(&sure, 7, 0);
    try testing.expectEqual(@as(u32, 2), pickRows(&c, &sure, 2, 8));
}
