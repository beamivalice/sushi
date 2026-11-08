//! DFlash round policy for a sparse target, where a verify row is not free: a round drafts once,
//! then verifies only the first R rows (t1 and R - 1 drafts), the R that maximizes emitted tokens
//! per ms under a per-position acceptance probability learned online from the drafter's own
//! confidence. Pure data, no MLX.
//!
//! The confidence calibration and the row choice follow mlx-serve's `dflash_policy.zig`
//! (https://github.com/ddalcu/mlx-serve, MIT); the cost ladder is Qwen3.8-Flash-Next's.
const std = @import("std");
const round_cost = @import("round_cost.zig");

/// t1 plus the drafts of the widest block a drafter serves.
pub const MAX_ROWS: u32 = 16;
pub const MAX_DRAFTS: u32 = MAX_ROWS - 1;

/// Round cost by verify rows relative to a one-row round; index = rows, 0 unused.
pub const Costs = [MAX_ROWS + 1]f32;

/// Verify forward ms by rows (index = rows, 0 unused) on an M5 Max, Qwen3.8-Flash-Next-SushiTrellisQ-2.75, kv8:
/// the decode meter's mean over 1K and 8K context, two boots each. Rows past 8 continue at the 2..8 slope.
const VERIFY_MS = [_]f32{ 0, 17.22, 20.38, 22.37, 25.07, 27.46, 30.78, 34.53, 37.37 };
/// What a round pays however many rows it verifies: the drafter, the head and selector, accept and append.
const FIXED_MS: f32 = 4.0;

fn verifyMs(rows: usize) f32 {
    if (rows < VERIFY_MS.len) return VERIFY_MS[rows];
    const last = VERIFY_MS.len - 1;
    return VERIFY_MS[last] + (VERIFY_MS[last] - VERIFY_MS[2]) / @as(f32, @floatFromInt(last - 2)) * @as(f32, @floatFromInt(rows - last));
}

pub fn ladder() Costs {
    var out: Costs = undefined;
    out[0] = 0;
    const one = verifyMs(1) + FIXED_MS;
    for (1..MAX_ROWS + 1) |r| out[r] = (verifyMs(r) + FIXED_MS) / one;
    return out;
}

/// Accepted drafts per round at which a round of `rows` rows (fractional for a mean) costs the same per
/// token as a serial step, which is one verify forward of a single row.
pub fn breakEvenDrafts(rows: f32) f32 {
    const c = ladder();
    const r = std.math.clamp(rows, 1, @as(f32, MAX_ROWS));
    const lo: usize = @intFromFloat(@floor(r));
    const hi: usize = @min(lo + 1, MAX_ROWS);
    const cost = c[lo] + (c[hi] - c[lo]) * (r - @as(f32, @floatFromInt(lo)));
    return cost * (verifyMs(1) + FIXED_MS) / verifyMs(1) - 1;
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
    return bestRows(&tokens, n, costs);
}

fn bestRows(tokens: *const [MAX_ROWS + 1]f32, n: usize, costs: *const Costs) Choice {
    var best: f32 = 0;
    for (1..n + 2) |r| best = @max(best, tokens[r] / costs[r]);
    for (1..n + 2) |r| {
        if (tokens[r] / costs[r] >= best * (1 - ROW_TIE)) return .{ .rows = @intCast(r), .tokens = tokens[r] };
    }
    return .{ .rows = 1, .tokens = 1 };
}

/// Round cost for a ladder that starts at `first` for one row and adds `step` per row.
pub fn linearLadder(first: f32, step: f32) Costs {
    var out: Costs = undefined;
    out[0] = 0;
    for (1..MAX_ROWS + 1) |r| out[r] = first + step * @as(f32, @floatFromInt(r - 1));
    return out;
}

/// Rows for a tree of drafts in the order taken (`parents[k]` is a draft index before k, or -1 under t1; `conf[k]`
/// the drafter's probability of draft k among its siblings): a draft adds the probability that the accepted path
/// runs through it, since at most one sibling does.
pub fn chooseTreeRows(calib: *const Calibration, parents: []const i32, conf: []const f32, costs: *const Costs) Choice {
    const n: usize = @min(@min(parents.len, conf.len), MAX_DRAFTS);
    var tokens: [MAX_ROWS + 1]f32 = undefined;
    var path: [MAX_DRAFTS]f32 = undefined;
    tokens[1] = 1;
    for (0..n) |k| {
        const own = calib.probability(conf[k]);
        path[k] = (if (own >= 0) @min(own, 1) else 0) * (if (parents[k] < 0) 1 else path[@intCast(parents[k])]);
        tokens[k + 2] = tokens[k + 1] + path[k];
    }
    return bestRows(&tokens, n, costs);
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

    /// A verified tree: row k (parent row `parents[k]`, 0 = t1) was reached when its parent was accepted, and is
    /// accepted when it is on the path `accepted` (rows in order, t1 first).
    pub fn observeTree(self: *Calibration, conf: []const f32, parents: []const i32, count: usize, accepted: []const u32) void {
        var on_path: [16]bool = @splat(false);
        for (accepted) |row| on_path[row] = true;
        for (1..@min(count, @min(conf.len, 16))) |row| {
            if (on_path[@intCast(parents[row])]) self.observe(conf[row], on_path[row]);
        }
    }
};

/// Rows (t1 included) for a tree of drafts in the order taken, kept inside `min..max`.
pub fn pickTreeRows(calib: *const Calibration, parents: []const i32, conf: []const f32, min: u32, max: u32, costs: *const Costs) u32 {
    return std.math.clamp(chooseTreeRows(calib, parents, conf, costs).rows, min, max);
}

/// Rows (t1 included) for a round whose drafts carry confidences `conf`, kept inside `min..max`.
pub fn pickRows(calib: *const Calibration, conf: []const f32, min: u32, max: u32) u32 {
    var p: [MAX_DRAFTS]f32 = undefined;
    const n = @min(conf.len, p.len);
    for (conf[0..n], p[0..n]) |c, *out| out.* = calib.probability(c);
    const costs = ladder();
    return std.math.clamp(chooseRows(p[0..n], &costs).rows, min, max);
}

const testing = std.testing;

// DFlash has different economics from sequential-token drafters: one
// assistant call proposes the whole block and the M5 NAX verify lane makes
// wide blocks cheap. Per-draft percentage therefore penalizes block 16
// even when it is decisively profitable. The stable cross-width signal is
// accepted drafts per verify round. M5 sweeps put code/tool traffic at
// 4.4-15.0 accepted/round and the regressing prose/vision class at
// 1.0-1.5; two is the measured break-even boundary. M4 block-5 model-card
// workloads accepting 62-86% remain above it as well.
// Warmup 8, not 3 (2026-09-03): three rounds tripped on ONE dry opening
// (llmprobe's 16k rung: 0/3, drafter off for the whole request at 25
// tok/s; the same prompt restored from cache decoded at 73).
pub const DFLASH_GATE_WARMUP: u64 = 8;
/// A DFlash round that costs more per emitted token than this multiple of a serial step is dropped.
pub const DFLASH_GATE_MARGIN: f32 = 0.05;

/// Accepted drafts per round at which a round of `rows` verifier rows costs one serial step per
/// token, used while the request's context bucket holds no serial sample. The policy's own cost ladder.
pub fn dflashBreakEvenAccepted(rows: f32) f32 {
    return breakEvenDrafts(rows);
}

/// Does the request's DFlash yield lose to serial decoding? With a measured serial price (the
/// round-cost table's cell) the test is ms per emitted token; without one it is the ladder's
/// break-even yield. Never decided before the warmup rounds.
pub fn dflashGateShouldDisable(attempted: u64, accepted: u64, wall_ms: f64, rows: f32, serial_ms_tok: ?f32) bool {
    if (attempted < DFLASH_GATE_WARMUP) return false;
    if (serial_ms_tok) |serial| {
        if (std.math.isFinite(serial) and serial > 0) {
            const emitted: f64 = @floatFromInt(attempted + accepted);
            return wall_ms / emitted > @as(f64, serial) * (1.0 + DFLASH_GATE_MARGIN);
        }
    }
    const avg = @as(f32, @floatFromInt(accepted)) / @as(f32, @floatFromInt(attempted));
    return avg < dflashBreakEvenAccepted(rows);
}

/// The rounds the gate judges: those after the warmup. The warmup's own wall carries one-time costs (graph builds,
/// first touches of a long context) that would decide a request the steady rounds win.
pub const DflashGateWindow = struct {
    open: bool = false,
    attempted: u64 = 0,
    accepted: u64 = 0,
    wall_ms: f64 = 0,

    pub fn shouldDisable(self: *DflashGateWindow, attempted: u64, accepted: u64, wall_ms: f64, rows: f32, serial_ms_tok: ?f32) bool {
        if (attempted < DFLASH_GATE_WARMUP) return false;
        if (!self.open) {
            self.* = .{ .open = true, .attempted = attempted, .accepted = accepted, .wall_ms = wall_ms };
            return false;
        }
        return dflashGateShouldDisable(attempted - self.attempted, accepted - self.accepted, wall_ms - self.wall_ms, rows, serial_ms_tok);
    }
};

/// The serial price a DFlash request is held to: the plain-tick cell at width 0, which this
/// path's own fallback ticks feed, else the serial row MTP's adaptive serial feeds.
pub fn dflashSerialPrice(t: *const round_cost.Table, bucket: usize) ?f32 {
    return t.msPerTok(0, bucket) orelse t.serialMsPerTok(bucket);
}

test "ladder: priced from the measured forward and the fixed round cost, never cheaper for more rows" {
    const c = ladder();
    try testing.expectEqual(@as(f32, 1), c[1]);
    try testing.expectApproxEqAbs(@as(f32, 1.15), c[2], 0.01);
    try testing.expectApproxEqAbs(@as(f32, 1.37), c[4], 0.01);
    try testing.expectApproxEqAbs(@as(f32, 1.95), c[8], 0.01);
    var r: u32 = 2;
    while (r <= MAX_ROWS) : (r += 1) try testing.expect(c[r] >= c[r - 1]);
}

test "breakEvenDrafts: the yield at which a round of N rows matches a serial step, rising with rows" {
    try testing.expectApproxEqAbs(@as(f32, 0.42), breakEvenDrafts(2), 0.01);
    try testing.expectApproxEqAbs(@as(f32, 0.69), breakEvenDrafts(4), 0.01);
    try testing.expectApproxEqAbs(@as(f32, 1.40), breakEvenDrafts(8), 0.01);
    try testing.expectApproxEqAbs((breakEvenDrafts(4) + breakEvenDrafts(5)) / 2, breakEvenDrafts(4.5), 1e-4);
    // Every row a round verifies must pay for itself: the bar never exceeds the drafts it carries.
    var r: u32 = 2;
    while (r <= MAX_ROWS) : (r += 1) try testing.expect(breakEvenDrafts(@floatFromInt(r)) < @as(f32, @floatFromInt(r - 1)));
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
    // 40% a draft: two drafts pay, the third does not.
    try testing.expectEqual(@as(u32, 3), chooseRows(&[_]f32{ 0.4, 0.4, 0.4 }, &costs).rows);
    try testing.expectEqual(@as(u32, 1), chooseRows(&[_]f32{ 0.15, 0.15, 0.15 }, &costs).rows);
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
    // three drafts, the second rejected: positions 0 and 1 were reached, 2 never. The second
    // observation shares the first's bin and so ages it once.
    c.observeRound(&[_]f32{ 0.9, 0.9, 0.99 }, 3, 1);
    var total: f32 = 0;
    for (c.seen) |v| total += v;
    try testing.expectApproxEqAbs(@as(f32, 1 + Calibration.KEEP), total, 1e-4);
    var hits: f32 = 0;
    for (c.hit) |v| hits += v;
    try testing.expectApproxEqAbs(Calibration.KEEP, hits, 1e-4);
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

test "linearLadder: a first row, then a fixed step per row" {
    const c = linearLadder(35, 8.5);
    try testing.expectApproxEqAbs(@as(f32, 35), c[1], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 43.5), c[2], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 94.5), c[8], 1e-3);
}

test "chooseTreeRows: a chain is chooseRows, a sibling adds only its own odds" {
    const costs = ladder();
    const calib = Calibration{};
    const conf = [_]f32{ 0.95, 0.9, 0.5, 0.2 };
    var p: [conf.len]f32 = undefined;
    for (conf, &p) |c, *out| out.* = calib.probability(c);
    const chain = chooseTreeRows(&calib, &.{ -1, 0, 1, 2 }, &conf, &costs);
    const want = chooseRows(&p, &costs);
    try testing.expectEqual(want.rows, chain.rows);
    try testing.expectApproxEqAbs(want.tokens, chain.tokens, 1e-5);
    // Two siblings under t1: the second row buys the second sibling's odds, not the chain's product.
    const forks = chooseTreeRows(&calib, &.{ -1, -1 }, conf[0..2], &costs);
    try testing.expect(forks.rows >= 1);
    const both = 1 + p[0] + p[1];
    var wide = Costs{ 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    wide[1] = 1;
    try testing.expectApproxEqAbs(both, chooseTreeRows(&calib, &.{ -1, -1 }, conf[0..2], &wide).tokens, 1e-5);
}

test "Calibration.observeTree: only drafts whose parent was accepted are reached" {
    var c = Calibration{};
    const prior_hit = c.probability(0.9);
    const prior_miss = c.probability(0.5);
    // rows: 0 = t1, 1 and 2 under t1, 3 under 1, 4 under 2 (never reached).
    const conf = [_]f32{ 1, 0.9, 0.5, 0.99, 0.2 };
    const parents = [_]i32{ -1, 0, 0, 1, 2 };
    for (0..30) |_| c.observeTree(&conf, &parents, 5, &.{ 0, 1 });
    try testing.expect(c.probability(0.9) > prior_hit);
    try testing.expect(c.probability(0.5) < prior_miss);
    try testing.expectEqual(@as(f32, 0), c.seen[Calibration.bin(0.2)]);
    try testing.expect(c.probability(0.99) < Calibration.prior(0.99));
}
