//! DSpark verify width: how many of a drafted block's rows a round verifies, and when a request stops drafting. Pure
//! data, no MLX; the generator's DSpark state is typed by it, the rounds themselves are the drafting arch's.
//!
//! The row choice and the confidence calibration are `dflash_policy`'s, the stop switch is `spec_cost_gate`'s.
//! DSpark's own part is the price of a verify, which grows with context and with every added row (each row's routed
//! experts are their own GEMVs), so it is fitted from the rounds the request runs.
const std = @import("std");
const dflash_policy = @import("dflash_policy.zig");
const spec_cost_gate = @import("spec_cost_gate.zig");
const depth_bounds = @import("mtp_depth_bounds.zig");
const mtp_plan = @import("mtp_plan.zig");
const sampler = @import("sampler.zig");

const max_rows = dflash_policy.MAX_ROWS;

/// Drafts a sampled DSpark round can verify, with room past a stage block.
pub const max_drafts = 16;

/// Forces the verify width of a request's rounds (the arch's diagnostic switch, and the tests).
pub const Diag = union(enum) { off, log, sweep, rows: u8 };

/// How a forced commit leaves a DSpark request.
pub const Resume = union(enum) {
    /// The slot decodes plain: not a DSpark slot, DSpark already off, or a grammar owns the ticks.
    refused,
    stopped,
    /// The closer `commitForcedTokens` left pending, published now: DSpark rounds follow.
    resumed: u32,
};

/// DSpark drafts argmax tokens, a point mass the verify rows accept exactly at any temperature. Penalties shift the
/// rows as in an MTP round, and a thinking bound fires between rounds (`thinkBoundTick`), after which the request
/// decodes serially.
pub fn eligible(sampling: sampler.SamplingParams, logprobs_n: u32) bool {
    return !mtp_plan.mtpUnsupported(sampling, logprobs_n);
}

/// Verify ms: `plain_ms` for the one-row step and `step_ms` for each row added to it (the decode kernels and the routed
/// experts both scale with rows from one row up). Each estimate moves by a bounded share per sample, so a cold
/// expert cache or a stall cannot pose as the machine's speed, and `step_ms` drifts back to its prior while no round
/// is measuring it, so a bad estimate cannot keep every wider verify out of reach.
pub const Cost = struct {
    plain_ms: f32 = 28,
    step_ms: f32 = prior_step,
    draft_ms: f32 = 6,
    /// The step `tick` drifts back to.
    prior: f32 = prior_step,

    /// One added row on the real pack at 1k to 16k of context.
    const prior_step: f32 = 9;
    const ema: f32 = 0.85;
    const drift: f32 = 0.03;

    fn bounded(old: f32, sample: f32) f32 {
        return ema * old + (1 - ema) * std.math.clamp(sample, old * 0.7, old * 1.5);
    }

    /// A one-row step's wall ms.
    pub fn observePlain(self: *Cost, ms: f32) void {
        if (!std.math.isFinite(ms) or ms <= 0) return;
        self.plain_ms = bounded(self.plain_ms, ms);
    }

    /// A drafted round that verified `rows` rows in `verify_ms` after a draft of `draft_ms`.
    pub fn observeRound(self: *Cost, rows: u32, verify_ms: f32, draft_ms: f32) void {
        if (std.math.isFinite(draft_ms) and draft_ms > 0) self.draft_ms = bounded(self.draft_ms, draft_ms);
        if (rows < 2 or !std.math.isFinite(verify_ms) or verify_ms <= 0) return;
        const step = (verify_ms - self.plain_ms) / @as(f32, @floatFromInt(rows - 1));
        self.step_ms = std.math.clamp(bounded(self.step_ms, @max(step, 1)), 3, 40);
    }

    /// Any tick: the step moves back toward its prior.
    pub fn tick(self: *Cost) void {
        self.step_ms += drift * (self.prior - self.step_ms);
    }

    /// Round ms by rows of a block that is already drafted: the draft is spent whatever the row choice, so the
    /// one-row round carries it too.
    pub fn costs(self: *const Cost) dflash_policy.Costs {
        var out: dflash_policy.Costs = undefined;
        out[0] = 0;
        out[1] = self.plain_ms + self.draft_ms;
        for (2..max_rows + 1) |r| out[r] = self.draft_ms + self.plain_ms + self.step_ms * @as(f32, @floatFromInt(r - 1));
        return out;
    }
};

/// The verify widths `--mtp-min-depth` / `--mtp-max-depth` leave a round: depth is the drafts a round verifies, so a
/// width is depth + 1. A depth past the checkpoint's block clamps to it (`clamped`), never an error.
pub const Range = struct {
    min_rows: u32 = 1,
    max_rows: u32,
    clamped: bool = false,

    pub fn of(b: depth_bounds.Bounds, block_drafts: u32) Range {
        const cap = if (b.max == 0) block_drafts else @min(b.max, block_drafts);
        if (b.pinned()) |d| {
            const rows = @min(d, block_drafts) + 1;
            return .{ .min_rows = rows, .max_rows = rows, .clamped = d > block_drafts };
        }
        const floor: u32 = if (b.min <= 1) 0 else @min(b.min, cap);
        return .{ .min_rows = floor + 1, .max_rows = cap + 1, .clamped = (b.max > block_drafts) or (b.min > block_drafts) };
    }

    /// A floor above one row: every drafted round verifies at least that, so no plain step is ever scheduled.
    fn floored(self: Range) bool {
        return self.min_rows > 1;
    }
};

pub const Policy = struct {
    gate: spec_cost_gate.Gate = .{},
    calib: dflash_policy.Calibration = identityCalibration(),
    cost: Cost = .{},
    /// Plain steps owed before the next draft, and the stretch the next losing stretch of drafts will owe.
    owed: u32 = 0,
    period: u32 = 1,
    /// The stretch's second plain step, until its third prices the serial step.
    first_fresh: f32 = 0,
    /// Decayed ms a drafted round saved against the plain steps for its tokens; negative while drafts lose.
    edge: f32 = edge_start,
    range: Range = .{ .max_rows = default_block + 1 },

    const max_period: u32 = 16;
    const edge_start: f32 = 10;
    const edge_keep: f32 = 0.8;
    /// The drafts a stage block holds when a caller does not say.
    const default_block: u32 = 5;

    pub fn init(bounds: depth_bounds.Bounds, block_drafts: u32) Policy {
        return .{ .range = Range.of(bounds, block_drafts) };
    }

    /// Whether this tick drafts a block or is a plain one-row step. A draft costs its forward whether or not any
    /// row is kept, so drafts that lose against plain steps back the next ones off, doubling to a cap. One
    /// dismissed block does not: blocks are bimodal, and the next one pays whatever the plain steps in between.
    pub fn next(self: *Policy) spec_cost_gate.Action {
        self.cost.tick();
        if (self.range.floored()) return .round;
        const action = self.gate.next();
        if (action == .round and self.owed > 0) {
            self.owed -= 1;
            return .plain;
        }
        return action;
    }

    /// Rows (t1 included) to verify of a block whose drafts carry confidence logits `conf`.
    pub fn rows(self: *const Policy, conf: []const f32) u32 {
        var p: [dflash_policy.MAX_DRAFTS]f32 = undefined;
        const n = @min(conf.len, p.len);
        for (conf[0..n], p[0..n]) |c, *o| o.* = self.calib.probability(sigmoid(c));
        const costs = self.cost.costs();
        const chosen = dflash_policy.chooseRows(p[0..@min(n, self.range.max_rows - 1)], &costs).rows;
        return std.math.clamp(chosen, self.range.min_rows, self.range.max_rows);
    }

    /// A drafted round that verified `verified` rows and kept `accepted` of their drafts.
    pub fn observeRound(self: *Policy, conf: []const f32, verified: u32, accepted: u32, draft_ms: f32, verify_ms: f32) void {
        var p: [dflash_policy.MAX_DRAFTS]f32 = undefined;
        const n = @min(conf.len, p.len);
        for (conf[0..n], p[0..n]) |c, *o| o.* = sigmoid(c);
        self.calib.observeRound(p[0..n], verified - 1, accepted);
        if (verified == 1) self.cost.observePlain(verify_ms) else self.cost.observeRound(verified, verify_ms, draft_ms);
        self.gate.observeRound(draft_ms + verify_ms, 1 + accepted);
        const saved = @as(f32, @floatFromInt(1 + accepted)) * self.cost.plain_ms - (draft_ms + verify_ms);
        if (std.math.isFinite(saved)) self.edge = edge_keep * self.edge + (1 - edge_keep) * saved;
        if (self.edge < 0) {
            self.owed = self.period;
            self.period = @min(self.period * 2, max_period);
        } else self.period = 1;
    }

    /// A plain one-row step.
    pub fn observePlain(self: *Policy, ms: f32) void {
        self.cost.observePlain(ms);
        self.gate.observePlain(ms);
        self.reprice(ms);
    }

    /// A trip into the retrial stretch measures the plain step afresh: the probe that priced it was taken once, early,
    /// and goes stale within a request (the experts of a new text load, the whole forward slows). The stretch's own
    /// second and third plain steps (the first follows a round, as in the probe) become the serial price; if the window
    /// that tripped pays against it, the stretch ends there and a trial of rounds follows, as it does after the full one.
    fn reprice(self: *Policy, ms: f32) void {
        if (self.gate.mode != .plain) return;
        switch (self.gate.plain_tokens) {
            2 => self.first_fresh = ms,
            3 => {
                const fresh: f64 = (self.first_fresh + ms) / 2;
                self.gate.serial_ms = fresh;
                if (!spec_cost_gate.roundLoses(self.gate.ms, self.gate.tokens, fresh)) self.gate.plain_tokens = self.gate.retrial;
            },
            else => {},
        }
    }
};

/// The confidence head's logit is a calibrated probability (the rounds measured bins of it landing within a few
/// points), so each bin starts as that many pseudo-rounds at its centre instead of the selector's fitted prior.
fn identityCalibration() dflash_policy.Calibration {
    var c: dflash_policy.Calibration = .{};
    const weight = 30;
    for (&c.hit, &c.seen, 0..) |*hit, *seen, b| {
        const axis = (@as(f32, @floatFromInt(b)) + 0.5) / 4;
        seen.* = weight;
        hit.* = weight * (1 - std.math.pow(f32, 10, -axis));
    }
    return c;
}

fn sigmoid(x: f32) f32 {
    return 1 / (1 + @exp(-x));
}

const testing = std.testing;

test "dspark rows: sure drafts buy the whole block, doubtful ones none" {
    const p = Policy{};
    try testing.expectEqual(@as(u32, 6), p.rows(&.{ 5, 5, 5, 5, 5 }));
    try testing.expectEqual(@as(u32, 1), p.rows(&.{ -2, -2, -2, -2, -2 }));
    try testing.expectEqual(@as(u32, 1), p.rows(&.{}));
}

test "dspark rows: a doubt mid-block cuts the verify there" {
    const p = Policy{};
    try testing.expectEqual(@as(u32, 3), p.rows(&.{ 5, 5, -3, 5, 5 }));
}

test "dspark rows: a dearer row verifies fewer drafts" {
    var cheap = Policy{};
    var dear = Policy{};
    for (0..40) |_| cheap.cost.observeRound(4, 60, 6);
    for (0..40) |_| dear.cost.observeRound(4, 300, 6);
    const conf = [_]f32{ 1.2, 1.2, 1.2, 1.2, 1.2 };
    try testing.expect(cheap.rows(&conf) > dear.rows(&conf));
}

test "dspark rows: the cost follows the rounds, a row at a time" {
    var c = Cost{};
    for (0..200) |i| {
        c.observePlain(30);
        c.observeRound(2 + @as(u32, @intCast(i % 5)), 30 + 10 * @as(f32, @floatFromInt(1 + i % 5)), 6);
    }
    const costs = c.costs();
    try testing.expectApproxEqAbs(@as(f32, 30 + 6), costs[1], 1);
    try testing.expectApproxEqAbs(@as(f32, 6 + 30 + 30), costs[4], 3);
}

test "dspark rows: a cold step moves the cost by a bounded share, and an unmeasured step drifts back" {
    var c = Cost{};
    c.observePlain(2000);
    c.observeRound(6, 2400, 900);
    try testing.expect(c.plain_ms < 42 * 1.5 and c.draft_ms < 6 * 1.5 and c.step_ms < 13 * 1.5);
    var p = Policy{};
    for (0..40) |_| p.cost.observeRound(3, 400, 6);
    const high = p.cost.step_ms;
    for (0..200) |_| _ = p.next();
    try testing.expect(p.cost.step_ms < high and p.cost.step_ms < 16);
}

test "dspark rows: the confidence is a probability until the rounds say otherwise" {
    var p = Policy{};
    try testing.expectApproxEqAbs(@as(f32, 0.88), p.calib.probability(0.88), 0.04);
    for (0..60) |_| p.calib.observe(0.88, false);
    try testing.expect(p.calib.probability(0.88) < 0.3);
}

test "dspark rows: drafts that keep losing back the next ones off, doubling to a cap, and a paying block clears it" {
    var p = Policy{};
    var owed_seen: [12]u32 = undefined;
    for (&owed_seen) |*o| {
        while (p.next() == .plain) p.observePlain(40);
        p.observeRound(&.{ -3, -3, -3, -3, -3 }, 1, 0, 6, 40);
        o.* = p.owed;
    }
    try testing.expectEqualSlices(u32, &.{ 0, 0, 1, 2, 4, 8, 16, 16, 16, 16, 16, 16 }, &owed_seen);
    var cleared = false;
    for (0..8) |_| {
        while (p.next() == .plain) p.observePlain(40);
        p.observeRound(&.{ 4, 4, 4, 4, 4 }, 6, 5, 6, 108);
        cleared = cleared or p.period == 1;
    }
    try testing.expect(cleared);
}

test "dspark rows: one dismissed block among paying ones owes no plain step" {
    var p = Policy{};
    for (0..30) |i| {
        while (p.next() == .plain) p.observePlain(25);
        if (i % 4 == 3) p.observeRound(&.{ -3, -3, -3, -3, -3 }, 1, 0, 5.5, 25) else p.observeRound(&.{ 2, 2, 2, 2, 2 }, 4, 3, 5.5, 53.5);
        try testing.expectEqual(@as(u32, 0), p.owed);
    }
}

test "dspark rows: a request whose drafts never pay is sent to plain steps, and a paying one is not" {
    var dry = Policy{};
    var plain_after: ?usize = null;
    for (0..200) |i| {
        switch (dry.next()) {
            .plain => {
                dry.observePlain(50);
                plain_after = plain_after orelse i;
            },
            .round => dry.observeRound(&.{ -1, -1, -1, -1, -1 }, 3, 0, 6, 95),
        }
    }
    try testing.expect(plain_after != null);

    var paying = Policy{};
    var plain_ticks: usize = 0;
    for (0..200) |_| {
        switch (paying.next()) {
            .plain => {
                paying.observePlain(50);
                plain_ticks += 1;
            },
            .round => paying.observeRound(&.{ 4, 4, 4, 4, 4 }, 6, 5, 6, 110),
        }
    }
    // Three probe ticks at the start and one set per 512 tokens are all a paying request spends on plain steps.
    try testing.expect(plain_ticks <= 12);
}

// Verify ms by forced rows on the shipped STQ pack after the routed chain is queued from the GPU cache map
// (median of a forced-width sweep run), and the draft ms of the same run.
const verify_table = [_]f32{ 0, 24.9, 36.4, 44.0, 53.5, 61.9, 68.2, 76.0, 84.9 };
const table_draft_ms: f32 = 5.5;

fn warm(p: *Policy) void {
    for (0..20) |_| {
        p.observePlain(verify_table[1]);
        for (2..7) |r| p.cost.observeRound(@intCast(r), verify_table[r], table_draft_ms);
    }
}

fn logit(a: f32) f32 {
    return @log(a / (1 - a));
}

test "dspark rows: at the cheap verify costs a block with 80-88% drafts verifies three rows or more" {
    for ([_]f32{ 0.8, 0.88 }) |a| {
        var p = Policy{};
        warm(&p);
        const l = logit(a);
        try testing.expect(p.rows(&.{ l, l, l, l, l }) >= 3);
    }
}

const Sim = struct {
    rng: std.Random.DefaultPrng = std.Random.DefaultPrng.init(7),
    tokens: u64 = 0,
    ms: f64 = 0,
    plain_ticks: u64 = 0,

    /// `paying` of the blocks carry confidence `hi` and accept each draft with probability `hi_p`; the rest are
    /// dismissed ones (`lo` confidence, nothing accepted).
    fn run(self: *Sim, p: *Policy, paying: f32, hi: f32, lo: f32, budget: u64) void {
        const r = self.rng.random();
        while (self.tokens < budget) {
            if (p.next() == .plain) {
                p.observePlain(verify_table[1]);
                self.tokens += 1;
                self.ms += verify_table[1];
                self.plain_ticks += 1;
                continue;
            }
            const good = r.float(f32) < paying;
            const conf: [5]f32 = @splat(if (good) logit(hi) else lo);
            const rows = p.rows(&conf);
            var accepted: u32 = 0;
            if (good) while (accepted + 1 < rows and r.float(f32) < hi) : (accepted += 1) {};
            const verify = verify_table[@min(rows, 8)];
            p.observeRound(&conf, rows, accepted, table_draft_ms, verify);
            self.tokens += 1 + accepted;
            self.ms += table_draft_ms + verify;
            if (rows == 1) self.plain_ticks += 1;
        }
    }

    fn tokPerS(self: Sim) f64 {
        return @as(f64, @floatFromInt(self.tokens)) / self.ms * 1000;
    }
};

test "dspark rows: mostly paying blocks with a dismissed one now and then keep drafting, as fast as the cheap verify allows" {
    var cur = Policy{};
    var a = Sim{};
    a.run(&cur, 0.75, 0.88, -3, 3000);
    try testing.expect(a.plain_ticks * 8 < a.tokens);
    try testing.expect(a.tokPerS() > 1.15 * 1000 / verify_table[1]);
}

test "dspark rows: novel text, where drafts are worth what their rows cost, stays on plain steps" {
    var p = Policy{};
    var s = Sim{};
    s.run(&p, 1.0, 0.3, -3, 3000);
    try testing.expect(s.plain_ticks * 2 > s.tokens);
    try testing.expect(s.tokPerS() >= 0.97 * 1000 / verify_table[1]);
}

/// One logged round of a greedy request (`SUSHI_DSV41_DSPARK_ROWS=log`): rows verified, drafts kept, draft ms, verify ms.
/// A draft ms of 0 is a plain step.
const Logged = struct { u32, u32, f32, f32 };

/// The first 93 rounds of the first greedy request on a fresh prompt, after the last of which the gate went plain: its
/// serial probe (three plain steps at rounds 8 to 10) read 18 ms, the verifies then ran 10 to 25 percent above the same
/// text's next run (experts not yet cached), and the rounds still paid 16 to 20 ms a token against a plain step's 24.
const fresh_prompt = [_]Logged{
    .{ 3, 1, 4.8, 33.6 }, .{ 2, 0, 4.7, 25.7 }, .{ 2, 0, 4.6, 25.9 }, .{ 3, 0, 4.6, 31.9 },
    .{ 3, 2, 4.6, 33.2 }, .{ 4, 0, 4.7, 39.9 }, .{ 3, 2, 4.6, 30.8 }, .{ 3, 2, 4.7, 32.5 },
    .{ 1, 0, 0, 17.7 }, .{ 1, 0, 0, 18.8 }, .{ 1, 0, 0, 17.6 }, .{ 3, 2, 4.5, 31.4 },
    .{ 3, 1, 4.8, 32.3 }, .{ 3, 0, 4.8, 34.0 }, .{ 2, 0, 4.8, 23.1 }, .{ 3, 2, 5.0, 39.4 },
    .{ 4, 3, 4.7, 56.2 }, .{ 2, 0, 4.8, 25.0 }, .{ 2, 1, 4.8, 24.7 }, .{ 2, 1, 4.8, 26.7 },
    .{ 3, 2, 4.7, 37.7 }, .{ 3, 2, 4.8, 31.4 }, .{ 3, 1, 4.9, 32.4 }, .{ 2, 0, 4.7, 23.3 },
    .{ 2, 0, 4.8, 26.5 }, .{ 2, 1, 4.8, 27.0 }, .{ 3, 1, 4.7, 37.6 }, .{ 2, 0, 4.7, 33.2 },
    .{ 2, 0, 4.7, 25.9 }, .{ 3, 2, 4.8, 31.8 }, .{ 3, 0, 5.0, 30.5 }, .{ 2, 1, 4.8, 25.1 },
    .{ 3, 2, 5.1, 33.9 }, .{ 2, 1, 4.7, 29.8 }, .{ 5, 4, 5.0, 67.1 }, .{ 3, 1, 4.8, 36.6 },
    .{ 5, 4, 4.8, 52.1 }, .{ 3, 1, 4.9, 33.0 }, .{ 3, 2, 5.0, 32.8 }, .{ 4, 1, 5.0, 48.8 },
    .{ 3, 2, 5.0, 38.7 }, .{ 3, 2, 5.1, 35.4 }, .{ 4, 3, 5.0, 58.3 }, .{ 4, 2, 4.8, 47.7 },
    .{ 5, 1, 4.7, 55.1 }, .{ 6, 5, 5.1, 74.6 }, .{ 6, 5, 5.0, 64.0 }, .{ 3, 0, 4.8, 29.1 },
    .{ 3, 2, 5.0, 35.2 }, .{ 6, 0, 5.2, 66.9 }, .{ 6, 5, 4.8, 63.6 }, .{ 5, 4, 5.0, 48.5 },
    .{ 2, 1, 4.9, 27.1 }, .{ 2, 1, 5.2, 33.7 }, .{ 2, 0, 4.9, 26.4 }, .{ 2, 1, 5.1, 27.2 },
    .{ 2, 0, 4.9, 24.1 }, .{ 2, 0, 4.8, 24.6 }, .{ 2, 1, 5.1, 30.6 }, .{ 2, 0, 4.8, 24.2 },
    .{ 2, 0, 5.1, 26.4 }, .{ 3, 2, 4.9, 36.1 }, .{ 6, 5, 4.9, 59.0 }, .{ 3, 0, 5.1, 34.2 },
    .{ 6, 5, 5.0, 57.5 }, .{ 3, 2, 5.1, 39.9 }, .{ 3, 2, 5.0, 39.4 }, .{ 2, 0, 4.9, 30.1 },
    .{ 2, 1, 5.0, 27.2 }, .{ 3, 1, 5.0, 33.3 }, .{ 2, 1, 5.0, 24.8 }, .{ 2, 0, 5.0, 23.8 },
    .{ 2, 1, 4.8, 24.9 }, .{ 2, 0, 5.0, 27.0 }, .{ 2, 0, 5.0, 30.6 }, .{ 2, 0, 4.8, 23.9 },
    .{ 3, 1, 5.2, 35.0 }, .{ 3, 0, 5.1, 32.2 }, .{ 3, 2, 5.0, 37.4 }, .{ 2, 1, 5.0, 26.3 },
    .{ 2, 0, 4.7, 30.1 }, .{ 3, 2, 5.0, 39.3 }, .{ 4, 0, 5.0, 75.0 }, .{ 1, 0, 0, 19.4 },
    .{ 4, 3, 4.7, 48.6 }, .{ 4, 0, 4.8, 57.6 }, .{ 1, 0, 0, 23.4 }, .{ 2, 1, 4.6, 29.1 },
    .{ 3, 2, 4.8, 45.4 }, .{ 3, 2, 4.8, 35.3 }, .{ 2, 1, 4.8, 31.1 }, .{ 2, 0, 4.9, 25.8 }, .{ 2, 0, 4.7, 25.7 },
};

test "dspark rows: a stretch of plain steps ends after the three that price the step afresh when the tripping window pays against them" {
    var p = Policy{};
    const conf: [5]f32 = @splat(0);
    for (fresh_prompt) |r| {
        const want: spec_cost_gate.Action = if (r[2] == 0) .plain else .round;
        try testing.expectEqual(want, p.next());
        if (r[2] == 0) p.observePlain(r[3]) else p.observeRound(&conf, r[0], r[1], r[2], r[3]);
    }
    // The gate trips here on its early probe's 18 ms; the stretch's own plain steps (the log's next three) price the
    // step at 22.5 ms, which the 20 ms a token the window paid beats, so the stretch ends after them.
    for ([_]f32{ 18.5, 20.7, 24.3 }) |ms| {
        try testing.expectEqual(spec_cost_gate.Action.plain, p.next());
        p.observePlain(ms);
    }
    try testing.expectEqual(spec_cost_gate.Action.round, p.next());
}

test "dspark rows: rounds dearer per token than the plain step it measures stop drafting for the whole stretch" {
    var p = Policy{};
    const conf: [5]f32 = @splat(0);
    // Rounds of 35.5 ms for one or two tokens (21.3 ms a token) against a measured plain step of 18.5 ms.
    var n: usize = 0;
    while (p.gate.mode != .plain) : (n += 1) {
        try testing.expect(n < 40);
        switch (p.next()) {
            .plain => p.observePlain(18.5),
            .round => p.observeRound(&conf, 3, if (n % 3 == 0) 0 else 1, 5, 30.5),
        }
    }
    var plain: usize = 0;
    while (p.next() == .plain) : (plain += 1) p.observePlain(18.5);
    try testing.expect(plain >= 60);
}

test "dspark rows: the depth flags give a verify width of depth + 1, clamped to the block" {
    const r = Range.of;
    try testing.expectEqual(Range{ .min_rows = 1, .max_rows = 6 }, r(.{}, 5));
    try testing.expectEqual(Range{ .min_rows = 1, .max_rows = 4 }, r(.{ .max = 3 }, 5));
    try testing.expectEqual(Range{ .min_rows = 1, .max_rows = 6, .clamped = true }, r(.{ .max = 8 }, 5));
    try testing.expectEqual(Range{ .min_rows = 3, .max_rows = 6 }, r(.{ .min = 2 }, 5));
    try testing.expectEqual(Range{ .min_rows = 6, .max_rows = 6, .clamped = true }, r(.{ .min = 8 }, 5));
    try testing.expectEqual(Range{ .min_rows = 3, .max_rows = 6 }, r(.{ .min = 2, .max = 5 }, 5));
    try testing.expectEqual(Range{ .min_rows = 3, .max_rows = 3 }, r(.{ .min = 2, .max = 2 }, 5));
    try testing.expectEqual(Range{ .min_rows = 2, .max_rows = 2 }, r(.{ .min = 1, .max = 1 }, 5));
    try testing.expectEqual(Range{ .min_rows = 6, .max_rows = 6, .clamped = true }, r(.{ .min = 7, .max = 7 }, 5));
}

test "dspark rows: a max depth caps the verify width, a min depth floors it, equal values pin it" {
    const sure = [_]f32{ 5, 5, 5, 5, 5 };
    const doubtful = [_]f32{ -3, -3, -3, -3, -3 };
    const capped = Policy.init(.{ .max = 2 }, 5);
    try testing.expectEqual(@as(u32, 3), capped.rows(&sure));
    try testing.expectEqual(@as(u32, 1), capped.rows(&doubtful));
    const floored = Policy.init(.{ .min = 3 }, 5);
    try testing.expectEqual(@as(u32, 6), floored.rows(&sure));
    try testing.expectEqual(@as(u32, 4), floored.rows(&doubtful));
    const pinned = Policy.init(.{ .min = 2, .max = 2 }, 5);
    try testing.expectEqual(@as(u32, 3), pinned.rows(&sure));
    try testing.expectEqual(@as(u32, 3), pinned.rows(&doubtful));
}

test "dspark rows: a floored width never schedules a plain step, however badly the drafts lose" {
    var p = Policy.init(.{ .min = 2 }, 5);
    for (0..200) |_| {
        try testing.expectEqual(spec_cost_gate.Action.round, p.next());
        p.observeRound(&.{ -3, -3, -3, -3, -3 }, 3, 0, 5.5, 44);
    }
    var free = Policy{};
    var plain: usize = 0;
    for (0..200) |_| {
        switch (free.next()) {
            .plain => {
                free.observePlain(25);
                plain += 1;
            },
            .round => free.observeRound(&.{ -3, -3, -3, -3, -3 }, 1, 0, 5.5, 25),
        }
    }
    try testing.expect(plain > 100);
}
