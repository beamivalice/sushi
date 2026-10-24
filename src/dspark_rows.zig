//! DeepSeek-V4.1 DSpark verify width: how many of a drafted block's rows a round verifies, and when a request
//! stops drafting. Pure data, no MLX.
//!
//! The row choice and the confidence calibration are `dflash_policy`'s, the stop switch is `spec_cost_gate`'s.
//! DSpark's own part is the price of a verify, which grows with context and with every added row (each row's routed
//! experts are their own GEMVs), so it is fitted from the rounds the request runs.
const std = @import("std");
const dflash_policy = @import("dflash_policy.zig");
const spec_cost_gate = @import("spec_cost_gate.zig");

const max_rows = dflash_policy.MAX_ROWS;

/// Verify ms: `plain_ms` for the one-row step and `step_ms` for each row added to it (the decode kernels and the routed
/// experts both scale with rows from one row up). Each estimate moves by a bounded share per sample, so a cold
/// expert cache or a stall cannot pose as the machine's speed, and `step_ms` drifts back to its prior while no round
/// is measuring it, so a bad estimate cannot keep every wider verify out of reach.
pub const Cost = struct {
    plain_ms: f32 = 42,
    step_ms: f32 = prior_step,
    draft_ms: f32 = 6,

    /// One added row on the real pack at 1k to 16k of context.
    const prior_step: f32 = 13;
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
        self.step_ms += drift * (prior_step - self.step_ms);
    }

    /// Round ms by rows; the draft is paid from the second row on.
    pub fn costs(self: *const Cost) dflash_policy.Costs {
        var out: dflash_policy.Costs = undefined;
        out[0] = 0;
        out[1] = self.plain_ms;
        for (2..max_rows + 1) |r| out[r] = self.draft_ms + self.plain_ms + self.step_ms * @as(f32, @floatFromInt(r - 1));
        return out;
    }
};

pub const Policy = struct {
    gate: spec_cost_gate.Gate = .{},
    calib: dflash_policy.Calibration = identityCalibration(),
    cost: Cost = .{},
    /// Plain steps owed before the next draft, and the stretch the next dismissed block will owe.
    owed: u32 = 0,
    period: u32 = 1,

    const max_period: u32 = 16;

    /// Whether this tick drafts a block or is a plain one-row step. A draft costs its forward whether or not any
    /// row is kept, so a block the row choice dismissed backs the next drafts off, doubling to a cap.
    pub fn next(self: *Policy) spec_cost_gate.Action {
        self.cost.tick();
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
        return dflash_policy.chooseRows(p[0..n], &costs).rows;
    }

    /// A drafted round that verified `verified` rows and kept `accepted` of their drafts.
    pub fn observeRound(self: *Policy, conf: []const f32, verified: u32, accepted: u32, draft_ms: f32, verify_ms: f32) void {
        var p: [dflash_policy.MAX_DRAFTS]f32 = undefined;
        const n = @min(conf.len, p.len);
        for (conf[0..n], p[0..n]) |c, *o| o.* = sigmoid(c);
        self.calib.observeRound(p[0..n], verified - 1, accepted);
        if (verified == 1) self.cost.observePlain(verify_ms) else self.cost.observeRound(verified, verify_ms, draft_ms);
        self.gate.observeRound(draft_ms + verify_ms, 1 + accepted);
        if (verified == 1) {
            self.owed = self.period;
            self.period = @min(self.period * 2, max_period);
        } else self.period = 1;
    }

    /// A plain one-row step.
    pub fn observePlain(self: *Policy, ms: f32) void {
        self.cost.observePlain(ms);
        self.gate.observePlain(ms);
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
    try testing.expectApproxEqAbs(@as(f32, 30), costs[1], 1);
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

test "dspark rows: a dismissed block backs the next drafts off, doubling to a cap, and a paying block clears it" {
    var p = Policy{};
    var owed_seen: [6]u32 = undefined;
    for (&owed_seen) |*o| {
        while (p.next() == .plain) p.observePlain(40);
        p.observeRound(&.{ -3, -3, -3, -3, -3 }, 1, 0, 6, 40);
        o.* = p.owed;
    }
    try testing.expectEqualSlices(u32, &.{ 1, 2, 4, 8, 16, 16 }, &owed_seen);
    while (p.next() == .plain) p.observePlain(40);
    p.observeRound(&.{ 4, 4, 4, 4, 4 }, 6, 5, 6, 108);
    try testing.expectEqual(@as(u32, 1), p.period);
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
