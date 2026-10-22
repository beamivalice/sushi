//! The cost test a speculative round must pass: ms per emitted token against a plain serial
//! token. `Gate` applies it per request for the GLM DFlash2 loop, which can run a plain
//! one-row tick without leaving speculation.
const std = @import("std");

/// A round that costs more than this share above a serial token is dropped.
pub const margin: f64 = 0.05;

pub fn roundLoses(round_ms: f64, tokens_per_round: f64, serial_ms: f64) bool {
    return round_ms / tokens_per_round > serial_ms * (1.0 + margin);
}

pub const Action = enum { round, plain };

/// Rounds folded before a verdict, and per trial: fewer is one dry opening.
const window_rounds = 8;
/// Plain ticks that price a serial token; the first follows a round and is not counted.
const probe_ticks = 3;
/// Tokens a losing request decodes plain before its first trial, doubled per failed trial up to the cap.
const retrial_tokens = 64;
const retrial_cap = 8 * retrial_tokens;
/// Tokens between serial re-probes while rounds pay: the price grows with context.
const reprobe_tokens = 512;
/// Weight a round keeps in the sums after the next one lands.
const decay = 0.95;

pub const Gate = struct {
    mode: enum { rounds, probe, plain } = .rounds,
    ms: f64 = 0,
    tokens: f64 = 0,
    window: u32 = 0,
    serial_ms: ?f64 = null,
    probe_seen: u32 = 0,
    probe_sum: f64 = 0,
    since_probe: u32 = 0,
    plain_tokens: u32 = 0,
    retrial: u32 = retrial_tokens,
    trial: bool = false,
    strike: bool = false,

    pub fn next(self: *Gate) Action {
        switch (self.mode) {
            .probe => return .plain,
            .plain => {
                if (self.plain_tokens < self.retrial) return .plain;
                self.mode = .rounds;
                self.trial = true;
                self.window = 0;
                self.ms = 0;
                self.tokens = 0;
                return .round;
            },
            .rounds => {
                if (self.window >= window_rounds and (self.serial_ms == null or self.since_probe >= reprobe_tokens)) {
                    self.mode = .probe;
                    self.probe_seen = 0;
                    self.probe_sum = 0;
                    return .plain;
                }
                return .round;
            },
        }
    }

    /// True while a probe tick runs: it must be a plain tick, not a verbatim-lookup round.
    pub fn probing(self: *const Gate) bool {
        return self.mode == .probe;
    }

    /// One assistant round: wall ms and tokens it emitted (the pending token plus accepted drafts).
    pub fn observeRound(self: *Gate, ms: f64, tokens: u32) void {
        self.ms = self.ms * decay + ms;
        self.tokens = self.tokens * decay + @as(f64, @floatFromInt(tokens));
        self.window += 1;
        self.since_probe += tokens;
        self.verdict();
    }

    /// One plain one-row tick: wall ms for its token.
    pub fn observePlain(self: *Gate, ms: f64) void {
        self.plain_tokens += 1;
        if (self.mode == .probe) {
            self.probe_seen += 1;
            if (self.probe_seen > 1) self.probe_sum += ms;
            if (self.probe_seen < probe_ticks) return;
            self.serial_ms = self.probe_sum / @as(f64, probe_ticks - 1);
            self.since_probe = 0;
            self.mode = .rounds;
            self.verdict();
        } else if (self.mode == .plain) {
            self.serial_ms = if (self.serial_ms) |s| 0.8 * s + 0.2 * ms else ms;
        }
    }

    /// A verbatim-lookup round: priced by neither side, but it moves the clocks.
    pub fn observeLookup(self: *Gate, tokens: u32) void {
        self.plain_tokens += tokens;
        self.since_probe += tokens;
    }

    fn verdict(self: *Gate) void {
        if (self.mode != .rounds or self.window < window_rounds) return;
        const serial = self.serial_ms orelse return;
        if (!roundLoses(self.ms, self.tokens, serial)) {
            self.trial = false;
            self.strike = false;
            self.retrial = retrial_tokens;
            return;
        }
        // One dry window is noise near the break-even; a trial's window is already the second look.
        if (!self.trial and !self.strike) {
            self.strike = true;
            self.window = 0;
            self.ms = 0;
            self.tokens = 0;
            return;
        }
        self.strike = false;
        if (self.trial) self.retrial = @min(self.retrial * 2, retrial_cap);
        self.trial = false;
        self.mode = .plain;
        self.plain_tokens = 0;
    }
};

// Medians of the GLM-5.3-Flash-Sushi-2.4bpw bank (3-row verify, M5 Max): round ms, tokens per
// round (1 + accepted drafts), serial ms per token, same boot.
const prose = .{ .round = 50.4, .tokens = 2.02, .serial = 28.08 };
const novel = .{ .round = 59.9, .tokens = 1.77, .serial = 31.60 };
const count32k = .{ .round = 82.6, .tokens = 2.98, .serial = 40.81 };
// Clearly under the break-even yield (about 0.9 accepted drafts per round there).
const dry = .{ .round = 59.9, .tokens = 1.5, .serial = 31.60 };

test "spec cost gate: a round loses only above the serial price plus the margin" {
    try std.testing.expect(!roundLoses(prose.round, prose.tokens, prose.serial));
    try std.testing.expect(roundLoses(novel.round, novel.tokens, novel.serial));
    try std.testing.expect(!roundLoses(count32k.round, count32k.tokens, count32k.serial));
    try std.testing.expect(!roundLoses(31.0 * 1.05, 1.0, 31.0));
    try std.testing.expect(roundLoses(31.0 * 1.051, 1.0, 31.0));
}

/// Runs `ticks` ticks of a workload; `plain_log` records which ticks were plain.
fn drive(g: *Gate, ticks: usize, w: anytype, plain_log: *std.ArrayList(bool)) !void {
    var emitted: f64 = 0;
    for (0..ticks) |_| {
        const a = g.next();
        try plain_log.append(std.testing.allocator, a == .plain);
        switch (a) {
            .round => {
                const before = @floor(emitted);
                emitted += w.tokens;
                g.observeRound(w.round, @intFromFloat(@floor(emitted) - before));
            },
            .plain => g.observePlain(w.serial),
        }
    }
}

fn longestPlainRun(log: []const bool) usize {
    var best: usize = 0;
    var run: usize = 0;
    for (log) |plain| {
        run = if (plain) run + 1 else 0;
        best = @max(best, run);
    }
    return best;
}

test "spec cost gate: warm-up rounds, three probe ticks, then rounds stay on for a paying text" {
    var g = Gate{};
    var log: std.ArrayList(bool) = .empty;
    defer log.deinit(std.testing.allocator);
    try drive(&g, 11, prose, &log);
    for (log.items[0..8]) |plain| try std.testing.expect(!plain);
    for (log.items[8..11]) |plain| try std.testing.expect(plain);
    log.clearRetainingCapacity();
    try drive(&g, 100, prose, &log);
    try std.testing.expectEqual(@as(usize, 0), longestPlainRun(log.items));
}

test "spec cost gate: the first probe tick is a transition and is not priced" {
    var g = Gate{};
    var log: std.ArrayList(bool) = .empty;
    defer log.deinit(std.testing.allocator);
    try drive(&g, 8, prose, &log);
    try std.testing.expectEqual(Action.plain, g.next());
    try std.testing.expect(g.probing());
    g.observePlain(400.0);
    _ = g.next();
    g.observePlain(28.0);
    _ = g.next();
    g.observePlain(30.0);
    try std.testing.expect(!g.probing());
    try std.testing.expectApproxEqAbs(@as(f64, 29.0), g.serial_ms.?, 1e-9);
    try std.testing.expectEqual(Action.round, g.next());
}

test "spec cost gate: a losing text goes plain, retries after a stretch, and backs off" {
    var g = Gate{};
    var log: std.ArrayList(bool) = .empty;
    defer log.deinit(std.testing.allocator);
    try drive(&g, 11, dry, &log);
    log.clearRetainingCapacity();
    try drive(&g, 400, dry, &log);
    // The first losing window is a strike; the second, a window later, sends the request plain.
    for (log.items[0..8]) |plain| try std.testing.expect(!plain);
    try std.testing.expect(log.items[8]);
    var plain_run: usize = 0;
    var runs: usize = 0;
    var first_run: usize = 0;
    var second_run: usize = 0;
    var prev = false;
    for (log.items) |plain| {
        if (plain) {
            plain_run += 1;
        } else if (prev) {
            runs += 1;
            if (runs == 1) first_run = plain_run else if (runs == 2) second_run = plain_run;
            plain_run = 0;
        }
        prev = plain;
    }
    try std.testing.expect(first_run > 0);
    try std.testing.expect(second_run >= 2 * first_run);
    var rounds: usize = 0;
    for (log.items) |plain| rounds += @intFromBool(!plain);
    try std.testing.expect(rounds < log.items.len / 4);
}

test "spec cost gate: a trial that pays returns to rounds" {
    var g = Gate{};
    var log: std.ArrayList(bool) = .empty;
    defer log.deinit(std.testing.allocator);
    try drive(&g, 11, dry, &log);
    log.clearRetainingCapacity();
    try drive(&g, 2000, prose, &log);
    try std.testing.expect(longestPlainRun(log.items[200..]) <= 3);
}

test "spec cost gate: a dry spell inside a paying text does not flip it" {
    var g = Gate{};
    var log: std.ArrayList(bool) = .empty;
    defer log.deinit(std.testing.allocator);
    try drive(&g, 11 + 40, prose, &log);
    log.clearRetainingCapacity();
    try drive(&g, 3, .{ .round = 53.0, .tokens = 1.0, .serial = 28.08 }, &log);
    try std.testing.expectEqual(@as(usize, 0), longestPlainRun(log.items));
}

test "spec cost gate: lookup rounds are not priced but count toward the stretch" {
    var g = Gate{};
    var log: std.ArrayList(bool) = .empty;
    defer log.deinit(std.testing.allocator);
    try drive(&g, 11 + 8, dry, &log);
    try std.testing.expectEqual(Action.plain, g.next());
    g.observeLookup(1000);
    try std.testing.expectEqual(Action.round, g.next());
}

test "spec cost gate: one losing window among paying ones does not send a request plain" {
    var g = Gate{};
    var log: std.ArrayList(bool) = .empty;
    defer log.deinit(std.testing.allocator);
    try drive(&g, 11 + 40, prose, &log);
    log.clearRetainingCapacity();
    try drive(&g, 8, .{ .round = 53.0, .tokens = 1.0, .serial = 28.08 }, &log);
    try drive(&g, 40, prose, &log);
    try std.testing.expectEqual(@as(usize, 0), longestPlainRun(log.items));
}
