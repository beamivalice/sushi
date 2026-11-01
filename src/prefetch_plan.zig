//! The expert prefetch's read budget, its pick order and the read-rate probe's arithmetic. Plain data, no MLX.
const std = @import("std");

/// The most candidates `rankWanted` takes (rows times ranked candidates of a prefetch job).
pub const candidates_max = 16 * 12;

/// The read budget of the prefetch, in bytes of SSD traffic. A forward's budget is `--ssd-prefetch` percent of the
/// machine's best read rate over the forward's wall time, less the demand fills the forward already read; one round's
/// unspent grant carries to the next. Each job (one layer's prediction) may spend its share of the grant plus what the
/// jobs before it left unspent, so late layers are not starved. A read is a whole expert or nothing.
pub const PrefetchCredit = struct {
    pct: f64,
    max_bps: f64,
    target_layers: f64,
    credit: f64 = 0,
    share: f64 = 0,
    slice: f64 = 0,

    pub fn init(pct: u8, max_bps: f64, target_layers: usize) PrefetchCredit {
        return .{ .pct = @floatFromInt(pct), .max_bps = max_bps, .target_layers = @floatFromInt(@max(target_layers, 1)) };
    }

    /// Every read is granted; tests that look at what the prefetch reads, not at its budget.
    pub const unmetered: PrefetchCredit = .{ .pct = 100, .max_bps = std.math.inf(f64), .target_layers = 1, .credit = std.math.inf(f64), .share = std.math.inf(f64) };

    /// Closes a forward of `round_ns` that read `demand_bytes` for its routes.
    pub fn refill(self: *PrefetchCredit, round_ns: u64, demand_bytes: u64) void {
        const allowed = self.pct / 100 * self.max_bps * @as(f64, @floatFromInt(round_ns)) / 1e9;
        const grant = @max(0, allowed - @as(f64, @floatFromInt(demand_bytes)));
        self.credit = @min(self.credit, grant) + grant;
        self.share = grant / self.target_layers;
        self.slice = 0;
    }

    /// Whether the next job could read `bytes` whole: a prediction is worth making only then.
    pub fn affords(self: *const PrefetchCredit, bytes: u64) bool {
        const b: f64 = @floatFromInt(bytes);
        return self.credit >= b and self.slice + self.share >= b;
    }

    pub fn beginJob(self: *PrefetchCredit) void {
        self.slice += self.share;
    }

    /// Spends `bytes` when the job's slice and the credit both hold them whole.
    pub fn take(self: *PrefetchCredit, bytes: u64) bool {
        const b: f64 = @floatFromInt(bytes);
        if (self.credit < b or self.slice < b) return false;
        self.credit -= b;
        self.slice -= b;
        return true;
    }
};

/// The distinct experts a prefetch job's rows want, best first, into `out`; returns how many. Each row offers the first
/// `m` of its `n` ranked candidates that are not `held` (cached, dropped or out of range). An expert ranks by how many
/// rows offer it, then by the highest predicted pick score any gave it, then by the earliest row and rank.
pub fn rankWanted(cand: []const u16, score: []const f32, held: []const bool, rows: usize, n: usize, m: usize, out: []u16) usize {
    const Entry = struct { expert: u16, rows: u16, score: f32, first: u16 };
    if (rows * n > candidates_max or cand.len != rows * n or score.len != cand.len or held.len != cand.len) return 0;
    var entries: [candidates_max]Entry = undefined;
    var len: usize = 0;
    for (0..rows) |r| {
        var taken: usize = 0;
        for (0..n) |k| {
            const at = r * n + k;
            if (held[at]) continue;
            taken += 1;
            const seen = for (entries[0..len]) |*e| {
                if (e.expert == cand[at]) break e;
            } else null;
            if (seen) |e| {
                e.rows += 1;
                e.score = @max(e.score, score[at]);
            } else {
                entries[len] = .{ .expert = cand[at], .rows = 1, .score = score[at], .first = @intCast(at) };
                len += 1;
            }
            if (taken == m) break;
        }
    }
    const Order = struct {
        fn before(_: void, a: Entry, b: Entry) bool {
            if (a.rows != b.rows) return a.rows > b.rows;
            if (a.score != b.score) return a.score > b.score;
            return a.first < b.first;
        }
    };
    std.sort.pdq(Entry, entries[0..len], {}, Order.before);
    const kept = @min(len, out.len);
    for (entries[0..kept], out[0..kept]) |e, *o| o.* = e.expert;
    return kept;
}

/// Bytes a second, for `bytes` read in `ns`.
pub fn readRate(bytes: u64, ns: u64) f64 {
    if (ns == 0) return 0;
    return @as(f64, @floatFromInt(bytes)) * 1e9 / @as(f64, @floatFromInt(ns));
}

/// The machine's read rate as the probe saw it: its fastest reading.
pub fn probeMax(rates: []const f64) f64 {
    var best: f64 = 0;
    for (rates) |r| best = @max(best, r);
    return best;
}

const testing = std.testing;
const expert: u64 = 1000;

fn creditOf(grant_experts: f64, target_layers: usize) PrefetchCredit {
    // A one second round at 100% of `max_bps`, no demand: the grant is `max_bps` bytes.
    var c = PrefetchCredit.init(100, grant_experts * @as(f64, expert), target_layers);
    c.refill(std.time.ns_per_s, 0);
    return c;
}

test "prefetch credit: a grant is the share of the read rate the demand fills left over" {
    var c = PrefetchCredit.init(65, 10e9, 1);
    c.refill(100 * std.time.ns_per_ms, 0);
    try testing.expectApproxEqAbs(@as(f64, 650e6), c.credit, 1);
    c.refill(100 * std.time.ns_per_ms, 700e6);
    try testing.expectEqual(@as(f64, 0), c.share);
    c.beginJob();
    try testing.expect(!c.take(1));
    c.refill(100 * std.time.ns_per_ms, 900e6);
    try testing.expectEqual(@as(f64, 0), c.share);
}

test "prefetch credit: 1.9 experts of credit read one and carry 0.9, 0.99 reads none" {
    var c = creditOf(1.9, 1);
    c.beginJob();
    try testing.expect(c.take(expert));
    try testing.expect(!c.take(expert));
    try testing.expectApproxEqAbs(@as(f64, 900), c.credit, 1e-6);
    var d = creditOf(0.99, 1);
    d.beginJob();
    try testing.expect(!d.take(expert));
}

test "prefetch credit: the carry is capped at one round's grant" {
    var c = creditOf(1.5, 1);
    for (0..5) |_| c.refill(std.time.ns_per_s, 0);
    try testing.expectApproxEqAbs(@as(f64, 3000), c.credit, 1e-6);
    c.refill(std.time.ns_per_s, 1500);
    try testing.expectApproxEqAbs(@as(f64, 0), c.credit, 1e-6);
}

test "prefetch credit: a job spends its share and what earlier jobs left, not the whole round" {
    var c = creditOf(4, 4);
    c.beginJob();
    try testing.expect(c.take(expert));
    try testing.expect(!c.take(expert));
    c.beginJob();
    try testing.expect(c.take(expert));
    try testing.expect(!c.take(expert));
    var half = creditOf(2, 4);
    half.beginJob();
    try testing.expect(!half.take(expert));
    half.beginJob();
    try testing.expect(half.take(expert));
}

test "prefetch credit: a job is afforded only when its slice and the credit both hold a whole expert" {
    var c = creditOf(4, 4);
    try testing.expect(c.affords(expert));
    try testing.expect(!c.affords(expert + 1));
    c.beginJob();
    _ = c.take(expert);
    try testing.expect(c.affords(expert));
    var starved = creditOf(0.99, 1);
    try testing.expect(!starved.affords(expert));
    var spent = creditOf(4, 4);
    spent.credit = 10;
    try testing.expect(!spent.affords(expert));
}

test "prefetch credit: 0 percent grants nothing" {
    var c = PrefetchCredit.init(0, 10e9, 4);
    c.refill(std.time.ns_per_s, 0);
    c.beginJob();
    try testing.expect(!c.take(1));
}

test "prefetch credit: unmetered grants every read after a refill too" {
    var c = PrefetchCredit.unmetered;
    c.refill(std.time.ns_per_ms, 1 << 30);
    c.beginJob();
    for (0..100) |_| try testing.expect(c.take(1 << 30));
}

test "rank wanted: rows that want an expert, then its predicted score, then the earlier row" {
    // Three rows of three candidates, one offered each. Row 0's first is held, so it offers 2, as row 1 does.
    const cand = [_]u16{ 5, 2, 9, 2, 5, 7, 7, 5, 1 };
    const score = [_]f32{ 3.0, 2.0, 1.0, 2.5, 2.4, 1.0, 1.5, 1.0, 0.5 };
    const held = [_]bool{ true, false, false, false, false, false, false, false, false };
    var out: [9]u16 = undefined;
    // 2 is wanted by rows 0 and 1; 7 by row 2 alone.
    var got = rankWanted(&cand, &score, &held, 3, 3, 1, &out);
    try testing.expectEqualSlices(u16, &.{ 2, 7 }, out[0..got]);
    // Every first candidate offered: three experts of one row each rank by score, the tie by the earlier row.
    const cand2 = [_]u16{ 4, 8, 6 };
    const score2 = [_]f32{ 3.0, 3.5, 3.5 };
    got = rankWanted(&cand2, &score2, &.{ false, false, false }, 3, 1, 1, &out);
    try testing.expectEqualSlices(u16, &.{ 8, 6, 4 }, out[0..got]);
    // A shorter `out` keeps the best.
    got = rankWanted(&cand2, &score2, &.{ false, false, false }, 3, 1, 1, out[0..2]);
    try testing.expectEqualSlices(u16, &.{ 8, 6 }, out[0..got]);
    // m = 2 offers a row's second uncached candidate too.
    got = rankWanted(&cand, &score, &held, 3, 3, 2, &out);
    try testing.expectEqualSlices(u16, &.{ 2, 5, 7, 9 }, out[0..got]);
}

test "probe max is the fastest reading and 0 for none" {
    try testing.expectEqual(@as(f64, 9.4e9), probeMax(&.{ 7.2e9, 9.4e9, 9.1e9 }));
    try testing.expectEqual(@as(f64, 0), probeMax(&.{}));
    try testing.expectEqual(@as(f64, 2e9), readRate(1 << 30, 536870912));
    try testing.expectEqual(@as(f64, 0), readRate(100, 0));
}
