//! Speculative-decoding totals the metrics endpoints export: per drafting mode, the rounds run, tokens
//! drafted and accepted, rounds that drafted nothing, and (DSpark) how many verifier rows each round
//! carried. They are the sums of the per-request figures `[spec-stats]` logs, added once when a request
//! ends, so the decode loop pays nothing. One `process` for the server; counting does not need `--metrics`.

const std = @import("std");

/// What drafted for a request: DeepSeek's DSpark, the MTP heads (Qwen, MiMo), or DFlash2 (GLM).
pub const Mode = enum { none, dspark, mtp, dflash };
const exported = @typeInfo(Mode).@"enum".field_names[1..];
const mode_count = exported.len;

/// Verifier rows a round can carry: the largest DSpark draft plus the always-committed token, plus a zero bucket.
pub const rows_cap = 18;

/// One request's speculative figures.
pub const Totals = struct {
    /// Rounds that ran a draft and one verify forward.
    rounds: u64 = 0,
    drafted: u64 = 0,
    /// Drafted tokens the verifier kept; the always-committed first token of a round is not counted.
    accepted: u64 = 0,
    /// DSpark: rounds with one verifier row (inside `rounds`). MTP: planner ticks that ran without a draft (outside it).
    plain: u64 = 0,

    pub fn acceptance(self: Totals) f64 {
        if (self.drafted == 0) return 0;
        return @as(f64, @floatFromInt(self.accepted)) / @as(f64, @floatFromInt(self.drafted));
    }
};

/// Drafts a mode proposed: its own count when it keeps one, else every round at the draft cap.
pub fn proposed(counted: u64, rounds: u64, cap: u64) u64 {
    return if (counted > 0) counted else rounds * cap;
}

const Counter = std.atomic.Value(u64);

const ModeStats = struct {
    requests: Counter = Counter.init(0),
    rounds: Counter = Counter.init(0),
    drafted: Counter = Counter.init(0),
    accepted: Counter = Counter.init(0),
    plain: Counter = Counter.init(0),
    rows: [rows_cap]Counter = @splat(Counter.init(0)),
};

pub const ModeSnapshot = struct {
    requests: u64 = 0,
    rounds: u64 = 0,
    drafted: u64 = 0,
    accepted: u64 = 0,
    plain: u64 = 0,
    /// Rounds by verifier rows (DSpark only).
    rows: [rows_cap]u64 = @splat(0),
};

pub const Snapshot = struct {
    /// Indexed by `Mode` minus one (`none` is not exported).
    modes: [mode_count]ModeSnapshot = @splat(.{}),
};

pub const Stats = struct {
    modes: [mode_count]ModeStats = @splat(.{}),

    /// Adds one finished request. `rows[r]` is the number of its rounds that carried `r` verifier rows.
    pub fn record(self: *Stats, mode: Mode, t: Totals, rows: []const u32) void {
        if (mode == .none or t.rounds == 0) return;
        const m = &self.modes[@intFromEnum(mode) - 1];
        _ = m.requests.fetchAdd(1, .monotonic);
        _ = m.rounds.fetchAdd(t.rounds, .monotonic);
        _ = m.drafted.fetchAdd(t.drafted, .monotonic);
        _ = m.accepted.fetchAdd(t.accepted, .monotonic);
        _ = m.plain.fetchAdd(t.plain, .monotonic);
        for (rows[0..@min(rows.len, rows_cap)], 0..) |n, r| if (n > 0) {
            _ = m.rows[r].fetchAdd(n, .monotonic);
        };
    }

    pub fn snapshot(self: *const Stats) Snapshot {
        var s: Snapshot = .{};
        for (&s.modes, &self.modes) |*dst, *src| {
            dst.requests = src.requests.load(.monotonic);
            dst.rounds = src.rounds.load(.monotonic);
            dst.drafted = src.drafted.load(.monotonic);
            dst.accepted = src.accepted.load(.monotonic);
            dst.plain = src.plain.load(.monotonic);
            for (&dst.rows, &src.rows) |*d, *c| d.* = c.load(.monotonic);
        }
        return s;
    }
};

pub var process: Stats = .{};

pub fn writePrometheus(snap: Snapshot, w: *std.Io.Writer) !void {
    inline for (.{
        .{ "sushi:spec_requests_total", "Requests that ran at least one speculative round, by drafting mode", "requests" },
        .{ "sushi:spec_rounds_total", "Speculative rounds (one verify forward each), by drafting mode", "rounds" },
        .{ "sushi:spec_drafted_tokens_total", "Tokens drafted for verification, by drafting mode", "drafted" },
        .{ "sushi:spec_accepted_tokens_total", "Drafted tokens the verifier accepted, by drafting mode", "accepted" },
        .{ "sushi:spec_plain_rounds_total", "Rounds that carried no draft where the engine counts them, by drafting mode", "plain" },
    }) |m| {
        try w.print("# HELP {s} {s}\n# TYPE {s} counter\n", .{ m[0], m[1], m[0] });
        inline for (exported, 0..) |name, i| try w.print("{s}{{mode=\"{s}\"}} {d}\n", .{ m[0], name, @field(snap.modes[i], m[2]) });
        try w.writeAll("\n");
    }
    try w.writeAll("# HELP sushi:spec_rows_total Speculative rounds by verifier rows carried (DSpark)\n# TYPE sushi:spec_rows_total counter\n");
    inline for (exported, 0..) |name, i| {
        for (snap.modes[i].rows, 0..) |n, r| if (n > 0) try w.print("sushi:spec_rows_total{{mode=\"{s}\",rows=\"{d}\"}} {d}\n", .{ name, r, n });
    }
    try w.writeAll("\n");
}

/// `"speculative":{...}` as one member of the `/metrics.json` object; `rows` appears for a mode that reports it.
pub fn writeJson(snap: Snapshot, w: *std.Io.Writer) !void {
    try w.writeAll("\"speculative\":{");
    inline for (exported, 0..) |name, i| {
        const m = snap.modes[i];
        try w.print("{s}\"{s}\":{{\"requests\":{d},\"rounds\":{d},\"drafted\":{d},\"accepted\":{d},\"plain_rounds\":{d}", .{ if (i == 0) "" else ",", name, m.requests, m.rounds, m.drafted, m.accepted, m.plain });
        var first = true;
        for (m.rows, 0..) |n, r| if (n > 0) {
            try w.print("{s}\"{d}\":{d}", .{ if (first) ",\"rows\":{" else ",", r, n });
            first = false;
        };
        if (!first) try w.writeAll("}");
        try w.writeAll("}");
    }
    try w.writeAll("}");
}

const testing = std.testing;

test "Stats: a request lands in its mode, and a request with no rounds is not counted" {
    var s: Stats = .{};
    s.record(.mtp, .{ .rounds = 10, .drafted = 30, .accepted = 21, .plain = 2 }, &.{});
    s.record(.mtp, .{ .rounds = 4, .drafted = 8, .accepted = 5 }, &.{});
    s.record(.dspark, .{ .rounds = 3, .drafted = 12, .accepted = 6, .plain = 1 }, &.{ 0, 0, 1, 2 });
    s.record(.dspark, .{ .rounds = 2, .drafted = 8, .accepted = 7 }, &.{ 0, 0, 0, 2 });
    s.record(.dflash, .{}, &.{});
    s.record(.none, .{ .rounds = 9 }, &.{});
    const snap = s.snapshot();
    const mtp = snap.modes[@intFromEnum(Mode.mtp) - 1];
    try testing.expectEqual(@as(u64, 2), mtp.requests);
    try testing.expectEqual(@as(u64, 14), mtp.rounds);
    try testing.expectEqual(@as(u64, 38), mtp.drafted);
    try testing.expectEqual(@as(u64, 26), mtp.accepted);
    try testing.expectEqual(@as(u64, 2), mtp.plain);
    const dspark = snap.modes[@intFromEnum(Mode.dspark) - 1];
    try testing.expectEqual(@as(u64, 2), dspark.requests);
    try testing.expectEqual(@as(u64, 1), dspark.rows[2]);
    try testing.expectEqual(@as(u64, 4), dspark.rows[3]);
    try testing.expectEqual(@as(u64, 0), snap.modes[@intFromEnum(Mode.dflash) - 1].requests);
}

test "the Prometheus and JSON renderings carry every counter of every mode" {
    var snap: Snapshot = .{};
    for (&snap.modes, 1..) |*m, i| m.* = .{ .requests = i, .rounds = 10 * i, .drafted = 100 * i, .accepted = 1000 * i, .plain = 7 * i };
    snap.modes[@intFromEnum(Mode.dspark) - 1].rows[3] = 42;
    var buf: [4096]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try writePrometheus(snap, &w);
    const prom = w.buffered();
    for ([_][]const u8{
        "sushi:spec_requests_total{mode=\"dspark\"} 1\n",
        "sushi:spec_rounds_total{mode=\"mtp\"} 20\n",
        "sushi:spec_drafted_tokens_total{mode=\"dflash\"} 300\n",
        "sushi:spec_accepted_tokens_total{mode=\"mtp\"} 2000\n",
        "sushi:spec_plain_rounds_total{mode=\"dspark\"} 7\n",
        "sushi:spec_rows_total{mode=\"dspark\",rows=\"3\"} 42\n",
    }) |want| try testing.expect(std.mem.indexOf(u8, prom, want) != null);
    try testing.expect(std.mem.indexOf(u8, prom, "rows=\"2\"") == null);
    var json_buf: [2048]u8 = undefined;
    var jw: std.Io.Writer = .fixed(&json_buf);
    try jw.writeAll("{");
    try writeJson(snap, &jw);
    try jw.writeAll("}");
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, jw.buffered(), .{});
    defer parsed.deinit();
    const spec = parsed.value.object.get("speculative").?.object;
    const mtp = spec.get("mtp").?.object;
    try testing.expectEqual(@as(i64, 2), mtp.get("requests").?.integer);
    try testing.expectEqual(@as(i64, 200), mtp.get("drafted").?.integer);
    try testing.expectEqual(@as(i64, 2000), mtp.get("accepted").?.integer);
    try testing.expectEqual(@as(i64, 14), mtp.get("plain_rounds").?.integer);
    try testing.expect(mtp.get("rows") == null);
    try testing.expectEqual(@as(i64, 42), spec.get("dspark").?.object.get("rows").?.object.get("3").?.integer);
}

test "proposed: the counted drafts when there are any, else the rounds at their cap" {
    try testing.expectEqual(@as(u64, 21), proposed(21, 10, 4));
    try testing.expectEqual(@as(u64, 40), proposed(0, 10, 4));
}

test "Totals: acceptance is accepted over drafted, and zero with nothing drafted" {
    try testing.expectEqual(@as(f64, 0.75), (Totals{ .rounds = 2, .drafted = 8, .accepted = 6 }).acceptance());
    try testing.expectEqual(@as(f64, 0), (Totals{ .rounds = 2 }).acceptance());
}
