//! Prefix-cache counters the metrics endpoints export beyond the request totals: which tier served a
//! hit, what each tier evicted, and the hot (RAM) tier's resident bytes. One `process` for the whole
//! server, written by the inference thread with relaxed atomics; counting does not depend on `--metrics`.

const std = @import("std");

/// Memory (the RAM tier and the live state a request keeps) or the SSD tier.
pub const Tier = enum { hot, ssd };
const tier_names = @typeInfo(Tier).@"enum".field_names;
const tier_count = tier_names.len;

pub const Snapshot = struct {
    /// Requests whose restored prefix came from the tier.
    hits: [tier_count]u64 = @splat(0),
    /// Entries a tier dropped for room: its budget, its entry cap, or a prefill being admitted.
    evictions: [tier_count]u64 = @splat(0),
    evicted_bytes: [tier_count]u64 = @splat(0),
    /// Bytes the hot tier of every ready model holds now.
    hot_bytes: u64 = 0,
};

pub const Stats = struct {
    hits: [tier_count]std.atomic.Value(u64) = @splat(std.atomic.Value(u64).init(0)),
    evictions: [tier_count]std.atomic.Value(u64) = @splat(std.atomic.Value(u64).init(0)),
    evicted_bytes: [tier_count]std.atomic.Value(u64) = @splat(std.atomic.Value(u64).init(0)),
    hot_bytes: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),

    pub fn recordHit(self: *Stats, tier: Tier) void {
        _ = self.hits[@intFromEnum(tier)].fetchAdd(1, .monotonic);
    }

    pub fn recordEviction(self: *Stats, tier: Tier, bytes: u64) void {
        _ = self.evictions[@intFromEnum(tier)].fetchAdd(1, .monotonic);
        _ = self.evicted_bytes[@intFromEnum(tier)].fetchAdd(bytes, .monotonic);
    }

    pub fn setHotBytes(self: *Stats, bytes: u64) void {
        self.hot_bytes.store(bytes, .monotonic);
    }

    pub fn snapshot(self: *const Stats) Snapshot {
        var s: Snapshot = .{ .hot_bytes = self.hot_bytes.load(.monotonic) };
        for (0..tier_count) |i| {
            s.hits[i] = self.hits[i].load(.monotonic);
            s.evictions[i] = self.evictions[i].load(.monotonic);
            s.evicted_bytes[i] = self.evicted_bytes[i].load(.monotonic);
        }
        return s;
    }
};

pub var process: Stats = .{};

pub fn writePrometheus(snap: Snapshot, w: *std.Io.Writer) !void {
    inline for (.{
        .{ "sushi:prefix_cache_tier_hits_total", "Requests whose restored prompt prefix came from the tier (hot = memory or the live state)", "hits" },
        .{ "sushi:prefix_cache_evictions_total", "Entries the tier dropped for room (its budget, its entry cap, a prefill being admitted)", "evictions" },
        .{ "sushi:prefix_cache_evicted_bytes_total", "Bytes of the entries the tier dropped for room", "evicted_bytes" },
    }) |m| {
        try w.print("# HELP {s} {s}\n# TYPE {s} counter\n", .{ m[0], m[1], m[0] });
        inline for (tier_names, 0..) |name, i| try w.print("{s}{{tier=\"{s}\"}} {d}\n", .{ m[0], name, @field(snap, m[2])[i] });
        try w.writeAll("\n");
    }
    try w.print("# HELP sushi:prefix_cache_hot_bytes Bytes the in-memory prefix caches of every ready model hold now\n# TYPE sushi:prefix_cache_hot_bytes gauge\nsushi:prefix_cache_hot_bytes {d}\n\n", .{snap.hot_bytes});
}

/// `"prefix_cache_tiers":{...}` as one member of the `/metrics.json` object.
pub fn writeJson(snap: Snapshot, w: *std.Io.Writer) !void {
    try w.writeAll("\"prefix_cache_tiers\":{");
    inline for (.{ "hits", "evictions", "evicted_bytes" }, 0..) |field, f| {
        try w.print("{s}\"{s}\":{{", .{ if (f == 0) "" else ",", field });
        inline for (tier_names, 0..) |name, i| try w.print("{s}\"{s}\":{d}", .{ if (i == 0) "" else ",", name, @field(snap, field)[i] });
        try w.writeAll("}");
    }
    try w.print(",\"hot_bytes\":{d}}}", .{snap.hot_bytes});
}

const testing = std.testing;

test "Stats: a hit and an eviction land in their own tier" {
    var s: Stats = .{};
    s.recordHit(.hot);
    s.recordHit(.ssd);
    s.recordHit(.ssd);
    s.recordEviction(.ssd, 4096);
    s.recordEviction(.hot, 100);
    s.recordEviction(.hot, 28);
    s.setHotBytes(777);
    const snap = s.snapshot();
    try testing.expectEqual([2]u64{ 1, 2 }, snap.hits);
    try testing.expectEqual([2]u64{ 2, 1 }, snap.evictions);
    try testing.expectEqual([2]u64{ 128, 4096 }, snap.evicted_bytes);
    try testing.expectEqual(@as(u64, 777), snap.hot_bytes);
}

test "the Prometheus and JSON renderings name every tier and counter" {
    const snap: Snapshot = .{ .hits = .{ 3, 5 }, .evictions = .{ 7, 11 }, .evicted_bytes = .{ 13, 17 }, .hot_bytes = 19 };
    var buf: [2048]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try writePrometheus(snap, &w);
    const prom = w.buffered();
    for ([_][]const u8{
        "sushi:prefix_cache_tier_hits_total{tier=\"hot\"} 3\n",
        "sushi:prefix_cache_tier_hits_total{tier=\"ssd\"} 5\n",
        "sushi:prefix_cache_evictions_total{tier=\"hot\"} 7\n",
        "sushi:prefix_cache_evictions_total{tier=\"ssd\"} 11\n",
        "sushi:prefix_cache_evicted_bytes_total{tier=\"hot\"} 13\n",
        "sushi:prefix_cache_evicted_bytes_total{tier=\"ssd\"} 17\n",
        "sushi:prefix_cache_hot_bytes 19\n",
    }) |want| try testing.expect(std.mem.indexOf(u8, prom, want) != null);
    var json_buf: [1024]u8 = undefined;
    var jw: std.Io.Writer = .fixed(&json_buf);
    try jw.writeAll("{");
    try writeJson(snap, &jw);
    try jw.writeAll("}");
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, jw.buffered(), .{});
    defer parsed.deinit();
    const obj = parsed.value.object.get("prefix_cache_tiers").?.object;
    try testing.expectEqual(@as(i64, 3), obj.get("hits").?.object.get("hot").?.integer);
    try testing.expectEqual(@as(i64, 5), obj.get("hits").?.object.get("ssd").?.integer);
    try testing.expectEqual(@as(i64, 7), obj.get("evictions").?.object.get("hot").?.integer);
    try testing.expectEqual(@as(i64, 17), obj.get("evicted_bytes").?.object.get("ssd").?.integer);
    try testing.expectEqual(@as(i64, 19), obj.get("hot_bytes").?.integer);
}
