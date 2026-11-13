//! Streamed-expert counters the metrics endpoints export: per forward, the routed experts a layer found in its cache
//! and the ones it filled from the SSD, the bytes and time the fills took, and every SSD read the expert engines issued
//! (the prefetcher's included). One `process` for the whole server, written with relaxed atomics by
//! `expert_stream.Engine.finishForward`; counting does not depend on `--metrics`.

const std = @import("std");

pub const Snapshot = struct {
    /// Expert engines alive now (one per loaded model that streams experts) and the cache bytes they hold.
    engines: u64 = 0,
    cache_bytes: u64 = 0,
    /// Forwards the engines ran, and their wall time.
    forwards: u64 = 0,
    forward_ns: u64 = 0,
    /// Routed experts the forwards needed (a layer's distinct experts per forward) and those already in the cache.
    routes: u64 = 0,
    hits: u64 = 0,
    /// Bytes the forwards waited on the SSD for, and the time they waited.
    fill_bytes: u64 = 0,
    fill_ns: u64 = 0,
    /// Every byte the expert engines read from the SSD, the prefetcher's reads ahead of need included.
    ssd_read_bytes: u64 = 0,
};

pub const Forward = struct { wall_ns: u64, routes: u64, hits: u64, fill_bytes: u64, fill_ns: u64, ssd_read_bytes: u64 };

const Counter = std.atomic.Value(u64);

pub const Stats = struct {
    engines: Counter = .init(0),
    cache_bytes: Counter = .init(0),
    forwards: Counter = .init(0),
    forward_ns: Counter = .init(0),
    routes: Counter = .init(0),
    hits: Counter = .init(0),
    fill_bytes: Counter = .init(0),
    fill_ns: Counter = .init(0),
    ssd_read_bytes: Counter = .init(0),

    /// An engine came up holding `cache_bytes` of expert slots, or (`up` false) went away with them.
    pub fn engine(self: *Stats, up: bool, cache_bytes: u64) void {
        if (up) {
            _ = self.engines.fetchAdd(1, .monotonic);
            _ = self.cache_bytes.fetchAdd(cache_bytes, .monotonic);
        } else {
            _ = self.engines.fetchSub(1, .monotonic);
            _ = self.cache_bytes.fetchSub(cache_bytes, .monotonic);
        }
    }

    pub fn recordForward(self: *Stats, f: Forward) void {
        _ = self.forwards.fetchAdd(1, .monotonic);
        _ = self.forward_ns.fetchAdd(f.wall_ns, .monotonic);
        _ = self.routes.fetchAdd(f.routes, .monotonic);
        _ = self.hits.fetchAdd(f.hits, .monotonic);
        _ = self.fill_bytes.fetchAdd(f.fill_bytes, .monotonic);
        _ = self.fill_ns.fetchAdd(f.fill_ns, .monotonic);
        _ = self.ssd_read_bytes.fetchAdd(f.ssd_read_bytes, .monotonic);
    }

    pub fn snapshot(self: *const Stats) Snapshot {
        var s: Snapshot = .{};
        inline for (@typeInfo(Snapshot).@"struct".field_names) |name| @field(s, name) = @field(self, name).load(.monotonic);
        return s;
    }
};

pub var process: Stats = .{};

const counters = .{
    .{ "forwards", "sushi:expert_stream_forwards_total", "Forwards run by a model that streams experts from the SSD" },
    .{ "forward_ns", "sushi:expert_stream_forward_seconds_total", "Wall time of those forwards" },
    .{ "routes", "sushi:expert_stream_routes_total", "Routed experts the forwards needed, counted once per layer and forward" },
    .{ "hits", "sushi:expert_stream_hits_total", "Routed experts the forwards found in the expert cache" },
    .{ "fill_bytes", "sushi:expert_stream_fill_bytes_total", "Expert bytes the forwards waited on the SSD for" },
    .{ "fill_ns", "sushi:expert_stream_fill_seconds_total", "Time the forwards waited on those SSD fills" },
    .{ "ssd_read_bytes", "sushi:expert_stream_ssd_read_bytes_total", "Every byte the expert engines read from the SSD, prefetch included" },
};

pub fn writePrometheus(snap: Snapshot, w: *std.Io.Writer) !void {
    try w.print("# HELP sushi:expert_stream_cache_bytes Expert cache bytes of every loaded model that streams experts\n# TYPE sushi:expert_stream_cache_bytes gauge\nsushi:expert_stream_cache_bytes {d}\n\n", .{snap.cache_bytes});
    inline for (counters) |c| {
        try w.print("# HELP {s} {s}\n# TYPE {s} counter\n", .{ c[1], c[2], c[1] });
        if (comptime std.mem.endsWith(u8, c[0], "_ns"))
            try w.print("{s} {d:.6}\n\n", .{ c[1], @as(f64, @floatFromInt(@field(snap, c[0]))) / 1e9 })
        else
            try w.print("{s} {d}\n\n", .{ c[1], @field(snap, c[0]) });
    }
}

/// `"expert_stream":{...}` as one member of the `/metrics.json` object.
pub fn writeJson(snap: Snapshot, w: *std.Io.Writer) !void {
    try w.writeAll("\"expert_stream\":{");
    inline for (@typeInfo(Snapshot).@"struct".field_names, 0..) |name, i| try w.print("{s}\"{s}\":{d}", .{ if (i == 0) "" else ",", name, @field(snap, name) });
    try w.writeAll("}");
}

const testing = std.testing;

test "expert stats: forwards add up and an engine's cache comes and goes with it" {
    var s: Stats = .{};
    s.engine(true, 1000);
    s.recordForward(.{ .wall_ns = 30, .routes = 10, .hits = 7, .fill_bytes = 300, .fill_ns = 12, .ssd_read_bytes = 500 });
    s.recordForward(.{ .wall_ns = 20, .routes = 4, .hits = 4, .fill_bytes = 0, .fill_ns = 0, .ssd_read_bytes = 64 });
    const up = s.snapshot();
    try testing.expectEqual(Snapshot{ .engines = 1, .cache_bytes = 1000, .forwards = 2, .forward_ns = 50, .routes = 14, .hits = 11, .fill_bytes = 300, .fill_ns = 12, .ssd_read_bytes = 564 }, up);
    s.engine(false, 1000);
    const down = s.snapshot();
    try testing.expectEqual(@as(u64, 0), down.engines);
    try testing.expectEqual(@as(u64, 0), down.cache_bytes);
    try testing.expectEqual(@as(u64, 14), down.routes);
}

test "expert stats: the Prometheus and JSON renderings carry every counter" {
    const snap: Snapshot = .{ .engines = 1, .cache_bytes = 3, .forwards = 5, .forward_ns = 2_500_000_000, .routes = 11, .hits = 13, .fill_bytes = 17, .fill_ns = 500_000_000, .ssd_read_bytes = 19 };
    var buf: [4096]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try writePrometheus(snap, &w);
    for ([_][]const u8{
        "sushi:expert_stream_cache_bytes 3\n",           "sushi:expert_stream_forwards_total 5\n",
        "sushi:expert_stream_forward_seconds_total 2.5", "sushi:expert_stream_routes_total 11\n",
        "sushi:expert_stream_hits_total 13\n",           "sushi:expert_stream_fill_bytes_total 17\n",
        "sushi:expert_stream_fill_seconds_total 0.5",    "sushi:expert_stream_ssd_read_bytes_total 19\n",
    }) |line| try testing.expect(std.mem.indexOf(u8, w.buffered(), line) != null);
    var jbuf: [1024]u8 = undefined;
    var jw: std.Io.Writer = .fixed(&jbuf);
    try jw.writeAll("{");
    try writeJson(snap, &jw);
    try jw.writeAll("}");
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, jw.buffered(), .{});
    defer parsed.deinit();
    const o = parsed.value.object.get("expert_stream").?.object;
    try testing.expectEqual(@as(i64, 13), o.get("hits").?.integer);
    try testing.expectEqual(@as(i64, 19), o.get("ssd_read_bytes").?.integer);
    try testing.expectEqual(@as(i64, 1), o.get("engines").?.integer);
}
