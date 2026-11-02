//! The dashboard's time series, kept by the server: the gauge sampler pushes one `Sample` per tick into a fixed ring,
//! and `GET /metrics.json?since=<unix ms>` returns the samples after that time. A page that was in the background
//! (and so did not poll) fills its charts from here, and plots each point at the server's own timestamp.
const std = @import("std");

/// The sampler's cadence: the token gauges move once per tick, so a faster ring would repeat values.
pub const interval_ms: u32 = 2000;
/// One hour of samples.
pub const capacity: usize = 1800;

pub const Sample = struct {
    /// Wall clock, unix milliseconds.
    t_ms: i64,
    /// `generation_tokens_live` and `prefill_tokens_live`: the page takes rates between samples.
    gen_tokens: u64,
    prefill_tokens: u64,
    mlx_active: u64,
    mlx_cache: u64,
    memory_mb: u32,
    gpu_pct: u16,
    /// 0 where the clock is unreadable.
    gpu_mhz: u16 = 0,
    running: u16,
    waiting: u16,
    prefilling: u16,
    batched: u16,
};

pub fn Ring(comptime cap: usize) type {
    return struct {
        const Self = @This();

        mu: std.Io.Mutex = .init,
        buf: [cap]Sample = undefined,
        /// Next write position.
        head: usize = 0,
        len: usize = 0,
        /// When the process started (`markStarted`); the first sample's time where nothing marked it; 0 before any.
        started_ms: i64 = 0,

        /// Unlike the sampler, which starts with the first enable, this runs at process start.
        pub fn markStarted(self: *Self, io: std.Io, t_ms: i64) void {
            self.mu.lockUncancelable(io);
            defer self.mu.unlock(io);
            self.started_ms = t_ms;
        }

        pub fn push(self: *Self, io: std.Io, s: Sample) void {
            self.mu.lockUncancelable(io);
            defer self.mu.unlock(io);
            if (self.started_ms == 0) self.started_ms = s.t_ms;
            self.buf[self.head] = s;
            self.head = (self.head + 1) % cap;
            self.len = @min(self.len + 1, cap);
        }

        /// The `i`th oldest kept sample. Caller holds `mu`.
        fn at(self: *const Self, i: usize) Sample {
            return self.buf[(self.head + cap - self.len + i) % cap];
        }

        /// `"history":{"interval_ms":N,"capacity":N,"samples":[[...],...]}` with the samples newer than `since_ms`, oldest first.
        pub fn writeSince(self: *Self, io: std.Io, since_ms: i64, w: *std.Io.Writer) !void {
            self.mu.lockUncancelable(io);
            defer self.mu.unlock(io);
            try w.print("\"history\":{{\"interval_ms\":{d},\"capacity\":{d},\"started_ms\":{d},\"fields\":[\"t_ms\",\"gen_tokens\",\"prefill_tokens\",\"running\",\"waiting\",\"prefilling\",\"batched\",\"gpu_pct\",\"memory_mb\",\"mlx_active\",\"mlx_cache\",\"gpu_mhz\"],\"samples\":[", .{ interval_ms, cap, self.started_ms });
            var first = true;
            for (0..self.len) |i| {
                const s = self.at(i);
                if (s.t_ms <= since_ms) continue;
                try w.print("{s}[{d},{d},{d},{d},{d},{d},{d},{d},{d},{d},{d},{d}]", .{
                    if (first) "" else ",", s.t_ms, s.gen_tokens, s.prefill_tokens, s.running, s.waiting,
                    s.prefilling,           s.batched, s.gpu_pct,  s.memory_mb, s.mlx_active, s.mlx_cache,
                    s.gpu_mhz,
                });
                first = false;
            }
            try w.writeAll("]}");
        }
    };
}

pub var process: Ring(capacity) = .{};

/// `since=<unix ms>` of a request path, or null when absent or not a number.
pub fn sinceParam(raw_path: []const u8) ?i64 {
    const q = raw_path[(std.mem.indexOfScalar(u8, raw_path, '?') orelse return null) + 1 ..];
    var it = std.mem.splitScalar(u8, q, '&');
    while (it.next()) |kv| {
        if (!std.mem.startsWith(u8, kv, "since=")) continue;
        return std.fmt.parseInt(i64, kv["since=".len..], 10) catch null;
    }
    return null;
}

const testing = std.testing;

fn sample(t: i64) Sample {
    return .{ .t_ms = t, .gen_tokens = @intCast(t), .prefill_tokens = 0, .mlx_active = 0, .mlx_cache = 0, .memory_mb = 1, .gpu_pct = 2, .gpu_mhz = 1200, .running = 3, .waiting = 4, .prefilling = 5, .batched = 6 };
}

fn rendered(ring: anytype, since: i64) !std.json.Parsed(std.json.Value) {
    var out: std.Io.Writer.Allocating = .init(testing.allocator);
    defer out.deinit();
    try out.writer.writeAll("{");
    try ring.writeSince(testing.io, since, &out.writer);
    try out.writer.writeAll("}");
    return std.json.parseFromSlice(std.json.Value, testing.allocator, out.written(), .{});
}

test "metrics history: a ring keeps the newest capacity samples, oldest first, and wraps" {
    var ring: Ring(4) = .{};
    for (1..7) |i| ring.push(testing.io, sample(@as(i64, @intCast(i)) * 1000));
    try testing.expectEqual(@as(usize, 4), ring.len);
    const parsed = try rendered(&ring, 0);
    defer parsed.deinit();
    const rows = parsed.value.object.get("history").?.object.get("samples").?.array.items;
    try testing.expectEqual(@as(usize, 4), rows.len);
    try testing.expectEqual(@as(i64, 3000), rows[0].array.items[0].integer);
    try testing.expectEqual(@as(i64, 6000), rows[3].array.items[0].integer);
    try testing.expectEqual(@as(i64, 3), rows[0].array.items[3].integer);
    const fields = parsed.value.object.get("history").?.object.get("fields").?.array.items;
    try testing.expectEqualStrings("gpu_mhz", fields[fields.len - 1].string);
    try testing.expectEqual(@as(i64, 1200), rows[0].array.items[fields.len - 1].integer);
}

test "metrics history: since returns only later samples, an empty ring and a future time return none" {
    var ring: Ring(8) = .{};
    {
        const parsed = try rendered(&ring, 0);
        defer parsed.deinit();
        try testing.expectEqual(@as(usize, 0), parsed.value.object.get("history").?.object.get("samples").?.array.items.len);
    }
    for (1..6) |i| ring.push(testing.io, sample(@as(i64, @intCast(i)) * 1000));
    const mid = try rendered(&ring, 3000);
    defer mid.deinit();
    const rows = mid.value.object.get("history").?.object.get("samples").?.array.items;
    try testing.expectEqual(@as(usize, 2), rows.len);
    try testing.expectEqual(@as(i64, 4000), rows[0].array.items[0].integer);
    const future = try rendered(&ring, 9000);
    defer future.deinit();
    try testing.expectEqual(@as(usize, 0), future.value.object.get("history").?.object.get("samples").?.array.items.len);
}

test "metrics history: started_ms is the first sample's time and survives the ring wrapping" {
    var ring: Ring(2) = .{};
    for (1..5) |i| ring.push(testing.io, sample(@as(i64, @intCast(i)) * 1000));
    const parsed = try rendered(&ring, 0);
    defer parsed.deinit();
    try testing.expectEqual(@as(i64, 1000), parsed.value.object.get("history").?.object.get("started_ms").?.integer);
}

test "metrics history: started_ms is the process start, not the late first sample of a server that enabled metrics at runtime" {
    var ring: Ring(4) = .{};
    ring.markStarted(testing.io, 500);
    ring.push(testing.io, sample(90_000));
    const parsed = try rendered(&ring, 0);
    defer parsed.deinit();
    try testing.expectEqual(@as(i64, 500), parsed.value.object.get("history").?.object.get("started_ms").?.integer);
}

test "metrics history: the ring is a fixed size however long it runs" {
    try testing.expect(@sizeOf(Ring(capacity)) < 128 * 1024);
    var ring: Ring(16) = .{};
    for (0..1000) |i| ring.push(testing.io, sample(@intCast(i)));
    try testing.expectEqual(@as(usize, 16), ring.len);
}

test "metrics history: since is read from the query" {
    try testing.expectEqual(@as(?i64, 1700000000000), sinceParam("/metrics.json?since=1700000000000"));
    try testing.expectEqual(@as(?i64, 5), sinceParam("/metrics.json?x=1&since=5"));
    try testing.expectEqual(@as(?i64, null), sinceParam("/metrics.json"));
    try testing.expectEqual(@as(?i64, null), sinceParam("/metrics.json?since=abc"));
}
