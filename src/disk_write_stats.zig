//! Cumulative bytes the SSD prefix cache wrote to the filesystem, by file kind. One `Stats` per tier
//! (`/props` per model) chains into `process` (`/metrics`), so a model that unloads keeps its bytes in
//! the process total. Counting does not depend on `--metrics`: the totals run from process start.

const std = @import("std");

/// What a written file holds. A new file kind is a new variant here, never a filename test.
pub const Kind = enum {
    /// `c*.safetensors` KV chunks.
    kv,
    /// `s*.safetensors` restore points and their `g*.safetensors` row segments.
    ssm,
    /// `qsa.safetensors` history.
    qsa,
    /// `spec.safetensors` draft-state sidecar.
    spec,
    /// `r*.safetensors` ring restore points.
    ring,
    /// `meta.json` and `tokens.bin`.
    meta,
};

const kind_names = @typeInfo(Kind).@"enum".field_names;
pub const kind_count = kind_names.len;

pub const Snapshot = struct {
    written: [kind_count]u64 = @splat(0),
    /// Bytes handed to a write that failed, or staged and then dropped; never part of `written`.
    failed: u64 = 0,

    pub fn totalWritten(self: Snapshot) u64 {
        var n: u64 = 0;
        for (self.written) |b| n +|= b;
        return n;
    }
};

pub const Stats = struct {
    written: [kind_count]std.atomic.Value(u64) = @splat(std.atomic.Value(u64).init(0)),
    failed: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    parent: ?*Stats = null,

    pub fn recordWritten(self: *Stats, kind: Kind, bytes: u64) void {
        _ = self.written[@intFromEnum(kind)].fetchAdd(bytes, .monotonic);
        if (self.parent) |p| p.recordWritten(kind, bytes);
    }

    pub fn recordFailed(self: *Stats, bytes: u64) void {
        _ = self.failed.fetchAdd(bytes, .monotonic);
        if (self.parent) |p| p.recordFailed(bytes);
    }

    pub fn snapshot(self: *const Stats) Snapshot {
        var s: Snapshot = .{ .failed = self.failed.load(.monotonic) };
        for (&s.written, &self.written) |*dst, *src| dst.* = src.load(.monotonic);
        return s;
    }
};

/// Every tier of this process.
pub var process: Stats = .{};

pub fn writePrometheus(snap: Snapshot, w: *std.Io.Writer) !void {
    try w.print("# HELP sushi:prefix_cache_write_bytes_total Bytes the SSD prefix cache wrote to the filesystem since process start, by file kind\n# TYPE sushi:prefix_cache_write_bytes_total counter\n", .{});
    inline for (kind_names, 0..) |name, i| {
        try w.print("sushi:prefix_cache_write_bytes_total{{kind=\"{s}\"}} {d}\n", .{ name, snap.written[i] });
    }
    try w.print("\n# HELP sushi:prefix_cache_write_failed_bytes_total Bytes of SSD prefix cache writes that failed or were dropped before landing\n# TYPE sushi:prefix_cache_write_failed_bytes_total counter\nsushi:prefix_cache_write_failed_bytes_total {d}\n\n", .{snap.failed});
}

/// `"prefix_cache_writes":{...}` as one member of the `/metrics.json` object.
pub fn writeJson(snap: Snapshot, w: *std.Io.Writer) !void {
    try w.print("\"prefix_cache_writes\":{{\"written_bytes\":{{", .{});
    inline for (kind_names, 0..) |name, i| {
        try w.print("{s}\"{s}\":{d}", .{ if (i == 0) "" else ",", name, snap.written[i] });
    }
    try w.print("}},\"failed_bytes\":{d}}}", .{snap.failed});
}

test "Stats: a write lands in its kind and in the parent, a failure in failed only" {
    var total: Stats = .{};
    var tier: Stats = .{ .parent = &total };
    tier.recordWritten(.kv, 100);
    tier.recordWritten(.kv, 28);
    tier.recordWritten(.qsa, 7);
    tier.recordFailed(5);
    const s = tier.snapshot();
    try std.testing.expectEqual(@as(u64, 128), s.written[@intFromEnum(Kind.kv)]);
    try std.testing.expectEqual(@as(u64, 7), s.written[@intFromEnum(Kind.qsa)]);
    try std.testing.expectEqual(@as(u64, 0), s.written[@intFromEnum(Kind.ssm)]);
    try std.testing.expectEqual(@as(u64, 135), s.totalWritten());
    try std.testing.expectEqual(@as(u64, 5), s.failed);
    try std.testing.expectEqual(s, total.snapshot());
}

test "the Prometheus and JSON renderings name every kind and the failed counter" {
    var snap: Snapshot = .{ .failed = 9 };
    for (&snap.written, 0..) |*b, i| b.* = 1000 + i;
    var buf: [2048]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try writePrometheus(snap, &w);
    const prom = w.buffered();
    var json_buf: [1024]u8 = undefined;
    var jw: std.Io.Writer = .fixed(&json_buf);
    try jw.writeAll("{");
    try writeJson(snap, &jw);
    try jw.writeAll("}");
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, jw.buffered(), .{});
    defer parsed.deinit();
    const obj = parsed.value.object.get("prefix_cache_writes").?.object;
    inline for (kind_names, 0..) |name, i| {
        var line: [96]u8 = undefined;
        const want = try std.fmt.bufPrint(&line, "sushi:prefix_cache_write_bytes_total{{kind=\"{s}\"}} {d}\n", .{ name, 1000 + i });
        try std.testing.expect(std.mem.indexOf(u8, prom, want) != null);
        try std.testing.expectEqual(@as(i64, 1000 + i), obj.get("written_bytes").?.object.get(name).?.integer);
    }
    try std.testing.expect(std.mem.indexOf(u8, prom, "sushi:prefix_cache_write_failed_bytes_total 9\n") != null);
    try std.testing.expectEqual(@as(i64, 9), obj.get("failed_bytes").?.integer);
}
