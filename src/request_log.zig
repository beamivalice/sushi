//! The most recently finished requests, kept by the server: `finishSlot` and the cleanup drain push one `Entry` per
//! request into a fixed ring, and `GET /metrics.json?since=<unix ms>` returns the entries finished after that time.
//! The page reads per-request speeds from here instead of differencing counters between polls, which merges two
//! requests that finish between one pair of polls.
const std = @import("std");
const json_text = @import("json_text.zig");
const spec_stats = @import("spec_stats.zig");

pub const capacity: usize = 128;
/// Longer model ids are cut (on a character boundary).
const model_cap = 128;

pub const End = enum {
    ok,
    cancelled,
    @"error",
    /// Refused before its first forward (`PrefillDoesNotFit`, `MediaRefused`).
    rejected,
};

pub const Entry = struct {
    request_id: u64,
    /// Wall clock, unix milliseconds; strictly increasing across the ring.
    t_ms: i64,
    /// Tokens the model forwarded (`prompt - cached`).
    prompt_tokens: u32,
    /// Tokens restored from the prefix cache.
    cached_tokens: u32,
    generated_tokens: u32,
    prefill_ns: u64,
    decode_ns: u64,
    end: End,
    /// What speculation did for this request; `.none` when nothing drafted.
    spec_mode: spec_stats.Mode = .none,
    spec_rounds: u32 = 0,
    spec_drafted: u32 = 0,
    spec_accepted: u32 = 0,
    model_len: u8 = 0,
    model_buf: [model_cap]u8 = undefined,

    pub fn setModel(self: *Entry, id: []const u8) void {
        var n = @min(id.len, model_cap);
        // A cut inside a multibyte character would make the JSON string invalid.
        if (n < id.len) while (n > 0 and id[n] & 0xC0 == 0x80) : (n -= 1) {};
        @memcpy(self.model_buf[0..n], id[0..n]);
        self.model_len = @intCast(n);
    }

    fn model(self: *const Entry) []const u8 {
        return self.model_buf[0..self.model_len];
    }
};

pub fn Ring(comptime cap: usize) type {
    return struct {
        const Self = @This();

        mu: std.Io.Mutex = .init,
        buf: [cap]Entry = undefined,
        /// Next write position.
        head: usize = 0,
        len: usize = 0,

        /// Stamps `e.t_ms` one past the newest entry's when the clock has not moved, so a page that polls with
        /// `since=<newest t_ms>` never skips a request that finished in the same millisecond.
        pub fn push(self: *Self, io: std.Io, e: Entry) void {
            self.mu.lockUncancelable(io);
            defer self.mu.unlock(io);
            var stored = e;
            if (self.len > 0) stored.t_ms = @max(stored.t_ms, self.at(self.len - 1).t_ms + 1);
            self.buf[self.head] = stored;
            self.head = (self.head + 1) % cap;
            self.len = @min(self.len + 1, cap);
        }

        /// The `i`th oldest kept entry. Caller holds `mu`.
        fn at(self: *const Self, i: usize) *const Entry {
            return &self.buf[(self.head + cap - self.len + i) % cap];
        }

        /// `"requests":[{...},...]` with the entries finished after `since_ms`, oldest first.
        pub fn writeSince(self: *Self, io: std.Io, since_ms: i64, w: *std.Io.Writer) !void {
            self.mu.lockUncancelable(io);
            defer self.mu.unlock(io);
            const ns_to_s = 1.0 / 1_000_000_000.0;
            try w.writeAll("\"requests\":[");
            var first = true;
            for (0..self.len) |i| {
                const e = self.at(i);
                if (e.t_ms <= since_ms) continue;
                try w.print("{s}{{\"request_id\":{d},\"model\":", .{ if (first) "" else ",", e.request_id });
                try json_text.write(w, e.model());
                try w.print(",\"t_ms\":{d},\"prompt_tokens\":{d},\"cached_tokens\":{d},\"prefill_seconds\":{d:.6},\"generated_tokens\":{d},\"decode_seconds\":{d:.6},\"outcome\":\"{s}\"", .{
                    e.t_ms,
                    e.prompt_tokens,
                    e.cached_tokens,
                    @as(f64, @floatFromInt(e.prefill_ns)) * ns_to_s,
                    e.generated_tokens,
                    @as(f64, @floatFromInt(e.decode_ns)) * ns_to_s,
                    outcomeName(e.end),
                });
                if (e.spec_mode != .none) try w.print(",\"spec\":{{\"mode\":\"{s}\",\"rounds\":{d},\"drafted\":{d},\"accepted\":{d}}}", .{ @tagName(e.spec_mode), e.spec_rounds, e.spec_drafted, e.spec_accepted });
                try w.writeAll("}");
                first = false;
            }
            try w.writeAll("]");
        }
    };
}

fn outcomeName(end: End) []const u8 {
    return if (end == .@"error") "error" else @tagName(end);
}

const testing = std.testing;

fn entry(id: u64, t: i64, model: []const u8) Entry {
    var e: Entry = .{ .request_id = id, .t_ms = t, .prompt_tokens = 10, .cached_tokens = 5, .generated_tokens = 7, .prefill_ns = 250_000_000, .decode_ns = 1_500_000_000, .end = .ok };
    e.setModel(model);
    return e;
}

fn rendered(ring: anytype, since: i64) !std.json.Parsed(std.json.Value) {
    var out: std.Io.Writer.Allocating = .init(testing.allocator);
    defer out.deinit();
    try out.writer.writeAll("{");
    try ring.writeSince(testing.io, since, &out.writer);
    try out.writer.writeAll("}");
    return std.json.parseFromSlice(std.json.Value, testing.allocator, out.written(), .{});
}

test "request log: an entry renders every field, the times in seconds" {
    var ring: Ring(4) = .{};
    var e = entry(42, 1000, "org/model");
    e.end = .cancelled;
    ring.push(testing.io, e);
    const parsed = try rendered(&ring, 0);
    defer parsed.deinit();
    const row = parsed.value.object.get("requests").?.array.items[0].object;
    try testing.expectEqual(@as(i64, 42), row.get("request_id").?.integer);
    try testing.expectEqualStrings("org/model", row.get("model").?.string);
    try testing.expectEqual(@as(i64, 1000), row.get("t_ms").?.integer);
    try testing.expectEqual(@as(i64, 10), row.get("prompt_tokens").?.integer);
    try testing.expectEqual(@as(i64, 5), row.get("cached_tokens").?.integer);
    try testing.expectEqual(@as(f64, 0.25), row.get("prefill_seconds").?.float);
    try testing.expectEqual(@as(i64, 7), row.get("generated_tokens").?.integer);
    try testing.expectEqual(@as(f64, 1.5), row.get("decode_seconds").?.float);
    try testing.expectEqualStrings("cancelled", row.get("outcome").?.string);
}

test "request log: the ring keeps the newest entries, oldest first, and since filters" {
    var ring: Ring(3) = .{};
    for (1..6) |i| ring.push(testing.io, entry(i, @as(i64, @intCast(i)) * 1000, "m"));
    {
        const all = try rendered(&ring, 0);
        defer all.deinit();
        const rows = all.value.object.get("requests").?.array.items;
        try testing.expectEqual(@as(usize, 3), rows.len);
        try testing.expectEqual(@as(i64, 3), rows[0].object.get("request_id").?.integer);
        try testing.expectEqual(@as(i64, 5), rows[2].object.get("request_id").?.integer);
    }
    const later = try rendered(&ring, 4000);
    defer later.deinit();
    try testing.expectEqual(@as(usize, 1), later.value.object.get("requests").?.array.items.len);
    const none = try rendered(&ring, 9000);
    defer none.deinit();
    try testing.expectEqual(@as(usize, 0), none.value.object.get("requests").?.array.items.len);
}

test "request log: requests finishing in one millisecond keep distinct times" {
    var ring: Ring(4) = .{};
    ring.push(testing.io, entry(1, 5000, "m"));
    ring.push(testing.io, entry(2, 5000, "m"));
    ring.push(testing.io, entry(3, 4000, "m"));
    const first = try rendered(&ring, 0);
    defer first.deinit();
    const rows = first.value.object.get("requests").?.array.items;
    try testing.expectEqual(@as(i64, 5000), rows[0].object.get("t_ms").?.integer);
    try testing.expectEqual(@as(i64, 5001), rows[1].object.get("t_ms").?.integer);
    try testing.expectEqual(@as(i64, 5002), rows[2].object.get("t_ms").?.integer);
    const after = try rendered(&ring, 5001);
    defer after.deinit();
    try testing.expectEqual(@as(usize, 1), after.value.object.get("requests").?.array.items.len);
}

test "request log: a model id with a quote, a backslash and a control character stays valid JSON" {
    var ring: Ring(2) = .{};
    const id = "a\"b\\c\x01d\n";
    ring.push(testing.io, entry(1, 1, id));
    const parsed = try rendered(&ring, 0);
    defer parsed.deinit();
    try testing.expectEqualStrings(id, parsed.value.object.get("requests").?.array.items[0].object.get("model").?.string);
}

test "request log: an over-long model id is cut on a character boundary" {
    var ring: Ring(2) = .{};
    var long: [model_cap + 8]u8 = undefined;
    @memset(&long, 'x');
    long[model_cap - 1] = 0xC3;
    long[model_cap] = 0xA9;
    ring.push(testing.io, entry(1, 1, &long));
    const parsed = try rendered(&ring, 0);
    defer parsed.deinit();
    try testing.expectEqual(@as(usize, model_cap - 1), parsed.value.object.get("requests").?.array.items[0].object.get("model").?.string.len);
}

test "request log: the ring is a fixed size however long it runs" {
    try testing.expect(@sizeOf(Ring(capacity)) < 64 * 1024);
    var ring: Ring(8) = .{};
    for (0..1000) |i| ring.push(testing.io, entry(i, @intCast(i), "m"));
    try testing.expectEqual(@as(usize, 8), ring.len);
}

test "request log: an empty ring renders an empty array" {
    var ring: Ring(2) = .{};
    const parsed = try rendered(&ring, 0);
    defer parsed.deinit();
    try testing.expectEqual(@as(usize, 0), parsed.value.object.get("requests").?.array.items.len);
}

test "request log: a speculative request renders its rounds, a plain one has no spec member" {
    var ring: Ring(4) = .{};
    var spec = entry(1, 1000, "m");
    spec.spec_mode = .dspark;
    spec.spec_rounds = 12;
    spec.spec_drafted = 96;
    spec.spec_accepted = 60;
    ring.push(testing.io, spec);
    ring.push(testing.io, entry(2, 2000, "m"));
    const parsed = try rendered(&ring, 0);
    defer parsed.deinit();
    const rows = parsed.value.object.get("requests").?.array.items;
    const got = rows[0].object.get("spec").?.object;
    try testing.expectEqualStrings("dspark", got.get("mode").?.string);
    try testing.expectEqual(@as(i64, 12), got.get("rounds").?.integer);
    try testing.expectEqual(@as(i64, 96), got.get("drafted").?.integer);
    try testing.expectEqual(@as(i64, 60), got.get("accepted").?.integer);
    try testing.expect(rows[1].object.get("spec") == null);
}

