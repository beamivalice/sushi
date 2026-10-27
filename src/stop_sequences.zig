//! Client stop sequences: ONE definition of which stop fires and where the text is cut, shared by
//! every non-streaming surface (`earliest`) and every streaming surface (`Gate`), so a stream and a
//! non-stream answer are the same bytes.
//!
//! The rule: the earliest occurrence in the generated text wins, whatever order the client listed
//! the stops in; at the same start the SHORTEST stop wins, because it is the one a stream sees
//! complete first (a stream cannot know a longer stop is coming).

const std = @import("std");
const log = @import("log.zig");

pub const Hit = struct { index: usize, stop: []const u8 };

pub fn earliest(text: []const u8, stops: []const []const u8) ?Hit {
    var best: ?Hit = null;
    for (stops) |stop| {
        if (stop.len == 0) continue;
        const idx = std.mem.indexOf(u8, text, stop) orelse continue;
        if (best) |b| if (b.index < idx or (b.index == idx and b.stop.len <= stop.len)) continue;
        best = .{ .index = idx, .stop = stop };
    }
    return best;
}

/// Start of the longest suffix of `text` that is a proper prefix of some stop (`text.len` if none):
/// the bytes a later token could still turn into a match.
fn pendingStart(text: []const u8, stops: []const []const u8) usize {
    var start: usize = 0;
    while (start < text.len) : (start += 1) {
        const tail = text[start..];
        for (stops) |stop| {
            if (stop.len > tail.len and std.mem.startsWith(u8, stop, tail)) return start;
        }
    }
    return text.len;
}

/// Number of trailing bytes of `s` that start a UTF-8 sequence the string ends before completing
/// (0 when it ends on a character boundary or in bytes no continuation can repair).
pub fn utf8TrailingIncomplete(s: []const u8) usize {
    if (s.len == 0) return 0;
    var i: usize = s.len;
    var cont: usize = 0;
    while (cont < 3 and i > 0) {
        i -= 1;
        if (s[i] & 0xC0 != 0x80) break;
        cont += 1;
    }
    if (i >= s.len) return 0;
    const lead = s[i];
    const expected: usize = if (lead & 0x80 == 0) 1 else if (lead & 0xE0 == 0xC0) 2 else if (lead & 0xF0 == 0xE0) 3 else if (lead & 0xF8 == 0xF0) 4 else return 0;
    const actual = s.len - i;
    return if (actual < expected) actual else 0;
}

/// Sits between the decoded token text and every delivery path (reasoning, content, tool buffer):
/// bytes that may still become part of a stop are held back until resolved, so a streamed delta is
/// never retracted by a stop that completes a token later. A token may also end inside a
/// character: those bytes wait in `carry` for the next token, so no delta ends mid-character.
pub const Gate = struct {
    stops: []const []const u8,
    held: std.ArrayList(u8) = .empty,
    carry: [3]u8 = undefined,
    carry_len: u8 = 0,
    /// Set once a stop fired; nothing is pushed after that.
    matched: ?[]const u8 = null,

    pub fn deinit(self: *Gate, allocator: std.mem.Allocator) void {
        self.held.deinit(allocator);
    }

    pub fn hasHeld(self: *const Gate) bool {
        return self.held.items.len > 0 or self.carry_len > 0;
    }

    /// Feed one token's decoded bytes; the result is the text safe to deliver now (caller frees).
    pub fn push(self: *Gate, allocator: std.mem.Allocator, text: []const u8) ![]u8 {
        try self.held.appendSlice(allocator, self.carry[0..self.carry_len]);
        self.carry_len = 0;
        try self.held.appendSlice(allocator, text);
        const tail = utf8TrailingIncomplete(self.held.items);
        @memcpy(self.carry[0..tail], self.held.items[self.held.items.len - tail ..]);
        self.carry_len = @intCast(tail);
        self.held.shrinkRetainingCapacity(self.held.items.len - tail);
        return self.release(allocator, false);
    }

    /// The generation ended: resolve what is held (a stop that completed, else the plain tail) with
    /// the unfinished character after it, delivered as the bytes the model produced.
    pub fn finish(self: *Gate, allocator: std.mem.Allocator) ![]u8 {
        try self.held.appendSlice(allocator, self.carry[0..self.carry_len]);
        self.carry_len = 0;
        return self.release(allocator, true);
    }

    fn release(self: *Gate, allocator: std.mem.Allocator, final: bool) ![]u8 {
        const buf = self.held.items;
        const wait_from = if (final) buf.len else pendingStart(buf, self.stops);
        var n = wait_from;
        // A completed stop waits for a held prefix that starts EARLIER: it could still complete as the earlier match.
        if (earliest(buf, self.stops)) |hit| if (hit.index <= wait_from) {
            n = hit.index;
            if (self.matched == null) {
                var line: [160]u8 = undefined;
                log.info("{s}\n", .{matchLine(&line, hit.stop)});
            }
            self.matched = hit.stop;
        };
        const out = try allocator.dupe(u8, buf[0..n]);
        if (self.matched != null) {
            self.held.clearRetainingCapacity();
        } else {
            std.mem.copyForwards(u8, buf[0 .. buf.len - n], buf[n..]);
            self.held.shrinkRetainingCapacity(buf.len - n);
        }
        return out;
    }
};

/// The one `[stop]` line a request's own stop sequence leaves, so a turn it cut is told from an EOS.
fn matchLine(out: []u8, stop: []const u8) []const u8 {
    const head = "[stop] reason=stop-seq match=";
    @memcpy(out[0..head.len], head);
    return out[0..log.appendQuoted(out, head.len, stop, 100)];
}

/// Test driver: what a streaming surface delivers for `fragments` (the decoded token texts, in
/// order), and which stop fired. Returns the concatenated delivered bytes (caller frees).
pub fn streamDelivered(allocator: std.mem.Allocator, fragments: []const []const u8, stops: []const []const u8, matched: *?[]const u8) ![]u8 {
    var gate = Gate{ .stops = stops };
    defer gate.deinit(allocator);
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(allocator);
    for (fragments) |f| {
        const piece = try gate.push(allocator, f);
        defer allocator.free(piece);
        try out.appendSlice(allocator, piece);
        if (gate.matched != null) break;
    }
    if (gate.matched == null and gate.hasHeld()) {
        const tail = try gate.finish(allocator);
        defer allocator.free(tail);
        try out.appendSlice(allocator, tail);
    }
    matched.* = gate.matched;
    return out.toOwnedSlice(allocator);
}

/// Test oracle for the non-streaming surfaces: the same cut they apply to the whole text.
pub fn nonStreamDelivered(text: []const u8, stops: []const []const u8, matched: *?[]const u8) []const u8 {
    const hit = earliest(text, stops) orelse {
        matched.* = null;
        return text;
    };
    matched.* = hit.stop;
    return text[0..hit.index];
}

fn expectParity(fragments: []const []const u8, stops: []const []const u8) !void {
    const t = std.testing;
    const whole = try std.mem.concat(t.allocator, u8, fragments);
    defer t.allocator.free(whole);
    var want_stop: ?[]const u8 = null;
    const want = nonStreamDelivered(whole, stops, &want_stop);
    var got_stop: ?[]const u8 = null;
    const got = try streamDelivered(t.allocator, fragments, stops, &got_stop);
    defer t.allocator.free(got);
    try t.expectEqualStrings(want, got);
    try t.expectEqual(want_stop != null, got_stop != null);
    if (want_stop) |w| try t.expectEqualStrings(w, got_stop.?);
}

/// Every way to cut `text` into two and into three consecutive fragments.
fn expectParityAtEverySplit(text: []const u8, stops: []const []const u8) !void {
    var i: usize = 0;
    while (i <= text.len) : (i += 1) {
        try expectParity(&.{ text[0..i], text[i..] }, stops);
        var j: usize = i;
        while (j <= text.len) : (j += 1) try expectParity(&.{ text[0..i], text[i..j], text[j..] }, stops);
    }
}

test "[stop] names the stop sequence that cut the turn" {
    var line: [160]u8 = undefined;
    try std.testing.expectEqualStrings("[stop] reason=stop-seq match=\"\\n\\nUser:\"", matchLine(&line, "\n\nUser:"));
}

test "a stop prefix is held until the stop completes, never delivered" {
    const t = std.testing;
    var matched: ?[]const u8 = null;
    const got = try streamDelivered(t.allocator, &.{ "alpha", "ST", "OP", "beta" }, &.{"STOP"}, &matched);
    defer t.allocator.free(got);
    try t.expectEqualStrings("alpha", got);
    try t.expectEqualStrings("STOP", matched.?);
}

test "a held prefix that never completes is delivered at the end of the generation" {
    const t = std.testing;
    var matched: ?[]const u8 = null;
    const got = try streamDelivered(t.allocator, &.{ "alpha", "ST", "O" }, &.{"STOP"}, &matched);
    defer t.allocator.free(got);
    try t.expectEqualStrings("alphaSTO", got);
    try t.expectEqual(@as(?[]const u8, null), matched);
    const wrong = try streamDelivered(t.allocator, &.{ "a", "ST", "X", "b" }, &.{"STOP"}, &matched);
    defer t.allocator.free(wrong);
    try t.expectEqualStrings("aSTXb", wrong);
}

test "the earliest occurrence wins, not the first listed stop" {
    const t = std.testing;
    const text = "alphaSTOPbetaEND";
    const hit = earliest(text, &.{ "END", "STOP" }).?;
    try t.expectEqual(@as(usize, 5), hit.index);
    try t.expectEqualStrings("STOP", hit.stop);
    // Same start: the shortest stop, whatever the request order.
    try t.expectEqualStrings("ST", earliest("xSTOP", &.{ "STOP", "ST" }).?.stop);
    try t.expectEqualStrings("ST", earliest("xSTOP", &.{ "ST", "STOP" }).?.stop);
    try t.expectEqual(@as(?Hit, null), earliest("all clear", &.{"STOP"}));
}

test "a completed stop waits for a held prefix that could complete earlier" {
    try expectParityAtEverySplit("ABCD", &.{ "ABCD", "BC" });
    try expectParityAtEverySplit("xABCXy", &.{ "ABCD", "BC" });
    try expectParityAtEverySplit("xABCD", &.{ "BC", "ABCD" });
}

test "stream and non-stream agree at every split of a multi-token stop" {
    try expectParityAtEverySplit("alpha STOP beta", &.{"STOP"});
    try expectParityAtEverySplit("alpha\n\nHuman: beta", &.{"\n\nHuman:"});
    try expectParityAtEverySplit("alpha STO beta", &.{"STOP"});
    try expectParityAtEverySplit("alpha ST", &.{"STOP"});
    try expectParityAtEverySplit("naïve STÖP tail", &.{"STÖP"});
}

test "stream and non-stream agree for every order of several stops" {
    const a = "END";
    const b = "STOP";
    const c = "ta ST";
    const text = "alphaSTOPbetaENDgammaST";
    const perms = [_][3][]const u8{
        .{ a, b, c }, .{ a, c, b }, .{ b, a, c }, .{ b, c, a }, .{ c, a, b }, .{ c, b, a },
    };
    for (perms) |p| try expectParityAtEverySplit(text, &p);
    for (perms) |p| try expectParityAtEverySplit("xENDySTOP", &p);
}

test "a stop inside reasoning cuts the raw stream where it falls" {
    try expectParityAtEverySplit("<think>weigh STOP more</think>answer", &.{"STOP"});
    try expectParityAtEverySplit("<think>weigh it</think>answer END", &.{ "END", "</think>ans" });
}

test "a stop that overlaps itself resolves at its first occurrence" {
    try expectParityAtEverySplit("aaab", &.{"aab"});
    try expectParityAtEverySplit("ababab", &.{"abab"});
}

test "utf8TrailingIncomplete reports only a sequence the string ends before completing" {
    const t = std.testing;
    try t.expectEqual(@as(usize, 0), utf8TrailingIncomplete(""));
    try t.expectEqual(@as(usize, 0), utf8TrailingIncomplete("hello"));
    try t.expectEqual(@as(usize, 0), utf8TrailingIncomplete("\xF0\x9F\x8E\x89"));
    try t.expectEqual(@as(usize, 3), utf8TrailingIncomplete("\xF0\x9F\x8E"));
    try t.expectEqual(@as(usize, 2), utf8TrailingIncomplete("\xF0\x9F"));
    try t.expectEqual(@as(usize, 1), utf8TrailingIncomplete("\xF0"));
    try t.expectEqual(@as(usize, 2), utf8TrailingIncomplete("hi\xF0\x9F"));
}

// Chat, Anthropic, Responses and completions all deliver through this Gate, so one parity run
// covers the four streaming loops.
test "a stream that ends inside a character delivers the same bytes as the non-stream text" {
    const tails = [_][]const u8{ "\xC3", "\xE2\x82", "\xE2", "\xF0\x9F\x8E", "\xF0\x9F", "\xF0" };
    for (tails) |tail| {
        for ([_][]const u8{ "alpha", "alphaST", "alphaSTO", "" }) |head| {
            const text = try std.mem.concat(std.testing.allocator, u8, &.{ head, tail });
            defer std.testing.allocator.free(text);
            try expectParityAtEverySplit(text, &.{"STOP"});
            try expectParityAtEverySplit(text, &.{});
        }
    }
}

test "an incomplete character completed by the next token is delivered whole, around a held stop prefix" {
    try expectParityAtEverySplit("alphaST\xC3\xA9tail", &.{"STOP"});
    try expectParityAtEverySplit("alpha\xF0\x9F\x8E\x89STO", &.{"STOP"});
    try expectParityAtEverySplit("alpha\xF0\x9F\x8E\x89STOP", &.{"STOP"});
}
