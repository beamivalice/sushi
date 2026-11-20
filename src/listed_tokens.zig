//! Which of a generation's tokens a reply lists logprobs for: the tokens whose bytes the reply returned. A stream
//! drains this as its text goes out and a non-stream reply reads it once; both ask `Listing`, so a stop's own tokens
//! and a thought the reply does not ship are listed on neither. A token the reply returned only part of (a stop or a
//! trimmed lead inside it) is listed with just those bytes: `head` and `tail` are what its entry drops.
const std = @import("std");

/// Tokens `start..end` (indexes into the generation); `head` bytes of the first and `tail` bytes of the last are not
/// returned text.
pub const Range = struct {
    start: usize,
    end: usize,
    head: usize = 0,
    tail: usize = 0,

    pub fn cut(self: Range) Cut {
        return .{ .head = self.head, .tail = self.tail };
    }
};

/// What a rendered list drops from its first token's start and its last token's end.
pub const Cut = struct { head: usize = 0, tail: usize = 0 };

pub const Listing = struct {
    /// Where each noted token starts, in bytes of the generation text as the reply sees it (hidden tokens are
    /// noted with length 0).
    starts: std.ArrayList(usize) = .empty,
    /// Bytes noted so far.
    total: usize = 0,
    /// Bytes the stop gate has let through: never a stop's own, never a prefix it still holds.
    released: usize = 0,
    /// Bytes through which a content chunk has gone out.
    content_to: usize = 0,
    /// The reply took no more text: `released` is final, so a token a stop cut into lists its returned bytes.
    ended: bool = false,
    /// The first token not yet shipped or retired.
    emitted: usize = 0,
    /// Bytes of that token the reply's content starts after.
    head: usize = 0,

    pub fn deinit(self: *Listing, allocator: std.mem.Allocator) void {
        self.starts.deinit(allocator);
    }

    pub fn note(self: *Listing, allocator: std.mem.Allocator, len: usize) !void {
        try self.starts.append(allocator, self.total);
        self.total += len;
    }

    /// The tokens that put a byte inside `from..to`; one straddling either edge is in, cut to the bytes inside.
    pub fn touching(self: *const Listing, from: usize, to: usize) Range {
        const start = self.firstEndingAfter(from);
        if (from >= to) return .{ .start = start, .end = start };
        const end = @max(start, self.firstStartingAtOrAfter(to));
        return self.cut(.{ .start = start, .end = end, .head = self.headOf(start, from) }, to);
    }

    /// The generation's own text from `from` is the reply's content: tokens wholly before it are never listed, and
    /// the one holding `from` lists only what follows it.
    pub fn skipTo(self: *Listing, from: usize) void {
        const first = self.firstEndingAfter(from);
        if (first < self.emitted) return;
        self.emitted = first;
        self.head = self.headOf(first, from);
    }

    fn headOf(self: *const Listing, token: usize, from: usize) usize {
        if (token >= self.starts.items.len) return 0;
        return from -| self.starts.items[token];
    }

    /// `r` with its last token's tail cut at `to`, and an emptied single token dropped.
    fn cut(self: *const Listing, r_in: Range, to: usize) Range {
        var r = r_in;
        if (r.end > r.start) r.tail = self.end_of(r.end - 1) -| to;
        if (r.end == r.start + 1 and r.head + r.tail >= self.end_of(r.start) - self.starts.items[r.start]) {
            r.end = r.start;
            r.head = 0;
            r.tail = 0;
        }
        return r;
    }

    /// Nothing released so far is content: retire it unlisted.
    pub fn retire(self: *Listing) void {
        const to = self.firstStartingAtOrAfter(self.released);
        if (to <= self.emitted) return;
        self.emitted = to;
        self.head = 0;
    }

    /// A content chunk went out, carrying every released byte but the last `withheld`. The tokens it lists are those
    /// up to `published` (a token's entry can lag its id), and only whole ones until the reply `ended`: the rest of
    /// a token a stop may still cut into, or the content arm still holds, is not returned yet.
    pub fn sent(self: *Listing, published: usize, withheld: usize) Range {
        self.content_to = self.released -| withheld;
        return self.drain(published);
    }

    /// The tokens of content already sent whose entries were not yet published when it went out.
    pub fn late(self: *Listing, published: usize) Range {
        return self.drain(published);
    }

    fn drain(self: *Listing, published: usize) Range {
        const start = self.emitted;
        var reach = self.firstStartingAtOrAfter(self.content_to);
        const straddles = reach > 0 and self.end_of(reach - 1) > self.content_to;
        if (straddles and !self.ended) reach -= 1;
        const end = @min(published, reach);
        if (end <= start) return .{ .start = start, .end = start };
        const r = self.cut(.{ .start = start, .end = end, .head = self.head }, if (end == reach) self.content_to else std.math.maxInt(usize));
        self.emitted = end;
        self.head = 0;
        return r;
    }

    fn end_of(self: *const Listing, i: usize) usize {
        return if (i + 1 < self.starts.items.len) self.starts.items[i + 1] else self.total;
    }

    /// The first token whose bytes end after `off` (a zero-length token ends where it starts).
    fn firstEndingAfter(self: *const Listing, off: usize) usize {
        var lo: usize = 0;
        var hi: usize = self.starts.items.len;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            if (self.end_of(mid) > off) hi = mid else lo = mid + 1;
        }
        return lo;
    }

    /// The first token that starts at or after `off`.
    fn firstStartingAtOrAfter(self: *const Listing, off: usize) usize {
        var lo: usize = 0;
        var hi: usize = self.starts.items.len;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            if (self.starts.items[mid] >= off) hi = mid else lo = mid + 1;
        }
        return lo;
    }
};

const testing = std.testing;

fn noteAll(l: *Listing, lens: []const usize) !void {
    for (lens) |n| try l.note(testing.allocator, n);
}

test "a stream lists a token only once all its bytes are released, a stop inside one lists its returned bytes at the end" {
    var l: Listing = .{};
    defer l.deinit(testing.allocator);
    // "[section" (8 bytes) then " x": the gate holds "ec" as a stop prefix.
    try noteAll(&l, &.{ 8, 2 });
    l.released = 2;
    const mid = l.sent(2, 0);
    try testing.expectEqual(@as(usize, 0), mid.end - mid.start);
    // The stop "ec" matched: the reply returned "[s", and the token carrying it lists only those bytes.
    l.ended = true;
    const last = l.late(2);
    try testing.expectEqual(@as(usize, 0), last.start);
    try testing.expectEqual(@as(usize, 1), last.end);
    try testing.expectEqual(@as(usize, 6), last.tail);
    try testing.expectEqual(@as(usize, 0), last.head);
}

test "a leading trim inside a token lists the token without the trimmed bytes, on both paths" {
    var l: Listing = .{};
    defer l.deinit(testing.allocator);
    // " 빵" is 4 bytes; the content starts after its space.
    try noteAll(&l, &.{4});
    const whole = l.touching(1, 4);
    try testing.expectEqual(@as(usize, 1), whole.end - whole.start);
    try testing.expectEqual(@as(usize, 1), whole.head);
    try testing.expectEqual(@as(usize, 0), whole.tail);

    l.released = 4;
    l.skipTo(1);
    const sent = l.sent(1, 0);
    try testing.expectEqual(@as(usize, 1), sent.end - sent.start);
    try testing.expectEqual(@as(usize, 1), sent.head);
}

test "a stop inside the token a reply ends on trims that token's tail, a stop at a token's edge trims nothing" {
    var l: Listing = .{};
    defer l.deinit(testing.allocator);
    try noteAll(&l, &.{ 3, 5, 2 });
    const mid = l.touching(0, 5);
    try testing.expectEqual(@as(usize, 2), mid.end);
    try testing.expectEqual(@as(usize, 3), mid.tail);
    const edge = l.touching(0, 8);
    try testing.expectEqual(@as(usize, 2), edge.end);
    try testing.expectEqual(@as(usize, 0), edge.tail);
    // A token with no returned byte is not listed.
    const none = l.touching(3, 3);
    try testing.expectEqual(@as(usize, 0), none.end - none.start);
}

test "a lead and a stop inside one token list that token with both cuts, and a token with no returned byte is not listed" {
    var l: Listing = .{};
    defer l.deinit(testing.allocator);
    try noteAll(&l, &.{10});
    const r = l.touching(2, 7);
    try testing.expectEqual(@as(usize, 1), r.end - r.start);
    try testing.expectEqual(@as(usize, 2), r.head);
    try testing.expectEqual(@as(usize, 3), r.tail);
    // Nothing of the token is returned: it is not listed.
    const gone = l.touching(5, 5);
    try testing.expectEqual(@as(usize, 0), gone.end - gone.start);
}

test "a trailing run the content arm withholds and a trimmed reply never ships is not listed" {
    var l: Listing = .{};
    defer l.deinit(testing.allocator);
    // "answer " then " more": the space after "answer" waits for text, and the reply ends trimmed.
    try noteAll(&l, &.{7});
    l.released = 7;
    const waiting = l.sent(1, 1);
    try testing.expectEqual(@as(usize, 0), waiting.end - waiting.start);
    l.ended = true;
    const last = l.late(1);
    try testing.expectEqual(@as(usize, 1), last.end - last.start);
    try testing.expectEqual(@as(usize, 1), last.tail);
}
