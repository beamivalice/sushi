//! Which of a generation's tokens a reply lists logprobs for: the tokens whose bytes the reply returned. A stream
//! drains this as its text goes out and a non-stream reply reads it once; both ask `Listing`, so a stop's own tokens
//! and a thought the reply does not ship are listed on neither. A token the reply returned only part of (a stop or a
//! trimmed lead inside it) is listed with just those bytes: `head` and `tail` are what its entry drops. A reply's text
//! can hold bytes no token produced (the "\n" that joins its content parts around a thought): no entry lists them, so
//! the listed bytes join to the returned content less exactly those joins.
const std = @import("std");

/// The bytes that join two content parts of a reply around a thought (`think_tags.normalizeEmbeddedThinkBlocks`, `ContentStream`).
pub const content_join = "\n";

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

/// Bytes `len` long at `at` inside the token `token` that the reply cut out of its content (a think closer between two
/// returned runs of one token): the token lists once, without them.
pub const Gap = struct { token: usize, at: usize, len: usize };

/// Where a rewritten reply text got its content bytes (`think_tags.Normalized`): runs of the text in order, each copied from
/// the generation at `src`, or, with no `src`, a join no token carries and no entry lists. A byte outside every run is not content (the
/// rewrite's framing and the thoughts it gathered), and a list of the text's content never reaches one.
pub const Run = struct { at: usize, len: usize, src: ?usize };

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
    /// Parts of the list already decided, in order, ahead of the tokens still being drained: a content part that ended.
    /// A stream drains them first (`items`); a non-stream reply reads them once (`queued`).
    queue: std.ArrayList(Range) = .empty,
    /// Raw bytes the reply drops from the middle of its content (a stray think closer): a token wholly inside lists nothing.
    holes: std.ArrayList(Hole) = .empty,
    /// The bytes inside a token the reply returned on both sides of, in order (`returned`).
    gaps: std.ArrayList(Gap) = .empty,
    /// Where the last returned bytes ended.
    returned_to: usize = 0,
    /// What `items` returned last.
    out: std.ArrayList(Range) = .empty,

    const Hole = struct { from: usize, to: usize };

    pub fn deinit(self: *Listing, allocator: std.mem.Allocator) void {
        self.starts.deinit(allocator);
        self.queue.deinit(allocator);
        self.holes.deinit(allocator);
        self.gaps.deinit(allocator);
        self.out.deinit(allocator);
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

    /// The reply returned the generation's bytes `from..to`: they list after what is already queued. A token the bytes
    /// before them also touched is listed once, with what lies between cut out (`gaps`).
    pub fn returned(self: *Listing, allocator: std.mem.Allocator, from: usize, to: usize) !void {
        var r = self.touching(from, to);
        defer self.returned_to = to;
        if (r.end <= r.start or self.queue.items.len == 0) return self.queueRange(allocator, r);
        const last = &self.queue.items[self.queue.items.len - 1];
        const shared = last.end > last.start and last.end - 1 == r.start and self.returned_to <= from and self.returned_to > self.starts.items[r.start];
        if (!shared) return self.queueRange(allocator, r);
        const token_at = self.starts.items[r.start];
        if (from > self.returned_to) try self.gaps.append(allocator, .{ .token = r.start, .at = self.returned_to - token_at, .len = from - self.returned_to });
        if (r.end == r.start + 1) {
            last.tail = r.tail;
            return;
        }
        last.tail = 0;
        r.start += 1;
        r.head = 0;
        try self.queueRange(allocator, r);
    }

    /// The gaps inside token `token`, in order, of the `gaps` a listing found.
    pub fn gapsOf(gaps: []const Gap, token: usize) []const Gap {
        var from: usize = 0;
        while (from < gaps.len and gaps[from].token < token) from += 1;
        var to = from;
        while (to < gaps.len and gaps[to].token == token) to += 1;
        return gaps[from..to];
    }

    /// The reply returned the bytes `from..to` of its text. `runs` says where a rewrite of the generation got them (null: the
    /// text is the generation's own); `base` is the generation offset the text, or the bytes `runs` copy from, starts at.
    pub fn returnedText(self: *Listing, allocator: std.mem.Allocator, runs: ?[]const Run, base: usize, from: usize, to: usize) !void {
        const rs = runs orelse return self.returned(allocator, base + from, base + to);
        for (rs) |run| {
            const lo = @max(from, run.at);
            const hi = @min(to, run.at + run.len);
            if (lo >= hi) continue;
            // A join has no token and no entry.
            if (run.src) |src| try self.returned(allocator, base + src + (lo - run.at), base + src + (hi - run.at));
        }
    }

    /// What the reply returned so far, in order (`returned`).
    pub fn queued(self: *const Listing) []const Range {
        return self.queue.items;
    }

    /// The content part being drained ends where its last sent byte does: what it returned lists, ahead of whatever follows,
    /// and the bytes it withheld (a trimmed tail) and the thought opening after it never do.
    pub fn endPart(self: *Listing, allocator: std.mem.Allocator) !void {
        const was = self.ended;
        self.ended = true;
        const r = self.drain(self.starts.items.len);
        self.ended = was;
        try self.queueRange(allocator, r);
    }

    /// The reply drops the raw bytes `from..to` from its content: a token wholly inside lists nothing.
    pub fn hide(self: *Listing, allocator: std.mem.Allocator, from: usize, to: usize) !void {
        if (from < to) try self.holes.append(allocator, .{ .from = from, .to = to });
    }

    fn hidden(self: *const Listing, token: usize) bool {
        const from = self.starts.items[token];
        const to = self.end_of(token);
        for (self.holes.items) |h| if (h.from <= from and to <= h.to) return true;
        return false;
    }

    /// `r` into `dest` as the runs of tokens a hole does not cut out of it.
    fn splitHoles(self: *const Listing, allocator: std.mem.Allocator, r: Range, dest: *std.ArrayList(Range)) !void {
        if (r.end <= r.start) return;
        var from = r.start;
        for (r.start..r.end + 1) |i| {
            if (i < r.end and !self.hidden(i)) continue;
            if (i > from) try dest.append(allocator, .{
                .start = from,
                .end = i,
                .head = if (from == r.start) r.head else 0,
                .tail = if (i == r.end) r.tail else 0,
            });
            from = i + 1;
        }
    }

    fn queueRange(self: *Listing, allocator: std.mem.Allocator, r: Range) !void {
        try self.splitHoles(allocator, r, &self.queue);
    }

    /// The runs of tokens a chat chunk going out now lists, in order, valid until the next call: the parts decided since the last chunk
    /// whose entries have landed (`published`), then, once they all have, the tokens of the content being drained
    /// (`sent` for a content chunk carrying all released bytes but the last `withheld`, `late` for one with none).
    pub fn items(self: *Listing, allocator: std.mem.Allocator, published: usize, content_chunk: bool, withheld: usize) ![]const Range {
        self.out.clearRetainingCapacity();
        var n: usize = 0;
        while (n < self.queue.items.len) : (n += 1) {
            if (self.queue.items[n].end > published) break;
            try self.out.append(allocator, self.queue.items[n]);
        }
        self.queue.replaceRangeAssumeCapacity(0, n, &.{});
        if (content_chunk) self.content_to = self.released -| withheld;
        if (self.queue.items.len == 0) try self.splitHoles(allocator, self.drain(published), &self.out);
        self.pruneHoles();
        return self.out.items;
    }

    fn pruneHoles(self: *Listing) void {
        const keep_from = if (self.emitted < self.starts.items.len) self.starts.items[self.emitted] else self.total;
        var n: usize = 0;
        for (self.holes.items) |h| {
            if (h.to <= keep_from) continue;
            self.holes.items[n] = h;
            n += 1;
        }
        self.holes.items.len = n;
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

test "a token the reply returned on both sides of cut bytes lists once, with the cut bytes as its gap" {
    var l: Listing = .{};
    defer l.deinit(testing.allocator);
    try noteAll(&l, &.{ 2, 10, 1 });
    try l.returned(testing.allocator, 0, 3);
    try l.returned(testing.allocator, 7, 12);
    try testing.expectEqual(@as(usize, 1), l.queued().len);
    try testing.expectEqual(@as(usize, 2), l.queued()[0].end);
    try testing.expectEqual(@as(usize, 0), l.queued()[0].tail);
    try testing.expectEqual(@as(usize, 1), l.gaps.items.len);
    try testing.expectEqual(Gap{ .token = 1, .at = 1, .len = 4 }, l.gaps.items[0]);
    // Bytes that continue into the next token do not share the first.
    try l.returned(testing.allocator, 12, 13);
    try testing.expectEqual(@as(usize, 2), l.queued().len);
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

test "a part that ends lists its returned bytes ahead of the part after, the withheld tail and a thought never" {
    var l: Listing = .{};
    defer l.deinit(testing.allocator);
    // "Sure.\n" (the "\n" withheld), "<think>", "hmm", "</think>", "Done.": 6 + 7 + 3 + 8 + 5 bytes.
    try noteAll(&l, &.{ 6, 7, 3, 8, 5 });
    l.released = 6;
    _ = l.sent(1, 1);
    l.released = 13;
    try l.endPart(testing.allocator);
    l.retire();
    l.released = 24;
    l.retire();
    l.released = 29;
    const items = try l.items(testing.allocator, 5, true, 0);
    try testing.expectEqual(@as(usize, 2), items.len);
    try testing.expectEqual(@as(usize, 0), items[0].start);
    try testing.expectEqual(@as(usize, 1), items[0].end);
    try testing.expectEqual(@as(usize, 1), items[0].tail);
    try testing.expectEqual(@as(usize, 4), items[1].start);
    try testing.expectEqual(@as(usize, 5), items[1].end);
}

test "a token wholly inside a hole lists nothing, the tokens around it list as runs with the cuts at the ends" {
    var l: Listing = .{};
    defer l.deinit(testing.allocator);
    try noteAll(&l, &.{ 3, 8, 4 });
    try l.hide(testing.allocator, 3, 11);
    l.released = 15;
    const items = try l.items(testing.allocator, 3, true, 0);
    try testing.expectEqual(@as(usize, 2), items.len);
    try testing.expectEqual(@as(usize, 1), items[0].end);
    try testing.expectEqual(@as(usize, 2), items[1].start);
}

test "a text rewritten by runs lists its content through them: a join lists nothing, a run of the generation its tokens" {
    var l: Listing = .{};
    defer l.deinit(testing.allocator);
    // "Sure." "<think>" "hmm" "</think>" "Done." as the rewrite "<think>\nhmm</think>\nSure.\nDone.".
    try noteAll(&l, &.{ 5, 7, 3, 8, 5 });
    const runs = [_]Run{ .{ .at = 20, .len = 5, .src = 0 }, .{ .at = 25, .len = 1, .src = null }, .{ .at = 26, .len = 5, .src = 23 } };
    try l.returnedText(testing.allocator, &runs, 0, 20, 31);
    const items = l.queued();
    try testing.expectEqual(@as(usize, 2), items.len);
    try testing.expectEqual(@as(usize, 0), items[0].start);
    try testing.expectEqual(@as(usize, 1), items[0].end);
    try testing.expectEqual(@as(usize, 4), items[1].start);
}
