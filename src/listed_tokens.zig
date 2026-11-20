//! Which of a generation's tokens a reply lists logprobs for: the tokens whose bytes the reply returned. A stream
//! drains this as its text goes out and a non-stream reply reads it once; both ask `Listing`, so a stop's own tokens
//! and a thought the reply does not ship are listed on neither.
const std = @import("std");

/// Tokens `start..end` (indexes into the generation).
pub const Range = struct { start: usize, end: usize };

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
    /// The first token not yet shipped or retired.
    emitted: usize = 0,

    pub fn deinit(self: *Listing, allocator: std.mem.Allocator) void {
        self.starts.deinit(allocator);
    }

    pub fn note(self: *Listing, allocator: std.mem.Allocator, len: usize) !void {
        try self.starts.append(allocator, self.total);
        self.total += len;
    }

    /// The tokens that put a byte inside `from..to`: one straddling either edge is in.
    pub fn touching(self: *const Listing, from: usize, to: usize) Range {
        const start = self.firstEndingAfter(from);
        if (from >= to) return .{ .start = start, .end = start };
        return .{ .start = start, .end = @max(start, self.firstStartingAtOrAfter(to)) };
    }

    /// The generation's own text from `from` is the reply's content: tokens wholly before it are never listed.
    pub fn skipTo(self: *Listing, from: usize) void {
        self.emitted = @max(self.emitted, self.firstEndingAfter(from));
    }

    /// Nothing released so far is content: retire it unlisted.
    pub fn retire(self: *Listing) void {
        self.emitted = @max(self.emitted, self.firstStartingAtOrAfter(self.released));
    }

    /// A content chunk carrying everything released went out. The tokens it lists are those up to `published`
    /// (a token's entry can lag its id).
    pub fn sent(self: *Listing, published: usize) Range {
        self.content_to = self.released;
        return self.drain(published);
    }

    /// The tokens of content already sent whose entries were not yet published when it went out.
    pub fn late(self: *Listing, published: usize) Range {
        return self.drain(published);
    }

    fn drain(self: *Listing, published: usize) Range {
        const start = self.emitted;
        const end = @min(published, self.firstStartingAtOrAfter(self.content_to));
        if (end <= start) return .{ .start = start, .end = start };
        self.emitted = end;
        return .{ .start = start, .end = end };
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
