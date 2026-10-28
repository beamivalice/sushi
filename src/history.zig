//! Request-owned proposal scratch over leased immutable external history.
const std = @import("std");
const tokens = @import("token_history.zig");
const lookup = @import("mtp_lookup.zig");

pub const Request = struct {
    allocator: std.mem.Allocator,
    drafter: tokens.Drafter,
    corpus: ?tokens.CorpusLease = null,
    persistent: ?tokens.PersistentLease = null,
    output: std.ArrayList(u32) = .empty,
    query: [tokens.max_order]u32 = undefined,
    query_len: usize = 0,
    query_cap: u32 = 0,
    matched_order: u32 = 0,
    queried: bool = false,

    pub fn init(io: std.Io, allocator: std.mem.Allocator, cache: *tokens.HistoryCache, config: tokens.Config, identity: ?[32]u8, vocab_limit: u32) !Request {
        const wanted = identity orelse return error.TokenizerRequired;
        var self: Request = .{ .allocator = allocator, .drafter = tokens.Drafter.init(allocator) };
        errdefer self.deinit();
        self.drafter.vocab_limit = vocab_limit;
        if (config.lookup_corpus) |path| {
            self.corpus = try cache.acquireCorpus(io, path, wanted);
            if (self.corpus) |*lease| self.drafter.corpus = lease.index();
        }
        if (config.lookup_store) |path| {
            if (config.lookup_store_write) try tokens.initializePersistent(io, allocator, path, wanted);
            self.persistent = try cache.acquirePersistent(io, path, wanted);
            self.drafter.persistent = self.persistent.?.index();
        }
        return self;
    }

    pub fn deinit(self: *Request) void {
        self.drafter.deinit();
        tokens.releaseBytes(self.output.capacity * @sizeOf(u32));
        self.output.deinit(self.allocator);
        if (self.corpus) |*lease| lease.deinit();
        if (self.persistent) |*lease| lease.deinit();
        self.* = undefined;
    }

    /// Result borrows request scratch until the next different query.
    pub fn match(self: *Request, committed: []const u32, t1: u32, cap: u32, lines: anytype) !?lookup.Match {
        const n: usize = @min(committed.len, tokens.max_order - 1);
        var query: [tokens.max_order]u32 = undefined;
        @memcpy(query[0..n], committed[committed.len - n ..]);
        query[n] = t1;
        const length: usize = n + 1;
        if (!self.queried or self.query_cap != cap or self.query_len != length or
            !std.mem.eql(u32, self.query[0..self.query_len], query[0..length]))
        {
            const previous_capacity = self.output.capacity;
            defer tokens.retainBytes((self.output.capacity - previous_capacity) * @sizeOf(u32));
            const got = try self.drafter.look(query[0..length], cap, &self.output);
            @memcpy(self.query[0..length], query[0..length]);
            self.query_len = length;
            self.query_cap = cap;
            self.matched_order = got.order;
            self.queried = true;
        }
        if (self.output.items.len == 0) return null;
        const order = @min(self.matched_order, length);
        var crosses_line = false;
        if (order > 1) for (query[length - order .. length - 1]) |token| {
            crosses_line = crosses_line or lines.hasNewline(token);
        };
        return .{ .draft = self.output.items, .suffix = self.matched_order, .source = .history, .crosses_line = crosses_line };
    }
};

const TestLines = struct {
    newline: u32 = 0,
    fn hasNewline(self: TestLines, token: u32) bool {
        return token == self.newline;
    }
};

test "history ordinary suffix does not invent line-crossing agreement" {
    const a = std.testing.allocator;
    var corpus = try tokens.Corpus.build(a, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, tokens.separator }, 64);
    defer corpus.deinit();
    var request = Request{ .allocator = a, .drafter = tokens.Drafter.init(a) };
    defer request.deinit();
    request.drafter.corpus = &corpus;
    const got = (try request.match(&.{ 1, 2, 3, 4, 5, 6, 7 }, 8, 2, TestLines{})).?;
    try std.testing.expectEqual(@as(u32, 8), got.suffix);
    try std.testing.expect(!got.crosses_line);
    const costs: lookup.Costs = .{ .mtp = 100, .lookup = @splat(1) };
    try std.testing.expectEqual(@as(u32, 0), lookup.gate(got, 8, 7, 1, false, costs));
    const across = (try request.match(&.{ 1, 2, 3, 4, 5, 6, 7 }, 8, 2, TestLines{ .newline = 3 })).?;
    try std.testing.expect(across.crosses_line);
    try std.testing.expectEqual(@as(u32, 2), lookup.gate(across, 8, 7, 1, false, costs));
    const anchor_break = (try request.match(&.{ 1, 2, 3, 4, 5, 6, 7 }, 8, 2, TestLines{ .newline = 8 })).?;
    try std.testing.expect(!anchor_break.crosses_line);
    try std.testing.expectEqual(@as(u32, 0), lookup.gate(anchor_break, 8, 7, 1, false, costs));
}

test "history request borrows snapshots across B-before-A publication and cache teardown" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var buffer: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const path = buffer[0..try tmp.dir.realPath(io, &buffer)];
    const identity: [32]u8 = @splat(7);
    try tokens.appendFinalTurn(io, a, path, identity, &.{ 1, 2, 3 }, &.{ 4, 5 });
    var cache = tokens.HistoryCache.init(a);
    const config: tokens.Config = .{ .lookup_store = path };
    var request_a = try Request.init(io, a, &cache, config, identity, 100);
    defer request_a.deinit();
    var request_b = try Request.init(io, a, &cache, config, identity, 100);
    defer request_b.deinit();
    try cache.appendFinalTurn(io, path, identity, &.{ 11, 12 }, &.{13});
    try cache.appendFinalTurn(io, path, identity, &.{ 21, 22 }, &.{23});
    var current = try cache.acquirePersistent(io, path, identity);
    defer current.deinit();
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3, 4, 5, tokens.separator, 11, 12, 13, tokens.separator, 21, 22, 23, tokens.separator }, current.index().tokens);
    try cache.deinit();
    try std.testing.expectEqualSlices(u32, &.{ 4, 5 }, (try request_a.match(&.{ 1, 2 }, 3, 8, TestLines{})).?.draft);
    try std.testing.expect((try request_b.match(&.{11}, 12, 8, TestLines{})) == null);
}

test "history request keeps the pending token below at and beyond maximum query order" {
    const a = std.testing.allocator;
    var document: [101]u32 = undefined;
    for (document[0..100], 0..) |*token, i| token.* = @intCast(i + 1);
    document[100] = tokens.separator;
    var corpus = try tokens.Corpus.build(a, &document, tokens.max_order);
    defer corpus.deinit();
    var request = Request{ .allocator = a, .drafter = tokens.Drafter.init(a) };
    defer request.deinit();
    request.drafter.corpus = &corpus;
    for ([_]usize{ 62, 63, 64, 80 }) |committed| {
        const got = (try request.match(document[0..committed], document[committed], 2, TestLines{})).?;
        try std.testing.expectEqual(@as(u32, @intCast(@min(committed + 1, tokens.max_order))), got.suffix);
        try std.testing.expectEqualSlices(u32, document[committed + 1 ..][0..2], got.draft);
    }
}
