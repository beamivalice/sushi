//! Token ids of prompt segments, per loaded model. A segment is the text between two specials the encoder
//! recognised (`Tokenizer.encodeSpans`): its ids are a function of its bytes alone, so an agent turn, which renders
//! every earlier turn again, encodes only what it has not seen. Bounded in bytes, least recently used out first;
//! every thread that formats a prompt shares it under one lock. Pageable host memory: reported in `cache_stats`,
//! never billed against the wired limit (docs/engine-memory-admission.md).
const std = @import("std");
const cache_stats = @import("cache_stats.zig");

/// What one model's cache may hold: segment bytes, their ids and each entry's bookkeeping.
pub const default_budget: usize = 64 << 20;
/// A shorter segment encodes faster than a lookup costs.
pub const min_segment: usize = 32;
/// One map slot: the key slice, the node pointer and the slot's metadata byte.
const slot_bytes: usize = @sizeOf([]const u8) + @sizeOf(*anyopaque) + 1;
/// Bookkeeping per entry, billed beside its bytes and ids: its node, and eight map slots. The map keeps at most
/// four slots per entry (it grows at 80% load to a power of two, and `rebuild` shrinks it), and while it grows or is
/// rebuilt the old and new slot arrays live together.
const entry_overhead: usize = @sizeOf(SegmentCache.Node) + 8 * slot_bytes;

pub const SegmentCache = struct {
    gpa: std.mem.Allocator,
    budget: usize,
    mu: std.Io.Mutex = .init,
    map: std.StringHashMapUnmanaged(*Node) = .empty,
    /// Most recently used first.
    head: ?*Node = null,
    tail: ?*Node = null,
    used: usize = 0,
    /// Entries removed since the map was last built: each left a tombstone that a miss probes past.
    dead: usize = 0,

    const Node = struct { key: []u8, ids: []u32, prev: ?*Node = null, next: ?*Node = null };

    pub fn init(gpa: std.mem.Allocator, budget: usize) SegmentCache {
        return .{ .gpa = gpa, .budget = budget };
    }

    pub fn deinit(self: *SegmentCache) void {
        self.clear();
        self.map.deinit(self.gpa);
    }

    fn lock(self: *SegmentCache) std.Io {
        const io = std.Io.Threaded.global_single_threaded.io();
        self.mu.lockUncancelable(io);
        return io;
    }

    fn cost(key_len: usize, ids_len: usize) usize {
        return key_len + ids_len * @sizeOf(u32) + entry_overhead;
    }

    /// Appends the ids of `segment` to `out` and says so, or leaves `out` as it was on a miss.
    pub fn appendIds(self: *SegmentCache, out_gpa: std.mem.Allocator, out: *std.ArrayList(u32), segment: []const u8) !bool {
        // Never kept, so never looked up or counted.
        if (cost(segment.len, 0) > self.budget / 16) return false;
        const io = self.lock();
        defer self.mu.unlock(io);
        const node = self.map.get(segment) orelse {
            cache_stats.process.recordTokenLookup(false);
            return false;
        };
        try out.appendSlice(out_gpa, node.ids);
        self.unlink(node);
        self.pushFront(node);
        cache_stats.process.recordTokenLookup(true);
        return true;
    }

    /// Keeps `segment`'s `ids`; a segment already held, or one too large for the budget, is left out.
    pub fn put(self: *SegmentCache, segment: []const u8, ids: []const u32) !void {
        const need = cost(segment.len, ids.len);
        // One entry may take a sixteenth of the budget: no single segment a client sends empties the cache.
        if (need > self.budget / 16) return;
        const io = self.lock();
        defer self.mu.unlock(io);
        if (self.map.contains(segment)) return;
        while (self.used + need > self.budget) self.drop(self.tail.?, true);
        if (self.dead > self.map.count() / 4 + 32 or self.map.capacity() > 4 * self.map.count() + 64) self.rebuild();
        const node = try self.gpa.create(Node);
        errdefer self.gpa.destroy(node);
        const key = try self.gpa.dupe(u8, segment);
        errdefer self.gpa.free(key);
        const owned = try self.gpa.dupe(u32, ids);
        errdefer self.gpa.free(owned);
        try self.map.put(self.gpa, key, node);
        node.* = .{ .key = key, .ids = owned };
        self.pushFront(node);
        self.grow(need);
    }

    /// The map built again from the live entries: no tombstones, and no more slots than they need.
    fn rebuild(self: *SegmentCache) void {
        var fresh: std.StringHashMapUnmanaged(*Node) = .empty;
        // Short of memory, the old map serves on; the next put tries again.
        fresh.ensureTotalCapacity(self.gpa, self.map.count() + 1) catch return;
        var it = self.head;
        while (it) |node| : (it = node.next) fresh.putAssumeCapacityNoClobber(node.key, node);
        self.map.deinit(self.gpa);
        self.map = fresh;
        self.dead = 0;
    }

    pub fn clear(self: *SegmentCache) void {
        while (self.tail) |node| self.drop(node, false);
    }

    /// Bytes held now.
    pub fn bytes(self: *SegmentCache) usize {
        const io = self.lock();
        defer self.mu.unlock(io);
        return self.used;
    }

    fn grow(self: *SegmentCache, n: usize) void {
        self.used += n;
        cache_stats.process.addTokenBytes(@intCast(n));
    }

    fn drop(self: *SegmentCache, node: *Node, evicted: bool) void {
        const n = cost(node.key.len, node.ids.len);
        _ = self.map.remove(node.key);
        self.dead += 1;
        self.unlink(node);
        self.used -= n;
        cache_stats.process.addTokenBytes(-@as(i64, @intCast(n)));
        if (evicted) cache_stats.process.recordTokenEviction();
        self.gpa.free(node.key);
        self.gpa.free(node.ids);
        self.gpa.destroy(node);
    }

    fn unlink(self: *SegmentCache, node: *Node) void {
        if (node.prev) |p| p.next = node.next else self.head = node.next;
        if (node.next) |n| n.prev = node.prev else self.tail = node.prev;
        node.prev = null;
        node.next = null;
    }

    fn pushFront(self: *SegmentCache, node: *Node) void {
        node.next = self.head;
        if (self.head) |h| h.prev = node else self.tail = node;
        self.head = node;
    }
};

const testing = std.testing;

fn idsOf(c: *SegmentCache, segment: []const u8) !?[]u32 {
    var out = std.ArrayList(u32).empty;
    errdefer out.deinit(testing.allocator);
    if (!try c.appendIds(testing.allocator, &out, segment)) return null;
    return try out.toOwnedSlice(testing.allocator);
}

test "segment cache: what it holds never passes its budget, whatever the client sends" {
    const one = SegmentCache.cost(40, 10);
    var c = SegmentCache.init(testing.allocator, one * 16);
    defer c.deinit();
    var key: [40]u8 = undefined;
    for (0..1000) |i| {
        @memset(&key, 'a');
        _ = std.fmt.bufPrint(&key, "{d}", .{i}) catch unreachable;
        try c.put(&key, &(@as([10]u32, @splat(@intCast(i)))));
        try testing.expect(c.bytes() <= c.budget);
    }
    // A segment past a sixteenth of the budget is not kept, and pushes nothing out.
    const held = c.bytes();
    const huge = try testing.allocator.alloc(u8, one);
    defer testing.allocator.free(huge);
    @memset(huge, 'z');
    try c.put(huge, &.{1});
    try testing.expectEqual(held, c.bytes());
    try testing.expect((try idsOf(&c, huge)) == null);
}

test "segment cache: a full cache drops the segment used least recently, and a hit counts as a use" {
    const one = SegmentCache.cost(2, 1);
    var c = SegmentCache.init(testing.allocator, one * 16);
    defer c.deinit();
    var keys: [17][2]u8 = undefined;
    for (&keys, 0..) |*k, i| k.* = .{ 'k', @intCast('A' + i) };
    for (keys[0..16], 0..) |*k, i| try c.put(k, &.{@intCast(i)});
    const first = (try idsOf(&c, &keys[0])).?;
    defer testing.allocator.free(first);
    try testing.expectEqualSlices(u32, &.{0}, first);
    try c.put(&keys[16], &.{16});
    // The second was used least recently: it goes; the first, touched by the hit, stays, and so does the rest.
    try testing.expect((try idsOf(&c, &keys[1])) == null);
    for ([_]usize{ 0, 2, 15, 16 }) |i| {
        const got = (try idsOf(&c, &keys[i])).?;
        defer testing.allocator.free(got);
        try testing.expectEqualSlices(u32, &.{@intCast(i)}, got);
    }
}

test "segment cache: clear drops everything and gives the bytes back to the gauge" {
    const before = cache_stats.process.snapshot().token_bytes;
    var c = SegmentCache.init(testing.allocator, 1 << 20);
    try c.put("segment one", &.{ 1, 2 });
    try c.put("segment two", &.{3});
    try testing.expect(cache_stats.process.snapshot().token_bytes > before);
    c.deinit();
    try testing.expectEqual(before, cache_stats.process.snapshot().token_bytes);
}

test "segment cache: a miss stays a short probe after the cache has turned over many times" {
    const one = SegmentCache.cost(32, 1);
    var c = SegmentCache.init(testing.allocator, one * 4096);
    defer c.deinit();
    var key: [32]u8 = undefined;
    for (0..200_000) |i| {
        _ = std.fmt.bufPrint(&key, "{d:0>32}", .{i}) catch unreachable;
        try c.put(&key, &.{@intCast(i & 0xffff)});
    }
    // A lookup probes until a free slot: a map whose removals left only tombstones scans all of it on every miss.
    const cap = c.map.capacity();
    var free: usize = 0;
    for (c.map.metadata.?[0..cap]) |m| free += @intFromBool(m.isFree());
    try testing.expect(free >= cap / 4);
}

/// Counts the bytes its child holds for the cache, as requested.
const Counting = struct {
    child: std.mem.Allocator,
    live: usize = 0,
    peak: usize = 0,

    fn allocator(self: *Counting) std.mem.Allocator {
        return .{ .ptr = self, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }
    fn note(self: *Counting, add: usize, sub: usize) void {
        self.live = self.live + add - sub;
        self.peak = @max(self.peak, self.live);
    }
    fn alloc(ctx: *anyopaque, len: usize, al: std.mem.Alignment, ra: usize) ?[*]u8 {
        const self: *Counting = @ptrCast(@alignCast(ctx));
        const p = self.child.rawAlloc(len, al, ra) orelse return null;
        self.note(len, 0);
        return p;
    }
    fn resize(ctx: *anyopaque, mem: []u8, al: std.mem.Alignment, new_len: usize, ra: usize) bool {
        const self: *Counting = @ptrCast(@alignCast(ctx));
        if (!self.child.rawResize(mem, al, new_len, ra)) return false;
        self.note(new_len, mem.len);
        return true;
    }
    fn remap(ctx: *anyopaque, mem: []u8, al: std.mem.Alignment, new_len: usize, ra: usize) ?[*]u8 {
        const self: *Counting = @ptrCast(@alignCast(ctx));
        const p = self.child.rawRemap(mem, al, new_len, ra) orelse return null;
        self.note(new_len, mem.len);
        return p;
    }
    fn free(ctx: *anyopaque, mem: []u8, al: std.mem.Alignment, ra: usize) void {
        const self: *Counting = @ptrCast(@alignCast(ctx));
        self.child.rawFree(mem, al, ra);
        self.note(0, mem.len);
    }
};

test "segment cache: everything it allocates stays within its budget, the map's own slots included" {
    var counting: Counting = .{ .child = testing.allocator };
    const budget: usize = 4 << 20;
    var c = SegmentCache.init(counting.allocator(), budget);
    // Many short segments grow the map; long ones after them must not find the map's slots unbilled.
    var key: [32]u8 = undefined;
    for (0..60_000) |i| {
        _ = std.fmt.bufPrint(&key, "{d:0>32}", .{i}) catch unreachable;
        try c.put(&key, &.{@intCast(i & 0xffff)});
    }
    var big: [3072]u8 = undefined;
    const ids: [768]u32 = @splat(7);
    for (0..3_000) |i| {
        @memset(&big, 'x');
        _ = std.fmt.bufPrint(&big, "{d}", .{i}) catch unreachable;
        try c.put(&big, &ids);
    }
    const peak = counting.peak;
    c.deinit();
    try testing.expectEqual(@as(usize, 0), counting.live);
    try testing.expect(peak <= budget);
}
