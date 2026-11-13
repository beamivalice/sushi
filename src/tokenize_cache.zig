//! Per-LoadedModel LRU cache for `chat_mod.formatChat` results.
//!
//! Phase 4 #3 win: the long-prompt warm-reuse case spends ~240 ms re-
//! rendering and re-tokenizing a 1813-token prompt on every turn, only to
//! discover the KV cache already has it. The Metal prefill on a full hit
//! is ~30 ms, so tokenize is 7× the actual work. Memoizing the tokenize
//! result over a small LRU drops the second hit to a clone of the cached
//! buffer — sub-millisecond regardless of prompt length.
//!
//! The key is the rendered chat-template inputs written as bytes, found by their Wyhash digest and matched by the
//! bytes themselves, so two prompts whose digests collide never share tokens:
//!     - per message: role, content
//!     - per tool_call: name + arguments
//!     - tools_json (raw bytes; null contributes a sentinel)
//!     - tool_choice_instruction
//!     - enable_thinking flag
//!     - reasoning_effort string (dsv4 templates render it)
//!     - preserve_thinking (Qwen3.8 drops prior-turn thinking when false)
//! Image-bearing messages are intentionally excluded — the vision pipeline
//! re-injects per-request token positions and any cache hit there would
//! point at stale token IDs.
//!
//! Thread safety: a single mutex guards entries. HTTP handlers each run
//! on their own thread; lookups + inserts are short (memcpy + map ops),
//! so a coarse-grained lock is fine.

const std = @import("std");
const chat_mod = @import("chat.zig");

pub const TokenizeCache = struct {
    allocator: std.mem.Allocator,
    /// Maximum entries kept resident. Past this, the least-recently-used
    /// entry is evicted on the next insert. 0 disables the cache entirely.
    capacity: u32,
    /// `std.Io.Mutex` — the codebase uses this everywhere for cross-thread
    /// locking (HTTP request threads consult the cache while the inference
    /// thread services them). `.init` is the default unlocked state.
    mutex: std.Io.Mutex = .init,
    entries: std.ArrayListUnmanaged(Entry) = .empty,
    /// Monotonic counter; each lookup/insert bumps the matching entry's
    /// `last_used` so LRU eviction is well-defined under any ordering.
    counter: u64 = 0,

    pub const Entry = struct {
        key: u64,
        key_bytes: []u8,
        tokens: []u32,
        last_used: u64,
    };

    /// A render's inputs as bytes (`keyFor`), and their digest.
    pub const Key = struct {
        hash: u64,
        bytes: []u8,

        fn matches(self: Key, e: *const Entry) bool {
            return e.key == self.hash and std.mem.eql(u8, e.key_bytes, self.bytes);
        }
    };

    pub fn init(allocator: std.mem.Allocator, capacity: u32) TokenizeCache {
        return .{ .allocator = allocator, .capacity = capacity };
    }

    pub fn deinit(self: *TokenizeCache) void {
        for (self.entries.items) |*e| {
            self.allocator.free(e.tokens);
            self.allocator.free(e.key_bytes);
        }
        self.entries.deinit(self.allocator);
    }

    /// Compute the cache key for a chat-template render. Media pixels never
    /// reach the rendered text; where each media part renders does. The caller
    /// frees `bytes`.
    pub fn keyFor(
        allocator: std.mem.Allocator,
        messages: []const chat_mod.Message,
        tools_json: ?[]const u8,
        tool_choice_instruction: ?[]const u8,
        enable_thinking: bool,
        reasoning_effort: ?[]const u8,
        continue_final: bool,
        preserve_thinking: ?bool,
    ) error{OutOfMemory}!Key {
        var list = std.ArrayList(u8).empty;
        errdefer list.deinit(allocator);
        const h: Writer = .{ .list = &list, .allocator = allocator };
        for (messages) |m| {
            try h.field(m.role);
            try h.update("\x1f"); // unit separator — keeps role/content boundary unambiguous
            try h.field(m.content);
            try h.update("\x1e"); // record separator
            if (m.tool_call_id) |id| try h.field(id);
            try h.update("\x1e");
            if (m.reasoning_content) |rc| try h.field(rc);
            try h.update("\x1e");
            if (m.media_parts) |parts| for (parts) |part| {
                try h.update(std.mem.asBytes(&part.at));
                try h.field(@tagName(part.kind));
            };
            try h.update("\x1e");
            if (m.tool_calls) |tcs| {
                for (tcs) |tc| {
                    try h.field(tc.id);
                    try h.field(tc.name);
                    try h.update("\x1f");
                    try h.field(tc.arguments);
                    try h.update("\x1f");
                }
            }
            try h.update("\x1d"); // group separator between messages
        }
        if (tools_json) |t| try h.field(t) else try h.update("(no-tools)");
        try h.update("\x1e");
        if (tool_choice_instruction) |t| try h.field(t) else try h.update("(no-tc)");
        try h.update("\x1e");
        try h.update(if (enable_thinking) "thinking=on" else "thinking=off");
        try h.update("\x1e");
        // The effort string changes the rendered prompt for templates that
        // map it (dsv4's high/max preamble) — two requests differing only in
        // effort must not share a cached tokenization.
        if (reasoning_effort) |e| try h.field(e) else try h.update("(no-effort)");
        try h.update("\x1e");
        // A continuation renders the SAME messages into a different prompt —
        // the trailing assistant becomes a prefill instead of history. Without
        // this the two collide in the cache and one of them is served the
        // other's tokens.
        try h.update(if (continue_final) "continue=on" else "continue=off");
        try h.update("\x1e");
        try h.update(if (preserve_thinking) |p| (if (p) "preserve=on" else "preserve=off") else "(no-preserve)");
        const bytes = try list.toOwnedSlice(allocator);
        return .{ .hash = std.hash.Wyhash.hash(0xC0DEC0DE, bytes), .bytes = bytes };
    }

    const Writer = struct {
        list: *std.ArrayList(u8),
        allocator: std.mem.Allocator,

        fn update(w: Writer, bytes: []const u8) error{OutOfMemory}!void {
            try w.list.appendSlice(w.allocator, bytes);
        }

        fn field(w: Writer, value: []const u8) error{OutOfMemory}!void {
            const len: u64 = value.len;
            try w.update(std.mem.asBytes(&len));
            try w.update(value);
        }
    };

    /// Return a freshly-allocated clone of the cached token IDs for `key`,
    /// or null on miss. Caller frees the returned slice.
    pub fn get(self: *TokenizeCache, io: std.Io, key: Key, allocator: std.mem.Allocator) !?[]u32 {
        if (self.capacity == 0) return null;
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        for (self.entries.items) |*e| {
            if (key.matches(e)) {
                self.counter += 1;
                e.last_used = self.counter;
                return try allocator.dupe(u32, e.tokens);
            }
        }
        return null;
    }

    /// Insert `tokens` under `key`, evicting the LRU entry if at capacity.
    /// The cache makes its own copy of `tokens` — caller retains ownership
    /// of the input slice.
    pub fn put(self: *TokenizeCache, io: std.Io, key: Key, tokens: []const u32) !void {
        if (self.capacity == 0) return;
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        // Skip if already present (rare race: two requests with same key
        // both miss + both put). Either copy is fine; keep the first.
        for (self.entries.items) |*e| {
            if (key.matches(e)) {
                self.counter += 1;
                e.last_used = self.counter;
                return;
            }
        }
        const owned = try self.allocator.dupe(u32, tokens);
        errdefer self.allocator.free(owned);
        const key_bytes = try self.allocator.dupe(u8, key.bytes);
        errdefer self.allocator.free(key_bytes);
        if (self.entries.items.len >= self.capacity) {
            // Evict LRU. We always have ≥ 1 entry here because capacity ≥ 1.
            var lru_idx: usize = 0;
            var lru_used: u64 = self.entries.items[0].last_used;
            for (self.entries.items, 0..) |*e, i| {
                if (e.last_used < lru_used) {
                    lru_used = e.last_used;
                    lru_idx = i;
                }
            }
            self.allocator.free(self.entries.items[lru_idx].tokens);
            self.allocator.free(self.entries.items[lru_idx].key_bytes);
            self.entries.items[lru_idx] = .{ .key = key.hash, .key_bytes = key_bytes, .tokens = owned, .last_used = blk: {
                self.counter += 1;
                break :blk self.counter;
            } };
        } else {
            self.counter += 1;
            try self.entries.append(self.allocator, .{ .key = key.hash, .key_bytes = key_bytes, .tokens = owned, .last_used = self.counter });
        }
    }
};

test "TokenizeCache key stability" {
    const allocator = std.testing.allocator;
    const m1 = [_]chat_mod.Message{
        .{ .role = "user", .content = "hello world" },
    };
    const m2 = [_]chat_mod.Message{
        .{ .role = "user", .content = "hello world" },
    };
    const m3 = [_]chat_mod.Message{
        .{ .role = "user", .content = "hello world!" }, // different content
    };
    const k1 = (try testKey(&m1, null, null, false, null, false, null));
    const k2 = (try testKey(&m2, null, null, false, null, false, null));
    const k3 = (try testKey(&m3, null, null, false, null, false, null));
    try std.testing.expectEqual(k1, k2);
    try std.testing.expect(k1 != k3);
    _ = allocator;
}

test "TokenizeCache key distinguishes assistant reasoning_content" {
    // reasoning_content changes the rendered prompt for templates that
    // persist reasoning across turns — two histories differing
    // only in reasoning must not collide on one cached tokenization.
    const m1 = [_]chat_mod.Message{
        .{ .role = "assistant", .content = "4" },
    };
    const m2 = [_]chat_mod.Message{
        .{ .role = "assistant", .content = "4", .reasoning_content = "two plus two is four" },
    };
    const k1 = (try testKey(&m1, null, null, true, null, false, null));
    const k2 = (try testKey(&m2, null, null, true, null, false, null));
    try std.testing.expect(k1 != k2);
}

test "TokenizeCache key distinguishes reasoning_effort" {
    // dsv4 templates render the mapped effort (high/max prepend a preamble) —
    // a request at effort "high" must never hit the tokenization cached for
    // the same messages at the default effort.
    const m = [_]chat_mod.Message{
        .{ .role = "user", .content = "prove it rigorously" },
    };
    const k_default = (try testKey(&m, null, null, true, null, false, null));
    const k_high = (try testKey(&m, null, null, true, "high", false, null));
    const k_max = (try testKey(&m, null, null, true, "max", false, null));
    try std.testing.expect(k_default != k_high);
    try std.testing.expect(k_high != k_max);
}

test "keyFor: a continuation does not share a key with the same messages as history" {
    // The trailing assistant renders as a PREFILL under continuation and as
    // history without it — same messages, different prompt. Colliding here
    // serves one request the other's tokens.
    const m = [_]chat_mod.Message{
        .{ .role = "user", .content = "count to three" },
        .{ .role = "assistant", .content = "one, two," },
    };
    const plain = (try testKey(&m, null, null, true, null, false, null));
    const cont = (try testKey(&m, null, null, true, null, true, null));
    try std.testing.expect(plain != cont);
}

test "TokenizeCache.keyFor separates the preserve_thinking arms" {
    // Off drops prior-turn thinking from the prompt: the same messages render
    // three ways, and each must keep its own tokens.
    const m = [_]chat_mod.Message{
        .{ .role = "user", .content = "q1" },
        .{ .role = "assistant", .content = "a1", .reasoning_content = "r1" },
        .{ .role = "user", .content = "q2" },
    };
    const absent = (try testKey(&m, null, null, true, null, false, null));
    const on = (try testKey(&m, null, null, true, null, false, true));
    const off = (try testKey(&m, null, null, true, null, false, false));
    try std.testing.expect(absent != on and absent != off and on != off);
}

test "TokenizeCache get/put + LRU eviction" {
    // Use the test-driver's io. std.Io.Threaded(.init) is the standard
    // single-threaded test rig that backs the std.Io.Mutex futex.
    var io_impl: std.Io.Threaded = .init(std.testing.allocator, .{});
    defer io_impl.deinit();
    const io = io_impl.io();

    var c = TokenizeCache.init(std.testing.allocator, 2);
    defer c.deinit();
    const a = [_]u32{ 1, 2, 3 };
    const b = [_]u32{ 4, 5, 6 };
    const cc = [_]u32{ 7, 8, 9 };
    try c.put(io, keyOf("A"), &a);
    try c.put(io, keyOf("B"), &b);
    // touch A so B is LRU
    if (try c.get(io, keyOf("A"), std.testing.allocator)) |got| std.testing.allocator.free(got);
    try c.put(io, keyOf("C"), &cc);
    // B evicted, A and C present
    try std.testing.expectEqual(@as(usize, 2), c.entries.items.len);
    const got_a = try c.get(io, keyOf("A"), std.testing.allocator) orelse return error.TestExpectedHit;
    defer std.testing.allocator.free(got_a);
    try std.testing.expectEqualSlices(u32, &a, got_a);
    try std.testing.expect((try c.get(io, keyOf("B"), std.testing.allocator)) == null);
}

test "TokenizeCache key keeps separator bytes inside their fields" {
    const a = [_]chat_mod.Message{.{ .role = "assistant", .content = "x\x1ey", .tool_call_id = "z" }};
    const b = [_]chat_mod.Message{.{ .role = "assistant", .content = "x", .tool_call_id = "y\x1ez" }};
    try std.testing.expect((try testKey(&a, null, null, false, null, false, null)) !=
        (try testKey(&b, null, null, false, null, false, null)));
}

test "TokenizeCache key includes tool call IDs" {
    const calls_a = [_]chat_mod.ToolCall{.{ .id = "call_a", .name = "lookup", .arguments = "{}" }};
    const calls_b = [_]chat_mod.ToolCall{.{ .id = "call_b", .name = "lookup", .arguments = "{}" }};
    const a = [_]chat_mod.Message{.{ .role = "assistant", .content = "", .tool_calls = &calls_a }};
    const b = [_]chat_mod.Message{.{ .role = "assistant", .content = "", .tool_calls = &calls_b }};
    try std.testing.expect((try testKey(&a, null, null, false, null, false, null)) !=
        (try testKey(&b, null, null, false, null, false, null)));
}

test "TokenizeCache: a hit is the same key bytes, never only the same hash" {
    const a = std.testing.allocator;
    var cache = TokenizeCache.init(a, 4);
    defer cache.deinit();
    const io = std.testing.io;
    var put_bytes = "first prompt".*;
    var other_bytes = "other prompt".*;
    try cache.put(io, .{ .hash = 7, .bytes = &put_bytes }, &.{ 1, 2, 3 });
    try std.testing.expectEqual(@as(?[]u32, null), try cache.get(io, .{ .hash = 7, .bytes = &other_bytes }, a));
    const hit = (try cache.get(io, .{ .hash = 7, .bytes = &put_bytes }, a)).?;
    defer a.free(hit);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3 }, hit);
}

pub fn testKey(messages: []const chat_mod.Message, tools_json: ?[]const u8, tool_choice_instruction: ?[]const u8, enable_thinking: bool, reasoning_effort: ?[]const u8, continue_final: bool, preserve_thinking: ?bool) !u64 {
    const key = try TokenizeCache.keyFor(std.testing.allocator, messages, tools_json, tool_choice_instruction, enable_thinking, reasoning_effort, continue_final, preserve_thinking);
    defer std.testing.allocator.free(key.bytes);
    return key.hash;
}

fn keyOf(comptime name: []const u8) TokenizeCache.Key {
    const bytes = struct {
        var b = name.*;
    };
    return .{ .hash = name[0], .bytes = &bytes.b };
}
