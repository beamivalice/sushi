//! The progress a stream shows while it buffers a tool call. The call reaches the client whole, since the parse chain
//! needs all of it, so a client that counts events (not SSE comments) saw nothing for the minutes a large call takes.
//! Once the model has named the first call, a due keepalive sends its opening (the name, `arguments` empty) and each
//! later one adds nothing to its arguments; the end adds them whole. Each surface writes the events; this file decides
//! what is sent and what the end owes.
//!
//! Only the first call goes out early. A call after a finished one cannot: the end decides whether a finished call
//! ships at all (a stop cut, a repetition loop, `parallel_tool_calls: false`), and a client cannot take one back.
const std = @import("std");
const reply = @import("reply.zig");
const tool_schema = @import("tool_schema.zig");
const think_tags = @import("think_tags.zig");
const io_util = @import("io_util.zig");
const http = @import("http.zig");

/// The first call, as the client was sent it before the end.
pub const Opened = struct { id: []u8, name: []u8 };

/// What a due keepalive sends.
pub const Beat = enum {
    /// The call's opening: its id and name, `arguments` empty.
    open,
    /// Nothing added to the arguments of the call that was opened.
    tick,
};

/// What the end owes a call of the reply.
pub const Owed = enum {
    /// The call that was opened: its arguments are left, as one fragment.
    arguments,
    /// A call the client has not seen.
    whole,
};

pub const Policy = struct {
    /// The surface sends a thought that closed before the call ahead of it. Reasoning that follows a call's opening would
    /// reach the client after it, and a surface whose blocks are sequential could not send it at all, so with this false
    /// no call goes out while the text holds a thought; a call inside a thought that never closed never goes out.
    closed_thought_first: bool = false,
    /// The prefix of the id the client is sent (`call`, `toolu`).
    id_prefix: []const u8 = "call",
};

pub const Progress = struct {
    opened: ?Opened = null,
    /// Set by `settle`: the opened call is not the reply's first, so the reply's calls follow it.
    detached: bool = false,

    pub fn deinit(self: *Progress, allocator: std.mem.Allocator) void {
        if (self.opened) |o| {
            allocator.free(o.id);
            allocator.free(o.name);
        }
        self.opened = null;
    }

    /// What a due keepalive sends, or null when no call is named yet (the keepalive stays a comment): the text holds
    /// no finished call, its first unfinished one has a declared name, and no thought stands in the way (`Policy`).
    /// `text` is everything the stream has buffered.
    pub fn beat(self: *Progress, allocator: std.mem.Allocator, io: std.Io, text: []const u8, origin: reply.Origin, tools_json: ?[]const u8, policy: Policy) !?Beat {
        if (self.opened != null) return .tick;
        const pending = (try reply.pendingCall(allocator, text, origin, tools_json)) orelse return null;
        defer pending.deinit(allocator);
        if (pending.finished.len > 0) return null;
        switch (thoughtBefore(origin.reading, text[0..pending.start], origin.opens_think)) {
            .none => {},
            .closed => if (!policy.closed_thought_first) return null,
            .open => return null,
        }
        const id = try io_util.callId(io, allocator, policy.id_prefix);
        errdefer allocator.free(id);
        const name = try allocator.dupe(u8, pending.name);
        self.opened = .{ .id = id, .name = name };
        return .open;
    }

    /// Decides once, when the generation has ended, whether the call that went out is the reply's first: the reply
    /// holds a first call of that name. It is then completed; otherwise it stays as it was sent, and the reply's calls
    /// follow it, each at its own place (`at`).
    pub fn settle(self: *Progress, first: ?tool_schema.ParsedToolCall) void {
        const o = self.opened orelse return;
        self.detached = if (first) |f| !std.mem.eql(u8, o.name, f.name) else true;
    }

    /// Where the reply's call `index` goes in the client's list.
    pub fn at(self: *const Progress, index: usize) usize {
        return index + @intFromBool(self.detached);
    }

    pub fn owed(self: *const Progress, index: usize) Owed {
        return if (self.opened != null and !self.detached and index == 0) .arguments else .whole;
    }

    /// The call that went out and was left unfinished (the generation ended inside it, or inside another call of the
    /// same name): the client holds it as it was sent, `arguments` empty, under a finish that never says `tool_calls`.
    pub fn cut(self: *const Progress) ?Opened {
        const o = self.opened orelse return null;
        return if (self.detached) o else null;
    }

    /// What closes that call: the arguments to send, or null (it stays as it was sent; the finish names the cut).
    pub fn cutEnding(self: *const Progress) ?[]const u8 {
        _ = self;
        return null;
    }
};

/// The thought a call's text stands after.
pub const Thought = enum { none, closed, open };

/// The thought in the text before a call: none, one that closed, or one the call stands inside.
pub fn thoughtBefore(reading: think_tags.Reading, before_call: []const u8, opens_think: bool) Thought {
    if (think_tags.splitThinkBlock(reading, before_call, true, opens_think).reasoning_content == null) return .none;
    const closed = think_tags.indexOfThinkCloseTag(reading, before_call, 0) != null or std.mem.indexOf(u8, before_call, "<channel|>") != null;
    return if (closed) .closed else .open;
}

/// A `tool_calls` delta that adds nothing to the arguments of call `index`.
pub fn emptyFragment(allocator: std.mem.Allocator, index: usize) ![]u8 {
    return std.fmt.allocPrint(allocator, "[{{\"index\":{d},\"function\":{{\"arguments\":\"\"}}}}]", .{index});
}

/// The arguments of call `index` as one fragment, which completes a call that was opened.
pub fn fragment(allocator: std.mem.Allocator, index: usize, arguments: []const u8) ![]u8 {
    const escaped = try http.jsonEscape(allocator, arguments);
    defer allocator.free(escaped);
    return std.fmt.allocPrint(allocator, "[{{\"index\":{d},\"function\":{{\"arguments\":\"{s}\"}}}}]", .{ index, quotedInner(escaped) });
}

fn quotedInner(escaped: []const u8) []const u8 {
    if (escaped.len >= 2 and escaped[0] == '"' and escaped[escaped.len - 1] == '"') return escaped[1 .. escaped.len - 1];
    return escaped;
}

/// A whole call as one `tool_calls` delta.
pub fn callDelta(allocator: std.mem.Allocator, index: usize, id: []const u8, name: []const u8, arguments: []const u8) ![]u8 {
    const escaped_name = try http.jsonEscape(allocator, name);
    defer allocator.free(escaped_name);
    const escaped_args = try http.jsonEscape(allocator, arguments);
    defer allocator.free(escaped_args);
    return std.fmt.allocPrint(allocator,
        \\[{{"index":{d},"id":"{s}","type":"function","function":{{"name":"{s}","arguments":"{s}"}}}}]
    , .{ index, id, quotedInner(escaped_name), quotedInner(escaped_args) });
}

/// The `tool_calls` array a due beat writes on the chat surface.
pub fn chatBeat(allocator: std.mem.Allocator, progress: *const Progress, beat: Beat) ![]u8 {
    const o = progress.opened.?;
    return switch (beat) {
        .open => callDelta(allocator, 0, o.id, o.name, ""),
        .tick => emptyFragment(allocator, 0),
    };
}

/// The `tool_calls` array that delivers call `index` of the reply on the chat surface at the end.
pub fn chatEnd(allocator: std.mem.Allocator, io: std.Io, progress: *const Progress, index: usize, call: tool_schema.ParsedToolCall) ![]u8 {
    switch (progress.owed(index)) {
        .arguments => return fragment(allocator, 0, call.arguments),
        .whole => {
            const id = try io_util.callId(io, allocator, "call");
            defer allocator.free(id);
            return callDelta(allocator, progress.at(index), id, call.name, call.arguments);
        },
    }
}

const testing = std.testing;

const weather_tools =
    \\[{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"notes":{"type":"string"}},"required":["city"]}}}]
;

/// One call of the same tool in each dialect a request is served in, cut where a long value starts: `head` up to the
/// second argument's value, `tail` after it, `first_call` a whole call of another value, and `unsettled` texts whose
/// tool name is not whole or not declared.
const Shape = struct {
    name: []const u8,
    head: []const u8,
    tail: []const u8,
    first_call: []const u8,
    unsettled: []const []const u8,
};

const shapes = [_]Shape{
    .{
        .name = "function tag",
        .head = "Let me check.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n<parameter=notes>\n",
        .tail = "\n</parameter>\n</function>\n</tool_call>",
        .first_call = "<tool_call>\n<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n</function>\n</tool_call>\n",
        .unsettled = &.{ "Let me check.\n<tool_call>\n<function=", "<tool_call>\n<function=get_stock>\n<parameter=symbol>\nAAPL" },
    },
    .{
        .name = "Hy4 wrapper, arg_key",
        .head = "Let me check.\n<tool_calls:opensource><tool_call>get_weather<arg_key>city</arg_key><arg_value>Tokyo</arg_value><arg_key>notes</arg_key><arg_value>",
        .tail = "</arg_value></tool_call></tool_calls:opensource>",
        .first_call = "<tool_calls:opensource><tool_call>get_weather<arg_key>city</arg_key><arg_value>Oslo</arg_value></tool_call></tool_calls:opensource>\n",
        .unsettled = &.{ "Let me check.\n<tool_calls:opensource>", "Let me check.\n<tool_calls:opensource><tool_call>", "<tool_calls:opensource><tool_call>get_wea", "<tool_calls:opensource><tool_call>get_stock<arg_key>symbol</arg_key><arg_value>AAPL" },
    },
    .{
        .name = "Hy4 wrapper, every tag :opensource",
        .head = "Let me check.\n<tool_calls:opensource><tool_call:opensource>get_weather<arg_key:opensource>city</arg_key:opensource><arg_value:opensource>Tokyo</arg_value:opensource><arg_key:opensource>notes</arg_key:opensource><arg_value:opensource>",
        .tail = "</arg_value:opensource></tool_call:opensource></tool_calls:opensource>",
        .first_call = "<tool_calls:opensource><tool_call:opensource>get_weather<arg_key:opensource>city</arg_key:opensource><arg_value:opensource>Oslo</arg_value:opensource></tool_call:opensource></tool_calls:opensource>\n",
        .unsettled = &.{ "Let me check.\n<tool_calls:opensource><tool_call:opensource>", "<tool_calls:opensource><tool_call:opensource>get_wea", "<tool_calls:opensource><tool_call:opensource>get_stock<arg_key:opensource>symbol</arg_key:opensource><arg_value:opensource>AAPL" },
    },
};

/// The head of a `write_file` call in each dialect a request is served in, up to its `content` value: what the surfaces'
/// beat cells stream.
pub const write_file_heads = [_][]const u8{
    "<tool_call>\n<function=write_file>\n<parameter=path>\nreport.md\n</parameter>\n<parameter=content>\n",
    "<tool_calls:opensource><tool_call>write_file<arg_key>path</arg_key><arg_value>report.md</arg_value><arg_key>content</arg_key><arg_value>",
    "<tool_calls:opensource><tool_call:opensource>write_file<arg_key:opensource>path</arg_key:opensource><arg_value:opensource>report.md</arg_value:opensource><arg_key:opensource>content</arg_key:opensource><arg_value:opensource>",
};

/// What an OpenAI client holds of the calls it is streamed, by index: `id` and `name` from the delta that has them,
/// the arguments as the concatenation of every fragment.
const Client = struct {
    const Call = struct { id: ?[]u8 = null, name: ?[]u8 = null, arguments: std.ArrayList(u8) = .empty };
    calls: std.ArrayList(Call) = .empty,
    events: usize = 0,

    fn deinit(self: *Client, allocator: std.mem.Allocator) void {
        for (self.calls.items) |*c| {
            if (c.id) |v| allocator.free(v);
            if (c.name) |v| allocator.free(v);
            c.arguments.deinit(allocator);
        }
        self.calls.deinit(allocator);
    }

    fn take(self: *Client, allocator: std.mem.Allocator, delta: []const u8) !void {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, delta, .{});
        defer parsed.deinit();
        for (parsed.value.array.items) |entry| {
            self.events += 1;
            const obj = entry.object;
            const index: usize = @intCast(obj.get("index").?.integer);
            while (self.calls.items.len <= index) try self.calls.append(allocator, .{});
            const call = &self.calls.items[index];
            if (obj.get("id")) |id| {
                try testing.expect(call.id == null);
                call.id = try allocator.dupe(u8, id.string);
            }
            const function = obj.get("function").?.object;
            if (function.get("name")) |name| {
                try testing.expect(call.name == null);
                call.name = try allocator.dupe(u8, name.string);
            }
            try call.arguments.appendSlice(allocator, function.get("arguments").?.string);
        }
    }
};

fn freeCalls(allocator: std.mem.Allocator, calls: []tool_schema.ParsedToolCall) void {
    for (calls) |c| {
        allocator.free(c.name);
        allocator.free(c.arguments);
    }
    allocator.free(calls);
}

test "tool progress: a call that takes longer than the beat interval yields data events at the interval, and the client assembles the non-stream call, in every dialect" {
    const allocator = testing.allocator;
    const interval_ms: i64 = 5000;
    const token_ms: i64 = 100;
    for (shapes) |shape| {
        errdefer std.debug.print("shape: {s}\n", .{shape.name});
        var progress = Progress{};
        defer progress.deinit(allocator);
        var client = Client{};
        defer client.deinit(allocator);
        var buffered = std.ArrayList(u8).empty;
        defer buffered.deinit(allocator);

        // One token every 100 ms for 23 s: the head, then the file the call writes.
        var tokens: std.ArrayList([]const u8) = .empty;
        defer tokens.deinit(allocator);
        try tokens.append(allocator, shape.head);
        for (0..229) |_| try tokens.append(allocator, "line of the file \n");
        var now: i64 = 0;
        var last_write: i64 = 0;
        var comments: usize = 0;
        var gaps_over: usize = 0;
        for (tokens.items) |tok| {
            now += token_ms;
            try buffered.appendSlice(allocator, tok);
            if (now - last_write < interval_ms) continue;
            if (try progress.beat(allocator, testing.io, buffered.items, .{ .reading = .text, .dialect = null }, weather_tools, .{})) |beat| {
                const delta = try chatBeat(allocator, &progress, beat);
                defer allocator.free(delta);
                try client.take(allocator, delta);
            } else comments += 1;
            if (now - last_write > interval_ms + token_ms) gaps_over += 1;
            last_write = now;
        }
        try buffered.appendSlice(allocator, shape.tail);

        // 23 s of silence: four beats, every one an event the client counts.
        try testing.expectEqual(@as(usize, 0), comments);
        try testing.expectEqual(@as(usize, 4), client.events);
        try testing.expectEqual(@as(usize, 0), gaps_over);
        try testing.expectEqualStrings("get_weather", client.calls.items[0].name.?);
        try testing.expectEqualStrings("", client.calls.items[0].arguments.items);

        const calls = (try reply.parseToolCallsForRequest(allocator, buffered.items, weather_tools, true, .{ .reading = .text, .dialect = null })).?;
        defer freeCalls(allocator, calls);
        progress.settle(calls[0]);
        const end = try chatEnd(allocator, testing.io, &progress, 0, calls[0]);
        defer allocator.free(end);
        try client.take(allocator, end);
        const got = client.calls.items[0];
        try testing.expectEqualStrings(calls[0].name, got.name.?);
        try testing.expectEqualStrings(calls[0].arguments, got.arguments.items);
        try testing.expectEqualStrings(progress.opened.?.id, got.id.?);
        try testing.expect(progress.cut() == null);
        const valid = try std.json.parseFromSlice(std.json.Value, allocator, got.arguments.items, .{});
        valid.deinit();
    }
}

test "tool progress: a call opened and then ended inside is cut, and the calls of the reply follow it, in every dialect" {
    const allocator = testing.allocator;
    for (shapes) |shape| {
        errdefer std.debug.print("shape: {s}\n", .{shape.name});
        var progress = Progress{};
        defer progress.deinit(allocator);
        _ = (try progress.beat(allocator, testing.io, shape.head, .{ .reading = .text, .dialect = null }, weather_tools, .{})).?;
        // The reply holds no call: the opened one stays as sent.
        progress.settle(null);
        try testing.expectEqualStrings("get_weather", progress.cut().?.name);
        try testing.expect(progress.cutEnding() == null);
        // A reply whose first call is another tool's: it goes after the opened one, at its own index.
        var other = Progress{};
        defer other.deinit(allocator);
        _ = (try other.beat(allocator, testing.io, shape.head, .{ .reading = .text, .dialect = null }, weather_tools, .{})).?;
        const call: tool_schema.ParsedToolCall = .{ .name = "get_stock", .arguments = "{\"symbol\":\"AAPL\"}" };
        other.settle(call);
        try testing.expect(other.cut() != null);
        const end = try chatEnd(allocator, testing.io, &other, 0, call);
        defer allocator.free(end);
        var client = Client{};
        defer client.deinit(allocator);
        try client.take(allocator, end);
        try testing.expectEqual(@as(usize, 2), client.calls.items.len);
        try testing.expectEqualStrings("get_stock", client.calls.items[1].name.?);
    }
}

test "tool progress: a call after a finished one is never sent early, in every dialect" {
    const allocator = testing.allocator;
    for (shapes) |shape| {
        errdefer std.debug.print("shape: {s}\n", .{shape.name});
        var progress = Progress{};
        defer progress.deinit(allocator);
        const text = try std.mem.concat(allocator, u8, &.{ shape.first_call, shape.head });
        defer allocator.free(text);
        try testing.expect((try progress.beat(allocator, testing.io, text, .{ .reading = .text, .dialect = null }, weather_tools, .{})) == null);
        try testing.expect(progress.opened == null);
    }
}

test "tool progress: no beat names a call before its name is settled and declared, in every dialect" {
    const allocator = testing.allocator;
    const common = [_][]const u8{ "", "Let me think about the weather." };
    for (common) |text| {
        var progress = Progress{};
        defer progress.deinit(allocator);
        try testing.expect((try progress.beat(allocator, testing.io, text, .{ .reading = .text, .dialect = null }, weather_tools, .{})) == null);
    }
    for (shapes) |shape| for (shape.unsettled) |text| {
        var progress = Progress{};
        defer progress.deinit(allocator);
        errdefer std.debug.print("[{s}] text: {f}\n", .{ shape.name, std.zig.fmtString(text) });
        try testing.expect((try progress.beat(allocator, testing.io, text, .{ .reading = .text, .dialect = null }, weather_tools, .{})) == null);
    };
}

test "tool progress: a thought stands in the way of the opening, a closed one only where the surface sends it first, in every dialect" {
    const allocator = testing.allocator;
    for (shapes) |shape| {
        const closed = try std.mem.concat(allocator, u8, &.{ "<think>I will call it.</think>\n\n", shape.head });
        defer allocator.free(closed);
        const open = try std.mem.concat(allocator, u8, &.{ "<think>I will call it. ", shape.head });
        defer allocator.free(open);
        const cases = [_]struct { text: []const u8, opens: bool, first: bool, goes: bool }{
            .{ .text = shape.head, .opens = false, .first = false, .goes = true },
            .{ .text = closed, .opens = false, .first = false, .goes = false },
            .{ .text = closed, .opens = false, .first = true, .goes = true },
            .{ .text = open, .opens = false, .first = true, .goes = false },
            // The prompt opened the thought: the call before any close stands inside it.
            .{ .text = shape.head, .opens = true, .first = true, .goes = false },
        };
        for (cases, 0..) |c, i| {
            var progress = Progress{};
            defer progress.deinit(allocator);
            errdefer std.debug.print("[{s}] case {d}\n", .{ shape.name, i });
            const got = try progress.beat(allocator, testing.io, c.text, .{ .reading = .text, .dialect = null, .opens_think = c.opens }, weather_tools, .{ .closed_thought_first = c.first });
            try testing.expectEqual(c.goes, got != null);
        }
    }
}

test "tool progress: a call delta escapes the function name" {
    const a = testing.allocator;
    const delta = try callDelta(a, 0, "call_1", "say\"hi", "{\"x\":1}");
    defer a.free(delta);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, delta, .{});
    defer parsed.deinit();
    try testing.expect(parsed.value == .array);
    const fn_obj = parsed.value.array.items[0].object.get("function").?.object;
    try testing.expectEqualStrings("say\"hi", fn_obj.get("name").?.string);
    try testing.expectEqualStrings("{\"x\":1}", fn_obj.get("arguments").?.string);
}
