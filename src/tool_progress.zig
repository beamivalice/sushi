//! The progress a chat stream shows while it buffers a tool call. The call reaches the client whole, since the parse
//! chain needs all of it, so a client that counts events (not SSE comments) saw nothing for the minutes a large call
//! takes. Once the model has named the call, a due keepalive opens it (index, id, name, `arguments: ""`) and each
//! later one is an empty `arguments` fragment; the end adds the whole arguments to the same call.
const std = @import("std");
const reply = @import("reply.zig");
const tool_schema = @import("tool_schema.zig");
const io_util = @import("io_util.zig");
const http = @import("http.zig");

/// A call the stream opened before it was whole.
const Opened = struct { id: []u8, name: []u8 };

pub const Progress = struct {
    opened: ?Opened = null,

    pub fn deinit(self: *Progress, allocator: std.mem.Allocator) void {
        if (self.opened) |o| {
            allocator.free(o.id);
            allocator.free(o.name);
        }
        self.opened = null;
    }

    /// The `tool_calls` array a due keepalive writes in place of its comment, or null when the buffer holds no call
    /// whose name is settled (the comment then). `text` is everything the stream has buffered.
    pub fn beat(self: *Progress, allocator: std.mem.Allocator, io: std.Io, text: []const u8, origin: reply.Origin, tools_json: ?[]const u8) !?[]u8 {
        if (self.opened != null) return try allocator.dupe(u8, empty_fragment);
        const name = (try reply.pendingCallName(allocator, text, origin, tools_json)) orelse return null;
        errdefer allocator.free(name);
        const id = try io_util.callId(io, allocator, "call");
        errdefer allocator.free(id);
        const delta = try callDelta(allocator, 0, id, name, "");
        self.opened = .{ .id = id, .name = name };
        return delta;
    }

    /// The `tool_calls` array that delivers call `index` at the end of the generation: the call whole, or, for the call
    /// the stream opened early, its arguments as the one fragment that completes it.
    pub fn deliver(self: *Progress, allocator: std.mem.Allocator, io: std.Io, index: usize, call: tool_schema.ParsedToolCall) ![]u8 {
        if (index == 0) if (self.opened) |o| {
            // A client holds the name it was sent: the call it completes must be the one it was told.
            if (!std.mem.eql(u8, o.name, call.name)) return error.PendingCallRenamed;
            return fragment(allocator, call.arguments);
        };
        const id = try io_util.callId(io, allocator, "call");
        defer allocator.free(id);
        return callDelta(allocator, index, id, call.name, call.arguments);
    }

    /// What closes an opened call the generation left unfinished (no call ships): the array to send, or null when it is
    /// left as the client holds it, `arguments` empty, under a finish that is never `tool_calls` and names the cut.
    pub fn cutEnding(self: *const Progress) ?[]const u8 {
        _ = self;
        return null;
    }
};

const empty_fragment = "[{\"index\":0,\"function\":{\"arguments\":\"\"}}]";

fn fragment(allocator: std.mem.Allocator, arguments: []const u8) ![]u8 {
    const escaped = try http.jsonEscape(allocator, arguments);
    defer allocator.free(escaped);
    return std.fmt.allocPrint(allocator, "[{{\"index\":0,\"function\":{{\"arguments\":\"{s}\"}}}}]", .{quotedInner(escaped)});
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

const testing = std.testing;

const weather_tools =
    \\[{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"notes":{"type":"string"}},"required":["city"]}}}]
;
const call_head = "Let me check.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n<parameter=notes>\n";
const call_tail = "\n</parameter>\n</function>\n</tool_call>";

/// What an OpenAI client holds of the one call it is streamed: `id` and `name` from the delta that has them, the
/// arguments as the concatenation of every fragment.
const Client = struct {
    id: ?[]u8 = null,
    name: ?[]u8 = null,
    arguments: std.ArrayList(u8) = .empty,
    events: usize = 0,

    fn deinit(self: *Client, allocator: std.mem.Allocator) void {
        if (self.id) |v| allocator.free(v);
        if (self.name) |v| allocator.free(v);
        self.arguments.deinit(allocator);
    }

    fn take(self: *Client, allocator: std.mem.Allocator, delta: []const u8) !void {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, delta, .{});
        defer parsed.deinit();
        for (parsed.value.array.items) |entry| {
            self.events += 1;
            const obj = entry.object;
            try testing.expectEqual(@as(i64, 0), obj.get("index").?.integer);
            if (obj.get("id")) |id| {
                try testing.expect(self.id == null);
                self.id = try allocator.dupe(u8, id.string);
            }
            const function = obj.get("function").?.object;
            if (function.get("name")) |name| {
                try testing.expect(self.name == null);
                self.name = try allocator.dupe(u8, name.string);
            }
            try self.arguments.appendSlice(allocator, function.get("arguments").?.string);
        }
    }
};

test "tool progress: a call that takes longer than the beat interval yields data events at the interval, and the client assembles the non-stream call" {
    const allocator = testing.allocator;
    const interval_ms: i64 = 5000;
    const token_ms: i64 = 100;
    var progress = Progress{};
    defer progress.deinit(allocator);
    var client = Client{};
    defer client.deinit(allocator);
    var buffered = std.ArrayList(u8).empty;
    defer buffered.deinit(allocator);

    // One token every 100 ms for 23 s: the head, then the file the call writes.
    var tokens: std.ArrayList([]const u8) = .empty;
    defer tokens.deinit(allocator);
    try tokens.append(allocator, call_head);
    for (0..229) |_| try tokens.append(allocator, "line of the file \n");
    var now: i64 = 0;
    var last_write: i64 = 0;
    var comments: usize = 0;
    var gaps_over: usize = 0;
    for (tokens.items) |tok| {
        now += token_ms;
        try buffered.appendSlice(allocator, tok);
        if (now - last_write < interval_ms) continue;
        if (try progress.beat(allocator, testing.io, buffered.items, .{ .dialect = null }, weather_tools)) |delta| {
            defer allocator.free(delta);
            try client.take(allocator, delta);
        } else comments += 1;
        if (now - last_write > interval_ms + token_ms) gaps_over += 1;
        last_write = now;
    }
    try buffered.appendSlice(allocator, call_tail);

    // 23 s of silence: four beats, every one an event the client counts.
    try testing.expectEqual(@as(usize, 0), comments);
    try testing.expectEqual(@as(usize, 4), client.events);
    try testing.expectEqual(@as(usize, 0), gaps_over);
    try testing.expectEqualStrings("get_weather", client.name.?);
    try testing.expectEqualStrings("", client.arguments.items);

    const calls = (try reply.parseToolCallsForRequest(allocator, buffered.items, weather_tools, true, .{ .dialect = null })).?;
    defer {
        for (calls) |c| {
            allocator.free(c.name);
            allocator.free(c.arguments);
        }
        allocator.free(calls);
    }
    const end = try progress.deliver(allocator, testing.io, 0, calls[0]);
    defer allocator.free(end);
    try client.take(allocator, end);
    try testing.expectEqualStrings(calls[0].name, client.name.?);
    try testing.expectEqualStrings(calls[0].arguments, client.arguments.items);
    try testing.expectEqualStrings(progress.opened.?.id, client.id.?);
    const valid = try std.json.parseFromSlice(std.json.Value, allocator, client.arguments.items, .{});
    valid.deinit();
}

test "tool progress: no beat names a call before its name is settled and declared" {
    const allocator = testing.allocator;
    const texts = [_][]const u8{
        "",
        "Let me think about the weather.",
        "Let me check.\n<tool_call>\n<function=",
        // A tool the request does not declare.
        "<tool_call>\n<function=get_stock>\n<parameter=symbol>\nAAPL",
    };
    for (texts) |text| {
        var progress = Progress{};
        defer progress.deinit(allocator);
        errdefer std.debug.print("text: {f}\n", .{std.zig.fmtString(text)});
        try testing.expect((try progress.beat(allocator, testing.io, text, .{ .dialect = null }, weather_tools)) == null);
    }
}

test "tool progress: the call the end delivers is the one the client was told" {
    const allocator = testing.allocator;
    var progress = Progress{};
    defer progress.deinit(allocator);
    const open = (try progress.beat(allocator, testing.io, call_head, .{ .dialect = null }, weather_tools)).?;
    allocator.free(open);
    try testing.expectError(error.PendingCallRenamed, progress.deliver(allocator, testing.io, 0, .{ .name = "get_stock", .arguments = "{}" }));
}

test "tool progress: a call after the opened one is delivered whole at its own index" {
    const allocator = testing.allocator;
    var progress = Progress{};
    defer progress.deinit(allocator);
    const open = (try progress.beat(allocator, testing.io, call_head, .{ .dialect = null }, weather_tools)).?;
    allocator.free(open);
    const second = try progress.deliver(allocator, testing.io, 1, .{ .name = "get_weather", .arguments = "{\"city\":\"Oslo\"}" });
    defer allocator.free(second);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, second, .{});
    defer parsed.deinit();
    const entry = parsed.value.array.items[0].object;
    try testing.expectEqual(@as(i64, 1), entry.get("index").?.integer);
    try testing.expect(!std.mem.eql(u8, progress.opened.?.id, entry.get("id").?.string));
    try testing.expectEqualStrings("{\"city\":\"Oslo\"}", entry.get("function").?.object.get("arguments").?.string);
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
