//! A text value as a JSON string, for the response bodies that splice a name the server did not write
//! (a model folder, an error name, a flag as typed).
const std = @import("std");

/// `{f}` writes `s` quoted and escaped; each invalid UTF-8 byte becomes U+FFFD, as `server.jsonEscape` does.
pub const Str = struct {
    s: []const u8,

    pub fn format(self: Str, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try write(w, self.s);
    }
};

pub fn write(w: *std.Io.Writer, s: []const u8) std.Io.Writer.Error!void {
    if (std.unicode.utf8ValidateSlice(s)) return std.json.Stringify.encodeJsonString(s, .{}, w);
    try w.writeByte('"');
    var i: usize = 0;
    while (i < s.len) {
        const n = std.unicode.utf8ByteSequenceLength(s[i]) catch 0;
        if (n > 0 and i + n <= s.len and std.unicode.utf8ValidateSlice(s[i..][0..n])) {
            try std.json.Stringify.encodeJsonStringChars(s[i..][0..n], .{}, w);
            i += n;
        } else {
            try w.writeAll("\\ufffd");
            i += 1;
        }
    }
    try w.writeByte('"');
}

/// The escaped content without its quotes, for a format string that supplies them. Caller frees.
/// `value` as JSON text appended to `buf`: a number keeps its digits as the parser read them (`std.json.Stringify`),
/// the one writer for a request's JSON that is read again later.
pub fn appendValue(allocator: std.mem.Allocator, buf: *std.ArrayList(u8), value: std.json.Value) std.mem.Allocator.Error!void {
    const text = try std.json.Stringify.valueAlloc(allocator, value, .{});
    defer allocator.free(text);
    try buf.appendSlice(allocator, text);
}

/// A tool call's `arguments` as the text a message carries: a string as sent, an object as its JSON, anything else
/// `{}`. `owned` is set when `text` was allocated here.
pub const CallArguments = struct { text: []const u8, owned: bool };

pub fn callArguments(allocator: std.mem.Allocator, value: ?std.json.Value) std.mem.Allocator.Error!CallArguments {
    switch (value orelse return .{ .text = "{}", .owned = false }) {
        .string => |s| return .{ .text = s, .owned = false },
        .object => |o| {
            var buf: std.ArrayList(u8) = .empty;
            errdefer buf.deinit(allocator);
            try appendValue(allocator, &buf, .{ .object = o });
            return .{ .text = try buf.toOwnedSlice(allocator), .owned = true };
        },
        else => return .{ .text = "{}", .owned = false },
    }
}

pub fn innerAlloc(allocator: std.mem.Allocator, s: []const u8) std.mem.Allocator.Error![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    write(&out.writer, s) catch return error.OutOfMemory;
    return allocator.dupe(u8, out.written()[1 .. out.written().len - 1]);
}

const testing = std.testing;

fn quoted(s: []const u8) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(testing.allocator);
    errdefer out.deinit();
    try out.writer.print("{f}", .{Str{ .s = s }});
    return out.toOwnedSlice();
}

test "json text: a quote, a backslash and a control character are escaped and the string parses back" {
    const input = "a\"b\\c\x01d\n";
    const text = try quoted(input);
    defer testing.allocator.free(text);
    const parsed = try std.json.parseFromSlice([]const u8, testing.allocator, text, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings(input, parsed.value);
}

test "json text: invalid UTF-8 becomes U+FFFD per bad byte and valid multibyte text passes" {
    const text = try quoted("é\xff\xc3x");
    defer testing.allocator.free(text);
    const parsed = try std.json.parseFromSlice([]const u8, testing.allocator, text, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("é\u{FFFD}\u{FFFD}x", parsed.value);
}

test "json text: the unquoted form drops only the quotes" {
    const inner = try innerAlloc(testing.allocator, "a\"b");
    defer testing.allocator.free(inner);
    try testing.expectEqualStrings("a\\\"b", inner);
}

fn numberValue(v: std.json.Value) !f64 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        .number_string => |s| try std.fmt.parseFloat(f64, s),
        else => error.NotANumber,
    };
}

test "json text: a re-serialized request value keeps every number's value" {
    const a = testing.allocator;
    const body = "{\"big\":1e40,\"one\":1.0,\"tiny\":-1e-30,\"huge_int\":12345678901234567890,\"pi\":3.141592653589793}";
    var parsed = try std.json.parseFromSlice(std.json.Value, a, body, .{});
    defer parsed.deinit();
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(a);
    try appendValue(a, &buf, parsed.value);
    var back = try std.json.parseFromSlice(std.json.Value, a, buf.items, .{});
    defer back.deinit();
    const o = back.value.object;
    try testing.expectEqual(@as(f64, 1e40), try numberValue(o.get("big").?));
    try testing.expectEqual(@as(f64, 1.0), try numberValue(o.get("one").?));
    try testing.expectEqual(@as(f64, -1e-30), try numberValue(o.get("tiny").?));
    try testing.expectEqual(@as(f64, 12345678901234567890.0), try numberValue(o.get("huge_int").?));
    try testing.expectEqual(@as(f64, 3.141592653589793), try numberValue(o.get("pi").?));
}

test "json text: call arguments sent as a JSON object are the object's text, a string is as sent, anything else is {}" {
    const a = testing.allocator;
    var parsed = try std.json.parseFromSlice(std.json.Value, a, "{\"o\":{\"x\":1.50},\"s\":\"{\\\"y\\\":2}\",\"n\":5,\"z\":null}", .{});
    defer parsed.deinit();
    const o = parsed.value.object;
    const obj = try callArguments(a, o.get("o"));
    defer if (obj.owned) a.free(obj.text);
    try testing.expectEqualStrings("{\"x\":1.5}", obj.text);
    try testing.expect(obj.owned);
    const str = try callArguments(a, o.get("s"));
    try testing.expectEqualStrings("{\"y\":2}", str.text);
    try testing.expect(!str.owned);
    for ([_]?std.json.Value{ o.get("n"), o.get("z"), null }) |other| {
        const c = try callArguments(a, other);
        try testing.expectEqualStrings("{}", c.text);
    }
}
