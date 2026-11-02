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
