//! The one parser of a request head: its request line, and its fields by name (matched whole and case-insensitively,
//! the value with its optional whitespace trimmed). Every reader of a head goes through it, so no two readers can
//! disagree on what a request said.
const std = @import("std");

const ows = " \t";

/// The fields named `name`, in order.
pub const Fields = struct {
    lines: std.mem.SplitIterator(u8, .sequence),
    name: []const u8,

    pub fn next(it: *Fields) ?[]const u8 {
        while (it.lines.next()) |line| {
            const colon = std.mem.indexOfScalar(u8, line, ':') orelse continue;
            if (!std.ascii.eqlIgnoreCase(line[0..colon], it.name)) continue;
            return std.mem.trim(u8, line[colon + 1 ..], ows);
        }
        return null;
    }
};

/// The fields of `head` named `name`. A request line never matches: its first colon follows a space.
pub fn fields(head: []const u8, name: []const u8) Fields {
    return .{ .lines = std.mem.splitSequence(u8, head, "\r\n"), .name = name };
}

/// The first field named `name`.
pub fn value(head: []const u8, name: []const u8) ?[]const u8 {
    var it = fields(head, name);
    return it.next();
}

/// The request line's method and target; empty when the line has not arrived whole.
pub const RequestLine = struct { method: []const u8 = "", target: []const u8 = "" };

pub fn requestLine(head: []const u8) RequestLine {
    const end = std.mem.indexOf(u8, head, "\r\n") orelse return .{};
    var words = std.mem.splitScalar(u8, head[0..end], ' ');
    const method = words.next() orelse return .{};
    return .{ .method = method, .target = words.next() orelse "" };
}

/// How a request's body is framed, read once from its head (RFC 9112 §6.3).
pub const Framing = union(enum) {
    length: usize,
    chunked,
};

pub const FramingError = error{
    /// A Content-Length that is not digits alone, two that differ, two Transfer-Encoding fields, `chunked` named
    /// twice, or Transfer-Encoding beside Content-Length: answered 400 and closed.
    BadFraming,
    /// A transfer coding other than `chunked`: answered 501.
    UnsupportedCoding,
};

/// The body framing `head` declares: no field is a length of 0.
pub fn framing(head: []const u8) FramingError!Framing {
    var te = fields(head, "transfer-encoding");
    const coding = te.next();
    if (te.next() != null) return error.BadFraming;
    const length = try contentLength(head);
    const c = coding orelse return .{ .length = length orelse 0 };
    // A request carrying both is refused, never framed by one and smuggled by the other.
    if (length != null) return error.BadFraming;
    var chunked: usize = 0;
    var items = std.mem.splitScalar(u8, c, ',');
    while (items.next()) |item| {
        if (!std.ascii.eqlIgnoreCase(std.mem.trim(u8, item, ows), "chunked")) return error.UnsupportedCoding;
        chunked += 1;
    }
    return if (chunked == 1) .chunked else error.BadFraming;
}

/// The Content-Length `head` declares, null without one; every field and list item must be the same digits.
fn contentLength(head: []const u8) FramingError!?usize {
    var found: ?usize = null;
    var it = fields(head, "content-length");
    while (it.next()) |v| {
        var items = std.mem.splitScalar(u8, v, ',');
        while (items.next()) |item| {
            const digits = std.mem.trim(u8, item, ows);
            if (digits.len == 0) return error.BadFraming;
            for (digits) |b| if (!std.ascii.isDigit(b)) return error.BadFraming;
            const n = std.fmt.parseInt(usize, digits, 10) catch return error.BadFraming;
            if (found) |f| if (f != n) return error.BadFraming;
            found = n;
        }
    }
    return found;
}

test "http_head: a field is found whatever its whitespace and case, by its whole name" {
    const head = "POST /x HTTP/1.1\r\nContent-Length:16\r\nX-Pad:\t a \t\r\nhost: h\r\nContent-Type: t";
    try std.testing.expectEqualStrings("16", value(head, "content-length").?);
    try std.testing.expectEqualStrings("a", value(head, "x-pad").?);
    try std.testing.expectEqualStrings("h", value(head, "Host").?);
    try std.testing.expect(value(head, "content") == null);
}

test "http_head: the framing a head declares, and the ones refused" {
    const t = std.testing;
    const line = "POST /x HTTP/1.1\r\n";
    try t.expectEqual(Framing{ .length = 0 }, try framing(line));
    inline for (.{ "Content-Length:16", "Content-Length:\t16", "Content-Length: 16\t", "Content-Length: 16\r\nContent-Length: 16", "Content-Length: 16, 16" }) |f|
        try t.expectEqual(Framing{ .length = 16 }, try framing(line ++ f));
    try t.expectEqual(Framing.chunked, try framing(line ++ "Transfer-Encoding: Chunked"));
    inline for (.{ "Content-Length: 1_6", "Content-Length: +16", "Content-Length: ", "Content-Length: 0\r\nContent-Length: 16", "Content-Length: 16, 17", "Transfer-Encoding: chunked\r\nTransfer-Encoding: chunked", "Transfer-Encoding: chunked, chunked", "Transfer-Encoding: chunked\r\nContent-Length: 5" }) |f|
        try t.expectError(error.BadFraming, framing(line ++ f));
    try t.expectError(error.UnsupportedCoding, framing(line ++ "Transfer-Encoding: gzip, chunked"));
}

test "http_head: the request line's method and target" {
    const l = requestLine("HEAD /v1/models?x=1 HTTP/1.1\r\nHost: h");
    try std.testing.expectEqualStrings("HEAD", l.method);
    try std.testing.expectEqualStrings("/v1/models?x=1", l.target);
    try std.testing.expectEqualStrings("", requestLine("HEAD /v1/mo").method);
}
