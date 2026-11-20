//! The one parser of a request head, strict and in one pass (RFC 9112 §2.2, §3, §5). `check` validates the request
//! line and every field line before anything reads them, and the request reader refuses (400) a head it does not
//! pass; every later read of the head (`value`, `hasToken`, `framing`) walks those checked lines with the same field
//! grammar, and the chunk-size and Content-Length digits go through `digits`. No reader has a grammar of its own.
const std = @import("std");

const ows = " \t";

/// The request line: method, target and version, split on single spaces.
pub const RequestLine = struct { method: []const u8 = "", target: []const u8 = "", version: []const u8 = "" };

/// The request line of the bytes that arrived: up to the first CRLF, or all of them when none came (a line over the
/// head limit), so even a refusal knows its route and method. Empty lines before it are skipped (§2.2).
pub fn requestLine(head: []const u8) RequestLine {
    const text = skipEmptyLines(head);
    return splitLine(text[0 .. std.mem.indexOf(u8, text, "\r\n") orelse text.len]);
}

fn splitLine(line: []const u8) RequestLine {
    var words = std.mem.splitScalar(u8, line, ' ');
    return .{ .method = words.next() orelse "", .target = words.next() orelse "", .version = words.next() orelse "" };
}

fn skipEmptyLines(head: []const u8) []const u8 {
    var text = head;
    while (std.mem.startsWith(u8, text, "\r\n")) text = text[2..];
    return text;
}

fn isTchar(b: u8) bool {
    return std.ascii.isAlphanumeric(b) or std.mem.indexOfScalar(u8, "!#$%&'*+-.^_`|~", b) != null;
}

fn isToken(s: []const u8) bool {
    if (s.len == 0) return false;
    for (s) |b| if (!isTchar(b)) return false;
    return true;
}

/// A field value's bytes: tab, space, visible ASCII and obs-text; no other control byte.
fn isFieldValue(s: []const u8) bool {
    for (s) |b| if (b != '\t' and (b < 0x20 or b == 0x7f)) return false;
    return true;
}

pub const Error = error{
    /// A request line that is not `method SP target SP HTTP/1.x`, a bare CR or LF, a field line folded (obs-fold),
    /// without a colon, with a name that is not a token (whitespace before the colon) or a control byte in its
    /// value, or two Host fields: answered 400 and closed.
    BadHead,
};

/// Checks `head` (through its blank line) whole.
pub fn check(head: []const u8) Error!void {
    for (head, 0..) |b, i| switch (b) {
        '\r' => if (i + 1 >= head.len or head[i + 1] != '\n') return error.BadHead,
        '\n' => if (i == 0 or head[i - 1] != '\r') return error.BadHead,
        else => {},
    };
    var lines = std.mem.splitSequence(u8, skipEmptyLines(head), "\r\n");
    const first = lines.next() orelse return error.BadHead;
    const line = splitLine(first);
    if (!isToken(line.method) or line.target.len == 0 or first.len != line.method.len + line.target.len + line.version.len + 2) return error.BadHead;
    for (line.target) |b| if (b <= 0x20 or b == 0x7f) return error.BadHead;
    if (!std.mem.eql(u8, line.version, "HTTP/1.1") and !std.mem.eql(u8, line.version, "HTTP/1.0")) return error.BadHead;
    var hosts: usize = 0;
    while (lines.next()) |field| {
        if (field.len == 0) break;
        const colon = std.mem.indexOfScalar(u8, field, ':') orelse return error.BadHead;
        if (!isToken(field[0..colon]) or !isFieldValue(field[colon + 1 ..])) return error.BadHead;
        if (std.ascii.eqlIgnoreCase(field[0..colon], "host")) hosts += 1;
    }
    if (hosts > 1) return error.BadHead;
}

/// The fields named `name`, in order, of a head `check` passed (the request line is no field: its name is no token).
pub const Fields = struct {
    lines: std.mem.SplitIterator(u8, .sequence),
    name: []const u8,

    pub fn next(it: *Fields) ?[]const u8 {
        while (it.lines.next()) |line| {
            const colon = std.mem.indexOfScalar(u8, line, ':') orelse continue;
            if (!isToken(line[0..colon]) or !std.ascii.eqlIgnoreCase(line[0..colon], it.name)) continue;
            return std.mem.trim(u8, line[colon + 1 ..], ows);
        }
        return null;
    }
};

pub fn fields(head: []const u8, name: []const u8) Fields {
    return .{ .lines = std.mem.splitSequence(u8, head, "\r\n"), .name = name };
}

/// The first field named `name`.
pub fn value(head: []const u8, name: []const u8) ?[]const u8 {
    var it = fields(head, name);
    return it.next();
}

/// Whether the list field `name`, across all its lines (§5.3), holds `token`.
pub fn hasToken(head: []const u8, name: []const u8, token: []const u8) bool {
    var it = fields(head, name);
    while (it.next()) |v| {
        var items = std.mem.splitScalar(u8, v, ',');
        while (items.next()) |item| if (std.ascii.eqlIgnoreCase(std.mem.trim(u8, item, ows), token)) return true;
    }
    return false;
}

/// `text` as a number in `base` (10 or 16): its digits alone, no sign, no separator; null otherwise or on overflow.
pub fn digits(text: []const u8, base: u8) ?usize {
    if (text.len == 0) return null;
    var n: usize = 0;
    for (text) |b| {
        const d = std.fmt.charToDigit(b, base) catch return null;
        n = std.math.mul(usize, n, base) catch return null;
        n = std.math.add(usize, n, d) catch return null;
    }
    return n;
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
            const n = digits(std.mem.trim(u8, item, ows), 10) orelse return error.BadFraming;
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

test "http_head: a head is checked whole: what a field line, a request line and a line end may be" {
    const t = std.testing;
    try check("POST /x HTTP/1.1\r\nContent-Length: 16\r\nX-A: b\r\n\r\n");
    try check("\r\nGET /health HTTP/1.0\r\n\r\n");
    inline for (.{
        "POST /x HTTP/1.1\r\nContent-Length : 16\r\n\r\n",
        "POST /x HTTP/1.1\r\nContent-Length\t: 16\r\n\r\n",
        "POST /x HTTP/1.1\r\nContent-Length: 1\r\n 6\r\n\r\n",
        "POST /x HTTP/1.1\r\nContent-Length: 1\r\n\t6\r\n\r\n",
        "POST /x HTTP/1.1\r\nHost: x\nContent-Length: 16\r\n\r\n",
        "POST /x HTTP/1.1\r\nX-A: b\rc\r\n\r\n",
        "POST /x HTTP/1.1\r\nNo colon here\r\n\r\n",
        "POST /x HTTP/1.1\r\nHost: a\r\nhost: b\r\n\r\n",
        "GET\r\n\r\n",
        "GET  /health HTTP/1.1\r\n\r\n",
        "GET /health HTTP/2\r\n\r\n",
    }) |h| try t.expectError(error.BadHead, check(h));
}

test "http_head: a list field is read across its lines" {
    const head = "GET / HTTP/1.1\r\nConnection: keep-alive\r\nConnection: Upgrade\r\n\r\n";
    try std.testing.expect(hasToken(head, "connection", "upgrade"));
    try std.testing.expect(!hasToken(head, "connection", "close"));
}

test "http_head: digits alone, in base 10 or 16" {
    const t = std.testing;
    try t.expectEqual(@as(?usize, 16), digits("16", 10));
    try t.expectEqual(@as(?usize, 16), digits("10", 16));
    try t.expectEqual(@as(?usize, 0), digits("0", 16));
    inline for (.{ "", "+10", "-0", "+0", "1_0", "0_10", "1 0", "0x10" }) |s| try t.expect(digits(s, 16) == null);
    try t.expect(digits("99999999999999999999999", 10) == null);
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

test "http_head: the request line's method and target, whole or cut short" {
    const l = requestLine("HEAD /v1/models?x=1 HTTP/1.1\r\nHost: h");
    try std.testing.expectEqualStrings("HEAD", l.method);
    try std.testing.expectEqualStrings("/v1/models?x=1", l.target);
    const cut = requestLine("HEAD /health?x=qqqq");
    try std.testing.expectEqualStrings("HEAD", cut.method);
    try std.testing.expectEqualStrings("/health?x=qqqq", cut.target);
}
