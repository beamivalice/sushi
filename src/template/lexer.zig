//! Tokens of a template as Jinja2's lexer reads them under transformers' settings: newlines normalized and one
//! trailing newline dropped, `trim_blocks` and `lstrip_blocks` on, `-` strips and `+` keeps whitespace.
const std = @import("std");
const value = @import("value.zig");

const Allocator = std.mem.Allocator;

pub const Tag = enum { data, var_begin, var_end, block_begin, block_end, name, string, integer, float, op, eof };

pub const Token = struct {
    tag: Tag,
    /// data: the text; string: its decoded value; name, integer, float, op: the source text.
    text: []const u8,
    pos: u32,
};

pub const SyntaxError = struct { pos: u32, message: []const u8 };

pub const Error = error{ TemplateSyntax, OutOfMemory };

/// Jinja2's operators, longest first.
const operators = [_][]const u8{ "//", "**", "==", "!=", ">=", "<=", "+", "-", "/", "*", "%", "~", "[", "]", "(", ")", "{", "}", ">", "<", "=", ".", ":", "|", ",", ";" };

pub const Lexer = struct {
    arena: Allocator,
    src: []const u8,
    pos: usize = 0,
    tokens: std.ArrayList(Token) = .empty,
    err: *SyntaxError,

    fn fail(self: *Lexer, at: usize, comptime fmt: []const u8, args: anytype) Error {
        @setRuntimeSafety(true);
        self.err.* = .{ .pos = @intCast(at), .message = std.fmt.allocPrint(self.arena, fmt, args) catch "out of memory" };
        return error.TemplateSyntax;
    }

    fn emit(self: *Lexer, kind: Tag, text: []const u8, at: usize) Error!void {
        @setRuntimeSafety(true);
        try self.tokens.append(self.arena, .{ .tag = kind, .text = text, .pos = @intCast(at) });
    }

    fn startsWith(self: *const Lexer, at: usize, s: []const u8) bool {
        @setRuntimeSafety(true);
        return at <= self.src.len and std.mem.startsWith(u8, self.src[at..], s);
    }

    /// The source in Jinja2's form: `\r\n` and `\r` as `\n`, one trailing newline dropped.
    pub fn normalize(arena: Allocator, raw: []const u8) Error![]const u8 {
        @setRuntimeSafety(true);
        var out = try std.ArrayList(u8).initCapacity(arena, raw.len);
        var i: usize = 0;
        while (i < raw.len) : (i += 1) {
            if (raw[i] == '\r') {
                out.appendAssumeCapacity('\n');
                if (i + 1 < raw.len and raw[i + 1] == '\n') i += 1;
            } else out.appendAssumeCapacity(raw[i]);
        }
        if (out.items.len > 0 and out.items[out.items.len - 1] == '\n') out.items.len -= 1;
        return out.items;
    }

    pub fn run(self: *Lexer) Error![]const Token {
        @setRuntimeSafety(true);
        var line_starting = true;
        while (self.pos < self.src.len) {
            const open = self.nextOpen(self.pos) orelse {
                try self.emit(.data, self.src[self.pos..], self.pos);
                break;
            };
            const kind = self.src[open + 1];
            const sign: u8 = if (open + 2 < self.src.len and (self.src[open + 2] == '-' or self.src[open + 2] == '+')) self.src[open + 2] else 0;
            var text = self.src[self.pos..open];
            if (sign == '-') {
                text = rstripSpace(text);
            } else if (sign != '+' and kind != '{') {
                const l_pos = if (std.mem.lastIndexOfScalar(u8, text, '\n')) |nl| nl + 1 else 0;
                if ((l_pos > 0 or line_starting) and l_pos < text.len and allSpace(text[l_pos..])) text = text[0..l_pos];
            }
            if (text.len > 0) try self.emit(.data, text, self.pos);
            const body = open + 2 + @intFromBool(sign != 0);
            switch (kind) {
                '#' => line_starting = try self.comment(open, body),
                '%' => {
                    if (self.isRaw(body)) return self.fail(open, "unsupported statement 'raw'", .{});
                    try self.emit(.block_begin, "{%", open);
                    line_starting = try self.tag(body, .block_end);
                },
                else => {
                    try self.emit(.var_begin, "{{", open);
                    line_starting = try self.tag(body, .var_end);
                },
            }
        }
        try self.emit(.eof, "", self.src.len);
        return self.tokens.items;
    }

    fn nextOpen(self: *const Lexer, from: usize) ?usize {
        @setRuntimeSafety(true);
        var i = from;
        while (std.mem.indexOfScalarPos(u8, self.src, i, '{')) |at| {
            if (at + 1 < self.src.len) switch (self.src[at + 1]) {
                '{', '%', '#' => return at,
                else => {},
            };
            i = at + 1;
        }
        return null;
    }

    fn isRaw(self: *const Lexer, body: usize) bool {
        @setRuntimeSafety(true);
        var i = body;
        while (i < self.src.len and std.ascii.isWhitespace(self.src[i])) i += 1;
        if (!self.startsWith(i, "raw")) return false;
        i += 3;
        while (i < self.src.len and std.ascii.isWhitespace(self.src[i])) i += 1;
        return self.startsWith(i, "%}") or self.startsWith(i, "-%}");
    }

    /// A comment to its first `#}`; returns whether what it consumed ends a line.
    fn comment(self: *Lexer, open: usize, body: usize) Error!bool {
        @setRuntimeSafety(true);
        var i = body;
        while (true) : (i += 1) {
            if (i >= self.src.len) return self.fail(open, "missing end of comment tag", .{});
            if (self.startsWith(i, "+#}")) {
                self.pos = i + 3;
                return false;
            }
            if (self.startsWith(i, "-#}")) {
                self.pos = skipSpace(self.src, i + 3);
                return self.pos > 0 and self.src[self.pos - 1] == '\n';
            }
            if (self.startsWith(i, "#}")) {
                self.pos = i + 2;
                if (self.startsWith(self.pos, "\n")) {
                    self.pos += 1;
                    return true;
                }
                return false;
            }
        }
    }

    /// The tokens of a `{% %}` or `{{ }}` tag through its end; returns whether what it consumed ends a line.
    fn tag(self: *Lexer, body: usize, end: Tag) Error!bool {
        @setRuntimeSafety(true);
        self.pos = body;
        var balance = std.ArrayList(u8).empty;
        while (true) {
            if (self.pos >= self.src.len) return self.fail(self.pos, "unexpected end of template, expected '{s}'", .{if (end == .block_end) "%}" else "}}"});
            if (balance.items.len == 0) {
                if (end == .block_end) {
                    if (self.startsWith(self.pos, "+%}")) {
                        try self.emit(.block_end, "%}", self.pos);
                        self.pos += 3;
                        return false;
                    }
                    if (self.startsWith(self.pos, "-%}")) {
                        try self.emit(.block_end, "%}", self.pos);
                        self.pos = skipSpace(self.src, self.pos + 3);
                        return self.src[self.pos - 1] == '\n';
                    }
                    if (self.startsWith(self.pos, "%}")) {
                        try self.emit(.block_end, "%}", self.pos);
                        self.pos += 2;
                        if (self.startsWith(self.pos, "\n")) {
                            self.pos += 1;
                            return true;
                        }
                        return false;
                    }
                } else {
                    if (self.startsWith(self.pos, "-}}")) {
                        try self.emit(.var_end, "}}", self.pos);
                        self.pos = skipSpace(self.src, self.pos + 3);
                        return self.src[self.pos - 1] == '\n';
                    }
                    if (self.startsWith(self.pos, "}}")) {
                        try self.emit(.var_end, "}}", self.pos);
                        self.pos += 2;
                        return false;
                    }
                }
            }
            const space_end = skipSpace(self.src, self.pos);
            if (space_end > self.pos) {
                self.pos = space_end;
                continue;
            }
            try self.token(&balance);
        }
    }

    fn token(self: *Lexer, balance: *std.ArrayList(u8)) Error!void {
        @setRuntimeSafety(true);
        const at = self.pos;
        const c = self.src[at];
        if (std.ascii.isDigit(c)) {
            if (self.float(at)) |end| {
                try self.emit(.float, try stripUnderscores(self.arena, self.src[at..end]), at);
                self.pos = end;
                return;
            }
            const end = self.integer(at) orelse return self.fail(at, "invalid integer literal", .{});
            try self.emit(.integer, self.src[at..end], at);
            self.pos = end;
            return;
        }
        if (std.ascii.isAlphabetic(c) or c == '_') {
            var end = at + 1;
            while (end < self.src.len and (std.ascii.isAlphanumeric(self.src[end]) or self.src[end] == '_')) end += 1;
            try self.emit(.name, self.src[at..end], at);
            self.pos = end;
            return;
        }
        if (c == '\'' or c == '"') return self.string(at, c);
        for (operators) |op| {
            if (!self.startsWith(at, op)) continue;
            switch (op[0]) {
                '(' => try balance.append(self.arena, ')'),
                '[' => try balance.append(self.arena, ']'),
                '{' => try balance.append(self.arena, '}'),
                ')', ']', '}' => {
                    const want = balance.pop() orelse return self.fail(at, "unexpected '{c}'", .{op[0]});
                    if (want != op[0]) return self.fail(at, "unexpected '{c}', expected '{c}'", .{ op[0], want });
                },
                else => {},
            }
            try self.emit(.op, op, at);
            self.pos = at + op.len;
            return;
        }
        return self.fail(at, "unexpected char '{s}'", .{charAt(self.src, at)});
    }

    /// Jinja2's float: digits, then a fraction and/or an exponent; never right after a `.` (`x.1.0` subscripts).
    fn float(self: *const Lexer, at: usize) ?usize {
        @setRuntimeSafety(true);
        if (at > 0 and self.src[at - 1] == '.') return null;
        var i = digitsEnd(self.src, at) orelse return null;
        var frac = false;
        if (i < self.src.len and self.src[i] == '.') {
            if (digitsEnd(self.src, i + 1)) |e| {
                i = e;
                frac = true;
            }
        }
        if (i < self.src.len and (self.src[i] == 'e' or self.src[i] == 'E')) {
            var j = i + 1;
            if (j < self.src.len and (self.src[j] == '+' or self.src[j] == '-')) j += 1;
            if (digitsEnd(self.src, j)) |e| return e;
        }
        return if (frac) i else null;
    }

    fn integer(self: *const Lexer, at: usize) ?usize {
        @setRuntimeSafety(true);
        const s = self.src;
        if (s[at] == '0' and at + 1 < s.len) {
            const base: u8 = switch (s[at + 1]) {
                'b', 'B' => 2,
                'o', 'O' => 8,
                'x', 'X' => 16,
                else => 10,
            };
            if (base != 10) {
                var i = at + 2;
                var n: usize = 0;
                while (i < s.len) {
                    if (s[i] == '_' and i + 1 < s.len and isBaseDigit(s[i + 1], base)) i += 1;
                    if (i < s.len and isBaseDigit(s[i], base)) {
                        i += 1;
                        n += 1;
                    } else break;
                }
                return if (n > 0) i else null;
            }
        }
        if (s[at] == '0') {
            var i = at + 1;
            while (i < s.len) {
                if (s[i] == '0') i += 1 else if (s[i] == '_' and i + 1 < s.len and s[i + 1] == '0') i += 2 else break;
            }
            return i;
        }
        return digitsEnd(s, at);
    }

    fn string(self: *Lexer, at: usize, quote: u8) Error!void {
        @setRuntimeSafety(true);
        var out = std.ArrayList(u8).empty;
        var i = at + 1;
        while (true) {
            if (i >= self.src.len) return self.fail(at, "unexpected char '{c}' (an unterminated string)", .{quote});
            const c = self.src[i];
            if (c == quote) break;
            if (c != '\\') {
                try out.append(self.arena, c);
                i += 1;
                continue;
            }
            if (i + 1 >= self.src.len) return self.fail(at, "unexpected char '{c}' (an unterminated string)", .{quote});
            i = try self.escape(&out, i);
        }
        self.pos = i + 1;
        try self.emit(.string, out.items, at);
    }

    /// Python's `unicode-escape` decoding of the escape at `i`; returns where the text goes on.
    fn escape(self: *Lexer, out: *std.ArrayList(u8), i: usize) Error!usize {
        @setRuntimeSafety(true);
        const e = self.src[i + 1];
        const simple: ?u8 = switch (e) {
            '\\' => '\\',
            '\'' => '\'',
            '"' => '"',
            'a' => 0x07,
            'b' => 0x08,
            'f' => 0x0C,
            'n' => '\n',
            'r' => '\r',
            't' => '\t',
            'v' => 0x0B,
            else => null,
        };
        if (simple) |c| {
            try out.append(self.arena, c);
            return i + 2;
        }
        if (e == '\n') return i + 2;
        if (e >= '0' and e <= '7') {
            var j = i + 1;
            var cp: u21 = 0;
            while (j < self.src.len and j < i + 4 and self.src[j] >= '0' and self.src[j] <= '7') : (j += 1) cp = cp * 8 + (self.src[j] - '0');
            try appendCodepoint(self.arena, out, cp);
            return j;
        }
        const width: usize = switch (e) {
            'x' => 2,
            'u' => 4,
            'U' => 8,
            else => {
                if (e == 'N') return self.fail(i, "unsupported string escape '\\N'", .{});
                try out.append(self.arena, '\\');
                return i + 1;
            },
        };
        if (i + 2 + width > self.src.len) return self.fail(i, "truncated \\{c} escape", .{e});
        const hex = self.src[i + 2 .. i + 2 + width];
        // Exactly `width` hex digits, as Python reads them: parseInt alone would take a sign or an underscore.
        for (hex) |c| if (!std.ascii.isHex(c)) return self.fail(i, "truncated \\{c} escape", .{e});
        const cp = std.fmt.parseInt(u32, hex, 16) catch unreachable;
        if (cp > 0x10FFFF or (cp >= 0xD800 and cp <= 0xDFFF)) return self.fail(i, "string escape \\{c}{s} is not a character", .{ e, hex });
        try appendCodepoint(self.arena, out, @intCast(cp));
        return i + 2 + width;
    }
};

fn appendCodepoint(arena: Allocator, out: *std.ArrayList(u8), cp: u21) Error!void {
    @setRuntimeSafety(true);
    var buf: [4]u8 = undefined;
    const n = std.unicode.utf8Encode(cp, &buf) catch return error.TemplateSyntax;
    try out.appendSlice(arena, buf[0..n]);
}

fn isBaseDigit(c: u8, base: u8) bool {
    @setRuntimeSafety(true);
    return switch (base) {
        2 => c == '0' or c == '1',
        8 => c >= '0' and c <= '7',
        else => std.ascii.isHex(c),
    };
}

/// The end of `\d+(_\d+)*` from `at`, or null.
fn digitsEnd(s: []const u8, at: usize) ?usize {
    @setRuntimeSafety(true);
    if (at >= s.len or !std.ascii.isDigit(s[at])) return null;
    var i = at + 1;
    while (i < s.len) {
        if (std.ascii.isDigit(s[i])) i += 1 else if (s[i] == '_' and i + 1 < s.len and std.ascii.isDigit(s[i + 1])) i += 2 else break;
    }
    return i;
}

fn stripUnderscores(arena: Allocator, s: []const u8) Error![]const u8 {
    @setRuntimeSafety(true);
    if (std.mem.indexOfScalar(u8, s, '_') == null) return s;
    var out = std.ArrayList(u8).empty;
    for (s) |c| if (c != '_') try out.append(arena, c);
    return out.items;
}

fn charAt(s: []const u8, at: usize) []const u8 {
    @setRuntimeSafety(true);
    const len = std.unicode.utf8ByteSequenceLength(s[at]) catch 1;
    return s[at..@min(s.len, at + len)];
}

/// One whitespace code point at `at` (Python's `\s`), or null.
fn spaceAt(s: []const u8, at: usize) ?usize {
    @setRuntimeSafety(true);
    if (at >= s.len) return null;
    const len = std.unicode.utf8ByteSequenceLength(s[at]) catch return null;
    if (at + len > s.len) return null;
    const cp = std.unicode.utf8Decode(s[at .. at + len]) catch return null;
    return if (value.isSpace(cp)) len else null;
}

pub fn skipSpace(s: []const u8, from: usize) usize {
    @setRuntimeSafety(true);
    var i = from;
    while (spaceAt(s, i)) |len| i += len;
    return i;
}

/// Python's `str.rstrip()`.
pub fn rstripSpace(s: []const u8) []const u8 {
    @setRuntimeSafety(true);
    var end = s.len;
    while (end > 0) {
        var start = end - 1;
        while (start > 0 and s[start] & 0xC0 == 0x80) start -= 1;
        if (spaceAt(s, start)) |len| {
            if (start + len != end) break;
            end = start;
        } else break;
    }
    return s[0..end];
}

fn allSpace(s: []const u8) bool {
    @setRuntimeSafety(true);
    return skipSpace(s, 0) == s.len;
}

pub fn tokenize(arena: Allocator, src: []const u8, err: *SyntaxError) Error![]const Token {
    @setRuntimeSafety(true);
    var lexer = Lexer{ .arena = arena, .src = src, .err = err };
    return lexer.run();
}
