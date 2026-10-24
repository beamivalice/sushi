//! DeepSeek-V4.x pre-tokenizer: the three isolating Splits of its tokenizer.json, in order, each applied to
//! the pieces the one before left. A piece never crosses a run's edge, so a `\s+(?!\S)` at the end of a piece
//! sees the end of input (`"x  1"` keeps its two spaces whole).
const std = @import("std");
const uni = @import("tokenizer_unicode.zig");

pub const digits_regex = "\\p{N}{1,3}";
pub const cjk_regex = "[\u{4e00}-\u{9fa5}\u{3040}-\u{309f}\u{30a0}-\u{30ff}]+";
pub const main_regex = "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+";

const Cp = struct { cp: u21, len: usize };

/// An invalid byte is its own one-byte codepoint of no class.
fn decode(text: []const u8, i: usize) ?Cp {
    if (i >= text.len) return null;
    const n = std.unicode.utf8ByteSequenceLength(text[i]) catch return .{ .cp = 0xFFFD, .len = 1 };
    if (i + n > text.len) return .{ .cp = 0xFFFD, .len = 1 };
    const cp = std.unicode.utf8Decode(text[i .. i + n]) catch return .{ .cp = 0xFFFD, .len = 1 };
    return .{ .cp = cp, .len = n };
}

fn isNumber(cp: u21) bool {
    return uni.category(cp) == .number;
}

fn isCjk(cp: u21) bool {
    return cp >= 0x4E00 and cp <= 0x9FA5 or cp >= 0x3040 and cp <= 0x30FF;
}

fn isAsciiPunct(cp: u21) bool {
    return cp >= 0x21 and cp <= 0x2F or cp >= 0x3A and cp <= 0x40 or cp >= 0x5B and cp <= 0x60 or cp >= 0x7B and cp <= 0x7E;
}

fn isAsciiLetter(cp: u21) bool {
    return cp >= 'a' and cp <= 'z' or cp >= 'A' and cp <= 'Z';
}

fn isPunctOrSymbol(cp: u21) bool {
    return switch (uni.category(cp)) {
        .punct, .symbol => true,
        else => false,
    };
}

fn isLetterOrMark(cp: u21) bool {
    return switch (uni.category(cp)) {
        .letter, .mark => true,
        else => false,
    };
}

fn isNewline(cp: u21) bool {
    return cp == '\r' or cp == '\n';
}

/// `[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+`
fn matchPunctThenLetters(text: []const u8, i: usize) ?usize {
    const first = decode(text, i) orelse return null;
    if (!isAsciiPunct(first.cp)) return null;
    var end = i + first.len;
    while (decode(text, end)) |c| : (end += c.len) if (!isAsciiLetter(c.cp)) break;
    return if (end > i + first.len) end else null;
}

/// `[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+`: the optional lead is tried first, then dropped.
fn matchWord(text: []const u8, i: usize) ?usize {
    const first = decode(text, i) orelse return null;
    if (!isNewline(first.cp) and uni.category(first.cp) != .letter and !isPunctOrSymbol(first.cp)) {
        var end = i + first.len;
        while (decode(text, end)) |c| : (end += c.len) if (!isLetterOrMark(c.cp)) break;
        if (end > i + first.len) return end;
    }
    var end = i;
    while (decode(text, end)) |c| : (end += c.len) if (!isLetterOrMark(c.cp)) break;
    return if (end > i) end else null;
}

/// ` ?[\p{P}\p{S}]+[\r\n]*`
fn matchPunctRun(text: []const u8, i: usize) ?usize {
    var end = i;
    if (end < text.len and text[end] == ' ') end += 1;
    const body = end;
    while (decode(text, end)) |c| : (end += c.len) if (!isPunctOrSymbol(c.cp)) break;
    if (end == body) return null;
    while (end < text.len and (text[end] == '\r' or text[end] == '\n')) end += 1;
    return end;
}

/// The end of the whitespace run at `i`, and where its last codepoint starts.
fn whitespaceRun(text: []const u8, i: usize) struct { end: usize, last: usize, newline_end: ?usize } {
    var end = i;
    var last = i;
    var newline_end: ?usize = null;
    while (decode(text, end)) |c| : (end += c.len) {
        if (!uni.whitespace(c.cp)) break;
        last = end;
        if (isNewline(c.cp)) newline_end = end + c.len;
    }
    return .{ .end = end, .last = last, .newline_end = newline_end };
}

fn matchMain(text: []const u8, i: usize) ?usize {
    if (matchPunctThenLetters(text, i)) |e| return e;
    if (matchWord(text, i)) |e| return e;
    if (matchPunctRun(text, i)) |e| return e;
    const run = whitespaceRun(text, i);
    if (run.end == i) return null;
    // `\s*[\r\n]+` ends after the run's last newline; `\s+(?!\S)` gives its last codepoint to what follows.
    if (run.newline_end) |e| return e;
    if (run.end == text.len) return run.end;
    if (run.last > i) return run.last;
    return run.end;
}

fn mainPieces(allocator: std.mem.Allocator, piece: []const u8, words: *std.ArrayList([]const u8)) !void {
    var i: usize = 0;
    var gap: ?usize = null;
    while (i < piece.len) {
        if (matchMain(piece, i)) |end| {
            if (gap) |g| try words.append(allocator, try allocator.dupe(u8, piece[g..i]));
            gap = null;
            try words.append(allocator, try allocator.dupe(u8, piece[i..end]));
            i = end;
        } else {
            if (gap == null) gap = i;
            i += (decode(piece, i) orelse break).len;
        }
    }
    if (gap) |g| try words.append(allocator, try allocator.dupe(u8, piece[g..]));
}

const Range = struct { start: usize, end: usize };

/// `\p{N}{1,3}` isolated: groups of up to three digits and the text between them.
fn digitPieces(allocator: std.mem.Allocator, text: []const u8, out: *std.ArrayList(Range)) !void {
    var i: usize = 0;
    var gap: usize = 0;
    while (decode(text, i)) |c| {
        if (!isNumber(c.cp)) {
            i += c.len;
            continue;
        }
        if (gap < i) try out.append(allocator, .{ .start = gap, .end = i });
        const start = i;
        var taken: usize = 0;
        while (taken < 3) : (taken += 1) {
            const d = decode(text, i) orelse break;
            if (!isNumber(d.cp)) break;
            i += d.len;
        }
        try out.append(allocator, .{ .start = start, .end = i });
        gap = i;
    }
    if (gap < text.len) try out.append(allocator, .{ .start = gap, .end = text.len });
}

/// The CJK class isolated: each run whole, and the text between runs.
fn cjkPieces(allocator: std.mem.Allocator, text: []const u8, base: usize, out: *std.ArrayList(Range)) !void {
    var i: usize = 0;
    var start: usize = 0;
    var in_run = false;
    while (decode(text, i)) |c| : (i += c.len) {
        const cjk = isCjk(c.cp);
        if (cjk != in_run and i > start) {
            try out.append(allocator, .{ .start = base + start, .end = base + i });
            start = i;
        }
        in_run = cjk;
    }
    if (start < text.len) try out.append(allocator, .{ .start = base + start, .end = base + text.len });
}

pub fn preTokenize(allocator: std.mem.Allocator, text: []const u8, words: *std.ArrayList([]const u8)) !void {
    var first: std.ArrayList(Range) = .empty;
    defer first.deinit(allocator);
    try digitPieces(allocator, text, &first);
    var second: std.ArrayList(Range) = .empty;
    defer second.deinit(allocator);
    for (first.items) |r| try cjkPieces(allocator, text[r.start..r.end], r.start, &second);
    for (second.items) |r| try mainPieces(allocator, text[r.start..r.end], words);
}

const testing = std.testing;

fn expectPieces(input: []const u8, expected: []const []const u8) !void {
    var words: std.ArrayList([]const u8) = .empty;
    defer {
        for (words.items) |w| testing.allocator.free(w);
        words.deinit(testing.allocator);
    }
    try preTokenize(testing.allocator, input, &words);
    try testing.expectEqual(expected.len, words.items.len);
    for (expected, words.items) |e, a| try testing.expectEqualStrings(e, a);
}

// The expected pieces are HF `tokenizers`' for the three Splits as the release ships them
// (`Sequence` of `Split(..., "isolated")`, no ByteLevel).
test "deepseek pre-tokenizer: digits, CJK and the punctuation grammar split as HF's Splits do" {
    try expectPieces("Hello world", &.{ "Hello", " world" });
    try expectPieces("x  1", &.{ "x", "  ", "1" });
    try expectPieces("12345 + 6789", &.{ "123", "45", " +", " ", "678", "9" });
    try expectPieces("foo.bar(baz)_qux", &.{ "foo", ".bar", "(baz", ")_", "qux" });
    try expectPieces("a\n\n  b", &.{ "a", "\n\n", " ", " b" });
    try expectPieces("天气 怎么样？今天は晴れです。", &.{ "天气", " ", "怎么样", "？", "今天は晴れです", "。" });
    try expectPieces("زار موسى عيسى ١٢٣", &.{ "زار", " موسى", " عيسى", " ", "١٢٣" });
    try expectPieces("$100, 50% off!!!", &.{ "$", "100", ",", " ", "50", "%", " off", "!!!" });
    try expectPieces("  indented\n", &.{ " ", " indented", "\n" });
    try expectPieces("a.привет b_c", &.{ "a", ".", "привет", " b", "_c" });
    try expectPieces("\u{200b}\u{200b}x", &.{ "\u{200b}", "\u{200b}x" });
    try expectPieces("ア・イ カタカナ", &.{ "ア", "・", "イ", " ", "カタカナ" });
    try expectPieces("tab\there\t\tx", &.{ "tab", "\there", "\t", "\tx" });
    try expectPieces("x\r\n\r\n y", &.{ "x", "\r\n\r\n", " y" });
    try expectPieces("word  \n  \nnext", &.{ "word", "  \n  \n", "next" });
    try expectPieces("été àéîõü ǅ ŉ ﬁ ①", &.{ "été", " àéîõü", " ǅ", " ŉ", " ﬁ", " ", "①" });
    try expectPieces("a1b22c333d4444", &.{ "a", "1", "b", "22", "c", "333", "d", "444", "4" });
    try expectPieces("https://example.com/a_b?x=1&y=2#frag", &.{ "https", "://", "example", ".com", "/a", "_b", "?x", "=", "1", "&y", "=", "2", "#frag" });
    try expectPieces("if (x == 0) { return -1; }", &.{ "if", " (", "x", " ==", " ", "0", ")", " {", " return", " -", "1", ";", " }" });
    try expectPieces(" \u{3000}x\u{3000} y", &.{ " ", "\u{3000}x", "\u{3000}", " y" });
    try expectPieces("e\u{301}a \u{301}b", &.{ "e\u{301}a", " \u{301}b" });
}
