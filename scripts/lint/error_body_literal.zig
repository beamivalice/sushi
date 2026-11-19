//! Rule `error-body-literal`: an error envelope spelled outside `src/http.zig`, whose `sendError` is the one error
//! sender: a body built by hand misses the request's API (an OpenAI body on an Anthropic route), the HEAD rule and the
//! WebSocket frame. The envelope is found by its key, however the literal is spaced or split: an `"error"` key opening
//! an object (in a literal, a format's quoted `{s}` key, or a `.@"error"` field built for a JSON writer), or `"type"`
//! set to `"error"`. Tests and `*_test.zig` files read envelopes and are not scanned. Token scan.
const std = @import("std");
const Ast = std.zig.Ast;
const rule = @import("rule.zig");

pub const name = "error-body-literal";
pub const reason = "every error body goes through http.sendError, which picks the envelope by the request's API";

/// One literal's text: a string literal decoded, or a multiline literal's lines joined. `first`/`last` are its tokens.
const Unit = struct { text: []const u8, first: u32, last: u32 };

fn isLiteral(tag: std.zig.Token.Tag) bool {
    return tag == .string_literal or tag == .multiline_string_literal_line;
}

/// The literal that starts at token `t`.
fn unitAt(arena: std.mem.Allocator, tree: *const Ast, t: u32) !Unit {
    if (tree.tokenTag(t) == .string_literal) {
        const raw = tree.tokenSlice(t);
        const text = std.zig.string_literal.parseAlloc(arena, raw) catch raw;
        return .{ .text = text, .first = t, .last = t };
    }
    var text: std.ArrayList(u8) = .empty;
    var last = t;
    while (last < tree.tokens.len and tree.tokenTag(last) == .multiline_string_literal_line) : (last += 1) {
        const line = tree.tokenSlice(last);
        try text.appendSlice(arena, std.mem.trimEnd(u8, line[2..], "\r\n"));
        try text.append(arena, '\n');
    }
    return .{ .text = text.items, .first = t, .last = last - 1 };
}

const ws = " \t\r\n";

/// `"<key>"` followed by a colon at `at`; returns the index past the colon.
fn keyAt(text: []const u8, at: usize, key: []const u8) ?usize {
    if (!std.mem.startsWith(u8, text[at..], key)) return null;
    var i = at + key.len;
    while (i < text.len and std.mem.indexOfScalar(u8, ws, text[i]) != null) i += 1;
    return if (i < text.len and text[i] == ':') i + 1 else null;
}

/// The last non-blank byte before `at`, or null when only blanks precede it.
fn before(text: []const u8, at: usize) ?u8 {
    var i = at;
    while (i > 0) {
        i -= 1;
        if (std.mem.indexOfScalar(u8, ws, text[i]) == null) return text[i];
    }
    return null;
}

/// A key `key` opening an object: after `{`, or first in a literal joined on by `++`.
fn opensObject(tree: *const Ast, u: Unit, key: []const u8) bool {
    var at: usize = 0;
    while (std.mem.indexOfPos(u8, u.text, at, key)) |i| : (at = i + 1) {
        if (keyAt(u.text, i, key) == null) continue;
        if (before(u.text, i)) |c| {
            if (c == '{') return true;
        } else if (u.first > 0 and tree.tokenTag(u.first - 1) == .plus_plus) return true;
    }
    return false;
}

/// `"type"` set to `"error"`.
fn typeError(text: []const u8) bool {
    var at: usize = 0;
    while (std.mem.indexOfPos(u8, text, at, "\"type\"")) |i| : (at = i + 1) {
        var j = keyAt(text, i, "\"type\"") orelse continue;
        while (j < text.len and std.mem.indexOfScalar(u8, ws, text[j]) != null) j += 1;
        if (std.mem.startsWith(u8, text[j..], "\"error\"")) return true;
    }
    return false;
}

/// The literal `"error"` as an argument of the `.{ ... }` that follows a format whose quoted placeholder is a key.
fn formatKeyArg(arena: std.mem.Allocator, tree: *const Ast, t: u32) !bool {
    var depth: u32 = 0;
    var i = t;
    while (i > 1) {
        i -= 1;
        switch (tree.tokenTag(i)) {
            .r_brace => depth += 1,
            .l_brace => if (depth == 0) break else {
                depth -= 1;
            },
            .semicolon => return false,
            else => {},
        }
    }
    if (tree.tokenTag(i) != .l_brace or tree.tokenTag(i - 1) != .period or i < 3 or tree.tokenTag(i - 2) != .comma) return false;
    var f = i - 3;
    if (!isLiteral(tree.tokenTag(f))) return false;
    while (f > 0 and tree.tokenTag(f) == .multiline_string_literal_line and tree.tokenTag(f - 1) == .multiline_string_literal_line) f -= 1;
    const u = try unitAt(arena, tree, f);
    for ([_][]const u8{ "\"{s}\"", "\"{}\"", "\"{any}\"" }) |key| if (opensObject(tree, u, key)) return true;
    return false;
}

pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    if (std.mem.eql(u8, report.file, "src/http.zig") or std.mem.endsWith(u8, report.file, "_test.zig")) return;
    const owners = try rule.owners(arena, tree);
    var t: u32 = 0;
    while (t < tree.tokens.len) : (t += 1) {
        const tag = tree.tokenTag(t);
        const hit, const next = if (isLiteral(tag)) blk: {
            const u = try unitAt(arena, tree, t);
            const found = opensObject(tree, u, "\"error\"") or typeError(u.text) or
                (std.mem.eql(u8, u.text, "error") and try formatKeyArg(arena, tree, t));
            break :blk .{ found, u.last };
        } else .{
            // `.{ .@"error" = ... }`: an envelope built for a JSON writer, the key opening the object.
            tag == .identifier and std.mem.eql(u8, tree.tokenSlice(t), "@\"error\"") and t > 1 and
                tree.tokenTag(t - 1) == .period and tree.tokenTag(t + 1) == .equal and tree.tokenTag(t - 2) == .l_brace,
            t,
        };
        defer t = next;
        if (!hit or rule.inTest(owners.items, t)) continue;
        const key = if (rule.ownerOf(owners.items, t)) |o| o.key else "-";
        try report.add(tree, t, key, "a hand-built error body: send it through http.sendError", .{});
    }
}
