//! Rule `error-body-literal`: an error envelope spelled outside `src/http.zig`, whose `ErrorEnvelope` is the one place
//! an error answer is built: a body spelled by hand misses the request's API (an OpenAI body on an Anthropic route),
//! the HEAD rule and the WebSocket frame. An envelope is an `error` key whose object holds a `message`, or a
//! `"type":"error"` marker beside an `error` object or a `message`, or an object opening with `"error":"text"`, spelled in any way a literal, a format, a JSON
//! writer or a struct can: the key alone as a literal (`"\"error\""`), as a format's quoted `{s}` key, as
//! `objectField("error")`, as `.@"error" = .{ .message = .. }` or as a non-optional `@"error": struct { message .. }`
//! field (a reader declares it optional). An `error` that is a plain value (`"error":null`, `.@"error" = n`), a schema
//! property or a marker a reader looks for is not one. Tests and `*_test.zig` files read envelopes and are not scanned.
//! Token scan.
const std = @import("std");
const Ast = std.zig.Ast;
const rule = @import("rule.zig");

pub const name = "error-body-literal";
pub const reason = "every error body is built by http.ErrorEnvelope, which picks the envelope by the request's API";

const message_found = "an error envelope spelled by hand: build it with http.ErrorEnvelope";

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

fn skipBlank(text: []const u8, from: usize) usize {
    var i = from;
    while (i < text.len and std.mem.indexOfScalar(u8, ws, text[i]) != null) i += 1;
    return i;
}

/// `key` (quotes included) followed by a colon at `at`; returns the index past the colon.
fn keyAt(text: []const u8, at: usize, key: []const u8) ?usize {
    if (!std.mem.startsWith(u8, text[at..], key)) return null;
    const i = skipBlank(text, at + key.len);
    return if (i < text.len and text[i] == ':') i + 1 else null;
}

/// Whether the text from `at` opens an object: one `{` (a literal) or two (a format's escaped brace), then a key.
/// A format's placeholder (`{s}`, `{f}`) is not one.
fn opensObject(text: []const u8, at: usize) bool {
    var i = skipBlank(text, at);
    if (i >= text.len or text[i] != '{') return false;
    i += 1;
    if (i < text.len and text[i] == '{') i += 1;
    i = skipBlank(text, i);
    return i < text.len and text[i] == '"';
}

/// An `"error"` key whose value is an object; `after` is where that object starts.
fn errorObject(text: []const u8) ?usize {
    var at: usize = 0;
    while (std.mem.indexOfPos(u8, text, at, "\"error\"")) |i| : (at = i + 1) {
        const v = keyAt(text, i, "\"error\"") orelse continue;
        if (opensObject(text, v)) return v;
    }
    return null;
}

/// `"type"` set to `"error"`.
fn typeError(text: []const u8) bool {
    var at: usize = 0;
    while (std.mem.indexOfPos(u8, text, at, "\"type\"")) |i| : (at = i + 1) {
        const j = skipBlank(text, keyAt(text, i, "\"type\"") orelse continue);
        if (std.mem.startsWith(u8, text[j..], "\"error\"")) return true;
    }
    return false;
}

fn hasMessage(text: []const u8) bool {
    var at: usize = 0;
    while (std.mem.indexOfPos(u8, text, at, "\"message\"")) |i| : (at = i + 1) {
        if (keyAt(text, i, "\"message\"") != null) return true;
    }
    return false;
}

/// An object that opens with an `"error"` key holding a string: `{"error":"..."}`, an error answer of no API.
fn opensWithErrorString(text: []const u8) bool {
    var at: usize = 0;
    while (std.mem.indexOfPos(u8, text, at, "\"error\"")) |i| : (at = i + 1) {
        const v = keyAt(text, i, "\"error\"") orelse continue;
        const j = skipBlank(text, v);
        if (j >= text.len or text[j] != '"') continue;
        var k = i;
        while (k > 0 and std.mem.indexOfScalar(u8, ws, text[k - 1]) != null) k -= 1;
        if (k > 0 and text[k - 1] == '{') return true;
    }
    return false;
}

/// A literal that spells an envelope whole: the error object with its message, or the type marker beside either.
fn spellsEnvelope(text: []const u8) bool {
    if (opensWithErrorString(text)) return true;
    const object = errorObject(text);
    if (object) |v| if (hasMessage(text[v..])) return true;
    return typeError(text) and (object != null or hasMessage(text));
}

/// The key alone, as a piece joined on or handed to a format: `"error"` or `"error":`, quotes included.
fn loneKey(text: []const u8) bool {
    const t = std.mem.trim(u8, text, ws);
    return std.mem.eql(u8, t, "\"error\"") or std.mem.eql(u8, t, "\"error\":");
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
    for ([_][]const u8{ "\"{s}\"", "\"{}\"", "\"{any}\"" }) |key| {
        var at: usize = 0;
        while (std.mem.indexOfPos(u8, u.text, at, key)) |k| : (at = k + 1) {
            if (keyAt(u.text, k, key)) |v| if (opensObject(u.text, v)) return true;
        }
    }
    return false;
}

/// The token past the `}` that closes the `{` at `open`, or null when it never closes.
fn closeOf(tree: *const Ast, open: u32) ?u32 {
    var depth: u32 = 0;
    var i = open;
    while (i < tree.tokens.len) : (i += 1) switch (tree.tokenTag(i)) {
        .l_brace => depth += 1,
        .r_brace => {
            depth -= 1;
            if (depth == 0) return i;
        },
        else => {},
    };
    return null;
}

/// Whether the braces from `open` to `close` hold a `message` field (`.message = ..` or `message: ..`).
fn holdsMessageField(tree: *const Ast, open: u32, close: u32) bool {
    var i = open;
    while (i + 1 < close) : (i += 1) {
        if (tree.tokenTag(i) == .identifier and std.mem.eql(u8, tree.tokenSlice(i), "message") and
            (tree.tokenTag(i + 1) == .equal or tree.tokenTag(i + 1) == .colon)) return true;
    }
    return false;
}

/// The innermost `{` that holds token `t`, or null.
fn enclosingBrace(tree: *const Ast, t: u32) ?u32 {
    var depth: u32 = 0;
    var i = t;
    while (i > 0) {
        i -= 1;
        switch (tree.tokenTag(i)) {
            .r_brace => depth += 1,
            .l_brace => if (depth == 0) return i else {
                depth -= 1;
            },
            else => {},
        }
    }
    return null;
}

fn isIdent(tree: *const Ast, t: u32, text: []const u8) bool {
    return tree.tokenTag(t) == .identifier and std.mem.eql(u8, tree.tokenSlice(t), text);
}

/// An envelope spelled in Zig syntax at token `t`: a JSON writer's `objectField("error")` then `beginObject`, a struct
/// literal `.@"error" = .{ .. message .. }`, a non-optional field `@"error": struct { .. message .. }`, or a
/// `.type = "error"` field beside a `message` field.
fn syntaxEnvelope(arena: std.mem.Allocator, tree: *const Ast, t: u32) !bool {
    const n = tree.tokens.len;
    if (isIdent(tree, t, "objectField") and t + 4 < n and tree.tokenTag(t + 1) == .l_paren and tree.tokenTag(t + 2) == .string_literal and
        std.mem.eql(u8, tree.tokenSlice(t + 2), "\"error\"") and tree.tokenTag(t + 3) == .r_paren)
    {
        var i = t + 4;
        while (i < n and i < t + 10) : (i += 1) if (isIdent(tree, i, "beginObject")) return true;
        return false;
    }
    if (isIdent(tree, t, "@\"error\"") and t + 3 < n) {
        // `.@"error" = .{ ... }` or `Env{ .@"error" = .{ ... } }`.
        if (tree.tokenTag(t - 1) == .period and tree.tokenTag(t + 1) == .equal and tree.tokenTag(t + 2) == .period and tree.tokenTag(t + 3) == .l_brace) {
            const close = closeOf(tree, t + 3) orelse return false;
            return holdsMessageField(tree, t + 3, close);
        }
        // `@"error": struct { message .. }`: a non-optional field; `?struct` (a reader's) is not one.
        if (tree.tokenTag(t + 1) == .colon and tree.tokenTag(t + 2) == .keyword_struct and tree.tokenTag(t + 3) == .l_brace) {
            const close = closeOf(tree, t + 3) orelse return false;
            return holdsMessageField(tree, t + 3, close);
        }
        return false;
    }
    // `.type = "error"` inside a struct literal that also sets `.message`.
    if (isIdent(tree, t, "type") and t > 0 and tree.tokenTag(t - 1) == .period and t + 2 < n and tree.tokenTag(t + 1) == .equal and
        tree.tokenTag(t + 2) == .string_literal and std.mem.eql(u8, tree.tokenSlice(t + 2), "\"error\""))
    {
        _ = arena;
        const open = enclosingBrace(tree, t) orelse return false;
        const close = closeOf(tree, open) orelse return false;
        return holdsMessageField(tree, open, close);
    }
    return false;
}

/// `ErrorEnvelope{ ... }`, or `: ErrorEnvelope = .{ ... }`, with the type named bare or as `http.ErrorEnvelope`. A fn
/// returning the type (`) ErrorEnvelope {`, `) !ErrorEnvelope {`) opens a body, not a literal.
fn builtEnvelope(tree: *const Ast, t: u32) bool {
    if (tree.tokenTag(t) != .identifier or !std.mem.eql(u8, tree.tokenSlice(t), "ErrorEnvelope") or t + 1 >= tree.tokens.len) return false;
    var before: u32 = t;
    if (before >= 2 and tree.tokenTag(before - 1) == .period and tree.tokenTag(before - 2) == .identifier) before -= 2;
    if (before == 0) return false;
    const next = tree.tokenTag(t + 1);
    if (next == .l_brace) {
        var b = before - 1;
        if (tree.tokenTag(b) == .bang or tree.tokenTag(b) == .question_mark) {
            if (b == 0) return true;
            b -= 1;
        }
        return tree.tokenTag(b) != .r_paren;
    }
    return next == .equal and t + 3 < tree.tokens.len and tree.tokenTag(t + 2) == .period and tree.tokenTag(t + 3) == .l_brace and
        tree.tokenTag(before - 1) == .colon;
}

pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    if (std.mem.eql(u8, report.file, "src/http.zig") or std.mem.endsWith(u8, report.file, "_test.zig")) return;
    const owners = try rule.owners(arena, tree);
    var t: u32 = 0;
    while (t < tree.tokens.len) : (t += 1) {
        const tag = tree.tokenTag(t);
        const hit, const next = if (builtEnvelope(tree, t)) .{ true, t } else if (isLiteral(tag)) blk: {
            const u = try unitAt(arena, tree, t);
            const found = spellsEnvelope(u.text) or loneKey(u.text) or
                (std.mem.eql(u8, u.text, "error") and try formatKeyArg(arena, tree, t));
            break :blk .{ found, u.last };
        } else .{ tag == .identifier and try syntaxEnvelope(arena, tree, t), t };
        defer t = next;
        if (!hit or rule.inTest(owners.items, t)) continue;
        const key = if (rule.ownerOf(owners.items, t)) |o| o.key else "-";
        try report.add(tree, t, key, message_found, .{});
    }
}
