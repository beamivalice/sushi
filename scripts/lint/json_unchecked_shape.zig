//! Rule `json-unchecked-shape`: a `std.json.Value` read by its shape without checking it: `.object`, `.array`,
//! `.string`, `.integer`, `.float` or `.bool` taken off an unwrapped optional (`x.get("k").?.object`), off a parsed
//! document's `.value` (`parsed.value.object`) or off a map entry's `value_ptr`, with no comparison of that same
//! receiver to a tag (`== .object`, `!= .array`) earlier in the function; and any `@intCast(<expr>.integer)` or
//! `@intCast` of an `.integer => |i|` prong's capture, which panics or wraps on a value out of range (`std.math.cast` it
//! and fail by name). A file whose JSON is valid but of another shape then panics or reads the
//! wrong union field: one bad pack folder took the server down. Test code is exempt: tests, test files, and a
//! private fn that only tests (or other such fns) call. Token scan, no Sema.
const std = @import("std");
const Ast = std.zig.Ast;
const rule = @import("rule.zig");

pub const name = "json-unchecked-shape";
pub const reason = "a JSON file of the wrong shape is a named error, never a wrong union field";

const tags = [_][]const u8{ "object", "array", "string", "integer", "float", "bool" };

fn isTag(text: []const u8) bool {
    for (tags) |t| if (std.mem.eql(u8, t, text)) return true;
    return false;
}

/// The first token of the postfix chain that ends at `end` (identifiers, `.`, `.?`, calls and indexing).
fn chainStart(tree: *const Ast, end: u32) u32 {
    var t = end;
    while (t > 0) {
        switch (tree.tokenTag(t)) {
            .r_paren, .r_bracket => {
                const open: std.zig.Token.Tag = if (tree.tokenTag(t) == .r_paren) .l_paren else .l_bracket;
                const close = tree.tokenTag(t);
                var depth: usize = 0;
                while (t > 0) : (t -= 1) {
                    if (tree.tokenTag(t) == close) depth += 1;
                    if (tree.tokenTag(t) == open) {
                        depth -= 1;
                        if (depth == 0) break;
                    }
                }
                if (t == 0) return 0;
                t -= 1;
            },
            .identifier, .question_mark => {
                if (tree.tokenTag(t - 1) != .period) return t;
                t -= 2;
            },
            else => return t + 1,
        }
    }
    return 0;
}

pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    if (rule.testFile(report.file)) return;
    const owners = try rule.owners(arena, tree);
    const test_only = try rule.testOnly(arena, tree, owners.items);
    var t: u32 = 4;
    while (t < tree.tokens.len) : (t += 1) {
        if (tree.tokenTag(t) != .identifier or tree.tokenTag(t - 1) != .period or !isTag(tree.tokenSlice(t))) continue;
        if (t + 1 < tree.tokens.len and tree.tokenTag(t + 1) == .l_paren) continue; // a method of the same name
        const unwrapped = tree.tokenTag(t - 2) == .question_mark and tree.tokenTag(t - 3) == .period;
        const parsed_value = tree.tokenTag(t - 2) == .identifier and std.mem.eql(u8, tree.tokenSlice(t - 2), "value") and
            tree.tokenTag(t - 3) == .period and tree.tokenTag(t - 4) == .identifier and std.mem.startsWith(u8, tree.tokenSlice(t - 4), "parsed");
        const entry_value = tree.tokenTag(t - 2) == .identifier and std.mem.eql(u8, tree.tokenSlice(t - 2), "value_ptr") or
            (tree.tokenTag(t - 2) == .period_asterisk and tree.tokenTag(t - 3) == .identifier and std.mem.eql(u8, tree.tokenSlice(t - 3), "value_ptr"));
        if (!unwrapped and !parsed_value and !entry_value) continue;
        if (rule.inTest(owners.items, t)) continue;
        const owner = rule.ownerOf(owners.items, t);
        var held = false;
        for (owners.items, test_only) |o, m| if (m and t >= o.first and t <= o.last) {
            held = true;
        };
        if (held) continue;
        // The receiver's text, and whether the function compares it to a tag before this access.
        const start = chainStart(tree, t - 2);
        const src = tree.source;
        const from = tree.tokenStart(start);
        const to = tree.tokenStart(t - 1);
        const receiver = std.mem.trim(u8, src[from..to], " \n");
        const fn_start: usize = if (owner) |o| tree.tokenStart(o.first) else 0;
        const before = src[fn_start..from];
        var guarded = false;
        for ([_][]const u8{ " == .", " != .", ".* == .", ".* != ." }) |op| {
            const needle = try std.fmt.allocPrint(arena, "{s}{s}", .{ receiver, op });
            if (std.mem.indexOf(u8, before, needle) != null) guarded = true;
        }
        if (guarded) continue;
        try report.add(tree, t, if (owner) |o| o.key else "-", "unchecked JSON shape '{s}.{s}': check the tag first and fail by name", .{ receiver, tree.tokenSlice(t) });
    }
    // `@intCast(<expr>.integer)`: the argument's last token is `integer` after a period.
    t = 0;
    while (t + 1 < tree.tokens.len) : (t += 1) {
        if (tree.tokenTag(t) != .builtin or !std.mem.eql(u8, tree.tokenSlice(t), "@intCast") or tree.tokenTag(t + 1) != .l_paren) continue;
        var depth: usize = 0;
        var close = t + 1;
        while (close < tree.tokens.len) : (close += 1) {
            if (tree.tokenTag(close) == .l_paren) depth += 1;
            if (tree.tokenTag(close) == .r_paren) {
                depth -= 1;
                if (depth == 0) break;
            }
        }
        if (close < 2 or tree.tokenTag(close - 1) != .identifier or !std.mem.eql(u8, tree.tokenSlice(close - 1), "integer") or tree.tokenTag(close - 2) != .period) continue;
        if (rule.inTest(owners.items, t)) continue;
        const owner = rule.ownerOf(owners.items, t);
        // Guarded when the fn compares that same integer to a bound earlier (`x.integer < 0 or x.integer > max`).
        const int_expr = tree.source[tree.tokenStart(chainStart(tree, close - 1))..tree.tokenStart(close)];
        const fn_from: usize = if (owner) |o| tree.tokenStart(o.first) else 0;
        const earlier = tree.source[fn_from..tree.tokenStart(t)];
        var bounded = false;
        for ([_][]const u8{ " < ", " > ", " <= ", " >= " }) |op| {
            const needle = try std.fmt.allocPrint(arena, "{s}{s}", .{ int_expr, op });
            if (std.mem.indexOf(u8, earlier, needle) != null) bounded = true;
        }
        if (bounded) continue;
        var held = false;
        for (owners.items, test_only) |o, m| if (m and t >= o.first and t <= o.last) {
            held = true;
        };
        if (held) continue;
        try report.add(tree, t, if (owner) |o| o.key else "-", "@intCast of a JSON integer: cast it with std.math.cast and fail by name", .{});
    }
    // `.integer => |i| ... @intCast(i)`: a prong's capture is the same integer, and a one-sided guard
    // (`if (i < 0) null else`) still wraps past the target's top, so the capture is never cast unchecked.
    t = 0;
    while (t + 5 < tree.tokens.len) : (t += 1) {
        if (tree.tokenTag(t) != .period or tree.tokenTag(t + 1) != .identifier or !std.mem.eql(u8, tree.tokenSlice(t + 1), "integer")) continue;
        if (tree.tokenTag(t + 2) != .equal_angle_bracket_right or tree.tokenTag(t + 3) != .pipe or tree.tokenTag(t + 4) != .identifier or tree.tokenTag(t + 5) != .pipe) continue;
        if (rule.inTest(owners.items, t)) continue;
        const capture = tree.tokenSlice(t + 4);
        // The prong body runs to the `,` or `}` that closes it at its own depth.
        var depth: isize = 0;
        var b = t + 6;
        while (b + 3 < tree.tokens.len) : (b += 1) {
            switch (tree.tokenTag(b)) {
                .l_paren, .l_brace, .l_bracket => depth += 1,
                .r_paren, .r_bracket => depth -= 1,
                .r_brace => {
                    if (depth == 0) break;
                    depth -= 1;
                },
                .comma => if (depth == 0) break,
                else => {},
            }
            if (tree.tokenTag(b) != .builtin or !std.mem.eql(u8, tree.tokenSlice(b), "@intCast")) continue;
            if (tree.tokenTag(b + 1) != .l_paren or tree.tokenTag(b + 2) != .identifier or tree.tokenTag(b + 3) != .r_paren) continue;
            if (!std.mem.eql(u8, tree.tokenSlice(b + 2), capture)) continue;
            const owner = rule.ownerOf(owners.items, b);
            var held = false;
            for (owners.items, test_only) |o, m| if (m and b >= o.first and b <= o.last) {
                held = true;
            };
            if (held) continue;
            try report.add(tree, b, if (owner) |o| o.key else "-", "@intCast of a JSON integer: cast it with std.math.cast and fail by name", .{});
        }
    }
}
