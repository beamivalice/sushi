//! Rule `log-line-end`: a `log.info|warn|debug|err` call whose format is one string literal with no line end. The
//! logger writes the bytes it is given, so the next line of the log starts on this one. `"{s}"` alone passes: it
//! prints a line that was built whole. Token scan of the parsed file, no Sema.
const std = @import("std");
const Ast = std.zig.Ast;
const rule = @import("rule.zig");

pub const name = "log-line-end";
pub const reason = "the logger adds no line end: a format without one runs the next log line into this one";

const levels = [_][]const u8{ "info", "warn", "debug", "err" };

pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    var owners: ?std.ArrayList(rule.Owner) = null;
    var t: u32 = 1;
    while (t + 5 < tree.tokens.len) : (t += 1) {
        if (tree.tokenTag(t) != .identifier or !std.mem.eql(u8, tree.tokenSlice(t), "log")) continue;
        if (tree.tokenTag(t - 1) == .period or tree.tokenTag(t + 1) != .period or tree.tokenTag(t + 2) != .identifier) continue;
        // A format joined with `++` ends in its last piece, which this scan does not follow.
        if (tree.tokenTag(t + 3) != .l_paren or tree.tokenTag(t + 4) != .string_literal or tree.tokenTag(t + 5) != .comma) continue;
        const level = tree.tokenSlice(t + 2);
        for (levels) |l| {
            if (std.mem.eql(u8, level, l)) break;
        } else continue;
        const format = tree.tokenSlice(t + 4);
        if (std.mem.endsWith(u8, format, "\\n\"") or std.mem.eql(u8, format, "\"{s}\"")) continue;
        if (owners == null) owners = try rule.owners(arena, tree);
        const key = if (rule.ownerOf(owners.?.items, t)) |o| o.key else "-";
        try report.add(tree, t + 4, key, "log.{s} format has no line end", .{level});
    }
}
