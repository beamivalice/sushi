//! Rule `error-fallback`: `(f(..) catch null) orelse break :label`. The idiom of a fused arm whose null means "keep
//! the composed chain": `catch null` turns an MLX failure into that fallback too, so a failed request runs on and
//! reads as a declined arm. Token scan of the parsed file, no Sema.
const std = @import("std");
const Ast = std.zig.Ast;
const rule = @import("rule.zig");

pub const name = "error-fallback";
pub const reason = "catch null before orelse break turns an error into the fallback a null asks for";

pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    const owners = try rule.owners(arena, tree);
    var t: u32 = 0;
    while (t + 4 < tree.tokens.len) : (t += 1) {
        if (tree.tokenTag(t) != .keyword_catch) continue;
        if (tree.tokenTag(t + 1) != .identifier or !std.mem.eql(u8, tree.tokenSlice(t + 1), "null")) continue;
        if (tree.tokenTag(t + 2) != .r_paren or tree.tokenTag(t + 3) != .keyword_orelse or tree.tokenTag(t + 4) != .keyword_break) continue;
        const owner = if (rule.ownerOf(owners.items, t)) |o| o.key else "-";
        try report.add(tree, t, owner, "an error caught into the null fallback: try the call, so only a null takes the fallback", .{});
    }
}
