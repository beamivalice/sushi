//! Rule `returned-handle-unguarded`: `var x = mlx.mlx_array_new();`, a `try` after it, then `return x;`, with no
//! `defer`/`errdefer` naming `x` before that first `try`. A checked call can fail after it filled `x` (a latched error),
//! and the error return then drops the handle the caller was to own. Token scan of the parsed file, no Sema.
const std = @import("std");
const Ast = std.zig.Ast;
const rule = @import("rule.zig");

pub const name = "returned-handle-unguarded";
pub const reason = "an error return between filling a returned handle and returning it leaks the handle";

pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    if (std.mem.endsWith(u8, report.file, "_test.zig")) return;
    const owners = try rule.owners(arena, tree);
    var t: u32 = 0;
    while (t + 8 < tree.tokens.len) : (t += 1) {
        if (tree.tokenTag(t) != .keyword_var or tree.tokenTag(t + 1) != .identifier or tree.tokenTag(t + 2) != .equal) continue;
        if (!isNew(tree, t + 3)) continue;
        const v = tree.tokenSlice(t + 1);
        if (unguardedReturn(tree, t + 9, v)) |at| {
            const owner = if (rule.ownerOf(owners.items, t)) |o| o.key else "-";
            try report.add(tree, at, owner, "'{s}' is returned, and an error before it leaks it: errdefer its free where it is made", .{v});
        }
    }
}

/// `mlx . mlx_array_new ( ) ;` at `at`.
fn isNew(tree: *const Ast, at: u32) bool {
    const tags = [_]std.zig.Token.Tag{ .identifier, .period, .identifier, .l_paren, .r_paren, .semicolon };
    for (tags, 0..) |tag, i| if (tree.tokenTag(at + @as(u32, @intCast(i))) != tag) return false;
    return std.mem.eql(u8, tree.tokenSlice(at), "mlx") and std.mem.eql(u8, tree.tokenSlice(at + 2), "mlx_array_new");
}

/// The `return` token of `return v;` reached after a `try` with no guard of `v` before that `try`, inside `v`'s block.
fn unguardedReturn(tree: *const Ast, from: u32, v: []const u8) ?u32 {
    var depth: i32 = 0;
    var tried = false;
    var t = from;
    while (t + 2 < tree.tokens.len) : (t += 1) {
        switch (tree.tokenTag(t)) {
            .l_brace => depth += 1,
            .r_brace => {
                depth -= 1;
                if (depth < 0) return null;
            },
            .keyword_try => tried = true,
            .keyword_defer, .keyword_errdefer => if (!tried and namesBeforeSemicolon(tree, t, v)) return null,
            .keyword_return => if (tree.tokenTag(t + 1) == .identifier and std.mem.eql(u8, tree.tokenSlice(t + 1), v) and
                tree.tokenTag(t + 2) == .semicolon)
            {
                return if (tried) t else null;
            },
            else => {},
        }
    }
    return null;
}

fn namesBeforeSemicolon(tree: *const Ast, from: u32, v: []const u8) bool {
    var t = from;
    while (t < tree.tokens.len and tree.tokenTag(t) != .semicolon) : (t += 1) {
        if (tree.tokenTag(t) == .identifier and std.mem.eql(u8, tree.tokenSlice(t), v)) return true;
    }
    return false;
}
