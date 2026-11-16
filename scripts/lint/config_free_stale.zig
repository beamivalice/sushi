//! Rule `config-free-stale`: `if (slot) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);` whose next statement does
//! not assign `slot`. A rebuild that fails after the free leaves the cache naming a freed config under the old key, and
//! the next call with that key launches it. Token scan of the parsed file, no Sema.
const std = @import("std");
const Ast = std.zig.Ast;
const rule = @import("rule.zig");

pub const name = "config-free-stale";
pub const reason = "a freed kernel config left in its cache slot is launched by the next call with that key";

pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    const owners = try rule.owners(arena, tree);
    var t: u32 = 0;
    while (t + 2 < tree.tokens.len) : (t += 1) {
        if (tree.tokenTag(t) != .keyword_if or tree.tokenTag(t + 1) != .l_paren) continue;
        const slot_first = t + 2;
        const close = matching(tree, t + 1) orelse continue;
        var u = close + 1;
        if (!seq(tree, u, &.{ .pipe, .identifier, .pipe, .identifier, .equal, .identifier, .period, .identifier, .l_paren, .identifier, .r_paren, .semicolon })) continue;
        if (!std.mem.eql(u8, tree.tokenSlice(u + 7), "mlx_fast_metal_kernel_config_free")) continue;
        u += 12;
        const n = close - slot_first;
        var same = u + n < tree.tokens.len and tree.tokenTag(u + n) == .equal;
        var i: u32 = 0;
        while (same and i < n) : (i += 1) same = std.mem.eql(u8, tree.tokenSlice(slot_first + i), tree.tokenSlice(u + i));
        if (same) continue;
        const owner = if (rule.ownerOf(owners.items, t)) |o| o.key else "-";
        try report.add(tree, t, owner, "a kernel config freed and left in its slot: assign the slot (null, or the new config) next", .{});
    }
}

fn matching(tree: *const Ast, open: u32) ?u32 {
    var depth: u32 = 0;
    var t = open;
    while (t < tree.tokens.len) : (t += 1) switch (tree.tokenTag(t)) {
        .l_paren => depth += 1,
        .r_paren => {
            depth -= 1;
            if (depth == 0) return t;
        },
        .semicolon, .l_brace => return null,
        else => {},
    };
    return null;
}

fn seq(tree: *const Ast, at: u32, tags: []const std.zig.Token.Tag) bool {
    if (at + tags.len > tree.tokens.len) return false;
    for (tags, 0..) |tag, i| if (tree.tokenTag(at + @as(u32, @intCast(i))) != tag) return false;
    return true;
}
