//! Rule `in-place-buffer`: a fn that makes a buffer with `mlx_zeros` and writes rows into one with `mlx_slice_update`,
//! outside `mlx_pages.zig`. A buffer written in place takes `mlx_pages.inPlaceBuffer`, which refuses one off MLX's page
//! grid: there MLX's pool can hand it a buffer it never donates, and every write copies all of it. Tests are exempt.
//! Token scan per fn.
const std = @import("std");
const Ast = std.zig.Ast;
const rule = @import("rule.zig");

pub const name = "in-place-buffer";
pub const reason = "a buffer off MLX's page grid copies on every write into it; make it with mlx_pages.inPlaceBuffer";

pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    if (std.mem.eql(u8, std.fs.path.basename(report.file), "mlx_pages.zig") or rule.testFile(report.file)) return;
    const owners = try rule.owners(arena, tree);
    const test_only = try rule.testOnly(arena, tree, owners.items);
    for (owners.items, test_only) |owner, only_tests| {
        if (owner.is_test or only_tests) continue;
        var zeros: ?u32 = null;
        var writes = false;
        var t = owner.first;
        while (t + 1 <= owner.last) : (t += 1) {
            if (tree.tokenTag(t) != .identifier or tree.tokenTag(t + 1) != .l_paren) continue;
            const callee = tree.tokenSlice(t);
            if (std.mem.eql(u8, callee, "mlx_zeros") and zeros == null) zeros = t;
            if (std.mem.eql(u8, callee, "mlx_slice_update")) writes = true;
        }
        const at = zeros orelse continue;
        if (!writes or innerOwner(owners.items, at, owner)) continue;
        try report.add(tree, at, owner.key, "a buffer made by mlx_zeros and written by slice_update: make it with mlx_pages.inPlaceBuffer", .{});
    }
}

/// Whether `t` sits in a fn nested inside `outer` (reported under that fn instead).
fn innerOwner(list: []const rule.Owner, t: u32, outer: rule.Owner) bool {
    const inner = rule.ownerOf(list, t) orelse return false;
    return inner.first != outer.first;
}
