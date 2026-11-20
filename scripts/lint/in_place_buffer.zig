//! Rule `in-place-buffer`: a fn that makes a buffer (`mlx_zeros`, `mlx_full`, a `mlx_concatenate_axis` that grows one, or
//! a scope's `zeros` / `join` over them) and writes rows into one with `mlx_slice_update`, outside `mlx_pages.zig`.
//! A buffer written in place takes `mlx_pages.inPlaceBuffer` or `grownBuffer`, which refuse one off MLX's page grid:
//! there MLX's pool can hand it a buffer it never donates, and every write copies all of it. Tests are exempt. Token
//! scan per fn: a buffer made in one fn and written in another is out of its reach.
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
            if (zeros == null and oneOf(callee, &makers)) zeros = t;
            if (oneOf(callee, &writers)) writes = true;
        }
        const at = zeros orelse continue;
        if (!writes or innerOwner(owners.items, at, owner)) continue;
        try report.add(tree, at, owner.key, "a buffer made by {s} and written by slice_update: make it with mlx_pages.inPlaceBuffer", .{tree.tokenSlice(at)});
    }
}

/// Calls that make a buffer: MLX's own, and the scope helpers a fn makes one through.
const makers = [_][]const u8{ "mlx_zeros", "mlx_full", "mlx_concatenate_axis", "zeros", "join" };
/// Calls that write rows into a buffer in place.
const writers = [_][]const u8{"mlx_slice_update"};

fn oneOf(callee: []const u8, set: []const []const u8) bool {
    for (set) |s| if (std.mem.eql(u8, callee, s)) return true;
    return false;
}

/// Whether `t` sits in a fn nested inside `outer` (reported under that fn instead).
fn innerOwner(list: []const rule.Owner, t: u32, outer: rule.Owner) bool {
    const inner = rule.ownerOf(list, t) orelse return false;
    return inner.first != outer.first;
}
