//! Rule `cache-room`: a call of `roomFor` outside `mlx_pages.zig` and `src/tensor/`. The engine's room for a cache
//! buffer is taken by the `mlx_pages` constructor that makes it, once: a growth path that also took it raised the
//! engine's pool by the buffer twice, past the request's bill. Tests are exempt. Token scan.
const std = @import("std");
const Ast = std.zig.Ast;
const rule = @import("rule.zig");

pub const name = "cache-room";
pub const reason = "a buffer's engine room is taken once, by the mlx_pages constructor that makes it";

pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    if (std.mem.eql(u8, std.fs.path.basename(report.file), "mlx_pages.zig") or rule.testFile(report.file)) return;
    if (std.mem.indexOf(u8, report.file, "src/tensor/") != null) return;
    const owners = try rule.owners(arena, tree);
    var t: u32 = 0;
    while (t + 1 < tree.tokens.len) : (t += 1) {
        if (tree.tokenTag(t) != .identifier or tree.tokenTag(t + 1) != .l_paren) continue;
        if (!std.mem.eql(u8, tree.tokenSlice(t), "roomFor")) continue;
        const owner = rule.ownerOf(owners.items, t) orelse continue;
        if (owner.is_test) continue;
        try report.add(tree, t, owner.key, "the engine's room taken outside mlx_pages: make the buffer with its constructor", .{});
    }
}
