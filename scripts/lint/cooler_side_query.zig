//! Rule `cooler-side-query`: a load's Sushi Cooler inputs (`cooler_ask`, `cooler_for_test`) read or written anywhere
//! but `src/cooler_side.zig`. Its side is one decision from them and the arch's gap: two archs each deciding from
//! their own copies is how the decision came to exist twice. Tests name a side directly and are exempt. Token scan,
//! no Sema.
const std = @import("std");
const Ast = std.zig.Ast;
const rule = @import("rule.zig");

pub const name = "cooler-side-query";
pub const reason = "a Sushi Cooler side input read outside src/cooler_side.zig decides the side a second time";

const fields = [_][]const u8{ "cooler_ask", "cooler_for_test" };

pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    if (std.mem.eql(u8, report.file, "src/cooler_side.zig")) return;
    if (rule.testFile(report.file)) return;
    const owners = try rule.owners(arena, tree);
    var t: u32 = 1;
    while (t < tree.tokens.len) : (t += 1) {
        if (tree.tokenTag(t) != .identifier or tree.tokenTag(t - 1) != .period) continue;
        const id = tree.tokenSlice(t);
        const hit = for (fields) |f| {
            if (std.mem.eql(u8, id, f)) break true;
        } else false;
        if (!hit) continue;
        const owner = rule.ownerOf(owners.items, t) orelse continue;
        if (rule.inTest(owners.items, t)) continue;
        try report.add(tree, t, owner.key, "{s} outside src/cooler_side.zig: ask cooler_side.sideOf, or stamp it there", .{id});
    }
}
