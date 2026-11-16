//! Rule `refusing-eager`: `core.refusing(..)` bound to a `const` or `var`. It sets the engine's refusal reason when it
//! runs, so a value made ahead of a refusal leaves a reason that a later, unrelated refusal reports as its own. Call it
//! where it refuses. Token scan of the parsed file, no Sema.
const std = @import("std");
const Ast = std.zig.Ast;
const rule = @import("rule.zig");

pub const name = "refusing-eager";
pub const reason = "refusing sets the refusal reason as it runs; bound ahead, the reason outlives its refusal";

pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    const owners = try rule.owners(arena, tree);
    var t: u32 = 3;
    while (t + 1 < tree.tokens.len) : (t += 1) {
        if (tree.tokenTag(t) != .identifier or !std.mem.eql(u8, tree.tokenSlice(t), "refusing")) continue;
        if (tree.tokenTag(t + 1) != .l_paren) continue;
        const start = if (tree.tokenTag(t - 1) == .period and tree.tokenTag(t - 2) == .identifier) t - 2 else t;
        if (start < 3 or tree.tokenTag(start - 1) != .equal or tree.tokenTag(start - 2) != .identifier) continue;
        if (tree.tokenTag(start - 3) != .keyword_const and tree.tokenTag(start - 3) != .keyword_var) continue;
        const owner = if (rule.ownerOf(owners.items, t)) |o| o.key else "-";
        try report.add(tree, t, owner, "core.refusing bound to '{s}' before it refuses", .{tree.tokenSlice(start - 2)});
    }
}
