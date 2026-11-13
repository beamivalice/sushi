//! Rule `engine-hold-end`: `engine.unhold` called anywhere but `engineHoldEnd` (`src/arch/handle.zig`) or the engine's
//! own `src/tensor/`. The last hold's end also gives back the EXL3 module's engine-side arrays; an exit that only
//! unholds keeps them, as a failed Cooler load did.
const std = @import("std");
const Ast = std.zig.Ast;
const rule = @import("rule.zig");

pub const name = "engine-hold-end";
pub const reason = "a hold ends in arch_handle.engineHoldEnd alone: another exit keeps the module arrays the last hold's end gives back";

const owner_dir = "/tensor/";
const owner_file = "/arch/handle.zig";
const owner_fn = "engineHoldEnd";

pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    if (std.mem.indexOf(u8, report.file, owner_dir) != null) return;
    const in_owner_file = std.mem.endsWith(u8, report.file, owner_file);
    var owners: ?std.ArrayList(rule.Owner) = null;
    const n = tree.tokens.len;
    for (0..n) |i| {
        const t: u32 = @intCast(i);
        if (t + 3 >= n or tree.tokenTag(t) != .identifier or tree.tokenTag(t + 1) != .period or tree.tokenTag(t + 2) != .identifier or tree.tokenTag(t + 3) != .l_paren) continue;
        if (!std.mem.eql(u8, tree.tokenSlice(t), "engine") or !std.mem.eql(u8, tree.tokenSlice(t + 2), "unhold")) continue;
        if (owners == null) owners = try rule.owners(arena, tree);
        const key = if (rule.ownerOf(owners.?.items, t)) |o| o.key else "-";
        if (in_owner_file and std.mem.eql(u8, key, owner_fn)) continue;
        try report.add(tree, t, key, "'engine.unhold' ends a hold by hand: call arch_handle.engineHoldEnd", .{});
    }
}
