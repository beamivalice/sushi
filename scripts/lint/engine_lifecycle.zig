//! Rule `engine-lifecycle`: a load's engine bookkeeping (`engine.enable`, `engine.hold`, `engine.own_weights`,
//! `engine.unsealed`, `engine.seal`) done anywhere but the shared seam. One arch kept a step another lacked, and a
//! second Cooler load then compiled every kernel as late; `arch_cooler.LoadScope` and `warmEnded` hold them for every
//! arch. The engine, the mlx seam and tests are exempt. Token scan, no Sema.
const std = @import("std");
const Ast = std.zig.Ast;
const rule = @import("rule.zig");

pub const name = "engine-lifecycle";
pub const reason = "a load's engine hold, weight wrap or seal moved outside the shared Cooler seam";

const allowed = [_][]const u8{ "src/arch/cooler.zig", "src/arch/handle.zig", "src/mlx.zig" };
const steps = [_][]const u8{ "enable", "hold", "own_weights", "unsealed", "seal" };

pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    for (allowed) |f| if (std.mem.eql(u8, report.file, f)) return;
    if (std.mem.startsWith(u8, report.file, "src/tensor/")) return;
    if (rule.testFile(report.file)) return;
    const owners = try rule.owners(arena, tree);
    var t: u32 = 2;
    while (t < tree.tokens.len) : (t += 1) {
        if (tree.tokenTag(t) != .identifier or tree.tokenTag(t - 1) != .period) continue;
        if (tree.tokenTag(t - 2) != .identifier or !std.mem.eql(u8, tree.tokenSlice(t - 2), "engine")) continue;
        const id = tree.tokenSlice(t);
        const hit = for (steps) |s| {
            if (std.mem.eql(u8, id, s)) break true;
        } else false;
        if (!hit) continue;
        // Reading whether the load wraps its regions is every wrap's to do; setting it is the seam's.
        if (std.mem.eql(u8, id, "own_weights") and (t + 1 >= tree.tokens.len or tree.tokenTag(t + 1) != .equal)) continue;
        const owner = rule.ownerOf(owners.items, t) orelse continue;
        if (rule.inTest(owners.items, t)) continue;
        try report.add(tree, t, owner.key, "engine.{s} outside the shared Cooler seam: use arch_cooler.LoadScope or warmEnded", .{id});
    }
}
