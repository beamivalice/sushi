//! Rule `closure-compile-seam`: a load's MLX closures (`compileGelu`, `compileGeglu`, `compileMoeRouting`,
//! `arch_state.compile`) compiled anywhere but `model_load.compileClosures`. That seam declines on a Cooler load; a
//! loader that compiles on its own applies an MLX closure to the engine's arrays (the KLD loader's did, and crashed).
//! Tests are exempt. Token scan, no Sema.
const std = @import("std");
const Ast = std.zig.Ast;
const rule = @import("rule.zig");

pub const name = "closure-compile-seam";
pub const reason = "a load's MLX closures compile in model_load.compileClosures only, which declines on a Cooler load";

const seam = "src/model_load.zig";
const methods = [_][]const u8{ "compileGelu", "compileGeglu", "compileMoeRouting" };

pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    if (std.mem.eql(u8, report.file, seam) or rule.testFile(report.file)) return;
    const owners = try rule.owners(arena, tree);
    var t: u32 = 1;
    while (t + 1 < tree.tokens.len) : (t += 1) {
        if (tree.tokenTag(t) != .identifier or tree.tokenTag(t - 1) != .period or tree.tokenTag(t + 1) != .l_paren) continue;
        const callee = tree.tokenSlice(t);
        const hit = for (methods) |m| {
            if (std.mem.eql(u8, callee, m)) break true;
        } else std.mem.eql(u8, callee, "compile") and t >= 2 and tree.tokenTag(t - 2) == .identifier and
            std.mem.eql(u8, tree.tokenSlice(t - 2), "arch_state");
        if (!hit) continue;
        const owner = rule.ownerOf(owners.items, t) orelse continue;
        if (owner.is_test) continue;
        try report.add(tree, t, owner.key, "'{s}' compiled outside model_load.compileClosures: call that seam", .{callee});
    }
}
