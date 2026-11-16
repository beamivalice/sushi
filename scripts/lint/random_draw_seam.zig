//! Rule `random-draw-seam`: `mlx_random_categorical` called anywhere but `src/sampler.zig`. A draw there picks its
//! key by the stream's side (the engine's per-position key, or MLX's `seedKey`); a draw elsewhere keyed from MLX
//! failed every sampled MTP round on a Cooler load. The route and the mlx-c seam declare it, and tests are exempt.
//! Token scan, no Sema.
const std = @import("std");
const Ast = std.zig.Ast;
const rule = @import("rule.zig");

pub const name = "random-draw-seam";
pub const reason = "a categorical draw outside src/sampler.zig takes MLX's key on the engine's stream";

const allowed = [_][]const u8{ "src/sampler.zig", "src/tensor/route.zig", "src/mlx.zig" };

pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    for (allowed) |f| if (std.mem.eql(u8, report.file, f)) return;
    if (rule.testFile(report.file)) return;
    const owners = try rule.owners(arena, tree);
    var t: u32 = 1;
    while (t + 1 < tree.tokens.len) : (t += 1) {
        if (tree.tokenTag(t) != .identifier or tree.tokenTag(t - 1) != .period or tree.tokenTag(t + 1) != .l_paren) continue;
        if (!std.mem.eql(u8, tree.tokenSlice(t), "mlx_random_categorical")) continue;
        const owner = rule.ownerOf(owners.items, t) orelse continue;
        if (owner.is_test) continue;
        try report.add(tree, t, owner.key, "a categorical draw outside src/sampler.zig: call sampler.drawRowsLogProbs", .{});
    }
}
