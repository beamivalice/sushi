//! Rule `ceiling-raw-read`: a raw read of the GPU limit, the margin or the working set outside `bills.Ceiling`
//! (`ceilingNow`). Two raw reads in one decision can take two values (a margin saved between them, a test's working-set
//! stand-in), and a figure read raw misses the RAM cap and the GPU floor the Ceiling folds in. A margin flag write and a test
//! are not reads; a device read that is no ceiling is an allowlist entry with its reason. Token scan, no Sema.
const std = @import("std");
const Ast = std.zig.Ast;
const rule = @import("rule.zig");

pub const name = "ceiling-raw-read";
pub const reason = "two raw reads of the GPU limit in one decision can take two values; read one bills.ceilingNow()";

const raw = [_][]const u8{ "wiredLimitBytes", "declaredWiredLimit", "staticGpuMemoryCeiling", "getGpuWorkingSetLimit", "maxRecommendedWorkingSet", "wiredMarginBytes" };
/// The Ceiling's own readers, and the files that hold the device reads it is made of.
const owner_fns = [_][]const u8{ "ceilingNow", "wiredLimitBytes", "declaredWiredLimit", "staticGpuMemoryCeiling" };
const owner_files = [_][]const u8{ "src/gpu_mem.zig", "src/mlx.zig" };

pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    if (rule.testFile(report.file)) return;
    for (owner_files) |f| if (std.mem.eql(u8, report.file, f)) return;
    var owners: ?std.ArrayList(rule.Owner) = null;
    const n = tree.tokens.len;
    var t: u32 = 1;
    while (t + 1 < n) : (t += 1) {
        if (tree.tokenTag(t) != .identifier) continue;
        const word = tree.tokenSlice(t);
        const is_raw = for (raw) |r| {
            if (std.mem.eql(u8, word, r)) break true;
        } else false;
        if (!is_raw) continue;
        // A declaration, or a write of the margin.
        switch (tree.tokenTag(t - 1)) {
            .keyword_fn, .keyword_var, .keyword_const => continue,
            else => {},
        }
        if (tree.tokenTag(t + 1) == .equal) continue;
        if (owners == null) owners = try rule.owners(arena, tree);
        if (rule.inTest(owners.?.items, t)) continue;
        const owner = rule.ownerOf(owners.?.items, t) orelse continue;
        if (std.mem.eql(u8, report.file, "src/bills.zig")) {
            const own = for (owner_fns) |f| {
                if (std.mem.eql(u8, owner.key, f)) break true;
            } else false;
            if (own) continue;
        }
        try report.add(tree, t, owner.key, "{s} read raw: read the GPU limit, margin and working set through one bills.ceilingNow()", .{word});
    }
}
