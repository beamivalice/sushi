//! Rule `cache-gpu-stream`: a call of `gpuStream()` in the prefix cache or the SSM cache outside `arraysStream`, the
//! one fallback. Their array work runs on the caller's stream (the engine's on Sushi Cooler, which refuses MLX's stream
//! inside its forward); MLX's own stream there fails the request or the commit. Tests are exempt. Token scan.
const std = @import("std");
const Ast = std.zig.Ast;
const rule = @import("rule.zig");

pub const name = "cache-gpu-stream";
pub const reason = "cache array work on MLX's stream is refused inside an engine forward; take the caller's stream";

const scoped = [_][]const u8{
    "prefix_cache.zig",
    "prefix_cache_index.zig",
    "prefix_cache_evict.zig",
    "prefix_cache_commit.zig",
    "ssm_cache.zig",
};

pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    const base = std.fs.path.basename(report.file);
    for (scoped) |f| {
        if (std.mem.eql(u8, base, f)) break;
    } else return;
    const owners = try rule.owners(arena, tree);
    const test_only = try rule.testOnly(arena, tree, owners.items);
    var t: u32 = 0;
    while (t + 1 < tree.tokens.len) : (t += 1) {
        if (tree.tokenTag(t) != .identifier or tree.tokenTag(t + 1) != .l_paren) continue;
        if (!std.mem.eql(u8, tree.tokenSlice(t), "gpuStream")) continue;
        if (rule.inTest(owners.items, t)) continue;
        if (inTestOnly(owners.items, test_only, t)) continue;
        const owner = if (rule.ownerOf(owners.items, t)) |o| o.key else "-";
        if (std.mem.eql(u8, owner, "arraysStream")) continue;
        try report.add(tree, t, owner, "MLX's stream in the cache: take the caller's stream, or arraysStream()", .{});
    }
}

/// Whether token `t` sits in a fn only tests reach (`rule.testOnly`).
fn inTestOnly(list: []const rule.Owner, test_only: []const bool, t: u32) bool {
    for (list, test_only) |o, m| if (m and t >= o.first and t <= o.last) return true;
    return false;
}
