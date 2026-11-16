//! Rule `disk-read-stream`: a disk-tier file (`*disk*.zig`) that makes its own CPU stream outside `reading`. A tier
//! read takes `DiskTier.reading()`, whose stream is the load's side (`arrays_on`); a Qwen Cooler restore that read on
//! MLX's CPU stream inside the request walked a map of the other side and crashed. Tests are exempt. Token scan.
const std = @import("std");
const Ast = std.zig.Ast;
const rule = @import("rule.zig");

pub const name = "disk-read-stream";
pub const reason = "a disk-tier read on its own CPU stream misses the load's side; take DiskTier.reading()";

pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    if (std.mem.indexOf(u8, std.fs.path.basename(report.file), "disk") == null or rule.testFile(report.file)) return;
    const owners = try rule.owners(arena, tree);
    var t: u32 = 1;
    while (t + 1 < tree.tokens.len) : (t += 1) {
        if (tree.tokenTag(t) != .identifier or tree.tokenTag(t - 1) != .period or tree.tokenTag(t + 1) != .l_paren) continue;
        if (!std.mem.eql(u8, tree.tokenSlice(t), "mlx_default_cpu_stream_new")) continue;
        const owner = rule.ownerOf(owners.items, t) orelse continue;
        if (owner.is_test or std.mem.eql(u8, owner.key, "reading") or std.mem.endsWith(u8, owner.key, ".reading")) continue;
        try report.add(tree, t, owner.key, "a CPU stream of its own in the disk tier: read through DiskTier.reading()", .{});
    }
}
