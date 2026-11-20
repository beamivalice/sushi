//! Rule `promise-active`: a fn that reads the GPU's active bytes (`.activeBytes()`) and never asks
//! `bills.promisedActive`. A promise (admission, a headroom, a load's fit, a pinned context or chunk) counts the decoder
//! pools a lease lent out of the active bytes in use; a reading that took them raw promised them free (the lessee's
//! own reading too: it frees only its own model's). A report or the lent term's own measure is an allowlist entry with
//! its reason. Token scan, no Sema.
const std = @import("std");
const Ast = std.zig.Ast;
const rule = @import("rule.zig");

pub const name = "promise-active";
pub const reason = "a promise read off the raw active bytes counts the lent decoder pools free";

pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    if (std.mem.eql(u8, report.file, "src/gpu_mem.zig") or rule.testFile(report.file)) return;
    var owners: ?std.ArrayList(rule.Owner) = null;
    const n = tree.tokens.len;
    var t: u32 = 1;
    while (t + 1 < n) : (t += 1) {
        if (tree.tokenTag(t) != .identifier or tree.tokenTag(t - 1) != .period or tree.tokenTag(t + 1) != .l_paren) continue;
        if (!std.mem.eql(u8, tree.tokenSlice(t), "activeBytes")) continue;
        if (owners == null) owners = try rule.owners(arena, tree);
        if (rule.inTest(owners.?.items, t)) continue;
        const owner = rule.ownerOf(owners.?.items, t) orelse continue;
        const asks = for (owner.first..owner.last + 1) |i| {
            const u: u32 = @intCast(i);
            if (tree.tokenTag(u) == .identifier and std.mem.eql(u8, tree.tokenSlice(u), "promisedActive")) break true;
        } else false;
        if (asks) continue;
        try report.add(tree, t, owner.key, "the active bytes read raw: a promise takes them through bills.promisedActive", .{});
    }
}
