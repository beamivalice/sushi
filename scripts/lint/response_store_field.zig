//! Rule `response-store-field`: the Responses store's map or LRU (`by_id`, `recency`) touched anywhere but
//! `src/responses.zig`. Every access goes through its locked methods, and a record is held by `acquire`/`release`:
//! a read or a move past the lock is a data race, and a pointer kept past it is freed by the next delete. Token scan.
const std = @import("std");
const Ast = std.zig.Ast;
const rule = @import("rule.zig");

pub const name = "response-store-field";
pub const reason = "the Responses store's map or LRU touched outside its lock races its deletes and evictions";

const fields = [_][]const u8{ "by_id", "recency" };

pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    if (std.mem.eql(u8, report.file, "src/responses.zig")) return;
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
        try report.add(tree, t, owner.key, "{s} outside src/responses.zig: go through ResponseStore's methods", .{id});
    }
}
