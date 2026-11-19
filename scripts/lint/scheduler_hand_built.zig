//! Rule `scheduler-hand-built`: a `Scheduler` declared `undefined` and set field by field, then given a `submit`.
//! `submit` reads queue fields such a fixture can leave unset (`draining`), so the call reads whatever an earlier
//! test left on the stack and can refuse with `error.Shutdown`. Start from `submitTestScheduler`. Token scan, no Sema.
const std = @import("std");
const Ast = std.zig.Ast;
const rule = @import("rule.zig");

pub const name = "scheduler-hand-built";
pub const reason = "submit on a field-by-field Scheduler reads the fields it left undefined";

pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    const owners = try rule.owners(arena, tree);
    var t: u32 = 0;
    while (t + 5 < tree.tokens.len) : (t += 1) {
        if (tree.tokenTag(t) != .keyword_var or tree.tokenTag(t + 1) != .identifier or tree.tokenTag(t + 2) != .colon) continue;
        var ty = t + 3;
        while (ty + 2 < tree.tokens.len and tree.tokenTag(ty) == .identifier and tree.tokenTag(ty + 1) == .period) ty += 2;
        if (tree.tokenTag(ty) != .identifier or !std.mem.eql(u8, tree.tokenSlice(ty), "Scheduler")) continue;
        if (tree.tokenTag(ty + 1) != .equal or tree.tokenTag(ty + 2) != .identifier) continue;
        if (!std.mem.eql(u8, tree.tokenSlice(ty + 2), "undefined")) continue;
        const owner = rule.ownerOf(owners.items, t) orelse continue;
        const var_name = tree.tokenSlice(t + 1);
        var u = ty + 3;
        while (u + 3 <= owner.last) : (u += 1) {
            if (tree.tokenTag(u) != .identifier or !std.mem.eql(u8, tree.tokenSlice(u), var_name)) continue;
            if (tree.tokenTag(u + 1) != .period or tree.tokenTag(u + 2) != .identifier or tree.tokenTag(u + 3) != .l_paren) continue;
            if (!std.mem.eql(u8, tree.tokenSlice(u + 2), "submit")) continue;
            try report.add(tree, t + 1, owner.key, "Scheduler '{s}' built field by field reaches submit: start from submitTestScheduler", .{var_name});
            break;
        }
    }
}
