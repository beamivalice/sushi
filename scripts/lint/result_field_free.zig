//! Rule `result-field-free`: a `.free(x.token_ids)` or `.free(x.logprobs)` call. A `GenerationResult` owns its text,
//! token ids and logprobs together and releases them with `deinit`; freeing a field by hand leaves the others behind.
//! Token scan of the parsed file, no Sema: the field names stand for the type.
const std = @import("std");
const Ast = std.zig.Ast;
const rule = @import("rule.zig");

pub const name = "result-field-free";
pub const reason = "a GenerationResult owns its fields together: release it with deinit, never one field at a time";

const fields = [_][]const u8{ "token_ids", "logprobs" };

pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    var owners: ?std.ArrayList(rule.Owner) = null;
    var t: u32 = 1;
    while (t + 3 < tree.tokens.len) : (t += 1) {
        if (tree.tokenTag(t) != .identifier or !std.mem.eql(u8, tree.tokenSlice(t), "free")) continue;
        if (tree.tokenTag(t - 1) != .period or tree.tokenTag(t + 1) != .l_paren) continue;
        // The argument is a plain `a.b.c` chain closed by the call's `)`.
        var a = t + 2;
        if (tree.tokenTag(a) != .identifier) continue;
        while (tree.tokenTag(a + 1) == .period and tree.tokenTag(a + 2) == .identifier) a += 2;
        if (tree.tokenTag(a + 1) != .r_paren or a == t + 2) continue;
        const field = tree.tokenSlice(a);
        for (fields) |f| {
            if (std.mem.eql(u8, field, f)) break;
        } else continue;
        if (owners == null) owners = try rule.owners(arena, tree);
        const key = if (rule.ownerOf(owners.?.items, t)) |o| o.key else "-";
        try report.add(tree, t, key, "'{s}' freed by hand: free the whole result with deinit", .{field});
    }
}
