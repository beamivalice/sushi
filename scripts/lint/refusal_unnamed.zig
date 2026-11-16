//! Rule `refusal-unnamed`: under `src/tensor/`, an `error.TensorUnsupported` or `error.TensorDtype` returned, or handed
//! to `fail`, without `core.refusing`. The error says only that the engine refused; its reason is what names the port a load still needs,
//! and `refusing` is what counts it. Token scan of the parsed file, no Sema.
const std = @import("std");
const Ast = std.zig.Ast;
const rule = @import("rule.zig");

pub const name = "refusal-unnamed";
pub const reason = "an engine refusal without core.refusing gives no reason and is not counted";

pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    if (!std.mem.startsWith(u8, report.file, "src/tensor/")) return;
    const owners = try rule.owners(arena, tree);
    var t: u32 = 3;
    while (t < tree.tokens.len) : (t += 1) {
        if (tree.tokenTag(t) != .identifier) continue;
        const what = tree.tokenSlice(t);
        if (!std.mem.eql(u8, what, "TensorUnsupported") and !std.mem.eql(u8, what, "TensorDtype")) continue;
        if (tree.tokenTag(t - 1) != .period or tree.tokenTag(t - 2) != .keyword_error) continue;
        const before = tree.tokenTag(t - 3);
        const refused = before == .keyword_return or (before == .comma and callee(tree, t - 3, "fail"));
        if (!refused) continue;
        const owner = if (rule.ownerOf(owners.items, t)) |o| o.key else "-";
        try report.add(tree, t - 2, owner, "error.{s} refused with no reason: give one through core.refusing", .{what});
    }
}

/// Whether the call whose argument list holds token `at` is `name(..)`.
fn callee(tree: *const Ast, at: u32, comptime fn_name: []const u8) bool {
    var depth: u32 = 0;
    var t = at;
    while (t > 0) {
        t -= 1;
        switch (tree.tokenTag(t)) {
            .r_paren => depth += 1,
            .l_paren => {
                if (depth == 0) return t > 0 and tree.tokenTag(t - 1) == .identifier and std.mem.eql(u8, tree.tokenSlice(t - 1), fn_name);
                depth -= 1;
            },
            .semicolon, .l_brace, .r_brace => return false,
            else => {},
        }
    }
    return false;
}
