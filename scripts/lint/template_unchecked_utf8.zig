//! Rule `template-unchecked-utf8`: under `src/template/`, `Utf8View.initUnchecked`. Its iterator reads past the end of
//! text that is not UTF-8; the engine walks code points with `value.codepoints`, which never does. Tests are exempt.
//! Token scan of the parsed file, no Sema.
const std = @import("std");
const Ast = std.zig.Ast;
const rule = @import("rule.zig");

pub const name = "template-unchecked-utf8";
pub const reason = "the template engine walks code points with value.codepoints, which never reads past the end";

const folder = "src/template/";

pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    if (!std.mem.startsWith(u8, report.file, folder)) return;
    const owners = try rule.owners(arena, tree);
    var t: u32 = 2;
    while (t < tree.tokens.len) : (t += 1) {
        if (tree.tokenTag(t) != .identifier or !std.mem.eql(u8, tree.tokenSlice(t), "initUnchecked")) continue;
        if (tree.tokenTag(t - 1) != .period or tree.tokenTag(t - 2) != .identifier) continue;
        if (!std.mem.eql(u8, tree.tokenSlice(t - 2), "Utf8View") or rule.inTest(owners.items, t)) continue;
        const owner = if (rule.ownerOf(owners.items, t)) |o| o.key else "-";
        try report.add(tree, t, owner, "Utf8View.initUnchecked under src/template/: walk code points with value.codepoints", .{});
    }
}
