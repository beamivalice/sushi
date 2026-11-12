//! Rule `template-text-search`: outside `src/template/` and tests, a `mem.` search (`indexOf`, `count`, `startsWith`,
//! ...) whose arguments name a template's source (`chat_template`, `template`, `tmpl`, `tpl`, `template_source`). A
//! word found in the text may sit in a comment or a comparison; what a template reads, prints or compares comes from
//! its compiled facts (`chat.asks`). Token scan of the parsed file, no Sema.
const std = @import("std");
const Ast = std.zig.Ast;
const rule = @import("rule.zig");

pub const name = "template-text-search";
pub const reason = "a template is asked through its compiled facts, never by searching its text";

const engine_folder = "src/template/";

const searches = [_][]const u8{
    "indexOf",       "indexOfPos",      "lastIndexOf", "indexOfPosLinear", "lastIndexOfLinear", "count",
    "containsAtLeast", "startsWith",    "endsWith",    "replace",          "replaceOwned",      "replacementSize",
    "splitSequence", "splitBackwardsSequence", "tokenizeSequence", "find", "findLast", "findPos",
};

const template_names = [_][]const u8{ "chat_template", "template", "tmpl", "tpl", "template_source" };

fn isIdent(tree: *const Ast, t: u32, list: []const []const u8) bool {
    if (tree.tokenTag(t) != .identifier) return false;
    for (list) |n| if (std.mem.eql(u8, n, tree.tokenSlice(t))) return true;
    return false;
}

pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    if (std.mem.startsWith(u8, report.file, engine_folder)) return;
    const owners = try rule.owners(arena, tree);
    const n: u32 = @intCast(tree.tokens.len);
    var t: u32 = 2;
    while (t + 1 < n) : (t += 1) {
        if (tree.tokenTag(t - 1) != .period or !isIdent(tree, t, &searches)) continue;
        if (tree.tokenTag(t - 2) != .identifier or !std.mem.eql(u8, tree.tokenSlice(t - 2), "mem")) continue;
        if (tree.tokenTag(t + 1) != .l_paren or rule.inTest(owners.items, t)) continue;
        var depth: u32 = 0;
        var a = t + 1;
        const names_template = while (a < n) : (a += 1) {
            switch (tree.tokenTag(a)) {
                .l_paren => depth += 1,
                .r_paren => {
                    depth -= 1;
                    if (depth == 0) break false;
                },
                else => if (isIdent(tree, a, &template_names)) break true,
            }
        } else false;
        if (!names_template) continue;
        const owner = if (rule.ownerOf(owners.items, t)) |o| o.key else "-";
        try report.add(tree, t, owner, "mem.{s} over the template text: ask its compiled facts (chat.asks)", .{tree.tokenSlice(t)});
    }
}
