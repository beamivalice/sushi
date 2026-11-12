//! Rule `template-superlinear-std`: under `src/template/`, `std.sort` named, or a `mem.` substring search or sort
//! (`indexOf`, `lastIndexOf`, `count`, `replace`, `splitSequence`, `sort`, ...). Those std searches take time in
//! text length times needle length and the std sorts assume a consistent comparator; the engine reads both from
//! pack and request text, so it searches with `value.Finder` and sorts with `value.stableSort`. Tests are exempt.
//! Token scan of the parsed file, no Sema.
const std = @import("std");
const Ast = std.zig.Ast;
const rule = @import("rule.zig");

pub const name = "template-superlinear-std";
pub const reason = "the template engine searches with value.Finder and sorts with value.stableSort: linear, and safe on any comparator";

const folder = "src/template/";

const mem_names = [_][]const u8{
    "indexOf",         "lastIndexOf",           "indexOfPos",       "indexOfPosLinear", "lastIndexOfLinear",
    "find",            "findLast",              "findPos",          "count",            "containsAtLeast",
    "replace",         "replaceOwned",          "replacementSize",  "splitSequence",    "splitBackwardsSequence",
    "tokenizeSequence", "sort",                 "sortUnstable",     "sortContext",      "sortUnstableContext",
};

fn named(tree: *const Ast, t: u32, text: []const u8) bool {
    return tree.tokenTag(t) == .identifier and std.mem.eql(u8, tree.tokenSlice(t), text);
}

pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    if (!std.mem.startsWith(u8, report.file, folder)) return;
    const owners = try rule.owners(arena, tree);
    var t: u32 = 2;
    while (t < tree.tokens.len) : (t += 1) {
        if (tree.tokenTag(t - 1) != .period or tree.tokenTag(t) != .identifier) continue;
        const what = tree.tokenSlice(t);
        const hit = if (named(tree, t - 2, "std"))
            std.mem.eql(u8, what, "sort")
        else if (named(tree, t - 2, "mem")) for (mem_names) |n| {
            if (std.mem.eql(u8, n, what)) break true;
        } else false else false;
        if (!hit or rule.inTest(owners.items, t)) continue;
        const owner = if (rule.ownerOf(owners.items, t)) |o| o.key else "-";
        try report.add(tree, t, owner, "{s}.{s} under src/template/: search with value.Finder, sort with value.stableSort", .{ tree.tokenSlice(t - 2), what });
    }
}
