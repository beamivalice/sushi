//! Rule `folder-bytes-compare`: in `src/model_registry.zig` and `src/downloads.zig`, a `std.mem` call (`eql`,
//! `trimEnd`, `startsWith`, `endsWith`) over a model folder's path bytes (`path`, `abs_path`, `dest`). Two spellings
//! of one folder differ in bytes (trailing slash, `//`, a symlink, APFS case), so folders are compared as
//! `FolderId`s (`src/folder_identity.zig`). Token scan of the parsed file, no Sema.
const std = @import("std");
const Ast = std.zig.Ast;
const rule = @import("rule.zig");

pub const name = "folder-bytes-compare";
pub const reason = "two spellings of one folder differ in bytes: compare FolderId (folder_identity.zig), never path bytes";

const files = [_][]const u8{ "src/model_registry.zig", "src/downloads.zig" };
const calls = [_][]const u8{ "eql", "trimEnd", "startsWith", "endsWith" };
const path_names = [_][]const u8{ "path", "abs_path", "dest" };

fn listed(set: []const []const u8, s: []const u8) bool {
    for (set) |x| if (std.mem.eql(u8, x, s)) return true;
    return false;
}

pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    var in_scope = false;
    for (files) |f| in_scope = in_scope or std.mem.eql(u8, report.file, f);
    if (!in_scope) return;
    const owners = try rule.owners(arena, tree);
    var t: u32 = 4;
    while (t < tree.tokens.len) : (t += 1) {
        if (tree.tokenTag(t) != .identifier or !listed(&calls, tree.tokenSlice(t))) continue;
        if (tree.tokenTag(t - 1) != .period or !std.mem.eql(u8, tree.tokenSlice(t - 2), "mem")) continue;
        if (tree.tokenTag(t - 3) != .period or !std.mem.eql(u8, tree.tokenSlice(t - 4), "std")) continue;
        if (tree.tokenTag(t + 1) != .l_paren) continue;
        var depth: u32 = 0;
        var i = t + 1;
        while (i < tree.tokens.len) : (i += 1) {
            switch (tree.tokenTag(i)) {
                .l_paren => depth += 1,
                .r_paren => {
                    depth -= 1;
                    if (depth == 0) break;
                },
                .identifier => if (listed(&path_names, tree.tokenSlice(i))) {
                    const owner = if (rule.ownerOf(owners.items, t)) |o| o.key else "-";
                    try report.add(tree, t, owner, "std.mem.{s} over '{s}': compare FolderId, never a folder's path bytes", .{ tree.tokenSlice(t), tree.tokenSlice(i) });
                    break;
                },
                else => {},
            }
        }
    }
}
