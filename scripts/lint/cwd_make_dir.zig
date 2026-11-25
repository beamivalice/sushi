//! Rule `cwd-make-dir`: `createDirPath` on the working directory (`std.Io.Dir.cwd()`, or a local named `cwd`) outside
//! `src/io_util.zig`. Such a path is the user's (a cache, a settings or log folder), and std refuses one whose last
//! component is a symlink to a folder, as `~/.sushi` linked to another volume is: `io_util.makeDirPath` takes it.
//! Token scan of the parsed file, no Sema.
const std = @import("std");
const Ast = std.zig.Ast;
const rule = @import("rule.zig");

pub const name = "cwd-make-dir";
pub const reason = "a user's folder may link to another volume: make it with io_util.makeDirPath";

const owner_file = "/io_util.zig";

/// `t` is `createDirPath` called on `cwd()` or on an identifier `cwd`.
fn onCwd(tree: *const Ast, t: u32) bool {
    if (t < 2 or tree.tokenTag(t - 1) != .period) return false;
    const r = t - 2;
    if (tree.tokenTag(r) == .identifier) return std.mem.eql(u8, tree.tokenSlice(r), "cwd");
    return r >= 2 and tree.tokenTag(r) == .r_paren and tree.tokenTag(r - 1) == .l_paren and
        tree.tokenTag(r - 2) == .identifier and std.mem.eql(u8, tree.tokenSlice(r - 2), "cwd");
}

pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    if (std.mem.endsWith(u8, report.file, owner_file)) return;
    var owners: ?std.ArrayList(rule.Owner) = null;
    for (0..tree.tokens.len) |i| {
        const t: u32 = @intCast(i);
        if (tree.tokenTag(t) != .identifier or !std.mem.eql(u8, tree.tokenSlice(t), "createDirPath") or !onCwd(tree, t)) continue;
        if (owners == null) owners = try rule.owners(arena, tree);
        const key = if (rule.ownerOf(owners.?.items, t)) |o| o.key else "-";
        try report.add(tree, t, key, "createDirPath on the working directory refuses a folder that is a symlink: use io_util.makeDirPath", .{});
    }
}
