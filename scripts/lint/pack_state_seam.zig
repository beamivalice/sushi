//! Rule `pack-state-seam`: the pull's marker and record are read by `src/pack_state.zig` alone. A second reader
//! re-derives "is this folder a finished pack?" by its own rule, as five callers once did. Outside `pack_state.zig` and
//! `hf_pull.zig` (the record's writer), the pull's own writer allowlisted per fn, and tests, it flags the names
//! `pull_marker` and `manifest_name`, the literals ".pull-incomplete" and ".sushi-pull.json", and, in a file that
//! imports `hf_pull.zig` or `pack_state.zig` (a file that deals with pack folders), a ".partial" literal. Other
//! `.partial` staging (the encoder's, a capture's, the updater's) lives in files that import neither.
//! Token scan of the parsed file, no Sema.
const std = @import("std");
const Ast = std.zig.Ast;
const rule = @import("rule.zig");

pub const name = "pack-state-seam";
pub const reason = "ask pack_state.of for a folder's state: it is the one reader of the pull's marker and record";

const owner_files = [_][]const u8{ "/pack_state.zig", "/hf_pull.zig" };
const pack_imports = [_][]const u8{ "\"hf_pull.zig\"", "\"pack_state.zig\"" };
const seam_names = [_][]const u8{ "pull_marker", "manifest_name" };
const seam_literals = [_][]const u8{ ".pull-incomplete", ".sushi-pull.json" };

fn dealsWithPacks(tree: *const Ast) bool {
    for (0..tree.tokens.len) |i| {
        const t: u32 = @intCast(i);
        if (tree.tokenTag(t) != .string_literal) continue;
        for (pack_imports) |imp| if (std.mem.eql(u8, tree.tokenSlice(t), imp)) return true;
    }
    return false;
}

fn contains(text: []const u8, needles: []const []const u8) ?[]const u8 {
    for (needles) |n| if (std.mem.indexOf(u8, text, n) != null) return n;
    return null;
}

pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    for (owner_files) |f| if (std.mem.endsWith(u8, report.file, f)) return;
    const packs = dealsWithPacks(tree);
    var owners: ?std.ArrayList(rule.Owner) = null;
    for (0..tree.tokens.len) |i| {
        const t: u32 = @intCast(i);
        const text = tree.tokenSlice(t);
        if (t > 0 and tree.tokenTag(t - 1) == .keyword_test) continue;
        const found: []const u8 = switch (tree.tokenTag(t)) {
            .identifier => for (seam_names) |n| {
                if (std.mem.eql(u8, text, n)) break n;
            } else continue,
            .string_literal => contains(text, &seam_literals) orelse
                (if (packs and std.mem.indexOf(u8, text, ".partial") != null) ".partial" else continue),
            else => continue,
        };
        if (owners == null) owners = try rule.owners(arena, tree);
        if (rule.inTest(owners.?.items, t)) continue;
        const key = if (rule.ownerOf(owners.?.items, t)) |o| o.key else "-";
        try report.add(tree, t, key, "{s} read outside pack_state: ask pack_state.of for the folder's state", .{found});
    }
}
