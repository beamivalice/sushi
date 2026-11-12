//! Rule `scope-state`: a write to the engine's scope state (`engine.active`, `engine.enabled`) outside `src/tensor/`.
//! The state is one owner's: a span that moves it takes a guard there (`engine.enter`, `makers`, `turnTo`, `hold`,
//! `enable`) that gives back what it found on every path. A write by hand is the leak the guards exist to end. A
//! test, or a helper in a test file, may turn `enabled` on for its span; `active` is never written by hand.
const std = @import("std");
const Ast = std.zig.Ast;
const rule = @import("rule.zig");

pub const name = "scope-state";
pub const reason = "the engine's scope state is written by its owner's guards alone: a write by hand leaks on the paths it forgot";

const owner_dir = "/tensor/";

pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    if (std.mem.indexOf(u8, report.file, owner_dir) != null) return;
    const test_file = std.mem.endsWith(u8, report.file, "_test.zig") or std.mem.endsWith(u8, report.file, "/test_support.zig");
    var owners: ?std.ArrayList(rule.Owner) = null;
    const n = tree.tokens.len;
    for (0..n) |i| {
        const t: u32 = @intCast(i);
        if (t + 3 >= n or tree.tokenTag(t) != .identifier or tree.tokenTag(t + 1) != .period or tree.tokenTag(t + 2) != .identifier or tree.tokenTag(t + 3) != .equal) continue;
        if (!std.mem.eql(u8, tree.tokenSlice(t), "engine")) continue;
        const field = tree.tokenSlice(t + 2);
        const is_active = std.mem.eql(u8, field, "active");
        if (!is_active and !std.mem.eql(u8, field, "enabled")) continue;
        if (owners == null) owners = try rule.owners(arena, tree);
        if (!is_active and (test_file or rule.inTest(owners.?.items, t))) continue;
        const key = if (rule.ownerOf(owners.?.items, t)) |o| o.key else "-";
        try report.add(tree, t, key, "'engine.{s}' is written by hand: take the engine's guard for the span instead", .{field});
    }
}
