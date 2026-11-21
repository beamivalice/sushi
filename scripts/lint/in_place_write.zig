//! Rule `in-place-write`: `mlx_slice_update`, the call that writes rows into a buffer in place, outside the three files
//! that own it: `mlx_pages.zig` (`InPlace.write` is the one in-place write, which refuses a buffer off MLX's page grid:
//! there MLX's pool can hand it a buffer it never donates, and every write copies all of it), `src/mlx.zig` (the
//! binding) and `src/tensor/` (the engine of our own, which writes in place at any size). Also an `InPlace` built
//! outside `mlx_pages.zig` (`InPlace{`, a `.capacity_bytes =` field): only its constructors record a buffer's bytes, so
//! a holder of the type holds a buffer whose grid was checked where it was made. Tests are exempt.
const std = @import("std");
const Ast = std.zig.Ast;
const rule = @import("rule.zig");

pub const name = "in-place-write";
pub const reason = "a buffer off MLX's page grid copies on every write into it; write it with mlx_pages.InPlace.write, which refuses one";

const callee = "mlx_slice_update";

fn owned(file: []const u8) bool {
    return std.mem.eql(u8, std.fs.path.basename(file), "mlx_pages.zig") or std.mem.eql(u8, file, "src/mlx.zig") or
        std.mem.startsWith(u8, file, "src/tensor/");
}

pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    if (owned(report.file) or rule.testFile(report.file)) return;
    const owners = try rule.owners(arena, tree);
    const test_only = try rule.testOnly(arena, tree, owners.items);
    for (0..tree.tokens.len) |i| {
        const t: u32 = @intCast(i);
        if (tree.tokenTag(t) != .identifier) continue;
        const word = tree.tokenSlice(t);
        const message = if (std.mem.eql(u8, word, callee))
            "an in-place write outside mlx_pages: write it with mlx_pages.InPlace.write"
        else if (builtBy(tree, t, word))
            "an InPlace built outside mlx_pages: make it with mlx_pages.inPlaceBuffer, ringBuffer, grownBuffer or InPlace.adopt"
        else
            continue;
        if (rule.inTest(owners.items, t)) continue;
        const owner = rule.ownerOf(owners.items, t);
        if (owner) |o| if (onlyTests(owners.items, test_only, o)) continue;
        const key = if (owner) |o| o.key else "-";
        try report.add(tree, t, key, "{s}", .{message});
    }
}

/// An `InPlace{ ... }` literal, or a `.capacity_bytes = ...` field of an anonymous one.
fn builtBy(tree: *const Ast, t: u32, word: []const u8) bool {
    if (std.mem.eql(u8, word, "InPlace")) {
        // `InPlace{ .field = ..` is a literal; a return type before a body and `[_]InPlace{ x, y }` are not.
        return t + 4 < tree.tokens.len and tree.tokenTag(t + 1) == .l_brace and tree.tokenTag(t + 2) == .period and
            tree.tokenTag(t + 3) == .identifier and tree.tokenTag(t + 4) == .equal;
    }
    return std.mem.eql(u8, word, "capacity_bytes") and t > 0 and tree.tokenTag(t - 1) == .period and t + 1 < tree.tokens.len and tree.tokenTag(t + 1) == .equal;
}

/// Whether `o` is a fn whose every use is in a test (`rule.testOnly`).
fn onlyTests(list: []const rule.Owner, marked: []const bool, o: rule.Owner) bool {
    for (list, marked) |c, only| if (c.first == o.first and c.last == o.last) return only;
    return false;
}
