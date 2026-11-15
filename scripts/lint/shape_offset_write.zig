//! Rule `shape-offset-write`: a local `[max_dims]c_int` (or `[core.max_dims]c_int`) written by `@memcpy` at an offset
//! other than a literal 0. Each such write was guarded by hand, and one guard missing let split-K write index 8 of an
//! 8-entry shape. Build the shape with `core.Dims`, whose pieces past `max_dims` are `TensorShape`. Token scan per fn.
const std = @import("std");
const Ast = std.zig.Ast;
const rule = @import("rule.zig");

pub const name = "shape-offset-write";
pub const reason = "a shape buffer written at an offset is one hand guard from a write past it; core.Dims checks every piece";

pub fn check(gpa: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const fns = try rule.owners(arena.allocator(), tree);
    for (fns.items) |owner| {
        var names: std.ArrayList([]const u8) = .empty;
        var t = owner.first;
        while (t + 5 < owner.last) : (t += 1) {
            if (tree.tokenTag(t) != .identifier or tree.tokenTag(t + 1) != .colon or tree.tokenTag(t + 2) != .l_bracket) continue;
            var at = t + 3;
            if (tree.tokenTag(at) == .identifier and std.mem.eql(u8, tree.tokenSlice(at), "core") and tree.tokenTag(at + 1) == .period) at += 2;
            if (tree.tokenTag(at) != .identifier or !std.mem.eql(u8, tree.tokenSlice(at), "max_dims")) continue;
            if (tree.tokenTag(at + 1) != .r_bracket or !std.mem.eql(u8, tree.tokenSlice(at + 2), "c_int")) continue;
            try names.append(arena.allocator(), tree.tokenSlice(t));
        }
        if (names.items.len == 0) continue;
        t = owner.first;
        while (t + 4 < owner.last) : (t += 1) {
            if (tree.tokenTag(t) != .builtin or !std.mem.eql(u8, tree.tokenSlice(t), "@memcpy")) continue;
            if (tree.tokenTag(t + 1) != .l_paren or tree.tokenTag(t + 2) != .identifier or tree.tokenTag(t + 3) != .l_bracket) continue;
            const buf = tree.tokenSlice(t + 2);
            var declared = false;
            for (names.items) |n| declared = declared or std.mem.eql(u8, n, buf);
            if (!declared) continue;
            const zero = tree.tokenTag(t + 4) == .number_literal and std.mem.eql(u8, tree.tokenSlice(t + 4), "0") and tree.tokenTag(t + 5) == .ellipsis2;
            if (zero) continue;
            try report.add(tree, t, owner.key, "@memcpy into shape buffer '{s}' at an offset: build it with core.Dims", .{buf});
        }
    }
}
