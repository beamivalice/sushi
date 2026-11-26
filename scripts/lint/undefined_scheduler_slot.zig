//! Rule `undefined-scheduler-slot`: a `Scheduler` or `Slot` local declared `= undefined`. A fixture built that way
//! reads every field it does not set from whatever an earlier test left on the stack, so a cull that walks one of
//! its lists segfaults or passes by test order. The type is read by its last field name, so `@import("...").Slot` and
//! `mod.Scheduler` are refused too; a pointer, an array or an optional of one is another value and stays allowed.
//! Parse only, no Sema. Start from `Scheduler.initForTest` / `Slot.initForTest`.
const std = @import("std");
const Ast = std.zig.Ast;
const Node = Ast.Node;
const rule = @import("rule.zig");

pub const name = "undefined-scheduler-slot";
pub const reason = "a Scheduler or Slot declared undefined reads fields the fixture never set";

/// The last field a type expression names: `Slot`, `mod.Slot`, `@import("scheduler.zig").Slot`.
fn lastField(tree: *const Ast, node: Node.Index) ?[]const u8 {
    return switch (tree.nodeTag(node)) {
        .identifier => tree.tokenSlice(tree.nodeMainToken(node)),
        .field_access => tree.tokenSlice(tree.nodeData(node).node_and_token[1]),
        else => null,
    };
}

fn isUndefined(tree: *const Ast, node: Node.Index) bool {
    return switch (tree.nodeTag(node)) {
        .identifier => std.mem.eql(u8, tree.tokenSlice(tree.nodeMainToken(node)), "undefined"),
        .grouped_expression => isUndefined(tree, tree.nodeData(node).node_and_token[0]),
        .builtin_call_two, .builtin_call_two_comma => blk: {
            const builtin = tree.nodeMainToken(node);
            if (tree.tokenTag(builtin) != .builtin or !std.mem.eql(u8, tree.tokenSlice(builtin), "@as")) break :blk false;
            var buf: [2]Node.Index = undefined;
            const params = tree.builtinCallParams(&buf, node) orelse break :blk false;
            break :blk params.len == 2 and isUndefined(tree, params[1]);
        },
        else => false,
    };
}

pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    const owners = try rule.owners(arena, tree);
    for (0..tree.nodes.len) |i| {
        const node: Node.Index = @fromBackingInt(@intCast(i));
        const decl = tree.fullVarDecl(node) orelse continue;
        if (tree.tokenTag(decl.ast.mut_token) != .keyword_var) continue;
        const type_node = decl.ast.type_node.unwrap() orelse continue;
        const type_name = lastField(tree, type_node) orelse continue;
        const ctor = if (std.mem.eql(u8, type_name, "Scheduler")) "Scheduler.initForTest" else if (std.mem.eql(u8, type_name, "Slot")) "Slot.initForTest" else continue;
        const init_node = decl.ast.init_node.unwrap() orelse continue;
        if (!isUndefined(tree, init_node)) continue;
        const owner = rule.ownerOf(owners.items, decl.ast.mut_token) orelse continue;
        const var_name = tree.tokenSlice(decl.ast.mut_token + 1);
        try report.add(tree, decl.ast.mut_token + 1, owner.key, "{s} '{s}' is undefined: start it from {s}", .{ type_name, var_name, ctor });
    }
}
