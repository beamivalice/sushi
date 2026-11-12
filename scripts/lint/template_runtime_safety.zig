//! Rule `template-runtime-safety`: a fn under `src/template/` whose body does not open with
//! `@setRuntimeSafety(true)`, or any `@setRuntimeSafety` there that is not `(true)`. The template engine reads text from downloaded packs; in a ReleaseFast build only that
//! statement keeps its bounds and overflow checks, so a bad template fails loudly instead of reading wild memory.
//! Parse only, no Sema.
const std = @import("std");
const Ast = std.zig.Ast;
const Node = Ast.Node;
const rule = @import("rule.zig");

pub const name = "template-runtime-safety";
pub const reason = "the template engine keeps runtime safety on in ReleaseFast: it reads untrusted template text";

const folder = "src/template/";

fn isSafetyCall(tree: *const Ast, node: Node.Index) bool {
    var buf: [2]Node.Index = undefined;
    _ = tree.builtinCallParams(&buf, node) orelse return false;
    return std.mem.eql(u8, tree.tokenSlice(tree.nodeMainToken(node)), "@setRuntimeSafety");
}

/// Whether `stmt` is `@setRuntimeSafety(true)`.
fn isSafetyOn(tree: *const Ast, stmt: Node.Index) bool {
    if (!isSafetyCall(tree, stmt)) return false;
    var buf: [2]Node.Index = undefined;
    const params = tree.builtinCallParams(&buf, stmt).?;
    return params.len == 1 and tree.nodeTag(params[0]) == .identifier and std.mem.eql(u8, tree.tokenSlice(tree.nodeMainToken(params[0])), "true");
}

pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    if (!std.mem.startsWith(u8, report.file, folder)) return;
    const owners = try rule.owners(arena, tree);
    var i: u32 = 0;
    while (i < tree.nodes.len) : (i += 1) {
        const node: Node.Index = @enumFromInt(i);
        if (isSafetyCall(tree, node) and !isSafetyOn(tree, node)) {
            const at = tree.nodeMainToken(node);
            const owner = if (rule.ownerOf(owners.items, at)) |o| o.key else "-";
            try report.add(tree, at, owner, "@setRuntimeSafety(false) under src/template/: the engine keeps its checks on", .{});
            continue;
        }
        if (tree.nodeTag(node) != .fn_decl) continue;
        var proto_buf: [1]Node.Index = undefined;
        const proto = tree.fullFnProto(&proto_buf, node) orelse continue;
        const fn_name = if (proto.name_token) |t| tree.tokenSlice(t) else "(anonymous)";
        const body = tree.nodeData(node).node_and_node[1];
        var buf: [2]Node.Index = undefined;
        const stmts = tree.blockStatements(&buf, body) orelse &.{};
        if (stmts.len > 0 and isSafetyOn(tree, stmts[0])) continue;
        try report.add(tree, proto.name_token orelse tree.firstToken(node), fn_name, "fn '{s}' does not start with @setRuntimeSafety(true)", .{fn_name});
    }
}
