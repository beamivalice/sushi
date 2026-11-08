//! Rule `const-write`: an `@constCast` whose operand is rooted at a pointer-to-const parameter or a `const` local of the
//! enclosing fn. ReleaseFast marks such a parameter readonly, so a store through the cast can be dropped, and no
//! behavioural test sees it. Parse only, no Sema.
const std = @import("std");
const Ast = std.zig.Ast;
const Node = Ast.Node;
const rule = @import("rule.zig");

pub const name = "const-write";
pub const reason = "a store through @constCast of a const parameter or local can be dropped by the optimizer";

const Fn = struct {
    name: []const u8,
    first: u32,
    last: u32,
    const_params: []const []const u8,
};

const Local = struct { name: []const u8, token: u32, fn_index: usize };

/// `*const T`, `[]const T`, `[*]const T`, `?*const T` and their sentinel/aligned forms.
fn isConstPointer(tree: *const Ast, type_node: Node.Index) bool {
    var node = type_node;
    while (tree.nodeTag(node) == .optional_type) node = tree.nodeData(node).node;
    const ptr = tree.fullPtrType(node) orelse return false;
    return ptr.const_token != null;
}

fn isMutablePointer(tree: *const Ast, type_node: Node.Index) bool {
    var node = type_node;
    while (tree.nodeTag(node) == .optional_type) node = tree.nodeData(node).node;
    const ptr = tree.fullPtrType(node) orelse return false;
    return ptr.const_token == null;
}

const Root = struct { name: []const u8, addr_taken: bool };

/// The identifier an operand is rooted at, looking through the projections and casts that keep the address derived
/// from it. A call result is a pointer the callee chose, so it roots nowhere.
fn rootIdentifier(tree: *const Ast, start: Node.Index) ?Root {
    var node = start;
    var addr_taken = false;
    while (true) switch (tree.nodeTag(node)) {
        .identifier => return .{ .name = tree.tokenSlice(tree.nodeMainToken(node)), .addr_taken = addr_taken },
        .address_of => {
            addr_taken = true;
            node = tree.nodeData(node).node;
        },
        .deref => node = tree.nodeData(node).node,
        .field_access, .unwrap_optional, .grouped_expression => node = tree.nodeData(node).node_and_token[0],
        .array_access, .slice_open => node = tree.nodeData(node).node_and_node[0],
        .slice, .slice_sentinel => node = tree.nodeData(node).node_and_extra[0],
        .builtin_call_two, .builtin_call_two_comma, .builtin_call, .builtin_call_comma => {
            const builtin = tree.tokenSlice(tree.nodeMainToken(node));
            const through = [_][]const u8{ "@ptrCast", "@alignCast", "@as" };
            for (through) |t| {
                if (!std.mem.eql(u8, builtin, t)) continue;
                var buf: [2]Node.Index = undefined;
                const params = tree.builtinCallParams(&buf, node) orelse return null;
                if (params.len == 0) return null;
                node = params[params.len - 1];
                break;
            } else return null;
        },
        else => return null,
    };
}

/// The token a `const` local goes out of scope at: the brace that closes its enclosing block.
fn scopeEnd(tree: *const Ast, decl_token: u32) u32 {
    var depth: i32 = 0;
    var t = decl_token;
    while (t < tree.tokens.len) : (t += 1) switch (tree.tokenTag(t)) {
        .l_brace => depth += 1,
        .r_brace => {
            depth -= 1;
            if (depth < 0) return t;
        },
        else => {},
    };
    return t;
}

fn innermostFn(fns: []const Fn, token: u32) ?usize {
    var best: ?usize = null;
    for (fns, 0..) |f, i| {
        if (token < f.first or token > f.last) continue;
        if (best == null or f.last - f.first < fns[best.?].last - fns[best.?].first) best = i;
    }
    return best;
}

/// `arena` outlives the call: the driver frees it once per file.
pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    var fns: std.ArrayList(Fn) = .empty;
    var locals: std.ArrayList(Local) = .empty;
    var casts: std.ArrayList(Node.Index) = .empty;

    for (0..tree.nodes.len) |i| {
        const node: Node.Index = @fromBackingInt(@intCast(i));
        switch (tree.nodeTag(node)) {
            .fn_decl => {
                var buf: [1]Node.Index = undefined;
                const proto = tree.fullFnProto(&buf, node) orelse continue;
                const body = tree.nodeData(node).node_and_node[1];
                var names: std.ArrayList([]const u8) = .empty;
                var it = proto.iterate(tree);
                while (it.next()) |param| {
                    const name_token = param.name_token orelse continue;
                    const type_node = param.type_expr orelse continue;
                    if (isConstPointer(tree, type_node)) try names.append(arena, tree.tokenSlice(name_token));
                }
                try fns.append(arena, .{
                    .name = if (proto.name_token) |t| tree.tokenSlice(t) else "(anonymous)",
                    .first = tree.firstToken(body),
                    .last = tree.lastToken(body),
                    .const_params = names.items,
                });
            },
            .builtin_call_two, .builtin_call_two_comma, .builtin_call, .builtin_call_comma => {
                if (std.mem.eql(u8, tree.tokenSlice(tree.nodeMainToken(node)), "@constCast")) try casts.append(arena, node);
            },
            else => {},
        }
    }
    if (casts.items.len == 0) return;

    for (0..tree.nodes.len) |i| {
        const node: Node.Index = @fromBackingInt(@intCast(i));
        const decl = tree.fullVarDecl(node) orelse continue;
        if (tree.tokenTag(decl.ast.mut_token) != .keyword_const) continue;
        const fn_index = innermostFn(fns.items, decl.ast.mut_token) orelse continue;
        // An explicitly typed mutable pointer is a handle to state, not the state itself.
        if (decl.ast.type_node.unwrap()) |t| if (isMutablePointer(tree, t)) continue;
        try locals.append(arena, .{ .name = tree.tokenSlice(decl.ast.mut_token + 1), .token = decl.ast.mut_token, .fn_index = fn_index });
    }

    for (casts.items) |cast| {
        var buf: [2]Node.Index = undefined;
        const params = tree.builtinCallParams(&buf, cast) orelse continue;
        if (params.len != 1) continue;
        const found = rootIdentifier(tree, params[0]) orelse continue;
        const root = found.name;
        const at = tree.nodeMainToken(cast);
        const fn_index = innermostFn(fns.items, at) orelse continue;
        const f = fns.items[fn_index];

        var kind: ?[]const u8 = null;
        for (f.const_params) |p| if (std.mem.eql(u8, p, root)) {
            kind = "param";
        };
        // A const local holding a pointer is no store of its own: only `&local...` casts the local's own storage.
        if (kind == null and found.addr_taken) for (locals.items) |l| {
            if (l.fn_index != fn_index or l.token >= at or !std.mem.eql(u8, l.name, root)) continue;
            if (at <= scopeEnd(tree, l.token)) kind = "local";
        };
        const k = kind orelse continue;
        try report.add(tree, at, f.name, "@constCast of const {s} '{s}'", .{ k, root });
    }
}
