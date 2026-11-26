//! Rule `undefined-partial-pass`: a local ARRAY declared `= undefined` whose elements are written field by field
//! (`x[i].f = ...`) and that is then passed to a call before any whole element or array assignment, `@memset` or method
//! call on it. The fields never written stay garbage in the callee, and a fixture change that makes the callee read
//! one turns a test's verdict into stack contents. Parse only: an element struct declared in this file is cleared by
//! writing every field; a plain struct local is not checked (a hand-built struct whose callee reads only what the test
//! wrote is the repository's common fixture, and no parse can tell it from the trap); writes through a capture or a
//! pointer are not followed.
const std = @import("std");
const Ast = std.zig.Ast;
const Node = Ast.Node;
const rule = @import("rule.zig");

pub const name = "undefined-partial-pass";
pub const reason = "an undefined local written field by field is passed on with its other fields garbage";

const Write = struct { root: []const u8, whole: bool, field: []const u8, token: u32 };

/// `x`, `x[i]`, `x.f`, `x[i].f.g` as its root, whether it is the whole root or an element, and the field nearest the root.
fn writeOf(tree: *const Ast, start: Node.Index, token: u32) ?Write {
    var node = start;
    var field: []const u8 = "";
    while (true) switch (tree.nodeTag(node)) {
        .field_access => {
            const data = tree.nodeData(node).node_and_token;
            field = tree.tokenSlice(data[1]);
            node = data[0];
        },
        .array_access => node = tree.nodeData(node).node_and_node[0],
        .identifier => return .{ .root = tree.tokenSlice(tree.nodeMainToken(node)), .whole = field.len == 0, .field = field, .token = token },
        else => return null,
    };
}

/// The local an argument names: `x`, `&x`, `&x[a..b]`, `x[i]`, `x[a..b]`.
fn argRoot(tree: *const Ast, node: Node.Index) ?[]const u8 {
    return switch (tree.nodeTag(node)) {
        .identifier => tree.tokenSlice(tree.nodeMainToken(node)),
        .address_of => argRoot(tree, tree.nodeData(node).node),
        .array_access => argRoot(tree, tree.nodeData(node).node_and_node[0]),
        .slice, .slice_open, .slice_sentinel => argRoot(tree, tree.fullSlice(node).?.ast.sliced),
        else => null,
    };
}

fn typeName(tree: *const Ast, node: Node.Index) ?[]const u8 {
    return switch (tree.nodeTag(node)) {
        .identifier => tree.tokenSlice(tree.nodeMainToken(node)),
        else => null,
    };
}

/// The number of fields of `const <name> = struct { ... }` at the top of this file; null when it is not declared here.
fn fieldCount(tree: *const Ast, type_name: []const u8) ?usize {
    for (tree.rootDecls()) |decl| {
        const v = tree.fullVarDecl(decl) orelse continue;
        if (!std.mem.eql(u8, tree.tokenSlice(v.ast.mut_token + 1), type_name)) continue;
        const init_node = v.ast.init_node.unwrap() orelse continue;
        var buf: [2]Node.Index = undefined;
        const container = tree.fullContainerDecl(&buf, init_node) orelse return null;
        var n: usize = 0;
        for (container.ast.members) |m| if (tree.fullContainerField(m) != null) {
            n += 1;
        };
        return n;
    }
    return null;
}

/// The token the block holding `decl_token` closes at.
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

const Use = struct { root: []const u8, token: u32 };

fn isUndefinedLiteral(tree: *const Ast, node: Node.Index) bool {
    return tree.nodeTag(node) == .identifier and std.mem.eql(u8, tree.tokenSlice(tree.nodeMainToken(node)), "undefined");
}

/// `arena` outlives the call: the driver frees it once per file.
pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    var writes: std.ArrayList(Write) = .empty;
    var uses: std.ArrayList(Use) = .empty;
    // A whole initialization: `@memset(x, ...)` or `x.method(...)`.
    var inits: std.ArrayList(Use) = .empty;
    for (0..tree.nodes.len) |i| {
        const node: Node.Index = @fromBackingInt(@intCast(i));
        switch (tree.nodeTag(node)) {
            .assign => {
                const w = writeOf(tree, tree.nodeData(node).node_and_node[0], tree.firstToken(node)) orelse continue;
                try writes.append(arena, w);
            },
            .call, .call_comma, .call_one, .call_one_comma => {
                var buf: [1]Node.Index = undefined;
                const call = tree.fullCall(&buf, node) orelse continue;
                const token = tree.firstToken(node);
                if (tree.nodeTag(call.ast.fn_expr) == .field_access) {
                    const callee = writeOf(tree, call.ast.fn_expr, token);
                    if (callee) |c| if (tree.nodeTag(tree.nodeData(call.ast.fn_expr).node_and_token[0]) == .identifier) {
                        try inits.append(arena, .{ .root = c.root, .token = token });
                    };
                }
                for (call.ast.params) |p| if (argRoot(tree, p)) |root| try uses.append(arena, .{ .root = root, .token = token });
            },
            .builtin_call, .builtin_call_comma, .builtin_call_two, .builtin_call_two_comma => {
                var buf: [2]Node.Index = undefined;
                const params = tree.builtinCallParams(&buf, node) orelse continue;
                if (params.len == 0 or !std.mem.eql(u8, tree.tokenSlice(tree.nodeMainToken(node)), "@memset")) continue;
                if (argRoot(tree, params[0])) |root| try inits.append(arena, .{ .root = root, .token = tree.firstToken(node) });
            },
            else => {},
        }
    }
    if (writes.items.len == 0 or uses.items.len == 0) return;
    const owners = try rule.owners(arena, tree);

    for (0..tree.nodes.len) |i| {
        const node: Node.Index = @fromBackingInt(@intCast(i));
        const decl = tree.fullVarDecl(node) orelse continue;
        if (tree.tokenTag(decl.ast.mut_token) != .keyword_var) continue;
        const type_node = decl.ast.type_node.unwrap() orelse continue;
        const init_node = decl.ast.init_node.unwrap() orelse continue;
        if (!isUndefinedLiteral(tree, init_node)) continue;
        const local = tree.tokenSlice(decl.ast.mut_token + 1);
        const end = scopeEnd(tree, decl.ast.mut_token);

        // The element or struct type, to count its fields.
        const array = tree.fullArrayType(type_node);
        const elem_name = typeName(tree, if (array) |a| a.ast.elem_type else type_node);
        const total: ?usize = if (elem_name) |n| fieldCount(tree, n) else null;
        if (array == null) continue;

        var whole_at: u32 = std.math.maxInt(u32);
        for (writes.items) |w| if (w.whole and w.token > decl.ast.mut_token and w.token < end and std.mem.eql(u8, w.root, local)) {
            whole_at = @min(whole_at, w.token);
        };
        for (inits.items) |w| if (w.token > decl.ast.mut_token and w.token < end and std.mem.eql(u8, w.root, local)) {
            whole_at = @min(whole_at, w.token);
        };

        var use_token: ?u32 = null;
        for (uses.items) |u| if (u.token > decl.ast.mut_token and u.token < end and u.token < whole_at and std.mem.eql(u8, u.root, local)) {
            use_token = if (use_token) |t| @min(t, u.token) else u.token;
        };
        const at = use_token orelse continue;

        var seen: std.ArrayList([]const u8) = .empty;
        for (writes.items) |w| {
            if (w.whole or w.token < decl.ast.mut_token or w.token > at or !std.mem.eql(u8, w.root, local)) continue;
            var dup = false;
            for (seen.items) |s| dup = dup or std.mem.eql(u8, s, w.field);
            if (!dup) try seen.append(arena, w.field);
        }
        if (seen.items.len == 0) continue;
        if (total) |t| if (seen.items.len >= t) continue;
        const owner = if (rule.ownerOf(owners.items, at)) |o| o.key else "-";
        try report.add(tree, at, owner, "'{s}' is declared undefined and only {d} of its fields are written before it is passed on; build it whole", .{ local, seen.items.len });
    }
}
