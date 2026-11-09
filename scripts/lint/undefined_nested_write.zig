//! Rule `undefined-nested-write`: in a test block, a local declared `= undefined` and then written through a nested
//! struct (`v.config.x = ...`) before any whole assignment of that struct (`v.config = ...`) or of the local. The
//! other fields of the nested struct stay garbage, and in ReleaseFast code that reads them depends on stack contents.
//! Parse only, no Sema: a field written through a pointer or an index is not followed, and a method called on the
//! local (`v.init()`) counts as initializing all of it.
const std = @import("std");
const Ast = std.zig.Ast;
const Node = Ast.Node;
const rule = @import("rule.zig");

pub const name = "undefined-nested-write";
pub const reason = "a field write into a nested struct of an undefined local leaves the rest of that struct garbage";

const Target = struct { root: []const u8, first_field: []const u8, depth: usize };

/// `v.a.b.c` as root `v`, first field `a`, depth 3; null when the chain is not rooted at a plain identifier.
fn targetOf(tree: *const Ast, start: Node.Index) ?Target {
    var node = start;
    var depth: usize = 0;
    var first: []const u8 = "";
    while (true) switch (tree.nodeTag(node)) {
        .field_access => {
            const data = tree.nodeData(node).node_and_token;
            first = tree.tokenSlice(data[1]);
            depth += 1;
            node = data[0];
        },
        .identifier => return .{ .root = tree.tokenSlice(tree.nodeMainToken(node)), .first_field = first, .depth = depth },
        else => return null,
    };
}

fn isUndefinedLiteral(tree: *const Ast, node: Node.Index) bool {
    return tree.nodeTag(node) == .identifier and std.mem.eql(u8, tree.tokenSlice(tree.nodeMainToken(node)), "undefined");
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

const Test = struct { key: []const u8, first: u32, last: u32 };

/// The allowlist key of a test: its name with everything but letters and digits as `_`.
fn testKey(arena: std.mem.Allocator, tree: *const Ast, name_token: ?u32) ![]const u8 {
    const t = name_token orelse return "(unnamed)";
    const slice = tree.tokenSlice(t);
    const inner = if (slice.len >= 2 and slice[0] == '"') slice[1 .. slice.len - 1] else slice;
    const out = try arena.dupe(u8, inner);
    for (out) |*c| if (!std.ascii.isAlphanumeric(c.*)) {
        c.* = '_';
    };
    return out;
}

const Assign = struct { token: u32, target: Target };

fn contains(list: []const []const u8, item: []const u8) bool {
    for (list) |l| if (std.mem.eql(u8, l, item)) return true;
    return false;
}

/// `arena` outlives the call: the driver frees it once per file.
pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    var assigns: std.ArrayList(Assign) = .empty;
    var tests: std.ArrayList(Test) = .empty;
    for (0..tree.nodes.len) |i| {
        const node: Node.Index = @fromBackingInt(@intCast(i));
        switch (tree.nodeTag(node)) {
            .assign => {
                const target = targetOf(tree, tree.nodeData(node).node_and_node[0]) orelse continue;
                try assigns.append(arena, .{ .token = tree.firstToken(node), .target = target });
            },
            .call, .call_comma, .call_one, .call_one_comma => {
                var buf: [1]Node.Index = undefined;
                const call = tree.fullCall(&buf, node) orelse continue;
                const callee = targetOf(tree, call.ast.fn_expr) orelse continue;
                // `v.init(...)` is taken to initialize the whole of `v`.
                if (callee.depth == 1) try assigns.append(arena, .{ .token = tree.firstToken(node), .target = .{ .root = callee.root, .first_field = "", .depth = 0 } });
            },
            .test_decl => {
                const data = tree.nodeData(node).opt_token_and_node;
                const name_token: ?u32 = data[0].unwrap();
                try tests.append(arena, .{
                    .key = try testKey(arena, tree, name_token),
                    .first = tree.firstToken(data[1]),
                    .last = tree.lastToken(data[1]),
                });
            },
            else => {},
        }
    }
    if (assigns.items.len == 0 or tests.items.len == 0) return;
    std.mem.sort(Assign, assigns.items, {}, struct {
        fn less(_: void, a: Assign, b: Assign) bool {
            return a.token < b.token;
        }
    }.less);

    for (0..tree.nodes.len) |i| {
        const node: Node.Index = @fromBackingInt(@intCast(i));
        const decl = tree.fullVarDecl(node) orelse continue;
        if (tree.tokenTag(decl.ast.mut_token) != .keyword_var or decl.ast.type_node == .none) continue;
        const init_node = decl.ast.init_node.unwrap() orelse continue;
        if (!isUndefinedLiteral(tree, init_node)) continue;
        var in_test: ?Test = null;
        for (tests.items) |t| if (decl.ast.mut_token > t.first and decl.ast.mut_token < t.last) {
            in_test = t;
        };
        const t = in_test orelse continue;

        const local = tree.tokenSlice(decl.ast.mut_token + 1);
        const end = scopeEnd(tree, decl.ast.mut_token);
        var whole_set = false;
        var seen: std.ArrayList([]const u8) = .empty;
        for (assigns.items) |a| {
            if (a.token < decl.ast.mut_token or a.token > end or !std.mem.eql(u8, a.target.root, local)) continue;
            if (a.target.depth == 0) {
                whole_set = true;
            } else if (a.target.depth == 1) {
                try seen.append(arena, a.target.first_field);
            } else if (!whole_set and !contains(seen.items, a.target.first_field)) {
                try seen.append(arena, a.target.first_field);
                try report.add(tree, a.token, t.key, "'{s}.{s}' is written field by field into an undefined '{s}'; assign the whole '{s}.{s}' first", .{ local, a.target.first_field, local, local, a.target.first_field });
            }
        }
    }
}
