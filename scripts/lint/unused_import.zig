//! Rule `unused-import`: a non-`pub` file-scope `const` that only renames something (`@import(..)`, `a.b.C`, `x`) and
//! whose name no identifier token of the file mentions again. A name after a `.` is a field or an enum literal, never
//! a use of the declaration. Parse only, no Sema.
const std = @import("std");
const Ast = std.zig.Ast;
const Node = Ast.Node;
const rule = @import("rule.zig");

pub const name = "unused-import";
pub const reason = "a file-scope import or alias that nothing in the file names";

/// `@import("x")`, an identifier, or a field-access chain rooted at either.
fn isAlias(tree: *const Ast, start: Node.Index) ?[]const u8 {
    var node = start;
    while (true) switch (tree.nodeTag(node)) {
        .identifier => return "alias",
        .field_access => node = tree.nodeData(node).node_and_token[0],
        .builtin_call_two, .builtin_call_two_comma => {
            if (!std.mem.eql(u8, tree.tokenSlice(tree.nodeMainToken(node)), "@import")) return null;
            return if (node == start) "import" else "alias";
        },
        else => return null,
    };
}

fn bareName(slice: []const u8) []const u8 {
    if (std.mem.startsWith(u8, slice, "@\"")) return slice[2 .. slice.len - 1];
    return slice;
}

const Candidate = struct { token: u32, kind: []const u8, used: bool = false };

/// `arena` outlives the call: the driver frees it once per file.
pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    var candidates: std.StringHashMapUnmanaged(Candidate) = .empty;
    for (tree.rootDecls()) |decl_node| {
        const decl = tree.fullVarDecl(decl_node) orelse continue;
        if (decl.visib_token != null or tree.tokenTag(decl.ast.mut_token) != .keyword_const) continue;
        const init_node = decl.ast.init_node.unwrap() orelse continue;
        const kind = isAlias(tree, init_node) orelse continue;
        const decl_name = bareName(tree.tokenSlice(decl.ast.mut_token + 1));
        if (std.mem.eql(u8, decl_name, "_")) continue;
        try candidates.put(arena, decl_name, .{ .token = decl.ast.mut_token + 1, .kind = kind });
    }
    if (candidates.count() == 0) return;

    var t: u32 = 0;
    while (t < tree.tokens.len) : (t += 1) {
        if (tree.tokenTag(t) != .identifier or (t > 0 and tree.tokenTag(t - 1) == .period)) continue;
        const c = candidates.getPtr(bareName(tree.tokenSlice(t))) orelse continue;
        if (c.token != t) c.used = true;
    }

    var unused: std.ArrayList(Candidate) = .empty;
    var it = candidates.valueIterator();
    while (it.next()) |c| if (!c.used) try unused.append(arena, c.*);
    std.mem.sort(Candidate, unused.items, {}, struct {
        fn less(_: void, a: Candidate, b: Candidate) bool {
            return a.token < b.token;
        }
    }.less);
    for (unused.items) |c| {
        const decl_name = bareName(tree.tokenSlice(c.token));
        try report.add(tree, c.token, decl_name, "unused {s} '{s}'", .{ c.kind, decl_name });
    }
}
