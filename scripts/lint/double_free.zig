//! Rule `double-free`: one name freed by an `errdefer` and, later in that errdefer's scope, by a `defer` with the same
//! call. On an error both run, so the handle is freed twice: a double delete in mlx-c, a tensor released twice on the
//! engine. Parse only: the statements compared are the plain forms `[err]defer [_ =] callee(name);`, and a name
//! reassigned between the two is still reported (allowlist it with the reason).
const std = @import("std");
const Ast = std.zig.Ast;
const Node = Ast.Node;
const rule = @import("rule.zig");

pub const name = "double-free";
pub const reason = "a handle freed by an errdefer and by a later defer is freed twice when an error unwinds both";

const Free = struct { token: u32, callee_first: u32, callee_last: u32, arg: u32, end: u32 };

/// `[err]defer [_ =] a.b.c(name);` at `start`, or null for any other statement.
fn freeAt(tree: *const Ast, start: u32) ?Free {
    var t = start + 1;
    const n = tree.tokens.len;
    if (t + 1 < n and tree.tokenTag(t) == .identifier and std.mem.eql(u8, tree.tokenSlice(t), "_") and tree.tokenTag(t + 1) == .equal) t += 2;
    if (t >= n or tree.tokenTag(t) != .identifier) return null;
    const first = t;
    while (t + 2 < n and tree.tokenTag(t + 1) == .period and tree.tokenTag(t + 2) == .identifier) t += 2;
    const last = t;
    if (t + 4 >= n or tree.tokenTag(t + 1) != .l_paren or tree.tokenTag(t + 2) != .identifier or
        tree.tokenTag(t + 3) != .r_paren or tree.tokenTag(t + 4) != .semicolon) return null;
    return .{ .token = start, .callee_first = first, .callee_last = last, .arg = t + 2, .end = scopeEnd(tree, start) };
}

/// The token the block holding `from` closes at.
fn scopeEnd(tree: *const Ast, from: u32) u32 {
    var depth: i32 = 0;
    var t = from;
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

fn sameCall(tree: *const Ast, a: Free, b: Free) bool {
    if (a.callee_last - a.callee_first != b.callee_last - b.callee_first) return false;
    var i: u32 = 0;
    while (a.callee_first + i <= a.callee_last) : (i += 1) {
        if (!std.mem.eql(u8, tree.tokenSlice(a.callee_first + i), tree.tokenSlice(b.callee_first + i))) return false;
    }
    return std.mem.eql(u8, tree.tokenSlice(a.arg), tree.tokenSlice(b.arg));
}

/// `arena` outlives the call: the driver frees it once per file.
pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    var guards: std.ArrayList(Free) = .empty;
    var frees: std.ArrayList(Free) = .empty;
    for (0..tree.tokens.len) |i| {
        const t: u32 = @intCast(i);
        switch (tree.tokenTag(t)) {
            .keyword_errdefer => if (freeAt(tree, t)) |f| try guards.append(arena, f),
            .keyword_defer => if (freeAt(tree, t)) |f| try frees.append(arena, f),
            else => {},
        }
    }
    if (guards.items.len == 0 or frees.items.len == 0) return;
    const owners = try rule.owners(arena, tree);
    for (guards.items) |g| for (frees.items) |f| {
        if (f.token < g.token or f.token > g.end or !sameCall(tree, g, f)) continue;
        const key = if (rule.ownerOf(owners.items, f.token)) |o| o.key else "-";
        try report.add(tree, f.token, key, "'{s}' is freed by the errdefer on line {d} and by this defer: an error runs both", .{ tree.tokenSlice(f.arg), tree.tokenLocation(0, g.token).line + 1 });
    };
}
