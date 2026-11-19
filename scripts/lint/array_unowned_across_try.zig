//! Rule `array-unowned-across-try`: an mlx array that a checked call has filled, held only by a local across a later
//! `try`, or made by one `try` of an array literal that a later `try` of the same literal can fail. The error return
//! drops the handle: on Sushi Cooler its buffer stays in the engine's pool for the life of the process. Owned means a
//! `defer`/`errdefer` that names it, or a hand-off (`= x;`, `return`) before that later `try`. Token scan, no Sema.
const std = @import("std");
const Ast = std.zig.Ast;
const rule = @import("rule.zig");

pub const name = "array-unowned-across-try";
pub const reason = "a later try that fails drops an array that only a local or a literal's slot holds";

pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    if (rule.testFile(report.file)) return;
    const owners = try rule.owners(arena, tree);
    const test_only = try rule.testOnly(arena, tree, owners.items);
    var t: u32 = 0;
    while (t + 8 < tree.tokens.len) : (t += 1) {
        if (rule.inTest(owners.items, t) or inTestOnly(owners.items, test_only, t)) continue;
        const owner = if (rule.ownerOf(owners.items, t)) |o| o.key else "-";
        if (literalOpen(tree, t)) |brace| {
            if (secondTry(tree, brace)) |at| try report.add(tree, at, owner, "an array literal of mlx arrays made by several trys: a later one that fails leaks the earlier slots; fill them one by one behind a defer", .{});
            continue;
        }
        if (tree.tokenTag(t) != .keyword_var or tree.tokenTag(t + 1) != .identifier or tree.tokenTag(t + 2) != .equal) continue;
        if (!isNew(tree, t + 3)) continue;
        const v = tree.tokenSlice(t + 1);
        if (unownedTry(tree, t + 9, v)) |at| {
            try report.add(tree, at, owner, "'{s}' holds a filled array across this try, which leaks it on an error: errdefer its free, or hand it to its owner first", .{v});
        }
    }
}

/// Whether token `t` sits in a fn only tests reach (`rule.testOnly`): a test's leak ends with its process.
fn inTestOnly(list: []const rule.Owner, test_only: []const bool, t: u32) bool {
    for (list, test_only) |o, m| if (m and t >= o.first and t <= o.last) return true;
    return false;
}

/// `mlx . mlx_array_new ( ) ;` at `at`.
fn isNew(tree: *const Ast, at: u32) bool {
    const tags = [_]std.zig.Token.Tag{ .identifier, .period, .identifier, .l_paren, .r_paren, .semicolon };
    for (tags, 0..) |tag, i| if (tree.tokenTag(at + @as(u32, @intCast(i))) != tag) return false;
    return std.mem.eql(u8, tree.tokenSlice(at), "mlx") and std.mem.eql(u8, tree.tokenSlice(at + 2), "mlx_array_new");
}

/// The `{` of a `[_]mlx.mlx_array{` or `[N]mlx.mlx_array{` literal starting at `at` (not a fn's return type and body).
fn literalOpen(tree: *const Ast, at: u32) ?u32 {
    if (at == 0 or tree.tokenTag(at) != .l_bracket or tree.tokenTag(at + 2) != .r_bracket) return null;
    switch (tree.tokenTag(at - 1)) {
        .equal, .l_paren, .comma, .ampersand, .keyword_return => {},
        else => return null,
    }
    const tags = [_]std.zig.Token.Tag{ .identifier, .period, .identifier, .l_brace };
    for (tags, 0..) |tag, i| if (tree.tokenTag(at + 3 + @as(u32, @intCast(i))) != tag) return null;
    if (!std.mem.eql(u8, tree.tokenSlice(at + 3), "mlx") or !std.mem.eql(u8, tree.tokenSlice(at + 5), "mlx_array")) return null;
    return at + 6;
}

/// The literal's second `try`, if it has one.
fn secondTry(tree: *const Ast, brace: u32) ?u32 {
    var depth: i32 = 0;
    var tries: u32 = 0;
    var t = brace;
    while (t < tree.tokens.len) : (t += 1) {
        switch (tree.tokenTag(t)) {
            .l_brace => depth += 1,
            .r_brace => {
                depth -= 1;
                if (depth == 0) return null;
            },
            .keyword_try => {
                tries += 1;
                if (tries == 2) return t;
            },
            else => {},
        }
    }
    return null;
}

/// The first `try` after a `try` statement that filled `v` (`&v`), with no guard of `v` and no hand-off between, in
/// `v`'s block.
fn unownedTry(tree: *const Ast, from: u32, v: []const u8) ?u32 {
    var depth: i32 = 0;
    var filled = false;
    var t = from;
    while (t + 2 < tree.tokens.len) : (t += 1) {
        switch (tree.tokenTag(t)) {
            .l_brace => depth += 1,
            .r_brace => {
                depth -= 1;
                if (depth < 0) return null;
            },
            .keyword_try => {
                if (keeps(tree, t, v)) return null;
                const n = fills(tree, t, v);
                // A later fill of `v` is read as another branch's (`if .. try f(&v) else try g(&v)`).
                if (filled and n == 0) return t;
                switch (n) {
                    0 => {},
                    1 => {
                        filled = true;
                        // A `try` inside the statement that fills it runs before the fill.
                        while (t + 1 < tree.tokens.len and tree.tokenTag(t + 1) != .semicolon) t += 1;
                    },
                    // `sc.res(op(&v, ..), &v)`: the call that fills it hands it to a scope.
                    else => return null,
                }
            },
            .keyword_defer, .keyword_errdefer => if (deferNames(tree, t, v)) return null,
            .keyword_return => if (namesBeforeSemicolon(tree, t, v)) return null,
            .equal => if (isName(tree, t + 1, v) and switch (tree.tokenTag(t + 2)) {
                .semicolon, .comma, .r_brace => true,
                else => false,
            }) return null,
            else => {},
        }
    }
    return null;
}

fn isName(tree: *const Ast, at: u32, v: []const u8) bool {
    return tree.tokenTag(at) == .identifier and std.mem.eql(u8, tree.tokenSlice(at), v);
}

/// How many times `& v` appears in the statement that starts at `from`.
fn fills(tree: *const Ast, from: u32, v: []const u8) u32 {
    var n: u32 = 0;
    var t = from;
    while (t + 1 < tree.tokens.len and tree.tokenTag(t) != .semicolon) : (t += 1) {
        if (tree.tokenTag(t) == .ampersand and isName(tree, t + 1, v)) n += 1;
    }
    return n;
}

/// `keep ( v )` in the statement at `from`: a scope's `keep` takes the array, and frees it when it cannot.
fn keeps(tree: *const Ast, from: u32, v: []const u8) bool {
    var t = from;
    while (t + 3 < tree.tokens.len and tree.tokenTag(t) != .semicolon) : (t += 1) {
        if (tree.tokenTag(t) == .identifier and std.mem.eql(u8, tree.tokenSlice(t), "keep") and
            tree.tokenTag(t + 1) == .l_paren and isName(tree, t + 2, v) and tree.tokenTag(t + 3) == .r_paren) return true;
    }
    return false;
}

/// Whether the `defer`/`errdefer` at `at` names `v`: its block when it has one, else its statement.
fn deferNames(tree: *const Ast, at: u32, v: []const u8) bool {
    if (tree.tokenTag(at + 1) != .l_brace) return namesBeforeSemicolon(tree, at, v);
    var depth: i32 = 0;
    var t = at + 1;
    while (t < tree.tokens.len) : (t += 1) {
        switch (tree.tokenTag(t)) {
            .l_brace => depth += 1,
            .r_brace => {
                depth -= 1;
                if (depth == 0) return false;
            },
            else => if (isName(tree, t, v)) return true,
        }
    }
    return false;
}

fn namesBeforeSemicolon(tree: *const Ast, from: u32, v: []const u8) bool {
    var t = from;
    while (t < tree.tokens.len and tree.tokenTag(t) != .semicolon) : (t += 1) {
        if (isName(tree, t, v)) return true;
    }
    return false;
}
