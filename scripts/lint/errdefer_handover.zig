//! Rule `errdefer-handover`: an `errdefer` that frees a handle stays armed past a statement that hands the handle to an
//! owner, a store into a field or through a pointer (`buf.* = next`, `self.x = h`, `@field(st, f) = h`), and a `try`
//! after that store unwinds into it: the errdefer frees the owner's live handle, and the owner frees it again. The
//! handle is handed off when the stored value names it, or names a local made from an expression that names it
//! (`const next = try buf.write(&up, ...)`). The fix ends the errdefer's block before the store (`blk: { ... }`).
//! Parse only, on tokens: the errdefers read are the plain `errdefer [_ =] callee(name);`. Not read: a handover by a
//! call (`list.append(gpa, h)`), an errdefer that frees a field or runs a block, and a store into an element
//! (`slots[made] = h`), whose owner is usually decided by a count the store does not move.
const std = @import("std");
const Ast = std.zig.Ast;
const rule = @import("rule.zig");

pub const name = "errdefer-handover";
pub const reason = "an errdefer still armed after its handle went to an owner frees the owner's handle, which the owner frees again";

/// The calls that free the one MLX array or engine tensor they are given (`core.double_releases` counts the second).
const frees = [_][]const u8{ "mlx_array_free", "release" };

/// The handle `errdefer [_ =] a.free(name);` at `start` frees, or null for any other errdefer.
fn freedAt(tree: *const Ast, start: u32) ?u32 {
    var t = start + 1;
    const n = tree.tokens.len;
    if (t + 1 < n and tree.tokenTag(t) == .identifier and std.mem.eql(u8, tree.tokenSlice(t), "_") and tree.tokenTag(t + 1) == .equal) t += 2;
    if (t >= n or tree.tokenTag(t) != .identifier) return null;
    while (t + 2 < n and tree.tokenTag(t + 1) == .period and tree.tokenTag(t + 2) == .identifier) t += 2;
    if (t + 4 >= n or tree.tokenTag(t + 1) != .l_paren or tree.tokenTag(t + 2) != .identifier or
        tree.tokenTag(t + 3) != .r_paren or tree.tokenTag(t + 4) != .semicolon) return null;
    for (frees) |f| if (std.mem.eql(u8, tree.tokenSlice(t), f)) return t + 2;
    return null;
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

/// The `;` that ends the statement holding `from`, at its own nesting.
fn statementEnd(tree: *const Ast, from: u32, limit: u32) u32 {
    var depth: i32 = 0;
    var t = from;
    while (t < limit) : (t += 1) switch (tree.tokenTag(t)) {
        .l_brace, .l_paren, .l_bracket => depth += 1,
        .r_brace, .r_paren, .r_bracket => depth -= 1,
        .semicolon => if (depth <= 0) return t,
        else => {},
    };
    return limit;
}

/// The first token of the statement whose `=` is at `eq`: just past the `;`, `{` or `}` before it.
fn statementStart(tree: *const Ast, eq: u32, floor: u32) u32 {
    var t = eq;
    while (t > floor) : (t -= 1) switch (tree.tokenTag(t - 1)) {
        .semicolon, .l_brace, .r_brace => return t,
        else => {},
    };
    return floor;
}

/// Whether `name` is one of `tainted`.
fn isTainted(tainted: []const []const u8, ident: []const u8) bool {
    for (tainted) |n| if (std.mem.eql(u8, ident, n)) return true;
    return false;
}

/// The fields of a handle that are the handle: `next.array`, an MLX array's `ctx`. Any other field is a value read from it.
const handle_fields = [_][]const u8{ "array", "ctx" };

/// Whether the value `from..to` holds a tainted handle itself, not a call's argument, an index or another field: `next`,
/// `next.array`, `.{ .stream = h }`, and the argument of a call named `adopt`, which takes it (`InPlace.adopt(h, ...)`).
/// A local made from it (`taints`) holds it only through its address, `&h`, or as `h` alone.
fn holds(tree: *const Ast, tainted: []const []const u8, from: u32, to: u32) bool {
    var depth: i32 = 0;
    var t = from;
    while (t < to) : (t += 1) switch (tree.tokenTag(t)) {
        .l_paren => depth += if (t > from and tree.tokenTag(t - 1) == .identifier and std.mem.eql(u8, tree.tokenSlice(t - 1), "adopt")) 0 else 1,
        .l_bracket => depth += 1,
        .r_paren, .r_bracket => depth -= 1,
        .identifier => if (depth == 0 and !(t > from and tree.tokenTag(t - 1) == .period) and isTainted(tainted, tree.tokenSlice(t)) and !otherField(tree, t, to)) return true,
        else => {},
    };
    return false;
}

fn otherField(tree: *const Ast, at: u32, to: u32) bool {
    if (at + 2 >= to or tree.tokenTag(at + 1) != .period or tree.tokenTag(at + 2) != .identifier) return false;
    for (handle_fields) |f| if (std.mem.eql(u8, tree.tokenSlice(at + 2), f)) return false;
    return true;
}

fn taints(tree: *const Ast, tainted: []const []const u8, from: u32, to: u32) bool {
    if (to == from + 1 and tree.tokenTag(from) == .identifier) return isTainted(tainted, tree.tokenSlice(from));
    var t = from;
    while (t + 1 < to) : (t += 1) {
        if (tree.tokenTag(t) == .ampersand and tree.tokenTag(t + 1) == .identifier and isTainted(tainted, tree.tokenSlice(t + 1)) and !intoMlx(tree, from, t)) return true;
    }
    return false;
}

/// Whether the `&` at `at` is an argument of an mlx-c call, which writes the handle in place and returns a status code.
fn intoMlx(tree: *const Ast, from: u32, at: u32) bool {
    var depth: i32 = 0;
    var t = at;
    while (t > from) {
        t -= 1;
        switch (tree.tokenTag(t)) {
            .r_paren => depth += 1,
            .l_paren => {
                if (depth == 0) return t > from and tree.tokenTag(t - 1) == .identifier and std.mem.startsWith(u8, tree.tokenSlice(t - 1), "mlx_");
                depth -= 1;
            },
            else => {},
        }
    }
    return false;
}

/// Whether the token at `at` starts a statement of its own, not the body of an `if`, `else`, prong or capture.
fn startsStatement(tree: *const Ast, at: u32) bool {
    if (at == 0) return true;
    return switch (tree.tokenTag(at - 1)) {
        .semicolon, .l_brace, .r_brace => true,
        else => false,
    };
}

/// Past an `else` at `t` and the branch it opens, `else if` chains included: a branch beside the one that stored.
fn pastElse(tree: *const Ast, at: u32, limit: u32) u32 {
    var t = at;
    while (t < limit and tree.tokenTag(t) == .keyword_else) {
        t += 1;
        if (t < limit and tree.tokenTag(t) == .keyword_if) {
            t = balanced(tree, t + 1, limit);
            if (t < limit and tree.tokenTag(t) == .pipe) t = afterPipe(tree, t, limit);
        } else if (t < limit and tree.tokenTag(t) == .pipe) t = afterPipe(tree, t, limit);
        t = if (t < limit and tree.tokenTag(t) == .l_brace) balanced(tree, t, limit) else statementEnd(tree, t, limit) + 1;
    }
    return t;
}

/// Past the group that opens at `at` (`(`, `{` or `[`).
fn balanced(tree: *const Ast, at: u32, limit: u32) u32 {
    var depth: i32 = 0;
    var t = at;
    while (t < limit) : (t += 1) switch (tree.tokenTag(t)) {
        .l_paren, .l_brace, .l_bracket => depth += 1,
        .r_paren, .r_brace, .r_bracket => {
            depth -= 1;
            if (depth == 0) return t + 1;
        },
        else => {},
    };
    return limit;
}

fn afterPipe(tree: *const Ast, at: u32, limit: u32) u32 {
    var t = at + 1;
    while (t < limit and tree.tokenTag(t) != .pipe) t += 1;
    return t + 1;
}

/// The first error exit after the store that ends at `from`, inside the errdefer's scope: a `try`, a `return error.X`, a
/// `catch |e| return e`. Null when the handle is set again first (the errdefer then frees what replaced it, never the
/// owner's), or when the path returns without an error first.
fn tryAfter(tree: *const Ast, handle: []const u8, from: u32, end: u32) ?u32 {
    var depth: i32 = 0;
    var t = from;
    while (t < end) {
        switch (tree.tokenTag(t)) {
            .keyword_try => return t,
            .keyword_return => {
                if (t + 1 < end and tree.tokenTag(t + 1) == .keyword_error) return t;
                if (t >= 4 and t + 1 < end and tree.tokenTag(t - 1) == .pipe and tree.tokenTag(t - 3) == .pipe and tree.tokenTag(t - 4) == .keyword_catch and
                    tree.tokenTag(t + 1) == .identifier and std.mem.eql(u8, tree.tokenSlice(t - 2), tree.tokenSlice(t + 1))) return t;
                // The path from the store ends here: only this statement's own exits are left.
                if (depth == 0 and startsStatement(tree, t)) {
                    const se = statementEnd(tree, t, end);
                    while (t < se) : (t += 1) if (tree.tokenTag(t) == .keyword_try) return t;
                    return null;
                }
            },
            .l_brace => depth += 1,
            .r_brace => {
                depth -= 1;
                if (depth < 0) {
                    depth = 0;
                    // A switch prong's block: the prongs beside it are other paths, so the scan goes on past the switch.
                    if (t + 1 < end and tree.tokenTag(t + 1) == .comma) {
                        t = scopeEnd(tree, t + 1) + 1;
                        continue;
                    }
                    t = pastElse(tree, t + 1, end);
                    continue;
                }
            },
            .identifier => if (std.mem.eql(u8, tree.tokenSlice(t), handle) and t + 1 < end and tree.tokenTag(t + 1) == .equal and
                (t == 0 or tree.tokenTag(t - 1) == .semicolon or tree.tokenTag(t - 1) == .l_brace or tree.tokenTag(t - 1) == .r_brace)) return null,
            else => {},
        }
        t += 1;
    }
    return null;
}

/// Whether the target `from..eq` of an assignment is a store into an owner: a field or a pointer's pointee, not a local
/// nor a field of a struct literal (`.{ .a = x }`).
fn ownerStore(tree: *const Ast, from: u32, eq: u32) bool {
    if (from >= eq) return false;
    switch (tree.tokenTag(from)) {
        .builtin => return std.mem.eql(u8, tree.tokenSlice(from), "@field"),
        .identifier => {},
        else => return false,
    }
    var t = from + 1;
    while (t < eq) : (t += 1) switch (tree.tokenTag(t)) {
        .period_asterisk, .period => return true,
        else => {},
    };
    return false;
}

/// Past the `if (...)`, `else` and `|x|` that guard a one-statement branch (`if (c) self.x = h;`): the store's target.
fn targetStart(tree: *const Ast, from: u32, eq: u32) u32 {
    var t = from;
    while (t < eq) switch (tree.tokenTag(t)) {
        .keyword_if, .keyword_while => t = balanced(tree, t + 1, eq),
        .keyword_else => t += 1,
        .pipe => t = afterPipe(tree, t, eq),
        else => return t,
    };
    return t;
}

/// Whether the target at `from..eq` is a part of a `var` value of the function's own (`var out: S = .{}; out.x = h`),
/// which owns nothing until the function returns it; a store through a pointer (`.*`, a `var p = &...`) is not one.
fn localValue(tree: *const Ast, from: u32, eq: u32) bool {
    if (tree.tokenTag(from) != .identifier) return false;
    var t = from;
    while (t < eq) : (t += 1) if (tree.tokenTag(t) == .period_asterisk) return false;
    const ident = tree.tokenSlice(from);
    var b = from;
    while (b > 1) {
        b -= 1;
        switch (tree.tokenTag(b)) {
            .keyword_fn => return false,
            .identifier => if (tree.tokenTag(b - 1) == .keyword_var and std.mem.eql(u8, tree.tokenSlice(b), ident)) {
                const after = b + 1;
                if (after + 1 < tree.tokens.len and tree.tokenTag(after) == .colon and tree.tokenTag(after + 1) == .asterisk) return false;
                if (after + 1 < tree.tokens.len and tree.tokenTag(after) == .equal and tree.tokenTag(after + 1) == .ampersand) return false;
                return true;
            },
            else => {},
        }
    }
    return false;
}

/// `arena` outlives the call: the driver frees it once per file.
pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    var owners: ?std.ArrayList(rule.Owner) = null;
    for (0..tree.tokens.len) |i| {
        const g: u32 = @intCast(i);
        if (tree.tokenTag(g) != .keyword_errdefer) continue;
        const handle = freedAt(tree, g) orelse continue;
        const end = scopeEnd(tree, g);
        var tainted: std.ArrayList([]const u8) = .empty;
        try tainted.append(arena, tree.tokenSlice(handle));
        var t = statementEnd(tree, g, end) + 1;
        const handed: ?u32 = while (t < end) : (t += 1) {
            switch (tree.tokenTag(t)) {
                .keyword_const, .keyword_var => if (t + 2 < end and tree.tokenTag(t + 1) == .identifier) {
                    var eq = t + 2;
                    while (eq < end and tree.tokenTag(eq) != .equal and tree.tokenTag(eq) != .semicolon) eq += 1;
                    if (eq < end and tree.tokenTag(eq) == .equal and taints(tree, tainted.items, eq + 1, statementEnd(tree, eq, end)))
                        try tainted.append(arena, tree.tokenSlice(t + 1));
                },
                .equal => {
                    const from = targetStart(tree, statementStart(tree, t, g), t);
                    // A store into the handle itself hands it to no one.
                    if (ownerStore(tree, from, t) and !isTainted(tainted.items, tree.tokenSlice(from)) and !localValue(tree, from, t) and
                        holds(tree, tainted.items, t + 1, statementEnd(tree, t, end))) break t;
                },
                else => {},
            }
        } else null;
        const store = handed orelse continue;
        const at = tryAfter(tree, tree.tokenSlice(handle), statementEnd(tree, store, end) + 1, end) orelse continue;
        if (owners == null) owners = try rule.owners(arena, tree);
        const key = if (rule.ownerOf(owners.?.items, at)) |o| o.key else "-";
        try report.add(tree, at, key, "'{s}' went to its owner on line {d}, and the errdefer on line {d} that frees it is still armed here: a failure frees the owner's handle", .{ tree.tokenSlice(handle), tree.tokenLocation(0, store).line + 1, tree.tokenLocation(0, g).line + 1 });
    }
}
