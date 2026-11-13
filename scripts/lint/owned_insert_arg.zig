//! Rule `owned-insert-arg`: an owned value made inside the argument of a fallible insert,
//! `try xs.append(a, try a.dupe(u8, s))`: when the insert fails to grow, the copy it was handed is never freed. Reserve
//! first and insert with `*AssumeCapacity`, or make the value first and free it if the insert fails. An insert is a
//! method whose name starts with `append`, `put` or `insert`; a maker, a call whose name starts with `dupe` or `alloc`.
//! Test code is exempt; an arena's insert leaks nothing and goes in the allowlist. Token scan, no Sema.
const std = @import("std");
const Ast = std.zig.Ast;
const rule = @import("rule.zig");

pub const name = "owned-insert-arg";
pub const reason = "a value made for an insert is freed when the insert fails";

fn startsWithAny(s: []const u8, prefixes: []const []const u8) bool {
    for (prefixes) |p| if (std.mem.startsWith(u8, s, p)) return true;
    return false;
}

/// The name token of the call `try <chain>(`; with `method`, only a call through a period (`xs.append(`).
fn calledName(tree: *const Ast, try_tok: u32, method: bool) ?u32 {
    var t = try_tok + 1;
    if (t >= tree.tokens.len or tree.tokenTag(t) != .identifier) return null;
    var dotted = false;
    while (t + 2 < tree.tokens.len and tree.tokenTag(t + 1) == .period and tree.tokenTag(t + 2) == .identifier) : (dotted = true) t += 2;
    if (t + 1 >= tree.tokens.len or tree.tokenTag(t + 1) != .l_paren or (method and !dotted)) return null;
    return t;
}

pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    if (rule.testFile(report.file)) return;
    const owners = try rule.owners(arena, tree);
    const test_only = try rule.testOnly(arena, tree, owners.items);
    var t: u32 = 0;
    while (t < tree.tokens.len) : (t += 1) {
        if (tree.tokenTag(t) != .keyword_try) continue;
        const insert = calledName(tree, t, true) orelse continue;
        if (!startsWithAny(tree.tokenSlice(insert), &.{ "append", "put", "insert" })) continue;
        var depth: usize = 0;
        var close = insert + 1;
        while (close < tree.tokens.len) : (close += 1) {
            if (tree.tokenTag(close) == .l_paren) depth += 1;
            if (tree.tokenTag(close) == .r_paren) {
                depth -= 1;
                if (depth == 0) break;
            }
        }
        var inner = insert + 2;
        while (inner < close) : (inner += 1) {
            if (tree.tokenTag(inner) != .keyword_try) continue;
            const maker = calledName(tree, inner, false) orelse continue;
            if (!startsWithAny(tree.tokenSlice(maker), &.{ "dupe", "alloc" })) continue;
            if (rule.inTest(owners.items, inner)) continue;
            var held = false;
            for (owners.items, test_only) |o, m| if (m and inner >= o.first and inner <= o.last) {
                held = true;
            };
            if (held) continue;
            const owner = rule.ownerOf(owners.items, inner);
            try report.add(tree, inner, if (owner) |o| o.key else "-", "'{s}' makes an owned value inside '{s}': make it first and free it if the insert fails", .{ tree.tokenSlice(maker), tree.tokenSlice(insert) });
        }
        t = insert;
    }
}
