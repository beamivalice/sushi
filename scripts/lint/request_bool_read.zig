//! Rule `request-bool-read`: `x == .bool` or `x != .bool` in a file that reads a request body. A bool field read that
//! way takes its default for a value of any other type, so `"stream":"true"` reads as absent; the reader is
//! `request_parse.boolField`, and `boolFieldFault` refuses the wrong type by name. A check that refuses the other type
//! itself is an allowlist entry. Token scan of the parsed file, no Sema.
const std = @import("std");
const Ast = std.zig.Ast;
const rule = @import("rule.zig");

pub const name = "request-bool-read";
pub const reason = "a request bool compared with .bool reads a wrong-typed value as absent; read it with request_parse.boolField";

const request_files = [_][]const u8{
    "src/server.zig",         "src/anthropic.zig",    "src/completions.zig", "src/responses_api.zig",
    "src/http_conn.zig",      "src/request_policy.zig", "src/generation_stream.zig",
};

pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    for (request_files) |f| {
        if (std.mem.eql(u8, report.file, f)) break;
    } else return;
    const owners = try rule.owners(arena, tree);
    var t: u32 = 0;
    while (t + 2 < tree.tokens.len) : (t += 1) {
        const tag = tree.tokenTag(t);
        if (tag != .equal_equal and tag != .bang_equal) continue;
        if (tree.tokenTag(t + 1) != .period or tree.tokenTag(t + 2) != .identifier) continue;
        if (!std.mem.eql(u8, tree.tokenSlice(t + 2), "bool")) continue;
        const owner = rule.ownerOf(owners.items, t) orelse rule.Owner{ .key = "-", .first = 0, .last = 0, .is_test = false };
        if (owner.is_test) continue;
        try report.add(tree, t, owner.key, "a bool compared with .bool: read a request's bool with request_parse.boolField", .{});
    }
}
