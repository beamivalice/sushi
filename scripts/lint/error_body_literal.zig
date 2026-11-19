//! Rule `error-body-literal`: a string literal that spells an error envelope (`{"error":` or `"type":"error"`) outside
//! `src/http.zig`, whose `sendError` is the one error sender: a body built by hand misses the request's API (an
//! OpenAI body on an Anthropic route), the HEAD rule and the WebSocket frame. Tests and `*_test.zig` files read
//! envelopes and are not scanned. Token scan.
const std = @import("std");
const Ast = std.zig.Ast;
const rule = @import("rule.zig");

pub const name = "error-body-literal";
pub const reason = "every error body goes through http.sendError, which picks the envelope by the request's API";

const spellings = [_][]const u8{ "{\\\"error\\\":", "{{\\\"error\\\":", "{\"error\":", "{{\"error\":", "\\\"type\\\":\\\"error\\\"", "\"type\":\"error\"" };

pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    if (std.mem.eql(u8, report.file, "src/http.zig") or std.mem.endsWith(u8, report.file, "_test.zig")) return;
    const owners = try rule.owners(arena, tree);
    for (0..tree.tokens.len) |i| {
        const t: u32 = @intCast(i);
        switch (tree.tokenTag(t)) {
            .string_literal, .multiline_string_literal_line => {},
            else => continue,
        }
        if (rule.inTest(owners.items, t)) continue;
        const text = tree.tokenSlice(t);
        for (spellings) |s| if (std.mem.indexOf(u8, text, s) != null) {
            const key = if (rule.ownerOf(owners.items, t)) |o| o.key else "-";
            try report.add(tree, t, key, "a hand-built error body: send it through http.sendError", .{});
            break;
        };
    }
}
