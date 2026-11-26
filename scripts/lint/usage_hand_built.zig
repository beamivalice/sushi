//! Rule `usage-hand-built`: an OpenAI `usage` object spelled inside a string literal, instead of built by
//! `generation_stream.formatChatUsage`. The Responses and Messages dialects keep different keys, so only the
//! `prompt_tokens` shape is this rule's. Token scan, no Sema.
const std = @import("std");
const Ast = std.zig.Ast;
const rule = @import("rule.zig");

pub const name = "usage-hand-built";
pub const reason = "an OpenAI usage object spelled by hand: call formatChatUsage";

pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    const owners = try rule.owners(arena, tree);
    var t: u32 = 0;
    while (t < tree.tokens.len) : (t += 1) {
        const tag = tree.tokenTag(t);
        if (tag != .string_literal and tag != .multiline_string_literal_line) continue;
        const text = tree.tokenSlice(t);
        const usage = std.mem.indexOf(u8, text, "usage") orelse continue;
        const rest = text[usage..];
        const prompt = std.mem.indexOf(u8, rest, "prompt_tokens") orelse continue;
        // A usage object names its first count a few characters after `usage`; a whole request body does not.
        if (prompt > 48) continue;
        if (rule.inTest(owners.items, t)) continue;
        const owner = rule.ownerOf(owners.items, t) orelse continue;
        if (std.mem.eql(u8, owner.key, "formatChatUsage") and std.mem.endsWith(u8, report.file, "generation_stream.zig")) continue;
        try report.add(tree, t, owner.key, "a usage object spelled by hand: call formatChatUsage", .{});
    }
}
