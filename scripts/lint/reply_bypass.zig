//! Rule `reply-bypass`: a reply surface that reads or builds a generation's outcome beside the shared steps. The raw
//! `parseToolCalls` skips the chokepoint's thought rule and schema coercion, a `GenerationResult` literal (it names
//! `.prefill_tps`) outside `generation_stream.zig` leaves the fields it forgets at their defaults, and an `Ask`'s
//! `.normalize` written as a literal lets a stream and a non-stream reply of one request settle by different rules. Token scan.
const std = @import("std");
const Ast = std.zig.Ast;
const rule = @import("rule.zig");

pub const name = "reply-bypass";
pub const reason = "a surface calls reply.parseToolCallsForRequest, builds a stream's result with StreamingTokenStream.result and reads an Ask's normalize from generate.payloadRouted";

const surfaces = [_][]const u8{ "src/anthropic.zig", "src/responses_api.zig", "src/completions.zig", "src/server.zig", "src/generation_stream.zig" };

pub fn check(arena: std.mem.Allocator, tree: *const Ast, report: rule.Report) !void {
    for (surfaces) |s| {
        if (std.mem.eql(u8, report.file, s)) break;
    } else return;
    const builds_results = !std.mem.eql(u8, report.file, "src/generation_stream.zig");
    const owners = try rule.owners(arena, tree);
    var t: u32 = 1;
    while (t + 1 < tree.tokens.len) : (t += 1) {
        if (tree.tokenTag(t) != .identifier or rule.inTest(owners.items, t)) continue;
        const key = if (rule.ownerOf(owners.items, t)) |o| o.key else "-";
        const text = tree.tokenSlice(t);
        if (std.mem.eql(u8, text, "parseToolCalls") and tree.tokenTag(t + 1) == .l_paren) {
            try report.add(tree, t, key, "parseToolCalls: read calls through reply.parseToolCallsForRequest", .{});
        } else if (builds_results and std.mem.eql(u8, text, "prefill_tps") and tree.tokenTag(t - 1) == .period and tree.tokenTag(t + 1) == .equal) {
            try report.add(tree, t, key, "a GenerationResult literal: build it with StreamingTokenStream.result", .{});
        } else if (std.mem.eql(u8, text, "normalize") and tree.tokenTag(t - 1) == .period and tree.tokenTag(t + 1) == .equal and t + 2 < tree.tokens.len and tree.tokenTag(t + 2) == .identifier and isBool(tree.tokenSlice(t + 2))) {
            try report.add(tree, t, key, "normalize: an Ask reads it from generate.payloadRouted, never a literal", .{});
        }
    }
}

fn isBool(text: []const u8) bool {
    return std.mem.eql(u8, text, "true") or std.mem.eql(u8, text, "false");
}
