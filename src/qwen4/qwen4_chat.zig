//! Flash-Next's chat face: the effort words its template takes (`arch/chat.zig` reads it).
const std = @import("std");
const eff = @import("../effort.zig");

pub const model_types = [_][]const u8{"qwen4_exp"};

pub const fixtures = @import("qwen4_fixtures.zig");

/// high and max are folds: the template has no such word, so they are served at xhigh, with a notice.
pub const effort: eff.Offer = .{ .table = eff.makeSpec("qwen4_exp", "Qwen", &.{ .low, .medium, .xhigh }, &.{}, true, .{ .default = .medium, .rows = .{
    .{ .offered = true },
    .{ .offered = true, .sends = .{ .word = .low }, .cap = 2048 },
    .{ .offered = true, .sends = .{ .word = .medium }, .cap = 8192 },
    .{ .offered = false, .sends = .{ .word = .xhigh }, .fold_to = .xhigh },
    .{ .offered = true, .sends = .{ .word = .xhigh } },
    .{ .offered = false, .sends = .{ .word = .xhigh }, .fold_to = .xhigh },
} }) };

/// The stock Flash-Next template raises on a system turn that is not first. Replace only that branch so the turn
/// renders in place; any other revision keeps its native contract and the fold.
pub fn lateSystemTemplate(allocator: std.mem.Allocator, tpl: []const u8) !?[]const u8 {
    const source = fixtures.template;
    const content = "render_content(message.content, false, true)|trim";
    const branch =
        \\{%- if message.role == "system" %}
        \\        {%- if not loop.first %}
        \\            {{- raise_exception('System message must be at the beginning.') }}
        \\        {%- endif %}
    ;
    if (!std.mem.eql(u8, tpl, source)) return null;
    if (std.mem.count(u8, tpl, branch) != 1) return null;
    const replacement = try std.mem.concat(allocator, u8, &.{
        "{%- if message.role == \"system\" %}{%- if not loop.first %}{{- '<|im_start|>system\\n' + (",
        content,
        ") + '<|im_end|>\\n' }}{%- endif %}",
    });
    defer allocator.free(replacement);
    return try std.mem.replaceOwned(u8, allocator, tpl, branch, replacement);
}

test "Flash-Next's built-in effort table: medium by default, low and medium capped, high and max fold to xhigh" {
    const testing = @import("std").testing;
    const q = effort.table.table;
    try testing.expectEqual(eff.Effort.medium, q.default);
    try testing.expectEqualDeep(eff.Row{ .offered = false, .sends = .{ .word = .xhigh }, .fold_to = .xhigh }, q.row(.high));
    try testing.expectEqualDeep(eff.Row{ .offered = false, .sends = .{ .word = .xhigh }, .fold_to = .xhigh }, q.row(.max));
    try testing.expectEqualDeep(eff.Row{ .offered = true, .sends = .{ .word = .low }, .cap = 2048 }, q.row(.low));
}
