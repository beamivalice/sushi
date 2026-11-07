//! Flash-Next's chat face: the effort words its template takes (`arch/chat.zig` reads it).
const eff = @import("../effort.zig");

pub const model_types = [_][]const u8{"qwen4_exp"};

/// high and max are folds: the template has no such word, so they are served at xhigh, with a notice.
pub const effort: eff.Offer = .{ .table = eff.makeSpec("qwen4_exp", "Qwen", &.{ .low, .medium, .xhigh }, &.{}, true, .{ .default = .medium, .rows = .{
    .{ .offered = true },
    .{ .offered = true, .sends = .{ .word = .low }, .cap = 2048 },
    .{ .offered = true, .sends = .{ .word = .medium }, .cap = 8192 },
    .{ .offered = false, .sends = .{ .word = .xhigh }, .fold_to = .xhigh },
    .{ .offered = true, .sends = .{ .word = .xhigh } },
    .{ .offered = false, .sends = .{ .word = .xhigh }, .fold_to = .xhigh },
} }) };

test "Flash-Next's built-in effort table: medium by default, low and medium capped, high and max fold to xhigh" {
    const testing = @import("std").testing;
    const q = effort.table.table;
    try testing.expectEqual(eff.Effort.medium, q.default);
    try testing.expectEqualDeep(eff.Row{ .offered = false, .sends = .{ .word = .xhigh }, .fold_to = .xhigh }, q.row(.high));
    try testing.expectEqualDeep(eff.Row{ .offered = false, .sends = .{ .word = .xhigh }, .fold_to = .xhigh }, q.row(.max));
    try testing.expectEqualDeep(eff.Row{ .offered = true, .sends = .{ .word = .low }, .cap = 2048 }, q.row(.low));
}
