//! GLM-5.3's chat face: the effort words its template takes (`arch/chat.zig` reads it).
const eff = @import("../effort.zig");

pub const model_types = [_][]const u8{"glm5_next"};

/// The template has no thinking-off switch. medium and xhigh are folds: served at high and max, with a notice.
pub const effort: eff.Offer = .{ .table = eff.makeSpec("glm5_next", "GLM", &.{ .low, .high, .max }, &.{}, false, .{ .default = .high, .rows = .{
    .{ .offered = false },
    .{ .offered = true, .sends = .{ .word = .low } },
    .{ .offered = false, .sends = .{ .word = .high }, .fold_to = .high },
    .{ .offered = true, .sends = .{ .word = .high } },
    .{ .offered = false, .sends = .{ .word = .max }, .fold_to = .max },
    .{ .offered = true, .sends = .{ .word = .max } },
} }) };

test "GLM's built-in effort table: no off, medium and xhigh fold to high and max" {
    const testing = @import("std").testing;
    const g = effort.table.table;
    try testing.expect(!g.row(.off).offered);
    try testing.expectEqualDeep(eff.Row{ .offered = false, .sends = .{ .word = .high }, .fold_to = .high }, g.row(.medium));
    try testing.expectEqualDeep(eff.Row{ .offered = false, .sends = .{ .word = .max }, .fold_to = .max }, g.row(.xhigh));
}
