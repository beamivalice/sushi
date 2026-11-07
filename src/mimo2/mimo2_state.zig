//! MiMo-V2.6-Flash keeps nothing beyond the trunk except, on a streamed load, the expert engine.
const std = @import("std");
const mlx = @import("../mlx.zig");
const model_mod = @import("../model.zig");
const trunk = @import("../trunk.zig");
const Trunk = trunk.Trunk;

/// The model-lifetime state: the trunk's shortlist cell, by pointer. The cell is a heap object with one owner
/// (`TrunkCells.destroy` retires its coarse head), so a copy of the state never duplicates the head.
pub const State = struct {
    shortlist: @FieldType(Trunk, "shortlist") = null,
};

/// What a forward context carries for MiMo: the verify-rows flag and the batched decode's per-slot contexts.
pub const Ctx = struct {
    verify_rows: bool = false,
    batch_rows: ?[]const *trunk.ForwardCtx = null,
    batch_row_offsets: []const u32 = &.{},
};

/// The expert engine of a streamed MiMo load, null for a resident one.
pub fn streamParts(allocator: std.mem.Allocator, config: *const model_mod.ModelConfig, s: mlx.mlx_stream) !?Trunk.ExpertStreamParts {
    if (!config.expert_streaming) return null;
    return try Trunk.initExpertStream(allocator, config, s);
}

test "MiMo state: State holds the trunk's shortlist cell pointer, not a copy of the coarse head" {
    try std.testing.expect(@FieldType(State, "shortlist") == @FieldType(Trunk, "shortlist"));
    try std.testing.expectEqual(@as(usize, 1), std.meta.fieldNames(State).len);
    const state: State = .{};
    try std.testing.expect(state.shortlist == null);
}

test "MiMo state: Ctx holds the types the forward context's MiMo fields hold today" {
    const testing = std.testing;
    const names = comptime std.meta.fieldNames(Ctx);
    inline for (names) |name| try testing.expect(@FieldType(Ctx, name) == @FieldType(trunk.ForwardCtx, name));
    try testing.expectEqual(@as(usize, 3), names.len);
    const ctx: Ctx = .{};
    try testing.expect(!ctx.verify_rows and ctx.batch_rows == null and ctx.batch_row_offsets.len == 0);
}
