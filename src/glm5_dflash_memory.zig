//! Conservative live scratch bounds for branch-local MLA views.
const std = @import("std");
pub const limit_bytes: usize = 256 * 1024 * 1024;
/// Trees up to this many rows read the committed latent plus an ancestry overlay.
pub const overlay_rows: usize = @import("glm5_attention_overlay.zig").max_rows;
pub const Plan = struct { branches: usize, common_bytes: usize, per_branch_bytes: usize, live_bytes: usize };
fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.GlmTreeScratchLimit;
}
fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.GlmTreeScratchLimit;
}
fn capacity(needed: usize) !usize {
    return mul((try add(needed, 255)) / 256, 256);
}

pub fn plan(processed: usize, latent_capacity: usize, pool_capacity: usize, width: usize, index_width: usize, heads: usize, rows: usize, bytes: usize) !Plan {
    if (rows == 0 or rows > 16 or width == 0 or index_width == 0 or heads == 0 or (bytes != 2 and bytes != 4) or latent_capacity < processed or pool_capacity < processed / 4) return error.InvalidGlmDraftShape;
    const needed = try add(processed, rows);
    const pool_needed = needed / 4;
    const latent_rows = @max(latent_capacity, try capacity(needed));
    const pool_rows = @max(pool_capacity, try capacity(pool_needed));
    // Growth can retain both concatenation storage and the slice-update copy. Overlay branches
    // write neither shared buffer; their completed pools ride in `compression`.
    const overlay = rows <= overlay_rows;
    const latent_copy = if (overlay) 0 else try mul(try mul(try mul(latent_rows, width), bytes), if (needed > latent_capacity) 2 else 1);
    const pool_copy = if (overlay) 0 else try mul(try mul(try mul(pool_rows, index_width), bytes), if (pool_needed > pool_capacity) 2 else 1);
    const partials = try mul(try mul(try mul(heads, 8), try add(width, 2)), 4);
    const scores = try mul(pool_needed, 12);
    const selected = try mul(try add(try mul(@min(pool_needed, 512), 4), 3), 4);
    const compression = try mul(try mul(try mul(try add(rows, 3), index_width), bytes), 5);
    const gathered = try mul(try mul(rows, width), bytes);
    // Reserve both directions' potential eight-way qvm split-K intermediates.
    const projections = try mul(try mul(try mul(heads, width), bytes), 16);
    var per_branch = try add(latent_copy, pool_copy);
    for ([_]usize{ partials, scores, selected, compression, gathered, projections }) |part| per_branch = try add(per_branch, part);
    const common = try add(try mul(try mul(try mul(try mul(rows, heads), width), bytes), 4), try mul(try mul(try mul(rows, index_width), bytes), 2));
    if (common >= limit_bytes or per_branch > limit_bytes - common) return error.GlmTreeScratchLimit;
    const branches = @min(rows, (limit_bytes - common) / per_branch);
    return .{ .branches = branches, .common_bytes = common, .per_branch_bytes = per_branch, .live_bytes = try add(common, try mul(branches, per_branch)) };
}

test "GLM DFlash overlay trees keep every overlay branch at a full-context reservation" {
    // A request without max_tokens reserves its whole context window (946,179 rows here).
    const full = try plan(7585, 946432, 236608, 512, 128, 64, overlay_rows, 2);
    try std.testing.expectEqual(@as(usize, overlay_rows), full.branches);
    try std.testing.expect(full.live_bytes <= limit_bytes);
    // The B3 batch keeps its three branches beside either arm's native scratch.
    const three = try plan(7585, 946432, 236608, 512, 128, 64, 3, 2);
    try std.testing.expect(three.live_bytes + @import("glm5_attention_decode_batch.zig").scratchLimit() <= limit_bytes);
    try std.testing.expectError(error.GlmTreeScratchLimit, plan(7585, 946432, 236608, 512, 128, 64, overlay_rows + 1, 2));
    // An eight-row chain keeps all its branches beside the wider native scratch.
    const decode = @import("glm5_attention_decode_batch.zig");
    for (5..overlay_rows + 1) |rows| {
        const wide = try plan(7585, 946432, 236608, 512, 128, 64, rows, 2);
        try std.testing.expectEqual(rows, wide.branches);
        try std.testing.expect(wide.live_bytes + decode.scratchLimitFor(rows) <= limit_bytes);
    }
}

test "GLM DFlash overlay branch scratch does not grow with the reservation" {
    for (1..overlay_rows + 1) |rows| {
        const full = try plan(7585, 946432, 236608, 512, 128, 64, rows, 2);
        const tight = try plan(7585, 7680, 1920, 512, 128, 64, rows, 2);
        try std.testing.expectEqual(tight.per_branch_bytes, full.per_branch_bytes);
    }
}

test "GLM DFlash MLA scratch covers cache copies and batches only what fits" {
    const short = try plan(32, 256, 256, 512, 128, 64, 16, 2);
    try std.testing.expectEqual(@as(usize, 16), short.branches);
    const boundary = try plan(65536, 65536, 16384, 512, 128, 64, 16, 2);
    try std.testing.expectEqual(@as(usize, 1), boundary.branches);
    try std.testing.expect(boundary.per_branch_bytes >= 2 * (65792 * 512 * 2 + 16640 * 128 * 2));
    try std.testing.expect(boundary.live_bytes <= limit_bytes);
    const room = try plan(65400, 65536, 16384, 512, 128, 64, 16, 2);
    try std.testing.expect(room.branches > boundary.branches);
    try std.testing.expectError(error.GlmTreeScratchLimit, plan(131072, 131072, 32768, 512, 128, 64, 16, 2));
    try std.testing.expectError(error.InvalidGlmDraftShape, plan(32, 16, 256, 512, 128, 64, 4, 2));
    try std.testing.expectError(error.InvalidGlmDraftShape, plan(32, 256, 256, 512, 128, 64, 17, 2));
}
