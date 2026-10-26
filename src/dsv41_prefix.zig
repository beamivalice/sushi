//! DeepSeek-V4.1 prefix-cache state. A restore point is the request's whole state at one position
//! (`dsv41_model.captureState`: window rings, open pooling groups, compressed and index rows, DSpark
//! stage windows), stored as an `SSMCheckpoint` whose layers are the model's slots, so the cache's
//! checkpoint list, thinning, merge and byte bill serve it unchanged.
const std = @import("std");
const mlx = @import("mlx.zig");
const transformer = @import("transformer.zig");
const model = @import("dsv41_model.zig");
const expert_usage = @import("expert_usage.zig");
const testing = std.testing;

/// Snapshot layers a restore point holds: each model slot's four arrays ride two layers' conv and ssm fields
/// (the aux and pooled fields mean a QSA history to the cache, so they stay empty). A slot array the
/// state does not hold is a one-byte placeholder, which keeps every field a real array for the file format.
pub fn snapshotLayers(m: *const model.Model) usize {
    return 2 * model.slotCount(m);
}

fn placeholder() mlx.mlx_array {
    const zero: [1]u8 = .{0};
    return mlx.mlx_array_new_data(&zero, &[_]c_int{1}, 1, .uint8);
}

fn orPlaceholder(a: mlx.mlx_array) mlx.mlx_array {
    return if (a.ctx != null) a else placeholder();
}

/// The model's state at `pos`, which must be where the request stands, as an owned checkpoint.
pub fn capture(allocator: std.mem.Allocator, m: *model.Model, pos: usize) !transformer.SSMCheckpoint {
    return captureWith(allocator, m, pos, false);
}

/// `capture` whose rows share the request's own (`model.captureStateShared`): for a restore point the SSD tier stages
/// and releases before the next forward, so the prefill holds no copy of the state.
pub fn captureShared(allocator: std.mem.Allocator, m: *model.Model, pos: usize) !transformer.SSMCheckpoint {
    return captureWith(allocator, m, pos, true);
}

fn captureWith(allocator: std.mem.Allocator, m: *model.Model, pos: usize, share: bool) !transformer.SSMCheckpoint {
    if ((try model.ensureState(m)).n != pos) return error.Dsv41CachePositionMismatch;
    const slots = try allocator.alloc(model.Slot, model.slotCount(m));
    defer allocator.free(slots);
    @memset(slots, model.nil_slot);
    try if (share) model.captureStateShared(m, slots) else model.captureState(m, slots);
    const layers = allocator.alloc(transformer.SSMCacheEntrySnapshot, 2 * slots.len) catch |err| {
        for (slots) |*sl| model.freeSlot(sl);
        return err;
    };
    for (slots, 0..) |sl, i| for (0..2) |h| {
        layers[2 * i + h] = .{
            .conv_state = orPlaceholder(sl[2 * h]),
            .ssm_state = orPlaceholder(sl[2 * h + 1]),
            .initialized = true,
        };
    };
    return .{ .pos = pos, .layers = layers };
}

/// Makes the request's state `cp` over the first `cp.pos` of `ids`; the checkpoint's arrays are shared.
pub fn restore(m: *model.Model, cp: *const transformer.SSMCheckpoint, ids: []const u32) !void {
    if (cp.layers.len != snapshotLayers(m)) return error.Dsv41CheckpointMismatch;
    const slots = try m.gpa.alloc(model.Slot, cp.layers.len / 2);
    defer m.gpa.free(slots);
    for (slots, 0..) |*sl, i| {
        const lo = cp.layers[2 * i];
        const hi = cp.layers[2 * i + 1];
        sl.* = .{ lo.conv_state, lo.ssm_state, hi.conv_state, hi.ssm_state };
    }
    try model.restoreState(m, slots, cp.pos, ids);
}

/// The SSD tier indexes a prefix by its KV rows, and this cache holds none: layer 0 carries one byte per token
/// so the tier's chunk files, sizes and scan cover the entry. Done before a commit; the forward never reads it.
pub fn markRows(cache: *transformer.KVCache) !void {
    const n = cache.step;
    if (n == 0 or cache.entries.len == 0) return;
    const e = &cache.entries[0];
    transformer.resetKVEntry(e);
    // A slot's cache takes the engine's KV scheme whatever the pack says, and a quantized entry owns six buffers.
    const slots = [_]*mlx.mlx_array{ &e.keys, &e.values, &e.keys_scales, &e.keys_biases, &e.values_scales, &e.values_biases };
    for (slots[0 .. if (cache.config.scheme == .off) 2 else 6]) |slot| {
        _ = mlx.mlx_array_free(slot.*);
        slot.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_zeros(slot, &[_]c_int{ 1, 1, @intCast(n), 1 }, 4, .uint8, mlx.gpuStream()));
        try mlx.check(mlx.mlx_array_eval(slot.*));
    }
    e.offset = n;
    e.initialized = true;
}

/// Names the stored layout for the SSD cache namespace: a restore point written under another layout, or
/// with a different set of slots, is never read.
pub fn namespaceTag(dspark: bool) []const u8 {
    return if (dspark) "dsv41-state-v1-dspark" else "dsv41-state-v1";
}

/// The SSD namespace of one boot: `namespaceTag` plus what changes the numbers a stored state was computed
/// under. `drop` is the ranked list in use; a non-zero `pick_tolerance` stores states built on cache-dependent
/// picks, so no exact boot reads them. A boot with neither keeps `namespaceTag`.
pub fn layoutKey(allocator: std.mem.Allocator, dspark: bool, drop: ?*const expert_usage.Drop, pick_tolerance: f32) ![]u8 {
    var key = std.ArrayList(u8).empty;
    errdefer key.deinit(allocator);
    try key.appendSlice(allocator, namespaceTag(dspark));
    if (drop) |d| try key.print(allocator, "-drop{d}-{x:0>16}", .{ d.n, d.digest() });
    if (pick_tolerance > 0) try key.print(allocator, "-pick{x:0>8}", .{@as(u32, @bitCast(pick_tolerance))});
    return key.toOwnedSlice(allocator);
}

test "layoutKey: a boot that drops other experts or picks leniently has its own namespace" {
    const a = testing.allocator;
    const full = try layoutKey(a, false, null, 0);
    defer a.free(full);
    try testing.expectEqualStrings(namespaceTag(false), full);

    var d8 = try expert_usage.fixedDrop(a, 1, &.{ &.{ 0, 1, 2, 3, 4, 5, 6, 7 }, &.{ 8, 9, 10, 11, 12, 13, 14, 15 } });
    defer d8.deinit();
    var d8_same = try expert_usage.fixedDrop(a, 1, &.{ &.{ 0, 1, 2, 3, 4, 5, 6, 7 }, &.{ 8, 9, 10, 11, 12, 13, 14, 15 } });
    defer d8_same.deinit();
    // Same count, another list: still another namespace.
    var d8_other = try expert_usage.fixedDrop(a, 1, &.{ &.{ 0, 1, 2, 3, 4, 5, 6, 7 }, &.{ 9, 8, 10, 11, 12, 13, 14, 16 } });
    defer d8_other.deinit();
    var d16 = try expert_usage.fixedDrop(a, 1, &.{ &.{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 }, &.{ 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 } });
    defer d16.deinit();

    const k8 = try layoutKey(a, false, &d8, 0);
    defer a.free(k8);
    const k8_same = try layoutKey(a, false, &d8_same, 0);
    defer a.free(k8_same);
    const k8_other = try layoutKey(a, false, &d8_other, 0);
    defer a.free(k8_other);
    const k16 = try layoutKey(a, false, &d16, 0);
    defer a.free(k16);
    const k8_dspark = try layoutKey(a, true, &d8, 0);
    defer a.free(k8_dspark);
    const k_lossy = try layoutKey(a, false, null, 0.3);
    defer a.free(k_lossy);
    const k_lossier = try layoutKey(a, false, null, 0.4);
    defer a.free(k_lossier);
    const k8_lossy = try layoutKey(a, false, &d8, 0.3);
    defer a.free(k8_lossy);

    try testing.expectEqualStrings(k8, k8_same);
    const distinct = [_][]const u8{ full, k8, k8_other, k16, k8_dspark, k_lossy, k_lossier, k8_lossy };
    for (distinct, 0..) |x, i| for (distinct[i + 1 ..]) |y| try testing.expect(!std.mem.eql(u8, x, y));
}

/// Thought tokens under which the prompt-end snapshot's re-prefill is cheap enough to skip a decode restore point.
pub const min_thought_tokens: usize = 512;

/// Whether the point behind a finished thought can match the next prompt. The server delivers the thought with
/// "\n" and " " trimmed off both ends and the template re-renders it as delivered, so a thought whose first or last
/// token carries one diverges from the generated tokens there.
pub fn thoughtEdgesSurviveTrim(first_token: []const u8, last_token: []const u8) bool {
    const trimmed = "\n ";
    if (first_token.len > 0 and std.mem.indexOfScalar(u8, trimmed, first_token[0]) != null) return false;
    if (last_token.len > 0 and std.mem.indexOfScalar(u8, trimmed, last_token[last_token.len - 1]) != null) return false;
    return true;
}

test "thoughtEdgesSurviveTrim: a thought the server trims leaves no point" {
    try std.testing.expect(thoughtEdgesSurviveTrim("The", "."));
    try std.testing.expect(thoughtEdgesSurviveTrim("<", "tools"));
    try std.testing.expect(!thoughtEdgesSurviveTrim("\nThe", "."));
    try std.testing.expect(!thoughtEdgesSurviveTrim(" The", "."));
    try std.testing.expect(!thoughtEdgesSurviveTrim("The", ".\n"));
    try std.testing.expect(!thoughtEdgesSurviveTrim("The", " "));
    // A tab is not trimmed.
    try std.testing.expect(thoughtEdgesSurviveTrim("\tThe", ".\t"));
}

/// Checkpoints one entry or one prefill keeps: each is a full copy of the request's state.
pub const checkpoint_cap: u32 = 4;

pub fn checkpointMax(global: u32) u32 {
    return if (global == 0) checkpoint_cap else @min(global, checkpoint_cap);
}
