//! DeepSeek-V4.1 prefix-cache state. A restore point is the request's whole state at one position
//! (`dsv41_model.captureState`: window rings, open pooling groups, compressed and index rows, DSpark
//! stage windows), stored as an `SSMCheckpoint` whose layers are the model's slots, so the cache's
//! checkpoint list, thinning, merge and byte bill serve it unchanged.
const std = @import("std");
const mlx = @import("mlx.zig");
const transformer = @import("transformer.zig");
const model = @import("dsv41_model.zig");

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

/// Checkpoints one entry or one prefill keeps: each is a full copy of the request's state.
pub const checkpoint_cap: u32 = 4;

pub fn checkpointMax(global: u32) u32 {
    return if (global == 0) checkpoint_cap else @min(global, checkpoint_cap);
}
