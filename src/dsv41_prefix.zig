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

/// Restore points a finished request keeps in RAM beside its live state: the deepest few, a point's fixed arrays only.
pub const live_point_cap: u32 = 8;

/// One kept restore point: the fixed arrays (`captureState`'s non-row slots) at `pos`.
pub const LivePoint = struct { pos: usize, slots: []model.Slot };

/// The slot's last request, as the next one can roll its live state back into: the points it took and the tokens the
/// state covers. The rows are the live state's own, append-only, so a point costs its fixed arrays alone. Held only
/// while the model's state is the one it was recorded for (`model.Model.state_epoch`).
pub const Live = struct {
    points: std.ArrayList(LivePoint) = .empty,
    tokens: []u32 = &.{},
    has_tools: bool = false,
    vision_key: u64 = 0,
    media_start: ?u32 = null,
    epoch: u64 = 0,
    /// A request finished over it; false while one is running, whose state is moving past the points.
    ready: bool = false,

    pub fn deinit(self: *Live, allocator: std.mem.Allocator) void {
        self.clear(allocator);
        self.points.deinit(allocator);
    }

    pub fn clear(self: *Live, allocator: std.mem.Allocator) void {
        for (self.points.items) |*p| freePoint(allocator, p);
        self.points.clearRetainingCapacity();
        allocator.free(self.tokens);
        self.tokens = &.{};
        self.ready = false;
    }

    fn freePoint(allocator: std.mem.Allocator, p: *LivePoint) void {
        for (p.slots) |*sl| model.freeSlot(sl);
        allocator.free(p.slots);
    }

    /// RAM the record holds beyond the model's own state: the points' arrays and the token ids.
    pub fn bytes(self: *const Live) u64 {
        var n: u64 = @as(u64, self.tokens.len) * @sizeOf(u32);
        for (self.points.items) |p| for (p.slots) |sl| for (sl) |a| {
            n += model.arrayBytes(a);
        };
        return n;
    }

    /// Keeps the fixed arrays of `cp`, a point of the request now running, and drops the shallowest past the cap.
    pub fn keep(self: *Live, allocator: std.mem.Allocator, m: *const model.Model, cp: *const transformer.SSMCheckpoint) !void {
        if (self.ready) self.clear(allocator);
        const n_slots = model.slotCount(m);
        if (cp.layers.len != 2 * n_slots) return error.Dsv41CheckpointMismatch;
        while (self.points.items.len > 0 and self.points.items[self.points.items.len - 1].pos >= cp.pos) {
            var p = self.points.pop().?;
            freePoint(allocator, &p);
        }
        if (self.points.items.len >= live_point_cap) {
            var p = self.points.orderedRemove(0);
            freePoint(allocator, &p);
        }
        const slots = try allocator.alloc(model.Slot, n_slots);
        @memset(slots, model.nil_slot);
        errdefer {
            for (slots) |*sl| model.freeSlot(sl);
            allocator.free(slots);
        }
        for (slots, 0..) |*sl, j| {
            if (j < 2 * m.cfg.layers and j % 2 == 1) continue;
            const lo = cp.layers[2 * j];
            const hi = cp.layers[2 * j + 1];
            for ([4]mlx.mlx_array{ lo.conv_state, lo.ssm_state, hi.conv_state, hi.ssm_state }, 0..) |a, k| {
                if (!model.present(a)) continue;
                sl[k] = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_array_set(&sl[k], a));
            }
        }
        try self.points.append(allocator, .{ .pos = cp.pos, .slots = slots });
    }

    /// The request over the state finished with `tokens` as its history: its points become what the next one may roll back to.
    pub fn publish(self: *Live, allocator: std.mem.Allocator, m: *const model.Model, tokens: []const u32, has_tools: bool, vision_key: u64, media_start: ?u32) void {
        const st = m.dec_state orelse return self.clear(allocator);
        const covered = @min(st.n, tokens.len);
        while (self.points.items.len > 0 and self.points.items[self.points.items.len - 1].pos > covered) {
            var p = self.points.pop().?;
            freePoint(allocator, &p);
        }
        allocator.free(self.tokens);
        self.tokens = &.{};
        if (self.points.items.len == 0) return self.clear(allocator);
        self.tokens = allocator.dupe(u32, tokens[0..covered]) catch return self.clear(allocator);
        self.has_tools = has_tools;
        self.vision_key = vision_key;
        self.media_start = media_start;
        self.epoch = m.state_epoch;
        self.ready = true;
    }

    /// The deepest point a request over `prompt` may roll the live state back to: at most `media_cap` tokens in,
    /// short of the prompt's end, and only while the state's rows outside the request's own bill would still fit it.
    pub fn pick(self: *const Live, m: *const model.Model, prompt: []const u32, has_tools: bool, media_cap: usize) ?usize {
        if (!self.ready or self.epoch != m.state_epoch or self.has_tools != has_tools) return null;
        const st = m.dec_state orelse return null;
        var shared: usize = 0;
        const max_shared = @min(self.tokens.len, prompt.len);
        while (shared < max_shared and self.tokens[shared] == prompt[shared]) shared += 1;
        const limit = @min(@min(shared, st.n), @min(media_cap, prompt.len -| 1));
        // The request bills its own prompt's state: rows kept past that are memory nobody paid for.
        if (!model.liveRowsFit(m, prompt.len)) return null;
        var i = self.points.items.len;
        while (i > 0) {
            i -= 1;
            if (self.points.items[i].pos <= limit) return i;
        }
        return null;
    }

    /// Rolls the live state back to point `idx`; the points past it go, and the record is open until its request publishes.
    pub fn rollback(self: *Live, allocator: std.mem.Allocator, m: *model.Model, idx: usize, prompt: []const u32) !void {
        const p = self.points.items[idx];
        try model.rollbackState(m, p.slots, p.pos, prompt);
        while (self.points.items.len > idx + 1) {
            var q = self.points.pop().?;
            freePoint(allocator, &q);
        }
        allocator.free(self.tokens);
        self.tokens = &.{};
        self.ready = false;
    }
};

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
    return if (dspark) "dsv41-state-v2-dspark" else "dsv41-state-v2";
}

/// Tokens of compressed rows and index keys one `g<n>` segment of an SSD entry holds.
pub const row_span: u32 = 8192;

/// Restore points one SSD entry keeps: the `row_span` grid out to a million tokens, with room for the prompt-end
/// and thought points of earlier turns. A point holds only its fixed arrays, so the cap bills a few MB apiece.
pub const disk_checkpoint_cap: u32 = 160;

/// SSD bytes one cached token adds to an entry: its rows once, and its share of the fixed arrays of the restore
/// point on each `row_span`.
pub fn diskBytesPerToken(c: *const @import("dsv41_config.zig").Config, dspark: bool) u64 {
    return c.stateBytesPerToken() +| (c.stateFixedBytes(dspark) +| row_span - 1) / row_span;
}

/// The SSD tier's split of this model's restore points (`kv_disk_cache.GrownSpec`): per snapshot layer the tokens
/// per row of the rows it holds, 0 for a layer of fixed arrays. Caller frees.
pub fn rowDivisors(allocator: std.mem.Allocator, c: *const @import("dsv41_config.zig").Config, dspark: bool) ![]u32 {
    const divs = try allocator.alloc(u32, 2 * model.slotCountFor(c, dspark));
    @memset(divs, 0);
    for (0..c.layers) |li| {
        const r = model.rowRatios(c, li);
        divs[4 * li + 2] = r[0];
        divs[4 * li + 3] = r[1];
    }
    return divs;
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

/// Checkpoints one RAM entry or one prefill keeps in memory: each is a full copy of the request's state.
pub const checkpoint_cap: u32 = 4;

pub fn checkpointMax(global: u32) u32 {
    return if (global == 0) checkpoint_cap else @min(global, checkpoint_cap);
}
