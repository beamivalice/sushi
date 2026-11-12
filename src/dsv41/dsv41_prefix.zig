//! DeepSeek-V4.1 prefix-cache state. A restore point is the request's whole state at one position
//! (`dsv41_model.captureState`: window rings, open pooling groups, compressed and index rows, DSpark
//! stage windows), stored as an `SSMCheckpoint` whose layers are the model's slots, so the cache's
//! checkpoint list, thinning, merge and byte bill serve it unchanged.
const std = @import("std");
const arch_handle = @import("../arch/handle.zig");
const mlx = @import("../mlx.zig");
const transformer = @import("../trunk.zig");
const model = @import("dsv41_model.zig");
const expert_usage = @import("../expert_usage.zig");
const testing = std.testing;
const KVCache = transformer.KVCache;
const SSMCheckpoint = transformer.SSMCheckpoint;
const dsv41_model = @import("dsv41_model.zig");
const io_util = @import("../io_util.zig");
const kv_disk_cache = @import("../kv_disk_cache.zig");
const log = @import("../log.zig");
const pc = @import("../prefix_cache.zig");
const kv_quant = @import("../kv_quant.zig");
const prefix_arch = @import("../arch/prefix.zig");
const restore_dump = @import("../restore_dump.zig");
const dsv41_state = @import("dsv41_state.zig");
const cfg_mod = @import("dsv41_config.zig");

/// Tokens a prompt's chunk loop holds back for its last forward: those after the last assistant opener (`opener`
/// its id), so the prompt-end snapshot sits before that opener. The pinned template renders everything before it the
/// same once the turn is history, where the think tag after it turns. Null (the shared count) when the last
/// forward would then be wider than a decode-width pass or the prompt's end holds no opener.
pub fn snapshotBackoff(opener: u32, prompt_ids: []const u32) ?usize {
    const n = prompt_ids.len;
    var i = n;
    while (i > 0 and n - i < cfg_mod.narrow_rows) {
        i -= 1;
        if (prompt_ids[i] == opener) return n - 1 - i;
    }
    return null;
}

/// The model whose request state a lookup restores into.
pub const State = *dsv41_model.Model;

/// What the hot cache keeps beside its entries: the slot's last request, which its state can be rolled back to.
pub const CacheState = Live;

pub fn liveOf(hc: *pc.HotPrefixCache) *Live {
    if (comptime !prefix_arch.has("dsv41")) unreachable;
    return &hc.arch.dsv41;
}

/// Bytes the cache's live record of the slot's last request holds.
pub fn liveBytes(hc: *pc.HotPrefixCache) u64 {
    return liveOf(hc).bytes();
}

pub fn reset(m: State) void {
    dsv41_model.resetState(m);
}

/// The model's state at its position as a restore tape digests it: the arrays a restore point holds.
pub fn stateDigest(m: State, h: *std.crypto.hash.sha2.Sha256, _: mlx.mlx_stream) !void {
    const st = try model.ensureState(m);
    h.update(std.mem.asBytes(&std.mem.nativeToLittle(u64, st.n)));
    const slots = try m.gpa.alloc(model.Slot, model.slotCount(m));
    defer m.gpa.free(slots);
    @memset(slots, model.nil_slot);
    try model.captureStateShared(m, slots);
    defer for (slots) |*sl| model.freeSlot(sl);
    for (slots) |sl| for (sl) |a| try restore_dump.feedArray(h, a, m.s);
}

/// The request state is a checkpoint: without checkpoints nothing restores.
pub fn checkpointsRequired(config: *const @import("../model.zig").ModelConfig) bool {
    _ = config;
    return true;
}

pub fn handover(self: *pc.HotPrefixCache, m: State, _: mlx.mlx_stream) void {
    releaseLive(self, m, "handover");
}

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

/// `a`, or for an absent slot the checkpoint's one placeholder, shared: `first` is the handle that made it.
fn orPlaceholder(a: mlx.mlx_array, first: *mlx.mlx_array) !mlx.mlx_array {
    if (a.ctx != null) return a;
    if (first.ctx == null) {
        first.* = placeholder();
        if (first.ctx == null) return error.Dsv41CheckpointPlaceholder;
        return first.*;
    }
    var again = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&again, first.*));
    return again;
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
    // The placeholder of the absent slots is made where the state is, and the pool is told of it like a copy.
    const host = mlx.engine.enter(model.stateStream(m));
    defer host.end();
    mlx.engine.room(&.{1});
    return fromSlots(allocator, slots, pos);
}

/// The checkpoint that takes ownership of the arrays in `slots`, which are freed on a failure.
fn fromSlots(allocator: std.mem.Allocator, slots: []model.Slot, pos: usize) !transformer.SSMCheckpoint {
    const layers = allocator.alloc(transformer.SSMCacheEntrySnapshot, 2 * slots.len) catch |err| {
        for (slots) |*sl| model.freeSlot(sl);
        return err;
    };
    // Filled in order, so a failure frees the snapshots made and the slots not yet taken.
    var first: mlx.mlx_array = .{ .ctx = null };
    for (0..2 * slots.len) |at| {
        const sl = &slots[at / 2];
        const h = at % 2;
        errdefer {
            for (layers[0..at]) |l| {
                _ = mlx.mlx_array_free(l.conv_state);
                _ = mlx.mlx_array_free(l.ssm_state);
            }
            for (slots[at / 2 ..]) |*rest| model.freeSlot(rest);
            allocator.free(layers);
        }
        const conv = try orPlaceholder(sl[2 * h], &first);
        sl[2 * h] = .{ .ctx = null };
        errdefer _ = mlx.mlx_array_free(conv);
        const ssm = try orPlaceholder(sl[2 * h + 1], &first);
        sl[2 * h + 1] = .{ .ctx = null };
        layers[at] = .{ .conv_state = conv, .ssm_state = ssm, .initialized = true };
    }
    return .{ .pos = pos, .layers = layers };
}

/// A kept point of the live state as the checkpoint `captureShared` took there, its rows views of the live ones.
fn captureLive(allocator: std.mem.Allocator, m: *model.Model, p: *const LivePoint) !transformer.SSMCheckpoint {
    const slots = try allocator.alloc(model.Slot, p.slots.len);
    defer allocator.free(slots);
    @memset(slots, model.nil_slot);
    try model.captureLiveShared(m, p.slots, p.pos, slots);
    const host = mlx.engine.enter(model.stateStream(m));
    defer host.end();
    mlx.engine.room(&.{1});
    return fromSlots(allocator, slots, p.pos);
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
/// `flushed`: the point's state is on SSD already (written through, or restored from there).
pub const LivePoint = struct { pos: usize, slots: []model.Slot, flushed: bool = false };

/// The slot's last request, as the next one can roll its live state back into: the points it took and the tokens the
/// state covers. The rows are the live state's own, append-only, so a point costs its fixed arrays alone. Held only
/// while the model's state is the one it was recorded for (`model.Model.state_epoch`).
pub const Live = struct {
    points: std.ArrayList(LivePoint) = .empty,
    tokens: []u32 = &.{},
    has_tools: bool = false,
    vision_key: u64 = 0,
    media_start: ?u32 = null,
    /// The KV scheme of the slot cache the record was published from: the SSD tier keys its entries on it.
    quant: kv_quant.KVQuantConfig = .dense,
    epoch: u64 = 0,
    /// Rows below this position are the request's restore or rollback: taken from a stored entry, not computed by it.
    base: usize = 0,
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
        self.base = 0;
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
    pub fn keep(self: *Live, allocator: std.mem.Allocator, m: *const model.Model, cp: *const transformer.SSMCheckpoint, flushed: bool) !void {
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
        try self.points.append(allocator, .{ .pos = cp.pos, .slots = slots, .flushed = flushed });
    }

    /// `keep` for the point a request restored from: its rows below it came from a stored entry.
    pub fn keepRestored(self: *Live, allocator: std.mem.Allocator, m: *const model.Model, cp: *const transformer.SSMCheckpoint, flushed: bool) !void {
        try self.keep(allocator, m, cp, flushed);
        self.base = cp.pos;
    }

    fn markFlushed(self: *Live, pos: usize) void {
        for (self.points.items) |*p| {
            if (p.pos == pos) p.flushed = true;
        }
    }

    /// Whether a point of the record is still only here, not on SSD.
    pub fn unflushed(self: *const Live) bool {
        if (!self.ready) return false;
        for (self.points.items) |p| {
            if (!p.flushed) return true;
        }
        return false;
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
        self.base = p.pos;
    }
};

/// The SSD tier indexes a prefix by its KV rows, and this cache holds none: layer 0 carries one byte per token
/// so the tier's chunk files, sizes and scan cover the entry. Done before a commit; the forward never reads it.
pub fn markRows(cache: *transformer.KVCache, s: mlx.mlx_stream) !void {
    const n = cache.step;
    if (n == 0 or cache.entries.len == 0) return;
    const e = &cache.entries[0];
    transformer.resetKVEntry(e);
    // A slot's cache takes the engine's KV scheme whatever the pack says, and a quantized entry owns six buffers.
    const slots = [_]*mlx.mlx_array{ &e.keys, &e.values, &e.keys_scales, &e.keys_biases, &e.values_scales, &e.values_biases };
    for (slots[0 .. if (cache.config.scheme == .off) 2 else 6]) |slot| {
        _ = mlx.mlx_array_free(slot.*);
        slot.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_zeros(slot, &[_]c_int{ 1, 1, @intCast(n), 1 }, 4, .uint8, s));
        try mlx.check(mlx.mlx_array_eval(slot.*));
    }
    e.offset = n;
    e.initialized = true;
}

/// Names the stored layout for the SSD cache namespace: a restore point written under another layout, or
/// with a different set of slots, is never read.
pub fn namespaceTag(dspark: bool) []const u8 {
    return if (dspark) "dsv41-state-v6-dspark" else "dsv41-state-v6";
}

/// Tokens of compressed rows and index keys one `g<n>` segment of an SSD entry holds.
pub const row_span: u32 = 8192;

/// Restore points one SSD entry keeps: the `row_span` grid out to a million tokens, with room for the prompt-end
/// and thought points of earlier turns. A point holds only its fixed arrays, so the cap bills a few MB apiece.
pub const disk_checkpoint_cap: u32 = 160;

pub fn restorePointModel(xfm: *const transformer.Trunk) bool {
    return dsv41_state.modelOf(xfm) != null;
}

pub fn captureCheckpoint(allocator: std.mem.Allocator, xfm: *transformer.Trunk, pos: usize, to_ssd: bool) !?SSMCheckpoint {
    const m = dsv41_state.modelOf(xfm) orelse return null;
    return if (to_ssd and pos >= kv_disk_cache.MIN_PERSIST_TOKENS) try captureShared(allocator, m, pos) else try capture(allocator, m, pos);
}

/// A Cooler load's state is the engine's, and so is every array the caches make of it.
pub fn arraysOn(config: *const @import("../model.zig").ModelConfig) ?mlx.mlx_stream {
    const c = config.dsv41 orelse return null;
    return if (@import("dsv41_borrow.zig").ownDecode(&c)) mlx.engine.stream() else null;
}

/// Gives the SSD tier this model's split of its restore points.
pub fn setDiskGrownSpec(allocator: std.mem.Allocator, config: *const @import("../model.zig").ModelConfig, disk: *kv_disk_cache.DiskTier) !void {
    const c = config.dsv41 orelse return;
    const divs = try rowDivisors(allocator, &c, config.dsparkOn());
    defer allocator.free(divs);
    try disk.setGrownSpec(row_span, divs, c.stateBytesPerToken());
}

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

/// The SSD namespace of a loaded DeepSeek model: `layoutKey` over the boot's arms and the expert drop list it names.
pub fn diskLayoutKey(allocator: std.mem.Allocator, config: *const @import("../model.zig").ModelConfig, model_dir: []const u8, xfm: *transformer.Trunk) !?[]u8 {
    var drop: ?expert_usage.Drop = if (config.drop_expert == 0) null else try expert_usage.load(allocator, model_dir, config.expertDropRequest());
    defer if (drop) |*d| d.deinit();
    const m = xfm.arch_state.dsv41.model.?;
    return try layoutKey(allocator, config.dsparkOn(), m.attn_nax, m.fused, m.dspark_exact, if (drop) |*d| d else null, @import("../expert_stream.zig").pick_tolerance, m.hc_mix_nax);
}

/// The SSD namespace of one boot: `namespaceTag` plus what changes the numbers a stored state was computed
/// under. `drop` is the ranked list in use; a non-zero `pick_tolerance` stores states built on cache-dependent
/// picks, so no exact boot reads them. `attn_nax` is the attention arm that serves: the SIMD arm sums in another
/// order, so its states are no NAX boot's, and so does the ALU hyper-connection mix (`hc_mix_nax` off). A composed boot (`fused` off) runs the op chains, and DSpark rounds that
/// sum by width (`dspark_exact` off) change the verify rows; neither writes into the served namespace. A boot with
/// none of these keeps `namespaceTag`.
pub fn layoutKey(allocator: std.mem.Allocator, dspark: bool, attn_nax: bool, fused: bool, dspark_exact: bool, drop: ?*const expert_usage.Drop, pick_tolerance: f32, hc_mix_nax: bool) ![]u8 {
    var key = std.ArrayList(u8).empty;
    errdefer key.deinit(allocator);
    try key.appendSlice(allocator, namespaceTag(dspark));
    if (!attn_nax) try key.appendSlice(allocator, "-simd");
    if (!hc_mix_nax) try key.appendSlice(allocator, "-hcalu");
    if (!fused) try key.appendSlice(allocator, "-composed");
    if (dspark and !dspark_exact) try key.appendSlice(allocator, "-dsparkinexact");
    if (drop) |d| try key.print(allocator, "-drop{d}-{x:0>16}", .{ d.n, d.digest() });
    if (pick_tolerance > 0) try key.print(allocator, "-pick{x:0>8}", .{@as(u32, @bitCast(pick_tolerance))});
    return key.toOwnedSlice(allocator);
}

/// A DeepSeek-V4.1 request restores its whole state from the entry's checkpoint into the model.
pub fn lookupAndRestoreDsv41(
    self: *pc.HotPrefixCache,
    target_cache: *KVCache,
    target_moe_seq_offset: *usize,
    dsv: *dsv41_model.Model,
    s: mlx.mlx_stream,
    prompt_ids: []const u32,
    has_tools: bool,
    vision_key: u64,
    media_start: ?usize,
    media_chain: []const pc.MediaSpan,
    slot_id: ?usize,
    skip: bool,
) !pc.LookupResult {
    return self.lookupImpl(target_cache, target_moe_seq_offset, prefix_arch.recurrent("dsv41", dsv), s, prompt_ids, has_tools, vision_key, media_start, media_chain, null, null, slot_id, skip);
}

/// DeepSeek: the live state rolled back in place to the deepest point of the slot's last request that this prompt
/// shares, when no stored point reaches further: no read, no copy. Null means the state is lost to this request:
/// the record was flushed to SSD and cleared.
pub fn rollbackLive(
    self: *pc.HotPrefixCache,
    m: State,
    target_cache: *KVCache,
    target_moe_seq_offset: *usize,
    s: mlx.mlx_stream,
    prompt_ids: []const u32,
    has_tools: bool,
    media_limits: kv_disk_cache.MediaLimits,
    ram_match: ?pc.RamMatch,
) ?pc.LookupResult {
    if (rollbackLiveInPlace(self, m, target_cache, target_moe_seq_offset, s, prompt_ids, has_tools, media_limits, ram_match)) |res| return res;
    releaseLive(self, m, "handover");
    return null;
}

fn rollbackLiveInPlace(
    self: *pc.HotPrefixCache,
    m: State,
    target_cache: *KVCache,
    target_moe_seq_offset: *usize,
    s: mlx.mlx_stream,
    prompt_ids: []const u32,
    has_tools: bool,
    media_limits: kv_disk_cache.MediaLimits,
    ram_match: ?pc.RamMatch,
) ?pc.LookupResult {
    const live = liveOf(self);
    const cap: usize = if (media_limits.rule) |r| (r.limitFor(r.ctx, live.vision_key, live.media_start) orelse 0) else kv_disk_cache.MediaLimits.UNBOUNDED;
    const idx = live.pick(m, prompt_ids, has_tools, cap) orelse return null;
    const pos = live.points.items[idx].pos;
    var stored: usize = if (ram_match) |rm| blk: {
        const cps = self.entries.items[rm.idx].ssm_checkpoints orelse break :blk 0;
        break :blk (pc.HotPrefixCache.highestCheckpointAtOrBelow(cps, rm.shared) orelse break :blk 0).pos;
    } else 0;
    if (self.disk) |*d| if (d.bestHybridMatch(prompt_ids, has_tools, target_cache.config, @intCast(prompt_ids.len), media_limits)) |hm| {
        stored = @max(stored, hm.cp);
    };
    if (stored > pos) return null;
    const handled = Handled.begin();
    live.rollback(self.allocator, m, idx, prompt_ids) catch |err| {
        log.warn("  [dsv41-prefix] live state not rolled back: {s}; a stored point serves\n", .{@errorName(err)});
        live.clear(self.allocator);
        handled.drop();
        return null;
    };
    target_cache.truncate(0, s) catch handled.drop();
    target_cache.step = pos;
    target_moe_seq_offset.* = pos;
    self.last_restored_live = true;
    return .{ .matched = pos, .full_match = false, .slot_owned = true };
}

/// DeepSeek's SSD arm: the stored point under the match that beats the RAM match restores into the model.
pub fn restoreDisk(
    self: *pc.HotPrefixCache,
    d: *kv_disk_cache.DiskTier,
    m: State,
    target_cache: *KVCache,
    ram_eff: usize,
    prompt_ids: []const u32,
    has_tools: bool,
    media_limits: kv_disk_cache.MediaLimits,
    dflash_target: ?pc.DflashTarget,
    _: mlx.mlx_stream,
) ?pc.LookupResult {
    _ = dflash_target;
    const s = model.stateStream(m);
    const hm = d.bestHybridMatch(prompt_ids, has_tools, target_cache.config, @intCast(prompt_ids.len), media_limits) orelse return null;
    if (@as(usize, hm.cp) < ram_eff + kv_disk_cache.MIN_DISK_ADVANTAGE_TOKENS) return null;
    const sw = io_util.Stopwatch.init(d.io);
    // The arrays read back are state: they take their own room in the engine's pool, whatever cap the last forward left.
    const cap = mlx.engine.uncapped();
    defer mlx.engine.capped(cap);
    const handled = Handled.begin();
    const refusals = mlx.memoryRefusals();
    var cp = d.restoreCheckpointInto(target_cache, hm.idx, hm.cp, snapshotLayers(m), s) catch |err| {
        log.warn("  [disk-cache] DeepSeek restore failed: {s} — falling back to RAM/cold path\n", .{@errorName(err)});
        pc.HotPrefixCache.poisonUnreadable(d, hm.idx, err, refusals);
        target_cache.truncate(0, s) catch {};
        handled.drop();
        return null;
    };
    defer cp.deinit(d.allocator);
    restore(m, &cp, prompt_ids) catch |err| {
        log.warn("  [disk-cache] DeepSeek state not installed: {s} — falling back to RAM/cold path\n", .{@errorName(err)});
        target_cache.truncate(0, s) catch {};
        reset(m);
        handled.drop();
        return null;
    };
    self.last_restored_disk_id = d.entries.items[hm.idx].id;
    liveOf(self).keepRestored(self.allocator, m, &cp, true) catch handled.drop();
    log.info("  [disk-cache] restored {d}/{d} tokens from SSD in {d}ms (dsv41@{d})\n", .{ hm.cp, prompt_ids.len, sw.read() / std.time.ns_per_ms, hm.cp });
    return .{ .matched = hm.cp, .full_match = false, .slot_owned = true };
}

/// Installs the entry's checkpoint under `restore_cap` into the model; 0 when none serves.
pub fn restoreEntry(self: *pc.HotPrefixCache, m: State, e: *pc.Entry, restore_cap: usize, prompt_ids: []const u32) !usize {
    const cps = e.ssm_checkpoints orelse return 0;
    const cp = pc.HotPrefixCache.highestCheckpointAtOrBelow(cps, restore_cap) orelse return 0;
    try restore(m, cp, prompt_ids);
    const handled = Handled.begin();
    liveOf(self).keepRestored(self.allocator, m, cp, false) catch handled.drop();
    return cp.pos;
}

/// A step this file handles itself when it fails (a log line and a fallback): what the failure latched goes with it,
/// so the next request, whose own first check would read the latch, does not meet it.
const Handled = struct {
    had: bool,

    fn begin() Handled {
        return .{ .had = mlx.errorPending() };
    }
    fn drop(self: Handled) void {
        mlx.dropLatchedErrorUnless(self.had);
    }
};

/// DeepSeek's write-through: the request's restore points on the tier's row-segment grid go to the SSD tier and every
/// point leaves RAM at once, the newest too: the writer has copied each into its own bounded host buffers when the
/// commit returns. A point off the grid is kept live only (`Live`), whatever the tier does. Below the tier's floor nothing
/// is written: the points stay for the commit and are kept live, so the next turn rolls back to them as above it.
/// `held` names the phase that waited on the writer, for the log.
pub fn writeThroughDsv41(self: *pc.HotPrefixCache, m: *const dsv41_model.Model, cache: *transformer.KVCache, prompt: []const u32, has_tools: bool, media: pc.MediaStamp, cps: []SSMCheckpoint, held: []const u8) void {
    const s = dsv41_model.stateStream(m);
    if (self.disk == null) return;
    const handled = Handled.begin();
    defer handled.drop();
    var pending = false;
    for (cps) |*cp| pending = pending or !cp.isReleased();
    if (!pending) return;
    if (prompt.len < kv_disk_cache.MIN_PERSIST_TOKENS) return keepLive(self, m, cps);
    // The points share the request's rows: one that outlived this call would make the next forward copy the store.
    defer for (cps) |*cp| cp.release();
    markRows(cache, s) catch |err| {
        log.warn("  [disk-cache] DeepSeek prefill write-through rows not marked: {s}\n", .{@errorName(err)});
        return;
    };
    keepLive(self, m, cps);
    // A point off the row-segment grid would copy the rows past the last segment into its own file; it stays live.
    const span: usize = if (self.disk.?.grown) |g| g.span else 0;
    for (cps) |*cp| if (!cp.isReleased() and span != 0 and cp.pos % span != 0) cp.release();
    for (cps) |*cp| if (!cp.isReleased()) liveOf(self).markFlushed(cp.pos);
    const staged_before = self.disk.?.point_bytes_written;
    var sw = io_util.Stopwatch.init(std.Io.Threaded.global_single_threaded.io());
    self.disk.?.rows_base = @intCast(liveOf(self).base);
    defer self.disk.?.rows_base = std.math.maxInt(u32);
    self.writeThroughHybrid(cache.entries, cache.config, prompt, has_tools, media, null, cps, &.{}, std.math.maxInt(u64), s);
    const bytes = self.disk.?.point_bytes_written - staged_before;
    // The prefill waits here while the writer's bounded buffer is full: the stall a slow SSD costs.
    log.info("[dsv41-prefix] restore points to SSD at {d}: {d:.2} GB staged, {s} held {d} ms\n", .{ prompt.len, @as(f64, @floatFromInt(bytes)) / 1e9, held, sw.read() / std.time.ns_per_ms });
}

fn keepLive(self: *pc.HotPrefixCache, m: *const dsv41_model.Model, cps: []SSMCheckpoint) void {
    for (cps) |*cp| if (!cp.isReleased()) liveOf(self).keep(self.allocator, m, cp, false) catch |err| {
        log.warn("  [dsv41-prefix] live point at {d} not kept: {s}\n", .{ cp.pos, @errorName(err) });
    };
}

/// The request's state at `tokens.len`, which must be where the model stands, as one restore point straight to the
/// SSD tier: shared rows, staged and released before the next forward appends.
pub fn writeDecodePointDsv41(self: *pc.HotPrefixCache, cache: *transformer.KVCache, m: *dsv41_model.Model, tokens: []const u32, has_tools: bool, media: pc.MediaStamp) !void {
    var cps = [1]SSMCheckpoint{try captureShared(self.allocator, m, tokens.len)};
    defer cps[0].deinit(self.allocator);
    writeThroughDsv41(self, m, cache, tokens, has_tools, media, &cps, "decode");
}

/// The request over `m`'s state finished with `tokens`: the points it kept are what the next prompt may roll back to.
/// True when the record holds points that are not on SSD.
pub fn publishLiveDsv41(self: *pc.HotPrefixCache, m: *const dsv41_model.Model, tokens: []const u32, has_tools: bool, media: pc.MediaStamp, quant: kv_quant.KVQuantConfig) bool {
    liveOf(self).publish(self.allocator, m, tokens, has_tools, media.key, media.start);
    liveOf(self).quant = quant;
    return liveOf(self).unflushed();
}

/// The state is about to be replaced: its record reaches SSD (`flushLiveDsv41`), then goes.
pub fn releaseLive(self: *pc.HotPrefixCache, m: *dsv41_model.Model, why: []const u8) void {
    flushLiveDsv41(self, m, why);
    liveOf(self).clear(self.allocator);
}

/// `flushLiveDsv41` for a resident model, if it is a DeepSeek one. Inference thread.
pub fn flushResident(entry: *@import("../model_registry.zig").LoadedModel, why: []const u8) void {
    const hc = if (entry.prefix_cache) |*p| p else return;
    const xfm = arch_handle.trunk(entry.handle orelse return);
    flushLiveDsv41(hc, xfm.arch_state.dsv41.model orelse return, why);
}

/// Writes the live record's points that are not on SSD yet, while the rows they lean on are still the state's own: all
/// of them (up to `live_point_cap`), so what a rewind of the last turns would restore from RAM restores from SSD too. Run
/// where the state is about to be lost (another conversation takes the slot, unload, shutdown, idle) and before anything
/// appends to it; a point a later prefill overwrote is no longer in the record. A record with images is written under
/// its vision key, which hashes the pixels, so only a prompt with the same images restores past the first image row.
pub fn flushLiveDsv41(self: *pc.HotPrefixCache, m: *dsv41_model.Model, why: []const u8) void {
    const s = dsv41_model.stateStream(m);
    const live = liveOf(self);
    if (self.disk == null or !self.ssd_first or !live.ready or live.epoch != m.state_epoch) return;
    // A pixel key with no boundary proves nothing about where its rows begin (`diskTextLen`).
    if (live.vision_key != 0 and live.media_start == null) return;
    const st = m.dec_state orelse return;
    const handled = Handled.begin();
    defer handled.drop();
    var cps: std.ArrayList(SSMCheckpoint) = .empty;
    defer {
        for (cps.items) |*cp| cp.deinit(self.allocator);
        cps.deinit(self.allocator);
    }
    for (live.points.items) |*p| {
        if (p.flushed or p.pos < kv_disk_cache.MIN_PERSIST_TOKENS or p.pos > @min(live.tokens.len, st.n)) continue;
        var cp = captureLive(self.allocator, m, p) catch |err| {
            log.warn("  [dsv41-prefix] live point at {d} not flushed: {s}\n", .{ p.pos, @errorName(err) });
            continue;
        };
        cps.append(self.allocator, cp) catch {
            cp.deinit(self.allocator);
            continue;
        };
    }
    if (cps.items.len == 0) return;
    const top = cps.items[cps.items.len - 1].pos;
    var cache = KVCache.initWithConfig(self.allocator, m.cfg.layers, live.quant) catch return;
    defer cache.deinit();
    cache.step = top;
    markRows(&cache, s) catch |err| {
        log.warn("  [dsv41-prefix] live points not flushed, rows not marked: {s}\n", .{@errorName(err)});
        return;
    };
    const staged_before = self.disk.?.point_bytes_written;
    var sw = io_util.Stopwatch.init(std.Io.Threaded.global_single_threaded.io());
    self.disk.?.rows_base = @intCast(live.base);
    defer self.disk.?.rows_base = std.math.maxInt(u32);
    const media = pc.truncatedMediaKey(&.{}, live.vision_key, if (live.media_start) |v| @as(usize, v) else null, top);
    self.writeThroughHybrid(cache.entries, cache.config, live.tokens[0..top], live.has_tools, media, null, cps.items, &.{}, std.math.maxInt(u64), s);
    for (cps.items) |*cp| live.markFlushed(cp.pos);
    const bytes = self.disk.?.point_bytes_written - staged_before;
    log.info("[dsv41-prefix] live points to SSD ({s}): {d} points to {d}, {d:.2} GB staged, held {d} ms\n", .{ why, cps.items.len, top, @as(f64, @floatFromInt(bytes)) / 1e9, sw.read() / std.time.ns_per_ms });
}

test "snapshotBackoff: the last forward starts at the prompt's last assistant opener when it fits a decode-width pass" {
    const A = 900;
    const T = 901;
    // A plain turn ends opener, think tag: one token is held back and the last forward is two rows.
    try testing.expectEqual(@as(?usize, 1), snapshotBackoff(A, &.{ 5, 6, 7, A, T }));
    // A restored remainder is those two rows alone.
    try testing.expectEqual(@as(?usize, 1), snapshotBackoff(A, &.{ A, T }));
    try testing.expectEqual(@as(?usize, 0), snapshotBackoff(A, &.{ 5, A }));
    // Text the server appended after the tag rides the same forward: opener + 15 more is the widest.
    var forced: [21]u32 = @splat(8);
    forced[4] = A;
    try testing.expectEqual(@as(?usize, 15), snapshotBackoff(A, forced[0..20]));
    try testing.expectEqual(@as(?usize, null), snapshotBackoff(A, &forced));
    // The last opener counts, an earlier turn's does not.
    try testing.expectEqual(@as(?usize, 1), snapshotBackoff(A, &.{ A, T, 9, 9, A, T }));
    try testing.expectEqual(@as(?usize, null), snapshotBackoff(A, &.{ 5, 6, 7 }));
    try testing.expectEqual(@as(?usize, null), snapshotBackoff(A, &.{}));
}

test "layoutKey: a boot that drops other experts or picks leniently has its own namespace" {
    const a = testing.allocator;
    const full = try layoutKey(a, false, true, true, true, null, 0, true);
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

    const k8 = try layoutKey(a, false, true, true, true, &d8, 0, true);
    defer a.free(k8);
    const k8_same = try layoutKey(a, false, true, true, true, &d8_same, 0, true);
    defer a.free(k8_same);
    const k8_other = try layoutKey(a, false, true, true, true, &d8_other, 0, true);
    defer a.free(k8_other);
    const k16 = try layoutKey(a, false, true, true, true, &d16, 0, true);
    defer a.free(k16);
    const k8_dspark = try layoutKey(a, true, true, true, true, &d8, 0, true);
    defer a.free(k8_dspark);
    const k_lossy = try layoutKey(a, false, true, true, true, null, 0.3, true);
    defer a.free(k_lossy);
    const k_lossier = try layoutKey(a, false, true, true, true, null, 0.4, true);
    defer a.free(k_lossier);
    const k8_lossy = try layoutKey(a, false, true, true, true, &d8, 0.3, true);
    defer a.free(k8_lossy);
    const k_simd = try layoutKey(a, false, false, true, true, null, 0, true);
    defer a.free(k_simd);
    const k8_simd = try layoutKey(a, false, false, true, true, &d8, 0, true);
    defer a.free(k8_simd);

    const k_composed = try layoutKey(a, false, true, false, true, null, 0, true);
    defer a.free(k_composed);
    const k_composed_simd = try layoutKey(a, false, false, false, true, null, 0, true);
    defer a.free(k_composed_simd);
    const k_dspark = try layoutKey(a, true, true, true, true, null, 0, true);
    defer a.free(k_dspark);
    const k_dspark_inexact = try layoutKey(a, true, true, true, false, null, 0, true);
    defer a.free(k_dspark_inexact);
    const k_dspark_composed = try layoutKey(a, true, true, false, false, null, 0, true);
    defer a.free(k_dspark_composed);

    // The served keys are unchanged, and a model without DSpark has no verify rows to sum differently.
    try testing.expectEqualStrings("dsv41-state-v6", full);
    try testing.expectEqualStrings("dsv41-state-v6-simd", k_simd);
    try testing.expectEqualStrings("dsv41-state-v6-dspark", k_dspark);
    // The ALU mix sums in another order than the matrix-unit one: its states are their own, beside either attention arm.
    const k_hcalu = try layoutKey(a, false, true, true, true, null, 0, false);
    defer a.free(k_hcalu);
    const k_simd_hcalu = try layoutKey(a, false, false, true, true, null, 0, false);
    defer a.free(k_simd_hcalu);
    try testing.expectEqualStrings("dsv41-state-v6-hcalu", k_hcalu);
    try testing.expectEqualStrings("dsv41-state-v6-simd-hcalu", k_simd_hcalu);
    const no_dspark = try layoutKey(a, false, true, true, false, null, 0, true);
    defer a.free(no_dspark);
    try testing.expectEqualStrings(full, no_dspark);

    try testing.expectEqualStrings(k8, k8_same);
    const distinct = [_][]const u8{ full, k8, k8_other, k16, k8_dspark, k_lossy, k_lossier, k8_lossy, k_simd, k8_simd, k_composed, k_composed_simd, k_dspark, k_dspark_inexact, k_dspark_composed };
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
