//! Hy4-Flash's prefix cache: its request state is the restore target. A sparse layer's rows grow in place
//! (`row_store`), so a point within the running state is its position and its small state alone (`Model.Point`: the
//! sliding rings, the open groups): the prefill keeps them live (`captureCheckpoint`), a finished request hands them to
//! the cache's live record (`Live`), and the next turn rolls the state back to the deepest one it shares
//! (`rollbackLive`) with no copy. A point stored past the state's life (`capture`, an SSMCheckpoint) holds views of the
//! rows too; the SSD tier stores it as a whole record.
const std = @import("std");
const mlx = @import("../mlx.zig");
const trunk = @import("../trunk.zig");
const pc = @import("../prefix_cache.zig");
const kv_disk_cache = @import("../kv_disk_cache.zig");
const ssm_cache = @import("../ssm_cache.zig");
const model = @import("../model.zig");
const hy4_model = @import("hy4_model.zig");
const pool_class = @import("../tensor/pool_class.zig");
const log = @import("../log.zig");
const live_points = @import("../live_points.zig");

pub const State = *hy4_model.Model;

const Point = [hy4_model.Model.point_arrays]?mlx.mlx_array;
/// A point rides `per_layer` snapshot entries a layer, each two of its arrays (`Model.point_fields` by index, null for
/// the placeholder): the latent and rope rows, the scales and biases, the pooled keys, the open group. An entry's arrays
/// grow at one rate, which the SSD tier's split reads (`rowDivisors`).
const per_layer = 4;
const snap_slots = [per_layer][2]?usize{ .{ 0, 1 }, .{ 2, 3 }, .{ 4, null }, .{ 5, null } };

/// Tokens of rows one `g<n>` segment of an SSD entry holds: a point on this grid stores its small state alone, and a
/// conversation's rows go to its segments once.
pub const row_span: u32 = 8192;

/// What the hot cache keeps beside its entries: the last finished request's tokens and the restore points its prefill
/// took, which the next turn may roll the state back to (`rollbackLive`). A fresh cache keeps none.
pub const CacheState = Live;

/// Points the live record keeps: the conversation's newest, beside the running request's own (`Model.live_point_cap`).
pub const record_point_cap = 8;

/// The most the points a model holds in RAM take, whatever the context (`hy4_ledger.pointsBytes`): the record's and the
/// running request's.
pub const ram_tier_max_bytes: u64 = 250_000_000;

pub const Live = struct {
    points: std.ArrayList(hy4_model.Model.Point) = .empty,
    tokens: std.ArrayList(u32) = .empty,
    /// What the record's rows were computed under: the next turn is held to it (`rollbackLive`).
    has_tools: bool = false,
    vision_key: u64 = 0,
    media_start: ?u32 = null,
    /// The model's `version` the record was published at: the model's own state is the record's tokens while it holds.
    version: ?u64 = null,
    /// The model's `epoch` the points were taken in: they stand on the model's rows while it holds (`Model.epoch`).
    epoch: ?u64 = null,
    /// Rows below this position are the running request's restore or rollback: a stored entry's, not computed by it
    /// (`DiskTier.rows_base`).
    base: usize = 0,
    gpa: ?std.mem.Allocator = null,

    /// The points' slots are the model's (`Model.point_fixed`), which may be gone by now: nothing of them is touched.
    pub fn deinit(self: *Live, allocator: std.mem.Allocator) void {
        _ = allocator;
        const gpa = self.gpa orelse return;
        self.points.deinit(gpa);
        self.tokens.deinit(gpa);
    }

    /// Lets go of the record, its points' slots given back to `m`, whose they are.
    pub fn clear(self: *Live, m: State) void {
        for (self.points.items) |*p| m.releasePoint(p);
        self.points.clearRetainingCapacity();
        self.tokens.clearRetainingCapacity();
        self.version = null;
        self.epoch = null;
        self.base = 0;
    }

    /// Whether a point of the record is only here, not on `disk`.
    pub fn unflushed(self: *const Live, disk: ?*const kv_disk_cache.DiskTier) bool {
        return live_points.anyUnflushed(self.points.items, disk);
    }

    /// Lets go of the points deeper than `pos`: a rollback to `pos` writes over their rows.
    fn dropPointsPast(self: *Live, pos: usize, m: State) void {
        while (self.points.items.len > 0 and self.points.items[self.points.items.len - 1].pos > pos) {
            const p = self.points.pop().?;
            m.releasePoint(&p);
        }
    }

    /// The request over `m` finished with `tokens`, run under `has_tools` and the media stamp: its points join the
    /// record's (those of an earlier turn it continued, which stand on the same rows), the newest `record_point_cap`
    /// kept; its tokens are copied, and the record is the model's state as it stands now.
    pub fn publish(self: *Live, m: State, tokens: []const u32, has_tools: bool, media: pc.MediaStamp) void {
        if (self.gpa == null) self.gpa = m.gpa;
        const gpa = self.gpa.?;
        // Points of another state, or past what this request kept of the last one, are not this state's.
        if (self.epoch != m.epoch) self.dropPointsPast(0, m);
        if (m.live_points.items.len > 0) self.dropPointsPast(m.live_points.items[0].pos -| 1, m);
        self.points.appendSlice(gpa, m.live_points.items) catch {
            self.clear(m);
            return m.dropLive();
        };
        m.live_points.clearRetainingCapacity();
        while (self.points.items.len > record_point_cap) {
            const p = self.points.orderedRemove(0);
            m.releasePoint(&p);
        }
        self.tokens.clearRetainingCapacity();
        self.tokens.appendSlice(gpa, tokens) catch return self.clear(m);
        self.has_tools = has_tools;
        self.vision_key = media.key;
        self.media_start = media.start;
        self.version = m.version;
        self.epoch = m.epoch;
    }
};

/// The request over `m` finished with `tokens`: a point where its state stands (behind the reply, which the next prompt
/// repeats, so a turn that restores it after a handover prefills only what is new) joins its own, and the record is
/// published.
pub fn publishLive(hc: *pc.HotPrefixCache, m: State, tokens: []const u32, has_tools: bool, media: pc.MediaStamp) void {
    const newest: usize = if (m.live_points.items.len > 0) m.live_points.items[m.live_points.items.len - 1].pos else 0;
    if (m.pos <= tokens.len and m.pos > newest) m.keepLivePoint() catch |err| {
        log.warn("  [hy4-prefix] restore point behind the reply at {d} not kept: {s}\n", .{ m.pos, @errorName(err) });
    };
    liveOf(hc).publish(m, tokens, has_tools, media);
}

pub fn liveOf(hc: *pc.HotPrefixCache) *Live {
    return &hc.arch.hy4flash;
}

/// The buffers the model holds, each billed to one owner. A stored point shares its rows' buffers with the state while
/// no forward has run past it, and its small state with the slot of the live point taken with it, so a buffer is
/// counted once, at its pool size class: the hot cache's entries bill theirs (`see`), the model bills the rest (`add`).
pub const Held = struct {
    gpa: std.mem.Allocator,
    seen: std.AutoHashMapUnmanaged(usize, void) = .empty,
    bytes: u64 = 0,

    pub fn deinit(self: *Held) void {
        self.seen.deinit(self.gpa);
    }

    fn take(self: *Held, a: mlx.mlx_array, bill: bool) !void {
        if (a.ctx == null) return;
        const store = if (mlx.engine.isOurs(a)) mlx.engine.storeKey(a) else @intFromPtr(a.ctx);
        const fresh = !(try self.seen.getOrPut(self.gpa, store)).found_existing;
        if (fresh and bill) self.bytes +|= pool_class.poolBytes(@intCast(ssm_cache.arrayBytes(a)));
    }

    /// An entry's points, whose bytes its `kv_bytes` bills.
    pub fn see(self: *Held, cps: []const ssm_cache.SSMCheckpoint) !void {
        for (cps) |cp| for (cp.layers) |l| for ([_]mlx.mlx_array{ l.conv_state, l.ssm_state }) |a| try self.take(a, false);
    }

    /// A point's arrays, in `Model.capturePoint`'s layout.
    pub fn addPoint(self: *Held, arrays: []const [hy4_model.Model.point_arrays]?mlx.mlx_array) !void {
        for (arrays) |row| for (row) |maybe| if (maybe) |a| try self.take(a, true);
    }

    /// The `Points` slots, which hold the running request's points and the live record's.
    fn addPoints(self: *Held, m: State) !void {
        for (&m.point_fixed, m.point_hidden) |*slot, hidden| {
            try self.addPoint(slot[0..m.config.layers]);
            if (hidden) |h| try self.take(h, true);
        }
    }

    /// The request's state (with a drafter's handover, picks and mark blocks) and the `Points` slots.
    pub fn addModel(self: *Held, m: State) !void {
        var arrays: [hy4_model.max_state_arrays]mlx.mlx_array = undefined;
        for (m.state[0..m.config.layers]) |*st| for (hy4_model.stateArrays(st, &arrays)) |a| try self.take(a, true);
        if (m.hidden_fixed) |h| try self.take(h, true);
        for (m.picks_fixed[0..m.config.layers]) |pf| for (pf) |b| if (b) |a| try self.take(a, true);
        for (m.picks_rows[0..m.config.layers]) |pf| for (pf) |b| if (b) |a| try self.take(a, true);
        for (m.key_journal[0..m.config.layers]) |b| if (b) |a| try self.take(a, true);
        if (m.rows_streams) |h| try self.take(h, true);
        try self.addPoint(m.mark_fixed[0..m.config.layers]);
        if (m.mark_hidden) |h| try self.take(h, true);
        try self.addPoints(m);
    }
};

/// What the model's `Points` slots hold past its state, each buffer once (the RAM tier, which `ram_tier_max_bytes`
/// bounds).
pub fn pointsHeldBytes(m: State) !u64 {
    var held: Held = .{ .gpa = m.gpa };
    defer held.deinit();
    var arrays: [hy4_model.max_state_arrays]mlx.mlx_array = undefined;
    for (m.state[0..m.config.layers]) |*st| for (hy4_model.stateArrays(st, &arrays)) |a| try held.take(a, false);
    try held.addPoints(m);
    return held.bytes;
}

/// What the model holds past what the hot cache's entries bill: the state it parks or runs, its running points and the
/// live record's. The scheduler reads it as the model's own state (`hy4_sched.modelStateBytes`, `parkedStateBytes`).
pub fn heldBytes(m: State, hc: ?*pc.HotPrefixCache) u64 {
    var held: Held = .{ .gpa = m.gpa };
    defer held.deinit();
    heldBytesInto(&held, m, hc) catch |err| std.log.err("[hy4-prefix] the model's held bytes are short: {s}", .{@errorName(err)});
    return held.bytes;
}

fn heldBytesInto(held: *Held, m: State, hc: ?*pc.HotPrefixCache) !void {
    if (hc) |h| {
        for (h.entries.items) |e| if (e.ssm_checkpoints) |cps| try held.see(cps);
    }
    try held.addModel(m);
}

pub fn reset(m: State) void {
    m.reset();
}

/// The request state restores only from its points.
pub fn checkpointsRequired(config: *const model.ModelConfig) bool {
    _ = config;
    return true;
}

pub fn restorePointModel(xfm: *trunk.Trunk) bool {
    return xfm.arch_state.hy4flash.model != null;
}

/// Restore points one prefill or one RAM entry lists: positions, their state released once a write-through has taken
/// it or at once (`captureCheckpoint`), their small state the live points' (`Live`).
pub const checkpoint_cap: u32 = 4;

pub fn checkpointMax(global: u32) u32 {
    return if (global == 0) checkpoint_cap else @min(global, checkpoint_cap);
}

/// A restore point of the request at `pos`, where it must stand: a live point, and a stored point whose rows are views
/// of the state's. Only a write-through (`to_ssd`) takes it before the next forward appends (`writeThrough`); any other
/// is its position alone, since one held past that forward would make it copy every store, and the live point serves.
pub fn captureCheckpoint(allocator: std.mem.Allocator, xfm: *trunk.Trunk, ctx: *const trunk.ForwardCtx, pos: usize, to_ssd: bool) !?ssm_cache.SSMCheckpoint {
    _ = ctx;
    const m = xfm.arch_state.hy4flash.model orelse return null;
    if (m.pos != pos) return error.Hy4CachePositionMismatch;
    var cp = try captureKept(allocator, m, pos);
    if (!to_ssd or pos < kv_disk_cache.MIN_PERSIST_TOKENS) cp.release();
    return cp;
}

/// Hy4's write-through: the request's points on the SSD tier's row-segment grid are written, their rows into the
/// entry's segments once a conversation (`setDiskGrownSpec`), and every point leaves RAM at once, the writer having
/// copied what it stages: one that outlived this call would make the next forward copy every store. A point off the
/// grid is its live point only, and reaches the tier at the handover (`flushLive`).
pub fn writeThrough(hc: *pc.HotPrefixCache, m: State, cache: *const trunk.KVCache, prompt: []const u32, has_tools: bool, media: pc.MediaStamp, restored: ?kv_disk_cache.RestoredRows, cps: []ssm_cache.SSMCheckpoint, s: mlx.mlx_stream) void {
    defer for (cps) |*cp| cp.release();
    const d = if (hc.disk) |*dd| dd else return;
    if (!live_points.releaseOffGrid(d, cps)) return;
    const w = live_points.write(hc, cache.entries, cache.config, prompt, has_tools, media, restored, cps, liveOf(hc).base, s);
    if (w.id) |id| for ([_][]hy4_model.Model.Point{ liveOf(hc).points.items, m.live_points.items }) |list| live_points.markOnDisk(list, d, id);
    log.info("[hy4-prefix] restore points to SSD at {d}: {d:.2} MB staged\n", .{ prompt.len, @as(f64, @floatFromInt(w.bytes)) / 1e6 });
}

/// Writes the live record's points that are not on SSD yet, while the rows they stand on are still the state's own:
/// run where the state is about to be lost (another conversation takes it, unload, shutdown, idle). A record with
/// images is written under its vision key (`live_points.flush`).
pub fn flushLive(hc: *pc.HotPrefixCache, m: State, why: []const u8) void {
    const live = liveOf(hc);
    if (hc.disk == null or live.epoch != m.epoch or live.points.items.len == 0) return;
    // A pixel key with no boundary proves nothing about where its rows begin.
    if (live.vision_key != 0 and live.media_start == null) return;
    const had = mlx.errorPending();
    defer mlx.dropLatchedErrorUnless(had);
    const rec: live_points.Record = .{ .tokens = live.tokens.items, .has_tools = live.has_tools, .vision_key = live.vision_key, .media_start = if (live.media_start) |v| @as(usize, v) else null, .base = live.base };
    const f = live_points.flush(FlushCtx{ .m = m }, hc, live.points.items, rec, @min(live.tokens.items.len, m.pos), m.config.layers, m.kv, m.s, "[hy4-prefix]");
    if (f.id) |id| live_points.markOnDisk(m.live_points.items, &hc.disk.?, id);
    if (f.points == 0) return;
    log.info("[hy4-prefix] live points to SSD ({s}): {d} points to {d}, {d:.2} MB staged\n", .{ why, f.points, f.top, @as(f64, @floatFromInt(f.bytes)) / 1e6 });
}

/// How `flushLive` captures a live point; its records need no rows marker (`DiskTier.points_hold_state`).
const FlushCtx = struct {
    m: State,

    pub fn capture(self: FlushCtx, allocator: std.mem.Allocator, p: *const hy4_model.Model.Point) !ssm_cache.SSMCheckpoint {
        return captureLive(allocator, self.m, p);
    }

    pub fn prepare(_: FlushCtx, _: *trunk.KVCache) !void {}
};

/// The state is about to be replaced: its record reaches SSD (`flushLive`), then goes.
pub fn releaseLive(hc: *pc.HotPrefixCache, m: State, why: []const u8) void {
    flushLive(hc, m, why);
    liveOf(hc).clear(m);
}

pub fn handover(hc: *pc.HotPrefixCache, m: State, _: mlx.mlx_stream) void {
    releaseLive(hc, m, "handover");
}

/// The live state serves the prompt in place: where the last request's tokens and the prompt part (one short of the
/// prompt, which must forward its last token, and short of the media the two turns do not share), the state continues
/// as it stands when it is still the record's and stands within that, else the deepest live point within it is
/// installed. A turn under other tools takes neither. Null when nothing serves.
pub fn rollbackLive(
    hc: *pc.HotPrefixCache,
    m: State,
    target_cache: *trunk.KVCache,
    target_moe_seq_offset: *usize,
    s: mlx.mlx_stream,
    prompt_ids: []const u32,
    has_tools: bool,
    media_limits: kv_disk_cache.MediaLimits,
    ram_match: ?pc.RamMatch,
) ?pc.LookupResult {
    if (rollbackLiveInPlace(hc, m, target_cache, target_moe_seq_offset, s, prompt_ids, has_tools, media_limits, ram_match)) |res| return res;
    // The request starts the state over or restores a stored point: the record's points reach SSD first.
    releaseLive(hc, m, "handover");
    return null;
}

fn rollbackLiveInPlace(
    hc: *pc.HotPrefixCache,
    m: State,
    target_cache: *trunk.KVCache,
    target_moe_seq_offset: *usize,
    s: mlx.mlx_stream,
    prompt_ids: []const u32,
    has_tools: bool,
    media_limits: kv_disk_cache.MediaLimits,
    ram_match: ?pc.RamMatch,
) ?pc.LookupResult {
    const live = liveOf(hc);
    const tokens = live.tokens.items;
    // A record of another state: its points' rows are gone.
    if (live.epoch != m.epoch) {
        live.clear(m);
        return null;
    }
    if (tokens.len == 0 or prompt_ids.len < 2 or live.has_tools != has_tools) return null;
    // The state and its points hold rows at the width they were made at; a request at another starts over.
    const want = target_cache.config;
    if (m.kv.scheme != want.scheme or m.kv.bits != want.bits or m.kv.group_size != want.group_size) return null;
    const shared = std.mem.indexOfDiff(u32, tokens, prompt_ids) orelse @min(tokens.len, prompt_ids.len);
    const limit = @min(@min(shared, prompt_ids.len - 1), media_limits.ceilingFor(live.vision_key, live.media_start));
    const in_place = live.version == m.version and m.pos > 0 and m.pos <= limit;
    var best: ?*const hy4_model.Model.Point = null;
    if (!in_place) for (live.points.items) |*p| if (p.pos <= limit and (best == null or p.pos > best.?.pos)) {
        best = p;
    };
    const pos: usize = if (in_place) m.pos else if (best) |p| p.pos else return null;
    // A stored point deeper than the live one serves instead: the RAM entry's, or the SSD tier's.
    var stored: usize = if (ram_match) |rm| blk: {
        const cps = hc.entries.items[rm.idx].ssm_checkpoints orelse break :blk 0;
        break :blk (pc.HotPrefixCache.highestCheckpointAtOrBelow(cps, rm.shared) orelse break :blk 0).pos;
    } else 0;
    if (hc.disk) |*d| if (d.bestHybridMatch(prompt_ids, has_tools, target_cache.config, @intCast(prompt_ids.len), media_limits)) |hm| {
        stored = @max(stored, hm.cp);
    };
    if (stored > pos) return null;
    // The points a request kept and never published are past the record's: not this request's.
    m.dropLive();
    if (!in_place) {
        m.rollbackPoint(best.?, prompt_ids.len) catch |err| {
            std.log.warn("[hy4-prefix] live state not rolled back: {s}", .{@errorName(err)});
            live.clear(m);
            m.reset();
            return null;
        };
    }
    live.dropPointsPast(pos, m);
    live.base = pos;
    target_cache.truncate(0, s) catch {};
    target_cache.step = pos;
    target_moe_seq_offset.* = pos;
    hc.last_restored_live = true;
    return .{ .matched = pos, .full_match = false, .slot_owned = true };
}

/// The state at `pos` kept as a live point, for the next turn to roll back to without a stored restore, and as a stored
/// point that shares the live one's small state (its slot) and the state's rows.
pub fn captureKept(allocator: std.mem.Allocator, m: State, pos: usize) !ssm_cache.SSMCheckpoint {
    if (m.pos != pos) return error.Hy4CachePositionMismatch;
    try m.keepLivePoint();
    const live = m.live_points.items;
    return captureLive(allocator, m, &live[live.len - 1]);
}

/// The state at `pos`, where it stands, as a stored point of its own copies: its small state and views of its rows
/// (`Model.capturePoint`).

pub fn capture(allocator: std.mem.Allocator, m: State, pos: usize) !ssm_cache.SSMCheckpoint {
    const points = try allocator.alloc(Point, m.config.layers);
    defer allocator.free(points);
    try m.capturePoint(points);
    return pack(allocator, m, points, pos);
}

/// The live point `p` as the stored point `capture` took there: its small state and views of the state's rows below it,
/// which are the point's own while the record stands on the state.
fn captureLive(allocator: std.mem.Allocator, m: State, p: *const hy4_model.Model.Point) !ssm_cache.SSMCheckpoint {
    const points = try allocator.alloc(Point, m.config.layers);
    defer allocator.free(points);
    try m.pointArrays(p, points);
    errdefer freePoints(points);
    try m.captureRows(points, p.pos);
    return pack(allocator, m, points, p.pos);
}

/// `points` (taken: their slots emptied) as a checkpoint at `pos`, `per_layer` snapshot entries a layer.
fn pack(allocator: std.mem.Allocator, m: State, points: []Point, pos: usize) !ssm_cache.SSMCheckpoint {
    const n = m.config.layers;
    const layers = allocator.alloc(ssm_cache.SSMCacheEntrySnapshot, per_layer * n) catch |err| {
        freePoints(points);
        return err;
    };
    // The absent arrays share the model's placeholder.
    const first = m.placeholder orelse return error.Hy4CheckpointPlaceholder;
    var done: usize = 0;
    errdefer {
        for (layers[0..done]) |l| {
            _ = mlx.mlx_array_free(l.conv_state);
            _ = mlx.mlx_array_free(l.ssm_state);
        }
        freePoints(points);
        allocator.free(layers);
    }
    for (points) |*row| for (snap_slots) |pair| {
        var none: ?mlx.mlx_array = null;
        const a = try orPlaceholder(&row[pair[0].?], first);
        errdefer _ = mlx.mlx_array_free(a);
        const b = try orPlaceholder(if (pair[1]) |i| &row[i] else &none, first);
        layers[done] = .{ .conv_state = a, .ssm_state = b, .initialized = true };
        done += 1;
    };
    return .{ .pos = pos, .layers = layers };
}

fn freePoints(points: []Point) void {
    for (points) |*row| for (row) |*a| if (a.*) |x| {
        _ = mlx.mlx_array_free(x);
        a.* = null;
    };
}

/// The slot's array, taken (the slot is emptied), or for an absent one a new handle on the placeholder.
fn orPlaceholder(slot: *?mlx.mlx_array, placeholder: mlx.mlx_array) !mlx.mlx_array {
    if (slot.*) |a| {
        slot.* = null;
        return a;
    }
    var again = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(again);
    try mlx.check(mlx.mlx_array_set(&again, placeholder));
    return again;
}

fn isPlaceholder(a: mlx.mlx_array) bool {
    return mlx.mlx_array_dtype(a) == .uint8 and mlx.getShape(a).len == 1 and mlx.mlx_array_size(a) == 1;
}

/// The state reserved for a request of `prompt` tokens and `max_tokens`, `past_end` speculative rows past its reply, as
/// its first forward reserves it (`KVCache.requestEnd`), made before its lookup: a restore into it asks for nothing. A
/// state deeper than that end is left to the lookup, which rolls it back or starts it over.
pub fn reserveRequest(m: State, prompt: usize, max_tokens: u64, past_end: u64) !void {
    const end: usize = @intCast(trunk.KVCache.requestEnd(prompt, max_tokens, prompt +| max_tokens, past_end));
    if (end >= m.pos) try m.reserveFor(end);
}

/// Makes the request's state the point `cp` over the first `cp.pos` of `ids`; the point's arrays are shared.
pub fn restore(m: State, cp: *const ssm_cache.SSMCheckpoint, ids: []const u32) !void {
    const n = m.config.layers;
    if (cp.layers.len != per_layer * n) return error.Hy4CheckpointMismatch;
    const points = try m.gpa.alloc(Point, n);
    defer m.gpa.free(points);
    for (points, 0..) |*row, l| {
        row.* = @splat(null);
        for (snap_slots, 0..) |pair, k| {
            const snap = cp.layers[per_layer * l + k];
            row[pair[0].?] = if (isPlaceholder(snap.conv_state)) null else snap.conv_state;
            if (pair[1]) |i| row[i] = if (isPlaceholder(snap.ssm_state)) null else snap.ssm_state;
        }
    }
    try m.restorePoint(points, cp.pos, ids);
    // The restored state's own point, which a next turn that repeats this prompt rolls back to.
    m.keepLivePoint() catch |err| std.log.warn("[hy4-prefix] live point at {d} not kept: {s}", .{ cp.pos, @errorName(err) });
}

/// Installs the deepest point of `e` at or below `restore_cap`; the position it stands at, 0 when none serves.
pub fn restoreEntry(hc: *pc.HotPrefixCache, m: State, e: *pc.Entry, restore_cap: usize, prompt_ids: []const u32) !usize {
    const cps = e.ssm_checkpoints orelse return 0;
    const cp = pc.HotPrefixCache.highestCheckpointAtOrBelow(cps, restore_cap) orelse return 0;
    if (cp.released_bytes != 0 or cp.pos > prompt_ids.len) return 0;
    try restore(m, cp, prompt_ids);
    liveOf(hc).base = cp.pos;
    return cp.pos;
}

/// The SSD namespace of a cache-biased load (`--expert-cache-bias`): its states come from picks the expert cache steered,
/// so no exact boot reads them. An exact load keeps the config's own namespace (null).
pub fn diskLayoutKey(allocator: std.mem.Allocator, config: *const model.ModelConfig, model_dir: []const u8, xfm: *trunk.Trunk) !?[]u8 {
    _ = .{ model_dir, xfm };
    const bias = config.expertCacheBias();
    if (!(bias > 0)) return null;
    const ns = config.cacheLayoutNamespace() orelse "";
    return try std.fmt.allocPrint(allocator, "{s}{s}cbias{x:0>8}", .{ ns, if (ns.len > 0) "-" else "", @as(u32, @bitCast(bias)) });
}

/// The SSD tier stores Hy4's points as records over their tokens (they hold the request's whole state, the trunk cache
/// no rows), split as DeepSeek's are: a point's small state in its own file, its rows in the entry's segments of
/// `row_span` tokens, each written once.
pub fn setDiskGrownSpec(allocator: std.mem.Allocator, config: *const model.ModelConfig, disk: *kv_disk_cache.DiskTier) !void {
    disk.points_hold_state = true;
    const c = config.hy4flash orelse return;
    try setRowSpan(allocator, &c, disk, row_span, @import("../bills.zig").defaultKvBits(config));
}

/// The tier's split of Hy4's points (`kv_disk_cache.GrownSpec`) on a grid of `span` tokens; its row bytes are the
/// served 8-bit width's.
pub fn setRowSpan(allocator: std.mem.Allocator, c: *const @import("hy4_config.zig").Config, disk: *kv_disk_cache.DiskTier, span: u32, kv_bits: u64) !void {
    const divs = try rowDivisors(allocator, c);
    defer allocator.free(divs);
    try disk.setGrownSpec(span, divs, @import("hy4_ledger.zig").stateBytesPerToken(c, kv_bits));
}

/// Per snapshot entry the tokens one of its rows holds: a sparse layer's latent and rope rows and its scales and
/// biases a token, its pooled keys a group; 0 for the rings and the open groups, which do not grow. Caller frees.
pub fn rowDivisors(allocator: std.mem.Allocator, c: *const @import("hy4_config.zig").Config) ![]u32 {
    const divs = try allocator.alloc(u32, per_layer * c.layers);
    @memset(divs, 0);
    for (0..c.layers) |l| if (c.kinds[l] == .sparse) {
        divs[per_layer * l] = 1;
        divs[per_layer * l + 1] = 1;
        divs[per_layer * l + 2] = c.indexer.group;
    };
    return divs;
}

/// The point at `pos` of the tier's entry `idx` read straight into the request's state (`Model.restoreBegin`), which
/// then stands at it over `ids`, with its own live point. No array is made: a restore asks a closed pool nothing.
pub fn restoreDiskInto(d: *kv_disk_cache.DiskTier, m: State, target_cache: *trunk.KVCache, idx: usize, pos: u32, ids: []const u32) !void {
    const Kv = @import("../kv_quant.zig").KVQuantConfig;
    const kv = target_cache.config;
    if (!std.meta.eql(kv, Kv.engine_default) and !std.meta.eql(kv, Kv.dense)) return error.Hy4KvWidthUnsupported;
    try m.restoreBegin(kv, pos);
    const Into = struct {
        m: State,
        rows: [@import("hy4_config.zig").max_layers][hy4_model.Model.point_arrays]usize = @splat(@splat(0)),
        /// What the state refused: the reader's own error may name only the read.
        failed: ?anyerror = null,

        fn dest(ctx: *anyopaque, li: usize, field: u1, row: usize, t: @import("../prefix_vfs_tensors.zig").Named) anyerror!?[]u8 {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            // An absent array's placeholder holds nothing of the state.
            if (t.dtype == .uint8 and t.shape.len == 1 and t.shape[0] == 1) return null;
            const i = snap_slots[li % per_layer][field] orelse return self.fail(error.Hy4PointMismatch);
            const l = li / per_layer;
            const out = self.m.restoreDest(l, i, row, t.dtype, t.shape) catch |e| return self.fail(e);
            self.rows[l][i] += @intCast(t.shape[0]);
            return out;
        }

        fn fail(self: *@This(), e: anyerror) anyerror {
            self.failed = e;
            return e;
        }
    };
    var into: Into = .{ .m = m };
    d.restorePointInto(target_cache, idx, pos, per_layer * m.config.layers, .{ .ctx = &into, .dest = Into.dest }, m.s) catch |e| return into.failed orelse e;
    try m.restoreEnd(pos, ids, into.rows[0..m.config.layers]);
    m.keepLivePoint() catch |err| std.log.warn("[hy4-prefix] live point at {d} not kept: {s}", .{ pos, @errorName(err) });
}

/// The deepest point the SSD tier holds for the prompt, when it beats the RAM match: its bytes are read into the
/// request's state. A point that will not read or install falls back to the RAM or cold path.
pub fn restoreDisk(
    hc: *pc.HotPrefixCache,
    d: *kv_disk_cache.DiskTier,
    m: State,
    target_cache: *trunk.KVCache,
    ram_eff: usize,
    prompt_ids: []const u32,
    has_tools: bool,
    media_limits: kv_disk_cache.MediaLimits,
    dflash_target: ?pc.DflashTarget,
    s: mlx.mlx_stream,
) ?pc.LookupResult {
    _ = dflash_target;
    _ = s;
    const hm = d.bestHybridMatch(prompt_ids, has_tools, target_cache.config, @intCast(prompt_ids.len), media_limits) orelse return null;
    if (@as(usize, hm.cp) < ram_eff + kv_disk_cache.MIN_DISK_ADVANTAGE_TOKENS) return null;
    // A reservation past the state's grows it: the state's own room, whatever cap the last forward left.
    const cap = mlx.engine.uncapped();
    defer mlx.engine.capped(cap);
    const had = mlx.errorPending();
    const refusals = mlx.memoryRefusals();
    restoreDiskInto(d, m, target_cache, hm.idx, hm.cp, prompt_ids) catch |err| {
        std.log.warn("  [disk-cache] Hy4 restore failed: {s}; RAM or cold path", .{@errorName(err)});
        pc.HotPrefixCache.poisonUnreadable(d, hm.idx, err, refusals);
        target_cache.truncate(0, m.s) catch {};
        m.reset();
        mlx.dropLatchedErrorUnless(had);
        return null;
    };
    hc.last_restored_disk_id = d.entries.items[hm.idx].id;
    liveOf(hc).base = hm.cp;
    // The point the request restored is on SSD where it was read.
    if (m.live_points.items.len > 0) m.live_points.items[m.live_points.items.len - 1].disk_id = d.entries.items[hm.idx].id;
    return .{ .matched = hm.cp, .full_match = false, .slot_owned = true };
}

test "hy4 prefix: a cache-biased load's SSD tier is its own namespace, apart from an exact load's, and the flag off keeps the exact one (HY4_STQ_TINY)" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
    const settings = @import("../model_settings.zig");
    const flag_was = settings.expert_cache_bias_flag;
    defer settings.expert_cache_bias_flag = flag_was;
    settings.expert_cache_bias_flag = null;
    const fingerprint = struct {
        fn of(al: std.mem.Allocator, i: std.Io, d: []const u8, bias: ?f32) ![]u8 {
            const config = model.ModelConfig{ .model_type = "hy_v4_flash", .expert_cache_bias_override = bias };
            const key = try diskLayoutKey(al, &config, d, undefined);
            defer if (key) |k| al.free(k);
            return @import("../disk_placement.zig").modelFingerprintFor(al, i, d, key orelse config.cacheLayoutNamespace(), null);
        }
    }.of;
    const exact = try fingerprint(a, io, dir, null);
    defer a.free(exact);
    const off = try fingerprint(a, io, dir, 0);
    defer a.free(off);
    const biased = try fingerprint(a, io, dir, 0.05);
    defer a.free(biased);
    const other = try fingerprint(a, io, dir, 0.1);
    defer a.free(other);
    try std.testing.expectEqualStrings(exact, off);
    try std.testing.expect(!std.mem.eql(u8, exact, biased));
    try std.testing.expect(!std.mem.eql(u8, biased, other));
    // The launch flag outranks the setting: a flag of 0 is an exact boot.
    settings.expert_cache_bias_flag = 0;
    const flag_off = try fingerprint(a, io, dir, 0.05);
    defer a.free(flag_off);
    try std.testing.expectEqualStrings(exact, flag_off);
}

test "hy4 prefix: setDiskGrownSpec sets row_bytes from the configured KV width (8 and 16)" {
    const bills = @import("../bills.zig");
    const B = @import("hy4_ledger.zig");
    const hy4_config = @import("hy4_config.zig");
    const kv_quant = @import("../kv_quant.zig");
    const a = std.testing.allocator;
    const io = std.testing.io;
    const prev_kv = bills.launch_state.configured_kv_quant;
    defer bills.launch_state.configured_kv_quant = prev_kv;
    bills.launch_state.configured_kv_quant = null;
    const c = try hy4_config.parseText(hy4_config.release_config);
    for ([_]struct { kv: kv_quant.KVQuantConfig, bits: u64 }{
        .{ .kv = .engine_default, .bits = 8 },
        .{ .kv = .dense, .bits = 16 },
    }) |case| {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const base = buf[0..try tmp.dir.realPath(io, &buf)];
        var disk = try kv_disk_cache.DiskTier.init(a, io, base, "hy4-kv-width", 0, 128);
        defer disk.deinit();
        var mc = model.ModelConfig{ .model_type = "hy_v4_flash", .hy4flash = c, .kv_quant_override = case.kv };
        try setDiskGrownSpec(a, &mc, &disk);
        try std.testing.expectEqual(B.stateBytesPerToken(&c, case.bits), disk.grown.?.row_bytes);
    }
}

test "hy4 prefix: the held bytes count each store once, two stores a dry recording made as two" {
    try mlx.requireGpu();
    const core = @import("../tensor/core.zig");
    const plan = @import("../tensor/plan/plan_api.zig");
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const scope = mlx.engine.enter(mlx.engine.stream());
    defer scope.end();
    var rec = try plan.Recorder.begin(std.testing.allocator, .{ .site = 1 });
    defer rec.deinit();
    var driver: plan.Driver = .{ .gpa = std.testing.allocator };
    defer driver.deinit();
    try core.beginRecording(.{ .dry = .{ .driver = &driver } }, &rec);
    defer core.endRecording() catch {};
    var a = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a);
    var b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b);
    try mlx.check(mlx.engine.freshArray(&a, &.{ 64, 64 }, .float32));
    try mlx.check(mlx.engine.freshArray(&b, &.{ 64, 64 }, .float32));
    var shared = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(shared);
    try mlx.check(mlx.mlx_array_set(&shared, a));
    var held: Held = .{ .gpa = std.testing.allocator };
    defer held.deinit();
    const row: [hy4_model.Model.point_arrays]?mlx.mlx_array = .{ a, b, shared, null, null, null };
    try held.addPoint(&.{row});
    try std.testing.expectEqual(2 * pool_class.poolBytes(64 * 64 * 4), held.bytes);
}
