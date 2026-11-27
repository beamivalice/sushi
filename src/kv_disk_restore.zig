//! SSD prefix cache lookup matching and restore paths (KV chunks, SSM/KDA points, rings, spec sidecars).

const std = @import("std");
const mlx = @import("mlx.zig");
const kv_quant = @import("kv_quant.zig");
const trunk = @import("trunk.zig");
const mlx_pages = @import("mlx_pages.zig");
const prefix_arch = @import("arch/prefix.zig");
const log = @import("log.zig");
const disk_placement = @import("disk_placement.zig");
const vt = @import("prefix_vfs_tensors.zig");
const safetensors = @import("safetensors.zig");
const disk = @import("kv_disk_cache.zig");

const KVCache = trunk.KVCache;
const DiskTier = disk.DiskTier;
const IndexEntry = disk.IndexEntry;
const MediaLimits = disk.MediaLimits;
const Match = disk.Match;
const HybridMatch = disk.HybridMatch;
const RingMatch = disk.RingMatch;
const SpecKind = disk.SpecKind;
const SpecMeta = disk.SpecMeta;
const Growth = DiskTier.Growth;
const specGrowth = DiskTier.specGrowth;
const commonPrefixLen = disk.commonPrefixLen;

// ── Lookup ──

/// Longest usable shared prefix across persisted entries with a matching
/// (has_tools, quant) key. Same filter semantics as the RAM cache: a
/// cross-config restore would hand SDPA a wrong buffer layout.
pub fn bestMatch(
    self: *const DiskTier,
    prompt_ids: []const u32,
    has_tools: bool,
    quant: kv_quant.KVQuantConfig,
    media: MediaLimits,
) ?Match {
    var best_idx: ?usize = null;
    var best_usable: u32 = 0;
    for (self.entries.items, 0..) |*e, i| {
        if (e.poisoned) continue; // a failed write killed it: this is a MISS
        if (e.rings.len > 0) continue; // its chunks lack the ringed layers: `bestRingMatch`
        if (e.has_tools != has_tools) continue;
        if (!std.meta.eql(e.quant, quant)) continue;
        const cap = media.at(e) orelse continue;
        const max_shared = @min(e.tokens.len, prompt_ids.len);
        var shared: usize = 0;
        while (shared < max_shared and e.tokens[shared] == prompt_ids[shared]) shared += 1;
        const usable: u32 = @intCast(@min(@min(shared, e.kv_len), @as(usize, cap)));
        if (usable > best_usable) {
            best_usable = usable;
            best_idx = i;
        }
    }
    if (best_idx) |idx| return .{ .idx = idx, .usable = best_usable };
    return null;
}

/// Hybrid targets rank disk entries by their RESTORABLE position — the
/// highest SSM checkpoint at or below the usable prefix — not by the raw
/// usable length (the RAM tier's #312 lesson: a longer raw match whose
/// checkpoints sit past the divergence restores nothing, and must not
/// shadow a shorter entry with a higher restorable position). Entries
/// with no checkpoint at or below their usable prefix are skipped.
pub fn bestHybridMatch(
    self: *const DiskTier,
    prompt_ids: []const u32,
    has_tools: bool,
    quant: kv_quant.KVQuantConfig,
    limit: u32,
    media: MediaLimits,
) ?HybridMatch {
    var best: ?HybridMatch = null;
    for (self.entries.items, 0..) |*e, i| {
        if (e.poisoned) continue; // a failed write killed it: this is a MISS
        if (e.has_tools != has_tools) continue;
        if (!std.meta.eql(e.quant, quant)) continue;
        const cap = media.at(e) orelse continue;
        const max_shared = @min(e.tokens.len, prompt_ids.len);
        var shared: usize = 0;
        while (shared < max_shared and e.tokens[shared] == prompt_ids[shared]) shared += 1;
        const usable: u32 = @intCast(@min(@min(shared, e.kv_len), @min(@as(usize, limit), @as(usize, cap))));
        const cp_limit: u32 = @intCast(@min(usable, prompt_ids.len -| 1));
        const cp = self.highestSsmPosAtOrBelow(i, cp_limit) orelse continue;
        if (best == null or cp > best.?.cp) best = .{ .idx = i, .usable = usable, .cp = cp };
    }
    return best;
}

/// A ringed target's lookup: the entry and ring file restoring the most of `prompt_ids`
/// within `limit`. A ring file at `pos` holding `rows` serves a clamp to `at <= pos` while
/// its rows still reach a window below `at` (a full match re-forwards its last token).
pub fn bestRingMatch(
    self: *const DiskTier,
    prompt_ids: []const u32,
    has_tools: bool,
    quant: kv_quant.KVQuantConfig,
    limit: u32,
    window: u32,
    media: MediaLimits,
) ?RingMatch {
    var best: ?RingMatch = null;
    for (self.entries.items, 0..) |*e, i| {
        if (e.poisoned or e.rings.len == 0) continue;
        if (e.has_tools != has_tools) continue;
        if (!std.meta.eql(e.quant, quant)) continue;
        const cap = media.at(e) orelse continue;
        const usable: u32 = @intCast(@min(@min(commonPrefixLen(e.tokens, prompt_ids), e.kv_len), @min(@as(usize, limit), @as(usize, cap))));
        for (e.rings) |r| {
            const at = @min(usable, r.pos);
            const clamp: u32 = if (at == prompt_ids.len and at > 1) at - 1 else at;
            const low = r.pos - r.rows;
            if (clamp < low or clamp - low < @min(clamp, window)) continue;
            if (best == null or at > best.?.at) best = .{ .idx = i, .pos = r.pos, .at = at };
        }
    }
    return best;
}

/// Which of the ascending restore points `steps` of a cache holding `tokens` an intact entry
/// keeps a ring file for. A file counts once staged: a restore drains its entry first, and a
/// write that then fails poisons the entry (`harvestWriteFailures`).
pub fn ringPointsHeld(
    self: *const DiskTier,
    tokens: []const u32,
    has_tools: bool,
    quant: kv_quant.KVQuantConfig,
    steps: []const usize,
    held: []bool,
    media: MediaLimits,
) void {
    @memset(held, false);
    for (self.entries.items) |*e| {
        if (e.poisoned or e.rings.len == 0) continue;
        if (e.has_tools != has_tools) continue;
        if (!std.meta.eql(e.quant, quant)) continue;
        const cap = media.at(e) orelse continue;
        const common = @min(commonPrefixLen(e.tokens, tokens), @as(usize, cap));
        for (steps, held) |step, *h| {
            if (h.* or step > common or step > e.kv_len) continue;
            for (e.rings) |r| {
                if (r.pos == step) h.* = true;
            }
        }
    }
}

/// Rebuild `entries[idx]` at `len` into a ringed cache: the chunked layers up to `len`, then
/// ring file `pos` (>= len) under the ringed layers (`KVCache.restoreRing`), clamped to
/// `len`. On error the cache may be half-rebuilt; the caller resets it.
pub fn restoreIntoRinged(self: *DiskTier, cache: *KVCache, idx: usize, pos: u32, len: u32, s: mlx.mlx_stream) !void {
    // Callers swallow a restore failure and prefill cold; the latch it raised must not fail that prefill.
    const had_error = mlx.errorPending();
    errdefer mlx.dropLatchedErrorUnless(had_error);
    self.drainEntry(self.entries.items[idx].id);
    const e = &self.entries.items[idx];
    if (len == 0 or len > pos) return error.DiskCacheNoCheckpoint;
    const known = for (e.rings) |r| {
        if (r.pos == pos) break true;
    } else false;
    if (!known) return error.DiskCacheNoCheckpoint;
    // Loaded FIRST, so a corrupt file fails before the cache is touched.
    var cp = try self.loadRingFile(e.id, pos, cache.entries.len, e.quant, cache.swa_ring_window);
    defer cp.deinit();
    try self.restoreKvInto(cache, e, len, s);
    for (cache.entries, cp.entries) |*dst, *src| {
        if (src.initialized and dst.initialized) return error.DiskCacheCorruptRing;
    }
    try cache.restoreRing(&cp);
    try cache.truncate(len, s);
    e.last_used = self.bump();
    self.writeMeta(e.*) catch {};
}

/// Load ring file `pos` as a checkpoint snapshot (`KVCache.ringCheckpoint`'s shape): each
/// ringed layer's rows `[pos - rows, pos)`, every other layer empty.
pub fn loadRingFile(self: *DiskTier, id: u64, pos: u32, n_layers: usize, quant: kv_quant.KVQuantConfig, window: u32) !trunk.KVCacheSnapshot {
    const rd = self.reading();
    defer rd.end();
    const cpu = rd.s;
    const path = try std.fmt.allocPrint(self.allocator, "{s}/e{d}/r{d:0>7}.safetensors", .{ self.root, id, pos });
    defer self.allocator.free(path);
    if (self.fs.size(path) == null) return error.DiskCacheNoCheckpoint;
    var loaded = try self.fs.loadTensors(path, cpu);
    defer loaded.deinit();
    const layers_s = loaded.getMeta("layers") orelse return error.DiskCacheCorruptRing;
    const recorded = std.fmt.parseInt(usize, layers_s, 10) catch return error.DiskCacheCorruptRing;
    if (recorded != n_layers) return error.DiskCacheCorruptRing;

    const entries = try self.allocator.alloc(trunk.KVCacheEntry, n_layers);
    for (entries) |*en| en.* = trunk.newEmptyKVEntry();
    var cp: trunk.KVCacheSnapshot = .{ .entries = entries, .step = pos, .allocator = self.allocator, .config = quant, .swa_ring_window = window };
    errdefer cp.deinit();
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    const kinds: []const []const u8 = if (quant.scheme == .off) &.{ "k", "v" } else &.{ "k", "v", "ks", "kb", "vs", "vb" };
    var key_buf: [32]u8 = undefined;
    for (entries, 0..) |*en, li| {
        const slots = [_]*mlx_pages.InPlace{ &en.keys, &en.values, &en.keys_scales, &en.keys_biases, &en.values_scales, &en.values_biases };
        for (kinds, 0..) |kind, ki| {
            const key = try std.fmt.bufPrint(&key_buf, "l{d}.{s}", .{ li, kind });
            const arr = loaded.take(key) orelse {
                if (ki == 0) break; // not a ringed layer
                return error.DiskCacheCorruptRing;
            };
            slots[ki].deinit();
            slots[ki].* = mlx_pages.InPlace.adopt(arr, .ring);
            const shape = mlx.getShape(arr);
            if (shape.len != 4 or shape[2] <= 0 or shape[2] > pos) return error.DiskCacheCorruptRing;
            const rows: usize = @intCast(shape[2]);
            if (ki == 0) {
                en.initialized = true;
                en.ringed = true;
                en.offset = rows;
                en.base = pos - rows;
            } else if (rows != en.offset) return error.DiskCacheCorruptRing;
            _ = mlx.mlx_vector_array_append_value(vec, arr);
        }
    }
    if (mlx.mlx_vector_array_size(vec) == 0) return error.DiskCacheCorruptRing;
    // A CHECKED eval: lazy Load reads the file here, so corruption surfaces before install.
    try mlx.check(mlx.mlx_eval(vec));
    return cp;
}

/// Rebuild the persisted KV state of `entries[idx]` into `cache`:
/// per-layer chunk tensors are loaded (CPU stream), concatenated along
/// the sequence axis, and installed as the cache's storage buffers.
/// Views are left empty — identical contract to `KVCache.restore` (the
/// next `update`/`truncate` rebuilds them). Returns kv_len.
pub fn restoreInto(self: *DiskTier, cache: *KVCache, idx: usize, s: mlx.mlx_stream) !u32 {
    const kv_len = self.entries.items[idx].kv_len;
    try self.restorePrefixInto(cache, idx, kv_len, s);
    return kv_len;
}

/// Rebuild ONLY positions [0, limit) of `entries[idx]` — `limit` must be
/// ≤ its kv_len. A short shared prefix against a long stored entry then
/// reads just the chunks covering `limit` instead of the whole entry (a
/// diverged-prefix "hit" that would otherwise read every stored chunk to
/// serve a few hundred tokens — slower than a cold prefill).
pub fn restorePrefixInto(self: *DiskTier, cache: *KVCache, idx: usize, limit: u32, s: mlx.mlx_stream) !void {
    // Callers swallow a restore failure and prefill cold; the latch it raised must not fail that prefill.
    const had_error = mlx.errorPending();
    errdefer mlx.dropLatchedErrorUnless(had_error);
    // This entry's staged chunks must be on disk before the readback.
    self.drainEntry(self.entries.items[idx].id);
    const e = &self.entries.items[idx];
    try self.restoreKvInto(cache, e, limit, s);
    e.last_used = self.bump();
    // Bump meta.json mtime so cross-restart LRU sees the use.
    self.writeMeta(e.*) catch {};
}

/// Phase 3 hybrid variant: rebuild the KV prefix covering [0, cp_pos)
/// AND install the SSM state persisted at `cp_pos` into `ssm_entries`.
/// Mirrors the RAM tier's hybrid-restore semantics — KV and SSM state
/// land at the SAME position, the caller continues prefill from cp_pos.
/// `cp_pos` must be one of the entry's persisted checkpoint positions
/// (pick via `highestSsmPosAtOrBelow`). On error the cache/entries may be
/// half-rebuilt — the caller resets both and falls back to cold prefill.
pub fn restoreIntoHybrid(
    self: *DiskTier,
    cache: *KVCache,
    ssm_entries: []trunk.SSMCacheEntry,
    idx: usize,
    cp_pos: u32,
    s: mlx.mlx_stream,
) !u32 {
    // Callers swallow a restore failure and prefill cold; the latch it raised must not fail that prefill.
    const had_error = mlx.errorPending();
    errdefer mlx.dropLatchedErrorUnless(had_error);
    self.drainEntry(self.entries.items[idx].id);
    const e = &self.entries.items[idx];
    if (cp_pos == 0 or cp_pos > e.kv_len) return error.DiskCacheNoCheckpoint;
    if (std.mem.indexOfScalar(u32, e.ssm_positions, cp_pos) == null) return error.DiskCacheNoCheckpoint;
    // Load the checkpoint FIRST (transient, no side effects on the live
    // state) so a corrupt/missing file fails before the cache is touched.
    var cp = try self.loadSsmFile(e.id, cp_pos, ssm_entries.len);
    defer cp.deinit(self.allocator);
    const want_rows = DiskTier.qsaHistoryRowsOf(&cp);
    // The overlay is owed unless the pooled bank is on THIS checkpoint: every checkpoint
    // carries its own raw leftover, and a mid-block one (backoff 30) is not the bank's home.
    const snap_has_pooled = prefix_arch.aux.checkpointHasQsaPooled(&cp);
    var qsa_overlay: ?trunk.SSMCheckpoint = null;
    defer if (qsa_overlay) |*q| q.deinit(self.allocator);
    if ((want_rows > 0 or e.qsa_history_rows > 0) and !snap_has_pooled) {
        var hist = self.loadQsaHistoryFile(e.id, ssm_entries.len) catch return error.DiskCacheQsaHistoryGap;
        if (!prefix_arch.aux.checkpointQsaCoversPos(&cp, &hist, cp_pos)) {
            hist.deinit(self.allocator);
            return error.DiskCacheQsaHistoryGap;
        }
        qsa_overlay = hist;
    } else if (!snap_has_pooled and e.ssm_positions.len > 0) {
        const latest = e.ssm_positions[e.ssm_positions.len - 1];
        if (latest != cp_pos) {
            if (self.loadSsmFile(e.id, latest, ssm_entries.len)) |latest_cp_val| {
                var latest_cp = latest_cp_val;
                if (prefix_arch.aux.checkpointHasQsaPooled(&latest_cp)) {
                    if (!prefix_arch.aux.checkpointQsaCoversPos(&cp, &latest_cp, cp_pos)) {
                        latest_cp.deinit(self.allocator);
                        return error.DiskCacheQsaHistoryGap;
                    }
                    qsa_overlay = latest_cp;
                } else {
                    latest_cp.deinit(self.allocator);
                }
            } else |_| return error.DiskCacheQsaHistoryGap;
        }
    }
    if (disk_placement.test_qsa_overlay_mismatch) {
        if (qsa_overlay) |*q| q.deinit(self.allocator);
        qsa_overlay = .{ .pos = 0, .layers = try self.allocator.alloc(trunk.SSMCacheEntrySnapshot, 0) };
    }
    try self.restoreKvInto(cache, e, cp_pos, s);
    errdefer cache.truncate(0, s) catch {};
    try trunk.restoreSsmCheckpoint(ssm_entries, &cp, s);
    errdefer {
        for (ssm_entries) |*ent| {
            _ = mlx.mlx_array_free(ent.conv_state);
            _ = mlx.mlx_array_free(ent.ssm_state);
            ent.conv_state = mlx.mlx_array_new();
            ent.ssm_state = mlx.mlx_array_new();
            ent.initialized = false;
            prefix_arch.aux.ssmFreeQsaState(ent);
            ent.ple_prev_valid = false;
        }
    }
    if (qsa_overlay) |*qsa_cp| {
        try prefix_arch.aux.applyQsaHistoryAt(ssm_entries, qsa_cp, cp_pos, s, true);
    }
    e.last_used = self.bump();
    self.writeMeta(e.*) catch {};
    return cp_pos;
}

/// `restoreIntoHybrid` for a state with no QSA history (a KDA state): the KV prefix and the
/// checkpoint at `cp_pos`, without reading any other checkpoint file.
pub fn restoreIntoKda(
    self: *DiskTier,
    cache: *KVCache,
    ssm_entries: []trunk.SSMCacheEntry,
    idx: usize,
    cp_pos: u32,
    s: mlx.mlx_stream,
) !u32 {
    const had_error = mlx.errorPending();
    errdefer mlx.dropLatchedErrorUnless(had_error);
    self.drainEntry(self.entries.items[idx].id);
    const e = &self.entries.items[idx];
    if (cp_pos == 0 or cp_pos > e.kv_len) return error.DiskCacheNoCheckpoint;
    if (std.mem.indexOfScalar(u32, e.ssm_positions, cp_pos) == null) return error.DiskCacheNoCheckpoint;
    var cp = try self.loadSsmFile(e.id, cp_pos, ssm_entries.len);
    defer cp.deinit(self.allocator);
    try self.restoreKvInto(cache, e, cp_pos, s);
    errdefer cache.truncate(0, s) catch {};
    try trunk.restoreSsmCheckpoint(ssm_entries, &cp, s);
    e.last_used = self.bump();
    self.writeMeta(e.*) catch {};
    return cp_pos;
}

/// The restore point at `cp_pos` of entry `idx` as a checkpoint the caller owns, for a state that has no KV
/// rows of its own (compressed rows): the entry's chunks are read only to prove they cover the position.
pub fn restoreCheckpointInto(
    self: *DiskTier,
    cache: *KVCache,
    idx: usize,
    cp_pos: u32,
    n_layers: usize,
    s: mlx.mlx_stream,
) !trunk.SSMCheckpoint {
    const had_error = mlx.errorPending();
    errdefer mlx.dropLatchedErrorUnless(had_error);
    self.drainEntry(self.entries.items[idx].id);
    const e = &self.entries.items[idx];
    if (cp_pos == 0 or cp_pos > e.kv_len) return error.DiskCacheNoCheckpoint;
    if (std.mem.indexOfScalar(u32, e.ssm_positions, cp_pos) == null) return error.DiskCacheNoCheckpoint;
    var cp = try self.loadSsmFile(e.id, cp_pos, n_layers);
    errdefer cp.deinit(self.allocator);
    try self.restoreKvInto(cache, e, cp_pos, s);
    e.last_used = self.bump();
    self.writeMeta(e.*) catch {};
    return cp;
}

/// Where a point's arrays are read (`restorePointInto`): for array `field` (0 a snapshot layer's `conv`, 1 its `ssm`)
/// of snapshot layer `li`, the bytes that `t` fills from the array's row `row` on, exactly its size, or null to pass
/// it over.
pub const PointSink = struct {
    ctx: *anyopaque,
    dest: *const fn (ctx: *anyopaque, li: usize, field: u1, row: usize, t: vt.Named) anyerror!?[]u8,
};

/// `restoreCheckpointInto` with no array made: the point's arrays are read straight into the bytes `sink` names, a
/// growing array's rows from the entry's row segments and the point's own file each at their rows.
pub fn restorePointInto(self: *DiskTier, cache: *KVCache, idx: usize, cp_pos: u32, n_layers: usize, sink: PointSink, s: mlx.mlx_stream) !void {
    const had_error = mlx.errorPending();
    errdefer mlx.dropLatchedErrorUnless(had_error);
    self.drainEntry(self.entries.items[idx].id);
    const e = &self.entries.items[idx];
    if (cp_pos == 0 or cp_pos > e.kv_len) return error.DiskCacheNoCheckpoint;
    if (std.mem.indexOfScalar(u32, e.ssm_positions, cp_pos) == null) return error.DiskCacheNoCheckpoint;
    try self.readPointInto(e.id, cp_pos, n_layers, sink);
    try self.restoreKvInto(cache, e, cp_pos, s);
    e.last_used = self.bump();
    self.writeMeta(e.*) catch {};
}

/// A point file's or a row segment's tensors handed to a `PointSink`, each at the row its array holds it from.
pub const PointRead = struct {
    sink: PointSink,
    divs: []const u32,
    span: u32,
    n_layers: usize,
    /// The row segments the point leans on, from its file's `grown`; null until its metadata is read.
    grown: ?u32 = null,
    /// The segment being read (1-based), 0 for the point's own file.
    segment: u32 = 0,

    fn meta(ctx: *anyopaque, pairs: []const vt.MetaPair) anyerror!void {
        const r: *PointRead = @ptrCast(@alignCast(ctx));
        if (r.segment > 0) return;
        const layers = metaValue(pairs, "layers") orelse return error.DiskCacheCorruptSsm;
        if ((std.fmt.parseInt(usize, layers, 10) catch return error.DiskCacheCorruptSsm) != r.n_layers) return error.DiskCacheSsmLayerMismatch;
        const k = metaValue(pairs, "grown") orelse "0";
        r.grown = std.fmt.parseInt(u32, k, 10) catch return error.DiskCacheCorruptSsm;
    }

    fn dest(ctx: *anyopaque, t: vt.Named) anyerror!?[]u8 {
        const r: *PointRead = @ptrCast(@alignCast(ctx));
        const dot = std.mem.indexOfScalar(u8, t.key, '.') orelse return error.DiskCacheCorruptSsm;
        if (t.key[0] != 'l') return error.DiskCacheCorruptSsm;
        const li = std.fmt.parseInt(usize, t.key[1..dot], 10) catch return error.DiskCacheCorruptSsm;
        const name = t.key[dot + 1 ..];
        const field: u1 = if (std.mem.eql(u8, name, "conv")) 0 else if (std.mem.eql(u8, name, "ssm")) 1 else return error.DiskCacheCorruptSsm;
        if (li >= r.n_layers) return error.DiskCacheCorruptSsm;
        // A growing array's rows past the segments sit in the point's own file (`encodePoint`); a fixed one is whole.
        const div: u32 = if (li < r.divs.len) r.divs[li] else 0;
        const grows = div > 0 and r.span % div == 0 and t.shape.len == 2 and t.shape[0] > 0;
        const row: usize = if (r.segment > 0) blk: {
            if (!grows) return error.DiskCacheCorruptSsm;
            break :blk (@as(usize, r.segment) - 1) * r.span / div;
        } else if (grows and r.grown.? > 0) @as(usize, r.grown.?) * r.span / div else 0;
        return r.sink.dest(r.sink.ctx, li, field, row, t);
    }
};

pub fn metaValue(pairs: []const vt.MetaPair, key: []const u8) ?[]const u8 {
    for (pairs) |p| if (std.mem.eql(u8, p.key, key)) return p.value;
    return null;
}

pub fn readPointInto(self: *DiskTier, id: u64, pos: u32, n_layers: usize, sink: PointSink) !void {
    var arena = std.heap.ArenaAllocator.init(self.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const g = self.grown;
    var r: PointRead = .{ .sink = sink, .divs = if (g) |x| x.divs else &.{}, .span = if (g) |x| x.span else 0, .n_layers = n_layers };
    const into: vt.Sink = .{ .ctx = &r, .dest = PointRead.dest, .meta = PointRead.meta };
    self.last_point_held = self.heldIndex(id, pos) != null;
    if (self.heldIndex(id, pos)) |hi| {
        _ = try vt.readBytesInto(self.held.items[hi].bytes, aa, into);
    } else {
        const path = try std.fmt.allocPrint(aa, "{s}/e{d}/s{d:0>7}.safetensors", .{ self.root, id, pos });
        if (self.fs.size(path) == null) return error.DiskCacheNoCheckpoint;
        _ = try self.fs.readTensorsInto(path, aa, into);
    }
    const k = r.grown orelse return error.DiskCacheCorruptSsm;
    if (k > 0 and r.span == 0) return error.DiskCacheCorruptSsm;
    for (1..@as(usize, k) + 1) |j| {
        r.segment = @intCast(j);
        const path = try std.fmt.allocPrint(aa, "{s}/e{d}/g{d:0>5}.safetensors", .{ self.root, id, j });
        if (self.fs.size(path) == null) return error.DiskCacheNoCheckpoint;
        _ = try self.fs.readTensorsInto(path, aa, into);
    }
}

/// Largest persisted SSM checkpoint position ≤ `limit` for entry `idx`;
/// null when none qualifies (hybrid KV without SSM state is unusable, so
/// the caller must skip the entry entirely).
pub fn highestSsmPosAtOrBelow(self: *const DiskTier, idx: usize, limit: u32) ?u32 {
    var best: ?u32 = null;
    for (self.entries.items[idx].ssm_positions) |p| {
        if (p > limit) break; // sorted ascending
        best = p;
    }
    return best;
}

/// Does entry `e`'s `tokens.bin` still begin with the `n` tokens its index matched? The index
/// is read once, at scan or commit; the rows a restore serves must be the ones recorded for it.
pub fn tokenRecordMatches(self: *DiskTier, e: *const IndexEntry, n: u32) !bool {
    const path = try std.fmt.allocPrint(self.allocator, "{s}/e{d}/tokens.bin", .{ self.root, e.id });
    defer self.allocator.free(path);
    return self.fs.startsWith(path, std.mem.sliceAsBytes(e.tokens[0..n]));
}


/// Shared chunk-loading body: rebuild positions [0, limit) of entry `e`
/// into `cache` (limit == e.kv_len for the plain-attention path; a
/// checkpoint position for the hybrid path — the final chunk is sliced
/// down so KV lands exactly at the checkpoint).
pub fn restoreKvInto(self: *DiskTier, cache: *KVCache, e: *IndexEntry, limit: u32, s: mlx.mlx_stream) !void {
    const quant = e.quant;
    if (!std.meta.eql(cache.config, quant)) return error.DiskCacheConfigMismatch;
    if (limit == 0 or limit > e.kv_len) return error.DiskCacheEmptyEntry;
    if (!try self.tokenRecordMatches(e, limit)) {
        self.poisonEntry(e, "token record differs from the index");
        return error.DiskCacheTokenMismatch;
    }
    const n_chunks: u32 = @intCast((@as(u64, limit) + self.chunk_tokens - 1) / self.chunk_tokens);
    if (n_chunks == 0) return error.DiskCacheEmptyEntry;
    self.chunks_loaded_last = n_chunks;

    const rd = self.reading();
    defer rd.end();
    const cpu = rd.s;

    const kinds: []const []const u8 = if (quant.scheme == .off)
        &.{ "k", "v" }
    else
        &.{ "k", "v", "ks", "kb", "vs", "vb" };

    // Per-layer per-kind buffers holding `limit` rows, filled chunk by chunk in place (on MLX's page grid,
    // `restoredRows`): the restore runs before any admission bill sees it, and a concatenation at the end held
    // every chunk beside the result (twice the restored KV). Layers absent from chunk 0 stay uninitialized: the
    // GatedDeltaNet layers of a hybrid arch have no KV (their state rides the SSM checkpoints).
    const n_layers = cache.entries.len;
    const dsts = try self.allocator.alloc(mlx_pages.InPlace, n_layers * kinds.len);
    for (dsts) |*d| d.* = .none;
    defer {
        for (dsts) |*d| d.deinit();
        self.allocator.free(dsts);
    }
    const present = try self.allocator.alloc(bool, n_layers);
    defer self.allocator.free(present);
    @memset(present, false);

    var chunk_i: u32 = 0;
    while (chunk_i < n_chunks) : (chunk_i += 1) {
        // A chunk the index names but the disk lacks fails every later restore the same way.
        errdefer if (!self.chunkOnDisk(e, chunk_i)) self.poisonEntry(e, "chunk file missing or short");
        const c0: u64 = @as(u64, chunk_i) * self.chunk_tokens;
        const need: u64 = @min(@as(u64, self.chunk_tokens), limit - c0);

        const path = try std.fmt.allocPrint(self.allocator, "{s}/e{d}/c{d:0>6}.safetensors", .{ self.root, e.id, chunk_i });
        defer self.allocator.free(path);
        var loaded = try self.fs.loadTensors(path, cpu);
        defer loaded.deinit();
        const chunk_arrays = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(chunk_arrays);

        var key_buf: [32]u8 = undefined;
        for (0..n_layers) |li| {
            const rows_held: c_int = if (chunk_i == 0) @intCast(try self.restoredRows(&loaded, li, kinds, limit, s)) else 0;
            for (kinds, 0..) |kind, ki| {
                const key = try std.fmt.bufPrint(&key_buf, "l{d}.{s}", .{ li, kind });
                var arr = loaded.take(key) orelse {
                    // A layer holds KV in every chunk or in none.
                    if (ki == 0 and !present[li]) break;
                    return error.DiskCacheCorruptChunk;
                };
                defer if (arr.ctx != null) {
                    _ = mlx.mlx_array_free(arr);
                };
                if (ki == 0) {
                    if (chunk_i == 0) present[li] = true;
                    if (!present[li]) return error.DiskCacheCorruptChunk;
                }
                // Crash-tolerance: a chunk file may hold MORE positions
                // than meta.json committed to (rewrite raced a crash).
                // Slice down to the committed range; never trust the file.
                const shape = mlx.getShape(arr);
                if (shape.len != 4) return error.DiskCacheCorruptChunk;
                const have: u64 = @intCast(shape[2]);
                if (have < need) return error.DiskCacheCorruptChunk;
                if (have > need) {
                    var sliced = mlx.mlx_array_new();
                    const st_i = [_]c_int{ 0, 0, 0, 0 };
                    const sp_i = [_]c_int{ shape[0], shape[1], @intCast(need), shape[3] };
                    const sd_i = [_]c_int{ 1, 1, 1, 1 };
                    const rc = mlx.mlx_slice(&sliced, arr, &st_i, 4, &sp_i, 4, &sd_i, 4, s);
                    _ = mlx.mlx_array_free(arr);
                    arr = .{};
                    if (rc != 0) {
                        _ = mlx.mlx_array_free(sliced);
                        return mlx.check(rc);
                    }
                    arr = sliced;
                }
                const dst = &dsts[li * kinds.len + ki];
                const rows = mlx.getShape(arr);
                if (chunk_i == 0) {
                    const made = try mlx_pages.inPlaceBuffer(&.{ rows[0], rows[1], rows_held, rows[3] }, mlx.mlx_array_dtype(arr), s);
                    dst.deinit();
                    dst.* = made;
                }
                const dshape = mlx.getShape(dst.array);
                if (dshape.len != 4 or dshape[0] != rows[0] or dshape[1] != rows[1] or dshape[3] != rows[3])
                    return error.DiskCacheCorruptChunk;
                var updated = mlx.mlx_array_new();
                const st = [_]c_int{ 0, 0, @intCast(c0), 0 };
                const sp = [_]c_int{ rows[0], rows[1], @intCast(c0 + need), rows[3] };
                const next = dst.write(&updated, arr, &st, &sp, s) catch |err| {
                    _ = mlx.mlx_array_free(updated);
                    return err;
                };
                _ = mlx.mlx_array_free(dst.array);
                dst.* = next;
                _ = mlx.mlx_vector_array_append_value(chunk_arrays, updated);
            }
        }
        // A CHECKED eval: lazy Load reads the file here, so a corrupt chunk surfaces its MLX
        // error before install and the caller falls back to cold prefill.
        try mlx.check(mlx.mlx_eval(chunk_arrays));
        // An eval returns before its command buffer lets go of the filled buffers; the next write
        // into a still-held buffer copies all of it instead of donating, so drain the stream first.
        _ = mlx.mlx_synchronize(s);
    }

    // Install the filled buffers as the cache's storage. Mirrors `KVCache.restore`: views stay
    // empty, offset/initialized set, step = kv_len.
    for (cache.entries, 0..) |*dst, li| {
        trunk.resetKVEntry(dst);
        if (!present[li]) continue;
        const slots = [_]*mlx_pages.InPlace{ &dst.keys, &dst.values, &dst.keys_scales, &dst.keys_biases, &dst.values_scales, &dst.values_biases };
        for (slots[0..kinds.len], dsts[li * kinds.len ..][0..kinds.len]) |slot, *filled| {
            slot.deinit();
            slot.* = filled.*;
            filled.* = .none;
        }
        dst.offset = limit;
        dst.initialized = true;
    }
    cache.step = limit;
}

/// The rows a restored layer's buffers hold for `limit` restored rows: on MLX the page grid of every kind's row
/// bytes, read off chunk 0 (`KVCache.heldRows`, which the cache's later grows take), so the fill and the request's
/// appends write in place. The engine of our own holds `limit`.
pub fn restoredRows(_: *DiskTier, chunk0: *const safetensors.Loaded, li: usize, kinds: []const []const u8, limit: u32, s: mlx.mlx_stream) !u64 {
    if (mlx.engine.isStream(s)) return limit;
    var parts: [6]u64 = @splat(0);
    var key_buf: [32]u8 = undefined;
    for (kinds, 0..) |kind, ki| {
        const key = try std.fmt.bufPrint(&key_buf, "l{d}.{s}", .{ li, kind });
        const arr = chunk0.get(key) orelse return limit;
        const sh = mlx.getShape(arr);
        if (sh.len != 4) return error.DiskCacheCorruptChunk;
        parts[ki] = @as(u64, @intCast(sh[0] * sh[1] * sh[3])) * mlx.mlx_array_itemsize(arr);
    }
    return trunk.KVCache.heldRows(0, limit, .{ .end = limit }, parts[0..kinds.len]);
}

/// Rebuilds a point's growing arrays: the entry's first `k` row segments in order, then the rows the point's own
/// file kept. An array the segments do not hold is a fixed one and stays as loaded.
pub fn joinGrownSegments(self: *DiskTier, id: u64, k: u32, layers: []trunk.SSMCacheEntrySnapshot) !void {
    const rd = self.reading();
    defer rd.end();
    const cpu = rd.s;
    // pieces[2 * li + f]: layer li's conv (f = 0) or ssm (f = 1) rows from each segment.
    const pieces = try self.allocator.alloc(mlx.mlx_vector_array, 2 * layers.len);
    defer self.allocator.free(pieces);
    for (pieces) |*p| p.* = mlx.mlx_vector_array_new();
    defer for (pieces) |p| {
        _ = mlx.mlx_vector_array_free(p);
    };
    var key_buf: [32]u8 = undefined;
    for (1..@as(usize, k) + 1) |j| {
        const path = try std.fmt.allocPrint(self.allocator, "{s}/e{d}/g{d:0>5}.safetensors", .{ self.root, id, j });
        defer self.allocator.free(path);
        if (self.fs.size(path) == null) return error.DiskCacheNoCheckpoint;
        var loaded = try self.fs.loadTensors(path, cpu);
        defer loaded.deinit();
        for (0..layers.len) |li| for ([_][]const u8{ "conv", "ssm" }, 0..) |name, f| {
            const key = try std.fmt.bufPrint(&key_buf, "l{d}.{s}", .{ li, name });
            const piece = loaded.get(key) orelse continue;
            try mlx.check(mlx.mlx_vector_array_append_value(pieces[2 * li + f], piece));
        };
    }
    for (layers, 0..) |*l, li| for ([_]*mlx.mlx_array{ &l.conv_state, &l.ssm_state }, 0..) |dst, f| {
        const vec = pieces[2 * li + f];
        if (mlx.mlx_vector_array_size(vec) == 0) continue;
        if (dst.ctx != null and mlx.mlx_array_size(dst.*) > 0) try mlx.check(mlx.mlx_vector_array_append_value(vec, dst.*));
        var joined = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(joined);
        try mlx.check(mlx.mlx_concatenate_axis(&joined, vec, 0, cpu));
        _ = mlx.mlx_array_free(dst.*);
        dst.* = joined;
    };
}

/// A read's stream, and the handles a read makes without one (maps, vectors) on that stream's side until `end`. Every
/// load of the tier reads through it: a Cooler load's restore read on MLX's CPU stream crashed on a map of the other
/// side.
pub const Reading = struct {
    s: mlx.mlx_stream,
    scope: mlx.engine.Scope,

    pub fn end(r: Reading) void {
        r.scope.end();
        _ = mlx.mlx_stream_free(r.s);
    }
};

pub fn reading(self: *const DiskTier) Reading {
    const s = self.arrays_on orelse mlx.mlx_default_cpu_stream_new();
    return .{ .s = s, .scope = mlx.engine.enter(s) };
}

/// Load a persisted SSM checkpoint file into a transient `SSMCheckpoint`
/// (caller frees via `deinit`). The recorded layer count must match the
/// target model's `ssm_entries` — a mismatch inside a fingerprint dir
/// means corruption, never a different model.
pub fn loadSsmFile(self: *DiskTier, id: u64, pos: u32, n_layers: usize) !trunk.SSMCheckpoint {
    const rd = self.reading();
    defer rd.end();
    const cpu = rd.s;
    const path = try std.fmt.allocPrint(self.allocator, "{s}/e{d}/s{d:0>7}.safetensors", .{ self.root, id, pos });
    defer self.allocator.free(path);
    self.last_point_held = self.heldIndex(id, pos) != null;
    var loaded = if (self.heldIndex(id, pos)) |hi|
        try vt.loadBytes(self.allocator, self.held.items[hi].bytes, cpu)
    else blk: {
        if (self.fs.size(path) == null) return error.DiskCacheNoCheckpoint;
        break :blk try self.fs.loadTensors(path, cpu);
    };
    defer loaded.deinit();

    const layers_s = loaded.getMeta("layers") orelse return error.DiskCacheCorruptSsm;
    const recorded = std.fmt.parseInt(usize, layers_s, 10) catch return error.DiskCacheCorruptSsm;
    if (recorded != n_layers) return error.DiskCacheSsmLayerMismatch;
    const init_str = loaded.getMeta("init") orelse return error.DiskCacheCorruptSsm;

    const layers = try self.allocator.alloc(trunk.SSMCacheEntrySnapshot, n_layers);
    for (layers) |*l| l.* = .{
        .conv_state = mlx.mlx_array_new(),
        .ssm_state = mlx.mlx_array_new(),
        .initialized = false,
    };
    var cp: trunk.SSMCheckpoint = .{ .pos = pos, .layers = layers };
    errdefer cp.deinit(self.allocator);

    // Absent key = null state (plain-attention layers in the hybrid have
    // neither) — that's a valid
    // shape, not corruption.
    var key_buf: [32]u8 = undefined;
    for (layers, 0..) |*l, li| {
        if (loaded.take(try std.fmt.bufPrint(&key_buf, "l{d}.conv", .{li}))) |conv| l.conv_state = conv;
        if (loaded.take(try std.fmt.bufPrint(&key_buf, "l{d}.ssm", .{li}))) |ssm| l.ssm_state = ssm;
        if (loaded.take(try std.fmt.bufPrint(&key_buf, "l{d}.aux", .{li}))) |aux| l.aux_state = aux;
        if (loaded.take(try std.fmt.bufPrint(&key_buf, "l{d}.pooled", .{li}))) |pooled| l.qsa_pooled = pooled;
        if (loaded.get(try std.fmt.bufPrint(&key_buf, "l{d}.ple", .{li}))) |ple| {
            if (mlx.mlx_array_dtype(ple) != .uint32 or mlx.mlx_array_size(ple) != 9) return error.DiskCacheCorruptSsm;
            try mlx.check(mlx.mlx_array_eval(ple));
            const d = mlx.mlx_array_data_uint32(ple) orelse return error.DiskCacheCorruptSsm;
            l.ple_prev_valid = d[0] != 0;
            for (0..8) |i| l.ple_prev[i] = d[1 + i];
        }
    }
    if (loaded.getMeta("qsa_ratio")) |ratio_s| {
        const ratio = std.fmt.parseInt(c_int, ratio_s, 10) catch return error.DiskCacheCorruptSsm;
        for (layers) |*l| l.qsa_ratio = ratio;
    }
    if (loaded.getMeta("qsa_rows")) |rows_s| {
        const rows = std.fmt.parseInt(c_int, rows_s, 10) catch return error.DiskCacheCorruptSsm;
        for (layers) |*l| l.qsa_rows = rows;
    } else {
        for (layers) |*l| {
            if (!prefix_arch.aux.snapshotHasQsaHistory(l)) continue;
            if (l.aux_state.ctx != null) {
                const sh = mlx.getShape(l.aux_state);
                if (sh.len >= 2) l.qsa_rows = sh[1];
            }
        }
    }

    // `initialized=true` with both states null is a valid shape, so the
    // flags can't derive from tensor presence — they ride the metadata.
    var it = std.mem.tokenizeScalar(u8, init_str, ',');
    while (it.next()) |tok| {
        const li = std.fmt.parseInt(usize, tok, 10) catch return error.DiskCacheCorruptSsm;
        if (li >= n_layers) return error.DiskCacheCorruptSsm;
        layers[li].initialized = true;
    }

    if (loaded.getMeta("grown")) |grown_s| {
        const k = std.fmt.parseInt(u32, grown_s, 10) catch return error.DiskCacheCorruptSsm;
        try self.joinGrownSegments(id, k, layers);
    }

    // Materialize with a CHECKED eval so a corrupt file surfaces HERE
    // (lazy Load reads data at eval), not mid-forward after install.
    {
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        var count: usize = 0;
        for (layers) |*l| {
            inline for (.{ l.conv_state, l.ssm_state, l.aux_state, l.qsa_pooled }) |arr| {
                if (arr.ctx != null) {
                    _ = mlx.mlx_vector_array_append_value(vec, arr);
                    count += 1;
                }
            }
        }
        if (count > 0) try mlx.check(mlx.mlx_eval(vec));
    }
    return cp;
}


/// False for a sidecar whose stamp is absent (written before stamps) or names other rows.
pub fn specStampMatches(loaded: *const safetensors.Loaded, prefix: []const u8, sm: SpecMeta) bool {
    var key: [8]u8 = undefined;
    var want: [48]u8 = undefined;
    const k = std.fmt.bufPrint(&key, "{s}.pos", .{prefix}) catch return false;
    const w = std.fmt.bufPrint(&want, "{d}:{d}", .{ sm.base, sm.step }) catch return false;
    const got = loaded.getMeta(k) orelse return false;
    return std.mem.eql(u8, got, w);
}


/// Load one persisted spec snapshot as a transient `KVCacheSnapshot` the
/// caller restores from and then deinits. Best-effort in every direction:
/// null when the entry has none, the recorded geometry doesn't fit the
/// target (layer count / quant config — `KVCache.restore` asserts equal
/// lengths, so the check lives here), or the file is unreadable. The
/// caller then starts blind, never wrong.
pub fn loadSpecSnap(
    self: *DiskTier,
    idx: usize,
    which: SpecKind,
    expected_layers: usize,
    target_config: kv_quant.KVQuantConfig,
) ?struct {
    snap: trunk.KVCacheSnapshot,
    base: usize,
    /// v5 in-checkpoint head half; null on every other entry.
    head_aux: ?trunk.SSMCacheEntrySnapshot = null,
    head_pos_base: c_int = 0,
    /// v8: the head's checkpoint-position leftovers. Empty on a v5..v7 sidecar, which
    /// clamps as it does today (a mid-block warm clamp declines, head starts blind).
    head_marks: trunk.QsaHeadMarkSet = .{},
} {
    const e = &self.entries.items[idx];
    const meta = (switch (which) {
        .dflash => e.spec_dflash,
        .mtp => e.spec_mtp,
    }) orelse return null;
    if (meta.layers != expected_layers) return null;
    if (!std.meta.eql(meta.quant, target_config)) {
        log.info("  [disk-cache] spec sidecar declined (quant mismatch)\n", .{});
        return null;
    }

    const rd = self.reading();
    defer rd.end();
    const cpu = rd.s;
    const path = std.fmt.allocPrint(self.allocator, "{s}/e{d}/spec.safetensors", .{ self.root, e.id }) catch return null;
    defer self.allocator.free(path);
    var loaded = self.fs.loadTensors(path, cpu) catch return null;
    defer loaded.deinit();
    if (which == .mtp and e.spec_segs.len > 0) {
        const ratio: u32 = if (meta.head) |h| @intCast(@max(h.ratio, 1)) else 1;
        const joined = self.joinSegments(&loaded, e.id, 'p', e.spec_segs.len, specGrowth, ratio, cpu) orelse {
            log.info("  [disk-cache] spec sidecar declined (a segment is unreadable)\n", .{});
            return null;
        };
        loaded.deinit();
        loaded = joined;
    }

    const prefix: []const u8 = switch (which) {
        .dflash => "d",
        .mtp => "m",
    };
    if (!specStampMatches(&loaded, prefix, meta)) {
        log.info("  [disk-cache] spec sidecar declined (stamp mismatch)\n", .{});
        return null;
    }
    const kinds: []const []const u8 = if (meta.quant.scheme == .off)
        &.{ "k", "v" }
    else
        &.{ "k", "v", "ks", "kb", "vs", "vb" };

    const entries = self.allocator.alloc(trunk.KVCacheEntry, expected_layers) catch return null;
    for (entries) |*en| en.* = trunk.newEmptyKVEntry();
    var snap: trunk.KVCacheSnapshot = .{
        .entries = entries,
        .step = meta.step,
        .allocator = self.allocator,
        .config = meta.quant,
    };
    var ok = true;
    var key_buf: [32]u8 = undefined;
    outer: for (entries, 0..) |*en, li| {
        for (kinds, 0..) |kind, ki| {
            const key = std.fmt.bufPrint(&key_buf, "{s}{d}.{s}", .{ prefix, li, kind }) catch {
                ok = false;
                break :outer;
            };
            const arr = loaded.take(key) orelse {
                if (ki == 0) continue :outer; // layer absent — stays uninitialized
                ok = false; // partial layer = corrupt
                break :outer;
            };
            // transfer the +1 handed by _get, replacing the empty handle
            switch (ki) {
                0 => {
                    _ = mlx.mlx_array_free(en.keys.array);
                    en.keys = mlx_pages.InPlace.adopt(arr, .paged);
                },
                1 => {
                    _ = mlx.mlx_array_free(en.values.array);
                    en.values = mlx_pages.InPlace.adopt(arr, .paged);
                },
                2 => {
                    _ = mlx.mlx_array_free(en.keys_scales.array);
                    en.keys_scales = mlx_pages.InPlace.adopt(arr, .paged);
                },
                3 => {
                    _ = mlx.mlx_array_free(en.keys_biases.array);
                    en.keys_biases = mlx_pages.InPlace.adopt(arr, .paged);
                },
                4 => {
                    _ = mlx.mlx_array_free(en.values_scales.array);
                    en.values_scales = mlx_pages.InPlace.adopt(arr, .paged);
                },
                5 => {
                    _ = mlx.mlx_array_free(en.values_biases.array);
                    en.values_biases = mlx_pages.InPlace.adopt(arr, .paged);
                },
                else => unreachable,
            }
        }
        en.offset = meta.step;
        en.initialized = true;
    }
    if (!ok) {
        snap.deinit();
        return null;
    }
    // Checked eval: a corrupt file surfaces its MLX error HERE (lazy Load
    // reads data at eval), not mid-forward after the restore.
    {
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        var count: usize = 0;
        for (entries) |*en| {
            if (!en.initialized) continue;
            _ = mlx.mlx_vector_array_append_value(vec, en.keys.array);
            _ = mlx.mlx_vector_array_append_value(vec, en.values.array);
            if (meta.quant.scheme != .off) {
                _ = mlx.mlx_vector_array_append_value(vec, en.keys_scales.array);
                _ = mlx.mlx_vector_array_append_value(vec, en.keys_biases.array);
                _ = mlx.mlx_vector_array_append_value(vec, en.values_scales.array);
                _ = mlx.mlx_vector_array_append_value(vec, en.values_biases.array);
            }
            count += 1;
        }
        if (count > 0) {
            mlx.check(mlx.mlx_eval(vec)) catch {
                snap.deinit();
                return null;
            };
        }
    }
    // v5 head half, best-effort: a pre-v5 sidecar returns none and the head declines the adoption.
    var head_aux: ?trunk.SSMCacheEntrySnapshot = null;
    var head_pos_base: c_int = 0;
    var head_marks: trunk.QsaHeadMarkSet = .{};
    if (meta.head) |hm| head: {
        const aux = getSpecArray(&loaded, prefix, "h.aux") orelse break :head;
        var snap_aux: trunk.SSMCacheEntrySnapshot = .{
            .conv_state = mlx.mlx_array_new(),
            .ssm_state = mlx.mlx_array_new(),
            .initialized = true,
            .aux_state = aux,
            .qsa_ratio = hm.ratio,
        };
        if (hm.pooled) {
            snap_aux.qsa_pooled = getSpecArray(&loaded, prefix, "h.pooled") orelse {
                trunk.ssmSnapshotDeinit(&snap_aux);
                break :head;
            };
        }
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        _ = mlx.mlx_vector_array_append_value(vec, snap_aux.aux_state);
        if (snap_aux.qsa_pooled.ctx != null) _ = mlx.mlx_vector_array_append_value(vec, snap_aux.qsa_pooled);
        mlx.check(mlx.mlx_eval(vec)) catch {
            trunk.ssmSnapshotDeinit(&snap_aux);
            break :head;
        };
        snap_aux.qsa_rows = hm.rows;
        head_aux = snap_aux;
        head_pos_base = hm.pos_base;
        for (hm.marks[0..hm.mark_count], 0..) |pos, mi| {
            var kind_buf: [16]u8 = undefined;
            const kind = std.fmt.bufPrint(&kind_buf, "h.lv{d}", .{mi}) catch break;
            const arr = getSpecArray(&loaded, prefix, kind) orelse break;
            const one = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(one);
            _ = mlx.mlx_vector_array_append_value(one, arr);
            mlx.check(mlx.mlx_eval(one)) catch {
                _ = mlx.mlx_array_free(arr);
                break;
            };
            head_marks.put(pos, arr);
        }
    }
    return .{ .snap = snap, .base = meta.base, .head_aux = head_aux, .head_pos_base = head_pos_base, .head_marks = head_marks };
}

/// `tail`'s tensors with each growing one's `n` segments of entry `id` put back in front of its rows, as a new
/// `Loaded` (preserving `tail`'s metadata) the caller owns. Null when a segment is unreadable or lacks a tensor.
pub fn joinSegments(
    self: *DiskTier,
    tail: *const safetensors.Loaded,
    id: u64,
    letter: u8,
    n: usize,
    growthOf: *const fn ([]const u8, u32) ?Growth,
    ratio: u32,
    cpu: mlx.mlx_stream,
) ?safetensors.Loaded {
    const segs = self.allocator.alloc(safetensors.Loaded, n) catch return null;
    defer self.allocator.free(segs);
    var loaded_n: usize = 0;
    defer for (segs[0..loaded_n]) |*m| m.deinit();
    for (segs, 0..) |*m, j| {
        const path = std.fmt.allocPrint(self.allocator, "{s}/e{d}/{c}{d:0>5}.safetensors", .{ self.root, id, letter, j }) catch return null;
        defer self.allocator.free(path);
        m.* = self.fs.loadTensors(path, cpu) catch return null;
        loaded_n += 1;
    }
    var out = safetensors.Loaded.init(self.allocator);
    var ok = false;
    defer if (!ok) out.deinit();
    {
        var mit = tail.meta.iterator();
        while (mit.next()) |e| out.putMeta(e.key_ptr.*, e.value_ptr.*) catch return null;
    }
    {
        var tit = tail.tensors.iterator();
        while (tit.next()) |e| {
            const k = e.key_ptr.*;
            const val = e.value_ptr.*;
            if (val.ctx == null or growthOf(k, ratio) != null) continue;
            out.putRetained(k, val) catch return null;
        }
    }
    if (n > 0) {
        var tit = segs[0].tensors.iterator();
        while (tit.next()) |e| {
            const k = e.key_ptr.*;
            const g = growthOf(k, ratio) orelse continue;
            const vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(vec);
            for (segs) |*m| {
                const part = m.get(k) orelse return null;
                _ = mlx.mlx_vector_array_append_value(vec, part);
            }
            if (tail.get(k)) |rest| _ = mlx.mlx_vector_array_append_value(vec, rest);
            var whole = mlx.mlx_array_new();
            mlx.check(mlx.mlx_concatenate_axis(&whole, vec, @intCast(g.axis), cpu)) catch {
                _ = mlx.mlx_array_free(whole);
                return null;
            };
            out.put(k, whole) catch {
                _ = mlx.mlx_array_free(whole);
                return null;
            };
        }
    }
    ok = true;
    return out;
}

pub fn getSpecArray(loaded: *safetensors.Loaded, prefix: []const u8, kind: []const u8) ?mlx.mlx_array {
    var key_buf: [32]u8 = undefined;
    const key = std.fmt.bufPrint(&key_buf, "{s}{s}", .{ prefix, kind }) catch return null;
    return loaded.take(key);
}

