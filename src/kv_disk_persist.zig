//! SSD prefix cache persistence: commit pieces, chunk/point/ring/spec serialization, held points, donor linking.

const std = @import("std");
const mlx = @import("mlx.zig");
const kv_quant = @import("kv_quant.zig");
const trunk = @import("trunk.zig");
const prefix_arch = @import("arch/prefix.zig");
const prefix_policy = @import("prefix_policy.zig");
const model = @import("model.zig");
const io_util = @import("io_util.zig");
const write_stats = @import("disk_write_stats.zig");
const log = @import("log.zig");
const disk_placement = @import("disk_placement.zig");
const disk_format = @import("disk_format.zig");
const vt = @import("prefix_vfs_tensors.zig");
const safetensors = @import("safetensors.zig");
const disk = @import("kv_disk_cache.zig");

const DiskTier = disk.DiskTier;
const IndexEntry = disk.IndexEntry;
const HeldPoint = disk.HeldPoint;
const RingFile = disk.RingFile;
const RingCommit = disk.RingCommit;
const SpecMeta = disk.SpecMeta;
const SpecHeadMeta = disk.SpecHeadMeta;
const SpecCommit = disk.SpecCommit;
const SideSeg = disk.SideSeg;
const PersistOutcome = disk.PersistOutcome;
const MediaLimits = disk.MediaLimits;
const RestoredRows = disk.RestoredRows;
const MIN_DISK_ADVANTAGE_TOKENS = disk.MIN_DISK_ADVANTAGE_TOKENS;
const MIN_PERSIST_TOKENS = disk.MIN_PERSIST_TOKENS;
const POINT_GRID = disk.POINT_GRID;
const RING_DISK_MAX_PER_ENTRY = disk.RING_DISK_MAX_PER_ENTRY;
const HELD_POINTS_MAX_BYTES = disk.HELD_POINTS_MAX_BYTES;
const HELD_POINTS_PERMIT_DIVISOR = disk.HELD_POINTS_PERMIT_DIVISOR;
const FILE_OVERHEAD_BYTES = disk.FILE_OVERHEAD_BYTES;
const TENSOR_OVERHEAD_BYTES = disk.TENSOR_OVERHEAD_BYTES;
const anyInitialized = disk.anyInitialized;
const persistTargetLen = disk.persistTargetLen;
const nbytesOf = disk.nbytesOf;
const mib = disk.mib;
const layerArrays = disk.layerArrays;
const rowBytes = disk.rowBytes;
const checkpointFileBytes = disk.checkpointFileBytes;
const specFileBytes = disk.specFileBytes;
const sliceAxis = disk.sliceAxis;
const nonChunkBytes = disk.nonChunkBytes;
const clampAdd = disk.clampAdd;
const commonPrefixLen = disk.commonPrefixLen;
const chunkShareEnabled = disk.chunkShareEnabled;

// ── Commit ──

/// Persist a cache state under `tokens`. Called on the inference thread
/// AFTER the response is finished (the write is bounded but synchronous).
/// Takes snapshot-shaped parts (entries + step + config) so callers can
/// flush either a live `KVCache` or a committed `KVCacheSnapshot` — the
/// hot cache flushes the RAM entry it just committed, post-markFinished,
/// so the client never waits on the SSD write. Skips ineligible states
/// silently; never fails the request.
pub fn appendCommit(
    self: *DiskTier,
    kv_entries: []const trunk.KVCacheEntry,
    step: usize,
    config: kv_quant.KVQuantConfig,
    tokens: []const u32,
    has_tools: bool,
    ssm_checkpoints: ?[]const trunk.SSMCheckpoint,
    s: mlx.mlx_stream,
) !PersistOutcome {
    return self.appendCommitWith(.{
        .kv_entries = kv_entries,
        .step = step,
        .config = config,
        .tokens = tokens,
        .has_tools = has_tools,
        .ssm_checkpoints = ssm_checkpoints,
    }, s, .{});
}

/// Everything one commit persists. A struct, not twelve positional parameters: the v10 media
/// identity would otherwise grow five signatures and every call site of each.
pub const CommitParams = struct {
    kv_entries: []const trunk.KVCacheEntry,
    step: usize,
    config: kv_quant.KVQuantConfig,
    tokens: []const u32,
    has_tools: bool,
    ssm_checkpoints: ?[]const trunk.SSMCheckpoint = null,
    dflash_snap: ?SpecCommit = null,
    mtp_snap: ?SpecCommit = null,
    ring: ?RingCommit = null,
    /// The media chain the rows were computed under: its last block's key and the first
    /// placeholder position. A record may claim only the blocks it covers
    /// (`prefix_cache.truncatedMediaKey`); 0 / null is a record with no media row.
    vision_key: u64 = 0,
    media_start: ?u32 = null,
    /// The request's media blocks (`prefix_policy.MediaSpan`): lets chunk and row-segment
    /// donors share every block this commit extends past, and narrows the stamp on a budget cut.
    media_chain: []const prefix_policy.MediaSpan = &.{},
    /// The SSD entry and rows the request's prefix was restored from: those rows are that entry's chunk bytes.
    restored: ?RestoredRows = null,
};

/// How much of the entry one call writes: by default a piece per call with the writer
/// finishing it, `whole` every piece now, `flush_bound` the write-through's one-chunk bound.
pub const CommitOpts = struct { whole: bool = false, flush_bound: ?u64 = null };

/// The one commit entry point; the positional wrappers above and below forward to it.
pub fn appendCommitWith(self: *DiskTier, p: CommitParams, s: mlx.mlx_stream, opts: CommitOpts) !PersistOutcome {
    if (opts.flush_bound) |bound| return self.appendCommitWithSpecBounded(p, s, bound);
    // With the writer armed the commit lands whole, piece by piece: each piece stages at most
    // `max_flush_bytes`, and the writer's permit bounds the host bytes in flight. Without it a
    // piece is a synchronous write, so one per commit bounds the stall.
    return self.appendCommitPieces(p, s, opts.whole or self.writer != null);
}

/// `appendCommit` with an explicit per-call flush bound (bytes): the loop stops after the
/// first chunk that crosses it. The write-through hook passes one byte (one chunk per boundary).
pub fn appendCommitBounded(
    self: *DiskTier,
    kv_entries: []const trunk.KVCacheEntry,
    step: usize,
    config: kv_quant.KVQuantConfig,
    tokens: []const u32,
    has_tools: bool,
    ssm_checkpoints: ?[]const trunk.SSMCheckpoint,
    s: mlx.mlx_stream,
    flush_bound: u64,
) !PersistOutcome {
    return self.appendCommitWith(.{
        .kv_entries = kv_entries,
        .step = step,
        .config = config,
        .tokens = tokens,
        .has_tools = has_tools,
        .ssm_checkpoints = ssm_checkpoints,
    }, s, .{ .flush_bound = flush_bound });
}

/// `appendCommit` plus the v4 spec snapshots (dflash assistant context /
/// MTP committed history). Eligibility is enforced UPSTREAM, same as the
/// RAM tier: the caller passes only what `commitWithState` was handed.
pub fn appendCommitWithSpec(
    self: *DiskTier,
    kv_entries: []const trunk.KVCacheEntry,
    step: usize,
    config: kv_quant.KVQuantConfig,
    tokens: []const u32,
    has_tools: bool,
    ssm_checkpoints: ?[]const trunk.SSMCheckpoint,
    dflash_snap: ?SpecCommit,
    mtp_snap: ?SpecCommit,
    s: mlx.mlx_stream,
) !PersistOutcome {
    return self.appendCommitWith(.{
        .kv_entries = kv_entries,
        .step = step,
        .config = config,
        .tokens = tokens,
        .has_tools = has_tools,
        .ssm_checkpoints = ssm_checkpoints,
        .dflash_snap = dflash_snap,
        .mtp_snap = mtp_snap,
    }, s, .{});
}

/// `appendCommitWithSpec` for a cache whose sliding layers ring: without `ring` such a
/// cache is skipped, since its chunks alone restore nothing.
pub fn appendCommitWithRing(
    self: *DiskTier,
    kv_entries: []const trunk.KVCacheEntry,
    step: usize,
    config: kv_quant.KVQuantConfig,
    tokens: []const u32,
    has_tools: bool,
    ssm_checkpoints: ?[]const trunk.SSMCheckpoint,
    dflash_snap: ?SpecCommit,
    mtp_snap: ?SpecCommit,
    ring: ?RingCommit,
    s: mlx.mlx_stream,
) !PersistOutcome {
    return self.appendCommitWith(.{
        .kv_entries = kv_entries,
        .step = step,
        .config = config,
        .tokens = tokens,
        .has_tools = has_tools,
        .ssm_checkpoints = ssm_checkpoints,
        .dflash_snap = dflash_snap,
        .mtp_snap = mtp_snap,
        .ring = ring,
    }, s, .{});
}

/// `appendCommitWithRing` that writes pieces until the entry is whole, writer or not: its
/// caller's source is gone when it returns, so nothing could resume a partial entry.
pub fn appendCommitWhole(
    self: *DiskTier,
    kv_entries: []const trunk.KVCacheEntry,
    step: usize,
    config: kv_quant.KVQuantConfig,
    tokens: []const u32,
    has_tools: bool,
    ssm_checkpoints: ?[]const trunk.SSMCheckpoint,
    dflash_snap: ?SpecCommit,
    mtp_snap: ?SpecCommit,
    ring: ?RingCommit,
    s: mlx.mlx_stream,
) !PersistOutcome {
    return self.appendCommitWith(.{
        .kv_entries = kv_entries,
        .step = step,
        .config = config,
        .tokens = tokens,
        .has_tools = has_tools,
        .ssm_checkpoints = ssm_checkpoints,
        .dflash_snap = dflash_snap,
        .mtp_snap = mtp_snap,
        .ring = ring,
    }, s, .{ .whole = true });
}

pub fn appendCommitPieces(self: *DiskTier, p: CommitParams, s: mlx.mlx_stream, whole: bool) !PersistOutcome {
    var out = try self.appendCommitWithSpecBounded(p, s, self.max_flush_bytes);
    if (!whole) return out;
    const first = self.last_store_id orelse return out;
    var pieces: usize = 1;
    while (out == .partial and pieces < disk_placement.MAX_FLUSH_PIECES) : (pieces += 1) {
        const before = self.storeProgress(first) orelse break;
        out = try self.appendCommitWithSpecBounded(p, s, self.max_flush_bytes);
        // A piece that landed nothing, or an entry the budget evicted, ends the flush.
        if (self.last_store_id != first) break;
        const after = self.storeProgress(first) orelse break;
        if (std.meta.eql(before, after)) break;
    }
    return out;
}

/// What an entry holds on disk so far: its flushed rows and its checkpoint and ring files.
pub fn storeProgress(self: *const DiskTier, id: u64) ?[3]usize {
    for (self.entries.items) |*e| {
        if (e.id == id) return .{ e.kv_len, e.ssm_positions.len, e.rings.len };
    }
    return null;
}

/// The longest restorable record of `kv_entries` that fits the tier's byte budget on its own:
/// `kv_target` when it fits, else the highest checkpoint or ring restore point that does (any
/// length when the cache has neither). Null when nothing of `MIN_PERSIST_TOKENS` or more fits.
pub fn budgetTarget(
    self: *const DiskTier,
    kv_entries: []const trunk.KVCacheEntry,
    kv_target: u32,
    cps: ?[]const trunk.SSMCheckpoint,
    ring_srcs: []const RingSource,
    spec_bytes: u64,
) ?u32 {
    if (self.max_bytes == 0 or self.ssm_budget != 0) return kv_target;
    if (self.entryBytesAt(kv_entries, kv_target, cps, ring_srcs, spec_bytes) <= self.max_bytes) return kv_target;
    var best: u32 = 0;
    if (ring_srcs.len > 0) {
        for (ring_srcs) |src| {
            if (src.pos < kv_target and src.pos > best and self.entryBytesAt(kv_entries, src.pos, cps, ring_srcs, 0) <= self.max_bytes) best = src.pos;
        }
    } else if (cps != null and cps.?.len > 0) {
        for (cps.?) |*cp| {
            const p: u32 = @intCast(cp.pos);
            if (p > best and p < kv_target and self.entryBytesAt(kv_entries, p, cps, ring_srcs, 0) <= self.max_bytes) best = p;
        }
    } else {
        var hi: u32 = kv_target;
        while (hi - best > 1) {
            const mid = best + (hi - best) / 2;
            if (self.entryBytesAt(kv_entries, mid, cps, ring_srcs, 0) <= self.max_bytes) best = mid else hi = mid;
        }
    }
    return if (best >= MIN_PERSIST_TOKENS) best else null;
}

/// What the checkpoint files of a record over the tier's budget may take once only its oldest points
/// go: the budget less everything but the points, when the newest point and the newest grid point
/// alone fit. Zero when the record fits whole, has no checkpoints, or its rows alone overflow
/// (`budgetTarget` then cuts its tail).
pub fn pointBudget(
    self: *const DiskTier,
    kv_entries: []const trunk.KVCacheEntry,
    kv_target: u32,
    cps_opt: ?[]const trunk.SSMCheckpoint,
    ring_srcs: []const RingSource,
    spec_bytes: u64,
) u64 {
    const cps = cps_opt orelse return 0;
    if (self.max_bytes == 0 or cps.len == 0 or ring_srcs.len > 0) return 0;
    if (self.entryBytesAt(kv_entries, kv_target, cps, ring_srcs, spec_bytes) <= self.max_bytes) return 0;
    var newest: ?*const trunk.SSMCheckpoint = null;
    var grid: ?*const trunk.SSMCheckpoint = null;
    for (cps) |*cp| {
        if (cp.pos == 0 or cp.pos > kv_target) continue;
        if (newest == null or cp.pos > newest.?.pos) newest = cp;
        if (cp.pos % POINT_GRID == 0 and (grid == null or cp.pos > grid.?.pos)) grid = cp;
    }
    const top = newest orelse return 0;
    var kept: [2]trunk.SSMCheckpoint = undefined;
    var n: usize = 0;
    var pts: u64 = 0;
    for ([_]?*const trunk.SSMCheckpoint{ grid, top }) |opt| {
        const cp = opt orelse continue;
        if (n == 1 and kept[0].pos == cp.pos) continue;
        kept[n] = cp.*;
        n += 1;
        pts += self.pointFileBytes(cp);
    }
    const with = self.entryBytesAt(kv_entries, kv_target, kept[0..n], ring_srcs, spec_bytes);
    if (with > self.max_bytes) return 0;
    return self.max_bytes - (with - pts);
}

/// What `cp`'s file occupies: whole, or with a split tier its fixed arrays and the rows past its last whole span.
pub fn pointFileBytes(self: *const DiskTier, cp: *const trunk.SSMCheckpoint) u64 {
    const whole = checkpointFileBytes(cp);
    const g = self.grown orelse return whole;
    return whole -| g.row_bytes *| cp.pos +| g.row_bytes *| (cp.pos % g.span);
}

/// What a record of rows [0, p) of `kv_entries` occupies with no chunk shared: its chunks, the
/// checkpoints and ring files the entry keeps at or below `p`, the token record, the manifest
/// and `extra` (its spec sidecar). Headers are an allowance, not an exact count.
pub fn entryBytesAt(
    self: *const DiskTier,
    kv_entries: []const trunk.KVCacheEntry,
    p: u32,
    cps: ?[]const trunk.SSMCheckpoint,
    ring_srcs: []const RingSource,
    extra: u64,
) u64 {
    var row: u64 = 0;
    var tensors: u64 = 0;
    for (kv_entries) |*e| {
        if (!e.initialized or e.ringed) continue;
        for (layerArrays(e)) |a| {
            const rb = rowBytes(a);
            if (rb == 0) continue;
            row += rb;
            tensors += 1;
        }
    }
    const chunks: u64 = (@as(u64, p) + self.chunk_tokens - 1) / self.chunk_tokens;
    var total: u64 = @as(u64, p) * (row + 4) + chunks * (FILE_OVERHEAD_BYTES + tensors * TENSOR_OVERHEAD_BYTES) + FILE_OVERHEAD_BYTES + extra;
    var cp_bytes: u64 = 0;
    var cp_count: u64 = 0;
    var top: u64 = 0;
    if (cps) |list| for (list) |*cp| {
        if (cp.pos == 0 or cp.pos > p) continue;
        cp_bytes += self.pointFileBytes(cp);
        cp_count += 1;
        top = @max(top, cp.pos);
    };
    // Retention keeps at most `ssm_max_per_entry` of them.
    if (cp_count > self.ssm_max_per_entry) cp_bytes = cp_bytes * self.ssm_max_per_entry / cp_count;
    total += cp_bytes;
    if (self.grown) |g| {
        var arrays: u64 = 0;
        for (g.divs) |d| arrays += @intFromBool(d != 0);
        const segs = top / g.span;
        total += segs * (g.span * g.row_bytes + FILE_OVERHEAD_BYTES + 2 * arrays * TENSOR_OVERHEAD_BYTES);
    }
    for (ring_srcs) |src| {
        if (src.pos <= p) total += ringFileBytes(src);
    }
    return total;
}

pub fn appendCommitWithSpecBounded(self: *DiskTier, p: CommitParams, s: mlx.mlx_stream, flush_bound: u64) !PersistOutcome {
    self.lost_points.clearRetainingCapacity();
    const kv_entries = p.kv_entries;
    const step = p.step;
    const config = p.config;
    const all_tokens = p.tokens;
    const has_tools = p.has_tools;
    const ssm_checkpoints = p.ssm_checkpoints;
    const ring = p.ring;
    var media: CommitMedia = .{ .key = p.vision_key, .start = p.media_start };
    var tokens = all_tokens;
    var dflash_snap = p.dflash_snap;
    var mtp_snap = p.mtp_snap;
    // Every production caller swallows our error (the disk tier is best-effort end to
    // end), and three writers inside can raise: persistSsmCheckpoints, writeChunkFile,
    // appendSsmOnly. Drop the latch THIS call raised at the one funnel they all pass
    // through — otherwise the next decode tick's checkErrorDecode charges it to an
    // unrelated request. `writeSpecSidecar` swallows internally, so it keeps its own pair.
    const had_error = mlx.errorPending();
    errdefer mlx.dropLatchedErrorUnless(had_error);
    self.last_store_id = null;
    const waited_before: u64 = if (self.writer) |w| w.waitedNs() else 0;
    // On EOS-terminated turns the cache runs 1-2 positions AHEAD of the
    // committed token record (forwarded terminator tokens that never
    // land in `tokens`). Persist the prefix covered by the record —
    // positions beyond it are unusable for matching anyway.
    // The KV extent is the initialized layers' offset, NOT `step`: on
    // hybrid archs (GDN) `cache.step` only bumps on layer 0,
    // which is a GatedDeltaNet layer that never writes KV, so it stays 0
    // while the full-attention layers carry offset == prompt position.
    // `max(step, max initialized offset)` is correct for both — equal on
    // pure attention, and the layer offset on hybrid.
    // Anything the writer lost since the last commit is attributed before the index is read.
    _ = self.harvestWriteFailures();
    self.dropPoisonedEntries();

    // An arch whose points hold its whole state persists them over its tokens with no KV rows (`points_hold_state`).
    const state_only = self.points_hold_state and !anyInitialized(kv_entries) and ssm_checkpoints != null and ssm_checkpoints.?.len > 0;
    const kv_target_u: usize = if (state_only) @min(step, tokens.len) else persistTargetLen(kv_entries, step, tokens.len);
    if (kv_target_u < MIN_PERSIST_TOKENS) return .skipped;
    var kv_target: u32 = @intCast(kv_target_u);
    // Every initialized layer must cover the persisted range with B == 1
    // — anything else (mid-spec-decode state, batched cache) is not a
    // persistable snapshot. A ringed layer covers it through ring files instead.
    var ringed = false;
    for (kv_entries) |*entry| {
        if (!entry.initialized) continue;
        if (entry.ringed) {
            ringed = true;
            continue;
        }
        if (entry.offset < kv_target_u) {
            log.debug("  [disk-cache] skip: layer offset {d} < kv_len {d}\n", .{ entry.offset, kv_target_u });
            return .skipped;
        }
        const shape = mlx.getShape(entry.keys.array);
        if (shape.len != 4 or shape[0] != 1) {
            log.debug("  [disk-cache] skip: non-B1 cache shape\n", .{});
            return .skipped;
        }
    }

    var ring_srcs: []const RingSource = if (ringed) blk: {
        const rc = ring orelse return .skipped;
        break :blk try self.ringSources(kv_entries, kv_target, rc);
    } else &.{};
    defer self.allocator.free(ring_srcs);
    if (ringed and ring_srcs.len == 0) {
        log.debug("  [disk-cache] skip: no ring restore point at or below {d}\n", .{kv_target});
        return .skipped;
    }

    const spec_bytes = specFileBytes(dflash_snap) + specFileBytes(mtp_snap);
    self.ssm_budget = self.pointBudget(kv_entries, kv_target, ssm_checkpoints, ring_srcs, spec_bytes);

    // Superseded check: an existing entry that already covers `tokens`
    // (same key, tokens is a prefix of its tokens, kv already >= ours)
    // makes this commit a no-op — UNLESS the entry is hybrid and still has
    // pending SSM checkpoints (byte-capped across turns), which take a
    // dedicated SSM-only append path (the KV chunks are all present, so
    // the extend machinery would pointlessly rewrite the tail chunk).
    var target = self.selectStoreTarget(tokens, kv_target, has_tools, config, media, ssm_checkpoints, dflash_snap, mtp_snap, ring_srcs);
    if (target.superseded) return .persisted;
    // Re-derive the budget before every store, and only a store: the probe is slow and the idle
    // spill reaches this point once per idle entry. It also gates the SSM-only append.
    if (self.ssd_first) {
        self.refreshDiskBudget();
        self.ssm_budget = self.pointBudget(kv_entries, kv_target, ssm_checkpoints, ring_srcs, spec_bytes);
        // Its eviction swap-removes: indices picked above may now name other entries.
        target = self.selectStoreTarget(tokens, kv_target, has_tools, config, media, ssm_checkpoints, dflash_snap, mtp_snap, ring_srcs);
        if (target.superseded) return .persisted;
    }
    // A sole entry never outgrows the tier: its oldest points go first (`ssm_budget`); a record whose rows
    // alone overflow is cut to the highest restore point that fits, without its spec state (that
    // describes the whole record).
    const fit = self.budgetTarget(kv_entries, kv_target, ssm_checkpoints, ring_srcs, spec_bytes) orelse {
        log.info("  [disk-cache] skip: no restore point of {d} tokens fits the {d:.1} MB budget\n", .{ kv_target, mib(self.max_bytes) });
        return .skipped;
    };
    if (fit < kv_target) {
        log.info("  [disk-cache] {d}-token record over the {d:.1} MB budget: persisting {d} tokens\n", .{ kv_target, mib(self.max_bytes), fit });
        kv_target = fit;
        tokens = all_tokens[0..fit];
        const cut_media = prefix_policy.truncatedMediaKey(p.media_chain, media.key, if (media.start) |ms| @as(usize, ms) else null, fit);
        media = .{ .key = cut_media.key, .start = cut_media.start };
        dflash_snap = null;
        mtp_snap = null;
        if (ringed) {
            const srcs = try self.ringSources(kv_entries, kv_target, ring.?);
            self.allocator.free(ring_srcs);
            ring_srcs = srcs;
        }
        target = self.selectStoreTarget(tokens, kv_target, has_tools, config, media, ssm_checkpoints, dflash_snap, mtp_snap, ring_srcs);
        if (target.superseded) return .persisted;
    }
    const extend_idx = target.extend_idx;
    const ssm_only_idx = target.ssm_only_idx;
    if (self.store_declined) return .skipped;
    if (ssm_only_idx) |i| return self.appendSsmOnly(i, ssm_checkpoints, dflash_snap, mtp_snap, ring_srcs, s);

    const sw = io_util.Stopwatch.init(self.io);

    const id: u64 = if (extend_idx) |i| self.entries.items[i].id else try self.claimEntryId();
    const dir_rel = try std.fmt.allocPrint(self.allocator, "{s}/e{d}", .{ self.root, id });
    defer self.allocator.free(dir_rel);
    try self.fs.mkdirp(dir_rel);

    // Chunks [0, keep) are full chunks already on disk from the entry we
    // extend; everything from `keep` on (the old partial tail + the new
    // positions) is (re)written — up to the per-flush byte cap. Stopping
    // early lands on a full-chunk boundary; the entry then records the
    // shorter kv_len and the NEXT flush resumes from there.
    //
    // A fresh entry may inherit its leading whole chunks from a resident entry that shares a
    // prefix, by hard link: a persisted entry's tokens are `prompt ++ generated`, so the next
    // turn diverges inside the generated span and used to rewrite every chunk. SSD-first only.
    const old_kv: u32 = if (extend_idx) |i| self.entries.items[i].kv_len else 0;
    const donor = if (extend_idx == null) self.chunkShareDonor(tokens, kv_target, has_tools, config, media, p.media_chain, null) else null;
    var inherited: u32 = if (extend_idx) |i| self.entries.items[i].inherited_chunks else if (donor) |d| d.chunks else 0;
    var keep: u32 = if (extend_idx != null) old_kv / self.chunk_tokens else inherited;
    const n_chunks: u32 = @intCast((@as(u64, kv_target) + self.chunk_tokens - 1) / self.chunk_tokens);

    var chunk_sizes = std.ArrayList(u64).empty;
    errdefer chunk_sizes.deinit(self.allocator);
    if (extend_idx) |i| {
        const old_cb = self.entries.items[i].chunk_bytes;
        try chunk_sizes.appendSlice(self.allocator, old_cb[0..@min(keep, old_cb.len)]);
    } else if (donor) |d| {
        // Only the donor's landed chunks are linked (a contiguous prefix); may be zero.
        const linked = self.linkInheritedChunks(d, id, 0, &chunk_sizes) catch 0;
        if (linked == 0) chunk_sizes.clearRetainingCapacity();
        inherited = linked;
        keep = linked;
    }
    const donor_linked = inherited;
    // Past the donor's share, a whole chunk below the restored rows holds the bytes this commit would write:
    // the slot read them from that file. An image turn's donor lends only the text before the first image.
    if (extend_idx == null) if (self.restoredSource(p.restored, tokens, kv_target, config)) |src| {
        if (src.chunks > inherited) {
            inherited += self.linkInheritedChunks(src, id, inherited, &chunk_sizes) catch 0;
            keep = inherited;
        }
    };
    var inherited_qsa = if (extend_idx) |i| self.entries.items[i].inherited_qsa else false;
    var qsa_own: []const SideSeg = if (extend_idx) |i| self.entries.items[i].qsa_segs else &.{};
    var qsa_own_inherited: u32 = if (extend_idx) |i| self.entries.items[i].qsa_segs_inherited else 0;
    var inherited_qsa_rows: u32 = if (extend_idx) |i| self.entries.items[i].qsa_history_rows else 0;
    var inherited_qsa_bytes: u64 = if (extend_idx) |i| self.entries.items[i].qsa_history_bytes else 0;
    // A real check: a rewrite must never land on a link (the sync arm truncates in place).
    if (keep < inherited) {
        self.unlinkChunkRange(id, keep, inherited);
        log.warn("  [disk-cache] chunk share: e{d} would rewrite an inherited chunk (keep {d} < inherited {d}) — writing from {d} instead\n", .{ id, keep, inherited, keep });
        inherited = keep;
        if (chunk_sizes.items.len > keep) chunk_sizes.shrinkRetainingCapacity(keep);
    }
    if (donor) |d| {
        if (donor_linked > 0 and self.linkInheritedQsa(d, id)) {
            inherited_qsa = true;
            inherited_qsa_rows = self.entries.items[d.idx].qsa_history_rows;
            inherited_qsa_bytes = self.entries.items[d.idx].qsa_history_bytes;
            qsa_own = self.entries.items[d.idx].qsa_segs;
            qsa_own_inherited = @intCast(qsa_own.len);
        }
    }

    // Phase 3 FIRST: checkpoints come off the TOP of the flush budget —
    // chunk-first budgeting starved them to ZERO on turns appending ≥
    // the cap in chunks (live 2026-09-07: cancel-salvage retries at
    // +16 chunks ≈ 544 MB vs a 512 MB cap left no checkpoint budget,
    // ever; every entry of the Sep-4 disk wave landed KV-only and
    // unrestorable). Their share is capped at half the budget so chunk
    // progress never stalls entirely. The eligibility bound is the
    // TARGET length, not the chunk progress: a checkpoint beyond the
    // chunks this flush reaches is still written (position-keyed,
    // immutable) and becomes restorable when a later flush extends
    // kv_len past it.
    const old_ssm_pos: []const u32 = if (extend_idx) |i| self.entries.items[i].ssm_positions else &[_]u32{};
    const old_ssm_bytes: []const u64 = if (extend_idx) |i| self.entries.items[i].ssm_bytes else &[_]u64{};
    var written_bytes: u64 = 0;
    var linked_grown = std.ArrayList(u64).empty;
    defer linked_grown.deinit(self.allocator);
    const seg_donor = if (extend_idx == null) self.segmentDonor(tokens, kv_target, has_tools, config, media, p.media_chain) else null;
    if (seg_donor) |d| self.linkInheritedSegments(d, id, ssm_checkpoints, kv_target, 0, &linked_grown) catch |err| {
        log.warn("  [disk-cache] segment share: e{d} writes its own row segments: {s}\n", .{ id, @errorName(err) });
    };
    // Past the donor's share, whole row segments below the restored rows came straight from that entry.
    if (extend_idx == null) if (self.restoredSegmentSource(p.restored, tokens, kv_target, config)) |src| {
        const have: u32 = @intCast(linked_grown.items.len);
        if (src.chunks > have) self.linkInheritedSegments(src, id, ssm_checkpoints, kv_target, have, &linked_grown) catch |err| {
            log.warn("  [disk-cache] segment share (restored): e{d} writes its own row segments: {s}\n", .{ id, @errorName(err) });
        };
    };
    const inherited_grown: u32 = if (extend_idx) |i| self.entries.items[i].inherited_grown else @intCast(linked_grown.items.len);
    const old_grown: []const u64 = if (extend_idx) |i| self.entries.items[i].grown else linked_grown.items;
    const old_span: u32 = if (extend_idx) |i| self.entries.items[i].grown_span else if (linked_grown.items.len > 0) self.grown.?.span else 0;
    var ssm_res = self.persistSsmCheckpoints(id, dir_rel, kv_target, old_ssm_pos, old_ssm_bytes, old_grown, old_span, ssm_checkpoints, &written_bytes, s, self.max_flush_bytes / 2, tokens) catch |err| {
        // errdefer already owns chunk_sizes + ssm_res; a manual deinit here
        // would double-free (the fault-injection test segfaulted on exactly this).
        return err;
    };
    errdefer ssm_res.deinit(self.allocator);
    // Ring files are a few MB each and the only way the entry restores: outside the bound.
    const old_rings: []const RingFile = if (extend_idx) |i| self.entries.items[i].rings else &.{};
    const rings = try self.persistRings(id, dir_rel, config, old_rings, ring_srcs, &written_bytes, s);
    errdefer self.allocator.free(rings);

    // Chunks [0, keep) are full chunks already on disk (see above); the
    // rewrite runs from `keep` up to the REMAINDER of the flush bound —
    // the checkpoints above already came off the top of it.
    var chunk_i: u32 = keep;
    while (chunk_i < n_chunks) : (chunk_i += 1) {
        if (written_bytes >= flush_bound and chunk_i > keep) break;
        const c0: u32 = chunk_i * self.chunk_tokens;
        const c1: u32 = @intCast(@min(@as(u64, c0) + self.chunk_tokens, kv_target));
        const csize = try self.writeChunkFile(kv_entries, config, dir_rel, chunk_i, c0, c1, s);
        written_bytes += csize;
        try chunk_sizes.append(self.allocator, csize);
    }
    const chunks_done: u32 = chunk_i;
    const chunk_complete = chunks_done == n_chunks;
    const kv_len: u32 = if (chunk_complete) kv_target else chunks_done * self.chunk_tokens;
    if (kv_len <= old_kv and std.mem.eql(u32, ssm_res.positions, old_ssm_pos) and ringsEqual(rings, old_rings)) {
        // Cap so tight nothing new landed — nothing to commit (a
        // checkpoint-only write still counts as progress).
        chunk_sizes.deinit(self.allocator);
        ssm_res.deinit(self.allocator);
        self.allocator.free(rings);
        return if (chunk_complete) .persisted else .partial;
    }

    const prefix_rows: u32 = if (donor) |d| @min(d.shared, kv_len) else kv_len;
    if (inherited_qsa) inherited_qsa_rows = @min(inherited_qsa_rows, prefix_rows);
    // An extended entry may hold fewer sidecar segments than the record it grew from: a prefill write-through
    // stores none. A segment links only on an equal hash, so any entry sharing the prefix may lend.
    const side_donor = donor orelse if (extend_idx) |x| self.chunkShareDonor(tokens, kv_target, has_tools, config, media, p.media_chain, x) else null;
    const qsa_prev: SegPrev = .{
        .id = id,
        .own = qsa_own,
        .own_inherited = qsa_own_inherited,
        .donor_id = if (side_donor) |d| d.id else null,
        .donor = if (side_donor) |d| self.entries.items[d.idx].qsa_segs else &.{},
    };
    const qsa_res = try self.persistQsaHistory(dir_rel, ssm_checkpoints, inherited_qsa, inherited_qsa_rows, inherited_qsa_bytes, prefix_rows, s, qsa_prev);
    errdefer self.allocator.free(qsa_res.segs);
    written_bytes += qsa_res.written;
    const complete = chunk_complete and ssm_res.complete;

    // v4 spec snapshots — one sidecar file, REPLACED wholesale by every
    // commit (a commit with no payload deletes a stale one, the RAM
    // tier's supersede rule), by the piece that completes its rows; an
    // earlier piece keeps what the entry had. Best-effort DRAFT-side
    // state: a failed write costs the entry its spec, never the entry.
    const spec_prev: SegPrev = .{
        .id = id,
        .own = if (extend_idx) |i| self.entries.items[i].spec_segs else &.{},
        .own_inherited = if (extend_idx) |i| self.entries.items[i].spec_segs_inherited else 0,
        .donor_id = if (side_donor) |d| d.id else null,
        .donor = if (side_donor) |d| self.entries.items[d.idx].spec_segs else &.{},
    };
    const spec_res: SpecSidecarResult = if (!chunk_complete) (if (extend_idx) |i| .{
        .bytes = self.entries.items[i].spec_bytes,
        .dflash = self.entries.items[i].spec_dflash,
        .mtp = self.entries.items[i].spec_mtp,
        .segs = try self.allocator.dupe(SideSeg, self.entries.items[i].spec_segs),
        .segs_inherited = self.entries.items[i].spec_segs_inherited,
    } else .{}) else self.writeSpecSidecar(dir_rel, dflash_snap, mtp_snap, s, spec_prev) catch |err| blk: {
        log.warn("  [disk-cache] spec persist failed: {s} — entry keeps no spec\n", .{@errorName(err)});
        break :blk .{};
    };
    errdefer self.allocator.free(spec_res.segs);
    written_bytes += spec_res.written;

    // Token record — the LONGER of the existing record and this commit's
    // tokens (a resumed incremental flush must not shrink the record its
    // earlier chunks were committed against). An extension appends only its new tail: the
    // manifest still on disk names the old count, and the scan reads a longer record at it.
    const record: []const u32 = if (extend_idx) |i| blk: {
        const et = self.entries.items[i].tokens;
        break :blk if (et.len >= tokens.len) et else tokens;
    } else tokens;
    if (extend_idx == null or record.ptr == tokens.ptr) {
        const tpath = try std.fmt.allocPrint(self.allocator, "{s}/tokens.bin", .{dir_rel});
        defer self.allocator.free(tpath);
        const have: usize = if (extend_idx) |i| self.entries.items[i].tokens.len else 0;
        const from: usize = if (have * 4 <= (self.fs.size(tpath) orelse 0)) have else 0;
        errdefer self.stats.recordFailed((record.len - from) * 4);
        // Bytes past the record (a torn or stale tail) would never be billed or swept.
        try self.fs.writeAt(tpath, from == 0, from * 4, std.mem.sliceAsBytes(record[from..]), record.len * 4);
        self.stats.recordWritten(.meta, (record.len - from) * 4);
    }

    // `bytes` is what this entry created on disk: inherited (linked) chunks bill 0.
    var non_chunk: u64 = @as(u64, record.len) * 4 + spec_res.bytes;
    for (spec_res.segs[spec_res.segs_inherited..]) |sg| non_chunk += sg.bytes;
    for (qsa_res.segs[qsa_res.segs_inherited..]) |sg| non_chunk += sg.bytes;
    for (ssm_res.bytes) |b| non_chunk += b;
    for (ssm_res.grown[inherited_grown..]) |b| non_chunk += b;
    for (rings) |r| non_chunk += r.bytes;
    if (!qsa_res.inherited) non_chunk += qsa_res.bytes;
    var bytes: u64 = non_chunk;
    for (chunk_sizes.items[@min(inherited, chunk_sizes.items.len)..]) |b| bytes += b;

    var new_entry: IndexEntry = .{
        .id = id,
        .tokens = try self.allocator.dupe(u32, record),
        .kv_len = kv_len,
        .has_tools = has_tools,
        .quant = config,
        .bytes = bytes,
        .chunk_bytes = try chunk_sizes.toOwnedSlice(self.allocator),
        .inherited_chunks = inherited,
        .ssm_positions = ssm_res.positions,
        .ssm_bytes = ssm_res.bytes,
        .grown = ssm_res.grown,
        .grown_span = ssm_res.grown_span,
        .inherited_grown = inherited_grown,
        .spec_bytes = spec_res.bytes,
        .spec_dflash = spec_res.dflash,
        .spec_mtp = spec_res.mtp,
        .spec_segs = spec_res.segs,
        .spec_segs_inherited = spec_res.segs_inherited,
        .qsa_history_bytes = if (qsa_res.inherited) inherited_qsa_bytes else qsa_res.bytes,
        .qsa_history_rows = if (qsa_res.inherited) inherited_qsa_rows else qsa_res.rows,
        .inherited_qsa = qsa_res.inherited,
        .qsa_segs = qsa_res.segs,
        .qsa_segs_inherited = qsa_res.segs_inherited,
        .rings = rings,
        .vision_key = media.key,
        .media_start = media.start,
        .last_used = self.bump(),
    };
    errdefer {
        self.allocator.free(new_entry.tokens);
        self.allocator.free(new_entry.chunk_bytes);
    }

    if (extend_idx) |i| {
        const e = &self.entries.items[i];
        // An extension's delta is file-based; `e.bytes` after a `scan` may include chunks
        // the manifest lists as inherited (their donor died first), so it is carried forward.
        var delta: i64 = @as(i64, @intCast(non_chunk)) - @as(i64, @intCast(nonChunkBytes(e)));
        for (new_entry.chunk_bytes[@min(keep, new_entry.chunk_bytes.len)..]) |b| delta += @as(i64, @intCast(b));
        if (e.chunk_bytes.len > keep) delta -= @as(i64, @intCast(e.chunk_bytes[keep]));
        new_entry.bytes = clampAdd(e.bytes, delta);
        // meta.json is the commit point: written last, atomically.
        try self.writeMeta(new_entry);
        self.total_bytes = clampAdd(self.total_bytes, delta);
        // ssm_positions/ssm_bytes ownership moved into new_entry.ssm_res —
        // free only the fields NOT carried forward.
        self.allocator.free(e.tokens);
        self.allocator.free(e.chunk_bytes);
        self.allocator.free(e.ssm_positions);
        self.allocator.free(e.ssm_bytes);
        self.allocator.free(e.grown);
        self.allocator.free(e.rings);
        self.allocator.free(e.spec_segs);
        self.allocator.free(e.qsa_segs);
        e.* = new_entry;
    } else {
        // meta.json is the commit point: written last, atomically.
        try self.writeMeta(new_entry);
        try self.entries.append(self.allocator, new_entry);
        self.total_bytes += new_entry.bytes;
    }
    self.last_store_id = id;
    self.settleHeld(s);
    self.enforceBudget();
    const listed: usize = for (self.entries.items) |e| {
        if (e.id == id) break e.ssm_positions.len;
    } else 0;

    self.last_persist_bytes = written_bytes;
    const wrote_mb = @as(f64, @floatFromInt(written_bytes)) / (1024.0 * 1024.0);
    const ms: u64 = sw.read() / std.time.ns_per_ms;
    const waited_ms: u64 = if (self.writer) |w| (w.waitedNs() -| waited_before) / std.time.ns_per_ms else 0;
    log.info("  [disk-cache] persisted {d}/{d} tokens (+{d} chunks, {d} ssm-cp, {d:.1} MB, {d}ms, {d}ms waiting on the writer); on SSD={d:.1} MB ({d} entries)\n", .{
        kv_len,                                                        kv_target,              chunks_done - keep, listed, wrote_mb, ms, waited_ms,
        @as(f64, @floatFromInt(self.total_bytes)) / (1024.0 * 1024.0), self.entries.items.len,
    });
    // The one completion marker: every chunk and every wanted checkpoint is staged. A
    // bounded flush leaves it out and `disk_dirty` set. A write that already failed for this
    // entry makes `.persisted` a promise nothing can keep.
    var whole = complete;
    if (whole) {
        _ = self.harvestWriteFailures();
        if (self.entryPoisoned(id)) whole = false;
    }
    if (whole) {
        log.info("  [disk-cache] e{d} complete on disk: {d} tokens, {d} chunks, {d} ssm-cp\n", .{ id, kv_len, new_entry.chunk_bytes.len, listed });
    }
    return if (whole) .persisted else .partial;
}

/// Does the index agree that this tier holds a complete, restorable copy of `tokens`? Two
/// bars before a RAM entry may be discarded: the token record covers `tokens` at the same
/// key with `kv_len` reaching the persist target, and `chunk_bytes` has one non-zero entry
/// per implied chunk.
pub fn holdsFullPrefix(
    self: *const DiskTier,
    kv_entries: []const trunk.KVCacheEntry,
    step: usize,
    tokens: []const u32,
    has_tools: bool,
    config: kv_quant.KVQuantConfig,
) bool {
    return self.fullPrefixEntryId(kv_entries, step, tokens, has_tools, config) != null;
}

/// `holdsFullPrefix`, returning the entry's id so the caller can ask `entryWritesPending`.
pub fn fullPrefixEntryId(
    self: *const DiskTier,
    kv_entries: []const trunk.KVCacheEntry,
    step: usize,
    tokens: []const u32,
    has_tools: bool,
    config: kv_quant.KVQuantConfig,
) ?u64 {
    const target = persistTargetLen(kv_entries, step, tokens.len);
    if (target == 0) return null;
    for (self.entries.items) |*e| {
        if (e.poisoned) continue;
        if (e.has_tools != has_tools) continue;
        if (!std.meta.eql(e.quant, config)) continue;
        if (e.tokens.len < tokens.len) continue;
        if (!std.mem.eql(u32, e.tokens[0..tokens.len], tokens)) continue;
        if (e.kv_len < target) continue;
        const want: usize = (@as(usize, e.kv_len) + self.chunk_tokens - 1) / self.chunk_tokens;
        if (e.chunk_bytes.len < want) continue;
        var whole = true;
        for (e.chunk_bytes[0..want]) |b| {
            if (b == 0) whole = false;
        }
        if (whole) return e.id;
    }
    return null;
}

/// Sidecar-only append: KV chunks are already fully on disk (superseded
/// on KV) but the entry has pending SSM checkpoints (byte-capped across
/// turns) and/or a missing spec snapshot this commit carries. Writes the
/// missing pieces into the existing dir + rewrites meta. Never touches
/// the KV chunks or the token record.
pub fn appendSsmOnly(
    self: *DiskTier,
    idx: usize,
    ssm_checkpoints: ?[]const trunk.SSMCheckpoint,
    dflash_snap: ?SpecCommit,
    mtp_snap: ?SpecCommit,
    ring_srcs: []const RingSource,
    s: mlx.mlx_stream,
) !PersistOutcome {
    const dir_rel = try std.fmt.allocPrint(self.allocator, "{s}/e{d}", .{ self.root, self.entries.items[idx].id });
    defer self.allocator.free(dir_rel);
    const live = &self.entries.items[idx];
    var updated = live.*;
    const e = &updated;
    const old_non_chunk = nonChunkBytes(e);
    var written_bytes: u64 = 0;
    // Write-ahead bound: the token record, not the flushed kv_len — a
    // checkpoint beyond the current chunks is position-keyed and becomes
    // restorable when a later extend raises kv_len past it.
    var ssm_res = try self.persistSsmCheckpoints(e.id, dir_rel, @intCast(e.tokens.len), e.ssm_positions, e.ssm_bytes, e.grown, e.grown_span, ssm_checkpoints, &written_bytes, s, self.max_flush_bytes, e.tokens);
    errdefer ssm_res.deinit(self.allocator);
    const rings = try self.persistRings(e.id, dir_rel, e.quant, e.rings, ring_srcs, &written_bytes, s);
    errdefer self.allocator.free(rings);
    const qsa_res = try self.persistQsaHistory(dir_rel, ssm_checkpoints, e.inherited_qsa, e.qsa_history_rows, e.qsa_history_bytes, e.kv_len, s, .{ .id = e.id, .own = e.qsa_segs, .own_inherited = e.qsa_segs_inherited });
    e.qsa_segs = qsa_res.segs;
    e.qsa_segs_inherited = qsa_res.segs_inherited;

    var spec_rewritten = false;
    if (specWorkPending(e, dflash_snap, mtp_snap)) {
        const prev: SegPrev = .{ .id = e.id, .own = e.spec_segs, .own_inherited = e.spec_segs_inherited };
        if (self.writeSpecSidecar(dir_rel, dflash_snap, mtp_snap, s, prev)) |spec_res| {
            e.spec_bytes = spec_res.bytes;
            e.spec_dflash = spec_res.dflash;
            e.spec_mtp = spec_res.mtp;
            e.spec_segs = spec_res.segs;
            e.spec_segs_inherited = spec_res.segs_inherited;
            spec_rewritten = true;
        } else |err| {
            log.warn("  [disk-cache] spec persist failed: {s} — entry keeps its old spec\n", .{@errorName(err)});
        }
    }

    e.ssm_positions = ssm_res.positions;
    e.ssm_bytes = ssm_res.bytes;
    e.grown = ssm_res.grown;
    e.grown_span = ssm_res.grown_span;
    e.rings = rings;
    if (qsa_res.inherited) {
        e.inherited_qsa = true;
    } else if (qsa_res.bytes > 0) {
        e.qsa_history_bytes = qsa_res.bytes;
        e.qsa_history_rows = qsa_res.rows;
        e.inherited_qsa = false;
    }
    // KV chunks and tokens did not change; bill the retained sidecars.
    const delta: i64 = @as(i64, @intCast(nonChunkBytes(e))) - @as(i64, @intCast(old_non_chunk));
    e.bytes = clampAdd(e.bytes, delta);
    e.last_used = self.bump();
    try self.writeMeta(e.*);
    self.allocator.free(live.ssm_positions);
    self.allocator.free(live.ssm_bytes);
    self.allocator.free(live.grown);
    self.allocator.free(live.rings);
    if (spec_rewritten) self.allocator.free(live.spec_segs);
    self.allocator.free(live.qsa_segs);
    live.* = updated;
    self.total_bytes = clampAdd(self.total_bytes, delta);
    self.last_store_id = updated.id;
    self.settleHeld(s);
    self.enforceBudget();
    return if (ssm_res.complete) .persisted else .partial;
}

// ── Spec-snapshot persistence (v4: dflash context / MTP history) ──

pub const SpecSidecarResult = struct {
    bytes: u64 = 0,
    dflash: ?SpecMeta = null,
    mtp: ?SpecMeta = null,
    /// The MTP history's whole segments (owned) and how many lead as links; `written` is what this call wrote.
    segs: []SideSeg = &.{},
    segs_inherited: u32 = 0,
    written: u64 = 0,
};

/// Does this commit carry a spec payload the entry lacks? Mirrors
/// `ssmWorkPending`'s role for the superseded no-op decision. A present
/// spec is never "updated" at the same tokens — same tokens, same
/// committed state.
pub fn specWorkPending(e: *const IndexEntry, dflash: ?SpecCommit, mtp: ?SpecCommit) bool {
    return (dflash != null and e.spec_dflash == null) or
        (mtp != null and e.spec_mtp == null) or
        // v5 upgrade: an entry persisted with a KV-only MTP snap gains the head's QSA half.
        (mtp != null and mtp.?.head_aux != null and
            (e.spec_mtp == null or e.spec_mtp.?.head == null)) or
        // A sidecar written under another KV scheme is declined at restore, so it
        // must be rewritten or the entry would draft blind forever.
        (mtp != null and e.spec_mtp != null and !std.meta.eql(e.spec_mtp.?.quant, mtp.?.config));
}

/// Write (or delete) the entry's ONE spec sidecar from this commit's
/// snapshots. Tensors are sliced to `step` positions (the snapshot buffer
/// can hold a stale draft tail past it) and keyed `d{layer}.*` /
/// `m{layer}.*` with the trunk chunks' kind suffixes.
pub fn writeSpecSidecar(self: *DiskTier, dir_abs: []const u8, dflash: ?SpecCommit, mtp: ?SpecCommit, s: mlx.mlx_stream, prev: SegPrev) !SpecSidecarResult {
    // Best-effort by contract: the callers log the failure and keep going. The
    // latch a raise plants is process-wide, so a write we already reported must
    // drop what it set — or the next decode tick fails an unrelated request.
    const had_error = mlx.errorPending();
    errdefer mlx.dropLatchedErrorUnless(had_error);
    if (dflash == null and mtp == null) {
        const path = try std.fmt.allocPrint(self.allocator, "{s}/spec.safetensors", .{dir_abs});
        defer self.allocator.free(path);
        // A staged copy of the old sidecar must not land after the delete.
        if (self.writer) |w| w.fence(path);
        self.fs.deleteFile(path);
        // The entry is spec-less, as before; its segments stay spare for the next commit's to keep.
        return .{ .segs = try self.allocator.dupe(SideSeg, prev.own), .segs_inherited = prev.own_inherited };
    }
    var list = std.ArrayList(NamedTensor).empty;
    defer self.freeNamed(&list);
    var res: SpecSidecarResult = .{};
    errdefer self.allocator.free(res.segs);
    if (dflash) |dc| res.dflash = try self.appendSpecTensors(&list, "d", dc, s);
    if (mtp) |mc| res.mtp = try self.appendSpecTensors(&list, "m", mc, s);
    const ratio: u32 = if (res.mtp) |m| if (m.head) |h| @intCast(@max(h.ratio, 1)) else 1 else 1;
    const seg = try self.writeSegments(dir_abs, 'p', &list, if (res.mtp) |m| m.step else 0, specGrowth, ratio, prev, .spec, s);
    res.segs = seg.segs;
    res.segs_inherited = seg.inherited;
    res.written = seg.written;
    // The file is replaced before the manifest commits; equal-shaped windows share a byte
    // size, so the load re-checks base/step from here.
    var stamp_buf: [2][48]u8 = undefined;
    var meta: [2]MetaPair = undefined;
    var n_meta: usize = 0;
    for ([_]?SpecMeta{ res.dflash, res.mtp }, [_][]const u8{ "d.pos", "m.pos" }) |sm_opt, key| {
        const sm = sm_opt orelse continue;
        meta[n_meta] = .{ .key = key, .value = try std.fmt.bufPrint(&stamp_buf[n_meta], "{d}:{d}", .{ sm.base, sm.step }) };
        n_meta += 1;
    }
    const path = try std.fmt.allocPrint(self.allocator, "{s}/spec.safetensors", .{dir_abs});
    res.bytes = try self.saveNamed(path, list.items, meta[0..n_meta], .spec, s);
    res.written += res.bytes;
    return res;
}

/// The segments a commit may keep or link instead of writing: the entry's own and a donor's.
pub const SegPrev = struct {
    id: u64 = 0,
    own: []const SideSeg = &.{},
    own_inherited: u32 = 0,
    donor_id: ?u64 = null,
    donor: []const SideSeg = &.{},
};

pub const SegResult = struct { segs: []SideSeg = &.{}, inherited: u32 = 0, written: u64 = 0 };

/// How a growing sidecar tensor grows: along `axis`, one row per `div` history rows.
pub const Growth = struct { axis: usize, div: u32 };

/// The MTP history's KV rows and the head's pooled bank grow; the head's ring and leftovers do not.
pub fn specGrowth(key: []const u8, ratio: u32) ?Growth {
    if (std.mem.eql(u8, key, "mh.pooled")) return .{ .axis = 1, .div = ratio };
    if (key.len < 2 or key[0] != 'm' or !std.ascii.isDigit(key[1])) return null;
    const kind = key[(std.mem.lastIndexOfScalar(u8, key, '.') orelse return null) + 1 ..];
    inline for (.{ "k", "v", "ks", "kb", "vs", "vb" }) |k| if (std.mem.eql(u8, kind, k)) return .{ .axis = 2, .div = 1 };
    return null;
}

/// Move the whole `seg_span`-row segments of `list`'s growing tensors into `{letter}{j}.safetensors`, leaving the
/// rows past them (at least one) in `list`. A segment whose file image the entry (`prev.own`) or, in a leading run,
/// the donor already holds is kept or hard-linked, so an extension writes only its new segments.
pub fn writeSegments(
    self: *DiskTier,
    dir_abs: []const u8,
    letter: u8,
    list: *std.ArrayList(NamedTensor),
    rows: u32,
    growthOf: *const fn ([]const u8, u32) ?Growth,
    ratio: u32,
    prev: SegPrev,
    kind: write_stats.Kind,
    s: mlx.mlx_stream,
) !SegResult {
    const span = self.seg_span;
    var n: u32 = if (span > 0 and rows > 0) (rows - 1) / span else 0;
    for (list.items) |*t| {
        const g = growthOf(t.key, ratio) orelse continue;
        if (span % g.div != 0 or mlx.getShape(t.arr)[g.axis] < @as(c_int, @intCast(@as(u64, n) * span / g.div))) n = 0;
    }
    var segs = std.ArrayList(SideSeg).empty;
    errdefer segs.deinit(self.allocator);
    var res: SegResult = .{};
    var leading = true;
    var j: u32 = 0;
    while (j < n) : (j += 1) {
        var part = std.ArrayList(NamedTensor).empty;
        defer self.freeNamed(&part);
        for (list.items) |*t| {
            const g = growthOf(t.key, ratio) orelse continue;
            const r0: c_int = @intCast(@as(u64, j) * span / g.div);
            const r1: c_int = @intCast(@as(u64, j + 1) * span / g.div);
            const key = try self.allocator.dupe(u8, t.key);
            errdefer self.allocator.free(key);
            var arr = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(arr);
            try sliceAxis(&arr, t.arr, g.axis, r0, r1, s);
            try part.append(self.allocator, .{ .key = key, .arr = arr, .rows = rowsFrom(t.rows, @intCast(r0)) });
        }
        const hash = try self.imageHash(part.items, s);
        const path = try std.fmt.allocPrint(self.allocator, "{s}/{c}{d:0>5}.safetensors", .{ dir_abs, letter, j });
        var path_owned: ?[]u8 = path;
        defer if (path_owned) |pp| self.allocator.free(pp);
        if (j < prev.own.len and prev.own[j].hash == hash and self.fs.size(path) == prev.own[j].bytes) {
            try segs.append(self.allocator, prev.own[j]);
            if (leading and j < prev.own_inherited) res.inherited += 1 else leading = false;
            continue;
        }
        if (leading and prev.donor_id != null and j < prev.donor.len and prev.donor[j].hash == hash and self.linkSeg(prev.donor_id.?, prev.id, letter, j)) {
            try segs.append(self.allocator, prev.donor[j]);
            res.inherited += 1;
            continue;
        }
        leading = false;
        // The synchronous arm writes in place: a link must go first, or the donor's file changes.
        if (self.writer == null) self.fs.deleteFile(path);
        path_owned = null;
        const sz = try self.saveNamed(path, part.items, &.{}, kind, s);
        res.written += sz;
        try segs.append(self.allocator, .{ .bytes = sz, .hash = hash });
    }
    if (prev.own.len > n) self.deleteSegFiles(dir_abs, letter, n, prev.own.len);
    if (n > 0) {
        for (list.items) |*t| {
            const g = growthOf(t.key, ratio) orelse continue;
            const from: usize = @intCast(@as(u64, n) * span / g.div);
            var tail = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(tail);
            try sliceAxis(&tail, t.arr, g.axis, @intCast(from), mlx.getShape(t.arr)[g.axis], s);
            _ = mlx.mlx_array_free(t.arr);
            t.arr = tail;
            t.rows = rowsFrom(t.rows, from);
        }
    }
    res.segs = try segs.toOwnedSlice(self.allocator);
    return res;
}

/// `rows` of a tensor for its slice from row `from` on.
pub fn rowsFrom(rows: ?vt.RowMap, from: usize) ?vt.RowMap {
    var m = rows orelse return null;
    m.pos0 += @as(u64, from) * m.per_row;
    return m;
}

/// Wyhash of the file image `tensors` would be saved as (no metadata).
pub fn imageHash(self: *DiskTier, tensors: []const NamedTensor, s: mlx.mlx_stream) !u64 {
    const views = try self.tensorViews(tensors);
    defer views.deinit(self.allocator);
    const conts = try safetensors.materialize(self.allocator, views.tensors, s);
    defer safetensors.freeArrays(self.allocator, conts);
    for (views.tensors, conts) |*v, c| v.arr = c;
    const head = try safetensors.encodeHeader(self.allocator, views.tensors, &.{});
    defer self.allocator.free(head);
    var h = std.hash.Wyhash.init(0);
    h.update(head);
    for (conts) |c| {
        const bytes = safetensors.bytesOf(c) orelse return error.DiskCacheUnreadable;
        if (bytes.len != 0) h.update(bytes);
    }
    return h.final();
}

/// Hard-link segment `j` of entry `from` into entry `to`; false leaves `to` to write it.
pub fn linkSeg(self: *DiskTier, from: u64, to: u64, letter: u8, j: u32) bool {
    var ab: [std.fs.max_path_bytes]u8 = undefined;
    var nb: [std.fs.max_path_bytes]u8 = undefined;
    const old_abs = std.fmt.bufPrint(&ab, "{s}/e{d}/{c}{d:0>5}.safetensors", .{ self.root, from, letter, j }) catch return false;
    const new_abs = std.fmt.bufPrint(&nb, "{s}/e{d}/{c}{d:0>5}.safetensors", .{ self.root, to, letter, j }) catch return false;
    if (self.writer) |w| if (w.isPending(old_abs)) return false;
    self.fs.deleteFile(new_abs);
    self.fs.hardLink(old_abs, new_abs) catch return false;
    return true;
}


pub fn linkSegFile(self: *DiskTier, from: u64, to: u64, letter: u8, j: u32) bool {
    return self.linkSeg(from, to, letter, j);
}

pub fn deleteSegFiles(self: *DiskTier, dir_abs: []const u8, letter: u8, from: usize, to: usize) void {
    for (from..to) |j| {
        const path = std.fmt.allocPrint(self.allocator, "{s}/{c}{d:0>5}.safetensors", .{ dir_abs, letter, j }) catch continue;
        defer self.allocator.free(path);
        if (self.writer) |w| w.fence(path);
        self.fs.deleteFile(path);
    }
}


pub fn appendSpecTensors(self: *DiskTier, list: *std.ArrayList(NamedTensor), prefix: []const u8, sc: SpecCommit, s: mlx.mlx_stream) !SpecMeta {
    if (sc.step == 0) return error.DiskCacheEmptyEntry;
    const limit: u32 = @intCast(sc.step);
    const affine = sc.config.scheme != .off;
    const first = list.items.len;
    for (sc.entries, 0..) |*entry, li| {
        if (!entry.initialized) continue;
        try self.appendSpecSlice(list, prefix, li, "k", entry.keys.array, limit, s);
        try self.appendSpecSlice(list, prefix, li, "v", entry.values.array, limit, s);
        if (affine) {
            try self.appendSpecSlice(list, prefix, li, "ks", entry.keys_scales.array, limit, s);
            try self.appendSpecSlice(list, prefix, li, "kb", entry.keys_biases.array, limit, s);
            try self.appendSpecSlice(list, prefix, li, "vs", entry.values_scales.array, limit, s);
            try self.appendSpecSlice(list, prefix, li, "vb", entry.values_biases.array, limit, s);
        }
    }
    for (list.items[first..]) |*t| t.rows = .{ .axis = 2, .pos0 = sc.base_pos };
    // v5 head half: the QSA raw-key history and pooled bank go in whole.
    var head: ?SpecHeadMeta = null;
    if (sc.head_aux) |a| {
        // A history that is not exactly `limit` rows is not persistable: drop the head half, keep the KV.
        const hist: c_int = if (a.qsa_rows > 0) a.qsa_rows else blk: {
            if (a.aux_state.ctx == null) break :blk 0;
            const sh = mlx.getShape(a.aux_state);
            break :blk if (sh.len >= 2) sh[1] else 0;
        };
        // `qsa_rows` reports the checkpoint position even when the raw ring is
        // gone, so a pooled-only head has no `h.aux` to write: the loader refuses
        // a head half without it, and an empty array raises inside mlx.
        const rows_ok = hist == @as(c_int, @intCast(limit)) and a.aux_state.ctx != null;
        if (rows_ok) {
            try self.appendSpecArray(list, prefix, "h.aux", a.aux_state);
            if (a.qsa_pooled.ctx != null) {
                try self.appendSpecArray(list, prefix, "h.pooled", a.qsa_pooled);
                list.items[list.items.len - 1].rows = .{ .axis = 1, .pos0 = @intCast(@max(sc.head_pos_base, 0)), .per_row = @intCast(@max(a.qsa_ratio, 1)) };
            }
            var hm: SpecHeadMeta = .{
                .pos_base = sc.head_pos_base,
                .ratio = a.qsa_ratio,
                .pooled = a.qsa_pooled.ctx != null,
                .rows = hist,
            };
            // v8: the leftovers a warm clamp reads, one tensor each (at most ratio-1 rows).
            for (sc.head_marks) |mk| {
                if (mk.rows.ctx == null or mk.pos > hist) continue;
                if (hm.mark_count == disk_format.HEAD_MARKS_MAX) break;
                var kind_buf: [16]u8 = undefined;
                const kind = try std.fmt.bufPrint(&kind_buf, "h.lv{d}", .{hm.mark_count});
                try self.appendSpecArray(list, prefix, kind, mk.rows);
                hm.marks[hm.mark_count] = mk.pos;
                hm.mark_count += 1;
            }
            head = hm;
        }
    }
    return .{
        .base = sc.base_pos,
        .step = limit,
        .layers = @intCast(sc.entries.len),
        .quant = sc.config,
        .head = head,
    };
}

/// A second handle on `arr`, keyed `{prefix}{kind}`; the list owns it.
pub fn appendSpecArray(self: *DiskTier, list: *std.ArrayList(NamedTensor), prefix: []const u8, kind: []const u8, arr: mlx.mlx_array) !void {
    const key = try std.fmt.allocPrint(self.allocator, "{s}{s}", .{ prefix, kind });
    errdefer self.allocator.free(key);
    var owned = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(owned);
    try mlx.check(mlx.mlx_array_set(&owned, arr));
    try list.append(self.allocator, .{ .key = key, .arr = owned });
}

pub fn appendSpecSlice(self: *DiskTier, list: *std.ArrayList(NamedTensor), prefix: []const u8, layer: usize, kind: []const u8, buf: mlx.mlx_array, limit: u32, s: mlx.mlx_stream) !void {
    const shape = mlx.getShape(buf);
    if (shape.len != 4) return error.DiskCacheBadShape;
    if (shape[2] < limit) return error.DiskCacheBadShape;
    var sliced = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(sliced);
    const st = [_]c_int{ 0, 0, 0, 0 };
    const sp = [_]c_int{ shape[0], shape[1], @intCast(limit), shape[3] };
    const sd = [_]c_int{ 1, 1, 1, 1 };
    try mlx.check(mlx.mlx_slice(&sliced, buf, &st, 4, &sp, 4, &sd, 4, s));
    const key = try std.fmt.allocPrint(self.allocator, "{s}{d}.{s}", .{ prefix, layer, kind });
    errdefer self.allocator.free(key);
    try list.append(self.allocator, .{ .key = key, .arr = sliced });
}


/// One staged tensor on the way to a safetensors file. `rows` names its token rows, which the content-addressed
/// layer stores by absolute position, so a rewrite shares the row groups it keeps; plain files ignore it.
pub const NamedTensor = struct { key: []u8, arr: mlx.mlx_array, rows: ?vt.RowMap = null };

pub fn freeNamed(self: *DiskTier, list: *std.ArrayList(NamedTensor)) void {
    for (list.items) |*t| {
        self.allocator.free(t.key);
        _ = mlx.mlx_array_free(t.arr);
    }
    list.deinit(self.allocator);
}

/// Write (or, under SSD-first, stage) one KV chunk. Returns the file's byte size.
pub fn writeChunkFile(
    self: *DiskTier,
    kv_entries: []const trunk.KVCacheEntry,
    config: kv_quant.KVQuantConfig,
    dir_abs: []const u8,
    chunk_idx: u32,
    c0: u32,
    c1: u32,
    s: mlx.mlx_stream,
) !u64 {
    var list = std.ArrayList(NamedTensor).empty;
    defer self.freeNamed(&list);

    for (kv_entries, 0..) |*entry, li| {
        // A ringed layer holds a window, never these rows: its restore points are ring files.
        if (!entry.initialized or entry.ringed) continue;
        try self.appendLayerSlices(&list, li, entry, config, c0, c1, s);
    }
    for (list.items) |*t| t.rows = .{ .axis = 2, .pos0 = c0 };
    const path = try std.fmt.allocPrint(self.allocator, "{s}/c{d:0>6}.safetensors", .{ dir_abs, chunk_idx });
    return self.saveNamed(path, list.items, no_meta, .kv, s);
}

/// One layer's K/V rows `[r0, r1)` of its own buffers, every stored kind.
pub fn appendLayerSlices(
    self: *DiskTier,
    list: *std.ArrayList(NamedTensor),
    li: usize,
    entry: *const trunk.KVCacheEntry,
    config: kv_quant.KVQuantConfig,
    r0: u32,
    r1: u32,
    s: mlx.mlx_stream,
) !void {
    try self.appendSlice(list, li, "k", entry.keys.array, r0, r1, s);
    try self.appendSlice(list, li, "v", entry.values.array, r0, r1, s);
    if (config.scheme != .off) {
        try self.appendSlice(list, li, "ks", entry.keys_scales.array, r0, r1, s);
        try self.appendSlice(list, li, "kb", entry.keys_biases.array, r0, r1, s);
        try self.appendSlice(list, li, "vs", entry.values_scales.array, r0, r1, s);
        try self.appendSlice(list, li, "vb", entry.values_biases.array, r0, r1, s);
    }
}

/// Write (or, with the writer, stage) one safetensors file at `path`, which it takes.
/// Returns the file's byte size.
pub fn saveNamed(self: *DiskTier, path: []u8, list: []NamedTensor, meta: []const MetaPair, kind: write_stats.Kind, s: mlx.mlx_stream) !u64 {
    if (self.writer) |w| {
        // Held points give way to a staged write: they go to the writer first, so they never hold its permit.
        while (self.held.items.len > 0 and w.pendingBytes() + self.held_bytes + dataLen(list) > w.permit_bytes) self.flushHeldAt(0, s) catch |err| {
            log.warn("  [disk-cache] a held restore point could not be written: {s}; dropped\n", .{@errorName(err)});
        };
        const views = self.tensorViews(list) catch |err| {
            self.allocator.free(path);
            return err;
        };
        defer views.deinit(self.allocator);
        return self.fs.stageTensors(w, path, views.tensors, meta, views.maps, kind, s);
    }
    const size = self.saveSync(path, list, meta, kind, s) catch |err| {
        self.stats.recordFailed(dataLen(list));
        return err;
    };
    self.stats.recordWritten(kind, size);
    return size;
}

pub const TensorViews = struct {
    tensors: []vt.Tensor,
    /// Empty when no tensor has rows, else parallel to `tensors`.
    maps: []?vt.RowMap,

    fn deinit(self: TensorViews, allocator: std.mem.Allocator) void {
        allocator.free(self.tensors);
        allocator.free(self.maps);
    }
};

pub fn tensorViews(self: *DiskTier, list: []const NamedTensor) !TensorViews {
    const views = try self.allocator.alloc(vt.Tensor, list.len);
    errdefer self.allocator.free(views);
    for (list, views) |t, *v| v.* = .{ .key = t.key, .arr = t.arr };
    const mapped = for (list) |t| {
        if (t.rows != null) break true;
    } else false;
    const maps = try self.allocator.alloc(?vt.RowMap, if (mapped) list.len else 0);
    if (mapped) for (list, maps) |t, *m| {
        m.* = t.rows;
    };
    return .{ .tensors = views, .maps = maps };
}

/// The synchronous arm of `saveNamed`: writes directly into the VFS. Takes `path`.
pub fn saveSync(self: *DiskTier, path: []u8, list: []NamedTensor, meta: []const MetaPair, kind: write_stats.Kind, s: mlx.mlx_stream) !u64 {
    defer self.allocator.free(path);
    const views = try self.tensorViews(list);
    defer views.deinit(self.allocator);
    return self.fs.saveTensorsNow(path, views.tensors, meta, views.maps, kind, s);
}

pub fn appendSlice(
    self: *DiskTier,
    list: *std.ArrayList(NamedTensor),
    layer: usize,
    kind: []const u8,
    buf: mlx.mlx_array,
    c0: u32,
    c1: u32,
    s: mlx.mlx_stream,
) !void {
    const shape = mlx.getShape(buf);
    if (shape.len != 4) return error.DiskCacheBadShape;
    var sliced = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(sliced);
    const st = [_]c_int{ 0, 0, @intCast(c0), 0 };
    const sp = [_]c_int{ shape[0], shape[1], @intCast(c1), shape[3] };
    const sd = [_]c_int{ 1, 1, 1, 1 };
    try mlx.check(mlx.mlx_slice(&sliced, buf, &st, 4, &sp, 4, &sd, 4, s));
    const key = try std.fmt.allocPrint(self.allocator, "l{d}.{s}", .{ layer, kind });
    errdefer self.allocator.free(key);
    try list.append(self.allocator, .{ .key = key, .arr = sliced });
}

pub const MetaPair = disk_format.MetaPair;
pub const no_meta: []const MetaPair = &[_]MetaPair{};

pub fn serializeNamed(self: *DiskTier, tensors: []const NamedTensor, meta: []const MetaPair, s: mlx.mlx_stream) ![]u8 {
    const views = try self.tensorViews(tensors);
    defer views.deinit(self.allocator);
    return safetensors.serialize(self.allocator, views.tensors, meta, s);
}

pub fn encodeNamed(self: *DiskTier, tensors: []const NamedTensor, meta: []const MetaPair) ![]u8 {
    const views = try self.tensorViews(tensors);
    defer views.deinit(self.allocator);
    return safetensors.encodeMaterialized(self.allocator, views.tensors, meta);
}

pub fn dataLen(tensors: []const NamedTensor) usize {
    var n: u64 = 0;
    for (tensors) |*t| n += nbytesOf(t.arr);
    return @intCast(n);
}

// ── SSM checkpoint persistence (Phase 3, hybrid archs) ──

pub const SsmPersistResult = struct {
    /// Persisted checkpoint positions, sorted ascending. Owned.
    positions: []u32,
    /// Per-file byte sizes, parallel to `positions`. Owned.
    bytes: []u64,
    /// The entry's row segments after this flush (`IndexEntry.grown`) and their span. Owned.
    grown: []u64,
    grown_span: u32,
    /// All target checkpoints made it to disk this flush (false → the
    /// per-flush byte cap deferred some; the caller keeps the entry dirty).
    complete: bool,

    fn deinit(self: *SsmPersistResult, allocator: std.mem.Allocator) void {
        allocator.free(self.positions);
        allocator.free(self.bytes);
        allocator.free(self.grown);
    }
};

/// Whether `arr` of snapshot layer `li` holds rows that grow with the prompt, `span` tokens to `divs[li]` apiece.
pub fn growable(divs: []const u32, span: u32, li: usize, arr: mlx.mlx_array) bool {
    if (arr.ctx == null or li >= divs.len or divs[li] == 0 or span % divs[li] != 0) return false;
    const sh = mlx.getShape(arr);
    return sh.len == 2 and sh[0] > 0;
}

/// Row segments a point of `cp` leans on: its whole spans, none when none of its arrays grow.
pub fn pointSegments(divs: []const u32, span: u32, cp: *const trunk.SSMCheckpoint) u32 {
    if (span == 0) return 0;
    for (cp.layers, 0..) |*l, li| {
        if (growable(divs, span, li, l.conv_state) or growable(divs, span, li, l.ssm_state)) return @intCast(cp.pos / span);
    }
    return 0;
}

/// A growing array's rows: row `i` holds tokens from `i * divs[li]`.
pub fn grownRows(divs: []const u32, li: usize) vt.RowMap {
    return .{ .axis = 0, .per_row = divs[li] };
}

pub fn sliceRows(out: *mlx.mlx_array, arr: mlx.mlx_array, from: usize, to: usize, s: mlx.mlx_stream) !void {
    const sh = mlx.getShape(arr);
    try mlx.check(mlx.mlx_slice(out, arr, &[_]c_int{ @intCast(from), 0 }, 2, &[_]c_int{ @intCast(to), sh[1] }, 2, &[_]c_int{ 1, 1 }, 2, s));
}

/// Segment `j` (1-based) of an entry: the rows of tokens ((j-1)*span, j*span] of every growing array of `cp`.
pub fn writeGrownSegment(self: *DiskTier, dir_rel: []const u8, cp: *const trunk.SSMCheckpoint, divs: []const u32, span: u32, j: u32, s: mlx.mlx_stream) !u64 {
    var list = std.ArrayList(NamedTensor).empty;
    defer self.freeNamed(&list);
    for (cp.layers, 0..) |l, li| {
        inline for (.{ "conv", "ssm" }, .{ l.conv_state, l.ssm_state }) |name, arr| {
            if (growable(divs, span, li, arr)) {
                if (@as(usize, @intCast(mlx.getShape(arr)[0])) < @as(usize, j) * span / divs[li]) return error.DiskCacheCorruptSsm;
                const from = (@as(usize, j) - 1) * span / divs[li];
                var piece = mlx.mlx_array_new();
                errdefer _ = mlx.mlx_array_free(piece);
                try sliceRows(&piece, arr, from, @as(usize, j) * span / divs[li], s);
                const key = try std.fmt.allocPrint(self.allocator, "l{d}." ++ name, .{li});
                errdefer self.allocator.free(key);
                try list.append(self.allocator, .{ .key = key, .arr = piece, .rows = rowsFrom(grownRows(divs, li), from) });
            }
        }
    }
    const path = try std.fmt.allocPrint(self.allocator, "{s}/g{d:0>5}.safetensors", .{ dir_rel, j });
    return self.saveNamed(path, list.items, no_meta, .ssm, s);
}

/// Writes the row segments `cp`'s point leans on that the entry lacks, from `cp`'s own rows; returns how many it leans on.
pub fn growSegments(self: *DiskTier, dir_rel: []const u8, cp: *const trunk.SSMCheckpoint, divs: []const u32, span: u32, grown: *std.ArrayList(u64), s: mlx.mlx_stream) !u32 {
    const k = pointSegments(divs, span, cp);
    while (grown.items.len < k) {
        const sz = try self.writeGrownSegment(dir_rel, cp, divs, span, @intCast(grown.items.len + 1), s);
        try grown.append(self.allocator, sz);
        self.point_bytes_written += sz;
    }
    return k;
}

pub fn findCp(cps: []const trunk.SSMCheckpoint, pos: u32) ?*const trunk.SSMCheckpoint {
    for (cps) |*cp| if (cp.pos == pos) return cp;
    return null;
}

/// The set of checkpoint positions that SHOULD be on disk after this
/// flush: `SSM_DISK_MAX_PER_ENTRY` of (already-persisted ∪ newly-eligible), thinned
/// span-preservingly. Eligible = a RAM checkpoint at a position within the
/// KV now on disk (a hybrid restore needs KV covering [0, cp_pos)).
/// Sorted ascending; caller frees.
pub fn ssmTargetPositions(self: *DiskTier, old_positions: []const u32, cps: []const trunk.SSMCheckpoint, kv_len: u32) ![]u32 {
    var set = std.ArrayList(u32).empty;
    errdefer set.deinit(self.allocator);
    try set.appendSlice(self.allocator, old_positions);
    for (cps) |*cp| {
        // A released state has no bytes to write: it is a restore point only where its file already is.
        if (cp.pos == 0 or cp.pos > kv_len or cp.isReleased()) continue;
        const p: u32 = @intCast(cp.pos);
        if (std.mem.indexOfScalar(u32, set.items, p) == null) try set.append(self.allocator, p);
    }
    std.mem.sort(u32, set.items, {}, std.sort.asc(u32));
    while (set.items.len > self.ssm_max_per_entry) {
        _ = set.orderedRemove(trunk.positionDropIndex(set.items, self.cp_thin));
    }
    self.ssm_budget_drops = 0;
    var unit: u64 = 0;
    for (cps) |*cp| unit = @max(unit, self.pointFileBytes(cp));
    // The newest point and the newest grid point stay; the oldest of the rest go first.
    while (self.ssm_budget != 0 and set.items.len > 0 and @as(u64, set.items.len) * unit > self.ssm_budget) {
        var grid_top: u32 = 0;
        for (set.items) |p| {
            if (p % POINT_GRID == 0) grid_top = p;
        }
        const newest = set.items[set.items.len - 1];
        const drop = for (set.items, 0..) |p, i| {
            if (p != newest and p != grid_top) break i;
        } else break;
        _ = set.orderedRemove(drop);
        self.ssm_budget_drops += 1;
    }
    return set.toOwnedSlice(self.allocator);
}

pub const StoreTarget = struct { superseded: bool = false, extend_idx: ?usize = null, ssm_only_idx: ?usize = null };

/// A commit's media identity, as `CommitParams` carries it.
pub const CommitMedia = struct { key: u64, start: ?u32 };

/// Leading rows of `e` that are also a commit's rows: `prefix_policy.crossKeyBoundary` with no
/// chain. An equal key shares everything; any other pair only the text before either side's
/// first media row, since two same-size images render the same placeholder ids.
pub fn commitShares(e: *const IndexEntry, media: CommitMedia) u32 {
    return commitSharesWithChain(e, media, &.{});
}

/// `commitShares` with the committing request's media chain: an entry whose key is the chain's
/// first k blocks shares through block k+1's start, letting chunk and row-segment donors link
/// past earlier images when a follow-up turn appends a new one.
pub fn commitSharesWithChain(e: *const IndexEntry, media: CommitMedia, chain: []const prefix_policy.MediaSpan) u32 {
    if (e.vision_key == media.key) return MediaLimits.UNBOUNDED;
    const es: ?usize = if (e.media_start) |v| @as(usize, v) else null;
    const ms: ?usize = if (media.start) |v| @as(usize, v) else null;
    const boundary = prefix_policy.crossKeyBoundary(e.vision_key, es, ms, chain) orelse return 0;
    return std.math.cast(u32, boundary) orelse MediaLimits.UNBOUNDED;
}

/// Which resident entry a commit of `tokens` supersedes, extends or tops up: only one whose
/// rows are the commit's wherever both records reach. Indices are valid only until the next eviction.
pub fn selectStoreTarget(
    self: *DiskTier,
    tokens: []const u32,
    kv_target: u32,
    has_tools: bool,
    config: kv_quant.KVQuantConfig,
    media: CommitMedia,
    ssm_checkpoints: ?[]const trunk.SSMCheckpoint,
    dflash_snap: ?SpecCommit,
    mtp_snap: ?SpecCommit,
    ring_srcs: []const RingSource,
) StoreTarget {
    var t: StoreTarget = .{};
    for (self.entries.items, 0..) |*e, i| {
        if (e.poisoned) continue; // dead: never superseded, never extended
        if (e.has_tools != has_tools) continue;
        if (!std.meta.eql(e.quant, config)) continue;
        // Covering the commit needs its rows through `tokens`; resuming the entry keeps all of its own.
        const shares = commitShares(e, media);
        if (e.tokens.len >= tokens.len) {
            if (std.mem.eql(u32, e.tokens[0..tokens.len], tokens)) {
                if (e.kv_len >= kv_target) {
                    if (shares < tokens.len) continue;
                    if (!self.ssmWorkPending(e, ssm_checkpoints, @intCast(e.tokens.len)) and
                        !specWorkPending(e, dflash_snap, mtp_snap) and
                        !self.ringWorkPending(e, ring_srcs))
                    {
                        // Superseded: the tier already holds this prefix in full.
                        e.last_used = self.bump();
                        return .{ .superseded = true };
                    }
                    t.ssm_only_idx = i;
                    break;
                }
                // Same token record, SHORTER persisted KV — a byte-capped
                // incremental flush in progress. Resume into its dir.
                if (shares >= e.tokens.len) t.extend_idx = i;
            }
        } else if (shares >= e.tokens.len and std.mem.eql(u32, e.tokens, tokens[0..e.tokens.len])) {
            // This commit extends `e` — reuse its directory and chunks.
            t.extend_idx = i;
        }
    }
    return t;
}

/// Would persisting `cps` add or drop any file for entry `e`? Drives the
/// superseded no-op vs SSM-only-append decision. Conservative on alloc
/// failure (returns false → the commit is a harmless no-op; the RAM tier
/// still holds the checkpoints).
pub fn ssmWorkPending(self: *DiskTier, e: *const IndexEntry, cps_opt: ?[]const trunk.SSMCheckpoint, kv_limit: u32) bool {
    const cps = cps_opt orelse return false;
    if (cps.len == 0) return false;
    const target = self.ssmTargetPositions(e.ssm_positions, cps, kv_limit) catch return false;
    defer self.allocator.free(target);
    // A target position missing from disk, OR an on-disk position no
    // longer in target (retention would drop it), is pending work.
    if (target.len != e.ssm_positions.len) return true;
    for (target) |p| {
        if (std.mem.indexOfScalar(u32, e.ssm_positions, p) == null) return true;
    }
    return false;
}

/// Persist the eligible SSM checkpoints for one entry: write target
/// positions not yet on disk (highest-first — the end-of-prompt
/// checkpoint is the most valuable), delete retention-dropped
/// positions, and return the resulting on-disk set. `written_bytes`
/// accumulates the bytes this call wrote; `max_spend` is this call's
/// own byte share of the flush budget (the caller decides how much of
/// `max_flush_bytes` the checkpoints get — chunk-first budgeting
/// starved them to zero on chunk-heavy turns, live 2026-09-07).
/// `kv_limit` is the WRITE-AHEAD bound, not the flushed kv_len: a
/// checkpoint beyond the chunks this flush reaches is still written
/// (position-keyed, immutable) and becomes restorable when a later
/// flush extends the entry's kv_len past it.
pub fn persistSsmCheckpoints(
    self: *DiskTier,
    id: u64,
    dir_rel: []const u8,
    kv_limit: u32,
    old_positions: []const u32,
    old_bytes: []const u64,
    old_grown: []const u64,
    old_span: u32,
    cps_opt: ?[]const trunk.SSMCheckpoint,
    written_bytes: *u64,
    s: mlx.mlx_stream,
    max_spend: u64,
    tokens: []const u32,
) !SsmPersistResult {
    const cps: []const trunk.SSMCheckpoint = cps_opt orelse &[_]trunk.SSMCheckpoint{};
    var grown = std.ArrayList(u64).empty;
    errdefer grown.deinit(self.allocator);
    try grown.appendSlice(self.allocator, old_grown);
    // An entry keeps the span its segments were cut at.
    const span: u32 = if (old_grown.len > 0) old_span else if (self.grown) |g| g.span else 0;
    const divs: []const u32 = if (self.grown) |g| g.divs else &.{};
    if (cps.len == 0 and old_positions.len == 0) {
        const positions = try self.allocator.alloc(u32, 0);
        errdefer self.allocator.free(positions);
        const bytes = try self.allocator.alloc(u64, 0);
        errdefer self.allocator.free(bytes);
        const kept = try grown.toOwnedSlice(self.allocator);
        return .{ .positions = positions, .bytes = bytes, .grown = kept, .grown_span = if (kept.len > 0) span else 0, .complete = true };
    }
    const target = try self.ssmTargetPositions(old_positions, cps, kv_limit);
    defer self.allocator.free(target);
    if (self.ssm_budget_drops > 0) {
        log.info("  [disk-cache] one conversation outgrows the {d:.1} MB budget: dropped its {d} oldest restore points (raise --prefix-cache-disk to keep them)\n", .{ mib(self.max_bytes), self.ssm_budget_drops });
    }

    // Delete positions retention drops (present on disk, absent from target).
    for (old_positions) |p| {
        if (std.mem.indexOfScalar(u32, target, p) == null) self.deleteSsmFile(id, p);
    }

    const Pair = struct { pos: u32, bytes: u64 };
    var pairs = std.ArrayList(Pair).empty;
    defer pairs.deinit(self.allocator);
    var complete = true;

    // Carry over old positions kept by retention (already on disk).
    for (target) |p| {
        if (std.mem.indexOfScalar(u32, old_positions, p)) |oi| {
            try pairs.append(self.allocator, .{ .pos = p, .bytes = old_bytes[oi] });
        }
    }
    // Write new target positions, highest-first.
    var ti = target.len;
    while (ti > 0) : (ti -= 1) {
        const p = target[ti - 1];
        if (std.mem.indexOfScalar(u32, old_positions, p) != null) continue; // already on disk
        const cp = findCp(cps, p) orelse continue;
        // Under SSD-first or the writer a checkpoint is written beside its chunk, outside the
        // byte budget; a synchronous flush bounds its share by `max_spend` (half the cap —
        // chunk-first budgeting starved them to zero, live 2026-09-07).
        if (!self.ssd_first and self.writer == null and written_bytes.* >= max_spend) {
            complete = false;
            continue; // share exhausted — persist on a later flush
        }
        if (self.grown == null and self.holdsPosition(p)) {
            if (self.holdPoint(id, dir_rel, cp, s)) |sz| {
                try pairs.append(self.allocator, .{ .pos = p, .bytes = sz });
                continue;
            } else |err| log.warn("  [disk-cache] restore point at {d} written, not held: {s}\n", .{ p, @errorName(err) });
        }
        const k = if (self.grown != null) try self.growSegments(dir_rel, cp, divs, span, &grown, s) else 0;
        const sz = try self.writePoint(dir_rel, cp, s, k, span);
        self.point_bytes_written += sz;
        written_bytes.* += sz;
        try pairs.append(self.allocator, .{ .pos = p, .bytes = sz });
    }
    // Bytes the segments took count against the flush like the points'.
    for (grown.items[old_grown.len..]) |b| written_bytes.* += b;
    // A held point this commit's points extend is restored again only by a regenerate of its turn: the newest such
    // point stays as a spare, and every older one goes unwritten.
    if (self.held.items.len > 0) {
        var top: u32 = 0;
        for (pairs.items) |pr| top = @max(top, pr.pos);
        var spare: ?usize = null;
        for (self.held.items, 0..) |h, j| {
            if (h.pos < top and self.heldExtendedBy(h, id, tokens) and (spare == null or h.pos > self.held.items[spare.?].pos)) spare = j;
        }
        const spare_key: ?struct { id: u64, pos: u32 } = if (spare) |j| .{ .id = self.held.items[j].id, .pos = self.held.items[j].pos } else null;
        var i: usize = 0;
        while (i < self.held.items.len) {
            const h = self.held.items[i];
            if (spare_key) |k| if (h.id == k.id and h.pos == k.pos) {
                self.held.items[i].spare = true;
                i += 1;
                continue;
            };
            if (h.pos < top and self.heldExtendedBy(h, id, tokens)) {
                if (h.id == id) {
                    for (pairs.items, 0..) |pr, j| if (pr.pos == h.pos) {
                        _ = pairs.orderedRemove(j);
                        break;
                    };
                } else self.unlistPoint(h.id, h.pos);
                self.dropHeldAt(i);
                continue;
            }
            i += 1;
        }
    }

    std.mem.sort(Pair, pairs.items, {}, struct {
        fn lt(_: void, a: Pair, b: Pair) bool {
            return a.pos < b.pos;
        }
    }.lt);
    const positions = try self.allocator.alloc(u32, pairs.items.len);
    errdefer self.allocator.free(positions);
    const bytes = try self.allocator.alloc(u64, pairs.items.len);
    for (pairs.items, 0..) |pr, i| {
        positions[i] = pr.pos;
        bytes[i] = pr.bytes;
    }
    errdefer self.allocator.free(bytes);
    const grown_out = try grown.toOwnedSlice(self.allocator);
    return .{ .positions = positions, .bytes = bytes, .grown = grown_out, .grown_span = if (grown_out.len > 0) span else 0, .complete = complete };
}

/// Write (or, under SSD-first, stage) one SSM checkpoint as `s{pos:0>7}.safetensors`.
/// Per-layer tensors keyed "l{i}.conv"/"l{i}.ssm"; the `initialized` bitmap rides the
/// metadata map. Staged because this runs inside the prefill chunk loop, where a
/// synchronous ~56 MB write was the largest term of the mid-prefill write-through.
pub fn writeSsmFile(
    self: *DiskTier,
    dir_rel: []const u8,
    cp: *const trunk.SSMCheckpoint,
    s: mlx.mlx_stream,
) !u64 {
    return self.writePoint(dir_rel, cp, s, 0, 0);
}

/// `writeSsmFile` for a point that leans on the entry's first `grown_k` row segments (`span` tokens each):
/// its growing arrays keep only the rows past them, and `grown` in the metadata names the count.
pub fn writePoint(
    self: *DiskTier,
    dir_rel: []const u8,
    cp: *const trunk.SSMCheckpoint,
    s: mlx.mlx_stream,
    grown_k: u32,
    span: u32,
) !u64 {
    return self.encodePoint(dir_rel, cp, s, grown_k, span, null);
}

/// `writePoint`, or with `held` its file's bytes into `held.*` instead of a write; returns the file's size.
pub fn encodePoint(
    self: *DiskTier,
    dir_rel: []const u8,
    cp: *const trunk.SSMCheckpoint,
    s: mlx.mlx_stream,
    grown_k: u32,
    span: u32,
    held: ?*[]u8,
) !u64 {
    const divs: []const u32 = if (self.grown) |g| g.divs else &.{};
    var list = std.ArrayList(NamedTensor).empty;
    defer self.freeNamed(&list);
    var meta = std.ArrayList(MetaPair).empty;
    defer meta.deinit(self.allocator);

    var lc_buf: [24]u8 = undefined;
    const lc = try std.fmt.bufPrint(&lc_buf, "{d}\x00", .{cp.layers.len});
    try meta.append(self.allocator, .{ .key = "layers", .value = lc[0 .. lc.len - 1] });

    var init_buf = std.ArrayList(u8).empty;
    defer init_buf.deinit(self.allocator);
    var num_buf: [16]u8 = undefined;
    for (cp.layers, 0..) |l, li| {
        if (!l.initialized) continue;
        if (init_buf.items.len > 0) try init_buf.append(self.allocator, ',');
        const ns = std.fmt.bufPrint(&num_buf, "{d}", .{li}) catch unreachable;
        try init_buf.appendSlice(self.allocator, ns);
    }
    const init_len = init_buf.items.len;
    try init_buf.append(self.allocator, 0); // NUL-terminate for the C API
    // Taken after the last append: an earlier slice would dangle across list growth.
    try meta.append(self.allocator, .{ .key = "init", .value = init_buf.items[0..init_len] });

    // Aux state rides the same file: `l{d}.aux` / `l{d}.pooled`
    // tensors and `l{d}.ple` = uint32 [9] (valid flag, then the 8 token
    // history slots); the compress ratio is one `qsa_ratio` metadata key.
    var ratio_buf: [16]u8 = undefined;
    var rows_buf: [16]u8 = undefined;
    var grown_buf: [16]u8 = undefined;
    var ratio_written = false;
    var rows_written = false;
    if (grown_k > 0) {
        const gs = try std.fmt.bufPrint(&grown_buf, "{d}", .{grown_k});
        try meta.append(self.allocator, .{ .key = "grown", .value = gs });
    }
    for (cp.layers, 0..) |l, li| {
        const qsa_hist = prefix_arch.aux.snapshotHasQsaHistory(&l);
        const names = .{ "conv", "ssm", "aux", "pooled" };
        const arrs = .{ l.conv_state, l.ssm_state, l.aux_state, l.qsa_pooled };
        inline for (names, arrs) |name, arr| {
            const skip_qsa = qsa_hist and !disk_placement.test_ssm_write_qsa_aux and (comptime std.mem.eql(u8, name, "pooled"));
            // A growing array keeps the rows its segments do not hold; nothing is written when none remain.
            const grows = growable(divs, span, li, arr);
            const cut = grown_k > 0 and grows;
            const from: usize = if (cut) @as(usize, grown_k) * span / divs[li] else 0;
            if (arr.ctx != null and !skip_qsa and !(cut and from == @as(usize, @intCast(mlx.getShape(arr)[0])))) {
                const key = try std.fmt.allocPrint(self.allocator, "l{d}." ++ name, .{li});
                errdefer self.allocator.free(key);
                // The list owns every handle it holds, so a borrowed checkpoint handle is retained first.
                var owned = mlx.mlx_array_new();
                errdefer _ = mlx.mlx_array_free(owned);
                if (cut) {
                    try sliceRows(&owned, arr, from, @intCast(mlx.getShape(arr)[0]), s);
                } else {
                    try mlx.check(mlx.mlx_array_set(&owned, arr));
                }
                try list.append(self.allocator, .{ .key = key, .arr = owned, .rows = if (grows) rowsFrom(grownRows(divs, li), from) else null });
            }
        }
        if (l.ple_prev_valid) {
            var ple: [9]u32 = undefined;
            ple[0] = 1;
            for (l.ple_prev, 0..) |t, i| ple[1 + i] = t;
            // `mlx_array_new_data` copies, so `ple` may die at the end of this block.
            const ple_arr = try mlx.newData(&ple, &[_]c_int{9}, .uint32);
            errdefer _ = mlx.mlx_array_free(ple_arr);
            const key = try std.fmt.allocPrint(self.allocator, "l{d}.ple", .{li});
            errdefer self.allocator.free(key);
            try list.append(self.allocator, .{ .key = key, .arr = ple_arr });
        }
        if (!ratio_written and (l.qsa_rows > 0 or l.aux_state.ctx != null or l.qsa_pooled.ctx != null)) {
            const rs = try std.fmt.bufPrint(&ratio_buf, "{d}\x00", .{l.qsa_ratio});
            try meta.append(self.allocator, .{ .key = "qsa_ratio", .value = rs[0 .. rs.len - 1] });
            ratio_written = true;
        }
        if (!rows_written and l.qsa_rows > 0) {
            const rs = try std.fmt.bufPrint(&rows_buf, "{d}\x00", .{l.qsa_rows});
            try meta.append(self.allocator, .{ .key = "qsa_rows", .value = rs[0 .. rs.len - 1] });
            rows_written = true;
        }
    }

    if (held) |out| {
        out.* = try self.serializeNamed(list.items, meta.items, s);
        return out.len;
    }
    const path = try std.fmt.allocPrint(self.allocator, "{s}/s{d:0>7}.safetensors", .{ dir_rel, cp.pos });
    return self.saveNamed(path, list.items, meta.items, .ssm, s);
}

/// `written`: the bytes this call put on disk (0 when the file was kept or linked).
pub const QsaHistoryResult = struct { bytes: u64 = 0, rows: u32 = 0, inherited: bool = false, written: u64 = 0, segs: []SideSeg = &.{}, segs_inherited: u32 = 0 };

pub fn deleteSsmFile(self: *DiskTier, id: u64, pos: u32) void {
    if (self.heldIndex(id, pos)) |i| return self.dropHeldAt(i);
    const path = std.fmt.allocPrint(self.allocator, "{s}/e{d}/s{d:0>7}.safetensors", .{ self.root, id, pos }) catch return;
    defer self.allocator.free(path);
    // A staged checkpoint may still be in the writer's queue; fence it before deleting, or
    // the write lands a file no index names.
    if (self.writer) |w| w.fence(path);
    self.fs.deleteFile(path);
}

// ── Ring restore points (ringed sliding layers) ──

/// Where a commit can write a ring file: `rows` below `pos` of `entries`' ringed layers.
pub const RingSource = struct { pos: u32, rows: u32, entries: []const trunk.KVCacheEntry };

pub fn ringFileBytes(src: RingSource) u64 {
    var n: u64 = FILE_OVERHEAD_BYTES;
    for (src.entries) |*e| {
        if (!e.initialized or !e.ringed) continue;
        for (layerArrays(e)) |a| {
            const rb = rowBytes(a);
            if (rb > 0) n += rb * src.rows + TENSOR_OVERHEAD_BYTES;
        }
    }
    return n;
}

/// Rows every ringed layer of `entries` can give a restore point at `pos`: window + backoff at
/// most, at least the window (or every row from 0). Null when one cannot, or none rings.
pub fn ringRowsAt(entries: []const trunk.KVCacheEntry, pos: usize, window: u32) ?u32 {
    const want: usize = @as(usize, window) + model.ModelConfig.SWA_RING_CHECKPOINT_BACKOFF;
    var rows: ?usize = null;
    for (entries) |*e| {
        if (!e.initialized or !e.ringed) continue;
        if (pos < e.base or pos > e.base + e.offset) return null;
        const n = @min(pos - e.base, want);
        if (n < @min(pos, @as(usize, window))) return null;
        rows = @min(rows orelse n, n);
    }
    return if (rows) |r| @intCast(r) else null;
}

/// The cache's own ring at the persisted length, then each checkpoint at its `step`; a
/// restore below `MIN_DISK_ADVANTAGE_TOKENS` never beats the RAM tier. Caller frees.
pub fn ringSources(self: *DiskTier, kv_entries: []const trunk.KVCacheEntry, kv_target: u32, ring: RingCommit) ![]RingSource {
    var out = std.ArrayList(RingSource).empty;
    errdefer out.deinit(self.allocator);
    if (kv_target >= MIN_DISK_ADVANTAGE_TOKENS) {
        if (ringRowsAt(kv_entries, kv_target, ring.window)) |rows|
            try out.append(self.allocator, .{ .pos = kv_target, .rows = rows, .entries = kv_entries });
    }
    for (ring.cps) |*cp| {
        if (cp.step < MIN_DISK_ADVANTAGE_TOKENS or cp.step > kv_target) continue;
        const pos: u32 = @intCast(cp.step);
        if (ringSourceAt(out.items, pos) != null) continue;
        const rows = ringRowsAt(cp.entries, pos, ring.window) orelse continue;
        try out.append(self.allocator, .{ .pos = pos, .rows = rows, .entries = cp.entries });
    }
    return out.toOwnedSlice(self.allocator);
}

pub fn ringSourceAt(srcs: []const RingSource, pos: u32) ?RingSource {
    for (srcs) |src| if (src.pos == pos) return src;
    return null;
}

/// The positions an entry holding `old` keeps after a commit offering `srcs`, ascending
/// (`thinRingPositions`). Caller frees.
pub fn ringTargetPositions(self: *DiskTier, old: []const RingFile, srcs: []const RingSource) ![]u32 {
    var set = std.ArrayList(u32).empty;
    defer set.deinit(self.allocator);
    for (old) |r| try set.append(self.allocator, r.pos);
    for (srcs) |src| {
        if (std.mem.indexOfScalar(u32, set.items, src.pos) == null) try set.append(self.allocator, src.pos);
    }
    std.mem.sort(u32, set.items, {}, std.sort.asc(u32));
    return self.allocator.dupe(u32, thinRingPositions(set.items));
}

/// Thin ascending `positions` in place to `RING_DISK_MAX_PER_ENTRY` the way the RAM entry
/// thins its ring checkpoints: the lowest (a shared preamble) and the newest stay.
pub fn thinRingPositions(positions: []u32) []u32 {
    var n = positions.len;
    while (n > RING_DISK_MAX_PER_ENTRY) : (n -= 1) {
        const k = trunk.positionDropIndex(positions[0..n], .min_span_recency);
        for (k..n - 1) |i| positions[i] = positions[i + 1];
    }
    return positions[0..n];
}

/// Would `srcs` add or drop a ring file of `e`? False on alloc failure (a harmless no-op).
pub fn ringWorkPending(self: *DiskTier, e: *const IndexEntry, srcs: []const RingSource) bool {
    if (srcs.len == 0) return false;
    const target = self.ringTargetPositions(e.rings, srcs) catch return false;
    defer self.allocator.free(target);
    if (target.len != e.rings.len) return true;
    for (target, e.rings) |pos, r| {
        if (pos != r.pos) return true;
    }
    return false;
}

pub fn ringsEqual(a: []const RingFile, b: []const RingFile) bool {
    if (a.len != b.len) return false;
    for (a, b) |x, y| {
        if (x.pos != y.pos) return false;
    }
    return true;
}

/// Write the ring files `srcs` adds, delete the ones retention drops, and return the entry's
/// resulting set (owned). A file is immutable once written, keyed by its position.
pub fn persistRings(
    self: *DiskTier,
    id: u64,
    dir_rel: []const u8,
    config: kv_quant.KVQuantConfig,
    old: []const RingFile,
    srcs: []const RingSource,
    written_bytes: *u64,
    s: mlx.mlx_stream,
) ![]RingFile {
    const target = try self.ringTargetPositions(old, srcs);
    defer self.allocator.free(target);
    for (old) |r| {
        if (std.mem.indexOfScalar(u32, target, r.pos) == null) self.deleteRingFile(id, r.pos);
    }
    const out = try self.allocator.alloc(RingFile, target.len);
    errdefer self.allocator.free(out);
    for (target, out) |pos, *o| {
        o.* = for (old) |r| {
            if (r.pos == pos) break r;
        } else blk: {
            const src = ringSourceAt(srcs, pos).?;
            const bytes = try self.writeRingFile(dir_rel, src, config, s);
            written_bytes.* += bytes;
            break :blk .{ .pos = pos, .rows = src.rows, .bytes = bytes };
        };
    }
    return out;
}

/// Write (or, under SSD-first, stage) the ringed layers' rows `[pos - rows, pos)` of `src`
/// as `r{pos:0>7}.safetensors`; every other layer is in the chunks.
pub fn writeRingFile(self: *DiskTier, dir_rel: []const u8, src: RingSource, config: kv_quant.KVQuantConfig, s: mlx.mlx_stream) !u64 {
    var list = std.ArrayList(NamedTensor).empty;
    defer self.freeNamed(&list);
    for (src.entries, 0..) |*entry, li| {
        if (!entry.initialized or !entry.ringed) continue;
        const hi: u32 = @intCast(src.pos - entry.base);
        try self.appendLayerSlices(&list, li, entry, config, hi - src.rows, hi, s);
    }
    for (list.items) |*t| t.rows = .{ .axis = 2, .pos0 = src.pos - src.rows };
    var lc_buf: [24]u8 = undefined;
    const lc = try std.fmt.bufPrint(&lc_buf, "{d}", .{src.entries.len});
    const path = try std.fmt.allocPrint(self.allocator, "{s}/r{d:0>7}.safetensors", .{ dir_rel, src.pos });
    return self.saveNamed(path, list.items, &.{.{ .key = "layers", .value = lc }}, .ring, s);
}

pub fn deleteRingFile(self: *DiskTier, id: u64, pos: u32) void {
    const path = std.fmt.allocPrint(self.allocator, "{s}/e{d}/r{d:0>7}.safetensors", .{ self.root, id, pos }) catch return;
    defer self.allocator.free(path);
    if (self.writer) |w| w.fence(path);
    self.fs.deleteFile(path);
}


// ── Held restore points (docs/engine-prefix-cache.md#held-points) ──

/// What the held points may take: the cap, or the permit's share when that is smaller; 0 without a writer.
pub fn heldCap(self: *const DiskTier) u64 {
    const w = self.writer orelse return 0;
    if (@import("builtin").is_test) if (self.held_cap_for_test) |c| return c;
    return @min(HELD_POINTS_MAX_BYTES, w.permit_bytes / HELD_POINTS_PERMIT_DIVISOR);
}

pub fn holdsPosition(self: *const DiskTier, pos: u32) bool {
    return self.hold_stride != 0 and pos % self.hold_stride != 0 and self.heldCap() > 0;
}

pub fn heldIndex(self: *const DiskTier, id: u64, pos: u32) ?usize {
    for (self.held.items, 0..) |h, i| if (h.id == id and h.pos == pos) return i;
    return null;
}

/// Keeps `cp`'s file bytes in RAM as entry `id`'s point; returns the file's size. A point past the cap is refused.
pub fn holdPoint(self: *DiskTier, id: u64, dir_rel: []const u8, cp: *const trunk.SSMCheckpoint, s: mlx.mlx_stream) !u64 {
    var bytes: []u8 = &.{};
    const sz = try self.encodePoint(dir_rel, cp, s, 0, 0, &bytes);
    errdefer self.allocator.free(bytes);
    if (sz > self.heldCap()) return error.HeldPointOverCap;
    const pos: u32 = @intCast(cp.pos);
    if (self.heldIndex(id, pos)) |i| self.dropHeldAt(i);
    try self.held.append(self.allocator, .{ .id = id, .pos = pos, .bytes = bytes });
    self.held_bytes += sz;
    return sz;
}

/// Entry `id`'s tokens extend held point `h`'s entry through `h.pos` (or `h` is `id`'s own).
pub fn heldExtendedBy(self: *const DiskTier, h: HeldPoint, id: u64, tokens: []const u32) bool {
    if (h.id == id) return true;
    const e = for (self.entries.items) |*e| {
        if (e.id == h.id) break e;
    } else return true; // its entry is gone: nothing restores it
    return h.pos <= e.tokens.len and h.pos <= tokens.len and std.mem.eql(u32, e.tokens[0..h.pos], tokens[0..h.pos]);
}

pub fn dropHeldAt(self: *DiskTier, i: usize) void {
    const h = self.held.orderedRemove(i);
    self.held_bytes -|= h.bytes.len;
    self.allocator.free(h.bytes);
}

pub fn dropHeldOf(self: *DiskTier, id: u64) void {
    var i: usize = 0;
    while (i < self.held.items.len) {
        if (self.held.items[i].id == id) self.dropHeldAt(i) else i += 1;
    }
}

/// Takes `pos` out of entry `id`'s listed points (a held point that goes unwritten); its manifest on disk still
/// names it, and the next scan drops it there, as it drops any point whose file is missing.
pub fn unlistPoint(self: *DiskTier, id: u64, pos: u32) void {
    const e = for (self.entries.items) |*e| {
        if (e.id == id) break e;
    } else return;
    const at = std.mem.indexOfScalar(u32, e.ssm_positions, pos) orelse return;
    const sz = e.ssm_bytes[at];
    std.mem.copyForwards(u32, e.ssm_positions[at..], e.ssm_positions[at + 1 ..]);
    std.mem.copyForwards(u64, e.ssm_bytes[at..], e.ssm_bytes[at + 1 ..]);
    e.ssm_positions = self.allocator.realloc(e.ssm_positions, e.ssm_positions.len - 1) catch e.ssm_positions[0 .. e.ssm_positions.len - 1];
    e.ssm_bytes = self.allocator.realloc(e.ssm_bytes, e.ssm_bytes.len - 1) catch e.ssm_bytes[0 .. e.ssm_bytes.len - 1];
    e.bytes -|= sz;
    self.total_bytes -|= sz;
}

/// Writes held point `i` as the file its bytes are; it leaves RAM.
pub fn flushHeldAt(self: *DiskTier, i: usize, s: mlx.mlx_stream) anyerror!void {
    const h = self.held.orderedRemove(i);
    self.held_bytes -|= h.bytes.len;
    defer self.allocator.free(h.bytes);
    // A point that is neither held nor written is not listed, nor by the record a commit in flight installs.
    errdefer {
        self.unlistPoint(h.id, h.pos);
        self.lost_points.append(self.allocator, .{ .id = h.id, .pos = h.pos }) catch {};
    }
    if (@import("builtin").is_test and self.fail_held_flush_for_test) return error.InjectedHeldFlushFailure;
    var loaded = try vt.loadBytes(self.allocator, h.bytes, s);
    defer loaded.deinit();
    var list = std.ArrayList(NamedTensor).empty;
    defer self.freeNamed(&list);
    var tit = loaded.tensors.iterator();
    while (tit.next()) |e| {
        if (e.value_ptr.ctx == null) continue;
        const key = try self.allocator.dupe(u8, e.key_ptr.*);
        errdefer self.allocator.free(key);
        const arr = e.value_ptr.*;
        e.value_ptr.* = .{};
        errdefer _ = mlx.mlx_array_free(arr);
        try list.append(self.allocator, .{ .key = key, .arr = arr });
    }
    var meta_pairs = std.ArrayList(MetaPair).empty;
    defer meta_pairs.deinit(self.allocator);
    var mit = loaded.meta.iterator();
    while (mit.next()) |e| try meta_pairs.append(self.allocator, .{ .key = e.key_ptr.*, .value = e.value_ptr.* });
    const path = try std.fmt.allocPrint(self.allocator, "{s}/e{d}/s{d:0>7}.safetensors", .{ self.root, h.id, h.pos });
    const sz = try self.saveNamed(path, list.items, meta_pairs.items, .ssm, s);
    self.point_bytes_written += sz;
    if (sz != h.bytes.len) log.warn("  [disk-cache] held point e{d}@{d} wrote {d} bytes, listed {d}\n", .{ h.id, h.pos, sz, h.bytes.len });
}

/// Runs once the index holds a commit: the points that left RAM unwritten during it, and those `fitHeld` drops,
/// are taken out of the record that names them, not the one it replaced.
pub fn settleHeld(self: *DiskTier, s: mlx.mlx_stream) void {
    for (self.lost_points.items) |k| if (self.heldIndex(k.id, k.pos) == null) self.unlistPoint(k.id, k.pos);
    self.lost_points.clearRetainingCapacity();
    self.fitHeld(s);
}

/// Fits the held points under the cap and the writer's share of the permit by writing them: a spare first (the
/// commit that made it holds the newer point after it), then the oldest.
pub fn fitHeld(self: *DiskTier, s: mlx.mlx_stream) void {
    while (self.held_bytes > self.heldCap()) {
        const j = for (self.held.items, 0..) |h, j| {
            if (h.spare) break j;
        } else break;
        self.flushHeldAt(j, s) catch |err| {
            log.warn("  [disk-cache] a held restore point could not be written: {s}; dropped\n", .{@errorName(err)});
        };
    }
    while (self.held.items.len > 0 and self.held_bytes > self.heldCap()) self.flushHeldAt(0, s) catch |err| {
        log.warn("  [disk-cache] a held restore point could not be written: {s}; dropped\n", .{@errorName(err)});
    };
}

/// Writes every held point: run where the process's RAM copy is about to go (handover, unload, idle, shutdown).
pub fn flushHeld(self: *DiskTier, why: []const u8, s: mlx.mlx_stream) void {
    if (self.held.items.len == 0) return;
    const n = self.held.items.len;
    const bytes = self.held_bytes;
    while (self.held.items.len > 0) self.flushHeldAt(0, s) catch |err| {
        log.warn("  [disk-cache] a held restore point could not be written ({s}): {s}; dropped\n", .{ why, @errorName(err) });
    };
    log.info("  [disk-cache] wrote {d} held restore points ({d:.1} MB) on {s}\n", .{ n, mib(bytes), why });
}


/// The resident entry whose leading chunk files a new entry for `tokens` may hard-link:
/// same tool flag and kv-quant config, most whole chunks below the prefix whose rows are the
/// commit's (`commitSharesWithChain`). Null on the synchronous legacy arm.
pub fn chunkShareDonor(self: *DiskTier, tokens: []const u32, kv_target: u32, has_tools: bool, config: kv_quant.KVQuantConfig, media: CommitMedia, chain: []const prefix_policy.MediaSpan, skip: ?usize) ?ChunkDonor {
    if (!(self.ssd_first or self.writer != null) or !chunkShareEnabled()) return null;
    var best: ?ChunkDonor = null;
    for (self.entries.items, 0..) |*e, i| {
        if (e.poisoned or i == skip) continue; // never inherit from a dead entry
        if (e.has_tools != has_tools) continue;
        if (!std.meta.eql(e.quant, config)) continue;
        const common: u32 = @min(@as(u32, @intCast(commonPrefixLen(e.tokens, tokens))), commitSharesWithChain(e, media, chain));
        const shared = @min(common, kv_target, e.kv_len);
        const whole: u32 = shared / self.chunk_tokens;
        const usable: u32 = @min(whole, @as(u32, @intCast(e.chunk_bytes.len)));
        if (usable == 0) continue;
        if (best == null or usable > best.?.chunks) best = .{ .idx = i, .id = e.id, .chunks = usable, .shared = common };
    }
    return best;
}

/// The resident entry holding the most row segments a new entry for `tokens` may link: whole spans inside the prefix
/// they share (`commitSharesWithChain`). Not the chunk donor: a turn that crossed no grid point holds chunks and no segments.
pub fn segmentDonor(self: *DiskTier, tokens: []const u32, kv_target: u32, has_tools: bool, config: kv_quant.KVQuantConfig, media: CommitMedia, chain: []const prefix_policy.MediaSpan) ?ChunkDonor {
    const g = self.grown orelse return null;
    if (!(self.ssd_first or self.writer != null) or !chunkShareEnabled()) return null;
    var best: ?ChunkDonor = null;
    for (self.entries.items, 0..) |*e, i| {
        if (e.poisoned or e.has_tools != has_tools or !std.meta.eql(e.quant, config) or e.grown_span != g.span) continue;
        const common: u32 = @min(@min(@as(u32, @intCast(commonPrefixLen(e.tokens, tokens))), commitSharesWithChain(e, media, chain)), self.rows_base);
        const usable: u32 = @min(@as(u32, @intCast(e.grown.len)), @min(common, kv_target) / g.span);
        if (usable == 0) continue;
        if (best == null or usable > best.?.chunks) best = .{ .idx = i, .id = e.id, .chunks = usable, .shared = common };
    }
    return best;
}

/// The entry a commit's rows were restored from, as a donor of the whole row segments below those rows.
pub fn restoredSegmentSource(self: *const DiskTier, restored: ?RestoredRows, tokens: []const u32, kv_target: u32, config: kv_quant.KVQuantConfig) ?ChunkDonor {
    const g = self.grown orelse return null;
    const r = restored orelse return null;
    if (!(self.ssd_first or self.writer != null) or !chunkShareEnabled()) return null;
    for (self.entries.items, 0..) |*e, i| {
        if (e.id != r.id) continue;
        if (e.poisoned or !std.meta.eql(e.quant, config) or e.grown_span != g.span) return null;
        const rows: u32 = @intCast(@min(@min(@min(r.rows, self.rows_base), kv_target), @min(e.kv_len, commonPrefixLen(e.tokens, tokens))));
        const usable: u32 = @min(rows / g.span, @as(u32, @intCast(e.grown.len)));
        if (usable == 0) return null;
        return .{ .idx = i, .id = e.id, .chunks = usable, .shared = rows };
    }
    return null;
}

/// `shared`: the leading tokens whose rows the donor and the heir provably agree on.
pub const ChunkDonor = struct { idx: usize, id: u64, chunks: u32, shared: u32 };

/// The entry a commit's rows were restored from, as a donor of the whole chunks below those rows.
pub fn restoredSource(self: *const DiskTier, restored: ?RestoredRows, tokens: []const u32, kv_target: u32, config: kv_quant.KVQuantConfig) ?ChunkDonor {
    const r = restored orelse return null;
    if (!(self.ssd_first or self.writer != null) or !chunkShareEnabled()) return null;
    for (self.entries.items, 0..) |*e, i| {
        if (e.id != r.id) continue;
        if (e.poisoned or !std.meta.eql(e.quant, config)) return null;
        const rows: u32 = @intCast(@min(@min(r.rows, kv_target), @min(e.kv_len, commonPrefixLen(e.tokens, tokens))));
        const whole: u32 = @min(rows / self.chunk_tokens, @as(u32, @intCast(e.chunk_bytes.len)));
        return .{ .idx = i, .id = e.id, .chunks = whole, .shared = rows };
    }
    return null;
}

/// Hard-link the donor's landed chunk files `[from, d.chunks)` into `e<id>/`; returns how many. Stops
/// at the first chunk not landed and never touches the donor's queue (`Writer.fence` would
/// discard it). A link failure unwinds this call's links and the caller writes those chunks itself.
pub fn linkInheritedChunks(self: *DiskTier, d: ChunkDonor, id: u64, from: u32, chunk_sizes: *std.ArrayList(u64)) !u32 {
    const donor_cb = self.entries.items[d.idx].chunk_bytes;
    var linked: u32 = 0;
    errdefer {
        self.unlinkChunkRange(id, from, from + linked);
        chunk_sizes.shrinkRetainingCapacity(from);
    }
    var i: u32 = from;
    while (i < d.chunks) : (i += 1) {
        const old_abs = try std.fmt.allocPrint(self.allocator, "{s}/e{d}/c{d:0>6}.safetensors", .{ self.root, d.id, i });
        defer self.allocator.free(old_abs);
        if (!self.chunkLanded(old_abs, donor_cb[i])) break;
        const new_abs = try std.fmt.allocPrint(self.allocator, "{s}/e{d}/c{d:0>6}.safetensors", .{ self.root, id, i });
        defer self.allocator.free(new_abs);
        self.fs.hardLink(old_abs, new_abs) catch |err| {
            log.warn("  [disk-cache] chunk share: link e{d}/c{d} -> e{d} failed: {s} — writing the chunks instead\n", .{ d.id, i, id, @errorName(err) });
            return error.ChunkShareLinkFailed;
        };
        linked += 1;
        try chunk_sizes.append(self.allocator, donor_cb[i]);
    }
    if (linked == 0) return 0;
    var mb: f64 = 0;
    for (donor_cb[from..][0..linked]) |b| mb += @as(f64, @floatFromInt(b));
    mb /= 1024.0 * 1024.0;
    log.info("  [disk-cache] chunk share: e{d} inherits {d} chunks ({d:.1} MB) from e{d} by hard link\n", .{ id, linked, mb, d.id });
    return linked;
}

/// Hard-link the donor's landed row segments `(from, limit]` that a point of this commit leans on and whose whole span lies inside
/// the prefix donor and heir share: rows are append-only, so those bytes are the heir's own. Appends their sizes
/// to `out`; a failure unwinds this call's links, leaving `out[0..from]` intact.
pub fn linkInheritedSegments(self: *DiskTier, d: ChunkDonor, id: u64, cps: ?[]const trunk.SSMCheckpoint, kv_target: u32, from: u32, out: *std.ArrayList(u64)) !void {
    const g = self.grown orelse return;
    const donor = &self.entries.items[d.idx];
    if (donor.grown_span != g.span) return;
    var lean: u32 = 0;
    for (cps orelse return) |*cp| {
        if (cp.isReleased() or cp.pos > kv_target) continue;
        lean = @max(lean, pointSegments(g.divs, g.span, cp));
    }
    const limit = @min(@min(lean, @as(u32, @intCast(donor.grown.len))), @min(d.shared, kv_target) / g.span);
    if (limit <= from) return;
    errdefer {
        var j: u32 = from + 1;
        while (j <= out.items.len) : (j += 1) self.unlinkSegment(id, j);
        out.shrinkRetainingCapacity(from);
    }
    var j: u32 = from + 1;
    while (j <= limit) : (j += 1) {
        const old_abs = try std.fmt.allocPrint(self.allocator, "{s}/e{d}/g{d:0>5}.safetensors", .{ self.root, d.id, j });
        defer self.allocator.free(old_abs);
        if (!self.chunkLanded(old_abs, donor.grown[j - 1])) break;
        const new_abs = try std.fmt.allocPrint(self.allocator, "{s}/e{d}/g{d:0>5}.safetensors", .{ self.root, id, j });
        defer self.allocator.free(new_abs);
        try self.fs.hardLink(old_abs, new_abs);
        try out.append(self.allocator, donor.grown[j - 1]);
    }
    if (out.items.len > from) log.info("  [disk-cache] segment share: e{d} inherits {d} row segments from e{d} by hard link\n", .{ id, out.items.len - from, d.id });
}

pub fn unlinkSegment(self: *DiskTier, id: u64, j: u32) void {
    const path = std.fmt.allocPrint(self.allocator, "{s}/e{d}/g{d:0>5}.safetensors", .{ self.root, id, j }) catch return;
    defer self.allocator.free(path);
    self.fs.deleteFile(path);
}

/// Has a chunk file landed: final name, recorded size, and no write to it queued or in flight?
pub fn chunkLanded(self: *DiskTier, abs_path: []const u8, want_size: u64) bool {
    const st = self.fs.stat(abs_path) orelse return false;
    if (st.size != want_size) return false;
    if (self.writer) |w| {
        if (w.isPending(abs_path)) return false;
    }
    return true;
}

pub fn unlinkChunkRange(self: *DiskTier, id: u64, from: u32, to: u32) void {
    var i: u32 = from;
    while (i < to) : (i += 1) {
        const path = std.fmt.allocPrint(self.allocator, "{s}/e{d}/c{d:0>6}.safetensors", .{ self.root, id, i }) catch continue;
        defer self.allocator.free(path);
        self.fs.deleteFile(path);
    }
}


/// Before `idx` is evicted: hard-link its highest point inside the prefix a resident entry provably holds
/// the same rows for into that entry, when it beats that entry's own. Heirs keep points only past where they
/// diverged, so once the oldest record of a conversation went a request that changed an early image restored
/// nothing. The point's bytes move to the heir's bill: the tier's total stays the bytes on disk.
pub fn bequeathPoint(self: *DiskTier, idx: usize) void {
    const dead = &self.entries.items[idx];
    if (dead.poisoned or dead.rings.len > 0) return;
    var best: ?struct { heir: usize, pos: u32, bytes: u64 } = null;
    for (self.entries.items, 0..) |*e, i| {
        if (i == idx or e.poisoned or e.rings.len > 0 or e.has_tools != dead.has_tools or !std.meta.eql(e.quant, dead.quant)) continue;
        const media_cap = commitShares(e, .{ .key = dead.vision_key, .start = dead.media_start });
        const token_shared: u32 = @intCast(@min(commonPrefixLen(e.tokens, dead.tokens), e.kv_len, dead.kv_len));
        const shared: u32 = if (self.grown != null) token_shared else @min(token_shared, media_cap);
        const own: u32 = self.highestSsmPosAtOrBelow(i, shared) orelse 0;
        for (dead.ssm_positions, dead.ssm_bytes) |p, b| {
            if (p > shared) break;
            if (p <= own or (best != null and p <= best.?.pos)) continue;
            // Past `media_cap` on a split tier, only a grid point whose every segment is a hard link
            // proves identical rows; an off-grid tail is unverified past `k * span`.
            if (p > media_cap and (dead.grown_span == 0 or p % dead.grown_span != 0)) continue;
            if (!self.sameRowsBelow(dead, e, p)) continue;
            best = .{ .heir = i, .pos = p, .bytes = b };
        }
    }
    const pick = best orelse return;
    self.linkPointInto(dead.id, pick.heir, pick.pos, pick.bytes) catch |err| {
        log.warn("  [disk-cache] keeping e{d}'s point at {d} failed: {s}\n", .{ dead.id, pick.pos, @errorName(err) });
        return;
    };
    log.info("  [disk-cache] e{d} keeps e{d}'s restore point at {d} by hard link\n", .{ self.entries.items[pick.heir].id, dead.id, pick.pos });
}

/// Do `a` and `b` hold the same bytes for every row a restore at `p` reads: the same KV chunk files,
/// the same row segments on a split tier, and on a QSA record the same history segments?
pub fn sameRowsBelow(self: *DiskTier, a: *const IndexEntry, b: *const IndexEntry, p: u32) bool {
    if (!self.points_hold_state or a.chunk_bytes.len > 0 or b.chunk_bytes.len > 0) {
        const n_chunks = (p + self.chunk_tokens - 1) / self.chunk_tokens;
        if (a.chunk_bytes.len < n_chunks or b.chunk_bytes.len < n_chunks) return false;
        for (0..n_chunks) |i| {
            var pa: [1024]u8 = undefined;
            var pb: [1024]u8 = undefined;
            const fa = std.fmt.bufPrint(&pa, "{s}/e{d}/c{d:0>6}.safetensors", .{ self.root, a.id, i }) catch return false;
            const fb = std.fmt.bufPrint(&pb, "{s}/e{d}/c{d:0>6}.safetensors", .{ self.root, b.id, i }) catch return false;
            const sa = self.fs.stat(fa) orelse return false;
            const sb = self.fs.stat(fb) orelse return false;
            if (sa.inode != sb.inode) return false;
        }
    } else if (self.grown == null) {
        return false;
    }
    if (self.grown != null) {
        if (a.grown_span == 0 or a.grown_span != b.grown_span) return false;
        const k = p / a.grown_span;
        if (a.grown.len < k or b.grown.len < k) return false;
        for (1..k + 1) |j| {
            var pa: [1024]u8 = undefined;
            var pb: [1024]u8 = undefined;
            const fa = std.fmt.bufPrint(&pa, "{s}/e{d}/g{d:0>5}.safetensors", .{ self.root, a.id, j }) catch return false;
            const fb = std.fmt.bufPrint(&pb, "{s}/e{d}/g{d:0>5}.safetensors", .{ self.root, b.id, j }) catch return false;
            const sa = self.fs.stat(fa) orelse return false;
            const sb = self.fs.stat(fb) orelse return false;
            if (sa.inode != sb.inode) return false;
        }
    }
    if (a.qsa_history_rows == 0 and b.qsa_history_rows == 0) return true;
    if (b.qsa_history_rows < p or self.seg_span == 0 or p % self.seg_span != 0) return false;
    const k = p / self.seg_span;
    if (a.qsa_segs.len < k or b.qsa_segs.len < k) return false;
    for (a.qsa_segs[0..k], b.qsa_segs[0..k]) |x, y| if (x.hash != y.hash or x.bytes != y.bytes) return false;
    return true;
}

/// Hard-link entry `from`'s point at `pos` into entry `heir_idx` and list it there, billed to it.
pub fn linkPointInto(self: *DiskTier, from: u64, heir_idx: usize, pos: u32, bytes: u64) !void {
    const heir = &self.entries.items[heir_idx];
    var ab: [std.fs.max_path_bytes]u8 = undefined;
    var nb: [std.fs.max_path_bytes]u8 = undefined;
    const old_abs = try std.fmt.bufPrint(&ab, "{s}/e{d}/s{d:0>7}.safetensors", .{ self.root, from, pos });
    const new_abs = try std.fmt.bufPrint(&nb, "{s}/e{d}/s{d:0>7}.safetensors", .{ self.root, heir.id, pos });
    if (self.fs.size(old_abs) != bytes) return error.PointNotLanded;
    try self.fs.hardLink(old_abs, new_abs);
    errdefer self.fs.deleteFile(new_abs);
    var at: usize = 0;
    while (at < heir.ssm_positions.len and heir.ssm_positions[at] < pos) at += 1;
    const positions = try self.allocator.alloc(u32, heir.ssm_positions.len + 1);
    errdefer self.allocator.free(positions);
    const sizes = try self.allocator.alloc(u64, heir.ssm_bytes.len + 1);
    errdefer self.allocator.free(sizes);
    @memcpy(positions[0..at], heir.ssm_positions[0..at]);
    positions[at] = pos;
    @memcpy(positions[at + 1 ..], heir.ssm_positions[at..]);
    @memcpy(sizes[0..at], heir.ssm_bytes[0..at]);
    sizes[at] = bytes;
    @memcpy(sizes[at + 1 ..], heir.ssm_bytes[at..]);
    var next = heir.*;
    next.ssm_positions = positions;
    next.ssm_bytes = sizes;
    next.bytes += bytes;
    try self.writeMeta(next);
    self.allocator.free(heir.ssm_positions);
    self.allocator.free(heir.ssm_bytes);
    heir.* = next;
    self.total_bytes += bytes;
}

