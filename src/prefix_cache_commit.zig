//! Hot prefix cache commit path, SSD spill/write-through/flush, and checkpoint/ring donor merging.

const std = @import("std");
const mlx = @import("mlx.zig");
const trunk_mod = @import("trunk.zig");
const kv_quant = @import("kv_quant.zig");
const kv_disk_cache = @import("kv_disk_cache.zig");
const log = @import("log.zig");
const prefix_arch = @import("arch/prefix.zig");
const prefix_policy = @import("prefix_policy.zig");
const pc = @import("prefix_cache.zig");

const KVCache = trunk_mod.KVCache;
const KVCacheSnapshot = trunk_mod.KVCacheSnapshot;
const SSMCacheEntry = trunk_mod.SSMCacheEntry;
const SSMCheckpoint = trunk_mod.SSMCheckpoint;
const checkpointHasQsaPooled = prefix_arch.aux.checkpointHasQsaPooled;
const checkpointListHasQsaPooled = prefix_arch.aux.checkpointListHasQsaPooled;
const sliceQsaHistoryOntoCheckpoint = prefix_arch.aux.sliceQsaHistoryOntoCheckpoint;
const keepOnlyLatestQsaHistory = prefix_arch.aux.keepOnlyLatestQsaHistory;
const ssmCheckpointBytes = trunk_mod.ssmCheckpointBytes;

const HotPrefixCache = pc.HotPrefixCache;
const Entry = pc.Entry;
const RamMatch = pc.RamMatch;
const TierRows = pc.TierRows;
const MediaSpan = pc.MediaSpan;
const MediaRule = pc.MediaRule;
const MediaStamp = pc.MediaStamp;
const TrimDecline = pc.TrimDecline;
const CommitStatus = pc.CommitStatus;
const SlotRingCps = pc.SlotRingCps;
const PendingDiskFlush = pc.PendingDiskFlush;
const DflashSnap = pc.DflashSnap;
const DflashCommit = pc.DflashCommit;
const diskTextLen = pc.diskTextLen;
const crossKeyBoundary = pc.crossKeyBoundary;
const RING_RESTORE_MIN_TOKENS = pc.RING_RESTORE_MIN_TOKENS;
const RING_CHECKPOINT_MAX = pc.RING_CHECKPOINT_MAX;
const RING_CHECKPOINT_RAM_KEEP = pc.RING_CHECKPOINT_RAM_KEEP;
const SLOT_RING_CHECKPOINTS = pc.SLOT_RING_CHECKPOINTS;
const CHECKPOINT_HOT_KEPT = pc.CHECKPOINT_HOT_KEPT;
const snapshotBytes = HotPrefixCache.snapshotBytes;
const specSnapBytes = HotPrefixCache.specSnapBytes;
const snapshotRowBytes = HotPrefixCache.snapshotRowBytes;
const ringedBytes = HotPrefixCache.ringedBytes;
const ringTrimLen = HotPrefixCache.ringTrimLen;
const logTrimInputs = HotPrefixCache.logTrimInputs;
const truncateCheckpoints = HotPrefixCache.truncateCheckpoints;
const highestCheckpointAtOrBelow = HotPrefixCache.highestCheckpointAtOrBelow;
const endCheckout = HotPrefixCache.endCheckout;

/// Commit the current `source_cache` state under the given key. Updates
/// the matching entry if one exists for this exact prefix, otherwise
/// inserts a new entry, evicting the oldest if at capacity. Snapshot is
/// taken here (cheap — refcount-share, no data copy).
pub fn commit(
    self: *HotPrefixCache,
    source_cache: *const KVCache,
    tokens: []const u32,
    has_tools: bool,
) !CommitStatus {
    return self.commitWithSsm(source_cache, tokens, has_tools, null, null, null);
}

/// Commit with optional SSM checkpoint array (Phase 1). The caller
/// transfers ownership of the slice — the entry frees it on eviction via
/// the shared `freeEntryOwnedState`. Pass null on plain-attention archs;
/// the entry stays SSM-free.
pub fn commitWithSsm(
    self: *HotPrefixCache,
    source_cache: *const KVCache,
    tokens: []const u32,
    has_tools: bool,
    ssm_cps: ?[]SSMCheckpoint,
    dflash: ?DflashCommit,
    mtp: ?DflashCommit,
) !CommitStatus {
    return self.commitWithState(source_cache, tokens, has_tools, 0, ssm_cps, dflash, mtp);
}


/// Commit with SSM checkpoints; ownership of the payload transfers to
/// the entry.
pub fn commitWithState(
    self: *HotPrefixCache,
    source_cache: *const KVCache,
    tokens: []const u32,
    has_tools: bool,
    vision_key: u64,
    ssm_cps: ?[]SSMCheckpoint,
    dflash: ?DflashCommit,
    mtp: ?DflashCommit,
) !CommitStatus {
    return self.commitWithMediaState(source_cache, tokens, has_tools, vision_key, 0, null, ssm_cps, dflash, mtp, tokens.len);
}

/// `prompt_len` is the committing request's PROMPT length inside `tokens`
/// (which may carry the generated tail too): inherited checkpoints are capped
/// at it, because a donor's checkpoints past the prompt hold the donor's own
/// generation and a greedy continuation can match those tokens verbatim.
/// `commitWithState` passes `tokens.len`, right only for a prompt-only array.
pub fn commitWithMediaState(
    self: *HotPrefixCache,
    source_cache: *const KVCache,
    tokens: []const u32,
    has_tools: bool,
    vision_key: u64,
    cache_key: u64,
    media_start: ?usize,
    ssm_cps: ?[]SSMCheckpoint,
    dflash: ?DflashCommit,
    mtp: ?DflashCommit,
    prompt_len: usize,
) !CommitStatus {
    return self.commitWithRing(source_cache, tokens, has_tools, vision_key, cache_key, media_start, ssm_cps, dflash, mtp, prompt_len, .{});
}

/// `ring_cps` are a ringed slot's restore points (`SlotRingCps`); ownership transfers to
/// the cache like `ssm_cps`.
pub fn commitWithRing(
    self: *HotPrefixCache,
    source_cache: *const KVCache,
    tokens: []const u32,
    has_tools: bool,
    vision_key: u64,
    cache_key: u64,
    media_start: ?usize,
    ssm_cps: ?[]SSMCheckpoint,
    dflash: ?DflashCommit,
    mtp: ?DflashCommit,
    prompt_len: usize,
    ring_cps: SlotRingCps,
) !CommitStatus {
    return self.commitImpl(source_cache, tokens, has_tools, vision_key, cache_key, media_start, ssm_cps, dflash, mtp, prompt_len, ring_cps, .{});
}

pub fn commitImpl(
    self: *HotPrefixCache,
    source_cache: *const KVCache,
    tokens: []const u32,
    has_tools: bool,
    vision_key: u64,
    cache_key: u64,
    media_start: ?usize,
    all_cps: ?[]SSMCheckpoint,
    dflash: ?DflashCommit,
    mtp: ?DflashCommit,
    prompt_len: usize,
    ring_cps: SlotRingCps,
    tier_rows: TierRows,
) !CommitStatus {
    const end_note = self.commit_end_note;
    self.commit_end_note = "";
    const restored = self.commit_restored;
    self.commit_restored = null;
    const media_chain = self.takeMediaChain();
    // Freed on every path that does not move it into an entry.
    var new_rings = ownRingCps(self.allocator, ring_cps);
    defer if (new_rings) |r| freeRingCps(self.allocator, r);
    var new_ring_bytes = ringCpsBytes(new_rings);
    var new_rows = tier_rows.ram;
    defer if (new_rows) |*rows| rows.deinit();
    var disk_rows = tier_rows.disk;
    defer if (disk_rows) |*rows| rows.deinit();
    var new_rows_bytes: u64 = if (new_rows) |*rows| rows.bytes() else 0;
    const quant_config = source_cache.config;
    var ssm_cps = all_cps;

    // An entry's pixel key applies only to rows it actually covers. When the committed range
    // ends before the request's first media row — a cancelled prefill that forwarded less than
    // the media position — the entry is pure text and must carry the NULL key, or the
    // boundary-less cross-key entry gets conservatively rejected by the lookup (live
    // 2026-09-07: the with-image and no-image retry shapes of one 122k session locked each
    // other out of a shared 121819-token text prefix). A null `media_start` keeps the caller's
    // key as-is (commitWithState's pixel-keyed, boundary-less contract). A caller holding the
    // chain narrows the key itself (`truncatedMediaKey`); this is the one case reachable here.
    var eff_media_start = media_start;
    var eff_vision_key = vision_key;
    if (media_start) |ms| {
        if (ms >= tokens.len) {
            eff_media_start = null;
            eff_vision_key = 0;
        }
    }

    // Taken before the budget trim can drop the first item.
    const disk_len = diskTextLen(tokens.len, eff_vision_key, eff_media_start);
    // Record what the live cache holds now, before any byte-budget trim: the SSD tier gets the
    // whole entry whatever the RAM tier keeps.
    var record_taken = false;
    if (self.capturesLiveForDisk() and disk_len > 0) {
        const rows: ?*const prefix_arch.Rows = if (disk_rows) |*r| r else if (new_rows) |*r| r else null;
        self.capturePendingDisk(source_cache, tokens, disk_len, has_tools, eff_vision_key, eff_media_start, media_chain, restored, prompt_len, ssm_cps, dflash, mtp, new_rings, rows);
        // Whatever the RAM tier then does with the candidate, the record is flushed.
        record_taken = self.pending_disk != null;
        if (record_taken) self.disk_dirty = true;
    }
    // The record shares the live KV; on an error return nothing consumes it and the slot's
    // KVCache deinit then frees nothing. Function scope on purpose.
    errdefer if (self.pending_disk) |*p| {
        p.deinit(self.allocator);
        self.pending_disk = null;
    };

    // An entry with SSD rows but none within the RAM budget keeps nothing in RAM.
    if (!self.ram_enabled or (tier_rows.disk != null and new_rows == null)) {
        if (ssm_cps) |cps| {
            for (cps) |*cp| cp.deinit(self.allocator);
            self.allocator.free(cps);
        }
        if (self.pending_disk == null) return .declined;
        self.disk_dirty = true;
        return .{ .disk_only = tokens.len };
    }
    // The RAM entry keeps only the checkpoints its own rows restore.
    if (new_rows) |*rows| if (ssm_cps) |cps| {
        ssm_cps = rows.keepCheckpoints(self.allocator, cps);
    };
    // The pending record holds every checkpoint for the SSD tier, so the hot entry keeps only its newest.
    // A GDN checkpoint past the record's text prefix is not in it and stays.
    const hot_keeps_newest = self.pending_disk != null and (new_rows != null or (tier_rows.ram == null and tier_rows.disk == null));
    const record_text: usize = if (new_rows != null) std.math.maxInt(usize) else disk_len;
    // GDN checkpoints are cut after the byte-budget trim below, which lands on one of them.
    if (hot_keeps_newest and new_rows != null) self.keepNewestCheckpoints(&ssm_cps, record_text);

    var replace_idx: ?usize = null;
    for (self.entries.items, 0..) |*e, i| {
        if (e.has_tools != has_tools) continue;
        if (e.vision_key != eff_vision_key) continue;
        if (!std.meta.eql(e.quant_config, quant_config)) continue;
        if (e.tokens.len <= tokens.len) {
            var shared: usize = 0;
            while (shared < e.tokens.len and e.tokens[shared] == tokens[shared]) shared += 1;
            if (shared == e.tokens.len) {
                replace_idx = i;
                break;
            }
        }
    }

    // Ownership of `ssm_cps` is ours from entry; no cleanup is armed until here.
    // The record holds this cache's retained rows already: sharing them spares a ring's second copy.
    var new_snap = (if (record_taken) self.pending_disk.?.snapshot.share() else source_cache.snapshotRetained(self.arraysStream())) catch |err| {
        if (ssm_cps) |cps| {
            for (cps) |*cp| cp.deinit(self.allocator);
            self.allocator.free(cps);
        }
        return err;
    };
    // The speculative-side payloads are best-effort: a snapshot failure
    // must not cost the trunk KV entry they ride on.
    var new_dflash: ?DflashSnap = null;
    var new_dflash_bytes: u64 = 0;
    if (dflash) |d| {
        if (d.cache.snapshot()) |snap| {
            new_dflash = .{ .snapshot = snap, .base_pos = d.base_pos };
            new_dflash_bytes = snapshotBytes(&new_dflash.?.snapshot);
        } else |err| {
            log.warn("  [hot-cache] dflash context snapshot failed: {s}\n", .{@errorName(err)});
        }
    }
    var new_mtp: ?DflashSnap = null;
    var new_mtp_bytes: u64 = 0;
    if (mtp) |m2| {
        if (m2.cache.snapshot()) |snap| {
            new_mtp = DflashSnap.ofCommit(m2, snap, self.arraysStream());
            new_mtp_bytes = specSnapBytes(&new_mtp.?);
        } else |err| {
            log.warn("  [hot-cache] mtp history snapshot failed: {s}\n", .{@errorName(err)});
        }
    }
    var new_kv_bytes = snapshotBytes(&new_snap);
    var new_ssm_bytes: u64 = 0;
    if (ssm_cps) |cps| {
        for (cps) |*cp| new_ssm_bytes += ssmCheckpointBytes(cp);
    }
    var new_bytes = new_kv_bytes + new_ssm_bytes + new_dflash_bytes + new_mtp_bytes + new_ring_bytes + new_rows_bytes;
    // Effective candidate: a byte-budget trim below shortens these.
    var eff_tokens = tokens;
    var eff_cps = ssm_cps;
    // The byte budget is a hard retention cap, including for the first (or
    // only) entry. The old eviction loop could empty the cache and then
    // append an entry larger than the cap, defeating the load-time clamp
    // precisely for long single-conversation prefixes (#326). But a flat
    // decline is a CLIFF (#330): a long agent session crosses the budget
    // once mid-conversation and then cold-prefills every turn while the
    // cap "holds" zero bytes. Retain the longest restorable prefix that
    // fits instead; decline only when nothing above the floor does.
    if (self.max_kv_bytes > 0 and new_bytes > self.max_kv_bytes) {
        var trimmed_ok = false;
        var decline: TrimDecline = .no_restorable_prefix;
        var decline_err: ?anyerror = null;
        var limit = if (eff_media_start) |ms| @min(tokens.len, ms) else tokens.len;
        var inputs_logged = false;
        trim_blk: while (true) {
            const row_bytes = snapshotRowBytes(&new_snap) + (if (new_rows) |*rows| rows.rowBytes() else 0);
            const ring_bytes = ringedBytes(&new_snap);
            var ring_cp: ?*const KVCacheSnapshot = null;
            const tl_opt = if (ring_bytes > 0) blk: {
                const rt = ringTrimLen(self.max_kv_bytes, limit, row_bytes, tokens.len, ring_bytes, new_rings orelse &.{}) orelse break :blk null;
                if (rt.cp) |k| ring_cp = &new_rings.?[k];
                break :blk rt.len;
            } else self.trimLenForBudget(self.max_kv_bytes -| (if (new_rows != null) new_dflash_bytes else 0), limit, row_bytes, eff_cps);
            if (!inputs_logged) {
                inputs_logged = true;
                logTrimInputs(tokens.len, row_bytes, self.max_kv_bytes, eff_cps, tl_opt, self.cp_thin != .min_span);
            }
            const tl = tl_opt orelse break :trim_blk;
            // One-shot: when the resident covered entry already retains
            // the trim target, keep it and drop the candidate — the
            // target is budget-derived and stable, so replacing would
            // re-copy an identical multi-GB prefix every turn.
            if (replace_idx) |idx| {
                // ...unless the entry is checked out: its snapshot is empty, so fall through to the trim/replace.
                if (self.entries.items[idx].checkout == null and
                    self.entries.items[idx].tokens.len >= tl)
                {
                    // The resident entry already covers the trim target;
                    // the candidate's EXTRA tokens still belong on disk.
                    self.spillDeclinedToDisk(&new_snap, tokens, diskTextLen(tokens.len, eff_vision_key, eff_media_start), has_tools, MediaStamp.of(eff_vision_key, eff_media_start), eff_cps, new_rings, if (new_rows) |*rows| rows else null);
                    var discarded = new_snap;
                    discarded.deinit();
                    if (new_dflash) |*d| d.deinit();
                    if (new_mtp) |*m3| m3.deinit();
                    if (eff_cps) |cps| {
                        for (cps) |*cp| cp.deinit(self.allocator);
                        self.allocator.free(cps);
                    }
                    log.info("  [hot-cache] kept resident {d}-token prefix; oversized candidate ({d} tokens, {d:.2} MB > {d:.2} MB budget) trims no further\n", .{
                        self.entries.items[idx].tokens.len,
                        tokens.len,
                        @as(f64, @floatFromInt(new_bytes)) / (1024.0 * 1024.0),
                        @as(f64, @floatFromInt(self.max_kv_bytes)) / (1024.0 * 1024.0),
                    });
                    return .{ .kept_resident = self.entries.items[idx].tokens.len };
                }
            }
            const trimmed = new_snap.trimmedCopy(tl, ring_cp, self.arraysStream()) catch |err| {
                // A copy that failed at this width is not a verdict on the entry: retry at the next-lower checkpoint.
                decline = .snapshot_copy_failed;
                decline_err = err;
                // Arch gate: a retry allocates again under memory pressure; ungated declines as before.
                if (self.cp_thin == .min_span) break :trim_blk;
                log.warn("  [hot-cache] trimmed copy to {d} tokens failed: {s}; retrying at the next-lower checkpoint\n", .{ tl, @errorName(err) });
                if (tl == 0) break :trim_blk;
                limit = tl - 1;
                continue;
            };
            new_snap.deinit();
            new_snap = trimmed;
            if (new_rows) |*rows| if (tl < rows.len()) {
                const short = rows.trimmedCopy(tl, self.arraysStream()) catch |err| {
                    decline = .snapshot_copy_failed;
                    decline_err = err;
                    break :trim_blk;
                };
                rows.deinit();
                new_rows = short;
                new_rows_bytes = short.bytes();
            };
            // Spec payloads describe the FULL-length state; a trimmed
            // prefix rebuilds them on its first reused turn. A rows arch's window is the prompt end's,
            // which a restore at any lower checkpoint clamps (`specAdoptPlan`).
            if (new_rows == null) if (new_dflash) |*d| {
                d.deinit();
                new_dflash = null;
                new_dflash_bytes = 0;
            };
            if (new_mtp) |*m3| {
                m3.deinit();
                new_mtp = null;
                new_mtp_bytes = 0;
            }
            if (eff_cps) |cps| {
                eff_cps = truncateCheckpoints(self.allocator, cps, tl, self.arraysStream()) catch |err| {
                    decline = switch (err) {
                        error.QsaHistorySliceFailed => .qsa_history_slice_failed,
                        error.CheckpointListCopyFailed => .checkpoint_list_copy_failed,
                    };
                    decline_err = err;
                    break :trim_blk;
                };
            }
            eff_tokens = tokens[0..tl];
            if (eff_media_start) |ms| {
                if (ms >= tl) {
                    // The trim cut below the media boundary: the entry
                    // is pure text now — re-key it, and drop the replace
                    // target (it was scanned under the pixel key, a
                    // different keyspace; the image-keyed entry stays
                    // behind for its own requests).
                    eff_media_start = null;
                    eff_vision_key = 0;
                    replace_idx = null;
                }
            }
            new_kv_bytes = snapshotBytes(&new_snap);
            new_ssm_bytes = 0;
            if (eff_cps) |cps| {
                for (cps) |*cp| new_ssm_bytes += ssmCheckpointBytes(cp);
            }
            // Only this slot's own checkpoints so far; inheritance runs below. A checkpoint
            // the trim landed on is the entry's ring now.
            if (new_rings) |r| {
                new_rings = ringCpsUpTo(self.allocator, r, if (ring_cp != null) tl - 1 else tl);
                new_ring_bytes = ringCpsBytes(new_rings);
            }
            new_bytes = new_kv_bytes + new_ssm_bytes + new_ring_bytes + new_rows_bytes + new_dflash_bytes;
            log.info("  [hot-cache] trimmed oversized entry to {d}/{d} tokens ({d:.2} MB before checkpoint shedding; {d:.2} MB budget)\n", .{
                tl,
                tokens.len,
                @as(f64, @floatFromInt(new_bytes)) / (1024.0 * 1024.0),
                @as(f64, @floatFromInt(self.max_kv_bytes)) / (1024.0 * 1024.0),
            });
            trimmed_ok = true;
            break;
        }
        if (!trimmed_ok) {
            // RAM decline is not a value verdict: offer the candidate to
            // the SSD tier before discarding it.
            self.spillDeclinedToDisk(&new_snap, tokens, diskTextLen(tokens.len, eff_vision_key, eff_media_start), has_tools, MediaStamp.of(eff_vision_key, eff_media_start), eff_cps, new_rings, if (new_rows) |*rows| rows else null);
            var discarded_snap = new_snap;
            discarded_snap.deinit();
            if (new_dflash) |*d| d.deinit();
            if (new_mtp) |*m3| m3.deinit();
            if (eff_cps) |cps| {
                for (cps) |*cp| cp.deinit(self.allocator);
                self.allocator.free(cps);
            }
            const err_sep: []const u8 = if (decline_err != null) ": " else "";
            const err_name: []const u8 = if (decline_err) |e| @errorName(e) else "";
            log.info("  [hot-cache] skipped oversized entry ({d} tokens, {d:.2} MB > {d:.2} MB budget): {s}{s}{s}\n", .{
                tokens.len,
                @as(f64, @floatFromInt(new_bytes)) / (1024.0 * 1024.0),
                @as(f64, @floatFromInt(self.max_kv_bytes)) / (1024.0 * 1024.0),
                decline.reason(),
                err_sep,
                err_name,
            });
            return .declined;
        }
    }
    // Ownership of the checkpoint slice transfers to the cache
    // UNCONDITIONALLY — success, decline, or error (#330 adjacent: the
    // scheduler's `catch` arm also freed them, so every failed commit was
    // a double free, with a different allocator). After a trim `eff_cps`
    // may be a cache-allocated replacement of the caller's slice, so the
    // cache is the only party that can still free correctly.
    const tokens_owned = self.allocator.dupe(u32, eff_tokens) catch |err| {
        var snap = new_snap;
        snap.deinit();
        if (new_dflash) |*d| d.deinit();
        if (new_mtp) |*m3| m3.deinit();
        if (eff_cps) |cps| {
            for (cps) |*cp| cp.deinit(self.allocator);
            self.allocator.free(cps);
        }
        return err;
    };

    // A commit built on a RESTORED prefix must be at least as restorable
    // as the entry it restored from. The replace path below inherits from
    // the entry it overwrites; a commit that lands as a NEW entry had no
    // inheritance at all, and that is the common shape whenever one prompt
    // is answered twice (an MTP arm then a serial arm, two clients, a
    // retry): the two entries agree on the whole prompt and diverge in
    // their generated tails, so neither is a prefix of the other. The
    // second entry then holds no snapshot of its own: a restored ~31-token
    // tail forwards as one span (`ssmSnapshotBackoff`). Evict the first and
    // the prompt becomes uncacheable.
    //
    // Inherit by refcount-SHARE, never copy: the buffers are already
    // resident, so this costs GPU memory only in the accounting, and only
    // until the donor is evicted.
    if (replace_idx == null) inherit: {
        const donor = self.bestCheckpointDonor(eff_tokens, has_tools, eff_vision_key, eff_media_start, media_chain, quant_config) orelse
            break :inherit;
        const budget: ?u64 = if (self.max_kv_bytes == 0)
            null
        else if (new_bytes >= self.max_kv_bytes)
            break :inherit
        else
            self.max_kv_bytes - new_bytes;
        const donor_cps = self.entries.items[donor.idx].ssm_checkpoints.?;
        const prompt_cap = @min(if (prompt_len == 0) eff_tokens.len else prompt_len, eff_tokens.len);
        const inherit_limit = @min(donor.shared, prompt_cap);
        const cloned = (cloneCheckpointsWithBank(self.allocator, donor_cps, inherit_limit, budget, self.arraysStream()) catch |err| {
            log.warn("  [hot-cache] checkpoint inheritance failed: {s}\n", .{@errorName(err)});
            break :inherit;
        }) orelse break :inherit;
        if (eff_cps) |own| {
            // Consumes both on every path; on error neither survives.
            const merged = self.mergeCheckpointLists(cloned, own) catch |err| {
                log.warn("  [hot-cache] checkpoint merge failed: {s}\n", .{@errorName(err)});
                eff_cps = null;
                new_bytes -= new_ssm_bytes;
                new_ssm_bytes = 0;
                break :inherit;
            };
            if (merged) |m| {
                eff_cps = m;
            } else {
                eff_cps = null;
                new_bytes -= new_ssm_bytes;
                new_ssm_bytes = 0;
                break :inherit;
            }
        } else {
            eff_cps = cloned;
        }
        if (eff_cps) |cps| {
            if (self.takeCpsIfQsaBank(cps)) |kept| {
                eff_cps = kept;
            } else {
                eff_cps = null;
                new_bytes -= new_ssm_bytes;
                new_ssm_bytes = 0;
                break :inherit;
            }
        }
        var inherited_bytes: u64 = 0;
        for (eff_cps.?) |*cp| inherited_bytes += ssmCheckpointBytes(cp);
        new_bytes = new_bytes - new_ssm_bytes + inherited_bytes;
        new_ssm_bytes = inherited_bytes;
        log.info("  [hot-cache] inherited {d} checkpoints (<= {d} tokens) from a shared prefix\n", .{
            eff_cps.?.len,
            inherit_limit,
        });
    }

    // The ring's twin of the checkpoint inheritance above: a request that restored off a
    // donor and appended its own tail keeps the donor's restore points below the fork, so a
    // later request diverging there restores after the donor is gone.
    if (replace_idx == null) ring_inherit: {
        const cap = @min(if (prompt_len == 0) eff_tokens.len else prompt_len, eff_tokens.len);
        const donor = self.bestRingDonor(eff_tokens, has_tools, eff_vision_key, eff_media_start, media_chain, quant_config, cap) orelse
            break :ring_inherit;
        const budget: ?u64 = if (self.max_kv_bytes == 0)
            null
        else if (new_bytes >= self.max_kv_bytes)
            break :ring_inherit
        else
            self.max_kv_bytes - new_bytes;
        const inherited = (shareRingCpsUpTo(self.allocator, self.entries.items[donor.idx].ring_cps.?, donor.limit, budget) catch |err| {
            log.warn("  [hot-cache] ring checkpoint inheritance failed: {s}\n", .{@errorName(err)});
            break :ring_inherit;
        }) orelse break :ring_inherit;
        const merged = mergeRingCps(self.allocator, inherited, new_rings);
        new_rings = merged catch |err| blk: {
            log.warn("  [hot-cache] ring checkpoint merge failed: {s}\n", .{@errorName(err)});
            break :blk null;
        };
        const merged_bytes = ringCpsBytes(new_rings);
        new_bytes = new_bytes - new_ring_bytes + merged_bytes;
        new_ring_bytes = merged_bytes;
        log.info("  [hot-cache] inherited ring checkpoints (<= {d} tokens) from a shared prefix\n", .{donor.limit});
    }

    if (replace_idx) |idx| {
        const e = &self.entries.items[idx];

        // Phase 1: SSM checkpoint inheritance on prefix-extend. The
        // replace path triggers when the new entry's tokens fully
        // extend the old's (i.e., e.tokens is a prefix of `tokens`).
        // The old SSM checkpoints were captured at positions inside
        // e.tokens, so they're still valid for the new entry — those
        // positions are a strict prefix of `tokens`. Inherit them and
        // append any new checkpoints from this turn that don't overlap.
        //
        // Without this, multi-turn flows lose checkpoints fast: turn 2's
        // prefill of the short tail captures few or no checkpoints, so
        // turn 3 has nothing to restore from even though turn 2's match
        // covered nearly the full prefix. (Reproducible by alternating
        // identical-prompt requests at ssm_checkpoint_stride > prompt_len.)
        var merged_cps: ?[]SSMCheckpoint = blk: {
            const old = e.ssm_checkpoints orelse break :blk eff_cps;
            // Detach old from its container either way: it is either moved
            // wholesale or consumed by the merge, and the free-below must
            // not touch it.
            e.ssm_checkpoints = null;
            const new = eff_cps orelse break :blk old;
            break :blk self.mergeCheckpointLists(old, new) catch |err| {
                // The merge consumed both lists, so the entry has no checkpoints to restore from: evict it.
                self.allocator.free(tokens_owned);
                var snap = new_snap;
                snap.deinit();
                if (new_dflash) |*d| d.deinit();
                if (new_mtp) |*m3| m3.deinit();
                self.evictAt(idx, "checkpoint merge failed");
                return err;
            };
        };

        // Free everything the old entry owned EXCEPT the (now-detached)
        // ssm_checkpoints, which were moved above.
        self.allocator.free(e.tokens);
        e.snapshot.deinit();
        // The old speculative payloads describe a strict PREFIX of the
        // new tokens, but they are keyed to their own base_pos and
        // length; the new ones supersede them outright. A commit with no
        // payload drops the old rather than keeping a shorter stale one.
        if (e.dflash) |*d| d.deinit();
        e.dflash = null;
        if (e.mtp) |*m4| m4.deinit();
        e.mtp = null;
        // The old restore points sit inside the new tokens too; at a tie the newer wins.
        const old_rings = e.ring_cps;
        e.ring_cps = null;
        const merged_rings = mergeRingCps(self.allocator, old_rings, new_rings);
        new_rings = null;
        e.ring_cps = merged_rings catch |err| blk: {
            log.warn("  [hot-cache] ring checkpoint merge failed: {s}\n", .{@errorName(err)});
            break :blk null;
        };
        const ring_bytes = ringCpsBytes(e.ring_cps);
        self.current_kv_bytes -|= e.kv_bytes;

        if (hot_keeps_newest) self.keepNewestCheckpoints(&merged_cps, record_text);
        // Recompute ssm bytes from the merged list.
        var merged_ssm_bytes: u64 = 0;
        if (merged_cps) |cps| {
            for (cps) |*cp| merged_ssm_bytes += ssmCheckpointBytes(cp);
        }
        e.tokens = tokens_owned;
        e.snapshot = new_snap;
        e.has_tools = has_tools;
        e.vision_key = eff_vision_key;
        e.cache_key = cache_key;
        e.media_start = eff_media_start;
        e.quant_config = quant_config;
        e.end_note = end_note;
        e.kv_bytes = new_kv_bytes + merged_ssm_bytes + new_dflash_bytes + new_mtp_bytes + ring_bytes + new_rows_bytes;
        if (e.rows) |*rows| rows.deinit();
        e.rows = new_rows;
        e.rows_bytes = new_rows_bytes;
        new_rows = null;
        e.ssm_checkpoints = merged_cps;
        e.ssm_bytes = merged_ssm_bytes;
        e.dflash = new_dflash;
        e.dflash_bytes = new_dflash_bytes;
        e.mtp = new_mtp;
        e.mtp_bytes = new_mtp_bytes;
        // Restore by move: the replacement the checkout promised; the entry is whole again.
        endCheckout(e);
        e.last_used = self.bumpCounter();
        self.current_kv_bytes += e.kv_bytes;
        // Inherited SSM checkpoints can make a replacement larger than
        // `new_bytes`, so enforce the cap again on the final entry — but
        // never by evicting the entry we just paid to update (#330
        // adjacent: near the budget that thrashed commit → evict all →
        // cold prefill, every turn). Evict OTHER entries first, then shed
        // this entry's checkpoints; eviction of the sole entry is the
        // last resort that keeps the load-time headroom clamp real.
        if (self.max_kv_bytes > 0) {
            while (self.current_kv_bytes > self.max_kv_bytes and
                self.entries.items.len > 1)
            {
                if (!self.reclaimFromLru("byte budget", null, self.current_kv_bytes - self.max_kv_bytes)) break;
            }
            self.shedCheckpointsToFit();
            while (self.current_kv_bytes > self.max_kv_bytes and
                self.entries.items.len > 0)
            {
                if (!self.evictOneLruProgress("byte budget", null)) break;
            }
        }
        if (self.disk != null) self.disk_dirty = true;
        self.logResident();
        return .{ .ok = eff_tokens.len };
    }

    if (hot_keeps_newest) {
        self.keepNewestCheckpoints(&eff_cps, record_text);
        var kept_bytes: u64 = 0;
        if (eff_cps) |cps| for (cps) |*cp| {
            kept_bytes += ssmCheckpointBytes(cp);
        };
        new_bytes -|= new_ssm_bytes -| kept_bytes;
        new_ssm_bytes = kept_bytes;
    }
    const incoming: Entry = .{
        .tokens = tokens_owned,
        .has_tools = has_tools,
        .vision_key = eff_vision_key,
        .cache_key = cache_key,
        .media_start = eff_media_start,
        .snapshot = new_snap,
        .id = self.bumpCounter(),
        .last_used = self.bumpCounter(),
        .quant_config = quant_config,
        .end_note = end_note,
        .kv_bytes = new_bytes,
        .ssm_checkpoints = eff_cps,
        .ssm_bytes = new_ssm_bytes,
        .dflash = new_dflash,
        .dflash_bytes = new_dflash_bytes,
        .mtp = new_mtp,
        .mtp_bytes = new_mtp_bytes,
        .ring_cps = new_rings,
        .rows = new_rows,
        .rows_bytes = new_rows_bytes,
    };
    new_rings = null;
    new_rows = null;
    if (!try self.retainNewEntry(incoming)) return .declined;
    // The trim prices a prefix against the checkpoints that survive a shed, so the shed runs here too.
    if (self.max_kv_bytes > 0) self.shedCheckpointsToFit();
    if (self.disk != null) self.disk_dirty = true;
    self.logResident();
    return .{ .ok = eff_tokens.len };
}



/// Offer a budget-declined candidate to the SSD tier before discarding
/// it. The RAM budget declines RETENTION, not value: under a small
/// --prefix-cache-mem a long agent session's cancelled-prefill prefixes
/// were thrown away on every retry while a much larger disk tier sat
/// idle (live 2026-09-07). Text-prefix candidates only — the disk never
/// holds pixel-keyed rows. Best-effort: a failed write costs nothing the
/// RAM decline hadn't already lost, and a byte-capped write continues
/// from the next commit of the same conversation (chunks are
/// content-deduped by token range).
pub fn spillDeclinedToDisk(
    self: *HotPrefixCache,
    snap: *const KVCacheSnapshot,
    all_tokens: []const u32,
    text_len: usize,
    has_tools: bool,
    media: MediaStamp,
    all_cps: ?[]SSMCheckpoint,
    ring_cps: ?[]const KVCacheSnapshot,
    rows: ?*const prefix_arch.Rows,
) void {
    const d = if (self.disk) |*dd| dd else return;
    if (text_len == 0) return;
    const tokens = all_tokens[0..text_len];
    const text = self.textCheckpoints(all_cps, text_len, all_tokens.len) orelse return;
    defer text.deinit(self.allocator);
    const cps = text.list;
    // The live state was captured as `pending_disk` before the trim; the flush after the
    // response lands it whole.
    if (self.capturesLiveForDisk()) {
        if (self.pending_disk != null) self.disk_dirty = true;
        return;
    }
    if (tokens.len < kv_disk_cache.MIN_PERSIST_TOKENS) return;
    // A decline-spill runs after the client is gone: the per-flush cap
    // exists to bound the stall a LIVE next request pays, and capping
    // here stranded most of each retry's work (live 2026-09-07: 512 MB
    // ≈ 13k tokens banked per retry while ~35k were recomputed every
    // time). Raise the budget for the spill's duration — the tier's
    // byte budget + LRU eviction is the real bound.
    const saved_cap = d.max_flush_bytes;
    d.max_flush_bytes = @max(saved_cap, kv_disk_cache.DECLINE_SPILL_FLUSH_FLOOR);
    defer d.max_flush_bytes = saved_cap;
    const outcome = (if (rows) |r| r.persist(d, tokens, has_tools, media, cps, null, false, self.arraysStream()) else d.appendCommitWith(.{
        .kv_entries = snap.entries,
        .step = snap.step,
        .config = snap.config,
        .tokens = tokens,
        .has_tools = has_tools,
        .ssm_checkpoints = cps,
        .ring = ringCommitOf(snap, ring_cps),
        .vision_key = media.key,
        .media_start = media.start,
    }, self.arraysStream(), .{})) catch |err| {
        log.warn("  [disk-cache] declined-candidate spill failed: {s}\n", .{@errorName(err)});
        return;
    };
    const note: []const u8 = switch (outcome) {
        .persisted => "complete",
        .partial => "byte-capped, continues on later commits",
        .skipped => "already persisted — nothing new to write",
    };
    log.info("  [disk-cache] declined RAM candidate spilled to SSD ({d} tokens, {s})\n", .{ tokens.len, note });
}

/// Snapshot the live cache (refcount-shared but for a ring's retained rows) plus the full token
/// record, this turn's checkpoints and those a RAM donor below the shared prefix holds, as the
/// RAM entry would inherit them. Best effort; the caller still owns `ssm_cps`/`dflash`/`mtp`.
pub fn capturePendingDisk(
    self: *HotPrefixCache,
    source_cache: *const KVCache,
    all_tokens: []const u32,
    text_len: usize,
    has_tools: bool,
    vision_key: u64,
    media_start: ?usize,
    media_chain: []const MediaSpan,
    restored: ?kv_disk_cache.RestoredRows,
    prompt_len: usize,
    ssm_cps: ?[]SSMCheckpoint,
    all_dflash: ?DflashCommit,
    all_mtp: ?DflashCommit,
    ring_cps: ?[]const KVCacheSnapshot,
    rows: ?*const prefix_arch.Rows,
) void {
    const tokens = all_tokens[0..text_len];
    // The spec snapshots cover rows past a cut record.
    const cut = text_len < all_tokens.len;
    const dflash = if (cut) null else all_dflash;
    const mtp = if (cut) null else all_mtp;
    // An earlier commit not flushed yet (two cancels culled in one pass) lands first.
    if (self.pending_disk != null) {
        if (self.disk) |*d| self.flushPendingRecord(d, self.arraysStream());
    }
    var snap = source_cache.snapshotRetained(self.arraysStream()) catch |err| {
        log.warn("  [disk-cache] live snapshot failed: {s} — flushing the RAM entry instead\n", .{@errorName(err)});
        return;
    };
    var rec: PendingDiskFlush = .{
        .snapshot = snap,
        .tokens = self.allocator.dupe(u32, tokens) catch {
            snap.deinit();
            return;
        },
        .has_tools = has_tools,
        .media = MediaStamp.of(vision_key, media_start),
        .media_chain = if (media_chain.len > 0) self.allocator.dupe(MediaSpan, media_chain) catch &.{} else &.{},
        .restored = restored,
    };
    if (ring_cps) |cps| rec.ring_cps = shareRingCpsUpTo(self.allocator, cps, std.math.maxInt(usize), null) catch null;
    if (rows) |r| rec.rows = r.share() catch {
        rec.deinit(self.allocator);
        return;
    };
    if (ssm_cps) |cps| {
        rec.ssm_cps = cloneCheckpointsWithBank(self.allocator, cps, text_len, null, self.arraysStream()) catch null;
    }
    const quant = source_cache.config;
    const cap = @min(if (prompt_len == 0) text_len else prompt_len, text_len);
    if (self.bestCheckpointDonor(tokens, has_tools, vision_key, media_start, media_chain, quant)) |donor| inherit: {
        const limit = @min(donor.shared, cap);
        const inherited = (cloneCheckpointsUpTo(self.allocator, self.entries.items[donor.idx].ssm_checkpoints.?, limit, null) catch break :inherit) orelse break :inherit;
        rec.ssm_cps = mergeDiskCheckpoints(self.allocator, inherited, rec.ssm_cps) catch null;
    }
    if (self.bestRingDonor(tokens, has_tools, vision_key, media_start, media_chain, quant, cap)) |donor| inherit: {
        const inherited = (shareRingCpsUpTo(self.allocator, self.entries.items[donor.idx].ring_cps.?, donor.limit, null) catch break :inherit) orelse break :inherit;
        rec.ring_cps = mergeRingCps(self.allocator, inherited, rec.ring_cps) catch null;
    }
    if (dflash) |d| {
        if (d.cache.snapshot()) |ds| {
            rec.dflash = .{ .snapshot = ds, .base_pos = d.base_pos };
        } else |_| {}
    }
    if (mtp) |m2| {
        if (m2.cache.snapshot()) |ms| {
            rec.mtp = DflashSnap.ofCommit(m2, ms, self.arraysStream());
        } else |_| {}
    }
    self.pending_disk = rec;
}

/// A ringed snapshot's restore points for the disk tier; null for a cache that does not ring.
pub fn ringCommitOf(snap: *const KVCacheSnapshot, cps: ?[]const KVCacheSnapshot) ?kv_disk_cache.RingCommit {
    if (snap.swa_ring_window == 0) return null;
    return .{ .window = snap.swa_ring_window, .cps = cps orelse &.{} };
}

pub const EntrySpecs = struct {
    dflash: ?kv_disk_cache.SpecCommit = null,
    mtp: ?kv_disk_cache.SpecCommit = null,
};

/// The disk-tier spec payloads for one RAM entry; the flush and the idle spill must not drift.
pub fn entrySpecCommits(e: *Entry) EntrySpecs {
    return .{
        .dflash = if (e.dflash) |*df| .{
            .entries = df.snapshot.entries,
            .step = df.snapshot.step,
            .config = df.snapshot.config,
            .base_pos = df.base_pos,
        } else null,
        .mtp = if (e.mtp) |*mm| .{
            .entries = mm.snapshot.entries,
            .step = mm.snapshot.step,
            .config = mm.snapshot.config,
            .base_pos = mm.base_pos,
            .head_aux = if (mm.head_aux) |*a| a else null,
            .head_pos_base = mm.head_pos_base,
            .head_marks = mm.head_marks.slice(),
        } else null,
    };
}

/// At the end of a request every idle entry is written to the SSD tier, and RAM is trimmed
/// back to the active session plus the idle allowance. Writing is unconditional; the
/// allowance is a hard cap shed in two tiers: entries with a proven durable copy first,
/// then (naming the reason) the rest, oldest first.
pub fn spillIdleEntries(self: *HotPrefixCache, caller_stream: mlx.mlx_stream) void {
    const s = self.arrays_on orelse caller_stream;
    if (!self.ssd_first) return;
    if (self.entries.items.len <= 1) return;
    const d = if (self.disk) |*dd| dd else return;

    // The active session is the single MRU entry; `last_used` is a strictly increasing counter.
    var newest_used: u64 = 0;
    for (self.entries.items) |*e| newest_used = @max(newest_used, e.last_used);

    // Pass 1: write, unconditionally. Attribute anything the writer lost since the last pass
    // before sampling the error counter: a failure between two passes is invisible to the delta.
    _ = d.harvestWriteFailures();
    const errs_before = d.writeErrors();
    var spilled: usize = 0;
    var idle_bytes: u64 = 0;
    for (self.entries.items) |*e| {
        e.spill_durable = false;
        if (e.last_used == newest_used) continue; // the active session
        // Checked out: the bytes are the slot's, the snapshot is empty. Skipped before the allowance count.
        if (e.checkout != null) continue;
        // Counted against the allowance before the vision skip: it occupies RAM either way.
        idle_bytes +|= e.kv_bytes;
        // Vision entries never spill: a token-only disk key is ambiguous.
        if (e.vision_key != 0) continue;
        const specs = entrySpecCommits(e);
        const outcome = d.appendCommitWithSpec(
            e.snapshot.entries,
            e.snapshot.step,
            e.snapshot.config,
            e.tokens,
            e.has_tools,
            e.ssm_checkpoints,
            specs.dflash,
            specs.mtp,
            s,
        ) catch |err| {
            log.warn("  [hot-cache] idle spill failed: {s} — entry stays resident\n", .{@errorName(err)});
            continue;
        };
        // Only `.persisted`: a silent skip or a `.partial` copy is not a copy.
        if (outcome != .persisted) continue;
        // `.persisted` is the write path's claim; the index has to agree.
        const disk_id = d.fullPrefixEntryId(e.snapshot.entries, e.snapshot.step, e.tokens, e.has_tools, e.snapshot.config) orelse {
            log.warn("  [hot-cache] idle spill: the tier does not hold the full prefix — entry stays resident\n", .{});
            continue;
        };
        // A staged copy is not durable yet, and the check must not be a drain (this runs on
        // the inference thread at every finish); the next pass asks again.
        if (d.entryWritesPending(disk_id)) continue;
        if (d.writeErrors() != errs_before) {
            log.warn("  [hot-cache] idle spill: background write failed — entry stays resident\n", .{});
            continue;
        }
        // Ask the filesystem too: one stat per chunk, against the eviction of a whole session.
        if (!d.entryWholeOnDisk(disk_id)) {
            log.warn("  [hot-cache] idle spill: the persisted chunks do not match the index — entry stays resident\n", .{});
            continue;
        }
        e.spill_durable = true;
        spilled += 1;
    }

    // Pass 2: evict the durable, oldest first, down to the allowance.
    var evicted: usize = 0;
    while (idle_bytes > self.ssd_idle_mem) {
        const idx = self.oldestIdleIndex(newest_used, true) orelse break;
        idle_bytes -|= self.entries.items[idx].kv_bytes;
        self.evictAt(idx, "SSD-first idle spill");
        evicted += 1;
    }

    // Pass 3: still over. Entries with no durable copy go too, oldest first, naming why.
    while (idle_bytes > self.ssd_idle_mem) {
        const idx = self.oldestIdleIndex(newest_used, false) orelse break;
        const e = &self.entries.items[idx];
        log.info("  [hot-cache] idle allowance exceeded: dropped unpersistable entry ({s}) {d} tokens, {d:.1} MB\n", .{
            unpersistableReason(d, e),
            e.tokens.len,
            @as(f64, @floatFromInt(e.kv_bytes)) / (1024.0 * 1024.0),
        });
        idle_bytes -|= e.kv_bytes;
        self.evictForSpace(idx, "SSD-first idle allowance");
        evicted += 1;
    }

    if (spilled > 0 or evicted > 0) {
        log.info("  [hot-cache] SSD-first: wrote {d} idle entries to disk, evicted {d}; RAM holds the active session + {d} MB idle allowance\n", .{
            spilled, evicted, self.ssd_idle_mem >> 20,
        });
    }
}

/// Least-recently-used idle entry (never the active session, never a vision entry).
pub fn oldestIdleIndex(self: *HotPrefixCache, newest_used: u64, durable_only: bool) ?usize {
    var best: ?usize = null;
    var best_used: u64 = std.math.maxInt(u64);
    for (self.entries.items, 0..) |*e, i| {
        if (e.last_used == newest_used) continue;
        if (e.vision_key != 0) continue;
        if (e.checkout != null) continue;
        if (durable_only and !e.spill_durable) continue;
        if (e.last_used < best_used) {
            best_used = e.last_used;
            best = i;
        }
    }
    return best;
}

/// Why pass 1 could not leave a durable copy of `e` (for the log).
pub fn unpersistableReason(d: *kv_disk_cache.DiskTier, e: *const Entry) []const u8 {
    if (d.store_declined) return "store declined: volume is short";
    if (e.tokens.len < @as(usize, kv_disk_cache.MIN_PERSIST_TOKENS)) return "under the persist floor";
    const target = kv_disk_cache.persistTargetLen(e.snapshot.entries, e.snapshot.step, e.tokens.len);
    for (e.snapshot.entries) |*le| {
        if (le.initialized and le.offset < target) return "layer offset short of the range";
    }
    return "partial copy";
}

/// The hybrid prefill write-through: the rows and checkpoints so far go to the SSD tier as the prefill produces
/// them, and every checkpoint but the newest leaves RAM once its file is written. Best effort: what a failed
/// write could not land is released too, since the admission bill counts on the bound.
pub fn writeThroughHybrid(
    self: *HotPrefixCache,
    kv_entries: []const trunk_mod.KVCacheEntry,
    config: kv_quant.KVQuantConfig,
    prompt: []const u32,
    has_tools: bool,
    media: MediaStamp,
    restored: ?kv_disk_cache.RestoredRows,
    cps: []SSMCheckpoint,
    live_ssm: []const SSMCacheEntry,
    flush_bound: u64,
    s: mlx.mlx_stream,
) void {
    const media_chain = self.takeMediaChain();
    const d = if (self.disk) |*dd| dd else return;
    // A hybrid record restores only at a stored SSM point: before the prefill holds one (or extends a record that
    // does), a chunk waits for the next record, which writes it too, instead of leaving an entry nothing restores.
    if (live_ssm.len > 0 and cps.len == 0 and restored == null) return;
    // A prefill point carries no QSA bank: the newest one borrows the live history for this write, so the SSD
    // history reaches every point the chunk lands.
    const lent = cps.len > 0 and live_ssm.len > 0 and !checkpointHasQsaPooled(&cps[cps.len - 1]);
    if (lent) prefix_arch.aux.handoffQsaHistoryToLatest(cps, live_ssm, s) catch |err| {
        log.warn("  [disk-cache] prefill write-through: QSA history not lent: {s}\n", .{@errorName(err)});
    };
    defer if (lent) prefix_arch.aux.dropQsaHistory(&cps[cps.len - 1]);
    // Below the tier's floor nothing is written yet, so no state may be freed.
    const floor = prompt.len >= kv_disk_cache.MIN_PERSIST_TOKENS;
    _ = d.appendCommitWith(.{
        .kv_entries = kv_entries,
        .step = prompt.len,
        .config = config,
        .tokens = prompt,
        .has_tools = has_tools,
        .ssm_checkpoints = if (cps.len > 0) cps else null,
        .vision_key = media.key,
        .media_start = media.start,
        .media_chain = media_chain,
        .restored = restored,
    }, s, .{ .flush_bound = flush_bound }) catch |err| {
        log.warn("  [disk-cache] prefill write-through failed: {s}\n", .{@errorName(err)});
    };
    if (floor) d.settleCheckpoints(d.last_store_id, cps);
}

/// Write the pending record whole and release it: its rows may be a finished request's own
/// buffers, which nothing could read again to resume a partial entry.
pub fn flushPendingRecord(self: *HotPrefixCache, d: *kv_disk_cache.DiskTier, s: mlx.mlx_stream) void {
    var pending = self.pending_disk orelse return;
    self.pending_disk = null;
    defer pending.deinit(self.allocator);
    const p_dflash: ?kv_disk_cache.SpecCommit = if (pending.dflash) |*df| .{
        .entries = df.snapshot.entries,
        .step = df.snapshot.step,
        .config = df.snapshot.config,
        .base_pos = df.base_pos,
    } else null;
    const p_mtp: ?kv_disk_cache.SpecCommit = if (pending.mtp) |*mm| .{
        .entries = mm.snapshot.entries,
        .step = mm.snapshot.step,
        .config = mm.snapshot.config,
        .base_pos = mm.base_pos,
        .head_aux = if (mm.head_aux) |*a| a else null,
        .head_pos_base = mm.head_pos_base,
        .head_marks = mm.head_marks.slice(),
    } else null;
    _ = (if (pending.rows) |*rows| rows.persist(d, pending.tokens, pending.has_tools, pending.media, pending.ssm_cps, p_dflash, true, s) else d.appendCommitWith(.{
        .kv_entries = pending.snapshot.entries,
        .step = pending.snapshot.step,
        .config = pending.snapshot.config,
        .tokens = pending.tokens,
        .has_tools = pending.has_tools,
        .ssm_checkpoints = pending.ssm_cps,
        .dflash_snap = p_dflash,
        .mtp_snap = p_mtp,
        .ring = ringCommitOf(&pending.snapshot, pending.ring_cps),
        .vision_key = pending.media.key,
        .media_start = pending.media.start,
        .media_chain = pending.media_chain,
        .restored = pending.restored,
    }, s, .{ .whole = true })) catch |err| {
        log.warn("  [disk-cache] persist failed: {s}\n", .{@errorName(err)});
    };
}

/// With both tiers on, a flushed entry keeps its newest `RING_CHECKPOINT_RAM_KEEP` ring
/// checkpoints in RAM; each older one the SSD tier holds a file for is released, and a lookup
/// that needs it restores from that file. A staged file counts: its bytes are the writer's, billed
/// by its permit. One with no file in the index (a cut or failed staging) stays.
pub fn releaseRingCpsToDisk(self: *HotPrefixCache) void {
    if (!self.ram_enabled) return;
    const d = if (self.disk) |*dd| dd else return;
    for (self.entries.items) |*e| {
        if (e.checkout != null) continue;
        const cps = e.ring_cps orelse continue;
        if (cps.len <= RING_CHECKPOINT_RAM_KEEP) continue;
        const older = cps.len - RING_CHECKPOINT_RAM_KEEP;
        var steps: [RING_CHECKPOINT_MAX]usize = undefined;
        var held: [RING_CHECKPOINT_MAX]bool = undefined;
        for (cps[0..older], steps[0..older]) |cp, *st| st.* = cp.step;
        // A disk entry holds this RAM entry's point only when their media identities agree:
        // the entry carries no chain, so an unequal key credits nothing past either side's
        // first media row.
        const rule: MediaRule = .{ .vision_key = e.vision_key, .media_start = e.media_start, .chain = &.{} };
        d.ringPointsHeld(e.tokens, e.has_tools, e.quant_config, steps[0..older], held[0..older], rule.limits());
        var released: usize = 0;
        for (held[0..older]) |h| released += @intFromBool(h);
        if (released == 0) continue;
        const kept = self.allocator.alloc(KVCacheSnapshot, cps.len - released) catch continue;
        var n: usize = 0;
        var freed: u64 = 0;
        for (cps, 0..) |*cp, i| {
            if (i < older and held[i]) {
                freed += snapshotBytes(cp);
                cp.deinit();
            } else {
                kept[n] = cp.*;
                n += 1;
            }
        }
        self.allocator.free(cps);
        e.ring_cps = kept;
        e.kv_bytes -|= freed;
        self.current_kv_bytes -|= freed;
        log.info("  [hot-cache] released {d} ring checkpoints to the SSD tier ({d} kept in RAM)\n", .{ released, kept.len });
    }
}

pub fn flushPendingDisk(self: *HotPrefixCache, caller_stream: mlx.mlx_stream) void {
    const s = self.arrays_on orelse caller_stream;
    if (!self.disk_dirty) return;
    self.disk_dirty = false;
    const d = if (self.disk) |*dd| dd else return;
    defer self.releaseRingCpsToDisk();
    // Flush the live state captured at commit, not what the RAM entry retained after its trim.
    if (self.pending_disk != null) return self.flushPendingRecord(d, s);
    if (self.entries.items.len == 0) return;
    const newest = &self.entries.items[prefix_policy.newestIndex(Entry, self.entries.items, .first).?];
    const text_len = diskTextLen(newest.tokens.len, newest.vision_key, newest.media_start);
    const media = MediaStamp.of(newest.vision_key, newest.media_start);
    if (text_len == 0) return;
    const tokens = newest.tokens[0..text_len];
    const text = self.textCheckpoints(newest.ssm_checkpoints, text_len, newest.tokens.len) orelse return;
    defer text.deinit(self.allocator);
    // Phase 3: hybrid entries persist their SSM checkpoints alongside the
    // KV chunks (immutable per-position s*.safetensors). The snapshot
    // arrays are refcount-shared with the RAM entry, so `appendCommit`
    // reads the same buffers the commit captured.
    // v4: the spec snapshots (dflash context / MTP history) ride along —
    // eligibility was enforced at commitWithState, so the disk tier
    // persists exactly what the RAM entry holds.
    const specs: EntrySpecs = if (text.cut) .{} else entrySpecCommits(newest);
    const dflash_spec = specs.dflash;
    const mtp_spec = specs.mtp;
    if (newest.rows) |*rows| {
        const done = rows.persist(d, tokens, newest.has_tools, media, text.list, dflash_spec, false, s) catch |err| {
            log.warn("  [disk-cache] persist failed: {s}\n", .{@errorName(err)});
            return;
        };
        if (!done.nothingPending()) self.disk_dirty = true;
        return;
    }
    const complete = d.appendCommitWith(.{
        .kv_entries = newest.snapshot.entries,
        .step = newest.snapshot.step,
        .config = newest.snapshot.config,
        .tokens = tokens,
        .has_tools = newest.has_tools,
        .ssm_checkpoints = text.list,
        .dflash_snap = dflash_spec,
        .mtp_snap = mtp_spec,
        .ring = ringCommitOf(&newest.snapshot, newest.ring_cps),
        .vision_key = media.key,
        .media_start = media.start,
    }, s, .{}) catch |err| {
        log.warn("  [disk-cache] persist failed: {s}\n", .{@errorName(err)});
        return;
    };
    // Byte-capped flush: a large entry persists incrementally — keep the
    // dirty flag set so the next finished request continues the write.
    if (!complete.nothingPending()) self.disk_dirty = true;
}

/// #330 adjacent: when the byte budget is exceeded with the just-updated
/// entry as the sole survivor, drop its checkpoints (the inherited bytes
/// the pre-check could not price) instead of evicting it. Reuses the
/// replace path's interior-thinning rule (keep the first and the newest).
/// The pre-check guarantees the entry's own KV + this turn's checkpoints
/// fit, so shedding converges under budget before the list empties in
/// practice; if it doesn't, the caller's eviction fallback decides.
/// Merge two OWNED checkpoint lists into one ascending, pos-deduped list
/// (on a tie the `new` state wins — it is the more recently observed one
/// at that position), re-apply `ssm_checkpoint_max`, and collapse to ONE
/// QSA history. Takes ownership of BOTH slices on every path; the caller
/// must have detached them from whatever owned them.
///
/// The cap is re-applied here because a merged list spans more than one
/// prefill, so `generate.zig`'s per-prefill cap no longer bounds it — and
/// NOT oldest-first. Within one prefill oldest-first is fine; across turns
/// it collapses the survivors onto the end of the prompt, and then a
/// request that diverges early finds no checkpoint at or below its match
/// and pays a FULL cold prefill:
///     [hot-cache] hybrid miss (no checkpoint <= 16382 of 178509)
/// That one cost 415 s. Oldest-first is also the expensive choice: a
/// checkpoint costs roughly a constant plus a term linear in its position,
/// so it discards the cheap early ones and keeps the large late ones.
///
/// Thin the interior instead, always keeping the first and the newest:
/// drop whichever checkpoint sits between the closest pair of neighbours,
/// i.e. the one whose removal widens the coverage gap least. Same count,
/// spread over the whole prompt, LESS memory. `n` is at most
/// `ssm_checkpoint_max`, so the quadratic scan is trivial. The selection is
/// `trunk_mod.ssmCheckpointDropIndex`, shared with every other thinning site.
pub fn mergeCheckpointLists(
    self: *HotPrefixCache,
    old: []SSMCheckpoint,
    new: []SSMCheckpoint,
) !?[]SSMCheckpoint {
    var merged = std.ArrayList(SSMCheckpoint).empty;
    var i: usize = 0;
    var j: usize = 0;
    var sources_freed = false;
    // Ownership of BOTH slices is ours from the first line, so the error
    // path owes the un-moved tails too — items still sitting in old[i..]
    // / new[j..] plus the two backing slices.
    errdefer {
        for (merged.items) |*c| c.deinit(self.allocator);
        merged.deinit(self.allocator);
        if (!sources_freed) {
            for (old[@min(i, old.len)..]) |*c| c.deinit(self.allocator);
            for (new[@min(j, new.len)..]) |*c| c.deinit(self.allocator);
            self.allocator.free(old);
            self.allocator.free(new);
        }
    }
    while (i < old.len or j < new.len) {
        if (i >= old.len) {
            try merged.append(self.allocator, new[j]);
            j += 1;
        } else if (j >= new.len) {
            try merged.append(self.allocator, old[i]);
            i += 1;
        } else if (old[i].pos < new[j].pos) {
            try merged.append(self.allocator, old[i]);
            i += 1;
        } else if (old[i].pos > new[j].pos) {
            try merged.append(self.allocator, new[j]);
            j += 1;
        } else {
            var dropped = old[i];
            dropped.deinit(self.allocator);
            i += 1;
            try merged.append(self.allocator, new[j]);
            j += 1;
        }
    }
    self.allocator.free(old);
    self.allocator.free(new);
    sources_freed = true;
    while (self.ssm_checkpoint_max > 0 and
        merged.items.len > self.ssm_checkpoint_max)
    {
        // Under three there is no interior to thin; honour the cap by
        // dropping the oldest, which is also the cheapest to redo.
        const drop = trunk_mod.ssmCheckpointDropIndex(merged.items, self.cp_thin, null);
        var dropped = merged.orderedRemove(drop);
        dropped.deinit(self.allocator);
    }
    const owned = try merged.toOwnedSlice(self.allocator);
    // The inherited latest and this turn's latest both carry the indexer
    // history: keep one.
    keepOnlyLatestQsaHistory(owned);
    return self.takeCpsIfQsaBank(owned);
}

/// The pending flush's checkpoints: `inherited` and `own` in one ascending list, `own`'s kept at
/// a tie. Unthinned: the SSD tier thins its own set. Consumes both on every path.
pub fn mergeDiskCheckpoints(allocator: std.mem.Allocator, inherited: []SSMCheckpoint, own_opt: ?[]SSMCheckpoint) ![]SSMCheckpoint {
    const own = own_opt orelse return inherited;
    defer {
        allocator.free(inherited);
        allocator.free(own);
    }
    var ties: usize = 0;
    for (inherited) |a| {
        for (own) |b| ties += @intFromBool(a.pos == b.pos);
    }
    const out = allocator.alloc(SSMCheckpoint, inherited.len + own.len - ties) catch |err| {
        for (inherited) |*c| c.deinit(allocator);
        for (own) |*c| c.deinit(allocator);
        return err;
    };
    var i: usize = 0;
    var j: usize = 0;
    for (out) |*o| {
        if (j == own.len or (i < inherited.len and inherited[i].pos < own[j].pos)) {
            o.* = inherited[i];
            i += 1;
            continue;
        }
        if (i < inherited.len and inherited[i].pos == own[j].pos) {
            inherited[i].deinit(allocator);
            i += 1;
        }
        o.* = own[j];
        j += 1;
    }
    return out;
}

pub fn takeCpsIfQsaBank(self: *HotPrefixCache, cps: []SSMCheckpoint) ?[]SSMCheckpoint {
    if (!self.qsa_history_required) return cps;
    if (checkpointListHasQsaPooled(cps)) return cps;
    for (cps) |*cp| cp.deinit(self.allocator);
    self.allocator.free(cps);
    log.info("  [hot-cache] dropped checkpoints: no QSA indexer bank\n", .{});
    return null;
}

/// The resident entry whose checkpoints a commit of `tokens` may inherit:
/// the key-compatible entry maximizing the highest checkpoint at or below
/// its shared prefix with `tokens`. Returns that entry's index and the
/// shared length (the inheritance limit — a checkpoint past it describes
/// state this prompt never reached).
///
/// Checkpoint inheritance was a property of the REPLACE path — an entry
/// whose tokens are a strict PREFIX of the new ones. It is really a
/// property of the TOKENS. A prompt sent twice (an MTP arm then a serial
/// arm; two clients; any retry) commits two entries that agree on the
/// whole prompt and diverge in their GENERATED tails, so neither replaces
/// the other — and the second one, having restored ~everything and
/// prefilled a ~31-token tail, earns no checkpoint of its own (a restored
/// tail inside the window forwards as one span, `ssmSnapshotBackoff`). Evict the first and the
/// prompt is uncacheable: 393k tokens, 560 s of cold prefill, every rung.
pub fn bestCheckpointDonor(
    self: *const HotPrefixCache,
    tokens: []const u32,
    has_tools: bool,
    vision_key: u64,
    media_start: ?usize,
    media_chain: []const MediaSpan,
    quant_config: kv_quant.KVQuantConfig,
) ?RamMatch {
    var best_idx: ?usize = null;
    var best_shared: usize = 0;
    var best_pos: usize = 0;
    for (self.entries.items, 0..) |*e, i| {
        const cps = e.ssm_checkpoints orelse continue;
        const shared = donorShared(e, tokens, has_tools, vision_key, media_start, media_chain, quant_config) orelse continue;
        const cp = highestCheckpointAtOrBelow(cps, shared) orelse continue;
        if (cp.pos > best_pos) {
            best_pos = cp.pos;
            best_shared = shared;
            best_idx = i;
        }
    }
    if (best_idx) |idx| return .{ .idx = idx, .shared = best_shared };
    return null;
}

/// Tokens a key-compatible entry provably shares with `tokens`; null = another key.
pub fn donorShared(
    e: *const Entry,
    tokens: []const u32,
    has_tools: bool,
    vision_key: u64,
    media_start: ?usize,
    media_chain: []const MediaSpan,
    quant_config: kv_quant.KVQuantConfig,
) ?usize {
    if (e.has_tools != has_tools) return null;
    if (!std.meta.eql(e.quant_config, quant_config)) return null;
    var max_shared = @min(e.tokens.len, tokens.len);
    if (e.vision_key != vision_key) {
        // Same rule as `findBestRestorableMatch`: rows before the first
        // media block the two prompts do not share cross keys.
        const safe_boundary = crossKeyBoundary(e.vision_key, e.media_start, media_start, media_chain) orelse return null;
        max_shared = @min(max_shared, safe_boundary);
    }
    var shared: usize = 0;
    while (shared < max_shared and e.tokens[shared] == tokens[shared]) shared += 1;
    return shared;
}

pub const RingDonor = struct { idx: usize, limit: usize };

/// The resident entry holding the highest ring checkpoint a commit of `tokens` may inherit:
/// one whose rows are all shared tokens, at or below `cap` (the committing prompt's length).
/// Returns that entry's index and the bound.
pub fn bestRingDonor(
    self: *const HotPrefixCache,
    tokens: []const u32,
    has_tools: bool,
    vision_key: u64,
    media_start: ?usize,
    media_chain: []const MediaSpan,
    quant_config: kv_quant.KVQuantConfig,
    cap: usize,
) ?RingDonor {
    var best: ?RingDonor = null;
    var best_step: usize = 0;
    for (self.entries.items, 0..) |*e, i| {
        const cps = e.ring_cps orelse continue;
        const shared = donorShared(e, tokens, has_tools, vision_key, media_start, media_chain, quant_config) orelse continue;
        const limit = @min(shared, cap);
        for (cps) |*cp| {
            if (cp.step < RING_RESTORE_MIN_TOKENS or cp.step > limit or cp.step <= best_step) continue;
            best_step = cp.step;
            best = .{ .idx = i, .limit = limit };
        }
    }
    return best;
}

/// Refcount-share `src`'s ring checkpoints with `step <= limit` into an ascending slice the
/// caller owns, thinned to `RING_CHECKPOINT_MAX` and to `budget` (null = unbounded). Null
/// when none qualifies. Billed again per entry, as `cloneCheckpointsUpTo` explains.
pub fn shareRingCpsUpTo(allocator: std.mem.Allocator, src: []const KVCacheSnapshot, limit: usize, budget: ?u64) !?[]KVCacheSnapshot {
    var n: usize = 0;
    for (src) |*cp| n += @intFromBool(cp.step <= limit);
    if (n == 0) return null;
    const out = try allocator.alloc(KVCacheSnapshot, n);
    var total: u64 = 0;
    {
        var filled: usize = 0;
        errdefer {
            for (out[0..filled]) |*c| c.deinit();
            allocator.free(out);
        }
        for (src) |*cp| {
            if (cp.step > limit) continue;
            out[filled] = try cp.share();
            total += snapshotBytes(cp);
            filled += 1;
        }
    }
    return thinRingCps(allocator, out, RING_CHECKPOINT_MAX, if (budget) |b| total -| b else 0);
}

pub fn ringStepOf(cp: *const KVCacheSnapshot) usize {
    return cp.step;
}

/// Thin an owned ascending list to `max_len` checkpoints and by at least `need_free` bytes,
/// dropping where coverage loses least: the lowest (a shared preamble) and the newest stay
/// longest. Consumes `cps` on every path; null when none survives.
pub fn thinRingCps(allocator: std.mem.Allocator, cps: []KVCacheSnapshot, max_len: usize, need_free: u64) !?[]KVCacheSnapshot {
    errdefer freeRingCps(allocator, cps);
    var live = try std.ArrayList(KVCacheSnapshot).initCapacity(allocator, cps.len);
    defer live.deinit(allocator);
    live.appendSliceAssumeCapacity(cps);
    var freed: u64 = 0;
    while (live.items.len > 0 and (live.items.len > max_len or freed < need_free)) {
        const k = trunk_mod.spanPreservingDropIndex(KVCacheSnapshot, live.items, ringStepOf, .min_span_recency, null);
        freed += snapshotBytes(&live.items[k]);
        _ = live.orderedRemove(k);
    }
    if (live.items.len == cps.len) return cps;
    const kept: ?[]KVCacheSnapshot = if (live.items.len == 0) null else try allocator.dupe(KVCacheSnapshot, live.items);
    for (cps) |*c| {
        const survives = for (live.items) |l| {
            if (l.entries.ptr == c.entries.ptr) break true;
        } else false;
        if (!survives) c.deinit();
    }
    allocator.free(cps);
    return kept;
}

/// One ascending list from two owned ones (a tie keeps `new`'s), thinned to
/// `RING_CHECKPOINT_MAX` (`thinRingCps`). Consumes both on every path.
pub fn mergeRingCps(allocator: std.mem.Allocator, old: ?[]KVCacheSnapshot, new: ?[]KVCacheSnapshot) !?[]KVCacheSnapshot {
    var none = [_]KVCacheSnapshot{};
    const a: []KVCacheSnapshot = old orelse &none;
    const b: []KVCacheSnapshot = new orelse &none;
    defer {
        if (old) |o| allocator.free(o);
        if (new) |n| allocator.free(n);
    }
    var ties: usize = 0;
    for (a) |x| {
        for (b) |y| ties += @intFromBool(x.step == y.step);
    }
    const n = a.len + b.len - ties;
    if (n == 0) return null;
    const out = allocator.alloc(KVCacheSnapshot, n) catch |err| {
        for (a) |*c| c.deinit();
        for (b) |*c| c.deinit();
        return err;
    };
    var i: usize = 0;
    var j: usize = 0;
    for (out) |*o| {
        if (j == b.len or (i < a.len and a[i].step < b[j].step)) {
            o.* = a[i];
            i += 1;
            continue;
        }
        if (i < a.len and a[i].step == b[j].step) {
            a[i].deinit();
            i += 1;
        }
        o.* = b[j];
        j += 1;
    }
    return thinRingCps(allocator, out, RING_CHECKPOINT_MAX, 0);
}

/// The slot's own checkpoints as an ascending owned list (a tie keeps the later one, so the
/// prompt end beats the fork); best effort, like their capture.
pub fn ownRingCps(allocator: std.mem.Allocator, cps: SlotRingCps) ?[]KVCacheSnapshot {
    var all: [SLOT_RING_CHECKPOINTS]KVCacheSnapshot = undefined;
    var n: usize = 0;
    if (cps.fork) |f| {
        all[n] = f;
        n += 1;
    }
    for (cps.marks[0..cps.n_marks]) |m| {
        all[n] = m;
        n += 1;
    }
    if (cps.prompt_end) |e| {
        all[n] = e;
        n += 1;
    }
    if (n == 0) return null;
    std.sort.insertion(KVCacheSnapshot, all[0..n], {}, struct {
        fn lt(_: void, x: KVCacheSnapshot, y: KVCacheSnapshot) bool {
            return x.step < y.step;
        }
    }.lt);
    var kept: usize = 0;
    for (all[0..n]) |cp| {
        if (kept > 0 and all[kept - 1].step == cp.step) {
            all[kept - 1].deinit();
            all[kept - 1] = cp;
        } else {
            all[kept] = cp;
            kept += 1;
        }
    }
    return allocator.dupe(KVCacheSnapshot, all[0..kept]) catch {
        for (all[0..kept]) |*c| c.deinit();
        return null;
    };
}

/// The checkpoints at or below `limit`, as an owned list; consumes `cps`.
pub fn ringCpsUpTo(allocator: std.mem.Allocator, cps: []KVCacheSnapshot, limit: usize) ?[]KVCacheSnapshot {
    var kept: usize = 0;
    while (kept < cps.len and cps[kept].step <= limit) kept += 1;
    if (kept == cps.len) return cps;
    for (cps[kept..]) |*c| c.deinit();
    defer allocator.free(cps);
    if (kept == 0) return null;
    return allocator.dupe(KVCacheSnapshot, cps[0..kept]) catch {
        for (cps[0..kept]) |*c| c.deinit();
        return null;
    };
}

pub fn freeRingCps(allocator: std.mem.Allocator, cps: []KVCacheSnapshot) void {
    for (cps) |*c| c.deinit();
    allocator.free(cps);
}

pub fn ringCpsBytes(cps: ?[]const KVCacheSnapshot) u64 {
    var total: u64 = 0;
    for (cps orelse return 0) |*c| total += snapshotBytes(c);
    return total;
}

/// `cloneCheckpointsUpTo`, plus the pooled indexer bank: it lives on the newest checkpoint,
/// which a `limit` below it drops, so it is sliced onto the last one kept.
pub fn cloneCheckpointsWithBank(allocator: std.mem.Allocator, src: []const SSMCheckpoint, limit: usize, budget: ?u64, s: mlx.mlx_stream) !?[]SSMCheckpoint {
    const cloned = (try cloneCheckpointsUpTo(allocator, src, limit, budget)) orelse return null;
    var bank: ?*const SSMCheckpoint = null;
    for (src) |*cp| {
        if (checkpointHasQsaPooled(cp)) bank = cp;
    }
    const b = bank orelse return cloned;
    if (b.pos <= limit) return cloned;
    const last = &cloned[cloned.len - 1];
    sliceQsaHistoryOntoCheckpoint(last, b, last.pos, s) catch |err| {
        for (cloned) |*c| c.deinit(allocator);
        allocator.free(cloned);
        return err;
    };
    return cloned;
}

/// The checkpoints a record cut to `len` rows persists. `.list` is owned only when `cut`.
pub const TextCps = struct {
    list: ?[]SSMCheckpoint,
    cut: bool,

    fn deinit(self: TextCps, allocator: std.mem.Allocator) void {
        if (!self.cut) return;
        const l = self.list orelse return;
        for (l) |*cp| cp.deinit(allocator);
        allocator.free(l);
    }
};

pub fn textCheckpoints(self: *const HotPrefixCache, cps: ?[]SSMCheckpoint, len: usize, tokens_len: usize) ?TextCps {
    if (len >= tokens_len) return .{ .list = cps, .cut = false };
    const src = cps orelse return .{ .list = null, .cut = true };
    const cloned = cloneCheckpointsWithBank(self.allocator, src, len, null, self.arraysStream()) catch |err| {
        log.warn("  [disk-cache] text checkpoints failed: {s} — nothing persisted\n", .{@errorName(err)});
        return null;
    };
    return .{ .list = cloned, .cut = true };
}

/// Refcount-share `src`'s checkpoints with `pos <= limit` into a fresh
/// ASCENDING slice the caller owns. Newest-first while a budget remains
/// (the highest position is the most valuable), then re-sorted; `budget`
/// null means unbounded. Null when nothing qualifies.
///
/// The clones share the donor's buffers, so this costs no GPU memory —
/// but `current_kv_bytes` bills them again, because the accounting is
/// per-entry and cannot see the sharing. That over-bills only while BOTH
/// entries are resident and self-corrects the moment the donor is evicted
/// (its bill goes, the buffers stay alive under the inheritor). Erring
/// toward eviction is the safe direction for a hard cap.
pub fn cloneCheckpointsUpTo(
    allocator: std.mem.Allocator,
    src: []const SSMCheckpoint,
    limit: usize,
    budget: ?u64,
) !?[]SSMCheckpoint {
    var out = std.ArrayList(SSMCheckpoint).empty;
    errdefer {
        for (out.items) |*c| c.deinit(allocator);
        out.deinit(allocator);
    }
    var spent: u64 = 0;
    var k = src.len;
    while (k > 0) {
        k -= 1;
        const cp = &src[k];
        if (cp.layers.len == 0) continue;
        if (cp.pos > limit) continue;
        const cost = ssmCheckpointBytes(cp);
        if (budget) |b| {
            if (spent + cost > b) break;
        }
        spent += cost;
        try out.ensureUnusedCapacity(allocator, 1);
        out.appendAssumeCapacity(try trunk_mod.shareSsmCheckpoint(allocator, cp));
    }
    if (out.items.len == 0) {
        out.deinit(allocator);
        return null;
    }
    // Collected newest-first; the list contract is ascending by pos.
    std.mem.reverse(SSMCheckpoint, out.items);
    return try out.toOwnedSlice(allocator);
}

/// Frees all but the newest `CHECKPOINT_HOT_KEPT` of an ascending checkpoint list. The older
/// ones are restored from the SSD tier, which holds them all; one past `text_len` is not in the SSD record
/// and stays.
pub fn keepNewestCheckpoints(self: *HotPrefixCache, list: *?[]SSMCheckpoint, text_len: usize) void {
    const cps = list.* orelse return;
    const newest: usize = CHECKPOINT_HOT_KEPT;
    var n_keep: usize = 0;
    for (cps, 0..) |*cp, i| {
        if (cps.len - i <= newest or cp.pos > text_len) n_keep += 1;
    }
    if (n_keep == cps.len) return;
    var at: usize = 0;
    for (cps, 0..) |*cp, i| {
        if (cps.len - i <= newest or cp.pos > text_len) {
            cps[at] = cp.*;
            at += 1;
        } else {
            cp.deinit(self.allocator);
        }
    }
    for (cps[at..]) |*stub| stub.* = .{ .pos = 0, .layers = &.{} };
    list.* = self.allocator.realloc(cps, at) catch cps;
}

