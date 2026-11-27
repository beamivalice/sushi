//! Hot prefix cache lookup, restorability ranking, restore orchestration, and residency digests.

const std = @import("std");
const mlx = @import("mlx.zig");
const trunk_mod = @import("trunk.zig");
const kv_quant = @import("kv_quant.zig");
const kv_disk_cache = @import("kv_disk_cache.zig");
const io_util = @import("io_util.zig");
const log = @import("log.zig");
const restore_dump = @import("restore_dump.zig");
const prefix_arch = @import("arch/prefix.zig");
const prefix_policy = @import("prefix_policy.zig");
const pc = @import("prefix_cache.zig");

const KVCache = trunk_mod.KVCache;
const KVCacheSnapshot = trunk_mod.KVCacheSnapshot;
const SSMCacheEntry = trunk_mod.SSMCacheEntry;
const SSMCheckpoint = trunk_mod.SSMCheckpoint;
const restoreSsmCheckpoint = trunk_mod.restoreSsmCheckpoint;
const applyQsaHistoryAt = prefix_arch.aux.applyQsaHistoryAt;
const checkpointHasQsaPooled = prefix_arch.aux.checkpointHasQsaPooled;
const entriesHaveQsaHistory = prefix_arch.aux.entriesHaveQsaHistory;
const qsaRestoreSatisfiesForward = prefix_arch.aux.qsaRestoreSatisfiesForward;

const HotPrefixCache = pc.HotPrefixCache;
const Entry = pc.Entry;
const Recurrent = pc.Recurrent;
const RamMatch = pc.RamMatch;
const MediaSpan = pc.MediaSpan;
const MediaRule = pc.MediaRule;
const MatchProbe = pc.MatchProbe;
const LookupResult = pc.LookupResult;
const DflashSnap = pc.DflashSnap;
const DflashTarget = pc.DflashTarget;
const specAdoptPlan = pc.specAdoptPlan;
const crossKeyBoundary = pc.crossKeyBoundary;
const missKind = pc.missKind;
const MIN_CANCELLED_COMMIT_TOKENS = pc.MIN_CANCELLED_COMMIT_TOKENS;
const RING_RESTORE_MIN_TOKENS = pc.RING_RESTORE_MIN_TOKENS;
const restoreWouldPinEntry = HotPrefixCache.restoreWouldPinEntry;

/// Restore a committed speculative-side cache (DFlash context or MTP
/// history) into the caller's cache, clamped to the trunk's restored
/// length. Returns the base position on success, null when there is
/// nothing to restore, the snap starts PAST what the trunk actually
/// reused, or the snap ends BEFORE it (a history with a gap right below
/// the generation point is worse than a blind start). Best-effort by
/// contract: a failure leaves the caller blind, never wrong.
pub fn restoreSpecSnap(snap_opt: ?*const DflashSnap, target: ?DflashTarget, matched: usize, s: mlx.mlx_stream, what: []const u8) ?usize {
    const t = target orelse return null;
    const snap = snap_opt orelse return null;
    const plan = specAdoptPlan(snap.base_pos, snap.snapshot.step, matched, t.head != null, snap.head_aux != null);
    if (plan == .skip) {
        log.info("  [hot-cache] {s} not adopted: want {d} tokens from base {d}, snap holds {d} (matched {d})\n", .{ what, matched -| snap.base_pos, snap.base_pos, snap.snapshot.step, matched });
        return null;
    }
    // An in-checkpoint head adopts its KV and aux halves together or not at all.
    if (t.head) |xfm| return prefix_arch.adoptSpecHead(xfm, plan, snap, t.base_pos);
    const want = plan.kv_only;
    t.cache.restore(&snap.snapshot) catch |err| {
        log.warn("  [hot-cache] {s} restore failed: {s} — starts blind\n", .{ what, @errorName(err) });
        return null;
    };
    t.cache.truncate(want, s) catch |err| {
        log.warn("  [hot-cache] {s} clamp failed: {s} — starts blind\n", .{ what, @errorName(err) });
        return null;
    };
    t.base_pos.* = snap.base_pos;
    log.debug("  [hot-cache] {s} restored: {d} tokens from base {d}\n", .{ what, want, snap.base_pos });
    return snap.base_pos;
}

pub fn restoreDflash(e: *const Entry, target: ?DflashTarget, matched: usize, s: mlx.mlx_stream) ?usize {
    return restoreSpecSnap(if (e.dflash) |*d| d else null, target, matched, s, "dflash context");
}

pub fn restoreMtp(e: *const Entry, target: ?DflashTarget, matched: usize, s: mlx.mlx_stream) ?usize {
    return restoreSpecSnap(if (e.mtp) |*m| m else null, target, matched, s, "mtp history");
}

/// Disk-tier variant: load the persisted spec snapshot (v4 sidecar) as a
/// transient and adopt it under the EXACT same clamp rule as the RAM
/// tier (`restoreSpecSnap`). The trunk restore already forwarded nothing,
/// so without this a disk hit drafted blind — the 92.6% → 66.5%
/// acceptance class the RAM tier already fixed.
pub fn diskRestoreSpec(
    d: *kv_disk_cache.DiskTier,
    idx: usize,
    target: ?DflashTarget,
    matched: usize,
    s: mlx.mlx_stream,
    which: kv_disk_cache.SpecKind,
) ?usize {
    const t = target orelse return null;
    const loaded = d.loadSpecSnap(idx, which, t.cache.entries.len, t.cache.config) orelse return null;
    // restore() refcount-shares the arrays into the target, so the
    // transient snapshot is freed right after. A pre-v5 sidecar carries no head half, so a
    // head target declines it.
    var snap = DflashSnap{
        .snapshot = loaded.snap,
        .base_pos = loaded.base,
        .head_aux = loaded.head_aux,
        .head_pos_base = loaded.head_pos_base,
        .head_marks = loaded.head_marks,
    };
    defer snap.deinit();
    return restoreSpecSnap(&snap, target, matched, s, switch (which) {
        .dflash => "dflash context",
        .mtp => "mtp history",
    });
}

/// The largest checkpoint whose `pos ≤ limit` (checkpoints are sorted
/// ascending). Shared by the RAM restore and the RAM-vs-disk fairness
/// comparison — both need the effective restorable length of a hybrid
/// entry, which is the highest snapshotted position ≤ the token match, NOT
/// the raw match length (SSM state only exists at snapshotted positions).
pub fn highestCheckpointAtOrBelow(cps: []const SSMCheckpoint, limit: usize) ?*const SSMCheckpoint {
    var picked: ?*const SSMCheckpoint = null;
    for (cps) |*cp| {
        // A deinit'd stub (`shedCheckpointsToFit`'s realloc-failure
        // leftover) has zero layers and restores nothing — skip it.
        if (cp.layers.len == 0 or cp.isReleased()) continue;
        if (cp.pos > limit) break;
        picked = cp;
    }
    return picked;
}

/// Why a restore stops short of the match: the request that committed `e` left no point at its end, or the
/// prompt diverges inside the entry past its last point below the match.
pub fn lostReason(e: *const Entry, shared: usize, restored: usize) []const u8 {
    if (shared <= restored) return "";
    const noted = e.end_note.len > 0 and !std.mem.eql(u8, e.end_note, "taken");
    return if (shared >= e.tokens.len and noted) e.end_note else "no-point";
}

pub fn highestCoveringCheckpoint(cps: []const SSMCheckpoint, limit: usize) ?*const SSMCheckpoint {
    var cap = limit;
    while (highestCheckpointAtOrBelow(cps, cap)) |cp| {
        const src = qsaHistorySource(cps, cp) orelse cp;
        if (prefix_arch.aux.checkpointQsaCoversPos(cp, src, cp.pos)) return cp;
        if (cp.pos == 0) break;
        cap = cp.pos - 1;
    }
    return null;
}


/// The rows a restore will deliver, which is not the rows it matched: a hybrid restore is
/// clamped to its highest SSM checkpoint at or below the match, so the lien test must weigh this.
pub fn deliverableShare(cps: ?[]const SSMCheckpoint, hybrid: bool, shared: usize) usize {
    if (!hybrid) return shared;
    const list = cps orelse return 0;
    const cp = highestCheckpointAtOrBelow(list, shared) orelse return 0;
    return cp.pos;
}

/// Latest checkpoint that carries the pooled indexer bank, unless it IS `restored`
/// (restoreSsmCheckpoint already installed that bank at full length). Every checkpoint
/// carries its own leftover; only one carries the bank.
pub fn qsaHistorySource(cps: []const SSMCheckpoint, restored: *const SSMCheckpoint) ?*const SSMCheckpoint {
    var i = cps.len;
    while (i > 0) {
        i -= 1;
        if (!checkpointHasQsaPooled(&cps[i])) continue;
        if (&cps[i] == restored) return null;
        return &cps[i];
    }
    return null;
}

/// Reset every SSM entry to the uninitialized (cold) state. Used on every
/// miss / failed-restore path so a subsequent prefill starts from a clean
/// recurrent state instead of stale conv/ssm buffers.
pub fn resetSsmEntries(entries: []SSMCacheEntry) void {
    for (entries) |*ssm| {
        _ = mlx.mlx_array_free(ssm.conv_state);
        _ = mlx.mlx_array_free(ssm.ssm_state);
        ssm.conv_state = mlx.mlx_array_new();
        ssm.ssm_state = mlx.mlx_array_new();
        ssm.initialized = false;
        prefix_arch.aux.ssmFreeQsaState(ssm);
        ssm.ple_prev_valid = false;
    }
    trunk_mod.ssmDetachFromGroup(entries);
}


/// Find the entry with the longest EFFECTIVELY RESTORABLE prefix shared
/// with `prompt_ids` and matching `(has_tools, quant_config)`. For a hybrid
/// target that means the highest SSM checkpoint at or below the raw token
/// match; a longer raw match with no usable checkpoint must not hide an
/// older entry that can actually restore. Returns the entry index and raw
/// shared-prefix length; null if no entry matches the key. Wave 1.A:
/// the config filter exists because cross-config buffer layouts differ
/// — a slot running `kv_quant=4` cannot restore from an entry committed
/// in dense (or 8-bit) mode and vice versa. The full `KVQuantConfig`
/// (scheme + bits + group_size) is compared because `Scheme.affine`
/// covers BOTH 4-bit and 8-bit packings: filtering on `Scheme` alone
/// would let a 4-bit entry alias to an 8-bit slot and crash SDPA on
/// restore. Different media keys may share only the text state before
/// the earliest known media-placeholder boundary; without a boundary the
/// lookup stays conservative and rejects the entry. See
/// `tests/test_kv_quant_per_request.sh`.
pub fn findBestRestorableMatch(
    self: *const HotPrefixCache,
    prompt_ids: []const u32,
    has_tools: bool,
    vision_key: u64,
    media_start: ?usize,
    media_chain: []const MediaSpan,
    quant_config: kv_quant.KVQuantConfig,
    require_ssm_checkpoint: bool,
    probe: ?*MatchProbe,
) ?RamMatch {
    var best: prefix_policy.BestMatch = .{};
    for (self.entries.items, 0..) |*e, i| {
        // A checked-out entry's buffers belong to another slot; its snapshot is empty.
        if (e.checkout != null) continue;
        if (e.has_tools != has_tools) continue;
        if (!std.meta.eql(e.quant_config, quant_config)) continue;

        var max_shared = @min(e.tokens.len, prompt_ids.len);
        if (e.vision_key != vision_key) {
            // Placeholder token IDs do not encode media pixels. Once an
            // image/audio/video row is forwarded, model state depends on
            // the media and cannot cross keys past the first block the
            // two prompts do not share.
            const safe_boundary = crossKeyBoundary(e.vision_key, e.media_start, media_start, media_chain) orelse continue;
            max_shared = @min(max_shared, safe_boundary);
        }
        const shared = prefix_policy.commonPrefixLen(e.tokens, prompt_ids, max_shared);

        // Record the RAW match before the restorability filter can drop
        // this candidate — a null return with a long raw match is the
        // expensive miss, and the only place that fact still exists.
        if (probe) |p| {
            p.candidates += 1;
            if (shared > p.best_raw) p.best_raw = shared;
        }

        // An un-restorable ringed candidate stays eligible at 0, so a lookup with nothing
        // better still lands on it and declines by name.
        const effective = if (require_ssm_checkpoint) blk: {
            const cps = e.ssm_checkpoints orelse continue;
            const cp = highestCheckpointAtOrBelow(cps, shared) orelse continue;
            break :blk cp.pos;
        } else if (ringRestore(e, shared, prompt_ids.len)) |rr| rr.len else 0;
        best.offer(i, effective, shared);
    }
    if (best.idx) |idx| return .{ .idx = idx, .shared = best.shared };
    return null;
}

pub const RingRestore = struct { len: usize, cp: ?*const KVCacheSnapshot };

/// Where an entry restores a match of `shared` tokens. A ringed entry's own ring holds the
/// window at its END; below that only a ring checkpoint (`cp`) can serve, the one reaching
/// highest at its position or the match, whichever is lower. Null = un-restorable. Any
/// other entry holds a prefix and restores at `shared`.
pub fn ringRestore(e: *const Entry, shared: usize, prompt_len: usize) ?RingRestore {
    if (e.snapshot.ringServes(clampLen(shared, prompt_len))) return .{ .len = shared, .cp = null };
    var best: ?RingRestore = null;
    for (e.ring_cps orelse return null) |*cp| {
        const at = @min(shared, cp.step);
        if (at < RING_RESTORE_MIN_TOKENS or !cp.ringServes(clampLen(at, prompt_len))) continue;
        if (best == null or at > best.?.len) best = .{ .len = at, .cp = cp };
    }
    return best;
}

/// The length a restore of `len` matched tokens clamps to: a full match re-forwards its last token.
pub fn clampLen(len: usize, prompt_len: usize) usize {
    return if (len == prompt_len and len > 1) len - 1 else len;
}

pub fn findBestMatch(self: *const HotPrefixCache, prompt_ids: []const u32, has_tools: bool, vision_key: u64, quant_config: kv_quant.KVQuantConfig) ?RamMatch {
    const match = self.findBestRestorableMatch(prompt_ids, has_tools, vision_key, null, &.{}, quant_config, false, null) orelse return null;
    return .{ .idx = match.idx, .shared = match.shared };
}

/// Try to restore a matching entry into `target_cache`. On success, returns
/// the matched prefix length. On miss, fully resets `target_cache` and
/// returns 0. Caller should prefill the trailing tokens after this.
///
/// The `target_*` parameters generalize the legacy single-slot path
/// (`xfm.cache`, `xfm.moe_seq_offset`, `xfm.ssm_entries`) so Phase 2
/// per-slot caches can reuse the same restore machinery.
pub fn lookupAndRestore(
    self: *HotPrefixCache,
    target_cache: *KVCache,
    target_moe_seq_offset: *usize,
    target_ssm_entries: ?[]SSMCacheEntry,
    s: mlx.mlx_stream,
    prompt_ids: []const u32,
    has_tools: bool,
    vision_key: u64,
    dflash_target: ?DflashTarget,
    mtp_target: ?DflashTarget,
) !LookupResult {
    return self.lookupAndRestoreWithMedia(
        target_cache,
        target_moe_seq_offset,
        target_ssm_entries,
        s,
        prompt_ids,
        has_tools,
        vision_key,
        null,
        &.{},
        dflash_target,
        mtp_target,
        null,
        false,
    );
}

/// The checkout-capable entry point: `slot_id` names the slot that will own the restored buffers.
pub fn lookupAndRestoreForSlot(
    self: *HotPrefixCache,
    target_cache: *KVCache,
    target_moe_seq_offset: *usize,
    target_ssm_entries: ?[]SSMCacheEntry,
    s: mlx.mlx_stream,
    prompt_ids: []const u32,
    has_tools: bool,
    vision_key: u64,
    media_start: ?usize,
    dflash_target: ?DflashTarget,
    mtp_target: ?DflashTarget,
    slot_id: usize,
) !LookupResult {
    return self.lookupAndRestoreWithMedia(
        target_cache,
        target_moe_seq_offset,
        target_ssm_entries,
        s,
        prompt_ids,
        has_tools,
        vision_key,
        media_start,
        &.{},
        dflash_target,
        mtp_target,
        slot_id,
        false,
    );
}

pub fn lookupAndRestoreWithMedia(
    self: *HotPrefixCache,
    target_cache: *KVCache,
    target_moe_seq_offset: *usize,
    target_ssm_entries: ?[]SSMCacheEntry,
    s: mlx.mlx_stream,
    prompt_ids: []const u32,
    has_tools: bool,
    vision_key: u64,
    media_start: ?usize,
    media_chain: []const MediaSpan,
    dflash_target: ?DflashTarget,
    mtp_target: ?DflashTarget,
    slot_id: ?usize,
    skip: bool,
) !LookupResult {
    const rec: Recurrent = if (target_ssm_entries) |e| .{ .ssm = e } else .none;
    return self.lookupImpl(target_cache, target_moe_seq_offset, rec, s, prompt_ids, has_tools, vision_key, media_start, media_chain, dflash_target, mtp_target, slot_id, skip);
}

/// An unreadable or corrupt file fails the same way on every later lookup, so its entry dies;
/// a coverage miss (no checkpoint, QSA gap) can still serve a shorter prefix, so it stays, and so does an entry
/// whose restore was refused for memory (`refusals_before`: `mlx.memoryRefusals()` as the restore began).
pub fn poisonUnreadable(d: *kv_disk_cache.DiskTier, idx: usize, err: anyerror, refusals_before: u64) void {
    switch (err) {
        error.MlxError, error.DiskCacheCorruptChunk, error.DiskCacheCorruptRing, error.DiskCacheCorruptSsm => {
            if (mlx.memoryRefusals() != refusals_before) {
                log.warn("  [disk-cache] e{d} kept: its restore was refused for memory\n", .{d.entries.items[idx].id});
                return;
            }
            _ = d.poisonId(d.entries.items[idx].id, @errorName(err));
        },
        else => {},
    }
}

pub fn lookupImpl(
    self: *HotPrefixCache,
    target_cache: *KVCache,
    target_moe_seq_offset: *usize,
    rec: Recurrent,
    caller_stream: mlx.mlx_stream,
    prompt_ids: []const u32,
    has_tools: bool,
    vision_key: u64,
    media_start: ?usize,
    /// The request's media blocks (`MediaSpan`); an entry holding a prefix
    /// of them restores up to the first block it lacks.
    media_chain: []const MediaSpan,
    dflash_target: ?DflashTarget,
    mtp_target: ?DflashTarget,
    /// Restore by move: non-null opts this request into the checkout (the caller promises
    /// `releaseCheckout` on every path that ends the slot).
    slot_id: ?usize,
    skip: bool,
) !LookupResult {
    const s = self.arrays_on orelse caller_stream;
    self.last_restored_used = null;
    self.last_restored_disk_id = null;
    self.last_restored_held = false;
    self.last_restored_live = false;
    if (rec == .arch) try prefix_arch.checkLent(rec.arch);
    if (skip) {
        if (rec == .arch) prefix_arch.handover(self, rec.arch, s);
        try target_cache.truncate(0, s);
        rec.reset();
        target_moe_seq_offset.* = 0;
        return .{ .matched = 0, .full_match = false };
    }
    // Each disk record carries the media chain its rows were computed under (manifest v10), so
    // these limits hold every candidate to what the ONE rule (`crossKeyBoundary`) allows: the
    // whole prefix under an equal key, up to the first block it lacks when its chain is this
    // request's first k blocks, and otherwise the text before either side's first media row.
    // The rule is load-bearing in BOTH directions — placeholder ids are ordinary vocabulary
    // entries, so tokens can match past a boundary while the rows there are the other side's
    // pixels.
    const media_rule: MediaRule = .{ .vision_key = vision_key, .media_start = media_start, .chain = media_chain };
    const media_limits = media_rule.limits();
    var probe: MatchProbe = .{};
    const match = self.findBestRestorableMatch(
        prompt_ids,
        has_tools,
        vision_key,
        media_start,
        media_chain,
        target_cache.config,
        rec.hybrid(),
        &probe,
    );

    if (rec == .arch) {
        if (prefix_arch.rollbackLive(self, rec.arch, target_cache, target_moe_seq_offset, s, prompt_ids, has_tools, media_limits, match)) |res| return res;
    }

    // ── SSD tier: consult when it can beat the RAM match meaningfully
    // (fresh boot, post-eviction). Phase 3 handles hybrid targets too —
    // the tier persists per-position SSM checkpoints beside the KV chunks
    // and restores both.
    if (self.disk) |*d| disk: {
        if (rec == .arch) {
            const ram_eff: usize = if (match) |m| blk: {
                const cps = self.entries.items[m.idx].ssm_checkpoints orelse break :blk 0;
                const cp = highestCheckpointAtOrBelow(cps, m.shared) orelse break :blk 0;
                break :blk cp.pos;
            } else 0;
            if (prefix_arch.restoreDisk(self, d, rec.arch, target_cache, ram_eff, prompt_ids, has_tools, media_limits, dflash_target, s)) |res| {
                target_moe_seq_offset.* = res.matched;
                return res;
            }
            break :disk;
        }
        const disk_limit: u32 = @intCast(prompt_ids.len);
        // A ringed cache restores only at a ring file: the chunks hold its other layers alone.
        if (target_cache.swa_ring_window > 0) {
            const ram_eff: usize = if (match) |m| blk: {
                const rr = ringRestore(&self.entries.items[m.idx], m.shared, prompt_ids.len) orelse break :blk 0;
                break :blk rr.len;
            } else 0;
            const rm = d.bestRingMatch(prompt_ids, has_tools, target_cache.config, disk_limit, target_cache.swa_ring_window, media_limits) orelse break :disk;
            if (@as(usize, rm.at) < ram_eff + kv_disk_cache.MIN_DISK_ADVANTAGE_TOKENS) break :disk;
            const full_match = rm.at == prompt_ids.len;
            const final_len: u32 = if (full_match and rm.at > 1) rm.at - 1 else rm.at;
            const sw = io_util.Stopwatch.init(d.io);
            const refusals = mlx.memoryRefusals();
            d.restoreIntoRinged(target_cache, rm.idx, rm.pos, final_len, s) catch |err| {
                log.warn("  [disk-cache] ring restore failed: {s} — falling back to RAM/cold path\n", .{@errorName(err)});
                poisonUnreadable(d, rm.idx, err, refusals);
                target_cache.truncate(0, s) catch {};
                break :disk;
            };
            target_moe_seq_offset.* = final_len;
            self.last_restored_disk_id = d.entries.items[rm.idx].id;
            log.info("  [disk-cache] restored {d}/{d} tokens from SSD in {d}ms (ring@{d})\n", .{ final_len, prompt_ids.len, sw.read() / std.time.ns_per_ms, rm.pos });
            const disk_mtp = diskRestoreSpec(d, rm.idx, mtp_target, final_len, s, .mtp);
            return .{
                .matched = final_len,
                .full_match = full_match,
                .dflash_base = diskRestoreSpec(d, rm.idx, dflash_target, final_len, s, .dflash),
                .mtp_base = disk_mtp,
            };
        }
        const dm = d.bestMatch(prompt_ids, has_tools, target_cache.config, media_limits) orelse {
            // Silent no-entry misses are why a 40 GB disk tier looked
            // dead in a live post-mortem (2026-09-07): nothing in the
            // log ever said the tier was consulted and found nothing.
            if (d.entryCount() > 0)
                log.debug("  [disk-cache] lookup miss: {d} entries on disk, none share a token prefix under this (tools, quant) key\n", .{d.entryCount()});
            break :disk;
        };
        const usable: u32 = @min(dm.usable, disk_limit);

        if (rec.ssmEntries()) |ssm_entries| {
            // Hybrid: compare EFFECTIVE restorable positions — the largest
            // SSM checkpoint ≤ the match on each tier, not the raw prefix
            // length (KV alone is useless without matching SSM state).
            // The disk side ranks entries by that checkpoint too (#312's
            // RAM lesson: a longer raw match whose checkpoints sit past
            // the divergence restores nothing — and it must not shadow a
            // shorter entry with a higher restorable position).
            const ram_eff: usize = if (match) |m| blk: {
                const e = &self.entries.items[m.idx];
                const cps = e.ssm_checkpoints orelse break :blk 0;
                const cp = highestCheckpointAtOrBelow(cps, m.shared) orelse break :blk 0;
                break :blk cp.pos;
            } else 0;
            const hm = d.bestHybridMatch(prompt_ids, has_tools, target_cache.config, disk_limit, media_limits) orelse break :disk;
            const disk_cp = hm.cp;
            if (@as(usize, disk_cp) < ram_eff + kv_disk_cache.MIN_DISK_ADVANTAGE_TOKENS) break :disk;
            const sw = io_util.Stopwatch.init(d.io);
            const refusals = mlx.memoryRefusals();
            const restored = d.restoreIntoHybrid(target_cache, ssm_entries, hm.idx, disk_cp, s) catch |err| {
                log.warn("  [disk-cache] hybrid restore failed: {s} — falling back to RAM/cold path\n", .{@errorName(err)});
                poisonUnreadable(d, hm.idx, err, refusals);
                // A failed restore can leave the cache AND ssm entries
                // half-rebuilt; reset both before the fall-through.
                target_cache.truncate(0, s) catch {};
                resetSsmEntries(ssm_entries);
                break :disk;
            };
            if (self.qsa_history_required and !(entriesHaveQsaHistory(ssm_entries) and qsaRestoreSatisfiesForward(ssm_entries, restored))) {
                log.warn("  [disk-cache] hybrid restore carries no QSA history — falling back to RAM/cold path\n", .{});
                target_cache.truncate(0, s) catch {};
                resetSsmEntries(ssm_entries);
                break :disk;
            }
            // A checkpoint is always ≤ prompt_len−1, so a hybrid restore
            // never takes the full-match branch (same as the RAM path).
            target_moe_seq_offset.* = restored;
            self.last_restored_disk_id = d.entries.items[hm.idx].id;
            const ms = sw.read() / std.time.ns_per_ms;
            const lost = @as(usize, hm.usable) -| restored;
            self.last_restored_held = d.last_point_held;
            log.info("  [disk-cache] restored {d}/{d} tokens from {s} in {d}ms (ssm@{d}), matched {d}, lost {d}{s}\n", .{
                restored, prompt_ids.len, if (d.last_point_held) "a held restore point (RAM)" else "SSD", ms, disk_cp, hm.usable, lost, if (lost > 0) ": no-point" else "",
            });
            const disk_mtp = diskRestoreSpec(d, hm.idx, mtp_target, restored, s, .mtp);
            return .{
                .matched = restored,
                .full_match = false,
                // The spec sidecar of the entry the TRUNK came from: `dm` ranks by raw
                // length, `hm` by restorable checkpoint, so they routinely differ.
                .dflash_base = diskRestoreSpec(d, hm.idx, dflash_target, restored, s, .dflash),
                .mtp_base = disk_mtp,
                .slot_owned = true,
                .disk_restore = .{ .id = d.entries.items[hm.idx].id, .rows = @intCast(restored) },
            };
        }

        const ram_len: usize = if (match) |m| m.shared else 0;
        if (usable <= ram_len) break :disk;
        if (usable - ram_len < kv_disk_cache.MIN_DISK_ADVANTAGE_TOKENS) break :disk;
        // `usable` is `dm.usable` clamped to the entry's kv_len (see
        // DiskTier.bestMatch) and to the request's media boundary, so it
        // IS the restorable length — restore
        // only the chunks covering it. Loading the whole entry
        // (restoreInto) would read a long stored prefix in full to serve a
        // short shared prefix, making a diverged-prefix "hit" slower than a
        // cold miss.
        const effective: usize = usable;
        const full_match = effective == prompt_ids.len;
        const final_len: usize = if (full_match and effective > 1) effective - 1 else effective;
        const sw = io_util.Stopwatch.init(d.io);
        const refusals = mlx.memoryRefusals();
        d.restorePrefixInto(target_cache, dm.idx, @intCast(final_len), s) catch |err| {
            log.warn("  [disk-cache] restore failed: {s} — falling back to RAM/cold path\n", .{@errorName(err)});
            poisonUnreadable(d, dm.idx, err, refusals);
            // A failed restore can leave a half-rebuilt cache; reset it.
            target_cache.truncate(0, s) catch {};
            break :disk;
        };
        target_moe_seq_offset.* = final_len;
        self.last_restored_disk_id = d.entries.items[dm.idx].id;
        const ms = sw.read() / std.time.ns_per_ms;
        log.info("  [disk-cache] restored {d}/{d} tokens from SSD ({d} chunks) in {d}ms\n", .{ final_len, prompt_ids.len, d.chunks_loaded_last, ms });
        const disk_mtp = diskRestoreSpec(d, dm.idx, mtp_target, final_len, s, .mtp);
        return .{
            .matched = final_len,
            .full_match = full_match,
            .dflash_base = diskRestoreSpec(d, dm.idx, dflash_target, final_len, s, .dflash),
            .mtp_base = disk_mtp,
            .disk_restore = .{ .id = d.entries.items[dm.idx].id, .rows = @intCast(final_len) },
        };
    }

    if (match == null) {
        try target_cache.truncate(0, s);
        rec.reset();
        target_moe_seq_offset.* = 0;
        // The filter dropped every candidate. This arm used to be silent,
        // so a 393k-token prompt that the cache almost had cold-prefilled
        // for 560 s with no `[hot-cache]` line at all. Same phrasing as
        // the one-entry miss below — one string to grep for.
        switch (missKind(probe.candidates, probe.best_raw)) {
            .cold => {},
            .no_checkpoint => log.info(
                "  [hot-cache] hybrid miss (no checkpoint ≤ {d} of {d} in {d} entries); cold prefill\n",
                .{ probe.best_raw, prompt_ids.len, probe.candidates },
            ),
        }
        return .{ .matched = 0, .full_match = false };
    }
    const m = match.?;
    const e = &self.entries.items[m.idx];
    // Decline a restore that is a lien (`restoreWouldPinEntry`), before the bump and before
    // `last_restored_used`; weighed on the DELIVERABLE share (a hybrid is clamped to its
    // highest checkpoint at or below the match). SSD-first only.
    const deliverable = deliverableShare(e.ssm_checkpoints, rec.hybrid(), m.shared);
    if (self.ssd_first and restoreWouldPinEntry(e.kv_bytes, self.restore_pin_min_bytes, e.tokens.len, deliverable)) {
        try target_cache.truncate(0, s);
        rec.reset();
        target_moe_seq_offset.* = 0;
        log.info("  [hot-cache] declined a {d}-token restore ({d} deliverable) from a {d}-token entry ({d} MB): the share is a lien on the whole entry; cold prefill\n", .{
            m.shared,
            deliverable,
            e.tokens.len,
            e.kv_bytes / (1024 * 1024),
        });
        return .{ .matched = 0, .full_match = false };
    }
    // A full reuse re-forwards the last token; a one-token prompt has nothing before it.
    if (prompt_ids.len == 1) {
        try target_cache.truncate(0, s);
        rec.reset();
        target_moe_seq_offset.* = 0;
        return .{ .matched = 0, .full_match = false };
    }
    const would_full = m.shared == prompt_ids.len and m.shared > 1;
    const restore_cap: usize = blk: {
        if (!rec.hybrid() or !would_full) break :blk m.shared;
        const cps = e.ssm_checkpoints orelse break :blk m.shared;
        const cp = highestCoveringCheckpoint(cps, m.shared) orelse break :blk m.shared;
        if (cp.pos == m.shared) break :blk m.shared - 1;
        break :blk m.shared;
    };
    if (e.ssm_checkpoints) |cps| {
        if (highestCoveringCheckpoint(cps, restore_cap) == null and highestCheckpointAtOrBelow(cps, restore_cap) != null) {
            return error.QsaHistoryGap;
        }
    }
    const used_before_restore = e.last_used;
    e.last_used = self.bumpCounter();
    // Identity of the entry this request runs on: evicting it frees nothing (shared buffers).
    self.last_restored_used = e.last_used;
    self.last_restored_shared = m.shared;

    // The sole caller reads an error as "no match" and cold-prefills the whole prompt, so a
    // failed restore must hand back an EMPTY cache, never a half-bound one.
    errdefer {
        target_cache.truncate(0, s) catch {};
        rec.reset();
        target_moe_seq_offset.* = 0;
        self.last_restored_used = null;
        e.last_used = used_before_restore;
    }
    try target_cache.restore(&e.snapshot);

    // Hybrid path: if the entry carries SSM checkpoints, restore the SSM
    // state at the largest stride-aligned position ≤ m.shared and clamp
    // the effective matched length to that position. KV is positionally
    // trimmable; SSM is only restorable at the snapshotted positions.
    // The two MUST stay in sync, so we rewind KV further too.
    var dump = restore_dump.RestoreDumpMeta{ .kind = "restore", .pos = 0 };
    var effective_matched: usize = restore_cap;
    if (rec == .arch) {
        effective_matched = try prefix_arch.restoreEntry(self, rec.arch, e, restore_cap, prompt_ids);
        if (effective_matched == 0) rec.reset();
    } else if (rec.ssmEntries()) |entries| {
        if (e.ssm_checkpoints) |cps| {
            if (highestCoveringCheckpoint(cps, restore_cap)) |cp| {
                try restoreSsmCheckpoint(entries, cp, s);
                effective_matched = cp.pos;
                const bank = qsaHistorySource(cps, cp);
                if (bank) |src| {
                    try applyQsaHistoryAt(entries, src, cp.pos, s, false);
                }
                const ring_rows: c_int = blk: {
                    for (entries) |*ent| {
                        if (ent.aux_state.ctx == null) continue;
                        const sh = mlx.getShape(ent.aux_state);
                        if (sh.len >= 2) break :blk sh[1];
                    }
                    break :blk 0;
                };
                dump.cp = cp.pos;
                dump.bank_from = if (bank) |b| b.pos else cp.pos;
                dump.source = if (bank == null) "own" else "donor";
                dump.entry_idx = m.idx + 1;
                dump.entry_count = self.entries.items.len;
                log.debug("  [hot-cache] restore pos={d} cp={d} bank_from=cp@{d} ring_rows={d} source={s}\n", .{
                    effective_matched,
                    dump.cp,
                    dump.bank_from,
                    ring_rows,
                    dump.source,
                });
                if (self.qsa_history_required and !(entriesHaveQsaHistory(entries) and qsaRestoreSatisfiesForward(entries, effective_matched))) {
                    resetSsmEntries(entries);
                    effective_matched = 0;
                }
            } else {
                // No checkpoint at or before this prefix length — reset
                // SSM and treat the match as zero-effective (we have to
                // cold-prefill anyway because SSM state would be wrong).
                resetSsmEntries(entries);
                effective_matched = 0;
            }
        } else {
            // Hybrid model without checkpoints (e.g., committed pre-Phase-1).
            // Reset and treat as cold prefill — we can't safely reuse.
            resetSsmEntries(entries);
            effective_matched = 0;
        }
    } else if (ringRestore(e, restore_cap, prompt_ids.len)) |rr| {
        if (rr.cp) |cp| {
            try target_cache.restoreRing(cp);
            effective_matched = rr.len;
            log.info("  [hot-cache] ring checkpoint @{d} serves the match of {d} (the entry's ring ends at {d})\n", .{ cp.step, m.shared, e.snapshot.step });
        }
    }
    target_moe_seq_offset.* = effective_matched;

    // Miss path (hybrid without a usable checkpoint, and the QSA-history
    // decline that funnels into it): also reset KV.
    if (effective_matched == 0) {
        try target_cache.truncate(0, s);
        // A 0-token outcome is not a restore: the marker was set above the restore (the hybrid
        // clamp needs the entry live) and would otherwise shield a fully reclaimable entry from
        // the admission pass. Promoted, the entry would make a usable one the next eviction victim.
        self.last_restored_used = null;
        e.last_used = used_before_restore;
        log.info("  [hot-cache] hybrid miss (no checkpoint ≤ {d} of {d}); cold prefill\n", .{ m.shared, prompt_ids.len });
        return .{ .matched = 0, .full_match = false };
    }

    const full_match = effective_matched == prompt_ids.len;
    const final_len: usize = if (full_match and effective_matched > 1) effective_matched - 1 else effective_matched;

    // ALWAYS clamp the restored cache to the matched length. The old guard
    // (`final_len < e.tokens.len`) skipped this on a WHOLE-entry match — but a
    // snapshot can be committed with a KV buffer LONGER than its logical token
    // count: PLD/speculative decode leaves stale draft positions in the buffer
    // past the committed step, and `commit` snapshots them. Restoring that
    // (then skipping the truncate) left `cache.offset` AHEAD of the matched
    // length that generation tracks (`moe_seq_offset`) — a silent drift that
    // corrupts RoPE positions and CRASHES the sliding-window prefill mask
    // (`broadcast_shapes` mask-vs-KV mismatch). truncate is a no-op when the buffer is
    // already `final_len`, so unconditional clamping is safe and restores the
    // invariant cache.offset == matched. The stale KV tail has no matching
    // token id, so the match can never reach into it — discarding it is correct.
    target_cache.truncate(final_len, s) catch |err| {
        // A ringed sliding layer holds a window, not a prefix: a clamp below
        // the rows it kept cannot be served, so this match is not restorable
        // and the request cold-prefills.
        log.warn("  [hot-cache] clamp to {d} declined: {s}; cold prefill\n", .{ final_len, @errorName(err) });
        try target_cache.truncate(0, s);
        rec.reset();
        target_moe_seq_offset.* = 0;
        self.last_restored_used = null;
        // Nothing was restored: promoted, this entry made a usable one the next eviction victim.
        e.last_used = used_before_restore;
        return .{ .matched = 0, .full_match = false };
    };

    const full_reuse = full_match and effective_matched > 1;
    const matched = if (full_reuse) effective_matched - 1 else effective_matched;
    if (full_reuse) {
        target_moe_seq_offset.* = matched;
        log.info("  [hot-cache] full reuse {d}/{d}, re-forwarding last token\n", .{ matched, prompt_ids.len });
    } else {
        log.info("  [hot-cache] reused {d}/{d} tokens (matched {d}; entry {d}/{d}; lost {d}{s}{s})\n", .{
            effective_matched,          prompt_ids.len, m.shared, m.idx + 1, self.entries.items.len, m.shared - effective_matched,
            if (m.shared > effective_matched) ": " else "", lostReason(e, m.shared, effective_matched),
        });
    }
    var res: LookupResult = .{
        .matched = matched,
        .full_match = full_match,
        .dflash_base = restoreDflash(e, dflash_target, matched, s),
        .mtp_base = restoreMtp(e, mtp_target, matched, s),
        .entry_id = e.id,
    };
    res.checked_out = self.checkoutIfEligible(m.idx, m.shared, slot_id);
    dump.pos = res.matched;
    dump.mtp_base = res.mtp_base;
    dump.entry_idx = m.idx + 1;
    dump.entry_count = self.entries.items.len;
    _ = restore_dump.dumpRestoreIfEnabled(target_cache, rec.ssmEntries(), s, dump);
    return res;
}

/// Prompt tokens the entries retained in RAM hold, never one a live slot has checked out (that slot's own
/// reservation already holds it). What a model's `max_kv_tokens` pool counts beside its live slots; the SSD
/// tier's records hold nothing in RAM and are not counted.
pub fn retainedTokens(self: *const HotPrefixCache) u64 {
    var tokens: u64 = 0;
    for (self.entries.items) |e| {
        if (e.checkout != null) continue;
        tokens +|= e.tokens.len;
    }
    return tokens;
}

/// The entry a request with `prompt` will restore from: the longest retained (not checked out) entry whose
/// tokens are a prefix of it. Its tokens are inside the request's own reservation, so the pool must not count
/// them a second time; it must also never be evicted for that request.
pub const RestoreMatch = struct { id: u64 = 0, tokens: u64 = 0 };

pub fn restoreEntry(self: *const HotPrefixCache, prompt: []const u32) RestoreMatch {
    var best: RestoreMatch = .{};
    for (self.entries.items) |e| {
        if (e.checkout != null or e.tokens.len <= best.tokens or e.tokens.len > prompt.len) continue;
        if (std.mem.eql(u32, e.tokens, prompt[0..e.tokens.len])) best = .{ .id = e.id, .tokens = e.tokens.len };
    }
    return best;
}

pub fn restoreEntryTokens(self: *const HotPrefixCache, prompt: []const u32) u64 {
    return self.restoreEntry(prompt).tokens;
}


/// Bytes an eviction pass can prove it will get back: the residency minus the largest
/// single entry (a restore pins at most one, and the guard cannot know which).
pub fn reclaimableBytes(self: *const HotPrefixCache) u64 {
    var largest: u64 = 0;
    // A checked-out entry is neither restorable nor evictable; it comes off the base.
    var checked_out: u64 = 0;
    for (self.entries.items) |*e| {
        if (e.checkout != null) {
            checked_out += e.kv_bytes;
            continue;
        }
        largest = @max(largest, e.kv_bytes);
    }
    return self.current_kv_bytes -| checked_out -| largest;
}

/// One resident entry, reduced to what a connection thread may know about it (it may never
/// dereference `hot_prefix_cache`). `fingerprint` hashes the first `DIGEST_PREFIX_TOKENS`
/// ids, the restore floor; a shorter entry can never be pinned and gets no digest.
pub const EntryDigest = struct {
    fingerprint: u64,
    len: u32,
    kv_bytes: u64,
};

pub const DIGEST_PREFIX_TOKENS: usize = MIN_CANCELLED_COMMIT_TOKENS;

/// FNV-1a over the first `DIGEST_PREFIX_TOKENS` ids; null under the restore floor.
pub fn prefixFingerprint(tokens: []const u32) ?u64 {
    if (tokens.len < DIGEST_PREFIX_TOKENS) return null;
    var h: u64 = 0xcbf29ce484222325;
    for (tokens[0..DIGEST_PREFIX_TOKENS]) |t| {
        h ^= t;
        h *%= 0x100000001b3;
    }
    return h;
}

/// Snapshot the resident entries for publication. Caller owns the slice.
pub fn digestsAlloc(self: *const HotPrefixCache, allocator: std.mem.Allocator) ![]EntryDigest {
    var out = std.ArrayList(EntryDigest).empty;
    errdefer out.deinit(allocator);
    for (self.entries.items) |*e| {
        if (e.checkout != null) continue;
        const fp = prefixFingerprint(e.tokens) orelse continue;
        try out.append(allocator, .{
            .fingerprint = fp,
            .len = @intCast(@min(e.tokens.len, std.math.maxInt(u32))),
            .kv_bytes = e.kv_bytes,
        });
    }
    return out.toOwnedSlice(allocator);
}

/// The connection thread's half of the rule: residency minus the largest entry this prompt
/// could restore from. The pin condition is the fingerprint match alone (a longer record
/// still restores, clamped); over-crediting is the unsafe direction.
pub fn reclaimableFromDigests(
    digests: []const EntryDigest,
    residency: u64,
    prompt_fingerprint: ?u64,
) u64 {
    const fp = prompt_fingerprint orelse return residency;
    var pinned: u64 = 0;
    for (digests) |d| {
        if (d.fingerprint == fp) pinned = @max(pinned, d.kv_bytes);
    }
    return residency -| pinned;
}

/// `reclaimableBytes` with the prompt in hand: only an entry this prompt could restore from
/// is unevictable. Conservative on both counts (key filters not applied, the LARGEST match
/// withheld), so never larger than the truth and never smaller than `reclaimableBytes()`.
pub fn reclaimableBytesFor(self: *const HotPrefixCache, prompt_tokens: []const u32) u64 {
    var pinned: u64 = 0;
    var checked_out: u64 = 0;
    for (self.entries.items) |*e| {
        if (e.checkout != null) {
            checked_out += e.kv_bytes;
            continue;
        }
        const shared = prefix_policy.commonPrefixLen(e.tokens, prompt_tokens, @min(e.tokens.len, prompt_tokens.len));
        if (shared < MIN_CANCELLED_COMMIT_TOKENS) continue;
        pinned = @max(pinned, e.kv_bytes);
    }
    return self.current_kv_bytes -| checked_out -| pinned;
}
