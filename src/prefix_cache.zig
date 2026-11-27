//! Plan 03 — hot prefix cache (Phase 1).
//!
//! Replaces the legacy single-slot `cached_prompt_ids` with a small bounded
//! LRU keyed by `(prompt_ids ++ generated_ids, has_tools)`. Each entry owns a
//! `KVCacheSnapshot` of the live cache at the moment the request finished —
//! refcount-shared handles point at the GPU buffers that filled positions
//! 0..len. On a new request we longest-prefix match across entries, restore
//! the best one back into `xfm.cache`, and let the existing
//! truncate-then-prefill path handle the diverged tail.
//!
//! Hybrid SSM/conv architectures (GatedDeltaNet) are excluded in v1: their recurrent state can't be rolled back, so
//! prefix reuse must reset on any divergence anyway. Plan 03's spec calls
//! these "hot tier only" — meaning we keep the single-slot path for them.
//! `HotPrefixCache.shouldUse(config)` returns false for those archs.


const std = @import("std");
const mlx = @import("mlx.zig");
const trunk_mod = @import("trunk.zig");
const model_mod = @import("model.zig");
const kv_quant = @import("kv_quant.zig");
const kv_disk_cache = @import("kv_disk_cache.zig");
const log = @import("log.zig");
const prefix_arch = @import("arch/prefix.zig");
const prefix_policy = @import("prefix_policy.zig");
const cache_stats = @import("cache_stats.zig");
const KVCache = trunk_mod.KVCache;
const KVCacheSnapshot = trunk_mod.KVCacheSnapshot;
const SSMCacheEntry = trunk_mod.SSMCacheEntry;
const SSMCheckpoint = trunk_mod.SSMCheckpoint;
const prefix_cache_index = @import("prefix_cache_index.zig");
const prefix_cache_evict = @import("prefix_cache_evict.zig");
const prefix_cache_commit = @import("prefix_cache_commit.zig");
const prefix_cache_helpers_test = @import("prefix_cache_helpers_test.zig");

pub const MIN_CANCELLED_COMMIT_TOKENS: usize = 256;

/// Shortest prefix an idle entry is trimmed down to for room. A restore under it saves about a
/// quarter of a second of prefill, and the remnant would still hold an entry slot.
pub const MIN_TRIMMED_VICTIM_TOKENS: usize = 512;

/// Shortest prefix a ring-checkpoint restore reuses; below it the request cold-prefills.
pub const RING_RESTORE_MIN_TOKENS = prefix_policy.RING_RESTORE_MIN_TOKENS;

/// Ring checkpoints one entry keeps, thinned so the lowest (a shared preamble) and the newest
/// (where the conversation goes on) stay (`thinRingCps`). One is ~16 MiB at kv8 on the ringed pack served.
pub const RING_CHECKPOINT_MAX: usize = 8;

/// Ring checkpoints a hot entry keeps in RAM when the SSD tier holds the older ones.
pub const RING_CHECKPOINT_RAM_KEEP: usize = 2;

/// Message starts one prefill takes a ring checkpoint at (`ringMarkPositions`).
pub const RING_MARKS_MAX = prefix_policy.RING_MARKS_MAX;

/// The special token that opens a message in the served chat templates: where the marks sit.
pub const RING_MARK_TOKEN = "<|im_start|>";

/// Ring checkpoints one slot holds at once: its restore, its marks and its prompt end
/// (`server.slotRingBytes` bills them).
pub const SLOT_RING_CHECKPOINTS: usize = 2 + RING_MARKS_MAX;

comptime {
    std.debug.assert(SLOT_RING_CHECKPOINTS <= RING_CHECKPOINT_MAX);
}

/// Why a lookup that found a real raw token match still restored nothing.
/// `findBestRestorableMatch` `continue`s every candidate whose highest SSM
/// checkpoint sits past the shared prefix, so a hybrid lookup can return null
/// with a 393k-token raw match behind it — and the `match == null` arm used to
/// log NOTHING, leaving a 560 s cold prefill with no `[hot-cache]` line at all.
/// Pure so the policy is unit-testable without a cache.
pub const MissKind = enum {
    /// Nothing worth naming: no key-compatible entry, or a shared prefix under
    /// the commit floor. An ordinary cold start; stays quiet.
    cold,
    /// Entries shared a real prefix and not one of them could restore it.
    /// This is the expensive miss and it owes a line.
    no_checkpoint,
};

/// What `findBestRestorableMatch` saw before its restorability filter ran:
/// how many key-compatible entries it considered and the longest RAW token
/// match among them. The filter's `continue`s destroy both, which is why a
/// hybrid miss could not name itself.
pub const MatchProbe = struct { candidates: usize = 0, best_raw: usize = 0 };

pub fn missKind(candidates: usize, best_raw: usize) MissKind {
    if (candidates == 0) return .cold;
    if (best_raw < MIN_CANCELLED_COMMIT_TOKENS) return .cold;
    return .no_checkpoint;
}

/// Why an oversized commit retained nothing; three outcomes used to print one identical line.
pub const TrimDecline = enum {
    no_restorable_prefix,
    snapshot_copy_failed,
    checkpoint_list_copy_failed,
    qsa_history_slice_failed,

    pub fn reason(self: TrimDecline) []const u8 {
        return switch (self) {
            .no_restorable_prefix => "no restorable prefix fits the budget",
            .snapshot_copy_failed => "every trimmed KV copy failed",
            .checkpoint_list_copy_failed => "the trimmed checkpoint list copy failed",
            .qsa_history_slice_failed => "the QSA indexer history could not be carried onto the kept checkpoint",
        };
    }
};

/// Outcome of a cache commit, so callers can log the truth. A budget
/// decline used to exit `commitWithMediaState` via plain return and the
/// scheduler logged "committed N/M" for it (live 2026-09-07: a 122k agent
/// session cold-prefilled ~95k tokens per retry while the log claimed
/// cancelled-prefill commits every time).
pub const CommitStatus = union(enum) {
    /// Entry committed (inserted or replaced) at this many tokens — the
    /// post-trim EFFECTIVE length, never the candidate's forwarded length.
    ok: usize,
    /// Full state captured for the SSD tier; no idle RAM entry retained.
    disk_only: usize,
    /// The budget decline kept a resident entry that already covers this
    /// many tokens; the longer candidate was discarded (details logged).
    kept_resident: usize,
    /// Declined: oversized with no restorable trim under the budget, or the
    /// trim copy failed. The decline site logs the reason.
    declined,
};

/// Result of a cache lookup. Tells the caller how many tokens of `prompt_ids`
/// are already in the live cache after a successful restore — the caller
/// then prefills only the trailing diverged tokens (`prompt_ids[matched..]`).
pub const LookupResult = struct {
    /// Tokens already in the live cache. Caller prefills `prompt_ids[matched..]`.
    matched: usize,
    /// Did the restore land on an entry whose tokens span the FULL new prompt?
    /// Then identical-re-issue logic kicks in (truncate to len-1 and re-forward
    /// the last token), matching the existing reuseKVCache behavior.
    full_match: bool,
    /// Non-null iff a DFlash assistant context was restored into the caller's
    /// target: the absolute trunk position its index 0 represents, with
    /// `base + cache.step == matched` on return. Both tiers serve it (the
    /// SSD tier persists the snapshot in the v4 spec sidecar). Null on EVERY
    /// other path (no target, no payload, miss) and the target is untouched —
    /// the caller then starts the assistant blind at `matched`.
    dflash_base: ?usize = null,
    /// Same contract for the MTP head's committed-history cache: the head's
    /// history is built from trunk hiddens, and a restore forwards NOTHING —
    /// without this every reused prefix drafts against an empty history
    /// (measured on echo: ~70 → ~38 tok/s on warm repeats).
    mtp_base: ?usize = null,
    /// Did this restore check out its entry (`upFrontTakeEligible`)? Only then does the first
    /// append donate in place; every other restore is a refcount share copied by that append.
    checked_out: bool = false,
    /// Did an SSD restore fill buffers the slot owns outright (`restoreKvInto`)?
    slot_owned: bool = false,
    /// `Entry.id` of the RAM entry restored from; 0 = none.
    entry_id: u64 = 0,
    /// The SSD entry and rows a disk restore read; null for every other source.
    disk_restore: ?kv_disk_cache.RestoredRows = null,

    /// Are the restored rows the slot's own? Only those the admission bill credits (`WarmPrefix.will_donate`).
    pub fn ownsRestoredRows(self: LookupResult) bool {
        return self.checked_out or self.slot_owned;
    }
};

/// A ringed slot's restore points (`KVCache.ringCheckpoint`), handed to the commit that owns
/// them from then on: where the request restored to, taken before its tail was forwarded (the
/// fork off another entry, which may be evicted before this commit), the message starts its
/// prefill crossed (`marks`, filled by the forward through `KVCache.ring_marks`), and its prompt end.
pub const SlotRingCps = struct {
    fork: ?KVCacheSnapshot = null,
    prompt_end: ?KVCacheSnapshot = null,
    marks: [RING_MARKS_MAX]KVCacheSnapshot = undefined,
    n_marks: usize = 0,

    pub fn deinit(self: *SlotRingCps) void {
        if (self.fork) |*r| r.deinit();
        if (self.prompt_end) |*r| r.deinit();
        self.dropMarks();
        self.* = .{};
    }

    fn dropMarks(self: *SlotRingCps) void {
        for (self.marks[0..self.n_marks]) |*m| m.deinit();
        self.n_marks = 0;
    }

    /// Empty marks at `positions` for the prefill to fill, replacing any left from an earlier
    /// attempt. Best effort: an allocation failure keeps the ones made so far.
    pub fn armMarks(self: *SlotRingCps, cache: *const KVCache, positions: []const usize) void {
        self.dropMarks();
        for (positions[0..@min(positions.len, RING_MARKS_MAX)]) |p| {
            self.marks[self.n_marks] = cache.ringMark(p) catch return;
            self.n_marks += 1;
        }
    }

    pub fn markSlice(self: *SlotRingCps) []KVCacheSnapshot {
        return self.marks[0..self.n_marks];
    }

    /// Free the marks the prefill did not fill on every ringed layer; only a whole one restores.
    pub fn keepCompleteMarks(self: *SlotRingCps, cache: *const KVCache) void {
        var kept: usize = 0;
        for (self.marks[0..self.n_marks]) |*m| {
            if (cache.ringMarkComplete(m)) {
                self.marks[kept] = m.*;
                kept += 1;
            } else m.deinit();
        }
        self.n_marks = kept;
    }

    pub fn bytes(self: *const SlotRingCps) u64 {
        var total: u64 = 0;
        if (self.fork) |*c| total += trunk_mod.kvEntriesBytes(c.entries);
        if (self.prompt_end) |*c| total += trunk_mod.kvEntriesBytes(c.entries);
        for (self.marks[0..self.n_marks]) |*c| total += trunk_mod.kvEntriesBytes(c.entries);
        return total;
    }
};

/// `prefix_policy.ringMarkPositions` with the model's prompt-end backoff.
pub fn ringMarkPositions(prompt: []const u32, restored: usize, mark_id: u32, out: *[RING_MARKS_MAX]usize) []usize {
    return prefix_policy.ringMarkPositions(prompt, restored, mark_id, model_mod.ModelConfig.SWA_RING_CHECKPOINT_BACKOFF, out);
}

/// Re-exported from `prefix_policy.zig` so the SSD tier shares the exact same media rules.
pub const MediaSpan = prefix_policy.MediaSpan;
pub const chainKey = prefix_policy.chainKey;
pub const crossKeyBoundary = prefix_policy.crossKeyBoundary;

/// A request's media identity, as the rule the SSD tier applies to each stored record. Both tiers
/// share one rule (`crossKeyBoundary`): an equal key shares the whole token prefix, a record whose
/// chain is this request's first k blocks shares up to block k+1, and anything else shares only the
/// text before either side's first media row.
pub const MediaRule = struct {
    vision_key: u64,
    media_start: ?usize,
    chain: []const MediaSpan,

    /// Always the rule, never an unconstrained shortcut: a TEXT request must be held to a media
    /// record's pre-media text too. Placeholder ids are ordinary vocabulary entries, so a text
    /// prompt can match a record's tokens past its first media row, and the rows there are the
    /// record's pixels, not text. Borrows `self`, which the caller keeps alive across the lookup.
    pub fn limits(self: *const MediaRule) kv_disk_cache.MediaLimits {
        return .{ .rule = .{ .ctx = self, .limitFor = limitFor } };
    }

    fn limitFor(ctx: *const anyopaque, entry_key: u64, entry_start: ?u32) ?u32 {
        const self: *const MediaRule = @ptrCast(@alignCast(ctx));
        if (entry_key == self.vision_key) return kv_disk_cache.MediaLimits.UNBOUNDED;
        const es: ?usize = if (entry_start) |v| @as(usize, v) else null;
        const boundary = crossKeyBoundary(entry_key, es, self.media_start, self.chain) orelse return null;
        return std.math.cast(u32, boundary) orelse kv_disk_cache.MediaLimits.UNBOUNDED;
    }
};

/// Rows of a record the SSD tier may hold. A record carries its media identity (manifest v10), so
/// an image-bearing turn persists whole and a hybrid restores at its checkpoints as a text one
/// does. A pixel-keyed entry with NO boundary is the one exception: nothing says where its media
/// rows begin, so nothing about it can be proven.
pub fn diskTextLen(tokens_len: usize, vision_key: u64, media_start: ?usize) usize {
    if (vision_key != 0 and media_start == null) return 0;
    return tokens_len;
}

pub const MediaStamp = prefix_policy.MediaStamp;
pub const truncatedMediaKey = prefix_policy.truncatedMediaKey;

pub const Entry = struct {
    /// `prompt_ids ++ generated_ids` from the request that produced this snapshot.
    /// Owned by the entry; freed on eviction.
    tokens: []u32,
    /// Whether the request had tools enabled (different chat template, can't
    /// share cache across).
    has_tools: bool,
    /// Hash of the request's media pixels (0 = text only). Image placeholder
    /// tokens are identical across images, so the KV under them is keyed on
    /// the pixels. The SSD tier holds only the text before the first item of a non-zero entry.
    vision_key: u64 = 0,
    /// Position of the first dynamic media placeholder in `tokens`. Prefix
    /// state strictly before this boundary is independent of the media pixels
    /// and can be shared across different `vision_key` values.
    media_start: ?usize = null,
    /// Workload the request belonged to (`server.requestCacheKey`, 0 = anonymous).
    /// Eviction is fair across keys: the key holding the most entries pays first.
    cache_key: u64 = 0,
    /// Stable identity for observers; `last_used` moves on every touch.
    id: u64 = 0,
    /// Snapshot of the live KVCache at end of generation. Owns refcount-shared
    /// handles to the GPU buffers backing positions 0..tokens.len.
    snapshot: KVCacheSnapshot,
    /// Monotonic counter for LRU. Higher = more recent.
    last_used: u64,
    /// Wave 1.A: full KV-quant config active when this entry was committed.
    /// A new request whose `KVQuantConfig` differs in any field cannot
    /// restore from this entry — the underlying buffer layout (dense bf16 vs
    /// packed uint32 triples; 4-bit vs 8-bit packing) differs, and
    /// dequantization would have happened at commit time anyway. Filter at
    /// lookup so per-request `kv_quant` overrides never produce a hit
    /// against an entry that was committed under another config.
    ///
    /// Storing the full `KVQuantConfig` (not just `Scheme`) is what
    /// distinguishes `affine 4` from `affine 8` and a `group_size` change —
    /// without that, a 4-bit entry would alias to an
    /// 8-bit slot's findBestMatch lookup and crash SDPA with a packed-shape
    /// mismatch on restore. Repro: `tests/test_kv_quant_per_request.sh`.
    quant_config: kv_quant.KVQuantConfig,
    /// Transient, one `spillIdleEntries` pass only: did this pass leave a durable copy on the SSD tier?
    spill_durable: bool = false,
    /// Why the request that committed this entry left no restore point at its end (`mtp_plan.EndPoint.label`),
    /// "taken" when it did, empty when it took none (a prefill, another arch). Named by a restore that loses tokens.
    end_note: []const u8 = "",
    /// KV-resident bytes for this entry, computed at commit time (Wave 1.B).
    /// Used for `--prefix-cache-mem` memory-budget enforcement; sum across
    /// all entries == `current_kv_bytes`.
    kv_bytes: u64,
    /// Phase 1 (perf-plan): SSM/conv state snapshots taken at stride-aligned
    /// positions during prefill. Sorted by `pos` ascending; the highest `pos`
    /// is at most `tokens.len`. Null for plain-attention archs. The hot
    /// cache restore picks the largest `pos ≤ matched` and rewinds both KV
    /// and SSM to it.
    ssm_checkpoints: ?[]SSMCheckpoint = null,
    /// Bytes resident in `ssm_checkpoints` (sum across all checkpoints and
    /// layers). Folded into `kv_bytes` for the byte-budget accounting so the
    /// memory cap covers both KV and SSM state.
    ssm_bytes: u64 = 0,
    /// DFlash assistant context for this prefix (dflash.zig). The assistant's
    /// K/V is built from trunk hiddens at `target_layer_ids`, and a restore
    /// forwards NOTHING — so without this the assistant starts every reused
    /// turn blind and drafts against an empty context. Optional in both
    /// directions: an entry committed by a non-dflash request has none, and a
    /// request that finds none simply starts blind (the state is DRAFT-side —
    /// a missing or stale context costs acceptance, never a token).
    dflash: ?DflashSnap = null,
    /// Bytes resident in `dflash`, folded into `kv_bytes` like `ssm_bytes`.
    dflash_bytes: u64 = 0,
    /// MTP committed-history cache for this prefix (the head's own dense
    /// KVCache, mtp.zig). Same lifecycle and contract as `dflash`: DRAFT-side
    /// state, best-effort in both directions — a missing or declined snap
    /// starts the history blind, which costs acceptance, never a token. The
    /// committer must snapshot ONLY committed history (no speculative draft
    /// tail — Generator.mtpCommittedHistoryLen is the boundary).
    mtp: ?DflashSnap = null,
    /// Bytes resident in `mtp`, folded into `kv_bytes` like `ssm_bytes`.
    mtp_bytes: u64 = 0,
    /// The arch's rows every checkpoint of `ssm_checkpoints` restores from (an arch whose KV lives in them),
    /// billed in `kv_bytes`.
    rows: ?prefix_arch.Rows = null,
    rows_bytes: u64 = 0,
    /// Ringed caches only: restore points below the entry's END (`KVCache.ringCheckpoint`),
    /// ascending by `step`, billed in `kv_bytes`: the prompt end, which a long reply carries the
    /// ring past, and those inherited from the entry this prompt forked off (`bestRingDonor`).
    ring_cps: ?[]KVCacheSnapshot = null,
    /// Restore by move: the slot that took this entry's buffers. While set the entry is invisible to every other reader.
    checkout: ?Checkout = null,
};

/// Restore by move, one slot's take of an entry's KV buffers (`KVCache.adopt`). Begun only by a checkout call
/// (`beginCheckout`), advanced only by `donateCheckout`, and ended only by `endCheckout` (the slot's release, the
/// commit that replaces the entry) or the entry's eviction: lint `entry-checkout` refuses any other writer.
pub const Checkout = struct {
    slot: usize,
    kind: Kind,
    /// Snapshot bytes the slot's append took in place, which `kv_bytes` bills until the release; null until it
    /// donates, and while null the entry's handles are live and a slot that ends hands the entry back intact.
    donated: ?u64 = null,

    pub const Kind = union(enum) {
        /// A full-prefix hit: the turn's commit replaces the entry.
        whole,
        /// A partial share: the tokens the prompt shares with the entry, all the donation keeps, so the turn's commit
        /// replaces the entry as it replaces a whole one.
        partial: usize,
    };
};

/// Where a hybrid restore puts its recurrent state: the slot's SSM entries, or the state an arch owns.
pub const Recurrent = union(enum) {
    none,
    ssm: []SSMCacheEntry,
    arch: prefix_arch.State,

    pub fn hybrid(self: Recurrent) bool {
        return self != .none;
    }
    pub fn ssmEntries(self: Recurrent) ?[]SSMCacheEntry {
        return if (self == .ssm) self.ssm else null;
    }
    pub fn reset(self: Recurrent) void {
        switch (self) {
            .none => {},
            .ssm => |entries| HotPrefixCache.resetSsmEntries(entries),
            .arch => |state| prefix_arch.reset(state),
        }
    }
};

/// The RAM entry a lookup matched: its index and the tokens it shares with the prompt.
pub const RamMatch = struct { idx: usize, shared: usize };

/// The rows a commit hands each tier: the RAM tier's within its budget, the SSD tier's through the newest checkpoint.
pub const TierRows = struct {
    ram: ?prefix_arch.Rows = null,
    disk: ?prefix_arch.Rows = null,
};

/// KDA-class checkpoints the RAM tier's hot entry keeps (the newest) when an SSD tier holds the rest.
pub const CHECKPOINT_HOT_KEPT: u32 = 2;

pub const SpecAdopt = prefix_policy.SpecAdopt;
pub const specAdoptPlan = prefix_policy.specAdoptPlan;

/// Test seam: forces SSD-first mode off (or on); null = on, armed only where
/// `ModelConfig.ssdFirstCapable()`.
pub var ssd_first_override: ?bool = null;

pub fn ssdFirstEnabled() bool {
    return ssd_first_override orelse true;
}

/// Test seam: forces the refcount share (false) whose first append copies the whole prefix;
/// null = restore by move, armed where `HotPrefixCache.ssd_first` is, or on demand when a share
/// does not fit (`checkoutRestored`).
pub var restore_move_override: ?bool = null;

pub fn restoreMoveEnabled() bool {
    return restore_move_override orelse true;
}

/// The SSD-first predicate: arch, the mode switch, AND a disk tier. Without the tier the mode used
/// to arm with nowhere to spill and a budget floor sized for a tier that did not exist. The
/// budget resolver asks `--prefix-cache-disk > 0`, the arming asks `disk != null`.
pub fn ssdFirstActive(config: *const model_mod.ModelConfig, has_disk: bool, ram_enabled: bool) bool {
    return has_disk and ssdFirstEnabled() and (config.ssdFirstCapable() or !ram_enabled);
}

/// What the live cache held at commit time, captured before the RAM byte-budget trim: the
/// disk tier gets the full prefix even when RAM keeps a trimmed one. Refcount-shared.
pub const PendingDiskFlush = struct {
    snapshot: KVCacheSnapshot,
    tokens: []u32,
    has_tools: bool,
    media: MediaStamp = .{},
    media_chain: []MediaSpan = &.{},
    ssm_cps: ?[]SSMCheckpoint = null,
    ring_cps: ?[]KVCacheSnapshot = null,
    dflash: ?DflashSnap = null,
    mtp: ?DflashSnap = null,
    rows: ?prefix_arch.Rows = null,
    restored: ?kv_disk_cache.RestoredRows = null,

    pub fn deinit(self: *PendingDiskFlush, allocator: std.mem.Allocator) void {
        self.snapshot.deinit();
        if (self.rows) |*rows| rows.deinit();
        allocator.free(self.tokens);
        allocator.free(self.media_chain);
        if (self.ssm_cps) |cps| {
            for (cps) |*cp| cp.deinit(allocator);
            allocator.free(cps);
        }
        if (self.ring_cps) |cps| HotPrefixCache.freeRingCps(allocator, cps);
        if (self.dflash) |*d| d.deinit();
        if (self.mtp) |*m| m.deinit();
    }
};

/// A committed speculative-side cache: the snapshot plus the absolute trunk
/// position its index 0 represents (nonzero when the committing request was
/// itself a cache hit). Shared by the DFlash assistant context and the MTP
/// committed-history cache — identical semantics, two Entry fields.
pub const DflashSnap = struct {
    snapshot: KVCacheSnapshot,
    base_pos: usize,
    /// In-checkpoint MTP head only: the head's QSA aux entry and the absolute position of its key
    /// row 0. A snap with one half and not the other is declined.
    head_aux: ?trunk_mod.SSMCacheEntrySnapshot = null,
    head_pos_base: c_int = 0,
    /// In-checkpoint MTP head only: the head's QSA leftovers at the trunk's checkpoint positions.
    /// A clamp to a restored trunk position lands far below the head's raw-key ring; this is
    /// what makes it exact there.
    head_marks: trunk_mod.QsaHeadMarkSet = .{},

    /// The MTP snapshot of a commit, over a snapshot of its cache.
    pub fn ofCommit(c: DflashCommit, snapshot: KVCacheSnapshot, s: mlx.mlx_stream) DflashSnap {
        return .{
            .snapshot = snapshot,
            .base_pos = c.base_pos,
            .head_aux = if (c.head) |h| trunk_mod.ssmSnapshot(h, s) else null,
            .head_pos_base = c.head_pos_base,
            .head_marks = trunk_mod.QsaHeadMarkSet.share(c.head_marks),
        };
    }

    pub fn deinit(self: *DflashSnap) void {
        self.snapshot.deinit();
        if (self.head_aux) |*a| trunk_mod.ssmSnapshotDeinit(a);
        self.head_aux = null;
        self.head_marks.deinit();
    }
};

/// What `commitWithState` reads to build a `DflashSnap`. `head` is set only by an in-checkpoint MTP head.
pub const DflashCommit = struct {
    cache: *const KVCache,
    base_pos: usize,
    head: ?*const SSMCacheEntry = null,
    head_pos_base: c_int = 0,
    head_marks: []const trunk_mod.QsaHeadMark = &.{},
};

/// Where `lookupAndRestore` puts a restored assistant context. `base_pos` is
/// written on every path so the caller can build `DflashCtx` from it. `head` is the Trunk
/// owning an in-checkpoint MTP head; the adoption then goes through the arch (`prefix_arch.adoptSpecHead`).
pub const DflashTarget = struct {
    cache: *KVCache,
    base_pos: *usize,
    head: ?*trunk_mod.Trunk = null,
};

/// What `evictLruToAdmit` gave up, and whether it was enough.
pub const EvictionReport = struct {
    entries: usize = 0,
    /// Bytes the allocator actually got back (live delta).
    bytes: u64 = 0,
    /// Bytes the cache had billed for those entries; larger whenever a snapshot is refcount-shared.
    accounted_bytes: u64 = 0,
    /// The pass stopped because an eviction returned nothing (shared with a live request).
    shared_stop: bool = false,
    /// False = the cache is empty (or down to this request's entry) and the request still does not fit.
    admitted: bool = false,
};

/// One model's hot cache as a connection thread may know it, published by the inference thread
/// (`scheduler.publishHotCacheResidency`): never the cache, which only the inference thread touches. Each model has its
/// own (`LoadedModel.hot_gauge`), so an admission bill reads the requested model's and no other.
pub const HotCacheGauge = struct {
    resident: std.atomic.Value(u64) = .init(0),
    /// The part of `resident` an eviction can prove it will return (residency minus the largest entry).
    reclaimable: std.atomic.Value(u64) = .init(0),
    mu: std.Io.Mutex = .init,
    digests: []HotPrefixCache.EntryDigest = &.{},
    /// The residency the digests describe, published under the same lock.
    digest_residency: u64 = 0,

    /// Publishes `hc` (null: the model holds none). An allocation failure keeps the previous digests: a stale digest is
    /// a hint, an empty one credits bytes that exist.
    pub fn publish(self: *HotCacheGauge, allocator: std.mem.Allocator, io: std.Io, hc: ?*const HotPrefixCache) void {
        const residency: u64 = if (hc) |c| c.residentBytes() else 0;
        self.resident.store(residency, .monotonic);
        self.reclaimable.store(if (hc) |c| c.reclaimableBytes() else 0, .monotonic);
        const fresh: []HotPrefixCache.EntryDigest = if (hc) |c| (c.digestsAlloc(allocator) catch return) else &.{};
        self.mu.lockUncancelable(io);
        const old = self.digests;
        self.digests = fresh;
        self.digest_residency = residency;
        self.mu.unlock(io);
        if (old.len > 0) allocator.free(old);
    }

    /// What an eviction pass can prove it will get back for this prompt.
    pub fn reclaimableFor(self: *HotCacheGauge, io: std.Io, prompt_tokens: []const u32) u64 {
        const fp = HotPrefixCache.prefixFingerprint(prompt_tokens);
        self.mu.lockUncancelable(io);
        defer self.mu.unlock(io);
        return HotPrefixCache.reclaimableFromDigests(self.digests, self.digest_residency, fp);
    }

    /// Back to nothing published, for a model nobody holds (its unload, its teardown).
    pub fn reset(self: *HotCacheGauge, allocator: std.mem.Allocator) void {
        if (self.digests.len > 0) allocator.free(self.digests);
        self.* = .{};
    }
};

/// A count of this keeps every entry the byte budget and admission allow; a served cache's count is its model's `max_concurrent`.
pub const unlimited_entries: u32 = std.math.maxInt(u32);

pub const HotPrefixCache = struct {
    entries: std.ArrayList(Entry),
    /// Conversations kept, oldest evicted past it; `unlimited_entries` keeps all. A served cache holds its model's
    /// `max_concurrent`, on top of `max_kv_bytes` and admission's reclaim.
    max_entries: u32,
    /// Wave 1.B: total KV bytes the cache is allowed to keep resident across
    /// all entries. 0 disables the byte budget (count cap still applies).
    /// Enforced on `commit`: evict LRU entries (in addition to the count
    /// cap) until `current_kv_bytes + new_entry_bytes <= max_kv_bytes`.
    max_kv_bytes: u64,
    /// Cap on SSM checkpoints kept per entry, mirroring `generate.zig`'s
    /// `ssm_checkpoint_max`. That one bounds a SINGLE prefill; this one bounds
    /// the replace path's merge, which concatenates the previous entry's
    /// checkpoints with this turn's. Without it an entry extended in place —
    /// every turn of an agent conversation — gains one checkpoint per turn for
    /// the life of the session. 0 = unlimited.
    ssm_checkpoint_max: u32 = 0,
    /// The next commit's `Entry.end_note`, set by the committer right before it; the commit consumes it.
    commit_end_note: []const u8 = "",
    /// The next commit's SSD restore source (`LookupResult.disk_restore`); the commit consumes it.
    commit_restored: ?kv_disk_cache.RestoredRows = null,
    /// The next commit's media chain (`MediaSpan`), borrowed from the request's slot. A consumer takes it
    /// (`takeMediaChain`) before any return, so it never outlives the request that set it.
    commit_media_chain: []const MediaSpan = &.{},
    /// Running total of `kv_bytes` across all live entries. Updated on
    /// commit/evict/invalidate.
    current_kv_bytes: u64,
    allocator: std.mem.Allocator,
    counter: u64 = 0,
    /// Set to true once we've called `xfm.resetCache()` at least once after
    /// init. The first commit on a fresh cache must seed an empty entry so
    /// future restores have something to land on.
    initialized: bool = false,
    /// SSD tier (kv_disk_cache.zig). Attached by the scheduler at model load
    /// when `--prefix-cache-disk` is non-zero and the arch is pure-attention.
    /// Lookup falls back to it when it beats the RAM match; commits mark
    /// `disk_dirty` and the scheduler flushes AFTER the response finishes so
    /// the client never waits on the SSD write.
    disk: ?kv_disk_cache.DiskTier = null,
    /// Where this cache's arrays are copied, trimmed, written and restored when the model's state is not MLX's: the
    /// stream the model's arch names as the cache is armed, which then outranks a caller's. Null is the caller's
    /// stream, or MLX's GPU stream where no caller gives one.
    arrays_on: ?mlx.mlx_stream = null,
    /// A commit landed since the last `flushPendingDisk`.
    disk_dirty: bool = false,
    /// `last_used` of the entry the current request restored from; `evictLruToAdmit` refuses to evict it.
    last_restored_used: ?u64 = null,
    /// The raw prefix that restore matched, before a hybrid clamp (`checkoutRestored`).
    last_restored_shared: usize = 0,
    last_restored_disk_id: ?u64 = null,
    /// The last SSD-tier restore came from a restore point the tier held in RAM (`DiskTier.held`), not a file.
    last_restored_held: bool = false,
    /// The last lookup rolled the model's live state back instead of restoring a stored point.
    last_restored_live: bool = false,
    /// What each arch keeps beside the entries (the points of a live state it can be rolled back to).
    arch: prefix_arch.CacheState = .{},
    /// The arch keeps a QSA indexer history beside its SSM state.
    /// A restore that leaves the live entries without it cannot prefill —
    /// `qsaMaskFromQk` errors on every turn on that prefix — so it is a MISS.
    qsa_history_required: bool = false,
    /// The arch carries recurrent state, so an entry restores only from a checkpoint. Set at
    /// load: an empty checkpoint list arrives as null, so the list cannot say.
    hybrid: bool = false,
    /// Checkpoint-retention policy, mirrored once at wiring from `ModelConfig.longCtxGated()`
    /// (this struct never sees a ModelConfig). The default is the previous behaviour.
    cp_thin: trunk_mod.ThinPolicy = .min_span,
    /// Whether completed requests retain reusable KV in RAM. The live slot still owns its
    /// working KV; false keeps reusable prefixes only on the SSD tier.
    ram_enabled: bool = true,
    /// SSD-first mode; set by the scheduler at load.
    ssd_first: bool = false,
    /// SSD-first: the RAM allowance for idle entries (the resolved `--prefix-cache-mem`).
    /// `spillIdleEntries` evicts only past it; 0 = nothing idle stays resident.
    ssd_idle_mem: u64 = 0,
    /// The live-cache state of the most recent commit, flushed instead of the (possibly trimmed) RAM entry.
    pending_disk: ?PendingDiskFlush = null,
    /// Byte floor for `restoreWouldPinEntry`; a field so a test can reproduce the live shape.
    restore_pin_min_bytes: u64 = RESTORE_PIN_MIN_BYTES,

    pub fn init(allocator: std.mem.Allocator, max_entries: u32) HotPrefixCache {
        return initWithMem(allocator, max_entries, 0);
    }

    /// The chain the committer set for this call, which the cache no longer holds.
    pub fn takeMediaChain(self: *HotPrefixCache) []const MediaSpan {
        const chain = self.commit_media_chain;
        self.commit_media_chain = &.{};
        return chain;
    }

    pub fn initWithMem(allocator: std.mem.Allocator, max_entries: u32, max_kv_bytes: u64) HotPrefixCache {
        return .{
            .entries = std.ArrayList(Entry).empty,
            .max_entries = max_entries,
            .ram_enabled = max_entries > 0,
            .max_kv_bytes = max_kv_bytes,
            .current_kv_bytes = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *HotPrefixCache) void {
        for (self.entries.items) |*e| {
            freeEntryOwnedState(self.allocator, e);
        }
        self.entries.deinit(self.allocator);
        if (self.pending_disk) |*p| p.deinit(self.allocator);
        self.pending_disk = null;
        prefix_arch.deinitCacheState(&self.arch, self.allocator);
        if (self.disk) |*d| d.deinit();
        self.disk = null;
    }

    /// Free everything an Entry owns: token buffer, KV snapshot, SSM
    /// checkpoint array. Used by `deinit`, eviction, and replace paths so
    /// they don't drift apart.
    pub fn freeEntryOwnedState(allocator: std.mem.Allocator, e: *Entry) void {
        allocator.free(e.tokens);
        e.snapshot.deinit();
        if (e.rows) |*rows| rows.deinit();
        e.rows = null;
        if (e.ssm_checkpoints) |cps| {
            for (cps) |*cp| cp.deinit(allocator);
            allocator.free(cps);
            e.ssm_checkpoints = null;
        }
        if (e.dflash) |*d| {
            d.deinit();
            e.dflash = null;
        }
        if (e.mtp) |*m| {
            m.deinit();
            e.mtp = null;
        }
        if (e.ring_cps) |r| {
            HotPrefixCache.freeRingCps(allocator, r);
            e.ring_cps = null;
        }
    }


    /// Pure-attention archs are eligible by default. Hybrid recurrent-state
    /// archs are gated by `enable_ssm_checkpoints` (set by the scheduler
    /// when `--ssm-checkpoint-stride > 0`): with checkpoints we can rewind
    /// both KV and SSM state to a stride-aligned position; without them
    /// every divergence would force a full reset, so we keep the legacy
    /// single-slot path.
    ///
    /// `full_attention_interval > 0` marks "every Nth layer is full attention,
    /// the rest are GatedDeltaNet": the model has SSM layers, so the
    /// SSM-checkpoint gate applies.
    pub fn shouldUse(
        config: *const model_mod.ModelConfig,
        enable_ssm_checkpoints: bool,
    ) bool {
        // An arch whose request state is a checkpoint restores nothing without them.
        if (prefix_arch.checkpointsRequired(config)) return enable_ssm_checkpoints;
        const has_ssm_layers = config.full_attention_interval > 0;
        if (has_ssm_layers and !enable_ssm_checkpoints) return false;
        return true;
    }

    pub fn bumpCounter(self: *HotPrefixCache) u64 {
        self.counter += 1;
        return self.counter;
    }


    /// The stream the cache's arrays live on: the engine's on Sushi Cooler, MLX's otherwise.
    pub fn arraysStream(self: *const HotPrefixCache) mlx.mlx_stream {
        return self.arrays_on orelse mlx.gpuStream();
    }

    /// The tier the last lookup's restored rows came from: memory (the RAM tier or the live state) unless the disk supplied them.
    pub fn lastRestoredTier(self: *const HotPrefixCache) cache_stats.Tier {
        return if (!self.last_restored_live and self.last_restored_disk_id != null and !self.last_restored_held) .ssd else .hot;
    }

    /// Does a commit record the live state for the SSD tier (`pending_disk`)? Wherever the flush
    /// lands it whole: SSD-first, or a tier with the background writer.
    pub fn capturesLiveForDisk(self: *const HotPrefixCache) bool {
        const d = if (self.disk) |*dd| dd else return false;
        return self.ssd_first or d.writer != null;
    }

    /// Bytes the cache currently holds resident. A hint for the connection thread's admission
    /// guard; the decision that matters is made on the inference thread by `evictLruToAdmit`.
    /// Host bytes this cache's SSD writer holds for files not yet written (up to the permit,
    /// ~1 GiB), peaking at a long prefill's chunk boundary. Inference thread only.
    pub fn stagedHostBytes(self: *HotPrefixCache) u64 {
        const d = if (self.disk) |*dd| dd else return 0;
        return d.stagedHostBytes();
    }

    pub fn residentBytes(self: *const HotPrefixCache) u64 {
        return self.current_kv_bytes;
    }

    pub fn logResident(self: *const HotPrefixCache) void {
        const mb = @as(f64, @floatFromInt(self.current_kv_bytes)) / (1024.0 * 1024.0);
        if (self.max_kv_bytes == 0) {
            if (self.max_entries == unlimited_entries) {
                log.info("  [hot-cache] resident={d:.2} MB ({d} entries)\n", .{ mb, self.entries.items.len });
            } else {
                log.info("  [hot-cache] resident={d:.2} MB ({d}/{d} entries)\n", .{ mb, self.entries.items.len, self.max_entries });
            }
        } else {
            const cap_mb = @as(f64, @floatFromInt(self.max_kv_bytes)) / (1024.0 * 1024.0);
            if (self.max_entries == unlimited_entries) {
                log.info("  [hot-cache] resident={d:.2} / {d:.2} MB ({d} entries)\n", .{ mb, cap_mb, self.entries.items.len });
            } else {
                log.info("  [hot-cache] resident={d:.2} / {d:.2} MB ({d}/{d} entries)\n", .{ mb, cap_mb, self.entries.items.len, self.max_entries });
            }
        }
    }


    /// Drop all entries. Called when the cache is suspect (pad-only generation, image-bearing
    /// prompt, tools toggle change).
    pub fn invalidateAll(self: *HotPrefixCache, reason: []const u8) void {
        // Suspect state must die on both tiers.
        if (self.disk) |*d| d.invalidateAll();
        self.disk_dirty = false;
        if (self.pending_disk) |*p| {
            p.deinit(self.allocator);
            self.pending_disk = null;
        }
        if (self.entries.items.len == 0) return;
        log.info("  [hot-cache] invalidating all {d} entries: {s}\n", .{ self.entries.items.len, reason });
        for (self.entries.items) |*e| {
            freeEntryOwnedState(self.allocator, e);
        }
        self.entries.clearRetainingCapacity();
        self.current_kv_bytes = 0;
    }

    /// Empty the SSD tier only: staged writes are fenced off, the index and the files go together. The RAM tier stays.
    pub fn clearDisk(self: *HotPrefixCache) void {
        const d = &(self.disk orelse return);
        if (d.writer) |w| w.fence(null);
        d.invalidateAll();
        self.disk_dirty = false;
        if (self.pending_disk) |*p| p.deinit(self.allocator);
        self.pending_disk = null;
        self.last_restored_disk_id = null;
        self.last_restored_held = false;
    }

    /// Drop the most recently committed entry — used after a pad-only
    /// generation: the entry we just wrote may have stale K/V from the bad
    /// generation in tail positions. Other entries from prior healthy
    /// requests remain untouched (improvement over the legacy nuke-everything).
    pub fn invalidateLatest(self: *HotPrefixCache, reason: []const u8) void {
        if (self.disk) |*d| d.invalidateNewest();
        self.disk_dirty = false;
        if (self.pending_disk) |*p| {
            p.deinit(self.allocator);
            self.pending_disk = null;
        }
        if (self.entries.items.len == 0) return;
        const newest_idx = prefix_policy.newestIndex(Entry, self.entries.items, .last).?;
        var evicted = self.entries.swapRemove(newest_idx);
        self.current_kv_bytes -|= evicted.kv_bytes;
        freeEntryOwnedState(self.allocator, &evicted);
        log.info("  [hot-cache] invalidated latest entry: {s}\n", .{reason});
    }

    pub fn entryCount(self: *const HotPrefixCache) usize {
        return self.entries.items.len;
    }

    // ── Lookup, Restore & Digests (prefix_cache_index.zig) ──
    pub const restoreSpecSnap = prefix_cache_index.restoreSpecSnap;
    pub const restoreDflash = prefix_cache_index.restoreDflash;
    pub const restoreMtp = prefix_cache_index.restoreMtp;
    pub const diskRestoreSpec = prefix_cache_index.diskRestoreSpec;
    pub const highestCheckpointAtOrBelow = prefix_cache_index.highestCheckpointAtOrBelow;
    pub const lostReason = prefix_cache_index.lostReason;
    pub const highestCoveringCheckpoint = prefix_cache_index.highestCoveringCheckpoint;
    pub const deliverableShare = prefix_cache_index.deliverableShare;
    pub const qsaHistorySource = prefix_cache_index.qsaHistorySource;
    pub const resetSsmEntries = prefix_cache_index.resetSsmEntries;
    pub const findBestRestorableMatch = prefix_cache_index.findBestRestorableMatch;
    pub const RingRestore = prefix_cache_index.RingRestore;
    pub const ringRestore = prefix_cache_index.ringRestore;
    pub const clampLen = prefix_cache_index.clampLen;
    pub const findBestMatch = prefix_cache_index.findBestMatch;
    pub const lookupAndRestore = prefix_cache_index.lookupAndRestore;
    pub const lookupAndRestoreForSlot = prefix_cache_index.lookupAndRestoreForSlot;
    pub const lookupAndRestoreWithMedia = prefix_cache_index.lookupAndRestoreWithMedia;
    pub const poisonUnreadable = prefix_cache_index.poisonUnreadable;
    pub const lookupImpl = prefix_cache_index.lookupImpl;
    pub const retainedTokens = prefix_cache_index.retainedTokens;
    pub const RestoreMatch = prefix_cache_index.RestoreMatch;
    pub const restoreEntry = prefix_cache_index.restoreEntry;
    pub const restoreEntryTokens = prefix_cache_index.restoreEntryTokens;
    pub const reclaimableBytes = prefix_cache_index.reclaimableBytes;
    pub const EntryDigest = prefix_cache_index.EntryDigest;
    pub const DIGEST_PREFIX_TOKENS = prefix_cache_index.DIGEST_PREFIX_TOKENS;
    pub const prefixFingerprint = prefix_cache_index.prefixFingerprint;
    pub const digestsAlloc = prefix_cache_index.digestsAlloc;
    pub const reclaimableFromDigests = prefix_cache_index.reclaimableFromDigests;
    pub const reclaimableBytesFor = prefix_cache_index.reclaimableBytesFor;

    // ── Takeover/Checkout, Trim & Eviction (prefix_cache_evict.zig) ──
    pub const liveCacheBytes = prefix_cache_evict.liveCacheBytes;
    pub const snapshotBytes = prefix_cache_evict.snapshotBytes;
    pub const specSnapBytes = prefix_cache_evict.specSnapBytes;
    pub const snapshotRowBytes = prefix_cache_evict.snapshotRowBytes;
    pub const ringedBytes = prefix_cache_evict.ringedBytes;
    pub const RingTrim = prefix_cache_evict.RingTrim;
    pub const ringTrimLen = prefix_cache_evict.ringTrimLen;
    pub const TRIM_LOG_MAX_POS = prefix_cache_evict.TRIM_LOG_MAX_POS;
    pub const appendTrimFmt = prefix_cache_evict.appendTrimFmt;
    pub const formatTrimInputs = prefix_cache_evict.formatTrimInputs;
    pub const logTrimInputs = prefix_cache_evict.logTrimInputs;
    pub const SHED_SIM_MAX = prefix_cache_evict.SHED_SIM_MAX;
    pub const trimmedCheckpointBytes = prefix_cache_evict.trimmedCheckpointBytes;
    pub const shedSurvivorBytes = prefix_cache_evict.shedSurvivorBytes;
    pub const trimLenForBudgetPure = prefix_cache_evict.trimLenForBudgetPure;
    pub const trimBillArm = prefix_cache_evict.trimBillArm;
    pub const trimLenBillingAllLower = prefix_cache_evict.trimLenBillingAllLower;
    pub const trimLenForBudget = prefix_cache_evict.trimLenForBudget;
    pub const wholeTakeEligible = prefix_cache_evict.wholeTakeEligible;
    pub const partialTakeEligible = prefix_cache_evict.partialTakeEligible;
    pub const movable = prefix_cache_evict.movable;
    pub const upFrontTakeEligible = prefix_cache_evict.upFrontTakeEligible;
    pub const beginCheckout = prefix_cache_evict.beginCheckout;
    pub const endCheckout = prefix_cache_evict.endCheckout;
    pub const checkoutIfEligible = prefix_cache_evict.checkoutIfEligible;
    pub const checkoutRestored = prefix_cache_evict.checkoutRestored;
    pub const donateCheckout = prefix_cache_evict.donateCheckout;
    pub const keepTakenPrefix = prefix_cache_evict.keepTakenPrefix;
    pub const donatedBytes = prefix_cache_evict.donatedBytes;
    pub const releaseCheckout = prefix_cache_evict.releaseCheckout;
    pub const TruncateCheckpointsError = prefix_cache_evict.TruncateCheckpointsError;
    pub const truncateCheckpoints = prefix_cache_evict.truncateCheckpoints;
    pub const checkpointListBytes = prefix_cache_evict.checkpointListBytes;
    pub const rebillCheckpoints = prefix_cache_evict.rebillCheckpoints;
    pub const retainNewEntry = prefix_cache_evict.retainNewEntry;
    pub const declineIncoming = prefix_cache_evict.declineIncoming;
    pub const shedCheckpointsToFit = prefix_cache_evict.shedCheckpointsToFit;
    pub const shedRingCheckpoints = prefix_cache_evict.shedRingCheckpoints;
    pub const setBudget = prefix_cache_evict.setBudget;
    pub const setMaxEntries = prefix_cache_evict.setMaxEntries;
    pub const evictOneLru = prefix_cache_evict.evictOneLru;
    pub const reclaimFromLru = prefix_cache_evict.reclaimFromLru;
    pub const victimTrimLen = prefix_cache_evict.victimTrimLen;
    pub const trimVictim = prefix_cache_evict.trimVictim;
    pub const evictForSpace = prefix_cache_evict.evictForSpace;
    pub const evictOneLruProgress = prefix_cache_evict.evictOneLruProgress;
    pub const dropLastRestored = prefix_cache_evict.dropLastRestored;
    pub const dropQsaGapEntry = prefix_cache_evict.dropQsaGapEntry;
    pub const evictAt = prefix_cache_evict.evictAt;
    pub const evictOldestForTokens = prefix_cache_evict.evictOldestForTokens;
    pub const SHARED_RATIO_MIN_BYTES = prefix_cache_evict.SHARED_RATIO_MIN_BYTES;
    pub const SHARED_RETURN_DIVISOR = prefix_cache_evict.SHARED_RETURN_DIVISOR;
    pub const RESTORE_PIN_MIN_BYTES = prefix_cache_evict.RESTORE_PIN_MIN_BYTES;
    pub const RESTORE_PIN_RATIO = prefix_cache_evict.RESTORE_PIN_RATIO;
    pub const restoreWouldPinEntry = prefix_cache_evict.restoreWouldPinEntry;
    pub const evictLruToAdmit = prefix_cache_evict.evictLruToAdmit;
    pub const lruIndexExcluding = prefix_cache_evict.lruIndexExcluding;

    // ── Commit, SSD Flush & Checkpoint Donors (prefix_cache_commit.zig) ──
    pub const commit = prefix_cache_commit.commit;
    pub const commitWithSsm = prefix_cache_commit.commitWithSsm;
    pub const commitWithState = prefix_cache_commit.commitWithState;
    pub const commitWithMediaState = prefix_cache_commit.commitWithMediaState;
    pub const commitWithRing = prefix_cache_commit.commitWithRing;
    pub const commitImpl = prefix_cache_commit.commitImpl;
    pub const spillDeclinedToDisk = prefix_cache_commit.spillDeclinedToDisk;
    pub const capturePendingDisk = prefix_cache_commit.capturePendingDisk;
    pub const ringCommitOf = prefix_cache_commit.ringCommitOf;
    pub const EntrySpecs = prefix_cache_commit.EntrySpecs;
    pub const entrySpecCommits = prefix_cache_commit.entrySpecCommits;
    pub const spillIdleEntries = prefix_cache_commit.spillIdleEntries;
    pub const oldestIdleIndex = prefix_cache_commit.oldestIdleIndex;
    pub const unpersistableReason = prefix_cache_commit.unpersistableReason;
    pub const writeThroughHybrid = prefix_cache_commit.writeThroughHybrid;
    pub const flushPendingRecord = prefix_cache_commit.flushPendingRecord;
    pub const releaseRingCpsToDisk = prefix_cache_commit.releaseRingCpsToDisk;
    pub const flushPendingDisk = prefix_cache_commit.flushPendingDisk;
    pub const mergeCheckpointLists = prefix_cache_commit.mergeCheckpointLists;
    pub const mergeDiskCheckpoints = prefix_cache_commit.mergeDiskCheckpoints;
    pub const takeCpsIfQsaBank = prefix_cache_commit.takeCpsIfQsaBank;
    pub const bestCheckpointDonor = prefix_cache_commit.bestCheckpointDonor;
    pub const donorShared = prefix_cache_commit.donorShared;
    pub const RingDonor = prefix_cache_commit.RingDonor;
    pub const bestRingDonor = prefix_cache_commit.bestRingDonor;
    pub const shareRingCpsUpTo = prefix_cache_commit.shareRingCpsUpTo;
    pub const ringStepOf = prefix_cache_commit.ringStepOf;
    pub const thinRingCps = prefix_cache_commit.thinRingCps;
    pub const mergeRingCps = prefix_cache_commit.mergeRingCps;
    pub const ownRingCps = prefix_cache_commit.ownRingCps;
    pub const ringCpsUpTo = prefix_cache_commit.ringCpsUpTo;
    pub const freeRingCps = prefix_cache_commit.freeRingCps;
    pub const ringCpsBytes = prefix_cache_commit.ringCpsBytes;
    pub const cloneCheckpointsWithBank = prefix_cache_commit.cloneCheckpointsWithBank;
    pub const TextCps = prefix_cache_commit.TextCps;
    pub const textCheckpoints = prefix_cache_commit.textCheckpoints;
    pub const cloneCheckpointsUpTo = prefix_cache_commit.cloneCheckpointsUpTo;
    pub const keepNewestCheckpoints = prefix_cache_commit.keepNewestCheckpoints;

};

/// The line every request logs once its prefix lookup is done; `sushi bench` parses it to take the restore out of the prefill time.
pub fn restoreLogLine(buf: []u8, tag: []const u8, matched: usize, prompt_len: usize, tier: []const u8, restore_ms: u64) ![]const u8 {
    return std.fmt.bufPrint(buf, "{s} reused {d} of {d} prompt tokens from {s}, prefilling {d}; lookup and restore {d} ms", .{ tag, matched, prompt_len, tier, prompt_len - matched, restore_ms });
}


// ── Shared test helper re-exports (prefix_cache_helpers_test.zig) ──
pub const testWriteCacheLayer = prefix_cache_helpers_test.testWriteCacheLayer;
pub const testFillCache = prefix_cache_helpers_test.testFillCache;
pub const conv_shape_pc = prefix_cache_helpers_test.conv_shape_pc;
pub const pcArange = prefix_cache_helpers_test.pcArange;
pub const pcSsmVal = prefix_cache_helpers_test.pcSsmVal;
pub const pcBuildHybrid = prefix_cache_helpers_test.pcBuildHybrid;
pub const pcFreeHybrid = prefix_cache_helpers_test.pcFreeHybrid;
pub const pcEmptySsm = prefix_cache_helpers_test.pcEmptySsm;
pub const pcRowBytes = prefix_cache_helpers_test.pcRowBytes;
pub const pcFreeQsaHybrid = prefix_cache_helpers_test.pcFreeQsaHybrid;
pub const pcQsaSliceAxis1 = prefix_cache_helpers_test.pcQsaSliceAxis1;
pub const pcQsaExpectViewEqual = prefix_cache_helpers_test.pcQsaExpectViewEqual;
pub const pcArrEqual = prefix_cache_helpers_test.pcArrEqual;
pub const mediaTurnTokens = prefix_cache_helpers_test.mediaTurnTokens;

// ── Pinned domain-routing smoke test (tests/test_test_bins.sh) ──
const testing = std.testing;
test "HotPrefixCache: shouldUse gates hybrid by enable_ssm_checkpoints" {
    var cfg = model_mod.ModelConfig{};
    // Plain attention: always allowed.
    try testing.expect(HotPrefixCache.shouldUse(&cfg, false));
    try testing.expect(HotPrefixCache.shouldUse(&cfg, true));
    // full_attention_interval-marks-hybrid: only with checkpoints enabled.
    cfg.full_attention_interval = 4;
    try testing.expect(!HotPrefixCache.shouldUse(&cfg, false));
    try testing.expect(HotPrefixCache.shouldUse(&cfg, true));
}

