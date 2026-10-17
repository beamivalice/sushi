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
//! Hybrid SSM/conv architectures (qwen3_5/qwen3_5_moe/qwen3_next/nemotron_h/
//! lfm2) are excluded in v1: their recurrent state can't be rolled back, so
//! prefix reuse must reset on any divergence anyway. Plan 03's spec calls
//! these "hot tier only" — meaning we keep the single-slot path for them.
//! `HotPrefixCache.shouldUse(config)` returns false for those archs.

const std = @import("std");
const mlx = @import("mlx.zig");
const transformer_mod = @import("transformer.zig");
const model_mod = @import("model.zig");
const kv_quant = @import("kv_quant.zig");
const kv_disk_cache = @import("kv_disk_cache.zig");
const io_util = @import("io_util.zig");
const log = @import("log.zig");
const restore_dump = @import("restore_dump.zig");
const glm5_forward = @import("glm5_forward.zig");
const glm5_prefix = @import("glm5_prefix.zig");

const KVCache = transformer_mod.KVCache;
const KVCacheSnapshot = transformer_mod.KVCacheSnapshot;
const SSMCacheEntry = transformer_mod.SSMCacheEntry;
const SSMCheckpoint = transformer_mod.SSMCheckpoint;
const restoreSsmCheckpoint = transformer_mod.restoreSsmCheckpoint;
const applyQsaHistoryAt = transformer_mod.applyQsaHistoryAt;
const checkpointHasQsaPooled = transformer_mod.checkpointHasQsaPooled;
const checkpointListHasQsaPooled = transformer_mod.checkpointListHasQsaPooled;
const sliceQsaHistoryOntoCheckpoint = transformer_mod.sliceQsaHistoryOntoCheckpoint;
const keepOnlyLatestQsaHistory = transformer_mod.keepOnlyLatestQsaHistory;
const entriesHaveQsaHistory = transformer_mod.entriesHaveQsaHistory;
const qsaRestoreSatisfiesForward = transformer_mod.qsaRestoreSatisfiesForward;
const ssmCheckpointBytes = transformer_mod.ssmCheckpointBytes;

/// Minimum forwarded-prefix length for committing a CANCELLED prefill
/// (client disconnect mid-prefill). Below this an entry is LRU pollution —
/// chat-template prologues (Gemma=12, Qwen=8, Llama=4 tokens) are identical
/// across every request and "reusable" only in a worthless sense. Same
/// rationale as an engine session pool's `min_prefix_to_claim`, applied at
/// commit time instead of claim time.
pub const MIN_CANCELLED_COMMIT_TOKENS: usize = 256;

/// Shortest prefix a ring-checkpoint restore reuses; below it the request cold-prefills.
pub const RING_RESTORE_MIN_TOKENS: usize = 64;

/// Ring checkpoints one entry keeps, thinned so the lowest (a shared preamble) and the newest
/// (where the conversation goes on) stay (`thinRingCps`). One is ~16 MiB on MiMo at kv8.
pub const RING_CHECKPOINT_MAX: usize = 8;

/// Ring checkpoints a hot entry keeps in RAM when the SSD tier holds the older ones.
pub const RING_CHECKPOINT_RAM_KEEP: usize = 2;

/// Message starts one prefill takes a ring checkpoint at (`ringMarkPositions`).
pub const RING_MARKS_MAX: usize = 4;

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
    /// (measured on Qwen3.6-27B echo: ~70 → ~38 tok/s on warm repeats).
    mtp_base: ?usize = null,
    /// Did this restore check out its entry (`checkoutEligible`)? Only then does the first
    /// append donate in place; every other restore is a refcount share copied by that append.
    checked_out: bool = false,
    /// Did an SSD restore fill buffers the slot owns outright (`restoreKvInto`)?
    slot_owned: bool = false,
    /// `Entry.id` of the RAM entry restored from; 0 = none.
    entry_id: u64 = 0,

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
        if (self.fork) |*c| total += transformer_mod.kvEntriesBytes(c.entries);
        if (self.prompt_end) |*c| total += transformer_mod.kvEntriesBytes(c.entries);
        for (self.marks[0..self.n_marks]) |*c| total += transformer_mod.kvEntriesBytes(c.entries);
        return total;
    }
};

/// Where a prefill takes ring checkpoints: the message starts (`mark_id`) it forwards past the
/// `restored` prefix, from the restore floor up to where the prompt-end checkpoint reaches;
/// thinned to `RING_MARKS_MAX` keeping the first (a shared preamble's end) and the last.
pub fn ringMarkPositions(prompt: []const u32, restored: usize, mark_id: u32, out: *[RING_MARKS_MAX]usize) []usize {
    var buf: [RING_MARKS_MAX + 1]usize = undefined;
    var n: usize = 0;
    var p = @max(restored + 1, RING_RESTORE_MIN_TOKENS);
    while (p + model_mod.ModelConfig.SWA_RING_CHECKPOINT_BACKOFF < prompt.len) : (p += 1) {
        if (prompt[p] != mark_id) continue;
        buf[n] = p;
        n += 1;
        if (n > RING_MARKS_MAX) {
            const k = transformer_mod.positionDropIndexUsize(buf[0..n], .min_span);
            for (k..n - 1) |i| buf[i] = buf[i + 1];
            n -= 1;
        }
    }
    @memcpy(out[0..n], buf[0..n]);
    return out[0..n];
}

/// One media block of a prompt: where its placeholder rows start, and a key
/// over that block and every block before it (pixels and positions).
pub const MediaSpan = struct { start: usize, key: u64 };

/// The key of a whole media chain: its last block's (0 = no media).
pub fn chainKey(chain: []const MediaSpan) u64 {
    return if (chain.len == 0) 0 else chain[chain.len - 1].key;
}

/// Rows an entry keyed `entry_key` may share with a request under a different
/// key. An entry whose media is the request's first k blocks shares up to the
/// request's block k+1; any other pair shares only the text before either
/// side's first media row. Null = nothing is provably shared.
pub fn crossKeyBoundary(entry_key: u64, entry_start: ?usize, request_start: ?usize, chain: []const MediaSpan) ?usize {
    if (entry_key != 0 and chain.len > 1) {
        for (chain[0 .. chain.len - 1], chain[1..]) |span, next| {
            if (span.key == entry_key) return next.start;
        }
    }
    const es = entry_start orelse return request_start;
    return if (request_start) |rs| @min(es, rs) else es;
}

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
fn diskTextLen(tokens_len: usize, vision_key: u64, media_start: ?usize) usize {
    if (vision_key != 0 and media_start == null) return 0;
    return tokens_len;
}

/// What a disk record claims about media: the chain's last covered key and the first placeholder,
/// in the tier's own width. `.{}` is a record with no media row.
pub const MediaStamp = struct {
    key: u64 = 0,
    start: ?u32 = null,

    pub fn of(vision_key: u64, media_start: ?usize) MediaStamp {
        const ms = media_start orelse return .{ .key = vision_key };
        // A boundary the tier's width cannot hold keeps the key and drops the boundary, which is
        // the shape `diskTextLen` persists nothing for. Stamping the text key instead would hand
        // a text prompt rows that only these pixels explain.
        const narrowed = std.math.cast(u32, ms) orelse return .{ .key = vision_key };
        return .{ .key = vision_key, .start = narrowed };
    }
};

/// The media identity a record covering `covered_len` of `chain`'s prompt may claim: the chain
/// truncated to the blocks whose placeholder rows are inside it. A record must never claim a block
/// it does not hold -- overclaiming is safe (an equal-key request is clamped by `kv_len`, a
/// differing one falls back to the pre-media text) but it costs the reuse of the blocks it does
/// hold. A record with no media row at all claims nothing, so a text request can share it whole.
pub fn truncatedMediaKey(
    chain: []const MediaSpan,
    vision_key: u64,
    media_start: ?usize,
    covered_len: usize,
) MediaStamp {
    const ms = media_start orelse return MediaStamp.of(vision_key, null);
    if (ms >= covered_len) return .{};
    if (chain.len == 0) return MediaStamp.of(vision_key, ms);
    var last: u64 = 0;
    for (chain) |span| {
        if (span.start >= covered_len) break;
        last = span.key;
    }
    return if (last == 0) .{} else MediaStamp.of(last, ms);
}

const Entry = struct {
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
    /// GLM only: the MLA rows every checkpoint of `ssm_checkpoints` restores from, billed in `kv_bytes`.
    glm: ?glm5_prefix.MlaRows = null,
    glm_bytes: u64 = 0,
    /// Ringed caches only: restore points below the entry's END (`KVCache.ringCheckpoint`),
    /// ascending by `step`, billed in `kv_bytes`: the prompt end, which a long reply carries the
    /// ring past, and those inherited from the entry this prompt forked off (`bestRingDonor`).
    ring_cps: ?[]KVCacheSnapshot = null,
    /// Restore by move: the slot that took ownership of this entry's KV buffers (`KVCache.adopt`).
    /// While set the snapshot holds empty handles and the entry is invisible to every other
    /// reader. Cleared by the commit that replaces it; dropped at slot end otherwise.
    checked_out_by: ?usize = null,
    /// Has the slot's append donated these buffers in place (`donateCheckout`)? Until then the
    /// entry's handles are live and a slot that ends hands the entry back; only a donated checkout is dropped.
    checkout_donated: bool = false,
    /// Snapshot bytes the donation handed the slot; `kv_bytes` keeps billing them until release.
    donated_bytes: u64 = 0,
};

/// Where a hybrid restore puts its recurrent state: the slot's SSM entries, or a GLM request.
const Recurrent = union(enum) {
    none,
    ssm: []SSMCacheEntry,
    glm: *glm5_forward.Request,

    fn hybrid(self: Recurrent) bool {
        return self != .none;
    }
    fn ssmEntries(self: Recurrent) ?[]SSMCacheEntry {
        return if (self == .ssm) self.ssm else null;
    }
    fn reset(self: Recurrent) void {
        switch (self) {
            .none => {},
            .ssm => |entries| HotPrefixCache.resetSsmEntries(entries),
            .glm => |request| request.reset(),
        }
    }
};

/// What a spec-snap adoption may do, decided before any mlx call.
pub const SpecAdopt = union(enum) {
    /// Nothing to adopt: no payload, or the snap does not cover the reused range.
    skip,
    /// A KV-only spec cache (dflash context, sidecar MTP head).
    kv_only: usize,
    /// The qwen4_exp in-checkpoint head: KV + QSA aux together.
    head: usize,
    /// A head target met a payload with no QSA half (pre-v5 sidecar). Head-only miss.
    decline_head_no_history,
};

pub fn specAdoptPlan(base_pos: usize, snap_step: usize, matched: usize, has_head_target: bool, has_head_aux: bool) SpecAdopt {
    if (base_pos > matched) return .skip;
    const want = matched - base_pos;
    if (want > snap_step) return .skip;
    if (!has_head_target) return .{ .kv_only = want };
    if (!has_head_aux) return .decline_head_no_history;
    return .{ .head = want };
}

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

/// A GLM commit's MLA rows for each tier; null for any other arch.
const GlmTierRows = struct {
    ram: ?glm5_prefix.MlaRows = null,
    disk: ?glm5_prefix.MlaRows = null,
};

/// What the live cache held at commit time, captured before the RAM byte-budget trim: the
/// disk tier gets the full prefix even when RAM keeps a trimmed one. Refcount-shared.
const PendingDiskFlush = struct {
    snapshot: KVCacheSnapshot,
    tokens: []u32,
    has_tools: bool,
    media: MediaStamp = .{},
    ssm_cps: ?[]SSMCheckpoint = null,
    ring_cps: ?[]KVCacheSnapshot = null,
    dflash: ?DflashSnap = null,
    mtp: ?DflashSnap = null,
    glm: ?glm5_prefix.MlaRows = null,

    fn deinit(self: *PendingDiskFlush, allocator: std.mem.Allocator) void {
        self.snapshot.deinit();
        if (self.glm) |*rows| rows.deinit();
        allocator.free(self.tokens);
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
    /// qwen4_exp MTP head only: the head's QSA aux entry and the absolute position of its key
    /// row 0. A snap with one half and not the other is declined.
    head_aux: ?transformer_mod.SSMCacheEntrySnapshot = null,
    head_pos_base: c_int = 0,
    /// qwen4_exp MTP head only: the head's QSA leftovers at the trunk's checkpoint positions.
    /// A clamp to a restored trunk position lands far below the head's raw-key ring; this is
    /// what makes it exact there.
    head_marks: transformer_mod.QsaHeadMarkSet = .{},

    pub fn deinit(self: *DflashSnap) void {
        self.snapshot.deinit();
        if (self.head_aux) |*a| transformer_mod.ssmSnapshotDeinit(a);
        self.head_aux = null;
        self.head_marks.deinit();
    }
};

/// What `commitWithState` reads to build a `DflashSnap`. `head` is set only by the qwen4_exp MTP head.
pub const DflashCommit = struct {
    cache: *const KVCache,
    base_pos: usize,
    head: ?*const SSMCacheEntry = null,
    head_pos_base: c_int = 0,
    head_marks: []const transformer_mod.QsaHeadMark = &.{},
};

/// Where `lookupAndRestore` puts a restored assistant context. `base_pos` is
/// written on every path so the caller can build `DflashCtx` from it. `head` is the qwen4_exp
/// Transformer owning the in-checkpoint MTP head; the adoption then goes through `qwen4MtpAdopt`.
pub const DflashTarget = struct {
    cache: *KVCache,
    base_pos: *usize,
    head: ?*transformer_mod.Transformer = null,
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

pub const HotPrefixCache = struct {
    entries: std.ArrayList(Entry),
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
    /// A commit landed since the last `flushPendingDisk`.
    disk_dirty: bool = false,
    /// `last_used` of the entry the current request restored from; `evictLruToAdmit` refuses to evict it.
    last_restored_used: ?u64 = null,
    /// The raw prefix that restore matched, before a hybrid clamp (`checkoutRestored`).
    last_restored_shared: usize = 0,
    last_restored_disk_id: ?u64 = null,
    /// The arch keeps a QSA indexer history beside its SSM state (qwen4_exp).
    /// A restore that leaves the live entries without it cannot prefill —
    /// `qsaMaskFromQk` errors on every turn on that prefix — so it is a MISS.
    qsa_history_required: bool = false,
    /// The arch carries recurrent state, so an entry restores only from a checkpoint. Set at
    /// load: an empty checkpoint list arrives as null, so the list cannot say.
    hybrid: bool = false,
    /// Checkpoint-retention policy, mirrored once at wiring from `ModelConfig.longCtxGated()`
    /// (this struct never sees a ModelConfig). The default is the previous behaviour.
    cp_thin: transformer_mod.ThinPolicy = .min_span,
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
        if (self.disk) |*d| d.deinit();
        self.disk = null;
    }

    /// Free everything an Entry owns: token buffer, KV snapshot, SSM
    /// checkpoint array. Used by `deinit`, eviction, and replace paths so
    /// they don't drift apart.
    fn freeEntryOwnedState(allocator: std.mem.Allocator, e: *Entry) void {
        allocator.free(e.tokens);
        e.snapshot.deinit();
        if (e.glm) |*rows| rows.deinit();
        e.glm = null;
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
            freeRingCps(allocator, r);
            e.ring_cps = null;
        }
    }

    /// Restore a committed speculative-side cache (DFlash context or MTP
    /// history) into the caller's cache, clamped to the trunk's restored
    /// length. Returns the base position on success, null when there is
    /// nothing to restore, the snap starts PAST what the trunk actually
    /// reused, or the snap ends BEFORE it (a history with a gap right below
    /// the generation point is worse than a blind start). Best-effort by
    /// contract: a failure leaves the caller blind, never wrong.
    fn restoreSpecSnap(snap_opt: ?*const DflashSnap, target: ?DflashTarget, matched: usize, s: mlx.mlx_stream, what: []const u8) ?usize {
        const t = target orelse return null;
        const snap = snap_opt orelse return null;
        const want = switch (specAdoptPlan(snap.base_pos, snap.snapshot.step, matched, t.head != null, snap.head_aux != null)) {
            .skip => {
                log.info("  [hot-cache] {s} not adopted: want {d} tokens from base {d}, snap holds {d} (matched {d})\n", .{ what, matched -| snap.base_pos, snap.base_pos, snap.snapshot.step, matched });
                return null;
            },
            .decline_head_no_history => {
                log.info("  [qwen4] MTP head restore declined (snapshot carries no QSA history) — head starts blind\n", .{});
                return null;
            },
            .kv_only, .head => |w| w,
        };
        // qwen4_exp in-checkpoint head: KV + QSA aux adopt together or not at all.
        if (t.head) |xfm| {
            const aux = &snap.head_aux.?;
            xfm.qwen4MtpAdopt(&snap.snapshot, aux, snap.head_marks.slice(), snap.head_pos_base, want) catch |err| {
                log.warn("  [qwen4] MTP head restore declined ({s}) — head starts blind\n", .{@errorName(err)});
                return null;
            };
            t.base_pos.* = snap.base_pos;
            log.info("  [qwen4] MTP head restored ({d} tokens from base {d})\n", .{ want, snap.base_pos });
            return snap.base_pos;
        }
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

    fn restoreDflash(e: *const Entry, target: ?DflashTarget, matched: usize, s: mlx.mlx_stream) ?usize {
        return restoreSpecSnap(if (e.dflash) |*d| d else null, target, matched, s, "dflash context");
    }

    fn restoreMtp(e: *const Entry, target: ?DflashTarget, matched: usize, s: mlx.mlx_stream) ?usize {
        return restoreSpecSnap(if (e.mtp) |*m| m else null, target, matched, s, "mtp history");
    }

    /// Disk-tier variant: load the persisted spec snapshot (v4 sidecar) as a
    /// transient and adopt it under the EXACT same clamp rule as the RAM
    /// tier (`restoreSpecSnap`). The trunk restore already forwarded nothing,
    /// so without this a disk hit drafted blind — the 92.6% → 66.5%
    /// acceptance class the RAM tier already fixed.
    fn diskRestoreSpec(
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
        // qwen4 head target declines it.
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
    fn highestCheckpointAtOrBelow(cps: []const SSMCheckpoint, limit: usize) ?*const SSMCheckpoint {
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

    fn highestCoveringCheckpoint(cps: []const SSMCheckpoint, limit: usize) ?*const SSMCheckpoint {
        var cap = limit;
        while (highestCheckpointAtOrBelow(cps, cap)) |cp| {
            const src = qsaHistorySource(cps, cp) orelse cp;
            if (transformer_mod.checkpointQsaCoversPos(cp, src, cp.pos)) return cp;
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
    fn qsaHistorySource(cps: []const SSMCheckpoint, restored: *const SSMCheckpoint) ?*const SSMCheckpoint {
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
            transformer_mod.ssmFreeQsaState(ssm);
            ssm.ple_prev_valid = false;
        }
        transformer_mod.ssmDetachFromGroup(entries);
    }

    /// Pure-attention + DSV4 are eligible by default. Hybrid recurrent-state
    /// archs are gated by `enable_ssm_checkpoints` (set by the scheduler
    /// when `--ssm-checkpoint-stride > 0`): with checkpoints we can rewind
    /// both KV and SSM state to a stride-aligned position; without them
    /// every divergence would force a full reset, so we keep the legacy
    /// single-slot path.
    ///
    /// Both `has_hybrid_layers` and `full_attention_interval > 0` signal
    /// the model has SSM/GatedDeltaNet layers somewhere — `has_hybrid_layers`
    /// is set explicitly by the parsers for lfm2 / nemotron_h; the qwen3_5
    /// family sets `full_attention_interval` to N to mark "every Nth layer
    /// is full attention, the rest are GatedDeltaNet". Either way the same
    /// SSM-checkpoint gate applies.
    pub fn shouldUse(
        config: *const model_mod.ModelConfig,
        enable_ssm_checkpoints: bool,
    ) bool {
        // dsv4 keeps its per-request state (raw-kv rings, compressed caches,
        // compressor pending windows) on the module-owned Dsv4Model, not in
        // the KVCache — a snapshot restore would advance cache.step without
        // rebuilding that state. Off until dsv4 state rides the ssm-entry
        // machinery (needsSsmEntries class).
        if (std.mem.eql(u8, config.model_type, "deepseek_v4")) return false;
        const has_ssm_layers = config.has_hybrid_layers or config.full_attention_interval > 0;
        if (has_ssm_layers and !enable_ssm_checkpoints) return false;
        return true;
    }

    fn bumpCounter(self: *HotPrefixCache) u64 {
        self.counter += 1;
        return self.counter;
    }

    /// Wave 1.B: total KV bytes held by a snapshot — sum of `size * itemsize`
    /// across every initialized entry's storage arrays. mlx-c arrays carry
    /// their shape + dtype so this is exact, not a heuristic. Quant schemes
    /// account for q, scales, biases together.
    /// What `snapshotBytes` will bill of `cache` once a commit snapshots it.
    pub fn liveCacheBytes(cache: *const KVCache) u64 {
        return snapshotBytes(&.{ .entries = cache.entries, .step = cache.step, .allocator = cache.allocator, .config = cache.config });
    }

    fn snapshotBytes(snap: *const KVCacheSnapshot) u64 {
        var total: u64 = 0;
        for (snap.entries) |e| {
            if (!e.initialized) continue;
            total += @as(u64, mlx.mlx_array_size(e.keys)) * @as(u64, mlx.mlx_array_itemsize(e.keys));
            total += @as(u64, mlx.mlx_array_size(e.values)) * @as(u64, mlx.mlx_array_itemsize(e.values));
            if (snap.config.scheme != .off) {
                total += @as(u64, mlx.mlx_array_size(e.keys_scales)) * @as(u64, mlx.mlx_array_itemsize(e.keys_scales));
                total += @as(u64, mlx.mlx_array_size(e.keys_biases)) * @as(u64, mlx.mlx_array_itemsize(e.keys_biases));
                total += @as(u64, mlx.mlx_array_size(e.values_scales)) * @as(u64, mlx.mlx_array_itemsize(e.values_scales));
                total += @as(u64, mlx.mlx_array_size(e.values_biases)) * @as(u64, mlx.mlx_array_itemsize(e.values_biases));
            }
        }
        return total;
    }

    /// Resident bytes of a speculative-side snap: the KV plus (qwen4_exp) the head's QSA aux half.
    fn specSnapBytes(snap: *const DflashSnap) u64 {
        var total = snapshotBytes(&snap.snapshot);
        if (snap.head_aux) |a| {
            // Only the two the head owns: its `conv_state`/`ssm_state` are empty handles.
            inline for (.{ a.aux_state, a.qsa_pooled }) |arr| {
                if (arr.ctx != null) total += @as(u64, mlx.mlx_array_size(arr)) * @as(u64, mlx.mlx_array_itemsize(arr));
            }
            total += snap.head_marks.bytes();
        }
        return total;
    }

    /// Issue #330: per-token bytes of a snapshot — what one retained token
    /// costs after a `trimmedCopy` materializes exactly `len` rows. Derived
    /// from each array's own shape (bytes / capacity rows), so it prices
    /// quantized triples correctly too. A ringed layer holds a window at any
    /// length (`ringedBytes`), so it has no per-token price.
    fn snapshotRowBytes(snap: *const KVCacheSnapshot) u64 {
        var total: u64 = 0;
        for (snap.entries) |e| {
            if (!e.initialized or e.ringed) continue;
            inline for (.{ e.keys, e.values, e.keys_scales, e.keys_biases, e.values_scales, e.values_biases }) |arr| {
                // A dense snapshot leaves the quant handles as empty 0-dim arrays; the ndim check makes axis 2 a fact.
                if (arr.ctx != null and mlx.mlx_array_ndim(arr) > 2) {
                    const rows: u64 = @intCast(mlx.mlx_array_shape(arr)[2]);
                    if (rows > 0) {
                        total += (@as(u64, mlx.mlx_array_size(arr)) * @as(u64, mlx.mlx_array_itemsize(arr))) / rows;
                    }
                }
            }
        }
        return total;
    }

    /// Bytes of a snapshot's ringed layers: the window they keep whatever the length.
    fn ringedBytes(snap: *const KVCacheSnapshot) u64 {
        var total: u64 = 0;
        for (snap.entries) |e| {
            if (!e.initialized or !e.ringed) continue;
            inline for (.{ e.keys, e.values, e.keys_scales, e.keys_biases, e.values_scales, e.values_biases }) |arr| {
                if (arr.ctx != null) total += @as(u64, mlx.mlx_array_size(arr)) * @as(u64, mlx.mlx_array_itemsize(arr));
            }
        }
        return total;
    }

    /// Where a ringed candidate trims to: its end, or ring checkpoint `cp`, whose rows become its ring.
    const RingTrim = struct { len: usize, cp: ?usize = null };

    /// A ringed candidate restores only where its sliding layers hold the window: at its `end`, or
    /// at one of its ring checkpoints (ascending). The longest that fits `budget`: the rows below it
    /// at `row_bytes`, the ring it keeps and the checkpoints below it. Null = none above the floor.
    fn ringTrimLen(budget: u64, limit: usize, row_bytes: u64, end: usize, ring_bytes: u64, cps: []const KVCacheSnapshot) ?RingTrim {
        var below: u64 = 0;
        for (cps) |*c| below += snapshotBytes(c);
        if (end <= limit and end >= MIN_CANCELLED_COMMIT_TOKENS and
            @as(u64, end) * row_bytes + ring_bytes + below <= budget) return .{ .len = end };
        var k = cps.len;
        while (k > 0) {
            k -= 1;
            const cp_bytes = snapshotBytes(&cps[k]);
            below -= cp_bytes;
            const p = cps[k].step;
            if (p > limit or p >= end) continue;
            if (p < MIN_CANCELLED_COMMIT_TOKENS) return null;
            if (@as(u64, p) * row_bytes + cp_bytes + below <= budget) return .{ .len = p, .cp = k };
        }
        return null;
    }

    /// Positions printed by the trim-inputs line before it elides; the count is always exact.
    const TRIM_LOG_MAX_POS: usize = 32;

    fn appendTrimFmt(buf: []u8, n: *usize, comptime fmt: []const u8, args: anytype) void {
        const s = std.fmt.bufPrint(buf[n.*..], fmt, args) catch return;
        n.* += s.len;
    }

    /// The trim decision's inputs as one line (the price, the positions, the chosen bill).
    fn formatTrimInputs(
        buf: []u8,
        tokens_len: usize,
        row_bytes: u64,
        budget: u64,
        positions: []const usize,
        cp_bytes: []const u64,
        total: usize,
        chosen: ?usize,
        gated: bool,
    ) []const u8 {
        var n: usize = 0;
        appendTrimFmt(buf, &n, "  [hot-cache] trim inputs: tokens={d} row_bytes={d} budget={d:.2} MB list_len={d} arm={s} survivors=[", .{
            tokens_len,
            row_bytes,
            @as(f64, @floatFromInt(budget)) / (1024.0 * 1024.0),
            total,
            trimBillArm(total, gated),
        });
        const shown = @min(positions.len, TRIM_LOG_MAX_POS);
        for (positions[0..shown], 0..) |p, i| {
            if (i > 0) appendTrimFmt(buf, &n, ",", .{});
            appendTrimFmt(buf, &n, "{d}", .{p});
        }
        if (shown < total) appendTrimFmt(buf, &n, ",...", .{});
        appendTrimFmt(buf, &n, "] ({d} of {d})", .{ shown, total });
        if (chosen) |tl| {
            appendTrimFmt(buf, &n, " chosen={d}", .{tl});
            var chosen_cp: u64 = 0;
            for (positions, 0..) |p, i| {
                if (p == tl and i < cp_bytes.len) {
                    chosen_cp = cp_bytes[i];
                    break;
                }
            }
            appendTrimFmt(buf, &n, " chosen_cp_bytes={d}", .{chosen_cp});
        } else {
            appendTrimFmt(buf, &n, " chosen=none", .{});
        }
        appendTrimFmt(buf, &n, "\n", .{});
        return buf[0..n];
    }

    /// Fires once per oversized commit.
    fn logTrimInputs(
        tokens_len: usize,
        row_bytes: u64,
        budget: u64,
        cps: ?[]const SSMCheckpoint,
        chosen: ?usize,
        gated: bool,
    ) void {
        var pos_buf: [SHED_SIM_MAX]usize = undefined;
        var byte_buf: [SHED_SIM_MAX]u64 = undefined;
        var k: usize = 0;
        if (cps) |list| {
            while (k < list.len and k < SHED_SIM_MAX) : (k += 1) {
                pos_buf[k] = list[k].pos;
                byte_buf[k] = trimmedCheckpointBytes(list, k);
            }
        }
        const total = if (cps) |list| list.len else 0;
        var line: [768]u8 = undefined;
        log.info("{s}", .{formatTrimInputs(&line, tokens_len, row_bytes, budget, pos_buf[0..k], byte_buf[0..k], total, chosen, gated)});
    }

    /// Stack bound for the shed simulation; a longer list falls back to billing every lower checkpoint.
    const SHED_SIM_MAX: usize = 128;

    /// The last retained checkpoint inherits the latest checkpoint's pooled bank.
    /// Price its sliced shape before allocating/copying any multi-GB KV prefix.
    /// Captured aux state already contains the destination's pos % ratio leftover.
    fn trimmedCheckpointBytes(list: []const SSMCheckpoint, index: usize) u64 {
        const dst = &list[index];
        const src = &list[list.len - 1];
        var bytes = ssmCheckpointBytes(dst);
        if (index == list.len - 1 or !checkpointHasQsaPooled(src)) return bytes;
        if (dst.layers.len != src.layers.len) return bytes;
        for (dst.layers, src.layers) |d, from| {
            if (d.conv_state.ctx != null and mlx.mlx_array_size(d.conv_state) > 0) continue;
            if (from.qsa_pooled.ctx == null) continue;
            if (d.qsa_pooled.ctx != null)
                bytes -= @as(u64, mlx.mlx_array_size(d.qsa_pooled)) * @as(u64, mlx.mlx_array_itemsize(d.qsa_pooled));
            const shape = mlx.getShape(from.qsa_pooled);
            const blocks: u64 = @intCast(shape[1]);
            if (blocks == 0) continue;
            const retained = @min(blocks, dst.pos / @as(usize, @intCast(@max(from.qsa_ratio, 1))));
            bytes += (@as(u64, mlx.mlx_array_size(from.qsa_pooled)) / blocks) *
                @as(u64, mlx.mlx_array_itemsize(from.qsa_pooled)) * retained;
        }
        return bytes;
    }

    /// Bytes the checkpoints at or below a candidate trim point cost after the commit path's
    /// span-preserving shed to `allowance`; null when even the last survivor is over.
    fn shedSurvivorBytes(positions: []const usize, bytes: []const u64, allowance: u64, policy: transformer_mod.ThinPolicy) ?u64 {
        var total: u64 = 0;
        for (bytes) |b| total += b;
        if (total <= allowance) return total;
        if (positions.len > SHED_SIM_MAX) return null;
        var pos_buf: [SHED_SIM_MAX]usize = undefined;
        var byte_buf: [SHED_SIM_MAX]u64 = undefined;
        @memcpy(pos_buf[0..positions.len], positions);
        @memcpy(byte_buf[0..bytes.len], bytes);
        var n = positions.len;
        while (total > allowance and n > 1) {
            const drop = transformer_mod.positionDropIndexUsize(pos_buf[0..n], policy);
            total -= byte_buf[drop];
            var k = drop;
            while (k + 1 < n) : (k += 1) {
                pos_buf[k] = pos_buf[k + 1];
                byte_buf[k] = byte_buf[k + 1];
            }
            n -= 1;
        }
        return if (total <= allowance) total else null;
    }

    /// Shared hybrid selector; checkpoints optionally supply the transferred QSA bank's bill.
    fn trimLenForBudgetPure(
        budget: u64,
        limit: usize,
        row_bytes: u64,
        positions: []const usize,
        cp_bytes: []const u64,
        policy: transformer_mod.ThinPolicy,
        cps: ?[]const SSMCheckpoint,
    ) ?usize {
        var byte_buf: [SHED_SIM_MAX]u64 = undefined;
        var k = positions.len;
        while (k > 0) {
            k -= 1;
            const p = positions[k];
            if (p > limit) continue;
            // A released state restores from the SSD, not from this entry.
            if (cps) |list| if (list[k].isReleased()) continue;
            if (p < MIN_CANCELLED_COMMIT_TOKENS) return null;
            const rows = @as(u64, p) * row_bytes;
            if (rows > budget) continue;
            const candidate_bytes = if (cps) |list| blk: {
                // Only the final survivor inherits the bank; lower checkpoints
                // keep their original costs throughout the shed simulation.
                @memcpy(byte_buf[0 .. k + 1], cp_bytes[0 .. k + 1]);
                byte_buf[k] = trimmedCheckpointBytes(list, k);
                break :blk byte_buf[0 .. k + 1];
            } else cp_bytes[0 .. k + 1];
            if (shedSurvivorBytes(positions[0 .. k + 1], candidate_bytes, budget - rows, policy) != null) return p;
        }
        return null;
    }

    /// The MLA rows (a checkpoint position) a GLM commit keeps, chosen before anything is copied.
    /// `budget` 0 keeps every row. A RAM tier keeps the rows, its checkpoints and the window within
    /// `budget`, as `commitImpl`'s trim does, shedding checkpoints to fit; the SSD tier (`disk`)
    /// keeps every checkpoint at or below the rows, so it prices all of them.
    pub fn glmCommitLenFor(
        policy: transformer_mod.ThinPolicy,
        budget: u64,
        disk: bool,
        newest: usize,
        row_bytes: u64,
        window_bytes: u64,
        positions: []const usize,
        cp_bytes: []const u64,
    ) ?usize {
        if (newest == 0) return null;
        if (budget == 0) return newest;
        if (!disk) {
            var cps_total: u64 = 0;
            for (cp_bytes) |b| cps_total += b;
            if (@as(u64, newest) * row_bytes + cps_total + window_bytes <= budget) return newest;
            return trimLenForBudgetPure(budget -| window_bytes, newest, row_bytes, positions, cp_bytes, policy, null);
        }
        const price = struct {
            fn at(p: usize, rb: u64, win: u64, pos: []const usize, bytes: []const u64) u64 {
                var total: u64 = @as(u64, p) * rb + win;
                for (pos, bytes) |q, b| {
                    if (q <= p) total += b;
                }
                return total;
            }
        }.at;
        if (price(newest, row_bytes, window_bytes, positions, cp_bytes) <= budget) return newest;
        var k = positions.len;
        while (k > 0) {
            k -= 1;
            if (positions[k] < newest and price(positions[k], row_bytes, 0, positions, cp_bytes) <= budget) return positions[k];
        }
        return null;
    }

    /// The rows a GLM commit copies into the RAM tier (`glmCommitLenFor` at its budget); null with
    /// RAM retention off or nothing that fits.
    pub fn glmCommitLen(self: *const HotPrefixCache, cps: []const SSMCheckpoint, offset: usize, row_bytes: u64, window_bytes: u64) ?usize {
        if (!self.ram_enabled) return null;
        return self.glmLenFor(cps, offset, self.max_kv_bytes, false, row_bytes, window_bytes);
    }

    /// The rows a GLM commit hands the SSD tier: through the newest checkpoint whose rows, every
    /// checkpoint at or below and the window fit the tier's byte budget. The flush reads them from
    /// the request's own buffers, piece by piece.
    pub fn glmDiskLen(self: *const HotPrefixCache, cps: []const SSMCheckpoint, offset: usize, row_bytes: u64, window_bytes: u64) ?usize {
        const d = if (self.disk) |*dd| dd else return null;
        return self.glmLenFor(cps, offset, d.max_bytes, true, row_bytes, window_bytes);
    }

    fn glmLenFor(self: *const HotPrefixCache, cps: []const SSMCheckpoint, offset: usize, budget: u64, disk: bool, row_bytes: u64, window_bytes: u64) ?usize {
        if (cps.len > SHED_SIM_MAX) return null;
        var pos_buf: [SHED_SIM_MAX]usize = undefined;
        var byte_buf: [SHED_SIM_MAX]u64 = undefined;
        for (cps, 0..) |*cp, i| {
            pos_buf[i] = cp.pos;
            // A released state is on the SSD tier already, and counts against its budget.
            byte_buf[i] = if (cp.isReleased()) cp.released_bytes else ssmCheckpointBytes(cp);
        }
        const newest = glm5_prefix.commitRows(cps, offset);
        // Beside an SSD tier the hot entry keeps only its newest checkpoints, so only those are priced.
        if (!disk and budget > 0 and newest > 0 and self.capturesLiveForDisk()) {
            var kept: u64 = 0;
            for (byte_buf[cps.len -| glm5_prefix.checkpoint_hot_kept .. cps.len]) |b| kept += b;
            if (@as(u64, newest) * row_bytes + kept + window_bytes <= budget) return newest;
        }
        return glmCommitLenFor(self.cp_thin, budget, disk, newest, row_bytes, window_bytes, pos_buf[0..cps.len], byte_buf[0..cps.len]);
    }

    /// Which arm `trimLenForBudget` bills a list of this length with (for the log).
    fn trimBillArm(list_len: usize, gated: bool) []const u8 {
        if (!gated) return "all_lower";
        return if (list_len > SHED_SIM_MAX) "all_lower" else "shed";
    }

    /// The pre-shed bill (every lower checkpoint), for a list past `SHED_SIM_MAX` or the ungated arm.
    fn trimLenBillingAllLower(budget: u64, limit: usize, row_bytes: u64, list: []const SSMCheckpoint) ?usize {
        var k = list.len;
        while (k > 0) {
            k -= 1;
            const p = list[k].pos;
            if (p > limit) continue;
            if (p < MIN_CANCELLED_COMMIT_TOKENS) return null;
            var cps_cost: u64 = 0;
            for (list[0 .. k + 1]) |*cp| cps_cost += ssmCheckpointBytes(cp);
            cps_cost = cps_cost - ssmCheckpointBytes(&list[k]) + trimmedCheckpointBytes(list, k);
            if (@as(u64, p) * row_bytes + cps_cost <= budget) return p;
        }
        return null;
    }

    /// Issue #330: the longest retainable prefix length under `budget`, or
    /// null when nothing at or above the commit floor fits. With checkpoints
    /// (hybrid entry) the trim point must be a RESTORABLE position — a
    /// checkpoint's own `pos`, and its cost includes the checkpoints that survive the commit's
    /// shed. Plain attention restores at any length. `limit` caps the answer.
    fn trimLenForBudget(
        self: *const HotPrefixCache,
        budget: u64,
        limit: usize,
        row_bytes: u64,
        cps: ?[]const SSMCheckpoint,
    ) ?usize {
        if (cps) |list| {
            if (list.len > 0) {
                // Arch gate: the ungated arm prices every lower checkpoint, as before.
                if (self.cp_thin == .min_span) return trimLenBillingAllLower(budget, limit, row_bytes, list);
                if (list.len > SHED_SIM_MAX) return trimLenBillingAllLower(budget, limit, row_bytes, list);
                var pos_buf: [SHED_SIM_MAX]usize = undefined;
                var byte_buf: [SHED_SIM_MAX]u64 = undefined;
                for (list, 0..) |*cp, i| {
                    pos_buf[i] = cp.pos;
                    byte_buf[i] = ssmCheckpointBytes(cp);
                }
                return trimLenForBudgetPure(budget, limit, row_bytes, pos_buf[0..list.len], byte_buf[0..list.len], self.cp_thin, list);
            }
        }
        if (row_bytes == 0) return null;
        const fit: usize = @intCast(budget / row_bytes);
        const len = @min(fit, limit);
        if (len < MIN_CANCELLED_COMMIT_TOKENS) return null;
        return len;
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
    fn findBestRestorableMatch(
        self: *const HotPrefixCache,
        prompt_ids: []const u32,
        has_tools: bool,
        vision_key: u64,
        media_start: ?usize,
        media_chain: []const MediaSpan,
        quant_config: kv_quant.KVQuantConfig,
        require_ssm_checkpoint: bool,
        probe: ?*MatchProbe,
    ) ?struct { idx: usize, shared: usize } {
        var best_idx: ?usize = null;
        var best_shared: usize = 0;
        var best_effective: usize = 0;
        for (self.entries.items, 0..) |*e, i| {
            // A checked-out entry's buffers belong to another slot; its snapshot is empty.
            if (e.checked_out_by != null) continue;
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
            var shared: usize = 0;
            while (shared < max_shared and e.tokens[shared] == prompt_ids[shared]) shared += 1;

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
            if (effective > best_effective or
                (effective == best_effective and shared > best_shared))
            {
                best_effective = effective;
                best_shared = shared;
                best_idx = i;
            }
        }
        if (best_idx) |idx| return .{ .idx = idx, .shared = best_shared };
        return null;
    }

    const RingRestore = struct { len: usize, cp: ?*const KVCacheSnapshot };

    /// Where an entry restores a match of `shared` tokens. A ringed entry's own ring holds the
    /// window at its END; below that only a ring checkpoint (`cp`) can serve, the one reaching
    /// highest at its position or the match, whichever is lower. Null = un-restorable. Any
    /// other entry holds a prefix and restores at `shared`.
    fn ringRestore(e: *const Entry, shared: usize, prompt_len: usize) ?RingRestore {
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
    fn clampLen(len: usize, prompt_len: usize) usize {
        return if (len == prompt_len and len > 1) len - 1 else len;
    }

    fn findBestMatch(self: *const HotPrefixCache, prompt_ids: []const u32, has_tools: bool, vision_key: u64, quant_config: kv_quant.KVQuantConfig) ?struct { idx: usize, shared: usize } {
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

    /// A GLM request restores its KDA state from the entry's checkpoints and its MLA rows from `Entry.glm`.
    pub fn lookupAndRestoreGlm(
        self: *HotPrefixCache,
        target_cache: *KVCache,
        target_moe_seq_offset: *usize,
        request: *glm5_forward.Request,
        s: mlx.mlx_stream,
        prompt_ids: []const u32,
        has_tools: bool,
        vision_key: u64,
        media_start: ?usize,
        media_chain: []const MediaSpan,
        dflash_target: ?DflashTarget,
        slot_id: ?usize,
        skip: bool,
    ) !LookupResult {
        return self.lookupImpl(target_cache, target_moe_seq_offset, .{ .glm = request }, s, prompt_ids, has_tools, vision_key, media_start, media_chain, dflash_target, null, slot_id, skip);
    }

    /// An unreadable or corrupt file fails the same way on every later lookup, so its entry dies;
    /// a coverage miss (no checkpoint, QSA gap) can still serve a shorter prefix, so it stays.
    fn poisonUnreadable(d: *kv_disk_cache.DiskTier, idx: usize, err: anyerror) void {
        switch (err) {
            error.MlxError, error.DiskCacheCorruptChunk, error.DiskCacheCorruptRing, error.DiskCacheCorruptSsm => {
                _ = d.poisonId(d.entries.items[idx].id, @errorName(err));
            },
            else => {},
        }
    }

    /// GLM's SSD arm: the entry whose highest persisted KDA checkpoint under the match beats the RAM
    /// match restores its MLA rows and that checkpoint into buffers the request then owns.
    fn restoreGlmFromDisk(
        self: *HotPrefixCache,
        d: *kv_disk_cache.DiskTier,
        request: *glm5_forward.Request,
        target_cache: *KVCache,
        ram_eff: usize,
        prompt_ids: []const u32,
        has_tools: bool,
        media_limits: kv_disk_cache.MediaLimits,
        dflash_target: ?DflashTarget,
        s: mlx.mlx_stream,
    ) ?LookupResult {
        const bits = target_cache.config.glmLatentBits() orelse return null;
        const quant = glm5_prefix.diskQuant(bits);
        const hm = d.bestHybridMatch(prompt_ids, has_tools, quant, @intCast(prompt_ids.len), media_limits) orelse return null;
        if (@as(usize, hm.cp) < ram_eff + kv_disk_cache.MIN_DISK_ADVANTAGE_TOKENS) return null;
        const sw = io_util.Stopwatch.init(d.io);
        var pseudo = KVCache.initWithConfig(self.allocator, @intCast(request.layers.len * 2), quant) catch return null;
        defer pseudo.deinit();
        const kda = self.allocator.alloc(SSMCacheEntry, request.layers.len) catch return null;
        for (kda) |*k| k.* = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false };
        defer {
            resetSsmEntries(kda);
            for (kda) |*k| {
                _ = mlx.mlx_array_free(k.conv_state);
                _ = mlx.mlx_array_free(k.ssm_state);
            }
            self.allocator.free(kda);
        }
        const restored = d.restoreIntoKda(&pseudo, kda, hm.idx, hm.cp, s) catch |err| {
            log.warn("  [disk-cache] GLM restore failed: {s} — falling back to RAM/cold path\n", .{@errorName(err)});
            poisonUnreadable(d, hm.idx, err);
            return null;
        };
        var rows = glm5_prefix.rowsFromDisk(self.allocator, &pseudo, bits) catch return null;
        defer rows.deinit();
        rows.restore(request, restored) catch |err| {
            log.warn("  [disk-cache] GLM restore failed: {s} — falling back to RAM/cold path\n", .{@errorName(err)});
            request.reset();
            return null;
        };
        for (request.layers, kda) |*layer, *k| {
            std.mem.swap(mlx.mlx_array, &layer.recurrent.conv_state, &k.conv_state);
            std.mem.swap(mlx.mlx_array, &layer.recurrent.ssm_state, &k.ssm_state);
            layer.recurrent.initialized = k.initialized;
        }
        target_cache.truncate(0, s) catch {};
        target_cache.step = restored;
        self.last_restored_disk_id = d.entries.items[hm.idx].id;
        log.info("  [disk-cache] restored {d}/{d} tokens from SSD in {d}ms (glm kda@{d})\n", .{ restored, prompt_ids.len, sw.read() / std.time.ns_per_ms, hm.cp });
        return .{
            .matched = restored,
            .full_match = false,
            .dflash_base = diskRestoreSpec(d, hm.idx, dflash_target, restored, s, .dflash),
            .slot_owned = true,
        };
    }

    fn lookupImpl(
        self: *HotPrefixCache,
        target_cache: *KVCache,
        target_moe_seq_offset: *usize,
        rec: Recurrent,
        s: mlx.mlx_stream,
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
        self.last_restored_used = null;
        self.last_restored_disk_id = null;
        if (skip) {
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

        // ── SSD tier: consult when it can beat the RAM match meaningfully
        // (fresh boot, post-eviction). Phase 3 handles hybrid targets too —
        // the tier persists per-position SSM checkpoints beside the KV chunks
        // and restores both.
        if (self.disk) |*d| disk: {
            if (rec == .glm) {
                const ram_eff: usize = if (match) |m| blk: {
                    const e = &self.entries.items[m.idx];
                    const cps = e.ssm_checkpoints orelse break :blk 0;
                    const cp = highestCheckpointAtOrBelow(cps, m.shared) orelse break :blk 0;
                    break :blk cp.pos;
                } else 0;
                if (self.restoreGlmFromDisk(d, rec.glm, target_cache, ram_eff, prompt_ids, has_tools, media_limits, dflash_target, s)) |res| {
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
                d.restoreIntoRinged(target_cache, rm.idx, rm.pos, final_len, s) catch |err| {
                    log.warn("  [disk-cache] ring restore failed: {s} — falling back to RAM/cold path\n", .{@errorName(err)});
                    poisonUnreadable(d, rm.idx, err);
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
                const restored = d.restoreIntoHybrid(target_cache, ssm_entries, hm.idx, disk_cp, s) catch |err| {
                    log.warn("  [disk-cache] hybrid restore failed: {s} — falling back to RAM/cold path\n", .{@errorName(err)});
                    poisonUnreadable(d, hm.idx, err);
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
                log.info("  [disk-cache] restored {d}/{d} tokens from SSD in {d}ms (ssm@{d})\n", .{ restored, prompt_ids.len, ms, disk_cp });
                const disk_mtp = diskRestoreSpec(d, hm.idx, mtp_target, restored, s, .mtp);
                return .{
                    .matched = restored,
                    .full_match = false,
                    // The spec sidecar of the entry the TRUNK came from: `dm` ranks by raw
                    // length, `hm` by restorable checkpoint, so they routinely differ.
                    .dflash_base = diskRestoreSpec(d, hm.idx, dflash_target, restored, s, .dflash),
                    .mtp_base = disk_mtp,
                    .slot_owned = true,
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
            d.restorePrefixInto(target_cache, dm.idx, @intCast(final_len), s) catch |err| {
                log.warn("  [disk-cache] restore failed: {s} — falling back to RAM/cold path\n", .{@errorName(err)});
                poisonUnreadable(d, dm.idx, err);
                // A failed restore can leave a half-rebuilt cache; reset it.
                target_cache.truncate(0, s) catch {};
                break :disk;
            };
            target_moe_seq_offset.* = final_len;
            const ms = sw.read() / std.time.ns_per_ms;
            log.info("  [disk-cache] restored {d}/{d} tokens from SSD ({d} chunks) in {d}ms\n", .{ final_len, prompt_ids.len, d.chunks_loaded_last, ms });
            const disk_mtp = diskRestoreSpec(d, dm.idx, mtp_target, final_len, s, .mtp);
            return .{
                .matched = final_len,
                .full_match = full_match,
                .dflash_base = diskRestoreSpec(d, dm.idx, dflash_target, final_len, s, .dflash),
                .mtp_base = disk_mtp,
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
        if (rec == .glm) {
            effective_matched = 0;
            if (e.glm) |*rows| if (e.ssm_checkpoints) |cps| if (highestCheckpointAtOrBelow(cps, @min(restore_cap, rows.len))) |cp| {
                try glm5_prefix.restore(rec.glm, rows, cp);
                effective_matched = cp.pos;
            };
            if (effective_matched == 0) rec.reset();
        } else if (rec.ssmEntries()) |entries| {
            if (e.ssm_checkpoints) |cps| {
                if (highestCoveringCheckpoint(cps, restore_cap)) |cp| {
                    try restoreSsmCheckpoint(entries, cp);
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
        // corrupts RoPE positions and CRASHES the Gemma sliding-window prefill mask
        // (`broadcast_shapes` mask-vs-KV mismatch; live 2026-07-09 on
        // gemma-4-26B-A4B at ~16K ctx). truncate is a no-op when the buffer is
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
            log.info("  [hot-cache] reused {d}/{d} tokens (matched {d}; entry {d}/{d})\n", .{ effective_matched, prompt_ids.len, m.shared, m.idx + 1, self.entries.items.len });
        }
        var res: LookupResult = .{
            .matched = matched,
            .full_match = full_match,
            .dflash_base = restoreDflash(e, dflash_target, matched, s),
            .mtp_base = restoreMtp(e, mtp_target, matched, s),
            .entry_id = e.id,
        };
        if (!full_reuse) {
            res.checked_out = self.checkoutIfEligible(m.idx, m.shared, prompt_ids.len, slot_id);
        }
        dump.pos = res.matched;
        dump.mtp_base = res.mtp_base;
        dump.entry_idx = m.idx + 1;
        dump.entry_count = self.entries.items.len;
        _ = restore_dump.dumpRestoreIfEnabled(target_cache, rec.ssmEntries(), s, dump);
        return res;
    }

    /// Restore by move, the decision: only on a full-prefix hit (the commit's replace path lands
    /// on this same entry) with something to append. A partial hit keeps the refcount share.
    pub fn checkoutEligible(
        ssd_first: bool,
        move_enabled: bool,
        pending_disk: bool,
        entry_tokens: usize,
        shared: usize,
        prompt_len: usize,
        has_slot: bool,
    ) bool {
        if (!ssd_first or !move_enabled or !has_slot) return false;
        // The pending disk record shares the same buffers, so a checkout could not donate anyway.
        if (pending_disk) return false;
        if (entry_tokens == 0 or shared != entry_tokens) return false;
        return prompt_len > shared;
    }

    /// Take the checkout: mark the entry as this slot's; `donateCheckout` releases the handles
    /// at the last moment before the slot's first write. Returns whether it was taken (the
    /// admission bill's input).
    fn checkoutIfEligible(self: *HotPrefixCache, idx: usize, shared: usize, prompt_len: usize, slot_id: ?usize) bool {
        const e = &self.entries.items[idx];
        if (!checkoutEligible(
            self.ssd_first,
            restoreMoveEnabled(),
            self.pending_disk != null,
            e.tokens.len,
            shared,
            prompt_len,
            slot_id != null,
        )) return false;
        e.checked_out_by = slot_id;
        log.info("  [hot-cache] checked out {d}-token entry to the slot (restore by move; the append donates in place)\n", .{e.tokens.len});
        return true;
    }

    /// Restore by move on demand: the admission pass found this slot's share does not fit (its first
    /// append copies the prefix). Takes, on the entry the slot just restored, the checkout
    /// `checkoutIfEligible` takes up front in SSD-first mode.
    pub fn checkoutRestored(self: *HotPrefixCache, slot_id: usize, prompt_len: usize) bool {
        const used = self.last_restored_used orelse return false;
        for (self.entries.items) |*e| {
            if (e.last_used != used) continue;
            if (e.checked_out_by != null) return false;
            if (!checkoutEligible(true, restoreMoveEnabled(), self.pending_disk != null, e.tokens.len, self.last_restored_shared, prompt_len, true)) return false;
            e.checked_out_by = slot_id;
            log.info("  [hot-cache] checked out {d}-token entry to the slot (the share does not fit; the append donates in place)\n", .{e.tokens.len});
            return true;
        }
        return false;
    }

    /// Restore by move, the transfer: give up the entry's own handles so the slot is the sole
    /// owner and its first `writeAtOffset` donates in place. Called right before
    /// `Generator.initWithOptions`, after the admission pass that can still refuse: releasing at
    /// restore time threw away a 364k session for a request that never ran. Idempotent.
    pub fn donateCheckout(self: *HotPrefixCache, slot_id: usize) void {
        for (self.entries.items) |*e| {
            if (e.checked_out_by != slot_id) continue;
            if (e.checkout_donated) continue;
            e.donated_bytes = snapshotBytes(&e.snapshot) + e.glm_bytes;
            e.snapshot.releaseHandles();
            if (e.glm) |*rows| rows.releaseHandles();
            e.checkout_donated = true;
        }
    }

    /// Bytes this cache still bills that `slot_id` now owns (its donated checkout). Inference thread only.
    pub fn donatedBytes(self: *const HotPrefixCache, slot_id: usize) u64 {
        var total: u64 = 0;
        for (self.entries.items) |*e| {
            if (e.checked_out_by == slot_id and e.checkout_donated) total += e.donated_bytes;
        }
        return total;
    }

    /// End of a slot's life. Not donated: the entry never gave its handles up, hand it back
    /// unchanged. Donated: the bytes die with the slot, drop the record. Reaching here with a
    /// mark set means the slot ended without committing. Idempotent.
    pub fn releaseCheckout(self: *HotPrefixCache, slot_id: usize, reason: []const u8) void {
        var i: usize = self.entries.items.len;
        while (i > 0) {
            i -= 1;
            const e = &self.entries.items[i];
            if (e.checked_out_by != slot_id) continue;
            const tokens_len = e.tokens.len;
            if (!e.checkout_donated) {
                // Nothing was handed over; only the mark is undone (no LRU bump for a request that never ran).
                e.checked_out_by = null;
                log.info("  [hot-cache] checked-out entry returned intact: {s} ({d} tokens; the append never ran)\n", .{ reason, tokens_len });
                continue;
            }
            // Clear the mark before `evictAt`; the snapshot is already empty.
            e.checked_out_by = null;
            e.checkout_donated = false;
            self.evictAt(i, "checked-out entry dropped");
            log.info("  [hot-cache] checked-out entry dropped: {s} ({d} tokens; its KV died with the slot)\n", .{ reason, tokens_len });
        }
    }

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

    /// A GLM commit: `cps` are its KDA checkpoints and `rows` the MLA rows below them, for both
    /// tiers; ownership of both transfers to the cache on every outcome.
    pub fn commitGlm(
        self: *HotPrefixCache,
        source_cache: *const KVCache,
        tokens: []const u32,
        has_tools: bool,
        vision_key: u64,
        cache_key: u64,
        media_start: ?usize,
        cps: []SSMCheckpoint,
        rows: glm5_prefix.MlaRows,
        dflash: ?DflashCommit,
        prompt_len: usize,
    ) !CommitStatus {
        return self.commitImpl(source_cache, tokens, has_tools, vision_key, cache_key, media_start, cps, dflash, null, prompt_len, .{}, .{ .ram = rows });
    }

    /// `commitGlm` with each tier's own rows: `ram_rows` within the RAM budget (`glmCommitLen`),
    /// `disk_rows` through the newest checkpoint the SSD tier keeps (`glmDiskLen`). Either may be
    /// null; `cps` cover the longer.
    pub fn commitGlmTiers(
        self: *HotPrefixCache,
        source_cache: *const KVCache,
        tokens: []const u32,
        has_tools: bool,
        vision_key: u64,
        cache_key: u64,
        media_start: ?usize,
        cps: []SSMCheckpoint,
        ram_rows: ?glm5_prefix.MlaRows,
        disk_rows: ?glm5_prefix.MlaRows,
        dflash: ?DflashCommit,
        prompt_len: usize,
    ) !CommitStatus {
        return self.commitImpl(source_cache, tokens, has_tools, vision_key, cache_key, media_start, cps, dflash, null, prompt_len, .{}, .{ .ram = ram_rows, .disk = disk_rows });
    }

    fn commitImpl(
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
        glm: GlmTierRows,
    ) !CommitStatus {
        // Freed on every path that does not move it into an entry.
        var new_rings = ownRingCps(self.allocator, ring_cps);
        defer if (new_rings) |r| freeRingCps(self.allocator, r);
        var new_ring_bytes = ringCpsBytes(new_rings);
        var new_glm = glm.ram;
        defer if (new_glm) |*rows| rows.deinit();
        var disk_glm = glm.disk;
        defer if (disk_glm) |*rows| rows.deinit();
        var new_glm_bytes: u64 = if (new_glm) |*rows| rows.bytes() else 0;
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
        if (self.capturesLiveForDisk() and disk_len > 0) {
            const rows: ?*const glm5_prefix.MlaRows = if (disk_glm) |*r| r else if (new_glm) |*r| r else null;
            self.capturePendingDisk(source_cache, tokens, disk_len, has_tools, eff_vision_key, eff_media_start, prompt_len, ssm_cps, dflash, mtp, new_rings, rows);
            // Whatever the RAM tier then does with the candidate, the record is flushed.
            if (self.pending_disk != null) self.disk_dirty = true;
        }
        // The record shares the live KV; on an error return nothing consumes it and the slot's
        // KVCache deinit then frees nothing. Function scope on purpose.
        errdefer if (self.pending_disk) |*p| {
            p.deinit(self.allocator);
            self.pending_disk = null;
        };

        // A GLM entry with no rows within the RAM budget keeps nothing in RAM.
        if (!self.ram_enabled or (glm.disk != null and new_glm == null)) {
            if (ssm_cps) |cps| {
                for (cps) |*cp| cp.deinit(self.allocator);
                self.allocator.free(cps);
            }
            if (self.pending_disk == null) return .declined;
            self.disk_dirty = true;
            return .{ .disk_only = tokens.len };
        }
        // The RAM entry keeps only the checkpoints its own rows restore.
        if (new_glm) |*rows| if (ssm_cps) |cps| {
            ssm_cps = glm5_prefix.keepThrough(self.allocator, cps, rows.len);
        };
        // The pending record holds every checkpoint for the SSD tier, so the hot entry keeps only its newest.
        // A GDN checkpoint past the record's text prefix is not in it and stays.
        const hot_keeps_newest = self.pending_disk != null and (new_glm != null or (glm.ram == null and glm.disk == null));
        const record_text: usize = if (new_glm != null) std.math.maxInt(usize) else disk_len;
        // GDN checkpoints are cut after the byte-budget trim below, which lands on one of them.
        if (hot_keeps_newest and new_glm != null) self.keepNewestCheckpoints(&ssm_cps, record_text);

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
        var new_snap = source_cache.snapshotRetained(mlx.gpuStream()) catch |err| {
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
                new_mtp = .{
                    .snapshot = snap,
                    .base_pos = m2.base_pos,
                    .head_aux = if (m2.head) |h| transformer_mod.ssmSnapshot(h) else null,
                    .head_pos_base = m2.head_pos_base,
                    .head_marks = transformer_mod.QsaHeadMarkSet.share(m2.head_marks),
                };
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
        var new_bytes = new_kv_bytes + new_ssm_bytes + new_dflash_bytes + new_mtp_bytes + new_ring_bytes + new_glm_bytes;
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
                const row_bytes = snapshotRowBytes(&new_snap) + (if (new_glm) |*rows| rows.rowBytes() else 0);
                const ring_bytes = ringedBytes(&new_snap);
                var ring_cp: ?*const KVCacheSnapshot = null;
                const tl_opt = if (ring_bytes > 0) blk: {
                    const rt = ringTrimLen(self.max_kv_bytes, limit, row_bytes, tokens.len, ring_bytes, new_rings orelse &.{}) orelse break :blk null;
                    if (rt.cp) |k| ring_cp = &new_rings.?[k];
                    break :blk rt.len;
                } else self.trimLenForBudget(self.max_kv_bytes -| (if (new_glm != null) new_dflash_bytes else 0), limit, row_bytes, eff_cps);
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
                    if (self.entries.items[idx].checked_out_by == null and
                        self.entries.items[idx].tokens.len >= tl)
                    {
                        // The resident entry already covers the trim target;
                        // the candidate's EXTRA tokens still belong on disk.
                        self.spillDeclinedToDisk(&new_snap, tokens, diskTextLen(tokens.len, eff_vision_key, eff_media_start), has_tools, MediaStamp.of(eff_vision_key, eff_media_start), eff_cps, new_rings, if (new_glm) |*rows| rows else null);
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
                const trimmed = new_snap.trimmedCopy(tl, ring_cp, mlx.gpuStream()) catch |err| {
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
                if (new_glm) |*rows| if (tl < rows.len) {
                    const short = rows.trimmedCopy(tl, mlx.gpuStream()) catch |err| {
                        decline = .snapshot_copy_failed;
                        decline_err = err;
                        break :trim_blk;
                    };
                    rows.deinit();
                    new_glm = short;
                    new_glm_bytes = short.bytes();
                };
                // Spec payloads describe the FULL-length state; a trimmed
                // prefix rebuilds them on its first reused turn. GLM's window is the prompt end's,
                // which a restore at any lower checkpoint clamps (`specAdoptPlan`).
                if (new_glm == null) if (new_dflash) |*d| {
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
                    var kept: usize = 0;
                    while (kept < cps.len and cps[kept].pos <= tl) kept += 1;
                    if (kept < cps.len) {
                        // The pooled indexer bank lives only on the latest snap (every snap
                        // carries its own leftover), so it is sliced onto the last KEPT snap
                        // before the tail is dropped.
                        if (kept > 0 and checkpointHasQsaPooled(&cps[cps.len - 1])) {
                            sliceQsaHistoryOntoCheckpoint(&cps[kept - 1], &cps[cps.len - 1], cps[kept - 1].pos, mlx.gpuStream()) catch |err| {
                                // Retaining a history-less hybrid entry makes every later
                                // restore a silent miss; decline and let the SSD tier take it.
                                decline = .qsa_history_slice_failed;
                                decline_err = err;
                                break :trim_blk;
                            };
                        }
                        const shrunk = self.allocator.dupe(SSMCheckpoint, cps[0..kept]) catch |err| {
                            decline = .checkpoint_list_copy_failed;
                            decline_err = err;
                            break :trim_blk;
                        };
                        for (cps[kept..]) |*cp| cp.deinit(self.allocator);
                        self.allocator.free(cps);
                        eff_cps = shrunk;
                    }
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
                new_bytes = new_kv_bytes + new_ssm_bytes + new_ring_bytes + new_glm_bytes + new_dflash_bytes;
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
                self.spillDeclinedToDisk(&new_snap, tokens, diskTextLen(tokens.len, eff_vision_key, eff_media_start), has_tools, MediaStamp.of(eff_vision_key, eff_media_start), eff_cps, new_rings, if (new_glm) |*rows| rows else null);
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
            const donor = self.bestCheckpointDonor(eff_tokens, has_tools, eff_vision_key, eff_media_start, quant_config) orelse
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
            const cloned = (cloneCheckpointsWithBank(self.allocator, donor_cps, inherit_limit, budget) catch |err| {
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
            const donor = self.bestRingDonor(eff_tokens, has_tools, eff_vision_key, eff_media_start, quant_config, cap) orelse
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
            e.kv_bytes = new_kv_bytes + merged_ssm_bytes + new_dflash_bytes + new_mtp_bytes + ring_bytes + new_glm_bytes;
            if (e.glm) |*rows| rows.deinit();
            e.glm = new_glm;
            e.glm_bytes = new_glm_bytes;
            new_glm = null;
            e.ssm_checkpoints = merged_cps;
            e.ssm_bytes = merged_ssm_bytes;
            e.dflash = new_dflash;
            e.dflash_bytes = new_dflash_bytes;
            e.mtp = new_mtp;
            e.mtp_bytes = new_mtp_bytes;
            // Restore by move: the replacement the checkout promised; the entry is whole again.
            e.checked_out_by = null;
            e.checkout_donated = false;
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
                    if (!self.evictOneLruProgress("byte budget", null)) break;
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
            .kv_bytes = new_bytes,
            .ssm_checkpoints = eff_cps,
            .ssm_bytes = new_ssm_bytes,
            .dflash = new_dflash,
            .dflash_bytes = new_dflash_bytes,
            .mtp = new_mtp,
            .mtp_bytes = new_mtp_bytes,
            .ring_cps = new_rings,
            .glm = new_glm,
            .glm_bytes = new_glm_bytes,
        };
        new_rings = null;
        new_glm = null;
        if (!try self.retainNewEntry(incoming)) return .declined;
        // The trim prices a prefix against the checkpoints that survive a shed, so the shed runs here too.
        if (self.max_kv_bytes > 0) self.shedCheckpointsToFit();
        if (self.disk != null) self.disk_dirty = true;
        self.logResident();
        return .{ .ok = eff_tokens.len };
    }

    fn retainNewEntry(self: *HotPrefixCache, entry: Entry) !bool {
        var incoming = entry;
        errdefer freeEntryOwnedState(self.allocator, &incoming);
        while (self.entries.items.len >= self.max_entries) {
            if (!self.evictOneLruProgress("count cap", incoming.cache_key)) break;
        }
        if (self.max_kv_bytes > 0) {
            while (self.current_kv_bytes + incoming.kv_bytes > self.max_kv_bytes and self.entries.items.len > 0) {
                if (!self.evictOneLruProgress("byte budget", incoming.cache_key)) break;
            }
        }
        if (self.entries.items.len >= self.max_entries) return self.declineIncoming(&incoming, "count cap");
        if (incoming.ssm_checkpoints) |cps| {
            if (self.takeCpsIfQsaBank(cps)) |kept| {
                incoming.ssm_checkpoints = kept;
            } else {
                incoming.kv_bytes -= incoming.ssm_bytes;
                incoming.ssm_bytes = 0;
                incoming.ssm_checkpoints = null;
            }
        }
        try self.entries.append(self.allocator, incoming);
        self.current_kv_bytes += incoming.kv_bytes;
        // A trim is priced against the checkpoints that survive the shed, so shed before judging.
        if (self.max_kv_bytes > 0 and self.current_kv_bytes > self.max_kv_bytes) {
            self.shedCheckpointsToFit();
            if (self.current_kv_bytes > self.max_kv_bytes) {
                var last = self.entries.pop().?;
                self.current_kv_bytes -|= last.kv_bytes;
                return self.declineIncoming(&last, "byte budget");
            }
        }
        return true;
    }

    fn declineIncoming(self: *HotPrefixCache, incoming: *Entry, cap: []const u8) bool {
        freeEntryOwnedState(self.allocator, incoming);
        log.info("  [hot-cache] skipped incoming snapshot: {s} full; checked-out entries prevent eviction\n", .{cap});
        return false;
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
    fn spillDeclinedToDisk(
        self: *HotPrefixCache,
        snap: *const KVCacheSnapshot,
        all_tokens: []const u32,
        text_len: usize,
        has_tools: bool,
        media: MediaStamp,
        all_cps: ?[]SSMCheckpoint,
        ring_cps: ?[]const KVCacheSnapshot,
        glm: ?*const glm5_prefix.MlaRows,
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
        const outcome = (if (glm) |rows| persistGlm(d, rows, tokens, has_tools, media, cps, null, false, mlx.gpuStream()) else d.appendCommitWith(.{
            .kv_entries = snap.entries,
            .step = snap.step,
            .config = snap.config,
            .tokens = tokens,
            .has_tools = has_tools,
            .ssm_checkpoints = cps,
            .ring = ringCommitOf(snap, ring_cps),
            .vision_key = media.key,
            .media_start = media.start,
        }, mlx.gpuStream(), .{})) catch |err| {
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
    fn capturePendingDisk(
        self: *HotPrefixCache,
        source_cache: *const KVCache,
        all_tokens: []const u32,
        text_len: usize,
        has_tools: bool,
        vision_key: u64,
        media_start: ?usize,
        prompt_len: usize,
        ssm_cps: ?[]SSMCheckpoint,
        all_dflash: ?DflashCommit,
        all_mtp: ?DflashCommit,
        ring_cps: ?[]const KVCacheSnapshot,
        glm: ?*const glm5_prefix.MlaRows,
    ) void {
        const tokens = all_tokens[0..text_len];
        // The spec snapshots cover rows past a cut record.
        const cut = text_len < all_tokens.len;
        const dflash = if (cut) null else all_dflash;
        const mtp = if (cut) null else all_mtp;
        // An earlier commit not flushed yet (two cancels culled in one pass) lands first.
        if (self.pending_disk != null) {
            if (self.disk) |*d| self.flushPendingRecord(d, mlx.gpuStream());
        }
        var snap = source_cache.snapshotRetained(mlx.gpuStream()) catch |err| {
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
        };
        if (ring_cps) |cps| rec.ring_cps = shareRingCpsUpTo(self.allocator, cps, std.math.maxInt(usize), null) catch null;
        if (glm) |rows| rec.glm = rows.share() catch {
            rec.deinit(self.allocator);
            return;
        };
        if (ssm_cps) |cps| {
            rec.ssm_cps = cloneCheckpointsWithBank(self.allocator, cps, text_len, null) catch null;
        }
        const quant = source_cache.config;
        const cap = @min(if (prompt_len == 0) text_len else prompt_len, text_len);
        if (self.bestCheckpointDonor(tokens, has_tools, vision_key, media_start, quant)) |donor| inherit: {
            const limit = @min(donor.shared, cap);
            const inherited = (cloneCheckpointsUpTo(self.allocator, self.entries.items[donor.idx].ssm_checkpoints.?, limit, null) catch break :inherit) orelse break :inherit;
            rec.ssm_cps = mergeDiskCheckpoints(self.allocator, inherited, rec.ssm_cps) catch null;
        }
        if (self.bestRingDonor(tokens, has_tools, vision_key, media_start, quant, cap)) |donor| inherit: {
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
                rec.mtp = .{
                    .snapshot = ms,
                    .base_pos = m2.base_pos,
                    .head_aux = if (m2.head) |h| transformer_mod.ssmSnapshot(h) else null,
                    .head_pos_base = m2.head_pos_base,
                    .head_marks = transformer_mod.QsaHeadMarkSet.share(m2.head_marks),
                };
            } else |_| {}
        }
        self.pending_disk = rec;
    }

    /// A ringed snapshot's restore points for the disk tier; null for a cache that does not ring.
    fn ringCommitOf(snap: *const KVCacheSnapshot, cps: ?[]const KVCacheSnapshot) ?kv_disk_cache.RingCommit {
        if (snap.swa_ring_window == 0) return null;
        return .{ .window = snap.swa_ring_window, .cps = cps orelse &.{} };
    }

    const EntrySpecs = struct {
        dflash: ?kv_disk_cache.SpecCommit = null,
        mtp: ?kv_disk_cache.SpecCommit = null,
    };

    /// The disk-tier spec payloads for one RAM entry; the flush and the idle spill must not drift.
    fn entrySpecCommits(e: *Entry) EntrySpecs {
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
    pub fn spillIdleEntries(self: *HotPrefixCache, s: mlx.mlx_stream) void {
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
            if (e.checked_out_by != null) continue;
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
            self.evictAt(idx, "SSD-first idle allowance");
            evicted += 1;
        }

        if (spilled > 0 or evicted > 0) {
            log.info("  [hot-cache] SSD-first: wrote {d} idle entries to disk, evicted {d}; RAM holds the active session + {d} MB idle allowance\n", .{
                spilled, evicted, self.ssd_idle_mem >> 20,
            });
        }
    }

    /// Least-recently-used idle entry (never the active session, never a vision entry).
    fn oldestIdleIndex(self: *HotPrefixCache, newest_used: u64, durable_only: bool) ?usize {
        var best: ?usize = null;
        var best_used: u64 = std.math.maxInt(u64);
        for (self.entries.items, 0..) |*e, i| {
            if (e.last_used == newest_used) continue;
            if (e.vision_key != 0) continue;
            if (e.checked_out_by != null) continue;
            if (durable_only and !e.spill_durable) continue;
            if (e.last_used < best_used) {
                best_used = e.last_used;
                best = i;
            }
        }
        return best;
    }

    /// Why pass 1 could not leave a durable copy of `e` (for the log).
    fn unpersistableReason(d: *kv_disk_cache.DiskTier, e: *const Entry) []const u8 {
        if (d.store_declined) return "store declined: volume is short";
        if (e.tokens.len < @as(usize, kv_disk_cache.MIN_PERSIST_TOKENS)) return "under the persist floor";
        const target = kv_disk_cache.persistTargetLen(e.snapshot.entries, e.snapshot.step, e.tokens.len);
        for (e.snapshot.entries) |*le| {
            if (le.initialized and le.offset < target) return "layer offset short of the range";
        }
        return "partial copy";
    }

    /// Flush the most recent commit to the SSD tier. Called by the scheduler
    /// AFTER `markFinished` (the client already has its response) — the
    /// chunk-append is bounded (partial tail + new chunks) but synchronous on
    /// the inference thread. Snapshot arrays are refcount-shared with the RAM
    /// entry, so slicing them here reads the same buffers the commit captured.
    /// A GLM entry goes to disk as its MLA rows in the dense disk layout beside its KDA checkpoints.
    /// `whole` writes every piece now, writer or not (`DiskTier.appendCommitWhole`).
    fn persistGlm(d: *kv_disk_cache.DiskTier, rows: *const glm5_prefix.MlaRows, tokens: []const u32, has_tools: bool, media: MediaStamp, cps: ?[]SSMCheckpoint, dflash: ?kv_disk_cache.SpecCommit, whole: bool, s: mlx.mlx_stream) !kv_disk_cache.PersistOutcome {
        const entries = try glm5_prefix.diskEntries(d.allocator, rows);
        defer glm5_prefix.freeDiskEntries(d.allocator, entries);
        const quant = glm5_prefix.diskQuant(rows.latent_bits);
        return d.appendCommitWith(.{
            .kv_entries = entries,
            .step = rows.len,
            .config = quant,
            .tokens = tokens,
            .has_tools = has_tools,
            .ssm_checkpoints = cps,
            .dflash_snap = dflash,
            .vision_key = media.key,
            .media_start = media.start,
        }, s, .{ .whole = whole });
    }

    /// GLM's prefill write-through: the rows through `request.offset` and the checkpoints so far go to the SSD tier
    /// as the prefill produces them, and every checkpoint but the newest leaves RAM. `entry_id` follows the entry
    /// across calls. Best effort: a failure leaves the checkpoints for the commit after the response.
    pub fn writeThroughGlm(
        self: *HotPrefixCache,
        entry_id: *?u64,
        request: *const glm5_forward.Request,
        prompt: []const u32,
        has_tools: bool,
        media: MediaStamp,
        cps: []SSMCheckpoint,
        s: mlx.mlx_stream,
    ) void {
        const d = if (self.disk) |*dd| dd else return;
        const len = prompt.len / glm5_prefix.pool_size * glm5_prefix.pool_size;
        // Below the tier's floor nothing is written yet, so no state may be freed.
        if (cps.len == 0 or len < kv_disk_cache.MIN_PERSIST_TOKENS) return;
        defer d.settleCheckpoints(entry_id.*, cps);
        var rows = glm5_prefix.MlaRows.shareLive(self.allocator, request, len, s) catch |err| {
            log.warn("  [disk-cache] GLM prefill write-through rows not shared: {s}\n", .{@errorName(err)});
            return;
        };
        defer rows.deinit();
        _ = persistGlm(d, &rows, prompt[0..len], has_tools, media, cps, null, false, s) catch |err| {
            log.warn("  [disk-cache] GLM prefill write-through failed: {s}\n", .{@errorName(err)});
            return;
        };
        if (d.last_store_id) |id| entry_id.* = id;
    }

    /// Qwen's prefill write-through: the rows and checkpoints so far go to the SSD tier as the prefill produces
    /// them, and every checkpoint but the newest leaves RAM once its file is written. Best effort: what a failed
    /// write could not land is released too, since the admission bill counts on the bound.
    pub fn writeThroughHybrid(
        self: *HotPrefixCache,
        kv_entries: []const transformer_mod.KVCacheEntry,
        config: kv_quant.KVQuantConfig,
        prompt: []const u32,
        has_tools: bool,
        media: MediaStamp,
        cps: []SSMCheckpoint,
        flush_bound: u64,
        s: mlx.mlx_stream,
    ) void {
        const d = if (self.disk) |*dd| dd else return;
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
        }, s, .{ .flush_bound = flush_bound }) catch |err| {
            log.warn("  [disk-cache] prefill write-through failed: {s}\n", .{@errorName(err)});
        };
        if (floor) d.settleCheckpoints(d.last_store_id, cps);
    }

    /// Write the pending record whole and release it: its rows may be a finished request's own
    /// buffers, which nothing could read again to resume a partial entry.
    fn flushPendingRecord(self: *HotPrefixCache, d: *kv_disk_cache.DiskTier, s: mlx.mlx_stream) void {
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
        _ = (if (pending.glm) |*rows| persistGlm(d, rows, pending.tokens, pending.has_tools, pending.media, pending.ssm_cps, p_dflash, true, s) else d.appendCommitWith(.{
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
        }, s, .{ .whole = true })) catch |err| {
            log.warn("  [disk-cache] persist failed: {s}\n", .{@errorName(err)});
        };
    }

    /// With both tiers on, a flushed entry keeps its newest `RING_CHECKPOINT_RAM_KEEP` ring
    /// checkpoints in RAM; each older one the SSD tier holds a file for is released, and a lookup
    /// that needs it restores from that file. A staged file counts: its bytes are the writer's, billed
    /// by its permit. One with no file in the index (a cut or failed staging) stays.
    fn releaseRingCpsToDisk(self: *HotPrefixCache) void {
        if (!self.ram_enabled) return;
        const d = if (self.disk) |*dd| dd else return;
        for (self.entries.items) |*e| {
            if (e.checked_out_by != null) continue;
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

    pub fn flushPendingDisk(self: *HotPrefixCache, s: mlx.mlx_stream) void {
        if (!self.disk_dirty) return;
        self.disk_dirty = false;
        const d = if (self.disk) |*dd| dd else return;
        defer self.releaseRingCpsToDisk();
        // Flush the live state captured at commit, not what the RAM entry retained after its trim.
        if (self.pending_disk != null) return self.flushPendingRecord(d, s);
        if (self.entries.items.len == 0) return;
        var newest: *Entry = &self.entries.items[0];
        for (self.entries.items[1..]) |*e| {
            if (e.last_used > newest.last_used) newest = e;
        }
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
        if (newest.glm) |*rows| {
            const done = persistGlm(d, rows, tokens, newest.has_tools, media, text.list, dflash_spec, false, s) catch |err| {
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
    /// `transformer.ssmCheckpointDropIndex`, shared with every other thinning site.
    fn mergeCheckpointLists(
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
            const drop = transformer_mod.ssmCheckpointDropIndex(merged.items, self.cp_thin, null);
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
    fn mergeDiskCheckpoints(allocator: std.mem.Allocator, inherited: []SSMCheckpoint, own_opt: ?[]SSMCheckpoint) ![]SSMCheckpoint {
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

    fn takeCpsIfQsaBank(self: *HotPrefixCache, cps: []SSMCheckpoint) ?[]SSMCheckpoint {
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
    fn bestCheckpointDonor(
        self: *const HotPrefixCache,
        tokens: []const u32,
        has_tools: bool,
        vision_key: u64,
        media_start: ?usize,
        quant_config: kv_quant.KVQuantConfig,
    ) ?struct { idx: usize, shared: usize } {
        var best_idx: ?usize = null;
        var best_shared: usize = 0;
        var best_pos: usize = 0;
        for (self.entries.items, 0..) |*e, i| {
            const cps = e.ssm_checkpoints orelse continue;
            const shared = donorShared(e, tokens, has_tools, vision_key, media_start, quant_config) orelse continue;
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
    fn donorShared(
        e: *const Entry,
        tokens: []const u32,
        has_tools: bool,
        vision_key: u64,
        media_start: ?usize,
        quant_config: kv_quant.KVQuantConfig,
    ) ?usize {
        if (e.has_tools != has_tools) return null;
        if (!std.meta.eql(e.quant_config, quant_config)) return null;
        var max_shared = @min(e.tokens.len, tokens.len);
        if (e.vision_key != vision_key) {
            // Same rule as `findBestRestorableMatch`: rows before the
            // earliest media placeholder are ordinary text and cross keys.
            const safe_boundary = crossKeyBoundary(e.vision_key, e.media_start, media_start, &.{}) orelse return null;
            max_shared = @min(max_shared, safe_boundary);
        }
        var shared: usize = 0;
        while (shared < max_shared and e.tokens[shared] == tokens[shared]) shared += 1;
        return shared;
    }

    const RingDonor = struct { idx: usize, limit: usize };

    /// The resident entry holding the highest ring checkpoint a commit of `tokens` may inherit:
    /// one whose rows are all shared tokens, at or below `cap` (the committing prompt's length).
    /// Returns that entry's index and the bound.
    fn bestRingDonor(
        self: *const HotPrefixCache,
        tokens: []const u32,
        has_tools: bool,
        vision_key: u64,
        media_start: ?usize,
        quant_config: kv_quant.KVQuantConfig,
        cap: usize,
    ) ?RingDonor {
        var best: ?RingDonor = null;
        var best_step: usize = 0;
        for (self.entries.items, 0..) |*e, i| {
            const cps = e.ring_cps orelse continue;
            const shared = donorShared(e, tokens, has_tools, vision_key, media_start, quant_config) orelse continue;
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
    fn shareRingCpsUpTo(allocator: std.mem.Allocator, src: []const KVCacheSnapshot, limit: usize, budget: ?u64) !?[]KVCacheSnapshot {
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

    fn ringStepOf(cp: *const KVCacheSnapshot) usize {
        return cp.step;
    }

    /// Thin an owned ascending list to `max_len` checkpoints and by at least `need_free` bytes,
    /// dropping where coverage loses least: the lowest (a shared preamble) and the newest stay
    /// longest. Consumes `cps` on every path; null when none survives.
    fn thinRingCps(allocator: std.mem.Allocator, cps: []KVCacheSnapshot, max_len: usize, need_free: u64) !?[]KVCacheSnapshot {
        errdefer freeRingCps(allocator, cps);
        var live = try std.ArrayList(KVCacheSnapshot).initCapacity(allocator, cps.len);
        defer live.deinit(allocator);
        live.appendSliceAssumeCapacity(cps);
        var freed: u64 = 0;
        while (live.items.len > 0 and (live.items.len > max_len or freed < need_free)) {
            const k = transformer_mod.spanPreservingDropIndex(KVCacheSnapshot, live.items, ringStepOf, .min_span_recency, null);
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
    fn mergeRingCps(allocator: std.mem.Allocator, old: ?[]KVCacheSnapshot, new: ?[]KVCacheSnapshot) !?[]KVCacheSnapshot {
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
    fn ownRingCps(allocator: std.mem.Allocator, cps: SlotRingCps) ?[]KVCacheSnapshot {
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
    fn ringCpsUpTo(allocator: std.mem.Allocator, cps: []KVCacheSnapshot, limit: usize) ?[]KVCacheSnapshot {
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

    fn freeRingCps(allocator: std.mem.Allocator, cps: []KVCacheSnapshot) void {
        for (cps) |*c| c.deinit();
        allocator.free(cps);
    }

    fn ringCpsBytes(cps: ?[]const KVCacheSnapshot) u64 {
        var total: u64 = 0;
        for (cps orelse return 0) |*c| total += snapshotBytes(c);
        return total;
    }

    /// `cloneCheckpointsUpTo`, plus the pooled indexer bank: it lives on the newest checkpoint,
    /// which a `limit` below it drops, so it is sliced onto the last one kept.
    fn cloneCheckpointsWithBank(allocator: std.mem.Allocator, src: []const SSMCheckpoint, limit: usize, budget: ?u64) !?[]SSMCheckpoint {
        const cloned = (try cloneCheckpointsUpTo(allocator, src, limit, budget)) orelse return null;
        var bank: ?*const SSMCheckpoint = null;
        for (src) |*cp| {
            if (checkpointHasQsaPooled(cp)) bank = cp;
        }
        const b = bank orelse return cloned;
        if (b.pos <= limit) return cloned;
        const last = &cloned[cloned.len - 1];
        sliceQsaHistoryOntoCheckpoint(last, b, last.pos, mlx.gpuStream()) catch |err| {
            for (cloned) |*c| c.deinit(allocator);
            allocator.free(cloned);
            return err;
        };
        return cloned;
    }

    /// The checkpoints a record cut to `len` rows persists. `.list` is owned only when `cut`.
    const TextCps = struct {
        list: ?[]SSMCheckpoint,
        cut: bool,

        fn deinit(self: TextCps, allocator: std.mem.Allocator) void {
            if (!self.cut) return;
            const l = self.list orelse return;
            for (l) |*cp| cp.deinit(allocator);
            allocator.free(l);
        }
    };

    fn textCheckpoints(self: *const HotPrefixCache, cps: ?[]SSMCheckpoint, len: usize, tokens_len: usize) ?TextCps {
        if (len >= tokens_len) return .{ .list = cps, .cut = false };
        const src = cps orelse return .{ .list = null, .cut = true };
        const cloned = cloneCheckpointsWithBank(self.allocator, src, len, null) catch |err| {
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
    fn cloneCheckpointsUpTo(
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
            out.appendAssumeCapacity(try transformer_mod.shareSsmCheckpoint(allocator, cp));
        }
        if (out.items.len == 0) {
            out.deinit(allocator);
            return null;
        }
        // Collected newest-first; the list contract is ascending by pos.
        std.mem.reverse(SSMCheckpoint, out.items);
        return try out.toOwnedSlice(allocator);
    }

    /// Frees all but the newest `glm5_prefix.checkpoint_hot_kept` of an ascending checkpoint list. The older
    /// ones are restored from the SSD tier, which holds them all; one past `text_len` is not in the SSD record
    /// and stays.
    fn keepNewestCheckpoints(self: *HotPrefixCache, list: *?[]SSMCheckpoint, text_len: usize) void {
        const cps = list.* orelse return;
        const newest: usize = glm5_prefix.checkpoint_hot_kept;
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

    fn shedCheckpointsToFit(self: *HotPrefixCache) void {
        if (self.max_kv_bytes == 0 or self.current_kv_bytes <= self.max_kv_bytes) return;
        if (self.entries.items.len == 0) return;
        var newest: *Entry = &self.entries.items[0];
        for (self.entries.items[1..]) |*e| {
            if (e.last_used > newest.last_used) newest = e;
        }
        self.shedRingCheckpoints(newest);
        const cps = newest.ssm_checkpoints orelse return;
        var n = cps.len;
        var shed: usize = 0;
        while (n > 1 and self.current_kv_bytes > self.max_kv_bytes) {
            const drop = transformer_mod.ssmCheckpointDropIndex(cps[0..n], self.cp_thin, null);
            const freed = ssmCheckpointBytes(&cps[drop]);
            if (checkpointHasQsaPooled(&cps[drop])) {
                if (drop > 0) {
                    sliceQsaHistoryOntoCheckpoint(&cps[drop - 1], &cps[drop], cps[drop - 1].pos, mlx.gpuStream()) catch |err| {
                        log.warn("  [hot-cache] shed dropped the QSA indexer history: {s}\n", .{@errorName(err)});
                    };
                }
            }
            cps[drop].deinit(self.allocator);
            var k = drop;
            while (k + 1 < n) : (k += 1) cps[k] = cps[k + 1];
            n -= 1;
            shed += 1;
            newest.ssm_bytes -|= freed;
            newest.kv_bytes -|= freed;
            self.current_kv_bytes -|= freed;
        }
        if (shed == 0) return;
        // Shrink-in-place realloc cannot practically fail; if it somehow
        // does, the deinit'd tail stubs (pos 0, zero layers) stay in the
        // slice — `highestCheckpointAtOrBelow` skips empty checkpoints and a
        // re-deinit of a stub is a no-op, so they are inert.
        newest.ssm_checkpoints = self.allocator.realloc(cps, n) catch cps;
        if (self.takeCpsIfQsaBank(newest.ssm_checkpoints.?)) |kept| {
            newest.ssm_checkpoints = kept;
            n = kept.len;
        } else {
            const leftover = newest.ssm_bytes;
            newest.ssm_checkpoints = null;
            newest.ssm_bytes = 0;
            newest.kv_bytes -|= leftover;
            self.current_kv_bytes -|= leftover;
            n = 0;
        }
        log.info("  [hot-cache] shed {d} checkpoints to fit the byte budget ({d} kept)\n", .{ shed, n });
    }

    /// While the cache is over its budget, the way `thinRingCps` drops them.
    fn shedRingCheckpoints(self: *HotPrefixCache, e: *Entry) void {
        const cps = e.ring_cps orelse return;
        const over = self.current_kv_bytes -| self.max_kv_bytes;
        if (over == 0) return;
        const before_n = cps.len;
        const before = ringCpsBytes(cps);
        e.ring_cps = null;
        e.ring_cps = thinRingCps(self.allocator, cps, RING_CHECKPOINT_MAX, over) catch null;
        const freed = before - ringCpsBytes(e.ring_cps);
        e.kv_bytes -|= freed;
        self.current_kv_bytes -|= freed;
        const kept = if (e.ring_cps) |r| r.len else 0;
        log.info("  [hot-cache] shed {d} ring checkpoints to fit the byte budget ({d} kept)\n", .{ before_n - kept, kept });
    }

    /// Re-clamp the byte budget after the machine's residency changed (a model loaded
    /// or unloaded beside this one). Shrinking evicts LRU down to the new cap; 0 = uncapped.
    pub fn setBudget(self: *HotPrefixCache, max_kv_bytes: u64) void {
        if (max_kv_bytes == self.max_kv_bytes) return;
        if (max_kv_bytes >> 20 != self.max_kv_bytes >> 20)
            log.info("  [hot-cache] budget revised {d} -> {d} MB\n", .{ self.max_kv_bytes >> 20, max_kv_bytes >> 20 });
        const shrank = max_kv_bytes != 0 and (self.max_kv_bytes == 0 or max_kv_bytes < self.max_kv_bytes);
        self.max_kv_bytes = max_kv_bytes;
        if (!shrank) return;
        while (self.current_kv_bytes > max_kv_bytes and self.entries.items.len > 0) {
            if (!self.evictOneLruProgress("budget revised", null)) break;
        }
        self.shedCheckpointsToFit();
        self.logResident();
    }

    fn evictOneLru(self: *HotPrefixCache, reason: []const u8, incoming_key: ?u64) void {
        const idx = self.lruIndexExcluding(null, incoming_key) orelse return;
        self.evictAt(idx, reason);
    }

    /// False when nothing was evictable (every survivor is checked out) — the termination
    /// condition every budget loop needs, since `evictOneLru` is then a no-op.
    fn evictOneLruProgress(self: *HotPrefixCache, reason: []const u8, incoming_key: ?u64) bool {
        const before = self.entries.items.len;
        self.evictOneLru(reason, incoming_key);
        return self.entries.items.len != before;
    }

    pub fn dropLastRestored(self: *HotPrefixCache) bool {
        var dropped = false;
        if (self.last_restored_disk_id) |id| {
            self.last_restored_disk_id = null;
            if (self.disk) |*d| {
                if (d.poisonId(id, "QSA history check")) dropped = true;
            }
        }
        const used = self.last_restored_used orelse return dropped;
        for (self.entries.items, 0..) |*e, i| {
            if (e.last_used != used) continue;
            self.last_restored_used = null;
            self.evictAt(i, "QSA history check");
            return true;
        }
        self.last_restored_used = null;
        return dropped;
    }

    pub fn dropQsaGapEntry(slot_cache: ?*HotPrefixCache) bool {
        if (slot_cache) |hc| return hc.dropLastRestored();
        return false;
    }

    fn evictAt(self: *HotPrefixCache, lru_idx: usize, reason: []const u8) void {
        var evicted = self.entries.swapRemove(lru_idx);
        const tokens_len = evicted.tokens.len;
        const kv_mb = @as(f64, @floatFromInt(evicted.kv_bytes)) / (1024.0 * 1024.0);
        const had_ssm = evicted.ssm_checkpoints != null;
        const ssm_mb = @as(f64, @floatFromInt(evicted.ssm_bytes)) / (1024.0 * 1024.0);
        const key = evicted.cache_key;
        self.current_kv_bytes -|= evicted.kv_bytes;
        freeEntryOwnedState(self.allocator, &evicted);
        if (had_ssm) {
            log.info("  [hot-cache] evicted LRU entry ({s}; key={x}; was {d} tokens, {d:.2} MB; ssm {d:.2} MB)\n", .{
                reason, key, tokens_len, kv_mb, ssm_mb,
            });
        } else {
            log.info("  [hot-cache] evicted LRU entry ({s}; key={x}; was {d} tokens, {d:.2} MB)\n", .{
                reason, key, tokens_len, kv_mb,
            });
        }
    }

    /// Does a commit record the live state for the SSD tier (`pending_disk`)? Wherever the flush
    /// lands it whole: SSD-first, or a tier with the background writer.
    fn capturesLiveForDisk(self: *const HotPrefixCache) bool {
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

    /// Bytes an eviction pass can prove it will get back: the residency minus the largest
    /// single entry (a restore pins at most one, and the guard cannot know which).
    pub fn reclaimableBytes(self: *const HotPrefixCache) u64 {
        var largest: u64 = 0;
        // A checked-out entry is neither restorable nor evictable; it comes off the base.
        var checked_out: u64 = 0;
        for (self.entries.items) |*e| {
            if (e.checked_out_by != null) {
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
            if (e.checked_out_by != null) continue;
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
            if (e.checked_out_by != null) {
                checked_out += e.kv_bytes;
                continue;
            }
            const max_shared = @min(e.tokens.len, prompt_tokens.len);
            var shared: usize = 0;
            while (shared < max_shared and e.tokens[shared] == prompt_tokens[shared]) shared += 1;
            if (shared < MIN_CANCELLED_COMMIT_TOKENS) continue;
            pinned = @max(pinned, e.kv_bytes);
        }
        return self.current_kv_bytes -| checked_out -| pinned;
    }

    /// Evict least-recently-used entries until `fits()` says the request fits (#353): a cached
    /// prefix is an optimization, the request is the work. Never evicts the entry this request
    /// restored from; `fits` is re-asked after every eviction.
    /// Smallest eviction whose live/billed ratio is meaningful.
    pub const SHARED_RATIO_MIN_BYTES: u64 = 1 << 20;

    /// An eviction returning less than 1/Nth of what the entry was billed gave the allocator
    /// nothing (refcount-shared with a live cache). A ratio, not a floor: a small exclusive
    /// entry also returns little.
    pub const SHARED_RETURN_DIVISOR: u64 = 4;

    /// A restore is a lien on the whole entry: `KVCache.restore` shares the entire allocation
    /// and `truncate` frees nothing, so a slot that matched 11 tokens of a 524k entry held all
    /// 11.5 GB and the admission pass could not touch it. A restore may not pin an entry that
    /// hands back less than 1/`RESTORE_PIN_RATIO` of what it pins; small entries never reach the test.
    pub const RESTORE_PIN_MIN_BYTES: u64 = 1 << 30;
    pub const RESTORE_PIN_RATIO: usize = 64;

    /// Would restoring `shared` rows from this entry be a lien rather than a hit?
    pub fn restoreWouldPinEntry(kv_bytes: u64, min_bytes: u64, entry_tokens: usize, shared: usize) bool {
        if (kv_bytes < min_bytes) return false;
        if (shared == 0) return true;
        return entry_tokens > shared *| RESTORE_PIN_RATIO;
    }

    pub fn evictLruToAdmit(
        self: *HotPrefixCache,
        seq_tokens: u64,
        ctx: ?*anyopaque,
        fits: *const fn (?*anyopaque) bool,
        protect_restored: bool,
    ) EvictionReport {
        var report = EvictionReport{};
        // A command buffer in flight keeps its inputs' buffers until it completes: drain the
        // stream so `fits` and every live delta below read settled memory.
        _ = mlx.mlx_synchronize(mlx.gpuStream());
        while (!fits(ctx)) {
            const idx = self.lruIndexExcluding(if (protect_restored) self.last_restored_used else null, null) orelse break;
            // Accounting bytes are what the entry was billed; live bytes are what the allocator got back.
            var live_before: usize = 0;
            _ = mlx.mlx_get_active_memory(&live_before);
            const acct_before = self.current_kv_bytes;
            self.evictAt(idx, "admitting a long prefill");
            var live_after: usize = 0;
            _ = mlx.mlx_get_active_memory(&live_after);
            const freed_live: u64 = @as(u64, live_before) -| @as(u64, live_after);
            report.entries += 1;
            report.bytes += freed_live;
            const acct_delta = acct_before -| self.current_kv_bytes;
            report.accounted_bytes += acct_delta;
            // Judge only entries big enough for the ratio to mean something.
            if (acct_delta >= SHARED_RATIO_MIN_BYTES and
                freed_live * SHARED_RETURN_DIVISOR < acct_delta)
            {
                report.shared_stop = true;
                break;
            }
        }
        report.admitted = fits(ctx);
        if (report.entries > 0) {
            log.info("  [hot-cache] evicted {d} entries ({d} MB live, {d} MB billed) to admit a {d}-token prefill{s}\n", .{
                report.entries,
                report.bytes / (1024 * 1024),
                report.accounted_bytes / (1024 * 1024),
                seq_tokens,
                if (report.shared_stop) " — stopped: the next entry is shared with a live request" else "",
            });
        }
        return report;
    }

    /// Least-recently-used entry index, skipping the one whose `last_used` equals `protect`.
    /// Workload-fair: only entries of the `cache_key` holding the most eligible entries
    /// (`incoming_key` counts as one more) are candidates, so a sweep evicts its own
    /// documents before another workload's conversation. One key = plain LRU.
    fn lruIndexExcluding(self: *const HotPrefixCache, protect: ?u64, incoming_key: ?u64) ?usize {
        var best: ?usize = null;
        var best_used: u64 = std.math.maxInt(u64);
        var max_count: usize = 0;
        for (self.entries.items, 0..) |*e, i| {
            // Held by a live slot that owns its buffers: evicting it frees nothing.
            if (e.checked_out_by != null) continue;
            if (protect) |p| {
                if (e.last_used == p) continue;
            }
            var count: usize = if (incoming_key != null and incoming_key.? == e.cache_key) 1 else 0;
            for (self.entries.items) |*o| {
                if (o.checked_out_by != null or o.cache_key != e.cache_key) continue;
                if (protect) |p| {
                    if (o.last_used == p) continue;
                }
                count += 1;
            }
            if (count > max_count or (count == max_count and e.last_used < best_used)) {
                max_count = count;
                best_used = e.last_used;
                best = i;
            }
        }
        return best;
    }

    fn logResident(self: *const HotPrefixCache) void {
        const mb = @as(f64, @floatFromInt(self.current_kv_bytes)) / (1024.0 * 1024.0);
        if (self.max_kv_bytes == 0) {
            log.info("  [hot-cache] resident={d:.2} MB ({d}/{d} entries)\n", .{ mb, self.entries.items.len, self.max_entries });
        } else {
            const cap_mb = @as(f64, @floatFromInt(self.max_kv_bytes)) / (1024.0 * 1024.0);
            log.info("  [hot-cache] resident={d:.2} / {d:.2} MB ({d}/{d} entries)\n", .{ mb, cap_mb, self.entries.items.len, self.max_entries });
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
        var newest_idx: usize = 0;
        var newest_used: u64 = 0;
        for (self.entries.items, 0..) |*e, i| {
            if (e.last_used >= newest_used) {
                newest_used = e.last_used;
                newest_idx = i;
            }
        }
        var evicted = self.entries.swapRemove(newest_idx);
        self.current_kv_bytes -|= evicted.kv_bytes;
        freeEntryOwnedState(self.allocator, &evicted);
        log.info("  [hot-cache] invalidated latest entry: {s}\n", .{reason});
    }

    pub fn entryCount(self: *const HotPrefixCache) usize {
        return self.entries.items.len;
    }
};

// ── Tests ──

const testing = std.testing;

test "HotPrefixCache: shouldUse gates hybrid by enable_ssm_checkpoints" {
    var cfg = model_mod.ModelConfig{};
    // Plain attention: always allowed.
    try testing.expect(HotPrefixCache.shouldUse(&cfg, false));
    try testing.expect(HotPrefixCache.shouldUse(&cfg, true));
    // Hybrid (lfm2/nemotron_h-style): only with checkpoints enabled.
    cfg.has_hybrid_layers = true;
    try testing.expect(!HotPrefixCache.shouldUse(&cfg, false));
    try testing.expect(HotPrefixCache.shouldUse(&cfg, true));
    // Qwen3.5-style full_attention_interval-marks-hybrid: same gate.
    cfg.has_hybrid_layers = false;
    cfg.full_attention_interval = 4;
    try testing.expect(!HotPrefixCache.shouldUse(&cfg, false));
    try testing.expect(HotPrefixCache.shouldUse(&cfg, true));
}

test "HotPrefixCache: shouldUse rejects deepseek_v4 (module-owned decode state)" {
    // dsv4's per-request state (raw-kv rings, compressed caches, compressor
    // pending windows) lives on the Dsv4Model, NOT in the 0-entry KVCache
    // shell — a snapshot restore would set cache.step without rebuilding that
    // state, silently serving a stale ring (or crashing on a null dec_state).
    var cfg = model_mod.ModelConfig{};
    cfg.model_type = "deepseek_v4";
    try testing.expect(!HotPrefixCache.shouldUse(&cfg, false));
    try testing.expect(!HotPrefixCache.shouldUse(&cfg, true));
}

test "HotPrefixCache: zero capacity disables RAM retention" {
    var cache = HotPrefixCache.init(testing.allocator, 0);
    defer cache.deinit();
    try testing.expectEqual(@as(u32, 0), cache.max_entries);
    try testing.expect(!cache.ram_enabled);
    try testing.expectEqual(@as(usize, 0), cache.entryCount());
}

test "HotPrefixCache: findBestMatch returns longest shared prefix" {
    var cache = HotPrefixCache.init(testing.allocator, 4);
    defer cache.deinit();

    // Two synthetic entries (snapshots are no-ops on freshly-zero KVCache; we
    // never restore in this unit test, so no GPU work).
    const ids_a = try testing.allocator.dupe(u32, &[_]u32{ 1, 2, 3, 4, 5 });
    const ids_b = try testing.allocator.dupe(u32, &[_]u32{ 1, 2, 3, 9, 9, 9 });
    try cache.entries.append(testing.allocator, .{
        .tokens = ids_a,
        .has_tools = false,
        .snapshot = .{ .entries = try testing.allocator.alloc(transformer_mod.KVCacheEntry, 0), .step = 0, .allocator = testing.allocator, .config = transformer_mod.KVQuantConfig.dense },
        .last_used = 1,
        .quant_config = kv_quant.KVQuantConfig.dense,
        .kv_bytes = 0,
        .ssm_checkpoints = null,
        .ssm_bytes = 0,
    });
    try cache.entries.append(testing.allocator, .{
        .tokens = ids_b,
        .has_tools = false,
        .snapshot = .{ .entries = try testing.allocator.alloc(transformer_mod.KVCacheEntry, 0), .step = 0, .allocator = testing.allocator, .config = transformer_mod.KVQuantConfig.dense },
        .last_used = 2,
        .quant_config = kv_quant.KVQuantConfig.dense,
        .kv_bytes = 0,
        .ssm_checkpoints = null,
        .ssm_bytes = 0,
    });

    // Looking up [1,2,3,4,5,6] should match entry A (5 shared tokens).
    const lookup_ids = [_]u32{ 1, 2, 3, 4, 5, 6 };
    const m = cache.findBestMatch(&lookup_ids, false, 0, kv_quant.KVQuantConfig.dense).?;
    try testing.expectEqual(@as(usize, 0), m.idx);
    try testing.expectEqual(@as(usize, 5), m.shared);

    // Looking up [1,2,3,9,9,9,7] should match entry B (6 shared).
    const lookup_ids2 = [_]u32{ 1, 2, 3, 9, 9, 9, 7 };
    const m2 = cache.findBestMatch(&lookup_ids2, false, 0, kv_quant.KVQuantConfig.dense).?;
    try testing.expectEqual(@as(usize, 1), m2.idx);
    try testing.expectEqual(@as(usize, 6), m2.shared);

    // has_tools mismatch returns null.
    try testing.expectEqual(@as(?@TypeOf(m), null), cache.findBestMatch(&lookup_ids, true, 0, kv_quant.KVQuantConfig.dense));
    // vision_key mismatch returns null both ways: a text entry never serves an
    // image request and an image entry only serves the same pixels.
    try testing.expectEqual(@as(?@TypeOf(m), null), cache.findBestMatch(&lookup_ids, false, 7, kv_quant.KVQuantConfig.dense));
    cache.entries.items[0].vision_key = 7;
    // Text lookup falls through to entry B (3 shared); the keyed lookup gets A.
    try testing.expectEqual(@as(usize, 1), cache.findBestMatch(&lookup_ids, false, 0, kv_quant.KVQuantConfig.dense).?.idx);
    try testing.expectEqual(@as(usize, 0), cache.findBestMatch(&lookup_ids, false, 7, kv_quant.KVQuantConfig.dense).?.idx);
    cache.entries.items[0].vision_key = 0;
    // Scheme mismatch returns null — entries are dense, a query for affine
    // 4-bit cannot match (Wave 1.A: cross-scheme cache hits never happen).
    try testing.expectEqual(@as(?@TypeOf(m), null), cache.findBestMatch(&lookup_ids, false, 0, kv_quant.KVQuantConfig.affine(4)));
}

test "HotPrefixCache: an entry holding the request's first images restores up to the new one" {
    // An agent turn appends a screenshot to a conversation that already holds
    // one: the previous turn's entry carries the first image's rows, keyed by
    // that image alone, and serves everything before the new image.
    var cache = HotPrefixCache.init(testing.allocator, 4);
    defer cache.deinit();
    const P = 99;
    const chain = [_]MediaSpan{ .{ .start = 3, .key = 0xA1 }, .{ .start = 8, .key = 0xB3 } };
    const request = [_]u32{ 1, 2, 3, P, P, 5, 6, 7, P, P, 9 };
    for ([_]u64{ 0xA1, 0xF0 }, 0..) |key, i| {
        try cache.entries.append(testing.allocator, .{
            .tokens = try testing.allocator.dupe(u32, &[_]u32{ 1, 2, 3, P, P, 5, 6, 7, 4 }),
            .has_tools = false,
            .vision_key = key,
            .media_start = 3,
            .snapshot = .{ .entries = try testing.allocator.alloc(transformer_mod.KVCacheEntry, 0), .step = 0, .allocator = testing.allocator, .config = transformer_mod.KVQuantConfig.dense },
            .last_used = i,
            .quant_config = kv_quant.KVQuantConfig.dense,
            .kv_bytes = 0,
            .ssm_checkpoints = null,
            .ssm_bytes = 0,
        });
    }
    const dense = kv_quant.KVQuantConfig.dense;
    const m = cache.findBestRestorableMatch(&request, false, chainKey(&chain), 3, &chain, dense, false, null).?;
    try testing.expectEqual(@as(usize, 0), m.idx);
    try testing.expectEqual(@as(usize, 8), m.shared);

    // Foreign pixels under the first placeholder share only the text before it.
    cache.entries.items[0].vision_key = 0xA2;
    try testing.expectEqual(@as(usize, 3), cache.findBestRestorableMatch(&request, false, chainKey(&chain), 3, &chain, dense, false, null).?.shared);
    // Without the chain the old rule holds: the first media row bounds it.
    cache.entries.items[0].vision_key = 0xA1;
    try testing.expectEqual(@as(usize, 3), cache.findBestRestorableMatch(&request, false, chainKey(&chain), 3, &.{}, dense, false, null).?.shared);
}

test "HotPrefixCache: restore clamps an inflated snapshot to the matched length (gemma mask crash)" {
    // Root cause of the live gemma-4-26B-A4B crash (2026-07-09, broadcast_shapes
    // mask 16890 vs KV 16892 at ~16K ctx): a snapshot committed with a KV buffer
    // LONGER than its logical token count — PLD/speculative decode leaves stale
    // draft positions in the buffer past the committed step. When the NEXT prompt
    // matches the entry's ENTIRE token sequence but is longer (a partial hit,
    // effective_matched == e.tokens.len < prompt_ids.len), the old truncate guard
    // `final_len < e.tokens.len` was FALSE, so the restored cache offset kept the
    // inflated snapshot length — drifting ahead of the matched length generation
    // tracks. That drift corrupts RoPE and crashes the sliding-window prefill mask.
    const s = mlx.gpuStream();

    var toks: [67]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 11);
    const logical_len: usize = 64;

    // Source cache: 64 logical tokens, then 2 STALE tokens (offset 66) — the
    // shape a PLD round leaves behind before commit.
    var src = try KVCache.init(testing.allocator, 2);
    defer src.deinit();
    try testFillCache(&src, s, 2, @intCast(logical_len));
    try testFillCache(&src, s, 2, 2); // stale draft tail → offset 66
    try testing.expectEqual(@as(usize, 66), src.step);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    // Commit with the LOGICAL token count (64) — but the snapshot carries 66.
    _ = try hc.commit(&src, toks[0..logical_len], false);

    // Reuse with a prompt that matches all 64 entry tokens but is LONGER (67):
    // effective_matched == e.tokens.len == 64 < prompt_ids.len — the crash path.
    var dst = try KVCache.init(testing.allocator, 2);
    defer dst.deinit();
    var moe_off: usize = 0;
    const res = try hc.lookupAndRestore(&dst, &moe_off, null, s, &toks, false, 0, null, null);

    try testing.expect(!res.full_match);
    try testing.expectEqual(@as(usize, 64), res.matched);
    // The invariant: restored cache offset == matched length, NOT the inflated 66.
    try testing.expectEqual(@as(usize, 64), moe_off);
    try testing.expectEqual(@as(usize, 64), dst.step);
    for (dst.entries) |*e| {
        try testing.expect(e.initialized);
        try testing.expectEqual(@as(usize, 64), e.offset); // clamped, not 66
    }
}

test "prefix cache: DFlash assistant context round-trips, clamped to the trunk's matched length" {
    const s = mlx.gpuStream();
    var toks: [64]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 5);

    // Trunk KV for 64 tokens; the assistant context covers the same span but
    // starts at 10 (the committing request was itself a partial cache hit).
    var trunk = try KVCache.init(testing.allocator, 2);
    defer trunk.deinit();
    try testFillCache(&trunk, s, 2, 64);
    var assist = try KVCache.init(testing.allocator, 2);
    defer assist.deinit();
    try testFillCache(&assist, s, 2, 54);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    _ = try hc.commitWithSsm(&trunk, &toks, false, null, .{ .cache = &assist, .base_pos = 10 }, null);
    // The assistant context is billed like SSM state, so the memory cap sees it.
    try testing.expect(hc.entries.items[0].dflash_bytes > 0);
    try testing.expect(hc.entries.items[0].kv_bytes > hc.entries.items[0].dflash_bytes);

    // A shorter prompt that the entry fully covers: the full-match path
    // re-forwards the last token, so the trunk lands at 31 and the assistant
    // context must be clamped to 31-10 = 21 — absLen == matched, or the first
    // round's `dctx.absLen() == anchor_pos` assert fires.
    var dst = try KVCache.init(testing.allocator, 2);
    defer dst.deinit();
    var dfl = try KVCache.init(testing.allocator, 2);
    defer dfl.deinit();
    var moe_off: usize = 0;
    var base: usize = 0;
    const res = try hc.lookupAndRestore(
        &dst,
        &moe_off,
        null,
        s,
        toks[0..32],
        false,
        0,
        .{ .cache = &dfl, .base_pos = &base },
        null,
    );
    try testing.expect(res.full_match);
    try testing.expectEqual(@as(usize, 31), res.matched);
    try testing.expectEqual(@as(?usize, 10), res.dflash_base);
    try testing.expectEqual(@as(usize, 10), base);
    try testing.expectEqual(@as(usize, 21), dfl.step);
    try testing.expectEqual(base + dfl.step, res.matched);

    // An entry with no assistant payload leaves the target untouched, and the
    // caller is told so — a blind start is a valid outcome, never a wrong one.
    var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc2.deinit();
    _ = try hc2.commit(&trunk, &toks, false);
    var dst2 = try KVCache.init(testing.allocator, 2);
    defer dst2.deinit();
    var dfl2 = try KVCache.init(testing.allocator, 2);
    defer dfl2.deinit();
    var moe_off2: usize = 0;
    var base2: usize = 7;
    const res2 = try hc2.lookupAndRestore(
        &dst2,
        &moe_off2,
        null,
        s,
        toks[0..32],
        false,
        0,
        .{ .cache = &dfl2, .base_pos = &base2 },
        null,
    );
    try testing.expectEqual(@as(usize, 31), res2.matched);
    try testing.expectEqual(@as(?usize, null), res2.dflash_base);
    try testing.expectEqual(@as(usize, 0), dfl2.step);
    try testing.expectEqual(@as(usize, 7), base2); // untouched
}

test "prefix cache: MTP committed history round-trips, clamped; a history ending short is declined" {
    // Same DflashSnap machinery, second Entry field: the head's history is
    // built from trunk hiddens and a restore forwards nothing, so without
    // this every reused prefix drafts blind (~70 -> ~38 tok/s on warm echo,
    // Qwen3.6-27B). Unlike the trunk KV, a history that ends BEFORE the
    // matched cursor cannot be adopted — the missing tail's hiddens are
    // unrecoverable, and a gap right below the generation point is worse
    // than a blind start.
    const s = mlx.gpuStream();
    var toks: [64]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 5);

    var trunk = try KVCache.init(testing.allocator, 2);
    defer trunk.deinit();
    try testFillCache(&trunk, s, 2, 64);
    // Committed history covers 60 of the 64 tokens (the deferred-stash lag).
    var hist = try KVCache.init(testing.allocator, 1);
    defer hist.deinit();
    try testFillCache(&hist, s, 1, 60);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    _ = try hc.commitWithSsm(&trunk, &toks, false, null, null, .{ .cache = &hist, .base_pos = 0 });
    try testing.expect(hc.entries.items[0].mtp_bytes > 0);
    try testing.expect(hc.entries.items[0].kv_bytes > hc.entries.items[0].mtp_bytes);

    // Shorter prompt fully covered by the entry: full-match arm lands the
    // trunk at 31 and the history clamps to 31 (base 0).
    var dst = try KVCache.init(testing.allocator, 2);
    defer dst.deinit();
    var mtp_dst = try KVCache.init(testing.allocator, 1);
    defer mtp_dst.deinit();
    var moe_off: usize = 0;
    var base: usize = 99;
    const res = try hc.lookupAndRestore(&dst, &moe_off, null, s, toks[0..32], false, 0, null, .{ .cache = &mtp_dst, .base_pos = &base });
    try testing.expect(res.full_match);
    try testing.expectEqual(@as(usize, 31), res.matched);
    try testing.expectEqual(@as(?usize, 0), res.mtp_base);
    try testing.expectEqual(@as(usize, 0), base);
    try testing.expectEqual(@as(usize, 31), mtp_dst.step);
    try testing.expectEqual(base + mtp_dst.step, res.matched);

    // Full 64-token re-issue: matched 63 > the 60 the history covers →
    // declined, target untouched, caller starts blind.
    var dst2 = try KVCache.init(testing.allocator, 2);
    defer dst2.deinit();
    var mtp2 = try KVCache.init(testing.allocator, 1);
    defer mtp2.deinit();
    var moe2: usize = 0;
    var base2: usize = 7;
    const res2 = try hc.lookupAndRestore(&dst2, &moe2, null, s, &toks, false, 0, null, .{ .cache = &mtp2, .base_pos = &base2 });
    try testing.expectEqual(@as(usize, 63), res2.matched);
    try testing.expectEqual(@as(?usize, null), res2.mtp_base);
    try testing.expectEqual(@as(usize, 0), mtp2.step);
    try testing.expectEqual(@as(usize, 7), base2); // untouched
}

fn testWriteCacheLayer(cache: *KVCache, s: mlx.mlx_stream, layer: u32, written: u32, step: u32) !void {
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    const count: f64 = @floatFromInt(step * 8);
    const base: f64 = @floatFromInt(written * 8 + layer * 1_000_000);
    try mlx.check(mlx.mlx_arange(&flat, base, base + count, 1.0, .float32, s));
    var k = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k);
    const shape = [_]c_int{ 1, 1, @intCast(step), 8 };
    try mlx.check(mlx.mlx_reshape(&k, flat, &shape, 4, s));
    var view = try cache.update(layer, k, k, s, 0);
    view.deinit();
}

fn testFillCache(cache: *KVCache, s: mlx.mlx_stream, n_layers: u32, tokens: u32) !void {
    var written: u32 = 0;
    while (written < tokens) {
        const step: u32 = @min(64, tokens - written);
        var li: u32 = 0;
        while (li < n_layers) : (li += 1) try testWriteCacheLayer(cache, s, li, written, step);
        written += step;
    }
}

/// Fill a head-shaped cache: one layer at `layer` (the qwen4_exp MTP head's layer is never 0),
/// driven through `Transformer.qwen4MtpAdvance` as `qwen4MtpForward` drives it.
fn testFillHeadCache(cache: *KVCache, s: mlx.mlx_stream, layer: u32, tokens: u32, seq_offset: *usize) !void {
    var written: u32 = 0;
    while (written < tokens) {
        const step: u32 = @min(64, tokens - written);
        try testWriteCacheLayer(cache, s, layer, written, step);
        transformer_mod.Transformer.qwen4MtpAdvance(cache, seq_offset, @intCast(step));
        written += step;
    }
}

/// One layer's chunk at absolute rows `[written, written+step)`, handed the
/// `max_seq` its attention would ask for: non-zero IS the ring predicate.
fn ringWriteLayer(cache: *KVCache, s: mlx.mlx_stream, layer: u32, written: u32, step: u32, window: u32) !transformer_mod.DenseKVView {
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    const count: f64 = @floatFromInt(step * 8);
    const base: f64 = @floatFromInt(written * 8 + layer * 1_000_000);
    try mlx.check(mlx.mlx_arange(&flat, base, base + count, 1.0, .float32, s));
    var k = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k);
    const shape = [_]c_int{ 1, 1, @intCast(step), 8 };
    try mlx.check(mlx.mlx_reshape(&k, flat, &shape, 4, s));
    const span: u32 = if (layer == 0) 0 else transformer_mod.slidingTailSpan(window, step, transformer_mod.SLIDING_TRIM_UNBOUNDED);
    return cache.update(layer, k, k, s, span);
}

/// MiMo shape: layer 0 global, every other layer sliding.
fn ringFill(cache: *KVCache, s: mlx.mlx_stream, n_layers: u32, window: u32, from: u32, to: u32, chunk: u32) !void {
    var written: u32 = from;
    while (written < to) {
        const step: u32 = @min(chunk, to - written);
        var li: u32 = 0;
        while (li < n_layers) : (li += 1) {
            var view = try ringWriteLayer(cache, s, li, written, step, window);
            view.deinit();
        }
        written += step;
    }
}

/// Bytes one sequence row of `arr` bills, from its own shape.
fn ringRowBytes(arr: mlx.mlx_array) u64 {
    const rows: u64 = @intCast(mlx.getShape(arr)[2]);
    return (@as(u64, mlx.mlx_array_size(arr)) * @as(u64, mlx.mlx_array_itemsize(arr))) / rows;
}

test "a hot entry holds a ringed layer's retained rows, not the ring buffer's capacity" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const window: u32 = 8;
    const n_layers: u32 = 4; // layer 0 global, 1..3 sliding
    const prompt_len: u32 = 700; // past the ring cap, so a compaction has run
    const chunk: u32 = 64;

    var toks: [prompt_len]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);

    var live = try KVCache.init(testing.allocator, n_layers);
    defer live.deinit();
    live.setSwaRing(window);
    try ringFill(&live, s, n_layers, window, 0, prompt_len, chunk);

    // The ring buffer is allocated at its cap from the first token, so the
    // rows past `offset` are rows no restore can ever read.
    var want_bytes: u64 = 0;
    var ring_rows: usize = 0;
    var ring_cap: usize = 0;
    for (live.entries) |*e| {
        if (!e.initialized) continue;
        const cap: usize = @intCast(mlx.getShape(e.keys)[2]);
        if (e.ringed) {
            ring_rows = e.offset;
            ring_cap = cap;
        }
        const rows: u64 = if (e.ringed) @intCast(e.offset) else @intCast(cap);
        want_bytes += rows * (ringRowBytes(e.keys) + ringRowBytes(e.values));
    }
    try testing.expect(ring_rows > 0 and ring_rows < ring_cap);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    _ = try hc.commitWithState(&live, &toks, false, 0, null, null, null);
    try testing.expectEqual(want_bytes, hc.current_kv_bytes);

    // A restore off that entry must still rebuild a ring and keep appending:
    // the reference cache walks the same absolute rows and never snapshots.
    var restored = try KVCache.init(testing.allocator, n_layers);
    defer restored.deinit();
    restored.setSwaRing(window);
    var moe_off: usize = 0;
    const hit = try hc.lookupAndRestore(&restored, &moe_off, null, s, &toks, false, 0, null, null);
    try testing.expectEqual(@as(usize, prompt_len - 1), hit.matched);

    var ref = try KVCache.init(testing.allocator, n_layers);
    defer ref.deinit();
    ref.setSwaRing(window);
    try ringFill(&ref, s, n_layers, window, 0, prompt_len + chunk, chunk);
    try ringFill(&restored, s, n_layers, window, @intCast(hit.matched), prompt_len + chunk, prompt_len + chunk);

    var li: u32 = 0;
    while (li < n_layers) : (li += 1) {
        var rv = try ringWriteLayer(&ref, s, li, prompt_len + chunk, 1, window);
        defer rv.deinit();
        var sv = try ringWriteLayer(&restored, s, li, prompt_len + chunk, 1, window);
        defer sv.deinit();
        try testing.expectEqual(@as(f32, 0), try transformer_mod.maxAbsDiffF32(rv.k, sv.k, s));
        try testing.expectEqual(@as(f32, 0), try transformer_mod.maxAbsDiffF32(rv.v, sv.v, s));
        try testing.expectEqual(ref.absSeqLen(li), restored.absSeqLen(li));
    }
}

/// A MiMo-shaped turn: `prompt` rows, a ring checkpoint at the prompt end, then a reply long
/// enough that the ring compacts past the prompt. The caller owns the checkpoint.
fn ringTurn(live: *KVCache, s: mlx.mlx_stream, n_layers: u32, window: u32, prompt: u32, reply: u32) !?KVCacheSnapshot {
    live.setSwaRing(window);
    try ringFill(live, s, n_layers, window, 0, prompt, 64);
    var cp = try live.ringCheckpoint(prompt, s);
    errdefer if (cp) |*c| c.deinit();
    try ringFill(live, s, n_layers, window, prompt, prompt + reply, 16);
    return cp;
}

test "a ringed entry restores at its prompt-end checkpoint when the next turn diverges past the ring" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const window: u32 = 8;
    const n_layers: u32 = 4; // layer 0 global, 1..3 sliding
    const prompt: u32 = 700;
    const reply: u32 = 600;

    var toks: [prompt + reply]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
    var live = try KVCache.init(testing.allocator, n_layers);
    defer live.deinit();
    const cp = try ringTurn(&live, s, n_layers, window, prompt, reply);

    var end = try live.snapshotRetained(s);
    const end_bytes = HotPrefixCache.snapshotBytes(&end);
    end.deinit();
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    _ = try hc.commitWithRing(&live, &toks, false, 0, 0, null, null, null, null, prompt, .{ .prompt_end = cp });
    // The bill: the window plus the backoff, per sliding layer, at the entry's KV width.
    var cp_bytes: u64 = 0;
    for (live.entries) |*e| {
        if (e.ringed) cp_bytes += (window + model_mod.ModelConfig.SWA_RING_CHECKPOINT_BACKOFF) * (ringRowBytes(e.keys) + ringRowBytes(e.values));
    }
    try testing.expectEqual(end_bytes + cp_bytes, hc.current_kv_bytes);

    // The next turn keeps the prompt and its first generated token, then diverges (the
    // template re-renders the reply), far below what the ring kept at the entry's end.
    var next: [prompt + 40]u32 = undefined;
    @memcpy(next[0 .. prompt + 1], toks[0 .. prompt + 1]);
    for (next[prompt + 1 ..], 0..) |*t, i| t.* = @intCast(900_000 + i);
    var restored = try KVCache.init(testing.allocator, n_layers);
    defer restored.deinit();
    restored.setSwaRing(window);
    var moe_off: usize = 0;
    const hit = try hc.lookupAndRestore(&restored, &moe_off, null, s, &next, false, 0, null, null);
    try testing.expectEqual(@as(usize, prompt), hit.matched);

    var ref = try KVCache.init(testing.allocator, n_layers);
    defer ref.deinit();
    ref.setSwaRing(window);
    try ringFill(&ref, s, n_layers, window, 0, prompt + 40, 64);
    try ringFill(&restored, s, n_layers, window, prompt, prompt + 40, 40);
    var li: u32 = 0;
    while (li < n_layers) : (li += 1) {
        var rv = try ringWriteLayer(&ref, s, li, prompt + 40, 1, window);
        defer rv.deinit();
        var sv = try ringWriteLayer(&restored, s, li, prompt + 40, 1, window);
        defer sv.deinit();
        try testing.expectEqual(@as(f32, 0), try transformer_mod.maxAbsDiffF32(rv.k, sv.k, s));
        try testing.expectEqual(@as(f32, 0), try transformer_mod.maxAbsDiffF32(rv.v, sv.v, s));
        try testing.expectEqual(ref.absSeqLen(li), restored.absSeqLen(li));
    }

    // A divergence below the checkpoint's rows is still un-restorable and cold-prefills.
    var edited: [prompt]u32 = undefined;
    @memcpy(edited[0 .. prompt - 40], toks[0 .. prompt - 40]);
    for (edited[prompt - 40 ..], 0..) |*t, i| t.* = @intCast(800_000 + i);
    var cold = try KVCache.init(testing.allocator, n_layers);
    defer cold.deinit();
    cold.setSwaRing(window);
    const miss = try hc.lookupAndRestore(&cold, &moe_off, null, s, &edited, false, 0, null, null);
    try testing.expectEqual(@as(usize, 0), miss.matched);
}

test "a ring checkpoint restore below the floor cold-prefills" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const window: u32 = 8;
    const n_layers: u32 = 4;
    const reply: u32 = 600;
    const cases = [_]struct { prompt: u32, shared: u32, want: usize }{
        // A short prompt's checkpoint covers it from row 0, so a shared chat header is servable.
        .{ .prompt = 24, .shared = 3, .want = 0 },
        .{ .prompt = RING_RESTORE_MIN_TOKENS + 20, .shared = RING_RESTORE_MIN_TOKENS + 21, .want = RING_RESTORE_MIN_TOKENS + 20 },
    };
    for (cases) |c| {
        var toks: [RING_RESTORE_MIN_TOKENS + 20 + reply]u32 = undefined;
        for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
        var live = try KVCache.init(testing.allocator, n_layers);
        defer live.deinit();
        const cp = try ringTurn(&live, s, n_layers, window, c.prompt, reply);
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        defer hc.deinit();
        _ = try hc.commitWithRing(&live, toks[0 .. c.prompt + reply], false, 0, 0, null, null, null, null, c.prompt, .{ .prompt_end = cp });

        var next: [RING_RESTORE_MIN_TOKENS + 60]u32 = undefined;
        @memcpy(next[0..c.shared], toks[0..c.shared]);
        for (next[c.shared..], 0..) |*t, i| t.* = @intCast(900_000 + i);
        var restored = try KVCache.init(testing.allocator, n_layers);
        defer restored.deinit();
        restored.setSwaRing(window);
        var moe_off: usize = 0;
        const hit = try hc.lookupAndRestore(&restored, &moe_off, null, s, &next, false, 0, null, null);
        try testing.expectEqual(c.want, hit.matched);
    }
}

test "a restore declined at the ring clamp leaves its entry's recency alone" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const window: u32 = 8;
    const n_layers: u32 = 4;
    const prompt: u32 = 700;
    const reply: u32 = 600;
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    // Two sessions that share only a chat header, the older one committed first.
    for ([_]u32{ 0, 500_000 }) |offset| {
        var toks: [prompt + reply]u32 = undefined;
        for (&toks, 0..) |*t, i| t.* = if (i < 3) @intCast(i + 1) else @intCast(offset + i + 1);
        var live = try KVCache.init(testing.allocator, n_layers);
        defer live.deinit();
        const cp = try ringTurn(&live, s, n_layers, window, prompt, reply);
        _ = try hc.commitWithRing(&live, &toks, false, 0, 0, null, null, null, null, prompt, .{ .prompt_end = cp });
    }
    const before = [_]u64{ hc.entries.items[0].last_used, hc.entries.items[1].last_used };
    try testing.expect(before[0] < before[1]);

    // A request sharing only the header lands on one of them and declines at the clamp.
    var header_only: [prompt]u32 = undefined;
    for (&header_only, 0..) |*t, i| t.* = if (i < 3) @intCast(i + 1) else @intCast(900_000 + i);
    var cold = try KVCache.init(testing.allocator, n_layers);
    defer cold.deinit();
    cold.setSwaRing(window);
    var moe_off: usize = 0;
    const miss = try hc.lookupAndRestore(&cold, &moe_off, null, s, &header_only, false, 0, null, null);
    try testing.expectEqual(@as(usize, 0), miss.matched);
    // A restore that did not happen is not a use: promoted, the older entry made the newer one the eviction victim.
    try testing.expectEqual(before[0], hc.entries.items[0].last_used);
    try testing.expectEqual(before[1], hc.entries.items[1].last_used);
}

test "ringed candidates rank by restorable position, not raw match" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const window: u32 = 8;
    const n_layers: u32 = 4;
    const prompt: u32 = 700;
    const reply: u32 = 600;
    const short: u32 = 400;

    var toks: [prompt + reply]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    // The longer raw match carries no checkpoint, so its ring cannot reach the divergence.
    var long = try KVCache.init(testing.allocator, n_layers);
    defer long.deinit();
    if (try ringTurn(&long, s, n_layers, window, prompt, reply)) |cp| {
        var doomed = cp;
        doomed.deinit();
    }
    _ = try hc.commitWithState(&long, &toks, false, 0, null, null, null);
    var shorter = try KVCache.init(testing.allocator, n_layers);
    defer shorter.deinit();
    shorter.setSwaRing(window);
    try ringFill(&shorter, s, n_layers, window, 0, short, 64);
    _ = try hc.commitWithState(&shorter, toks[0..short], false, 0, null, null, null);

    var next: [prompt + 40]u32 = undefined;
    @memcpy(next[0 .. prompt + 1], toks[0 .. prompt + 1]);
    for (next[prompt + 1 ..], 0..) |*t, i| t.* = @intCast(900_000 + i);
    var restored = try KVCache.init(testing.allocator, n_layers);
    defer restored.deinit();
    restored.setSwaRing(window);
    var moe_off: usize = 0;
    const hit = try hc.lookupAndRestore(&restored, &moe_off, null, s, &next, false, 0, null, null);
    try testing.expectEqual(@as(usize, short), hit.matched);
}

/// A ring checkpoint with no rows; `mark` tells two at one position apart.
fn stubRingCp(step: usize, mark: usize) !KVCacheSnapshot {
    const entries = try testing.allocator.alloc(transformer_mod.KVCacheEntry, 1);
    entries[0] = transformer_mod.newEmptyKVEntry();
    entries[0].offset = mark;
    return .{ .entries = entries, .step = step, .allocator = testing.allocator, .config = kv_quant.KVQuantConfig.dense };
}

fn stubRingCps(steps: []const usize, mark: usize) ![]KVCacheSnapshot {
    const out = try testing.allocator.alloc(KVCacheSnapshot, steps.len);
    for (out, steps) |*o, st| o.* = try stubRingCp(st, mark);
    return out;
}

test "ring checkpoints merge ascending, a tie keeps the newer, and past RING_CHECKPOINT_MAX the interior thins: the lowest and the newest stay" {
    const merged = (try HotPrefixCache.mergeRingCps(
        testing.allocator,
        try stubRingCps(&.{ 100, 300, 500, 1100, 1300, 1500 }, 1),
        try stubRingCps(&.{ 300, 1700, 1900, 2100, 2300 }, 2),
    )).?;
    defer HotPrefixCache.freeRingCps(testing.allocator, merged);
    try testing.expectEqual(RING_CHECKPOINT_MAX, merged.len);
    try testing.expectEqual(@as(usize, 100), merged[0].step);
    try testing.expectEqual(@as(usize, 2100), merged[merged.len - 2].step);
    try testing.expectEqual(@as(usize, 2300), merged[merged.len - 1].step);
    for (merged[0 .. merged.len - 1], merged[1..]) |a, b| try testing.expect(a.step < b.step);
    for (merged) |cp| {
        if (cp.step == 300) try testing.expectEqual(@as(usize, 2), cp.entries[0].offset);
    }

    const small = (try HotPrefixCache.mergeRingCps(testing.allocator, try stubRingCps(&.{ 100, 500 }, 1), try stubRingCps(&.{ 300, 500 }, 2))).?;
    defer HotPrefixCache.freeRingCps(testing.allocator, small);
    try testing.expectEqual(@as(usize, 3), small.len);
    for (small, [_]usize{ 100, 300, 500 }, [_]usize{ 1, 2, 2 }) |cp, step, mark| {
        try testing.expectEqual(step, cp.step);
        try testing.expectEqual(mark, cp.entries[0].offset);
    }
    try testing.expectEqual(@as(?[]KVCacheSnapshot, null), try HotPrefixCache.mergeRingCps(testing.allocator, null, null));

    // An inheritance shares what lies at or below the fork, thinned the same way.
    const src = try stubRingCps(&.{ 100, 300, 500, 700, 900, 1100, 1300, 1500, 1700, 1900 }, 1);
    defer HotPrefixCache.freeRingCps(testing.allocator, src);
    const shared = (try HotPrefixCache.shareRingCpsUpTo(testing.allocator, src, 1800, null)).?;
    defer HotPrefixCache.freeRingCps(testing.allocator, shared);
    try testing.expectEqual(RING_CHECKPOINT_MAX, shared.len);
    try testing.expectEqual(@as(usize, 100), shared[0].step);
    try testing.expectEqual(@as(usize, 1700), shared[shared.len - 1].step);
}

/// A ring checkpoint whose one entry holds 4 bf16 rows of 8 in K and in V.
fn sizedRingCp(step: usize) !KVCacheSnapshot {
    var cp = try stubRingCp(step, 0);
    errdefer cp.deinit();
    const e = &cp.entries[0];
    try mlx.check(mlx.mlx_zeros(&e.keys, &[_]c_int{ 1, 1, 4, 8 }, 4, .bfloat16, mlx.gpuStream()));
    try mlx.check(mlx.mlx_zeros(&e.values, &[_]c_int{ 1, 1, 4, 8 }, 4, .bfloat16, mlx.gpuStream()));
    e.initialized = true;
    return cp;
}

test "a ring checkpoint list sheds bytes toward its lowest and its newest" {
    const bytes_each: u64 = 2 * 4 * 8 * 2;
    const cps = try testing.allocator.alloc(KVCacheSnapshot, 5);
    var made: usize = 0;
    errdefer {
        for (cps[0..made]) |*c| c.deinit();
        testing.allocator.free(cps);
    }
    for (cps, [_]usize{ 100, 300, 500, 700, 900 }) |*c, step| {
        c.* = try sizedRingCp(step);
        made += 1;
    }
    const kept = (try HotPrefixCache.thinRingCps(testing.allocator, cps, RING_CHECKPOINT_MAX, 3 * bytes_each)).?;
    defer HotPrefixCache.freeRingCps(testing.allocator, kept);
    try testing.expectEqual(@as(usize, 2), kept.len);
    try testing.expectEqual(@as(usize, 100), kept[0].step);
    try testing.expectEqual(@as(usize, 900), kept[1].step);
}

test "ring marks sit at the message starts a prefill forwards, thinned to RING_MARKS_MAX keeping the first and the last" {
    const im: u32 = 7;
    var prompt: [2000]u32 = @splat(1);
    for ([_]usize{ 0, 40, 300, 500, 900, 1100, 1300, 1985 }) |p| prompt[p] = im;
    var buf: [RING_MARKS_MAX]usize = undefined;
    // Under the restore floor and inside the prompt end's own reach are skipped.
    try testing.expectEqualSlices(usize, &.{ 300, 500, 900, 1300 }, ringMarkPositions(&prompt, 0, im, &buf));
    // A restored prefix is already in the cache: only the tail past it is marked.
    try testing.expectEqualSlices(usize, &.{ 900, 1100, 1300 }, ringMarkPositions(&prompt, 500, im, &buf));
    try testing.expectEqual(@as(usize, 0), ringMarkPositions(&prompt, 0, 99, &buf).len);
    try testing.expectEqual(@as(usize, 0), ringMarkPositions(prompt[0..60], 0, im, &buf).len);
}

test "a slot's ring restore points are freed by the slot and by a commit that cannot take them" {
    var held: SlotRingCps = .{ .fork = try stubRingCp(10, 0), .prompt_end = try stubRingCp(90, 0) };
    held.marks[0] = try stubRingCp(50, 0);
    held.n_marks = 1;
    held.deinit();
    try testing.expectEqual(@as(usize, 0), held.n_marks);

    var cps: SlotRingCps = .{ .fork = try stubRingCp(10, 0), .prompt_end = try stubRingCp(90, 0) };
    cps.marks[0] = try stubRingCp(50, 0);
    cps.marks[1] = try stubRingCp(70, 0);
    cps.n_marks = 2;
    var failing = std.testing.FailingAllocator.init(testing.allocator, .{ .fail_index = 0 });
    try testing.expectEqual(@as(?[]KVCacheSnapshot, null), HotPrefixCache.ownRingCps(failing.allocator(), cps));

    var ok: SlotRingCps = .{ .fork = try stubRingCp(10, 0), .prompt_end = try stubRingCp(90, 0) };
    ok.marks[0] = try stubRingCp(50, 0);
    ok.n_marks = 1;
    const owned = HotPrefixCache.ownRingCps(testing.allocator, ok).?;
    defer HotPrefixCache.freeRingCps(testing.allocator, owned);
    try testing.expectEqual(@as(usize, 3), owned.len);
    for (owned, [_]usize{ 10, 50, 90 }) |cp, step| try testing.expectEqual(step, cp.step);
}

test "a new session sharing only another's preamble restores at the message mark that prefill took" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const window: u32 = 8;
    const n_layers: u32 = 4;
    const preamble: u32 = 500;
    const prompt: u32 = 800;
    const reply: u32 = 600;

    var toks: [prompt + reply]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
    var live = try KVCache.init(testing.allocator, n_layers);
    defer live.deinit();
    live.setSwaRing(window);
    var cps: SlotRingCps = .{};
    defer cps.deinit();
    // The second mark lies past what this prefill forwards, so it never fills.
    cps.armMarks(&live, &.{ preamble, prompt + 10 });
    try testing.expectEqual(@as(usize, 2), cps.n_marks);
    live.ring_marks = cps.markSlice();
    // 300-row writes: the one that crosses the mark also compacts the ring past it.
    try ringFill(&live, s, n_layers, window, 0, prompt, 300);
    live.ring_marks = &.{};
    cps.keepCompleteMarks(&live);
    try testing.expectEqual(@as(usize, 1), cps.n_marks);
    cps.prompt_end = try live.ringCheckpoint(prompt, s);
    try ringFill(&live, s, n_layers, window, prompt, prompt + reply, 16);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    const moved = cps;
    cps = .{};
    _ = try hc.commitWithRing(&live, &toks, false, 0, 0, null, null, null, null, prompt, moved);
    try testing.expectEqual(@as(usize, 2), hc.entries.items[0].ring_cps.?.len);

    // Same preamble and message header, then another task.
    var next: [prompt]u32 = undefined;
    @memcpy(next[0 .. preamble + 3], toks[0 .. preamble + 3]);
    for (next[preamble + 3 ..], 0..) |*t, i| t.* = @intCast(900_000 + i);
    var restored = try KVCache.init(testing.allocator, n_layers);
    defer restored.deinit();
    restored.setSwaRing(window);
    var moe_off: usize = 0;
    const hit = try hc.lookupAndRestore(&restored, &moe_off, null, s, &next, false, 0, null, null);
    try testing.expectEqual(@as(usize, preamble), hit.matched);

    var ref = try KVCache.init(testing.allocator, n_layers);
    defer ref.deinit();
    ref.setSwaRing(window);
    try ringFill(&ref, s, n_layers, window, 0, prompt, 64);
    try ringFill(&restored, s, n_layers, window, preamble, prompt, 64);
    var li: u32 = 0;
    while (li < n_layers) : (li += 1) {
        var rv = try ringWriteLayer(&ref, s, li, prompt, 1, window);
        defer rv.deinit();
        var sv = try ringWriteLayer(&restored, s, li, prompt, 1, window);
        defer sv.deinit();
        try testing.expectEqual(@as(f32, 0), try transformer_mod.maxAbsDiffF32(rv.k, sv.k, s));
        try testing.expectEqual(@as(f32, 0), try transformer_mod.maxAbsDiffF32(rv.v, sv.v, s));
        try testing.expectEqual(ref.absSeqLen(li), restored.absSeqLen(li));
    }
}

test "a slot holding every ring restore point it may bills SLOT_RING_CHECKPOINTS checkpoints" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const window: u32 = 8;
    const n_layers: u32 = 4;
    var live = try KVCache.init(testing.allocator, n_layers);
    defer live.deinit();
    live.setSwaRing(window);
    var cps: SlotRingCps = .{};
    defer cps.deinit();
    try ringFill(&live, s, n_layers, window, 0, 400, 64);
    cps.fork = try live.ringCheckpoint(400, s);
    cps.armMarks(&live, &.{ 700, 1000, 1300, 1600, 1700 });
    try testing.expectEqual(RING_MARKS_MAX, cps.n_marks);
    live.ring_marks = cps.markSlice();
    try ringFill(&live, s, n_layers, window, 400, 1900, 256);
    live.ring_marks = &.{};
    cps.keepCompleteMarks(&live);
    cps.prompt_end = try live.ringCheckpoint(1900, s);
    try testing.expectEqual(RING_MARKS_MAX, cps.n_marks);

    var cp_bytes: u64 = 0;
    for (live.entries) |*e| {
        if (e.ringed) cp_bytes += (window + model_mod.ModelConfig.SWA_RING_CHECKPOINT_BACKOFF) * (ringRowBytes(e.keys) + ringRowBytes(e.values));
    }
    try testing.expectEqual(@as(u64, SLOT_RING_CHECKPOINTS) * cp_bytes, cps.bytes());
}

test "an entry extended in place keeps its older ring checkpoints" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const window: u32 = 8;
    const n_layers: u32 = 4;
    const prompt: u32 = 700;
    const reply: u32 = 600;
    const turn2 = prompt + reply + 100;

    var toks: [turn2 + 300]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    var live = try KVCache.init(testing.allocator, n_layers);
    defer live.deinit();
    const cp = try ringTurn(&live, s, n_layers, window, prompt, reply);
    _ = try hc.commitWithRing(&live, toks[0 .. prompt + reply], false, 0, 0, null, null, null, null, prompt, .{ .prompt_end = cp });

    // A client that sends the whole reply back extends the entry, which the commit replaces.
    var ext = try KVCache.init(testing.allocator, n_layers);
    defer ext.deinit();
    ext.setSwaRing(window);
    var moe_off: usize = 0;
    const hit = try hc.lookupAndRestore(&ext, &moe_off, null, s, toks[0..turn2], false, 0, null, null);
    try testing.expectEqual(@as(usize, prompt + reply), hit.matched);
    try ringFill(&ext, s, n_layers, window, prompt + reply, turn2, 64);
    var cp2 = try ext.ringCheckpoint(turn2, s);
    errdefer if (cp2) |*c| c.deinit();
    try ringFill(&ext, s, n_layers, window, turn2, toks.len, 16);
    const moved = cp2;
    cp2 = null;
    _ = try hc.commitWithRing(&ext, &toks, false, 0, 0, null, null, null, null, turn2, .{ .prompt_end = moved });
    try testing.expectEqual(@as(usize, 1), hc.entries.items.len);
    try testing.expectEqual(@as(usize, 2), hc.entries.items[0].ring_cps.?.len);

    // A turn diverging right after the first prompt still restores there.
    var next: [prompt + 40]u32 = undefined;
    @memcpy(next[0 .. prompt + 1], toks[0 .. prompt + 1]);
    for (next[prompt + 1 ..], 0..) |*t, i| t.* = @intCast(900_000 + i);
    var restored = try KVCache.init(testing.allocator, n_layers);
    defer restored.deinit();
    restored.setSwaRing(window);
    const again = try hc.lookupAndRestore(&restored, &moe_off, null, s, &next, false, 0, null, null);
    try testing.expectEqual(@as(usize, prompt), again.matched);
}

/// Bytes one row of `e`'s K and V bill, from their own shapes.
fn ringEntryRowBytes(e: *const transformer_mod.KVCacheEntry) u64 {
    return ringRowBytes(e.keys) + ringRowBytes(e.values);
}

/// The next turn's first write through both caches: identical views prove the restore.
fn ringExpectSameNext(ref: *KVCache, restored: *KVCache, s: mlx.mlx_stream, n_layers: u32, window: u32, at: u32) !void {
    var li: u32 = 0;
    while (li < n_layers) : (li += 1) {
        var rv = try ringWriteLayer(ref, s, li, at, 1, window);
        defer rv.deinit();
        var sv = try ringWriteLayer(restored, s, li, at, 1, window);
        defer sv.deinit();
        try testing.expectEqual(@as(f32, 0), try transformer_mod.maxAbsDiffF32(rv.k, sv.k, s));
        try testing.expectEqual(@as(f32, 0), try transformer_mod.maxAbsDiffF32(rv.v, sv.v, s));
        try testing.expectEqual(ref.absSeqLen(li), restored.absSeqLen(li));
    }
}

test "an oversized ringed entry trims to its end: the ring is a constant, not a per-token price" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const window: u32 = 8;
    const n_layers: u32 = 4; // layer 0 global, 1..3 sliding
    const prompt: u32 = 2000;

    var toks: [prompt]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
    var live = try KVCache.init(testing.allocator, n_layers);
    defer live.deinit();
    live.setSwaRing(window);
    try ringFill(&live, s, n_layers, window, 0, prompt, 64);

    // The global layer's buffer holds spare capacity past its rows, as a reservation does.
    var probe = try live.snapshotRetained(s);
    const cap: u64 = @intCast(mlx.getShape(probe.entries[0].keys)[2]);
    const rows_bytes = HotPrefixCache.snapshotBytes(&probe) - (cap - prompt) * ringEntryRowBytes(&probe.entries[0]);
    probe.deinit();
    try testing.expect(cap > prompt);

    // Room for the rows, not the capacity they sit in.
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, rows_bytes);
    defer hc.deinit();
    const status = try hc.commitWithState(&live, &toks, false, 0, null, null, null);
    try testing.expectEqual(std.meta.Tag(CommitStatus).ok, std.meta.activeTag(status));
    try testing.expectEqual(@as(usize, 1), hc.entries.items.len);
    try testing.expectEqual(@as(usize, prompt), hc.entries.items[0].tokens.len);
    try testing.expectEqual(rows_bytes, hc.current_kv_bytes);

    var restored = try KVCache.init(testing.allocator, n_layers);
    defer restored.deinit();
    restored.setSwaRing(window);
    var moe_off: usize = 0;
    const hit = try hc.lookupAndRestore(&restored, &moe_off, null, s, &toks, false, 0, null, null);
    try testing.expectEqual(@as(usize, prompt - 1), hit.matched);
    var ref = try KVCache.init(testing.allocator, n_layers);
    defer ref.deinit();
    ref.setSwaRing(window);
    try ringFill(&ref, s, n_layers, window, 0, prompt + 64, 64);
    try ringFill(&restored, s, n_layers, window, @intCast(hit.matched), prompt + 64, prompt + 64);
    try ringExpectSameNext(&ref, &restored, s, n_layers, window, prompt + 64);
}

test "an oversized ringed entry trims to the highest ring checkpoint that fits, and declines below every one" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const window: u32 = 8;
    const n_layers: u32 = 4;
    const prompt: u32 = 2000;
    const reply: u32 = 600;

    var toks: [prompt + reply]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
    var live = try KVCache.init(testing.allocator, n_layers);
    defer live.deinit();
    const cp = try ringTurn(&live, s, n_layers, window, prompt, reply);
    const cp_bytes = HotPrefixCache.snapshotBytes(&cp.?);
    var probe = try live.snapshotRetained(s);
    const global_row = ringEntryRowBytes(&probe.entries[0]);
    probe.deinit();

    // The prompt's global rows plus the checkpoint's window: the entry ends where the checkpoint is.
    const budget = prompt * global_row + cp_bytes;
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, budget);
    defer hc.deinit();
    const status = try hc.commitWithRing(&live, &toks, false, 0, 0, null, null, null, null, prompt, .{ .prompt_end = cp });
    try testing.expectEqual(std.meta.Tag(CommitStatus).ok, std.meta.activeTag(status));
    try testing.expectEqual(@as(usize, 1), hc.entries.items.len);
    try testing.expectEqual(@as(usize, prompt), hc.entries.items[0].tokens.len);
    try testing.expect(hc.entries.items[0].ring_cps == null);
    try testing.expectEqual(budget, hc.current_kv_bytes);

    // The next turn keeps the prompt and diverges after it.
    var next: [prompt + 40]u32 = undefined;
    @memcpy(next[0 .. prompt + 1], toks[0 .. prompt + 1]);
    for (next[prompt + 1 ..], 0..) |*t, i| t.* = @intCast(900_000 + i);
    var restored = try KVCache.init(testing.allocator, n_layers);
    defer restored.deinit();
    restored.setSwaRing(window);
    var moe_off: usize = 0;
    const hit = try hc.lookupAndRestore(&restored, &moe_off, null, s, &next, false, 0, null, null);
    try testing.expectEqual(@as(usize, prompt), hit.matched);
    var ref = try KVCache.init(testing.allocator, n_layers);
    defer ref.deinit();
    ref.setSwaRing(window);
    try ringFill(&ref, s, n_layers, window, 0, prompt + 40, 64);
    try ringFill(&restored, s, n_layers, window, prompt, prompt + 40, 40);
    try ringExpectSameNext(&ref, &restored, s, n_layers, window, prompt + 40);

    // No room for the checkpoint's window: nothing restorable fits.
    var live2 = try KVCache.init(testing.allocator, n_layers);
    defer live2.deinit();
    const cp2 = try ringTurn(&live2, s, n_layers, window, prompt, reply);
    var tight = HotPrefixCache.initWithMem(testing.allocator, 4, prompt * global_row);
    defer tight.deinit();
    const declined = try tight.commitWithRing(&live2, &toks, false, 0, 0, null, null, null, null, prompt, .{ .prompt_end = cp2 });
    try testing.expectEqual(std.meta.Tag(CommitStatus).declined, std.meta.activeTag(declined));
    try testing.expectEqual(@as(usize, 0), tight.entries.items.len);
}

test "a replacement over the budget sheds its lowest ring checkpoints before the entry" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const window: u32 = 8;
    const n_layers: u32 = 4;
    const prompt: u32 = 700;
    const reply: u32 = 600;
    const turn2 = prompt + reply + 100;

    var toks: [turn2 + 300]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);

    // Turn 2's cache first, to price a budget that holds it and its own checkpoint.
    var ext = try KVCache.init(testing.allocator, n_layers);
    defer ext.deinit();
    ext.setSwaRing(window);
    try ringFill(&ext, s, n_layers, window, 0, turn2, 64);
    var cp2 = try ext.ringCheckpoint(turn2, s);
    errdefer if (cp2) |*c| c.deinit();
    try ringFill(&ext, s, n_layers, window, turn2, toks.len, 16);
    var probe = try ext.snapshotRetained(s);
    const alone = HotPrefixCache.snapshotBytes(&probe) + HotPrefixCache.snapshotBytes(&cp2.?);
    probe.deinit();

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, alone);
    defer hc.deinit();
    var live = try KVCache.init(testing.allocator, n_layers);
    defer live.deinit();
    const cp = try ringTurn(&live, s, n_layers, window, prompt, reply);
    _ = try hc.commitWithRing(&live, toks[0 .. prompt + reply], false, 0, 0, null, null, null, null, prompt, .{ .prompt_end = cp });
    try testing.expectEqual(@as(usize, 1), hc.entries.items.len);

    // Turn 2 extends the entry, and the merge brings turn 1's checkpoint along: one too many.
    const moved = cp2;
    cp2 = null;
    const status = try hc.commitWithRing(&ext, &toks, false, 0, 0, null, null, null, null, turn2, .{ .prompt_end = moved });
    try testing.expectEqual(std.meta.Tag(CommitStatus).ok, std.meta.activeTag(status));
    try testing.expectEqual(@as(usize, 1), hc.entries.items.len);
    try testing.expectEqual(@as(usize, toks.len), hc.entries.items[0].tokens.len);
    try testing.expectEqual(@as(usize, 1), hc.entries.items[0].ring_cps.?.len);
    try testing.expectEqual(@as(usize, turn2), hc.entries.items[0].ring_cps.?[0].step);
    try testing.expectEqual(alone, hc.current_kv_bytes);
}

test "a ring checkpoint taken at the restore keeps the fork when the donor is evicted before the commit" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const window: u32 = 8;
    const n_layers: u32 = 4;
    const prompt: u32 = 700;
    const reply: u32 = 600;
    const fork_prompt = prompt + 100 + 330;

    var toks: [fork_prompt + reply]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
    var hc = HotPrefixCache.initWithMem(testing.allocator, 1, 0);
    defer hc.deinit();
    var donor = try KVCache.init(testing.allocator, n_layers);
    defer donor.deinit();
    const donor_cp = try ringTurn(&donor, s, n_layers, window, prompt, reply);
    var donor_toks: [prompt + reply]u32 = undefined;
    @memcpy(donor_toks[0..prompt], toks[0..prompt]);
    for (donor_toks[prompt..], 0..) |*t, i| t.* = @intCast(700_000 + i);
    _ = try hc.commitWithRing(&donor, &donor_toks, false, 0, 0, null, null, null, null, prompt, .{ .prompt_end = donor_cp });

    var fork = try KVCache.init(testing.allocator, n_layers);
    defer fork.deinit();
    fork.setSwaRing(window);
    var moe_off: usize = 0;
    const fork_hit = try hc.lookupAndRestore(&fork, &moe_off, null, s, toks[0..fork_prompt], false, 0, null, null);
    try testing.expectEqual(@as(usize, prompt), fork_hit.matched);
    var at_restore = try fork.ringCheckpoint(fork_hit.matched, s);
    errdefer if (at_restore) |*c| c.deinit();
    try ringFill(&fork, s, n_layers, window, prompt, fork_prompt, 64);
    var at_end = try fork.ringCheckpoint(fork_prompt, s);
    errdefer if (at_end) |*c| c.deinit();
    try ringFill(&fork, s, n_layers, window, fork_prompt, fork_prompt + reply, 16);

    // Another slot commits first and the count cap evicts the donor: nothing is left to inherit.
    var other = try KVCache.init(testing.allocator, n_layers);
    defer other.deinit();
    other.setSwaRing(window);
    try ringFill(&other, s, n_layers, window, 0, 300, 64);
    var other_toks: [300]u32 = undefined;
    for (&other_toks, 0..) |*t, i| t.* = @intCast(500_000 + i);
    _ = try hc.commitWithState(&other, &other_toks, false, 0, null, null, null);
    const cps: SlotRingCps = .{ .fork = at_restore, .prompt_end = at_end };
    at_restore = null;
    at_end = null;
    _ = try hc.commitWithRing(&fork, &toks, false, 0, 0, null, null, null, null, fork_prompt, cps);
    try testing.expectEqual(@as(usize, 1), hc.entries.items.len);
    try testing.expectEqual(@as(usize, 2), hc.entries.items[0].ring_cps.?.len);

    var next: [prompt + 100 + 40]u32 = undefined;
    @memcpy(next[0 .. prompt + 100], toks[0 .. prompt + 100]);
    for (next[prompt + 100 ..], 0..) |*t, i| t.* = @intCast(900_000 + i);
    var restored = try KVCache.init(testing.allocator, n_layers);
    defer restored.deinit();
    restored.setSwaRing(window);
    const hit = try hc.lookupAndRestore(&restored, &moe_off, null, s, &next, false, 0, null, null);
    try testing.expectEqual(@as(usize, prompt), hit.matched);
    try expectRingContinuesCold(&restored, s, n_layers, window, prompt, next.len);
}

test "a fork entry keeps its donor's ring checkpoint, so evicting the donor still restores at the fork" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const window: u32 = 8;
    const n_layers: u32 = 4;
    const prompt: u32 = 700;
    const reply: u32 = 600;
    const tail: u32 = 100; // the conversation past the donor's prompt (its reply, re-rendered)
    const reminder: u32 = 330;
    const fork_prompt = prompt + tail + reminder;

    var toks: [fork_prompt + reply]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
    var hc = HotPrefixCache.initWithMem(testing.allocator, 2, 0);
    defer hc.deinit();

    var donor = try KVCache.init(testing.allocator, n_layers);
    defer donor.deinit();
    const donor_cp = try ringTurn(&donor, s, n_layers, window, prompt, reply);
    var donor_toks: [prompt + reply]u32 = undefined;
    @memcpy(donor_toks[0..prompt], toks[0..prompt]);
    for (donor_toks[prompt..], 0..) |*t, i| t.* = @intCast(700_000 + i);
    _ = try hc.commitWithRing(&donor, &donor_toks, false, 0, 0, null, null, null, null, prompt, .{ .prompt_end = donor_cp });

    // The side request: the conversation plus an appended reminder, restored off the donor.
    var fork = try KVCache.init(testing.allocator, n_layers);
    defer fork.deinit();
    fork.setSwaRing(window);
    var moe_off: usize = 0;
    const fork_hit = try hc.lookupAndRestore(&fork, &moe_off, null, s, toks[0..fork_prompt], false, 0, null, null);
    try testing.expectEqual(@as(usize, prompt), fork_hit.matched);
    try ringFill(&fork, s, n_layers, window, prompt, fork_prompt, 64);
    var fork_cp = try fork.ringCheckpoint(fork_prompt, s);
    errdefer if (fork_cp) |*c| c.deinit();
    try ringFill(&fork, s, n_layers, window, fork_prompt, fork_prompt + reply, 16);
    const moved_cp = fork_cp;
    fork_cp = null;
    _ = try hc.commitWithRing(&fork, &toks, false, 0, 0, null, null, null, null, fork_prompt, .{ .prompt_end = moved_cp });

    // The fork entry bills its own end, its own checkpoint and the inherited one.
    var end = try fork.snapshotRetained(s);
    const end_bytes = HotPrefixCache.snapshotBytes(&end);
    end.deinit();
    var cp_bytes: u64 = 0;
    for (fork.entries) |*e| {
        if (e.ringed) cp_bytes += (window + model_mod.ModelConfig.SWA_RING_CHECKPOINT_BACKOFF) * (ringRowBytes(e.keys) + ringRowBytes(e.values));
    }
    for (hc.entries.items) |*e| {
        if (e.tokens.len == toks.len) try testing.expectEqual(end_bytes + 2 * cp_bytes, e.kv_bytes);
    }

    // An unrelated request under the count cap evicts the donor, the LRU entry.
    var other = try KVCache.init(testing.allocator, n_layers);
    defer other.deinit();
    other.setSwaRing(window);
    try ringFill(&other, s, n_layers, window, 0, 300, 64);
    var other_toks: [300]u32 = undefined;
    for (&other_toks, 0..) |*t, i| t.* = @intCast(500_000 + i);
    _ = try hc.commitWithState(&other, &other_toks, false, 0, null, null, null);
    for (hc.entries.items) |*e| try testing.expect(e.tokens.len != donor_toks.len);

    // The next main turn shares the conversation and diverges where the reminder was appended.
    var next: [prompt + tail + 40]u32 = undefined;
    @memcpy(next[0 .. prompt + tail], toks[0 .. prompt + tail]);
    for (next[prompt + tail ..], 0..) |*t, i| t.* = @intCast(900_000 + i);
    var restored = try KVCache.init(testing.allocator, n_layers);
    defer restored.deinit();
    restored.setSwaRing(window);
    const hit = try hc.lookupAndRestore(&restored, &moe_off, null, s, &next, false, 0, null, null);
    try testing.expectEqual(@as(usize, prompt), hit.matched);
    try expectRingContinuesCold(&restored, s, n_layers, window, prompt, next.len);
}

/// Feed `restored` from `from` to `to`, then one more row: every layer must read exactly what
/// a cache fed from row 0 reads.
fn expectRingContinuesCold(restored: *KVCache, s: mlx.mlx_stream, n_layers: u32, window: u32, from: u32, to: u32) !void {
    var ref = try KVCache.init(testing.allocator, n_layers);
    defer ref.deinit();
    ref.setSwaRing(window);
    try ringFill(&ref, s, n_layers, window, 0, to, 64);
    try ringFill(restored, s, n_layers, window, from, to, 40);
    var li: u32 = 0;
    while (li < n_layers) : (li += 1) {
        var rv = try ringWriteLayer(&ref, s, li, to, 1, window);
        defer rv.deinit();
        var sv = try ringWriteLayer(restored, s, li, to, 1, window);
        defer sv.deinit();
        try testing.expectEqual(@as(f32, 0), try transformer_mod.maxAbsDiffF32(rv.k, sv.k, s));
        try testing.expectEqual(@as(f32, 0), try transformer_mod.maxAbsDiffF32(rv.v, sv.v, s));
        try testing.expectEqual(ref.absSeqLen(li), restored.absSeqLen(li));
    }
}

test "a ringed entry persists to the SSD tier and restores only at a ring checkpoint after a restart" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    const window: u32 = 8;
    const n_layers: u32 = 4;
    const prompt: u32 = 700;
    const reply: u32 = 600;

    var toks: [prompt + reply]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
    try persistRingTurn(io, base, "fp-ring", &toks, prompt, window, n_layers);

    var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-ring", 0, 128);
    defer hc2.deinit();
    var moe_off: usize = 0;

    // The next turn diverges right after the prompt: the prompt-end checkpoint serves it.
    var next: [prompt + 40]u32 = undefined;
    @memcpy(next[0 .. prompt + 1], toks[0 .. prompt + 1]);
    for (next[prompt + 1 ..], 0..) |*t, i| t.* = @intCast(900_000 + i);
    var a = try KVCache.init(testing.allocator, n_layers);
    defer a.deinit();
    a.setSwaRing(window);
    const hit = try hc2.lookupAndRestore(&a, &moe_off, null, s, &next, false, 0, null, null);
    try testing.expectEqual(@as(usize, prompt), hit.matched);
    // A ring restore's bill was never measured, so admission credits none of its rows.
    try testing.expect(!hit.ownsRestoredRows());
    try expectRingContinuesCold(&a, s, n_layers, window, prompt, next.len);

    // A verbatim re-send restores at its last token off the ring the entry ended with.
    var b = try KVCache.init(testing.allocator, n_layers);
    defer b.deinit();
    b.setSwaRing(window);
    const full = try hc2.lookupAndRestore(&b, &moe_off, null, s, &toks, false, 0, null, null);
    try testing.expect(full.full_match);
    try testing.expectEqual(@as(usize, toks.len - 1), full.matched);
    try testing.expect(!full.ownsRestoredRows());
    try expectRingContinuesCold(&b, s, n_layers, window, toks.len - 1, toks.len);

    // A divergence no ring checkpoint reaches cold-prefills.
    var early: [prompt]u32 = undefined;
    @memcpy(early[0..400], toks[0..400]);
    for (early[400..], 0..) |*t, i| t.* = @intCast(800_000 + i);
    var c = try KVCache.init(testing.allocator, n_layers);
    defer c.deinit();
    c.setSwaRing(window);
    const miss = try hc2.lookupAndRestore(&c, &moe_off, null, s, &early, false, 0, null, null);
    try testing.expectEqual(@as(usize, 0), miss.matched);
}

/// One MiMo-shaped turn (`ringTurn`) committed and flushed to a fresh SSD tier at `base/fp`.
fn persistRingTurn(io: std.Io, base: []const u8, fp: []const u8, toks: []const u32, prompt: u32, window: u32, n_layers: u32) !void {
    const s = mlx.gpuStream();
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, fp, 0, 128);
    defer hc.deinit();
    var live = try KVCache.init(testing.allocator, n_layers);
    defer live.deinit();
    const cp = try ringTurn(&live, s, n_layers, window, prompt, @intCast(toks.len - prompt));
    // Without its restore points a ringed cache is not persistable.
    try testing.expectEqual(kv_disk_cache.PersistOutcome.skipped, try hc.disk.?.appendCommit(live.entries, live.step, live.config, toks, false, null, s));
    _ = try hc.commitWithRing(&live, toks, false, 0, 0, null, null, null, null, prompt, .{ .prompt_end = cp });
    hc.flushPendingDisk(s);
    try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
}

test "a ringed SSD entry bills its ring files, and a truncated one is salvaged away at scan" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    const window: u32 = 8;
    const n_layers: u32 = 4;
    const prompt: u32 = 700;

    var toks: [prompt + 600]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
    try persistRingTurn(io, base, "fp-ring-salvage", &toks, prompt, window, n_layers);

    {
        // The tier bills exactly the files it wrote: chunks, the token record and both ring files.
        var d = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-ring-salvage", 0, 128);
        defer d.deinit();
        const e = &d.entries.items[0];
        try testing.expectEqual(@as(usize, 2), e.rings.len);
        try testing.expectEqual(@as(u32, prompt), e.rings[0].pos);
        try testing.expectEqual(@as(u32, toks.len), e.rings[1].pos);
        var on_disk: u64 = 0;
        var dir = try tmp.dir.openDir(io, "fp-ring-salvage/e1", .{ .iterate = true });
        defer dir.close(io);
        var it = dir.iterate();
        while (try it.next(io)) |f| {
            if (std.mem.eql(u8, f.name, "meta.json")) continue;
            on_disk += (try dir.statFile(io, f.name, .{})).size;
        }
        try testing.expectEqual(on_disk, d.total_bytes);
    }

    // kill -9 mid-write: the prompt-end file no longer matches its record.
    try tmp.dir.writeFile(io, .{ .sub_path = "fp-ring-salvage/e1/r0000700.safetensors", .data = "trunc" });
    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-ring-salvage", 0, 128);
        defer hc.deinit();
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entries.items[0].rings.len);
        var next: [prompt + 40]u32 = undefined;
        @memcpy(next[0 .. prompt + 1], toks[0 .. prompt + 1]);
        for (next[prompt + 1 ..], 0..) |*t, i| t.* = @intCast(900_000 + i);
        var a = try KVCache.init(testing.allocator, n_layers);
        defer a.deinit();
        a.setSwaRing(window);
        var moe_off: usize = 0;
        try testing.expectEqual(@as(usize, 0), (try hc.lookupAndRestore(&a, &moe_off, null, s, &next, false, 0, null, null)).matched);
        var b = try KVCache.init(testing.allocator, n_layers);
        defer b.deinit();
        b.setSwaRing(window);
        try testing.expectEqual(@as(usize, toks.len - 1), (try hc.lookupAndRestore(&b, &moe_off, null, s, &toks, false, 0, null, null)).matched);
    }

    // With no ring file left the chunks restore nothing: the entry goes.
    try tmp.dir.writeFile(io, .{ .sub_path = "fp-ring-salvage/e1/r0001300.safetensors", .data = "trunc" });
    var d = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-ring-salvage", 0, 128);
    defer d.deinit();
    try testing.expectEqual(@as(usize, 0), d.entryCount());
}

test "an SSD entry without ring rows is never restored into a ringed cache" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    const window: u32 = 8;
    const n_layers: u32 = 4;

    var toks: [700]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
    {
        // Every layer holds the whole prefix (no ring), as a pre-ring manifest describes it.
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-ring-old", 0, 128);
        defer hc.deinit();
        var plain = try KVCache.init(testing.allocator, n_layers);
        defer plain.deinit();
        try testFillCache(&plain, s, n_layers, toks.len);
        _ = try hc.commit(&plain, &toks, false);
        hc.flushPendingDisk(s);
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    }
    var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-ring-old", 0, 128);
    defer hc2.deinit();
    var next: [800]u32 = undefined;
    @memcpy(next[0..toks.len], &toks);
    for (next[toks.len..], 0..) |*t, i| t.* = @intCast(900_000 + i);
    var ringed = try KVCache.init(testing.allocator, n_layers);
    defer ringed.deinit();
    ringed.setSwaRing(window);
    var moe_off: usize = 0;
    const miss = try hc2.lookupAndRestore(&ringed, &moe_off, null, s, &next, false, 0, null, null);
    try testing.expectEqual(@as(usize, 0), miss.matched);
    for (ringed.entries) |*e| try testing.expect(!e.initialized);
}

test "HotPrefixCache: disk tier restores across a fresh cache instance (restart shape)" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const base = buf[0..root_len];

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // Session 1: commit through the NORMAL RAM path, then flush to disk
    // (the post-markFinished call the scheduler makes).
    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-hc", 0, 128);
        defer hc.deinit();

        var cache = try KVCache.init(testing.allocator, 2);
        defer cache.deinit();
        try testFillCache(&cache, s, 2, 600);
        _ = try hc.commit(&cache, &tokens, false);
        try testing.expect(hc.disk_dirty);
        hc.flushPendingDisk(s);
        try testing.expect(!hc.disk_dirty);
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    }

    // Session 2 ("server restart"): fresh RAM cache, fresh tier over the same
    // root. The lookup must land on the SSD tier and restore the prefix.
    {
        var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-hc", 0, 128);
        defer hc2.deinit();
        try testing.expectEqual(@as(usize, 0), hc2.entryCount()); // RAM empty
        try testing.expectEqual(@as(usize, 1), hc2.disk.?.entryCount());

        var cache2 = try KVCache.init(testing.allocator, 2);
        defer cache2.deinit();
        var moe_off: usize = 0;
        const res = try hc2.lookupAndRestore(&cache2, &moe_off, null, s, &tokens, false, 0, null, null);
        // Full match: identical re-issue semantics — truncate to len-1 and
        // re-forward the last token, exactly like a RAM full-match hit.
        try testing.expect(res.full_match);
        try testing.expectEqual(@as(usize, 599), res.matched);
        try testing.expectEqual(@as(usize, 599), cache2.step);
        try testing.expectEqual(@as(usize, 599), moe_off);
        for (cache2.entries) |*e| {
            try testing.expect(e.initialized);
            try testing.expectEqual(@as(usize, 599), e.offset);
        }

        // Diverged-tail shape: shares the first 400 tokens, then differs.
        // Restore must land at 400 and leave the tail to prefill.
        var tokens_div: [700]u32 = undefined;
        for (&tokens_div, 0..) |*t, i| t.* = if (i < 400) tokens[i] else @intCast(i + 500_000);
        var cache3 = try KVCache.init(testing.allocator, 2);
        defer cache3.deinit();
        var moe_off3: usize = 0;
        const res3 = try hc2.lookupAndRestore(&cache3, &moe_off3, null, s, &tokens_div, false, 0, null, null);
        try testing.expect(!res3.full_match);
        try testing.expectEqual(@as(usize, 400), res3.matched);
        try testing.expectEqual(@as(usize, 400), cache3.step);
        // A diverged short prefix must read ONLY the chunks covering the
        // usable 400 positions (ceil(400/128) = 4), NOT the whole 600-token
        // stored entry (5 chunks). Loading the full entry to serve a short
        // shared prefix makes a diverged "hit" slower than a cold prefill.
        try testing.expectEqual(@as(u32, 4), hc2.disk.?.chunks_loaded_last);
    }
}

test "HotPrefixCache: budget decline reports a status, not a silent success" {
    // Live 2026-09-07: every oversized decline from a cancelled prefill
    // logged "[hot-cache] committed N/M" at the scheduler because the
    // decline exits commitWithMediaState via plain return — a 122k agent
    // session looked cached while it cold-prefilled ~95k tokens per retry.
    // The commit contract must be observable: ok / kept_resident / declined.
    const s = mlx.gpuStream();
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 16 * 1024);
    defer hc.deinit();

    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try testFillCache(&cache, s, 2, 600);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // 16 KB budget against ~128 B/token rows: even the 256-token commit
    // floor cannot fit, so the candidate must be DECLINED — and the caller
    // must be able to SEE that instead of mistaking it for a commit.
    const st = try hc.commit(&cache, &tokens, false);
    try testing.expect(st == .declined);
    try testing.expectEqual(@as(usize, 0), hc.entryCount());

    // A candidate that fits commits with its effective token length.
    var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc2.deinit();
    var cache2 = try KVCache.init(testing.allocator, 2);
    defer cache2.deinit();
    try testFillCache(&cache2, s, 2, 600);
    const st2 = try hc2.commit(&cache2, &tokens, false);
    try testing.expect(st2 == .ok);
    try testing.expectEqual(@as(usize, 600), st2.ok);
}

test "HotPrefixCache: an entry shorter than its media boundary is pure text" {
    // Live 2026-09-07: a cancelled prefill that forwarded fewer tokens than
    // the request's first media position kept the PIXEL vision_key while
    // the boundary was dropped (scheduler shape: media_start=null, key kept)
    // — a boundary-less cross-key entry the lookup conservatively rejects.
    // The with-image and no-image retry shapes of one conversation locked
    // each other out of their shared 121819-token text prefix. An entry
    // whose [0, len) covers no media rows is text: key it as text.
    const s = mlx.gpuStream();
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();

    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try testFillCache(&cache, s, 2, 600);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // Cancel shape (the scheduler passes the boundary raw): media declared
    // at 500, entry covers [0, 400) — no media rows inside, so the entry
    // must be text-keyed.
    const st = try hc.commitWithMediaState(&cache, tokens[0..400], false, 0xABCD, 0, 500, null, null, null, 400);
    try testing.expect(st == .ok);
    try testing.expectEqual(@as(u64, 0), hc.entries.items[0].vision_key);
    try testing.expect(hc.entries.items[0].media_start == null);

    // Legacy shape (commitWithState's contract): a null boundary KEEPS the
    // caller's pixel key — boundary-less keyed entries stay possible for
    // callers that model their own media state.
    var cache_b = try KVCache.init(testing.allocator, 2);
    defer cache_b.deinit();
    try testFillCache(&cache_b, s, 2, 600);
    _ = try hc.commitWithState(&cache_b, tokens[0..400], false, 0xBEEF, null, null, null);
    try testing.expectEqual(@as(u64, 0xBEEF), hc.entries.items[1].vision_key);

    // The no-image retry shape (vision_key 0, no media) must restore it.
    {
        var c2 = try KVCache.init(testing.allocator, 2);
        defer c2.deinit();
        var moe: usize = 0;
        const res = try hc.lookupAndRestoreWithMedia(&c2, &moe, null, s, &tokens, false, 0, null, &.{}, null, null, null, false);
        try testing.expectEqual(@as(usize, 400), res.matched);
    }
    // A different-image request caps at ITS media boundary (500 > 400 here,
    // so the whole text entry is reusable through it).
    {
        var c3 = try KVCache.init(testing.allocator, 2);
        defer c3.deinit();
        var moe: usize = 0;
        const res = try hc.lookupAndRestoreWithMedia(&c3, &moe, null, s, &tokens, false, 0xFFFF, 500, &.{}, null, null, null, false);
        try testing.expectEqual(@as(usize, 400), res.matched);
    }

    // Trim shape: media INSIDE the candidate, budget forces a trim below the
    // boundary — the trimmed entry is text and must re-key to 0.
    var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 32 * 1024);
    defer hc2.deinit();
    var cache2 = try KVCache.init(testing.allocator, 2);
    defer cache2.deinit();
    try testFillCache(&cache2, s, 2, 600);
    const st2 = try hc2.commitWithMediaState(&cache2, &tokens, false, 0xABCD, 0, 300, null, null, null, tokens.len);
    try testing.expect(st2 == .ok);
    try testing.expectEqual(@as(usize, 256), st2.ok); // 32 KB / 128 B rows
    try testing.expectEqual(@as(u64, 0), hc2.entries.items[0].vision_key);
    try testing.expect(hc2.entries.items[0].media_start == null);
}

test "HotPrefixCache: media request restores the pre-media text prefix from SSD" {
    // Live 2026-09-07: the SSD-tier lookup skipped ANY request with a
    // non-zero vision_key (`vision_key != 0 → break :disk`) — a 122k agent
    // session whose LAST turn carries one screenshot never touched the
    // 40 GB of persisted text prefixes, though 99.8% of the prompt is
    // pixel-independent text. Disk entries are text-only by construction
    // (the flush refuses vision-keyed entries), so the restore is exactly
    // the RAM path's pre-media semantics: cap at the request's media_start.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const base = buf[0..root_len];

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // Session 1: a text-only conversation commits + flushes to disk.
    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-vision-disk", 0, 128);
        defer hc.deinit();
        var cache = try KVCache.init(testing.allocator, 2);
        defer cache.deinit();
        try testFillCache(&cache, s, 2, 600);
        _ = try hc.commit(&cache, &tokens, false);
        hc.flushPendingDisk(s);
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    }

    // Session 2 (RAM empty): an image-bearing request over the same text —
    // the disk tier must serve the shared text prefix, capped at the
    // request's media boundary (400), reading only the covering chunks.
    {
        var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-vision-disk", 0, 128);
        defer hc2.deinit();

        var cache2 = try KVCache.init(testing.allocator, 2);
        defer cache2.deinit();
        var moe: usize = 0;
        const res = try hc2.lookupAndRestoreWithMedia(&cache2, &moe, null, s, &tokens, false, 0xDEAD, 400, &.{}, null, null, null, false);
        try testing.expect(!res.full_match);
        try testing.expectEqual(@as(usize, 400), res.matched);
        try testing.expectEqual(@as(usize, 400), cache2.step);
        try testing.expectEqual(@as(u32, 4), hc2.disk.?.chunks_loaded_last); // ceil(400/128)
    }
}

test "HotPrefixCache: dflash + mtp snapshots survive the SSD tier across a restart" {
    // A disk-tier restore forwards NO trunk layers, so state derived from
    // trunk hiddens (dflash context, MTP history) started EMPTY on every
    // disk hit — multi-turn across a restart drafted blind (the same
    // 92.6% → 66.5% acceptance class the RAM tier fixed). v4 persists both
    // in the entry's spec sidecar and restores them under the RAM tier's
    // exact clamp rule.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const base = buf[0..root_len];

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // Session 1: commit with BOTH spec payloads through the normal RAM path,
    // then flush (the post-markFinished call the scheduler makes).
    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-spec-hc", 0, 128);
        defer hc.deinit();

        var trunk = try KVCache.init(testing.allocator, 2);
        defer trunk.deinit();
        try testFillCache(&trunk, s, 2, 600);
        var assist = try KVCache.init(testing.allocator, 2);
        defer assist.deinit();
        try testFillCache(&assist, s, 2, 600);
        var hist = try KVCache.init(testing.allocator, 1);
        defer hist.deinit();
        try testFillCache(&hist, s, 1, 600);

        _ = try hc.commitWithSsm(&trunk, &tokens, false, null, .{ .cache = &assist, .base_pos = 0 }, .{ .cache = &hist, .base_pos = 0 });
        hc.flushPendingDisk(s);
        try testing.expect(!hc.disk_dirty);
        try testing.expect(hc.disk.?.entries.items[0].spec_dflash != null);
        try testing.expect(hc.disk.?.entries.items[0].spec_mtp != null);
    }

    // Session 2 ("server restart"): RAM empty, disk serves the prefix AND
    // both spec snapshots, clamped to the trunk's matched length.
    {
        var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-spec-hc", 0, 128);
        defer hc2.deinit();
        try testing.expectEqual(@as(usize, 0), hc2.entryCount());

        var trunk2 = try KVCache.init(testing.allocator, 2);
        defer trunk2.deinit();
        var dfl = try KVCache.init(testing.allocator, 2);
        defer dfl.deinit();
        var mtp_dst = try KVCache.init(testing.allocator, 1);
        defer mtp_dst.deinit();
        var moe_off: usize = 0;
        var dbase: usize = 99;
        var mbase: usize = 99;
        const res = try hc2.lookupAndRestore(
            &trunk2,
            &moe_off,
            null,
            s,
            &tokens,
            false,
            0,
            .{ .cache = &dfl, .base_pos = &dbase },
            .{ .cache = &mtp_dst, .base_pos = &mbase },
        );
        // Full match: identical re-issue → trunk lands at 599; both spec
        // caches clamp to base + step == matched.
        try testing.expect(res.full_match);
        try testing.expectEqual(@as(usize, 599), res.matched);
        try testing.expectEqual(@as(?usize, 0), res.dflash_base);
        try testing.expectEqual(@as(usize, 0), dbase);
        try testing.expectEqual(@as(usize, 599), dfl.step);
        try testing.expectEqual(@as(?usize, 0), res.mtp_base);
        try testing.expectEqual(@as(usize, 599), mtp_dst.step);

        // A geometry the persisted snap doesn't fit starts BLIND, never
        // wrong: a 3-layer dflash target declines.
        var trunk3 = try KVCache.init(testing.allocator, 2);
        defer trunk3.deinit();
        var dfl3 = try KVCache.init(testing.allocator, 3);
        defer dfl3.deinit();
        var moe3: usize = 0;
        var dbase3: usize = 42;
        const res3 = try hc2.lookupAndRestore(
            &trunk3,
            &moe3,
            null,
            s,
            &tokens,
            false,
            0,
            .{ .cache = &dfl3, .base_pos = &dbase3 },
            null,
        );
        try testing.expectEqual(@as(usize, 599), res3.matched);
        try testing.expectEqual(@as(?usize, null), res3.dflash_base);
        try testing.expectEqual(@as(usize, 0), dfl3.step);
        try testing.expectEqual(@as(usize, 42), dbase3); // untouched
    }
}

test "HotPrefixCache: an SSD entry that fails to restore is poisoned, not retried" {
    mlx.installErrorHandler();
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const base = buf[0..root_len];

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-bad", 0, 128);
        defer hc.deinit();
        var cache = try KVCache.init(testing.allocator, 2);
        defer cache.deinit();
        try testFillCache(&cache, s, 2, 600);
        _ = try hc.commit(&cache, &tokens, false);
        hc.flushPendingDisk(s);
    }
    // A chunk whose bytes rotted at the same size passes the scan's size check and fails the load.
    const chunk = "fp-bad/e1/c000000.safetensors";
    const st = try tmp.dir.statFile(io, chunk, .{});
    const junk = try testing.allocator.alloc(u8, @intCast(st.size));
    defer testing.allocator.free(junk);
    @memset(junk, 0xff);
    try tmp.dir.writeFile(io, .{ .sub_path = chunk, .data = junk });

    var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-bad", 0, 128);
    defer hc2.deinit();
    const id = hc2.disk.?.entries.items[0].id;
    var cache2 = try KVCache.init(testing.allocator, 2);
    defer cache2.deinit();
    var moe_off: usize = 0;
    const res = try hc2.lookupAndRestore(&cache2, &moe_off, null, s, &tokens, false, 0, null, null);
    try testing.expectEqual(@as(usize, 0), res.matched);
    try testing.expect(hc2.disk.?.entryPoisoned(id));
    mlx.dropLatchedErrorUnless(false);
}

test "HotPrefixCache: RAM match at least as long as disk skips the SSD read" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const base = buf[0..root_len];

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-skip", 0, 128);
    defer hc.deinit();

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);
    _ = try hc.commit(&cache, &tokens, false);
    hc.flushPendingDisk(s);
    const disk_uses_before = hc.disk.?.counter;

    // Same prompt again: the RAM entry covers it fully, so the disk tier's
    // LRU counter must not move (no restore happened).
    var cache2 = try KVCache.init(testing.allocator, 1);
    defer cache2.deinit();
    var moe_off: usize = 0;
    const res = try hc.lookupAndRestore(&cache2, &moe_off, null, s, &tokens, false, 0, null, null);
    try testing.expect(res.full_match);
    try testing.expectEqual(disk_uses_before, hc.disk.?.counter);
}

test "HotPrefixCache: invalidation propagates to the disk tier" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const base = buf[0..root_len];

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-inv", 0, 128);
    defer hc.deinit();

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);
    _ = try hc.commit(&cache, &tokens, false);
    hc.flushPendingDisk(s);
    try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());

    hc.invalidateAll("test poison");
    try testing.expectEqual(@as(usize, 0), hc.entryCount());
    try testing.expectEqual(@as(usize, 0), hc.disk.?.entryCount());
    try testing.expect(!hc.disk_dirty);
}

test "HotPrefixCache: findBestMatch isolates affine 4-bit from affine 8-bit" {
    // Regression for the cross-bit-width hit that crashed SDPA in
    // tests/test_kv_quant_per_request.sh. With Entry.scheme tracking only
    // the `Scheme` enum, `affine(4)` and `affine(8)` both matched as
    // `.affine` and a 4-bit snapshot would be restored into an 8-bit slot
    // → broadcast_shapes (1,H,1,64) vs (1,H,1,32) MLX abort. After moving
    // to a full-`KVQuantConfig` filter, the two are distinct keys and
    // can't alias.
    var cache = HotPrefixCache.init(testing.allocator, 4);
    defer cache.deinit();

    const ids = try testing.allocator.dupe(u32, &[_]u32{ 1, 2, 3, 4, 5 });
    try cache.entries.append(testing.allocator, .{
        .tokens = ids,
        .has_tools = false,
        .snapshot = .{ .entries = try testing.allocator.alloc(transformer_mod.KVCacheEntry, 0), .step = 0, .allocator = testing.allocator, .config = kv_quant.KVQuantConfig.affine(4) },
        .last_used = 1,
        .quant_config = kv_quant.KVQuantConfig.affine(4),
        .kv_bytes = 0,
        .ssm_checkpoints = null,
        .ssm_bytes = 0,
    });

    const lookup_ids = [_]u32{ 1, 2, 3, 4, 5, 6 };
    // Matching config (affine 4) hits the entry.
    const hit = cache.findBestMatch(&lookup_ids, false, 0, kv_quant.KVQuantConfig.affine(4)).?;
    try testing.expectEqual(@as(usize, 0), hit.idx);
    try testing.expectEqual(@as(usize, 5), hit.shared);
    // Same Scheme (.affine) but different bits MUST NOT hit — that's the
    // cross-scheme buffer-layout crash this filter guards against.
    try testing.expectEqual(@as(?@TypeOf(hit), null), cache.findBestMatch(&lookup_ids, false, 0, kv_quant.KVQuantConfig.affine(8)));
    try testing.expectEqual(@as(?@TypeOf(hit), null), cache.findBestMatch(&lookup_ids, false, 0, kv_quant.KVQuantConfig.affine(6)));
    // Dense query against an affine entry: also null (existing guarantee).
    try testing.expectEqual(@as(?@TypeOf(hit), null), cache.findBestMatch(&lookup_ids, false, 0, kv_quant.KVQuantConfig.dense));
}

// ── Phase 3: two-tier hybrid restore (Qwen 3.5/3.6 GatedDeltaNet) ──

const conv_shape_pc = [_]c_int{ 1, 3, 8 };
const ssm_shape_pc = [_]c_int{ 1, 2, 4, 4 };

fn pcArange(s: mlx.mlx_stream, shape: []const c_int, base: f64) mlx.mlx_array {
    var count: f64 = 1;
    for (shape) |d| count *= @floatFromInt(d);
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    _ = mlx.mlx_arange(&flat, base, base + count, 1.0, .float32, s);
    var out = mlx.mlx_array_new();
    _ = mlx.mlx_reshape(&out, flat, shape.ptr, @intCast(shape.len), s);
    _ = mlx.mlx_array_eval(out);
    return out;
}

fn pcSsmVal(arr: mlx.mlx_array, idx: usize, s: mlx.mlx_stream) f32 {
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    _ = mlx.mlx_astype(&f, arr, .float32, s);
    _ = mlx.mlx_array_eval(f);
    return mlx.mlx_array_data_float32(f).?[idx];
}

fn pcBuildHybrid(s: mlx.mlx_stream, conv_base: f64, ssm_base: f64) [3]SSMCacheEntry {
    return .{
        .{ .conv_state = pcArange(s, &conv_shape_pc, conv_base), .ssm_state = pcArange(s, &ssm_shape_pc, ssm_base), .initialized = true },
        .{ .conv_state = pcArange(s, &conv_shape_pc, conv_base + 10_000), .ssm_state = mlx.mlx_array_new(), .initialized = true },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
    };
}

fn pcFreeHybrid(e: *[3]SSMCacheEntry) void {
    for (e) |*x| {
        _ = mlx.mlx_array_free(x.conv_state);
        _ = mlx.mlx_array_free(x.ssm_state);
    }
}

fn pcEmptySsm() [3]SSMCacheEntry {
    return .{
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
    };
}

// A new image changes the KV only when its dynamic placeholder rows are
// forwarded. The text prefix before that boundary remains valid even though
// the media hash changes. Hybrid models must restore the last checkpoint at
// or before the boundary, never a later checkpoint whose SSM state has seen
// the old pixels.
test "HotPrefixCache: hybrid lookup reuses only the prefix before changed media" {
    const s = mlx.gpuStream();
    const media_start: usize = 8;
    const cached_tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 900, 900, 20, 21 };
    const lookup_tokens = cached_tokens;

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();

    var source_cache = try KVCache.init(testing.allocator, 3);
    defer source_cache.deinit();
    try testFillCache(&source_cache, s, 3, cached_tokens.len);
    var source_ssm = pcBuildHybrid(s, 123.0, 456.0);
    defer pcFreeHybrid(&source_ssm);
    const checkpoints = try testing.allocator.alloc(SSMCheckpoint, 2);
    checkpoints[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &source_ssm, media_start, s);
    checkpoints[1] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &source_ssm, media_start + 2, s);
    _ = try hc.commitWithMediaState(
        &source_cache,
        &cached_tokens,
        false,
        0x1111,
        0,
        media_start,
        checkpoints,
        null,
        null,
        cached_tokens.len,
    );

    var target_cache = try KVCache.init(testing.allocator, 3);
    defer target_cache.deinit();
    var target_ssm = pcEmptySsm();
    defer pcFreeHybrid(&target_ssm);
    var moe_off: usize = 0;
    const result = try hc.lookupAndRestoreWithMedia(
        &target_cache,
        &moe_off,
        &target_ssm,
        s,
        &lookup_tokens,
        false,
        0x2222,
        media_start,
        &.{},
        null,
        null,
        null,
        false,
    );

    try testing.expectEqual(media_start, result.matched);
    try testing.expectEqual(media_start, target_cache.step);
    try testing.expectEqual(media_start, moe_off);
    try testing.expectEqual(@as(f32, 123.0), pcSsmVal(target_ssm[0].conv_state, 0, s));
}

// A vision turn can move the current image span when the same image becomes
// conversation history. The newest entry then has the longest raw token match
// (ending exactly at the old image boundary), but its first SSM checkpoint can
// sit just AFTER that boundary. An older entry for the same pixels may have a
// slightly shorter token match with a usable checkpoint. Picking by raw token
// match alone turns that recoverable case into a full cold prefill.
test "HotPrefixCache: hybrid lookup falls back to the best restorable RAM entry" {
    const s = mlx.gpuStream();
    const vision_key: u64 = 0xdecaf;
    const older_tokens = [_]u32{ 1, 2, 3, 4, 5, 90, 91, 92, 93, 94 };
    const newer_tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 80, 81, 82 };
    const lookup_tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 70, 71, 72 };

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();

    var older_cache = try KVCache.init(testing.allocator, 3);
    defer older_cache.deinit();
    try testFillCache(&older_cache, s, 3, older_tokens.len);
    var older_ssm = pcBuildHybrid(s, 100.0, 500.0);
    defer pcFreeHybrid(&older_ssm);
    const older_cps = try testing.allocator.alloc(SSMCheckpoint, 1);
    older_cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &older_ssm, 4, s);
    _ = try hc.commitWithState(&older_cache, &older_tokens, false, vision_key, older_cps, null, null);

    var newer_cache = try KVCache.init(testing.allocator, 3);
    defer newer_cache.deinit();
    try testFillCache(&newer_cache, s, 3, newer_tokens.len);
    var newer_ssm = pcBuildHybrid(s, 300.0, 700.0);
    defer pcFreeHybrid(&newer_ssm);
    const newer_cps = try testing.allocator.alloc(SSMCheckpoint, 1);
    // The raw match with this entry is 7, so this checkpoint cannot restore it.
    newer_cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &newer_ssm, 8, s);
    _ = try hc.commitWithState(&newer_cache, &newer_tokens, false, vision_key, newer_cps, null, null);

    var target_cache = try KVCache.init(testing.allocator, 3);
    defer target_cache.deinit();
    var target_ssm = pcEmptySsm();
    defer pcFreeHybrid(&target_ssm);
    var moe_off: usize = 0;
    const result = try hc.lookupAndRestore(
        &target_cache,
        &moe_off,
        &target_ssm,
        s,
        &lookup_tokens,
        false,
        vision_key,
        null,
        null,
    );

    // The newer 7-token raw match is unusable; the older checkpoint at 4 is
    // still vastly better than a cold prefill and must win the hybrid lookup.
    try testing.expectEqual(@as(usize, 4), result.matched);
    try testing.expectEqual(@as(usize, 4), target_cache.step);
    try testing.expectEqual(@as(usize, 4), moe_off);
    try testing.expectEqual(@as(f32, 100.0), pcSsmVal(target_ssm[0].conv_state, 0, s));
}

// Bar: an image turn commits under the PIXEL key, so the pre-media checkpoints
// it can only get from the text entry must survive that entry's eviction.
test "HotPrefixCache: a text turn after image turns restores the pre-media prefix" {
    const s = mlx.gpuStream();
    const media_start: usize = 12;
    const image_key: u64 = 0xF00D;

    // Turn N: the text conversation, with checkpoints spanning the prefix.
    const text_tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    // Turn N+1: the same conversation plus an image at `media_start`.
    const image_tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 900, 900, 900, 900, 61, 62, 63, 64 };
    // Turn N+2: a text-only turn. The image message is history now, so the
    // token stream still agrees far past the boundary; only the pixel key
    // caps the reusable prefix at `media_start`.
    const later_tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 900, 900, 900, 900, 61, 62, 71, 72 };

    var hc = HotPrefixCache.initWithMem(testing.allocator, 1, 0);
    defer hc.deinit();

    var text_cache = try KVCache.init(testing.allocator, 3);
    defer text_cache.deinit();
    try testFillCache(&text_cache, s, 3, text_tokens.len);
    var text_ssm = pcBuildHybrid(s, 100.0, 500.0);
    defer pcFreeHybrid(&text_ssm);
    const text_cps = try testing.allocator.alloc(SSMCheckpoint, 3);
    text_cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &text_ssm, 4, s);
    text_cps[1] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &text_ssm, 8, s);
    text_cps[2] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &text_ssm, 10, s);
    _ = try hc.commitWithMediaState(&text_cache, &text_tokens, false, 0, 0, null, text_cps, null, null, text_tokens.len);

    // The image turn restores the pre-media text prefix (cross-key, capped
    // at the boundary) and prefills the rest.
    var image_cache = try KVCache.init(testing.allocator, 3);
    defer image_cache.deinit();
    var image_ssm = pcEmptySsm();
    defer pcFreeHybrid(&image_ssm);
    var image_off: usize = 0;
    const restored = try hc.lookupAndRestoreWithMedia(
        &image_cache,
        &image_off,
        &image_ssm,
        s,
        &image_tokens,
        false,
        image_key,
        media_start,
        &.{},
        null,
        null,
        null,
        false,
    );
    try testing.expectEqual(@as(usize, 10), restored.matched);

    // ... and commits under the pixel key. Its OWN prefill only reached
    // positions past the image, so its own checkpoints all sit above the
    // boundary; the count cap evicts the text entry.
    try testFillCache(&image_cache, s, 3, image_tokens.len);
    var image_state = pcBuildHybrid(s, 300.0, 700.0);
    defer pcFreeHybrid(&image_state);
    const image_cps = try testing.allocator.alloc(SSMCheckpoint, 2);
    image_cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &image_state, 16, s);
    image_cps[1] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &image_state, 18, s);
    _ = try hc.commitWithMediaState(&image_cache, &image_tokens, false, image_key, 0, media_start, image_cps, null, null, image_tokens.len);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());

    // The next text-only turn must still restore the pre-media prefix: the
    // rows below `media_start` are pure text and pixel-independent.
    var later_cache = try KVCache.init(testing.allocator, 3);
    defer later_cache.deinit();
    var later_ssm = pcEmptySsm();
    defer pcFreeHybrid(&later_ssm);
    var later_off: usize = 0;
    const result = try hc.lookupAndRestoreWithMedia(
        &later_cache,
        &later_off,
        &later_ssm,
        s,
        &later_tokens,
        false,
        0,
        null,
        &.{},
        null,
        null,
        null,
        false,
    );
    try testing.expectEqual(@as(usize, 10), result.matched);
    try testing.expectEqual(@as(usize, 10), later_cache.step);
    try testing.expectEqual(@as(usize, 10), later_off);
    try testing.expectEqual(@as(f32, 100.0), pcSsmVal(later_ssm[0].conv_state, 0, s));
}

// Class guard: no retention policy may thin away the highest checkpoint at or
// below an entry's media boundary.
test "HotPrefixCache: thinning spreads checkpoints across the whole record, media or not" {
    // The thinning used to protect the highest checkpoint at or below the first media row,
    // because the SSD record ended there. A record now persists whole, so that rule only sacrificed
    // the checkpoints ABOVE the image -- the very ones the next image-bearing turn restores from.
    const s = mlx.gpuStream();
    const media_start: usize = 8;

    var tokens: [24]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 1);

    var kept_positions: [2][3]usize = undefined;
    for ([_]bool{ true, false }, 0..) |media, arm| {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 2, 0);
        hc.ssm_checkpoint_max = 3;
        defer hc.deinit();

        var ssm = pcBuildHybrid(s, 100.0, 500.0);
        defer pcFreeHybrid(&ssm);

        var c1 = try KVCache.init(testing.allocator, 3);
        defer c1.deinit();
        try testFillCache(&c1, s, 3, 16);
        const first = try testing.allocator.alloc(SSMCheckpoint, 3);
        for (first, 0..) |*c, i| c.* = try transformer_mod.captureSsmCheckpoint(testing.allocator, &ssm, 2 + i * 2, s);
        _ = try hc.commitWithMediaState(&c1, tokens[0..16], false, if (media) 0xF00D else 0, 0, if (media) media_start else null, first, null, null, 16);

        // The next turn extends the same conversation, so the commit replaces the
        // entry and merges: seven checkpoints thinned down to the cap of three.
        var c2 = try KVCache.init(testing.allocator, 3);
        defer c2.deinit();
        try testFillCache(&c2, s, 3, 24);
        const second = try testing.allocator.alloc(SSMCheckpoint, 4);
        for (second, 0..) |*c, i| c.* = try transformer_mod.captureSsmCheckpoint(testing.allocator, &ssm, 12 + i * 2, s);
        _ = try hc.commitWithMediaState(&c2, &tokens, false, if (media) 0xF00D else 0, 0, if (media) media_start else null, second, null, null, tokens.len);

        try testing.expectEqual(@as(usize, 1), hc.entryCount());
        const kept = hc.entries.items[0].ssm_checkpoints.?;
        try testing.expectEqual(@as(usize, 3), kept.len);
        for (kept, &kept_positions[arm]) |*cp, *out| out.* = cp.pos;
        // Restore points above the image survive, which is what the SSD record needs.
        try testing.expect(kept[kept.len - 1].pos > media_start);
    }
    // The media boundary changes nothing about which checkpoints are kept.
    try testing.expectEqualSlices(usize, &kept_positions[1], &kept_positions[0]);
}

test "HotPrefixCache: hybrid SSM state restores from the SSD tier across a restart" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const base = buf[0..root_len];

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // Session 1: hybrid commit (KV + two SSM checkpoints) through the RAM
    // path, then the post-markFinished flush the scheduler makes.
    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-hyb", 0, 128);
        defer hc.deinit();

        var cache = try KVCache.init(testing.allocator, 3);
        defer cache.deinit();
        try testFillCache(&cache, s, 3, 600);

        var src256 = pcBuildHybrid(s, 100.0, 500.0);
        defer pcFreeHybrid(&src256);
        var src512 = pcBuildHybrid(s, 300.0, 700.0);
        defer pcFreeHybrid(&src512);
        const cps = try testing.allocator.alloc(SSMCheckpoint, 2);
        cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &src256, 256, s);
        cps[1] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &src512, 512, s);
        // commitWithSsm takes ownership of `cps`.
        _ = try hc.commitWithSsm(&cache, &tokens, false, cps, null, null);
        try testing.expect(hc.disk_dirty);
        hc.flushPendingDisk(s);
        try testing.expect(!hc.disk_dirty);
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    }

    // Session 2 ("restart"): fresh RAM cache + fresh tier over the same root.
    // A hybrid lookup must restore BOTH KV and SSM state from disk at the
    // highest checkpoint ≤ the match.
    {
        var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-hyb", 0, 128);
        defer hc2.deinit();
        try testing.expectEqual(@as(usize, 0), hc2.entryCount());
        try testing.expectEqual(@as(usize, 1), hc2.disk.?.entryCount());

        var cache2 = try KVCache.init(testing.allocator, 3);
        defer cache2.deinit();
        var ssm2 = pcEmptySsm();
        defer pcFreeHybrid(&ssm2);
        var moe_off: usize = 0;
        const res = try hc2.lookupAndRestore(&cache2, &moe_off, &ssm2, s, &tokens, false, 0, null, null);
        // Highest checkpoint ≤ 600 is 512 — never a full match on hybrid.
        try testing.expect(!res.full_match);
        try testing.expectEqual(@as(usize, 512), res.matched);
        try testing.expectEqual(@as(usize, 512), cache2.step);
        try testing.expectEqual(@as(usize, 512), moe_off);
        // SSM state at pos 512 installed (conv base 300 / ssm base 700).
        try testing.expect(ssm2[0].initialized);
        try testing.expectEqual(@as(f32, 300.0), pcSsmVal(ssm2[0].conv_state, 0, s));
        try testing.expectEqual(@as(f32, 700.0), pcSsmVal(ssm2[0].ssm_state, 0, s));
        try testing.expect(ssm2[1].ssm_state.ctx == null);
        try testing.expect(!ssm2[2].initialized);

        // Diverged tail: shares the first 400 tokens → clamps to the largest
        // checkpoint ≤ 400, which is 256.
        var tokens_div: [700]u32 = undefined;
        for (&tokens_div, 0..) |*t, i| t.* = if (i < 400) tokens[i] else @intCast(i + 500_000);
        var cache3 = try KVCache.init(testing.allocator, 3);
        defer cache3.deinit();
        var ssm3 = pcEmptySsm();
        defer pcFreeHybrid(&ssm3);
        var moe_off3: usize = 0;
        const res3 = try hc2.lookupAndRestore(&cache3, &moe_off3, &ssm3, s, &tokens_div, false, 0, null, null);
        try testing.expect(!res3.full_match);
        try testing.expectEqual(@as(usize, 256), res3.matched);
        try testing.expectEqual(@as(usize, 256), cache3.step);
        try testing.expectEqual(@as(f32, 100.0), pcSsmVal(ssm3[0].conv_state, 0, s));
        try testing.expectEqual(@as(f32, 500.0), pcSsmVal(ssm3[0].ssm_state, 0, s));
    }
}

test "HotPrefixCache: chunk-heavy hybrid flush still lands its SSM checkpoints" {
    // Live 2026-09-07: checkpoints were written AFTER the chunks from the
    // REMAINING per-flush budget — a turn appending ≥ the cap in chunks
    // (cancel-salvage retries: +16 chunks ≈ 544 MB vs a 512 MB cap) left
    // ZERO checkpoint budget every time, and the same-token catch-up path
    // never fires for a growing conversation. Every entry of the Sep-4
    // disk wave landed KV-only: unrestorable, gigabytes of dead chunks.
    // Checkpoints must come off the TOP of the budget, not the bottom.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const base = buf[0..root_len];

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // Session 1: hybrid commit + a flush budget tight enough that the KV
    // chunks alone would consume it (5 chunks ≈ 24 KB each vs a 2-chunk
    // cap) — the exact starvation shape, scaled down.
    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-cp-starve", 0, 128);
        defer hc.deinit();
        hc.disk.?.max_flush_bytes = 2 * 128 * 3 * 2 * 8 * 4; // two KV chunks

        var cache = try KVCache.init(testing.allocator, 3);
        defer cache.deinit();
        try testFillCache(&cache, s, 3, 600);
        var src = pcBuildHybrid(s, 100.0, 500.0);
        defer pcFreeHybrid(&src);
        const cps = try testing.allocator.alloc(SSMCheckpoint, 2);
        cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &src, 256, s);
        cps[1] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &src, 512, s);
        _ = try hc.commitWithSsm(&cache, &tokens, false, cps, null, null);
        hc.flushPendingDisk(s);
    }

    // Session 2 (RAM empty): the hybrid lookup must restore from the disk
    // entry's highest checkpoint ≤ its flushed kv_len (256 of 5 chunks) —
    // not cold-miss on a checkpoint-less entry.
    {
        var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-cp-starve", 0, 128);
        defer hc2.deinit();
        try testing.expectEqual(@as(usize, 1), hc2.disk.?.entryCount());

        var cache2 = try KVCache.init(testing.allocator, 3);
        defer cache2.deinit();
        var ssm2 = pcEmptySsm();
        defer pcFreeHybrid(&ssm2);
        var moe_off: usize = 0;
        const res = try hc2.lookupAndRestore(&cache2, &moe_off, &ssm2, s, &tokens, false, 0, null, null);
        try testing.expect(!res.full_match);
        try testing.expectEqual(@as(usize, 256), res.matched);
        try testing.expectEqual(@as(usize, 256), cache2.step);
        try testing.expectEqual(@as(f32, 100.0), pcSsmVal(ssm2[0].conv_state, 0, s));
    }
}

test "HotPrefixCache: under SSD-first a decline rides pending_disk, never a second synchronous spill" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 16 * 1024);
    hc.ssd_first = true;
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-spill-ssdfirst", 0, 128);
    defer hc.deinit();
    const cap_before = hc.disk.?.max_flush_bytes;

    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try testFillCache(&cache, s, 2, 600);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    const st = try hc.commit(&cache, &tokens, false);
    try testing.expect(st == .declined);
    // The capture at commit time is the one record; nothing lands until the flush.
    try testing.expect(hc.pending_disk != null);
    try testing.expectEqual(@as(usize, 0), hc.disk.?.entryCount());
    try testing.expectEqual(cap_before, hc.disk.?.max_flush_bytes);
    hc.flushPendingDisk(s);
    try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
}

test "HotPrefixCache: with an SSD tier the RAM entry keeps only its newest two checkpoints" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tokens: [900]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var srcs: [8][3]SSMCacheEntry = undefined;
    for (&srcs, 0..) |*e, i| {
        const f: f64 = @floatFromInt(i + 1);
        e.* = pcBuildHybrid(s, 100.0 * f, 500.0 * f);
    }
    defer for (&srcs) |*e| pcFreeHybrid(e);

    for ([_]bool{ true, false }) |with_disk| {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        defer hc.deinit();
        if (with_disk) {
            hc.ssd_first = true;
            hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-ram-keep", 0, 128);
            hc.disk.?.ssm_max_per_entry = kv_disk_cache.SSM_DISK_MAX_PER_ENTRY;
        }

        var c1 = try KVCache.init(testing.allocator, 3);
        defer c1.deinit();
        try testFillCache(&c1, s, 3, 700);
        const cps1 = try testing.allocator.alloc(SSMCheckpoint, 5);
        for (cps1, 0..) |*c, i| c.* = try transformer_mod.captureSsmCheckpoint(testing.allocator, &srcs[i], 512 + 32 * i, s);
        _ = try hc.commitWithSsm(&c1, tokens[0..700], false, cps1, null, null);
        const kept1 = hc.entries.items[0].ssm_checkpoints.?;
        // Without a tier there is nowhere else to restore from: today's behaviour.
        try testing.expectEqual(@as(usize, if (with_disk) 2 else 5), kept1.len);
        try testing.expectEqual(@as(usize, 640), kept1[kept1.len - 1].pos);
        try testing.expectEqual(@as(usize, if (with_disk) 608 else 512), kept1[0].pos);

        // The next turn extends the entry and brings three more: still the newest two.
        var c2 = try KVCache.init(testing.allocator, 3);
        defer c2.deinit();
        try testFillCache(&c2, s, 3, 900);
        const cps2 = try testing.allocator.alloc(SSMCheckpoint, 3);
        for (cps2, 0..) |*c, i| c.* = try transformer_mod.captureSsmCheckpoint(testing.allocator, &srcs[5 + i], 704 + 32 * i, s);
        _ = try hc.commitWithSsm(&c2, tokens[0..900], false, cps2, null, null);
        try testing.expectEqual(@as(usize, 1), hc.entries.items.len);
        const kept2 = hc.entries.items[0].ssm_checkpoints.?;
        try testing.expectEqual(@as(usize, if (with_disk) 2 else 8), kept2.len);
        try testing.expectEqual(@as(usize, 768), kept2[kept2.len - 1].pos);
        var bytes: u64 = 0;
        for (kept2) |*cp| bytes += ssmCheckpointBytes(cp);
        try testing.expectEqual(bytes, hc.entries.items[0].ssm_bytes);

        if (with_disk) {
            // Everything RAM let go is on the SSD record, to restore from there.
            hc.flushPendingDisk(s);
            try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
            try testing.expectEqual(@as(usize, 8), hc.disk.?.entries.items[0].ssm_positions.len);
        }
    }
}



fn pcHeld(cps: []const SSMCheckpoint) usize {
    var n: usize = 0;
    for (cps) |*cp| n += @intFromBool(!cp.isReleased());
    return n;
}

test "Qwen SSD-only: a prefill holds at most two checkpoint states, lists every position, and any of them restores from SSD exactly" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    var tokens: [1500]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try testFillCache(&cache, s, 3, 1500);
    var srcs: [8][3]SSMCacheEntry = undefined;
    for (&srcs, 0..) |*e, i| {
        const f: f64 = @floatFromInt(i + 1);
        e.* = pcBuildHybrid(s, 100.0 * f, 500.0 * f);
    }
    defer for (&srcs) |*e| pcFreeHybrid(e);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 0, 0);
    hc.ssd_first = true;
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-qwen-wt", 0, 128);
    defer hc.deinit();
    hc.disk.?.ssd_first = true;
    hc.disk.?.ssm_max_per_entry = kv_disk_cache.SSM_DISK_MAX_PER_ENTRY;
    hc.disk.?.armTestSpace(256 << 30, 1 << 40);
    hc.disk.?.enableBackgroundWriter();

    var cps = std.ArrayList(SSMCheckpoint).empty;
    errdefer {
        for (cps.items) |*cp| cp.deinit(testing.allocator);
        cps.deinit(testing.allocator);
    }
    var peak: usize = 0;
    // The prefill loop: a checkpoint after each chunk, then the write-through.
    for (0..7) |i| {
        const pos = 512 + 128 * i;
        try cps.append(testing.allocator, try transformer_mod.captureSsmCheckpoint(testing.allocator, &srcs[i], pos, s));
        peak = @max(peak, pcHeld(cps.items));
        hc.writeThroughHybrid(cache.entries, cache.config, tokens[0..pos], false, .{}, cps.items, std.math.maxInt(u64), s);
        try testing.expectEqual(@as(usize, 1), pcHeld(cps.items));
    }
    // The prompt-end snapshot is taken after the loop, past the hook.
    try cps.append(testing.allocator, try transformer_mod.captureSsmCheckpoint(testing.allocator, &srcs[7], 1400, s));
    try testing.expectEqual(@as(usize, 2), pcHeld(cps.items));
    try testing.expect(peak <= 2);
    try testing.expectEqual(@as(usize, 8), cps.items.len);

    // The commit lists every position and the SSD record writes only the states still held.
    const owned = try cps.toOwnedSlice(testing.allocator);
    try testing.expectEqual(CommitStatus{ .disk_only = 1500 }, try hc.commitWithSsm(&cache, &tokens, false, owned, null, null));
    hc.flushPendingDisk(s);
    hc.disk.?.drainWriter();
    try testing.expectEqual(@as(u64, 0), hc.disk.?.writeErrors());
    try testing.expectEqual(@as(usize, 0), hc.entries.items.len);
    try testing.expectEqualSlices(u32, &.{ 512, 640, 768, 896, 1024, 1152, 1280, 1400 }, hc.disk.?.entries.items[0].ssm_positions);

    // Released and held alike: each position restores its own state from the file.
    for ([_]struct { pos: u32, conv: f32, ssm: f32 }{
        .{ .pos = 512, .conv = 100, .ssm = 500 },
        .{ .pos = 896, .conv = 400, .ssm = 2000 },
        .{ .pos = 1280, .conv = 700, .ssm = 3500 },
        .{ .pos = 1400, .conv = 800, .ssm = 4000 },
    }) |w| {
        var cache2 = try KVCache.init(testing.allocator, 3);
        defer cache2.deinit();
        var dst = pcEmptySsm();
        defer pcFreeHybrid(&dst);
        try testing.expectEqual(w.pos, try hc.disk.?.restoreIntoHybrid(&cache2, &dst, 0, w.pos, s));
        try testing.expectEqual(w.conv, pcSsmVal(dst[0].conv_state, 0, s));
        try testing.expectEqual(w.ssm, pcSsmVal(dst[0].ssm_state, 0, s));
    }
}

test "HotPrefixCache: a byte-budget trim never lands on a released checkpoint, which this entry cannot restore from" {
    const cps = [_]SSMCheckpoint{
        .{ .pos = 1024, .layers = &.{}, .released_bytes = 100 },
        .{ .pos = 2048, .layers = &.{} },
    };
    const positions = [_]usize{ 1024, 2048 };
    const bytes = [_]u64{ 0, 100 };
    // 1024 is the highest position under the limit, and a placeholder: the entry declines to trim there.
    try testing.expectEqual(@as(?usize, null), HotPrefixCache.trimLenForBudgetPure(10_000_000, 1500, 1000, &positions, &bytes, .min_span_recency, &cps));
    const held = [_]SSMCheckpoint{ .{ .pos = 1024, .layers = &.{} }, .{ .pos = 2048, .layers = &.{} } };
    try testing.expectEqual(@as(?usize, 1024), HotPrefixCache.trimLenForBudgetPure(10_000_000, 1500, 1000, &positions, &bytes, .min_span_recency, &held));
}

test "HotPrefixCache: a budget-declined candidate spills to the SSD tier" {
    // Live 2026-09-07: an oversized decline discarded the cancelled
    // prefill's work entirely — under a 1 GB hot budget a 122k session's
    // 76-82k-token forwarded prefixes were thrown away on every retry
    // while a 40 GB SSD tier sat idle. The decline must OFFER the
    // candidate to the disk tier instead of discarding it.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const base = buf[0..root_len];

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // A RAM budget too small for even the commit floor forces the decline.
    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 16 * 1024);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-spill", 0, 128);
        defer hc.deinit();

        var cache = try KVCache.init(testing.allocator, 2);
        defer cache.deinit();
        try testFillCache(&cache, s, 2, 600);
        const st = try hc.commit(&cache, &tokens, false);
        try testing.expect(st == .declined);
        try testing.expectEqual(@as(usize, 0), hc.entryCount()); // RAM keeps nothing
        // The candidate's work must live on the SSD tier.
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    }

    // The next request (fresh RAM, generous budget) restores from SSD
    // instead of cold-prefilling the whole prefix.
    {
        var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-spill", 0, 128);
        defer hc2.deinit();
        try testing.expectEqual(@as(usize, 0), hc2.entryCount());

        var cache2 = try KVCache.init(testing.allocator, 2);
        defer cache2.deinit();
        var moe: usize = 0;
        const res = try hc2.lookupAndRestore(&cache2, &moe, null, s, &tokens, false, 0, null, null);
        try testing.expect(res.full_match);
        try testing.expectEqual(@as(usize, 599), res.matched);
    }
}

test "HotPrefixCache: with RAM retention on, the SSD tier gets the whole entry the RAM budget trims, after the response" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const base = buf[0..root_len];

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 40 * 1024);
        defer hc.deinit();
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-ram-trim", 0, 128);
        hc.disk.?.enableBackgroundWriter();
        hc.disk.?.max_flush_bytes = 1; // one chunk per piece

        var cache = try KVCache.init(testing.allocator, 2);
        defer cache.deinit();
        try testFillCache(&cache, s, 2, 600);
        const st = try hc.commit(&cache, &tokens, false);
        try testing.expect(st == .ok and st.ok < 600);
        // Nothing is written on the response path.
        try testing.expectEqual(@as(usize, 0), hc.disk.?.entryCount());
        hc.flushPendingDisk(s);
        hc.disk.?.drainWriter();
        try testing.expect(!hc.disk_dirty);
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
        try testing.expectEqual(@as(u32, 600), hc.disk.?.entries.items[0].kv_len);
    }
    // A fresh process restores the whole prompt from the SSD tier.
    var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc2.deinit();
    hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-ram-trim", 0, 128);
    var cache2 = try KVCache.init(testing.allocator, 2);
    defer cache2.deinit();
    var moe: usize = 0;
    const res = try hc2.lookupAndRestore(&cache2, &moe, null, s, &tokens, false, 0, null, null);
    try testing.expect(res.full_match);
    try testing.expectEqual(@as(usize, 599), res.matched);
}

test "HotPrefixCache: a commit that checked-out residents decline still reaches the SSD tier" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    const big: u64 = 1 << 30;
    // The count cap, then the byte cap, held by an entry a live slot checked out.
    const Cap = struct { entries: u32, bytes: u64, fp: []const u8 };
    for ([_]Cap{ .{ .entries = 1, .bytes = 0, .fp = "fp-held-count" }, .{ .entries = 8, .bytes = big, .fp = "fp-held-bytes" } }) |cap| {
        var hc = HotPrefixCache.initWithMem(testing.allocator, cap.entries, cap.bytes);
        defer hc.deinit();
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, cap.fp, 0, 128);
        hc.disk.?.enableBackgroundWriter();
        try pcAppendCheckedOut(&hc, &.{ 1, 2 }, cap.bytes, 1);
        var cache = try KVCache.init(testing.allocator, 2);
        defer cache.deinit();
        try testFillCache(&cache, s, 2, 600);
        try testing.expect(try hc.commit(&cache, &tokens, false) == .declined);
        hc.flushPendingDisk(s);
        hc.disk.?.drainWriter();
        try testing.expect(hc.pending_disk == null);
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
        try testing.expectEqual(@as(u32, 600), hc.disk.?.entries.items[0].kv_len);
    }
}

test "HotPrefixCache: a fork's SSD entry keeps the checkpoints its donor holds in RAM" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];

    var prompt: [600]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(i + 1);
    const a_tokens = prompt ++ [_]u32{ 200, 201 };
    const b_tokens = prompt ++ [_]u32{ 210, 211 };
    var srcs: [4][3]SSMCacheEntry = undefined;
    for (&srcs, 0..) |*e, i| {
        const f: f64 = @floatFromInt(i + 1);
        e.* = pcBuildHybrid(s, 100.0 * f, 500.0 * f);
    }
    defer for (&srcs) |*e| pcFreeHybrid(e);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 2, 0);
    hc.ssm_checkpoint_max = 8;
    defer hc.deinit();
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-fork-cps", 0, 128);
    hc.disk.?.enableBackgroundWriter();

    // A: checkpoints at 256 and 512 inside the shared prompt, 601 inside its own reply.
    var a_cache = try KVCache.init(testing.allocator, 3);
    defer a_cache.deinit();
    try testFillCache(&a_cache, s, 3, a_tokens.len);
    const a_cps = try testing.allocator.alloc(SSMCheckpoint, 3);
    a_cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &srcs[0], 256, s);
    a_cps[1] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &srcs[1], 512, s);
    a_cps[2] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &srcs[2], 601, s);
    _ = try hc.commitWithState(&a_cache, &a_tokens, false, 0, a_cps, null, null);
    hc.flushPendingDisk(s);

    // B restored A's prompt and checkpointed only its own prompt end.
    var b_cache = try KVCache.init(testing.allocator, 3);
    defer b_cache.deinit();
    try testFillCache(&b_cache, s, 3, b_tokens.len);
    const b_cps = try testing.allocator.alloc(SSMCheckpoint, 1);
    b_cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &srcs[3], 600, s);
    _ = try hc.commitWithState(&b_cache, &b_tokens, false, 0, b_cps, null, null);
    hc.flushPendingDisk(s);
    hc.disk.?.drainWriter();

    const d = &hc.disk.?;
    const b_disk = for (d.entries.items) |*e| {
        if (e.tokens.len > 600 and e.tokens[600] == 210) break e;
    } else return error.TestExpectedForkEntry;
    // A's RAM entry kept its newest two (512, 601): 256 stays only in A's own SSD entry.
    try testing.expectEqualSlices(u32, &.{ 512, 600 }, b_disk.ssm_positions);
}

test "a fork's SSD entry keeps its donor's ring checkpoint, so a restart restores there with the donor gone" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    const window: u32 = 8;
    const n_layers: u32 = 4;
    const prompt: u32 = 700;
    const reply: u32 = 600;
    const tail: u32 = 100;
    const reminder: u32 = 330;
    const fork_prompt = prompt + tail + reminder;

    var toks: [fork_prompt + reply]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
    var donor_toks: [prompt + reply]u32 = undefined;
    @memcpy(donor_toks[0..prompt], toks[0..prompt]);
    for (donor_toks[prompt..], 0..) |*t, i| t.* = @intCast(700_000 + i);
    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 2, 0);
        defer hc.deinit();
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-ring-fork-disk", 0, 128);
        hc.disk.?.enableBackgroundWriter();
        var donor = try KVCache.init(testing.allocator, n_layers);
        defer donor.deinit();
        const donor_cp = try ringTurn(&donor, s, n_layers, window, prompt, reply);
        _ = try hc.commitWithRing(&donor, &donor_toks, false, 0, 0, null, null, null, null, prompt, .{ .prompt_end = donor_cp });
        hc.flushPendingDisk(s);

        var fork = try KVCache.init(testing.allocator, n_layers);
        defer fork.deinit();
        fork.setSwaRing(window);
        var moe_off: usize = 0;
        const fork_hit = try hc.lookupAndRestore(&fork, &moe_off, null, s, toks[0..fork_prompt], false, 0, null, null);
        try testing.expectEqual(@as(usize, prompt), fork_hit.matched);
        try ringFill(&fork, s, n_layers, window, prompt, fork_prompt, 64);
        var fork_cp = try fork.ringCheckpoint(fork_prompt, s);
        errdefer if (fork_cp) |*c| c.deinit();
        try ringFill(&fork, s, n_layers, window, fork_prompt, fork_prompt + reply, 16);
        const moved_cp = fork_cp;
        fork_cp = null;
        _ = try hc.commitWithRing(&fork, &toks, false, 0, 0, null, null, null, null, fork_prompt, .{ .prompt_end = moved_cp });
        hc.flushPendingDisk(s);
        hc.disk.?.drainWriter();
        try testing.expectEqual(@as(usize, 2), hc.disk.?.entryCount());
    }
    // The donor's SSD entry goes; the fork's linked chunks outlive it.
    try tmp.dir.deleteTree(io, "fp-ring-fork-disk/e1");

    var hc2 = HotPrefixCache.initWithMem(testing.allocator, 2, 0);
    defer hc2.deinit();
    hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-ring-fork-disk", 0, 128);
    try testing.expectEqual(@as(usize, 1), hc2.disk.?.entryCount());
    var next: [prompt + tail + 40]u32 = undefined;
    @memcpy(next[0 .. prompt + tail], toks[0 .. prompt + tail]);
    for (next[prompt + tail ..], 0..) |*t, i| t.* = @intCast(900_000 + i);
    var restored = try KVCache.init(testing.allocator, n_layers);
    defer restored.deinit();
    restored.setSwaRing(window);
    var moe_off: usize = 0;
    const hit = try hc2.lookupAndRestore(&restored, &moe_off, null, s, &next, false, 0, null, null);
    try testing.expectEqual(@as(usize, prompt), hit.matched);
    try expectRingContinuesCold(&restored, s, n_layers, window, prompt, next.len);
}

/// A MiMo-shaped turn whose slot took a ring checkpoint at each of `at` (ascending, the last the
/// prompt end, the rest marks), then `reply` rows.
fn ringTurnAt(live: *KVCache, s: mlx.mlx_stream, n_layers: u32, window: u32, at: []const u32, reply: u32) !SlotRingCps {
    live.setSwaRing(window);
    var cps: SlotRingCps = .{};
    errdefer cps.deinit();
    var from: u32 = 0;
    for (at, 0..) |p, i| {
        try ringFill(live, s, n_layers, window, from, p, 64);
        from = p;
        const cp = (try live.ringCheckpoint(p, s)).?;
        if (i + 1 == at.len) {
            cps.prompt_end = cp;
        } else {
            cps.marks[cps.n_marks] = cp;
            cps.n_marks += 1;
        }
    }
    try ringFill(live, s, n_layers, window, from, from + reply, 16);
    return cps;
}

fn ringCpSteps(hc: *const HotPrefixCache, entry: usize, out: *[RING_CHECKPOINT_MAX]usize) []const usize {
    const cps = hc.entries.items[entry].ring_cps orelse return out[0..0];
    for (cps, 0..) |cp, i| out[i] = cp.step;
    return out[0..cps.len];
}

test "RAM and SSD: a flushed ringed entry keeps its newest two ring checkpoints, and an older point restores its exact rows from SSD" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    const window: u32 = 8;
    const n_layers: u32 = 4;
    const reply: u32 = 600;

    var toks: [700 + reply]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-ring-keep", 0, 128);
    hc.disk.?.enableBackgroundWriter();
    var live = try KVCache.init(testing.allocator, n_layers);
    defer live.deinit();
    const cps = try ringTurnAt(&live, s, n_layers, window, &.{ 300, 450, 600, 700 }, reply);
    _ = try hc.commitWithRing(&live, &toks, false, 0, 0, null, null, null, null, 700, cps);

    var steps: [RING_CHECKPOINT_MAX]usize = undefined;
    try testing.expectEqualSlices(usize, &.{ 300, 450, 600, 700 }, ringCpSteps(&hc, 0, &steps));
    const before = hc.current_kv_bytes;
    hc.flushPendingDisk(s);
    hc.disk.?.drainWriter();
    try testing.expectEqual(@as(u64, 0), hc.disk.?.writeErrors());
    try testing.expectEqualSlices(usize, &.{ 600, 700 }, ringCpSteps(&hc, 0, &steps));

    // The bill follows the storage: two checkpoints' bytes left it, and the cache total agrees.
    var cp_bytes: u64 = 0;
    for (live.entries) |*e| {
        if (e.ringed) cp_bytes += (window + model_mod.ModelConfig.SWA_RING_CHECKPOINT_BACKOFF) * (ringRowBytes(e.keys) + ringRowBytes(e.values));
    }
    try testing.expectEqual(before - 2 * cp_bytes, hc.current_kv_bytes);
    try testing.expectEqual(hc.entries.items[0].kv_bytes, hc.current_kv_bytes);

    // Between two points no RAM checkpoint reaches: the SSD tier's file at 450 serves it.
    var next: [520]u32 = undefined;
    @memcpy(next[0..460], toks[0..460]);
    for (next[460..], 0..) |*t, i| t.* = @intCast(900_000 + i);
    var restored = try KVCache.init(testing.allocator, n_layers);
    defer restored.deinit();
    restored.setSwaRing(window);
    var moe_off: usize = 0;
    const hit = try hc.lookupAndRestore(&restored, &moe_off, null, s, &next, false, 0, null, null);
    try testing.expectEqual(@as(usize, 450), hit.matched);
    try expectRingContinuesCold(&restored, s, n_layers, window, 450, next.len);

    // A newest point still restores from RAM.
    var newest: [660]u32 = undefined;
    @memcpy(newest[0..610], toks[0..610]);
    for (newest[610..], 0..) |*t, i| t.* = @intCast(800_000 + i);
    var from_ram = try KVCache.init(testing.allocator, n_layers);
    defer from_ram.deinit();
    from_ram.setSwaRing(window);
    const ram_hit = try hc.lookupAndRestore(&from_ram, &moe_off, null, s, &newest, false, 0, null, null);
    try testing.expectEqual(@as(usize, 600), ram_hit.matched);
    try expectRingContinuesCold(&from_ram, s, n_layers, window, 600, newest.len);
}

test "RAM tier without an SSD tier keeps every ring checkpoint" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const window: u32 = 8;
    const n_layers: u32 = 4;
    var toks: [1300]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    var live = try KVCache.init(testing.allocator, n_layers);
    defer live.deinit();
    const cps = try ringTurnAt(&live, s, n_layers, window, &.{ 300, 450, 600, 700 }, 600);
    _ = try hc.commitWithRing(&live, &toks, false, 0, 0, null, null, null, null, 700, cps);
    hc.flushPendingDisk(s);
    var steps: [RING_CHECKPOINT_MAX]usize = undefined;
    try testing.expectEqualSlices(usize, &.{ 300, 450, 600, 700 }, ringCpSteps(&hc, 0, &steps));
}

test "RAM and SSD: a ring checkpoint whose write failed stays in RAM" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    const window: u32 = 8;
    const n_layers: u32 = 4;
    var toks: [1300]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-ring-fail", 0, 128);
    hc.disk.?.enableBackgroundWriter();
    hc.disk.?.writer.?.injectFailure("r0000450", .submit);
    var live = try KVCache.init(testing.allocator, n_layers);
    defer live.deinit();
    const cps = try ringTurnAt(&live, s, n_layers, window, &.{ 300, 450, 600, 700 }, 600);
    _ = try hc.commitWithRing(&live, &toks, false, 0, 0, null, null, null, null, 700, cps);
    hc.flushPendingDisk(s);
    hc.disk.?.drainWriter();
    // The entry the failure invalidated holds no restore point, so RAM gives none up.
    var steps: [RING_CHECKPOINT_MAX]usize = undefined;
    try testing.expectEqualSlices(usize, &.{ 300, 450, 600, 700 }, ringCpSteps(&hc, 0, &steps));
}

test "a ring file past the rows its entry persisted is not a held restore point" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    const window: u32 = 8;
    const n_layers: u32 = 4;
    var toks: [1300]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-ring-kvlen", 0, 128);
    hc.disk.?.enableBackgroundWriter();
    var live = try KVCache.init(testing.allocator, n_layers);
    defer live.deinit();
    const cps = try ringTurnAt(&live, s, n_layers, window, &.{ 300, 450, 600, 700 }, 600);
    _ = try hc.commitWithRing(&live, &toks, false, 0, 0, null, null, null, null, 700, cps);
    hc.flushPendingDisk(s);
    hc.disk.?.drainWriter();

    // A flush the byte cap stopped leaves ring files above the rows the entry holds.
    hc.disk.?.entries.items[0].kv_len = 500;
    var held: [2]bool = undefined;
    hc.disk.?.ringPointsHeld(&toks, false, live.config, &.{ 450, 600 }, &held, .{});
    try testing.expectEqualSlices(bool, &.{ true, false }, &held);
}

test "RAM and SSD: a fork inherits the donor's newest checkpoints, and the donor's older points stay on its SSD entry" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    const window: u32 = 8;
    const n_layers: u32 = 4;
    const prompt: u32 = 700;
    const reply: u32 = 600;
    const fork_prompt: u32 = prompt + 100 + 330;

    var toks: [fork_prompt + reply]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
    var donor_toks: [prompt + reply]u32 = undefined;
    @memcpy(donor_toks[0..prompt], toks[0..prompt]);
    for (donor_toks[prompt..], 0..) |*t, i| t.* = @intCast(700_000 + i);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 2, 0);
    defer hc.deinit();
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-ring-donor", 0, 128);
    hc.disk.?.enableBackgroundWriter();
    var donor = try KVCache.init(testing.allocator, n_layers);
    defer donor.deinit();
    const donor_cps = try ringTurnAt(&donor, s, n_layers, window, &.{ 300, 450, 600, prompt }, reply);
    _ = try hc.commitWithRing(&donor, &donor_toks, false, 0, 0, null, null, null, null, prompt, donor_cps);
    hc.flushPendingDisk(s);
    var steps: [RING_CHECKPOINT_MAX]usize = undefined;
    try testing.expectEqualSlices(usize, &.{ 600, prompt }, ringCpSteps(&hc, 0, &steps));

    // The side request restores off the donor and appends its own tail.
    var fork = try KVCache.init(testing.allocator, n_layers);
    defer fork.deinit();
    fork.setSwaRing(window);
    var moe_off: usize = 0;
    const fork_hit = try hc.lookupAndRestore(&fork, &moe_off, null, s, toks[0..fork_prompt], false, 0, null, null);
    try testing.expectEqual(@as(usize, prompt), fork_hit.matched);
    try ringFill(&fork, s, n_layers, window, prompt, fork_prompt, 64);
    var fork_cp = try fork.ringCheckpoint(fork_prompt, s);
    errdefer if (fork_cp) |*c| c.deinit();
    try ringFill(&fork, s, n_layers, window, fork_prompt, fork_prompt + reply, 16);
    const moved_cp = fork_cp;
    fork_cp = null;
    _ = try hc.commitWithRing(&fork, &toks, false, 0, 0, null, null, null, null, fork_prompt, .{ .prompt_end = moved_cp });
    hc.flushPendingDisk(s);
    hc.disk.?.drainWriter();
    const fork_idx: usize = for (hc.entries.items, 0..) |*e, i| {
        if (e.tokens.len == toks.len) break i;
    } else return error.TestUnexpectedResult;
    // It inherited the donor's 600 and prompt end, then released the older of its three.
    try testing.expectEqualSlices(usize, &.{ prompt, fork_prompt }, ringCpSteps(&hc, fork_idx, &steps));

    // An unrelated request under the count cap evicts the donor from RAM.
    var other = try KVCache.init(testing.allocator, n_layers);
    defer other.deinit();
    other.setSwaRing(window);
    try ringFill(&other, s, n_layers, window, 0, 300, 64);
    var other_toks: [300]u32 = undefined;
    for (&other_toks, 0..) |*t, i| t.* = @intCast(500_000 + i);
    _ = try hc.commitWithState(&other, &other_toks, false, 0, null, null, null);
    for (hc.entries.items) |*e| try testing.expect(e.tokens.len != donor_toks.len);

    // A turn diverging at the inherited 600 restores there from the fork's SSD entry.
    var at_600: [660]u32 = undefined;
    @memcpy(at_600[0..610], toks[0..610]);
    for (at_600[610..], 0..) |*t, i| t.* = @intCast(900_000 + i);
    var a = try KVCache.init(testing.allocator, n_layers);
    defer a.deinit();
    a.setSwaRing(window);
    const hit_a = try hc.lookupAndRestore(&a, &moe_off, null, s, &at_600, false, 0, null, null);
    try testing.expectEqual(@as(usize, 600), hit_a.matched);
    try expectRingContinuesCold(&a, s, n_layers, window, 600, at_600.len);

    // One diverging at the donor's released 450 restores from the donor's SSD entry.
    var at_450: [520]u32 = undefined;
    @memcpy(at_450[0..460], toks[0..460]);
    for (at_450[460..], 0..) |*t, i| t.* = @intCast(800_000 + i);
    var b = try KVCache.init(testing.allocator, n_layers);
    defer b.deinit();
    b.setSwaRing(window);
    const hit_b = try hc.lookupAndRestore(&b, &moe_off, null, s, &at_450, false, 0, null, null);
    try testing.expectEqual(@as(usize, 450), hit_b.matched);
    try expectRingContinuesCold(&b, s, n_layers, window, 450, at_450.len);
}

test "HotPrefixCache: a decline-spill is not bounded by the per-flush byte cap" {
    // The per-flush cap bounds the stall a LIVE next request pays after a
    // response. A decline-spill runs on a request whose client is already
    // gone — capping it (live 2026-09-07: 512 MB ≈ 13k tokens banked per
    // retry) stranded most of each retry's work while ~35k tokens were
    // recomputed every time. The spill owes no latency; the tier's byte
    // budget + LRU is the real bound.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const base = buf[0..root_len];

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 16 * 1024);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-spill-cap", 0, 128);
        defer hc.deinit();
        // A cap of two KV chunks — the exact starvation shape, scaled down.
        hc.disk.?.max_flush_bytes = 2 * 128 * 2 * 2 * 8 * 4;

        var cache = try KVCache.init(testing.allocator, 2);
        defer cache.deinit();
        try testFillCache(&cache, s, 2, 600);
        const st = try hc.commit(&cache, &tokens, false);
        try testing.expect(st == .declined);
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    }

    // The WHOLE candidate must be restorable, not just the cap's worth.
    {
        var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-spill-cap", 0, 128);
        defer hc2.deinit();
        var cache2 = try KVCache.init(testing.allocator, 2);
        defer cache2.deinit();
        var moe: usize = 0;
        const res = try hc2.lookupAndRestore(&cache2, &moe, null, s, &tokens, false, 0, null, null);
        try testing.expect(res.full_match);
        try testing.expectEqual(@as(usize, 599), res.matched);
    }
}

test "HotPrefixCache: hybrid disk restore ranks entries by restorable checkpoint, not raw length" {
    // The RAM tier learned this the hard way (#312): a longer raw match
    // whose checkpoints sit past the divergence restores nothing. The disk
    // tier's hybrid arm used bestMatch (ranked by usable length) and then
    // took that ONE entry's checkpoint — an entry with a high usable length
    // but low checkpoints shadowed a shorter entry with a higher
    // restorable position (live 2026-09-07: the growing conversation entry
    // would have shadowed the old entry's cp@51200 with cp@49152).
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const base = buf[0..root_len];

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // Entry A: shares 550 tokens with the prompt (longer) but its only
    // checkpoint is at 256. Diverges from the prompt at 550.
    var a_tokens: [600]u32 = undefined;
    for (&a_tokens, 0..) |*t, i| t.* = if (i < 550) @intCast(i + 7) else @intCast(i + 700);
    // Entry B: shares only 512 tokens (shorter) but carries a checkpoint
    // at 512. Diverges from the prompt at 512 — so neither entry is a
    // prefix of the other and the flush cannot merge them.
    var b_tokens: [520]u32 = undefined;
    for (&b_tokens, 0..) |*t, i| t.* = if (i < 512) @intCast(i + 7) else @intCast(i + 900);
    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-hyb-rank", 0, 128);
        defer hc.deinit();
        var cache = try KVCache.init(testing.allocator, 3);
        defer cache.deinit();
        try testFillCache(&cache, s, 3, 600);
        var src = pcBuildHybrid(s, 100.0, 500.0);
        defer pcFreeHybrid(&src);
        const cps_a = try testing.allocator.alloc(SSMCheckpoint, 1);
        cps_a[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &src, 256, s);
        _ = try hc.commitWithSsm(&cache, &a_tokens, false, cps_a, null, null);
        hc.flushPendingDisk(s);
        var cache_b = try KVCache.init(testing.allocator, 3);
        defer cache_b.deinit();
        try testFillCache(&cache_b, s, 3, 600);
        const cps_b = try testing.allocator.alloc(SSMCheckpoint, 1);
        cps_b[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &src, 512, s);
        _ = try hc.commitWithSsm(&cache_b, &b_tokens, false, cps_b, null, null);
        hc.flushPendingDisk(s);
    }

    // RAM empty; the hybrid lookup must restore from B's cp@512, not A's
    // cp@256 — even though A's raw usable length (600) beats B's (512).
    {
        var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-hyb-rank", 0, 128);
        defer hc2.deinit();
        try testing.expectEqual(@as(usize, 2), hc2.disk.?.entryCount());

        var cache2 = try KVCache.init(testing.allocator, 3);
        defer cache2.deinit();
        var ssm2 = pcEmptySsm();
        defer pcFreeHybrid(&ssm2);
        var moe_off: usize = 0;
        const res = try hc2.lookupAndRestore(&cache2, &moe_off, &ssm2, s, &tokens, false, 0, null, null);
        try testing.expect(!res.full_match);
        try testing.expectEqual(@as(usize, 512), res.matched);
        try testing.expectEqual(@as(usize, 512), cache2.step);
    }
}

test "HotPrefixCache: a hybrid SSD restore hands the slot rows it owns, a RAM share does not" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const base = buf[0..root_len];

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var entry_tokens: [520]u32 = undefined;
    for (&entry_tokens, 0..) |*t, i| t.* = if (i < 512) @intCast(i + 7) else @intCast(i + 900);
    var src = pcBuildHybrid(s, 100.0, 500.0);
    defer pcFreeHybrid(&src);
    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-hyb-own", 0, 128);
        defer hc.deinit();
        var cache = try KVCache.init(testing.allocator, 3);
        defer cache.deinit();
        try testFillCache(&cache, s, 3, 520);
        const cps = try testing.allocator.alloc(SSMCheckpoint, 1);
        cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &src, 512, s);
        _ = try hc.commitWithSsm(&cache, &entry_tokens, false, cps, null, null);
        hc.flushPendingDisk(s);

        var slot = try KVCache.init(testing.allocator, 3);
        defer slot.deinit();
        var ssm = pcEmptySsm();
        defer pcFreeHybrid(&ssm);
        var moe_off: usize = 0;
        const ram = try hc.lookupAndRestore(&slot, &moe_off, &ssm, s, &tokens, false, 0, null, null);
        try testing.expectEqual(@as(usize, 512), ram.matched);
        // The entry keeps the buffers: the first append copies them, so nothing is credited.
        try testing.expect(!ram.ownsRestoredRows());
    }

    var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-hyb-own", 0, 128);
    defer hc2.deinit();
    var slot2 = try KVCache.init(testing.allocator, 3);
    defer slot2.deinit();
    var ssm2 = pcEmptySsm();
    defer pcFreeHybrid(&ssm2);
    var moe_off2: usize = 0;
    const disk = try hc2.lookupAndRestore(&slot2, &moe_off2, &ssm2, s, &tokens, false, 0, null, null);
    try testing.expectEqual(@as(usize, 512), disk.matched);
    try testing.expect(disk.ownsRestoredRows());
}

test "HotPrefixCache: a hybrid disk restore adopts the spec sidecar of the entry it restored" {
    // The trunk verifies, so no wrong token ships — but an MTP history describing another
    // entry's tokens collapses acceptance.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const base = buf[0..root_len];

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    // A wins the raw match (550) but checkpoints at 256; B matches 512 and checkpoints there.
    var a_tokens: [600]u32 = undefined;
    for (&a_tokens, 0..) |*t, i| t.* = if (i < 550) @intCast(i + 7) else @intCast(i + 700);
    var b_tokens: [520]u32 = undefined;
    for (&b_tokens, 0..) |*t, i| t.* = if (i < 512) @intCast(i + 7) else @intCast(i + 900);
    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-hyb-spec", 0, 128);
        defer hc.deinit();
        var src = pcBuildHybrid(s, 100.0, 500.0);
        defer pcFreeHybrid(&src);

        var cache_a = try KVCache.init(testing.allocator, 3);
        defer cache_a.deinit();
        try testFillCache(&cache_a, s, 3, 600);
        var hist_a = try KVCache.init(testing.allocator, 1);
        defer hist_a.deinit();
        try testFillCache(&hist_a, s, 1, 600);
        const cps_a = try testing.allocator.alloc(SSMCheckpoint, 1);
        cps_a[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &src, 256, s);
        _ = try hc.commitWithSsm(&cache_a, &a_tokens, false, cps_a, null, .{ .cache = &hist_a, .base_pos = 100 });
        hc.flushPendingDisk(s);

        var cache_b = try KVCache.init(testing.allocator, 3);
        defer cache_b.deinit();
        try testFillCache(&cache_b, s, 3, 600);
        var hist_b = try KVCache.init(testing.allocator, 1);
        defer hist_b.deinit();
        try testFillCache(&hist_b, s, 1, 520);
        const cps_b = try testing.allocator.alloc(SSMCheckpoint, 1);
        cps_b[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &src, 512, s);
        _ = try hc.commitWithSsm(&cache_b, &b_tokens, false, cps_b, null, .{ .cache = &hist_b, .base_pos = 0 });
        hc.flushPendingDisk(s);
    }

    var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-hyb-spec", 0, 128);
    defer hc2.deinit();
    try testing.expectEqual(@as(usize, 2), hc2.disk.?.entryCount());
    var cache2 = try KVCache.init(testing.allocator, 3);
    defer cache2.deinit();
    var ssm2 = pcEmptySsm();
    defer pcFreeHybrid(&ssm2);
    var mtp_dst = try KVCache.init(testing.allocator, 1);
    defer mtp_dst.deinit();
    var moe_off: usize = 0;
    var mbase: usize = 99;
    const res = try hc2.lookupAndRestore(&cache2, &moe_off, &ssm2, s, &tokens, false, 0, null, .{ .cache = &mtp_dst, .base_pos = &mbase });
    try testing.expectEqual(@as(usize, 512), res.matched);
    // B's history, not A's (which would adopt at base 100 for 412 rows).
    try testing.expectEqual(@as(?usize, 0), res.mtp_base);
    try testing.expectEqual(@as(usize, 0), mbase);
    try testing.expectEqual(@as(usize, 512), mtp_dst.step);
}

test "HotPrefixCache: hybrid RAM match at least as good as disk skips the SSD read" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const base = buf[0..root_len];

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-hybskip", 0, 128);
    defer hc.deinit();

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try testFillCache(&cache, s, 3, 600);
    var src512 = pcBuildHybrid(s, 300.0, 700.0);
    defer pcFreeHybrid(&src512);
    const cps = try testing.allocator.alloc(SSMCheckpoint, 1);
    cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &src512, 512, s);
    _ = try hc.commitWithSsm(&cache, &tokens, false, cps, null, null);
    hc.flushPendingDisk(s);
    const disk_uses_before = hc.disk.?.counter;

    // Same prompt again: the RAM entry's checkpoint at 512 ties the disk's, so
    // the disk advantage gate fails and the SSD read is skipped (counter
    // unchanged) — the RAM path serves the restore.
    var cache2 = try KVCache.init(testing.allocator, 3);
    defer cache2.deinit();
    var ssm2 = pcEmptySsm();
    defer pcFreeHybrid(&ssm2);
    var moe_off: usize = 0;
    const res = try hc.lookupAndRestore(&cache2, &moe_off, &ssm2, s, &tokens, false, 0, null, null);
    try testing.expectEqual(@as(usize, 512), res.matched);
    try testing.expectEqual(disk_uses_before, hc.disk.?.counter);
    // RAM restore installed the SSM state just the same.
    try testing.expectEqual(@as(f32, 300.0), pcSsmVal(ssm2[0].conv_state, 0, s));
}

// Regression: an entry that is EXTENDED in place must not accumulate SSM
// checkpoints without bound, and bounding them must not collapse the survivors
// onto the end of the prompt. `generate.zig` caps what a single prefill
// captures, but the replace path merges the previous entry's checkpoints with
// this turn's, and nothing re-applied a cap to the merged list — so an agent
// conversation gained one checkpoint per turn forever. Observed on
// Qwen3.8-Flash-Next (36 GDN layers): 31237 MB of SSM state in ONE entry under
// `--ssm-checkpoint-max 8`, which starved the prompt-admission check.
//
// Capping oldest-first fixes the size but leaves every survivor near the end,
// and a request diverging earlier then pays a full cold prefill
// ("hybrid miss (no checkpoint <= 16382 of 178509)", 415 s). So the cap thins
// the interior and keeps a spread.
test "HotPrefixCache: replace path bounds SSM checkpoints and keeps them spread" {
    const s = mlx.gpuStream();

    var tokens: [900]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 3);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.ssm_checkpoint_max = 4;
    defer hc.deinit();

    var srcs: [8][3]SSMCacheEntry = undefined;
    for (&srcs, 0..) |*e, i| {
        const f: f64 = @floatFromInt(i + 1);
        e.* = pcBuildHybrid(s, 100.0 * f, 500.0 * f);
    }
    defer {
        for (&srcs) |*e| pcFreeHybrid(e);
    }

    // Turn 1: four checkpoints at 100..400 over a 450-token prefix.
    var c1 = try KVCache.init(testing.allocator, 3);
    defer c1.deinit();
    try testFillCache(&c1, s, 3, 450);
    const cps1 = try testing.allocator.alloc(SSMCheckpoint, 4);
    for (cps1, 0..) |*c, i| {
        c.* = try transformer_mod.captureSsmCheckpoint(testing.allocator, &srcs[i], (i + 1) * 100, s);
    }
    _ = try hc.commitWithSsm(&c1, tokens[0..450], false, cps1, null, null);
    try testing.expectEqual(@as(usize, 4), hc.entries.items[0].ssm_checkpoints.?.len);

    // Turn 2 extends that exact prefix and brings four more at 500..800.
    // Merged that is eight against a cap of four.
    var c2 = try KVCache.init(testing.allocator, 3);
    defer c2.deinit();
    try testFillCache(&c2, s, 3, 900);
    const cps2 = try testing.allocator.alloc(SSMCheckpoint, 4);
    for (cps2, 0..) |*c, i| {
        c.* = try transformer_mod.captureSsmCheckpoint(testing.allocator, &srcs[i + 4], (i + 5) * 100, s);
    }
    _ = try hc.commitWithSsm(&c2, tokens[0..900], false, cps2, null, null);

    // Extended, not appended: still one entry, and the cap holds.
    try testing.expectEqual(@as(usize, 1), hc.entries.items.len);
    const kept = hc.entries.items[0].ssm_checkpoints.?;
    try testing.expectEqual(@as(usize, 4), kept.len);

    // The first and the newest always survive, the interior is thinned to keep
    // coverage. Oldest-first would have left 500/600/700/800, and a request
    // matching at 150 would then have nothing at or below it to restore from.
    try testing.expectEqual(@as(usize, 100), kept[0].pos);
    try testing.expectEqual(@as(usize, 300), kept[1].pos);
    try testing.expectEqual(@as(usize, 500), kept[2].pos);
    try testing.expectEqual(@as(usize, 800), kept[3].pos);
}

test "HotPrefixCache: a revised budget evicts down to fit and a raised one keeps everything" {
    const s = mlx.gpuStream();
    var a = try KVCache.init(testing.allocator, 2);
    defer a.deinit();
    try testFillCache(&a, s, 2, 600);
    var b = try KVCache.init(testing.allocator, 2);
    defer b.deinit();
    try testFillCache(&b, s, 2, 600);
    var toks_a: [600]u32 = undefined;
    for (&toks_a, 0..) |*t2, i| t2.* = @intCast(i + 1);
    var toks_b: [600]u32 = undefined;
    for (&toks_b, 0..) |*t2, i| t2.* = @intCast(i + 1000);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    _ = try hc.commit(&a, &toks_a, false);
    _ = try hc.commit(&b, &toks_b, false);
    try testing.expectEqual(@as(usize, 2), hc.entryCount());
    const one = hc.entries.items[0].kv_bytes;

    // Shrink to one entry's worth: the LRU (A) goes, B stays whole.
    hc.setBudget(one);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try testing.expect(hc.current_kv_bytes <= hc.max_kv_bytes);
    try testing.expectEqual(@as(u32, 1000), hc.entries.items[0].tokens[0]);
    // Raise (0 = uncapped): nothing moves.
    hc.setBudget(0);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try testing.expectEqual(@as(u64, 0), hc.max_kv_bytes);
}

test "HotPrefixCache: byte budget rejects an oversized sole entry and preserves a smaller prefix" {
    const s = mlx.gpuStream();

    var a = try KVCache.init(testing.allocator, 2);
    defer a.deinit();
    try testFillCache(&a, s, 2, 8);
    // KV buffers grow in 256-token chunks, so the sizes must straddle a chunk
    // boundary for the two entries' snapshot bytes to differ.
    var b = try KVCache.init(testing.allocator, 2);
    defer b.deinit();
    try testFillCache(&b, s, 2, 600);

    var toks_a: [8]u32 = undefined;
    for (&toks_a, 0..) |*t2, i| t2.* = @intCast(i + 1);
    var toks_b: [600]u32 = undefined;
    for (&toks_b, 0..) |*t2, i| t2.* = @intCast(i + 100);

    var a_snap = try a.snapshot();
    defer a_snap.deinit();
    var b_snap = try b.snapshot();
    defer b_snap.deinit();
    const small = HotPrefixCache.snapshotBytes(&a_snap);
    const big = HotPrefixCache.snapshotBytes(&b_snap);
    try testing.expect(big > small);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, small);
    defer hc.deinit();

    _ = try hc.commit(&a, &toks_a, false);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try testing.expect(hc.current_kv_bytes <= hc.max_kv_bytes);

    // B extends A, so this exercises the replacement path. It is too large
    // for the cap: the byte budget still holds — but as a TRIM (#330), not a
    // decline. The retained prefix is longer than A's, shorter than B's.
    @memcpy(toks_b[0..toks_a.len], &toks_a);
    _ = try hc.commit(&b, &toks_b, false);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    const kept_len = hc.entries.items[0].tokens.len;
    try testing.expect(kept_len >= MIN_CANCELLED_COMMIT_TOKENS);
    try testing.expect(kept_len < toks_b.len);
    try testing.expect(hc.current_kv_bytes <= hc.max_kv_bytes);

    // An empty cache retains the same trimmed prefix from the oversized
    // candidate rather than staying empty (#330: the pre-fix decline held the
    // cap by holding zero bytes).
    hc.invalidateAll("test");
    _ = try hc.commit(&b, &toks_b, false);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try testing.expectEqual(kept_len, hc.entries.items[0].tokens.len);
    try testing.expect(hc.current_kv_bytes <= hc.max_kv_bytes);
}

// ── Issue #330: the oversized-entry decline is a cliff, not a cap ──

/// Test-local: dense per-token KV bytes of a snapshot (k + v across layers).
fn pcRowBytes(snap: *const transformer_mod.KVCacheSnapshot) u64 {
    var total: u64 = 0;
    for (snap.entries) |e| {
        if (!e.initialized) continue;
        const rows: u64 = @intCast(mlx.mlx_array_shape(e.keys)[2]);
        if (rows == 0) continue;
        const kb = @as(u64, mlx.mlx_array_size(e.keys)) * @as(u64, mlx.mlx_array_itemsize(e.keys));
        const vb = @as(u64, mlx.mlx_array_size(e.values)) * @as(u64, mlx.mlx_array_itemsize(e.values));
        total += (kb + vb) / rows;
    }
    return total;
}

/// Test-local: assert rows [0:rows] of `a` and `b` (axis 2) are identical.
fn pcExpectRowsEqual(s: mlx.mlx_stream, a: mlx.mlx_array, b: mlx.mlx_array, rows: usize) !void {
    var sliced_a = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sliced_a);
    var sliced_b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sliced_b);
    const sh_a = mlx.mlx_array_shape(a);
    const sh_b = mlx.mlx_array_shape(b);
    const start = [_]c_int{ 0, 0, 0, 0 };
    const strides = [_]c_int{ 1, 1, 1, 1 };
    const stop_a = [_]c_int{ sh_a[0], sh_a[1], @intCast(rows), sh_a[3] };
    const stop_b = [_]c_int{ sh_b[0], sh_b[1], @intCast(rows), sh_b[3] };
    try mlx.check(mlx.mlx_slice(&sliced_a, a, &start, 4, &stop_a, 4, &strides, 4, s));
    try mlx.check(mlx.mlx_slice(&sliced_b, b, &start, 4, &stop_b, 4, &strides, 4, s));
    var eq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eq);
    try mlx.check(mlx.mlx_equal(&eq, sliced_a, sliced_b, s));
    var all = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(all);
    try mlx.check(mlx.mlx_all(&all, eq, false, s));
    var ok: bool = false;
    try mlx.check(mlx.mlx_array_item_bool(&ok, all));
    try testing.expect(ok);
}

test "HotPrefixCache: oversized entry trims to the longest prefix that fits (#330)" {
    const s = mlx.gpuStream();

    var src = try KVCache.init(testing.allocator, 2);
    defer src.deinit();
    try testFillCache(&src, s, 2, 600);
    var toks: [600]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);

    var probe = try src.snapshot();
    defer probe.deinit();
    const row = pcRowBytes(&probe);
    // Room for exactly 400 tokens — over the 256-token commit floor, under
    // the 600-token candidate.
    const budget = row * 400;

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, budget);
    defer hc.deinit();
    _ = try hc.commit(&src, &toks, false);

    // The cliff: pre-fix this declines outright and the cache stays empty.
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try testing.expectEqual(@as(usize, 400), hc.entries.items[0].tokens.len);
    try testing.expect(hc.current_kv_bytes <= hc.max_kv_bytes);

    // The trimmed prefix restores: same prompt matches 400 tokens.
    var dst = try KVCache.init(testing.allocator, 2);
    defer dst.deinit();
    var moe_off: usize = 0;
    const res = try hc.lookupAndRestore(&dst, &moe_off, null, s, &toks, false, 0, null, null);
    try testing.expect(!res.full_match);
    try testing.expectEqual(@as(usize, 400), res.matched);
    try testing.expectEqual(@as(usize, 400), dst.step);
    for (dst.entries, src.entries) |*d, *e| {
        try testing.expectEqual(@as(usize, 400), d.offset);
        // Trimmed rows must be the SOURCE's rows — a slice-math bug here
        // serves a wrong prefix as a cache hit.
        try pcExpectRowsEqual(s, d.keys, e.keys, 400);
        try pcExpectRowsEqual(s, d.values, e.values, 400);
    }
}

test "HotPrefixCache: trimmed entry is one-shot — a covered re-commit keeps the resident copy (#330)" {
    const s = mlx.gpuStream();

    var src = try KVCache.init(testing.allocator, 2);
    defer src.deinit();
    try testFillCache(&src, s, 2, 600);
    var toks: [700]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);

    var probe = try src.snapshot();
    defer probe.deinit();
    const budget = pcRowBytes(&probe) * 400;

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, budget);
    defer hc.deinit();
    _ = try hc.commit(&src, toks[0..600], false);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try testing.expectEqual(@as(usize, 400), hc.entries.items[0].tokens.len);
    const resident_keys_ctx = hc.entries.items[0].snapshot.entries[0].keys.ctx;

    // Next turn: the conversation grew, the trim target did not. Re-copying
    // an identical prefix every turn would be a per-turn multi-GB memcpy.
    var src2 = try KVCache.init(testing.allocator, 2);
    defer src2.deinit();
    try testFillCache(&src2, s, 2, 700);
    _ = try hc.commit(&src2, &toks, false);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try testing.expectEqual(@as(usize, 400), hc.entries.items[0].tokens.len);
    try testing.expectEqual(resident_keys_ctx, hc.entries.items[0].snapshot.entries[0].keys.ctx);
}

test "HotPrefixCache: oversized hybrid entry trims to the highest checkpoint that fits (#330)" {
    const s = mlx.gpuStream();

    var toks: [900]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 3);

    var srcs: [8][3]SSMCacheEntry = undefined;
    for (&srcs, 0..) |*e, i| {
        const f: f64 = @floatFromInt(i + 1);
        e.* = pcBuildHybrid(s, 100.0 * f, 500.0 * f);
    }
    defer {
        for (&srcs) |*e| pcFreeHybrid(e);
    }

    var c1 = try KVCache.init(testing.allocator, 3);
    defer c1.deinit();
    try testFillCache(&c1, s, 3, 900);
    const cps = try testing.allocator.alloc(SSMCheckpoint, 8);
    for (cps, 0..) |*c, i| {
        c.* = try transformer_mod.captureSsmCheckpoint(testing.allocator, &srcs[i], (i + 1) * 100, s);
    }

    var probe = try c1.snapshot();
    defer probe.deinit();
    const row = pcRowBytes(&probe);
    var cps_at_or_below_500: u64 = 0;
    for (cps[0..5]) |*c| cps_at_or_below_500 += transformer_mod.ssmCheckpointBytes(c);
    // Exactly the cost of a 500-token prefix plus its five checkpoints: the
    // trim point must be a RESTORABLE position, so 500 is the answer even
    // though a few more raw tokens would fit.
    const budget = row * 500 + cps_at_or_below_500;

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, budget);
    hc.ssm_checkpoint_max = 8;
    defer hc.deinit();
    _ = try hc.commitWithSsm(&c1, &toks, false, cps, null, null);

    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    const e = &hc.entries.items[0];
    try testing.expectEqual(@as(usize, 500), e.tokens.len);
    try testing.expect(hc.current_kv_bytes <= hc.max_kv_bytes);
    // Checkpoints past the trim point are gone; the one AT it survives.
    const kept = e.ssm_checkpoints.?;
    try testing.expectEqual(@as(usize, 5), kept.len);
    try testing.expectEqual(@as(usize, 500), kept[kept.len - 1].pos);
}

test "HotPrefixCache: a trim priced after its checkpoint shed is retained, then shed" {
    // qwen4_exp prices a trim against the checkpoints that survive the shed; retaining the new
    // entry declined it on its unshed size, so a session past the budget was never cached.
    const s = mlx.gpuStream();
    var toks: [900]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 3);
    var srcs: [8][3]SSMCacheEntry = undefined;
    for (&srcs, 0..) |*e, i| {
        const f: f64 = @floatFromInt(i + 1);
        e.* = pcBuildHybrid(s, 100.0 * f, 500.0 * f);
    }
    defer {
        for (&srcs) |*e| pcFreeHybrid(e);
    }
    var c1 = try KVCache.init(testing.allocator, 3);
    defer c1.deinit();
    try testFillCache(&c1, s, 3, 900);
    const cps = try testing.allocator.alloc(SSMCheckpoint, 8);
    for (cps, 0..) |*c, i| {
        c.* = try transformer_mod.captureSsmCheckpoint(testing.allocator, &srcs[i], (i + 1) * 100, s);
    }
    var probe = try c1.snapshot();
    defer probe.deinit();
    const cp_bytes = transformer_mod.ssmCheckpointBytes(&cps[0]);
    // Room for a 500-token prefix and three of its five checkpoints.
    const budget = pcRowBytes(&probe) * 500 + 3 * cp_bytes;

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, budget);
    hc.ssm_checkpoint_max = 8;
    hc.cp_thin = .min_span_recency;
    defer hc.deinit();
    const status = try hc.commitWithSsm(&c1, &toks, false, cps, null, null);
    try testing.expectEqual(std.meta.Tag(CommitStatus).ok, std.meta.activeTag(status));
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    const e = &hc.entries.items[0];
    try testing.expect(e.tokens.len >= 500);
    try testing.expect(hc.current_kv_bytes <= hc.max_kv_bytes);
    const kept = e.ssm_checkpoints.?;
    try testing.expect(kept.len < 5);
    try testing.expectEqual(e.tokens.len, kept[kept.len - 1].pos);
}

/// A two-layer QSA hybrid: the layer counts differ from `pcBuildQsaHybrid`'s, which is how a
/// slice between two snaps of this list fails deterministically.
fn pcBuildQsaPair(s: mlx.mlx_stream, rows: c_int) [2]SSMCacheEntry {
    const aux_shape = [_]c_int{ 1, rows, 8 };
    const pooled_shape = [_]c_int{ 1, @divTrunc(rows, 4), 8 };
    return .{
        .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true, .aux_state = pcArange(s, &aux_shape, 0.0), .qsa_pooled = pcArange(s, &pooled_shape, 1000.0), .qsa_ratio = 4 },
        .{ .conv_state = pcArange(s, &conv_shape_pc, 100.0), .ssm_state = mlx.mlx_array_new(), .initialized = true },
    };
}

fn pcFreeSsmSlice(entries: []SSMCacheEntry) void {
    for (entries) |*x| {
        if (x.conv_state.ctx != null) _ = mlx.mlx_array_free(x.conv_state);
        if (x.ssm_state.ctx != null) _ = mlx.mlx_array_free(x.ssm_state);
        if (x.aux_state.ctx != null) _ = mlx.mlx_array_free(x.aux_state);
        if (x.qsa_pooled.ctx != null) _ = mlx.mlx_array_free(x.qsa_pooled);
    }
}

test "HotPrefixCache: a trim that cannot carry the QSA history over declines instead of committing" {
    // The bank lives only on the latest snap; a trim that drops it without slicing it onto the
    // last kept snap leaves an entry whose every restore is a silent miss.
    const s = mlx.gpuStream();

    var toks: [900]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 3);

    var wide = pcBuildQsaHybrid(s, 600, 100.0);
    defer pcFreeQsaHybrid(&wide);
    var narrow = pcBuildQsaPair(s, 600);
    defer pcFreeSsmSlice(&narrow);

    var c1 = try KVCache.init(testing.allocator, 3);
    defer c1.deinit();
    try testFillCache(&c1, s, 3, 900);
    const cps = try testing.allocator.alloc(SSMCheckpoint, 2);
    cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &wide, 300, s);
    cps[1] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &narrow, 600, s);
    try transformer_mod.attachQsaHistoryToLatest(cps, &narrow, s);
    try testing.expect(checkpointHasQsaPooled(&cps[1]));

    var probe = try c1.snapshot();
    defer probe.deinit();
    const budget = pcRowBytes(&probe) * 300 + transformer_mod.ssmCheckpointBytes(&cps[0]);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, budget);
    defer hc.deinit();
    const status = try hc.commitWithSsm(&c1, &toks, false, cps, null, null);
    try testing.expect(status == .declined);
    try testing.expectEqual(@as(usize, 0), hc.entryCount());
}

test "HotPrefixCache: oversized hybrid entry with no checkpoint under budget declines (#330)" {
    const s = mlx.gpuStream();

    var toks: [900]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 3);

    var hyb = pcBuildHybrid(s, 100.0, 500.0);
    defer pcFreeHybrid(&hyb);

    var c1 = try KVCache.init(testing.allocator, 3);
    defer c1.deinit();
    try testFillCache(&c1, s, 3, 900);
    const cps = try testing.allocator.alloc(SSMCheckpoint, 1);
    cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &hyb, 800, s);

    var probe = try c1.snapshot();
    defer probe.deinit();
    // Sole checkpoint sits at 800; a 100-token budget cannot retain a
    // restorable hybrid prefix, so the commit declines like before.
    const budget = pcRowBytes(&probe) * 100;

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, budget);
    defer hc.deinit();
    _ = try hc.commitWithSsm(&c1, &toks, false, cps, null, null);
    try testing.expectEqual(@as(usize, 0), hc.entryCount());
    try testing.expectEqual(@as(u64, 0), hc.current_kv_bytes);
}

/// qwen4-shaped hybrid: layer 0 is a QSA full-attention layer (no conv/ssm,
/// `aux_state` = `[1, rows, 8]` indexer key history), layer 1 GDN, layer 2 idle.
fn pcBuildQsaHybrid(s: mlx.mlx_stream, rows: c_int, conv_base: f64) [3]SSMCacheEntry {
    const aux_shape = [_]c_int{ 1, rows, 8 };
    const pooled_shape = [_]c_int{ 1, @divTrunc(rows, 4), 8 };
    return .{
        .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true, .aux_state = pcArange(s, &aux_shape, 0.0), .qsa_pooled = pcArange(s, &pooled_shape, 1000.0), .qsa_ratio = 4 },
        .{ .conv_state = pcArange(s, &conv_shape_pc, conv_base), .ssm_state = mlx.mlx_array_new(), .initialized = true },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
    };
}

fn pcFreeQsaHybrid(e: *[3]SSMCacheEntry) void {
    for (e) |*x| {
        if (x.conv_state.ctx != null) _ = mlx.mlx_array_free(x.conv_state);
        if (x.ssm_state.ctx != null) _ = mlx.mlx_array_free(x.ssm_state);
        if (x.aux_state.ctx != null) _ = mlx.mlx_array_free(x.aux_state);
        if (x.qsa_pooled.ctx != null) _ = mlx.mlx_array_free(x.qsa_pooled);
    }
}

test "HotPrefixCache: restore dump writes named tensors and is a no-op when unset" {
    const s = mlx.gpuStream();
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try testFillCache(&cache, s, 3, 8);
    var live = pcBuildQsaHybrid(s, 8, 100.0);
    defer pcFreeQsaHybrid(&live);

    restore_dump.dump_restore_override = null;
    try testing.expectEqual(@as(?u64, null), restore_dump.dumpRestoreIfEnabled(&cache, &live, s, .{
        .kind = "restore",
        .pos = 8,
    }));

    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const dir = buf[0..root_len];
    restore_dump.dump_restore_override = dir;
    defer restore_dump.dump_restore_override = null;
    const n = restore_dump.dumpRestoreIfEnabled(&cache, &live, s, .{
        .kind = "restore",
        .pos = 8,
        .cp = 8,
        .bank_from = 8,
        .source = "own",
        .entry_idx = 1,
        .entry_count = 1,
    }) orelse return error.TestExpectedEqual;

    const st_path = try std.fmt.allocPrintSentinel(testing.allocator, "{s}/restore-{d}-pos8.safetensors", .{ dir, n }, 0);
    defer testing.allocator.free(st_path);
    const json_path = try std.fmt.allocPrintSentinel(testing.allocator, "{s}/restore-{d}-pos8.json", .{ dir, n }, 0);
    defer testing.allocator.free(json_path);
    {
        const sf = std.c.fopen(st_path, "r") orelse return error.TestExpectedEqual;
        _ = std.c.fclose(sf);
        const jf = std.c.fopen(json_path, "r") orelse return error.TestExpectedEqual;
        _ = std.c.fclose(jf);
    }
    var tmap = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(tmap);
    var mmap = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(mmap);
    try mlx.check(mlx.mlx_load_safetensors(&tmap, &mmap, st_path, s));
    const names = [_][:0]const u8{
        "pos",
        "layers.0.aux_state",
        "layers.0.qsa_pooled",
        "layers.0.ple_prev",
        "layers.0.qsa_rows",
        "layers.0.qsa_key_rows",
        "layers.0.ring_start",
        "layers.1.conv_state",
        "kv.0.k_tail",
        "kv.0.v_tail",
    };
    for (names) |name| {
        var got = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(got);
        try mlx.check(mlx.mlx_map_string_to_array_get(&got, tmap, name));
        try testing.expect(got.ctx != null);
    }
}

test "HotPrefixCache: a QSA arch restore with no indexer history is a miss, never a poisoned entry" {
    // A snap without QSA history (a cancel handoff whose attach failed, an
    // old on-disk entry) used to restore aux-less; the next prefill then died
    // in qsaMaskFromQk with QsaHistoryGap on EVERY turn on that prefix. A
    // QSA arch treats "no history after restore" like "no checkpoint".
    const s = mlx.gpuStream();
    const tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const lookup_tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 70, 71 };
    for ([_]bool{ false, true }) |with_history| {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        defer hc.deinit();
        hc.qsa_history_required = true;
        var cache = try KVCache.init(testing.allocator, 3);
        defer cache.deinit();
        try testFillCache(&cache, s, 3, tokens.len);
        var live = pcBuildQsaHybrid(s, 10, 100.0);
        defer pcFreeQsaHybrid(&live);
        const cps = try testing.allocator.alloc(SSMCheckpoint, 1);
        cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 4, s);
        if (with_history) try transformer_mod.attachQsaHistoryToLatest(cps, &live, s);
        _ = try hc.commitWithState(&cache, &tokens, false, 0, cps, null, null);

        var target_cache = try KVCache.init(testing.allocator, 3);
        defer target_cache.deinit();
        var target = pcEmptySsm();
        defer pcFreeQsaHybrid(&target);
        var moe_off: usize = 0;
        const r = try hc.lookupAndRestore(&target_cache, &moe_off, &target, s, &lookup_tokens, false, 0, null, null);
        if (with_history) {
            try testing.expectEqual(@as(usize, 4), r.matched);
            try testing.expect(target[0].aux_state.ctx == null);
            try testing.expectEqual(@as(c_int, 4), target[0].qsa_hist_rows);
        } else {
            try testing.expectEqual(@as(usize, 0), r.matched);
            try testing.expectEqual(@as(usize, 0), moe_off);
            try testing.expect(target[0].aux_state.ctx == null);
        }
    }
}

test "HotPrefixCache: a history tensor shorter than the checkpoint is a miss, not a short hit" {
    const s = mlx.gpuStream();
    var tokens: [70]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 1);
    var lookup_latest: [70]u32 = tokens;
    lookup_latest[64] = 999;
    var lookup_interior: [70]u32 = tokens;
    lookup_interior[16] = 999;

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.qsa_history_required = true;
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try testFillCache(&cache, s, 3, tokens.len);
    var live = pcBuildQsaHybrid(s, 8, 100.0);
    defer pcFreeQsaHybrid(&live);
    const cps = try testing.allocator.alloc(SSMCheckpoint, 2);
    cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 16, s);
    cps[1] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 64, s);
    try transformer_mod.attachQsaHistoryToLatest(cps, &live, s);
    try testing.expect(cps[1].layers[0].aux_state.ctx == null);
    try testing.expectEqual(@as(c_int, 64), cps[1].layers[0].qsa_rows);
    _ = try hc.commitWithState(&cache, &tokens, false, 0, cps, null, null);

    for ([_][]const u32{ &lookup_latest, &lookup_interior }) |lookup| {
        var target_cache = try KVCache.init(testing.allocator, 3);
        defer target_cache.deinit();
        var target = pcEmptySsm();
        defer pcFreeQsaHybrid(&target);
        var moe_off: usize = 0;
        try testing.expectError(error.QsaHistoryGap, hc.lookupAndRestore(&target_cache, &moe_off, &target, s, lookup, false, 0, null, null));
        try testing.expectEqual(@as(usize, 0), target_cache.step);
        try testing.expect(target[0].aux_state.ctx == null);
    }
}

/// Two QSA layers, the second holding a SHORTER pooled bank than the first.
fn pcBuildQsaHybridUneven(s: mlx.mlx_stream, rows: c_int, short_blocks: c_int) [3]SSMCacheEntry {
    const aux_shape = [_]c_int{ 1, rows, 8 };
    const pooled_shape = [_]c_int{ 1, @divTrunc(rows, 4), 8 };
    const short_shape = [_]c_int{ 1, short_blocks, 8 };
    return .{
        .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true, .aux_state = pcArange(s, &aux_shape, 0.0), .qsa_pooled = pcArange(s, &pooled_shape, 1000.0), .qsa_ratio = 4 },
        .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true, .aux_state = pcArange(s, &aux_shape, 2000.0), .qsa_pooled = pcArange(s, &short_shape, 3000.0), .qsa_ratio = 4 },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
    };
}

test "HotPrefixCache: a lookup that fails after binding the KV leaves nothing behind" {
    // The sole caller reads any lookup error as "no match" and cold-prefills the WHOLE prompt:
    // a half-restored cache would hold the prefix twice, at the wrong RoPE positions.
    const s = mlx.gpuStream();
    var tokens: [16]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 1);
    var lookup: [16]u32 = tokens;
    lookup[10] = 999;

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.qsa_history_required = true;
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try testFillCache(&cache, s, 3, tokens.len);
    // The coverage pre-check maxes the pooled bank over layers; the apply is per layer.
    var live = pcBuildQsaHybridUneven(s, 16, 1);
    defer pcFreeQsaHybrid(&live);
    const cps = try testing.allocator.alloc(SSMCheckpoint, 2);
    cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 8, s);
    cps[1] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 12, s);
    try transformer_mod.attachQsaHistoryToLatest(cps, &live, s);
    _ = try hc.commitWithState(&cache, &tokens, false, 0, cps, null, null);

    var target_cache = try KVCache.init(testing.allocator, 3);
    defer target_cache.deinit();
    var target = pcEmptySsm();
    defer pcFreeQsaHybrid(&target);
    var moe_off: usize = 7;
    try testing.expectError(error.QsaHistoryGap, hc.lookupAndRestore(&target_cache, &moe_off, &target, s, &lookup, false, 0, null, null));
    try testing.expectEqual(@as(usize, 0), target_cache.step);
    try testing.expectEqual(@as(usize, 0), moe_off);
    for (&target) |*e| {
        try testing.expect(!e.initialized);
        try testing.expect(e.aux_state.ctx == null);
        try testing.expect(e.qsa_pooled.ctx == null);
    }
}

test "HotPrefixCache: prefix-extend keeps ONE QSA history across turns" {
    // The replace path inherits the old entry's checkpoints. Its latest snap
    // carried the full history and the new latest gets another one: one copy
    // per committed turn, the leak the stride fix closed by another door.
    const s = mlx.gpuStream();
    const t1 = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const t2 = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 };
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.qsa_history_required = true;

    var c1 = try KVCache.init(testing.allocator, 3);
    defer c1.deinit();
    try testFillCache(&c1, s, 3, t1.len);
    var l1 = pcBuildQsaHybrid(s, 10, 100.0);
    defer pcFreeQsaHybrid(&l1);
    const cps1 = try testing.allocator.alloc(SSMCheckpoint, 1);
    cps1[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &l1, 4, s);
    try transformer_mod.attachQsaHistoryToLatest(cps1, &l1, s);
    _ = try hc.commitWithState(&c1, &t1, false, 0, cps1, null, null);

    var c2 = try KVCache.init(testing.allocator, 3);
    defer c2.deinit();
    try testFillCache(&c2, s, 3, t2.len);
    var l2 = pcBuildQsaHybrid(s, 14, 200.0);
    defer pcFreeQsaHybrid(&l2);
    const cps2 = try testing.allocator.alloc(SSMCheckpoint, 1);
    cps2[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &l2, 8, s);
    try transformer_mod.attachQsaHistoryToLatest(cps2, &l2, s);
    _ = try hc.commitWithState(&c2, &t2, false, 0, cps2, null, null);

    try testing.expectEqual(@as(usize, 1), hc.entries.items.len);
    const merged = hc.entries.items[0].ssm_checkpoints.?;
    try testing.expectEqual(@as(usize, 2), merged.len);
    try testing.expect(!transformer_mod.checkpointHasQsaHistory(&merged[0]));
    try testing.expect(transformer_mod.checkpointHasQsaHistory(&merged[1]));
    try testing.expect(merged[1].layers[0].aux_state.ctx == null);
    try testing.expect(merged[1].layers[0].qsa_pooled.ctx != null);
}

test "HotPrefixCache: a handed-off QSA history commits, restores and bills exactly like the prefill-end copy" {
    // The commit handoff gives the newest snap a view of the slot's live history; the entry must
    // be indistinguishable from the copy arm and outlive the slot.
    const s = mlx.gpuStream();
    const tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const lookup_tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 70, 71 };
    var restored_val: [2]f32 = .{ -1.0, -2.0 };
    var billed: [2]u64 = .{ 0, 0 };
    for ([_]bool{ false, true }, 0..) |handoff, arm| {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        defer hc.deinit();
        hc.qsa_history_required = true;
        var cache = try KVCache.init(testing.allocator, 3);
        defer cache.deinit();
        try testFillCache(&cache, s, 3, tokens.len);
        var live = pcBuildQsaHybrid(s, 10, 100.0);
        const cps = try testing.allocator.alloc(SSMCheckpoint, 1);
        cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 4, s);
        if (handoff) {
            try transformer_mod.handoffQsaHistoryToLatest(cps, &live, s);
        } else {
            try transformer_mod.attachQsaHistoryToLatest(cps, &live, s);
        }
        _ = try hc.commitWithState(&cache, &tokens, false, 0, cps, null, null);
        pcFreeQsaHybrid(&live);
        billed[arm] = hc.entries.items[0].ssm_bytes;

        var target_cache = try KVCache.init(testing.allocator, 3);
        defer target_cache.deinit();
        var target = pcEmptySsm();
        defer pcFreeQsaHybrid(&target);
        var moe_off: usize = 0;
        const r = try hc.lookupAndRestore(&target_cache, &moe_off, &target, s, &lookup_tokens, false, 0, null, null);
        try testing.expectEqual(@as(usize, 4), r.matched);
        try testing.expect(target[0].aux_state.ctx == null);
        try testing.expectEqual(@as(c_int, 4), target[0].qsa_hist_rows);
        try testing.expectEqual(@as(c_int, 1), target[0].qsa_pooled_blocks);
        restored_val[arm] = 0;
    }
    try testing.expectEqual(restored_val[0], restored_val[1]);
    try testing.expect(billed[0] > 0);
    try testing.expectEqual(billed[0], billed[1]);
}

test "HotPrefixCache: replace path sheds inherited checkpoints instead of evicting its own entry (#330)" {
    const s = mlx.gpuStream();

    var toks: [900]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 3);

    var srcs: [8][3]SSMCacheEntry = undefined;
    for (&srcs, 0..) |*e, i| {
        const f: f64 = @floatFromInt(i + 1);
        e.* = pcBuildHybrid(s, 100.0 * f, 500.0 * f);
    }
    defer {
        for (&srcs) |*e| pcFreeHybrid(e);
    }

    // Turn 1: 450 tokens, checkpoints at 100..400.
    var c1 = try KVCache.init(testing.allocator, 3);
    defer c1.deinit();
    try testFillCache(&c1, s, 3, 450);
    const cps1 = try testing.allocator.alloc(SSMCheckpoint, 4);
    var cps1_bytes: u64 = 0;
    for (cps1, 0..) |*c, i| {
        c.* = try transformer_mod.captureSsmCheckpoint(testing.allocator, &srcs[i], (i + 1) * 100, s);
        cps1_bytes += transformer_mod.ssmCheckpointBytes(c);
    }

    // Turn 2 extends to 900 with checkpoints at 500..800.
    var c2 = try KVCache.init(testing.allocator, 3);
    defer c2.deinit();
    try testFillCache(&c2, s, 3, 900);
    const cps2 = try testing.allocator.alloc(SSMCheckpoint, 4);
    var c2_bytes: u64 = 0;
    for (cps2, 0..) |*c, i| {
        c.* = try transformer_mod.captureSsmCheckpoint(testing.allocator, &srcs[i + 4], (i + 5) * 100, s);
        c2_bytes += transformer_mod.ssmCheckpointBytes(c);
    }
    var c2_snap = try c2.snapshot();
    c2_bytes += HotPrefixCache.snapshotBytes(&c2_snap);
    c2_snap.deinit();

    // Turn 2 alone fits the budget; turn 2 plus the INHERITED turn-1
    // checkpoints does not. The pre-check cannot price the inheritance, so
    // pre-fix the post-merge loop evicted the sole, just-updated entry —
    // commit → evict everything → cold prefill, every turn.
    const budget = c2_bytes + cps1_bytes / 2;

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, budget);
    hc.ssm_checkpoint_max = 8;
    defer hc.deinit();
    _ = try hc.commitWithSsm(&c1, toks[0..450], false, cps1, null, null);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    _ = try hc.commitWithSsm(&c2, &toks, false, cps2, null, null);

    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    const e = &hc.entries.items[0];
    try testing.expectEqual(@as(usize, 900), e.tokens.len);
    try testing.expect(hc.current_kv_bytes <= hc.max_kv_bytes);
    // Shedding trimmed the checkpoint list, it did not empty it.
    try testing.expect(e.ssm_checkpoints.?.len >= 1);
}

test "HotPrefixCache: a failed commit still frees the checkpoints it was handed (#330 adjacent)" {
    const s = mlx.gpuStream();

    var src = try KVCache.init(testing.allocator, 1);
    defer src.deinit();
    try testFillCache(&src, s, 1, 8);

    // fail_index 0: the first cache-side allocation (the tokens dupe) fails.
    // Ownership of the checkpoints transfers to the cache UNCONDITIONALLY —
    // the scheduler's catch arm frees nothing (scan-pinned in scheduler.zig;
    // pre-fix it freed too, a double free with a different allocator). The
    // cache's error paths therefore MUST free the slice; std.testing.allocator
    // flags both the leak and a double free.
    var failing = std.testing.FailingAllocator.init(testing.allocator, .{ .fail_index = 0 });
    var hc = HotPrefixCache.initWithMem(failing.allocator(), 1, 0);
    defer hc.deinit();

    const cps = try testing.allocator.alloc(SSMCheckpoint, 1);
    cps[0] = .{ .pos = 4, .layers = try testing.allocator.alloc(transformer_mod.SSMCacheEntrySnapshot, 0) };
    var toks = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    try testing.expectError(error.OutOfMemory, hc.commitWithMediaState(&src, &toks, false, 0, 0, null, cps, null, null, toks.len));
    // No frees here: the cache owns the checkpoints on every outcome.
}

test "HotPrefixCache: a commit from a restored prefix inherits the donor's checkpoints" {
    // The 64k-ladder miss. Each rung sends the same growing prompt twice: an
    // MTP arm cold-prefills and commits entry A (checkpoints at stride), then
    // a serial arm restores ~the whole prompt from A, prefills the ~31-token
    // tail and commits its OWN entry B. B's tokens are NOT a prefix-extension
    // of A's (the two arms generate different tails), so the replace path —
    // the only checkpoint inheritance there was — never runs, and B's own
    // prefill was too short to earn a checkpoint (a restored tail inside the
    // window forwards as one span, `ssmSnapshotBackoff`). Once the byte budget evicted A, the
    // next rung found only B, every candidate `continue`d in
    // findBestRestorableMatch, and a 393k-token prompt cold-prefilled for
    // 560 s with no `[hot-cache]` line at all.
    const s = mlx.gpuStream();

    // Shared prompt P, then two different generated tails.
    var prompt: [20]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(i + 1);
    const a_tokens = prompt ++ [_]u32{ 200, 201 };
    const b_tokens = prompt ++ [_]u32{ 210, 211 };

    var srcs: [4][3]SSMCacheEntry = undefined;
    for (&srcs, 0..) |*e, i| {
        const f: f64 = @floatFromInt(i + 1);
        e.* = pcBuildHybrid(s, 100.0 * f, 500.0 * f);
    }
    defer {
        for (&srcs) |*e| pcFreeHybrid(e);
    }

    var hc = HotPrefixCache.initWithMem(testing.allocator, 2, 0);
    hc.ssm_checkpoint_max = 8;
    defer hc.deinit();

    // Entry A: the MTP arm's cold prefill. Checkpoints at 8 and 16 (inside
    // the shared prompt) and one at 21 (inside its OWN generated tail, which
    // B never saw and must not inherit).
    var a_cache = try KVCache.init(testing.allocator, 3);
    defer a_cache.deinit();
    try testFillCache(&a_cache, s, 3, a_tokens.len);
    const a_cps = try testing.allocator.alloc(SSMCheckpoint, 3);
    a_cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &srcs[0], 8, s);
    a_cps[1] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &srcs[1], 16, s);
    a_cps[2] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &srcs[2], 21, s);
    _ = try hc.commitWithState(&a_cache, &a_tokens, false, 0, a_cps, null, null);

    // Entry B: the serial arm. It restored from A and prefilled a tail too
    // short for a backoff, so its only checkpoint sits at the prompt end.
    var b_cache = try KVCache.init(testing.allocator, 3);
    defer b_cache.deinit();
    try testFillCache(&b_cache, s, 3, b_tokens.len);
    const b_cps = try testing.allocator.alloc(SSMCheckpoint, 1);
    b_cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &srcs[3], 20, s);
    _ = try hc.commitWithState(&b_cache, &b_tokens, false, 0, b_cps, null, null);

    try testing.expectEqual(@as(usize, 2), hc.entryCount());
    // B carries A's in-prompt checkpoints, and NOT the one at 21 (a position
    // only A's own generated tail ever reached).
    const b_idx: usize = if (hc.entries.items[0].tokens[20] == 210) 0 else 1;
    const b_merged = hc.entries.items[b_idx].ssm_checkpoints.?;
    try testing.expectEqual(@as(usize, 3), b_merged.len);
    try testing.expectEqual(@as(usize, 8), b_merged[0].pos);
    try testing.expectEqual(@as(usize, 16), b_merged[1].pos);
    try testing.expectEqual(@as(usize, 20), b_merged[2].pos);

    // The count cap evicts A (the byte budget is the same mechanism).
    var c_cache = try KVCache.init(testing.allocator, 3);
    defer c_cache.deinit();
    try testFillCache(&c_cache, s, 3, 4);
    const c_tokens = [_]u32{ 90, 91, 92, 93 };
    _ = try hc.commitWithState(&c_cache, &c_tokens, true, 0, null, null, null);
    try testing.expectEqual(@as(usize, 2), hc.entryCount());
    for (hc.entries.items) |*e| {
        try testing.expect(e.tokens.len != a_tokens.len or e.tokens[20] != 200);
    }

    // The next rung: shares the first 17 prompt tokens, then diverges (the
    // template's generation suffix renders differently once the turn enters
    // history). B's own checkpoint at 20 cannot serve it; A's at 16 can, and
    // B now carries it.
    const next = prompt[0..17].* ++ [_]u32{ 50, 51, 52 };
    var target_cache = try KVCache.init(testing.allocator, 3);
    defer target_cache.deinit();
    var target_ssm = pcEmptySsm();
    defer pcFreeHybrid(&target_ssm);
    var moe_off: usize = 0;
    const result = try hc.lookupAndRestore(&target_cache, &moe_off, &target_ssm, s, &next, false, 0, null, null);

    try testing.expectEqual(@as(usize, 16), result.matched);
    try testing.expectEqual(@as(usize, 16), target_cache.step);
    try testing.expectEqual(@as(usize, 16), moe_off);
    // The restored state is the checkpoint A captured at 16 (srcs[1]).
    try testing.expectEqual(@as(f32, 200.0), pcSsmVal(target_ssm[0].conv_state, 0, s));
}

test "HotPrefixCache: inheriting a 1-token-tail commit cannot restore a bank-less leftover" {
    const s = mlx.gpuStream();
    var donor_toks: [30]u32 = undefined;
    for (&donor_toks, 0..) |*t, i| t.* = @intCast(i + 1);
    var heir_toks: [19]u32 = undefined;
    @memcpy(heir_toks[0..17], donor_toks[0..17]);
    heir_toks[17] = 900;
    heir_toks[18] = 901;

    var live = pcBuildQsaHybrid(s, 25, 100.0);
    defer pcFreeQsaHybrid(&live);
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.qsa_history_required = true;

    var donor_cache = try KVCache.init(testing.allocator, 3);
    defer donor_cache.deinit();
    try testFillCache(&donor_cache, s, 3, donor_toks.len);
    const donor_cps = try testing.allocator.alloc(SSMCheckpoint, 3);
    donor_cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 8, s);
    donor_cps[1] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 17, s);
    donor_cps[2] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 25, s);
    try transformer_mod.attachQsaHistoryToLatest(donor_cps, &live, s);
    _ = try hc.commitWithState(&donor_cache, &donor_toks, false, 0, donor_cps, null, null);

    var heir_cache = try KVCache.init(testing.allocator, 3);
    defer heir_cache.deinit();
    try testFillCache(&heir_cache, s, 3, heir_toks.len);
    _ = try hc.commitWithState(&heir_cache, &heir_toks, false, 0, null, null, null);

    var lookup = heir_toks;
    var target_cache = try KVCache.init(testing.allocator, 3);
    defer target_cache.deinit();
    var target = pcEmptySsm();
    defer pcFreeQsaHybrid(&target);
    var moe_off: usize = 0;
    const r = try hc.lookupAndRestore(&target_cache, &moe_off, &target, s, &lookup, false, 0, null, null);
    const pos: c_int = @intCast(r.matched);
    const ratio: c_int = if (target[0].qsa_ratio > 0) target[0].qsa_ratio else 4;
    const blocks: c_int = if (target[0].qsa_pooled.ctx != null) mlx.getShape(target[0].qsa_pooled)[1] else 0;
    const held: c_int = if (target[0].aux_state.ctx != null) mlx.getShape(target[0].aux_state)[1] else 0;
    const backed = target[0].qsa_hist_rows == pos and blocks * ratio + held >= pos;
    try testing.expect(r.matched == 0 or backed);
}

test "HotPrefixCache: a 1-token-tail heir restores a bank that sits at prompt_end-1" {
    const s = mlx.gpuStream();
    var prefix: [18]u32 = undefined;
    for (&prefix, 0..) |*t, i| t.* = @intCast(i + 1);
    const donor_toks = prefix ++ [_]u32{ 90, 91 };
    const heir_toks = prefix ++ [_]u32{ 200, 201, 202, 203, 204, 205, 206, 207 };
    var live = pcBuildQsaHybrid(s, 18, 100.0);
    defer pcFreeQsaHybrid(&live);
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.qsa_history_required = true;

    var donor_cache = try KVCache.init(testing.allocator, 3);
    defer donor_cache.deinit();
    try testFillCache(&donor_cache, s, 3, donor_toks.len);
    const donor_cps = try testing.allocator.alloc(SSMCheckpoint, 2);
    donor_cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 8, s);
    donor_cps[1] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 18, s);
    try transformer_mod.attachQsaHistoryToLatest(donor_cps, &live, s);
    _ = try hc.commitWithState(&donor_cache, &donor_toks, false, 0, donor_cps, null, null);

    var heir_live = pcBuildQsaHybrid(s, 21, 100.0);
    defer pcFreeQsaHybrid(&heir_live);
    var heir_cache = try KVCache.init(testing.allocator, 3);
    defer heir_cache.deinit();
    try testFillCache(&heir_cache, s, 3, heir_toks.len);
    const heir_own = try testing.allocator.alloc(SSMCheckpoint, 1);
    heir_own[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &heir_live, 21, s);
    _ = try hc.commitWithState(&heir_cache, &heir_toks, false, 0, heir_own, null, null);

    var lookup = heir_toks ++ [_]u32{ 300, 301 };
    var target_cache = try KVCache.init(testing.allocator, 3);
    defer target_cache.deinit();
    var target = pcEmptySsm();
    defer pcFreeQsaHybrid(&target);
    var moe_off: usize = 0;
    const r = try hc.lookupAndRestore(&target_cache, &moe_off, &target, s, &lookup, false, 0, null, null);
    try testing.expectEqual(@as(usize, 18), r.matched);
    try testing.expectEqual(@as(c_int, 18), target[0].qsa_hist_rows);
    try testing.expect(target[0].qsa_pooled.ctx != null);
    const ratio: c_int = if (target[0].qsa_ratio > 0) target[0].qsa_ratio else 4;
    const blocks: c_int = mlx.getShape(target[0].qsa_pooled)[1];
    const held: c_int = if (target[0].aux_state.ctx != null) mlx.getShape(target[0].aux_state)[1] else 0;
    try testing.expect(blocks * ratio + held >= 18);
}

test "HotPrefixCache: a leftover-only checkpoint list is a lookup miss" {
    const s = mlx.gpuStream();
    var toks: [19]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
    var live = pcBuildQsaHybrid(s, 17, 100.0);
    defer pcFreeQsaHybrid(&live);
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.qsa_history_required = true;
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try testFillCache(&cache, s, 3, toks.len);
    const cps = try testing.allocator.alloc(SSMCheckpoint, 2);
    cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 8, s);
    cps[1] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 17, s);
    _ = try hc.commitWithState(&cache, &toks, false, 0, cps, null, null);

    var target_cache = try KVCache.init(testing.allocator, 3);
    defer target_cache.deinit();
    var target = pcEmptySsm();
    defer pcFreeQsaHybrid(&target);
    var moe_off: usize = 0;
    const r = hc.lookupAndRestore(&target_cache, &moe_off, &target, s, &toks, false, 0, null, null) catch |err| blk: {
        try testing.expectEqual(error.QsaHistoryGap, err);
        break :blk LookupResult{ .matched = 0, .full_match = false };
    };
    try testing.expectEqual(@as(usize, 0), r.matched);
}

test "HotPrefixCache: v3 heir full-reuse clamp satisfies QSA at the restored position, all four residues" {
    const s = mlx.gpuStream();
    var donor_toks: [30]u32 = undefined;
    for (&donor_toks, 0..) |*t, i| t.* = @intCast(i + 1);
    var heir_toks: [28]u32 = undefined;
    @memcpy(heir_toks[0..21], donor_toks[0..21]);
    for (heir_toks[21..], 0..) |*t, i| t.* = @intCast(900 + i);

    var donor_live = pcBuildQsaHybrid(s, 25, 100.0);
    defer pcFreeQsaHybrid(&donor_live);
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.qsa_history_required = true;

    var donor_cache = try KVCache.init(testing.allocator, 3);
    defer donor_cache.deinit();
    try testFillCache(&donor_cache, s, 3, donor_toks.len);
    const donor_cps = try testing.allocator.alloc(SSMCheckpoint, 3);
    donor_cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &donor_live, 8, s);
    donor_cps[1] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &donor_live, 16, s);
    donor_cps[2] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &donor_live, 25, s);
    try transformer_mod.attachQsaHistoryToLatest(donor_cps, &donor_live, s);
    _ = try hc.commitWithState(&donor_cache, &donor_toks, false, 0, donor_cps, null, null);

    var heir_live = pcBuildQsaHybrid(s, 21, 200.0);
    defer pcFreeQsaHybrid(&heir_live);
    var heir_cache = try KVCache.init(testing.allocator, 3);
    defer heir_cache.deinit();
    try testFillCache(&heir_cache, s, 3, heir_toks.len);
    const heir_own = try testing.allocator.alloc(SSMCheckpoint, 1);
    heir_own[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &heir_live, 20, s);
    try transformer_mod.attachQsaHistoryToLatest(heir_own, &heir_live, s);
    _ = try hc.commitWithState(&heir_cache, &heir_toks, false, 0, heir_own, null, null);

    var n: usize = 17;
    var hits: usize = 0;
    while (n <= 20) : (n += 1) {
        var target_cache = try KVCache.init(testing.allocator, 3);
        defer target_cache.deinit();
        var target = pcEmptySsm();
        defer pcFreeQsaHybrid(&target);
        var moe_off: usize = 0;
        const r = hc.lookupAndRestore(&target_cache, &moe_off, &target, s, heir_toks[0..n], false, 0, null, null) catch |err| blk: {
            try testing.expectEqual(error.QsaHistoryGap, err);
            break :blk LookupResult{ .matched = 0, .full_match = false };
        };
        try testing.expect(!r.full_match);
        if (r.matched > 0) {
            hits += 1;
            try testing.expect(transformer_mod.qsaRestoreSatisfiesForward(&target, r.matched));
            try testing.expectEqual(r.matched, moe_off);
        }
    }
    try testing.expect(hits == 4);
}

fn pcQsaArangeKeys(s: mlx.mlx_stream, rows: c_int, hd: c_int) !mlx.mlx_array {
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    mlx.check(mlx.mlx_arange(&flat, 0.0, @floatFromInt(rows * hd), 1.0, .float32, s)) catch return error.TestUnexpectedResult;
    var shaped = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(shaped);
    mlx.check(mlx.mlx_reshape(&shaped, flat, &[_]c_int{ 1, rows, hd }, 3, s)) catch return error.TestUnexpectedResult;
    var out = mlx.mlx_array_new();
    mlx.check(mlx.mlx_astype(&out, shaped, .bfloat16, s)) catch return error.TestUnexpectedResult;
    _ = mlx.mlx_array_eval(out);
    return out;
}

fn pcQsaBf16Equal(s: mlx.mlx_stream, a: mlx.mlx_array, b: mlx.mlx_array) !void {
    try testing.expect(a.ctx != null);
    try testing.expect(b.ctx != null);
    _ = mlx.mlx_array_eval(a);
    _ = mlx.mlx_array_eval(b);
    var eq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eq);
    try mlx.check(mlx.mlx_array_equal(&eq, a, b, false, s));
    _ = mlx.mlx_array_eval(eq);
    var ok: bool = false;
    try mlx.check(mlx.mlx_array_item_bool(&ok, eq));
    try testing.expect(ok);
}

fn pcQsaSliceAxis1(s: mlx.mlx_stream, arr: mlx.mlx_array, from: c_int, to: c_int) !mlx.mlx_array {
    const sh = mlx.getShape(arr);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&out, arr, &[_]c_int{ 0, from, 0 }, 3, &[_]c_int{ sh[0], to, sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    return out;
}

fn pcQsaExpectViewEqual(s: mlx.mlx_stream, warm: *SSMCacheEntry, cold: *SSMCacheEntry, pos: c_int) !void {
    try testing.expectEqual(pos, warm.qsa_hist_rows);
    try testing.expectEqual(pos, cold.qsa_hist_rows);
    const ratio = @max(@max(warm.qsa_ratio, cold.qsa_ratio), 1);
    const need_blocks = @divTrunc(pos, ratio);
    const leftover_n = @mod(pos, ratio);
    if (need_blocks > 0) {
        try testing.expect(warm.qsa_pooled.ctx != null);
        try testing.expect(cold.qsa_pooled.ctx != null);
        const w = try pcQsaSliceAxis1(s, warm.qsa_pooled, 0, need_blocks);
        defer _ = mlx.mlx_array_free(w);
        const c = try pcQsaSliceAxis1(s, cold.qsa_pooled, 0, need_blocks);
        defer _ = mlx.mlx_array_free(c);
        pcQsaBf16Equal(s, w, c) catch {
            log.warn("qsa pooled mismatch at pos {d} need_blocks {d} warm_blocks {d} cold_blocks {d}\n", .{
                pos,
                need_blocks,
                mlx.getShape(warm.qsa_pooled)[1],
                mlx.getShape(cold.qsa_pooled)[1],
            });
            return error.TestExpectedEqual;
        };
    }
    if (leftover_n > 0) {
        try testing.expect(warm.aux_state.ctx != null);
        try testing.expect(cold.aux_state.ctx != null);
        const wh = mlx.getShape(warm.aux_state)[1];
        const ch = mlx.getShape(cold.aux_state)[1];
        try testing.expect(wh >= leftover_n);
        try testing.expect(ch >= leftover_n);
        const w = try pcQsaSliceAxis1(s, warm.aux_state, wh - leftover_n, wh);
        defer _ = mlx.mlx_array_free(w);
        const c = try pcQsaSliceAxis1(s, cold.aux_state, ch - leftover_n, ch);
        defer _ = mlx.mlx_array_free(c);
        pcQsaBf16Equal(s, w, c) catch {
            log.warn("qsa leftover mismatch at pos {d} leftover {d} warm_aux {d} cold_aux {d}\n", .{ pos, leftover_n, wh, ch });
            return error.TestExpectedEqual;
        };
    }
}

fn pcQsaFeed(xfm: *transformer_mod.Transformer, entry: *SSMCacheEntry, keys: mlx.mlx_array, n: c_int, s: mlx.mlx_stream) !void {
    const chunk = try pcQsaSliceAxis1(s, keys, 0, n);
    defer _ = mlx.mlx_array_free(chunk);
    try transformer_mod.qsaTestAppendPool(xfm, entry, chunk, 0, s);
}

fn pcQsaSweep(hc: *HotPrefixCache, toks: []const u32, keys: mlx.mlx_array, lo: usize, hi: usize, s: mlx.mlx_stream) !void {
    var xfm: transformer_mod.Transformer = undefined;
    xfm.s = s;
    xfm.allocator = testing.allocator;
    var r = lo;
    while (r <= hi) : (r += 1) {
        var target_cache = try KVCache.init(testing.allocator, 3);
        defer target_cache.deinit();
        var target = pcEmptySsm();
        defer pcFreeQsaHybrid(&target);
        var moe_off: usize = 0;
        const got = hc.lookupAndRestore(&target_cache, &moe_off, &target, s, toks[0..r], false, 0, null, null) catch |err| {
            log.warn("qsa value sweep lookup at r={d}: {s}\n", .{ r, @errorName(err) });
            return err;
        };
        try testing.expect(got.matched > 0);
        const pos: c_int = @intCast(got.matched);
        var cold = pcEmptySsm();
        defer pcFreeQsaHybrid(&cold);
        cold[0].qsa_ratio = 4;
        try pcQsaFeed(&xfm, &cold[0], keys, pos, s);
        pcQsaExpectViewEqual(s, &target[0], &cold[0], pos) catch {
            log.warn("qsa value sweep mismatch r={d} restored={d} full_match={}\n", .{ r, got.matched, got.full_match });
            return error.TestExpectedEqual;
        };
    }
}

test "HotPrefixCache: restored QSA history values match a cold feed at every position" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const hd: c_int = 8;
    const P: usize = 80;
    const keys = try pcQsaArangeKeys(s, @intCast(P), hd);
    defer _ = mlx.mlx_array_free(keys);
    var toks: [P]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);

    var xfm: transformer_mod.Transformer = undefined;
    xfm.s = s;
    xfm.allocator = testing.allocator;

    // Plain entry: checkpoints at every r in [P-40, P], bank on latest.
    {
        var live: [3]SSMCacheEntry = .{
            .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true, .qsa_ratio = 4 },
            .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = false },
            .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = false },
        };
        defer pcFreeQsaHybrid(&live);
        const n_cps: usize = 41;
        const cps = try testing.allocator.alloc(SSMCheckpoint, n_cps);
        var i: usize = 0;
        while (i < n_cps) : (i += 1) {
            const pos: c_int = @intCast(P - 40 + i);
            const chunk = try pcQsaSliceAxis1(s, keys, 0, pos);
            defer _ = mlx.mlx_array_free(chunk);
            transformer_mod.ssmFreeQsaState(&live[0]);
            live[0] = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true, .qsa_ratio = 4 };
            try transformer_mod.qsaTestAppendPool(&xfm, &live[0], chunk, 0, s);
            cps[i] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, @intCast(pos), s);
        }
        try transformer_mod.attachQsaHistoryToLatest(cps, &live, s);
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        defer hc.deinit();
        hc.qsa_history_required = true;
        var cache = try KVCache.init(testing.allocator, 3);
        defer cache.deinit();
        try testFillCache(&cache, s, 3, toks.len);
        _ = try hc.commitWithState(&cache, &toks, false, 0, cps, null, null);
        try pcQsaSweep(&hc, &toks, keys, P - 39, P, s);
    }

    // Inherited entry: donor bank sits above shared; heir is a short-tail commit.
    {
        const shared: usize = 48;
        const donor_len: usize = 64;
        const donor_keys = try pcQsaArangeKeys(s, @intCast(donor_len), hd);
        defer _ = mlx.mlx_array_free(donor_keys);
        var donor_toks: [64]u32 = undefined;
        for (&donor_toks, 0..) |*t, i| t.* = @intCast(i + 1);
        var heir_toks: [56]u32 = undefined;
        @memcpy(heir_toks[0..shared], donor_toks[0..shared]);
        for (heir_toks[shared..], 0..) |*t, i| t.* = @intCast(10_000 + i);

        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        defer hc.deinit();
        hc.qsa_history_required = true;

        var donor_live: [3]SSMCacheEntry = .{
            .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true, .qsa_ratio = 4 },
            .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = false },
            .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = false },
        };
        defer pcFreeQsaHybrid(&donor_live);
        const donor_cps = try testing.allocator.alloc(SSMCheckpoint, 5);
        for ([_]c_int{ 16, 32, 48, 56, 64 }, 0..) |pos, i| {
            const chunk = try pcQsaSliceAxis1(s, donor_keys, 0, pos);
            defer _ = mlx.mlx_array_free(chunk);
            transformer_mod.ssmFreeQsaState(&donor_live[0]);
            donor_live[0] = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true, .qsa_ratio = 4 };
            try transformer_mod.qsaTestAppendPool(&xfm, &donor_live[0], chunk, 0, s);
            donor_cps[i] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &donor_live, @intCast(pos), s);
        }
        try transformer_mod.attachQsaHistoryToLatest(donor_cps, &donor_live, s);
        var donor_cache = try KVCache.init(testing.allocator, 3);
        defer donor_cache.deinit();
        try testFillCache(&donor_cache, s, 3, donor_toks.len);
        _ = try hc.commitWithState(&donor_cache, &donor_toks, false, 0, donor_cps, null, null);

        var heir_live: [3]SSMCacheEntry = .{
            .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true, .qsa_ratio = 4 },
            .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = false },
            .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = false },
        };
        defer pcFreeQsaHybrid(&heir_live);
        {
            const chunk = try pcQsaSliceAxis1(s, donor_keys, 0, @intCast(shared));
            defer _ = mlx.mlx_array_free(chunk);
            try transformer_mod.qsaTestAppendPool(&xfm, &heir_live[0], chunk, 0, s);
        }
        const heir_own = try testing.allocator.alloc(SSMCheckpoint, 1);
        heir_own[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &heir_live, shared, s);
        var heir_cache = try KVCache.init(testing.allocator, 3);
        defer heir_cache.deinit();
        try testFillCache(&heir_cache, s, 3, heir_toks.len);
        _ = try hc.commitWithState(&heir_cache, &heir_toks, false, 0, heir_own, null, null);

        try pcQsaSweep(&hc, heir_toks[0..shared], donor_keys, 17, shared, s);
    }
}

test "HotPrefixCache: QSA trim checkpoint bill matches materialized history" {
    const s = mlx.gpuStream();
    var live = pcBuildQsaHybrid(s, 1027, 100.0);
    defer pcFreeQsaHybrid(&live);
    var cps: [2]SSMCheckpoint = undefined;
    cps[1] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 1027, s);
    defer cps[1].deinit(testing.allocator);
    try transformer_mod.attachQsaHistoryToLatest(cps[1..], &live, s);
    for ([_]usize{ 256, 257, 258, 259, 512, 1024, 1027 }) |pos| {
        cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, pos, s);
        defer cps[0].deinit(testing.allocator);
        const predicted = HotPrefixCache.trimmedCheckpointBytes(&cps, 0);
        try sliceQsaHistoryOntoCheckpoint(&cps[0], &cps[1], pos, s);
        try testing.expectEqual(predicted, ssmCheckpointBytes(&cps[0]));
        // Pricing a checkpoint that already carries its bank must not double-bill it.
        try testing.expectEqual(predicted, HotPrefixCache.trimmedCheckpointBytes(&cps, 0));
    }
}

test "HotPrefixCache: oversized QSA trim bills the transferred bank and stays reusable" {
    const s = mlx.gpuStream();
    for ([_]transformer_mod.ThinPolicy{ .min_span_recency, .min_span }) |policy| {
        for ([_]bool{ false, true }) |replace| {
            var toks: [1024]u32 = undefined;
            for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
            var live = pcBuildQsaHybrid(s, 1024, 100.0);
            defer pcFreeQsaHybrid(&live);
            var cache = try KVCache.init(testing.allocator, 3);
            defer cache.deinit();
            try testFillCache(&cache, s, 3, toks.len);
            var snap = try cache.snapshot();
            defer snap.deinit();
            const cps = try testing.allocator.alloc(SSMCheckpoint, 4);
            for (cps, 0..) |*cp, i| cp.* = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, (i + 1) * 256, s);
            try transformer_mod.attachQsaHistoryToLatest(cps, &live, s);
            // 768 KV rows and all small checkpoints fit, but the 768-row bank does not.
            const budget = HotPrefixCache.snapshotRowBytes(&snap) * 768 + ssmCheckpointBytes(&cps[0]) * 4 + 1024;
            var hc = HotPrefixCache.initWithMem(testing.allocator, 1, budget);
            defer hc.deinit();
            hc.cp_thin = policy;
            hc.qsa_history_required = true;
            if (replace) {
                var seed = try KVCache.init(testing.allocator, 3);
                defer seed.deinit();
                try testFillCache(&seed, s, 3, 256);
                const seed_cps = try testing.allocator.alloc(SSMCheckpoint, 1);
                seed_cps[0] = try transformer_mod.shareSsmCheckpoint(testing.allocator, &cps[0]);
                try sliceQsaHistoryOntoCheckpoint(&seed_cps[0], &cps[3], 256, s);
                _ = try hc.commitWithState(&seed, toks[0..256], false, 0, seed_cps, null, null);
            }
            const chosen = hc.trimLenForBudget(budget, toks.len, HotPrefixCache.snapshotRowBytes(&snap), cps);
            _ = try hc.commitWithState(&cache, &toks, false, 0, cps, null, null);
            try testing.expectEqual(@as(?usize, 512), chosen);
            try testing.expectEqual(@as(usize, 1), hc.entryCount());
            try testing.expect(hc.current_kv_bytes <= budget);
            try testing.expectEqual(@as(usize, 512), hc.entries.items[0].tokens.len);
            var restored = pcEmptySsm();
            defer pcFreeQsaHybrid(&restored);
            var target = try KVCache.init(testing.allocator, 3);
            defer target.deinit();
            var moe_off: usize = 0;
            const hit = try hc.lookupAndRestore(&target, &moe_off, &restored, s, &toks, false, 0, null, null);
            try testing.expectEqual(@as(usize, 512), hit.matched);
        }
    }
}

test "HotPrefixCache: shed rescues an interior QSA bank" {
    const s = mlx.gpuStream();
    var toks: [30]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
    var live = pcBuildQsaHybrid(s, 25, 100.0);
    defer pcFreeQsaHybrid(&live);
    var probe = try KVCache.init(testing.allocator, 3);
    defer probe.deinit();
    try testFillCache(&probe, s, 3, toks.len);
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.qsa_history_required = true;
    hc.ssm_checkpoint_max = 8;
    const cps = try testing.allocator.alloc(SSMCheckpoint, 3);
    cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 8, s);
    cps[1] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 17, s);
    cps[2] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 25, s);
    try transformer_mod.attachQsaHistoryToLatest(cps, &live, s);
    try transformer_mod.sliceQsaHistoryOntoCheckpoint(&cps[1], &cps[2], cps[1].pos, s);
    for (cps[2].layers) |*l| {
        if (l.qsa_pooled.ctx != null) _ = mlx.mlx_array_free(l.qsa_pooled);
        l.qsa_pooled = .{ .ctx = null };
    }
    _ = try hc.commitWithState(&probe, &toks, false, 0, cps, null, null);
    const before = hc.current_kv_bytes;
    const listed = hc.entries.items[0].ssm_checkpoints.?;
    hc.max_kv_bytes = before - ssmCheckpointBytes(&listed[1]) + 1;
    hc.shedCheckpointsToFit();
    try testing.expect(hc.entryCount() == 1);
    const kept = hc.entries.items[0].ssm_checkpoints.?;
    try testing.expect(kept.len >= 1);
    try testing.expect(transformer_mod.checkpointListHasQsaPooled(kept));
}

test "HotPrefixCache: dropLastRestored removes the restored entry so the next lookup is cold" {
    const s = mlx.gpuStream();
    const tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.qsa_history_required = true;
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try testFillCache(&cache, s, 3, tokens.len);
    var live = pcBuildQsaHybrid(s, 10, 100.0);
    defer pcFreeQsaHybrid(&live);
    const cps = try testing.allocator.alloc(SSMCheckpoint, 1);
    cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 8, s);
    try transformer_mod.attachQsaHistoryToLatest(cps, &live, s);
    _ = try hc.commitWithState(&cache, &tokens, false, 0, cps, null, null);

    var target_cache = try KVCache.init(testing.allocator, 3);
    defer target_cache.deinit();
    var target = pcEmptySsm();
    defer pcFreeQsaHybrid(&target);
    var moe_off: usize = 0;
    const first = try hc.lookupAndRestore(&target_cache, &moe_off, &target, s, &tokens, false, 0, null, null);
    try testing.expect(first.matched > 0);
    try testing.expect(hc.dropLastRestored());
    try testing.expectEqual(@as(usize, 0), hc.entryCount());
    var target2 = pcEmptySsm();
    defer pcFreeQsaHybrid(&target2);
    var moe2: usize = 0;
    var cache2 = try KVCache.init(testing.allocator, 3);
    defer cache2.deinit();
    const second = try hc.lookupAndRestore(&cache2, &moe2, &target2, s, &tokens, false, 0, null, null);
    try testing.expectEqual(@as(usize, 0), second.matched);
}

test "HotPrefixCache: QSA self-heal drops the slot model cache and leaves a sibling cache" {
    const s = mlx.gpuStream();
    const tokens_a = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const tokens_b = [_]u32{ 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };

    var last_loaded = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer last_loaded.deinit();
    last_loaded.qsa_history_required = true;
    var cache_a = try KVCache.init(testing.allocator, 3);
    defer cache_a.deinit();
    try testFillCache(&cache_a, s, 3, tokens_a.len);
    var live_a = pcBuildQsaHybrid(s, 10, 100.0);
    defer pcFreeQsaHybrid(&live_a);
    const cps_a = try testing.allocator.alloc(SSMCheckpoint, 1);
    cps_a[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live_a, 8, s);
    try transformer_mod.attachQsaHistoryToLatest(cps_a, &live_a, s);
    _ = try last_loaded.commitWithState(&cache_a, &tokens_a, false, 0, cps_a, null, null);

    var slot_hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer slot_hc.deinit();
    slot_hc.qsa_history_required = true;
    var cache_b = try KVCache.init(testing.allocator, 3);
    defer cache_b.deinit();
    try testFillCache(&cache_b, s, 3, tokens_b.len);
    var live_b = pcBuildQsaHybrid(s, 10, 200.0);
    defer pcFreeQsaHybrid(&live_b);
    const cps_b = try testing.allocator.alloc(SSMCheckpoint, 1);
    cps_b[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live_b, 8, s);
    try transformer_mod.attachQsaHistoryToLatest(cps_b, &live_b, s);
    _ = try slot_hc.commitWithState(&cache_b, &tokens_b, false, 0, cps_b, null, null);

    var tgt_a = try KVCache.init(testing.allocator, 3);
    defer tgt_a.deinit();
    var ssm_a = pcEmptySsm();
    defer pcFreeQsaHybrid(&ssm_a);
    var moe_a: usize = 0;
    const ra = try last_loaded.lookupAndRestore(&tgt_a, &moe_a, &ssm_a, s, &tokens_a, false, 0, null, null);
    try testing.expect(ra.matched > 0);

    var tgt_b = try KVCache.init(testing.allocator, 3);
    defer tgt_b.deinit();
    var ssm_b = pcEmptySsm();
    defer pcFreeQsaHybrid(&ssm_b);
    var moe_b: usize = 0;
    const rb = try slot_hc.lookupAndRestore(&tgt_b, &moe_b, &ssm_b, s, &tokens_b, false, 0, null, null);
    try testing.expect(rb.matched > 0);

    try testing.expect(HotPrefixCache.dropQsaGapEntry(&slot_hc));
    try testing.expectEqual(@as(usize, 0), slot_hc.entryCount());
    try testing.expectEqual(@as(usize, 1), last_loaded.entryCount());
}

test "HotPrefixCache: inherited checkpoints never exceed the shared prefix" {
    const s = mlx.gpuStream();
    var prefix: [18]u32 = undefined;
    for (&prefix, 0..) |*t, i| t.* = @intCast(i + 1);
    const donor_toks = prefix ++ [_]u32{ 90, 91 };
    const prompt_len: usize = 19;
    const heir_toks = prefix ++ [_]u32{200} ++ [_]u32{ 300, 301, 302, 303, 304, 305, 306, 307 };
    var live = pcBuildQsaHybrid(s, 18, 100.0);
    defer pcFreeQsaHybrid(&live);
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.qsa_history_required = true;

    var donor_cache = try KVCache.init(testing.allocator, 3);
    defer donor_cache.deinit();
    try testFillCache(&donor_cache, s, 3, donor_toks.len);
    const donor_cps = try testing.allocator.alloc(SSMCheckpoint, 2);
    donor_cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 8, s);
    donor_cps[1] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 18, s);
    try transformer_mod.attachQsaHistoryToLatest(donor_cps, &live, s);
    _ = try hc.commitWithState(&donor_cache, &donor_toks, false, 0, donor_cps, null, null);

    var heir_cache = try KVCache.init(testing.allocator, 3);
    defer heir_cache.deinit();
    try testFillCache(&heir_cache, s, 3, heir_toks.len);
    _ = try hc.commitWithState(&heir_cache, &heir_toks, false, 0, null, null, null);

    var found = false;
    for (hc.entries.items) |*e| {
        if (e.tokens.len != heir_toks.len) continue;
        found = true;
        const cps = e.ssm_checkpoints orelse continue;
        for (cps) |*cp| try testing.expect(cp.pos <= prompt_len - 1);
    }
    try testing.expect(found);
}

fn pcArrEqual(s: mlx.mlx_stream, a: mlx.mlx_array, b: mlx.mlx_array) !void {
    if (a.ctx == null and b.ctx == null) return;
    try testing.expect(a.ctx != null);
    try testing.expect(b.ctx != null);
    _ = mlx.mlx_array_eval(a);
    _ = mlx.mlx_array_eval(b);
    var eq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eq);
    try mlx.check(mlx.mlx_array_equal(&eq, a, b, false, s));
    _ = mlx.mlx_array_eval(eq);
    var ok: bool = false;
    try mlx.check(mlx.mlx_array_item_bool(&ok, eq));
    try testing.expect(ok);
}

fn pcSsmDiffField(s: mlx.mlx_stream, a: []const SSMCacheEntry, b: []const SSMCacheEntry) ![]const u8 {
    if (a.len != b.len) return "layer_count";
    for (a, b, 0..) |*la, *lb, i| {
        _ = i;
        if (la.initialized != lb.initialized) return "initialized";
        if (la.ple_prev_valid != lb.ple_prev_valid) return "ple_prev_valid";
        if (!std.mem.eql(u32, &la.ple_prev, &lb.ple_prev)) return "ple_prev";
        if (la.qsa_ratio != lb.qsa_ratio) return "qsa_ratio";
        if (la.qsa_hist_rows != lb.qsa_hist_rows) return "qsa_hist_rows";
        if (la.qsa_key_rows != lb.qsa_key_rows) return "qsa_key_rows";
        if (la.qsa_pooled_blocks != lb.qsa_pooled_blocks) return "qsa_pooled_blocks";
        pcArrEqual(s, la.conv_state, lb.conv_state) catch return "conv_state";
        pcArrEqual(s, la.ssm_state, lb.ssm_state) catch return "ssm_state";
        pcArrEqual(s, la.qsa_pooled, lb.qsa_pooled) catch return "qsa_pooled";
        pcArrEqual(s, la.aux_state, lb.aux_state) catch return "aux_state";
    }
    return "";
}

fn pcCaptureFed(
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,
    xfm: *transformer_mod.Transformer,
    keys: mlx.mlx_array,
    pos: c_int,
    conv_base: f64,
    ple0: u32,
) !SSMCheckpoint {
    var live: [3]SSMCacheEntry = .{
        .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true, .qsa_ratio = 4 },
        .{ .conv_state = pcArange(s, &conv_shape_pc, conv_base), .ssm_state = mlx.mlx_array_new(), .initialized = true, .ple_prev = .{ ple0, 0, 0, 0, 0, 0, 0, 0 }, .ple_prev_valid = true },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
    };
    defer pcFreeQsaHybrid(&live);
    try pcQsaFeed(xfm, &live[0], keys, pos, s);
    return transformer_mod.captureSsmCheckpoint(allocator, &live, @intCast(pos), s);
}

test "HotPrefixCache: inherit does not clone checkpoints past donor.shared" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const P: usize = 20;
    const hd: c_int = 8;
    var xfm: transformer_mod.Transformer = undefined;
    xfm.s = s;
    xfm.allocator = testing.allocator;
    const keys = try pcQsaArangeKeys(s, 40, hd);
    defer _ = mlx.mlx_array_free(keys);
    var donor_toks: [40]u32 = undefined;
    var heir_toks: [40]u32 = undefined;
    for (0..P) |i| {
        donor_toks[i] = @intCast(i + 1);
        heir_toks[i] = @intCast(i + 1);
    }
    for (P..40) |i| {
        donor_toks[i] = @intCast(1000 + i);
        heir_toks[i] = @intCast(2000 + i);
    }
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.qsa_history_required = true;

    var donor_cache = try KVCache.init(testing.allocator, 3);
    defer donor_cache.deinit();
    try testFillCache(&donor_cache, s, 3, donor_toks.len);
    const donor_cps = try testing.allocator.alloc(SSMCheckpoint, 4);
    donor_cps[0] = try pcCaptureFed(testing.allocator, s, &xfm, keys, 8, 100.0, 8);
    donor_cps[1] = try pcCaptureFed(testing.allocator, s, &xfm, keys, 16, 160.0, 16);
    donor_cps[2] = try pcCaptureFed(testing.allocator, s, &xfm, keys, 30, 300.0, 30);
    donor_cps[3] = try pcCaptureFed(testing.allocator, s, &xfm, keys, 40, 400.0, 40);
    {
        var live: [3]SSMCacheEntry = .{
            .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true, .qsa_ratio = 4 },
            .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = false },
            .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = false },
        };
        defer pcFreeQsaHybrid(&live);
        try pcQsaFeed(&xfm, &live[0], keys, 40, s);
        try transformer_mod.attachQsaHistoryToLatest(donor_cps, &live, s);
    }
    _ = try hc.commitWithState(&donor_cache, &donor_toks, false, 0, donor_cps, null, null);

    var heir_cache = try KVCache.init(testing.allocator, 3);
    defer heir_cache.deinit();
    try testFillCache(&heir_cache, s, 3, heir_toks.len);
    _ = try hc.commitWithState(&heir_cache, &heir_toks, false, 0, null, null, null);

    var found = false;
    for (hc.entries.items) |*e| {
        if (e.tokens.len != heir_toks.len or e.tokens[P] != heir_toks[P]) continue;
        found = true;
        const cps = e.ssm_checkpoints orelse return error.TestExpectedEqual;
        for (cps) |*cp| try testing.expect(cp.pos <= P);
    }
    try testing.expect(found);

    var rpos: usize = P - 4;
    while (rpos <= P) : (rpos += 1) {
        var target_cache = try KVCache.init(testing.allocator, 3);
        defer target_cache.deinit();
        var target = pcEmptySsm();
        defer pcFreeQsaHybrid(&target);
        var moe_off: usize = 0;
        const got = try hc.lookupAndRestore(&target_cache, &moe_off, &target, s, heir_toks[0..rpos], false, 0, null, null);
        try testing.expect(got.matched > 0);
        try testing.expect(got.matched <= P);
        var cold = pcEmptySsm();
        defer pcFreeQsaHybrid(&cold);
        cold[0].qsa_ratio = 4;
        try pcQsaFeed(&xfm, &cold[0], keys, @intCast(got.matched), s);
        pcQsaExpectViewEqual(s, &target[0], &cold[0], @intCast(got.matched)) catch {
            log.warn("inherit divergence restore mismatch pos={d} restored={d}\n", .{ rpos, got.matched });
            return error.TestExpectedEqual;
        };
    }
}

test "HotPrefixCache: inherit of a greedy continuation stops at the prompt" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const P: usize = 48;
    const prompt_len: usize = P - 1;
    const hd: c_int = 8;
    var xfm: transformer_mod.Transformer = undefined;
    xfm.s = s;
    xfm.allocator = testing.allocator;
    const keys = try pcQsaArangeKeys(s, 80, hd);
    defer _ = mlx.mlx_array_free(keys);

    var donor_toks: [64]u32 = undefined;
    for (&donor_toks, 0..) |*t, i| t.* = @intCast(i + 1);
    var heir_toks: [59]u32 = undefined;
    @memcpy(&heir_toks, donor_toks[0..59]);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.qsa_history_required = true;

    var donor_cache = try KVCache.init(testing.allocator, 3);
    defer donor_cache.deinit();
    try testFillCache(&donor_cache, s, 3, donor_toks.len);
    const donor_cps = try testing.allocator.alloc(SSMCheckpoint, 3);
    donor_cps[0] = try pcCaptureFed(testing.allocator, s, &xfm, keys, 19, 190.0, 19);
    donor_cps[1] = try pcCaptureFed(testing.allocator, s, &xfm, keys, @intCast(P), 480.0, 48);
    donor_cps[2] = try pcCaptureFed(testing.allocator, s, &xfm, keys, 64, 640.0, 64);
    {
        var live: [3]SSMCacheEntry = .{
            .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true, .qsa_ratio = 4 },
            .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = false },
            .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = false },
        };
        defer pcFreeQsaHybrid(&live);
        try pcQsaFeed(&xfm, &live[0], keys, 64, s);
        try transformer_mod.attachQsaHistoryToLatest(donor_cps, &live, s);
    }
    _ = try hc.commitWithState(&donor_cache, &donor_toks, false, 0, donor_cps, null, null);

    var heir_cache = try KVCache.init(testing.allocator, 3);
    defer heir_cache.deinit();
    try testFillCache(&heir_cache, s, 3, heir_toks.len);
    _ = try hc.commitWithMediaState(&heir_cache, &heir_toks, false, 0, 0, null, null, null, null, prompt_len);

    var found = false;
    for (hc.entries.items) |*e| {
        if (e.tokens.len != heir_toks.len) continue;
        found = true;
        const cps = e.ssm_checkpoints orelse return error.TestExpectedEqual;
        for (cps) |*cp| try testing.expect(cp.pos <= prompt_len);
    }
    try testing.expect(found);
}

test "HotPrefixCache: sliced qsa bank apply at backoff matches the full bank at every residue" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const hd: c_int = 8;
    const ratio: c_int = 4;
    const B: usize = 19;
    const H: usize = 80;
    var xfm: transformer_mod.Transformer = undefined;
    xfm.s = s;
    xfm.allocator = testing.allocator;
    const keys = try pcQsaArangeKeys(s, @intCast(H), hd);
    defer _ = mlx.mlx_array_free(keys);

    var live: [3]SSMCacheEntry = .{
        .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true, .qsa_ratio = ratio },
        .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = false },
        .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = false },
    };
    defer pcFreeQsaHybrid(&live);
    try pcQsaFeed(&xfm, &live[0], keys, @intCast(B), s);
    var cps = [_]SSMCheckpoint{
        try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, B, s),
        undefined,
    };
    transformer_mod.ssmFreeQsaState(&live[0]);
    live[0] = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true, .qsa_ratio = ratio };
    try pcQsaFeed(&xfm, &live[0], keys, @intCast(H), s);
    cps[1] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, H, s);
    defer for (&cps) |*c| c.deinit(testing.allocator);
    try transformer_mod.attachQsaHistoryToLatest(&cps, &live, s);
    if (live[0].aux_state.ctx != null) {
        if (cps[1].layers[0].aux_state.ctx != null) _ = mlx.mlx_array_free(cps[1].layers[0].aux_state);
        cps[1].layers[0].aux_state = try transformer_mod.materializedOwnedCopy(s, live[0].aux_state);
        _ = mlx.mlx_array_eval(cps[1].layers[0].aux_state);
    }

    var cold = pcEmptySsm();
    defer pcFreeQsaHybrid(&cold);
    cold[0].qsa_ratio = ratio;
    try pcQsaFeed(&xfm, &cold[0], keys, @intCast(B), s);

    var from_full = pcEmptySsm();
    defer pcFreeQsaHybrid(&from_full);
    try transformer_mod.restoreSsmCheckpoint(&from_full, &cps[0]);
    transformer_mod.applyQsaHistoryAt(&from_full, &cps[1], B, s, false) catch |err| {
        std.debug.print("sliced-apply residue fail K=full field=apply {s}\n", .{@errorName(err)});
        return err;
    };
    pcQsaExpectViewEqual(s, &from_full[0], &cold[0], @intCast(B)) catch {
        std.debug.print("sliced-apply residue fail K=full field=cold_view hist={d}/{d} blocks={d}\n", .{
            from_full[0].qsa_hist_rows,
            from_full[0].qsa_key_rows,
            from_full[0].qsa_pooled_blocks,
        });
        return error.TestExpectedEqual;
    };

    var K: usize = B + 1;
    while (K < H) : (K += 1) {
        var sliced = try transformer_mod.shareSsmCheckpoint(testing.allocator, &cps[0]);
        defer sliced.deinit(testing.allocator);
        sliced.pos = K;
        try transformer_mod.sliceQsaHistoryOntoCheckpoint(&sliced, &cps[1], K, s);

        var from_sliced = pcEmptySsm();
        defer pcFreeQsaHybrid(&from_sliced);
        try transformer_mod.restoreSsmCheckpoint(&from_sliced, &cps[0]);
        transformer_mod.applyQsaHistoryAt(&from_sliced, &sliced, B, s, false) catch |err| {
            std.debug.print("sliced-apply residue fail K={d} field=apply {s}\n", .{ K, @errorName(err) });
            return err;
        };

        const field = try pcSsmDiffField(s, &from_full, &from_sliced);
        if (field.len != 0) {
            std.debug.print("sliced-apply residue fail K={d} field={s} full_hist={d}/{d} sliced_hist={d}/{d} full_blocks={d} sliced_blocks={d}\n", .{
                K,
                field,
                from_full[0].qsa_hist_rows,
                from_full[0].qsa_key_rows,
                from_sliced[0].qsa_hist_rows,
                from_sliced[0].qsa_key_rows,
                from_full[0].qsa_pooled_blocks,
                from_sliced[0].qsa_pooled_blocks,
            });
            return error.TestExpectedEqual;
        }
        pcQsaExpectViewEqual(s, &from_sliced[0], &cold[0], @intCast(B)) catch {
            std.debug.print("sliced-apply residue fail K={d} field=cold_view\n", .{K});
            return error.TestExpectedEqual;
        };
    }
}

test "HotPrefixCache: heir restore at backoff matches the donor bit for bit" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const hd: c_int = 8;
    const P: usize = 48;
    const backoff: usize = 19;
    const donor_gen: usize = 16;
    const heir_gen: usize = 12;
    const a2_extra: usize = 8;
    var xfm: transformer_mod.Transformer = undefined;
    xfm.s = s;
    xfm.allocator = testing.allocator;

    const donor_len = P + donor_gen;
    const a2_len = donor_len + a2_extra;
    const heir_len = (P - 1) + heir_gen;
    const keys = try pcQsaArangeKeys(s, @intCast(a2_len), hd);
    defer _ = mlx.mlx_array_free(keys);

    var donor_toks: [P + donor_gen]u32 = undefined;
    for (&donor_toks, 0..) |*t, i| t.* = @intCast(i + 1);
    var a2_toks: [P + donor_gen + a2_extra]u32 = undefined;
    @memcpy(a2_toks[0..donor_toks.len], &donor_toks);
    for (a2_toks[donor_toks.len..], 0..) |*t, i| t.* = @intCast(50_000 + i);
    var heir_toks: [(P - 1) + heir_gen]u32 = undefined;
    @memcpy(heir_toks[0 .. P - 1], donor_toks[0 .. P - 1]);
    for (heir_toks[P - 1 ..], 0..) |*t, i| t.* = @intCast(90_000 + i);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.qsa_history_required = true;

    var donor_cache = try KVCache.init(testing.allocator, 3);
    defer donor_cache.deinit();
    try testFillCache(&donor_cache, s, 3, donor_toks.len);
    const donor_cps = try testing.allocator.alloc(SSMCheckpoint, 3);
    donor_cps[0] = try pcCaptureFed(testing.allocator, s, &xfm, keys, @intCast(backoff), 190.0, 19);
    donor_cps[1] = try pcCaptureFed(testing.allocator, s, &xfm, keys, @intCast(P), 480.0, 48);
    donor_cps[2] = try pcCaptureFed(testing.allocator, s, &xfm, keys, @intCast(donor_len), 640.0, 64);
    {
        var live: [3]SSMCacheEntry = .{
            .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true, .qsa_ratio = 4 },
            .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = false },
            .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = false },
        };
        defer pcFreeQsaHybrid(&live);
        try pcQsaFeed(&xfm, &live[0], keys, @intCast(donor_len), s);
        try transformer_mod.attachQsaHistoryToLatest(donor_cps, &live, s);
    }
    _ = try hc.commitWithState(&donor_cache, &donor_toks, false, 0, donor_cps, null, null);

    var a2_cache = try KVCache.init(testing.allocator, 3);
    defer a2_cache.deinit();
    try testFillCache(&a2_cache, s, 3, a2_toks.len);
    const a2_own = try testing.allocator.alloc(SSMCheckpoint, 1);
    a2_own[0] = try pcCaptureFed(testing.allocator, s, &xfm, keys, @intCast(a2_len), 720.0, 72);
    {
        var live: [3]SSMCacheEntry = .{
            .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true, .qsa_ratio = 4 },
            .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = false },
            .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = false },
        };
        defer pcFreeQsaHybrid(&live);
        try pcQsaFeed(&xfm, &live[0], keys, @intCast(a2_len), s);
        try transformer_mod.attachQsaHistoryToLatest(a2_own, &live, s);
    }
    _ = try hc.commitWithState(&a2_cache, &a2_toks, false, 0, a2_own, null, null);

    var donor_restored = pcEmptySsm();
    defer pcFreeQsaHybrid(&donor_restored);
    var donor_kv = try KVCache.init(testing.allocator, 3);
    defer donor_kv.deinit();
    var donor_moe: usize = 0;
    const donor_hit = try hc.lookupAndRestore(&donor_kv, &donor_moe, &donor_restored, s, a2_toks[0..P], false, 0, null, null);
    try testing.expectEqual(backoff, donor_hit.matched);

    var heir_cache = try KVCache.init(testing.allocator, 3);
    defer heir_cache.deinit();
    try testFillCache(&heir_cache, s, 3, heir_toks.len);
    const heir_own = try testing.allocator.alloc(SSMCheckpoint, 1);
    heir_own[0] = try pcCaptureFed(testing.allocator, s, &xfm, keys, 44, 440.0, 44);
    _ = try hc.commitWithState(&heir_cache, &heir_toks, false, 0, heir_own, null, null);

    var heir_restored = pcEmptySsm();
    defer pcFreeQsaHybrid(&heir_restored);
    var heir_kv = try KVCache.init(testing.allocator, 3);
    defer heir_kv.deinit();
    var heir_moe: usize = 0;
    const heir_hit = try hc.lookupAndRestore(&heir_kv, &heir_moe, &heir_restored, s, &heir_toks, false, 0, null, null);
    try testing.expectEqual(backoff, heir_hit.matched);

    const field = try pcSsmDiffField(s, &donor_restored, &heir_restored);
    if (field.len != 0) {
        log.warn("heir vs donor restore differed in {s} (donor_pos={d} heir_pos={d} heir_hist={d}/{d} donor_hist={d}/{d})\n", .{
            field,
            donor_hit.matched,
            heir_hit.matched,
            heir_restored[0].qsa_hist_rows,
            heir_restored[0].qsa_key_rows,
            donor_restored[0].qsa_hist_rows,
            donor_restored[0].qsa_key_rows,
        });
        return error.TestExpectedEqual;
    }
    _ = heir_len;
}

test "HotPrefixCache: a verbatim re-send restores at prompt_len-1" {
    const s = mlx.gpuStream();
    var toks: [19]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
    var live = pcBuildQsaHybrid(s, 18, 100.0);
    defer pcFreeQsaHybrid(&live);
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.qsa_history_required = true;
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try testFillCache(&cache, s, 3, toks.len);
    const cps = try testing.allocator.alloc(SSMCheckpoint, 2);
    cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 8, s);
    cps[1] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, toks.len - 1, s);
    try transformer_mod.attachQsaHistoryToLatest(cps, &live, s);
    _ = try hc.commitWithState(&cache, &toks, false, 0, cps, null, null);

    var target_cache = try KVCache.init(testing.allocator, 3);
    defer target_cache.deinit();
    var target = pcEmptySsm();
    defer pcFreeQsaHybrid(&target);
    var moe_off: usize = 0;
    const r = try hc.lookupAndRestore(&target_cache, &moe_off, &target, s, &toks, false, 0, null, null);
    try testing.expectEqual(toks.len - 1, r.matched);
    try testing.expect(!r.full_match);
    try testing.expect(transformer_mod.qsaRestoreSatisfiesForward(&target, r.matched));
}

test "HotPrefixCache: a hybrid full-prefix extend checks out under SSD-first" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    restore_move_override = true;
    defer restore_move_override = null;
    var toks: [10]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
    var prompt: [12]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(i + 1);
    var live = pcBuildQsaHybrid(s, 8, 100.0);
    defer pcFreeQsaHybrid(&live);
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.ssd_first = true;
    defer hc.deinit();
    hc.qsa_history_required = true;
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try testFillCache(&cache, s, 3, toks.len);
    const cps = try testing.allocator.alloc(SSMCheckpoint, 1);
    cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 8, s);
    try transformer_mod.attachQsaHistoryToLatest(cps, &live, s);
    _ = try hc.commitWithState(&cache, &toks, false, 0, cps, null, null);

    var slot = try KVCache.init(testing.allocator, 3);
    defer slot.deinit();
    var target = pcEmptySsm();
    defer pcFreeQsaHybrid(&target);
    var moe_off: usize = 0;
    const r = try hc.lookupAndRestoreForSlot(&slot, &moe_off, &target, s, &prompt, false, 0, null, null, null, 0xA11CE);
    try testing.expect(r.matched > 0);
    try testing.expect(r.checked_out);
    try testing.expectEqual(@as(?usize, 0xA11CE), hc.entries.items[0].checked_out_by);
}

test "HotPrefixCache: walk-down restores the higher covering checkpoint, not a lower one" {
    const s = mlx.gpuStream();
    var toks: [20]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
    var live = pcBuildQsaHybrid(s, 16, 100.0);
    defer pcFreeQsaHybrid(&live);
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.qsa_history_required = true;
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try testFillCache(&cache, s, 3, toks.len);
    const cps = try testing.allocator.alloc(SSMCheckpoint, 2);
    cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 8, s);
    cps[1] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 16, s);
    try transformer_mod.attachQsaHistoryToLatest(cps, &live, s);
    _ = try hc.commitWithState(&cache, &toks, false, 0, cps, null, null);

    var target_cache = try KVCache.init(testing.allocator, 3);
    defer target_cache.deinit();
    var target = pcEmptySsm();
    defer pcFreeQsaHybrid(&target);
    var moe_off: usize = 0;
    const r = try hc.lookupAndRestore(&target_cache, &moe_off, &target, s, &toks, false, 0, null, null);
    try testing.expectEqual(@as(usize, 16), r.matched);
    try testing.expect(target[0].qsa_pooled.ctx != null);
}

test "HotPrefixCache: skip set on the cache without a lookup does not poison the next request" {
    const s = mlx.gpuStream();
    const tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.qsa_history_required = true;
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try testFillCache(&cache, s, 3, tokens.len);
    var live = pcBuildQsaHybrid(s, 10, 100.0);
    defer pcFreeQsaHybrid(&live);
    const cps = try testing.allocator.alloc(SSMCheckpoint, 1);
    cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 8, s);
    try transformer_mod.attachQsaHistoryToLatest(cps, &live, s);
    _ = try hc.commitWithState(&cache, &tokens, false, 0, cps, null, null);

    var target_cache = try KVCache.init(testing.allocator, 3);
    defer target_cache.deinit();
    var target = pcEmptySsm();
    defer pcFreeQsaHybrid(&target);
    var moe_off: usize = 0;
    const skipped = try hc.lookupAndRestoreWithMedia(&target_cache, &moe_off, &target, s, &tokens, false, 0, null, &.{}, null, null, null, true);
    try testing.expectEqual(@as(usize, 0), skipped.matched);
    var cache2 = try KVCache.init(testing.allocator, 3);
    defer cache2.deinit();
    var target2 = pcEmptySsm();
    defer pcFreeQsaHybrid(&target2);
    var moe2: usize = 0;
    const leaked = try hc.lookupAndRestoreWithMedia(&cache2, &moe2, &target2, s, &tokens, false, 0, null, &.{}, null, null, null, false);
    try testing.expect(leaked.matched > 0);
}

test "HotPrefixCache: skip_prefix_cache lookup is cold even when a covering entry exists" {
    const s = mlx.gpuStream();
    const tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.qsa_history_required = true;
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try testFillCache(&cache, s, 3, tokens.len);
    var live = pcBuildQsaHybrid(s, 10, 100.0);
    defer pcFreeQsaHybrid(&live);
    const cps = try testing.allocator.alloc(SSMCheckpoint, 1);
    cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 8, s);
    try transformer_mod.attachQsaHistoryToLatest(cps, &live, s);
    _ = try hc.commitWithState(&cache, &tokens, false, 0, cps, null, null);

    var target_cache = try KVCache.init(testing.allocator, 3);
    defer target_cache.deinit();
    var target = pcEmptySsm();
    defer pcFreeQsaHybrid(&target);
    var moe_off: usize = 0;
    const skipped = try hc.lookupAndRestoreWithMedia(&target_cache, &moe_off, &target, s, &tokens, false, 0, null, &.{}, null, null, null, true);
    try testing.expectEqual(@as(usize, 0), skipped.matched);

    var target2 = pcEmptySsm();
    defer pcFreeQsaHybrid(&target2);
    var moe2: usize = 0;
    var cache2 = try KVCache.init(testing.allocator, 3);
    defer cache2.deinit();
    const second = try hc.lookupAndRestore(&cache2, &moe2, &target2, s, &tokens, false, 0, null, null);
    try testing.expect(second.matched > 0);
}

test "HotPrefixCache: a skipped lookup of a covering prompt matches a cold cache" {
    const s = mlx.gpuStream();
    const tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.qsa_history_required = true;
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try testFillCache(&cache, s, 3, tokens.len);
    var live = pcBuildQsaHybrid(s, 10, 100.0);
    defer pcFreeQsaHybrid(&live);
    const cps = try testing.allocator.alloc(SSMCheckpoint, 1);
    cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 8, s);
    try transformer_mod.attachQsaHistoryToLatest(cps, &live, s);
    _ = try hc.commitWithState(&cache, &tokens, false, 0, cps, null, null);

    var skipped_cache = try KVCache.init(testing.allocator, 3);
    defer skipped_cache.deinit();
    var skipped_ssm = pcEmptySsm();
    defer pcFreeQsaHybrid(&skipped_ssm);
    var skip_off: usize = 7;
    const skipped = try hc.lookupAndRestoreWithMedia(&skipped_cache, &skip_off, &skipped_ssm, s, &tokens, false, 0, null, &.{}, null, null, null, true);

    var cold = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer cold.deinit();
    cold.qsa_history_required = true;
    var cold_cache = try KVCache.init(testing.allocator, 3);
    defer cold_cache.deinit();
    var cold_ssm = pcEmptySsm();
    defer pcFreeQsaHybrid(&cold_ssm);
    var cold_off: usize = 7;
    const cold_r = try cold.lookupAndRestore(&cold_cache, &cold_off, &cold_ssm, s, &tokens, false, 0, null, null);
    try testing.expectEqual(cold_r.matched, skipped.matched);
    try testing.expectEqual(cold_off, skip_off);
    try testing.expectEqual(@as(usize, 0), skipped.matched);
}

test "HotPrefixCache: dropping a disk hybrid restore poisons that disk entry" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const base = buf[0..root_len];

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-qsa-poison", 0, 128);
        defer hc.deinit();
        var cache = try KVCache.init(testing.allocator, 3);
        defer cache.deinit();
        try testFillCache(&cache, s, 3, 600);
        var src256 = pcBuildHybrid(s, 100.0, 500.0);
        defer pcFreeHybrid(&src256);
        var src512 = pcBuildHybrid(s, 300.0, 700.0);
        defer pcFreeHybrid(&src512);
        const cps = try testing.allocator.alloc(SSMCheckpoint, 2);
        cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &src256, 256, s);
        cps[1] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &src512, 512, s);
        _ = try hc.commitWithSsm(&cache, &tokens, false, cps, null, null);
        hc.flushPendingDisk(s);
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    }

    var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-qsa-poison", 0, 128);
    defer hc2.deinit();
    try testing.expectEqual(@as(usize, 0), hc2.entryCount());
    try testing.expectEqual(@as(usize, 1), hc2.disk.?.entryCount());
    const disk_id = hc2.disk.?.entries.items[0].id;

    var cache2 = try KVCache.init(testing.allocator, 3);
    defer cache2.deinit();
    var ssm2 = pcEmptySsm();
    defer pcFreeHybrid(&ssm2);
    var moe_off: usize = 0;
    const res = try hc2.lookupAndRestore(&cache2, &moe_off, &ssm2, s, &tokens, false, 0, null, null);
    try testing.expect(res.matched > 0);
    try testing.expect(hc2.dropLastRestored());
    try testing.expect(hc2.disk.?.entryPoisoned(disk_id));

    var cache3 = try KVCache.init(testing.allocator, 3);
    defer cache3.deinit();
    var ssm3 = pcEmptySsm();
    defer pcFreeHybrid(&ssm3);
    var moe3: usize = 0;
    const res3 = try hc2.lookupAndRestore(&cache3, &moe3, &ssm3, s, &tokens, false, 0, null, null);
    try testing.expectEqual(@as(usize, 0), res3.matched);
}

test "HotPrefixCache: inherit-branch QSA invariant drop unbills the dropped SSM bytes" {
    const s = mlx.gpuStream();
    var donor_toks: [30]u32 = undefined;
    for (&donor_toks, 0..) |*t, i| t.* = @intCast(i + 1);
    var heir_toks: [19]u32 = undefined;
    @memcpy(heir_toks[0..17], donor_toks[0..17]);
    heir_toks[17] = 900;
    heir_toks[18] = 901;

    var live = pcBuildQsaHybrid(s, 17, 100.0);
    defer pcFreeQsaHybrid(&live);
    var ctrl = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer ctrl.deinit();
    var ctrl_cache = try KVCache.init(testing.allocator, 3);
    defer ctrl_cache.deinit();
    try testFillCache(&ctrl_cache, s, 3, heir_toks.len);
    _ = try ctrl.commitWithState(&ctrl_cache, &heir_toks, false, 0, null, null, null);
    const snap_only = ctrl.entries.items[0].kv_bytes;

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    var donor_cache = try KVCache.init(testing.allocator, 3);
    defer donor_cache.deinit();
    try testFillCache(&donor_cache, s, 3, donor_toks.len);
    const donor_cps = try testing.allocator.alloc(SSMCheckpoint, 1);
    donor_cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 17, s);
    for (donor_cps[0].layers) |*l| {
        if (l.qsa_pooled.ctx != null) _ = mlx.mlx_array_free(l.qsa_pooled);
        l.qsa_pooled = .{ .ctx = null };
    }
    _ = try hc.commitWithState(&donor_cache, &donor_toks, false, 0, donor_cps, null, null);

    hc.qsa_history_required = true;
    var heir_live = pcBuildQsaHybrid(s, 19, 100.0);
    defer pcFreeQsaHybrid(&heir_live);
    var heir_cache = try KVCache.init(testing.allocator, 3);
    defer heir_cache.deinit();
    try testFillCache(&heir_cache, s, 3, heir_toks.len);
    const heir_own = try testing.allocator.alloc(SSMCheckpoint, 1);
    heir_own[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &heir_live, 18, s);
    for (heir_own[0].layers) |*l| {
        if (l.qsa_pooled.ctx != null) _ = mlx.mlx_array_free(l.qsa_pooled);
        l.qsa_pooled = .{ .ctx = null };
    }
    _ = try hc.commitWithState(&heir_cache, &heir_toks, false, 0, heir_own, null, null);

    var found = false;
    for (hc.entries.items) |*e| {
        if (e.tokens.len != heir_toks.len) continue;
        found = true;
        try testing.expect(e.ssm_checkpoints == null);
        try testing.expectEqual(@as(u64, 0), e.ssm_bytes);
        try testing.expectEqual(snap_only, e.kv_bytes);
    }
    try testing.expect(found);
}

test "HotPrefixCache: shedding the oldest QSA bank with no lower destination drops the list" {
    const s = mlx.gpuStream();
    var toks: [30]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
    var live = pcBuildQsaHybrid(s, 8, 100.0);
    defer pcFreeQsaHybrid(&live);
    var live2 = pcBuildQsaHybrid(s, 25, 200.0);
    defer pcFreeQsaHybrid(&live2);
    var probe = try KVCache.init(testing.allocator, 3);
    defer probe.deinit();
    try testFillCache(&probe, s, 3, toks.len);
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.qsa_history_required = true;
    hc.ssm_checkpoint_max = 8;
    const cps = try testing.allocator.alloc(SSMCheckpoint, 2);
    cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 8, s);
    cps[1] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live2, 25, s);
    try transformer_mod.attachQsaHistoryToLatest(cps[0..1], &live, s);
    for (cps[1].layers) |*l| {
        if (l.qsa_pooled.ctx != null) _ = mlx.mlx_array_free(l.qsa_pooled);
        l.qsa_pooled = .{ .ctx = null };
    }
    _ = try hc.commitWithState(&probe, &toks, false, 0, cps, null, null);
    hc.max_kv_bytes = 1;
    hc.shedCheckpointsToFit();
    try testing.expect(hc.entryCount() == 1);
    try testing.expect(hc.entries.items[0].ssm_checkpoints == null);
}

test "HotPrefixCache: a QSA-required restore without indexer history is a miss even when forward-satisfy is vacuous" {
    const s = mlx.gpuStream();
    const tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try testFillCache(&cache, s, 3, tokens.len);
    var live = pcBuildHybrid(s, 123.0, 456.0);
    defer pcFreeHybrid(&live);
    const cps = try testing.allocator.alloc(SSMCheckpoint, 1);
    cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, 8, s);
    _ = try hc.commitWithState(&cache, &tokens, false, 0, cps, null, null);
    hc.qsa_history_required = true;

    var target_cache = try KVCache.init(testing.allocator, 3);
    defer target_cache.deinit();
    var target = pcEmptySsm();
    defer pcFreeHybrid(&target);
    var moe_off: usize = 0;
    const r = try hc.lookupAndRestore(&target_cache, &moe_off, &target, s, &tokens, false, 0, null, null);
    try testing.expectEqual(@as(usize, 0), r.matched);
}

test "prefix cache: a hybrid miss with a raw token match names itself" {
    // The silent null: `findBestRestorableMatch` rejects every candidate that
    // has no SSM checkpoint at or below its shared prefix, so the lookup can
    // return null with a LONG raw match behind it — and the `match == null`
    // arm logged nothing at all. `missKind` is the seam: a genuinely cold
    // cache stays quiet, an expensive miss gets a line.
    try testing.expectEqual(MissKind.cold, missKind(0, 0));
    try testing.expectEqual(MissKind.cold, missKind(0, 100_000));
    try testing.expectEqual(MissKind.cold, missKind(3, 0));
    // Below the commit floor there was never a prefix worth restoring.
    try testing.expectEqual(MissKind.cold, missKind(3, MIN_CANCELLED_COMMIT_TOKENS - 1));
    // At and above it, a cold prefill of that many tokens owes an explanation.
    try testing.expectEqual(MissKind.no_checkpoint, missKind(1, MIN_CANCELLED_COMMIT_TOKENS));
    try testing.expectEqual(MissKind.no_checkpoint, missKind(4, 393_000));
}

test "prefix cache: the no-match lookup arm consults missKind, never returns silently" {
    // Class guard for the 560 s unexplained cold prefill: every early return
    // from the lookup owes a reason. The `match == null` arm is the one that
    // had none, and it is reachable only through this file.
    const source = @embedFile("prefix_cache.zig");
    const start = std.mem.indexOf(u8, source, "if (match == null) {") orelse
        return error.MissingNoMatchArm;
    const arm = source[start .. start + 900];
    const end = std.mem.indexOf(u8, arm, "return .{ .matched = 0, .full_match = false };") orelse
        return error.MissingNoMatchReturn;
    try testing.expect(std.mem.indexOf(u8, arm[0..end], "missKind(") != null);
    try testing.expect(std.mem.indexOf(u8, arm[0..end], "[hot-cache] hybrid miss") != null);
    // The probe must be filled BEFORE the restorability filter can `continue`
    // a candidate away, or `best_raw` is always 0 and the line never fires.
    const fbr = std.mem.indexOf(u8, source, "fn findBestRestorableMatch(") orelse
        return error.MissingFinder;
    const body = source[fbr .. fbr + 2600];
    const probe_at = std.mem.indexOf(u8, body, "if (probe) |p| {") orelse return error.MissingProbe;
    const filter_at = std.mem.indexOf(u8, body, "const effective = if (require_ssm_checkpoint)") orelse
        return error.MissingFilter;
    try testing.expect(probe_at < filter_at);
}

test "prefix cache: an inherited checkpoint SHARES the donor's buffers and is budget-bounded" {
    // The two claims inheritance rests on. (1) Sharing: a clone must outlive
    // the donor — the ladder's whole point is that evicting the entry we
    // inherited from frees nothing the inheritor still needs. (2) Bounding:
    // the per-entry accounting bills shared bytes again, so an unbounded
    // inherit could book an entry past a hard cap; the clone takes the
    // HIGHEST positions that fit and stops.
    const s = mlx.gpuStream();

    var srcs: [3][3]SSMCacheEntry = undefined;
    for (&srcs, 0..) |*e, i| {
        const f: f64 = @floatFromInt(i + 1);
        e.* = pcBuildHybrid(s, 100.0 * f, 500.0 * f);
    }
    defer {
        for (&srcs) |*e| pcFreeHybrid(e);
    }

    const donor = try testing.allocator.alloc(SSMCheckpoint, 3);
    for (donor, 0..) |*c, i| {
        c.* = try transformer_mod.captureSsmCheckpoint(testing.allocator, &srcs[i], (i + 1) * 8, s);
    }
    const one = transformer_mod.ssmCheckpointBytes(&donor[0]);

    // Unbounded, limit past everything: all three, ascending.
    {
        const all = (try HotPrefixCache.cloneCheckpointsUpTo(testing.allocator, donor, 100, null)).?;
        defer {
            for (all) |*c| c.deinit(testing.allocator);
            testing.allocator.free(all);
        }
        try testing.expectEqual(@as(usize, 3), all.len);
        try testing.expectEqual(@as(usize, 8), all[0].pos);
        try testing.expectEqual(@as(usize, 24), all[2].pos);
    }

    // A position past the shared prefix describes state this prompt never
    // reached and must not be inherited.
    {
        const two = (try HotPrefixCache.cloneCheckpointsUpTo(testing.allocator, donor, 16, null)).?;
        defer {
            for (two) |*c| c.deinit(testing.allocator);
            testing.allocator.free(two);
        }
        try testing.expectEqual(@as(usize, 2), two.len);
        try testing.expectEqual(@as(usize, 16), two[1].pos);
    }

    // Budget for one and a half: the HIGHEST reachable position wins.
    {
        const one_only = (try HotPrefixCache.cloneCheckpointsUpTo(testing.allocator, donor, 100, one + one / 2)).?;
        defer {
            for (one_only) |*c| c.deinit(testing.allocator);
            testing.allocator.free(one_only);
        }
        try testing.expectEqual(@as(usize, 1), one_only.len);
        try testing.expectEqual(@as(usize, 24), one_only[0].pos);
    }
    // Nothing fits: null, never an empty slice the caller must special-case.
    try testing.expectEqual(@as(?[]SSMCheckpoint, null), try HotPrefixCache.cloneCheckpointsUpTo(testing.allocator, donor, 100, 0));

    // (1) The clone outlives the donor. Free the donor list entirely, then
    // read the shared state back — a copy would be fine here too, but a
    // DANGLING handle would not, and the restore below is what the ladder's
    // next rung actually does.
    const kept = (try HotPrefixCache.cloneCheckpointsUpTo(testing.allocator, donor, 100, null)).?;
    defer {
        for (kept) |*c| c.deinit(testing.allocator);
        testing.allocator.free(kept);
    }
    for (donor) |*c| c.deinit(testing.allocator);
    testing.allocator.free(donor);

    var dst = pcEmptySsm();
    defer pcFreeHybrid(&dst);
    try transformer_mod.restoreSsmCheckpoint(&dst, &kept[1]);
    try testing.expectEqual(@as(f32, 200.0), pcSsmVal(dst[0].conv_state, 0, s));
    try testing.expectEqual(@as(f32, 1000.0), pcSsmVal(dst[0].ssm_state, 0, s));
}

test "evictLruToAdmit: oldest first, never the entry THIS request restored, and shared bytes are not counted as freed" {
    // "Most recently used" is not "the entry this request restored", and a restored entry's
    // shared buffers return nothing to the allocator when evicted.
    const s = mlx.gpuStream();
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();

    var toks_a: [4096]u32 = undefined;
    for (&toks_a, 0..) |*t, i| t.* = @intCast(i + 1);
    var toks_b: [4096]u32 = undefined;
    for (&toks_b, 0..) |*t, i| t.* = @intCast(i + 1_000_001);
    var toks_c: [4096]u32 = undefined;
    for (&toks_c, 0..) |*t, i| t.* = @intCast(i + 2_000_001);

    inline for (.{ &toks_a, &toks_b, &toks_c }) |toks| {
        var cache = try KVCache.init(testing.allocator, 8);
        defer cache.deinit();
        try testFillCache(&cache, s, 8, 4096);
        // Materialize before committing: an unevaluated cache owns no Metal buffer at all.
        for (cache.entries) |*e| {
            if (e.keys.ctx != null) _ = mlx.mlx_array_eval(e.keys);
            if (e.values.ctx != null) _ = mlx.mlx_array_eval(e.values);
        }
        _ = try hc.commit(&cache, toks, false);
    }
    try testing.expectEqual(@as(usize, 3), hc.entryCount());
    var live_resident: usize = 0;
    _ = mlx.mlx_get_active_memory(&live_resident);
    try testing.expect(live_resident > 4 * 1024 * 1024);

    // This request restores B, the middle entry. The target cache stays alive: that is what makes B shared.
    var live_b = try KVCache.init(testing.allocator, 8);
    defer live_b.deinit();
    var moe_off: usize = 0;
    const hit = try hc.lookupAndRestore(&live_b, &moe_off, null, s, &toks_b, false, 0, null, null);
    try testing.expect(hit.full_match);
    try testing.expect(hc.last_restored_used != null);

    const Never = struct {
        fn call(ctx: ?*anyopaque) bool {
            _ = ctx;
            return false;
        }
    };
    const rep = hc.evictLruToAdmit(458_832, null, Never.call, true);
    try testing.expect(!rep.admitted);
    try testing.expectEqual(@as(usize, 2), rep.entries); // A and C, oldest first
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    var probe_cache = try KVCache.init(testing.allocator, 8);
    defer probe_cache.deinit();
    var off2: usize = 0;
    const still_b = try hc.lookupAndRestore(&probe_cache, &off2, null, s, &toks_b, false, 0, null, null);
    try testing.expect(still_b.full_match);
    // An exclusive eviction may return MORE than it was billed (capacity rounding).
    try testing.expect(rep.bytes > 0);
    try testing.expect(rep.bytes * HotPrefixCache.SHARED_RETURN_DIVISOR >= rep.accounted_bytes);
    try testing.expect(!rep.shared_stop);

    // Unprotected, B goes too, and the allocator gets ~nothing back, which the pass notices.
    const rest = hc.evictLruToAdmit(458_832, null, Never.call, false);
    try testing.expect(!rest.admitted);
    try testing.expectEqual(@as(usize, 0), hc.entryCount());
    try testing.expect(rest.accounted_bytes > 0);
    try testing.expect(rest.bytes * HotPrefixCache.SHARED_RETURN_DIVISOR < rest.accounted_bytes);
    try testing.expect(rest.shared_stop);
}

test "a trivial share of a whole session is a LIEN, and the admission pass gets the entry back" {
    // Live shape: a cold 786k prompt shared eleven tokens with the one resident 524k entry, the
    // restore shared the whole 11.5 GB into the slot, and the admission pass could not touch it.
    const t = testing;
    try t.expect(23_692 > 14_713); // refused
    try t.expect(14_713 + 11_476 >= 23_692); // admitted

    const MB: u64 = 1 << 20;
    const floor = HotPrefixCache.RESTORE_PIN_MIN_BYTES;
    try t.expect(HotPrefixCache.restoreWouldPinEntry(11_476 * MB, floor, 524_464, 11));
    try t.expect(!HotPrefixCache.restoreWouldPinEntry(11_476 * MB, floor, 524_464, 100_000));
    try t.expect(!HotPrefixCache.restoreWouldPinEntry(60 * MB, floor, 218, 11));

    const s = mlx.gpuStream();
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.ssd_first = true; // the arm this was measured on
    hc.restore_pin_min_bytes = 1 << 20; // a 2 MB entry stands in for 11.5 GB

    var session: [4096]u32 = undefined;
    for (&session, 0..) |*x, i| x.* = @intCast(i + 1);
    var cold: [4096]u32 = undefined;
    for (&cold, 0..) |*x, i| x.* = if (i < 11) @as(u32, @intCast(i + 1)) else @intCast(i + 5_000_001);

    // One cache: the slot that served the session is handed the next request.
    var slot_cache = try KVCache.init(testing.allocator, 8);
    defer slot_cache.deinit();
    try testFillCache(&slot_cache, s, 8, 4096);
    for (slot_cache.entries) |*e| {
        if (e.keys.ctx != null) _ = mlx.mlx_array_eval(e.keys);
        if (e.values.ctx != null) _ = mlx.mlx_array_eval(e.values);
    }
    _ = try hc.commit(&slot_cache, &session, false);
    try t.expectEqual(@as(usize, 1), hc.entryCount());
    try t.expect(hc.entries.items[0].kv_bytes > hc.restore_pin_min_bytes);

    var moe_off: usize = 0;
    const hit = try hc.lookupAndRestore(&slot_cache, &moe_off, null, s, &cold, false, 0, null, null);

    // Declined: eleven rows are not worth a lien on the session.
    try t.expectEqual(@as(usize, 0), hit.matched);
    try t.expect(!hit.full_match);
    try t.expectEqual(@as(usize, 0), moe_off);
    try t.expect(hc.last_restored_used == null);
    try t.expectEqual(@as(usize, 0), slot_cache.step);

    // ...so the admission pass reclaims it, and reclaims real bytes.
    const Fits = struct {
        fn call(ctx: ?*anyopaque) bool {
            const cache: *HotPrefixCache = @ptrCast(@alignCast(ctx.?));
            return cache.entryCount() == 0;
        }
    };
    const rep = hc.evictLruToAdmit(786_369, &hc, Fits.call, true);
    try t.expect(rep.admitted);
    try t.expectEqual(@as(usize, 1), rep.entries);
    try t.expect(rep.bytes > 0);
    try t.expect(!rep.shared_stop);
}

test "a PROPORTIONATE share still restores, and is still protected" {
    // The inverse: an entry a request genuinely continues hands back most of what it pins.
    const t = testing;
    const s = mlx.gpuStream();
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.ssd_first = true;
    hc.restore_pin_min_bytes = 1 << 20;

    var session: [4096]u32 = undefined;
    for (&session, 0..) |*x, i| x.* = @intCast(i + 1);
    var warm: [4096]u32 = undefined;
    for (&warm, 0..) |*x, i| x.* = if (i < 2048) @as(u32, @intCast(i + 1)) else @intCast(i + 5_000_001);

    var src_cache = try KVCache.init(testing.allocator, 8);
    defer src_cache.deinit();
    try testFillCache(&src_cache, s, 8, 4096);
    for (src_cache.entries) |*e| {
        if (e.keys.ctx != null) _ = mlx.mlx_array_eval(e.keys);
        if (e.values.ctx != null) _ = mlx.mlx_array_eval(e.values);
    }
    _ = try hc.commit(&src_cache, &session, false);

    var slot_cache = try KVCache.init(testing.allocator, 8);
    defer slot_cache.deinit();
    var moe_off: usize = 0;
    const hit = try hc.lookupAndRestore(&slot_cache, &moe_off, null, s, &warm, false, 0, null, null);
    try t.expectEqual(@as(usize, 2048), hit.matched);
    try t.expect(hc.last_restored_used != null);

    const Never = struct {
        fn call(ctx: ?*anyopaque) bool {
            _ = ctx;
            return false;
        }
    };
    const rep = hc.evictLruToAdmit(786_369, null, Never.call, true);
    try t.expectEqual(@as(usize, 0), rep.entries);
    try t.expectEqual(@as(usize, 1), hc.entryCount());

    // The rule is SSD-first's. Off it, even the lien restores.
    var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc2.deinit();
    hc2.restore_pin_min_bytes = 1 << 20;
    var src2 = try KVCache.init(testing.allocator, 8);
    defer src2.deinit();
    try testFillCache(&src2, s, 8, 4096);
    _ = try hc2.commit(&src2, &session, false);
    var cold: [4096]u32 = undefined;
    for (&cold, 0..) |*x, i| x.* = if (i < 11) @as(u32, @intCast(i + 1)) else @intCast(i + 6_000_001);
    var slot2 = try KVCache.init(testing.allocator, 8);
    defer slot2.deinit();
    var off2: usize = 0;
    const hit2 = try hc2.lookupAndRestore(&slot2, &off2, null, s, &cold, false, 0, null, null);
    try t.expectEqual(@as(usize, 11), hit2.matched);
}

test "a 0-token outcome is not a restore: no LRU bump, no protection, and the entry stays evictable" {
    // A 0-token restore outcome used to leave `last_restored_used` set, shielding a fully
    // reclaimable entry from the admission pass trying to admit the request its miss condemned.
    const t = testing;
    const s = mlx.gpuStream();
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.ssd_first = true;
    // The qwen4_exp arm: the indexer history travels with the KV or the
    // restore is a miss (`MtpHeadQsaHistoryGap`).
    hc.qsa_history_required = true;

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*x, i| x.* = @intCast(i + 7);

    // A DECOY, committed first so it is the LRU victim the pass should take.
    var decoy_ids: [600]u32 = undefined;
    for (&decoy_ids, 0..) |*x, i| x.* = @intCast(i + 900_007);
    var decoy_cache = try KVCache.init(testing.allocator, 3);
    defer decoy_cache.deinit();
    try testFillCache(&decoy_cache, s, 3, 600);
    _ = try hc.commit(&decoy_cache, &decoy_ids, false);

    // The hybrid entry: KV plus one SSM checkpoint, and NO QSA history on it.
    var src = try KVCache.init(testing.allocator, 3);
    defer src.deinit();
    try testFillCache(&src, s, 3, 600);
    var src512 = pcBuildHybrid(s, 300.0, 700.0);
    defer pcFreeHybrid(&src512);
    const cps = try testing.allocator.alloc(SSMCheckpoint, 1);
    cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &src512, 512, s);
    _ = try hc.commitWithSsm(&src, &tokens, false, cps, null, null);
    try t.expectEqual(@as(usize, 2), hc.entryCount());
    const hybrid_idx: usize = 1;
    const used_before = hc.entries.items[hybrid_idx].last_used;

    var slot_cache = try KVCache.init(testing.allocator, 3);
    defer slot_cache.deinit();
    var ssm = pcEmptySsm();
    defer pcFreeHybrid(&ssm);
    var moe_off: usize = 0;
    const hit = try hc.lookupAndRestoreWithMedia(&slot_cache, &moe_off, &ssm, s, &tokens, false, 0, null, &.{}, null, null, 0xF5, false);

    // The outcome: nothing delivered.
    try t.expectEqual(@as(usize, 0), hit.matched);
    try t.expect(!hit.full_match);
    try t.expect(!hit.checked_out);
    try t.expectEqual(@as(usize, 0), moe_off);
    try t.expectEqual(@as(usize, 0), slot_cache.step);

    // ...therefore nothing to protect, nothing to promote, and nothing
    // checked out to another slot.
    try t.expect(hc.last_restored_used == null);
    try t.expectEqual(used_before, hc.entries.items[hybrid_idx].last_used);
    try t.expectEqual(@as(?usize, null), hc.entries.items[hybrid_idx].checked_out_by);

    // The admission pass takes the decoy first, then the entry that delivered nothing.
    const Fits = struct {
        fn call(ctx: ?*anyopaque) bool {
            const cache: *HotPrefixCache = @ptrCast(@alignCast(ctx.?));
            return cache.entryCount() == 0;
        }
    };
    const rep = hc.evictLruToAdmit(600_000, &hc, Fits.call, true);
    try t.expect(rep.admitted);
    try t.expectEqual(@as(usize, 2), rep.entries);
}

test "off SSD-first too, a 0-token hybrid outcome keeps its entry's recency: the count cap takes it, not another session" {
    const t = testing;
    const s = mlx.gpuStream();
    var hc = HotPrefixCache.initWithMem(testing.allocator, 2, 0);
    defer hc.deinit();
    try t.expect(!hc.ssd_first);

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*x, i| x.* = @intCast(i + 7);
    // The entry the lookup lands on, committed first: a checkpoint but no QSA history, so on a
    // QSA arch it matches and then restores nothing.
    var src = try KVCache.init(testing.allocator, 3);
    defer src.deinit();
    try testFillCache(&src, s, 3, 600);
    var src512 = pcBuildHybrid(s, 300.0, 700.0);
    defer pcFreeHybrid(&src512);
    const cps = try testing.allocator.alloc(SSMCheckpoint, 1);
    cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &src512, 512, s);
    _ = try hc.commitWithSsm(&src, &tokens, false, cps, null, null);
    // Another session's entry, committed after it.
    var other_ids: [600]u32 = undefined;
    for (&other_ids, 0..) |*x, i| x.* = @intCast(i + 900_007);
    var other = try KVCache.init(testing.allocator, 3);
    defer other.deinit();
    try testFillCache(&other, s, 3, 600);
    _ = try hc.commit(&other, &other_ids, false);
    hc.qsa_history_required = true;
    const used_before = hc.entries.items[0].last_used;

    var slot_cache = try KVCache.init(testing.allocator, 3);
    defer slot_cache.deinit();
    var ssm = pcEmptySsm();
    defer pcFreeHybrid(&ssm);
    var moe_off: usize = 0;
    const miss = try hc.lookupAndRestoreWithMedia(&slot_cache, &moe_off, &ssm, s, &tokens, false, 0, null, &.{}, null, null, 0xF5, false);
    try t.expectEqual(@as(usize, 0), miss.matched);
    try t.expectEqual(used_before, hc.entries.items[0].last_used);

    var third_ids: [600]u32 = undefined;
    for (&third_ids, 0..) |*x, i| x.* = @intCast(i + 500_007);
    var third = try KVCache.init(testing.allocator, 3);
    defer third.deinit();
    try testFillCache(&third, s, 3, 600);
    _ = try hc.commit(&third, &third_ids, false);
    try t.expectEqual(@as(usize, 2), hc.entryCount());
    for (hc.entries.items) |e| try t.expect(e.tokens[0] != tokens[0]);
}

test "the lien weighs the share a restore DELIVERS, not the one it matched" {
    // `findBestRestorableMatch` returns the raw token match but the restore clamps to the highest
    // checkpoint, so `restoreWouldPinEntry` weighing `shared` let a 100k match that delivers 1,024
    // rows pin 11.5 GB.
    const t = testing;

    // The decision, pure. Pure attention delivers what it matched.
    try t.expectEqual(@as(usize, 100_000), HotPrefixCache.deliverableShare(null, false, 100_000));
    // A hybrid delivers its checkpoint, and nothing without one. (One
    // null-handle layer: `highestCheckpointAtOrBelow` skips a zero-layer stub.)
    var pure_layers = [_]transformer_mod.SSMCacheEntrySnapshot{.{
        .conv_state = .{ .ctx = null },
        .ssm_state = .{ .ctx = null },
        .initialized = false,
    }};
    var cps_pure = [_]SSMCheckpoint{.{ .pos = 1024, .layers = &pure_layers }};
    try t.expectEqual(@as(usize, 1024), HotPrefixCache.deliverableShare(&cps_pure, true, 100_000));
    try t.expectEqual(@as(usize, 0), HotPrefixCache.deliverableShare(&cps_pure, true, 512));
    try t.expectEqual(@as(usize, 0), HotPrefixCache.deliverableShare(null, true, 100_000));

    // ...and the lien test at the live numbers, both ways round.
    const MB: u64 = 1 << 20;
    const floor = HotPrefixCache.RESTORE_PIN_MIN_BYTES;
    // The raw share alone acquits it...
    try t.expect(!HotPrefixCache.restoreWouldPinEntry(11_476 * MB, floor, 524_464, 100_000));
    // ...the deliverable share convicts it.
    try t.expect(HotPrefixCache.restoreWouldPinEntry(11_476 * MB, floor, 524_464, HotPrefixCache.deliverableShare(&cps_pure, true, 100_000)));

    // One hybrid entry, a prompt sharing 400 of 600 tokens, whose only checkpoint is at 8.
    const s = mlx.gpuStream();
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    hc.ssd_first = true;
    hc.restore_pin_min_bytes = 1 << 20; // a 2 MB entry stands in for 11.5 GB

    var tokens: [4096]u32 = undefined;
    for (&tokens, 0..) |*x, i| x.* = @intCast(i + 7);

    var src = try KVCache.init(testing.allocator, 8);
    defer src.deinit();
    try testFillCache(&src, s, 8, 4096);
    for (src.entries) |*e| {
        if (e.keys.ctx != null) _ = mlx.mlx_array_eval(e.keys);
        if (e.values.ctx != null) _ = mlx.mlx_array_eval(e.values);
    }
    var src8 = pcBuildHybrid(s, 300.0, 700.0);
    defer pcFreeHybrid(&src8);
    const cps = try testing.allocator.alloc(SSMCheckpoint, 1);
    cps[0] = try transformer_mod.captureSsmCheckpoint(testing.allocator, &src8, 8, s);
    _ = try hc.commitWithSsm(&src, &tokens, false, cps, null, null);
    try t.expectEqual(@as(usize, 1), hc.entryCount());
    try t.expect(hc.entries.items[0].kv_bytes > hc.restore_pin_min_bytes);
    const used_before = hc.entries.items[0].last_used;

    // 2048 raw shared rows of a 4096-row entry — a ratio of 2, far inside
    // RESTORE_PIN_RATIO — but the only checkpoint sits at 8.
    var diverged: [4096]u32 = undefined;
    for (&diverged, 0..) |*x, i| x.* = if (i < 2048) tokens[i] else @intCast(i + 800_000);

    var slot_cache = try KVCache.init(testing.allocator, 8);
    defer slot_cache.deinit();
    var ssm = pcEmptySsm();
    defer pcFreeHybrid(&ssm);
    var moe_off: usize = 0;
    const hit = try hc.lookupAndRestore(&slot_cache, &moe_off, &ssm, s, &diverged, false, 0, null, null);

    // Declined at the gate, before the bump — eight rows are not worth a lien
    // on the session.
    try t.expectEqual(@as(usize, 0), hit.matched);
    try t.expectEqual(@as(usize, 0), moe_off);
    try t.expectEqual(@as(usize, 0), slot_cache.step);
    try t.expect(hc.last_restored_used == null);
    try t.expectEqual(used_before, hc.entries.items[0].last_used);

    // ...and the admission pass gets real bytes back for it.
    const Fits = struct {
        fn call(ctx: ?*anyopaque) bool {
            const cache: *HotPrefixCache = @ptrCast(@alignCast(ctx.?));
            return cache.entryCount() == 0;
        }
    };
    const rep = hc.evictLruToAdmit(600_000, &hc, Fits.call, true);
    try t.expect(rep.admitted);
    try t.expectEqual(@as(usize, 1), rep.entries);
    try t.expect(rep.bytes > 0);
}
test "the admission pass weighs an eviction after the GPU work still reading the entry drains" {
    // A command buffer in flight keeps its inputs' buffers until it completes: read before the
    // stream drains, the eviction returns nothing live and the pass stops as if it were shared.
    const t = testing;
    const s = mlx.gpuStream();
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();

    var tokens: [4096]u32 = undefined;
    for (&tokens, 0..) |*x, i| x.* = @intCast(i + 7);
    var src = try KVCache.init(testing.allocator, 8);
    var src_live = true;
    defer if (src_live) src.deinit();
    try testFillCache(&src, s, 8, 4096);
    _ = try hc.commit(&src, &tokens, false);
    try t.expect(hc.entries.items[0].kv_bytes >= HotPrefixCache.SHARED_RATIO_MIN_BYTES);

    // A matmul chain queued ahead of a read of every buffer the entry shares.
    const pending = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(pending);
    const dim = [_]c_int{ 1024, 1024 };
    var chain = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(chain);
    try mlx.check(mlx.mlx_ones(&chain, &dim, 2, .float32, s));
    for (0..64) |_| {
        var next = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_matmul(&next, chain, chain, s));
        _ = mlx.mlx_array_free(chain);
        chain = next;
    }
    _ = mlx.mlx_vector_array_append_value(pending, chain);
    for (src.entries) |*e| for ([_]mlx.mlx_array{ e.keys, e.values }) |buf| {
        if (buf.ctx == null) continue;
        var total = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(total);
        try mlx.check(mlx.mlx_sum(&total, buf, false, s));
        _ = mlx.mlx_vector_array_append_value(pending, total);
    };
    _ = mlx.mlx_async_eval(pending);
    // The entry is now the buffers' only owner.
    src.deinit();
    src_live = false;

    const Fits = struct {
        fn call(ctx: ?*anyopaque) bool {
            const cache: *HotPrefixCache = @ptrCast(@alignCast(ctx.?));
            return cache.entryCount() == 0;
        }
    };
    const rep = hc.evictLruToAdmit(600_000, &hc, Fits.call, true);
    try t.expect(rep.admitted);
    try t.expect(!rep.shared_stop);
    try t.expect(rep.bytes * HotPrefixCache.SHARED_RETURN_DIVISOR >= rep.accounted_bytes);
}
test "spec adopt: a qwen4 head target declines a payload with no QSA half; KV-only targets are unaffected" {
    // The qwen4_exp head's KV is meaningless without its index-key history, so the two halves
    // adopt together or not at all. Everything else is KV-only and keeps the old rule.
    const Tag = std.meta.Tag(SpecAdopt);
    const Plan = struct {
        fn tag(p: SpecAdopt) Tag {
            return std.meta.activeTag(p);
        }
        fn len(p: SpecAdopt) usize {
            return switch (p) {
                .kv_only, .head => |w| w,
                else => std.math.maxInt(usize),
            };
        }
    };
    try testing.expectEqual(Tag.kv_only, Plan.tag(specAdoptPlan(10, 40, 31, false, false)));
    try testing.expectEqual(@as(usize, 21), Plan.len(specAdoptPlan(10, 40, 31, false, false)));
    try testing.expectEqual(Tag.skip, Plan.tag(specAdoptPlan(40, 40, 31, false, false))); // starts past the reuse
    try testing.expectEqual(Tag.skip, Plan.tag(specAdoptPlan(0, 20, 31, false, false))); // ends short of it
    try testing.expectEqual(@as(usize, 31), Plan.len(specAdoptPlan(0, 31, 31, false, false))); // exact

    try testing.expectEqual(Tag.head, Plan.tag(specAdoptPlan(10, 40, 31, true, true)));
    try testing.expectEqual(@as(usize, 21), Plan.len(specAdoptPlan(10, 40, 31, true, true)));
    try testing.expectEqual(Tag.decline_head_no_history, Plan.tag(specAdoptPlan(10, 40, 31, true, false)));
    // A payload the trunk cannot use is skipped before the aux question.
    try testing.expectEqual(Tag.skip, Plan.tag(specAdoptPlan(40, 40, 31, true, false)));
    try testing.expectEqual(Tag.skip, Plan.tag(specAdoptPlan(0, 20, 31, true, false)));
}

test "qwen4 MTP head persist: the head's row count IS its cache step, so a committed history adopts" {
    // `KVCache.update` advances `step` only at layer 0, so `qwen4MtpAdvance` sets the head's;
    // the commit snapshots it and `qwen4MtpAdopt` demands the key history be exactly that long.
    const s = mlx.gpuStream();
    const head_layer: u32 = 3; // stands in for `num_hidden_layers`
    var kv = try KVCache.init(testing.allocator, head_layer + 1);
    defer kv.deinit();
    var seq_offset: usize = 0;
    try testFillHeadCache(&kv, s, head_layer, 100, &seq_offset);
    try testing.expectEqual(@as(usize, 100), seq_offset);
    try testing.expectEqual(seq_offset, kv.step);

    var snap = DflashSnap{ .snapshot = try kv.snapshot(), .base_pos = 0 };
    defer snap.deinit();
    try testing.expectEqual(seq_offset, snap.snapshot.step);

    const Tag = std.meta.Tag(SpecAdopt);
    const plan = specAdoptPlan(snap.base_pos, snap.snapshot.step, seq_offset, true, true);
    try testing.expectEqual(Tag.head, std.meta.activeTag(plan));
    try testing.expectEqual(seq_offset, plan.head);
    try testing.expectEqual(@as(usize, 100), snap.snapshot.step);

    // The shape the bug had: the step `KVCache.update` left at a non-zero layer is not adoptable.
    try testing.expectEqual(Tag.skip, std.meta.activeTag(specAdoptPlan(0, 0, seq_offset, true, true)));
}

test "spec snap bytes: the qwen4 head's QSA half is billed into the entry" {
    const s = mlx.gpuStream();
    // Head-shaped: one layer at the head's own index, never layer 0.
    const head_layer: u32 = 3;
    var kv = try KVCache.init(testing.allocator, head_layer + 1);
    defer kv.deinit();
    var head_rows: usize = 0;
    try testFillHeadCache(&kv, s, head_layer, 16, &head_rows);
    try testing.expectEqual(head_rows, kv.step);
    var snap = DflashSnap{ .snapshot = try kv.snapshot(), .base_pos = 0 };
    defer snap.deinit();
    const kv_only = HotPrefixCache.specSnapBytes(&snap);
    try testing.expect(kv_only > 0);

    var entry: SSMCacheEntry = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = true };
    defer transformer_mod.ssmFreeQsaState(&entry);
    defer _ = mlx.mlx_array_free(entry.conv_state);
    defer _ = mlx.mlx_array_free(entry.ssm_state);
    const shape = [_]c_int{ 1, 16, 128 };
    entry.aux_state = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_zeros(&entry.aux_state, &shape, 3, .bfloat16, s));
    entry.qsa_ratio = 4;
    snap.head_aux = transformer_mod.ssmSnapshot(&entry);
    const with_head = HotPrefixCache.specSnapBytes(&snap);
    try testing.expectEqual(kv_only + 16 * 128 * 2, with_head);
}

test "SSD-first: the disk flush carries the full prefix while RAM keeps a trim" {
    const io = std.testing.io;
    const s = mlx.gpuStream();

    var tokens: [1200]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 1200);
    var probe = try cache.snapshot();
    const row_bytes = HotPrefixCache.snapshotRowBytes(&probe);
    probe.deinit();
    try testing.expect(row_bytes > 0);
    const budget: u64 = row_bytes * 768;

    // Arm A (ssd_first on): RAM trims, the disk entry covers the full prompt.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);

        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, budget);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-ssd-on", 0, 128);
        defer hc.deinit();

        _ = try hc.commit(&cache, &tokens, false);
        try testing.expectEqual(@as(usize, 1), hc.entries.items.len);
        try testing.expect(hc.entries.items[0].tokens.len < tokens.len);
        hc.flushPendingDisk(s);
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
        try testing.expectEqual(@as(u32, tokens.len), hc.disk.?.entries.items[0].kv_len);
        try testing.expectEqual(@as(usize, tokens.len), hc.disk.?.entries.items[0].tokens.len);
        try testing.expect(hc.pending_disk == null);
    }

    // Arm B (ssd_first off): the disk copy is exactly what RAM retained.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);

        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, budget);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-ssd-off", 0, 128);
        defer hc.deinit();

        _ = try hc.commit(&cache, &tokens, false);
        hc.flushPendingDisk(s);
        try testing.expect(hc.pending_disk == null);
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
        try testing.expect(hc.disk.?.entries.items[0].kv_len < tokens.len);
    }
}

test "SSD-first companion: a restore adopts the entry's buffer when its capacity suffices" {
    // A grow is not in place, so a restore must land in the donor's buffer (which already
    // carries the previous turn's reservation) rather than allocate the entry's whole size.
    const s = mlx.gpuStream();
    const Grows = &transformer_mod.KVCache.kv_cap_buf_grows;

    // Turn 1: a reserved cache grows once, to the reservation.
    var donor = try KVCache.init(testing.allocator, 1);
    defer donor.deinit();
    donor.reserve(4096);
    const g0 = Grows.*;
    try testFillCache(&donor, s, 1, 600);
    try testing.expectEqual(@as(usize, 1), Grows.* - g0);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.ssd_first = true;
    defer hc.deinit();
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    _ = try hc.commit(&donor, &tokens, false);

    // Turn 2: restore into a fresh slot cache that reserves the same length: nothing allocates.
    var slot = try KVCache.init(testing.allocator, 1);
    defer slot.deinit();
    var moe_off: usize = 0;
    const res = try hc.lookupAndRestore(&slot, &moe_off, null, s, &tokens, false, 0, null, null);
    try testing.expect(res.full_match);
    slot.reserve(4096);
    const g1 = Grows.*;
    try testFillCache(&slot, s, 1, 8); // the diverged tail
    try testing.expectEqual(@as(usize, 0), Grows.* - g1);

    // A reservation is not retroactive: reserving more than the donor holds still allocates nothing.
    var slot2 = try KVCache.init(testing.allocator, 1);
    defer slot2.deinit();
    var moe_off2: usize = 0;
    _ = try hc.lookupAndRestore(&slot2, &moe_off2, null, s, &tokens, false, 0, null, null);
    slot2.reserve(65536);
    const g2 = Grows.*;
    try testFillCache(&slot2, s, 1, 8);
    try testing.expectEqual(@as(usize, 0), Grows.* - g2);

    // Negative arm: writing past the donor's capacity does grow, exactly once.
    const g3 = Grows.*;
    try testFillCache(&slot2, s, 1, 4096);
    try testing.expect(Grows.* - g3 >= 1);
}

test "SSD-first: an idle entry spills to disk and leaves RAM; the active session stays" {
    // RAM floors at one entry under SSD-first: everything but the active session goes to the
    // SSD, and only once its copy is complete.
    const io = std.testing.io;
    const s = mlx.gpuStream();

    var tokens_a: [600]u32 = undefined;
    for (&tokens_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tokens_b: [600]u32 = undefined;
    for (&tokens_b, 0..) |*t, i| t.* = @intCast(i + 90_000);

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);

    // Arm A: SSD-first spills the idle session.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);

        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-spill", 0, 128);
        defer hc.deinit();

        _ = try hc.commit(&cache, &tokens_a, false);
        hc.flushPendingDisk(s);
        _ = try hc.commit(&cache, &tokens_b, false);
        hc.flushPendingDisk(s);
        try testing.expectEqual(@as(usize, 2), hc.entryCount());

        hc.spillIdleEntries(s);
        try testing.expectEqual(@as(usize, 1), hc.entryCount());
        try testing.expectEqualSlices(u32, &tokens_b, hc.entries.items[0].tokens);
        // ...and A is still served, from disk.
        try testing.expectEqual(@as(usize, 2), hc.disk.?.entryCount());
        hc.disk.?.drainWriter();
        var back = try KVCache.init(testing.allocator, 1);
        defer back.deinit();
        var moe_off: usize = 0;
        const res = try hc.lookupAndRestore(&back, &moe_off, null, s, &tokens_a, false, 0, null, null);
        // A restore always leaves the last token to forward.
        try testing.expectEqual(@as(usize, 599), res.matched);
    }

    // Arm B: every other arch keeps both entries resident.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);

        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-nospill", 0, 128);
        defer hc.deinit();
        _ = try hc.commit(&cache, &tokens_a, false);
        hc.flushPendingDisk(s);
        _ = try hc.commit(&cache, &tokens_b, false);
        hc.flushPendingDisk(s);
        hc.spillIdleEntries(s);
        try testing.expectEqual(@as(usize, 2), hc.entryCount());
    }
}

test "SSD-first: an in-flight write does not stall the tick — the entry is re-checked next pass" {
    // The durability check must not drain the writer on the inference thread: an entry whose
    // files are still staged is not evictable on this pass, and the next pass evicts.
    const io = std.testing.io;
    const s = mlx.gpuStream();

    var tok_a: [600]u32 = undefined;
    for (&tok_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tok_b: [600]u32 = undefined;
    for (&tok_b, 0..) |*t, i| t.* = @intCast(i + 90_000);

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);

    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 8, 0);
    hc.ssd_first = true;
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-inflight", 0, 128);
    defer hc.deinit();
    hc.disk.?.ssd_first = true;
    hc.disk.?.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
    hc.disk.?.enableBackgroundWriter();
    // Generous allowance: this test is about the write state, not the cap.
    hc.ssd_idle_mem = 64 * 1024 * 1024 * 1024;

    _ = try hc.commit(&cache, &tok_a, false);
    _ = try hc.commit(&cache, &tok_b, false);

    hc.disk.?.writer.?.setPaused(true);
    defer hc.disk.?.writer.?.setPaused(false);
    hc.spillIdleEntries(s);
    // It returned, the index knows the entry, and RAM still holds it.
    try testing.expect(hc.disk.?.writer.?.pendingBytes() > 0);
    try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    try testing.expectEqual(@as(usize, 2), hc.entryCount());

    // Let the writer run and drop the allowance: the next pass evicts.
    hc.disk.?.writer.?.setPaused(false);
    hc.disk.?.drainWriter(); // test-side only: the engine never waits here
    hc.ssd_idle_mem = 0;
    hc.spillIdleEntries(s);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try testing.expectEqualSlices(u32, &tok_b, hc.entries.items[0].tokens);
}

test "SSD-first: the idle ALLOWANCE bounds eviction, not the fact of being idle" {
    // The spill used to evict every non-newest entry on every finish, ignoring
    // `--prefix-cache-mem`; writing stays unconditional, evicting is what the allowance bounds.
    const io = std.testing.io;
    const s = mlx.gpuStream();

    var tok_a: [600]u32 = undefined;
    for (&tok_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tok_b: [600]u32 = undefined;
    for (&tok_b, 0..) |*t, i| t.* = @intCast(i + 90_000);
    var tok_c: [600]u32 = undefined;
    for (&tok_c, 0..) |*t, i| t.* = @intCast(i + 300_000);

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);

    // Two sessions, an allowance that covers the idle one: both stay.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);
        var hc = HotPrefixCache.initWithMem(testing.allocator, 8, 0);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-allow2", 0, 128);
        defer hc.deinit();

        _ = try hc.commit(&cache, &tok_a, false);
        _ = try hc.commit(&cache, &tok_b, false);
        hc.ssd_idle_mem = hc.entries.items[0].kv_bytes; // room for one idle entry
        hc.spillIdleEntries(s);
        try testing.expectEqual(@as(usize, 2), hc.entryCount());
        // ...and the write still happened.
        hc.disk.?.drainWriter();
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    }

    // A third session past the allowance: the oldest idle entry goes, and only that one.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);
        var hc = HotPrefixCache.initWithMem(testing.allocator, 8, 0);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-allow3", 0, 128);
        defer hc.deinit();

        _ = try hc.commit(&cache, &tok_a, false); // oldest
        _ = try hc.commit(&cache, &tok_b, false);
        _ = try hc.commit(&cache, &tok_c, false); // active
        hc.ssd_idle_mem = hc.entries.items[0].kv_bytes; // room for ONE of the two idle
        hc.spillIdleEntries(s);
        try testing.expectEqual(@as(usize, 2), hc.entryCount());
        for (hc.entries.items) |*e| try testing.expect(!std.mem.eql(u32, e.tokens, &tok_a));
        var saw_b = false;
        var saw_c = false;
        for (hc.entries.items) |*e| {
            if (std.mem.eql(u32, e.tokens, &tok_b)) saw_b = true;
            if (std.mem.eql(u32, e.tokens, &tok_c)) saw_c = true;
        }
        try testing.expect(saw_b and saw_c);
    }

    // Allowance 0 means what it says: nothing idle stays resident.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);
        var hc = HotPrefixCache.initWithMem(testing.allocator, 8, 0);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-allow0", 0, 128);
        defer hc.deinit();

        _ = try hc.commit(&cache, &tok_a, false);
        _ = try hc.commit(&cache, &tok_b, false);
        _ = try hc.commit(&cache, &tok_c, false);
        hc.ssd_idle_mem = 0;
        hc.spillIdleEntries(s);
        try testing.expectEqual(@as(usize, 1), hc.entryCount());
        try testing.expectEqualSlices(u32, &tok_c, hc.entries.items[0].tokens);
    }
}

test "SSD-first: the allowance is a HARD cap, shed in two tiers (durable first)" {
    // Shed the entries that have a durable copy first, then the rest: an unpersistable entry
    // survives while the cache is under the cap and is dropped only past it.
    const io = std.testing.io;
    const s = mlx.gpuStream();

    var tok_a: [600]u32 = undefined;
    for (&tok_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tok_b: [600]u32 = undefined;
    for (&tok_b, 0..) |*t, i| t.* = @intCast(i + 90_000);
    var tok_c: [600]u32 = undefined;
    for (&tok_c, 0..) |*t, i| t.* = @intCast(i + 300_000);

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);

    // A is the oldest and unpersistable (a layer offset short of the persist target); B is newer and persists; C is active.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);
        var hc = HotPrefixCache.initWithMem(testing.allocator, 8, 0);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-tier2", 0, 128);
        defer hc.deinit();

        _ = try hc.commit(&cache, &tok_a, false);
        _ = try hc.commit(&cache, &tok_b, false);
        _ = try hc.commit(&cache, &tok_c, false);
        for (hc.entries.items) |*e| {
            if (std.mem.eql(u32, e.tokens, &tok_a)) e.snapshot.entries[0].offset = 300;
        }
        hc.ssd_idle_mem = hc.entries.items[0].kv_bytes;
        hc.spillIdleEntries(s);

        // Tier 1 shed B, the durable one, even though A is older.
        try testing.expectEqual(@as(usize, 2), hc.entryCount());
        var saw_a = false;
        for (hc.entries.items) |*e| {
            if (std.mem.eql(u32, e.tokens, &tok_a)) saw_a = true;
            try testing.expect(!std.mem.eql(u32, e.tokens, &tok_b));
        }
        try testing.expect(saw_a);

        // Allowance zero: A has nowhere to go and the cap is hard, so tier 2 drops it.
        hc.ssd_idle_mem = 0;
        hc.spillIdleEntries(s);
        try testing.expectEqual(@as(usize, 1), hc.entryCount());
        try testing.expectEqualSlices(u32, &tok_c, hc.entries.items[0].tokens);
    }

    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);
        var hc = HotPrefixCache.initWithMem(testing.allocator, 8, 0);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-tier2b", 0, 128);
        defer hc.deinit();
        hc.disk.?.ssd_first = true;
        hc.disk.?.armTestSpace(3 * 1024 * 1024 * 1024, 512 * 1024 * 1024 * 1024);

        _ = try hc.commit(&cache, &tok_a, false); // oldest
        _ = try hc.commit(&cache, &tok_b, false);
        _ = try hc.commit(&cache, &tok_c, false); // active
        hc.ssd_idle_mem = hc.entries.items[0].kv_bytes;
        hc.spillIdleEntries(s);
        try testing.expectEqual(@as(usize, 0), hc.disk.?.entryCount());
        try testing.expectEqual(@as(usize, 2), hc.entryCount());
        for (hc.entries.items) |*e| try testing.expect(!std.mem.eql(u32, e.tokens, &tok_a));
    }
}

/// A commit whose SSD flush never ran, so `spillIdleEntries` is the tier's only writer.
fn commitUnflushed(hc: *HotPrefixCache, cache: *const KVCache, tokens: []const u32) !CommitStatus {
    const r = try hc.commit(cache, tokens, false);
    if (hc.pending_disk) |*p| p.deinit(hc.allocator);
    hc.pending_disk = null;
    hc.disk_dirty = false;
    return r;
}

test "SSD-first: a silent SKIP is not a durable copy — the idle entry stays resident" {
    // Every silent skip used to read as "the SSD holds this session" and evicted the RAM copy.
    // Four skip reasons, each asserting both halves: the tier holds nothing, RAM still holds the entry.
    const io = std.testing.io;
    const s = mlx.gpuStream();

    var tokens_a: [600]u32 = undefined;
    for (&tokens_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tokens_b: [600]u32 = undefined;
    for (&tokens_b, 0..) |*t, i| t.* = @intCast(i + 90_000);

    // Arm 1: the volume declined the store.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);
        var cache = try KVCache.init(testing.allocator, 1);
        defer cache.deinit();
        try testFillCache(&cache, s, 1, 600);

        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-declined", 0, 128);
        defer hc.deinit();
        hc.ssd_idle_mem = 64 * 1024 * 1024 * 1024;
        hc.disk.?.ssd_first = true;
        hc.disk.?.armTestSpace(3 * 1024 * 1024 * 1024, 512 * 1024 * 1024 * 1024);

        _ = try commitUnflushed(&hc, &cache, &tokens_a);
        _ = try commitUnflushed(&hc, &cache, &tokens_b);
        hc.spillIdleEntries(s);
        try testing.expectEqual(@as(usize, 0), hc.disk.?.entryCount());
        try testing.expectEqual(@as(usize, 2), hc.entryCount());
    }

    // Arm 2: under `MIN_PERSIST_TOKENS`.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);
        var cache = try KVCache.init(testing.allocator, 1);
        defer cache.deinit();
        try testFillCache(&cache, s, 1, 400);

        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-short", 0, 128);
        defer hc.deinit();
        hc.ssd_idle_mem = 64 * 1024 * 1024 * 1024;
        _ = try commitUnflushed(&hc, &cache, tokens_a[0..400]);
        _ = try commitUnflushed(&hc, &cache, tokens_b[0..400]);
        hc.spillIdleEntries(s);
        try testing.expectEqual(@as(usize, 0), hc.disk.?.entryCount());
        try testing.expectEqual(@as(usize, 2), hc.entryCount());
    }

    // Arm 3: a layer offset short of the persist target.
    {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root_len = try tmp.dir.realPath(io, &buf);
        var cache = try KVCache.init(testing.allocator, 2);
        defer cache.deinit();
        try testFillCache(&cache, s, 2, 600);

        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-offset", 0, 128);
        defer hc.deinit();
        hc.ssd_idle_mem = 64 * 1024 * 1024 * 1024;
        _ = try commitUnflushed(&hc, &cache, &tokens_a);
        _ = try commitUnflushed(&hc, &cache, &tokens_b);
        for (hc.entries.items) |*e| {
            if (std.mem.eql(u32, e.tokens, &tokens_a)) e.snapshot.entries[1].offset = 300;
        }
        hc.spillIdleEntries(s);
        try testing.expectEqual(@as(usize, 0), hc.disk.?.entryCount());
        try testing.expectEqual(@as(usize, 2), hc.entryCount());
    }
}

test "SSD-first: a PARTIAL copy is not a copy — the idle entry stays resident" {
    // A byte-capped flush lands real bytes and stops on a chunk boundary: the entry on disk is short and RAM keeps it.
    const io = std.testing.io;
    const s = mlx.gpuStream();

    var tokens_a: [600]u32 = undefined;
    for (&tokens_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tokens_b: [600]u32 = undefined;
    for (&tokens_b, 0..) |*t, i| t.* = @intCast(i + 90_000);

    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.ssd_first = true;
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-partial", 0, 128);
    defer hc.deinit();
    hc.ssd_idle_mem = 64 * 1024 * 1024 * 1024;
    // One byte: the loop writes chunk 0 and stops.
    hc.disk.?.max_flush_bytes = 1;

    _ = try commitUnflushed(&hc, &cache, &tokens_a);
    _ = try commitUnflushed(&hc, &cache, &tokens_b);
    hc.spillIdleEntries(s);
    hc.disk.?.drainWriter();
    try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    try testing.expect(hc.disk.?.entries.items[0].kv_len < 600);
    try testing.expectEqual(@as(usize, 2), hc.entryCount());

    // Lift the cap and the allowance: the next pass completes the copy and the durable entry goes.
    hc.disk.?.max_flush_bytes = 512 * 1024 * 1024;
    hc.ssd_idle_mem = 0;
    hc.spillIdleEntries(s);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try testing.expectEqualSlices(u32, &tokens_b, hc.entries.items[0].tokens);
    try testing.expectEqual(@as(u32, 600), hc.disk.?.entries.items[0].kv_len);
}

test "DiskTier.holdsFullPrefix: the INDEX must agree before a RAM copy is discarded" {
    // `.persisted` is the write path's claim; this is the manifest's.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    var tier = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-holds", 0, 128);
    defer tier.deinit();

    try testing.expect(!tier.holdsFullPrefix(cache.entries, cache.step, &tokens, false, cache.config));

    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
    tier.drainWriter();
    try testing.expect(tier.holdsFullPrefix(cache.entries, cache.step, &tokens, false, cache.config));
    try testing.expect(!tier.holdsFullPrefix(cache.entries, cache.step, &tokens, true, cache.config));
    try testing.expect(!tier.holdsFullPrefix(cache.entries, cache.step, &tokens, false, .{ .scheme = .affine, .bits = 4, .group_size = 64 }));

    // A truncated tail chunk, the shape `scan` records after a kill -9.
    const cb = tier.entries.items[0].chunk_bytes;
    const keep = cb[cb.len - 1];
    cb[cb.len - 1] = 0;
    try testing.expect(!tier.holdsFullPrefix(cache.entries, cache.step, &tokens, false, cache.config));
    cb[cb.len - 1] = keep;
    try testing.expect(tier.holdsFullPrefix(cache.entries, cache.step, &tokens, false, cache.config));

    tier.entries.items[0].kv_len = 400;
    try testing.expect(!tier.holdsFullPrefix(cache.entries, cache.step, &tokens, false, cache.config));
}

test "SSD-first: one resident session makes reclaimableBytes truthfully ZERO" {
    // Under SSD-first RAM holds exactly the active session at rest, so the largest entry is the
    // only entry and the provable discount is 0 (its buffers are the live KV).
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);

    var tokens_a: [600]u32 = undefined;
    for (&tokens_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tokens_b: [600]u32 = undefined;
    for (&tokens_b, 0..) |*t, i| t.* = @intCast(i + 90_000);

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.ssd_first = true;
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-reclaim", 0, 128);
    defer hc.deinit();

    _ = try hc.commit(&cache, &tokens_a, false);
    hc.flushPendingDisk(s);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try testing.expect(hc.residentBytes() > 0);
    try testing.expectEqual(@as(u64, 0), hc.reclaimableBytes());

    // Mid-switch, two sessions are briefly resident and the non-active one is reclaimable.
    _ = try hc.commit(&cache, &tokens_b, false);
    hc.flushPendingDisk(s);
    try testing.expectEqual(@as(usize, 2), hc.entryCount());
    try testing.expect(hc.reclaimableBytes() > 0);

    // ...and the idle spill returns it to 0 without evicting the session being served.
    hc.spillIdleEntries(s);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try testing.expectEqual(@as(u64, 0), hc.reclaimableBytes());
}

test "reclaimableBytesFor: only an entry the PROMPT could restore from is unevictable" {
    // The guard's credit with the prompt in hand: the prompt-blind rule always subtracts the
    // whole cache under one-session-resident.
    const s = mlx.gpuStream();

    var tokens_a: [600]u32 = undefined;
    for (&tokens_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tokens_b: [600]u32 = undefined;
    for (&tokens_b, 0..) |*t, i| t.* = @intCast(i + 90_000);

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.ssd_first = true;
    defer hc.deinit();
    _ = try hc.commit(&cache, &tokens_a, false);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    const resident = hc.residentBytes();
    try testing.expect(resident > 0);

    // (1) The prompt extends the resident session: nothing to reclaim.
    try testing.expectEqual(@as(u64, 0), hc.reclaimableBytesFor(&tokens_a));
    try testing.expectEqual(@as(u64, 0), hc.reclaimableBytes());

    // (2) A different session's prompt: the bytes are reclaimable.
    try testing.expectEqual(resident, hc.reclaimableBytesFor(&tokens_b));
    try testing.expect(hc.reclaimableBytesFor(&tokens_b) > hc.reclaimableBytes());

    // A prefix too short to restore from does not pin the entry either.
    var barely: [600]u32 = undefined;
    for (&barely, 0..) |*t, i| t.* = @intCast(i + 7);
    for (barely[MIN_CANCELLED_COMMIT_TOKENS - 8 ..]) |*t| t.* = 424_242;
    try testing.expectEqual(resident, hc.reclaimableBytesFor(&barely));

    // (3) Two entries, prompt matches one: only that one is withheld.
    _ = try hc.commit(&cache, &tokens_b, false);
    try testing.expectEqual(@as(usize, 2), hc.entryCount());
    const both = hc.residentBytes();
    const credit_a = hc.reclaimableBytesFor(&tokens_a);
    try testing.expect(credit_a > 0 and credit_a < both);
    try testing.expect(credit_a >= hc.reclaimableBytes());
}

test "EntryDigest: the published snapshot answers the reclaimable question without the cache" {
    // The guard reads a published snapshot of these digests instead of the cache.
    const s = mlx.gpuStream();
    const A = testing.allocator;

    var tokens_a: [600]u32 = undefined;
    for (&tokens_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tokens_b: [600]u32 = undefined;
    for (&tokens_b, 0..) |*t, i| t.* = @intCast(i + 90_000);
    var short: [64]u32 = undefined;
    for (&short, 0..) |*t, i| t.* = @intCast(i + 7);

    var cache = try KVCache.init(A, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);

    var hc = HotPrefixCache.initWithMem(A, 4, 0);
    hc.ssd_first = true;
    defer hc.deinit();
    _ = try hc.commit(&cache, &tokens_a, false);
    const resident = hc.residentBytes();

    // Publish, then replace: the superseded slice is the caller's to free.
    var d1 = try hc.digestsAlloc(A);
    try testing.expectEqual(@as(usize, 1), d1.len);
    try testing.expectEqual(resident, d1[0].kv_bytes);
    _ = try hc.commit(&cache, &tokens_b, false);
    const d2 = try hc.digestsAlloc(A);
    A.free(d1);
    d1 = d2;
    defer A.free(d1);
    try testing.expectEqual(@as(usize, 2), d1.len);

    const fp_a = HotPrefixCache.prefixFingerprint(&tokens_a).?;
    const both = hc.residentBytes();
    const credit_a = HotPrefixCache.reclaimableFromDigests(d1, both, fp_a);
    try testing.expect(credit_a > 0 and credit_a < both);

    // A prompt matching neither session credits the whole residency.
    var tokens_c: [600]u32 = undefined;
    for (&tokens_c, 0..) |*t, i| t.* = @intCast(i + 500_000);
    const fp_c = HotPrefixCache.prefixFingerprint(&tokens_c);
    try testing.expectEqual(both, HotPrefixCache.reclaimableFromDigests(d1, both, fp_c));

    // A prompt under the restore floor pins nothing, and hashes to null.
    try testing.expectEqual(@as(?u64, null), HotPrefixCache.prefixFingerprint(&short));
    try testing.expectEqual(both, HotPrefixCache.reclaimableFromDigests(d1, both, null));

    try testing.expectEqual(hc.reclaimableBytesFor(&tokens_a), credit_a);
    try testing.expectEqual(hc.reclaimableBytesFor(&tokens_c), both);
}

test "prefix cache: the trim bill prices only the checkpoints a shed would keep" {
    // `shedCheckpointsToFit` thins the interior the moment an entry lands over the cap, so
    // billing every lower checkpoint prices memory the entry never holds.
    const positions = [_]usize{ 100, 200, 300, 400, 500 };
    const bytes = [_]u64{ 10, 10, 10, 10, 10 };
    try testing.expectEqual(@as(?u64, 50), HotPrefixCache.shedSurvivorBytes(&positions, &bytes, 50, .min_span_recency));
    try testing.expectEqual(@as(?u64, 20), HotPrefixCache.shedSurvivorBytes(&positions, &bytes, 25, .min_span_recency));
    try testing.expectEqual(@as(?u64, 10), HotPrefixCache.shedSurvivorBytes(&positions, &bytes, 15, .min_span_recency));
    try testing.expectEqual(@as(?u64, null), HotPrefixCache.shedSurvivorBytes(&positions, &bytes, 5, .min_span_recency));
}

test "prefix cache: a 383k oversized hybrid entry trims instead of flat-declining" {
    // The live #330 follow-up shape: qwen4_exp, stride 4096, a 383,069-token entry at 13,056
    // bytes per KV row, ~26 MB per checkpoint, a 3,873.54 MB budget; it flat-declined.
    const row_bytes: u64 = 13_056;
    const per_cp: u64 = 26 * 1024 * 1024;
    const budget: u64 = 3873 * 1024 * 1024;
    const tokens: usize = 383_069;

    // 93 stride captures plus the end-of-prompt snap.
    var all_pos: [94]usize = undefined;
    for (all_pos[0..93], 0..) |*p, i| p.* = (i + 1) * 4096;
    all_pos[93] = 383_039;
    var bytes: [94]u64 = undefined;
    for (&bytes) |*b| b.* = per_cp;

    // (a) Drop-oldest retention: the lowest survivor already prices past the budget.
    {
        const end_anchored = all_pos[78..94];
        try testing.expect(@as(u64, end_anchored[0]) * row_bytes > budget);
        try testing.expectEqual(
            @as(?usize, null),
            HotPrefixCache.trimLenForBudgetPure(budget, tokens, row_bytes, end_anchored, bytes[0..16], .min_span_recency, null),
        );
    }

    // (b) Span-preserving retention: the survivors spread over the whole prompt.
    var pos: [94]usize = all_pos;
    var n: usize = pos.len;
    while (n > 16) {
        const drop = transformer_mod.positionDropIndexUsize(pos[0..n], .min_span_recency);
        var k = drop;
        while (k + 1 < n) : (k += 1) pos[k] = pos[k + 1];
        n -= 1;
    }
    try testing.expectEqual(@as(usize, 4096), pos[0]);
    try testing.expectEqual(@as(usize, 383_039), pos[n - 1]);
    const tl = HotPrefixCache.trimLenForBudgetPure(budget, tokens, row_bytes, pos[0..n], bytes[0..n], .min_span_recency, null) orelse
        return error.NoTrimPoint;
    try testing.expect(tl >= 126_976);
    try testing.expect(std.mem.indexOfScalar(usize, pos[0..n], tl) != null);
    var kept: usize = 0;
    while (kept < n and pos[kept] <= tl) kept += 1;
    const survivors = HotPrefixCache.shedSurvivorBytes(
        pos[0..kept],
        bytes[0..kept],
        budget - @as(u64, tl) * row_bytes,
        .min_span_recency,
    ) orelse return error.ShedDoesNotFit;
    try testing.expect(@as(u64, tl) * row_bytes + survivors <= budget);

    // (c) Pricing every lower checkpoint at the same point buys a strictly shorter prefix.
    var all_lower: ?usize = null;
    var k = n;
    while (k > 0) {
        k -= 1;
        const p = pos[k];
        if (p < MIN_CANCELLED_COMMIT_TOKENS) break;
        if (@as(u64, p) * row_bytes + @as(u64, k + 1) * per_cp <= budget) {
            all_lower = p;
            break;
        }
    }
    try testing.expect(all_lower != null);
    try testing.expect(tl > all_lower.?);
}

test "prefix cache: a failed trimmed copy retries at the next-lower checkpoint" {
    // A `trimmedCopy` failure at one width is not a verdict on the entry.
    const positions = [_]usize{ 4096, 8192, 12288 };
    const bytes = [_]u64{ 1024, 1024, 1024 };
    const budget: u64 = 60_000;
    const tl = HotPrefixCache.trimLenForBudgetPure(budget, 100_000, 4, &positions, &bytes, .min_span_recency, null) orelse
        return error.NoTrimPoint;
    try testing.expectEqual(@as(usize, 12288), tl);
    try testing.expectEqual(
        @as(?usize, 8192),
        HotPrefixCache.trimLenForBudgetPure(budget, tl - 1, 4, &positions, &bytes, .min_span_recency, null),
    );
    try testing.expectEqual(
        @as(?usize, null),
        HotPrefixCache.trimLenForBudgetPure(budget, 255, 4, &positions, &bytes, .min_span_recency, null),
    );
}

test "prefix cache: an oversized commit names WHICH outcome declined it" {
    const a = TrimDecline.no_restorable_prefix.reason();
    const b = TrimDecline.snapshot_copy_failed.reason();
    const c = TrimDecline.checkpoint_list_copy_failed.reason();
    try testing.expect(a.len > 0 and b.len > 0 and c.len > 0);
    try testing.expect(!std.mem.eql(u8, a, b));
    try testing.expect(!std.mem.eql(u8, a, c));
    try testing.expect(!std.mem.eql(u8, b, c));
}

test "prefix cache: the ungated retention + trim arms reproduce the previous policy exactly" {
    // Characterization of the ungated arms: min-span over the whole interior with no recency,
    // and a trim bill of every lower checkpoint.
    const t = std.testing;

    // The cache's default is the ungated policy.
    var hc = HotPrefixCache.init(t.allocator, 4);
    defer hc.deinit();
    try t.expectEqual(transformer_mod.ThinPolicy.min_span, hc.cp_thin);

    try t.expectEqualStrings("all_lower", HotPrefixCache.trimBillArm(4, false));
    try t.expectEqualStrings("all_lower", HotPrefixCache.trimBillArm(32, false));
    try t.expectEqualStrings("shed", HotPrefixCache.trimBillArm(32, true));

    // The two arms really disagree: four checkpoints of 10 bytes, row_bytes 0.
    const positions = [_]usize{ 256, 512, 768, 1024 };
    const bytes = [_]u64{ 10, 10, 10, 10 };
    // shed arm: at position 1024 the shed can thin down to 20 bytes.
    try t.expectEqual(
        @as(?usize, 1024),
        HotPrefixCache.trimLenForBudgetPure(25, 4096, 0, &positions, &bytes, .min_span, null),
    );
    // ungated arm bills every lower checkpoint: 1024 costs all four (40), over the 25-byte budget.
    var all_lower_at_1024: u64 = 0;
    for (bytes) |b| all_lower_at_1024 += b;
    try t.expectEqual(@as(u64, 40), all_lower_at_1024);
    try t.expect(all_lower_at_1024 > 25);
    try t.expectEqual(
        @as(?u64, 20),
        HotPrefixCache.shedSurvivorBytes(&positions, &bytes, 25, .min_span),
    );
}

test "prefix cache: the trim-inputs line carries the price, the positions and the chosen bill" {
    // Format pinned on the live 383k fixture.
    const row_bytes: u64 = 13_056;
    const per_cp: u64 = 26 * 1024 * 1024;
    const budget: u64 = 3873 * 1024 * 1024;

    var all_pos: [94]usize = undefined;
    for (all_pos[0..93], 0..) |*p, i| p.* = (i + 1) * 4096;
    all_pos[93] = 383_039;
    var bytes: [94]u64 = undefined;
    for (&bytes) |*b| b.* = per_cp;

    var buf: [768]u8 = undefined;
    // Long list: elided at TRIM_LOG_MAX_POS, but the count stays exact.
    {
        const line = HotPrefixCache.formatTrimInputs(&buf, 383_069, row_bytes, budget, &all_pos, &bytes, all_pos.len, 126_976, true);
        try testing.expect(std.mem.startsWith(u8, line, "  [hot-cache] trim inputs: tokens=383069 row_bytes=13056 budget=3873.00 MB list_len=94 arm=shed survivors=["));
        try testing.expect(std.mem.endsWith(u8, line, "\n"));
        try testing.expect(std.mem.indexOf(u8, line, "[4096,8192,12288,") != null);
        try testing.expect(std.mem.indexOf(u8, line, ",...] (32 of 94)") != null);
        try testing.expect(std.mem.indexOf(u8, line, " chosen=126976") != null);
        try testing.expect(std.mem.indexOf(u8, line, " chosen_cp_bytes=27262976") != null);
        try testing.expectEqual(@as(usize, 1), std.mem.count(u8, line, "\n"));
        try testing.expect(line.len < buf.len);
    }
    {
        const line = HotPrefixCache.formatTrimInputs(&buf, 900, row_bytes, budget, all_pos[0..3], bytes[0..3], 3, 8192, true);
        try testing.expect(std.mem.indexOf(u8, line, "survivors=[4096,8192,12288] (3 of 3)") != null);
        try testing.expect(std.mem.indexOf(u8, line, "...") == null);
    }
    {
        const line = HotPrefixCache.formatTrimInputs(&buf, 900, row_bytes, budget, all_pos[0..2], bytes[0..2], 2, null, true);
        try testing.expect(std.mem.indexOf(u8, line, " chosen=none") != null);
        try testing.expect(std.mem.indexOf(u8, line, "chosen_cp_bytes") == null);
    }
    {
        const line = HotPrefixCache.formatTrimInputs(&buf, 900, row_bytes, budget, all_pos[0..0], bytes[0..0], 0, 512, true);
        try testing.expect(std.mem.indexOf(u8, line, "survivors=[] (0 of 0)") != null);
        try testing.expect(std.mem.indexOf(u8, line, " chosen=512 chosen_cp_bytes=0") != null);
    }
    try testing.expectEqualStrings("shed", HotPrefixCache.trimBillArm(32, true));
    try testing.expectEqualStrings("shed", HotPrefixCache.trimBillArm(HotPrefixCache.SHED_SIM_MAX, true));
    try testing.expectEqualStrings("all_lower", HotPrefixCache.trimBillArm(HotPrefixCache.SHED_SIM_MAX + 1, true));
    {
        const line = HotPrefixCache.formatTrimInputs(&buf, 383_069, row_bytes, budget, &all_pos, &bytes, 200, 126_976, true);
        try testing.expect(std.mem.indexOf(u8, line, "list_len=200 arm=all_lower") != null);
        try testing.expect(std.mem.indexOf(u8, line, ",...] (32 of 200)") != null);
    }
}

test "prefix cache: the trim's row price is the entry's own bytes divided by its rows" {
    // The trim price divides each of the six quantized arrays by its own `shape[2]`, a layout
    // assumption; pin the invariant (price x capacity == snapshot bytes) rather than the constant.
    const s = mlx.gpuStream();

    for ([_]kv_quant.KVQuantConfig{
        kv_quant.KVQuantConfig.dense,
        kv_quant.KVQuantConfig.affine(8),
        kv_quant.KVQuantConfig.affine(6),
        kv_quant.KVQuantConfig.affine(4),
    }) |cfg| {
        var cache = try KVCache.initWithConfig(testing.allocator, 2, cfg);
        defer cache.deinit();

        // qwen4_exp's own attention shape: 2 kv heads, head_dim 256.
        const mk = struct {
            fn f(str: mlx.mlx_stream, len: c_int) !mlx.mlx_array {
                const shape = [_]c_int{ 1, 2, len, 256 };
                var a = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_ones(&a, &shape, 4, .bfloat16, str));
                return a;
            }
        }.f;
        // Two writes so the second crosses a growth event.
        for ([_]c_int{ 64, 40 }) |n| {
            const k = try mk(s, n);
            defer _ = mlx.mlx_array_free(k);
            var dv = try cache.update(0, k, k, s, 0);
            dv.deinit();
            const k2 = try mk(s, n);
            defer _ = mlx.mlx_array_free(k2);
            var dv2 = try cache.update(1, k2, k2, s, 0);
            dv2.deinit();
        }

        var snap = try cache.snapshot();
        defer snap.deinit();

        const row_bytes = HotPrefixCache.snapshotRowBytes(&snap);
        const total = HotPrefixCache.snapshotBytes(&snap);
        try testing.expect(row_bytes > 0);

        const cap: u64 = @intCast(mlx.getShape(snap.entries[0].keys)[2]);
        try testing.expect(cap >= 104);
        try testing.expectEqual(total, row_bytes * cap);

        // Per token: 2 layers of (K+V) at 2 heads x 256 dims; affine packs to `bits` plus scale and bias per group of 64.
        const per_layer: u64 = switch (cfg.scheme) {
            .off => 2 * (2 * 256 * 2),
            else => blk: {
                const packed_b: u64 = 2 * 256 * @as(u64, cfg.bits) / 8;
                const groups: u64 = 2 * 256 / cfg.group_size;
                break :blk 2 * (packed_b + groups * 2 * 2);
            },
        };
        try testing.expectEqual(2 * per_layer, row_bytes);
    }
}

test "HotPrefixCache: a bounded disk flush leaves disk_dirty set and later flushes complete the entry — never a whole-entry claim in between" {
    // A long entry reaches the tier in pieces; each piece is a valid shorter entry and the next
    // `flushPendingDisk` extends it.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const base = buf[0..root_len];

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-partial", 0, 128);
    defer hc.deinit();
    hc.disk.?.max_flush_bytes = 1; // one chunk per flush: the bounded shape

    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try testFillCache(&cache, s, 2, 600);
    _ = try hc.commit(&cache, &tokens, false);
    try testing.expect(hc.disk_dirty);

    // 600 tokens at 128/chunk = 5 chunks; each flush lands one.
    var flushes: usize = 0;
    var last_kv: u32 = 0;
    while (hc.disk_dirty and flushes < 10) : (flushes += 1) {
        hc.flushPendingDisk(s);
        const d = &hc.disk.?;
        try testing.expectEqual(@as(usize, 1), d.entryCount());
        const kv = d.entries.items[0].kv_len;
        try testing.expect(kv >= last_kv);
        if (hc.disk_dirty) try testing.expectEqual(@as(u32, 0), kv % 128);
        const m = d.bestMatch(&tokens, false, kv_quant.KVQuantConfig.dense, .{}).?;
        try testing.expectEqual(kv, m.usable);
        last_kv = kv;
    }
    try testing.expect(!hc.disk_dirty);
    try testing.expectEqual(@as(usize, 5), flushes);
    try testing.expectEqual(@as(u32, 600), hc.disk.?.entries.items[0].kv_len);
}

// ── Restore by move (checkout) ──

/// Live Metal bytes, with the allocator pool pinned first.
fn testLiveBytes(s: mlx.mlx_stream) u64 {
    _ = mlx.mlx_synchronize(s);
    _ = mlx.mlx_clear_cache();
    var live: usize = 0;
    _ = mlx.mlx_get_active_memory(&live);
    return @intCast(live);
}

/// Bytes of one layer's key buffer (capacity, not logical rows).
fn testKeyBufferBytes(cache: *KVCache, layer: usize) u64 {
    const sh = mlx.getShape(cache.entries[layer].keys);
    return @as(u64, @intCast(sh[0])) * @as(u64, @intCast(sh[1])) *
        @as(u64, @intCast(sh[2])) * @as(u64, @intCast(sh[3])) * 4; // f32
}

fn testCheckoutCache(hc: *HotPrefixCache, s: mlx.mlx_stream, tokens: []const u32, reserve: usize) !void {
    var donor = try KVCache.init(testing.allocator, 1);
    defer donor.deinit();
    donor.reserve(reserve);
    try testFillCache(&donor, s, 1, @intCast(tokens.len));
    _ = try hc.commit(&donor, tokens, false);
}

test "restore by move: a full-prefix hit checks the entry out and the append donates in place" {
    // `KVCache.restore` binds through `mlx_array_set`, so the entry keeps a second reference and
    // the first `writeAtOffset` cannot donate: `copy_gpu` privatised the whole prefix (5.13 GB /
    // ~110 ms at 393k, 45% of the warm TTFT). The observable is allocation, not address: under
    // suite-wide pressure the buffer pool recycles addresses.
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    // The prompt extends the entry: a full-prefix hit whose commit will replace this same entry.
    var prompt: [608]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(i + 7);
    // A reservation big enough that one buffer dwarfs any pool noise.
    const reserve: usize = 1 << 20;

    var moved_bytes: [64]f32 = undefined;
    var copied_bytes: [64]f32 = undefined;
    var moved_delta: u64 = 0;
    var copied_delta: u64 = 0;
    var buf_bytes: u64 = 0;

    // Arm A: the move.
    {
        restore_move_override = true;
        defer restore_move_override = null;
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.ssd_first = true;
        defer hc.deinit();
        try testCheckoutCache(&hc, s, &tokens, reserve);

        var slot = try KVCache.init(testing.allocator, 1);
        defer slot.deinit();
        var moe_off: usize = 0;
        const res = try hc.lookupAndRestoreForSlot(&slot, &moe_off, null, s, &prompt, false, 0, null, null, null, 0xA11CE);
        try testing.expectEqual(@as(usize, 600), res.matched);
        // The transfer is the scheduler's second step, taken at the last point
        // before the first write (see `donateCheckout`), and the slot adopts the buffers.
        hc.donateCheckout(0xA11CE);
        slot.adoptRestored();
        // A share would size its copy by this reservation; the move ignores it.
        slot.reserve(reserve / 4);

        // The entry gave the buffers up: its handles are empty and it names the slot.
        const e = &hc.entries.items[0];
        try testing.expectEqual(@as(?usize, 0xA11CE), e.checked_out_by);
        try testing.expect(e.snapshot.entries[0].keys.ctx == null);
        try testing.expect(e.snapshot.entries[0].values.ctx == null);

        slot.evalState();
        buf_bytes = testKeyBufferBytes(&slot, 0);
        const before = testLiveBytes(s);
        try testWriteCacheLayer(&slot, s, 0, 600, 8);
        slot.evalState();
        moved_delta = testLiveBytes(s) -| before;
        try testReadKeyRows(&slot, 0, 596, &moved_bytes);
    }

    // Arm B: the refcount share, same bytes by a different buffer.
    {
        restore_move_override = false;
        defer restore_move_override = null;
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.ssd_first = true;
        defer hc.deinit();
        try testCheckoutCache(&hc, s, &tokens, reserve);

        var slot = try KVCache.init(testing.allocator, 1);
        defer slot.deinit();
        var moe_off: usize = 0;
        const res = try hc.lookupAndRestoreForSlot(&slot, &moe_off, null, s, &prompt, false, 0, null, null, null, 0xA11CE);
        try testing.expectEqual(@as(usize, 600), res.matched);
        const e = &hc.entries.items[0];
        try testing.expectEqual(@as(?usize, null), e.checked_out_by);
        try testing.expect(e.snapshot.entries[0].keys.ctx != null);

        slot.evalState();
        try testing.expectEqual(buf_bytes, testKeyBufferBytes(&slot, 0));
        slot.reserve(reserve / 4);
        const before = testLiveBytes(s);
        try testWriteCacheLayer(&slot, s, 0, 600, 8);
        slot.evalState();
        copied_delta = testLiveBytes(s) -| before;
        try testReadKeyRows(&slot, 0, 596, &copied_bytes);
    }

    // The share arm copied into a buffer the request's reservation sizes, not the donor's capacity.
    try testing.expect(copied_delta * 4 > buf_bytes and copied_delta < buf_bytes);
    // The move arm allocated nothing beyond the tail; a quarter of one buffer is a loose ceiling.
    try testing.expect(moved_delta * 4 < buf_bytes);
    try testing.expectEqualSlices(f32, &copied_bytes, &moved_bytes);
}

fn testReadKeyRows(cache: *KVCache, layer: usize, row: usize, out: []f32) !void {
    cache.evalState();
    const p = mlx.mlx_array_data_float32(cache.entries[layer].keys) orelse return error.NotEvaluated;
    for (out, 0..) |*v, i| v.* = p[row * 8 + i];
}

test "a one-token prompt that hits its own entry prefills cold: a restore leaves a token to forward" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const tokens = [_]u32{42};
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    try testCheckoutCache(&hc, s, &tokens, 64);
    var slot = try KVCache.init(testing.allocator, 1);
    defer slot.deinit();
    var moe_off: usize = 0;
    const res = try hc.lookupAndRestoreForSlot(&slot, &moe_off, null, s, &tokens, false, 0, null, null, null, 7);
    try testing.expectEqual(@as(usize, 0), res.matched);
    try testing.expectEqual(@as(usize, 0), moe_off);
    try testing.expectEqual(@as(usize, 0), slot.step);
    try testing.expect(hc.last_restored_used == null);
}

test "restore by move ON DEMAND: a share that does not fit is taken over, off SSD-first" {
    // The admission pass converts a full-entry hit into a checkout; a partial hit and a second call decline.
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    restore_move_override = true;
    defer restore_move_override = null;
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var prompt: [608]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(i + 7);
    var diverged = prompt;
    diverged[500] = 999_999;

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    try testCheckoutCache(&hc, s, &tokens, 4096);

    var slot = try KVCache.init(testing.allocator, 1);
    defer slot.deinit();
    var moe_off: usize = 0;
    const partial = try hc.lookupAndRestoreForSlot(&slot, &moe_off, null, s, &diverged, false, 0, null, null, null, 7);
    try testing.expectEqual(@as(usize, 500), partial.matched);
    try testing.expect(!hc.checkoutRestored(7, diverged.len));

    const res = try hc.lookupAndRestoreForSlot(&slot, &moe_off, null, s, &prompt, false, 0, null, null, null, 7);
    try testing.expectEqual(@as(usize, 600), res.matched);
    try testing.expect(!res.checked_out);
    try testing.expect(hc.checkoutRestored(7, prompt.len));
    try testing.expectEqual(@as(?usize, 7), hc.entries.items[0].checked_out_by);
    try testing.expect(!hc.checkoutRestored(7, prompt.len));

    hc.releaseCheckout(7, "prefill refused");
    try testing.expectEqual(@as(usize, 1), hc.entries.items.len);
    try testing.expectEqual(@as(?usize, null), hc.entries.items[0].checked_out_by);
    try testing.expect(hc.entries.items[0].snapshot.entries[0].keys.ctx != null);
}

test "restore by move: a partial-prefix hit keeps the refcount-share" {
    // A prompt that diverges from the entry makes no replace promise: its commit lands as a new entry.
    const s = mlx.gpuStream();
    restore_move_override = true;
    defer restore_move_override = null;
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var prompt: [600]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(i + 7);
    prompt[500] = 999_999; // diverge

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.ssd_first = true;
    defer hc.deinit();
    try testCheckoutCache(&hc, s, &tokens, 4096);

    var slot = try KVCache.init(testing.allocator, 1);
    defer slot.deinit();
    var moe_off: usize = 0;
    const res = try hc.lookupAndRestoreForSlot(&slot, &moe_off, null, s, &prompt, false, 0, null, null, null, 7);
    try testing.expectEqual(@as(usize, 500), res.matched);
    try testing.expectEqual(@as(?usize, null), hc.entries.items[0].checked_out_by);
    try testing.expect(hc.entries.items[0].snapshot.entries[0].keys.ctx != null);
}

test "restore by move: a refusal BEFORE the append hands the entry back INTACT" {
    // Turn B restored a 364k entry by move, the admission pass refused before any forward, and
    // `finishSlot` dropped the checked-out entry. The checkout is a promise, not a transfer:
    // a slot that ends before the donate hands the entry back as it was.
    const s = mlx.gpuStream();
    restore_move_override = true;
    defer restore_move_override = null;
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var prompt: [608]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(i + 7);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.ssd_first = true;
    defer hc.deinit();
    try testCheckoutCache(&hc, s, &tokens, 4096);
    const billed_before = hc.current_kv_bytes;
    try testing.expect(billed_before > 0);

    var slot = try KVCache.init(testing.allocator, 1);
    defer slot.deinit();
    var moe_off: usize = 0;
    const res = try hc.lookupAndRestoreForSlot(&slot, &moe_off, null, s, &prompt, false, 0, null, null, null, 42);
    try testing.expect(res.checked_out);
    const lru_before = hc.entries.items[0].last_used;

    // The refusal: `runPrefill` returns before `Generator.initWithOptions`, so
    // `donateCheckout` never ran and the entry never gave its handles up.
    try testing.expect(!hc.entries.items[0].checkout_donated);
    hc.releaseCheckout(42, "prefill refused");

    // The entry is whole: same tokens, live handles, same bill, same LRU position, restorable.
    try testing.expectEqual(@as(usize, 1), hc.entries.items.len);
    const e = &hc.entries.items[0];
    try testing.expectEqual(@as(usize, 600), e.tokens.len);
    try testing.expectEqual(@as(?usize, null), e.checked_out_by);
    try testing.expect(e.snapshot.entries[0].keys.ctx != null);
    try testing.expect(e.snapshot.entries[0].values.ctx != null);
    try testing.expectEqual(billed_before, hc.current_kv_bytes);
    try testing.expectEqual(lru_before, e.last_used);

    // ...and the proof: a second slot restores the same prefix from it.
    var slot2 = try KVCache.init(testing.allocator, 1);
    defer slot2.deinit();
    var moe_off2: usize = 0;
    const again = try hc.lookupAndRestoreForSlot(&slot2, &moe_off2, null, s, &prompt, false, 0, null, null, null, 43);
    try testing.expectEqual(@as(usize, 600), again.matched);
    hc.releaseCheckout(43, "second slot ended");
}

test "restore by move: a slot that ends without committing DROPS its checked-out entry" {
    // The bytes die with the slot; `finishSlot` releases unconditionally. `testing.allocator` is the free-once bar.
    const s = mlx.gpuStream();
    restore_move_override = true;
    defer restore_move_override = null;
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var prompt: [608]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(i + 7);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.ssd_first = true;
    defer hc.deinit();
    try testCheckoutCache(&hc, s, &tokens, 4096);
    const billed_before = hc.current_kv_bytes;
    try testing.expect(billed_before > 0);

    var slot = try KVCache.init(testing.allocator, 1);
    var moe_off: usize = 0;
    _ = try hc.lookupAndRestoreForSlot(&slot, &moe_off, null, s, &prompt, false, 0, null, null, null, 42);
    try testing.expectEqual(@as(usize, 1), hc.entries.items.len);

    // Admitted and about to write: the handles go over.
    hc.donateCheckout(42);
    try testing.expect(hc.entries.items[0].checkout_donated);
    try testing.expect(hc.entries.items[0].snapshot.entries[0].keys.ctx == null);

    hc.releaseCheckout(42, "cancelled");
    try testing.expectEqual(@as(usize, 0), hc.entries.items.len);
    try testing.expectEqual(@as(u64, 0), hc.current_kv_bytes);
    // Idempotent.
    hc.releaseCheckout(42, "cancelled");
    hc.releaseCheckout(43, "cancelled");
    slot.deinit();
}

test "restore by move: a commit RECLAIMS the checked-out entry with the grown buffers" {
    // The happy path: the replace arm installs the grown snapshot and clears the mark.
    const s = mlx.gpuStream();
    restore_move_override = true;
    defer restore_move_override = null;
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var prompt: [608]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(i + 7);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.ssd_first = true;
    defer hc.deinit();
    try testCheckoutCache(&hc, s, &tokens, 4096);

    var slot = try KVCache.init(testing.allocator, 1);
    defer slot.deinit();
    var moe_off: usize = 0;
    _ = try hc.lookupAndRestoreForSlot(&slot, &moe_off, null, s, &prompt, false, 0, null, null, null, 42);
    try testing.expect(hc.entries.items[0].checked_out_by != null);
    try testWriteCacheLayer(&slot, s, 0, 600, 8);

    _ = try hc.commit(&slot, &prompt, false);
    try testing.expectEqual(@as(usize, 1), hc.entries.items.len);
    const e = &hc.entries.items[0];
    try testing.expectEqual(@as(?usize, null), e.checked_out_by);
    try testing.expectEqual(@as(usize, 608), e.tokens.len);
    try testing.expect(e.snapshot.entries[0].keys.ctx != null);
    hc.releaseCheckout(42, "finished");
    try testing.expectEqual(@as(usize, 1), hc.entries.items.len);

    var slot2 = try KVCache.init(testing.allocator, 1);
    defer slot2.deinit();
    var moe_off2: usize = 0;
    const res2 = try hc.lookupAndRestore(&slot2, &moe_off2, null, s, &prompt, false, 0, null, null);
    try testing.expectEqual(@as(usize, 607), res2.matched);
}

test "restore by move: a checked-out entry is invisible to a second slot, to eviction and to the published residency" {
    // Its snapshot is empty: invisible to a second slot, to eviction and to the guard.
    const s = mlx.gpuStream();
    restore_move_override = true;
    defer restore_move_override = null;
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var prompt: [608]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(i + 7);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.ssd_first = true;
    defer hc.deinit();
    try testCheckoutCache(&hc, s, &tokens, 4096);

    var slot = try KVCache.init(testing.allocator, 1);
    defer slot.deinit();
    var moe_off: usize = 0;
    _ = try hc.lookupAndRestoreForSlot(&slot, &moe_off, null, s, &prompt, false, 0, null, null, null, 42);
    const billed = hc.entries.items[0].kv_bytes;
    try testing.expect(billed > 0);

    // (c) a second slot misses.
    var slot2 = try KVCache.init(testing.allocator, 1);
    defer slot2.deinit();
    var moe_off2: usize = 0;
    const res2 = try hc.lookupAndRestoreForSlot(&slot2, &moe_off2, null, s, &prompt, false, 0, null, null, null, 43);
    try testing.expectEqual(@as(usize, 0), res2.matched);
    try testing.expect(!res2.full_match);
    try testing.expectEqual(@as(usize, 0), slot2.step);
    try testing.expectEqual(@as(?usize, 42), hc.entries.items[0].checked_out_by);

    // (d) nothing to reclaim, no digest to publish.
    const digests = try hc.digestsAlloc(testing.allocator);
    defer testing.allocator.free(digests);
    try testing.expectEqual(@as(usize, 0), digests.len);
    try testing.expectEqual(@as(u64, 0), hc.reclaimableBytes());
    try testing.expectEqual(@as(u64, 0), hc.reclaimableBytesFor(&prompt));
    // The bill still counts it: the bytes really are resident, in the slot.
    try testing.expectEqual(billed, hc.residentBytes());

    // Eviction cannot take it.
    const Never = struct {
        fn fits(_: ?*anyopaque) bool {
            return false;
        }
    };
    const report = hc.evictLruToAdmit(608, null, Never.fits, false);
    try testing.expectEqual(@as(usize, 0), report.entries);
    try testing.expectEqual(@as(usize, 1), hc.entries.items.len);
    try testing.expectEqual(@as(?usize, 42), hc.entries.items[0].checked_out_by);

    hc.releaseCheckout(42, "test teardown");
}

test "restore by move: the policy is off outside the SSD-first arm and with the move forced off" {
    // Eligible: SSD-first, enabled, no pending flush, whole entry matched, something to append.
    try testing.expect(HotPrefixCache.checkoutEligible(true, true, false, 600, 600, 608, true));
    try testing.expect(!HotPrefixCache.checkoutEligible(false, true, false, 600, 600, 608, true));
    try testing.expect(!HotPrefixCache.checkoutEligible(true, false, false, 600, 600, 608, true));
    try testing.expect(!HotPrefixCache.checkoutEligible(true, true, false, 600, 600, 608, false));
    try testing.expect(!HotPrefixCache.checkoutEligible(true, true, true, 600, 600, 608, true));
    try testing.expect(!HotPrefixCache.checkoutEligible(true, true, false, 600, 500, 608, true));
    try testing.expect(!HotPrefixCache.checkoutEligible(true, true, false, 600, 600, 600, true));
    try testing.expect(!HotPrefixCache.checkoutEligible(true, true, false, 0, 0, 608, true));
}

/// The resident entry whose token record is exactly `toks`.
fn testEntryFor(hc: *HotPrefixCache, toks: []const u32) !*Entry {
    for (hc.entries.items) |*e| {
        if (std.mem.eql(u32, e.tokens, toks)) return e;
    }
    return error.EntryGone;
}

test "SSD-first: a chunk write that fails AFTER the pass invalidates the entry — RAM is never dropped against it" {
    // Pass N stages A and skips it (pending). The writer then loses `c000003` but, FIFO, still
    // lands meta.json, so the index keeps a non-zero `chunk_bytes[3]`. Pass N+1 must not call
    // A durable. The bar is the verdict (`spill_durable`), not the eviction it licenses.
    const io = std.testing.io;
    const s = mlx.gpuStream();

    var tok_a: [600]u32 = undefined;
    for (&tok_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tok_b: [600]u32 = undefined;
    for (&tok_b, 0..) |*t, i| t.* = @intCast(i + 90_000);

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);

    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 8, 0);
    hc.ssd_first = true;
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-poison", 0, 128);
    defer hc.deinit();
    hc.disk.?.ssd_first = true;
    hc.disk.?.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
    hc.disk.?.enableBackgroundWriter();
    hc.ssd_idle_mem = 64 * 1024 * 1024 * 1024;

    _ = try commitUnflushed(&hc, &cache, &tok_a);
    _ = try commitUnflushed(&hc, &cache, &tok_b);

    // Pass N: the writer is held, so A's files stay staged.
    hc.disk.?.writer.?.setPaused(true);
    defer hc.disk.?.writer.?.setPaused(false);
    hc.disk.?.writer.?.injectFailure("c000003", .write);
    hc.spillIdleEntries(s);
    try testing.expect(!(try testEntryFor(&hc, &tok_a)).spill_durable);
    try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    const dead_id = hc.disk.?.entries.items[0].id;

    // Between the passes: chunk 3 dies, meta.json lands.
    hc.disk.?.writer.?.setPaused(false);
    hc.disk.?.drainWriter(); // test-side only: the engine never waits here
    try testing.expect(hc.disk.?.writeErrors() > 0);
    try testing.expectEqual(@as(usize, 5), hc.disk.?.entries.items[0].chunk_bytes.len);

    // The failure names its entry, so a restore from it misses.
    try testing.expectEqual(@as(usize, 1), hc.disk.?.harvestWriteFailures());
    try testing.expect(hc.disk.?.entries.items[0].poisoned);
    try testing.expect(hc.disk.?.bestMatch(&tok_a, false, cache.config, .{}) == null);
    try testing.expect(!hc.disk.?.holdsFullPrefix(cache.entries, cache.step, &tok_a, false, cache.config));

    // Pass N+1: A is not durable; the dead directory is reclaimed and the rebuild is staged.
    hc.disk.?.writer.?.setPaused(true);
    defer hc.disk.?.writer.?.setPaused(false);
    hc.spillIdleEntries(s);
    try testing.expect(!(try testEntryFor(&hc, &tok_a)).spill_durable);
    try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    try testing.expect(hc.disk.?.entries.items[0].id != dead_id);

    // Control: with the writer healthy the same entry is durable and restores whole.
    hc.disk.?.writer.?.injectFailure(null, .write);
    hc.disk.?.writer.?.setPaused(false);
    hc.disk.?.drainWriter();
    hc.spillIdleEntries(s);
    try testing.expect((try testEntryFor(&hc, &tok_a)).spill_durable);

    hc.ssd_idle_mem = 0;
    hc.spillIdleEntries(s);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try testing.expectEqualSlices(u32, &tok_b, hc.entries.items[0].tokens);

    var back = try KVCache.init(testing.allocator, 1);
    defer back.deinit();
    var moe_off: usize = 0;
    const res = try hc.lookupAndRestore(&back, &moe_off, null, s, &tok_a, false, 0, null, null);
    try testing.expectEqual(@as(usize, 599), res.matched);
}

test "SSD-first: a write failure inside the SAME pass still keeps the entry resident" {
    // The same-pass interleaving (`writeErrors() != errs_before`), via a submit-time injection.
    const io = std.testing.io;
    const s = mlx.gpuStream();

    var tok_a: [600]u32 = undefined;
    for (&tok_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tok_b: [600]u32 = undefined;
    for (&tok_b, 0..) |*t, i| t.* = @intCast(i + 90_000);

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);

    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 8, 0);
    hc.ssd_first = true;
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-samepass", 0, 128);
    defer hc.deinit();
    hc.disk.?.ssd_first = true;
    hc.disk.?.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
    hc.disk.?.enableBackgroundWriter();
    hc.ssd_idle_mem = 64 * 1024 * 1024 * 1024;

    _ = try commitUnflushed(&hc, &cache, &tok_a);
    _ = try commitUnflushed(&hc, &cache, &tok_b);
    hc.disk.?.writer.?.injectFailure("c000002", .submit);
    hc.spillIdleEntries(s);
    hc.disk.?.drainWriter();
    try testing.expect(hc.disk.?.writeErrors() > 0);
    try testing.expect(!(try testEntryFor(&hc, &tok_a)).spill_durable);
    try testing.expectEqual(@as(usize, 2), hc.entryCount());
    try testing.expect(!hc.disk.?.holdsFullPrefix(cache.entries, cache.step, &tok_a, false, cache.config));
}

test "SSD-first: the durability check STATS the chunks — a truncated file is never durable" {
    // A byte can go missing with no write error at all; one stat per chunk catches it.
    const io = std.testing.io;
    const s = mlx.gpuStream();

    var tok_a: [600]u32 = undefined;
    for (&tok_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tok_b: [600]u32 = undefined;
    for (&tok_b, 0..) |*t, i| t.* = @intCast(i + 90_000);

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);

    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 8, 0);
    hc.ssd_first = true;
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-stat", 0, 128);
    defer hc.deinit();
    hc.disk.?.ssd_first = true;
    hc.disk.?.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
    hc.disk.?.enableBackgroundWriter();
    hc.ssd_idle_mem = 64 * 1024 * 1024 * 1024;

    _ = try hc.commit(&cache, &tok_a, false);
    _ = try hc.commit(&cache, &tok_b, false);

    // Pass 1 stages A with the writer held; pass 2, after the files land, is the healthy control.
    hc.disk.?.writer.?.setPaused(true);
    defer hc.disk.?.writer.?.setPaused(false);
    hc.spillIdleEntries(s);
    try testing.expect(!(try testEntryFor(&hc, &tok_a)).spill_durable);
    hc.disk.?.writer.?.setPaused(false);
    hc.disk.?.drainWriter();
    hc.spillIdleEntries(s);
    try testing.expectEqual(@as(u64, 0), hc.disk.?.writeErrors());
    try testing.expect((try testEntryFor(&hc, &tok_a)).spill_durable);

    // One chunk loses its bytes behind the tier's back; every in-memory bar still passes.
    const id = hc.disk.?.entries.items[0].id;
    var sub: [64]u8 = undefined;
    const rel = try std.fmt.bufPrint(&sub, "fp-stat/e{d}/c000002.safetensors", .{id});
    try tmp.dir.writeFile(io, .{ .sub_path = rel, .data = "short" });

    hc.spillIdleEntries(s);
    try testing.expect(!(try testEntryFor(&hc, &tok_a)).spill_durable);
    try testing.expectEqual(@as(usize, 2), hc.entryCount());
}

/// A resident entry a live slot holds: eviction can never take it.
fn pcAppendCheckedOut(hc: *HotPrefixCache, tokens: []const u32, kv_bytes: u64, used: u64) !void {
    try hc.entries.append(testing.allocator, .{
        .tokens = try testing.allocator.dupe(u32, tokens),
        .has_tools = false,
        .snapshot = .{ .entries = try testing.allocator.alloc(transformer_mod.KVCacheEntry, 0), .step = 0, .allocator = testing.allocator, .config = transformer_mod.KVQuantConfig.dense },
        .last_used = used,
        .quant_config = kv_quant.KVQuantConfig.dense,
        .kv_bytes = kv_bytes,
        .ssm_checkpoints = null,
        .ssm_bytes = 0,
        .checked_out_by = 1,
    });
    hc.current_kv_bytes += kv_bytes;
}

test "HotPrefixCache: a commit whose only eviction candidates are checked out still returns" {
    // Every budget loop in the commit path must stop when eviction can make no progress —
    // SSD-first with two slots reaches exactly that state on the inference thread.
    const s = mlx.gpuStream();
    const fresh = [_]u32{ 90, 91, 92, 93 };
    const held = [_]u32{ 1, 2 };
    const extend = [_]u32{ 1, 2, 3, 4 };
    const big: u64 = 1 << 30;

    // Count cap: the incoming entry has nowhere to go.
    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 1, 0);
        defer hc.deinit();
        var cache = try KVCache.init(testing.allocator, 3);
        defer cache.deinit();
        try testFillCache(&cache, s, 3, fresh.len);
        try pcAppendCheckedOut(&hc, &held, 0, 1);
        _ = try hc.commit(&cache, &fresh, false);
        try testing.expect(hc.entries.items[0].checked_out_by != null);
    }

    // Append byte budget: already over the cap, nothing reclaimable.
    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 8, big);
        defer hc.deinit();
        var cache = try KVCache.init(testing.allocator, 3);
        defer cache.deinit();
        try testFillCache(&cache, s, 3, fresh.len);
        try pcAppendCheckedOut(&hc, &held, big, 1);
        _ = try hc.commit(&cache, &fresh, false);
        try testing.expect(hc.entries.items[0].checked_out_by != null);
    }

    // Replace path: the target is evictable, the survivors are not.
    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 8, big);
        defer hc.deinit();
        var cache = try KVCache.init(testing.allocator, 3);
        defer cache.deinit();
        try testFillCache(&cache, s, 3, extend.len);
        try pcAppendCheckedOut(&hc, &fresh, big, 1);
        try pcAppendCheckedOut(&hc, &fresh, big, 2);
        var target = try KVCache.init(testing.allocator, 3);
        defer target.deinit();
        try testFillCache(&target, s, 3, held.len);
        _ = try hc.commit(&target, &held, false);
        _ = try hc.commit(&cache, &extend, false);
        try testing.expect(hc.entries.items.len >= 2);
    }
}

test "HotPrefixCache: eviction picks the LRU of the key holding the most entries" {
    var cache = HotPrefixCache.init(testing.allocator, 8);
    defer cache.deinit();
    const key_a: u64 = 0xa;
    const key_b: u64 = 0xb;
    // C (conversation, key A, the global LRU) then D1..D3 (sweep docs, key B).
    const keys = [_]u64{ key_a, key_b, key_b, key_b };
    for (keys, 1..) |k, used| {
        try cache.entries.append(testing.allocator, .{
            .tokens = try testing.allocator.dupe(u32, &[_]u32{ 1, @intCast(used) }),
            .has_tools = false,
            .cache_key = k,
            .snapshot = .{ .entries = try testing.allocator.alloc(transformer_mod.KVCacheEntry, 0), .step = 0, .allocator = testing.allocator, .config = transformer_mod.KVQuantConfig.dense },
            .last_used = used,
            .quant_config = kv_quant.KVQuantConfig.dense,
            .kv_bytes = 0,
            .ssm_checkpoints = null,
            .ssm_bytes = 0,
        });
    }
    // A fourth sweep doc arriving: the sweep evicts its own oldest, never C.
    try testing.expectEqual(@as(?usize, 1), cache.lruIndexExcluding(null, key_b));
    // No incoming key: the largest existing group (B) still pays.
    try testing.expectEqual(@as(?usize, 1), cache.lruIndexExcluding(null, null));
    // D1 protected / checked out: next LRU within B.
    try testing.expectEqual(@as(?usize, 2), cache.lruIndexExcluding(2, key_b));
    cache.entries.items[1].checked_out_by = 7;
    try testing.expectEqual(@as(?usize, 2), cache.lruIndexExcluding(null, key_b));
    cache.entries.items[1].checked_out_by = null;
    // One workload = plain LRU: C goes first.
    for (cache.entries.items) |*e| e.cache_key = 0;
    try testing.expectEqual(@as(?usize, 0), cache.lruIndexExcluding(null, 0));
}

test "HotPrefixCache: checkpoint clone allocation failure releases owned layers" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var accounting = testing.FailingAllocator.init(arena.allocator(), .{ .fail_index = 1 });
    var layers = [_]transformer_mod.SSMCacheEntrySnapshot{.{
        .conv_state = .{ .ctx = null },
        .ssm_state = .{ .ctx = null },
        .initialized = false,
    }};
    const checkpoints = [_]SSMCheckpoint{.{ .pos = 512, .layers = &layers }};

    try testing.expectError(error.OutOfMemory, HotPrefixCache.cloneCheckpointsUpTo(accounting.allocator(), &checkpoints, 512, null));
    try testing.expectEqual(accounting.allocated_bytes, accounting.freed_bytes);
}

test "hot cache declines when checked out entries fill count cap" {
    try testCheckedOutRetentionCap(1, 0);
}

test "hot cache declines when checked out entries fill byte cap" {
    try testCheckedOutRetentionCap(8, 4096);
}

fn testCheckedOutRetentionCap(max_entries: u32, max_bytes: u64) !void {
    var hc = HotPrefixCache.initWithMem(testing.allocator, max_entries, max_bytes);
    defer hc.deinit();
    try pcAppendCheckedOut(&hc, &.{ 1, 2 }, max_bytes, 1);
    const retained = try hc.retainNewEntry(.{
        .tokens = try testing.allocator.dupe(u32, &.{ 90, 91, 92, 93 }),
        .has_tools = false,
        .snapshot = .{ .entries = try testing.allocator.alloc(transformer_mod.KVCacheEntry, 0), .step = 4, .allocator = testing.allocator, .config = kv_quant.KVQuantConfig.dense },
        .last_used = 2,
        .quant_config = kv_quant.KVQuantConfig.dense,
        .kv_bytes = 128,
        .ssm_checkpoints = try testing.allocator.alloc(SSMCheckpoint, 0),
        .ssm_bytes = 0,
    });
    try testing.expect(!retained);
    try testing.expectEqual(@as(usize, 1), hc.entries.items.len);
    try testing.expectEqual(max_bytes, hc.current_kv_bytes);
    try testing.expect(hc.entries.items[0].checked_out_by != null);
}

test "a restore names its entry, and a commit that extends it in place keeps the id" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var tokens: [64]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var longer: [80]u32 = undefined;
    for (&longer, 0..) |*t, i| t.* = @intCast(i + 7);
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    try testCheckoutCache(&hc, s, &tokens, 64);
    const id = hc.entries.items[0].id;
    try testing.expect(id != 0);

    var slot = try KVCache.init(testing.allocator, 1);
    defer slot.deinit();
    var moe_off: usize = 0;
    const res = try hc.lookupAndRestoreForSlot(&slot, &moe_off, null, s, &longer, false, 0, null, null, null, 0);
    try testing.expectEqual(id, res.entry_id);

    try testFillCache(&slot, s, 1, longer.len);
    _ = try hc.commit(&slot, &longer, false);
    try testing.expectEqual(@as(usize, 1), hc.entries.items.len);
    try testing.expectEqual(id, hc.entries.items[0].id);
}

test "a donated checkout names the bytes its slot now holds; a share names none" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var prompt: [608]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(i + 7);
    restore_move_override = true;
    defer restore_move_override = null;
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.ssd_first = true;
    defer hc.deinit();
    try testCheckoutCache(&hc, s, &tokens, 1024);

    var slot = try KVCache.init(testing.allocator, 1);
    defer slot.deinit();
    var moe_off: usize = 0;
    const res = try hc.lookupAndRestoreForSlot(&slot, &moe_off, null, s, &prompt, false, 0, null, null, null, 0xA11CE);
    try testing.expect(res.checked_out);
    // Until the transfer the slot shares the entry's buffers, which the entry bills.
    try testing.expectEqual(@as(u64, 0), slot.residentBytes());
    try testing.expectEqual(@as(u64, 0), hc.donatedBytes(0xA11CE));

    const billed = hc.residentBytes();
    hc.donateCheckout(0xA11CE);
    slot.adoptRestored();
    // The entry keeps billing what the slot now owns, so a live count subtracts it.
    try testing.expectEqual(billed, hc.residentBytes());
    try testing.expect(slot.residentBytes() > 0);
    try testing.expectEqual(slot.residentBytes(), hc.donatedBytes(0xA11CE));
    try testing.expectEqual(@as(u64, 0), hc.donatedBytes(0xB0B));

    hc.releaseCheckout(0xA11CE, "test");
    try testing.expectEqual(@as(u64, 0), hc.donatedBytes(0xA11CE));
}

test "HotPrefixCache: disk-only commit persists the full prefix and retains no RAM entry" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const base = buf[0..root_len];

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 0, 0);
        try testing.expectEqual(@as(u32, 0), hc.max_entries);
        try testing.expect(!hc.ram_enabled);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-disk-only", 0, 128);
        hc.disk.?.ssd_first = true;
        hc.disk.?.enableBackgroundWriter();
        try testing.expect(hc.disk.?.writer != null);
        defer hc.deinit();

        var cache = try KVCache.init(testing.allocator, 2);
        defer cache.deinit();
        try testFillCache(&cache, s, 2, tokens.len);
        const status = try hc.commit(&cache, &tokens, false);
        try testing.expectEqual(std.meta.Tag(CommitStatus).disk_only, std.meta.activeTag(status));
        try testing.expectEqual(@as(usize, tokens.len), status.disk_only);
        try testing.expectEqual(@as(usize, 0), hc.entryCount());
        try testing.expectEqual(@as(u64, 0), hc.residentBytes());
        try testing.expect(hc.pending_disk != null);

        hc.flushPendingDisk(s);
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    }

    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 0, 0);
        try testing.expectEqual(@as(u32, 0), hc.max_entries);
        try testing.expect(!hc.ram_enabled);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-disk-only", 0, 128);
        hc.disk.?.ssd_first = true;
        hc.disk.?.enableBackgroundWriter();
        try testing.expect(hc.disk.?.writer != null);
        defer hc.deinit();

        var cache = try KVCache.init(testing.allocator, 2);
        defer cache.deinit();
        var moe_offset: usize = 0;
        const restored = try hc.lookupAndRestore(&cache, &moe_offset, null, s, &tokens, false, 0, null, null);
        try testing.expectEqual(@as(usize, tokens.len - 1), restored.matched);
        try testing.expectEqual(@as(usize, 0), hc.entryCount());
        try testing.expectEqual(@as(u64, 0), hc.residentBytes());
    }
}

test "HotPrefixCache: SSD-only LFM2 conv state survives background flush and restart" {
    const a = testing.allocator;
    const io = testing.io;
    const s = mlx.gpuStream();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var path: [512]u8 = undefined;
    const base = path[0..try tmp.dir.realPath(io, &path)];
    var config = model_mod.ModelConfig{ .model_type = "lfm2", .has_hybrid_layers = true };
    try testing.expect(HotPrefixCache.shouldUse(&config, true));
    try testing.expect(ssdFirstActive(&config, true, false));
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*token, i| token.* = @intCast(i + 7);
    for (0..2) |boot| {
        var hc = HotPrefixCache.initWithMem(a, 0, 0);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(a, io, base, "lfm-conv", 0, 128);
        hc.disk.?.ssd_first = true;
        hc.disk.?.enableBackgroundWriter();
        defer hc.deinit();
        try testing.expect(hc.disk.?.writer != null);
        var cache = try KVCache.init(a, 3);
        defer cache.deinit();
        if (boot == 0) {
            var written: u32 = 0;
            while (written < tokens.len) : (written += 64) {
                try testWriteCacheLayer(&cache, s, 1, written, @intCast(@min(64, tokens.len - written)));
            }
            var source = pcEmptySsm();
            defer pcFreeHybrid(&source);
            _ = mlx.mlx_array_free(source[0].conv_state);
            source[0].conv_state = pcArange(s, &conv_shape_pc, 300);
            source[0].initialized = true;
            const cps = try a.alloc(SSMCheckpoint, 1);
            cps[0] = try transformer_mod.captureSsmCheckpoint(a, &source, 512, s);
            const status = try hc.commitWithSsm(&cache, &tokens, false, cps, null, null);
            try testing.expectEqual(std.meta.Tag(CommitStatus).disk_only, std.meta.activeTag(status));
            try testing.expectEqual(std.meta.Tag(CommitStatus).disk_only, std.meta.activeTag(status));
            try testing.expectEqual(@as(usize, tokens.len), status.disk_only);
            hc.flushPendingDisk(s);
            hc.disk.?.drainWriter();
            try testing.expectEqual(@as(u64, 0), hc.disk.?.writeErrors());
        } else {
            var restored = pcEmptySsm();
            defer pcFreeHybrid(&restored);
            var offset: usize = 0;
            const hit = try hc.lookupAndRestore(&cache, &offset, &restored, s, &tokens, false, 0, null, null);
            try testing.expectEqual(@as(usize, 512), hit.matched);
            try testing.expectEqual(@as(usize, 512), offset);
            try testing.expect(!cache.entries[0].initialized);
            try testing.expect(cache.entries[1].initialized);
            try testing.expectEqual(@as(usize, 512), cache.entries[1].offset);
            try testing.expect(restored[0].initialized);
            try testing.expectEqual(@as(f32, 300), pcSsmVal(restored[0].conv_state, 0, s));
            try testing.expect(restored[0].ssm_state.ctx == null);
        }
        try testing.expectEqual(@as(usize, 0), hc.entryCount());
        try testing.expectEqual(@as(u64, 0), hc.residentBytes());
    }
}

test "SSD-only ring checkpoints survive restart without idle RAM" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = testing.io;
    const s = mlx.gpuStream();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var path: [512]u8 = undefined;
    const base = path[0..try tmp.dir.realPath(io, &path)];
    const prompt: u32 = 700;
    const total: u32 = 1300;
    const window: u32 = 8;
    const layers: u32 = 4;
    var tokens: [total]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 1);
    for (0..2) |boot| {
        var hc = HotPrefixCache.initWithMem(a, 0, 0);
        hc.ssd_first = true;
        hc.disk = try kv_disk_cache.DiskTier.init(a, io, base, "ssd-only-ring", 0, 128);
        hc.disk.?.ssd_first = true;
        hc.disk.?.enableBackgroundWriter();
        defer hc.deinit();
        var cache = try KVCache.init(a, layers);
        defer cache.deinit();
        cache.setSwaRing(window);
        if (boot == 0) {
            try ringFill(&cache, s, layers, window, 0, prompt, 16);
            const cp = try cache.ringCheckpoint(prompt, s);
            try ringFill(&cache, s, layers, window, prompt, total, 16);
            const status = try hc.commitWithRing(&cache, &tokens, false, 0, 0, null, null, null, null, prompt, .{ .prompt_end = cp });
            try testing.expectEqual(std.meta.Tag(CommitStatus).disk_only, std.meta.activeTag(status));
            hc.flushPendingDisk(s);
            hc.disk.?.drainWriter();
            try testing.expectEqual(@as(u64, 0), hc.disk.?.writeErrors());
            // Once the writer has landed them, no restore point is held in RAM.
            try testing.expect(hc.pending_disk == null);
            try testing.expectEqual(@as(u64, 0), hc.disk.?.stagedHostBytes());
        } else {
            var next: [800]u32 = undefined;
            @memcpy(next[0..prompt], tokens[0..prompt]);
            for (next[prompt..], 0..) |*t, i| t.* = @intCast(90_000 + i);
            var offset: usize = 0;
            const hit = try hc.lookupAndRestore(&cache, &offset, null, s, &next, false, 0, null, null);
            try testing.expectEqual(@as(usize, prompt), hit.matched);
            try testing.expect(!hit.ownsRestoredRows());
            try testing.expectEqual(@as(usize, prompt), offset);
        }
        try testing.expectEqual(@as(usize, 0), hc.entryCount());
        try testing.expectEqual(@as(u64, 0), hc.residentBytes());
    }
}

/// A GLM fixture turn: `tokens` prefilled in `chunks`, a KDA checkpoint after each chunk listed in
/// `checkpoints` (chunk indices), MLA rows to the request's last pool boundary.
const GlmTurn = struct {
    cps: []SSMCheckpoint,
    rows: glm5_prefix.MlaRows,
    cache: KVCache,

    fn run(net: *const glm5_forward.Model, bits: u8, tokens: []const u32, chunks: []const usize, checkpoints: []const usize) !GlmTurn {
        var req = try glm5_prefix.servedRequest(bits);
        defer req.deinit();
        var cps = try testing.allocator.alloc(SSMCheckpoint, checkpoints.len);
        var built: usize = 0;
        errdefer {
            for (cps[0..built]) |*cp| cp.deinit(testing.allocator);
            testing.allocator.free(cps);
        }
        var at: usize = 0;
        for (chunks, 0..) |n, i| {
            _ = mlx.mlx_array_free(try glm5_prefix.prefill(net, &req, tokens[at .. at + n], &.{n}));
            at += n;
            if (std.mem.indexOfScalar(usize, checkpoints, i) != null) {
                cps[built] = try glm5_prefix.captureKda(testing.allocator, &req, mlx.gpuStream());
                built += 1;
            }
        }
        var rows = try glm5_prefix.MlaRows.capture(testing.allocator, &req, req.offset / 4 * 4, mlx.gpuStream());
        errdefer rows.deinit();
        var cache = try KVCache.initWithConfig(testing.allocator, 4, glmKvConfig(bits));
        cache.step = req.offset;
        return .{ .cps = cps, .rows = rows, .cache = cache };
    }

    /// Hands `cps` and `rows` to the cache.
    fn commit(self: *GlmTurn, hc: *HotPrefixCache, tokens: []const u32, vision_key: u64, media_start: ?usize) !CommitStatus {
        defer self.cache.deinit();
        return hc.commitGlm(&self.cache, tokens, false, vision_key, 0, media_start, self.cps, self.rows, null, tokens.len);
    }
};

fn glmKvConfig(bits: u8) kv_quant.KVQuantConfig {
    return if (bits == 0) kv_quant.KVQuantConfig.dense else kv_quant.KVQuantConfig.affine(8);
}

const glm_first = [_]u32{ 1, 2, 3, 0, 2, 2, 1, 3, 0, 1, 1, 2, 3, 3, 0, 2, 1, 3, 0, 0 };
const glm_next = [_]u32{ 1, 2, 3, 0, 2, 2, 1, 3, 0, 1, 1, 2, 3, 3, 0, 2, 1, 3, 2, 1, 3, 1, 0, 2 };
const glm_other = [_]u32{ 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 };

fn glmLookup(hc: *HotPrefixCache, req: *glm5_forward.Request, bits: u8, prompt: []const u32, vision_key: u64, media_start: ?usize, chain: []const MediaSpan) !LookupResult {
    var target = try KVCache.initWithConfig(testing.allocator, 4, glmKvConfig(bits));
    defer target.deinit();
    var moe: usize = 0;
    const res = try hc.lookupAndRestoreGlm(&target, &moe, req, mlx.gpuStream(), prompt, false, vision_key, media_start, chain, null, null, false);
    try testing.expectEqual(res.matched, target.step);
    try testing.expectEqual(res.matched, req.offset);
    return res;
}

const GlmNet = struct {
    weights: model_mod.Weights,
    net: glm5_forward.Model,

    fn load(self: *GlmNet) !void {
        self.weights = model_mod.Weights.init(testing.allocator);
        errdefer self.weights.deinit();
        var cfg = try glm5_forward.nonzeroDecodeFixture(&self.weights);
        cfg.max_position_embeddings = 4096;
        self.net = try glm5_forward.Model.load(testing.allocator, cfg, &self.weights, mlx.gpuStream());
    }
    fn deinit(self: *GlmNet) void {
        self.net.deinit();
        self.weights.deinit();
    }
};

test "GLM hot cache restores the highest checkpoint under the match and continues like a cold prefill" {
    var fx: GlmNet = undefined;
    try fx.load();
    defer fx.deinit();
    for ([_]u8{ 0, 8 }) |bits| {
        var hc = HotPrefixCache.init(testing.allocator, 4);
        defer hc.deinit();
        var turn = try GlmTurn.run(&fx.net, bits, &glm_first, &.{ 8, 8, 4 }, &.{ 0, 1 });
        try testing.expectEqual(CommitStatus{ .ok = glm_first.len }, try turn.commit(&hc, &glm_first, 0, null));
        const e = &hc.entries.items[0];
        var cps_bytes: u64 = 0;
        for (e.ssm_checkpoints.?) |*cp| cps_bytes += ssmCheckpointBytes(cp);
        try testing.expectEqual(e.glm.?.bytes() + cps_bytes, e.kv_bytes);
        try testing.expectEqual(e.kv_bytes, hc.current_kv_bytes);

        var warm = try glm5_prefix.servedRequest(bits);
        defer warm.deinit();
        const res = try glmLookup(&hc, &warm, bits, &glm_next, 0, null, &.{});
        try testing.expectEqual(@as(usize, 16), res.matched);
        const got = try glm5_prefix.prefill(&fx.net, &warm, glm_next[16..], &.{8});
        defer _ = mlx.mlx_array_free(got);
        var cold = try glm5_prefix.servedRequest(bits);
        defer cold.deinit();
        const want = try glm5_prefix.prefill(&fx.net, &cold, &glm_next, &.{ 8, 8, 8 });
        defer _ = mlx.mlx_array_free(want);
        try glm5_forward.expectArrayBits(want, got);
        try glm5_prefix.expectSameLogicalState(&cold, &warm);

        // The other latent width never restores this entry, and its miss leaves the request cold.
        const other_bits: u8 = if (bits == 0) 8 else 0;
        var other = try glm5_prefix.servedRequest(other_bits);
        defer other.deinit();
        try testing.expectEqual(@as(usize, 0), (try glmLookup(&hc, &other, other_bits, &glm_next, 0, null, &.{})).matched);
    }
}

test "GLM hot cache trims an entry over the byte budget to a checkpoint and bills what it keeps" {
    var fx: GlmNet = undefined;
    try fx.load();
    defer fx.deinit();
    var long: [300]u32 = undefined;
    for (&long, 0..) |*t, i| t.* = @intCast((i * 7 + i / 5) % 4);
    var probe = try GlmTurn.run(&fx.net, 8, &long, &.{ 264, 32, 4 }, &.{ 0, 1 });
    const row = probe.rows.rowBytes();
    const per_cp = ssmCheckpointBytes(&probe.cps[0]);
    probe.cache.deinit();
    probe.rows.deinit();
    for (probe.cps) |*cp| cp.deinit(testing.allocator);
    testing.allocator.free(probe.cps);

    // The assistant window survives the trim and is priced inside the budget.
    var window = try KVCache.init(testing.allocator, 2);
    defer window.deinit();
    try testFillCache(&window, mlx.gpuStream(), 2, 64);
    var window_snap = try window.snapshot();
    defer window_snap.deinit();
    const window_bytes = HotPrefixCache.snapshotBytes(&window_snap);

    // Room for the window, the rows to 264 and one checkpoint, not for 300 rows and two.
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, window_bytes + 264 * row + per_cp);
    defer hc.deinit();
    var turn = try GlmTurn.run(&fx.net, 8, &long, &.{ 264, 32, 4 }, &.{ 0, 1 });
    defer turn.cache.deinit();
    try testing.expectEqual(CommitStatus{ .ok = 264 }, try hc.commitGlm(&turn.cache, &long, false, 0, 0, null, turn.cps, turn.rows, .{ .cache = &window, .base_pos = 236 }, long.len));
    const e = &hc.entries.items[0];
    try testing.expectEqual(@as(usize, 264), e.glm.?.len);
    try testing.expect(e.dflash != null);
    try testing.expectEqual(window_bytes + 264 * row + per_cp, e.kv_bytes);
    try testing.expect(hc.current_kv_bytes <= hc.max_kv_bytes);
    var warm = try glm5_prefix.servedRequest(8);
    defer warm.deinit();
    try testing.expectEqual(@as(usize, 264), (try glmLookup(&hc, &warm, 8, &long, 0, null, &.{})).matched);
}

test "GLM hot cache evicts by count, invalidates, and a miss leaves the request cold" {
    var fx: GlmNet = undefined;
    try fx.load();
    defer fx.deinit();
    var hc = HotPrefixCache.init(testing.allocator, 1);
    defer hc.deinit();
    var a = try GlmTurn.run(&fx.net, 0, &glm_first, &.{ 8, 8, 4 }, &.{ 0, 1 });
    _ = try a.commit(&hc, &glm_first, 0, null);
    var b = try GlmTurn.run(&fx.net, 0, &glm_other, &.{ 8, 4 }, &.{0});
    _ = try b.commit(&hc, &glm_other, 0, null);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    var warm = try glm5_prefix.servedRequest(0);
    defer warm.deinit();
    try testing.expectEqual(@as(usize, 0), (try glmLookup(&hc, &warm, 0, &glm_next, 0, null, &.{})).matched);
    const again = glm_other ++ [_]u32{1};
    try testing.expectEqual(@as(usize, 8), (try glmLookup(&hc, &warm, 0, &again, 0, null, &.{})).matched);
    hc.invalidateAll("test");
    try testing.expectEqual(@as(u64, 0), hc.current_kv_bytes);
    try testing.expectEqual(@as(usize, 0), (try glmLookup(&hc, &warm, 0, &again, 0, null, &.{})).matched);
    try testing.expectEqual(@as(usize, 0), warm.layers[3].attention.processed);
}

test "GLM hot cache gives entries back LRU first to admit a long prefill, sparing the one it restored from" {
    var fx: GlmNet = undefined;
    try fx.load();
    defer fx.deinit();
    var hc = HotPrefixCache.init(testing.allocator, 4);
    defer hc.deinit();
    var a = try GlmTurn.run(&fx.net, 0, &glm_first, &.{ 8, 8, 4 }, &.{ 0, 1 });
    _ = try a.commit(&hc, &glm_first, 0, null);
    var b = try GlmTurn.run(&fx.net, 0, &glm_other, &.{ 8, 4 }, &.{0});
    _ = try b.commit(&hc, &glm_other, 0, null);
    var warm = try glm5_prefix.servedRequest(0);
    defer warm.deinit();
    const again = glm_other ++ [_]u32{1};
    try testing.expectEqual(@as(usize, 8), (try glmLookup(&hc, &warm, 0, &again, 0, null, &.{})).matched);
    const Room = struct {
        fn oneEntry(ctx: ?*anyopaque) bool {
            const cache: *HotPrefixCache = @ptrCast(@alignCast(ctx.?));
            return cache.entryCount() <= 1;
        }
        fn never(_: ?*anyopaque) bool {
            return false;
        }
    };
    const report = hc.evictLruToAdmit(again.len, &hc, Room.oneEntry, true);
    try testing.expect(report.admitted);
    try testing.expectEqual(@as(usize, 1), report.entries);
    try testing.expectEqual(@as(usize, 8), (try glmLookup(&hc, &warm, 0, &again, 0, null, &.{})).matched);
    try testing.expect(!hc.evictLruToAdmit(again.len, &hc, Room.never, true).admitted);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
}

test "GLM hot cache shares across media keys only below the first block the request lacks" {
    var fx: GlmNet = undefined;
    try fx.load();
    defer fx.deinit();
    var hc = HotPrefixCache.init(testing.allocator, 4);
    defer hc.deinit();
    // One media block at 10 (key 0xA); the next turn appends a second block at 20 (key 0xB).
    var turn = try GlmTurn.run(&fx.net, 0, &glm_first, &.{ 8, 8, 4 }, &.{ 0, 1 });
    _ = try turn.commit(&hc, &glm_first, 0xA, 10);
    var warm = try glm5_prefix.servedRequest(0);
    defer warm.deinit();
    const chain = [_]MediaSpan{ .{ .start = 10, .key = 0xA }, .{ .start = 20, .key = 0xB } };
    try testing.expectEqual(@as(usize, 16), (try glmLookup(&hc, &warm, 0, &glm_next, 0xB, 10, &chain)).matched);
    const stranger = [_]MediaSpan{.{ .start = 10, .key = 0xC }};
    try testing.expectEqual(@as(usize, 8), (try glmLookup(&hc, &warm, 0, &glm_next, 0xC, 10, &stranger)).matched);
}

test "GLM hot cache checkout hands the MLA rows to the slot so its first append donates" {
    var fx: GlmNet = undefined;
    try fx.load();
    defer fx.deinit();
    restore_move_override = true;
    defer restore_move_override = null;
    var hc = HotPrefixCache.init(testing.allocator, 4);
    defer hc.deinit();
    hc.ssd_first = true;
    var turn = try GlmTurn.run(&fx.net, 8, &glm_first, &.{ 8, 8, 4 }, &.{ 0, 1 });
    _ = try turn.commit(&hc, &glm_first, 0, null);
    const billed = hc.entries.items[0].glm_bytes;
    var warm = try glm5_prefix.servedRequest(8);
    defer warm.deinit();
    var target = try KVCache.initWithConfig(testing.allocator, 4, kv_quant.KVQuantConfig.affine(8));
    defer target.deinit();
    var moe: usize = 0;
    const longer = glm_first ++ [_]u32{ 2, 2, 2, 2 };
    const res = try hc.lookupAndRestoreGlm(&target, &moe, &warm, mlx.gpuStream(), &longer, false, 0, null, &.{}, null, 7, false);
    try testing.expect(res.checked_out);
    try testing.expectEqual(@as(usize, 16), res.matched);
    hc.donateCheckout(7);
    try testing.expectEqual(@as(u64, 0), hc.entries.items[0].glm.?.bytes());
    try testing.expectEqual(billed, hc.donatedBytes(7));
    hc.releaseCheckout(7, "test");
    try testing.expectEqual(@as(usize, 0), hc.entryCount());
}

test "GLM SSD tier restores a checkpoint after the RAM copy is gone and continues like a cold prefill" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    var fx: GlmNet = undefined;
    try fx.load();
    defer fx.deinit();
    var first: [700]u32 = undefined;
    for (&first, 0..) |*t, i| t.* = @intCast((i * 7 + i / 5) % 4);
    var next: [750]u32 = undefined;
    @memcpy(next[0..650], first[0..650]);
    for (next[650..], 0..) |*t, i| t.* = @intCast((i * 3 + 1) % 4);
    for ([_]u8{ 0, 8 }) |bits| {
        const fp: []const u8 = if (bits == 0) "fp-glm-bf16" else "fp-glm-kv8";
        {
            var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
            hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, fp, 0, 128);
            defer hc.deinit();
            var turn = try GlmTurn.run(&fx.net, bits, &first, &.{ 256, 256, 188 }, &.{ 0, 1 });
            _ = try turn.commit(&hc, &first, 0, null);
            hc.flushPendingDisk(s);
            try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
        }
        // A restart: nothing in RAM.
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, fp, 0, 128);
        defer hc.deinit();
        var warm = try glm5_prefix.servedRequest(bits);
        defer warm.deinit();
        const res = try glmLookup(&hc, &warm, bits, &next, 0, null, &.{});
        try testing.expectEqual(@as(usize, 512), res.matched);
        try testing.expect(res.ownsRestoredRows());
        const got = try glm5_prefix.prefill(&fx.net, &warm, next[512..], &.{238});
        defer _ = mlx.mlx_array_free(got);
        var cold = try glm5_prefix.servedRequest(bits);
        defer cold.deinit();
        const want = try glm5_prefix.prefill(&fx.net, &cold, &next, &.{ 256, 256, 238 });
        defer _ = mlx.mlx_array_free(want);
        try glm5_forward.expectArrayBits(want, got);
        try glm5_prefix.expectSameLogicalState(&cold, &warm);
        // The other latent width finds nothing on disk.
        var other = try glm5_prefix.servedRequest(if (bits == 0) 8 else 0);
        defer other.deinit();
        try testing.expectEqual(@as(usize, 0), (try glmLookup(&hc, &other, if (bits == 0) 8 else 0, &next, 0, null, &.{})).matched);
    }
}

test "GLM entries carry the assistant window, and a restore hands back its rows below the checkpoint, RAM and SSD" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    var fx: GlmNet = undefined;
    try fx.load();
    defer fx.deinit();
    var first: [700]u32 = undefined;
    for (&first, 0..) |*t, i| t.* = @intCast((i * 7 + i / 5) % 4);
    var next: [750]u32 = undefined;
    @memcpy(next[0..650], first[0..650]);
    for (next[650..], 0..) |*t, i| t.* = @intCast((i * 3 + 1) % 4);
    // The window as prefill left it: 300 rows ending at the prompt end.
    var window = try KVCache.init(testing.allocator, 2);
    defer window.deinit();
    try testFillCache(&window, s, 2, 300);
    var committing = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    committing.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-spec", 0, 128);
    var turn = try GlmTurn.run(&fx.net, 8, &first, &.{ 256, 256, 188 }, &.{ 0, 1 });
    defer turn.cache.deinit();
    _ = try committing.commitGlm(&turn.cache, &first, false, 0, 0, null, turn.cps, turn.rows, .{ .cache = &window, .base_pos = 400 }, first.len);
    committing.flushPendingDisk(s);
    var restarted = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer restarted.deinit();
    for ([_]*HotPrefixCache{ &committing, &restarted }) |hc| {
        if (hc == &restarted) {
            committing.deinit();
            hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-spec", 0, 128);
        }
        var drafter = try KVCache.init(testing.allocator, 2);
        defer drafter.deinit();
        var drafter_base: usize = 0;
        var warm = try glm5_prefix.servedRequest(8);
        defer warm.deinit();
        var target = try KVCache.initWithConfig(testing.allocator, 4, kv_quant.KVQuantConfig.affine(8));
        defer target.deinit();
        var moe: usize = 0;
        const res = try hc.lookupAndRestoreGlm(&target, &moe, &warm, s, &next, false, 0, null, &.{}, .{ .cache = &drafter, .base_pos = &drafter_base }, null, false);
        try testing.expectEqual(@as(usize, 512), res.matched);
        try testing.expectEqual(@as(?usize, 400), res.dflash_base);
        try testing.expectEqual(@as(usize, 112), drafter.step);
    }
}

fn glmSsdOnlyCache(io: std.Io, base: []const u8, fp: []const u8, max_bytes: u64, budget_from_volume: bool) !HotPrefixCache {
    var hc = HotPrefixCache.initWithMem(testing.allocator, 0, 0);
    errdefer hc.deinit();
    hc.ssd_first = true;
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, fp, max_bytes, 128);
    hc.disk.?.ssd_first = budget_from_volume;
    if (budget_from_volume) hc.disk.?.armTestSpace(256 << 30, 1 << 40);
    hc.disk.?.enableBackgroundWriter();
    return hc;
}

test "GLM SSD-only: a commit reaches disk with no idle RAM entry, and a restart restores it like a cold prefill" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    var fx: GlmNet = undefined;
    try fx.load();
    defer fx.deinit();
    var first: [700]u32 = undefined;
    for (&first, 0..) |*t, i| t.* = @intCast((i * 7 + i / 5) % 4);
    var next: [750]u32 = undefined;
    @memcpy(next[0..650], first[0..650]);
    for (next[650..], 0..) |*t, i| t.* = @intCast((i * 3 + 1) % 4);
    var window = try KVCache.init(testing.allocator, 2);
    defer window.deinit();
    try testFillCache(&window, s, 2, 300);
    for ([_]u8{ 0, 8 }) |bits| {
        const fp: []const u8 = if (bits == 0) "glm-ssd-only-bf16" else "glm-ssd-only-kv8";
        {
            var hc = try glmSsdOnlyCache(io, base, fp, 0, true);
            defer hc.deinit();
            var turn = try GlmTurn.run(&fx.net, bits, &first, &.{ 256, 256, 188 }, &.{ 0, 1 });
            defer turn.cache.deinit();
            const status = try hc.commitGlm(&turn.cache, &first, false, 0, 0, null, turn.cps, turn.rows, .{ .cache = &window, .base_pos = 400 }, first.len);
            try testing.expectEqual(CommitStatus{ .disk_only = first.len }, status);
            try testing.expectEqual(@as(usize, 0), hc.entryCount());
            try testing.expectEqual(@as(u64, 0), hc.residentBytes());
            hc.flushPendingDisk(s);
            hc.disk.?.drainWriter();
            try testing.expectEqual(@as(u64, 0), hc.disk.?.writeErrors());
            try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
            try testing.expect(hc.pending_disk == null);
        }
        var hc = try glmSsdOnlyCache(io, base, fp, 0, true);
        defer hc.deinit();
        var drafter = try KVCache.init(testing.allocator, 2);
        defer drafter.deinit();
        var drafter_base: usize = 0;
        var warm = try glm5_prefix.servedRequest(bits);
        defer warm.deinit();
        var target = try KVCache.initWithConfig(testing.allocator, 4, glmKvConfig(bits));
        defer target.deinit();
        var moe: usize = 0;
        const res = try hc.lookupAndRestoreGlm(&target, &moe, &warm, s, &next, false, 0, null, &.{}, .{ .cache = &drafter, .base_pos = &drafter_base }, 9, false);
        try testing.expectEqual(@as(usize, 512), res.matched);
        try testing.expect(res.ownsRestoredRows());
        try testing.expectEqual(@as(?usize, 400), res.dflash_base);
        try testing.expectEqual(@as(usize, 112), drafter.step);
        try testing.expectEqual(@as(usize, 0), hc.entryCount());
        const got = try glm5_prefix.prefill(&fx.net, &warm, next[512..], &.{238});
        defer _ = mlx.mlx_array_free(got);
        var cold = try glm5_prefix.servedRequest(bits);
        defer cold.deinit();
        const want = try glm5_prefix.prefill(&fx.net, &cold, &next, &.{ 256, 256, 238 });
        defer _ = mlx.mlx_array_free(want);
        try glm5_forward.expectArrayBits(want, got);
        try glm5_prefix.expectSameLogicalState(&cold, &warm);
    }
}

test "GLM: the SSD tier gets every row in one flush whatever the RAM tier keeps, and the longer tier restores" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    var fx: GlmNet = undefined;
    try fx.load();
    defer fx.deinit();
    var first: [700]u32 = undefined;
    for (&first, 0..) |*t, i| t.* = @intCast((i * 7 + i / 5) % 4);
    var next: [750]u32 = undefined;
    @memcpy(next[0..650], first[0..650]);
    for (next[650..], 0..) |*t, i| t.* = @intCast((i * 3 + 1) % 4);
    const Arm = struct { ram: bool, fp: []const u8 };
    for ([_]u8{ 0, 8 }) |bits| for ([_]Arm{ .{ .ram = true, .fp = "glm-tiers-both" }, .{ .ram = false, .fp = "glm-tiers-ssd" } }) |arm| {
        var fp_buf: [32]u8 = undefined;
        const fp = try std.fmt.bufPrint(&fp_buf, "{s}-{d}", .{ arm.fp, bits });
        var hc = if (arm.ram) HotPrefixCache.initWithMem(testing.allocator, 4, 0) else try glmSsdOnlyCache(io, base, fp, 0, false);
        defer hc.deinit();
        if (arm.ram) {
            hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, fp, 0, 128);
            hc.disk.?.enableBackgroundWriter();
        }
        hc.disk.?.max_flush_bytes = 1; // one chunk per piece
        var turn = try GlmTurn.run(&fx.net, bits, &first, &.{ 256, 256, 188 }, &.{ 0, 1 });
        defer turn.cache.deinit();
        // The RAM tier keeps the rows through the first checkpoint; the SSD tier gets them all.
        const ram_rows: ?glm5_prefix.MlaRows = if (arm.ram) try turn.rows.trimmedCopy(256, s) else null;
        const status = try hc.commitGlmTiers(&turn.cache, &first, false, 0, 0, null, turn.cps, ram_rows, turn.rows, null, first.len);
        if (arm.ram) {
            try testing.expectEqual(CommitStatus{ .ok = first.len }, status);
            try testing.expectEqual(@as(usize, 256), hc.entries.items[0].glm.?.len);
            try testing.expectEqual(@as(usize, 1), hc.entries.items[0].ssm_checkpoints.?.len);
        } else {
            try testing.expectEqual(CommitStatus{ .disk_only = first.len }, status);
            try testing.expectEqual(@as(usize, 0), hc.entryCount());
        }
        hc.flushPendingDisk(s);
        hc.disk.?.drainWriter();
        try testing.expect(!hc.disk_dirty);
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
        const on_disk = &hc.disk.?.entries.items[0];
        try testing.expectEqual(@as(u32, 700), on_disk.kv_len);
        try testing.expectEqualSlices(u32, &.{ 256, 512 }, on_disk.ssm_positions);

        // The SSD tier's checkpoint at 512 beats the RAM entry's at 256.
        var warm = try glm5_prefix.servedRequest(bits);
        defer warm.deinit();
        const res = try glmLookup(&hc, &warm, bits, &next, 0, null, &.{});
        try testing.expectEqual(@as(usize, 512), res.matched);
        try testing.expect(res.ownsRestoredRows());
        const got = try glm5_prefix.prefill(&fx.net, &warm, next[512..], &.{238});
        defer _ = mlx.mlx_array_free(got);
        var cold = try glm5_prefix.servedRequest(bits);
        defer cold.deinit();
        const want = try glm5_prefix.prefill(&fx.net, &cold, &next, &.{ 256, 256, 238 });
        defer _ = mlx.mlx_array_free(want);
        try glm5_forward.expectArrayBits(want, got);
        try glm5_prefix.expectSameLogicalState(&cold, &warm);
    };
}

/// A GLM prefill driven chunk by chunk as the scheduler drives it: a KDA checkpoint after each chunk, then the write-through.
const GlmPrefill = struct {
    req: glm5_forward.Request,
    cps: std.ArrayList(SSMCheckpoint) = .empty,
    entry_id: ?u64 = null,
    /// Most checkpoint states in RAM at once, read right after a capture.
    peak_held: usize = 0,

    /// The last chunk is the loop's end: its checkpoint is taken after the loop, past the hook, and the
    /// rest of `tokens` is the final span.
    fn run(hc: *HotPrefixCache, net: *const glm5_forward.Model, bits: u8, tokens: []const u32, chunks: []const usize) !GlmPrefill {
        var self = GlmPrefill{ .req = try glm5_prefix.servedRequest(bits) };
        errdefer self.deinit();
        var at: usize = 0;
        for (chunks, 0..) |n, i| {
            _ = mlx.mlx_array_free(try glm5_prefix.prefill(net, &self.req, tokens[at .. at + n], &.{n}));
            at += n;
            if (i + 1 < chunks.len) try self.capture();
            hc.writeThroughGlm(&self.entry_id, &self.req, tokens[0..at], false, .{}, self.cps.items, mlx.gpuStream());
        }
        try self.capture();
        if (at < tokens.len) _ = mlx.mlx_array_free(try glm5_prefix.prefill(net, &self.req, tokens[at..], &.{tokens.len - at}));
        return self;
    }

    fn capture(self: *GlmPrefill) !void {
        try self.cps.append(testing.allocator, try glm5_prefix.captureKda(testing.allocator, &self.req, mlx.gpuStream()));
        self.peak_held = @max(self.peak_held, self.held());
    }

    fn held(self: *const GlmPrefill) usize {
        var n: usize = 0;
        for (self.cps.items) |*cp| n += @intFromBool(!cp.isReleased());
        return n;
    }

    /// The request's end-of-request commit for the SSD tier alone, as `commitGlmSlot` builds it.
    fn commitSsdOnly(self: *GlmPrefill, hc: *HotPrefixCache, bits: u8, tokens: []const u32) !CommitStatus {
        var cache = try KVCache.initWithConfig(testing.allocator, 4, glmKvConfig(bits));
        defer cache.deinit();
        cache.step = self.req.offset;
        const disk_rows = try glm5_prefix.MlaRows.shareLive(testing.allocator, &self.req, glm5_prefix.commitRows(self.cps.items, self.req.offset), mlx.gpuStream());
        const owned = try self.cps.toOwnedSlice(testing.allocator);
        return hc.commitGlmTiers(&cache, tokens, false, 0, 0, null, owned, null, disk_rows, null, tokens.len);
    }

    fn deinit(self: *GlmPrefill) void {
        for (self.cps.items) |*cp| cp.deinit(testing.allocator);
        self.cps.deinit(testing.allocator);
        self.req.deinit();
    }
};

/// A prompt that shares `shared` tokens with `first` and then differs.
fn glmDiverging(buf: []u32, first: []const u32, shared: usize) []const u32 {
    @memcpy(buf[0..shared], first[0..shared]);
    for (buf[shared..], 0..) |*t, i| t.* = @intCast((i * 3 + 1) % 4);
    return buf;
}

/// Restores `prompt` from `hc` at `at` and checks the continuation against a cold prefill of it, bit for bit.
fn expectGlmRestoreLikeCold(hc: *HotPrefixCache, net: *const glm5_forward.Model, bits: u8, prompt: []const u32, at: usize) !void {
    var warm = try glm5_prefix.servedRequest(bits);
    defer warm.deinit();
    const res = try glmLookup(hc, &warm, bits, prompt, 0, null, &.{});
    try testing.expectEqual(at, res.matched);
    const got = try glm5_prefix.prefill(net, &warm, prompt[at..], &.{prompt.len - at});
    defer _ = mlx.mlx_array_free(got);
    var cold = try glm5_prefix.servedRequest(bits);
    defer cold.deinit();
    const want = try glm5_prefix.prefill(net, &cold, prompt, &.{ at, prompt.len - at });
    defer _ = mlx.mlx_array_free(want);
    try glm5_forward.expectArrayBits(want, got);
    try glm5_prefix.expectSameLogicalState(&cold, &warm);
}

test "GLM SSD-only: a prefill keeps at most two checkpoint states in RAM, and a restore at a released one reads SSD like a cold prefill" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    var fx: GlmNet = undefined;
    try fx.load();
    defer fx.deinit();
    var first: [700]u32 = undefined;
    for (&first, 0..) |*t, i| t.* = @intCast((i * 7 + i / 5) % 4);
    for ([_]u8{ 0, 8 }) |bits| {
        var fp_buf: [32]u8 = undefined;
        const fp = try std.fmt.bufPrint(&fp_buf, "glm-wt-{d}", .{bits});
        var hc = try glmSsdOnlyCache(io, base, fp, 0, true);
        defer hc.deinit();
        var turn = try GlmPrefill.run(&hc, &fx.net, bits, &first, &.{ 512, 64, 64, 48 });
        defer turn.deinit();
        try testing.expect(turn.peak_held <= 2);
        // The checkpoint of the last hooked chunk and the loop-end one beside it.
        try testing.expectEqual(@as(usize, 2), turn.held());
        for (turn.cps.items[0 .. turn.cps.items.len - 2]) |cp| try testing.expect(cp.isReleased());
        try testing.expectEqual(CommitStatus{ .disk_only = first.len }, try turn.commitSsdOnly(&hc, bits, &first));
        hc.flushPendingDisk(s);
        hc.disk.?.drainWriter();
        try testing.expectEqual(@as(u64, 0), hc.disk.?.writeErrors());
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
        const on_disk = &hc.disk.?.entries.items[0];
        try testing.expectEqual(@as(u32, 688), on_disk.kv_len);
        try testing.expectEqualSlices(u32, &.{ 512, 576, 640, 688 }, on_disk.ssm_positions);

        // The oldest, a middle one and the newest: each state comes off the disk.
        var other: [800]u32 = undefined;
        try expectGlmRestoreLikeCold(&hc, &fx.net, bits, glmDiverging(other[0..600], &first, 530), 512);
        try expectGlmRestoreLikeCold(&hc, &fx.net, bits, glmDiverging(other[0..680], &first, 650), 640);
        @memcpy(other[0..700], &first);
        for (other[700..750], 0..) |*t, i| t.* = @intCast((i * 3 + 1) % 4);
        try expectGlmRestoreLikeCold(&hc, &fx.net, bits, other[0..750], 688);
    }
}

test "GLM RAM tier beside an SSD tier judges the byte budget by the two checkpoints that stay" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    var fx: GlmNet = undefined;
    try fx.load();
    defer fx.deinit();
    var first: [700]u32 = undefined;
    for (&first, 0..) |*t, i| t.* = @intCast((i * 7 + i / 5) % 4);
    var probe = try GlmTurn.run(&fx.net, 8, &first, &.{ 512, 64, 64, 60 }, &.{ 0, 1, 2, 3 });
    // Room for every row and two checkpoints, not for four.
    const budget = probe.rows.bytes() + 2 * ssmCheckpointBytes(&probe.cps[0]) + 1;
    probe.rows.deinit();
    for (probe.cps) |*cp| cp.deinit(testing.allocator);
    testing.allocator.free(probe.cps);
    probe.cache.deinit();

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, budget);
    defer hc.deinit();
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "glm-price", 0, 128);
    hc.disk.?.enableBackgroundWriter();
    // The same session twice: the second commit replaces the first instead of being judged oversized and dropped.
    for (0..2) |_| {
        var turn = try GlmTurn.run(&fx.net, 8, &first, &.{ 512, 64, 64, 60 }, &.{ 0, 1, 2, 3 });
        defer turn.cache.deinit();
        const disk_rows = try turn.rows.share();
        try testing.expectEqual(CommitStatus{ .ok = first.len }, try hc.commitGlmTiers(&turn.cache, &first, false, 0, 0, null, turn.cps, turn.rows, disk_rows, null, first.len));
        try testing.expectEqual(@as(usize, 2), hc.entries.items[0].ssm_checkpoints.?.len);
        try testing.expect(hc.current_kv_bytes <= budget);
    }
}

test "GLM SSD-only: past the checkpoint cap the loop-end checkpoint still reaches disk" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    var fx: GlmNet = undefined;
    try fx.load();
    defer fx.deinit();
    var first: [1400]u32 = undefined;
    for (&first, 0..) |*t, i| t.* = @intCast((i * 7 + i / 5) % 4);
    var hc = try glmSsdOnlyCache(io, base, "glm-wt-cap", 0, true);
    defer hc.deinit();
    hc.disk.?.ssm_max_per_entry = glm5_prefix.checkpoint_cap;
    var chunks: [15]usize = @splat(64);
    chunks[0] = 512;
    chunks[14] = 40;
    // 512 + 13 x 64 + 40 = 1384: thirteen grid checkpoints, the loop-end one after them, a 16-token final span.
    var turn = try GlmPrefill.run(&hc, &fx.net, 8, &first, &chunks);
    defer turn.deinit();
    try testing.expect(turn.peak_held <= 2);
    _ = try turn.commitSsdOnly(&hc, 8, first[0..1400]);
    hc.flushPendingDisk(s);
    hc.disk.?.drainWriter();
    try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    const on_disk = &hc.disk.?.entries.items[0];
    try testing.expectEqual(@as(usize, glm5_prefix.checkpoint_cap), on_disk.ssm_positions.len);
    try testing.expectEqual(@as(u32, 1384), on_disk.ssm_positions[on_disk.ssm_positions.len - 1]);
    var other: [1500]u32 = undefined;
    @memcpy(other[0..1400], &first);
    for (other[1400..], 0..) |*t, i| t.* = @intCast((i * 3 + 1) % 4);
    try expectGlmRestoreLikeCold(&hc, &fx.net, 8, &other, 1384);
}

test "GLM SSD-only: a checkpoint whose write failed is not a restore point" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    var fx: GlmNet = undefined;
    try fx.load();
    defer fx.deinit();
    var first: [700]u32 = undefined;
    for (&first, 0..) |*t, i| t.* = @intCast((i * 7 + i / 5) % 4);
    var hc = try glmSsdOnlyCache(io, base, "glm-wt-fail", 0, true);
    defer hc.deinit();
    hc.disk.?.writer.?.injectFailure("s0000512", .write);
    var turn = try GlmPrefill.run(&hc, &fx.net, 8, &first, &.{ 512, 64, 112 });
    defer turn.deinit();
    try testing.expect(turn.peak_held <= 2);
    _ = try turn.commitSsdOnly(&hc, 8, &first);
    hc.flushPendingDisk(s);
    hc.disk.?.drainWriter();
    try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    const on_disk = &hc.disk.?.entries.items[0];
    try testing.expect(!on_disk.poisoned);
    try testing.expectEqualSlices(u32, &.{ 576, 688 }, on_disk.ssm_positions);
    try testing.expectEqual(@as(u32, 688), on_disk.kv_len);

    var other: [800]u32 = undefined;
    var warm = try glm5_prefix.servedRequest(8);
    defer warm.deinit();
    try testing.expectEqual(@as(usize, 0), (try glmLookup(&hc, &warm, 8, glmDiverging(other[0..600], &first, 530), 0, null, &.{})).matched);
    try expectGlmRestoreLikeCold(&hc, &fx.net, 8, glmDiverging(other[0..680], &first, 650), 576);
}

test "GLM RAM tier beside an SSD tier: the hot entry keeps its two newest checkpoints, the SSD tier all of them" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    var fx: GlmNet = undefined;
    try fx.load();
    defer fx.deinit();
    var first: [700]u32 = undefined;
    for (&first, 0..) |*t, i| t.* = @intCast((i * 7 + i / 5) % 4);
    for ([_]u8{ 0, 8 }) |bits| {
        var fp_buf: [32]u8 = undefined;
        const fp = try std.fmt.bufPrint(&fp_buf, "glm-ram-newest-{d}", .{bits});
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        defer hc.deinit();
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, fp, 0, 128);
        hc.disk.?.enableBackgroundWriter();
        var turn = try GlmTurn.run(&fx.net, bits, &first, &.{ 512, 64, 64, 60 }, &.{ 0, 1, 2, 3 });
        defer turn.cache.deinit();
        const disk_rows = try turn.rows.share();
        try testing.expectEqual(CommitStatus{ .ok = first.len }, try hc.commitGlmTiers(&turn.cache, &first, false, 0, 0, null, turn.cps, turn.rows, disk_rows, null, first.len));
        const e = &hc.entries.items[0];
        try testing.expectEqual(@as(usize, 2), e.ssm_checkpoints.?.len);
        try testing.expectEqualSlices(usize, &.{ 640, 700 }, &.{ e.ssm_checkpoints.?[0].pos, e.ssm_checkpoints.?[1].pos });
        var bytes: u64 = 0;
        for (e.ssm_checkpoints.?) |*cp| bytes += ssmCheckpointBytes(cp);
        try testing.expectEqual(bytes, e.ssm_bytes);
        try testing.expectEqual(e.glm.?.bytes() + bytes, e.kv_bytes);
        try testing.expectEqual(e.kv_bytes, hc.current_kv_bytes);
        hc.flushPendingDisk(s);
        hc.disk.?.drainWriter();
        try testing.expectEqualSlices(u32, &.{ 512, 576, 640, 700 }, hc.disk.?.entries.items[0].ssm_positions);

        // Under the RAM entry's oldest checkpoint the SSD tier restores.
        var other: [800]u32 = undefined;
        try expectGlmRestoreLikeCold(&hc, &fx.net, bits, glmDiverging(other[0..600], &first, 530), 512);
        try expectGlmRestoreLikeCold(&hc, &fx.net, bits, glmDiverging(other[0..680], &first, 600), 576);
    }
}

test "GLM SSD-only: two commits before one flush (two cancels culled in one pass) both reach disk" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    var fx: GlmNet = undefined;
    try fx.load();
    defer fx.deinit();
    var prompts: [2][600]u32 = undefined;
    for (&prompts, 0..) |*p, k| for (p, 0..) |*t, i| {
        t.* = @intCast((i * (k + 3) + k + i / 7) % 4);
    };
    var hc = try glmSsdOnlyCache(io, base, "glm-two-cancels", 0, false);
    defer hc.deinit();
    for (&prompts) |*p| {
        var turn = try GlmTurn.run(&fx.net, 8, p, &.{ 512, 88 }, &.{0});
        try testing.expectEqual(CommitStatus{ .disk_only = p.len }, try turn.commit(&hc, p, 0, null));
    }
    hc.flushPendingDisk(s);
    hc.disk.?.drainWriter();
    try testing.expect(hc.pending_disk == null);
    try testing.expectEqual(@as(usize, 2), hc.disk.?.entryCount());
    var warm = try glm5_prefix.servedRequest(8);
    defer warm.deinit();
    for (&prompts) |*p| {
        const again = p.* ++ [_]u32{1};
        try testing.expectEqual(@as(usize, 512), (try glmLookup(&hc, &warm, 8, &again, 0, null, &.{})).matched);
    }
}

test "GLM SSD-only without the writer: one flush finishes the whole entry before its source goes" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    var fx: GlmNet = undefined;
    try fx.load();
    defer fx.deinit();
    var first: [700]u32 = undefined;
    for (&first, 0..) |*t, i| t.* = @intCast((i * 7 + i / 5) % 4);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 0, 0);
    defer hc.deinit();
    hc.ssd_first = true;
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "glm-no-writer", 0, 128);
    hc.disk.?.ssd_first = true;
    hc.disk.?.armTestSpace(256 << 30, 1 << 40);
    hc.disk.?.max_flush_bytes = 1; // one chunk per synchronous piece
    try testing.expect(hc.disk.?.writer == null);
    var turn = try GlmTurn.run(&fx.net, 8, &first, &.{ 256, 256, 188 }, &.{ 0, 1 });
    try testing.expectEqual(CommitStatus{ .disk_only = first.len }, try turn.commit(&hc, &first, 0, null));
    hc.flushPendingDisk(s);
    try testing.expect(hc.pending_disk == null);
    const e = &hc.disk.?.entries.items[0];
    try testing.expectEqual(@as(u32, 700), e.kv_len);
    try testing.expectEqualSlices(u32, &.{ 256, 512 }, e.ssm_positions);
    try testing.expect(hc.disk.?.entryWholeOnDisk(e.id));
}

test "a GLM commit hands the SSD tier every row through its newest checkpoint, past any one flush" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    const cp = struct {
        fn at(pos: usize) SSMCheckpoint {
            return .{ .pos = pos, .layers = &.{} };
        }
    }.at;
    const cps = [_]SSMCheckpoint{ cp(2048), cp(524_288), cp(1_040_000) };
    const row: u64 = 6688;
    {
        var hc = try glmSsdOnlyCache(io, base, "glm-len-ssd", 20 << 30, false);
        defer hc.deinit();
        try testing.expectEqual(@as(?usize, null), hc.glmCommitLen(&cps, 1_040_003, row, 0));
        try testing.expectEqual(@as(?usize, 1_040_000), hc.glmDiskLen(&cps, 1_040_003, row, 0));
    }
    {
        // Both tiers: the RAM tier keeps what its 1 GiB holds, the SSD tier every row.
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 1 << 30);
        defer hc.deinit();
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "glm-len-both", 20 << 30, 128);
        try testing.expectEqual(@as(?usize, 2048), hc.glmCommitLen(&cps, 1_040_003, row, 0));
        try testing.expectEqual(@as(?usize, 1_040_000), hc.glmDiskLen(&cps, 1_040_003, row, 0));
    }
}

test "GLM SSD-only: the disk budget evicts the least recently used conversation" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    var fx: GlmNet = undefined;
    try fx.load();
    defer fx.deinit();
    var prompts: [3][600]u32 = undefined;
    for (&prompts, 0..) |*p, k| for (p, 0..) |*t, i| {
        t.* = @intCast((i * (k + 3) + k + i / 7) % 4);
    };
    // Measure one entry, then allow two.
    var one: u64 = 0;
    {
        var hc = try glmSsdOnlyCache(io, base, "glm-ssd-probe", 0, false);
        defer hc.deinit();
        var turn = try GlmTurn.run(&fx.net, 8, &prompts[0], &.{ 512, 88 }, &.{0});
        _ = try turn.commit(&hc, &prompts[0], 0, null);
        hc.flushPendingDisk(s);
        hc.disk.?.drainWriter();
        one = hc.disk.?.total_bytes;
    }
    var hc = try glmSsdOnlyCache(io, base, "glm-ssd-budget", one * 5 / 2, false);
    defer hc.deinit();
    for (&prompts) |*p| {
        var turn = try GlmTurn.run(&fx.net, 8, p, &.{ 512, 88 }, &.{0});
        _ = try turn.commit(&hc, p, 0, null);
        hc.flushPendingDisk(s);
        hc.disk.?.drainWriter();
    }
    try testing.expectEqual(@as(usize, 2), hc.disk.?.entryCount());
    try testing.expect(hc.disk.?.total_bytes <= one * 5 / 2);
    var warm = try glm5_prefix.servedRequest(8);
    defer warm.deinit();
    const again: [3][601]u32 = .{ prompts[0] ++ [_]u32{1}, prompts[1] ++ [_]u32{1}, prompts[2] ++ [_]u32{1} };
    try testing.expectEqual(@as(usize, 0), (try glmLookup(&hc, &warm, 8, &again[0], 0, null, &.{})).matched);
    try testing.expectEqual(@as(usize, 512), (try glmLookup(&hc, &warm, 8, &again[1], 0, null, &.{})).matched);
    try testing.expectEqual(@as(usize, 512), (try glmLookup(&hc, &warm, 8, &again[2], 0, null, &.{})).matched);
}

test "GLM's 1 GiB RAM tier keeps a 30K or 60K session at its prompt end and sheds checkpoints, a longer one trims" {
    const cp: u64 = 147_619_840;
    const row: u64 = 6688;
    const window: u64 = 2047 * 5 * 8 * 128 * 2 * 2;
    const budget: u64 = (1 << 30) - window;
    const Case = struct { positions: [8]usize, kept: usize, cps: u64 };
    const cases = [_]Case{
        .{ .positions = .{ 2048, 8192, 14336, 20480, 24576, 26624, 28672, 30688 }, .kept = 30688, .cps = 5 },
        .{ .positions = .{ 2048, 14336, 26624, 38912, 51200, 55296, 59392, 61408 }, .kept = 61408, .cps = 4 },
        .{ .positions = .{ 2048, 34816, 67584, 100352, 120832, 133120, 137216, 140256 }, .kept = 120832, .cps = 1 },
    };
    const bytes: [8]u64 = @splat(cp);
    for (cases) |c| {
        const tl = HotPrefixCache.trimLenForBudgetPure(budget, std.math.maxInt(usize), row, &c.positions, &bytes, .min_span_recency, null);
        try testing.expectEqual(@as(?usize, c.kept), tl);
        const k = std.mem.indexOfScalar(usize, &c.positions, c.kept).? + 1;
        const survivors = HotPrefixCache.shedSurvivorBytes(c.positions[0..k], bytes[0..k], budget - c.kept * row, .min_span_recency).?;
        try testing.expectEqual(c.cps * cp, survivors);
    }
}

test "a GLM commit copies only the rows its destination tier keeps, chosen before the copy" {
    const t = testing;
    const row: u64 = 6688;
    const cp: u64 = 147_619_840;
    const gib: u64 = 1 << 30;
    const positions = [_]usize{ 2048, 131072, 262144, 524288, 1_040_000 };
    const bytes: [5]u64 = @splat(cp);
    const policy: transformer_mod.ThinPolicy = .min_span_recency;
    // Everything fits: every row to the newest checkpoint.
    try t.expectEqual(@as(?usize, 64), HotPrefixCache.glmCommitLenFor(policy, gib, false, 64, row, 0, &.{64}, &.{cp}));
    try t.expectEqual(@as(?usize, 1_040_000), HotPrefixCache.glmCommitLenFor(policy, 0, false, 1_040_000, row, 0, &positions, &bytes));
    // A 1 GiB RAM tier keeps the newest checkpoint whose rows and a surviving checkpoint fit, never the full 7 GB.
    const ram = HotPrefixCache.glmCommitLenFor(policy, gib, false, 1_040_000, row, 0, &positions, &bytes).?;
    try t.expectEqual(@as(usize, 131072), ram);
    try t.expect(@as(u64, ram) * row + cp <= gib);
    // The window comes out of the budget first.
    try t.expectEqual(@as(?usize, 2048), HotPrefixCache.glmCommitLenFor(policy, gib, false, 1_040_000, row, 200 << 20, &positions, &bytes));
    // The SSD tier keeps every row through the newest checkpoint; only its own byte budget bounds
    // them, priced with every checkpoint it keeps at or below.
    try t.expectEqual(@as(?usize, 1_040_000), HotPrefixCache.glmCommitLenFor(policy, 0, true, 1_040_000, row, 0, &positions, &bytes));
    try t.expectEqual(@as(?usize, 1_040_000), HotPrefixCache.glmCommitLenFor(policy, 20 * gib, true, 1_040_000, row, 0, &positions, &bytes));
    // 262,144 rows (1.75 GB) fit 2 GiB alone; with their three checkpoints (0.44 GB) they do not.
    try t.expectEqual(@as(?usize, 131072), HotPrefixCache.glmCommitLenFor(policy, 2 * gib, true, 1_040_000, row, 0, &positions, &bytes));
    // A 1 GiB tier and ~131K rows with eight checkpoints: 0.82 GiB of rows, 1.92 GiB in all.
    const eight = [_]usize{ 2048, 18432, 43008, 69632, 92160, 116736, 126976, 131040 };
    const eight_bytes: [8]u64 = @splat(cp);
    const kept = HotPrefixCache.glmCommitLenFor(policy, gib, true, 131040, row, 0, &eight, &eight_bytes).?;
    var priced: u64 = @as(u64, kept) * row;
    for (eight) |p| priced += if (p <= kept) cp else 0;
    try t.expect(priced <= gib);
    try t.expect(kept < 131040);
    try t.expectEqual(@as(?usize, null), HotPrefixCache.glmCommitLenFor(policy, gib, false, 0, row, 0, &.{}, &.{}));
}

test "a GLM commit whose retention snapshot fails still destroys every transferred checkpoint once" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const cps = try testing.allocator.alloc(SSMCheckpoint, 3);
    for (cps, 0..) |*cp, i| {
        const layers = try testing.allocator.alloc(transformer_mod.SSMCacheEntrySnapshot, 2);
        for (layers) |*l| l.* = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false };
        cp.* = .{ .pos = (i + 1) * 8, .layers = layers };
    }
    const rows: glm5_prefix.MlaRows = .{
        .allocator = testing.allocator,
        .layers = try testing.allocator.alloc(glm5_prefix.MlaRows.Layer, 2),
        .len = 8,
        .latent_bits = 8,
    };
    @memset(rows.layers, .{});

    var failing = testing.FailingAllocator.init(testing.allocator, .{ .fail_index = std.math.maxInt(usize) });
    var source = try KVCache.initWithConfig(failing.allocator(), 2, glmKvConfig(8));
    // The next allocation is the retention snapshot's.
    failing.fail_index = failing.alloc_index;
    defer source.deinit();

    var hc = HotPrefixCache.init(testing.allocator, 2);
    defer hc.deinit();
    const tokens: [24]u32 = @splat(1);
    try testing.expectError(error.OutOfMemory, hc.commitGlm(&source, &tokens, false, 0, 0, null, cps, rows, null, tokens.len));
    try testing.expectEqual(@as(usize, 0), hc.entryCount());
}

test "a GLM replace commit that fails at any allocation leaks nothing and leaves the cache consistent" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const Stub = struct {
        fn cps(pos: usize) ![]SSMCheckpoint {
            const out = try testing.allocator.alloc(SSMCheckpoint, 1);
            const layers = try testing.allocator.alloc(transformer_mod.SSMCacheEntrySnapshot, 1);
            layers[0] = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false };
            out[0] = .{ .pos = pos, .layers = layers };
            return out;
        }
        fn rows() !glm5_prefix.MlaRows {
            const layers = try testing.allocator.alloc(glm5_prefix.MlaRows.Layer, 1);
            @memset(layers, .{});
            return .{ .allocator = testing.allocator, .layers = layers, .len = 8, .latent_bits = 8 };
        }
    };
    const tokens: [24]u32 = @splat(1);
    var k: usize = 0;
    while (k < 64) : (k += 1) {
        var failing = testing.FailingAllocator.init(testing.allocator, .{ .fail_index = std.math.maxInt(usize) });
        var hc = HotPrefixCache.init(failing.allocator(), 2);
        defer hc.deinit();
        var source = try KVCache.initWithConfig(testing.allocator, 1, glmKvConfig(8));
        defer source.deinit();
        _ = try hc.commitGlm(&source, tokens[0..16], false, 0, 0, null, try Stub.cps(8), try Stub.rows(), null, 16);
        try testing.expectEqual(@as(usize, 1), hc.entryCount());

        failing.fail_index = failing.alloc_index + k;
        const res = hc.commitGlm(&source, &tokens, false, 0, 0, null, try Stub.cps(16), try Stub.rows(), null, 24);
        var sum: u64 = 0;
        for (hc.entries.items) |e| sum += e.kv_bytes;
        try testing.expectEqual(sum, hc.current_kv_bytes);
        if (res) |_| break else |err| try testing.expectEqual(error.OutOfMemory, err);
    }
    try testing.expect(k < 64);
}

fn mediaTurnTokens(comptime n: usize) [n]u32 {
    var tokens: [n]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    return tokens;
}

test "an image turn persists its WHOLE turn to the SSD tier, image rows included" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const io = std.testing.io;
    const s = mlx.gpuStream();
    const tokens = mediaTurnTokens(3072);
    // The record carries the media identity (manifest v10), so it holds the whole turn whatever
    // the media boundary is; a hybrid still restores at its highest checkpoint, now above the item.
    // The record always holds the whole turn (3072 rows). `want` = what the SAME image restores:
    // a full token match leaves its last token to forward (3071), and a hybrid stops at its
    // highest checkpoint instead. `cross` = what another key restores, which is the text before
    // the item, again clamped to a hybrid's highest checkpoint at or below it.
    const Arm = struct { ssd_first: bool = false, mem: u64 = 0, cps: ?[]const usize = null, ms: usize = 2600, want: u32, cross: u32 };
    const arms = [_]Arm{
        .{ .ssd_first = true, .want = 3071, .cross = 2600 },
        .{ .want = 3071, .cross = 2600 },
        .{ .mem = 16 * 1024, .want = 3071, .cross = 2600 },
        .{ .ssd_first = true, .mem = 16 * 1024, .want = 3071, .cross = 2600 },
        .{ .cps = &.{ 1024, 2048, 3000 }, .want = 3000, .cross = 2048 },
        .{ .ssd_first = true, .cps = &.{ 2048, 3000 }, .want = 3000, .cross = 2048 },
        .{ .mem = 16 * 1024, .cps = &.{ 1024, 2048, 3000 }, .want = 3000, .cross = 2048 },
        .{ .cps = &.{ 1024, 2048, 3000 }, .ms = 2048, .want = 3000, .cross = 2048 },
        .{ .ssd_first = true, .cps = &.{ 1024, 2048, 3000 }, .ms = 2047, .want = 3000, .cross = 1024 },
        .{ .cps = &.{3000}, .want = 3000, .cross = 0 },
        .{ .ssd_first = true, .cps = &.{3000}, .want = 3000, .cross = 0 },
    };
    for (arms) |arm| {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root = buf[0..try tmp.dir.realPath(io, &buf)];
        var cold = try KVCache.init(testing.allocator, 3);
        defer cold.deinit();
        try testFillCache(&cold, s, 3, tokens.len);
        {
            var hc = HotPrefixCache.initWithMem(testing.allocator, 4, arm.mem);
            hc.ssd_first = arm.ssd_first;
            hc.hybrid = arm.cps != null;
            hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, root, "fp-media", 0, 1024);
            defer hc.deinit();
            var cache = try KVCache.init(testing.allocator, 3);
            defer cache.deinit();
            try testFillCache(&cache, s, 3, tokens.len);
            var src = pcBuildHybrid(s, 100.0, 500.0);
            defer pcFreeHybrid(&src);
            var cps: ?[]SSMCheckpoint = null;
            if (arm.cps) |positions| {
                cps = try testing.allocator.alloc(SSMCheckpoint, positions.len);
                for (positions, cps.?) |p, *cp| cp.* = try transformer_mod.captureSsmCheckpoint(testing.allocator, &src, p, s);
            }
            _ = try hc.commitWithMediaState(&cache, &tokens, false, 0xABCD, 0, arm.ms, cps, null, null, tokens.len);
            hc.flushPendingDisk(s);
        }
        // A cold RAM tier (restart, eviction) restores, under the SAME image, everything the
        // record holds — image rows included. Another image whose placeholders carry the same
        // ids, a later first item, and no image at all get the text before the item and no more.
        const Ask = struct { key: u64, ms: ?usize, want: u32 };
        for ([_]Ask{
            .{ .key = 0xABCD, .ms = arm.ms, .want = arm.want },
            .{ .key = 0x1234, .ms = arm.ms, .want = arm.cross },
            .{ .key = 0x1234, .ms = 2900, .want = arm.cross },
            .{ .key = 0, .ms = null, .want = arm.cross },
        }) |ask| {
            var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
            hc2.ssd_first = arm.ssd_first;
            hc2.hybrid = arm.cps != null;
            hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, root, "fp-media", 0, 1024);
            defer hc2.deinit();
            try testing.expectEqual(@as(usize, 1), hc2.disk.?.entryCount());
            // The whole turn, not a prefix cut at the item.
            try testing.expectEqual(@as(u32, tokens.len), hc2.disk.?.entries.items[0].kv_len);
            // The record names the media it was computed under, which is what bounds a foreign key.
            try testing.expectEqual(@as(u64, 0xABCD), hc2.disk.?.entries.items[0].vision_key);
            try testing.expectEqual(@as(?u32, @intCast(arm.ms)), hc2.disk.?.entries.items[0].media_start);
            var dst = try KVCache.init(testing.allocator, 3);
            defer dst.deinit();
            var ssm = pcEmptySsm();
            defer pcFreeHybrid(&ssm);
            var moe: usize = 0;
            const res = try hc2.lookupAndRestoreWithMedia(&dst, &moe, if (arm.cps != null) &ssm else null, s, &tokens, false, ask.key, ask.ms, &.{}, null, null, null, false);
            try testing.expectEqual(@as(usize, ask.want), res.matched);
            if (ask.want == 0) continue;
            for (0..3) |li| try pcExpectRowsEqual(s, dst.entries[li].keys, cold.entries[li].keys, ask.want);
            if (arm.cps != null) try testing.expectEqual(@as(f32, 100.0), pcSsmVal(ssm[0].conv_state, 0, s));
        }
    }
}

test "truncatedMediaKey: a record claims exactly the media blocks its rows cover" {
    const chain = [_]MediaSpan{ .{ .start = 100, .key = 0xA1 }, .{ .start = 2000, .key = 0xB2 } };
    const key = chainKey(&chain);

    // Whole prompt: the request's own key.
    try testing.expectEqual(@as(u64, 0xB2), truncatedMediaKey(&chain, key, 100, 3000).key);
    try testing.expectEqual(@as(?u32, 100), truncatedMediaKey(&chain, key, 100, 3000).start);
    // Cut between the two blocks: the first block's key, which is what a later request's chain
    // can recognise as its own prefix. Claiming 0xB2 here would be safe but would cost the reuse
    // of the first image's rows.
    try testing.expectEqual(@as(u64, 0xA1), truncatedMediaKey(&chain, key, 100, 1500).key);
    // Exactly at the second block's first row: that block's rows are NOT covered.
    try testing.expectEqual(@as(u64, 0xA1), truncatedMediaKey(&chain, key, 100, 2000).key);
    // Cut below every block: no media row, so a text prompt may share the record whole.
    try testing.expectEqual(@as(u64, 0), truncatedMediaKey(&chain, key, 100, 100).key);
    try testing.expectEqual(@as(?u32, null), truncatedMediaKey(&chain, key, 100, 50).start);
    // No chain (a caller that does not track blocks) keeps the caller's key and boundary.
    try testing.expectEqual(key, truncatedMediaKey(&.{}, key, 100, 3000).key);
    // A text request claims nothing, whatever it covers.
    try testing.expectEqual(@as(u64, 0), truncatedMediaKey(&.{}, 0, null, 3000).key);
}

test "the prefill write-through persists an image turn, keyed by the media it has forwarded" {
    // Step 4: a media turn's chunks were skipped outright, so a long image conversation banked
    // nothing until its end-of-request commit. Each chunk record now claims the blocks inside it.
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const io = std.testing.io;
    const s = mlx.gpuStream();
    const tokens = mediaTurnTokens(2048);
    const first_at: usize = 600;
    const k1: u64 = 0xA1A1_A1A1_A1A1_A1A1;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(io, &buf)];
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.ssd_first = true;
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, root, "fp-media-wt", 0, 1024);
    defer hc.deinit();
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try testFillCache(&cache, s, 3, tokens.len);

    // The first chunk stops below the item: it holds no media row, so it claims none.
    hc.writeThroughHybrid(cache.entries, cache.config, tokens[0..512], false, truncatedMediaKey(&.{}, k1, first_at, 512), &.{}, std.math.maxInt(u64), s);
    try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    try testing.expectEqual(@as(u64, 0), hc.disk.?.entries.items[0].vision_key);

    // A later chunk covers the item, so the record names the pixels its rows were computed under.
    hc.writeThroughHybrid(cache.entries, cache.config, tokens[0..2048], false, truncatedMediaKey(&.{}, k1, first_at, 2048), &.{}, std.math.maxInt(u64), s);
    const e = &hc.disk.?.entries.items[0];
    try testing.expectEqual(@as(u64, k1), e.vision_key);
    try testing.expectEqual(@as(?u32, @intCast(first_at)), e.media_start);
    try testing.expectEqual(@as(u32, 2048), e.kv_len);
}

test "SSD: a record holding the request's first image restores up to the new one" {
    // The disk half of the chain rule (`crossKeyBoundary`), which only the RAM tier could serve
    // before: an agent turn that appends a second screenshot to a conversation already holding
    // one reuses the persisted record through the FIRST image's rows, and stops at the new block.
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const io = std.testing.io;
    const s = mlx.gpuStream();
    const tokens = mediaTurnTokens(3072);
    const first_at: usize = 600;
    const second_at: usize = 2000;
    const k1: u64 = 0xA1A1_A1A1_A1A1_A1A1;
    const k2: u64 = 0xB2B2_B2B2_B2B2_B2B2;

    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(io, &buf)];
    var cold = try KVCache.init(testing.allocator, 3);
    defer cold.deinit();
    try testFillCache(&cold, s, 3, tokens.len);
    {
        // The earlier turn: one image at 100, keyed by its pixels.
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, root, "fp-media-chain", 0, 1024);
        defer hc.deinit();
        var cache = try KVCache.init(testing.allocator, 3);
        defer cache.deinit();
        try testFillCache(&cache, s, 3, tokens.len);
        _ = try hc.commitWithMediaState(&cache, &tokens, false, k1, 0, first_at, null, null, null, tokens.len);
        hc.flushPendingDisk(s);
    }

    // The next turn appends a second image at 2000. Its chain's first block IS the record's.
    const chain = [_]MediaSpan{ .{ .start = first_at, .key = k1 }, .{ .start = second_at, .key = k2 } };
    var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, root, "fp-media-chain", 0, 1024);
    defer hc2.deinit();
    var dst = try KVCache.init(testing.allocator, 3);
    defer dst.deinit();
    var moe: usize = 0;
    const res = try hc2.lookupAndRestoreWithMedia(&dst, &moe, null, s, &tokens, false, chainKey(&chain), first_at, &chain, null, null, null, false);
    try testing.expectEqual(second_at, res.matched);
    for (0..3) |li| try pcExpectRowsEqual(s, dst.entries[li].keys, cold.entries[li].keys, second_at);

    // Without the chain, the same request can only be held to the text before its first image:
    // nothing then proves the record's block is this request's.
    var bare = try KVCache.init(testing.allocator, 3);
    defer bare.deinit();
    var moe2: usize = 0;
    const res2 = try hc2.lookupAndRestoreWithMedia(&bare, &moe2, null, s, &tokens, false, chainKey(&chain), first_at, &.{}, null, null, null, false);
    try testing.expectEqual(first_at, res2.matched);
}

test "an image turn's speculative snapshots ride its whole SSD record" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const io = std.testing.io;
    const s = mlx.gpuStream();
    const tokens = mediaTurnTokens(1200);
    for ([_]bool{ true, false }) |media| {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root = buf[0..try tmp.dir.realPath(io, &buf)];
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, root, "fp-media-spec", 0, 128);
        defer hc.deinit();
        var trunk = try KVCache.init(testing.allocator, 2);
        defer trunk.deinit();
        try testFillCache(&trunk, s, 2, tokens.len);
        var assist = try KVCache.init(testing.allocator, 2);
        defer assist.deinit();
        try testFillCache(&assist, s, 2, tokens.len);
        var hist = try KVCache.init(testing.allocator, 1);
        defer hist.deinit();
        try testFillCache(&hist, s, 1, tokens.len);
        _ = try hc.commitWithMediaState(&trunk, &tokens, false, if (media) 0xABCD else 0, 0, if (media) 800 else null, null, .{ .cache = &assist, .base_pos = 0 }, .{ .cache = &hist, .base_pos = 0 }, tokens.len);
        hc.flushPendingDisk(s);
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
        const e = &hc.disk.?.entries.items[0];
        // The record is no longer cut at the media boundary, so the dflash context and the MTP
        // history cover exactly its rows and persist with it either way.
        try testing.expectEqual(@as(u32, 1200), e.kv_len);
        try testing.expect(e.spec_dflash != null);
        try testing.expect(e.spec_mtp != null);
    }
}

test "an image turn on a ringed cache persists whole and restores at its ring point under its own key" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const io = std.testing.io;
    const s = mlx.gpuStream();
    const window: u32 = 8;
    const n_layers: u32 = 4;
    const prompt: u32 = 700;
    const reply: u32 = 600;
    var toks: [prompt + reply]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
    // The only ring point sits at the prompt end (700). Under the record's OWN key it serves
    // whatever the item's position; a foreign key is held to the text before the item, and the
    // point's rows cannot reach back that far when the item sits at 500.
    // Under its own key the record's END point serves the whole turn less the token a restore
    // leaves to forward; a foreign key is held to the text before the item, which only the
    // prompt-end point can reach when the item sits at 700.
    for ([_]struct { ms: usize, want: u32, cross: u32 }{
        .{ .ms = 700, .want = prompt + reply - 1, .cross = 700 },
        .{ .ms = 500, .want = prompt + reply - 1, .cross = 0 },
    }) |arm| {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const root = buf[0..try tmp.dir.realPath(io, &buf)];
        {
            var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
            hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, root, "fp-media-ring", 0, 128);
            defer hc.deinit();
            var live = try KVCache.init(testing.allocator, n_layers);
            defer live.deinit();
            const cp = try ringTurn(&live, s, n_layers, window, prompt, reply);
            _ = try hc.commitWithRing(&live, &toks, false, 0xABCD, 0, arm.ms, null, null, null, prompt, .{ .prompt_end = cp });
            hc.flushPendingDisk(s);
        }
        for ([_]struct { key: u64, want: u32 }{ .{ .key = 0xABCD, .want = arm.want }, .{ .key = 0x1234, .want = arm.cross } }) |ask| {
            var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
            hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, root, "fp-media-ring", 0, 128);
            defer hc2.deinit();
            // The record exists either way now: it is keyed by its pixels, not cut at them.
            try testing.expectEqual(@as(usize, 1), hc2.disk.?.entryCount());
            var dst = try KVCache.init(testing.allocator, n_layers);
            defer dst.deinit();
            dst.setSwaRing(window);
            var moe: usize = 0;
            const res = try hc2.lookupAndRestoreWithMedia(&dst, &moe, null, s, &toks, false, ask.key, arm.ms, &.{}, null, null, null, false);
            try testing.expectEqual(@as(usize, ask.want), res.matched);
            // The restored ring keeps serving from WHEREVER the restore landed, which under the
            // record's own key is now past the prompt end.
            if (ask.want > 0) try expectRingContinuesCold(&dst, s, n_layers, window, ask.want, ask.want + 40);
        }
    }
}

test "an image turn on a QSA hybrid persists the indexer history its checkpoint needs" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(io, &buf)];
    const tokens = mediaTurnTokens(1024);
    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.hybrid = true;
        hc.qsa_history_required = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, root, "fp-media-qsa", 0, 128);
        defer hc.deinit();
        var live = pcBuildQsaHybrid(s, 1024, 100.0);
        defer pcFreeQsaHybrid(&live);
        const cps = try testing.allocator.alloc(SSMCheckpoint, 4);
        for (cps, 0..) |*cp, i| cp.* = try transformer_mod.captureSsmCheckpoint(testing.allocator, &live, (i + 1) * 256, s);
        // The pooled bank lives on the newest checkpoint, which is past the item.
        try transformer_mod.attachQsaHistoryToLatest(cps, &live, s);
        var cache = try KVCache.init(testing.allocator, 3);
        defer cache.deinit();
        try testFillCache(&cache, s, 3, tokens.len);
        _ = try hc.commitWithMediaState(&cache, &tokens, false, 0xABCD, 0, 600, cps, null, null, tokens.len);
        hc.flushPendingDisk(s);
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
        // Whole, with the pooled bank's checkpoint; the foreign-key restore below still needs the
        // indexer history for its OWN checkpoint, which is the point of this test.
        try testing.expectEqual(@as(u32, tokens.len), hc.disk.?.entries.items[0].kv_len);
    }
    var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc2.hybrid = true;
    hc2.qsa_history_required = true;
    hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, root, "fp-media-qsa", 0, 128);
    defer hc2.deinit();
    var dst = try KVCache.init(testing.allocator, 3);
    defer dst.deinit();
    var ssm = pcEmptySsm();
    defer pcFreeQsaHybrid(&ssm);
    var moe: usize = 0;
    const res = try hc2.lookupAndRestoreWithMedia(&dst, &moe, &ssm, s, &tokens, false, 0x1234, 600, &.{}, null, null, null, false);
    try testing.expectEqual(@as(usize, 512), res.matched);
}

test "GLM image turn persists its whole turn, and a foreign key restarts like a cold prefill below its item" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    var fx: GlmNet = undefined;
    try fx.load();
    defer fx.deinit();
    var first: [700]u32 = undefined;
    for (&first, 0..) |*t, i| t.* = @intCast((i * 7 + i / 5) % 4);
    var next: [750]u32 = undefined;
    @memcpy(next[0..650], first[0..650]);
    for (next[650..], 0..) |*t, i| t.* = @intCast((i * 3 + 1) % 4);
    var window = try KVCache.init(testing.allocator, 2);
    defer window.deinit();
    try testFillCache(&window, s, 2, 300);
    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.hybrid = true;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-glm-media", 0, 128);
        defer hc.deinit();
        var turn = try GlmTurn.run(&fx.net, 8, &first, &.{ 256, 256, 188 }, &.{ 0, 1 });
        defer turn.cache.deinit();
        _ = try hc.commitGlm(&turn.cache, &first, false, 0xABCD, 0, 600, turn.cps, turn.rows, .{ .cache = &window, .base_pos = 400 }, first.len);
        hc.flushPendingDisk(s);
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
        try testing.expectEqual(@as(u32, first.len), hc.disk.?.entries.items[0].kv_len);
        // The record is not cut at the item any more, so the dflash window covers its rows.
        try testing.expect(hc.disk.?.entries.items[0].spec_dflash != null);
    }
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.hybrid = true;
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-glm-media", 0, 128);
    defer hc.deinit();
    var warm = try glm5_prefix.servedRequest(8);
    defer warm.deinit();
    const res = try glmLookup(&hc, &warm, 8, &next, 0x1234, 600, &.{});
    try testing.expectEqual(@as(usize, 512), res.matched);
    const got = try glm5_prefix.prefill(&fx.net, &warm, next[512..], &.{238});
    defer _ = mlx.mlx_array_free(got);
    var cold = try glm5_prefix.servedRequest(8);
    defer cold.deinit();
    const want = try glm5_prefix.prefill(&fx.net, &cold, &next, &.{ 256, 256, 238 });
    defer _ = mlx.mlx_array_free(want);
    try glm5_forward.expectArrayBits(want, got);
    try glm5_prefix.expectSameLogicalState(&cold, &warm);
}
