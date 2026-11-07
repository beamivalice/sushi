//! The prefix cache's rules, apart from the arrays they act on: which checkpoint a thinning drops,
//! which entry a lookup picks, which one an eviction takes, and what a spec adoption may do.
//! Every function reads positions, lengths and byte counts only, so the host tier runs them
//! (`engine-prefix-cache.md`). `prefix_cache.zig`, `ssm_cache.zig` and the arch bodies keep their
//! old names for these through re-exports.

const std = @import("std");

/// Which retention policy a checkpoint list is thinned with. The span-preserving thin moves
/// where a warm turn restores from and was measured on the long-context archs only, so the policy is chosen
/// by `longCtxGated()` at each site and the ungated value names that site's previous
/// behaviour: `.oldest` (prefill capture, disk tier), `.min_span` (hot-cache merge/shed),
/// `.min_span_recency` (`.min_span` plus a dense newest quarter; the long-context archs).
pub const ThinPolicy = enum { oldest, min_span, min_span_recency };

/// Shortest list that reserves a dense newest quarter.
const RECENCY_DENSE_MIN: usize = 8;

/// The checkpoint index to drop when a retention list is over its cap: the interior position
/// whose removal widens the coverage gap least; index 0 and the last are always kept. One
/// selection for every retention site (#330: end-anchored survivors left no affordable trim point).
pub fn spanPreservingDropIndex(
    comptime T: type,
    items: []const T,
    comptime posOf: fn (*const T) usize,
    policy: ThinPolicy,
    /// An index the caller knows a later turn must restore from (a media boundary): never
    /// selected. When it is the only candidate the oldest goes instead — that one sits below
    /// it, so the protected position survives either way.
    protect: ?usize,
) usize {
    std.debug.assert(items.len > 0);
    if (policy == .oldest) return 0;
    if (items.len < 3) return 0;
    // Recency bias: the newest quarter stays at capture density (a warm turn that edits near
    // the end restores from there); everything below it is thinned span-preservingly.
    const scan_end = if (policy == .min_span_recency and items.len >= RECENCY_DENSE_MIN)
        items.len - items.len / 4
    else
        items.len;
    var best_at: usize = 1;
    var best_span: usize = std.math.maxInt(usize);
    var k: usize = 1;
    while (k + 1 < items.len and k < scan_end) : (k += 1) {
        if (protect) |p| if (k == p) continue;
        const span = posOf(&items[k + 1]) -| posOf(&items[k - 1]);
        if (span < best_span) {
            best_span = span;
            best_at = k;
        }
    }
    if (best_span == std.math.maxInt(usize)) return 0;
    return best_at;
}

fn fieldPosOf(comptime T: type) fn (*const T) usize {
    return struct {
        fn at(cp: *const T) usize {
            return cp.pos;
        }
    }.at;
}

fn u32PosOf(p: *const u32) usize {
    return p.*;
}

fn usizePosOf(p: *const usize) usize {
    return p.*;
}

/// `spanPreservingDropIndex` over any checkpoint list whose items carry a `pos`.
pub fn checkpointDropIndex(comptime T: type, cps: []const T, policy: ThinPolicy, protect: ?usize) usize {
    return spanPreservingDropIndex(T, cps, fieldPosOf(T), policy, protect);
}

/// `spanPreservingDropIndex` over a bare ascending position list.
pub fn positionDropIndex(positions: []const u32, policy: ThinPolicy) usize {
    return spanPreservingDropIndex(u32, positions, u32PosOf, policy, null);
}

pub fn positionDropIndexUsize(positions: []const usize, policy: ThinPolicy) usize {
    return spanPreservingDropIndex(usize, positions, usizePosOf, policy, null);
}

/// What a spec-snap adoption may do, decided before any mlx call.
pub const SpecAdopt = union(enum) {
    /// Nothing to adopt: no payload, or the snap does not cover the reused range.
    skip,
    /// A KV-only spec cache (dflash context, sidecar MTP head).
    kv_only: usize,
    /// An in-checkpoint head: KV + QSA aux together.
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

/// Shortest prefix a ring-checkpoint restore reuses; below it the request cold-prefills.
pub const RING_RESTORE_MIN_TOKENS: usize = 64;

/// Message starts one prefill takes a ring checkpoint at (`ringMarkPositions`).
pub const RING_MARKS_MAX: usize = 4;

/// Where a prefill takes ring checkpoints: the message starts (`mark_id`) it forwards past the
/// `restored` prefix, from the restore floor up to where the prompt-end checkpoint reaches
/// (`backoff` tokens short of the prompt's end); thinned to `RING_MARKS_MAX` keeping the first
/// (a shared preamble's end) and the last.
pub fn ringMarkPositions(prompt: []const u32, restored: usize, mark_id: u32, backoff: usize, out: *[RING_MARKS_MAX]usize) []usize {
    var buf: [RING_MARKS_MAX + 1]usize = undefined;
    var n: usize = 0;
    var p = @max(restored + 1, RING_RESTORE_MIN_TOKENS);
    while (p + backoff < prompt.len) : (p += 1) {
        if (prompt[p] != mark_id) continue;
        buf[n] = p;
        n += 1;
        if (n > RING_MARKS_MAX) {
            const k = positionDropIndexUsize(buf[0..n], .min_span);
            for (k..n - 1) |i| buf[i] = buf[i + 1];
            n -= 1;
        }
    }
    @memcpy(out[0..n], buf[0..n]);
    return out[0..n];
}

/// The longest candidate list a trim simulates the span-preserving shed over.
pub const SHED_SIM_MAX: usize = 192;

/// Bytes the checkpoints at or below a candidate trim point cost after the commit path's
/// span-preserving shed to `allowance`; null when even the last survivor is over.
pub fn shedSurvivorBytes(positions: []const usize, bytes: []const u64, allowance: u64, policy: ThinPolicy) ?u64 {
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
        const drop = positionDropIndexUsize(pos_buf[0..n], policy);
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

/// Tokens two sequences share from the start, capped at `max`.
pub fn commonPrefixLen(a: []const u32, b: []const u32, max: usize) usize {
    var shared: usize = 0;
    while (shared < max and a[shared] == b[shared]) shared += 1;
    return shared;
}

/// The lookup's running winner: the candidate with the highest EFFECTIVE restorable length, the
/// longer raw match breaking a tie, the earlier entry breaking a full one.
pub const BestMatch = struct {
    idx: ?usize = null,
    shared: usize = 0,
    effective: usize = 0,

    pub fn offer(self: *BestMatch, idx: usize, effective: usize, shared: usize) void {
        if (effective > self.effective or (effective == self.effective and shared > self.shared)) {
            self.effective = effective;
            self.shared = shared;
            self.idx = idx;
        }
    }
};

/// Least-recently-used entry index, skipping the one whose `last_used` equals `protect`.
/// Workload-fair: only entries of the `cache_key` holding the most eligible entries
/// (`incoming_key` counts as one more) are candidates, so a sweep evicts its own
/// documents before another workload's conversation. One key = plain LRU.
/// `E` carries `checked_out_by`, `cache_key` and `last_used`; a checked-out entry owns its buffers
/// (evicting it frees nothing) and is never a candidate.
pub fn lruIndexExcluding(comptime E: type, entries: []const E, protect: ?u64, incoming_key: ?u64) ?usize {
    var best: ?usize = null;
    var best_used: u64 = std.math.maxInt(u64);
    var max_count: usize = 0;
    for (entries, 0..) |*e, i| {
        if (e.checked_out_by != null) continue;
        if (protect) |p| {
            if (e.last_used == p) continue;
        }
        var count: usize = if (incoming_key != null and incoming_key.? == e.cache_key) 1 else 0;
        for (entries) |*o| {
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

/// Which of two entries tied on `last_used` is the newest.
pub const NewestTie = enum { first, last };

/// The most recently used entry (`last_used` is a strictly increasing counter, so a tie only
/// arises in a hand-built list).
pub fn newestIndex(comptime E: type, entries: []const E, tie: NewestTie) ?usize {
    if (entries.len == 0) return null;
    var best: usize = 0;
    for (entries[1..], 1..) |*e, i| {
        const newer = switch (tie) {
            .first => e.last_used > entries[best].last_used,
            .last => e.last_used >= entries[best].last_used,
        };
        if (newer) best = i;
    }
    return best;
}

/// Smallest eviction whose live/billed ratio is meaningful.
pub const SHARED_RATIO_MIN_BYTES: u64 = 1 << 20;

/// An eviction returning less than 1/Nth of what the entry was billed gave the allocator
/// nothing (refcount-shared with a live cache). A ratio, not a floor: a small exclusive
/// entry also returns little.
pub const SHARED_RETURN_DIVISOR: u64 = 4;

/// Did an eviction that was billed `billed` bytes hand back so little that the next one would too?
pub fn evictionWasShared(billed: u64, freed_live: u64) bool {
    return billed >= SHARED_RATIO_MIN_BYTES and freed_live * SHARED_RETURN_DIVISOR < billed;
}

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

const testing = std.testing;

const CheckpointStub = struct { pos: usize };

test "ssm retention: the span-preserving thin keeps both ends and spreads the survivors" {
    // Drop-oldest survivors covered only the last `max * stride` tokens (#330).
    const t = std.testing;
    var positions: [94]usize = undefined;
    for (&positions, 0..) |*p, i| p.* = (i + 1) * 4096;
    var n: usize = positions.len;
    while (n > 16) {
        const drop = positionDropIndexUsize(positions[0..n], .min_span_recency);
        try t.expect(drop != 0);
        try t.expect(drop + 1 != n);
        var k = drop;
        while (k + 1 < n) : (k += 1) positions[k] = positions[k + 1];
        n -= 1;
    }
    try t.expectEqual(@as(usize, 16), n);
    try t.expectEqual(@as(usize, 4096), positions[0]);
    try t.expectEqual(@as(usize, 94 * 4096), positions[n - 1]);
    // Spread, but not evenly: the newest quarter stays at capture density.
    const ideal = (positions[n - 1] - positions[0]) / (n - 1);
    var i: usize = 1;
    while (i < n) : (i += 1) try t.expect(positions[i] - positions[i - 1] <= 2 * ideal);
    try t.expectEqual(@as(usize, 4096), positions[n - 1] - positions[n - 2]);
    try t.expectEqual(@as(usize, 4096), positions[n - 2] - positions[n - 3]);
    try t.expect(positions[1] - positions[0] > 4 * 4096);
    try t.expectEqual(@as(usize, 0), positionDropIndexUsize(positions[0..2], .min_span_recency));
    try t.expectEqual(@as(usize, 0), positionDropIndexUsize(positions[0..1], .min_span_recency));
    const clustered = [_]u32{ 0, 1000, 1010, 2000 };
    try t.expectEqual(@as(usize, 2), positionDropIndex(&clustered, .min_span_recency));

    // Every other arch keeps the previous policy of its own site: `.oldest` (prefill capture,
    // disk tier) and `.min_span` (hot-cache sites, no recency quarter).
    try t.expectEqual(@as(usize, 0), positionDropIndex(&clustered, .oldest));
    try t.expectEqual(@as(usize, 0), positionDropIndexUsize(positions[0..n], .oldest));
    {
        // 12 entries; the tightest pair is inside the newest quarter (starts at 9).
        var pos12: [12]usize = .{ 0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 9010, 10_000 };
        try t.expectEqual(@as(usize, 10), positionDropIndexUsize(&pos12, .min_span));
        try t.expect(positionDropIndexUsize(&pos12, .min_span_recency) < 9);
        try t.expectEqual(
            positionDropIndexUsize(pos12[0..7], .min_span),
            positionDropIndexUsize(pos12[0..7], .min_span_recency),
        );
    }
}

test "drop index: the recorded position lists drop the same index through every wrapper" {
    const recorded = [_][]const usize{
        &.{ 4096, 8192, 8200, 16384 },
        &.{ 100, 200, 300, 400, 500, 600, 700, 800, 810, 1000 },
        &.{ 0, 512, 1024, 1030, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120 },
    };
    const expect_oldest = [_]usize{ 0, 0, 0 };
    const expect_min_span = [_]usize{ 1, 7, 3 };
    for (recorded, 0..) |positions, r| {
        var wide: [12]u32 = undefined;
        var cps: [12]CheckpointStub = undefined;
        for (positions, 0..) |p, i| {
            wide[i] = @intCast(p);
            cps[i] = .{ .pos = p };
        }
        const n = positions.len;
        try testing.expectEqual(expect_oldest[r], positionDropIndexUsize(positions, .oldest));
        try testing.expectEqual(expect_min_span[r], positionDropIndexUsize(positions, .min_span));
        try testing.expectEqual(expect_min_span[r], positionDropIndex(wide[0..n], .min_span));
        try testing.expectEqual(expect_min_span[r], checkpointDropIndex(CheckpointStub, cps[0..n], .min_span, null));
    }
}

test "drop index: a protected position is never dropped, and the oldest goes when it is the only interior one" {
    const cps = [_]CheckpointStub{ .{ .pos = 100 }, .{ .pos = 110 }, .{ .pos = 5000 } };
    try testing.expectEqual(@as(usize, 1), checkpointDropIndex(CheckpointStub, &cps, .min_span, null));
    try testing.expectEqual(@as(usize, 0), checkpointDropIndex(CheckpointStub, &cps, .min_span, 1));
    const four = [_]CheckpointStub{ .{ .pos = 100 }, .{ .pos = 110 }, .{ .pos = 120 }, .{ .pos = 5000 } };
    try testing.expectEqual(@as(usize, 1), checkpointDropIndex(CheckpointStub, &four, .min_span, null));
    try testing.expectEqual(@as(usize, 2), checkpointDropIndex(CheckpointStub, &four, .min_span, 1));
}

test "spec adopt plan: skip, KV-only, head and the pre-v5 decline" {
    const A = SpecAdopt;
    try testing.expectEqual(A{ .kv_only = 21 }, specAdoptPlan(10, 40, 31, false, false));
    try testing.expectEqual(@as(A, .skip), specAdoptPlan(40, 40, 31, false, false));
    try testing.expectEqual(@as(A, .skip), specAdoptPlan(0, 20, 31, false, false));
    try testing.expectEqual(A{ .kv_only = 31 }, specAdoptPlan(0, 31, 31, false, false));
    try testing.expectEqual(A{ .head = 21 }, specAdoptPlan(10, 40, 31, true, true));
    try testing.expectEqual(@as(A, .decline_head_no_history), specAdoptPlan(10, 40, 31, true, false));
    try testing.expectEqual(@as(A, .skip), specAdoptPlan(40, 40, 31, true, false));
}

test "ring marks sit at the message starts a prefill forwards, thinned to RING_MARKS_MAX keeping the first and the last" {
    const im: u32 = 7;
    const backoff: usize = 30;
    var prompt: [2000]u32 = @splat(1);
    for ([_]usize{ 0, 40, 300, 500, 900, 1100, 1300, 1985 }) |p| prompt[p] = im;
    var buf: [RING_MARKS_MAX]usize = undefined;
    // Under the restore floor and inside the prompt end's own reach are skipped.
    try testing.expectEqualSlices(usize, &.{ 300, 500, 900, 1300 }, ringMarkPositions(&prompt, 0, im, backoff, &buf));
    // A restored prefix is already in the cache: only the tail past it is marked.
    try testing.expectEqualSlices(usize, &.{ 900, 1100, 1300 }, ringMarkPositions(&prompt, 500, im, backoff, &buf));
    try testing.expectEqual(@as(usize, 0), ringMarkPositions(&prompt, 0, 99, backoff, &buf).len);
    try testing.expectEqual(@as(usize, 0), ringMarkPositions(prompt[0..60], 0, im, backoff, &buf).len);
}

test "shed survivor bytes: the span-preserving shed prices what a trim keeps" {
    const positions = [_]usize{ 1000, 2000, 2010, 3000, 4000 };
    const bytes = [_]u64{ 10, 10, 10, 10, 10 };
    try testing.expectEqual(@as(?u64, 50), shedSurvivorBytes(&positions, &bytes, 50, .min_span));
    try testing.expectEqual(@as(?u64, 40), shedSurvivorBytes(&positions, &bytes, 45, .min_span));
    try testing.expectEqual(@as(?u64, 10), shedSurvivorBytes(&positions, &bytes, 10, .min_span));
    try testing.expectEqual(@as(?u64, null), shedSurvivorBytes(&positions, &bytes, 9, .min_span));
    var many: [SHED_SIM_MAX + 1]usize = undefined;
    var cost: [SHED_SIM_MAX + 1]u64 = @splat(1);
    for (&many, 0..) |*p, i| p.* = (i + 1) * 100;
    try testing.expectEqual(@as(?u64, null), shedSurvivorBytes(&many, &cost, 10, .min_span));
}

test "best match: the effective length decides, the raw match breaks a tie, the earlier entry a full one" {
    var best: BestMatch = .{};
    best.offer(0, 0, 0);
    try testing.expectEqual(@as(?usize, null), best.idx);
    best.offer(1, 100, 120);
    best.offer(2, 100, 110);
    try testing.expectEqual(@as(?usize, 1), best.idx);
    best.offer(3, 100, 130);
    try testing.expectEqual(@as(?usize, 3), best.idx);
    best.offer(4, 100, 130);
    try testing.expectEqual(@as(?usize, 3), best.idx);
    best.offer(5, 101, 101);
    try testing.expectEqual(@as(?usize, 5), best.idx);
    try testing.expectEqual(@as(usize, 101), best.shared);
}

test "common prefix: capped at the shorter side" {
    try testing.expectEqual(@as(usize, 3), commonPrefixLen(&.{ 1, 2, 3, 4 }, &.{ 1, 2, 3, 9 }, 4));
    try testing.expectEqual(@as(usize, 2), commonPrefixLen(&.{ 1, 2, 3, 4 }, &.{ 1, 2, 3, 4 }, 2));
    try testing.expectEqual(@as(usize, 0), commonPrefixLen(&.{ 5 }, &.{ 6 }, 1));
}

const EntryStub = struct { checked_out_by: ?usize = null, cache_key: u64 = 0, last_used: u64 };

test "eviction order: the LRU of the key holding the most eligible entries goes" {
    const key_a: u64 = 0xa;
    const key_b: u64 = 0xb;
    var entries = [_]EntryStub{
        .{ .cache_key = key_a, .last_used = 1 },
        .{ .cache_key = key_b, .last_used = 2 },
        .{ .cache_key = key_b, .last_used = 3 },
        .{ .cache_key = key_b, .last_used = 4 },
    };
    try testing.expectEqual(@as(?usize, 1), lruIndexExcluding(EntryStub, &entries, null, key_b));
    try testing.expectEqual(@as(?usize, 1), lruIndexExcluding(EntryStub, &entries, null, null));
    // The incoming key counts as one more: A (1 + 1) still loses to B (3 + 0).
    try testing.expectEqual(@as(?usize, 1), lruIndexExcluding(EntryStub, &entries, null, key_a));
    try testing.expectEqual(@as(?usize, 2), lruIndexExcluding(EntryStub, &entries, 2, key_b));
    entries[1].checked_out_by = 7;
    try testing.expectEqual(@as(?usize, 2), lruIndexExcluding(EntryStub, &entries, null, key_b));
    entries[1].checked_out_by = null;
    for (&entries) |*e| e.cache_key = 0;
    try testing.expectEqual(@as(?usize, 0), lruIndexExcluding(EntryStub, &entries, null, 0));
    for (&entries) |*e| e.checked_out_by = 1;
    try testing.expectEqual(@as(?usize, null), lruIndexExcluding(EntryStub, &entries, null, null));
}

test "eviction order: a tie in count goes to the older entry, then to the earlier one" {
    var entries = [_]EntryStub{
        .{ .cache_key = 1, .last_used = 9 },
        .{ .cache_key = 2, .last_used = 5 },
        .{ .cache_key = 1, .last_used = 8 },
        .{ .cache_key = 2, .last_used = 7 },
    };
    try testing.expectEqual(@as(?usize, 1), lruIndexExcluding(EntryStub, &entries, null, null));
    entries[3].last_used = 5;
    try testing.expectEqual(@as(?usize, 1), lruIndexExcluding(EntryStub, &entries, null, null));
}

test "newest entry: a tie goes to the side the caller names" {
    const entries = [_]EntryStub{ .{ .last_used = 3 }, .{ .last_used = 9 }, .{ .last_used = 9 }, .{ .last_used = 4 } };
    try testing.expectEqual(@as(?usize, 1), newestIndex(EntryStub, &entries, .first));
    try testing.expectEqual(@as(?usize, 2), newestIndex(EntryStub, &entries, .last));
    try testing.expectEqual(@as(?usize, null), newestIndex(EntryStub, entries[0..0], .first));
}

test "admission: an eviction that handed back under a quarter of its bill stops the pass, small ones never judge" {
    try testing.expect(evictionWasShared(8 << 20, (2 << 20) - 1));
    try testing.expect(!evictionWasShared(8 << 20, 2 << 20));
    try testing.expect(!evictionWasShared((1 << 20) - 1, 0));
}

test "restore pin: only a large entry whose share is a sliver of it is a lien" {
    try testing.expect(!restoreWouldPinEntry(RESTORE_PIN_MIN_BYTES - 1, RESTORE_PIN_MIN_BYTES, 1 << 20, 1));
    try testing.expect(restoreWouldPinEntry(RESTORE_PIN_MIN_BYTES, RESTORE_PIN_MIN_BYTES, 1 << 20, 0));
    try testing.expect(restoreWouldPinEntry(RESTORE_PIN_MIN_BYTES, RESTORE_PIN_MIN_BYTES, 6500, 100));
    try testing.expect(!restoreWouldPinEntry(RESTORE_PIN_MIN_BYTES, RESTORE_PIN_MIN_BYTES, 6400, 100));
}
