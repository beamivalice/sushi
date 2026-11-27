//! Hot prefix cache takeover/checkout lifecycle, trim byte accounting, and budget/LRU eviction policy.

const std = @import("std");
const mlx = @import("mlx.zig");
const trunk_mod = @import("trunk.zig");
const log = @import("log.zig");
const prefix_arch = @import("arch/prefix.zig");
const prefix_policy = @import("prefix_policy.zig");
const cache_stats = @import("cache_stats.zig");
const pc = @import("prefix_cache.zig");

const KVCache = trunk_mod.KVCache;
const KVCacheSnapshot = trunk_mod.KVCacheSnapshot;
const SSMCheckpoint = trunk_mod.SSMCheckpoint;
const checkpointHasQsaPooled = prefix_arch.aux.checkpointHasQsaPooled;
const sliceQsaHistoryOntoCheckpoint = prefix_arch.aux.sliceQsaHistoryOntoCheckpoint;
const ssmCheckpointBytes = trunk_mod.ssmCheckpointBytes;

const HotPrefixCache = pc.HotPrefixCache;
const Entry = pc.Entry;
const Checkout = pc.Checkout;
const DflashSnap = pc.DflashSnap;
const EvictionReport = pc.EvictionReport;
const restoreMoveEnabled = pc.restoreMoveEnabled;
const unlimited_entries = pc.unlimited_entries;
const MIN_CANCELLED_COMMIT_TOKENS = pc.MIN_CANCELLED_COMMIT_TOKENS;
const MIN_TRIMMED_VICTIM_TOKENS = pc.MIN_TRIMMED_VICTIM_TOKENS;
const RING_CHECKPOINT_MAX = pc.RING_CHECKPOINT_MAX;
const freeEntryOwnedState = HotPrefixCache.freeEntryOwnedState;
const ringCpsBytes = HotPrefixCache.ringCpsBytes;
const thinRingCps = HotPrefixCache.thinRingCps;

/// Wave 1.B: total KV bytes held by a snapshot — sum of `size * itemsize`
/// across every initialized entry's storage arrays. mlx-c arrays carry
/// their shape + dtype so this is exact, not a heuristic. Quant schemes
/// account for q, scales, biases together.
/// What `snapshotBytes` will bill of `cache` once a commit snapshots it.
pub fn liveCacheBytes(cache: *const KVCache) u64 {
    return snapshotBytes(&.{ .entries = cache.entries, .step = cache.step, .allocator = cache.allocator, .config = cache.config });
}

pub fn snapshotBytes(snap: *const KVCacheSnapshot) u64 {
    var total: u64 = 0;
    for (snap.entries) |e| {
        if (!e.initialized) continue;
        total += @as(u64, mlx.mlx_array_size(e.keys.array)) * @as(u64, mlx.mlx_array_itemsize(e.keys.array));
        total += @as(u64, mlx.mlx_array_size(e.values.array)) * @as(u64, mlx.mlx_array_itemsize(e.values.array));
        if (snap.config.scheme != .off) {
            total += @as(u64, mlx.mlx_array_size(e.keys_scales.array)) * @as(u64, mlx.mlx_array_itemsize(e.keys_scales.array));
            total += @as(u64, mlx.mlx_array_size(e.keys_biases.array)) * @as(u64, mlx.mlx_array_itemsize(e.keys_biases.array));
            total += @as(u64, mlx.mlx_array_size(e.values_scales.array)) * @as(u64, mlx.mlx_array_itemsize(e.values_scales.array));
            total += @as(u64, mlx.mlx_array_size(e.values_biases.array)) * @as(u64, mlx.mlx_array_itemsize(e.values_biases.array));
        }
    }
    return total;
}

/// Resident bytes of a speculative-side snap: the KV plus an in-checkpoint head's QSA aux half.
pub fn specSnapBytes(snap: *const DflashSnap) u64 {
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
pub fn snapshotRowBytes(snap: *const KVCacheSnapshot) u64 {
    var total: u64 = 0;
    for (snap.entries) |e| {
        if (!e.initialized or e.ringed) continue;
        inline for (.{ e.keys.array, e.values.array, e.keys_scales.array, e.keys_biases.array, e.values_scales.array, e.values_biases.array }) |arr| {
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
pub fn ringedBytes(snap: *const KVCacheSnapshot) u64 {
    var total: u64 = 0;
    for (snap.entries) |e| {
        if (!e.initialized or !e.ringed) continue;
        inline for (.{ e.keys.array, e.values.array, e.keys_scales.array, e.keys_biases.array, e.values_scales.array, e.values_biases.array }) |arr| {
            if (arr.ctx != null) total += @as(u64, mlx.mlx_array_size(arr)) * @as(u64, mlx.mlx_array_itemsize(arr));
        }
    }
    return total;
}

/// Where a ringed candidate trims to: its end, or ring checkpoint `cp`, whose rows become its ring.
pub const RingTrim = struct { len: usize, cp: ?usize = null };

/// A ringed candidate restores only where its sliding layers hold the window: at its `end`, or
/// at one of its ring checkpoints (ascending). The longest that fits `budget`: the rows below it
/// at `row_bytes`, the ring it keeps and the checkpoints below it. Null = none above the floor.
pub fn ringTrimLen(budget: u64, limit: usize, row_bytes: u64, end: usize, ring_bytes: u64, cps: []const KVCacheSnapshot) ?RingTrim {
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
pub const TRIM_LOG_MAX_POS: usize = 32;

pub fn appendTrimFmt(buf: []u8, n: *usize, comptime fmt: []const u8, args: anytype) void {
    const s = std.fmt.bufPrint(buf[n.*..], fmt, args) catch return;
    n.* += s.len;
}

/// The trim decision's inputs as one line (the price, the positions, the chosen bill).
pub fn formatTrimInputs(
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
pub fn logTrimInputs(
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
pub const SHED_SIM_MAX = prefix_policy.SHED_SIM_MAX;

/// The last retained checkpoint inherits the latest checkpoint's pooled bank.
/// Price its sliced shape before allocating/copying any multi-GB KV prefix.
/// Captured aux state already contains the destination's pos % ratio leftover.
pub fn trimmedCheckpointBytes(list: []const SSMCheckpoint, index: usize) u64 {
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

pub const shedSurvivorBytes = prefix_policy.shedSurvivorBytes;

/// Shared hybrid selector; checkpoints optionally supply the transferred QSA bank's bill.
pub fn trimLenForBudgetPure(
    budget: u64,
    limit: usize,
    row_bytes: u64,
    positions: []const usize,
    cp_bytes: []const u64,
    policy: trunk_mod.ThinPolicy,
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

/// Which arm `trimLenForBudget` bills a list of this length with (for the log).
pub fn trimBillArm(list_len: usize, gated: bool) []const u8 {
    if (!gated) return "all_lower";
    return if (list_len > SHED_SIM_MAX) "all_lower" else "shed";
}

/// The pre-shed bill (every lower checkpoint), for a list past `SHED_SIM_MAX` or the ungated arm.
pub fn trimLenBillingAllLower(budget: u64, limit: usize, row_bytes: u64, list: []const SSMCheckpoint) ?usize {
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
pub fn trimLenForBudget(
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

/// Restore by move, a whole take: a full-prefix hit, whose commit (prompt and reply) replaces this same entry. A
/// prompt that is exactly the entry qualifies: its slot re-forwards the last token from below the share.
pub fn wholeTakeEligible(entry_tokens: usize, shared: usize) bool {
    return entry_tokens > 0 and shared == entry_tokens;
}

/// A partial take keeps only the shared prefix at donation and its slot appends from a restore point at or below
/// the share, so it needs only a share, whatever the prompt's length. A ringed entry's restore points are
/// snapshots past any prefix cut: it is taken whole only.
pub fn partialTakeEligible(entry_tokens: usize, shared: usize, ringed: bool) bool {
    return shared > 0 and shared < entry_tokens and !ringed;
}

/// Can a slot take any entry over? A pending disk record shares the buffers, so its append could not donate.
pub fn movable(move_enabled: bool, pending_disk: bool) bool {
    return move_enabled and !pending_disk;
}

/// The SSD-first lookup's take, made up front: a whole take, for a slot.
pub fn upFrontTakeEligible(ssd_first: bool, has_slot: bool, move_enabled: bool, pending_disk: bool, entry_tokens: usize, shared: usize) bool {
    return ssd_first and has_slot and movable(move_enabled, pending_disk) and wholeTakeEligible(entry_tokens, shared);
}

pub fn beginCheckout(e: *Entry, slot_id: usize, kind: Checkout.Kind) void {
    std.debug.assert(e.checkout == null);
    e.checkout = .{ .slot = slot_id, .kind = kind };
}

pub fn endCheckout(e: *Entry) void {
    e.checkout = null;
}

/// Take the checkout: mark the entry as this slot's; `donateCheckout` releases the handles
/// at the last moment before the slot's first write. Returns whether it was taken (the
/// admission bill's input).
pub fn checkoutIfEligible(self: *HotPrefixCache, idx: usize, shared: usize, slot_id: ?usize) bool {
    const e = &self.entries.items[idx];
    if (!upFrontTakeEligible(self.ssd_first, slot_id != null, restoreMoveEnabled(), self.pending_disk != null, e.tokens.len, shared)) return false;
    beginCheckout(e, slot_id.?, .whole);
    log.info("  [hot-cache] checked out {d}-token entry to the slot (restore by move; the append donates in place)\n", .{e.tokens.len});
    return true;
}

/// Restore by move on demand: the admission pass found this slot's share does not fit (its first
/// append copies the prefix). Takes, on the entry the slot just restored, the checkout
/// `checkoutIfEligible` takes up front in SSD-first mode.
/// With `partial_ok` a partial share qualifies too, keeping only the shared prefix at donation (#109): the caller
/// asks for it only once evicting the other entries left no room, since it drops what the entry held past the share.
pub fn checkoutRestored(self: *HotPrefixCache, slot_id: usize, partial_ok: bool) bool {
    const used = self.last_restored_used orelse return false;
    if (!movable(restoreMoveEnabled(), self.pending_disk != null)) return false;
    for (self.entries.items) |*e| {
        if (e.last_used != used) continue;
        if (e.checkout != null) return false;
        const shared = @min(self.last_restored_shared, e.tokens.len);
        const kind: Checkout.Kind = if (wholeTakeEligible(e.tokens.len, shared))
            .whole
        else if (partial_ok and partialTakeEligible(e.tokens.len, shared, e.ring_cps != null))
            .{ .partial = shared }
        else
            return false;
        beginCheckout(e, slot_id, kind);
        log.info("  [hot-cache] checked out {d}-token entry to the slot (the share of {d} does not fit; the append donates in place)\n", .{ e.tokens.len, shared });
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
        const c = if (e.checkout) |*taken| taken else continue;
        if (c.slot != slot_id or c.donated != null) continue;
        c.donated = snapshotBytes(&e.snapshot) + e.rows_bytes;
        e.snapshot.releaseHandles();
        if (e.rows) |*rows| rows.releaseHandles();
        // The slot restored its draft-side history already; nothing reads a checked-out record's copy.
        const draft = e.dflash_bytes + e.mtp_bytes;
        if (e.dflash) |*d| d.deinit();
        e.dflash = null;
        if (e.mtp) |*m| m.deinit();
        e.mtp = null;
        e.dflash_bytes = 0;
        e.mtp_bytes = 0;
        e.kv_bytes -|= draft;
        self.current_kv_bytes -|= draft;
        switch (c.kind) {
            .whole => {},
            .partial => |keep| self.keepTakenPrefix(e, keep),
        }
    }
}

/// A partial takeover's record keeps only the prefix its prompt shares, and the checkpoints inside it: the rest
/// describes a text the slot is overwriting. Where that cut fails the record stays whole, and the turn's commit
/// lands beside it as a new entry until the slot's release drops the record.
pub fn keepTakenPrefix(self: *HotPrefixCache, e: *Entry, keep: usize) void {
    if (keep >= e.tokens.len) return;
    const tokens = self.allocator.dupe(u32, e.tokens[0..keep]) catch return;
    if (e.ssm_checkpoints) |cps| {
        const kept = truncateCheckpoints(self.allocator, cps, keep, self.arraysStream()) catch {
            self.allocator.free(tokens);
            // The slice may already have moved the QSA bank onto a kept checkpoint.
            self.rebillCheckpoints(e, checkpointListBytes(cps));
            return;
        };
        e.ssm_checkpoints = kept;
        self.rebillCheckpoints(e, checkpointListBytes(kept));
    }
    self.allocator.free(e.tokens);
    e.tokens = tokens;
}

/// Bytes this cache still bills that `slot_id` now owns (its donated checkout). Inference thread only.
pub fn donatedBytes(self: *const HotPrefixCache, slot_id: usize) u64 {
    var total: u64 = 0;
    for (self.entries.items) |*e| {
        const c = e.checkout orelse continue;
        if (c.slot == slot_id) total += c.donated orelse 0;
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
        const c = e.checkout orelse continue;
        if (c.slot != slot_id) continue;
        const tokens_len = e.tokens.len;
        endCheckout(e);
        if (c.donated == null) {
            // Nothing was handed over; only the mark is undone (no LRU bump for a request that never ran).
            log.info("  [hot-cache] checked-out entry returned intact: {s} ({d} tokens; the append never ran)\n", .{ reason, tokens_len });
            continue;
        }
        // The snapshot is already empty.
        self.evictAt(i, "checked-out entry dropped");
        log.info("  [hot-cache] checked-out entry dropped: {s} ({d} tokens; its KV died with the slot)\n", .{ reason, tokens_len });
    }
}

pub const TruncateCheckpointsError = error{ QsaHistorySliceFailed, CheckpointListCopyFailed };

/// The checkpoints at or below `tl`, the rest freed. The pooled indexer bank lives only on the latest
/// checkpoint, so it is sliced onto the last KEPT one first; a history-less hybrid entry would make
/// every later restore a silent miss. On error `cps` stays owned by the caller.
pub fn truncateCheckpoints(allocator: std.mem.Allocator, cps: []SSMCheckpoint, tl: usize, s: mlx.mlx_stream) TruncateCheckpointsError![]SSMCheckpoint {
    var kept: usize = 0;
    while (kept < cps.len and cps[kept].pos <= tl) kept += 1;
    if (kept == cps.len) return cps;
    if (kept > 0 and checkpointHasQsaPooled(&cps[cps.len - 1])) {
        sliceQsaHistoryOntoCheckpoint(&cps[kept - 1], &cps[cps.len - 1], cps[kept - 1].pos, s) catch
            return error.QsaHistorySliceFailed;
    }
    const shrunk = allocator.dupe(SSMCheckpoint, cps[0..kept]) catch return error.CheckpointListCopyFailed;
    for (cps[kept..]) |*cp| cp.deinit(allocator);
    allocator.free(cps);
    return shrunk;
}

pub fn checkpointListBytes(cps: []const SSMCheckpoint) u64 {
    var bytes: u64 = 0;
    for (cps) |*cp| bytes += ssmCheckpointBytes(cp);
    return bytes;
}

/// Bills `e`'s checkpoints at `bytes`, moving its bill and the cache's by the change.
pub fn rebillCheckpoints(self: *HotPrefixCache, e: *Entry, bytes: u64) void {
    e.kv_bytes = e.kv_bytes - e.ssm_bytes + bytes;
    self.current_kv_bytes = self.current_kv_bytes - e.ssm_bytes + bytes;
    e.ssm_bytes = bytes;
}

pub fn retainNewEntry(self: *HotPrefixCache, entry: Entry) !bool {
    var incoming = entry;
    errdefer freeEntryOwnedState(self.allocator, &incoming);
    while (self.max_entries != unlimited_entries and self.entries.items.len >= self.max_entries) {
        if (!self.evictOneLruProgress("count cap", incoming.cache_key)) break;
    }
    if (self.max_kv_bytes > 0) {
        while (self.current_kv_bytes + incoming.kv_bytes > self.max_kv_bytes and self.entries.items.len > 0) {
            const need = self.current_kv_bytes + incoming.kv_bytes - self.max_kv_bytes;
            if (!self.reclaimFromLru("byte budget", incoming.cache_key, need)) break;
        }
    }
    if (self.max_entries != unlimited_entries and self.entries.items.len >= self.max_entries) return self.declineIncoming(&incoming, "count cap");
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

pub fn declineIncoming(self: *HotPrefixCache, incoming: *Entry, cap: []const u8) bool {
    freeEntryOwnedState(self.allocator, incoming);
    log.info("  [hot-cache] skipped incoming snapshot: {s} full; checked-out entries prevent eviction\n", .{cap});
    return false;
}

pub fn shedCheckpointsToFit(self: *HotPrefixCache) void {
    if (self.max_kv_bytes == 0 or self.current_kv_bytes <= self.max_kv_bytes) return;
    if (self.entries.items.len == 0) return;
    const newest = &self.entries.items[prefix_policy.newestIndex(Entry, self.entries.items, .first).?];
    self.shedRingCheckpoints(newest);
    const cps = newest.ssm_checkpoints orelse return;
    var n = cps.len;
    var shed: usize = 0;
    while (n > 1 and self.current_kv_bytes > self.max_kv_bytes) {
        const drop = trunk_mod.ssmCheckpointDropIndex(cps[0..n], self.cp_thin, null);
        if (checkpointHasQsaPooled(&cps[drop])) {
            if (drop > 0) {
                sliceQsaHistoryOntoCheckpoint(&cps[drop - 1], &cps[drop], cps[drop - 1].pos, self.arraysStream()) catch |err| {
                    log.warn("  [hot-cache] shed dropped the QSA indexer history: {s}\n", .{@errorName(err)});
                };
            }
        }
        cps[drop].deinit(self.allocator);
        var k = drop;
        while (k + 1 < n) : (k += 1) cps[k] = cps[k + 1];
        n -= 1;
        shed += 1;
        // What the kept checkpoints hold, the bank a slice moved onto the one below included.
        self.rebillCheckpoints(newest, checkpointListBytes(cps[0..n]));
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
pub fn shedRingCheckpoints(self: *HotPrefixCache, e: *Entry) void {
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
        if (!self.reclaimFromLru("budget revised", null, self.current_kv_bytes - max_kv_bytes)) break;
    }
    self.shedCheckpointsToFit();
    self.logResident();
}

/// A shrink evicts now, not at the next retention; a live slot's checkout or restore keeps its entry until release.
pub fn setMaxEntries(self: *HotPrefixCache, max_entries: u32) void {
    self.max_entries = max_entries;
    if (max_entries == unlimited_entries or self.entries.items.len <= max_entries) return;
    while (self.entries.items.len > max_entries) {
        const idx = self.lruIndexExcluding(self.last_restored_used, null) orelse break;
        self.evictForSpace(idx, "count cap revised");
    }
    self.logResident();
}

pub fn evictOneLru(self: *HotPrefixCache, reason: []const u8, incoming_key: ?u64) void {
    const idx = self.lruIndexExcluding(null, incoming_key) orelse return;
    self.evictForSpace(idx, reason);
}

/// Frees at least `need` bytes from the LRU victim: trimmed to the last restore point that leaves it
/// small enough, else evicted whole. False when nothing was evictable (the budget loops' exit).
pub fn reclaimFromLru(self: *HotPrefixCache, reason: []const u8, incoming_key: ?u64, need: u64) bool {
    const idx = self.lruIndexExcluding(null, incoming_key) orelse return false;
    const why = self.trimVictim(idx, reason, need) orelse return true;
    var buf: [160]u8 = undefined;
    self.evictForSpace(idx, std.fmt.bufPrint(&buf, "{s}; not trimmed: {s}", .{ reason, why }) catch reason);
    return true;
}

/// The longest restore point of `list` (ascending) at or under `limit` and over the useful minimum
/// whose prefix, with the checkpoints at or below it, bills at most `allow`. No checkpoint is shed or added.
pub fn victimTrimLen(allow: u64, limit: usize, row_bytes: u64, list: []const SSMCheckpoint) ?usize {
    var k = list.len;
    while (k > 0) {
        k -= 1;
        const p = list[k].pos;
        if (p > limit or list[k].isReleased()) continue;
        if (p < MIN_TRIMMED_VICTIM_TOKENS) return null;
        var cps_cost: u64 = trimmedCheckpointBytes(list, k);
        for (list[0..k]) |*cp| cps_cost += ssmCheckpointBytes(cp);
        if (@as(u64, p) * row_bytes + cps_cost <= allow) return p;
    }
    return null;
}

/// Shrinks idle entry `idx` to a real copy of its prefix so that it frees `need` bytes, dropping its
/// spec payloads. Null when trimmed; otherwise why not, with the entry untouched.
pub fn trimVictim(self: *HotPrefixCache, idx: usize, reason: []const u8, need: u64) ?[]const u8 {
    const e = &self.entries.items[idx];
    const old_snap_bytes = snapshotBytes(&e.snapshot);
    // Only an entry whose whole bill is rows, checkpoints and spec payloads is re-billed here.
    if (e.ring_cps != null or ringedBytes(&e.snapshot) > 0 or
        e.kv_bytes != old_snap_bytes + e.ssm_bytes + e.dflash_bytes + e.mtp_bytes) return "state this trim does not bill";
    if (need >= e.kv_bytes) return "needs more than the entry holds";
    const allow = e.kv_bytes - need;
    const row_bytes = snapshotRowBytes(&e.snapshot);
    if (row_bytes == 0) return "no per-row price";
    // A trim never keeps media rows, as the incoming candidate's does not.
    const limit = @min(e.tokens.len -| 1, e.media_start orelse e.tokens.len);
    var tl: usize = undefined;
    if (e.ssm_checkpoints) |list| {
        if (self.qsa_history_required and !checkpointHasQsaPooled(&list[list.len - 1])) return "no QSA bank on the newest checkpoint";
        tl = victimTrimLen(allow, limit, row_bytes, list) orelse return "no restore point fits";
    } else {
        tl = @min(allow / row_bytes, limit);
        if (tl < MIN_TRIMMED_VICTIM_TOKENS) return "no prefix fits";
    }
    const stream = self.arraysStream();
    const tokens = self.allocator.dupe(u32, e.tokens[0..tl]) catch return "token copy failed";
    var snap = e.snapshot.trimmedCopy(tl, null, stream) catch {
        self.allocator.free(tokens);
        return "snapshot copy failed";
    };
    var new_cps: ?[]SSMCheckpoint = null;
    if (e.ssm_checkpoints) |cps| {
        new_cps = truncateCheckpoints(self.allocator, cps, tl, self.arraysStream()) catch |err| {
            self.allocator.free(tokens);
            snap.deinit();
            // The slice may already have moved the QSA bank onto a kept checkpoint.
            self.rebillCheckpoints(e, checkpointListBytes(cps));
            return @errorName(err);
        };
    }
    var ssm: u64 = 0;
    if (new_cps) |cps| for (cps) |*cp| {
        ssm += ssmCheckpointBytes(cp);
    };
    const old_len = e.tokens.len;
    const old_bytes = e.kv_bytes;
    const new_bytes = snapshotBytes(&snap) + ssm;
    self.allocator.free(e.tokens);
    e.tokens = tokens;
    e.snapshot.deinit();
    e.snapshot = snap;
    e.ssm_checkpoints = new_cps;
    e.ssm_bytes = ssm;
    if (e.dflash) |*d| d.deinit();
    e.dflash = null;
    e.dflash_bytes = 0;
    if (e.mtp) |*m| m.deinit();
    e.mtp = null;
    e.mtp_bytes = 0;
    if (e.media_start != null) {
        e.media_start = null;
        e.vision_key = 0;
    }
    e.kv_bytes = new_bytes;
    self.current_kv_bytes = self.current_kv_bytes - old_bytes + new_bytes;
    log.info("  [hot-cache] trimmed idle entry ({s}; key={x}): {d} -> {d} tokens, freed {d:.2} MB\n", .{
        reason, e.cache_key, old_len, tl, @as(f64, @floatFromInt(old_bytes - new_bytes)) / (1024.0 * 1024.0),
    });
    return null;
}

/// An entry dropped for room (a budget, the entry cap, a prefill): a tier-counter eviction, which an SSD-first spill is not.
pub fn evictForSpace(self: *HotPrefixCache, idx: usize, reason: []const u8) void {
    cache_stats.process.recordEviction(.hot, self.entries.items[idx].kv_bytes);
    self.evictAt(idx, reason);
}



/// False when nothing was evictable (every survivor is checked out) — the termination
/// condition every budget loop needs, since `evictOneLru` is then a no-op.
pub fn evictOneLruProgress(self: *HotPrefixCache, reason: []const u8, incoming_key: ?u64) bool {
    const before = self.entries.items.len;
    self.evictOneLru(reason, incoming_key);
    return self.entries.items.len != before;
}

pub fn dropLastRestored(self: *HotPrefixCache) bool {
    var dropped = false;
    if (self.last_restored_disk_id) |id| {
        self.last_restored_disk_id = null;
        self.last_restored_held = false;
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

pub fn evictAt(self: *HotPrefixCache, lru_idx: usize, reason: []const u8) void {
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


/// Evicts whole retained entries oldest-first until `tokens` are freed. `protect` names the entries a live slot
/// uses and the candidate's own restore match; nothing else is protected (a previous request's restore is not).
/// Returns the tokens freed.
pub fn evictOldestForTokens(self: *HotPrefixCache, tokens: u64, protect: []const u64) u64 {
    var freed: u64 = 0;
    while (freed < tokens) {
        var idx: ?usize = null;
        var oldest: u64 = std.math.maxInt(u64);
        for (self.entries.items, 0..) |e, i| {
            if (e.checkout != null) continue;
            if (std.mem.indexOfScalar(u64, protect, e.id) != null) continue;
            if (e.last_used < oldest) {
                oldest = e.last_used;
                idx = i;
            }
        }
        const i = idx orelse break;
        const held: u64 = self.entries.items[i].tokens.len;
        self.evictAt(i, "max_kv_tokens");
        freed +|= held;
    }
    return freed;
}


/// Evict least-recently-used entries until `fits()` says the request fits (#353): a cached
/// prefix is an optimization, the request is the work. Never evicts the entry this request
/// restored from; `fits` is re-asked after every eviction.
pub const SHARED_RATIO_MIN_BYTES = prefix_policy.SHARED_RATIO_MIN_BYTES;
pub const SHARED_RETURN_DIVISOR = prefix_policy.SHARED_RETURN_DIVISOR;
pub const RESTORE_PIN_MIN_BYTES = prefix_policy.RESTORE_PIN_MIN_BYTES;
pub const RESTORE_PIN_RATIO = prefix_policy.RESTORE_PIN_RATIO;
pub const restoreWouldPinEntry = prefix_policy.restoreWouldPinEntry;

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
    _ = mlx.mlx_synchronize(self.arraysStream());
    while (!fits(ctx)) {
        const idx = self.lruIndexExcluding(if (protect_restored) self.last_restored_used else null, null) orelse break;
        // Accounting bytes are what the entry was billed; live bytes are what the allocator got back.
        var live_before: usize = 0;
        _ = mlx.mlx_get_active_memory(&live_before);
        const acct_before = self.current_kv_bytes;
        self.evictForSpace(idx, "admitting a long prefill");
        var live_after: usize = 0;
        _ = mlx.mlx_get_active_memory(&live_after);
        const freed_live: u64 = @as(u64, live_before) -| @as(u64, live_after);
        report.entries += 1;
        report.bytes += freed_live;
        const acct_delta = acct_before -| self.current_kv_bytes;
        report.accounted_bytes += acct_delta;
        // Judge only entries big enough for the ratio to mean something.
        if (prefix_policy.evictionWasShared(acct_delta, freed_live)) {
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

pub fn lruIndexExcluding(self: *const HotPrefixCache, protect: ?u64, incoming_key: ?u64) ?usize {
    return prefix_policy.lruIndexExcluding(Entry, self.entries.items, protect, incoming_key);
}

