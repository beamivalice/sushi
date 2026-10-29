const trf = @import("transformer.zig");
const std = @import("std");
const mlx = @import("mlx.zig");
const kv_quant = @import("kv_quant.zig");
const KVQuantConfig = kv_quant.KVQuantConfig;
const model_mod = @import("model.zig");
const log = @import("log.zig");
const ModelConfig = model_mod.ModelConfig;
const testing = std.testing;
const test_support = @import("test_support.zig");
const attn256RandBf16 = test_support.attn256RandBf16;
const qkvExpectBitsEqual = test_support.qkvExpectBitsEqual;
const qkvIdentityArray = test_support.qkvIdentityArray;
const qsaArraysAllEqual = test_support.qsaArraysAllEqual;
const snapshotsBitEqual = test_support.snapshotsBitEqual;
const testKVWide = test_support.testKVWide;
const ForwardCtx = trf.ForwardCtx;
const Transformer = trf.Transformer;
const kvAppendDummyAttn = trf.kvAppendDummyAttn;
const kvAppendOperandsEqual = trf.kvAppendOperandsEqual;
const kvAppendPacked = trf.kvAppendPacked;
const kvAppendPrefill = trf.kvAppendPrefill;
const kvEntriesBytes = trf.kvEntriesBytes;
const materializedOwnedCopy = trf.materializedOwnedCopy;
const maxAbsDiffF32 = trf.maxAbsDiffF32;
const ringCheckpointFeed = trf.ringCheckpointFeed;
const sliceAttentionSeq = trf.sliceAttentionSeq;
const swaRingChunk = trf.swaRingChunk;
const verifyRollbackEvalStorage = trf.verifyRollbackEvalStorage;
const verifyRollbackValidPrefixEqual = trf.verifyRollbackValidPrefixEqual;

pub const KvPrefixView = struct {
    view: DenseKVView,
    owned: [8]mlx.mlx_array = @splat(.{}),

    pub fn init(s: mlx.mlx_stream, source: DenseKVView, length: c_int) !KvPrefixView {
        return initRange(s, source, 0, length);
    }

    /// Rows `[start, end)` of `source`: the view a decode tick saw when `end` was its last key.
    pub fn initRange(s: mlx.mlx_stream, source: DenseKVView, start: c_int, end: c_int) !KvPrefixView {
        var result = KvPrefixView{ .view = source };
        errdefer result.deinit();
        inline for (.{ "k", "v", "k_triple_q", "k_triple_scales", "k_triple_biases", "v_triple_q", "v_triple_scales", "v_triple_biases" }, 0..) |name, i| {
            if (i < 2 or source.has_quant_triple) {
                result.owned[i] = try sliceAttentionSeq(s, @field(source, name), start, end);
                @field(result.view, name) = result.owned[i];
            }
        }
        result.view.owned = false;
        return result;
    }
    pub fn deinit(self: *KvPrefixView) void {
        for (&self.owned) |*value| if (value.ctx != null) {
            _ = mlx.mlx_array_free(value.*);
            value.* = .{};
        };
    }
};

pub const KVCacheEntry = struct {
    // Storage. In `off` (dense bf16) mode `keys`/`values` are the full
    // [B,H,T,D] buffers and the `*_scales`/`*_biases` fields stay null. In
    // `affine` mode `keys`/`values` hold packed uint32 codes
    // ([B,H,T, D*bits/32]) and the matching scales/biases hold
    // [B,H,T, D/group_size] bf16. Switched on `KVCache.config.scheme`.
    keys: mlx.mlx_array,
    values: mlx.mlx_array,
    keys_scales: mlx.mlx_array,
    keys_biases: mlx.mlx_array,
    values_scales: mlx.mlx_array,
    values_biases: mlx.mlx_array,

    // Views: same layout as the storage fields above but trimmed to
    // [..., offset, ...] (or last `sw` entries during sliding-window decode).
    // SDPA reads dense arrays via `KVCache.denseView`; in `off` mode the
    // dense pair aliases `key_view`/`value_view`, in `affine` mode the
    // dense pair is freshly dequantized from these triples on read.
    key_view: mlx.mlx_array,
    value_view: mlx.mlx_array,
    key_scales_view: mlx.mlx_array,
    key_biases_view: mlx.mlx_array,
    value_scales_view: mlx.mlx_array,
    value_biases_view: mlx.mlx_array,

    offset: usize, // rows STORED (may be < buffer capacity)
    initialized: bool,
    shared_view: bool = false,
    /// Rows dropped off the FRONT by the sliding ring. The absolute token count
    /// this entry describes is `base + offset`; every caller that speaks in
    /// prompt positions (truncate, trimmedCopy) goes through `base`.
    base: usize = 0,
    /// This layer's storage is a ring: its attention reads the tail only, so
    /// rows below the window were dropped and no rewind can reach them.
    ringed: bool = false,
};

/// Materialized dense `[B,H,T,D]` K/V pair handed to SDPA. Owns its arrays
/// only when `owned == true` (i.e. when the cache stores quantized data and
/// `KVCache.denseView` had to dequantize on read). In dense mode `k`/`v`
/// alias the cache's `key_view`/`value_view` and `deinit` is a no-op.
///
/// Phase 2 (fused-attn): in `.affine` mode the view ALSO carries borrowed
/// references to the cache's quantized K/V triples (`k_triple_q`, etc.).
/// MiMo's packed attention arms read the triples directly;
/// everyone else uses `k`/`v` as before
/// and pays for the dense materialization. The arrays are non-owning
/// borrows of the cache's `key_view` / `key_scales_view` / `key_biases_view`
/// (and the V trio) — the cache keeps them alive for the request's lifetime.
pub const DenseKVView = struct {
    k: mlx.mlx_array,
    v: mlx.mlx_array,
    owned: bool,

    /// Borrowed quant triples. Set to `.ctx = null` when not applicable
    /// (scheme == .off). Read-only; the cache owns these handles.
    k_triple_q: mlx.mlx_array = .{ .ctx = null },
    k_triple_scales: mlx.mlx_array = .{ .ctx = null },
    k_triple_biases: mlx.mlx_array = .{ .ctx = null },
    v_triple_q: mlx.mlx_array = .{ .ctx = null },
    v_triple_scales: mlx.mlx_array = .{ .ctx = null },
    v_triple_biases: mlx.mlx_array = .{ .ctx = null },
    /// True iff the triple fields above are populated. Lets call sites
    /// avoid checking `.ctx == null` on every field.
    has_quant_triple: bool = false,
    /// Quant params copied off the cache config so call sites don't need
    /// a pointer to it.
    bits: u8 = 0,
    group_size: u32 = 0,
    /// The triples slice a ring's pre-compaction buffer and are this view's to free
    /// (`KVCache.handOffRingViews`); otherwise they are the cache's.
    owns_triple: bool = false,

    pub fn deinit(self: *DenseKVView) void {
        if (self.owned) {
            _ = mlx.mlx_array_free(self.k);
            _ = mlx.mlx_array_free(self.v);
            self.k = mlx.mlx_array_new();
            self.v = mlx.mlx_array_new();
            self.owned = false;
        }
        if (self.owns_triple) {
            for ([_]*mlx.mlx_array{ &self.k_triple_q, &self.k_triple_scales, &self.k_triple_biases, &self.v_triple_q, &self.v_triple_scales, &self.v_triple_biases }) |a| {
                _ = mlx.mlx_array_free(a.*);
                a.* = .{ .ctx = null };
            }
            self.owns_triple = false;
        }
    }
};

pub fn newEmptyKVEntry() KVCacheEntry {
    return .{
        .keys = mlx.mlx_array_new(),
        .values = mlx.mlx_array_new(),
        .keys_scales = mlx.mlx_array_new(),
        .keys_biases = mlx.mlx_array_new(),
        .values_scales = mlx.mlx_array_new(),
        .values_biases = mlx.mlx_array_new(),
        .key_view = mlx.mlx_array_new(),
        .value_view = mlx.mlx_array_new(),
        .key_scales_view = mlx.mlx_array_new(),
        .key_biases_view = mlx.mlx_array_new(),
        .value_scales_view = mlx.mlx_array_new(),
        .value_biases_view = mlx.mlx_array_new(),
        .offset = 0,
        .initialized = false,
    };
}

/// Can this entry still show the window a query at `len` reads? A ringed entry's `base`
/// rows are gone for good; every other entry holds a prefix and clamps anywhere.
fn ringEntryServes(e: *const KVCacheEntry, window: u32, len: usize) bool {
    if (!e.initialized or !e.ringed) return true;
    if (len < e.base) return false;
    return len - e.base >= @min(len, @as(usize, window));
}

/// Reset a cache entry to the empty state, freeing all storage + view
/// handles. Mirror of the per-entry reset in `KVCache.restore`; used by the
/// disk tier (kv_disk_cache.zig) when installing restored buffers.
pub fn resetKVEntry(e: *KVCacheEntry) void {
    freeKVEntry(e);
    e.* = newEmptyKVEntry();
}

pub fn freeKVEntry(e: *KVCacheEntry) void {
    _ = mlx.mlx_array_free(e.keys);
    _ = mlx.mlx_array_free(e.values);
    _ = mlx.mlx_array_free(e.keys_scales);
    _ = mlx.mlx_array_free(e.keys_biases);
    _ = mlx.mlx_array_free(e.values_scales);
    _ = mlx.mlx_array_free(e.values_biases);
    _ = mlx.mlx_array_free(e.key_view);
    _ = mlx.mlx_array_free(e.value_view);
    _ = mlx.mlx_array_free(e.key_scales_view);
    _ = mlx.mlx_array_free(e.key_biases_view);
    _ = mlx.mlx_array_free(e.value_scales_view);
    _ = mlx.mlx_array_free(e.value_biases_view);
}

/// "No width bound" for `slidingTailSpan`: every consumer derives its query offset as
/// `kv_len - q_len`, so width needs no bound.
pub const SLIDING_TRIM_UNBOUNDED: usize = std.math.maxInt(usize);

/// How many TAIL KV entries a forward of `q_len` queries needs from a SLIDING
/// layer. The last query reaches back `window`; the first of the block reaches
/// back `window + q_len - 1`, so the block's union is that many entries.
/// Returns 0 — meaning "hand back the whole cache, no trim" — when the layer
/// does not slide or the block is wider than `max_width`.
///
/// This is what makes a spec verify cost what a decode step costs on a sliding
/// layer: at width 1 it has always returned `window`, but a block-wide verify
/// used to fall through untrimmed and read the ENTIRE cache on every sliding
/// layer, so a 3:1 local:global model paid full attention on 75% of its layers
/// once past the window.
pub fn slidingTailSpan(window: u32, q_len: usize, max_width: usize) u32 {
    if (window == 0 or q_len == 0 or q_len > max_width) return 0;
    return window + @as(u32, @intCast(q_len)) - 1;
}

/// What a sliding layer's K/V view looks like for THIS forward: the `max_kv`
/// to hand `KVCache.update` and the length the view will actually have.
///
/// ONE chokepoint for the policy — every arch path asks it rather than
/// re-deriving the predicate, which is how the standard and MoE paths drifted
/// apart in the first place.
pub const SlidingView = struct {
    /// `max_kv` for `KVCache.update`; 0 = hand back the whole cache.
    span: u32,
    /// The K/V length a sliding layer will see. EVERY mask builder must be
    /// handed this, not `total_kv`: they derive the query offset relatively
    /// (`offset_val = kv_len - q_len`), so a trimmed view with the trimmed
    /// length is correct and a trimmed view with the full length is not.
    kv_len: c_int,
};

var sliding_block_trim_logged: bool = false; // one-shot log guard

pub var row_axis_verify_logged: bool = false;
pub var verify_fail_after_flush: ?usize = null;
pub var deferred_ple_group_syncs: u64 = 0;
/// Slots one batched forward carries (twin of `slot_admission.MAX_BATCH_GROUP`).
pub const MAX_BATCH_ROWS = 32;
pub fn slidingViewFor(cfg: *const ModelConfig, total_kv: c_int, seq_len: c_int) SlidingView {
    const span: u32 = if (cfg.has_sliding_window)
        slidingTailSpan(cfg.sliding_window, @intCast(seq_len), SLIDING_TRIM_UNBOUNDED)
    else
        0;
    const kv_len: c_int = if (span > 0) @min(total_kv, @as(c_int, @intCast(span))) else total_kv;
    // One-shot: fires only when the BLOCK half of the trim did something — decode has always
    // trimmed to the window.
    if (!sliding_block_trim_logged and span > cfg.sliding_window and kv_len < total_kv) {
        sliding_block_trim_logged = true;
        log.info("[sliding] block trim engaged: window={d} q_len={d} span={d} kv={d} of {d}\n", .{ cfg.sliding_window, seq_len, span, kv_len, total_kv });
    }
    return .{ .span = span, .kv_len = kv_len };
}

pub const KVCache = struct {
    entries: []KVCacheEntry,
    step: usize, // absolute sequence position (not affected by sliding window trimming)
    allocator: std.mem.Allocator,
    config: KVQuantConfig,
    /// Capacity, in tokens, this cache was asked to hold up front (the admission bill's
    /// `reservedCacheTokens`). A grow is not in place, so a long prefill's peak carried a
    /// second copy of ~6 GB of KV at 458k; reserving on the first grow removes the transient.
    reserve_tokens: usize = 0,
    /// Window a sliding layer's storage rings at (`ModelConfig.sliding_window`);
    /// 0 = every layer stores the whole sequence. A layer rings only on a
    /// forward that hands `update` a non-zero `max_seq` — the one predicate
    /// that says its attention reads the tail and nothing below it.
    swa_ring_window: u32 = 0,
    /// Restore points (`ringMark`) each ringed write fills once it reaches their `step`, before
    /// its compaction drops the rows. Caller-owned; empty outside a prefill.
    ring_marks: []KVCacheSnapshot = &.{},

    pub fn setSwaRing(self: *KVCache, window: u32) void {
        self.swa_ring_window = window;
    }

    /// Rows a ringed entry may hold before it compacts.
    fn ringCap(window: u32) usize {
        return @as(usize, window) + ModelConfig.SWA_RING_SLACK;
    }

    /// Rows a compaction keeps. The gap below `ringCap` is rewind headroom:
    /// a full-reuse restore clamps back a token and a prefix match lands
    /// wherever the histories diverged, and neither may fall off the ring.
    fn ringKeep(window: u32) usize {
        return @as(usize, window) + ModelConfig.SWA_RING_SLACK / 2;
    }

    /// Absolute tokens this layer's state describes, rings included.
    pub fn absSeqLen(self: *const KVCache, layer: u32) usize {
        const entry = &self.entries[layer];
        if (!entry.initialized) return 0;
        return entry.base + entry.offset;
    }

    pub fn init(allocator: std.mem.Allocator, num_layers: u32) !KVCache {
        return initWithConfig(allocator, num_layers, KVQuantConfig.dense);
    }

    pub fn initWithConfig(allocator: std.mem.Allocator, num_layers: u32, config: KVQuantConfig) !KVCache {
        const entries = try allocator.alloc(KVCacheEntry, num_layers);
        errdefer allocator.free(entries);
        for (entries) |*e| {
            e.* = newEmptyKVEntry();
        }
        return .{ .entries = entries, .step = 0, .allocator = allocator, .config = config };
    }

    pub fn deinit(self: *KVCache) void {
        for (self.entries) |*e| {
            freeKVEntry(e);
        }
        self.allocator.free(self.entries);
    }

    /// Swap this cache for a freshly-built one under `config`. BUILD FIRST,
    /// then free: `initWithConfig` is fallible, and a caller that reached it
    /// through `cache.deinit(); cache = try init(...)` left a FREED cache
    /// installed on the Transformer on the error path, whose own `deinit`
    /// then freed every handle a second time. The class is "a fallible
    /// re-init behind a deinit"; keeping the order inside ONE helper is what
    /// stops it recurring: every live re-init goes through here, never `.cache = try init` behind a deinit.
    pub fn reinit(self: *KVCache, num_layers: u32, config: KVQuantConfig) !void {
        var fresh = try initWithConfig(self.allocator, num_layers, config);
        fresh.swa_ring_window = self.swa_ring_window;
        self.deinit();
        self.* = fresh;
    }

    /// GPU bytes of the buffers this cache owns, at their capacity (a ringed layer's ring). A
    /// restored share is billed to the hot-cache entry that owns its buffers (`shared_view`); the
    /// COW grow that materializes the slot's own copy clears the flag and bills here.
    pub fn residentBytes(self: *const KVCache) u64 {
        return kvEntriesBytes(self.entries);
    }

    /// Capture cache state for speculative-decoding rollback (PLD/drafter).
    /// Snapshots own array handles that share the underlying buffer with the
    /// source via refcount — cheap (no data copy) and immune to subsequent
    /// `update()` calls (which create new buffer handles when growing).
    /// `*_view` fields are excluded because `update()` recreates them every
    /// call.
    pub fn snapshot(self: *const KVCache) !KVCacheSnapshot {
        const out = try self.allocator.alloc(KVCacheEntry, self.entries.len);
        var built: usize = 0;
        errdefer {
            for (out[0..built]) |*e| freeKVEntry(e);
            self.allocator.free(out);
        }
        for (self.entries, 0..) |src, i| {
            out[i] = newEmptyKVEntry();
            built = i + 1;
            out[i].offset = src.offset;
            out[i].base = src.base;
            out[i].ringed = src.ringed;
            out[i].initialized = src.initialized;
            if (src.initialized) {
                try mlx.check(mlx.mlx_array_set(&out[i].keys, src.keys));
                try mlx.check(mlx.mlx_array_set(&out[i].values, src.values));
                if (self.config.scheme != .off) {
                    try mlx.check(mlx.mlx_array_set(&out[i].keys_scales, src.keys_scales));
                    try mlx.check(mlx.mlx_array_set(&out[i].keys_biases, src.keys_biases));
                    try mlx.check(mlx.mlx_array_set(&out[i].values_scales, src.values_scales));
                    try mlx.check(mlx.mlx_array_set(&out[i].values_biases, src.values_biases));
                }
            }
        }
        return .{ .entries = out, .step = self.step, .allocator = self.allocator, .config = self.config, .swa_ring_window = self.swa_ring_window };
    }

    /// The retention snapshot: `snapshot`, but a ringed entry's storage is
    /// copied down to the rows it actually holds. A ring buffer is allocated at
    /// `ringCap` from its first token, so the refcount share billed — and
    /// pinned — the dead capacity too (a 25-token MiMo prompt cost a full
    /// 640-row ring per sliding layer). The retained rows are unchanged, so
    /// `base`, `offset` and every rewind the ring's slack allows are as well.
    pub fn snapshotRetained(self: *const KVCache, s: mlx.mlx_stream) !KVCacheSnapshot {
        var snap = try self.snapshot();
        errdefer snap.deinit();
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        var copied: usize = 0;
        for (snap.entries) |*e| {
            if (!e.initialized or !e.ringed) continue;
            if (e.offset >= bufferCapacity(e.keys)) continue;
            var bufs: [6]*mlx.mlx_array = undefined;
            var n: usize = 2;
            bufs[0] = &e.keys;
            bufs[1] = &e.values;
            if (self.config.scheme != .off) {
                bufs[2] = &e.keys_scales;
                bufs[3] = &e.keys_biases;
                bufs[4] = &e.values_scales;
                bufs[5] = &e.values_biases;
                n = 6;
            }
            for (bufs[0..n]) |buf| {
                const owned = try trimRowsOwned(buf.*, e.offset, s);
                _ = mlx.mlx_array_free(buf.*);
                buf.* = owned;
                _ = mlx.mlx_vector_array_append_value(vec, owned);
            }
            copied += 1;
        }
        // Without the eval the lazy copies pin the ring buffers this exists to release.
        if (copied > 0) _ = mlx.mlx_eval(vec);
        return snap;
    }

    /// Replace cache state with `snap`. Frees current entries' arrays first;
    /// re-binds via refcount-share from snapshot. After restore, the next
    /// `update()` will recreate `*_view` fields from the restored buffers.
    pub fn restore(self: *KVCache, snap: *const KVCacheSnapshot) !void {
        std.debug.assert(self.entries.len == snap.entries.len);
        for (self.entries, snap.entries) |*dst, src| try self.bindEntry(dst, src);
        self.step = snap.step;
    }

    fn bindEntry(self: *const KVCache, dst: *KVCacheEntry, src: KVCacheEntry) !void {
        freeKVEntry(dst);
        dst.* = newEmptyKVEntry();
        dst.offset = src.offset;
        dst.base = src.base;
        dst.ringed = src.ringed;
        dst.initialized = src.initialized;
        if (src.initialized) {
            try mlx.check(mlx.mlx_array_set(&dst.keys, src.keys));
            try mlx.check(mlx.mlx_array_set(&dst.values, src.values));
            if (self.config.scheme != .off) {
                try mlx.check(mlx.mlx_array_set(&dst.keys_scales, src.keys_scales));
                try mlx.check(mlx.mlx_array_set(&dst.keys_biases, src.keys_biases));
                try mlx.check(mlx.mlx_array_set(&dst.values_scales, src.values_scales));
                try mlx.check(mlx.mlx_array_set(&dst.values_biases, src.values_biases));
            }
            dst.shared_view = true;
        }
    }

    /// A restore point of a ringed cache: each ringed layer's rows `[pos - n, pos)` as owned
    /// copies, n = window + `SWA_RING_CHECKPOINT_BACKOFF`, or as many as the ring holds down to
    /// the window (a ring restored off a checkpoint holds no more). The global layers stay
    /// empty: they hold a prefix and clamp anywhere. Null when no layer rings.
    pub fn ringCheckpoint(self: *const KVCache, pos: usize, s: mlx.mlx_stream) !?KVCacheSnapshot {
        if (self.swa_ring_window == 0) return null;
        var cp = try self.ringMark(pos);
        errdefer cp.deinit();
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        for (self.entries, cp.entries) |*src, *dst| {
            if (!src.initialized or !src.ringed) continue;
            const rows = self.ringRowsAt(src, pos) orelse return error.SlidingRingRewindPastWindow;
            try self.copyRingRows(src, dst, pos, rows, s, vec);
        }
        if (mlx.mlx_vector_array_size(vec) == 0) {
            cp.deinit();
            return null;
        }
        // Without the eval the lazy copies pin the ring buffers they were sliced from.
        _ = mlx.mlx_eval(vec);
        return cp;
    }

    /// An empty restore point at `pos`, for `ring_marks` to fill.
    pub fn ringMark(self: *const KVCache, pos: usize) !KVCacheSnapshot {
        const out = try self.allocator.alloc(KVCacheEntry, self.entries.len);
        for (out) |*e| e.* = newEmptyKVEntry();
        return .{ .entries = out, .step = pos, .allocator = self.allocator, .config = self.config, .swa_ring_window = self.swa_ring_window };
    }

    /// Did every ringed layer fill `mark`? Only then does it restore.
    pub fn ringMarkComplete(self: *const KVCache, mark: *const KVCacheSnapshot) bool {
        var any = false;
        for (self.entries, mark.entries) |*e, *m| {
            if (!e.initialized or !e.ringed) continue;
            if (!m.initialized) return false;
            any = true;
        }
        return any;
    }

    /// Rows a restore point of `src` at `pos` holds: window + backoff at most, at least the window
    /// (or every row from 0). Null when the ring does not hold them (yet, or any more).
    fn ringRowsAt(self: *const KVCache, src: *const KVCacheEntry, pos: usize) ?usize {
        if (pos < src.base or pos > src.base + src.offset) return null;
        const want: usize = @as(usize, self.swa_ring_window) + ModelConfig.SWA_RING_CHECKPOINT_BACKOFF;
        const rows = @min(pos - src.base, want);
        if (rows < @min(pos, @as(usize, self.swa_ring_window))) return null;
        return rows;
    }

    /// Owned copies of `src`'s rows `[pos - rows, pos)` into the empty `dst`, each appended to `vec`.
    fn copyRingRows(self: *const KVCache, src: *const KVCacheEntry, dst: *KVCacheEntry, pos: usize, rows: usize, s: mlx.mlx_stream, vec: mlx.mlx_vector_array) !void {
        const from = pos - rows - src.base;
        dst.initialized = true;
        dst.ringed = true;
        dst.base = pos - rows;
        dst.offset = rows;
        const pairs = [_]struct { *mlx.mlx_array, mlx.mlx_array }{
            .{ &dst.keys, src.keys },                   .{ &dst.values, src.values },
            .{ &dst.keys_scales, src.keys_scales },     .{ &dst.keys_biases, src.keys_biases },
            .{ &dst.values_scales, src.values_scales }, .{ &dst.values_biases, src.values_biases },
        };
        const n: usize = if (self.config.scheme != .off) 6 else 2;
        for (pairs[0..n]) |p| {
            const owned = try rowsOwned(p[1], from, from + rows, s);
            _ = mlx.mlx_array_free(p[0].*);
            p[0].* = owned;
            _ = mlx.mlx_vector_array_append_value(vec, owned);
        }
    }

    /// Fill each ring mark this layer's write has reached, from rows the ring still holds. Reads
    /// only; a failed copy leaves its mark incomplete, and an incomplete mark is never kept.
    fn captureRingMarks(self: *KVCache, layer: u32, s: mlx.mlx_stream) void {
        const src = &self.entries[layer];
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        for (self.ring_marks) |*m| {
            const dst = &m.entries[layer];
            if (dst.initialized) continue;
            const rows = self.ringRowsAt(src, m.step) orelse continue;
            self.copyRingRows(src, dst, m.step, rows, s, vec) catch {
                freeKVEntry(dst);
                dst.* = newEmptyKVEntry();
            };
        }
        // Left lazy, the copies would pin the pre-compaction buffer until the chunk's eval.
        if (mlx.mlx_vector_array_size(vec) > 0) _ = mlx.mlx_async_eval(vec);
    }

    /// Put a ring checkpoint's rows under the ringed layers of a cache restored from the
    /// same entry. `step` and the global layers stay; the caller's `truncate` clamps both.
    pub fn restoreRing(self: *KVCache, cp: *const KVCacheSnapshot) !void {
        std.debug.assert(self.entries.len == cp.entries.len);
        for (self.entries, cp.entries) |*dst, src| {
            if (src.initialized) try self.bindEntry(dst, src);
        }
    }

    const chunk_step = 256;

    /// Cap on the slack a single growth event adds, in tokens. Keeps the
    /// proportional term from over-allocating at very long context (25% of
    /// 200 K would be 50 K tokens of headroom).
    const max_grow_tokens = 8192;

    /// PURE: the capacity the K/V buffers must be grown to so `needed` tokens
    /// fit.
    ///
    /// Growing by a fixed `chunk_step` (the pre-#110 policy) means one grow-copy-free every 256 tokens, and
    /// every superseded buffer is orphaned: MLX only reuses a cached buffer
    /// when `cached_size < requested + 2*page_size` (`buffer_cache.h`), and the
    /// next KV buffer is a full chunk past that ~32 KB tolerance. On a 64
    /// KiB/token model at 89 K context that strands ~5.6 GB per 256 generated
    /// tokens in MLX's allocator pool — invisible to `active_bytes`, which is
    /// how the reporter's process reached 81 GB while the panel read 19.6.
    ///
    /// Proportional growth (+25%, floored at one chunk, capped at
    /// `max_grow_tokens`) turns that into ~27 growth events over a 100 K-token
    /// walk instead of 391.
    pub fn nextCapacityPolicy(current_cap: usize, needed: usize) usize {
        if (needed <= current_cap) return current_cap;
        const grow = @min(@max(current_cap / 4, chunk_step), max_grow_tokens);
        const target = @max(needed, current_cap + grow);
        const n_chunks = (target + chunk_step - 1) / chunk_step;
        return n_chunks * chunk_step;
    }

    fn nextCapacity(current_cap: usize, needed: usize) usize {
        return nextCapacityPolicy(current_cap, needed);
    }

    /// Tokens of capacity a request reserves: prompt + allowed generation + one chunk of slack.
    /// The one definition; `server.reservedCacheTokens` delegates here.
    pub fn reservedTokens(seq: u64, max_tokens: u64, chunk: u64, ctx: u64) u64 {
        // Short prompts keep the proportional policy: reserving costs them `max_tokens` of KV they may never use.
        if (seq < RESERVE_MIN_TOKENS) return 0;
        // An omitted `max_tokens` is maxInt(u32)/4 (~26 TB of KV), so bound by the context AND
        // by the generation headroom: reserving `ctx - prompt` billed 13 GB at 383k on a
        // 1M-context qwen4_exp. `ctx == 0` = unknown, no clamp.
        const want: u64 = @min(max_tokens, RESERVE_GEN_HEADROOM);
        const head_room: u64 = if (ctx > seq) @min(want, ctx - seq) else if (ctx > 0) 0 else want;
        return seq +| head_room +| @max(chunk, 1);
    }

    /// Tokens of generation the reservation pre-buys. Past it a decode grow takes the ordinary
    /// +25% policy, which happens between decode steps, never inside a chunk's lazy graph.
    /// 8192 covers essentially every real request at ~100 MB of KV on qwen4_exp at 8-bit.
    pub const RESERVE_GEN_HEADROOM: u64 = 8192;

    /// Prompt length at or above which a request reserves its capacity; below it the flat
    /// runtime floor already covers the transient.
    pub const RESERVE_MIN_TOKENS: u64 = 32768;

    /// How many times a KV capacity buffer actually grew (the moments a second copy of the whole cache exists).
    pub var kv_cap_buf_grows: usize = 0;

    /// Ask this cache to hold `tokens` up front. Idempotent and monotone.
    pub fn reserve(self: *KVCache, tokens: usize) void {
        if (tokens > self.reserve_tokens) self.reserve_tokens = tokens;
    }

    /// The capacity a grow takes: the proportional policy, raised to the reservation when there is one.
    fn nextCapacityReserved(self: *const KVCache, current_cap: usize, needed: usize) usize {
        const policy = nextCapacity(current_cap, needed);
        if (self.reserve_tokens <= policy) return policy;
        // Round the reservation the same way the policy rounds.
        const n_chunks = (self.reserve_tokens + chunk_step - 1) / chunk_step;
        return n_chunks * chunk_step;
    }

    /// Grow every restored non-ringed layer below the reservation to it, one layer per eval.
    /// Called before a warm prefill writes: left to the decode, the first step past the restored
    /// capacity grows every layer inside one forward, all the old buffers alive beside the new.
    pub fn growToReservation(self: *KVCache, s: mlx.mlx_stream) !void {
        if (self.reserve_tokens == 0) return;
        const new_cap: c_int = @intCast((self.reserve_tokens + chunk_step - 1) / chunk_step * chunk_step);
        const n: usize = if (self.config.scheme == .off) 2 else 6;
        for (self.entries) |*entry| {
            if (!entry.initialized or entry.ringed) continue;
            if (bufferCapacity(entry.keys) >= self.reserve_tokens) continue;
            const views = [_]*mlx.mlx_array{ &entry.key_view, &entry.value_view, &entry.key_scales_view, &entry.key_biases_view, &entry.value_scales_view, &entry.value_biases_view };
            for (views) |view| {
                _ = mlx.mlx_array_free(view.*);
                view.* = mlx.mlx_array_new();
            }
            const bufs = [_]*mlx.mlx_array{ &entry.keys, &entry.values, &entry.keys_scales, &entry.keys_biases, &entry.values_scales, &entry.values_biases };
            const vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(vec);
            for (bufs[0..n]) |buf| {
                const sh = mlx.getShape(buf.*);
                try growQuantBuf(s, buf, true, entry.offset, new_cap, sh[0], sh[1], sh[3], mlx.mlx_array_dtype(buf.*));
                _ = mlx.mlx_vector_array_append_value(vec, buf.*);
            }
            // The grown buffers are this cache's own; a shared restore's rows stay with the entry.
            entry.shared_view = false;
            kv_cap_buf_grows += 1;
            try mlx.check(mlx.mlx_eval(vec));
            _ = mlx.mlx_clear_cache();
        }
    }

    pub fn update(self: *KVCache, layer: u32, new_k: mlx.mlx_array, new_v: mlx.mlx_array, s: mlx.mlx_stream, max_seq: u32) !DenseKVView {
        switch (self.config.scheme) {
            .off => return self.updateDense(layer, new_k, new_v, s, max_seq),
            .affine => return self.updateAffine(layer, new_k, new_v, s, max_seq),
        }
    }

    pub fn appendBatched(
        stacked_k: mlx.mlx_array,
        stacked_v: mlx.mlx_array,
        caches: []const *KVCache,
        layer: u32,
        s: mlx.mlx_stream,
        max_seq: u32,
        scheme: KVQuantConfig,
    ) !void {
        if (scheme.scheme == .affine and (scheme.bits == 8 or scheme.bits == 6)) {
            const Once = struct {
                var logged: bool = false;
            };
            if (!Once.logged) {
                Once.logged = true;
                log.info("[batched] affine-{d} kv append engaged (slots={d}) (batched_kv_append_override=false restores per-slot appends)\n", .{ scheme.bits, caches.len });
            }
            try appendAffineBatched(stacked_k, stacked_v, caches, layer, s, max_seq, scheme);
            return;
        }
        try appendBatchedPerSlot(stacked_k, stacked_v, caches, layer, s, max_seq);
    }

    pub fn appendFromStacked(
        stacked_k: mlx.mlx_array,
        stacked_v: mlx.mlx_array,
        caches: []const *KVCache,
        layer: u32,
        s: mlx.mlx_stream,
        max_seq: u32,
        use_batched: bool,
    ) !void {
        if (use_batched and caches.len > 0 and sameConfig(caches)) {
            try appendBatched(stacked_k, stacked_v, caches, layer, s, max_seq, caches[0].config);
            return;
        }
        try appendBatchedPerSlot(stacked_k, stacked_v, caches, layer, s, max_seq);
    }

    fn sameConfig(caches: []const *KVCache) bool {
        for (caches[1..]) |c| if (!std.meta.eql(c.config, caches[0].config)) return false;
        return true;
    }

    fn appendBatchedPerSlot(
        stacked_k: mlx.mlx_array,
        stacked_v: mlx.mlx_array,
        caches: []const *KVCache,
        layer: u32,
        s: mlx.mlx_stream,
        max_seq: u32,
    ) !void {
        for (caches, 0..) |cache, i| {
            const k_slot = try sliceStackedRow(s, stacked_k, i);
            defer _ = mlx.mlx_array_free(k_slot);
            const v_slot = try sliceStackedRow(s, stacked_v, i);
            defer _ = mlx.mlx_array_free(v_slot);
            var view = try cache.update(layer, k_slot, v_slot, s, max_seq);
            view.deinit();
        }
    }

    fn sliceStackedRow(s: mlx.mlx_stream, stacked: mlx.mlx_array, i: usize) !mlx.mlx_array {
        const sh = mlx.getShape(stacked);
        const i_c: c_int = @intCast(i);
        const start = [_]c_int{ i_c, 0, 0, 0 };
        const stop = [_]c_int{ i_c + 1, sh[1], sh[2], sh[3] };
        const strides = [_]c_int{ 1, 1, 1, 1 };
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_slice(&out, stacked, &start, 4, &stop, 4, &strides, 4, s));
        return out;
    }

    fn appendAffineBatched(
        stacked_k: mlx.mlx_array,
        stacked_v: mlx.mlx_array,
        caches: []const *KVCache,
        layer: u32,
        s: mlx.mlx_stream,
        max_seq: u32,
        scheme: KVQuantConfig,
    ) !void {
        var new_kq = try kv_quant.quantizeAffine(s, stacked_k, scheme.group_size, scheme.bits);
        defer new_kq.deinit();
        var new_vq = try kv_quant.quantizeAffine(s, stacked_v, scheme.group_size, scheme.bits);
        defer new_vq.deinit();
        for (caches, 0..) |cache, i| {
            var row_kq: kv_quant.QuantizedKV = .{
                .q = mlx.mlx_array_new(),
                .scales = mlx.mlx_array_new(),
                .biases = mlx.mlx_array_new(),
            };
            defer row_kq.deinit();
            _ = mlx.mlx_array_free(row_kq.q);
            row_kq.q = try sliceStackedRow(s, new_kq.q, i);
            _ = mlx.mlx_array_free(row_kq.scales);
            row_kq.scales = try sliceStackedRow(s, new_kq.scales, i);
            _ = mlx.mlx_array_free(row_kq.biases);
            row_kq.biases = try sliceStackedRow(s, new_kq.biases, i);
            var row_vq: kv_quant.QuantizedKV = .{
                .q = mlx.mlx_array_new(),
                .scales = mlx.mlx_array_new(),
                .biases = mlx.mlx_array_new(),
            };
            defer row_vq.deinit();
            _ = mlx.mlx_array_free(row_vq.q);
            row_vq.q = try sliceStackedRow(s, new_vq.q, i);
            _ = mlx.mlx_array_free(row_vq.scales);
            row_vq.scales = try sliceStackedRow(s, new_vq.scales, i);
            _ = mlx.mlx_array_free(row_vq.biases);
            row_vq.biases = try sliceStackedRow(s, new_vq.biases, i);
            _ = try cache.writeAffineChunk(layer, row_kq, row_vq, s, max_seq);
        }
    }

    fn updateAffine(self: *KVCache, layer: u32, new_k: mlx.mlx_array, new_v: mlx.mlx_array, s: mlx.mlx_stream, max_seq: u32) !DenseKVView {
        const cfg = self.config;
        const group_size: u32 = cfg.group_size;
        const bits: u8 = cfg.bits;

        // 1. Free stale views (6 of them — dense + 4 quant scale/bias views).
        // 2. Quantize incoming K/V.
        var new_kq = try kv_quant.quantizeAffine(s, new_k, group_size, bits);
        defer new_kq.deinit();
        var new_vq = try kv_quant.quantizeAffine(s, new_v, group_size, bits);
        defer new_vq.deinit();

        // 3. Shape info: new_k is [B, heads, new_len, head_dim]. The V widths
        //    are solved from new_v's OWN head dim, not K's: an MLA arch scores
        //    over nope+rope (192) and stores values at v_head_dim (128), so a
        //    K-derived packed width writes a 16-u32 chunk into a 24-u32 window
        //    (an mlx slice_update mismatch we cannot catch). Identical on every
        //    symmetric arch — the same generalization `updateDense` carries.
        return self.appendQuantized(layer, new_kq, new_vq, s, max_seq);
    }

    /// `update` for rows the caller already quantized with this cache's scheme.
    pub fn appendQuantized(self: *KVCache, layer: u32, new_kq: kv_quant.QuantizedKV, new_vq: kv_quant.QuantizedKV, s: mlx.mlx_stream, max_seq: u32) !DenseKVView {
        const entry = &self.entries[layer];
        const group_size: u32 = self.config.group_size;
        const bits: u8 = self.config.bits;
        const compacted = try self.writeAffineChunk(layer, new_kq, new_vq, s, max_seq);
        var trip = [6]mlx.mlx_array{ entry.key_view, entry.key_scales_view, entry.key_biases_view, entry.value_view, entry.value_scales_view, entry.value_biases_view };
        if (compacted) trip = try handOffRingViews(entry, s, 6);
        errdefer if (compacted) for (trip) |a| {
            _ = mlx.mlx_array_free(a);
        };

        // 8. Dequantize K/V for SDPA. Owner of these dense arrays is the
        //    DenseKVView returned to the caller. The quant triples ride along
        //    as borrows so fused call sites can skip the dense read — mlx is
        //    lazy, so the dequant graph below costs nothing unless SDPA
        //    actually reads `.k`/`.v`. (Engagement bug class: the decode
        //    forward calls update(), not denseView() — omitting the triples
        //    here made the packed arms a silent no-op.)
        const dense_k = try kv_quant.dequantizeAffine(s, trip[0], trip[1], trip[2], group_size, bits);
        errdefer _ = mlx.mlx_array_free(dense_k);
        const dense_v = try kv_quant.dequantizeAffine(s, trip[3], trip[4], trip[5], group_size, bits);
        return .{
            .k = dense_k,
            .v = dense_v,
            .owned = true,
            .k_triple_q = trip[0],
            .k_triple_scales = trip[1],
            .k_triple_biases = trip[2],
            .v_triple_q = trip[3],
            .v_triple_scales = trip[4],
            .v_triple_biases = trip[5],
            .has_quant_triple = true,
            .bits = bits,
            .group_size = group_size,
            .owns_triple = compacted,
        };
    }

    /// A compaction leaves the entry's views slicing the pre-compaction buffer, which this
    /// forward still reads. Hand them to the forward (the caller frees them) and re-point the
    /// entry at its ring, so only the forward's graph keeps the old buffer alive.
    fn handOffRingViews(entry: *KVCacheEntry, s: mlx.mlx_stream, comptime n: usize) ![n]mlx.mlx_array {
        const views = [6]*mlx.mlx_array{ &entry.key_view, &entry.key_scales_view, &entry.key_biases_view, &entry.value_view, &entry.value_scales_view, &entry.value_biases_view };
        const bufs = [6]mlx.mlx_array{ entry.keys, entry.keys_scales, entry.keys_biases, entry.values, entry.values_scales, entry.values_biases };
        const pick = if (n == 2) [2]usize{ 0, 3 } else [6]usize{ 0, 1, 2, 3, 4, 5 };
        var out: [n]mlx.mlx_array = undefined;
        for (pick, 0..) |vi, i| {
            out[i] = views[vi].*;
            views[vi].* = mlx.mlx_array_new();
        }
        errdefer for (out) |a| {
            _ = mlx.mlx_array_free(a);
        };
        for (pick) |vi| try buildSliceView(s, views[vi], bufs[vi], @intCast(entry.offset), 0);
        return out;
    }

    fn writeAffineChunk(
        self: *KVCache,
        layer: u32,
        new_kq: kv_quant.QuantizedKV,
        new_vq: kv_quant.QuantizedKV,
        s: mlx.mlx_stream,
        max_seq: u32,
    ) !bool {
        const entry = &self.entries[layer];
        // A non-zero `max_seq` IS the ring predicate: it says this forward will
        // only ever read the tail of this layer.
        if (self.swa_ring_window > 0 and max_seq > 0) entry.ringed = true;
        _ = mlx.mlx_array_free(entry.key_view);
        _ = mlx.mlx_array_free(entry.value_view);
        _ = mlx.mlx_array_free(entry.key_scales_view);
        _ = mlx.mlx_array_free(entry.key_biases_view);
        _ = mlx.mlx_array_free(entry.value_scales_view);
        _ = mlx.mlx_array_free(entry.value_biases_view);
        entry.key_view = mlx.mlx_array_new();
        entry.value_view = mlx.mlx_array_new();
        entry.key_scales_view = mlx.mlx_array_new();
        entry.key_biases_view = mlx.mlx_array_new();
        entry.value_scales_view = mlx.mlx_array_new();
        entry.value_biases_view = mlx.mlx_array_new();

        const q_shape = mlx.getShape(new_kq.q);
        const vq_shape = mlx.getShape(new_vq.q);
        const sc_shape = mlx.getShape(new_kq.scales);
        const vsc_shape = mlx.getShape(new_vq.scales);
        const new_len: usize = @intCast(q_shape[2]);
        const B = q_shape[0];
        const heads = q_shape[1];
        const q_last = q_shape[3];
        const vq_last = vq_shape[3];
        const sc_last = sc_shape[3];
        const vsc_last = vsc_shape[3];

        const shared = self.copiesSharedRestore(entry, entry.offset + new_len);
        const will_grow = !entry.initialized or shared or entry.offset + new_len > bufferCapacity(entry.keys);
        if (entry.initialized and entry.shared_view and !will_grow) try cowAffineBuffers(s, entry);
        entry.shared_view = false;

        // 4. Grow buffers if needed (6 of them, in lockstep on the seq axis).
        if (will_grow) {
            const needed = entry.offset + new_len;
            const cur_cap = if (entry.initialized and !shared) bufferCapacity(entry.keys) else 0;
            const new_cap: c_int = @intCast(self.growCapacityFor(entry, cur_cap, needed));
            kv_cap_buf_grows += 1;
            try growQuantBuf(s, &entry.keys, entry.initialized, entry.offset, new_cap, B, heads, q_last, .uint32);
            try growQuantBuf(s, &entry.values, entry.initialized, entry.offset, new_cap, B, heads, vq_last, .uint32);
            // Quantizer scales retain the activation dtype through allocation and growth.
            const ksc_dt = mlx.mlx_array_dtype(new_kq.scales);
            const vsc_dt = mlx.mlx_array_dtype(new_vq.scales);
            try growQuantBuf(s, &entry.keys_scales, entry.initialized, entry.offset, new_cap, B, heads, sc_last, ksc_dt);
            try growQuantBuf(s, &entry.keys_biases, entry.initialized, entry.offset, new_cap, B, heads, sc_last, ksc_dt);
            try growQuantBuf(s, &entry.values_scales, entry.initialized, entry.offset, new_cap, B, heads, vsc_last, vsc_dt);
            try growQuantBuf(s, &entry.values_biases, entry.initialized, entry.offset, new_cap, B, heads, vsc_last, vsc_dt);
            entry.initialized = true;
        }

        // 5. slice_update each buffer at offset.
        try writeAtOffset(s, &entry.keys, entry.offset, new_kq.q);
        try writeAtOffset(s, &entry.values, entry.offset, new_vq.q);
        try writeAtOffset(s, &entry.keys_scales, entry.offset, new_kq.scales);
        try writeAtOffset(s, &entry.keys_biases, entry.offset, new_kq.biases);
        try writeAtOffset(s, &entry.values_scales, entry.offset, new_vq.scales);
        try writeAtOffset(s, &entry.values_biases, entry.offset, new_vq.biases);

        // 6. Update offset / step.
        entry.offset += new_len;
        if (layer == 0) self.step += new_len;

        // 7. Build views for all 6 buffers.
        const total: c_int = @intCast(entry.offset);
        // `max_seq` is the TAIL SPAN this forward needs, not the raw window:
        // the caller sizes it with `slidingTailSpan` so a block-wide verify
        // gets `window + q_len - 1` and decode still gets `window`. 0 = keep
        // the whole cache. Honored at EVERY width — the old `new_len == 1`
        // gate is what silently made every spec verify read the full cache on
        // sliding layers.
        const view_start: c_int = if (max_seq > 0 and entry.offset > max_seq)
            total - @as(c_int, @intCast(max_seq))
        else
            0;
        try buildSliceView(s, &entry.key_view, entry.keys, total, view_start);
        try buildSliceView(s, &entry.value_view, entry.values, total, view_start);
        try buildSliceView(s, &entry.key_scales_view, entry.keys_scales, total, view_start);
        try buildSliceView(s, &entry.key_biases_view, entry.keys_biases, total, view_start);
        try buildSliceView(s, &entry.value_scales_view, entry.values_scales, total, view_start);
        try buildSliceView(s, &entry.value_biases_view, entry.values_biases, total, view_start);
        return self.ringCompact(layer, s);
    }

    /// Does a shared restore's first append copy its rows into a buffer sized for THIS request? Yes
    /// when the donor's buffer is longer than the request reserves (or its rows, unreserved): the
    /// admission bills the copy at that, never at the donor's capacity. A ringed layer keeps its
    /// ring; a donated checkout owns its buffers (`adoptRestored`) and appends in place.
    fn copiesSharedRestore(self: *const KVCache, entry: *const KVCacheEntry, needed: usize) bool {
        if (!entry.initialized or !entry.shared_view or entry.ringed) return false;
        return bufferCapacity(entry.keys) > @max(self.reserve_tokens, needed);
    }

    /// The restored buffers are this cache's alone now (a donated checkout): append in place.
    pub fn adoptRestored(self: *KVCache) void {
        for (self.entries) |*e| e.shared_view = false;
    }

    fn cowAffineBuffers(s: mlx.mlx_stream, entry: *KVCacheEntry) !void {
        const bufs = [_]*mlx.mlx_array{
            &entry.keys,
            &entry.values,
            &entry.keys_scales,
            &entry.keys_biases,
            &entry.values_scales,
            &entry.values_biases,
        };
        for (bufs) |buf| {
            const owned = try materializedOwnedCopy(s, buf.*);
            _ = mlx.mlx_array_free(buf.*);
            buf.* = owned;
        }
    }

    fn updateDense(self: *KVCache, layer: u32, new_k: mlx.mlx_array, new_v: mlx.mlx_array, s: mlx.mlx_stream, max_seq: u32) !DenseKVView {
        const entry = &self.entries[layer];
        // A non-zero `max_seq` IS the ring predicate: it says this forward will
        // only ever read the tail of this layer.
        if (self.swa_ring_window > 0 and max_seq > 0) entry.ringed = true;

        // 1. Free stale views — drops refcount on buffer → enables buffer donation.
        //    Reset the handles at once: a failing op below must not leave freed
        //    handles for `freeKVEntry` to free again.
        _ = mlx.mlx_array_free(entry.key_view);
        _ = mlx.mlx_array_free(entry.value_view);
        entry.key_view = mlx.mlx_array_new();
        entry.value_view = mlx.mlx_array_new();

        // 2. Get shape info from new_k: [B, heads, new_len, head_dim]. The V
        //    head dim is read off new_v, NOT new_k: an MLA arch scores over
        //    nope+rope (192) but stores values at v_head_dim (128), and MLX's
        //    SDPA serves exactly that pair with a fused vector kernel. Every
        //    buffer below is sized from its own operand; the two are identical
        //    on every symmetric arch, so this is a strict generalization.
        const new_shape = mlx.getShape(new_k);
        const new_len: usize = @intCast(new_shape[2]);
        const v_head_dim = mlx.getShape(new_v)[3];

        const shared = self.copiesSharedRestore(entry, entry.offset + new_len);
        const will_grow = !entry.initialized or shared or entry.offset + new_len > bufferCapacity(entry.keys);
        if (entry.initialized and entry.shared_view and !will_grow) {
            const k_owned = try materializedOwnedCopy(s, entry.keys);
            _ = mlx.mlx_array_free(entry.keys);
            entry.keys = k_owned;
            const v_owned = try materializedOwnedCopy(s, entry.values);
            _ = mlx.mlx_array_free(entry.values);
            entry.values = v_owned;
        }
        entry.shared_view = false;

        // 3. Grow buffer if needed
        if (will_grow) {
            const B = new_shape[0];
            const heads = new_shape[1];
            const head_dim = new_shape[3];
            const dtype = mlx.mlx_array_dtype(new_k);
            const needed = entry.offset + new_len;
            const cur_cap = if (entry.initialized and !shared) bufferCapacity(entry.keys) else 0;
            const new_cap: c_int = @intCast(self.growCapacityFor(entry, cur_cap, needed));
            kv_cap_buf_grows += 1;
            const initialized = entry.initialized;
            try growQuantBuf(s, &entry.keys, initialized, entry.offset, new_cap, B, heads, head_dim, dtype);
            try growQuantBuf(s, &entry.values, initialized, entry.offset, new_cap, B, heads, v_head_dim, dtype);
            entry.initialized = true;
        }

        // 4. slice_update — write new_k/new_v into buffer at offset
        try writeAtOffset(s, &entry.keys, entry.offset, new_k);
        try writeAtOffset(s, &entry.values, entry.offset, new_v);

        // 5. Update offset and absolute step
        entry.offset += new_len;
        if (layer == 0) self.step += new_len;

        // 6. Create views for attention.
        //    Skip slicing when the view covers the entire buffer — just reference it directly.
        //    This saves 2 C API calls per layer per token (84 calls/token for 42-layer models).
        const total: c_int = @intCast(entry.offset);
        // `max_seq` is the TAIL SPAN this forward needs, not the raw window:
        // the caller sizes it with `slidingTailSpan` so a block-wide verify
        // gets `window + q_len - 1` and decode still gets `window`. 0 = keep
        // the whole cache. Honored at EVERY width — the old `new_len == 1`
        // gate is what silently made every spec verify read the full cache on
        // sliding layers.
        const view_start: c_int = if (max_seq > 0 and entry.offset > max_seq)
            total - @as(c_int, @intCast(max_seq))
        else
            0;

        // buildSliceView takes the whole-buffer shortcut itself (mlx-lm's
        // optimization: no slice when the view covers the buffer) and reads
        // each buffer's own last dim, so K and V may differ in width.
        try buildSliceView(s, &entry.key_view, entry.keys, total, view_start);
        try buildSliceView(s, &entry.value_view, entry.values, total, view_start);
        if (try self.ringCompact(layer, s)) {
            const kv = try handOffRingViews(entry, s, 2);
            return .{ .k = kv[0], .v = kv[1], .owned = true };
        }

        return .{ .k = entry.key_view, .v = entry.value_view, .owned = false };
    }

    /// Read-side accessor: return a dense `[B,H,T,D]` K/V pair for the layer.
    /// In dense mode this aliases `key_view`/`value_view` (no-op deinit).
    /// In quant mode this dequantizes on the fly from the cache's stored
    /// triples (the returned arrays are owned and freed by `deinit`).
    /// SDPA call sites use this so they don't have to know the scheme.
    pub fn denseView(self: *KVCache, layer: u32, s: mlx.mlx_stream) !DenseKVView {
        const entry = &self.entries[layer];
        switch (self.config.scheme) {
            .off => return .{ .k = entry.key_view, .v = entry.value_view, .owned = false },
            .affine => {
                if (!entry.initialized) {
                    return .{ .k = entry.key_view, .v = entry.value_view, .owned = false };
                }
                const dense_k = try kv_quant.dequantizeAffine(s, entry.key_view, entry.key_scales_view, entry.key_biases_view, self.config.group_size, self.config.bits);
                errdefer _ = mlx.mlx_array_free(dense_k);
                const dense_v = try kv_quant.dequantizeAffine(s, entry.value_view, entry.value_scales_view, entry.value_biases_view, self.config.group_size, self.config.bits);
                return .{
                    .k = dense_k,
                    .v = dense_v,
                    .owned = true,
                    // Borrow the cache's quant triples so fused-attn call
                    // sites can skip the dense materialization above (the
                    // dequant arrays still get computed — mlx is lazy, so
                    // the cost is only paid if SDPA actually reads them).
                    .k_triple_q = entry.key_view,
                    .k_triple_scales = entry.key_scales_view,
                    .k_triple_biases = entry.key_biases_view,
                    .v_triple_q = entry.value_view,
                    .v_triple_scales = entry.value_scales_view,
                    .v_triple_biases = entry.value_biases_view,
                    .has_quant_triple = true,
                    .bits = self.config.bits,
                    .group_size = self.config.group_size,
                };
            },
        }
    }

    pub fn bufferCapacity(arr: mlx.mlx_array) usize {
        const shape = mlx.getShape(arr);
        if (shape.len < 3) return 0;
        return @intCast(shape[2]);
    }

    /// Pure half of `residentCapacityTokens`: the minimum capacity across the entries that
    /// hold one, 0 when none does. An uninitialized entry is skipped, never a veto: on a
    /// hybrid trunk only the attention layers ever call `update` (12 of 48 on qwen4_exp).
    pub const CapacityFold = struct {
        cap: usize = std.math.maxInt(usize),
        seen: usize = 0,

        /// `null` = this layer holds no buffer (a linear layer, or a cold cache).
        pub fn add(self: *CapacityFold, entry_cap: ?usize) void {
            const v = entry_cap orelse return;
            self.seen += 1;
            self.cap = @min(self.cap, v);
        }

        pub fn result(self: CapacityFold) usize {
            return if (self.seen == 0) 0 else self.cap;
        }
    };

    /// Tokens of capacity every caching layer's buffer already holds: the warm bill's one
    /// gate. Restored rows are free only while the buffer is reused in place; past this
    /// capacity a grow allocates a whole new buffer beside the old one. The MINIMUM across
    /// layers, since the credit must be provable for every buffer the prefill writes.
    pub fn residentCapacityTokens(self: *const KVCache) usize {
        var fold = CapacityFold{};
        for (self.entries) |*e| fold.add(if (e.initialized and !e.ringed) bufferCapacity(e.keys) else null);
        return fold.result();
    }

    /// Buffer-grow / slice-update / view-build, parameterized over the
    /// buffer's last dim and dtype: six calls per `updateAffine` (3 buffers ×
    /// K and V) and two per `updateDense`. Every caller passes the width of
    /// the operand it is writing, which is what lets K and V differ.
    fn growQuantBuf(s: mlx.mlx_stream, buf: *mlx.mlx_array, initialized: bool, offset: usize, new_cap: c_int, B: c_int, heads: c_int, last_dim: c_int, dtype: mlx.mlx_dtype) !void {
        const buf_shape = [_]c_int{ B, heads, new_cap, last_dim };
        if (initialized and offset > 0) {
            var new_buf = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(new_buf);
            try mlx.check(mlx.mlx_zeros(&new_buf, &buf_shape, 4, dtype, s));
            const off_c: c_int = @intCast(offset);
            const su_start = [_]c_int{ 0, 0, 0, 0 };
            const su_stop = [_]c_int{ B, heads, off_c, last_dim };
            const su_strides = [_]c_int{ 1, 1, 1, 1 };
            var old_data = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(old_data);
            try mlx.check(mlx.mlx_slice(&old_data, buf.*, &su_start, 4, &su_stop, 4, &su_strides, 4, s));
            var updated = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(updated);
            try mlx.check(mlx.mlx_slice_update(&updated, new_buf, old_data, &su_start, 4, &su_stop, 4, &su_strides, 4, s));
            _ = mlx.mlx_array_free(old_data);
            old_data = .{ .ctx = null };
            _ = mlx.mlx_array_free(new_buf);
            new_buf = .{ .ctx = null };
            _ = mlx.mlx_array_free(buf.*);
            buf.* = updated;
            updated = .{ .ctx = null };
        } else {
            var new_buf = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(new_buf);
            try mlx.check(mlx.mlx_zeros(&new_buf, &buf_shape, 4, dtype, s));
            _ = mlx.mlx_array_free(buf.*);
            buf.* = new_buf;
            new_buf = .{ .ctx = null };
        }
    }

    /// Replace `buf` with a fresh `cap`-row buffer holding rows `[drop, offset)`
    /// at its front. A real copy, not a slice: the point is to release the
    /// buffer the dropped rows live in.
    fn ringTailBuf(s: mlx.mlx_stream, buf: *mlx.mlx_array, drop: usize, offset: usize, cap: usize) !void {
        const sh = mlx.getShape(buf.*);
        if (sh.len < 4) return;
        const keep: c_int = @intCast(offset - drop);
        const buf_shape = [_]c_int{ sh[0], sh[1], @intCast(cap), sh[3] };
        var fresh = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(fresh);
        try mlx.check(mlx.mlx_zeros(&fresh, &buf_shape, 4, mlx.mlx_array_dtype(buf.*), s));
        const strides = [_]c_int{ 1, 1, 1, 1 };
        const tail_start = [_]c_int{ 0, 0, @intCast(drop), 0 };
        const tail_stop = [_]c_int{ sh[0], sh[1], @intCast(offset), sh[3] };
        var tail = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(tail);
        try mlx.check(mlx.mlx_slice(&tail, buf.*, &tail_start, 4, &tail_stop, 4, &strides, 4, s));
        const head_start = [_]c_int{ 0, 0, 0, 0 };
        const head_stop = [_]c_int{ sh[0], sh[1], keep, sh[3] };
        var updated = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(updated);
        try mlx.check(mlx.mlx_slice_update(&updated, fresh, tail, &head_start, 4, &head_stop, 4, &strides, 4, s));
        _ = mlx.mlx_array_free(fresh);
        fresh = .{ .ctx = null };
        _ = mlx.mlx_array_free(buf.*);
        buf.* = updated;
    }

    /// Drop the rows below the retained window once a ringed entry passes its
    /// cap. Runs AFTER this forward's view is built: the view slices the
    /// pre-compaction buffer and holds it alive, so the rows it reads are never
    /// the rows this drops.
    fn ringCompact(self: *KVCache, layer: u32, s: mlx.mlx_stream) !bool {
        const entry = &self.entries[layer];
        if (self.swa_ring_window == 0 or !entry.ringed) return false;
        if (self.ring_marks.len > 0) self.captureRingMarks(layer, s);
        if (entry.offset <= ringCap(self.swa_ring_window)) return false;
        const keep = ringKeep(self.swa_ring_window);
        const drop = entry.offset - keep;
        const cap = ringCap(self.swa_ring_window);
        try ringTailBuf(s, &entry.keys, drop, entry.offset, cap);
        try ringTailBuf(s, &entry.values, drop, entry.offset, cap);
        if (self.config.scheme != .off) {
            try ringTailBuf(s, &entry.keys_scales, drop, entry.offset, cap);
            try ringTailBuf(s, &entry.keys_biases, drop, entry.offset, cap);
            try ringTailBuf(s, &entry.values_scales, drop, entry.offset, cap);
            try ringTailBuf(s, &entry.values_biases, drop, entry.offset, cap);
        }
        entry.base += drop;
        entry.offset = keep;
        // Nothing in this forward's output reads the ring, so left lazy it would hold the
        // pre-compaction buffer until the next forward. Evaluated now, it lets go of it.
        const quant = self.config.scheme != .off;
        const bufs = [6]mlx.mlx_array{ entry.keys, entry.values, entry.keys_scales, entry.keys_biases, entry.values_scales, entry.values_biases };
        const vec = mlx.mlx_vector_array_new_data(&bufs, if (quant) 6 else 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_async_eval(vec));
        return true;
    }

    /// The capacity a grow takes for THIS entry. A ringed entry never takes the
    /// session reservation (it holds a window, not a context) and floors at the
    /// ring cap so a cold decode does not grow twice to reach it.
    fn growCapacityFor(self: *const KVCache, entry: *const KVCacheEntry, cur_cap: usize, needed: usize) usize {
        if (!entry.ringed) return self.nextCapacityReserved(cur_cap, needed);
        return @max(nextCapacity(cur_cap, needed), ringCap(self.swa_ring_window));
    }

    fn writeAtOffset(s: mlx.mlx_stream, buf: *mlx.mlx_array, offset: usize, new_chunk: mlx.mlx_array) !void {
        const new_shape = mlx.getShape(new_chunk);
        const new_len: c_int = new_shape[2];
        const buf_shape = mlx.getShape(buf.*);
        const off: c_int = @intCast(offset);
        const off_end: c_int = off + new_len;
        const su_start = [_]c_int{ 0, 0, off, 0 };
        const su_stop = [_]c_int{ buf_shape[0], buf_shape[1], off_end, buf_shape[3] };
        const su_strides = [_]c_int{ 1, 1, 1, 1 };
        var updated = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(updated);
        try mlx.check(mlx.mlx_slice_update(&updated, buf.*, new_chunk, &su_start, 4, &su_stop, 4, &su_strides, 4, s));
        _ = mlx.mlx_array_free(buf.*);
        buf.* = updated;
    }

    fn buildSliceView(s: mlx.mlx_stream, view: *mlx.mlx_array, buf: mlx.mlx_array, total: c_int, view_start: c_int) !void {
        const buf_cap = bufferCapacity(buf);
        if (view_start == 0 and @as(usize, @intCast(total)) == buf_cap) {
            try mlx.check(mlx.mlx_array_set(view, buf));
        } else {
            const cur_shape = mlx.getShape(buf);
            const v_start = [_]c_int{ 0, 0, view_start, 0 };
            const v_stop = [_]c_int{ cur_shape[0], cur_shape[1], total, cur_shape[3] };
            const v_strides = [_]c_int{ 1, 1, 1, 1 };
            try mlx.check(mlx.mlx_slice(view, buf, &v_start, 4, &v_stop, 4, &v_strides, 4, s));
        }
    }

    pub fn seqLen(self: *const KVCache, layer: u32) usize {
        const entry = &self.entries[layer];
        if (!entry.initialized) return 0;
        return entry.offset;
    }

    pub fn appendEvalArrays(self: *const KVCache, vec: mlx.mlx_vector_array) void {
        const affine = self.config.scheme != .off;
        for (self.entries) |*entry| {
            if (!entry.initialized) continue;
            _ = mlx.mlx_vector_array_append_value(vec, entry.keys);
            _ = mlx.mlx_vector_array_append_value(vec, entry.values);
            if (affine) {
                _ = mlx.mlx_vector_array_append_value(vec, entry.keys_scales);
                _ = mlx.mlx_vector_array_append_value(vec, entry.keys_biases);
                _ = mlx.mlx_vector_array_append_value(vec, entry.values_scales);
                _ = mlx.mlx_vector_array_append_value(vec, entry.values_biases);
            }
        }
    }

    /// Evaluate all KV cache entries to materialize them on GPU.
    /// Called after prefill to ensure the cache is in optimal memory layout for decode.
    pub fn evalState(self: *KVCache) void {
        // Collect all initialized entries into a vector and batch-eval them.
        // This matches mlx_lm's `mx.eval([c.state for c in cache])` pattern.
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        self.appendEvalArrays(vec);
        if (mlx.mlx_vector_array_size(vec) > 0) {
            _ = mlx.mlx_eval(vec);
        }
    }

    /// The sequence length of the attention state this cache holds. `step` is maintained by
    /// `update` on layer 0 only: the absolute position on an attention-first trunk, a
    /// permanent zero on a linear-layer-0 trunk (GDN, Mamba2, gated-conv, KDA), where the
    /// first initialized entry's `offset` is the KV length. Attention-first trunks keep `step`.
    pub fn kvLenForBatching(self: *const KVCache) usize {
        if (self.entries.len > 0 and self.entries[0].initialized) return self.step;
        for (self.entries) |*e| {
            if (e.initialized) return e.base + e.offset;
        }
        return self.step;
    }

    /// Can every ringed entry still show the window a query at `len` reads?
    /// `base` rows are gone for good, so the answer is a property of what the
    /// ring kept, not of the request.
    fn ringServesClamp(self: *const KVCache, len: usize) !void {
        for (self.entries) |*entry| {
            if (!ringEntryServes(entry, self.swa_ring_window, len)) return error.SlidingRingRewindPastWindow;
        }
    }

    /// Truncate the KV cache to keep only the first `len` tokens on the sequence axis.
    pub fn truncate(self: *KVCache, len: usize, s: mlx.mlx_stream) !void {
        // A ringed layer holds a window, not a prefix: a clamp below the rows it
        // kept cannot be served. Decide BEFORE mutating anything — a half-clamped
        // cache is a state no caller can hand back.
        if (len > 0) try self.ringServesClamp(len);
        self.step = len;
        for (self.entries) |*entry| {
            if (!entry.initialized) continue;
            const local: usize = len -| entry.base;
            if (local >= entry.offset) continue;

            // Free stale views (all 6: dense + 4 quant scale/bias views)
            _ = mlx.mlx_array_free(entry.key_view);
            _ = mlx.mlx_array_free(entry.value_view);
            entry.key_view = mlx.mlx_array_new();
            entry.value_view = mlx.mlx_array_new();
            if (self.config.scheme != .off) {
                _ = mlx.mlx_array_free(entry.key_scales_view);
                _ = mlx.mlx_array_free(entry.key_biases_view);
                _ = mlx.mlx_array_free(entry.value_scales_view);
                _ = mlx.mlx_array_free(entry.value_biases_view);
                entry.key_scales_view = mlx.mlx_array_new();
                entry.key_biases_view = mlx.mlx_array_new();
                entry.value_scales_view = mlx.mlx_array_new();
                entry.value_biases_view = mlx.mlx_array_new();
            }

            if (local == 0) {
                _ = mlx.mlx_array_free(entry.keys);
                _ = mlx.mlx_array_free(entry.values);
                entry.keys = mlx.mlx_array_new();
                entry.values = mlx.mlx_array_new();
                if (self.config.scheme != .off) {
                    _ = mlx.mlx_array_free(entry.keys_scales);
                    _ = mlx.mlx_array_free(entry.keys_biases);
                    _ = mlx.mlx_array_free(entry.values_scales);
                    _ = mlx.mlx_array_free(entry.values_biases);
                    entry.keys_scales = mlx.mlx_array_new();
                    entry.keys_biases = mlx.mlx_array_new();
                    entry.values_scales = mlx.mlx_array_new();
                    entry.values_biases = mlx.mlx_array_new();
                }
                entry.initialized = false;
                entry.offset = 0;
                entry.base = 0;
                continue;
            }

            // Just update offset — the buffer still holds data but views will
            // only expose [0:local]. No need to shrink the pre-allocated buffer.
            entry.offset = local;

            // Recreate views for the truncated range. Each buffer is sliced
            // against its OWN shape — an MLA cache's V is narrower than its K.
            const shape = mlx.getShape(entry.keys);
            if (shape.len < 4) continue;
            const seq_end: c_int = @intCast(local);
            const v_start = [_]c_int{ 0, 0, 0, 0 };
            const v_stop = [_]c_int{ shape[0], shape[1], seq_end, shape[3] };
            const v_strides = [_]c_int{ 1, 1, 1, 1 };
            try mlx.check(mlx.mlx_slice(&entry.key_view, entry.keys, &v_start, 4, &v_stop, 4, &v_strides, 4, s));
            const val_shape = mlx.getShape(entry.values);
            const val_stop = [_]c_int{ val_shape[0], val_shape[1], seq_end, val_shape[3] };
            try mlx.check(mlx.mlx_slice(&entry.value_view, entry.values, &v_start, 4, &val_stop, 4, &v_strides, 4, s));
            if (self.config.scheme != .off) {
                // K and V scale/bias groups are as wide as their OWN payload —
                // an MLA cache's V is narrower than its K, so the value views
                // cannot be sliced against the key scales' shape.
                const sc_shape = mlx.getShape(entry.keys_scales);
                const sv_stop = [_]c_int{ sc_shape[0], sc_shape[1], seq_end, sc_shape[3] };
                try mlx.check(mlx.mlx_slice(&entry.key_scales_view, entry.keys_scales, &v_start, 4, &sv_stop, 4, &v_strides, 4, s));
                try mlx.check(mlx.mlx_slice(&entry.key_biases_view, entry.keys_biases, &v_start, 4, &sv_stop, 4, &v_strides, 4, s));
                const vsc_shape = mlx.getShape(entry.values_scales);
                const vsv_stop = [_]c_int{ vsc_shape[0], vsc_shape[1], seq_end, vsc_shape[3] };
                try mlx.check(mlx.mlx_slice(&entry.value_scales_view, entry.values_scales, &v_start, 4, &vsv_stop, 4, &v_strides, 4, s));
                try mlx.check(mlx.mlx_slice(&entry.value_biases_view, entry.values_biases, &v_start, 4, &vsv_stop, 4, &v_strides, 4, s));
            }
        }
    }
};

/// Snapshot of a `KVCache` at a point in time. Owns its array handles (which
/// share buffers with the source via refcount) and frees them in `deinit`.
/// Created by `KVCache.snapshot()` and consumed by `KVCache.restore()`.
pub const KVCacheSnapshot = struct {
    entries: []KVCacheEntry,
    step: usize,
    allocator: std.mem.Allocator,
    config: KVQuantConfig,
    /// The ring window the source cache held, so a trim can ask whether the
    /// rows it would keep still cover a query at the trimmed length.
    swa_ring_window: u32 = 0,

    pub fn deinit(self: *KVCacheSnapshot) void {
        for (self.entries) |*e| {
            freeKVEntry(e);
        }
        self.allocator.free(self.entries);
    }

    /// A second snapshot holding the same buffers by refcount (no data copy).
    pub fn share(self: *const KVCacheSnapshot) !KVCacheSnapshot {
        const view: KVCache = .{ .entries = self.entries, .step = self.step, .allocator = self.allocator, .config = self.config, .swa_ring_window = self.swa_ring_window };
        return view.snapshot();
    }

    /// Could a cache restored from this snapshot clamp to `len` (`KVCache.truncate`)?
    pub fn ringServes(self: *const KVCacheSnapshot, len: usize) bool {
        for (self.entries) |*e| {
            if (!ringEntryServes(e, self.swa_ring_window, len)) return false;
        }
        return true;
    }

    /// Give up every array handle this snapshot holds, leaving it empty but deinit-able. The
    /// second half of restore by move: `KVCache.restore` binds through `mlx_array_set`, so the
    /// snapshot's second reference made `is_donatable()` fail and the first `writeAtOffset`
    /// privatised the whole prefix (5.13 GB / ~110 ms at 393k, 45% of the warm TTFT). Called
    /// only from `donateCheckout`, right before the slot's first write, so a request refused
    /// before then still hands back an intact entry.
    pub fn releaseHandles(self: *KVCacheSnapshot) void {
        for (self.entries) |*e| {
            freeKVEntry(e);
            e.* = newEmptyKVEntry();
        }
        self.step = 0;
    }

    /// Issue #330: a materialized copy of the first `len` tokens (sequence
    /// axis 2). `KVCache.truncate` is offset-only — the views shrink but a
    /// refcount-shared snapshot keeps the full buffer alive and
    /// `snapshotBytes` keeps billing its capacity — so a byte-budget trim
    /// must be a real slice + copy into fresh buffers. Each array is sliced
    /// against its OWN shape (affine scales/biases are [B,H,T,D/gs]; the
    /// packed codes [B,H,T,D*bits/32] — all carry T at axis 2).
    /// `ring_cp` (`KVCache.ringCheckpoint`) supplies the ringed layers' rows below the ring.
    pub fn trimmedCopy(self: *const KVCacheSnapshot, len: usize, ring_cp: ?*const KVCacheSnapshot, s: mlx.mlx_stream) !KVCacheSnapshot {
        if (ring_cp) |cp| {
            std.debug.assert(cp.entries.len == self.entries.len);
            if (cp.step < len) return error.SlidingRingRewindPastWindow;
        }
        const out = try self.allocator.alloc(KVCacheEntry, self.entries.len);
        var built: usize = 0;
        errdefer {
            for (out[0..built]) |*e| freeKVEntry(e);
            self.allocator.free(out);
        }
        for (self.entries, 0..) |own, i| {
            const src = if (ring_cp != null and own.ringed and ring_cp.?.entries[i].initialized) ring_cp.?.entries[i] else own;
            out[i] = newEmptyKVEntry();
            built = i + 1;
            out[i].initialized = src.initialized;
            out[i].base = src.base;
            out[i].ringed = src.ringed;
            out[i].offset = @min(src.offset, len -| src.base);
            // A ringed entry keeps a window, so a trim below what it retained
            // has nothing to hand the restore. Declining is the contract: the
            // caller retries at a shorter candidate or keeps the resident one.
            if (src.ringed and src.initialized and
                out[i].offset < @min(len, @as(usize, self.swa_ring_window)))
                return error.SlidingRingRewindPastWindow;
            if (src.initialized) {
                const keep = out[i].offset;
                out[i].keys = try trimRowsOwned(src.keys, keep, s);
                out[i].values = try trimRowsOwned(src.values, keep, s);
                if (self.config.scheme != .off) {
                    out[i].keys_scales = try trimRowsOwned(src.keys_scales, keep, s);
                    out[i].keys_biases = try trimRowsOwned(src.keys_biases, keep, s);
                    out[i].values_scales = try trimRowsOwned(src.values_scales, keep, s);
                    out[i].values_biases = try trimRowsOwned(src.values_biases, keep, s);
                }
            }
        }
        // One batched eval, like `captureSsmCheckpoint`: without it the lazy
        // copy nodes pin the parent buffers — the capacity the trim exists
        // to release.
        {
            const vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(vec);
            var count: usize = 0;
            for (out) |*e| {
                inline for (.{ e.keys, e.values, e.keys_scales, e.keys_biases, e.values_scales, e.values_biases }) |arr| {
                    if (arr.ctx != null) {
                        _ = mlx.mlx_vector_array_append_value(vec, arr);
                        count += 1;
                    }
                }
            }
            if (count > 0) _ = mlx.mlx_eval(vec);
        }
        return .{ .entries = out, .step = @min(self.step, len), .allocator = self.allocator, .config = self.config, .swa_ring_window = self.swa_ring_window };
    }
};

/// Slice `[0:rows]` on axis 2 and force a real copy (see
/// `materializedOwnedCopy` for why a plain slice keeps the parent alive).
fn trimRowsOwned(x: mlx.mlx_array, rows: usize, s: mlx.mlx_stream) !mlx.mlx_array {
    return rowsOwned(x, 0, rows, s);
}

/// An owned copy of sequence rows `[from, to)` (axis 2).
fn rowsOwned(x: mlx.mlx_array, from: usize, to: usize, s: mlx.mlx_stream) !mlx.mlx_array {
    const sh = mlx.mlx_array_shape(x);
    var sliced = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sliced);
    const start = [_]c_int{ 0, 0, @intCast(from), 0 };
    const stop = [_]c_int{ sh[0], sh[1], @intCast(to), sh[3] };
    const strides = [_]c_int{ 1, 1, 1, 1 };
    try mlx.check(mlx.mlx_slice(&sliced, x, &start, 4, &stop, 4, &strides, 4, s));
    return materializedOwnedCopy(s, sliced);
}

/// Helper: create a dummy K or V tensor of shape [1, 1, seq_len, 1].
fn testKV(seq_len: usize, s: mlx.mlx_stream) mlx.mlx_array {
    const sl: c_int = @intCast(seq_len);
    const shape = [_]c_int{ 1, 1, sl, 1 };
    var arr = mlx.mlx_array_new();
    _ = mlx.mlx_zeros(&arr, &shape, 4, .float32, s);
    return arr;
}

/// A donor that reserved `reserve` rows and wrote `rows`, and a cache restored from its snapshot.
fn sharedRestoreFixture(s: mlx.mlx_stream, cfg: KVQuantConfig, reserve: usize, rows: c_int, donor: *KVCache, snap: *KVCacheSnapshot, warm: *KVCache) !mlx.mlx_array {
    donor.* = try KVCache.initWithConfig(testing.allocator, 1, cfg);
    donor.reserve(reserve);
    const k = try qkvIdentityArray(s, 2, rows, 64, 0.5, .bfloat16);
    var dv = try donor.update(0, k, k, s, 0);
    dv.deinit();
    snap.* = try donor.snapshot();
    warm.* = try KVCache.initWithConfig(testing.allocator, 1, cfg);
    try warm.restore(snap);
    return k;
}

/// Give a restored cache its own copy of every buffer, as a donated checkout owns them.
fn cowAffineOrDense(cache: *KVCache, s: mlx.mlx_stream) !void {
    for (cache.entries) |*e| {
        if (!e.initialized) continue;
        const bufs = [_]*mlx.mlx_array{ &e.keys, &e.values, &e.keys_scales, &e.keys_biases, &e.values_scales, &e.values_biases };
        const n: usize = if (cache.config.scheme == .off) 2 else 6;
        for (bufs[0..n]) |buf| {
            const owned = try materializedOwnedCopy(s, buf.*);
            _ = mlx.mlx_array_free(buf.*);
            buf.* = owned;
        }
        e.shared_view = false;
    }
}

fn batchedAppendMatchesSolo(bits: u8) !void {
    try mlx.requireGpu();
    const alloc = testing.allocator;
    const s = mlx.gpuStream();
    const cfg = kv_quant.KVQuantConfig.affine(bits);
    const ns = [_]usize{ 2, 4, 8 };
    const lens = [_]usize{ 2, 5, 3, 8, 4, 7, 6, 9 };
    for (ns) |n| {
        const solo = try alloc.alloc(KVCache, n);
        const batched = try alloc.alloc(KVCache, n);
        defer {
            for (solo) |*c| c.deinit();
            alloc.free(solo);
            for (batched) |*c| c.deinit();
            alloc.free(batched);
        }
        const solo_ptrs = try alloc.alloc(*KVCache, n);
        defer alloc.free(solo_ptrs);
        const batched_ptrs = try alloc.alloc(*KVCache, n);
        defer alloc.free(batched_ptrs);
        for (0..n) |i| {
            solo[i] = try KVCache.initWithConfig(alloc, 1, cfg);
            batched[i] = try KVCache.initWithConfig(alloc, 1, cfg);
            try kvAppendPrefill(&solo[i], lens[i], @intCast(0xA000 + i), s);
            try kvAppendPrefill(&batched[i], lens[i], @intCast(0xA000 + i), s);
            solo_ptrs[i] = &solo[i];
            batched_ptrs[i] = &batched[i];
        }
        const stacked_k = try kvAppendPacked(@intCast(n), 1, 0xC0FFE, s);
        defer _ = mlx.mlx_array_free(stacked_k);
        const stacked_v = try kvAppendPacked(@intCast(n), 1, 0x0D15EA, s);
        defer _ = mlx.mlx_array_free(stacked_v);
        try KVCache.appendFromStacked(stacked_k, stacked_v, solo_ptrs, 0, s, 0, false);
        try KVCache.appendFromStacked(stacked_k, stacked_v, batched_ptrs, 0, s, 0, true);
        for (0..n) |i| {
            try kvAppendOperandsEqual(&solo[i].entries[0], &batched[i].entries[0], s);
            const attn_s = try kvAppendDummyAttn(&solo[i], s);
            defer _ = mlx.mlx_array_free(attn_s);
            const attn_b = try kvAppendDummyAttn(&batched[i], s);
            defer _ = mlx.mlx_array_free(attn_b);
            try testing.expect(try qsaArraysAllEqual(attn_s, attn_b, s));
        }
    }
}

fn batchedAppendOpCount(bits: u8) !void {
    try mlx.requireGpu();
    const alloc = testing.allocator;
    const s = mlx.gpuStream();
    const cfg = kv_quant.KVQuantConfig.affine(bits);
    const ns = [_]usize{ 2, 4, 8 };
    const lens = [_]usize{ 2, 5, 3, 8, 4, 7, 6, 9 };
    var solo_at: [3]u64 = undefined;
    var bat_at: [3]u64 = undefined;
    for (ns, 0..) |n, ni| {
        const solo = try alloc.alloc(KVCache, n);
        const batched = try alloc.alloc(KVCache, n);
        defer {
            for (solo) |*c| c.deinit();
            alloc.free(solo);
            for (batched) |*c| c.deinit();
            alloc.free(batched);
        }
        const solo_ptrs = try alloc.alloc(*KVCache, n);
        defer alloc.free(solo_ptrs);
        const batched_ptrs = try alloc.alloc(*KVCache, n);
        defer alloc.free(batched_ptrs);
        for (0..n) |i| {
            solo[i] = try KVCache.initWithConfig(alloc, 1, cfg);
            batched[i] = try KVCache.initWithConfig(alloc, 1, cfg);
            try kvAppendPrefill(&solo[i], lens[i], @intCast(0xB000 + i), s);
            try kvAppendPrefill(&batched[i], lens[i], @intCast(0xB000 + i), s);
            solo_ptrs[i] = &solo[i];
            batched_ptrs[i] = &batched[i];
        }
        const stacked_k = try kvAppendPacked(@intCast(n), 1, 0x1111, s);
        defer _ = mlx.mlx_array_free(stacked_k);
        const stacked_v = try kvAppendPacked(@intCast(n), 1, 0x2222, s);
        defer _ = mlx.mlx_array_free(stacked_v);
        const c_solo = mlx.op_count.load(.monotonic);
        try KVCache.appendFromStacked(stacked_k, stacked_v, solo_ptrs, 0, s, 0, false);
        solo_at[ni] = mlx.op_count.load(.monotonic) - c_solo;
        const c_bat = mlx.op_count.load(.monotonic);
        try KVCache.appendFromStacked(stacked_k, stacked_v, batched_ptrs, 0, s, 0, true);
        bat_at[ni] = mlx.op_count.load(.monotonic) - c_bat;
        errdefer std.debug.print("batched KV append op-count N={d}: solo {d} batched {d}\n", .{ n, solo_at[ni], bat_at[ni] });
        try testing.expectEqual(24 * @as(u64, @intCast(n)), solo_at[ni]);
        try testing.expectEqual(8 + 18 * @as(u64, @intCast(n)), bat_at[ni]);
    }
    try testing.expect((bat_at[2] - bat_at[0]) < (solo_at[2] - solo_at[0]));
}

/// Walk a full-length cache and a ringed one through the same chunk schedule,
/// asserting the sliding view they hand SDPA is identical at every step.
fn swaRingWalk(kv_cfg: KVQuantConfig) !void {
    const alloc = std.testing.allocator;
    const s = mlx.gpuStream();
    const window: u32 = 8;
    const heads: c_int = 2;
    const dim: c_int = 64;

    var plain = try KVCache.initWithConfig(alloc, 1, kv_cfg);
    defer plain.deinit();
    var ringed = try KVCache.initWithConfig(alloc, 1, kv_cfg);
    defer ringed.deinit();
    ringed.setSwaRing(window);

    // Three prefill chunks past the compaction cap, then decode steps past it
    // again: both crossings, and chunk 2 straddles the window boundary.
    var schedule: [3 + 400]c_int = undefined;
    for (schedule[0..3]) |*w| w.* = 300;
    for (schedule[3..]) |*w| w.* = 1;

    var pos: c_int = 0;
    for (schedule) |q_len| {
        const k = try swaRingChunk(s, pos, q_len, heads, dim);
        defer _ = mlx.mlx_array_free(k);
        const v = try swaRingChunk(s, pos + 7919, q_len, heads, dim);
        defer _ = mlx.mlx_array_free(v);
        const span = slidingTailSpan(window, @intCast(q_len), SLIDING_TRIM_UNBOUNDED);

        var pv = try plain.update(0, k, v, s, span);
        defer pv.deinit();
        var rv = try ringed.update(0, k, v, s, span);
        defer rv.deinit();

        try std.testing.expectEqual(@as(f32, 0), try maxAbsDiffF32(pv.k, rv.k, s));
        try std.testing.expectEqual(@as(f32, 0), try maxAbsDiffF32(pv.v, rv.v, s));
        pos += q_len;
    }

    // The point of the ring: the sliding layer's storage stays bounded while the
    // full-length cache has grown to the whole walk.
    try std.testing.expectEqual(@as(usize, @intCast(pos)), plain.seqLen(0));
    try std.testing.expect(ringed.seqLen(0) <= KVCache.ringCap(window));
    // Absolute position is preserved even though the rows are not.
    try std.testing.expectEqual(plain.absSeqLen(0), ringed.absSeqLen(0));
}

/// Two prefill chunks through `layers` ringed sliding layers at MiMo's sliding geometry: every
/// layer's view is read (standing in for its attention) and each chunk's reads are evaluated.
/// Returns the bytes still active after the second chunk, over what was active before either.
fn swaRingChunkResidue(kv_cfg: KVQuantConfig, layers: u32, chunk: c_int) !usize {
    const alloc = std.testing.allocator;
    const s = mlx.gpuStream();
    const window: u32 = 128;
    var cache = try KVCache.initWithConfig(alloc, layers, kv_cfg);
    defer cache.deinit();
    cache.setSwaRing(window);
    var ks: [2]mlx.mlx_array = undefined;
    var vs: [2]mlx.mlx_array = undefined;
    for (&ks, &vs, 0..) |*k, *v, i| {
        k.* = try qkvIdentityArray(s, 8, chunk, 192, 0.3 + @as(f32, @floatFromInt(i)), .bfloat16);
        v.* = try qkvIdentityArray(s, 8, chunk, 128, 1.1 + @as(f32, @floatFromInt(i)), .bfloat16);
    }
    defer for (ks, vs) |k, v| {
        _ = mlx.mlx_array_free(k);
        _ = mlx.mlx_array_free(v);
    };
    for ([_][2]mlx.mlx_array{ ks, vs }) |pair| {
        const vec = mlx.mlx_vector_array_new_data(&pair, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_eval(vec));
    }
    try mlx.check(mlx.mlx_synchronize(s));
    _ = mlx.mlx_clear_cache();
    var before: usize = 0;
    try mlx.check(mlx.mlx_get_active_memory(&before));
    for (ks, vs) |k, v| {
        const reads = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(reads);
        for (0..layers) |li| {
            var view = try cache.update(@intCast(li), k, v, s, slidingTailSpan(window, @intCast(chunk), SLIDING_TRIM_UNBOUNDED));
            defer view.deinit();
            for ([_]mlx.mlx_array{ view.k, view.v }) |side| {
                var read = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(read);
                try mlx.check(mlx.mlx_sum(&read, side, false, s));
                _ = mlx.mlx_vector_array_append_value(reads, read);
            }
        }
        try mlx.check(mlx.mlx_eval(reads));
    }
    try mlx.check(mlx.mlx_synchronize(s));
    var after: usize = 0;
    try mlx.check(mlx.mlx_get_active_memory(&after));
    return after -| before;
}

fn ringCheckpointRestore(kv_cfg: KVQuantConfig) !void {
    const alloc = std.testing.allocator;
    const s = mlx.gpuStream();
    const window: u32 = 8;
    const prompt: usize = 700;
    const backoff: usize = ModelConfig.SWA_RING_CHECKPOINT_BACKOFF;

    var live = try KVCache.initWithConfig(alloc, 2, kv_cfg);
    defer live.deinit();
    live.setSwaRing(window);
    var plain = try KVCache.initWithConfig(alloc, 2, kv_cfg);
    defer plain.deinit();
    try ringCheckpointFeed(&live, &plain, window, 0, prompt, 250);

    var cp = (try live.ringCheckpoint(prompt, s)) orelse return error.TestExpectedRingCheckpoint;
    defer cp.deinit();
    try std.testing.expect(cp.ringServes(prompt) and cp.ringServes(prompt - backoff));
    try std.testing.expect(!cp.ringServes(prompt - backoff - 1));

    // A reply long enough that the ring compacts past the prompt end.
    try ringCheckpointFeed(&live, &plain, window, prompt, prompt + 600, 50);
    var end = try live.snapshotRetained(s);
    defer end.deinit();
    try std.testing.expect(!end.ringServes(prompt));

    var restored = try KVCache.initWithConfig(alloc, 2, kv_cfg);
    defer restored.deinit();
    restored.setSwaRing(window);
    try restored.restore(&end);
    try std.testing.expectError(error.SlidingRingRewindPastWindow, restored.truncate(prompt - 1, s));
    // The checkpoint's rows take the ringed layers; the global layer clamps as always.
    try restored.restoreRing(&cp);
    try restored.truncate(prompt - 1, s);
    try plain.truncate(prompt - 1, s);
    try std.testing.expectEqual(prompt - 1, restored.absSeqLen(0));
    try std.testing.expectEqual(prompt - 1, restored.absSeqLen(1));
    // Every view after the restore is the full-length cache's, across a compaction.
    try ringCheckpointFeed(&restored, &plain, window, prompt - 1, prompt + 700, 100);
}

fn ringMarksCapture(kv_cfg: KVQuantConfig) !void {
    const alloc = std.testing.allocator;
    const s = mlx.gpuStream();
    const window: u32 = 8;
    const end: usize = 1400;
    // Inside a write that does not compact, two inside one that does, and one never reached.
    const at = [_]usize{ 100, 610, 690, end + 5 };

    var live = try KVCache.initWithConfig(alloc, 2, kv_cfg);
    defer live.deinit();
    live.setSwaRing(window);
    var twin = try KVCache.initWithConfig(alloc, 2, kv_cfg);
    defer twin.deinit();
    twin.setSwaRing(window);
    var marks: [at.len]KVCacheSnapshot = undefined;
    var n: usize = 0;
    defer for (marks[0..n]) |*m| m.deinit();
    for (at) |p| {
        marks[n] = try live.ringMark(p);
        n += 1;
    }
    live.ring_marks = &marks;
    // Every view the marked cache hands a forward is its unmarked twin's.
    try ringCheckpointFeed(&live, &twin, window, 0, end, 300);
    live.ring_marks = &.{};
    var kept = try live.snapshotRetained(s);
    defer kept.deinit();
    var twin_kept = try twin.snapshotRetained(s);
    defer twin_kept.deinit();
    try std.testing.expect(try snapshotsBitEqual(&kept, &twin_kept, s));

    for (marks[0 .. at.len - 1]) |*m| {
        try std.testing.expect(live.ringMarkComplete(m));
        var ref = try KVCache.initWithConfig(alloc, 2, kv_cfg);
        defer ref.deinit();
        ref.setSwaRing(window);
        var scratch = try KVCache.initWithConfig(alloc, 2, kv_cfg);
        defer scratch.deinit();
        try ringCheckpointFeed(&ref, &scratch, window, 0, m.step, 300);
        var want = (try ref.ringCheckpoint(m.step, s)) orelse return error.TestExpectedRingCheckpoint;
        defer want.deinit();
        try std.testing.expect(try snapshotsBitEqual(m, &want, s));
    }
    try std.testing.expect(!live.ringMarkComplete(&marks[at.len - 1]));
}

/// `mlx_metal_is_available` is true for a Metal-enabled build with no usable
/// GPU, so this probes a real stream plus a 1x1 zeros+eval instead.
fn mlxDeviceUsable() bool {
    mlx.installErrorHandler();
    var trash: [256]u8 = undefined;
    if (mlx.errorPending()) _ = mlx.takeError(&trash);
    if (mlx.noGpuBackend()) return false;
    const s = mlx.gpuStream();
    if (s.ctx == null) {
        _ = mlx.takeError(&trash);
        return false;
    }
    const shape = [_]c_int{ 1, 1 };
    var a = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a);
    if (mlx.mlx_zeros(&a, &shape, 2, .float32, s) != 0) {
        _ = mlx.takeError(&trash);
        return false;
    }
    if (mlx.mlx_array_eval(a) != 0) {
        _ = mlx.takeError(&trash);
        return false;
    }
    return true;
}

fn verifyRollbackKvCase(config: KVQuantConfig, prefix_len: usize, force_growth: bool) !void {
    const allocator = testing.allocator;
    const stream = mlx.gpuStream();
    var cache = try KVCache.initWithConfig(allocator, 1, config);
    defer cache.deinit();
    var reference = try KVCache.initWithConfig(allocator, 1, config);
    defer reference.deinit();
    var prng = std.Random.DefaultPrng.init(1911 + prefix_len + @intFromBool(force_growth));
    if (prefix_len != 0) {
        const prefix = try attn256RandBf16(prng.random(), &.{ 1, 1, @intCast(prefix_len), 64 }, stream);
        defer _ = mlx.mlx_array_free(prefix);
        var cv = try cache.update(0, prefix, prefix, stream, 0);
        cv.deinit();
        var rv = try reference.update(0, prefix, prefix, stream, 0);
        rv.deinit();
        try verifyRollbackEvalStorage(&cache.entries[0]);
        try verifyRollbackEvalStorage(&reference.entries[0]);
    }
    const tail_len = if (force_growth and prefix_len != 0)
        KVCache.bufferCapacity(cache.entries[0].keys) - cache.entries[0].offset + 1
    else
        4;
    const tail = try attn256RandBf16(prng.random(), &.{ 1, 1, @intCast(tail_len), 64 }, stream);
    defer _ = mlx.mlx_array_free(tail);
    var offset = prefix_len;
    var ctx = ForwardCtx{ .cache = &cache, .moe_seq_offset = &offset, .ssm_entries = null, .capture_hidden = null, .vision_embeddings = null };
    var xfm: Transformer = undefined;
    xfm.allocator = allocator;
    xfm.fwd_gen = 0;
    var rollback = try Transformer.VerifyRollback.capture(&xfm, &ctx);
    defer rollback.deinit();
    var appended = try cache.update(0, tail, tail, stream, 0);
    appended.deinit();
    offset += tail_len;
    try verifyRollbackEvalStorage(&cache.entries[0]);
    rollback.restore();
    try testing.expectEqual(prefix_len, offset);
    try testing.expectEqual(reference.step, cache.step);
    try verifyRollbackValidPrefixEqual(&cache.entries[0], &reference.entries[0], stream);
    var resumed = try cache.update(0, tail, tail, stream, 0);
    resumed.deinit();
    var expected = try reference.update(0, tail, tail, stream, 0);
    expected.deinit();
    try verifyRollbackEvalStorage(&cache.entries[0]);
    try verifyRollbackEvalStorage(&reference.entries[0]);
    try testing.expectEqual(reference.step, cache.step);
    try verifyRollbackValidPrefixEqual(&cache.entries[0], &reference.entries[0], stream);
}

test "slidingTailSpan covers the whole block, not just the last query" {
    // Decode width: unchanged, exactly the window.
    try testing.expectEqual(@as(u32, 2048), slidingTailSpan(2048, 1, SLIDING_TRIM_UNBOUNDED));
    // A block of q queries needs window + q - 1 tail entries: the FIRST query
    // in the block reaches back furthest.
    try testing.expectEqual(@as(u32, 2052), slidingTailSpan(2048, 5, SLIDING_TRIM_UNBOUNDED));
    try testing.expectEqual(@as(u32, 2063), slidingTailSpan(2048, 16, SLIDING_TRIM_UNBOUNDED));
    // A PREFILL CHUNK is not special-cased any more: the old 32-wide cap kept
    // a chunk from ever taking the trim.
    try testing.expectEqual(@as(u32, 2080), slidingTailSpan(2048, 33, SLIDING_TRIM_UNBOUNDED));
    try testing.expectEqual(@as(u32, 2559), slidingTailSpan(512, 2048, SLIDING_TRIM_UNBOUNDED));
    // Non-sliding layer and empty forward still decline.
    try testing.expectEqual(@as(u32, 0), slidingTailSpan(0, 5, SLIDING_TRIM_UNBOUNDED));
    try testing.expectEqual(@as(u32, 0), slidingTailSpan(2048, 0, SLIDING_TRIM_UNBOUNDED));
    // max_width 1 is how the policy pins a forward to decode-only trimming.
    try testing.expectEqual(@as(u32, 4), slidingTailSpan(4, 1, 1));
    try testing.expectEqual(@as(u32, 0), slidingTailSpan(4, 2, 1));
}

test "slidingViewFor trims a sliding layer to the window tail and leaves a non-sliding one whole" {
    var cfg = ModelConfig{};
    cfg.has_sliding_window = true;
    cfg.sliding_window = 512;

    {
        const decode = slidingViewFor(&cfg, 20000, 1);
        try testing.expectEqual(@as(u32, 512), decode.span);
        try testing.expectEqual(@as(c_int, 512), decode.kv_len);

        const verify = slidingViewFor(&cfg, 20000, 8);
        try testing.expectEqual(@as(u32, 519), verify.span);
        try testing.expectEqual(@as(c_int, 519), verify.kv_len);

        const chunk = slidingViewFor(&cfg, 32768, 2048);
        try testing.expectEqual(@as(u32, 2559), chunk.span);
        try testing.expectEqual(@as(c_int, 2559), chunk.kv_len);
    }

    // A cache shorter than the span is not trimmed, and the reported kv_len is
    // what the view will actually be — masks are built from it.
    {
        const short = slidingViewFor(&cfg, 300, 8);
        try testing.expectEqual(@as(c_int, 300), short.kv_len);
    }

    cfg.has_sliding_window = false;
    {
        const full = slidingViewFor(&cfg, 20000, 8);
        try testing.expectEqual(@as(u32, 0), full.span);
        try testing.expectEqual(@as(c_int, 20000), full.kv_len);
    }
}

test "KVCache trims the tail view for a BLOCK-wide update, not just decode" {
    // The spec-verify shape: a block of queries lands at the tail of a cache
    // already longer than the window. Before the block trim, `new_len > 1`
    // fell through with view_start 0 and the layer read the ENTIRE cache.
    const s = mlx.gpuStream();
    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();

    // 10 prefill positions, untrimmed (a wide chunk keeps the full view).
    {
        const k = testKV(10, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKV(10, s);
        defer _ = mlx.mlx_array_free(v);
        var dv = try cache.update(0, k, v, s, 0);
        dv.deinit();
    }
    try testing.expectEqual(@as(usize, 10), cache.entries[0].offset);

    // Now a 3-wide verify block against a window of 4 → span 4 + 3 - 1 = 6.
    const span = slidingTailSpan(4, 3, SLIDING_TRIM_UNBOUNDED);
    try testing.expectEqual(@as(u32, 6), span);
    {
        const k = testKV(3, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKV(3, s);
        defer _ = mlx.mlx_array_free(v);
        var dv = try cache.update(0, k, v, s, span);
        defer dv.deinit();
        const view_shape = mlx.getShape(dv.k);
        try testing.expectEqual(@as(c_int, 6), view_shape[2]);
        const v_shape = mlx.getShape(dv.v);
        try testing.expectEqual(@as(c_int, 6), v_shape[2]);
    }
    // Absolute bookkeeping is untouched by the view trim.
    try testing.expectEqual(@as(usize, 13), cache.entries[0].offset);
    try testing.expectEqual(@as(usize, 13), cache.step);
}

test "KVCache carries a V head dim narrower than K (MLA 192/128)" {
    // MLA scores over nope+rope (192) but stores values at v_head_dim (128).
    // The dense path used to derive BOTH buffers from the K shape, so the V
    // slice_update wrote a 128-wide chunk into a 192-wide window and the cache
    // silently mis-shaped — grow, write, view and truncate each have to read
    // their OWN buffer's last dim. MLX's SDPA has a 192/128 vector kernel, so
    // this geometry is served, not padded.
    //
    // A cache entry's geometry is fixed by its FIRST write, so this needs its
    // own cache — the 192/128 chunk cannot land in a buffer another test's
    // narrow prefill already shaped.
    const s = mlx.gpuStream();
    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();

    // Prefill 3 tokens, then two decode steps that grow both buffers.
    {
        const k = testKVWide(3, 192, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKVWide(3, 128, s);
        defer _ = mlx.mlx_array_free(v);
        var dv = try cache.update(0, k, v, s, 0);
        defer dv.deinit();
        try testing.expectEqual(@as(c_int, 192), mlx.getShape(dv.k)[3]);
        try testing.expectEqual(@as(c_int, 128), mlx.getShape(dv.v)[3]);
        try testing.expectEqual(@as(c_int, 3), mlx.getShape(dv.v)[2]);
    }
    for (0..2) |_| {
        const k = testKVWide(1, 192, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKVWide(1, 128, s);
        defer _ = mlx.mlx_array_free(v);
        var dv = try cache.update(0, k, v, s, 0);
        defer dv.deinit();
        try testing.expectEqual(@as(c_int, 192), mlx.getShape(dv.k)[3]);
        try testing.expectEqual(@as(c_int, 128), mlx.getShape(dv.v)[3]);
    }
    try testing.expectEqual(@as(usize, 5), cache.entries[0].offset);
    // The views must be evaluable: a mis-shaped slice_update throws here.
    try mlx.check(mlx.mlx_array_eval(cache.entries[0].key_view));
    try mlx.check(mlx.mlx_array_eval(cache.entries[0].value_view));

    // Truncation rebuilds both views from their own buffers.
    try cache.truncate(2, s);
    try testing.expectEqual(@as(c_int, 2), mlx.getShape(cache.entries[0].value_view)[2]);
    try testing.expectEqual(@as(c_int, 128), mlx.getShape(cache.entries[0].value_view)[3]);
    try testing.expectEqual(@as(c_int, 192), mlx.getShape(cache.entries[0].key_view)[3]);
    try mlx.check(mlx.mlx_array_eval(cache.entries[0].value_view));
}

test "KVCache affine quant carries an asymmetric K/V too (--kv-quant on MLA)" {
    // The QUANTIZED write path solved every buffer's packed and scale widths
    // from K's head dim, so at K 192 / V 128 the value buffer was 24 u32 wide
    // while the incoming chunk was 16 — an mlx slice_update mismatch inside a
    // path we cannot catch. Nothing gated the SCHEME (the "MLA never opts in"
    // note covers the fused READ kernels only), so `--kv-quant 8` plus this
    // arch was a request-killing shape error. Each of the six buffers is sized
    // from its own operand; identical on every symmetric arch.
    const s = mlx.gpuStream();
    var cache = try KVCache.initWithConfig(testing.allocator, 1, kv_quant.KVQuantConfig.affine(8));
    defer cache.deinit();

    const mk = struct {
        fn f(str: mlx.mlx_stream, len: c_int, d: c_int) !mlx.mlx_array {
            const shape = [_]c_int{ 1, 2, len, d };
            var a = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_ones(&a, &shape, 4, .bfloat16, str));
            return a;
        }
    }.f;

    // Prefill 4, then two decode steps (the second crosses a growth event).
    {
        const k = try mk(s, 4, 192);
        defer _ = mlx.mlx_array_free(k);
        const v = try mk(s, 4, 128);
        defer _ = mlx.mlx_array_free(v);
        var dv = try cache.update(0, k, v, s, 0);
        defer dv.deinit();
        try testing.expect(dv.has_quant_triple);
        // 8-bit affine, group 64: K packs 192/4 = 48 u32 with 3 groups, V
        // 128/4 = 32 u32 with 2 — the point of the fix.
        try testing.expectEqual(@as(c_int, 48), mlx.getShape(dv.k_triple_q)[3]);
        try testing.expectEqual(@as(c_int, 32), mlx.getShape(dv.v_triple_q)[3]);
        try testing.expectEqual(@as(c_int, 3), mlx.getShape(dv.k_triple_scales)[3]);
        try testing.expectEqual(@as(c_int, 2), mlx.getShape(dv.v_triple_scales)[3]);
        // The dequantized views must be evaluable at their own widths, and the
        // values must survive the round trip (ones in, ones out).
        try mlx.check(mlx.mlx_array_eval(dv.v));
        try testing.expectEqual(@as(c_int, 192), mlx.getShape(dv.k)[3]);
        try testing.expectEqual(@as(c_int, 128), mlx.getShape(dv.v)[3]);
        var vf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(vf);
        try mlx.check(mlx.mlx_astype(&vf, dv.v, .float32, s));
        try mlx.check(mlx.mlx_array_eval(vf));
        const vd = mlx.mlx_array_data_float32(vf) orelse return error.InvalidDtype;
        try testing.expect(@abs(vd[0] - 1.0) < 1e-2);
        try testing.expect(@abs(vd[127] - 1.0) < 1e-2);
    }
    for (0..2) |_| {
        const k = try mk(s, 1, 192);
        defer _ = mlx.mlx_array_free(k);
        const v = try mk(s, 1, 128);
        defer _ = mlx.mlx_array_free(v);
        var dv = try cache.update(0, k, v, s, 0);
        defer dv.deinit();
        try mlx.check(mlx.mlx_array_eval(dv.k));
        try mlx.check(mlx.mlx_array_eval(dv.v));
    }
    try testing.expectEqual(@as(usize, 6), cache.entries[0].offset);

    // Truncation slices the value scale/bias views against THEIR width, not the
    // key scales' (2 groups vs 3).
    try cache.truncate(3, s);
    try testing.expectEqual(@as(c_int, 2), mlx.getShape(cache.entries[0].value_scales_view)[3]);
    try testing.expectEqual(@as(c_int, 3), mlx.getShape(cache.entries[0].key_scales_view)[3]);
    var restored = try cache.denseView(0, s);
    defer restored.deinit();
    try mlx.check(mlx.mlx_array_eval(restored.v));
    try testing.expectEqual(@as(c_int, 3), mlx.getShape(restored.v)[2]);
    try testing.expectEqual(@as(c_int, 128), mlx.getShape(restored.v)[3]);
}

test "a REFUSED KVCache re-init leaves the live cache intact, not freed" {
    // Every re-init site once read `cache.deinit(); cache = try init(...)`, so
    // a failed init left a FREED cache installed on the Transformer and its
    // owner's deinit freed every handle a second time — EXC_BAD_ACCESS in
    // `mlx_array_free`, several frames from the refusal that caused it. The
    // refusal here is the allocator's (the entries alloc of the second init);
    // under the testing allocator the pre-fix order trips the same double
    // free on this test's own `defer`.
    const s = mlx.gpuStream();
    var failing = std.testing.FailingAllocator.init(testing.allocator, .{ .fail_index = 1 });
    var cache = try KVCache.initWithConfig(failing.allocator(), 2, kv_quant.KVQuantConfig.affine(8));
    defer cache.deinit();

    // Real handles in the cache, so a double free has something to land on.
    {
        const k = testKVWide(3, 192, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKVWide(3, 128, s);
        defer _ = mlx.mlx_array_free(v);
        var dv = try cache.update(0, k, v, s, 0);
        dv.deinit();
    }
    try testing.expectEqual(@as(usize, 3), cache.entries[0].offset);

    try testing.expectError(error.OutOfMemory, cache.reinit(2, kv_quant.KVQuantConfig.affine(4)));
    failing.fail_index = std.math.maxInt(usize);

    // Untouched: same scheme, same contents, same entry count...
    try testing.expectEqual(kv_quant.Scheme.affine, cache.config.scheme);
    try testing.expectEqual(@as(u8, 8), cache.config.bits);
    try testing.expectEqual(@as(usize, 2), cache.entries.len);
    try testing.expectEqual(@as(usize, 3), cache.entries[0].offset);
    // ...and still SERVING, which a freed-but-readable cache would not be.
    {
        const k = testKVWide(1, 192, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKVWide(1, 128, s);
        defer _ = mlx.mlx_array_free(v);
        var dv = try cache.update(0, k, v, s, 0);
        defer dv.deinit();
        try mlx.check(mlx.mlx_array_eval(dv.k));
        try mlx.check(mlx.mlx_array_eval(dv.v));
    }
    try testing.expectEqual(@as(usize, 4), cache.entries[0].offset);

    // A re-init that SUCCEEDS still swaps — the fix must not turn the helper
    // into a no-op: new scheme, fresh entries, offsets back to zero.
    try cache.reinit(2, kv_quant.KVQuantConfig.affine(4));
    try testing.expectEqual(@as(u8, 4), cache.config.bits);
    try testing.expectEqual(@as(usize, 0), cache.entries[0].offset);
    try testing.expect(!cache.entries[0].initialized);
    try testing.expectEqual(@as(usize, 0), cache.step);
}

test "KVCache sliding window views return last max_seq entries" {
    const s = mlx.gpuStream();
    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();

    // Simulate 3 prefill tokens (max_seq=4 sliding window)
    {
        const k = testKV(3, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKV(3, s);
        defer _ = mlx.mlx_array_free(v);
        var dv = try cache.update(0, k, v, s, 4);
        dv.deinit();
    }
    // After prefill: offset=3, step=3
    try testing.expectEqual(@as(usize, 3), cache.entries[0].offset);
    try testing.expectEqual(@as(usize, 3), cache.step);

    // Decode token 4 — still within window
    {
        const k = testKV(1, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKV(1, s);
        defer _ = mlx.mlx_array_free(v);
        var dv = try cache.update(0, k, v, s, 4);
        dv.deinit();
    }
    try testing.expectEqual(@as(usize, 4), cache.entries[0].offset);
    try testing.expectEqual(@as(usize, 4), cache.step);

    // Decode token 5 — exceeds window, but buffer grows (no trimming)
    // Views return last 4 entries only
    {
        const k = testKV(1, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKV(1, s);
        defer _ = mlx.mlx_array_free(v);
        var dv = try cache.update(0, k, v, s, 4);
        defer dv.deinit();
        // View should be 4 entries (max_seq), not 5
        const view_shape = mlx.getShape(dv.k);
        try testing.expectEqual(@as(c_int, 4), view_shape[2]);
    }
    // Buffer has 5 entries, but view shows 4. step=5 (absolute).
    try testing.expectEqual(@as(usize, 5), cache.entries[0].offset);
    try testing.expectEqual(@as(usize, 5), cache.step);

    // Decode tokens 6,7,8 — step keeps incrementing, views stay at max_seq
    for (0..3) |_| {
        const k = testKV(1, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKV(1, s);
        defer _ = mlx.mlx_array_free(v);
        var dv = try cache.update(0, k, v, s, 4);
        defer dv.deinit();
        const view_shape = mlx.getShape(dv.k);
        try testing.expectEqual(@as(c_int, 4), view_shape[2]);
    }
    try testing.expectEqual(@as(usize, 8), cache.entries[0].offset);
    try testing.expectEqual(@as(usize, 8), cache.step);
}

test "KVCache updateAffine returns quant triples so fused attention can engage" {
    // Engagement-bug guard (docs/kv-quant-perf.md Phase 0/1): the decode
    // forward calls `cache.update(...)`, NOT `denseView`, so the triples must
    // ride the update path's DenseKVView too — otherwise the packed arms
    // silently never engage (output-identical to dense mode, which is
    // why an equivalence test can't see it).
    const s = mlx.gpuStream();
    var cache = try KVCache.initWithConfig(testing.allocator, 1, kv_quant.KVQuantConfig.affine(8));
    defer cache.deinit();

    const shape = [_]c_int{ 1, 2, 4, 64 };
    var k = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k);
    try mlx.check(mlx.mlx_ones(&k, &shape, 4, .bfloat16, s));
    var v = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v);
    try mlx.check(mlx.mlx_ones(&v, &shape, 4, .bfloat16, s));

    var dv = try cache.update(0, k, v, s, 0);
    defer dv.deinit();
    try testing.expect(dv.has_quant_triple);
    try testing.expect(dv.k_triple_q.ctx != null);
    try testing.expect(dv.v_triple_q.ctx != null);
    try testing.expectEqual(@as(u8, 8), dv.bits);
    try testing.expectEqual(@as(u32, 64), dv.group_size);
}

test "KVCache step resets on truncate" {
    const s = mlx.gpuStream();
    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();

    // Add 5 tokens
    {
        const k = testKV(5, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKV(5, s);
        defer _ = mlx.mlx_array_free(v);
        _ = try cache.update(0, k, v, s, 0);
    }
    try testing.expectEqual(@as(usize, 5), cache.step);

    // Truncate to 3 (simulating KV cache reuse)
    try cache.truncate(3, s);
    try testing.expectEqual(@as(usize, 3), cache.entries[0].offset);
    try testing.expectEqual(@as(usize, 3), cache.step);
}

test "KVCache step without trimming matches offset" {
    const s = mlx.gpuStream();
    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();

    // Add tokens without max_seq (no trimming)
    for (0..10) |_| {
        const k = testKV(1, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKV(1, s);
        defer _ = mlx.mlx_array_free(v);
        _ = try cache.update(0, k, v, s, 0);
    }
    // Without trimming, step and offset should be equal
    try testing.expectEqual(@as(usize, 10), cache.entries[0].offset);
    try testing.expectEqual(@as(usize, 10), cache.step);
}

test "kvLenForBatching: the batched cap reads the ATTENTION layers, never layer 0's step" {
    // `KVCache.step` advances on global layer 0 only, so on a linear-layer-0 trunk it reads 0
    // forever; the probe is "is entry 0 initialized?", not an arch list.
    const families = [_]struct { layers: u32, first_attn: u32, len: usize }{
        .{ .layers = 48, .first_attn = 3, .len = 60_000 }, // qwen4_exp GDN
        .{ .layers = 32, .first_attn = 7, .len = 131_072 }, // glm5_next KDA
    };
    for (families) |f| {
        var cache = try KVCache.init(testing.allocator, f.layers);
        defer cache.deinit();
        var li = f.first_attn;
        while (li < f.layers) : (li += 4) {
            cache.entries[li].initialized = true;
            cache.entries[li].offset = f.len;
        }
        try testing.expectEqual(@as(usize, 0), cache.step); // the trap
        try testing.expectEqual(f.len, cache.kvLenForBatching());
    }

    // Attention-first trunk: `step` stays the answer.
    var attn = try KVCache.init(testing.allocator, 4);
    defer attn.deinit();
    attn.step = 4096;
    attn.entries[0].initialized = true;
    attn.entries[0].offset = 4096;
    try testing.expectEqual(@as(usize, 4096), attn.kvLenForBatching());

    var cold = try KVCache.init(testing.allocator, 4);
    defer cold.deinit();
    try testing.expectEqual(@as(usize, 0), cold.kvLenForBatching());
}

test "KVCache step with multi-layer only increments once per update" {
    const s = mlx.gpuStream();
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();

    // Update all 3 layers with 2 tokens each
    for (0..3) |layer| {
        const k = testKV(2, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKV(2, s);
        defer _ = mlx.mlx_array_free(v);
        _ = try cache.update(@intCast(layer), k, v, s, 0);
    }
    // step should be 2 (one sequence worth), not 6 (3 layers × 2)
    try testing.expectEqual(@as(usize, 2), cache.step);
}

// ── Cache snapshot/restore (for spec-decode rollback) ──────────────────────

test "KVCache snapshot/restore round-trip preserves entries and step" {
    const s = mlx.gpuStream();
    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();

    // Build state: 4 tokens across 2 layers.
    for (0..2) |layer| {
        const k = testKV(4, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKV(4, s);
        defer _ = mlx.mlx_array_free(v);
        _ = try cache.update(@intCast(layer), k, v, s, 0);
    }
    try testing.expectEqual(@as(usize, 4), cache.step);
    try testing.expectEqual(@as(usize, 4), cache.entries[0].offset);

    // Snapshot at this point.
    var snap = try cache.snapshot();
    defer snap.deinit();

    // Mutate cache: add 2 more tokens to layer 0 only (to verify per-layer
    // offset is captured, not just global step).
    {
        const k = testKV(2, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKV(2, s);
        defer _ = mlx.mlx_array_free(v);
        _ = try cache.update(0, k, v, s, 0);
    }
    try testing.expectEqual(@as(usize, 6), cache.step);
    try testing.expectEqual(@as(usize, 6), cache.entries[0].offset);

    // Restore — step and per-layer offsets revert to snapshot.
    try cache.restore(&snap);
    try testing.expectEqual(@as(usize, 4), cache.step);
    try testing.expectEqual(@as(usize, 4), cache.entries[0].offset);
    try testing.expectEqual(@as(usize, 4), cache.entries[1].offset);
}

test "KVCache snapshot then more updates does not corrupt the snapshot" {
    // Critical invariant for spec-decode rollback: if we snapshot, then verify, then
    // (rejected) restore, snapshot must NOT have been mutated by the intervening
    // updates. The buffer is shared via refcount but must not be aliased through
    // the cache entry pointer.
    const s = mlx.gpuStream();
    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();

    {
        const k = testKV(2, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKV(2, s);
        defer _ = mlx.mlx_array_free(v);
        _ = try cache.update(0, k, v, s, 0);
    }

    var snap = try cache.snapshot();
    defer snap.deinit();
    try testing.expectEqual(@as(usize, 2), snap.entries[0].offset);

    // Run several updates to grow the cache buffer (forces buffer reallocation
    // inside update(), which would invalidate a naive snapshot).
    for (0..6) |_| {
        const k = testKV(1, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKV(1, s);
        defer _ = mlx.mlx_array_free(v);
        _ = try cache.update(0, k, v, s, 0);
    }
    try testing.expectEqual(@as(usize, 8), cache.entries[0].offset);

    // Snapshot still reports its captured state.
    try testing.expectEqual(@as(usize, 2), snap.entries[0].offset);
    try testing.expectEqual(@as(usize, 2), snap.step);

    try cache.restore(&snap);
    try testing.expectEqual(@as(usize, 2), cache.entries[0].offset);
    try testing.expectEqual(@as(usize, 2), cache.step);
}

test "KVCache: an append after a short restore sizes its copy from the prefix, not the donor (#492)" {
    // A 4k "hi" session restored from an 80k agent session's entry kept an 80k-capacity copy,
    // and its commit evicted the agent session to fit the budget.
    const s = mlx.gpuStream();
    for ([_]KVQuantConfig{ KVQuantConfig.dense, KVQuantConfig.affine(8) }) |cfg| {
        var donor: KVCache = undefined;
        var snap: KVCacheSnapshot = undefined;
        var cache: KVCache = undefined;
        const k = try sharedRestoreFixture(s, cfg, 0, 8192, &donor, &snap, &cache);
        defer _ = mlx.mlx_array_free(k);
        defer donor.deinit();
        defer snap.deinit();
        defer cache.deinit();
        try cache.truncate(100, s);
        const one = try sliceAttentionSeq(s, k, 100, 101);
        defer _ = mlx.mlx_array_free(one);
        var cv = try cache.update(0, one, one, s, 0);
        cv.deinit();
        try testing.expectEqual(@as(usize, 101), cache.entries[0].offset);
        try testing.expect(KVCache.bufferCapacity(cache.entries[0].keys) <= 512);
        try testing.expect(KVCache.bufferCapacity(snap.entries[0].keys) >= 8192);
    }
}

test "KVCache: a shared restore's first append allocates what the request reserves, not the donor's capacity" {
    // The admission bills a shared restore's copy at THIS request's reservation; the donor's buffer
    // is longer when it reserved for its own reply or decoded past its reservation.
    const s = mlx.gpuStream();
    for ([_]KVQuantConfig{ KVQuantConfig.dense, KVQuantConfig.affine(8) }) |cfg| {
        var donor: KVCache = undefined;
        var snap: KVCacheSnapshot = undefined;
        var warm: KVCache = undefined;
        const k = try sharedRestoreFixture(s, cfg, 16384, 1024, &donor, &snap, &warm);
        defer _ = mlx.mlx_array_free(k);
        defer donor.deinit();
        defer snap.deinit();
        defer warm.deinit();
        try testing.expectEqual(@as(usize, 16384), KVCache.bufferCapacity(donor.entries[0].keys));
        warm.reserve(2048);
        const tail = try sliceAttentionSeq(s, k, 0, 8);
        defer _ = mlx.mlx_array_free(tail);
        var wv = try warm.update(0, tail, tail, s, 0);
        defer wv.deinit();
        const want = warm.nextCapacityReserved(0, 1024 + 8);
        try testing.expect(want < 16384);
        try testing.expectEqual(want, KVCache.bufferCapacity(warm.entries[0].keys));
        try testing.expectEqual(want, KVCache.bufferCapacity(warm.entries[0].values));
        try testing.expectEqual(@as(usize, 1024 + 8), warm.entries[0].offset);
        if (cfg.scheme == .off) {
            const head = try sliceAttentionSeq(s, wv.k, 0, 1024);
            defer _ = mlx.mlx_array_free(head);
            try testing.expectEqual(@as(f32, 0), try maxAbsDiffF32(head, k, s));
        }
        // The donor's rows and buffer are untouched.
        try testing.expectEqual(@as(usize, 1024), donor.entries[0].offset);
        try testing.expectEqual(@as(usize, 16384), KVCache.bufferCapacity(donor.entries[0].keys));
    }
}

test "KVCache: a donated restore appends in place, at the entry's own capacity" {
    const s = mlx.gpuStream();
    for ([_]KVQuantConfig{ KVQuantConfig.dense, KVQuantConfig.affine(8) }) |cfg| {
        var donor: KVCache = undefined;
        var snap: KVCacheSnapshot = undefined;
        var warm: KVCache = undefined;
        const k = try sharedRestoreFixture(s, cfg, 16384, 1024, &donor, &snap, &warm);
        defer _ = mlx.mlx_array_free(k);
        defer donor.deinit();
        defer snap.deinit();
        defer warm.deinit();
        donor.deinit();
        donor = try KVCache.initWithConfig(testing.allocator, 1, cfg);
        snap.releaseHandles();
        warm.adoptRestored();
        warm.reserve(2048);
        const tail = try sliceAttentionSeq(s, k, 0, 8);
        defer _ = mlx.mlx_array_free(tail);
        var wv = try warm.update(0, tail, tail, s, 0);
        defer wv.deinit();
        try testing.expectEqual(@as(usize, 16384), KVCache.bufferCapacity(warm.entries[0].keys));
        try testing.expectEqual(@as(usize, 1024 + 8), warm.entries[0].offset);
    }
}

test "KVCache.growToReservation: the attention reads the same rows, and the decode never grows" {
    // A warm restore holding the prompt (C = 1024) but not the reservation (R = 2048): grown
    // before the prefill, or left to the decode step past C, every view must be bit-identical.
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    for ([_]KVQuantConfig{ KVQuantConfig.dense, KVQuantConfig.affine(8) }) |cfg| {
        for ([_]bool{ true, false }) |donate| {
            var donor: KVCache = undefined;
            var snap: KVCacheSnapshot = undefined;
            var eager: KVCache = undefined;
            const k = try sharedRestoreFixture(s, cfg, 1024, 1000, &donor, &snap, &eager);
            defer _ = mlx.mlx_array_free(k);
            defer donor.deinit();
            defer snap.deinit();
            defer eager.deinit();
            var lazy = try KVCache.initWithConfig(testing.allocator, 1, cfg);
            defer lazy.deinit();
            try lazy.restore(&snap);
            if (donate) {
                // Each cache owns its buffers, as a donated checkout does.
                donor.deinit();
                donor = try KVCache.initWithConfig(testing.allocator, 1, cfg);
                snap.releaseHandles();
                eager.adoptRestored();
                try cowAffineOrDense(&lazy, s);
            }
            eager.reserve(2048);
            lazy.reserve(2048);
            try eager.growToReservation(s);
            try testing.expectEqual(@as(usize, 2048), KVCache.bufferCapacity(eager.entries[0].keys));
            try testing.expectEqual(@as(usize, 1000), eager.entries[0].offset);

            const rows = try qkvIdentityArray(s, 2, 64, 64, 0.9, .bfloat16);
            defer _ = mlx.mlx_array_free(rows);
            var lazy_grows: usize = 0;
            // A 16-row prefill, then single-row decode steps past C = 1024.
            var at: c_int = 0;
            while (at < 64) {
                const w: c_int = if (at == 0) 16 else 1;
                const step = try sliceAttentionSeq(s, rows, at, at + w);
                defer _ = mlx.mlx_array_free(step);
                const g0 = KVCache.kv_cap_buf_grows;
                var ev = try eager.update(0, step, step, s, 0);
                defer ev.deinit();
                try testing.expectEqual(g0, KVCache.kv_cap_buf_grows);
                var lv = try lazy.update(0, step, step, s, 0);
                defer lv.deinit();
                lazy_grows += KVCache.kv_cap_buf_grows - g0;
                try qkvExpectBitsEqual(s, ev.k, lv.k);
                try qkvExpectBitsEqual(s, ev.v, lv.v);
                at += w;
            }
            try testing.expect(lazy_grows > 0);
        }
    }
}

test "KVCache snapshot/restore in a tight loop does not leak" {
    // testing.allocator is a TrackingAllocator — any unfreed allocation here
    // surfaces as a test failure at the leak-detection step.
    const s = mlx.gpuStream();
    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();

    {
        const k = testKV(3, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKV(3, s);
        defer _ = mlx.mlx_array_free(v);
        _ = try cache.update(0, k, v, s, 0);
        const k2 = testKV(3, s);
        defer _ = mlx.mlx_array_free(k2);
        const v2 = testKV(3, s);
        defer _ = mlx.mlx_array_free(v2);
        _ = try cache.update(1, k2, v2, s, 0);
    }

    var i: usize = 0;
    while (i < 50) : (i += 1) {
        var snap = try cache.snapshot();
        defer snap.deinit();
        const k = testKV(1, s);
        defer _ = mlx.mlx_array_free(k);
        const v = testKV(1, s);
        defer _ = mlx.mlx_array_free(v);
        _ = try cache.update(0, k, v, s, 0);
        try cache.restore(&snap);
    }
}

test "batched KV append: affine-8 N=2/4/8 matches N solo updates at production geometry" {
    try batchedAppendMatchesSolo(8);
}

test "batched KV append: affine-6 N=2/4/8 matches N solo updates at production geometry" {
    try batchedAppendMatchesSolo(6);
}

test "batched KV append: caches of different widths append per slot, byte-equal to solo updates" {
    try mlx.requireGpu();
    const alloc = testing.allocator;
    const s = mlx.gpuStream();
    const mixes = [_][3]u8{ .{ 6, 8, 0 }, .{ 8, 6, 0 }, .{ 8, 8, 6 }, .{ 6, 6, 4 } };
    for (mixes) |mix| {
        const n: usize = if (mix[2] == 0) 2 else 3;
        var solo: [3]KVCache = undefined;
        var batched: [3]KVCache = undefined;
        var solo_ptrs: [3]*KVCache = undefined;
        var batched_ptrs: [3]*KVCache = undefined;
        for (0..n) |i| {
            const cfg = kv_quant.KVQuantConfig.affine(mix[i]);
            solo[i] = try KVCache.initWithConfig(alloc, 1, cfg);
            batched[i] = try KVCache.initWithConfig(alloc, 1, cfg);
            try kvAppendPrefill(&solo[i], 3 + i, @intCast(0xB000 + i), s);
            try kvAppendPrefill(&batched[i], 3 + i, @intCast(0xB000 + i), s);
            solo_ptrs[i] = &solo[i];
            batched_ptrs[i] = &batched[i];
        }
        defer for (0..n) |i| {
            solo[i].deinit();
            batched[i].deinit();
        };
        const stacked_k = try kvAppendPacked(@intCast(n), 1, 0xC0FFE, s);
        defer _ = mlx.mlx_array_free(stacked_k);
        const stacked_v = try kvAppendPacked(@intCast(n), 1, 0x0D15EA, s);
        defer _ = mlx.mlx_array_free(stacked_v);
        try KVCache.appendFromStacked(stacked_k, stacked_v, solo_ptrs[0..n], 0, s, 0, false);
        try KVCache.appendFromStacked(stacked_k, stacked_v, batched_ptrs[0..n], 0, s, 0, true);
        for (0..n) |i| try kvAppendOperandsEqual(&solo[i].entries[0], &batched[i].entries[0], s);
    }
}

test "batched KV append: two restored siblings append and truncate independently; the snapshot stays byte-identical" {
    try mlx.requireGpu();
    const alloc = testing.allocator;
    const s = mlx.gpuStream();
    const cfg = kv_quant.KVQuantConfig.affine(8);
    var src = try KVCache.initWithConfig(alloc, 1, cfg);
    defer src.deinit();
    try kvAppendPrefill(&src, 6, 0x51A0, s);
    var snap = try src.snapshot();
    defer snap.deinit();
    const snap_k = try materializedOwnedCopy(s, snap.entries[0].keys);
    defer _ = mlx.mlx_array_free(snap_k);
    const snap_v = try materializedOwnedCopy(s, snap.entries[0].values);
    defer _ = mlx.mlx_array_free(snap_v);
    const snap_ks = try materializedOwnedCopy(s, snap.entries[0].keys_scales);
    defer _ = mlx.mlx_array_free(snap_ks);
    const snap_kb = try materializedOwnedCopy(s, snap.entries[0].keys_biases);
    defer _ = mlx.mlx_array_free(snap_kb);
    const snap_vs = try materializedOwnedCopy(s, snap.entries[0].values_scales);
    defer _ = mlx.mlx_array_free(snap_vs);
    const snap_vb = try materializedOwnedCopy(s, snap.entries[0].values_biases);
    defer _ = mlx.mlx_array_free(snap_vb);
    try mlx.check(mlx.mlx_array_eval(snap_k));
    var a = try KVCache.initWithConfig(alloc, 1, cfg);
    defer a.deinit();
    var b = try KVCache.initWithConfig(alloc, 1, cfg);
    defer b.deinit();
    try a.restore(&snap);
    try b.restore(&snap);
    const stacked_k = try kvAppendPacked(2, 1, 0xABBA, s);
    defer _ = mlx.mlx_array_free(stacked_k);
    const stacked_v = try kvAppendPacked(2, 1, 0xBEEF, s);
    defer _ = mlx.mlx_array_free(stacked_v);
    var sibs = [_]*KVCache{ &a, &b };
    try KVCache.appendBatched(stacked_k, stacked_v, &sibs, 0, s, 0, cfg);
    try testing.expectEqual(@as(usize, 7), a.entries[0].offset);
    try testing.expectEqual(@as(usize, 7), b.entries[0].offset);
    try a.truncate(6, s);
    try testing.expectEqual(@as(usize, 6), a.entries[0].offset);
    try testing.expectEqual(@as(usize, 7), b.entries[0].offset);
    try testing.expect(try qsaArraysAllEqual(snap.entries[0].keys, snap_k, s));
    try testing.expect(try qsaArraysAllEqual(snap.entries[0].values, snap_v, s));
    try testing.expect(try qsaArraysAllEqual(snap.entries[0].keys_scales, snap_ks, s));
    try testing.expect(try qsaArraysAllEqual(snap.entries[0].keys_biases, snap_kb, s));
    try testing.expect(try qsaArraysAllEqual(snap.entries[0].values_scales, snap_vs, s));
    try testing.expect(try qsaArraysAllEqual(snap.entries[0].values_biases, snap_vb, s));
    const k1 = try KVCache.sliceStackedRow(s, stacked_k, 0);
    defer _ = mlx.mlx_array_free(k1);
    const v1 = try KVCache.sliceStackedRow(s, stacked_v, 0);
    defer _ = mlx.mlx_array_free(v1);
    var dv = try a.update(0, k1, v1, s, 0);
    dv.deinit();
    try testing.expect(try qsaArraysAllEqual(snap.entries[0].keys, snap_k, s));
}

test "batched KV append: per-layer op count is quantize-once plus per-slot writes, not N x (2 slices + update)" {
    try batchedAppendOpCount(8);
    try batchedAppendOpCount(6);
}

test "batched KV append: dense and affine-4 stay on the per-slot path" {
    try mlx.requireGpu();
    const alloc = testing.allocator;
    const s = mlx.gpuStream();
    const schemes = [_]kv_quant.KVQuantConfig{
        kv_quant.KVQuantConfig.dense,
        kv_quant.KVQuantConfig.affine(4),
    };
    for (schemes) |cfg| {
        var solo: [2]KVCache = .{ try KVCache.initWithConfig(alloc, 1, cfg), try KVCache.initWithConfig(alloc, 1, cfg) };
        var batched: [2]KVCache = .{ try KVCache.initWithConfig(alloc, 1, cfg), try KVCache.initWithConfig(alloc, 1, cfg) };
        defer for (&solo) |*c| c.deinit();
        defer for (&batched) |*c| c.deinit();
        try kvAppendPrefill(&solo[0], 3, 0x10, s);
        try kvAppendPrefill(&solo[1], 5, 0x11, s);
        try kvAppendPrefill(&batched[0], 3, 0x10, s);
        try kvAppendPrefill(&batched[1], 5, 0x11, s);
        var solo_ptrs = [_]*KVCache{ &solo[0], &solo[1] };
        var batched_ptrs = [_]*KVCache{ &batched[0], &batched[1] };
        const stacked_k = try kvAppendPacked(2, 1, 0x30, s);
        defer _ = mlx.mlx_array_free(stacked_k);
        const stacked_v = try kvAppendPacked(2, 1, 0x31, s);
        defer _ = mlx.mlx_array_free(stacked_v);
        const c0 = mlx.op_count.load(.monotonic);
        try KVCache.appendFromStacked(stacked_k, stacked_v, &solo_ptrs, 0, s, 0, false);
        const solo_ops = mlx.op_count.load(.monotonic) - c0;
        const c1 = mlx.op_count.load(.monotonic);
        try KVCache.appendFromStacked(stacked_k, stacked_v, &batched_ptrs, 0, s, 0, true);
        const batched_ops = mlx.op_count.load(.monotonic) - c1;
        try testing.expectEqual(solo_ops, batched_ops);
        for (0..2) |i| {
            try testing.expectEqual(solo[i].entries[0].offset, batched[i].entries[0].offset);
            try testing.expect(try qsaArraysAllEqual(solo[i].entries[0].keys, batched[i].entries[0].keys, s));
            try testing.expect(try qsaArraysAllEqual(solo[i].entries[0].values, batched[i].entries[0].values, s));
            if (cfg.scheme == .affine) {
                try kvAppendOperandsEqual(&solo[i].entries[0], &batched[i].entries[0], s);
            }
        }
    }
}

test "a ringed sliding layer serves the same view as a full-length cache" {
    try mlx.requireGpu();
    try swaRingWalk(KVQuantConfig.dense);
}

test "a ringed sliding layer serves the same view under kv-quant" {
    try mlx.requireGpu();
    try swaRingWalk(.{ .scheme = .affine, .bits = 8, .group_size = 32 });
    try swaRingWalk(.{ .scheme = .affine, .bits = 4, .group_size = 32 });
}

test "a ringed layer keeps only its ring once the chunk that compacted it is evaluated" {
    // The ring bill (`ModelConfig.swaRingBytes`) holds `ringCap` rows per sliding layer across
    // chunks; the pre-compaction buffer is billed only for the width of the forward.
    try mlx.requireGpu();
    const layers: u32 = 6;
    for ([_]KVQuantConfig{ KVQuantConfig.dense, KVQuantConfig.affine(8) }) |cfg| {
        const residue = try swaRingChunkResidue(cfg, layers, 2048);
        const row_bytes: usize = if (cfg.scheme == .off) 8 * (192 + 128) * 2 else 8 * (192 + 128) + 8 * (192 + 128) / 64 * 2 * 2;
        const ring = layers * KVCache.ringCap(128) * row_bytes;
        std.testing.expect(residue <= ring + ring / 20) catch |err| {
            std.debug.print("ring residue {d} B over the {d} B ring bill ({s})\n", .{ residue, ring, @tagName(cfg.scheme) });
            return err;
        };
    }
}

test "a ring rewinds inside its retained window and declines below it" {
    try mlx.requireGpu();
    const alloc = std.testing.allocator;
    const s = mlx.gpuStream();
    const window: u32 = 8;
    var c = try KVCache.init(alloc, 1);
    defer c.deinit();
    c.setSwaRing(window);

    var pos: c_int = 0;
    while (pos < 2000) : (pos += 250) {
        const k = try swaRingChunk(s, pos, 250, 1, 4);
        defer _ = mlx.mlx_array_free(k);
        var view = try c.update(0, k, k, s, slidingTailSpan(window, 250, SLIDING_TRIM_UNBOUNDED));
        view.deinit();
    }
    const abs = c.absSeqLen(0);
    // A clamp inside the retained rows serves; the full-reuse rewind of one
    // token is the common case and must never decline.
    try c.truncate(abs - 1, s);
    try std.testing.expectEqual(abs - 1, c.absSeqLen(0));
    // A rewind past the retained window has no rows to serve it and says so.
    try std.testing.expectError(error.SlidingRingRewindPastWindow, c.truncate(window, s));
    // A byte-budget trim asks the same question of a snapshot and declines the
    // same way, so an oversized candidate retries shorter instead of restoring
    // a window it cannot reproduce.
    var snap = try c.snapshot();
    defer snap.deinit();
    var kept = try snap.trimmedCopy(c.absSeqLen(0), null, s);
    kept.deinit();
    try std.testing.expectError(error.SlidingRingRewindPastWindow, snap.trimmedCopy(window, null, s));
    // Zero is a reset, not a rewind.
    try c.truncate(0, s);
    try std.testing.expectEqual(@as(usize, 0), c.absSeqLen(0));
}

test "KVCache resident bytes are the buffers it owns, never a restored donor's" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testing.expectEqual(@as(u64, 0), cache.residentBytes());
    const k = testKVWide(3, 64, s);
    defer _ = mlx.mlx_array_free(k);
    var dv = try cache.update(0, k, k, s, 0);
    dv.deinit();
    const e = cache.entries[0];
    const want = 2 * @as(u64, mlx.mlx_array_size(e.keys)) * @as(u64, mlx.mlx_array_itemsize(e.keys));
    try testing.expect(want > 0);
    try testing.expectEqual(want, cache.residentBytes());
    cache.entries[0].shared_view = true;
    try testing.expectEqual(@as(u64, 0), cache.residentBytes());
    cache.entries[0].shared_view = false;
}

test "a ringed layer's resident bytes are its ring, not the history it has seen" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const window: u32 = 8;
    var c = try KVCache.init(testing.allocator, 1);
    defer c.deinit();
    c.setSwaRing(window);
    var pos: c_int = 0;
    while (pos < 2000) : (pos += 250) {
        const k = try swaRingChunk(s, pos, 250, 1, 4);
        defer _ = mlx.mlx_array_free(k);
        var view = try c.update(0, k, k, s, slidingTailSpan(window, 250, SLIDING_TRIM_UNBOUNDED));
        view.deinit();
    }
    const e = c.entries[0];
    const row: u64 = 4 * @as(u64, mlx.mlx_array_itemsize(e.keys));
    try testing.expectEqual(2 * @as(u64, @intCast(KVCache.bufferCapacity(e.keys))) * row, c.residentBytes());
    try testing.expect(c.residentBytes() < 2 * 2000 * row);
}

test "a trim below the ring takes the ringed rows from a checkpoint at the trim point" {
    try mlx.requireGpu();
    const alloc = std.testing.allocator;
    const s = mlx.gpuStream();
    const window: u32 = 8;
    const prompt: usize = 700;

    var live = try KVCache.init(alloc, 2);
    defer live.deinit();
    live.setSwaRing(window);
    var plain = try KVCache.init(alloc, 2);
    defer plain.deinit();
    try ringCheckpointFeed(&live, &plain, window, 0, prompt, 250);
    var cp = (try live.ringCheckpoint(prompt, s)) orelse return error.TestExpectedRingCheckpoint;
    defer cp.deinit();
    try ringCheckpointFeed(&live, &plain, window, prompt, prompt + 600, 50);
    var end = try live.snapshotRetained(s);
    defer end.deinit();

    try std.testing.expectError(error.SlidingRingRewindPastWindow, end.trimmedCopy(prompt, null, s));
    // A checkpoint short of the trim point has no rows for it either.
    try std.testing.expectError(error.SlidingRingRewindPastWindow, end.trimmedCopy(prompt + 1, &cp, s));
    var kept = try end.trimmedCopy(prompt, &cp, s);
    defer kept.deinit();
    try std.testing.expectEqual(prompt, kept.step);

    var restored = try KVCache.init(alloc, 2);
    defer restored.deinit();
    restored.setSwaRing(window);
    try restored.restore(&kept);
    try restored.truncate(prompt - 1, s);
    try plain.truncate(prompt - 1, s);
    // Every view after the trim is the full-length cache's, across a compaction.
    try ringCheckpointFeed(&restored, &plain, window, prompt - 1, prompt + 700, 100);
}

test "a ring checkpoint serves the clamp the ring's end state declines" {
    try mlx.requireGpu();
    try ringCheckpointRestore(KVQuantConfig.dense);
    try ringCheckpointRestore(.{ .scheme = .affine, .bits = 8, .group_size = 32 });
}

test "a ring restored from a checkpoint below its position still checkpoints a short tail's end" {
    try mlx.requireGpu();
    const alloc = std.testing.allocator;
    const s = mlx.gpuStream();
    const window: u32 = 8;
    const prompt: usize = 700;
    const backoff: usize = ModelConfig.SWA_RING_CHECKPOINT_BACKOFF;

    var live = try KVCache.init(alloc, 2);
    defer live.deinit();
    live.setSwaRing(window);
    var plain = try KVCache.init(alloc, 2);
    defer plain.deinit();
    try ringCheckpointFeed(&live, &plain, window, 0, prompt, 250);
    var cp = (try live.ringCheckpoint(prompt, s)) orelse return error.TestExpectedRingCheckpoint;
    defer cp.deinit();
    var entry = try live.snapshotRetained(s);
    defer entry.deinit();

    // The next turn restores 20 rows below the checkpoint and forwards a 10-token tail: the
    // ring then holds fewer than window + backoff rows below the new prompt end.
    var restored = try KVCache.init(alloc, 2);
    defer restored.deinit();
    restored.setSwaRing(window);
    try restored.restore(&entry);
    try restored.restoreRing(&cp);
    try restored.truncate(prompt - 20, s);
    try plain.truncate(prompt - 20, s);
    try ringCheckpointFeed(&restored, &plain, window, prompt - 20, prompt - 10, 10);

    var tail_cp = (try restored.ringCheckpoint(prompt - 10, s)) orelse return error.TestExpectedRingCheckpoint;
    defer tail_cp.deinit();
    const low = prompt - window - backoff;
    try std.testing.expect(tail_cp.ringServes(prompt - 10) and tail_cp.ringServes(low + window));
    try std.testing.expect(!tail_cp.ringServes(low + window - 1));

    // Fewer rows than the window below the position is still no checkpoint.
    var short = try KVCache.init(alloc, 2);
    defer short.deinit();
    short.setSwaRing(window);
    try short.restore(&entry);
    try short.restoreRing(&cp);
    try std.testing.expectError(error.SlidingRingRewindPastWindow, short.ringCheckpoint(low + window - 1, s));
}

test "a ring mark holds the checkpoint at its position, taken as the writes pass it, and changes no view" {
    try mlx.requireGpu();
    try ringMarksCapture(KVQuantConfig.dense);
    try ringMarksCapture(.{ .scheme = .affine, .bits = 8, .group_size = 32 });
}

test "KVCache growth strands bounded bytes" {
    // Walk a decode from empty to 100_000 tokens one token at a time and count
    // what the grow-and-copy policy leaves behind. Every growth event allocates
    // a fresh capacity-sized buffer, copies the old contents in, and frees the
    // old one — and MLX only reuses a cached buffer when
    // `cached_size < requested + 2*page_size` (`buffer_cache.h`), so under the
    // old fixed +256 policy the superseded buffer is always 512 KB-plus past a
    // 32 KB tolerance and can never be reused for the NEXT KV buffer either.
    // On the reporter's 64-KiB/token model that is a whole KV cache orphaned
    // per 256 generated tokens.
    const walk = struct {
        const Result = struct { events: usize, superseded: usize, final: usize };
        fn run() Result {
            var cap: usize = 0;
            var events: usize = 0;
            var superseded: usize = 0;
            var n: usize = 1;
            while (n <= 100_000) : (n += 1) {
                const next = KVCache.nextCapacityPolicy(cap, n);
                if (next != cap) {
                    events += 1;
                    superseded += cap;
                    cap = next;
                }
            }
            return .{ .events = events, .superseded = superseded, .final = cap };
        }
    };

    const prop = walk.run();
    try testing.expect(prop.final >= 100_000);
    try testing.expect(prop.events <= 40);
    try testing.expect(prop.superseded <= 8 * prop.final);
}

test "KV growth does not ratchet the MLX buffer pool" {
    // The reporter's mechanism (issue #110) at a size that runs in seconds and
    // needs no checkpoint: single-token `update` calls against a real KVCache,
    // sampling MLX's allocator pool. Nothing here calls `mlx_clear_cache` —
    // that is the point. What the pool holds is exactly what the growth policy
    // orphaned.
    //
    // 4 layers x 2 kv heads x 64 head_dim, bf16 = 2 KiB per token across all K
    // and V buffers. Under the old +256 policy the walk to 20_000 allocates
    // ~810_000 token-slots and frees all but the last ≈ 41 MB. Measured peak
    // pool: 1544 MB linear vs 274 MB proportional, so the bar sits between them
    // with ~2x margin either side rather than on top of the passing number.
    const s = mlx.gpuStream();
    const LAYERS: u32 = 4;
    const N: usize = 20_000;
    const shape = [_]c_int{ 1, 2, 1, 64 };

    var cache = try KVCache.init(testing.allocator, LAYERS);
    defer cache.deinit();

    _ = mlx.mlx_clear_cache();

    var peak_cache: usize = 0;
    var t: usize = 0;
    while (t < N) : (t += 1) {
        var layer: u32 = 0;
        while (layer < LAYERS) : (layer += 1) {
            var k = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(k);
            var v = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(v);
            try mlx.check(mlx.mlx_zeros(&k, &shape, 4, .bfloat16, s));
            try mlx.check(mlx.mlx_zeros(&v, &shape, 4, .bfloat16, s));
            var view = try cache.update(layer, k, v, s, 0);
            view.deinit();
        }
        if (t % 32 == 0) {
            cache.evalState();
            var c: usize = 0;
            _ = mlx.mlx_get_cache_memory(&c);
            if (c > peak_cache) peak_cache = c;
        }
    }
    cache.evalState();
    var final_cache: usize = 0;
    _ = mlx.mlx_get_cache_memory(&final_cache);
    if (final_cache > peak_cache) peak_cache = final_cache;

    if (peak_cache > 512 << 20) {
        std.debug.print(
            "[kv-pool] peak MLX buffer pool {d} MB after {d} single-token updates\n",
            .{ peak_cache >> 20, N },
        );
        return error.BufferPoolRatcheted;
    }
}

test "growQuantBuf else-arm: a faulted zeros leaves the old buffer owned (no double free)" {
    // A failed mlx_zeros leaves the caller's old buffer owned. The free below
    // is then legal; freeing it first would free it twice.
    if (!mlxDeviceUsable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const B: c_int = 1;
    const heads: c_int = 1;
    const last_dim: c_int = 4;
    const old_cap: c_int = 2;
    const new_cap: c_int = 8;
    const old_shape = [_]c_int{ B, heads, old_cap, last_dim };

    var warm = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_zeros(&warm, &old_shape, 4, .float32, s));
    try mlx.check(mlx.mlx_array_eval(warm));
    const c0 = mlx.op_count.load(.monotonic);
    try KVCache.growQuantBuf(s, &warm, false, 0, new_cap, B, heads, last_dim, .float32);
    const n_ops = mlx.op_count.load(.monotonic) - c0;
    try mlx.check(mlx.mlx_array_eval(warm));
    try testing.expectEqual(@as(c_int, new_cap), mlx.getShape(warm)[2]);
    _ = mlx.mlx_array_free(warm);
    try testing.expect(n_ops >= 1);

    var fired: usize = 0;
    var k: u64 = 1;
    while (k <= n_ops) : (k += 1) {
        var buf = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_zeros(&buf, &old_shape, 4, .float32, s));
        try mlx.check(mlx.mlx_array_eval(buf));
        mlx.fault.arm(k);
        const r = KVCache.growQuantBuf(s, &buf, false, 0, new_cap, B, heads, last_dim, .float32);
        const did = mlx.fault.didFire();
        mlx.fault.disarm();
        if (r) |_| {
            // Not reached for k in 1..=n_ops; free the replacement in case.
            _ = mlx.mlx_array_free(buf);
        } else |err| {
            try testing.expectEqual(error.MlxError, err);
            try testing.expect(did);
            fired += 1;
            // This free must be legal; it is the assertion.
            _ = mlx.mlx_array_free(buf);
        }
    }
    try testing.expectEqual(n_ops, fired);
}

test "KVCache dense update: a faulted op leaves no freed view handle in the entry" {
    // The bar: after any failing op, `deinit` frees each handle the entry holds exactly once.
    if (!mlxDeviceUsable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const first = testKV(4, s);
    defer _ = mlx.mlx_array_free(first);

    // 4 more tokens fit the first update's buffer; 300 more take the grow path.
    var ops = [_]u64{ 0, 0 };
    for ([_]usize{ 4, 300 }, &ops) |len, *n_ops| {
        const kv = testKV(len, s);
        defer _ = mlx.mlx_array_free(kv);
        {
            var cache = try KVCache.init(testing.allocator, 1);
            defer cache.deinit();
            var dv = try cache.update(0, first, first, s, 0);
            dv.deinit();
            const c0 = mlx.op_count.load(.monotonic);
            var dv2 = try cache.update(0, kv, kv, s, 0);
            dv2.deinit();
            n_ops.* = mlx.op_count.load(.monotonic) - c0;
        }

        var k: u64 = 1;
        while (k <= n_ops.*) : (k += 1) {
            var cache = try KVCache.init(testing.allocator, 1);
            defer cache.deinit();
            var dv = try cache.update(0, first, first, s, 0);
            dv.deinit();
            mlx.fault.arm(k);
            const r = cache.update(0, kv, kv, s, 0);
            const did = mlx.fault.didFire();
            mlx.fault.disarm();
            try testing.expectError(error.MlxError, r);
            try testing.expect(did);
            // The views are built last, K then V: a fault before a view's own build leaves it empty.
            if (k + 1 < n_ops.*) try testing.expect(cache.entries[0].key_view.ctx == null);
            if (k < n_ops.*) try testing.expect(cache.entries[0].value_view.ctx == null);
        }
    }
    try testing.expect(ops[0] >= 4 and ops[1] > ops[0]);
}

test "generation past the reserved headroom takes the +25% growth policy, never a mid-prefill grow" {
    const t = std.testing;
    // The reservation is bounded at `RESERVE_GEN_HEADROOM` tokens of generation; past it the
    // ordinary proportional policy grows between decode steps.
    const reserve_headroom = KVCache.RESERVE_GEN_HEADROOM;
    const chunk: usize = 4096;
    const seq: usize = 40_000;
    const reserved: usize = @intCast(KVCache.reservedTokens(seq, 1_000_000, chunk, 0));
    try t.expectEqual(seq + reserve_headroom + chunk, reserved);

    const s_gpu = mlx.gpuStream();
    var cache = try KVCache.init(t.allocator, 1);
    defer cache.deinit();
    cache.reserve(reserved);
    var written: usize = 0;
    var allocs: usize = 0;
    var last_cap: usize = 0;
    var cap_at_prefill_end: usize = 0;
    const total: usize = reserved + reserve_headroom;
    while (written < total) {
        const n: c_int = @intCast(@min(chunk, total - written));
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        const count: f64 = @floatFromInt(n * 8);
        try mlx.check(mlx.mlx_arange(&flat, 0.0, count, 1.0, .float32, s_gpu));
        var k = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k);
        const shape = [_]c_int{ 1, 1, n, 8 };
        try mlx.check(mlx.mlx_reshape(&k, flat, &shape, 4, s_gpu));
        var view = try cache.update(0, k, k, s_gpu, 0);
        view.deinit();
        const cap_now: usize = KVCache.bufferCapacity(cache.entries[0].keys);
        if (cap_now != last_cap) {
            allocs += 1;
            last_cap = cap_now;
        }
        written += @intCast(n);
        if (written <= seq) cap_at_prefill_end = last_cap;
    }
    // The prefill never grew.
    try t.expect(cap_at_prefill_end >= reserved);
    try t.expectEqual(total, cache.entries[0].offset);
    try t.expect(allocs >= 2);
    try t.expectEqual(KVCache.nextCapacityPolicy(cap_at_prefill_end, cap_at_prefill_end + 1), cache.nextCapacityReserved(cap_at_prefill_end, cap_at_prefill_end + 1));
    try t.expect(last_cap >= total);
}

test "a reserved KV cache grows ONCE: no old+new buffer coexists during a long prefill" {
    const t = std.testing;
    // Walking a 458,832-token prompt at chunk 4096 under the proportional policy is a ladder
    // of grows, each holding two copies inside the chunk's lazy graph.
    const seq: usize = 458_832;
    const chunk: usize = 4096;
    var cap: usize = 0;
    var pos: usize = 0;
    var grows: usize = 0;
    while (pos < seq) {
        const end = @min(pos + chunk, seq);
        if (end > cap) {
            cap = KVCache.nextCapacityPolicy(cap, end);
            grows += 1;
        }
        pos = end;
    }
    try t.expect(grows > 20);

    // Reserved: one allocation, driven through the real write path. ctx 0 = no context named.
    const reserved: usize = @intCast(KVCache.reservedTokens(seq, 2048, chunk, 0));
    try t.expectEqual(seq + 2048 + chunk, reserved);
    var rcache = try KVCache.init(t.allocator, 1);
    defer rcache.deinit();
    rcache.reserve(reserved);
    var rcap: usize = 0;
    var rpos: usize = 0;
    var rgrows: usize = 0;
    while (rpos < seq) {
        const end = @min(rpos + chunk, seq);
        if (end > rcap) {
            rcap = rcache.nextCapacityReserved(rcap, end);
            rgrows += 1;
        }
        rpos = end;
    }
    try t.expectEqual(@as(usize, 1), rgrows);
    try t.expect(rcap >= reserved);

    const s_gpu = mlx.gpuStream();
    const walk: usize = 33_000; // past RESERVE_MIN_TOKENS
    const step: usize = 4096;
    const live_reserve: usize = @intCast(KVCache.reservedTokens(walk, 512, step, 0));
    var live = try KVCache.init(t.allocator, 1);
    defer live.deinit();
    live.reserve(live_reserve);
    var written: usize = 0;
    var caps: usize = 0;
    var last_cap: usize = 0;
    while (written < walk) {
        const n: c_int = @intCast(@min(step, walk - written));
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        const count: f64 = @floatFromInt(n * 8);
        try mlx.check(mlx.mlx_arange(&flat, 0.0, count, 1.0, .float32, s_gpu));
        var k = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k);
        const shape = [_]c_int{ 1, 1, n, 8 };
        try mlx.check(mlx.mlx_reshape(&k, flat, &shape, 4, s_gpu));
        var view = try live.update(0, k, k, s_gpu, 0);
        view.deinit();
        const cap_now: usize = KVCache.bufferCapacity(live.entries[0].keys);
        if (cap_now != last_cap) {
            caps += 1;
            last_cap = cap_now;
        }
        written += @intCast(n);
    }
    try t.expectEqual(@as(usize, 1), caps);
    try t.expect(last_cap >= live_reserve);
    try t.expectEqual(walk, live.entries[0].offset);
    try t.expect(rcap >= seq + 2048);

    try t.expectEqual(@as(u64, 0), KVCache.reservedTokens(4096, 2048, chunk, 0));
    var small = try KVCache.init(t.allocator, 1);
    defer small.deinit();
    small.reserve(@intCast(KVCache.reservedTokens(4096, 2048, chunk, 0)));
    try t.expectEqual(
        KVCache.nextCapacityPolicy(1024, 2048),
        small.nextCapacityReserved(1024, 2048),
    );
}

test "the warm-credit capacity gate survives a hybrid trunk's uninitialized layers (W-1)" {
    const t = std.testing;
    // On a GDN trunk only the attention layers reach `update` (12 of 48 on qwen4_exp); a veto
    // on the first uninitialized entry made the warm credit unreachable on the arch it was for.
    const Fold = KVCache.CapacityFold;
    const layers: usize = 48;
    const interval: usize = 4; // full_attention_interval — layer 3, 7, 11, …
    const cap: usize = 787_456;

    var fold = Fold{};
    var attention_layers: usize = 0;
    for (0..layers) |i| {
        const is_attention = (i % interval) == interval - 1;
        if (is_attention) attention_layers += 1;
        fold.add(if (is_attention) cap else null);
    }
    try t.expectEqual(@as(usize, 12), attention_layers);
    try t.expectEqual(cap, fold.result());

    // The minimum still governs among the layers that hold a buffer.
    var ragged = Fold{};
    ragged.add(cap);
    ragged.add(null);
    ragged.add(cap - 4096);
    ragged.add(null);
    try t.expectEqual(cap - 4096, ragged.result());

    var cold = Fold{};
    for (0..layers) |_| cold.add(null);
    try t.expectEqual(@as(usize, 0), cold.result());
    const empty = Fold{};
    try t.expectEqual(@as(usize, 0), empty.result());

    var dense = Fold{};
    for (0..32) |_| dense.add(cap);
    try t.expectEqual(cap, dense.result());
}

test "offset rollback preserves dense affine growth and cold initialization" {
    try mlx.requireGpu();
    inline for (.{ KVQuantConfig.dense, KVQuantConfig.affine(8) }) |config| {
        try verifyRollbackKvCase(config, 8, false);
        try verifyRollbackKvCase(config, 8, true);
        try verifyRollbackKvCase(config, 0, false);
    }
}

test "upstream bugfix: KVCache affine quant keeps its dtype through growth" {
    const s = mlx.gpuStream();
    for ([_]mlx.mlx_dtype{ .float16, .bfloat16 }) |dt| {
        var cache = try KVCache.initWithConfig(testing.allocator, 1, kv_quant.KVQuantConfig.affine(8));
        defer cache.deinit();
        for (0..2) |pass| {
            const count: c_int = if (pass == 0) 4 else @intCast(KVCache.bufferCapacity(cache.entries[0].keys) + 1);
            const old_cap = if (pass == 0) 0 else KVCache.bufferCapacity(cache.entries[0].keys);
            const shape = [_]c_int{ 1, 2, count, 64 };
            var k = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(k);
            try mlx.check(mlx.mlx_ones(&k, &shape, 4, dt, s));
            var dv = try cache.update(0, k, k, s, 0);
            defer dv.deinit();
            try testing.expect(KVCache.bufferCapacity(cache.entries[0].keys) > old_cap);
            try testing.expectEqual(dt, mlx.mlx_array_dtype(dv.k));
            try testing.expectEqual(dt, mlx.mlx_array_dtype(dv.v));
            var restored = try cache.denseView(0, s);
            defer restored.deinit();
            try testing.expectEqual(dt, mlx.mlx_array_dtype(restored.k));
            try testing.expectEqual(dt, mlx.mlx_array_dtype(restored.v));
            for ([_]mlx.mlx_array{ cache.entries[0].keys_scales, cache.entries[0].keys_biases, cache.entries[0].values_scales, cache.entries[0].values_biases }) |arr| {
                try testing.expectEqual(dt, mlx.mlx_array_dtype(arr));
            }
        }
    }
}
