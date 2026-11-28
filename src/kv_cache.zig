const mlx_util = @import("mlx_util.zig");
const model_handle = @import("model_handle.zig");
const trunk = @import("trunk.zig");
const trunk_helpers_test = @import("trunk_helpers_test.zig");
const std = @import("std");
const mlx = @import("mlx.zig");
const kv_quant = @import("kv_quant.zig");
const mlx_pages = @import("mlx_pages.zig");
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
const ForwardCtx = model_handle.ForwardCtx;
const Trunk = trunk.Trunk;
const kvAppendDummyAttn = trunk_helpers_test.kvAppendDummyAttn;
const kvAppendOperandsEqual = trunk_helpers_test.kvAppendOperandsEqual;
const kvAppendPacked = trunk_helpers_test.kvAppendPacked;
const kvAppendPrefill = trunk_helpers_test.kvAppendPrefill;
const kvEntriesBytes = trunk.kvEntriesBytes;
const materializedOwnedCopy = mlx_util.materializedOwnedCopy;
const maxAbsDiffF32 = trunk_helpers_test.maxAbsDiffF32;
const ringCheckpointFeed = trunk_helpers_test.ringCheckpointFeed;
const sliceAttentionSeq = mlx_util.sliceAttentionSeq;
const swaRingChunk = trunk_helpers_test.swaRingChunk;
const verifyRollbackEvalStorage = trunk_helpers_test.verifyRollbackEvalStorage;
const verifyRollbackValidPrefixEqual = trunk_helpers_test.verifyRollbackValidPrefixEqual;

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
    keys: mlx_pages.InPlace,
    values: mlx_pages.InPlace,
    keys_scales: mlx_pages.InPlace,
    keys_biases: mlx_pages.InPlace,
    values_scales: mlx_pages.InPlace,
    values_biases: mlx_pages.InPlace,

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

    /// The dense pair is not built yet (`KVCache.updatePacked`): `dense` builds it.
    deferred: bool = false,

    /// Builds the dense pair of a deferred view; a view that has it is left as it is. MLX would skip an unread dequant;
    /// the engine runs every op it is given, so a reader that needs the pair asks for it here.
    pub fn dense(self: *DenseKVView, s: mlx.mlx_stream) !void {
        if (!self.deferred) return;
        const k = try kv_quant.dequantizeAffine(s, self.k_triple_q, self.k_triple_scales, self.k_triple_biases, self.group_size, self.bits);
        errdefer _ = mlx.mlx_array_free(k);
        const v = try kv_quant.dequantizeAffine(s, self.v_triple_q, self.v_triple_scales, self.v_triple_biases, self.group_size, self.bits);
        self.k = k;
        self.v = v;
        self.owned = true;
        self.deferred = false;
    }

    /// The dense K's shape `[B, heads, len, dim]`, built or not.
    pub fn kShape(self: *const DenseKVView) [4]c_int {
        return if (self.deferred) denseShapeOf(self.k_triple_q, self.k_triple_scales, self.group_size) else mlx.getShape(self.k)[0..4].*;
    }

    pub fn vShape(self: *const DenseKVView) [4]c_int {
        return if (self.deferred) denseShapeOf(self.v_triple_q, self.v_triple_scales, self.group_size) else mlx.getShape(self.v)[0..4].*;
    }

    fn denseShapeOf(q: mlx.mlx_array, scales: mlx.mlx_array, group_size: u32) [4]c_int {
        const qs = mlx.getShape(q);
        return .{ qs[0], qs[1], qs[2], mlx.getShape(scales)[3] * @as(c_int, @intCast(group_size)) };
    }

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
        .keys = .none,
        .values = .none,
        .keys_scales = .none,
        .keys_biases = .none,
        .values_scales = .none,
        .values_biases = .none,
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
    _ = mlx.mlx_array_free(e.keys.array);
    _ = mlx.mlx_array_free(e.values.array);
    _ = mlx.mlx_array_free(e.keys_scales.array);
    _ = mlx.mlx_array_free(e.keys_biases.array);
    _ = mlx.mlx_array_free(e.values_scales.array);
    _ = mlx.mlx_array_free(e.values_biases.array);
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

/// What a request asks of a cache buffer's growth (`KVCache.heldRows`).
pub const Hold = struct {
    /// Rows reserved up front (`KVCache.reservedTokens`, `prefill_plan.reservedPrefillTokens`); 0 = none.
    reserve: u64 = 0,
    /// The request's last row (`KVCache.requestEnd`): no grow passes it. Every hold names one; a buffer no request
    /// has bound has no hold and cannot grow (`KVCache.hold`, `qwen4_qsa_state.qsaHold`).
    end: u64,

    /// The hold of a buffer whose rows each hold `ratio` tokens (a QSA bank's pooled blocks).
    pub fn per(self: Hold, ratio: u64) Hold {
        const r = @max(ratio, 1);
        return .{ .reserve = (self.reserve +| r - 1) / r, .end = (self.end +| r - 1) / r };
    }
};

/// Cache layers whose own row stores a `Room` reports (`Room.rows`): DeepSeek's backbone layers.
pub const store_layers_max = 64;

/// What a prefill chunk finds in the cache it grows (`KVCache.roomAt`), which `bills.chunkGrowthBytes` prices the growth
/// from. The default is a full cache no request has bound: the growth the policy makes from the rows stored.
pub const Room = struct {
    /// The `hold.end` of a room no request has bound.
    pub const unbound_end = std.math.maxInt(u32);

    /// KV rows the buffers hold past the rows stored; 0 = full.
    kv_free: u64 = 0,
    /// History rows (one per `indexer_compress_ratio` tokens) the buffers hold past those the stored tokens fill.
    state_free: u64 = 0,
    hold: Hold = .{ .end = unbound_end },
    /// Rows each layer's own row stores hold (an arch that keeps them beside the KV cache: DeepSeek's compressed rows),
    /// by layer; 0 where a layer has none or none is made yet. The arch's bill grows them by the rule they grow by.
    rows: [store_layers_max]u32 = @splat(0),
};

pub const KVCache = struct {
    entries: []KVCacheEntry,
    step: usize, // absolute sequence position (not affected by sliding window trimming)
    allocator: std.mem.Allocator,
    config: KVQuantConfig,
    /// Capacity, in tokens, this cache was asked to hold up front (the admission bill's
    /// `reservedCacheTokens`). A grow is not in place, so a long prefill's peak carried a
    /// second copy of ~6 GB of KV at 458k; reserving on the first grow removes the transient.
    reserve_tokens: usize = 0,
    /// The request's last row (`requestEnd`): no grow passes it (`heldRows`). 0 = no request has bound this cache yet,
    /// and it cannot grow (`grownCapacity`): a request binds it before its first write (`generate.reserveRequestCapacity`).
    end_tokens: usize = 0,
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
    pub fn ringKeep(window: u32) usize {
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
    /// installed on the Trunk on the error path, whose own `deinit`
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
                out[i].keys = src.keys.share();
                out[i].values = src.values.share();
                if (self.config.scheme != .off) {
                    out[i].keys_scales = src.keys_scales.share();
                    out[i].keys_biases = src.keys_biases.share();
                    out[i].values_scales = src.values_scales.share();
                    out[i].values_biases = src.values_biases.share();
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
        const scope = mlx.engine.enter(s);
        defer scope.end();
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        var copied: usize = 0;
        for (snap.entries) |*e| {
            if (!e.initialized or !e.ringed) continue;
            if (e.offset >= bufferCapacity(e.keys.array)) continue;
            var bufs: [6]*mlx_pages.InPlace = undefined;
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
                const owned = try trimRowsOwned(buf.array, e.offset, s);
                buf.deinit();
                buf.* = mlx_pages.InPlace.adopt(owned, .ring);
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
            dst.keys = src.keys.share();
            dst.values = src.values.share();
            if (self.config.scheme != .off) {
                dst.keys_scales = src.keys_scales.share();
                dst.keys_biases = src.keys_biases.share();
                dst.values_scales = src.values_scales.share();
                dst.values_biases = src.values_biases.share();
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
        const pairs = [_]struct { *mlx_pages.InPlace, mlx.mlx_array }{
            .{ &dst.keys, src.keys.array },                   .{ &dst.values, src.values.array },
            .{ &dst.keys_scales, src.keys_scales.array },     .{ &dst.keys_biases, src.keys_biases.array },
            .{ &dst.values_scales, src.values_scales.array }, .{ &dst.values_biases, src.values_biases.array },
        };
        const n: usize = if (self.config.scheme != .off) 6 else 2;
        for (pairs[0..n]) |p| {
            const owned = try rowsOwned(p[1], from, from + rows, s);
            p[0].deinit();
            p[0].* = mlx_pages.InPlace.adopt(owned, .ring);
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

    /// The rows a cache buffer's capacity moves in.
    pub const chunk_step = 256;

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

    /// PURE: the rows a growing cache buffer holds once it grows from `cur_cap` to take `needed`: the +25% policy,
    /// raised to the reservation and capped at the request's end, on MLX's page grid for buffers of `parts` bytes a row
    /// (none: any rows). The allocators grow to it and the bills price it at the request's end, so from any start,
    /// restored or cold, a cache holds at most `heldRows(0, asked.end, asked, parts)`.
    pub fn heldRows(cur_cap: u64, needed: u64, asked: Hold, parts: []const u64) u64 {
        std.debug.assert(asked.end > 0);
        var rows: u64 = nextCapacityPolicy(@intCast(cur_cap), @intCast(needed));
        if (asked.reserve > rows) rows = roundToStep(asked.reserve);
        rows = @min(rows, roundToStep(@max(asked.end, asked.reserve)));
        rows = @max(rows, roundToStep(needed));
        return if (parts.len == 0) rows else mlx_pages.pagedRows(rows, parts);
    }

    fn roundToStep(rows: u64) u64 {
        return (rows +| chunk_step - 1) / chunk_step * chunk_step;
    }

    /// The request's last row: its prompt, the reply the context leaves it (at least one token, as
    /// `request_policy.clampMaxTokens` grants) and the rows a speculative round writes past it.
    pub fn requestEnd(seq: u64, max_tokens: u64, ctx: u64, past_end: u64) u64 {
        const reply = if (ctx > seq) @min(max_tokens, ctx - seq) else @min(max_tokens, 1);
        return seq +| reply +| past_end;
    }

    /// What a request asks of a cache's growth: its reservation (`reservedTokens`) and its last row (`requestEnd`);
    /// null until a request binds it (`bound`).
    pub fn hold(self: *const KVCache) ?Hold {
        if (self.end_tokens == 0) return null;
        return .{ .reserve = self.reserve_tokens, .end = self.end_tokens };
    }

    /// Bound every later grow at the request's last row. A cache outlives its request in the hot cache: a new
    /// request sets its own.
    pub fn bound(self: *KVCache, end: usize) void {
        self.end_tokens = end;
    }

    /// Tokens of capacity a request reserves: prompt + allowed generation + one chunk of slack.
    /// The one definition; `server.reservedCacheTokens` delegates here.
    pub fn reservedTokens(seq: u64, max_tokens: u64, chunk: u64, ctx: u64) u64 {
        // Short prompts keep the proportional policy: their whole cache is still billed (`bills.cacheCapacityTokens`),
        // but allocating `max_tokens` of KV up front costs every short request it.
        if (seq < RESERVE_MIN_TOKENS) return 0;
        // Every served request carries a concrete `max_tokens` and context (an omitted one is the engine default,
        // 32768): the reservation is the request's whole cache, so its decode never grows it.
        const generation: u64 = if (ctx > seq) @min(max_tokens, ctx - seq) else 0;
        return seq +| generation +| @max(chunk, 1);
    }

    /// Prompt length at or above which a request reserves its capacity; below it the flat
    /// runtime floor already covers the transient.
    pub const RESERVE_MIN_TOKENS: u64 = 32768;

    /// How many times a KV capacity buffer actually grew (the moments a second copy of the whole cache exists).
    pub var kv_cap_buf_grows: usize = 0;

    /// Ask this cache to hold `tokens` up front. Idempotent and monotone.
    pub fn reserve(self: *KVCache, tokens: usize) void {
        if (tokens > self.reserve_tokens) self.reserve_tokens = tokens;
    }

    /// Grow every restored non-ringed layer below the reservation to it, one layer per eval.
    /// Called before a warm prefill writes: left to the decode, the first step past the restored
    /// capacity grows every layer inside one forward, all the old buffers alive beside the new.
    pub fn growToReservation(self: *KVCache, s: mlx.mlx_stream) !void {
        if (self.reserve_tokens == 0) return;
        const n: usize = if (self.config.scheme == .off) 2 else 6;
        for (self.entries) |*entry| {
            if (!entry.initialized or entry.ringed) continue;
            const cur_cap = bufferCapacity(entry.keys.array);
            if (inPlaceCapacity(entry) >= self.reserve_tokens) continue;
            const new_cap: c_int = @intCast(try self.grownCapacity(entry, cur_cap, self.reserve_tokens, &entryRowBytes(entry, n), s));
            const views = [_]*mlx.mlx_array{ &entry.key_view, &entry.value_view, &entry.key_scales_view, &entry.key_biases_view, &entry.value_scales_view, &entry.value_biases_view };
            for (views) |view| {
                _ = mlx.mlx_array_free(view.*);
                view.* = mlx.mlx_array_new();
            }
            const bufs = [_]*mlx_pages.InPlace{ &entry.keys, &entry.values, &entry.keys_scales, &entry.keys_biases, &entry.values_scales, &entry.values_biases };
            const vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(vec);
            for (bufs[0..n]) |buf| {
                const sh = mlx.getShape(buf.array);
                try growQuantBuf(s, buf, true, false, entry.offset, new_cap, sh[0], sh[1], sh[3], mlx.mlx_array_dtype(buf.array));
                _ = mlx.mlx_vector_array_append_value(vec, buf.array);
            }
            // The grown buffers are this cache's own; a shared restore's rows stay with the entry.
            entry.shared_view = false;
            kv_cap_buf_grows += 1;
            try mlx.check(mlx.mlx_eval(vec));
            _ = mlx.mlx_clear_cache();
        }
    }

    /// Cut every owned, non-ringed layer to the rows it holds (capacity == rows), for a cache about to be committed to the hot
    /// cache: a request reserves its last row (`reservedTokens`, `prefill_plan.reservedPrefillTokens`), and a hot-cache entry bills
    /// the capacity of its buffers (`prefix_cache.snapshotBytes`). A buffer at a time, each evaluated and its old copy released
    /// before the next, so the copy beside the cache is the widest buffer's rows (`bills.commitTrimBytes`). A shared restore's
    /// buffers are the entry's, not this cache's to cut.
    pub fn shrinkToRows(self: *KVCache, s: mlx.mlx_stream) !void {
        const n: usize = if (self.config.scheme == .off) 2 else 6;
        for (self.entries) |*entry| {
            if (!entry.initialized or entry.ringed or entry.shared_view or entry.offset == 0) continue;
            if (bufferCapacity(entry.keys.array) <= entry.offset) continue;
            const views = [_]*mlx.mlx_array{ &entry.key_view, &entry.value_view, &entry.key_scales_view, &entry.key_biases_view, &entry.value_scales_view, &entry.value_biases_view };
            for (views) |view| {
                _ = mlx.mlx_array_free(view.*);
                view.* = mlx.mlx_array_new();
            }
            const bufs = [_]*mlx_pages.InPlace{ &entry.keys, &entry.values, &entry.keys_scales, &entry.keys_biases, &entry.values_scales, &entry.values_biases };
            for (bufs[0..n]) |buf| {
                try cutBuf(s, buf, entry.offset);
                try mlx.check(mlx.mlx_array_eval(buf.array));
                try mlx_pages.settleGrown(s);
            }
        }
    }

    /// Replace `buf` with a copy of its first `rows` rows; the slice that names them is released on return, with the old buffer.
    fn cutBuf(s: mlx.mlx_stream, buf: *mlx_pages.InPlace, rows: usize) !void {
        const sh = mlx.getShape(buf.array);
        const start = [_]c_int{ 0, 0, 0, 0 };
        const stop = [_]c_int{ sh[0], sh[1], @intCast(rows), sh[3] };
        const strides = [_]c_int{ 1, 1, 1, 1 };
        var kept = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(kept);
        try mlx.check(mlx.mlx_slice(&kept, buf.array, &start, 4, &stop, 4, &strides, 4, s));
        const cut = try mlx_pages.cutBuffer(kept, s);
        buf.deinit();
        buf.* = cut;
    }

    /// Grow every initialized, unshared, non-ringed layer whose next `rows` would pass its capacity, one layer per
    /// eval, before a forward builds. Left to the forward, a decode step past the capacity grew every layer inside its
    /// lazy graph, each old buffer alive beside its new one until the step ran: the whole cache twice, unbilled.
    pub fn growAhead(self: *KVCache, rows: usize, s: mlx.mlx_stream) !void {
        const n: usize = if (self.config.scheme == .off) 2 else 6;
        for (self.entries) |*entry| {
            if (!entry.initialized or entry.ringed or entry.shared_view) continue;
            const cur_cap = bufferCapacity(entry.keys.array);
            const needed = entry.offset + rows;
            if (needed <= inPlaceCapacity(entry)) continue;
            const new_cap: c_int = @intCast(try self.grownCapacity(entry, cur_cap, needed, &entryRowBytes(entry, n), s));
            const views = [_]*mlx.mlx_array{ &entry.key_view, &entry.value_view, &entry.key_scales_view, &entry.key_biases_view, &entry.value_scales_view, &entry.value_biases_view };
            for (views) |view| {
                _ = mlx.mlx_array_free(view.*);
                view.* = mlx.mlx_array_new();
            }
            const bufs = [_]*mlx_pages.InPlace{ &entry.keys, &entry.values, &entry.keys_scales, &entry.keys_biases, &entry.values_scales, &entry.values_biases };
            // A buffer at a time, each evaluated and its old copy released before the next: kept as free buffers the
            // old copies held the cache twice until a trim.
            for (bufs[0..n]) |buf| {
                const sh = mlx.getShape(buf.array);
                try growQuantBuf(s, buf, true, false, entry.offset, new_cap, sh[0], sh[1], sh[3], mlx.mlx_array_dtype(buf.array));
                try mlx.check(mlx.mlx_array_eval(buf.array));
                try mlx_pages.settleGrown(s);
            }
            kv_cap_buf_grows += 1;
        }
    }

    /// `update` for a reader of the quant triples: a quantized view's dense pair is built only when the reader asks
    /// (`DenseKVView.dense`).
    pub fn updatePacked(self: *KVCache, layer: u32, new_k: mlx.mlx_array, new_v: mlx.mlx_array, s: mlx.mlx_stream, max_seq: u32) !DenseKVView {
        switch (self.config.scheme) {
            .off => return self.updateDense(layer, new_k, new_v, s, max_seq),
            .affine => return self.updateAffine(layer, new_k, new_v, s, max_seq, true),
        }
    }

    pub fn update(self: *KVCache, layer: u32, new_k: mlx.mlx_array, new_v: mlx.mlx_array, s: mlx.mlx_stream, max_seq: u32) !DenseKVView {
        switch (self.config.scheme) {
            .off => return self.updateDense(layer, new_k, new_v, s, max_seq),
            .affine => return self.updateAffine(layer, new_k, new_v, s, max_seq, false),
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

    fn updateAffine(self: *KVCache, layer: u32, new_k: mlx.mlx_array, new_v: mlx.mlx_array, s: mlx.mlx_stream, max_seq: u32, deferred: bool) !DenseKVView {
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
        return self.appendQuantizedMode(layer, new_kq, new_vq, s, max_seq, deferred);
    }

    /// `update` for rows the caller already quantized with this cache's scheme.
    pub fn appendQuantized(self: *KVCache, layer: u32, new_kq: kv_quant.QuantizedKV, new_vq: kv_quant.QuantizedKV, s: mlx.mlx_stream, max_seq: u32) !DenseKVView {
        return self.appendQuantizedMode(layer, new_kq, new_vq, s, max_seq, false);
    }

    fn appendQuantizedMode(self: *KVCache, layer: u32, new_kq: kv_quant.QuantizedKV, new_vq: kv_quant.QuantizedKV, s: mlx.mlx_stream, max_seq: u32, deferred: bool) !DenseKVView {
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
        //    actually reads `.k`/`.v`; the engine is not, so a reader of the
        //    triples defers the pair (`updatePacked`). (Engagement bug class: the decode
        //    forward calls update(), not denseView() — omitting the triples
        //    here made the packed arms a silent no-op.)
        if (deferred) return .{
            .k = .{ .ctx = null },
            .v = .{ .ctx = null },
            .owned = false,
            .deferred = true,
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
        const bufs = [6]mlx.mlx_array{ entry.keys.array, entry.keys_scales.array, entry.keys_biases.array, entry.values.array, entry.values_scales.array, entry.values_biases.array };
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
        const will_grow = !entry.initialized or shared or entry.offset + new_len > inPlaceCapacity(entry);
        if (entry.initialized and entry.shared_view and !will_grow) try cowAffineBuffers(s, entry);
        entry.shared_view = false;

        // 4. Grow buffers if needed (6 of them, in lockstep on the seq axis).
        if (will_grow) {
            const needed = entry.offset + new_len;
            const cur_cap = if (entry.initialized and !shared) bufferCapacity(entry.keys.array) else 0;
            const ksc_dt = mlx.mlx_array_dtype(new_kq.scales);
            const vsc_dt = mlx.mlx_array_dtype(new_vq.scales);
            const parts = [_]u64{
                rowBytes(B, heads, q_last, .uint32),   rowBytes(B, heads, vq_last, .uint32),
                rowBytes(B, heads, sc_last, ksc_dt),   rowBytes(B, heads, vsc_last, vsc_dt),
            };
            const new_cap: c_int = @intCast(try self.grownCapacity(entry, cur_cap, needed, &parts, s));
            kv_cap_buf_grows += 1;
            try growQuantBuf(s, &entry.keys, entry.initialized, entry.ringed, entry.offset, new_cap, B, heads, q_last, .uint32);
            try growQuantBuf(s, &entry.values, entry.initialized, entry.ringed, entry.offset, new_cap, B, heads, vq_last, .uint32);
            // Quantizer scales retain the activation dtype through allocation and growth.
            try growQuantBuf(s, &entry.keys_scales, entry.initialized, entry.ringed, entry.offset, new_cap, B, heads, sc_last, ksc_dt);
            try growQuantBuf(s, &entry.keys_biases, entry.initialized, entry.ringed, entry.offset, new_cap, B, heads, sc_last, ksc_dt);
            try growQuantBuf(s, &entry.values_scales, entry.initialized, entry.ringed, entry.offset, new_cap, B, heads, vsc_last, vsc_dt);
            try growQuantBuf(s, &entry.values_biases, entry.initialized, entry.ringed, entry.offset, new_cap, B, heads, vsc_last, vsc_dt);
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
        try buildSliceView(s, &entry.key_view, entry.keys.array, total, view_start);
        try buildSliceView(s, &entry.value_view, entry.values.array, total, view_start);
        try buildSliceView(s, &entry.key_scales_view, entry.keys_scales.array, total, view_start);
        try buildSliceView(s, &entry.key_biases_view, entry.keys_biases.array, total, view_start);
        try buildSliceView(s, &entry.value_scales_view, entry.values_scales.array, total, view_start);
        try buildSliceView(s, &entry.value_biases_view, entry.values_biases.array, total, view_start);
        return self.ringCompact(layer, s);
    }

    /// Does a shared restore's first append copy its rows into a buffer sized for THIS request? Yes
    /// when the donor's buffer is longer than the request reserves (or its rows, unreserved): the
    /// admission bills the copy at that, never at the donor's capacity. A ringed layer keeps its
    /// ring; a donated checkout owns its buffers (`adoptRestored`) and appends in place.
    fn copiesSharedRestore(self: *const KVCache, entry: *const KVCacheEntry, needed: usize) bool {
        if (!entry.initialized or !entry.shared_view or entry.ringed) return false;
        return bufferCapacity(entry.keys.array) > @max(self.reserve_tokens, needed);
    }

    /// The restored buffers are this cache's alone now (a donated checkout): append in place.
    pub fn adoptRestored(self: *KVCache) void {
        for (self.entries) |*e| e.shared_view = false;
    }

    fn cowAffineBuffers(s: mlx.mlx_stream, entry: *KVCacheEntry) !void {
        const bufs = [_]*mlx_pages.InPlace{
            &entry.keys,
            &entry.values,
            &entry.keys_scales,
            &entry.keys_biases,
            &entry.values_scales,
            &entry.values_biases,
        };
        for (bufs) |buf| {
            // A copy of the whole buffer keeps its bytes, so its record stands.
            const owned = try materializedOwnedCopy(s, buf.array);
            _ = mlx.mlx_array_free(buf.array);
            buf.* = buf.replaced(owned);
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
        const will_grow = !entry.initialized or shared or entry.offset + new_len > inPlaceCapacity(entry);
        if (entry.initialized and entry.shared_view and !will_grow) {
            const k_owned = try materializedOwnedCopy(s, entry.keys.array);
            _ = mlx.mlx_array_free(entry.keys.array);
            entry.keys = entry.keys.replaced(k_owned);
            const v_owned = try materializedOwnedCopy(s, entry.values.array);
            _ = mlx.mlx_array_free(entry.values.array);
            entry.values = entry.values.replaced(v_owned);
        }
        entry.shared_view = false;

        // 3. Grow buffer if needed
        if (will_grow) {
            const B = new_shape[0];
            const heads = new_shape[1];
            const head_dim = new_shape[3];
            const dtype = mlx.mlx_array_dtype(new_k);
            const needed = entry.offset + new_len;
            const cur_cap = if (entry.initialized and !shared) bufferCapacity(entry.keys.array) else 0;
            const parts = [_]u64{ rowBytes(B, heads, head_dim, dtype), rowBytes(B, heads, v_head_dim, dtype) };
            const new_cap: c_int = @intCast(try self.grownCapacity(entry, cur_cap, needed, &parts, s));
            kv_cap_buf_grows += 1;
            const initialized = entry.initialized;
            try growQuantBuf(s, &entry.keys, initialized, entry.ringed, entry.offset, new_cap, B, heads, head_dim, dtype);
            try growQuantBuf(s, &entry.values, initialized, entry.ringed, entry.offset, new_cap, B, heads, v_head_dim, dtype);
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
        try buildSliceView(s, &entry.key_view, entry.keys.array, total, view_start);
        try buildSliceView(s, &entry.value_view, entry.values.array, total, view_start);
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
        return self.denseViewMode(layer, s, false);
    }

    /// `denseView` for a reader of the quant triples: the dense pair is built when it asks (`DenseKVView.dense`).
    pub fn denseViewPacked(self: *KVCache, layer: u32, s: mlx.mlx_stream) !DenseKVView {
        return self.denseViewMode(layer, s, true);
    }

    fn denseViewMode(self: *KVCache, layer: u32, s: mlx.mlx_stream, deferred: bool) !DenseKVView {
        const entry = &self.entries[layer];
        switch (self.config.scheme) {
            .off => return .{ .k = entry.key_view, .v = entry.value_view, .owned = false },
            .affine => {
                if (!entry.initialized) {
                    return .{ .k = entry.key_view, .v = entry.value_view, .owned = false };
                }
                if (deferred) return .{
                    .k = .{ .ctx = null },
                    .v = .{ .ctx = null },
                    .owned = false,
                    .deferred = true,
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

    /// Rows an entry takes in place: its capacity, or only the rows it holds when one of its MLX buffers sits off the
    /// page grid (a restore's exact copy, `trimmedCopy`), where every write could copy the whole buffer: the next write
    /// grows it onto the grid (`heldRows`) once. A ring keeps its fixed rows.
    fn inPlaceCapacity(entry: *const KVCacheEntry) usize {
        const cap = bufferCapacity(entry.keys.array);
        if (entry.ringed) return cap;
        for ([_]mlx_pages.InPlace{ entry.keys, entry.values, entry.keys_scales, entry.keys_biases, entry.values_scales, entry.values_biases }) |buf| {
            if (buf.array.ctx != null and buf.offGrid()) return entry.offset;
        }
        return cap;
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
        for (self.entries) |*e| fold.add(if (e.initialized and !e.ringed) inPlaceCapacity(e) else null);
        return fold.result();
    }

    /// The room the KV buffers give a chunk that starts `kv_len` rows in: the rows they hold past those stored (the
    /// fewest across the layers, `residentCapacityTokens`) and what the request asks of their growth. The history's
    /// rows are the arch's own to read (`state_free`).
    pub fn roomAt(self: *const KVCache, kv_len: u64) Room {
        var room: Room = .{ .kv_free = @as(u64, self.residentCapacityTokens()) -| kv_len };
        if (self.hold()) |h| room.hold = h;
        return room;
    }

    /// Buffer-grow / slice-update / view-build, parameterized over the
    /// buffer's last dim and dtype: six calls per `updateAffine` (3 buffers ×
    /// K and V) and two per `updateDense`. Every caller passes the width of
    /// the operand it is writing, which is what lets K and V differ.
    /// `ring`: a ring's fixed rows, which take no page grid (`pagedCapacity`).
    fn growQuantBuf(s: mlx.mlx_stream, buf: *mlx_pages.InPlace, initialized: bool, ring: bool, offset: usize, new_cap: c_int, B: c_int, heads: c_int, last_dim: c_int, dtype: mlx.mlx_dtype) !void {
        const buf_shape = [_]c_int{ B, heads, new_cap, last_dim };
        // A cold prefill's first chunk grows each layer to the whole reservation inside its forward.
        const new_buf = if (initialized and offset > 0) blk: {
            const su_start = [_]c_int{ 0, 0, 0, 0 };
            const su_stop = [_]c_int{ B, heads, @intCast(offset), last_dim };
            const su_strides = [_]c_int{ 1, 1, 1, 1 };
            var old_data = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(old_data);
            try mlx.check(mlx.mlx_slice(&old_data, buf.array, &su_start, 4, &su_stop, 4, &su_strides, 4, s));
            break :blk try mlx_pages.grownBuffer(old_data, &buf_shape, 2, dtype, ring, s);
        } else if (ring) try mlx_pages.ringBuffer(&buf_shape, dtype, s) else try mlx_pages.inPlaceBuffer(&buf_shape, dtype, s);
        buf.deinit();
        buf.* = new_buf;
    }

    /// Replace `buf` with a fresh `cap`-row buffer holding rows `[drop, offset)`
    /// at its front. A real copy, not a slice: the point is to release the
    /// buffer the dropped rows live in.
    fn ringTailBuf(s: mlx.mlx_stream, buf: *mlx_pages.InPlace, drop: usize, offset: usize, cap: usize) !void {
        const sh = mlx.getShape(buf.array);
        if (sh.len < 4) return;
        const keep: c_int = @intCast(offset - drop);
        const buf_shape = [_]c_int{ sh[0], sh[1], @intCast(cap), sh[3] };
        var zeros = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(zeros);
        try mlx.check(mlx.mlx_zeros(&zeros, &buf_shape, 4, mlx.mlx_array_dtype(buf.array), s));
        var fresh = mlx_pages.InPlace.adopt(zeros, .ring);
        errdefer fresh.deinit();
        const strides = [_]c_int{ 1, 1, 1, 1 };
        const tail_start = [_]c_int{ 0, 0, @intCast(drop), 0 };
        const tail_stop = [_]c_int{ sh[0], sh[1], @intCast(offset), sh[3] };
        var tail = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(tail);
        try mlx.check(mlx.mlx_slice(&tail, buf.array, &tail_start, 4, &tail_stop, 4, &strides, 4, s));
        const head_start = [_]c_int{ 0, 0, 0, 0 };
        const head_stop = [_]c_int{ sh[0], sh[1], keep, sh[3] };
        var updated = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(updated);
        const next = try fresh.write(&updated, tail, &head_start, &head_stop, s);
        fresh.deinit();
        buf.deinit();
        buf.* = next;
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
        const bufs = [6]mlx.mlx_array{ entry.keys.array, entry.values.array, entry.keys_scales.array, entry.keys_biases.array, entry.values_scales.array, entry.values_biases.array };
        const vec = mlx.mlx_vector_array_new_data(&bufs, if (quant) 6 else 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_async_eval(vec));
        return true;
    }

    /// The capacity a grow takes for THIS entry (`heldRows`): on MLX on its page grid for buffers of `parts` bytes a
    /// row, so every write lands in place; the engine of our own appends in place at any size. A ringed entry holds a
    /// window, not a context: it keeps its fixed rows, floored at the ring cap so a cold decode does not grow twice.
    fn grownCapacity(self: *const KVCache, entry: *const KVCacheEntry, cur_cap: usize, needed: usize, parts: []const u64, s: mlx.mlx_stream) error{KvCacheUnbound}!usize {
        if (entry.ringed) return @max(nextCapacity(cur_cap, needed), ringCap(self.swa_ring_window));
        const asked = self.hold() orelse return error.KvCacheUnbound;
        return @intCast(heldRows(cur_cap, needed, asked, if (mlx.engine.isStream(s)) &.{} else parts));
    }

    /// One cache row's bytes in a `[B, heads, rows, last]` buffer of `dtype`.
    fn rowBytes(B: c_int, heads: c_int, last: c_int, dtype: mlx.mlx_dtype) u64 {
        return @as(u64, @intCast(B * heads * last)) * @import("tensor/core.zig").itemSize(dtype);
    }

    /// One cache row's bytes in each of an entry's buffers.
    fn entryRowBytes(entry: *const KVCacheEntry, n: usize) [6]u64 {
        const bufs = [_]mlx.mlx_array{ entry.keys.array, entry.values.array, entry.keys_scales.array, entry.keys_biases.array, entry.values_scales.array, entry.values_biases.array };
        var parts: [6]u64 = @splat(0);
        for (bufs[0..n], parts[0..n]) |buf, *part| {
            const sh = mlx.getShape(buf);
            part.* = rowBytes(sh[0], sh[1], sh[3], mlx.mlx_array_dtype(buf));
        }
        return parts;
    }

    fn writeAtOffset(s: mlx.mlx_stream, buf: *mlx_pages.InPlace, offset: usize, new_chunk: mlx.mlx_array) !void {
        const new_shape = mlx.getShape(new_chunk);
        const new_len: c_int = new_shape[2];
        const buf_shape = mlx.getShape(buf.array);
        const off: c_int = @intCast(offset);
        const off_end: c_int = off + new_len;
        const su_start = [_]c_int{ 0, 0, off, 0 };
        const su_stop = [_]c_int{ buf_shape[0], buf_shape[1], off_end, buf_shape[3] };
        var updated = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(updated);
        const next = try buf.write(&updated, new_chunk, &su_start, &su_stop, s);
        _ = mlx.mlx_array_free(buf.array);
        buf.* = next;
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
            _ = mlx.mlx_vector_array_append_value(vec, entry.keys.array);
            _ = mlx.mlx_vector_array_append_value(vec, entry.values.array);
            if (affine) {
                _ = mlx.mlx_vector_array_append_value(vec, entry.keys_scales.array);
                _ = mlx.mlx_vector_array_append_value(vec, entry.keys_biases.array);
                _ = mlx.mlx_vector_array_append_value(vec, entry.values_scales.array);
                _ = mlx.mlx_vector_array_append_value(vec, entry.values_biases.array);
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
                entry.keys.deinit();
                entry.values.deinit();
                if (self.config.scheme != .off) {
                    entry.keys_scales.deinit();
                    entry.keys_biases.deinit();
                    entry.values_scales.deinit();
                    entry.values_biases.deinit();
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
            const shape = mlx.getShape(entry.keys.array);
            if (shape.len < 4) continue;
            const seq_end: c_int = @intCast(local);
            const v_start = [_]c_int{ 0, 0, 0, 0 };
            const v_stop = [_]c_int{ shape[0], shape[1], seq_end, shape[3] };
            const v_strides = [_]c_int{ 1, 1, 1, 1 };
            try mlx.check(mlx.mlx_slice(&entry.key_view, entry.keys.array, &v_start, 4, &v_stop, 4, &v_strides, 4, s));
            const val_shape = mlx.getShape(entry.values.array);
            const val_stop = [_]c_int{ val_shape[0], val_shape[1], seq_end, val_shape[3] };
            try mlx.check(mlx.mlx_slice(&entry.value_view, entry.values.array, &v_start, 4, &val_stop, 4, &v_strides, 4, s));
            if (self.config.scheme != .off) {
                // K and V scale/bias groups are as wide as their OWN payload —
                // an MLA cache's V is narrower than its K, so the value views
                // cannot be sliced against the key scales' shape.
                const sc_shape = mlx.getShape(entry.keys_scales.array);
                const sv_stop = [_]c_int{ sc_shape[0], sc_shape[1], seq_end, sc_shape[3] };
                try mlx.check(mlx.mlx_slice(&entry.key_scales_view, entry.keys_scales.array, &v_start, 4, &sv_stop, 4, &v_strides, 4, s));
                try mlx.check(mlx.mlx_slice(&entry.key_biases_view, entry.keys_biases.array, &v_start, 4, &sv_stop, 4, &v_strides, 4, s));
                const vsc_shape = mlx.getShape(entry.values_scales.array);
                const vsv_stop = [_]c_int{ vsc_shape[0], vsc_shape[1], seq_end, vsc_shape[3] };
                try mlx.check(mlx.mlx_slice(&entry.value_scales_view, entry.values_scales.array, &v_start, 4, &vsv_stop, 4, &v_strides, 4, s));
                try mlx.check(mlx.mlx_slice(&entry.value_biases_view, entry.values_biases.array, &v_start, 4, &vsv_stop, 4, &v_strides, 4, s));
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
                const grid: mlx_pages.Grid = if (src.ringed) .ring else .paged;
                out[i].keys = mlx_pages.InPlace.adopt(try trimRowsOwned(src.keys.array, keep, s), grid);
                out[i].values = mlx_pages.InPlace.adopt(try trimRowsOwned(src.values.array, keep, s), grid);
                if (self.config.scheme != .off) {
                    out[i].keys_scales = mlx_pages.InPlace.adopt(try trimRowsOwned(src.keys_scales.array, keep, s), grid);
                    out[i].keys_biases = mlx_pages.InPlace.adopt(try trimRowsOwned(src.keys_biases.array, keep, s), grid);
                    out[i].values_scales = mlx_pages.InPlace.adopt(try trimRowsOwned(src.values_scales.array, keep, s), grid);
                    out[i].values_biases = mlx_pages.InPlace.adopt(try trimRowsOwned(src.values_biases.array, keep, s), grid);
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
                inline for (.{ e.keys.array, e.values.array, e.keys_scales.array, e.keys_biases.array, e.values_scales.array, e.values_biases.array }) |arr| {
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
    donor.*.bound(test_request_end);
    donor.reserve(reserve);
    const k = try qkvIdentityArray(s, 2, rows, 64, 0.5, .bfloat16);
    var dv = try donor.update(0, k, k, s, 0);
    dv.deinit();
    snap.* = try donor.snapshot();
    warm.* = try KVCache.initWithConfig(testing.allocator, 1, cfg);
    warm.*.bound(test_request_end);
    try warm.restore(snap);
    return k;
}

/// Give a restored cache its own copy of every buffer, as a donated checkout owns them.
fn cowAffineOrDense(cache: *KVCache, s: mlx.mlx_stream) !void {
    for (cache.entries) |*e| {
        if (!e.initialized) continue;
        const bufs = [_]*mlx_pages.InPlace{ &e.keys, &e.values, &e.keys_scales, &e.keys_biases, &e.values_scales, &e.values_biases };
        const n: usize = if (cache.config.scheme == .off) 2 else 6;
        for (bufs[0..n]) |buf| {
            const owned = try materializedOwnedCopy(s, buf.array);
            _ = mlx.mlx_array_free(buf.array);
            buf.* = buf.replaced(owned);
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
            solo[i].bound(test_request_end);
            batched[i] = try KVCache.initWithConfig(alloc, 1, cfg);
            batched[i].bound(test_request_end);
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
            solo[i].bound(test_request_end);
            batched[i] = try KVCache.initWithConfig(alloc, 1, cfg);
            batched[i].bound(test_request_end);
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
    plain.bound(test_request_end);
    var ringed = try KVCache.initWithConfig(alloc, 1, kv_cfg);
    defer ringed.deinit();
    ringed.bound(test_request_end);
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
    cache.bound(test_request_end);
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
    live.bound(test_request_end);
    live.setSwaRing(window);
    var plain = try KVCache.initWithConfig(alloc, 2, kv_cfg);
    defer plain.deinit();
    plain.bound(test_request_end);
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
    restored.bound(test_request_end);
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
    live.bound(test_request_end);
    live.setSwaRing(window);
    var twin = try KVCache.initWithConfig(alloc, 2, kv_cfg);
    defer twin.deinit();
    twin.bound(test_request_end);
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
        ref.bound(test_request_end);
        ref.setSwaRing(window);
        var scratch = try KVCache.initWithConfig(alloc, 2, kv_cfg);
        defer scratch.deinit();
        scratch.bound(test_request_end);
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
    cache.bound(test_request_end);
    var reference = try KVCache.initWithConfig(allocator, 1, config);
    defer reference.deinit();
    reference.bound(test_request_end);
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
        KVCache.bufferCapacity(cache.entries[0].keys.array) - cache.entries[0].offset + 1
    else
        4;
    const tail = try attn256RandBf16(prng.random(), &.{ 1, 1, @intCast(tail_len), 64 }, stream);
    defer _ = mlx.mlx_array_free(tail);
    var offset = prefix_len;
    var ctx = ForwardCtx{ .cache = &cache, .moe_seq_offset = &offset, .ssm_entries = null, .capture_hidden = null, .vision_embeddings = null };
    var xfm: Trunk = undefined;
    xfm.allocator = allocator;
    xfm.fwd_gen = 0;
    var rollback = try Trunk.VerifyRollback.capture(&xfm, &ctx);
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
    cache.bound(test_request_end);

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
    cache.bound(test_request_end);

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
    cache.bound(test_request_end);

    const mk = struct {
        fn f(str: mlx.mlx_stream, len: c_int, d: c_int) !mlx.mlx_array {
            const shape = [_]c_int{ 1, 2, len, d };
            var a = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(a);
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
    // a failed init left a FREED cache installed on the Trunk and its
    // owner's deinit freed every handle a second time — EXC_BAD_ACCESS in
    // `mlx_array_free`, several frames from the refusal that caused it. The
    // refusal here is the allocator's (the entries alloc of the second init);
    // under the testing allocator the pre-fix order trips the same double
    // free on this test's own `defer`.
    const s = mlx.gpuStream();
    var failing = std.testing.FailingAllocator.init(testing.allocator, .{ .fail_index = 1 });
    var cache = try KVCache.initWithConfig(failing.allocator(), 2, kv_quant.KVQuantConfig.affine(8));
    defer cache.deinit();
    cache.bound(test_request_end);

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
    cache.bound(test_request_end);

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
    cache.bound(test_request_end);

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

test "KVCache updatePacked on the engine builds no dense pair until asked, then update's shapes and bytes" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("test_support.zig").restoreEngine(was);
    const s = mlx.engine.stream();
    var eager = try KVCache.initWithConfig(testing.allocator, 1, kv_quant.KVQuantConfig.affine(8));
    defer eager.deinit();
    eager.bound(test_request_end);
    var deferred = try KVCache.initWithConfig(testing.allocator, 1, kv_quant.KVQuantConfig.affine(8));
    defer deferred.deinit();
    deferred.bound(test_request_end);
    const k = try qkvIdentityArray(s, 2, 300, 256, 0.5, .bfloat16);
    defer _ = mlx.mlx_array_free(k);
    const v = try qkvIdentityArray(s, 2, 300, 256, 0.25, .bfloat16);
    defer _ = mlx.mlx_array_free(v);
    const at_eager = mlx.engine.stats().dispatches;
    var whole = try eager.update(0, k, v, s, 0);
    defer whole.deinit();
    const eager_dispatches = mlx.engine.stats().dispatches - at_eager;
    const at_packed = mlx.engine.stats().dispatches;
    var view = try deferred.updatePacked(0, k, v, s, 0);
    defer view.deinit();
    try testing.expect(mlx.engine.stats().dispatches - at_packed < eager_dispatches);
    try testing.expect(view.k.ctx == null and view.v.ctx == null and view.has_quant_triple);
    try testing.expectEqualSlices(c_int, mlx.getShape(whole.k), &view.kShape());
    try testing.expectEqualSlices(c_int, mlx.getShape(whole.v), &view.vShape());
    try view.dense(s);
    for ([_][2]mlx.mlx_array{ .{ whole.k, view.k }, .{ whole.v, view.v } }) |pair| {
        try mlx.check(mlx.mlx_array_eval(pair[0]));
        try mlx.check(mlx.mlx_array_eval(pair[1]));
        const n = mlx.mlx_array_size(pair[0]) * mlx.mlx_array_itemsize(pair[0]);
        try testing.expectEqualSlices(u8, mlx.mlx_array_data_uint8(pair[0]).?[0..n], mlx.mlx_array_data_uint8(pair[1]).?[0..n]);
    }
}

test "KVCache step resets on truncate" {
    const s = mlx.gpuStream();
    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    cache.bound(test_request_end);

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
    cache.bound(test_request_end);

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
    cache.bound(test_request_end);

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
    cache.bound(test_request_end);

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
    cache.bound(test_request_end);

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
        try testing.expect(KVCache.bufferCapacity(cache.entries[0].keys.array) <= 512);
        try testing.expect(KVCache.bufferCapacity(snap.entries[0].keys.array) >= 8192);
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
        try testing.expectEqual(@as(usize, 16384), KVCache.bufferCapacity(donor.entries[0].keys.array));
        warm.reserve(2048);
        warm.bound(2048);
        const tail = try sliceAttentionSeq(s, k, 0, 8);
        defer _ = mlx.mlx_array_free(tail);
        var wv = try warm.update(0, tail, tail, s, 0);
        defer wv.deinit();
        const want = KVCache.heldRows(0, 1024 + 8, warm.hold().?, &.{});
        try testing.expect(want < 16384);
        try testing.expectEqual(want, KVCache.bufferCapacity(warm.entries[0].keys.array));
        try testing.expectEqual(want, KVCache.bufferCapacity(warm.entries[0].values.array));
        try testing.expectEqual(@as(usize, 1024 + 8), warm.entries[0].offset);
        if (cfg.scheme == .off) {
            const head = try sliceAttentionSeq(s, wv.k, 0, 1024);
            defer _ = mlx.mlx_array_free(head);
            try testing.expectEqual(@as(f32, 0), try maxAbsDiffF32(head, k, s));
        }
        // The donor's rows and buffer are untouched.
        try testing.expectEqual(@as(usize, 1024), donor.entries[0].offset);
        try testing.expectEqual(@as(usize, 16384), KVCache.bufferCapacity(donor.entries[0].keys.array));
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
        donor.bound(test_request_end);
        snap.releaseHandles();
        warm.adoptRestored();
        warm.reserve(2048);
        warm.bound(2048);
        const tail = try sliceAttentionSeq(s, k, 0, 8);
        defer _ = mlx.mlx_array_free(tail);
        var wv = try warm.update(0, tail, tail, s, 0);
        defer wv.deinit();
        try testing.expectEqual(@as(usize, 16384), KVCache.bufferCapacity(warm.entries[0].keys.array));
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
            lazy.bound(test_request_end);
            try lazy.restore(&snap);
            if (donate) {
                // Each cache owns its buffers, as a donated checkout does.
                donor.deinit();
                donor = try KVCache.initWithConfig(testing.allocator, 1, cfg);
                donor.bound(test_request_end);
                snap.releaseHandles();
                eager.adoptRestored();
                try cowAffineOrDense(&lazy, s);
            }
            eager.reserve(2048);
            lazy.reserve(2048);
            try eager.growToReservation(s);
            try testing.expectEqual(@as(usize, 2048), KVCache.bufferCapacity(eager.entries[0].keys.array));
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
    cache.bound(test_request_end);

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
            solo[i].bound(test_request_end);
            batched[i] = try KVCache.initWithConfig(alloc, 1, cfg);
            batched[i].bound(test_request_end);
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
    src.bound(test_request_end);
    try kvAppendPrefill(&src, 6, 0x51A0, s);
    var snap = try src.snapshot();
    defer snap.deinit();
    const snap_k = try materializedOwnedCopy(s, snap.entries[0].keys.array);
    defer _ = mlx.mlx_array_free(snap_k);
    const snap_v = try materializedOwnedCopy(s, snap.entries[0].values.array);
    defer _ = mlx.mlx_array_free(snap_v);
    const snap_ks = try materializedOwnedCopy(s, snap.entries[0].keys_scales.array);
    defer _ = mlx.mlx_array_free(snap_ks);
    const snap_kb = try materializedOwnedCopy(s, snap.entries[0].keys_biases.array);
    defer _ = mlx.mlx_array_free(snap_kb);
    const snap_vs = try materializedOwnedCopy(s, snap.entries[0].values_scales.array);
    defer _ = mlx.mlx_array_free(snap_vs);
    const snap_vb = try materializedOwnedCopy(s, snap.entries[0].values_biases.array);
    defer _ = mlx.mlx_array_free(snap_vb);
    try mlx.check(mlx.mlx_array_eval(snap_k));
    var a = try KVCache.initWithConfig(alloc, 1, cfg);
    defer a.deinit();
    a.bound(test_request_end);
    var b = try KVCache.initWithConfig(alloc, 1, cfg);
    defer b.deinit();
    b.bound(test_request_end);
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
    try testing.expect(try qsaArraysAllEqual(snap.entries[0].keys.array, snap_k, s));
    try testing.expect(try qsaArraysAllEqual(snap.entries[0].values.array, snap_v, s));
    try testing.expect(try qsaArraysAllEqual(snap.entries[0].keys_scales.array, snap_ks, s));
    try testing.expect(try qsaArraysAllEqual(snap.entries[0].keys_biases.array, snap_kb, s));
    try testing.expect(try qsaArraysAllEqual(snap.entries[0].values_scales.array, snap_vs, s));
    try testing.expect(try qsaArraysAllEqual(snap.entries[0].values_biases.array, snap_vb, s));
    const k1 = try KVCache.sliceStackedRow(s, stacked_k, 0);
    defer _ = mlx.mlx_array_free(k1);
    const v1 = try KVCache.sliceStackedRow(s, stacked_v, 0);
    defer _ = mlx.mlx_array_free(v1);
    var dv = try a.update(0, k1, v1, s, 0);
    dv.deinit();
    try testing.expect(try qsaArraysAllEqual(snap.entries[0].keys.array, snap_k, s));
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
        for (&solo) |*c| c.bound(test_request_end);
        for (&batched) |*c| c.bound(test_request_end);
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
            try testing.expect(try qsaArraysAllEqual(solo[i].entries[0].keys.array, batched[i].entries[0].keys.array, s));
            try testing.expect(try qsaArraysAllEqual(solo[i].entries[0].values.array, batched[i].entries[0].values.array, s));
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
    cache.bound(test_request_end);
    try testing.expectEqual(@as(u64, 0), cache.residentBytes());
    const k = testKVWide(3, 64, s);
    defer _ = mlx.mlx_array_free(k);
    var dv = try cache.update(0, k, k, s, 0);
    dv.deinit();
    const e = cache.entries[0];
    const want = 2 * @as(u64, mlx.mlx_array_size(e.keys.array)) * @as(u64, mlx.mlx_array_itemsize(e.keys.array));
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
    const row: u64 = 4 * @as(u64, mlx.mlx_array_itemsize(e.keys.array));
    try testing.expectEqual(2 * @as(u64, @intCast(KVCache.bufferCapacity(e.keys.array))) * row, c.residentBytes());
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
    live.bound(test_request_end);
    live.setSwaRing(window);
    var plain = try KVCache.init(alloc, 2);
    defer plain.deinit();
    plain.bound(test_request_end);
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
    restored.bound(test_request_end);
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
    live.bound(test_request_end);
    live.setSwaRing(window);
    var plain = try KVCache.init(alloc, 2);
    defer plain.deinit();
    plain.bound(test_request_end);
    try ringCheckpointFeed(&live, &plain, window, 0, prompt, 250);
    var cp = (try live.ringCheckpoint(prompt, s)) orelse return error.TestExpectedRingCheckpoint;
    defer cp.deinit();
    var entry = try live.snapshotRetained(s);
    defer entry.deinit();

    // The next turn restores 20 rows below the checkpoint and forwards a 10-token tail: the
    // ring then holds fewer than window + backoff rows below the new prompt end.
    var restored = try KVCache.init(alloc, 2);
    defer restored.deinit();
    restored.bound(test_request_end);
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
    short.bound(test_request_end);
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
    cache.bound(test_request_end);

    _ = mlx.mlx_clear_cache();

    var k = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k);
    var v = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v);
    try mlx.check(mlx.mlx_zeros(&k, &shape, 4, .bfloat16, s));
    try mlx.check(mlx.mlx_zeros(&v, &shape, 4, .bfloat16, s));
    var peak_cache: usize = 0;
    var t: usize = 0;
    while (t < N) : (t += 1) {
        var layer: u32 = 0;
        while (layer < LAYERS) : (layer += 1) {
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

    var warm_arr = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_zeros(&warm_arr, &old_shape, 4, .float32, s));
    try mlx.check(mlx.mlx_array_eval(warm_arr));
    var warm = mlx_pages.InPlace.adopt(warm_arr, .paged);
    const c0 = mlx.op_count.load(.monotonic);
    try KVCache.growQuantBuf(s, &warm, false, false, 0, new_cap, B, heads, last_dim, .float32);
    const n_ops = mlx.op_count.load(.monotonic) - c0;
    try mlx.check(mlx.mlx_array_eval(warm.array));
    try testing.expectEqual(@as(c_int, new_cap), mlx.getShape(warm.array)[2]);
    warm.deinit();
    try testing.expect(n_ops >= 1);

    var fired: usize = 0;
    var k: u64 = 1;
    while (k <= n_ops) : (k += 1) {
        var made = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_zeros(&made, &old_shape, 4, .float32, s));
        try mlx.check(mlx.mlx_array_eval(made));
        var buf = mlx_pages.InPlace.adopt(made, .paged);
        mlx.fault.arm(k);
        const r = KVCache.growQuantBuf(s, &buf, false, false, 0, new_cap, B, heads, last_dim, .float32);
        const did = mlx.fault.didFire();
        mlx.fault.disarm();
        if (r) |_| {
            // Not reached for k in 1..=n_ops; free the replacement in case.
            buf.deinit();
        } else |err| {
            try testing.expectEqual(error.MlxError, err);
            try testing.expect(did);
            fired += 1;
            // This free must be legal; it is the assertion.
            buf.deinit();
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
            cache.bound(test_request_end);
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
            cache.bound(test_request_end);
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

test "a reservation holds the request's whole generation: neither its prefill nor its decode grows the cache" {
    const t = std.testing;
    const chunk: usize = 4096;
    const seq: usize = 40_000;
    const max_tokens: usize = 32_768;
    const reserved: usize = @intCast(KVCache.reservedTokens(seq, max_tokens, chunk, 1_048_576));
    try t.expectEqual(seq + max_tokens + chunk, reserved);
    // Bounded by the context's room past the prompt.
    try t.expectEqual(seq + 1000 + chunk, KVCache.reservedTokens(seq, max_tokens, chunk, seq + 1000));

    const s_gpu = mlx.gpuStream();
    var cache = try KVCache.init(t.allocator, 1);
    defer cache.deinit();
    cache.bound(test_request_end);
    cache.reserve(reserved);
    var written: usize = 0;
    var allocs: usize = 0;
    var last_cap: usize = 0;
    const total: usize = seq + max_tokens;
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
        const cap_now: usize = KVCache.bufferCapacity(cache.entries[0].keys.array);
        if (cap_now != last_cap) {
            allocs += 1;
            last_cap = cap_now;
        }
        written += @intCast(n);
    }
    // One allocation, at the reservation, and no grow after it.
    try t.expectEqual(@as(usize, 1), allocs);
    try t.expectEqual(total, cache.entries[0].offset);
    try t.expect(last_cap >= reserved);
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

    // Reserved: one allocation, driven through the real write path, in a context with room for the reply.
    const reserved: usize = @intCast(KVCache.reservedTokens(seq, 2048, chunk, 1 << 20));
    try t.expectEqual(seq + 2048 + chunk, reserved);
    var rcache = try KVCache.init(t.allocator, 1);
    defer rcache.deinit();
    rcache.bound(test_request_end);
    rcache.reserve(reserved);
    rcache.bound(reserved);
    var rcap: usize = 0;
    var rpos: usize = 0;
    var rgrows: usize = 0;
    while (rpos < seq) {
        const end = @min(rpos + chunk, seq);
        if (end > rcap) {
            rcap = KVCache.heldRows(rcap, end, rcache.hold().?, &.{});
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
    live.bound(test_request_end);
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
        const cap_now: usize = KVCache.bufferCapacity(live.entries[0].keys.array);
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
    small.bound(test_request_end);
    small.reserve(@intCast(KVCache.reservedTokens(4096, 2048, chunk, 0)));
    small.bound(@intCast(KVCache.requestEnd(4096, 2048, 4096 + 2048, 0)));
    try t.expectEqual(
        KVCache.nextCapacityPolicy(1024, 2048),
        KVCache.heldRows(1024, 2048, small.hold().?, &.{}),
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
        cache.bound(test_request_end);
        for (0..2) |pass| {
            const count: c_int = if (pass == 0) 4 else @intCast(KVCache.bufferCapacity(cache.entries[0].keys.array) + 1);
            const old_cap = if (pass == 0) 0 else KVCache.bufferCapacity(cache.entries[0].keys.array);
            const shape = [_]c_int{ 1, 2, count, 64 };
            var k = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(k);
            try mlx.check(mlx.mlx_ones(&k, &shape, 4, dt, s));
            var dv = try cache.update(0, k, k, s, 0);
            defer dv.deinit();
            try testing.expect(KVCache.bufferCapacity(cache.entries[0].keys.array) > old_cap);
            try testing.expectEqual(dt, mlx.mlx_array_dtype(dv.k));
            try testing.expectEqual(dt, mlx.mlx_array_dtype(dv.v));
            var restored = try cache.denseView(0, s);
            defer restored.deinit();
            try testing.expectEqual(dt, mlx.mlx_array_dtype(restored.k));
            try testing.expectEqual(dt, mlx.mlx_array_dtype(restored.v));
            for ([_]mlx.mlx_array{ cache.entries[0].keys_scales.array, cache.entries[0].keys_biases.array, cache.entries[0].values_scales.array, cache.entries[0].values_biases.array }) |arr| {
                try testing.expectEqual(dt, mlx.mlx_array_dtype(arr));
            }
        }
    }
}

test "KVCache on MLX appends kv8 rows in place when MLX's pool holds buffers two pages past their scale rows" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var cache = try KVCache.initWithConfig(testing.allocator, 1, KVQuantConfig.affine(8));
    defer cache.deinit();
    cache.bound(test_request_end);
    // Two heads of 256: 16 B of scales a row, so 1,280 rows are 1.25 pages, which a free 3-page buffer could hold.
    var prompt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(prompt);
    try mlx.check(mlx.mlx_ones(&prompt, &[_]c_int{ 1, 2, 1100, 256 }, 4, .bfloat16, s));
    var row = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(row);
    try mlx.check(mlx.mlx_ones(&row, &[_]c_int{ 1, 2, 1, 256 }, 4, .bfloat16, s));
    try mlx.check(mlx.mlx_array_eval(prompt));
    try mlx.check(mlx.mlx_array_eval(row));
    _ = mlx.mlx_clear_cache();
    var held: [64]mlx.mlx_array = undefined;
    for (&held) |*h| {
        h.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_zeros(h, &[_]c_int{3 * 16384}, 1, .uint8, s));
        try mlx.check(mlx.mlx_array_eval(h.*));
    }
    try mlx.check(mlx.mlx_synchronize(s));
    for (held) |h| _ = mlx.mlx_array_free(h);
    var pointers: [6]?[*]const u8 = undefined;
    for ([_]mlx.mlx_array{ prompt, row }, 0..) |rows, step| {
        var view = try cache.update(0, rows, rows, s, 0);
        view.deinit();
        const e = &cache.entries[0];
        const bufs = [_]mlx.mlx_array{ e.keys.array, e.values.array, e.keys_scales.array, e.keys_biases.array, e.values_scales.array, e.values_biases.array };
        const vec = mlx.mlx_vector_array_new_data(&bufs, bufs.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_eval(vec));
        try mlx.check(mlx.mlx_synchronize(s));
        for (bufs, &pointers) |buf, *p| {
            if (step == 0) p.* = mlx.mlx_array_data_uint8(buf) else try testing.expectEqual(p.*, mlx.mlx_array_data_uint8(buf));
        }
    }
}

/// One Flash-Next KV layer (2 heads of 256, kv8) restored at exactly `restored` rows (as an SSD restore or a trimmed hot
/// entry installs it; 0: cold), reserved and bounded as the engine runs the request (`reserveRequestCapacity`), then the
/// prompt's tail and the reply appended a row at a time: the rows it holds at the end.
fn requestHeldRows(cfg: *const @import("model.zig").ModelConfig, restored: c_int, seq: c_int, reply: c_int, chunk: u64) !usize {
    const s = mlx.gpuStream();
    var donor = try KVCache.initWithConfig(testing.allocator, 1, KVQuantConfig.affine(8));
    defer donor.deinit();
    donor.bound(test_request_end);
    var rows = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rows);
    try mlx.check(mlx.mlx_ones(&rows, &[_]c_int{ 1, 2, if (restored > 0) restored else 1, 256 }, 4, .bfloat16, s));
    var warm = try KVCache.initWithConfig(testing.allocator, 1, KVQuantConfig.affine(8));
    defer warm.deinit();
    warm.bound(test_request_end);
    if (restored > 0) {
        var dv = try donor.update(0, rows, rows, s, 0);
        dv.deinit();
        var snap = try donor.snapshot();
        defer snap.deinit();
        var trimmed = try snap.trimmedCopy(@intCast(restored), null, s);
        defer trimmed.deinit();
        try warm.restore(&trimmed);
        warm.adoptRestored();
    }
    _ = try @import("generate.zig").reserveRequestCapacity(&warm, cfg, @intCast(seq), @intCast(reply), chunk, 0, s);
    var tail = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tail);
    try mlx.check(mlx.mlx_ones(&tail, &[_]c_int{ 1, 2, seq - restored, 256 }, 4, .bfloat16, s));
    var tv = try warm.update(0, tail, tail, s, 0);
    tv.deinit();
    var one = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(one);
    try mlx.check(mlx.mlx_ones(&one, &[_]c_int{ 1, 2, 1, 256 }, 4, .bfloat16, s));
    var i: c_int = 0;
    while (i < reply) : (i += 1) {
        try warm.growAhead(1, s);
        var ov = try warm.update(0, one, one, s, 0);
        ov.deinit();
    }
    return KVCache.bufferCapacity(warm.entries[0].keys.array);
}

test "a Flash-Next cache restored at its rows or cold holds no more rows than its request's bill, nor than a prompt filling its context" {
    try mlx.requireGpu();
    const bills = @import("bills.zig");
    var cfg = @import("bills_fixtures_test.zig").qwen4ExpOomConfig();
    // {context, restored, prompt, reply}: up to a chunk no reservation (`prefill_plan.reservedPrefillTokens`), so the
    // cache grows by the policy from where the restore left it; the last three end at their context.
    const cases = [_][4]c_int{
        .{ 0, 0, 1950, 50 },    .{ 0, 1900, 1950, 50 },    .{ 0, 1600, 1700, 300 }, .{ 0, 1000, 1100, 900 },
        .{ 4100, 0, 3000, 1100 }, .{ 4100, 0, 2000, 2100 }, .{ 4100, 1900, 1950, 2150 },
    };
    var over: usize = 0;
    for (cases) |c| {
        cfg.pinned_context = @intCast(c[0]);
        const ctx = bills.getEffectiveContextLength(&cfg);
        const chunk: u64 = 2048;
        const held = try requestHeldRows(&cfg, c[1], c[2], c[3], chunk);
        const billed = bills.cacheCapacityTokens(&cfg, @intCast(c[2]), @intCast(c[3]), chunk, ctx, 0, 8);
        if (held > billed or billed > bills.cacheCapacityTokens(&cfg, ctx, 0, chunk, ctx, 0, 8)) over += 1;
    }
    try testing.expectEqual(@as(usize, 0), over);
}

/// Empties MLX's pool and leaves 16 free buffers of each page count in `pages`.
fn seedPoolPages(s: mlx.mlx_stream, pages: []const c_int) !void {
    try mlx.check(mlx.mlx_synchronize(s));
    _ = mlx.mlx_clear_cache();
    var held: [64]mlx.mlx_array = undefined;
    var n: usize = 0;
    for (pages) |p| for (0..16) |_| {
        held[n] = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_zeros(&held[n], &[_]c_int{p * 16384}, 1, .uint8, s));
        try mlx.check(mlx.mlx_array_eval(held[n]));
        n += 1;
    };
    try mlx.check(mlx.mlx_synchronize(s));
    for (held[0..n]) |h| _ = mlx.mlx_array_free(h);
}

/// A 1,100-row kv8 layer (2 heads of 256) trimmed to its rows (`trimmedCopy`, as a byte-budget trim or an
/// SSD restore leaves it), restored, clamped 30 rows back, then appended a row at a time inside its capacity. How many
/// of those appends moved any of the six buffers (a copy of the whole buffer).
fn clampedRestoreMoves(donated: bool) !usize {
    const s = mlx.gpuStream();
    var donor = try KVCache.initWithConfig(testing.allocator, 1, KVQuantConfig.affine(8));
    defer donor.deinit();
    donor.bound(test_request_end);
    var rows = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rows);
    try mlx.check(mlx.mlx_ones(&rows, &[_]c_int{ 1, 2, 1100, 256 }, 4, .bfloat16, s));
    var one = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(one);
    try mlx.check(mlx.mlx_ones(&one, &[_]c_int{ 1, 2, 1, 256 }, 4, .bfloat16, s));
    try mlx.check(mlx.mlx_array_eval(one));
    var dv = try donor.update(0, rows, rows, s, 0);
    dv.deinit();
    donor.evalState();
    var snap = try donor.snapshot();
    defer snap.deinit();
    // Codes 563,200 B (35 pages), scales 17,600 B (2 pages): free buffers one page past each.
    try seedPoolPages(s, &.{ 36, 3 });
    var trimmed = try snap.trimmedCopy(1100, null, s);
    defer trimmed.deinit();
    var warm = try KVCache.initWithConfig(testing.allocator, 1, KVQuantConfig.affine(8));
    defer warm.deinit();
    warm.bound(test_request_end);
    try warm.restore(&trimmed);
    if (donated) {
        warm.adoptRestored();
        trimmed.releaseHandles();
    }
    try warm.truncate(1070, s);
    var moved: usize = 0;
    var last: [6]?[*]const u8 = @splat(null);
    for (0..6) |step| {
        var ov = try warm.update(0, one, one, s, 0);
        ov.deinit();
        warm.evalState();
        try mlx.check(mlx.mlx_synchronize(s));
        const e = &warm.entries[0];
        const bufs = [_]mlx.mlx_array{ e.keys.array, e.values.array, e.keys_scales.array, e.keys_biases.array, e.values_scales.array, e.values_biases.array };
        var any = false;
        for (bufs, &last) |b, *p| {
            const now = mlx.mlx_array_data_uint8(b);
            if (step > 0 and now != p.*) any = true;
            p.* = now;
        }
        if (any) moved += 1;
    }
    return moved;
}

test "a trimmed restore clamped below its rows appends in place when MLX's pool holds buffers a page past it" {
    try mlx.requireGpu();
    const donated = try clampedRestoreMoves(true);
    const shared = try clampedRestoreMoves(false);
    try testing.expectEqual(@as(usize, 0), donated + shared);
}

/// Rows a warm cache holds once a request restored at `restored` of a `seq`-token prompt writes its tail and decodes
/// `reply` tokens, bound as `generate` binds every request, its tail one token or more.
fn warmTailHeldRows(cfg: *const @import("model.zig").ModelConfig, restored: c_int, seq: c_int, reply: c_int, chunk: u64) !usize {
    const s = mlx.gpuStream();
    var donor = try KVCache.initWithConfig(testing.allocator, 1, KVQuantConfig.affine(8));
    defer donor.deinit();
    donor.bound(@intCast(restored));
    var rows = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rows);
    try mlx.check(mlx.mlx_ones(&rows, &[_]c_int{ 1, 2, restored, 256 }, 4, .bfloat16, s));
    var warm = try KVCache.initWithConfig(testing.allocator, 1, KVQuantConfig.affine(8));
    defer warm.deinit();
    var dv = try donor.update(0, rows, rows, s, 0);
    dv.deinit();
    var snap = try donor.snapshot();
    defer snap.deinit();
    var trimmed = try snap.trimmedCopy(@intCast(restored), null, s);
    defer trimmed.deinit();
    try warm.restore(&trimmed);
    warm.adoptRestored();
    _ = try @import("generate.zig").reserveRequestCapacity(&warm, cfg, @intCast(seq), @intCast(reply), chunk, 0, s);
    var tail = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tail);
    try mlx.check(mlx.mlx_ones(&tail, &[_]c_int{ 1, 2, seq - restored, 256 }, 4, .bfloat16, s));
    var tv = try warm.update(0, tail, tail, s, 0);
    tv.deinit();
    var one = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(one);
    try mlx.check(mlx.mlx_ones(&one, &[_]c_int{ 1, 2, 1, 256 }, 4, .bfloat16, s));
    var i: c_int = 0;
    while (i < reply) : (i += 1) {
        try warm.growAhead(1, s);
        var ov = try warm.update(0, one, one, s, 0);
        ov.deinit();
    }
    return KVCache.bufferCapacity(warm.entries[0].keys.array);
}

test "a warm Flash-Next request restored at all but its last token holds no more rows than its request's bill" {
    try mlx.requireGpu();
    const bills = @import("bills.zig");
    var cfg = @import("bills_fixtures_test.zig").qwen4ExpOomConfig();
    // {restored, prompt, reply}: a regenerate's trimmed RAM restore (off the grid, exact rows), an SSD-sized one, and a
    // 50-token tail beside them.
    const cases = [_][3]c_int{ .{ 1900, 1901, 100 }, .{ 8000, 8001, 300 }, .{ 1900, 1950, 100 } };
    for (cases) |c| {
        const ctx = bills.getEffectiveContextLength(&cfg);
        const chunk: u64 = 2048;
        const held = try warmTailHeldRows(&cfg, c[0], c[1], c[2], chunk);
        const billed = bills.cacheCapacityTokens(&cfg, @intCast(c[1]), @intCast(c[2]), chunk, ctx, 0, 8);
        if (held > billed) std.debug.print("restored {d} seq {d} reply {d}: held {d} rows, bill {d}\n", .{ c[0], c[1], c[2], held, billed });
        try testing.expect(held <= billed);
    }
}

test "a KV cache no request has bound cannot grow" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    for ([_]KVQuantConfig{ KVQuantConfig.dense, KVQuantConfig.affine(8) }) |q| {
        var cache = try KVCache.initWithConfig(testing.allocator, 1, q);
        defer cache.deinit();
        var rows = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(rows);
        try mlx.check(mlx.mlx_ones(&rows, &[_]c_int{ 1, 2, 8, 256 }, 4, .bfloat16, s));
        try testing.expect(cache.hold() == null);
        try testing.expectError(error.KvCacheUnbound, cache.update(0, rows, rows, s, 0));
        cache.bound(8);
        var v = try cache.update(0, rows, rows, s, 0);
        v.deinit();
    }
}

test "an append writes in place only into a buffer on MLX's page grid, whichever function made the buffer" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var row = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(row);
    try mlx.check(mlx.mlx_ones(&row, &[_]c_int{ 1, 1, 1, 512 }, 4, .bfloat16, s));
    // 1,000 rows of 1 KiB are 62.5 pages: off the grid, so every write into it could copy all of it.
    var made = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_zeros(&made, &[_]c_int{ 1, 1, 1000, 512 }, 4, .bfloat16, s));
    var off = mlx_pages.InPlace.adopt(made, .paged);
    defer off.deinit();
    try testing.expectError(error.InPlaceBufferOffPageGrid, KVCache.writeAtOffset(s, &off, 0, row));
    // 1,008 rows are 63 pages.
    var on = try mlx_pages.inPlaceBuffer(&.{ 1, 1, 1008, 512 }, .bfloat16, s);
    defer on.deinit();
    try KVCache.writeAtOffset(s, &on, 0, row);
    // A ring keeps its fixed rows off the grid.
    var ring_made = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_zeros(&ring_made, &[_]c_int{ 1, 1, 1000, 512 }, 4, .bfloat16, s));
    var ring = mlx_pages.InPlace.adopt(ring_made, .ring);
    defer ring.deinit();
    try KVCache.writeAtOffset(s, &ring, 0, row);
}

/// A unit test's cache stands for one request whose end lies past every row the test writes: the +25% policy alone
/// sizes its grows (`KVCache.heldRows`).
const test_request_end: usize = 1 << 20;

/// Writes every KV layer of `cfg`'s cache from its stored rows to `rows` on `s`: the layers a regrow rebuilds.
fn fillGrowingLayers(cfg: *const ModelConfig, cache: *KVCache, rows: usize, s: mlx.mlx_stream) !void {
    var li: u32 = 0;
    while (li < cfg.num_hidden_layers) : (li += 1) {
        if (!cfg.isKvPerTokenLayer(li)) continue;
        const add: c_int = @intCast(rows - cache.seqLen(li));
        if (add == 0) continue;
        var k = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k);
        var v = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v);
        try mlx.check(mlx.mlx_ones(&k, &[_]c_int{ 1, @intCast(cfg.layerKVHeads(li)), add, @intCast(cfg.layerHeadDim(li)) }, 4, .bfloat16, s));
        try mlx.check(mlx.mlx_ones(&v, &[_]c_int{ 1, @intCast(cfg.layerKVHeads(li)), add, @intCast(cfg.layerVHeadDim(li)) }, 4, .bfloat16, s));
        var view = try cache.update(li, k, v, s, 0);
        view.deinit();
    }
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    cache.appendEvalArrays(vec);
    try mlx.check(mlx.mlx_eval(vec));
}

const mlxPeakOver = test_support.mlxPeakOver;

/// One MLX page: a buffer past it is whole pages, and a step's scalars are the few bytes beside them.
const regrow_page_slack: u64 = mlx_pages.page;

test "the dearest decode regrow of a request on MLX holds beside the grown cache no more than the request's bill carries beyond it, at every prompt and reply" {
    try mlx.requireGpu();
    const bills = @import("bills.zig");
    const fixtures = @import("bills_fixtures_test.zig");
    const s = mlx.gpuStream();
    var over: usize = 0;
    var regrows: usize = 0;
    inline for (.{ .{ "qwen4", fixtures.qwen4ExpLive364kConfig }, .{ "mimo2", fixtures.mimoV2FlashBillConfig } }) |arch| for ([_]u64{ 8, 16 }) |bits| for ([_]u64{ 4096, 8192 }) |seq| for ([_]u64{ 1, 4096 }) |max_tokens| {
        if (!bills.archBuilt(arch[0])) continue;
        var cfg = arch[1]();
        const chunk: u64 = 2048;
        const terms = bills.prefillRequestTerms(&cfg, seq, max_tokens, bits, chunk, .{});
        var cache = try KVCache.initWithConfig(testing.allocator, cfg.num_hidden_layers, if (bits == 16) KVQuantConfig.dense else KVQuantConfig.affine(8));
        defer cache.deinit();
        _ = try @import("generate.zig").reserveRequestCapacity(&cache, &cfg, seq, max_tokens, chunk, 0, s);
        try fillGrowingLayers(&cfg, &cache, seq, s);
        var last: usize = cfg.num_hidden_layers;
        while (!cfg.isKvPerTokenLayer(@intCast(last - 1))) last -= 1;
        last -= 1;
        const Grow = struct {
            cache: *KVCache,
            s: mlx.mlx_stream,
            pub fn run(self: @This()) !void {
                try self.cache.growAhead(1, self.s);
            }
        };
        // The request's decode: every row past a capacity step regrows the cache, as `forwardWith` grows it ahead of the forward.
        var offset: usize = seq;
        var dearest: u64 = 0;
        var later_most: u64 = 0;
        while (offset < seq + max_tokens) {
            const old_rows = KVCache.bufferCapacity(cache.entries[last].keys.array);
            const peak = try mlxPeakOver(s, Grow, .{ .cache = &cache, .s = s });
            const new_rows = KVCache.bufferCapacity(cache.entries[last].keys.array);
            dearest = @max(dearest, peak);
            if (new_rows != old_rows) {
                regrows += 1;
                // The peak is the widest buffer's old copy and tail, less what the layer's buffers after it have not yet grown by.
                const e = &cache.entries[last];
                const bufs = [_]mlx.mlx_array{ e.keys.array, e.values.array, e.keys_scales.array, e.keys_biases.array, e.values_scales.array, e.values_biases.array };
                var row: [6]u64 = @splat(0);
                var widest: u64 = 0;
                for (bufs[0..if (bits == 16) 2 else 6], row[0..if (bits == 16) 2 else 6]) |b, *r| {
                    r.* = mlx.mlx_array_size(b) * mlx.mlx_array_itemsize(b) / new_rows;
                    widest = @max(widest, r.*);
                }
                var after_widest: u64 = 0;
                var seen = false;
                for (0..6) |i| {
                    if (seen) after_widest += row[i] * (new_rows - old_rows);
                    if (row[i] == widest) {
                        seen = true;
                        after_widest = 0;
                    }
                }
                later_most = @max(later_most, after_widest);
            }
            offset = @min(new_rows, seq + max_tokens);
            try fillGrowingLayers(&cfg, &cache, offset, s);
        }
        const billed = terms.decode_regrow_bytes;
        // A ringed arch bills its sliding rings' compaction beside, which a decode step's regrow of the global layers never meets.
        if (dearest > billed + regrow_page_slack or billed > @max(dearest + later_most + regrow_page_slack, bills.ringCompactionBytes(&cfg, bits))) {
            over += 1;
            std.debug.print("{s} kv{d}: a {d}-token prompt + {d}: the dearest regrow holds {d} B past the grown buffers, billed {d} B\n", .{ cfg.model_type, bits, seq, max_tokens, dearest, billed });
        }
    };
    try testing.expect(regrows > 0);
    try testing.expectEqual(@as(usize, 0), over);
}

test "the dearest decode regrow of the MTP head's cache on MLX holds beside the grown cache no more than the request's bill carries beyond it, at every prompt and reply" {
    try mlx.requireGpu();
    const bills = @import("bills.zig");
    if (!bills.archBuilt("qwen4")) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const cfg = @import("bills_fixtures_test.zig").qwen4ExpLive364kConfig();
    var first: u32 = 0;
    while (!cfg.isKvPerTokenLayer(first)) first += 1;
    // The head's own scheme is dense unless the kv-quant lever is on: its buffers are K and V alone, one layer of the trunk's shape.
    const kv_heads: u64 = cfg.layerKVHeads(first);
    if (bills.mtpHeadKvBytesPerToken(&cfg) != kv_heads * (cfg.layerHeadDim(first) + cfg.layerVHeadDim(first)) * 2) return error.SkipZigTest;
    const chunk: u64 = 2048;
    var over: usize = 0;
    for ([_]u64{ 4096, 8192 }) |seq| for ([_]u64{ 1, 4096 }) |max_tokens| {
        const terms = bills.prefillRequestTerms(&cfg, seq, max_tokens, 8, chunk, .{ .mtp_on = true });
        const end = KVCache.requestEnd(seq, max_tokens, bills.getEffectiveContextLength(&cfg), bills.speculativeRowsPastEnd(&cfg, .{ .mtp_on = true }));
        var cache = try KVCache.initWithConfig(testing.allocator, 1, KVQuantConfig.dense);
        defer cache.deinit();
        cache.reserve(@intCast(@import("prefill_plan.zig").reservedPrefillTokens(&cfg, seq, max_tokens, chunk, 0)));
        cache.bound(@intCast(end));
        const Fill = struct {
            cache: *KVCache,
            cfg: *const ModelConfig,
            layer: u32,
            rows: usize,
            s: mlx.mlx_stream,
            // Writes the head's layer from its stored rows to `rows`; the write grows the cache inside the graph, as the head's forward does.
            pub fn run(self: @This()) !void {
                const add: c_int = @intCast(self.rows - self.cache.seqLen(0));
                if (add == 0) return;
                var k = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(k);
                var v = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(v);
                try mlx.check(mlx.mlx_ones(&k, &[_]c_int{ 1, @intCast(self.cfg.layerKVHeads(self.layer)), add, @intCast(self.cfg.layerHeadDim(self.layer)) }, 4, .bfloat16, self.s));
                try mlx.check(mlx.mlx_ones(&v, &[_]c_int{ 1, @intCast(self.cfg.layerKVHeads(self.layer)), add, @intCast(self.cfg.layerVHeadDim(self.layer)) }, 4, .bfloat16, self.s));
                var view = try self.cache.update(0, k, v, self.s, 0);
                view.deinit();
                const vec = mlx.mlx_vector_array_new();
                defer _ = mlx.mlx_vector_array_free(vec);
                self.cache.appendEvalArrays(vec);
                try mlx.check(mlx.mlx_eval(vec));
            }
        };
        try (Fill{ .cache = &cache, .cfg = &cfg, .layer = first, .rows = seq, .s = s }).run();
        var dearest: u64 = 0;
        var offset: usize = seq;
        while (offset < end) {
            dearest = @max(dearest, try mlxPeakOver(s, Fill, .{ .cache = &cache, .cfg = &cfg, .layer = first, .rows = offset + 1, .s = s }));
            offset = @min(KVCache.bufferCapacity(cache.entries[0].keys.array), end);
            try (Fill{ .cache = &cache, .cfg = &cfg, .layer = first, .rows = offset, .s = s }).run();
        }
        // The peak is the cache's old copies and zero tails whole.
        const billed = terms.decode_regrow_bytes;
        if (dearest > billed + regrow_page_slack or billed > dearest + regrow_page_slack) {
            over += 1;
            std.debug.print("a {d}-token prompt + {d}: the head's dearest regrow holds {d} B past its grown cache, billed {d} B\n", .{ seq, max_tokens, dearest, billed });
        }
    };
    try testing.expectEqual(@as(usize, 0), over);
}

test "the compaction of a ringed arch's sliding layers in a decode step on MLX holds beside the live rings no more than the request's bill carries, every layer's in one graph" {
    try mlx.requireGpu();
    const bills = @import("bills.zig");
    if (!bills.archBuilt("mimo2")) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const cfg = @import("bills_fixtures_test.zig").mimoV2FlashBillConfig();
    const cap = @as(usize, cfg.sliding_window) + ModelConfig.SWA_RING_SLACK;
    var over: usize = 0;
    var compactions: usize = 0;
    for ([_]u64{ 8, 16 }) |bits| {
        var cache = try KVCache.initWithConfig(testing.allocator, cfg.num_hidden_layers, if (bits == 16) KVQuantConfig.dense else KVQuantConfig.affine(8));
        defer cache.deinit();
        cache.setSwaRing(cfg.sliding_window);
        cache.bound(1 << 20);
        const Step = struct {
            cache: *KVCache,
            cfg: *const ModelConfig,
            add: usize,
            s: mlx.mlx_stream,
            // Every sliding layer takes `add` rows, as one decode step's forward builds them before it is evaluated.
            pub fn run(self: @This()) !void {
                var li: u32 = 0;
                while (li < self.cfg.num_hidden_layers) : (li += 1) {
                    if (self.cfg.isGlobalLayer(li)) continue;
                    var k = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(k);
                    var v = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(v);
                    try mlx.check(mlx.mlx_ones(&k, &[_]c_int{ 1, @intCast(self.cfg.layerKVHeads(li)), @intCast(self.add), @intCast(self.cfg.layerHeadDim(li)) }, 4, .bfloat16, self.s));
                    try mlx.check(mlx.mlx_ones(&v, &[_]c_int{ 1, @intCast(self.cfg.layerKVHeads(li)), @intCast(self.add), @intCast(self.cfg.layerVHeadDim(li)) }, 4, .bfloat16, self.s));
                    var view = try self.cache.update(li, k, v, self.s, @intCast(self.cfg.sliding_window));
                    view.deinit();
                }
                const vec = mlx.mlx_vector_array_new();
                defer _ = mlx.mlx_vector_array_free(vec);
                self.cache.appendEvalArrays(vec);
                try mlx.check(mlx.mlx_eval(vec));
            }
        };
        _ = try mlxPeakOver(s, Step, .{ .cache = &cache, .cfg = &cfg, .add = cap, .s = s });
        var dearest: u64 = 0;
        var rows: usize = cap;
        while (rows < cap + 3 * ModelConfig.SWA_RING_SLACK) : (rows += 1) {
            const before = KVCache.bufferCapacity(cache.entries[1].keys.array);
            const peak = try mlxPeakOver(s, Step, .{ .cache = &cache, .cfg = &cfg, .add = 1, .s = s });
            // A step past the ring's cap compacts every sliding layer's buffers.
            if (cache.entries[1].offset < before) {
                compactions += 1;
                dearest = @max(dearest, peak);
            }
        }
        // Every layer's compaction in one graph is the worst MLX can hold: it frees a layer's old rings as its command buffers complete.
        var probe = cfg;
        probe.pinned_context = 262_144;
        const billed = bills.prefillRequestTerms(&probe, 100, 4096, bits, 2048, .{}).decode_regrow_bytes;
        if (dearest > billed + regrow_page_slack) {
            over += 1;
            std.debug.print("kv{d}: the rings' compaction holds {d} B past the live rings, billed {d} B\n", .{ bits, dearest, billed });
        }
    }
    try testing.expect(compactions > 0);
    try testing.expectEqual(@as(usize, 0), over);
}
