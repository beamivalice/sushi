//! Shared test helpers for the hot prefix cache test suite.

const std = @import("std");
const testing = std.testing;
const mlx = @import("mlx.zig");
const trunk_mod = @import("trunk.zig");
const trunk_helpers_test = @import("trunk_helpers_test.zig");
const kv_quant = @import("kv_quant.zig");
const log = @import("log.zig");
const pc = @import("prefix_cache.zig");

const KVCache = trunk_mod.KVCache;
const KVCacheSnapshot = trunk_mod.KVCacheSnapshot;
const SSMCacheEntry = trunk_mod.SSMCacheEntry;
const HotPrefixCache = pc.HotPrefixCache;
const CommitStatus = pc.CommitStatus;


pub fn testWriteCacheLayer(cache: *KVCache, s: mlx.mlx_stream, layer: u32, written: u32, step: u32) !void {
    // A write is a request whose last row is its own: it binds the cache there, past any earlier request's end.
    cache.bound(@max(cache.end_tokens, written + step));
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

pub fn testFillCache(cache: *KVCache, s: mlx.mlx_stream, n_layers: u32, tokens: u32) !void {
    var written: u32 = 0;
    while (written < tokens) {
        const step: u32 = @min(64, tokens - written);
        var li: u32 = 0;
        while (li < n_layers) : (li += 1) try testWriteCacheLayer(cache, s, li, written, step);
        written += step;
    }
}

/// One layer's chunk at absolute rows `[written, written+step)`, handed the
/// `max_seq` its attention would ask for: non-zero IS the ring predicate.
pub fn ringWriteLayer(cache: *KVCache, s: mlx.mlx_stream, layer: u32, written: u32, step: u32, window: u32) !trunk_mod.DenseKVView {
    cache.bound(@max(cache.end_tokens, written + step));
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    const count: f64 = @floatFromInt(step * 8);
    const base: f64 = @floatFromInt(written * 8 + layer * 1_000_000);
    try mlx.check(mlx.mlx_arange(&flat, base, base + count, 1.0, .float32, s));
    var k = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k);
    const shape = [_]c_int{ 1, 1, @intCast(step), 8 };
    try mlx.check(mlx.mlx_reshape(&k, flat, &shape, 4, s));
    const span: u32 = if (layer == 0) 0 else trunk_mod.slidingTailSpan(window, step, trunk_mod.SLIDING_TRIM_UNBOUNDED);
    return cache.update(layer, k, k, s, span);
}

/// Ringed-trunk shape: layer 0 global, every other layer sliding.
pub fn ringFill(cache: *KVCache, s: mlx.mlx_stream, n_layers: u32, window: u32, from: u32, to: u32, chunk: u32) !void {
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
pub fn ringRowBytes(arr: mlx.mlx_array) u64 {
    const rows: u64 = @intCast(mlx.getShape(arr)[2]);
    return (@as(u64, mlx.mlx_array_size(arr)) * @as(u64, mlx.mlx_array_itemsize(arr))) / rows;
}

/// A ringed-trunk turn: `prompt` rows, a ring checkpoint at the prompt end, then a reply long
/// enough that the ring compacts past the prompt. The caller owns the checkpoint.
pub fn ringTurn(live: *KVCache, s: mlx.mlx_stream, n_layers: u32, window: u32, prompt: u32, reply: u32) !?KVCacheSnapshot {
    live.setSwaRing(window);
    try ringFill(live, s, n_layers, window, 0, prompt, 64);
    var cp = try live.ringCheckpoint(prompt, s);
    errdefer if (cp) |*c| c.deinit();
    try ringFill(live, s, n_layers, window, prompt, prompt + reply, 16);
    return cp;
}

/// Feed `restored` from `from` to `to`, then one more row: every layer must read exactly what
/// a cache fed from row 0 reads.
pub fn expectRingContinuesCold(restored: *KVCache, s: mlx.mlx_stream, n_layers: u32, window: u32, from: u32, to: u32) !void {
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
        try testing.expectEqual(@as(f32, 0), try trunk_helpers_test.maxAbsDiffF32(rv.k, sv.k, s));
        try testing.expectEqual(@as(f32, 0), try trunk_helpers_test.maxAbsDiffF32(rv.v, sv.v, s));
        try testing.expectEqual(ref.absSeqLen(li), restored.absSeqLen(li));
    }
}

// ── Phase 3: two-tier hybrid restore (GatedDeltaNet) ──

pub const conv_shape_pc = [_]c_int{ 1, 3, 8 };

pub const ssm_shape_pc = [_]c_int{ 1, 2, 4, 4 };

pub fn pcArange(s: mlx.mlx_stream, shape: []const c_int, base: f64) mlx.mlx_array {
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

pub fn pcSsmVal(arr: mlx.mlx_array, idx: usize, s: mlx.mlx_stream) f32 {
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    _ = mlx.mlx_astype(&f, arr, .float32, s);
    _ = mlx.mlx_array_eval(f);
    return mlx.mlx_array_data_float32(f).?[idx];
}

pub fn pcBuildHybrid(s: mlx.mlx_stream, conv_base: f64, ssm_base: f64) [3]SSMCacheEntry {
    return .{
        .{ .conv_state = pcArange(s, &conv_shape_pc, conv_base), .ssm_state = pcArange(s, &ssm_shape_pc, ssm_base), .initialized = true },
        .{ .conv_state = pcArange(s, &conv_shape_pc, conv_base + 10_000), .ssm_state = mlx.mlx_array_new(), .initialized = true },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
    };
}

pub fn pcFreeHybrid(e: *[3]SSMCacheEntry) void {
    for (e) |*x| {
        _ = mlx.mlx_array_free(x.conv_state);
        _ = mlx.mlx_array_free(x.ssm_state);
    }
}

pub fn pcEmptySsm() [3]SSMCacheEntry {
    return .{
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
    };
}

// ── Issue #330: the oversized-entry decline is a cliff, not a cap ──

/// Test-local: dense per-token KV bytes of a snapshot (k + v across layers).
pub fn pcRowBytes(snap: *const trunk_mod.KVCacheSnapshot) u64 {
    var total: u64 = 0;
    for (snap.entries) |e| {
        if (!e.initialized) continue;
        const rows: u64 = @intCast(mlx.mlx_array_shape(e.keys.array)[2]);
        if (rows == 0) continue;
        const kb = @as(u64, mlx.mlx_array_size(e.keys.array)) * @as(u64, mlx.mlx_array_itemsize(e.keys.array));
        const vb = @as(u64, mlx.mlx_array_size(e.values.array)) * @as(u64, mlx.mlx_array_itemsize(e.values.array));
        total += (kb + vb) / rows;
    }
    return total;
}

/// Test-local: assert rows [0:rows] of `a` and `b` (axis 2) are identical.
pub fn pcExpectRowsEqual(s: mlx.mlx_stream, a: mlx.mlx_array, b: mlx.mlx_array, rows: usize) !void {
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

pub fn pcFreeQsaHybrid(e: *[3]SSMCacheEntry) void {
    for (e) |*x| {
        if (x.conv_state.ctx != null) _ = mlx.mlx_array_free(x.conv_state);
        if (x.ssm_state.ctx != null) _ = mlx.mlx_array_free(x.ssm_state);
        if (x.aux_state.ctx != null) _ = mlx.mlx_array_free(x.aux_state);
        if (x.qsa_pooled.ctx != null) _ = mlx.mlx_array_free(x.qsa_pooled);
    }
}

pub fn pcQsaBf16Equal(s: mlx.mlx_stream, a: mlx.mlx_array, b: mlx.mlx_array) !void {
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

pub fn pcQsaSliceAxis1(s: mlx.mlx_stream, arr: mlx.mlx_array, from: c_int, to: c_int) !mlx.mlx_array {
    const sh = mlx.getShape(arr);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_slice(&out, arr, &[_]c_int{ 0, from, 0 }, 3, &[_]c_int{ sh[0], to, sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    return out;
}

pub fn pcQsaExpectViewEqual(s: mlx.mlx_stream, warm: *SSMCacheEntry, cold: *SSMCacheEntry, pos: c_int) !void {
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

pub fn pcArrEqual(s: mlx.mlx_stream, a: mlx.mlx_array, b: mlx.mlx_array) !void {
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

/// A commit whose SSD flush never ran, so `spillIdleEntries` is the tier's only writer.
pub fn commitUnflushed(hc: *HotPrefixCache, cache: *const KVCache, tokens: []const u32) !CommitStatus {
    const r = try hc.commit(cache, tokens, false);
    if (hc.pending_disk) |*p| p.deinit(hc.allocator);
    hc.pending_disk = null;
    hc.disk_dirty = false;
    return r;
}

pub fn testCheckoutCache(hc: *HotPrefixCache, s: mlx.mlx_stream, tokens: []const u32, reserve: usize) !void {
    var donor = try KVCache.init(testing.allocator, 1);
    defer donor.deinit();
    donor.reserve(reserve);
    try testFillCache(&donor, s, 1, @intCast(tokens.len));
    _ = try hc.commit(&donor, tokens, false);
}

/// A resident entry a live slot holds: eviction can never take it.
pub fn pcAppendCheckedOut(hc: *HotPrefixCache, tokens: []const u32, kv_bytes: u64, used: u64) !void {
    try hc.entries.append(testing.allocator, .{
        .tokens = try testing.allocator.dupe(u32, tokens),
        .has_tools = false,
        .snapshot = .{ .entries = try testing.allocator.alloc(trunk_mod.KVCacheEntry, 0), .step = 0, .allocator = testing.allocator, .config = trunk_mod.KVQuantConfig.dense },
        .last_used = used,
        .quant_config = kv_quant.KVQuantConfig.dense,
        .kv_bytes = kv_bytes,
        .ssm_checkpoints = null,
        .ssm_bytes = 0,
        .checkout = .{ .slot = 1, .kind = .whole },
    });
    hc.current_kv_bytes += kv_bytes;
}

pub fn mediaTurnTokens(comptime n: usize) [n]u32 {
    var tokens: [n]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    return tokens;
}
