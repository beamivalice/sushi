//! Hot prefix cache trimming, eviction, takeover/checkout, and admission tests.

const std = @import("std");
const testing = std.testing;
const mlx = @import("mlx.zig");
const trunk_mod = @import("trunk.zig");
const kv_quant = @import("kv_quant.zig");
const kv_disk_cache = @import("kv_disk_cache.zig");
const cache_stats = @import("cache_stats.zig");
const pc = @import("prefix_cache.zig");
const util = @import("prefix_cache_helpers_test.zig");

const KVCache = trunk_mod.KVCache;
const SSMCacheEntry = trunk_mod.SSMCacheEntry;
const SSMCheckpoint = trunk_mod.SSMCheckpoint;
const ssmCheckpointBytes = trunk_mod.ssmCheckpointBytes;
const HotPrefixCache = pc.HotPrefixCache;
const Entry = pc.Entry;
const CommitStatus = pc.CommitStatus;
const TrimDecline = pc.TrimDecline;
const MissKind = pc.MissKind;
const missKind = pc.missKind;
const unlimited_entries = pc.unlimited_entries;
const MIN_CANCELLED_COMMIT_TOKENS = pc.MIN_CANCELLED_COMMIT_TOKENS;

const testWriteCacheLayer = util.testWriteCacheLayer;
const testFillCache = util.testFillCache;
const pcSsmVal = util.pcSsmVal;
const pcBuildHybrid = util.pcBuildHybrid;
const pcFreeHybrid = util.pcFreeHybrid;
const pcEmptySsm = util.pcEmptySsm;
const pcRowBytes = util.pcRowBytes;
const pcExpectRowsEqual = util.pcExpectRowsEqual;
const testCheckoutCache = util.testCheckoutCache;
const pcAppendCheckedOut = util.pcAppendCheckedOut;

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

/// Commit an idle hybrid entry of `n` tokens (ids from `base`) with a restore point at each of `points`.
fn pcCommitHybridAt(hc: *HotPrefixCache, s: mlx.mlx_stream, base: u32, n: u32, points: []const usize) !void {
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try testFillCache(&cache, s, 3, n);
    const toks = try testing.allocator.alloc(u32, n);
    defer testing.allocator.free(toks);
    for (toks, 0..) |*t, i| t.* = base + @as(u32, @intCast(i));
    const cps = try testing.allocator.alloc(SSMCheckpoint, points.len);
    for (cps, points, 0..) |*cp, pos, i| {
        var src = pcBuildHybrid(s, 100.0 * @as(f64, @floatFromInt(i + 1)), 500.0);
        defer pcFreeHybrid(&src);
        cp.* = try trunk_mod.captureSsmCheckpoint(testing.allocator, &src, pos, s);
    }
    _ = try hc.commitWithSsm(&cache, toks, false, cps, null, null);
}

fn pcCommitPlain(hc: *HotPrefixCache, s: mlx.mlx_stream, base: u32, n: u32) !void {
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try testFillCache(&cache, s, 3, n);
    const toks = try testing.allocator.alloc(u32, n);
    defer testing.allocator.free(toks);
    for (toks, 0..) |*t, i| t.* = base + @as(u32, @intCast(i));
    _ = try hc.commit(&cache, toks, false);
}

fn pcBilledSum(hc: *const HotPrefixCache) u64 {
    var sum: u64 = 0;
    for (hc.entries.items) |*e| sum += e.kv_bytes;
    return sum;
}

/// What entry `e` bills when trimmed to its restore point `p`.
fn pcBillAt(e: *const Entry, p: usize) u64 {
    var sum = @as(u64, p) * HotPrefixCache.snapshotRowBytes(&e.snapshot);
    for (e.ssm_checkpoints.?) |*cp| {
        if (cp.pos <= p) sum += ssmCheckpointBytes(cp);
    }
    return sum;
}

fn pcPlainBytes(s: mlx.mlx_stream, n: u32) !u64 {
    var scratch = HotPrefixCache.initWithMem(testing.allocator, 1, 0);
    defer scratch.deinit();
    try pcCommitPlain(&scratch, s, 900_000, n);
    return scratch.current_kv_bytes;
}

/// Set the budget so that a new `n`-token plain entry fits only if the first entry shrinks to `allow` bytes.
fn pcCapLeaving(hc: *HotPrefixCache, e: *const Entry, s: mlx.mlx_stream, n: u32, allow: u64) !void {
    hc.max_kv_bytes = hc.current_kv_bytes + try pcPlainBytes(s, n) - (e.kv_bytes - allow);
}

test "HotPrefixCache: a byte-budget eviction trims the idle LRU entry to the restore point that fits" {
    const s = mlx.gpuStream();
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    try pcCommitHybridAt(&hc, s, 1, 8300, &.{ 512, 8192 });
    const e0 = &hc.entries.items[0];
    // Room for the 512 prefix and not for the 8192 one.
    try pcCapLeaving(&hc, e0, s, 1000, (pcBillAt(e0, 512) + pcBillAt(e0, 8192)) / 2);
    try pcCommitPlain(&hc, s, 900_000, 1000);

    try testing.expectEqual(@as(usize, 2), hc.entryCount());
    try testing.expect(hc.current_kv_bytes <= hc.max_kv_bytes);
    try testing.expectEqual(hc.current_kv_bytes, pcBilledSum(&hc));
    const old = &hc.entries.items[0];
    try testing.expectEqual(@as(u32, 1), old.tokens[0]);
    try testing.expectEqual(@as(usize, 512), old.tokens.len);
    try testing.expectEqual(@as(usize, 1), old.ssm_checkpoints.?.len);
    try testing.expectEqual(@as(usize, 512), old.ssm_checkpoints.?[0].pos);
    // The copy owns exactly its rows: a view of the 8448-row buffer would bill the full capacity.
    try testing.expectEqual(old.kv_bytes, HotPrefixCache.snapshotBytes(&old.snapshot) + old.ssm_bytes);
    try testing.expectEqual(@as(u64, 512) * HotPrefixCache.snapshotRowBytes(&old.snapshot), HotPrefixCache.snapshotBytes(&old.snapshot));

    var prompt: [700]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = if (i < 600) @as(u32, @intCast(i + 1)) else 700_000 + @as(u32, @intCast(i));
    var target = try KVCache.init(testing.allocator, 3);
    defer target.deinit();
    var restored = pcEmptySsm();
    defer pcFreeHybrid(&restored);
    var moe_off: usize = 0;
    const hit = try hc.lookupAndRestore(&target, &moe_off, &restored, s, &prompt, false, 0, null, null);
    try testing.expectEqual(@as(usize, 512), hit.matched);
}

test "HotPrefixCache: a victim with no restore point that fits, or only one under the useful minimum, is evicted whole" {
    const s = mlx.gpuStream();
    const cases = [_][]const usize{ &.{8192}, &.{ 256, 8192 } };
    for (cases) |points| {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        defer hc.deinit();
        try pcCommitHybridAt(&hc, s, 1, 8300, points);
        try pcCapLeaving(&hc, &hc.entries.items[0], s, 1000, pcBillAt(&hc.entries.items[0], 8192) - 1);
        try pcCommitPlain(&hc, s, 900_000, 1000);
        try testing.expectEqual(@as(usize, 1), hc.entryCount());
        try testing.expectEqual(@as(u32, 900_000), hc.entries.items[0].tokens[0]);
        try testing.expect(hc.current_kv_bytes <= hc.max_kv_bytes);
    }
}

test "HotPrefixCache: a checked-out entry is neither trimmed nor evicted for room; the idle one is trimmed" {
    const s = mlx.gpuStream();
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    const held = [_]u32{ 5, 6, 7, 8 };
    try pcAppendCheckedOut(&hc, &held, 1000, 1);
    try pcCommitHybridAt(&hc, s, 1, 8300, &.{ 512, 8192 });
    const e1 = &hc.entries.items[1];
    try pcCapLeaving(&hc, e1, s, 1000, (pcBillAt(e1, 512) + pcBillAt(e1, 8192)) / 2);
    try pcCommitPlain(&hc, s, 900_000, 1000);

    try testing.expectEqual(@as(usize, 3), hc.entryCount());
    try testing.expectEqual(@as(usize, 4), hc.entries.items[0].tokens.len);
    try testing.expect(hc.entries.items[0].checkout != null);
    try testing.expectEqual(@as(u64, 1000), hc.entries.items[0].kv_bytes);
    try testing.expectEqual(@as(usize, 512), hc.entries.items[1].tokens.len);
    try testing.expect(hc.current_kv_bytes <= hc.max_kv_bytes);
}

test "HotPrefixCache: a replacement that outgrows the budget trims another idle entry instead of evicting it" {
    const s = mlx.gpuStream();
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    try pcCommitHybridAt(&hc, s, 1, 8300, &.{ 512, 8192 });
    try pcCommitPlain(&hc, s, 900_000, 1000);
    const grown = try pcPlainBytes(s, 1500);
    const e0 = &hc.entries.items[0];
    const allow = (pcBillAt(e0, 512) + pcBillAt(e0, 8192)) / 2;
    hc.max_kv_bytes = hc.current_kv_bytes - hc.entries.items[1].kv_bytes + grown - (e0.kv_bytes - allow);
    try pcCommitPlain(&hc, s, 900_000, 1500);

    try testing.expectEqual(@as(usize, 2), hc.entryCount());
    try testing.expectEqual(@as(usize, 512), hc.entries.items[0].tokens.len);
    try testing.expectEqual(@as(usize, 1500), hc.entries.items[1].tokens.len);
    try testing.expect(hc.current_kv_bytes <= hc.max_kv_bytes);
}

test "HotPrefixCache: a revised budget trims the idle LRU entry instead of evicting it" {
    const s = mlx.gpuStream();
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    try pcCommitHybridAt(&hc, s, 1, 8300, &.{ 512, 8192 });
    try pcCommitPlain(&hc, s, 900_000, 1000);
    const newest = hc.entries.items[1].kv_bytes;
    const e0 = &hc.entries.items[0];
    hc.setBudget(hc.current_kv_bytes - e0.kv_bytes + (pcBillAt(e0, 512) + pcBillAt(e0, 8192)) / 2);
    try testing.expectEqual(@as(usize, 2), hc.entryCount());
    try testing.expectEqual(@as(usize, 512), hc.entries.items[0].tokens.len);
    try testing.expectEqual(newest, hc.entries.items[1].kv_bytes);
    try testing.expect(hc.current_kv_bytes <= hc.max_kv_bytes);
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
        try pcExpectRowsEqual(s, d.keys.array, e.keys.array, 400);
        try pcExpectRowsEqual(s, d.values.array, e.values.array, 400);
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
    const resident_keys_ctx = hc.entries.items[0].snapshot.entries[0].keys.array.ctx;

    // Next turn: the conversation grew, the trim target did not. Re-copying
    // an identical prefix every turn would be a per-turn multi-GB memcpy.
    var src2 = try KVCache.init(testing.allocator, 2);
    defer src2.deinit();
    try testFillCache(&src2, s, 2, 700);
    _ = try hc.commit(&src2, &toks, false);
    try testing.expectEqual(@as(usize, 1), hc.entryCount());
    try testing.expectEqual(@as(usize, 400), hc.entries.items[0].tokens.len);
    try testing.expectEqual(resident_keys_ctx, hc.entries.items[0].snapshot.entries[0].keys.array.ctx);
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
        c.* = try trunk_mod.captureSsmCheckpoint(testing.allocator, &srcs[i], (i + 1) * 100, s);
    }

    var probe = try c1.snapshot();
    defer probe.deinit();
    const row = pcRowBytes(&probe);
    var cps_at_or_below_500: u64 = 0;
    for (cps[0..5]) |*c| cps_at_or_below_500 += trunk_mod.ssmCheckpointBytes(c);
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
    // The long-context archs price a trim against the checkpoints that survive the shed; retaining the new
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
        c.* = try trunk_mod.captureSsmCheckpoint(testing.allocator, &srcs[i], (i + 1) * 100, s);
    }
    var probe = try c1.snapshot();
    defer probe.deinit();
    const cp_bytes = trunk_mod.ssmCheckpointBytes(&cps[0]);
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
    cps[0] = try trunk_mod.captureSsmCheckpoint(testing.allocator, &hyb, 800, s);

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
        c.* = try trunk_mod.captureSsmCheckpoint(testing.allocator, &srcs[i], (i + 1) * 100, s);
        cps1_bytes += trunk_mod.ssmCheckpointBytes(c);
    }

    // Turn 2 extends to 900 with checkpoints at 500..800.
    var c2 = try KVCache.init(testing.allocator, 3);
    defer c2.deinit();
    try testFillCache(&c2, s, 3, 900);
    const cps2 = try testing.allocator.alloc(SSMCheckpoint, 4);
    var c2_bytes: u64 = 0;
    for (cps2, 0..) |*c, i| {
        c.* = try trunk_mod.captureSsmCheckpoint(testing.allocator, &srcs[i + 4], (i + 5) * 100, s);
        c2_bytes += trunk_mod.ssmCheckpointBytes(c);
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
    // the scheduler's catch arm frees nothing (commitCancelledPrefillSlot detaches them first;
    // pre-fix it freed too, a double free with a different allocator). The
    // cache's error paths therefore MUST free the slice; std.testing.allocator
    // flags both the leak and a double free.
    var failing = std.testing.FailingAllocator.init(testing.allocator, .{ .fail_index = 0 });
    var hc = HotPrefixCache.initWithMem(failing.allocator(), 1, 0);
    defer hc.deinit();

    const cps = try testing.allocator.alloc(SSMCheckpoint, 1);
    cps[0] = .{ .pos = 4, .layers = try testing.allocator.alloc(trunk_mod.SSMCacheEntrySnapshot, 0) };
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
    a_cps[0] = try trunk_mod.captureSsmCheckpoint(testing.allocator, &srcs[0], 8, s);
    a_cps[1] = try trunk_mod.captureSsmCheckpoint(testing.allocator, &srcs[1], 16, s);
    a_cps[2] = try trunk_mod.captureSsmCheckpoint(testing.allocator, &srcs[2], 21, s);
    _ = try hc.commitWithState(&a_cache, &a_tokens, false, 0, a_cps, null, null);

    // Entry B: the serial arm. It restored from A and prefilled a tail too
    // short for a backoff, so its only checkpoint sits at the prompt end.
    var b_cache = try KVCache.init(testing.allocator, 3);
    defer b_cache.deinit();
    try testFillCache(&b_cache, s, 3, b_tokens.len);
    const b_cps = try testing.allocator.alloc(SSMCheckpoint, 1);
    b_cps[0] = try trunk_mod.captureSsmCheckpoint(testing.allocator, &srcs[3], 20, s);
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
        cps[0] = try trunk_mod.captureSsmCheckpoint(testing.allocator, &src256, 256, s);
        cps[1] = try trunk_mod.captureSsmCheckpoint(testing.allocator, &src512, 512, s);
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
        c.* = try trunk_mod.captureSsmCheckpoint(testing.allocator, &srcs[i], (i + 1) * 8, s);
    }
    const one = trunk_mod.ssmCheckpointBytes(&donor[0]);

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
    try trunk_mod.restoreSsmCheckpoint(&dst, &kept[1], s);
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
            if (e.keys.array.ctx != null) _ = mlx.mlx_array_eval(e.keys.array);
            if (e.values.array.ctx != null) _ = mlx.mlx_array_eval(e.values.array);
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
        if (e.keys.array.ctx != null) _ = mlx.mlx_array_eval(e.keys.array);
        if (e.values.array.ctx != null) _ = mlx.mlx_array_eval(e.values.array);
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
        if (e.keys.array.ctx != null) _ = mlx.mlx_array_eval(e.keys.array);
        if (e.values.array.ctx != null) _ = mlx.mlx_array_eval(e.values.array);
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
    // The SSM-checkpoint arm: the indexer history travels with the KV or the
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
    cps[0] = try trunk_mod.captureSsmCheckpoint(testing.allocator, &src512, 512, s);
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
    try t.expect(hc.entries.items[hybrid_idx].checkout == null);

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
    cps[0] = try trunk_mod.captureSsmCheckpoint(testing.allocator, &src512, 512, s);
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
    var pure_layers = [_]trunk_mod.SSMCacheEntrySnapshot{.{
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
        if (e.keys.array.ctx != null) _ = mlx.mlx_array_eval(e.keys.array);
        if (e.values.array.ctx != null) _ = mlx.mlx_array_eval(e.values.array);
    }
    var src8 = pcBuildHybrid(s, 300.0, 700.0);
    defer pcFreeHybrid(&src8);
    const cps = try testing.allocator.alloc(SSMCheckpoint, 1);
    cps[0] = try trunk_mod.captureSsmCheckpoint(testing.allocator, &src8, 8, s);
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
    for (src.entries) |*e| for ([_]mlx.mlx_array{ e.keys.array, e.values.array }) |buf| {
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
    // The live #330 follow-up shape: a long-context arch, stride 4096, a 383,069-token entry at 13,056
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
        const drop = trunk_mod.positionDropIndexUsize(pos[0..n], .min_span_recency);
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
    try t.expectEqual(trunk_mod.ThinPolicy.min_span, hc.cp_thin);

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

        // A long-context attention shape: 2 kv heads, head_dim 256.
        const mk = struct {
            fn f(str: mlx.mlx_stream, len: c_int) !mlx.mlx_array {
                const shape = [_]c_int{ 1, 2, len, 256 };
                var a = mlx.mlx_array_new();
                errdefer _ = mlx.mlx_array_free(a);
                try mlx.check(mlx.mlx_ones(&a, &shape, 4, .bfloat16, str));
                return a;
            }
        }.f;
        // Two writes so the second crosses a growth event; the request ends at their last row.
        cache.bound(64 + 40);
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

        const cap: u64 = @intCast(mlx.getShape(snap.entries[0].keys.array)[2]);
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
    const sh = mlx.getShape(cache.entries[layer].keys.array);
    return @as(u64, @intCast(sh[0])) * @as(u64, @intCast(sh[1])) *
        @as(u64, @intCast(sh[2])) * @as(u64, @intCast(sh[3])) * 4; // f32
}

test "restore by move: a full-prefix hit checks the entry out and the append donates in place" {
    // `KVCache.restore` binds through `mlx_array_set`, so the entry keeps a second reference and
    // the first `writeAtOffset` cannot donate: `copy_gpu` privatised the whole prefix (5.13 GB /
    // ~110 ms at 393k, 45% of the warm TTFT). The observable is allocation, not address: under
    // suite-wide pressure the buffer pool recycles addresses.
    try mlx.requireGpu();
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
        pc.restore_move_override = true;
        defer pc.restore_move_override = null;
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
        try testing.expectEqual(@as(usize, 0xA11CE), e.checkout.?.slot);
        try testing.expect(e.snapshot.entries[0].keys.array.ctx == null);
        try testing.expect(e.snapshot.entries[0].values.array.ctx == null);

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
        pc.restore_move_override = false;
        defer pc.restore_move_override = null;
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
        try testing.expect(e.checkout == null);
        try testing.expect(e.snapshot.entries[0].keys.array.ctx != null);

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
    const p = mlx.mlx_array_data_float32(cache.entries[layer].keys.array) orelse return error.NotEvaluated;
    for (out, 0..) |*v, i| v.* = p[row * 8 + i];
}

test "a one-token prompt that hits its own entry prefills cold: a restore leaves a token to forward" {
    try mlx.requireGpu();
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
    // The admission pass converts a restored share, whole or partial, into a checkout; a second call declines.
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    pc.restore_move_override = true;
    defer pc.restore_move_override = null;
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
    try testing.expect(!hc.checkoutRestored(7, false));
    try testing.expect(hc.checkoutRestored(7, true));
    // Refused before it wrote: the entry comes back whole.
    hc.releaseCheckout(7, "prefill refused");
    try testing.expectEqual(@as(usize, 600), hc.entries.items[0].tokens.len);
    try testing.expect(hc.entries.items[0].snapshot.entries[0].keys.array.ctx != null);

    const res = try hc.lookupAndRestoreForSlot(&slot, &moe_off, null, s, &prompt, false, 0, null, null, null, 7);
    try testing.expectEqual(@as(usize, 600), res.matched);
    try testing.expect(!res.checked_out);
    try testing.expect(hc.checkoutRestored(7, false));
    try testing.expectEqual(@as(usize, 7), hc.entries.items[0].checkout.?.slot);
    try testing.expect(!hc.checkoutRestored(7, true));

    hc.releaseCheckout(7, "prefill refused");
    try testing.expectEqual(@as(usize, 1), hc.entries.items.len);
    try testing.expect(hc.entries.items[0].checkout == null);
    try testing.expect(hc.entries.items[0].snapshot.entries[0].keys.array.ctx != null);
}

test "restore by move: a prompt that is exactly the entry, a full reuse, is taken whole, up front in SSD-first and on demand off it" {
    // Continuing a cut reply resends the entry's tokens: the slot re-forwards the last one from below the share.
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    pc.restore_move_override = true;
    defer pc.restore_move_override = null;
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    for ([_]bool{ true, false }) |ssd_first| {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        defer hc.deinit();
        hc.ssd_first = ssd_first;
        try testCheckoutCache(&hc, s, &tokens, 4096);
        var slot = try KVCache.init(testing.allocator, 1);
        defer slot.deinit();
        var moe_off: usize = 0;
        const res = try hc.lookupAndRestoreForSlot(&slot, &moe_off, null, s, &tokens, false, 0, null, null, null, 7);
        try testing.expectEqual(@as(usize, 599), res.matched);
        try testing.expectEqual(ssd_first, res.checked_out);
        if (!ssd_first) try testing.expect(hc.checkoutRestored(7, false));
        try testing.expect(hc.entries.items[0].checkout.?.kind == .whole);
        hc.releaseCheckout(7, "prefill refused");
        try testing.expect(hc.entries.items[0].checkout == null);
    }
}

test "restore by move: a partial take refused before it donates leaves no cut, so the next whole take keeps the whole entry" {
    // The refused take's keep stayed on the entry and cut the next, whole, take's record to the old share.
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    pc.restore_move_override = true;
    defer pc.restore_move_override = null;
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
    _ = try hc.lookupAndRestoreForSlot(&slot, &moe_off, null, s, &diverged, false, 0, null, null, null, 7);
    try testing.expect(hc.checkoutRestored(7, true));
    hc.releaseCheckout(7, "prefill refused");

    hc.ssd_first = true;
    const res = try hc.lookupAndRestoreForSlot(&slot, &moe_off, null, s, &prompt, false, 0, null, null, null, 8);
    try testing.expect(res.checked_out);
    hc.donateCheckout(8);
    try testing.expectEqual(@as(usize, 600), hc.entries.items[0].tokens.len);
    hc.releaseCheckout(8, "finished");
}

test "restore by move: a partial-prefix hit keeps the refcount-share" {
    // A prompt that diverges from the entry makes no replace promise: its commit lands as a new entry.
    const s = mlx.gpuStream();
    pc.restore_move_override = true;
    defer pc.restore_move_override = null;
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
    try testing.expect(hc.entries.items[0].checkout == null);
    try testing.expect(hc.entries.items[0].snapshot.entries[0].keys.array.ctx != null);
}

test "restore by move: a refusal BEFORE the append hands the entry back INTACT" {
    // Turn B restored a 364k entry by move, the admission pass refused before any forward, and
    // `finishSlot` dropped the checked-out entry. The checkout is a promise, not a transfer:
    // a slot that ends before the donate hands the entry back as it was.
    const s = mlx.gpuStream();
    pc.restore_move_override = true;
    defer pc.restore_move_override = null;
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
    try testing.expect(hc.entries.items[0].checkout.?.donated == null);
    hc.releaseCheckout(42, "prefill refused");

    // The entry is whole: same tokens, live handles, same bill, same LRU position, restorable.
    try testing.expectEqual(@as(usize, 1), hc.entries.items.len);
    const e = &hc.entries.items[0];
    try testing.expectEqual(@as(usize, 600), e.tokens.len);
    try testing.expect(e.checkout == null);
    try testing.expect(e.snapshot.entries[0].keys.array.ctx != null);
    try testing.expect(e.snapshot.entries[0].values.array.ctx != null);
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
    pc.restore_move_override = true;
    defer pc.restore_move_override = null;
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
    try testing.expect(hc.entries.items[0].checkout.?.donated != null);
    try testing.expect(hc.entries.items[0].snapshot.entries[0].keys.array.ctx == null);

    hc.releaseCheckout(42, "cancelled");
    try testing.expectEqual(@as(usize, 0), hc.entries.items.len);
    try testing.expectEqual(@as(u64, 0), hc.current_kv_bytes);
    // Idempotent.
    hc.releaseCheckout(42, "cancelled");
    hc.releaseCheckout(43, "cancelled");
    slot.deinit();
}

test "restore by move ON DEMAND: a partial share taken over keeps the shared prefix, and the turn's commit replaces it" {
    // A long conversation's next turn re-renders the last reply's tail, so its prompt shares only part of the entry (#109).
    const s = mlx.gpuStream();
    pc.restore_move_override = true;
    defer pc.restore_move_override = null;
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var prompt: [608]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(i + 7);
    prompt[500] = 999_999;

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    try testCheckoutCache(&hc, s, &tokens, 4096);

    var slot = try KVCache.init(testing.allocator, 1);
    defer slot.deinit();
    var moe_off: usize = 0;
    const res = try hc.lookupAndRestoreForSlot(&slot, &moe_off, null, s, &prompt, false, 0, null, null, null, 42);
    try testing.expectEqual(@as(usize, 500), res.matched);
    try testing.expect(!res.checked_out);
    try testing.expect(hc.checkoutRestored(42, true));
    hc.donateCheckout(42);
    // What the entry held past the shared prefix is another text: only the shared prefix is kept.
    try testing.expectEqual(@as(usize, 500), hc.entries.items[0].tokens.len);
    try testWriteCacheLayer(&slot, s, 0, 500, 108);

    _ = try hc.commit(&slot, &prompt, false);
    try testing.expectEqual(@as(usize, 1), hc.entries.items.len);
    const e = &hc.entries.items[0];
    try testing.expect(e.checkout == null);
    try testing.expectEqualSlices(u32, &prompt, e.tokens);
    try testing.expect(e.snapshot.entries[0].keys.array.ctx != null);
    try testing.expectEqual(e.kv_bytes, hc.current_kv_bytes);
    hc.releaseCheckout(42, "finished");
    try testing.expectEqual(@as(usize, 1), hc.entries.items.len);
}

test "restore by move: a donation frees the record's MTP history, which nothing reads once the slot owns the entry" {
    // The slot restored its own from it; the turn's commit brings a new one or the slot's release drops the record.
    const s = mlx.gpuStream();
    pc.restore_move_override = true;
    defer pc.restore_move_override = null;
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var prompt: [608]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(i + 7);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    var donor = try KVCache.init(testing.allocator, 1);
    defer donor.deinit();
    donor.reserve(4096);
    try testFillCache(&donor, s, 1, 600);
    var hist = try KVCache.init(testing.allocator, 1);
    defer hist.deinit();
    try testFillCache(&hist, s, 1, 590);
    _ = try hc.commitWithSsm(&donor, &tokens, false, null, null, .{ .cache = &hist, .base_pos = 0 });
    const mtp_bytes = hc.entries.items[0].mtp_bytes;
    try testing.expect(mtp_bytes > 0);
    const billed = hc.current_kv_bytes;

    var slot = try KVCache.init(testing.allocator, 1);
    defer slot.deinit();
    var moe_off: usize = 0;
    _ = try hc.lookupAndRestoreForSlot(&slot, &moe_off, null, s, &prompt, false, 0, null, null, null, 42);
    try testing.expect(hc.checkoutRestored(42, true));
    hc.donateCheckout(42);
    const e = &hc.entries.items[0];
    try testing.expect(e.mtp == null);
    try testing.expectEqual(@as(u64, 0), e.mtp_bytes);
    try testing.expectEqual(billed - mtp_bytes, hc.current_kv_bytes);
    try testing.expectEqual(e.kv_bytes, hc.current_kv_bytes);
    hc.releaseCheckout(42, "finished");
    try testing.expectEqual(@as(u64, 0), hc.current_kv_bytes);
}

test "restore by move: a commit RECLAIMS the checked-out entry with the grown buffers" {
    // The happy path: the replace arm installs the grown snapshot and clears the mark.
    const s = mlx.gpuStream();
    pc.restore_move_override = true;
    defer pc.restore_move_override = null;
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
    try testing.expect(hc.entries.items[0].checkout != null);
    try testWriteCacheLayer(&slot, s, 0, 600, 8);

    _ = try hc.commit(&slot, &prompt, false);
    try testing.expectEqual(@as(usize, 1), hc.entries.items.len);
    const e = &hc.entries.items[0];
    try testing.expect(e.checkout == null);
    try testing.expectEqual(@as(usize, 608), e.tokens.len);
    try testing.expect(e.snapshot.entries[0].keys.array.ctx != null);
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
    pc.restore_move_override = true;
    defer pc.restore_move_override = null;
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
    try testing.expectEqual(@as(usize, 42), hc.entries.items[0].checkout.?.slot);

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
    try testing.expectEqual(@as(usize, 42), hc.entries.items[0].checkout.?.slot);

    hc.releaseCheckout(42, "test teardown");
}

test "restore by move: the policy is off outside the SSD-first arm and with the move forced off" {
    // The up-front take: SSD-first, a slot, the move on, no pending flush, the whole entry matched.
    try testing.expect(HotPrefixCache.upFrontTakeEligible(true, true, true, false, 600, 600));
    try testing.expect(!HotPrefixCache.upFrontTakeEligible(false, true, true, false, 600, 600));
    try testing.expect(!HotPrefixCache.upFrontTakeEligible(true, true, false, false, 600, 600));
    try testing.expect(!HotPrefixCache.upFrontTakeEligible(true, false, true, false, 600, 600));
    try testing.expect(!HotPrefixCache.upFrontTakeEligible(true, true, true, true, 600, 600));
    try testing.expect(!HotPrefixCache.upFrontTakeEligible(true, true, true, false, 600, 500));
    try testing.expect(!HotPrefixCache.upFrontTakeEligible(true, true, true, false, 0, 0));
    // Any take, on demand too, needs the move on and no pending flush.
    try testing.expect(HotPrefixCache.movable(true, false));
    try testing.expect(!HotPrefixCache.movable(false, false));
    try testing.expect(!HotPrefixCache.movable(true, true));
    // A partial take needs only a share to keep, whatever the prompt's length, of an entry with no ring.
    try testing.expect(HotPrefixCache.partialTakeEligible(600, 500, false));
    try testing.expect(!HotPrefixCache.partialTakeEligible(600, 0, false));
    try testing.expect(!HotPrefixCache.partialTakeEligible(600, 600, false));
    try testing.expect(!HotPrefixCache.partialTakeEligible(600, 500, true));
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
        try testing.expect(hc.entries.items[0].checkout != null);
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
        try testing.expect(hc.entries.items[0].checkout != null);
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
            .snapshot = .{ .entries = try testing.allocator.alloc(trunk_mod.KVCacheEntry, 0), .step = 0, .allocator = testing.allocator, .config = trunk_mod.KVQuantConfig.dense },
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
    cache.entries.items[1].checkout = .{ .slot = 7, .kind = .whole };
    try testing.expectEqual(@as(?usize, 2), cache.lruIndexExcluding(null, key_b));
    cache.entries.items[1].checkout = null;
    // One workload = plain LRU: C goes first.
    for (cache.entries.items) |*e| e.cache_key = 0;
    try testing.expectEqual(@as(?usize, 0), cache.lruIndexExcluding(null, 0));
}

/// Two layerless entries of 100 and 200 bytes, the first the LRU.
fn appendCounterStubs(cache: *HotPrefixCache) !void {
    for (1..3) |used| {
        try cache.entries.append(testing.allocator, .{
            .tokens = try testing.allocator.dupe(u32, &[_]u32{ 1, @intCast(used) }),
            .has_tools = false,
            .cache_key = 0,
            .snapshot = .{ .entries = try testing.allocator.alloc(trunk_mod.KVCacheEntry, 0), .step = 0, .allocator = testing.allocator, .config = trunk_mod.KVQuantConfig.dense },
            .last_used = used,
            .quant_config = kv_quant.KVQuantConfig.dense,
            .kv_bytes = used * 100,
            .ssm_checkpoints = null,
            .ssm_bytes = 0,
        });
    }
    cache.current_kv_bytes = 300;
}

/// A layerless entry of `tokens.len` prompt tokens, `last_used = used`, no checkpoints.
fn appendTokenStub(hc: *HotPrefixCache, tokens: []const u32, used: u64) !void {
    try hc.entries.append(testing.allocator, .{
        .tokens = try testing.allocator.dupe(u32, tokens),
        .has_tools = false,
        .snapshot = .{ .entries = try testing.allocator.alloc(trunk_mod.KVCacheEntry, 0), .step = 0, .allocator = testing.allocator, .config = trunk_mod.KVQuantConfig.dense },
        .last_used = used,
        .quant_config = kv_quant.KVQuantConfig.dense,
        .kv_bytes = tokens.len * 10,
        .ssm_checkpoints = null,
        .ssm_bytes = 0,
    });
    hc.current_kv_bytes += tokens.len * 10;
}

test "a hot cache with no count limit keeps the 33rd conversation where the old default of 32 dropped it" {
    var cache = HotPrefixCache.initWithMem(testing.allocator, unlimited_entries, 0);
    defer cache.deinit();
    for (0..33) |i| {
        const tokens = [_]u32{ @intCast(i), @intCast(i + 1) };
        try testing.expect(try cache.retainNewEntry(.{
            .tokens = try testing.allocator.dupe(u32, &tokens),
            .has_tools = false,
            .snapshot = .{ .entries = try testing.allocator.alloc(trunk_mod.KVCacheEntry, 0), .step = 0, .allocator = testing.allocator, .config = trunk_mod.KVQuantConfig.dense },
            .last_used = i + 1,
            .quant_config = kv_quant.KVQuantConfig.dense,
            .kv_bytes = 20,
            .ssm_checkpoints = null,
            .ssm_bytes = 0,
        }));
    }
    try testing.expectEqual(@as(usize, 33), cache.entries.items.len);
}

fn retainConversation(cache: *HotPrefixCache, id: u32) !bool {
    const tokens = [_]u32{ id, id + 100 };
    return cache.retainNewEntry(.{
        .tokens = try testing.allocator.dupe(u32, &tokens),
        .has_tools = false,
        .snapshot = .{ .entries = try testing.allocator.alloc(trunk_mod.KVCacheEntry, 0), .step = 0, .allocator = testing.allocator, .config = trunk_mod.KVQuantConfig.dense },
        .last_used = id,
        .quant_config = kv_quant.KVQuantConfig.dense,
        .kv_bytes = 20,
        .ssm_checkpoints = null,
        .ssm_bytes = 0,
    });
}

test "a hot cache counted at the model's max_concurrent evicts the oldest conversation past it" {
    for ([_]u32{ 1, 3 }) |count| {
        var cache = HotPrefixCache.initWithMem(testing.allocator, count, 0);
        defer cache.deinit();
        for (1..count + 1) |i| try testing.expect(try retainConversation(&cache, @intCast(i)));
        try testing.expectEqual(@as(usize, count), cache.entries.items.len);
        try testing.expect(try retainConversation(&cache, count + 1));
        try testing.expectEqual(@as(usize, count), cache.entries.items.len);
        for (cache.entries.items) |e| try testing.expect(e.tokens[0] != 1);
        try testing.expectEqual(count + 1, cache.entries.items[cache.entries.items.len - 1].tokens[0]);
    }
}

test "retained tokens count every entry in RAM, and eviction takes the oldest first" {
    var cache = HotPrefixCache.initWithMem(testing.allocator, unlimited_entries, 0);
    defer cache.deinit();
    try appendTokenStub(&cache, &.{ 1, 2 }, 1);
    try appendTokenStub(&cache, &.{ 3, 4 }, 2);
    try appendTokenStub(&cache, &.{ 5, 6 }, 3);
    try testing.expectEqual(@as(u64, 6), cache.retainedTokens());
    try testing.expectEqual(@as(u64, 4), cache.evictOldestForTokens(4, &.{}));
    try testing.expectEqual(@as(u64, 2), cache.retainedTokens());
    try testing.expectEqual(@as(u32, 5), cache.entries.items[0].tokens[0]);
}

test "a retained entry a live slot has checked out is never evicted for tokens" {
    var cache = HotPrefixCache.initWithMem(testing.allocator, unlimited_entries, 0);
    defer cache.deinit();
    try appendTokenStub(&cache, &.{ 1, 2 }, 1);
    try appendTokenStub(&cache, &.{ 3, 4 }, 2);
    try pcAppendCheckedOut(&cache, &.{ 5, 6 }, 20, 3);
    // The checked-out entry is the live slot's own reservation now, so the pool holds only the two free ones.
    try testing.expectEqual(@as(u64, 4), cache.retainedTokens());
    // Six asked: the two free entries give four, the checked-out one stays.
    try testing.expectEqual(@as(u64, 4), cache.evictOldestForTokens(6, &.{}));
    try testing.expectEqual(@as(usize, 1), cache.entries.items.len);
    try testing.expect(cache.entries.items[0].checkout != null);
    try testing.expectEqual(@as(u64, 0), cache.retainedTokens());
}

test "HotPrefixCache: a capacity eviction is counted with its bytes in the hot tier" {
    var cache = HotPrefixCache.init(testing.allocator, 8);
    defer cache.deinit();
    try appendCounterStubs(&cache);
    const before = cache_stats.process.snapshot();
    cache.evictOneLru("count cap", null);
    const after = cache_stats.process.snapshot();
    try testing.expectEqual(before.evictions[0] + 1, after.evictions[0]);
    try testing.expectEqual(before.evicted_bytes[0] + 100, after.evicted_bytes[0]);
    try testing.expectEqual(before.evictions[1], after.evictions[1]);
    // An invalidation (a failed check) is not a capacity eviction.
    cache.evictAt(0, "QSA history check");
    try testing.expectEqual(before.evictions[0] + 1, cache_stats.process.snapshot().evictions[0]);
}

test "HotPrefixCache: entries evicted to admit a prefill are hot-tier evictions" {
    var cache = HotPrefixCache.init(testing.allocator, 8);
    defer cache.deinit();
    try appendCounterStubs(&cache);
    const Never = struct {
        fn call(_: ?*anyopaque) bool {
            return false;
        }
    };
    const before = cache_stats.process.snapshot();
    const rep = cache.evictLruToAdmit(1 << 20, null, Never.call, false);
    try testing.expectEqual(@as(usize, 2), rep.entries);
    const after = cache_stats.process.snapshot();
    try testing.expectEqual(before.evictions[0] + 2, after.evictions[0]);
    try testing.expectEqual(before.evicted_bytes[0] + 300, after.evicted_bytes[0]);
}

test "HotPrefixCache: the tier that served a restore is the disk's only when the disk supplied the rows" {
    var cache = HotPrefixCache.init(testing.allocator, 8);
    defer cache.deinit();
    try testing.expectEqual(cache_stats.Tier.hot, cache.lastRestoredTier());
    cache.last_restored_disk_id = 5;
    try testing.expectEqual(cache_stats.Tier.ssd, cache.lastRestoredTier());
    cache.last_restored_live = true;
    try testing.expectEqual(cache_stats.Tier.hot, cache.lastRestoredTier());
}

test "HotPrefixCache: checkpoint clone allocation failure releases owned layers" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var accounting = testing.FailingAllocator.init(arena.allocator(), .{ .fail_index = 1 });
    var layers = [_]trunk_mod.SSMCacheEntrySnapshot{.{
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
        .snapshot = .{ .entries = try testing.allocator.alloc(trunk_mod.KVCacheEntry, 0), .step = 4, .allocator = testing.allocator, .config = kv_quant.KVQuantConfig.dense },
        .last_used = 2,
        .quant_config = kv_quant.KVQuantConfig.dense,
        .kv_bytes = 128,
        .ssm_checkpoints = try testing.allocator.alloc(SSMCheckpoint, 0),
        .ssm_bytes = 0,
    });
    try testing.expect(!retained);
    try testing.expectEqual(@as(usize, 1), hc.entries.items.len);
    try testing.expectEqual(max_bytes, hc.current_kv_bytes);
    try testing.expect(hc.entries.items[0].checkout != null);
}

test "a restore names its entry, and a commit that extends it in place keeps the id" {
    try mlx.requireGpu();
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
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var prompt: [608]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(i + 7);
    pc.restore_move_override = true;
    defer pc.restore_move_override = null;
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
