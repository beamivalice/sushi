//! SSD prefix cache restore, lookup, and salvage tests.

const std = @import("std");
const testing = std.testing;
const mlx = @import("mlx.zig");
const kv_quant = @import("kv_quant.zig");
const trunk = @import("trunk.zig");
const write_stats = @import("disk_write_stats.zig");
const cache_stats = @import("cache_stats.zig");
const kv_disk_fs = @import("kv_disk_fs.zig");
const disk = @import("kv_disk_cache.zig");
const util = @import("kv_disk_helpers_test.zig");

const KVCache = trunk.KVCache;
const SSMCacheEntry = trunk.SSMCacheEntry;
const DiskTier = disk.DiskTier;
const IndexEntry = disk.IndexEntry;
const PersistOutcome = disk.PersistOutcome;
const POINT_GRID = disk.POINT_GRID;
const SSM_DISK_MAX_PER_ENTRY = disk.SSM_DISK_MAX_PER_ENTRY;

const fillCache = util.fillCache;
const cacheValueAt = util.cacheValueAt;
const cacheBufValueAt = util.cacheBufValueAt;
const tmpRoot = util.tmpRoot;
const buildHybridEntries = util.buildHybridEntries;
const freeHybridEntries = util.freeHybridEntries;

test "DiskTier: a base folder that links to a folder on another volume opens the tier" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root = try tmpRoot(&tmp, io, &buf);
    try tmp.dir.createDirPath(io, "volume/kv-cache");
    try tmp.dir.symLink(io, "volume/kv-cache", "kv-cache", .{ .is_directory = true });
    var p: [600]u8 = undefined;
    var tier = try DiskTier.init(testing.allocator, io, try std.fmt.bufPrint(&p, "{s}/kv-cache", .{root}), "fp-linked", 0, 128);
    defer tier.deinit();
    var ns = try tmp.dir.openDir(io, "volume/kv-cache/fp-linked", .{});
    ns.close(io);
}

test "DiskTier: failed checkpoint manifest keeps index ownership" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);
    var tier = try DiskTier.init(testing.allocator, io, base, "fp-owned", 0, 128);
    defer tier.deinit();
    var staging: [64]u8 = undefined;
    try kv_disk_fs.testMkdirAt(io, base, try std.fmt.bufPrint(&staging, "fp-owned/e1/meta.json.{d}.tmp", .{std.c.getpid()}));
    const tokens = try testing.allocator.alloc(u32, 600);
    @memset(tokens, 7);
    const positions = try testing.allocator.dupe(u32, &.{512});
    const sizes = try testing.allocator.dupe(u64, &.{64});
    try tier.entries.append(testing.allocator, .{
        .id = 1,
        .tokens = tokens,
        .kv_len = 600,
        .has_tools = false,
        .quant = kv_quant.KVQuantConfig.dense,
        .bytes = 2464,
        .chunk_bytes = try testing.allocator.alloc(u64, 0),
        .ssm_positions = positions,
        .ssm_bytes = sizes,
        .last_used = 1,
    });
    tier.total_bytes = 2464;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var accounting = testing.FailingAllocator.init(arena.allocator(), .{});
    tier.allocator = accounting.allocator();
    defer tier.allocator = testing.allocator;

    if (tier.appendSsmOnly(0, null, null, null, &.{}, .{ .ctx = null })) |_| {
        return error.TestExpectedError;
    } else |_| {}

    try testing.expectEqual(@as(usize, 0), accounting.allocated_bytes - accounting.freed_bytes);
    try testing.expect(tier.entries.items[0].ssm_positions.ptr == positions.ptr);
    try testing.expect(tier.entries.items[0].ssm_bytes.ptr == sizes.ptr);
    try testing.expectEqual(@as(u64, 2464), tier.total_bytes);
}

test "DiskTier: chunked commit + restore round-trips exact KV, step, offsets" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-test", 0, 128);

    // 600 tokens => 5 chunks at 128 (last partial: 88).
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    try testing.expectEqual(@as(usize, 600), cache.step);

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
    try testing.expectEqual(@as(usize, 1), tier.entryCount());

    // Restore into a fresh cache (fresh tier too — proves the restart path).
    tier.deinit();
    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-test", 0, 128);
    defer tier2.deinit();
    try testing.expectEqual(@as(usize, 1), tier2.entryCount());

    const m = tier2.bestMatch(&tokens, false, kv_quant.KVQuantConfig.dense, .{}).?;
    try testing.expectEqual(@as(u32, 600), m.usable);

    var cache2 = try KVCache.init(testing.allocator, 3);
    defer cache2.deinit();
    const restored = try tier2.restoreInto(&cache2, m.idx, s);
    try testing.expectEqual(@as(u32, 600), restored);
    try testing.expectEqual(@as(usize, 600), cache2.step);
    for (cache2.entries) |*e| {
        try testing.expect(e.initialized);
        try testing.expectEqual(@as(usize, 600), e.offset);
    }

    // Spot-check exact values across chunk boundaries and layers.
    const probes = [_][2]u32{ .{ 0, 0 }, .{ 127, 7 }, .{ 128, 0 }, .{ 300, 3 }, .{ 511, 7 }, .{ 512, 0 }, .{ 599, 7 } };
    for (probes) |p| {
        var li: u32 = 0;
        while (li < 3) : (li += 1) {
            const want = try cacheValueAt(&cache, li, p[0], p[1], s);
            const got = try cacheValueAt(&cache2, li, p[0], p[1], s);
            try testing.expectEqual(want, got);
            // V was written as -K: restored values must mirror that, so a
            // restore-side K/V swap or shared-buffer mixup fails here.
            const got_v = try cacheBufValueAt(&cache2, li, p[0], p[1], s, true);
            try testing.expectEqual(-want, got_v);
        }
    }

    // Mismatched key never matches.
    try testing.expect(tier2.bestMatch(&tokens, true, kv_quant.KVQuantConfig.dense, .{}) == null);
    try testing.expect(tier2.bestMatch(&tokens, false, kv_quant.KVQuantConfig.affine(4), .{}) == null);
}

test "DiskTier: extend commit appends only new chunks (full chunks untouched)" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-ext", 0, 128);
    defer tier.deinit();

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try fillCache(&cache, s, 1, 600, 8, 0.0, .float32);
    var tokens: [900]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, tokens[0..600], false, null, s);

    // Tamper-mark chunk 0 (a FULL chunk): record its mtime, then extend the
    // entry and assert chunk 0 was not rewritten while chunk 4 (the old
    // partial) was, and new chunks appeared.
    const c0_path = try std.fmt.allocPrint(testing.allocator, "{s}/fp-ext/e1/c000000.safetensors", .{base});
    defer testing.allocator.free(c0_path);
    const c4_path = try std.fmt.allocPrint(testing.allocator, "{s}/fp-ext/e1/c000004.safetensors", .{base});
    defer testing.allocator.free(c4_path);
    const c6_path = try std.fmt.allocPrint(testing.allocator, "{s}/fp-ext/e1/c000006.safetensors", .{base});
    defer testing.allocator.free(c6_path);
    const c0_before = kv_disk_fs.testStat(io, c0_path).?.mtime_ns;
    const c4_before = kv_disk_fs.testStat(io, c4_path).?.mtime_ns;
    try testing.expect(kv_disk_fs.testSize(io, c6_path) == null);

    // Ensure the extend write lands at a measurably later mtime.
    std.Io.sleep(io, .fromMilliseconds(20), .real) catch {};

    // Same prefix, 300 more tokens.
    try fillCache(&cache, s, 1, 300, 8, 4800.0, .float32);
    try testing.expectEqual(@as(usize, 900), cache.step);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
    try testing.expectEqual(@as(usize, 1), tier.entryCount());
    try testing.expectEqual(@as(u32, 900), tier.entries.items[0].kv_len);

    try testing.expectEqual(c0_before, kv_disk_fs.testStat(io, c0_path).?.mtime_ns); // untouched
    try testing.expect(kv_disk_fs.testStat(io, c4_path).?.mtime_ns != c4_before); // partial rewritten
    try testing.expect(kv_disk_fs.testSize(io, c6_path) != null); // new tail chunk

    // Restore the extended entry and check a value in the extension range.
    var cache2 = try KVCache.init(testing.allocator, 1);
    defer cache2.deinit();
    const restored = try tier.restoreInto(&cache2, 0, s);
    try testing.expectEqual(@as(u32, 900), restored);
    const want = try cacheValueAt(&cache, 0, 750, 5, s);
    const got = try cacheValueAt(&cache2, 0, 750, 5, s);
    try testing.expectEqual(want, got);
}

test "DiskTier: identical re-commit is a no-op; shorter prefix is superseded" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-noop", 0, 128);
    defer tier.deinit();

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try fillCache(&cache, s, 1, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
    const c0_path = try std.fmt.allocPrint(testing.allocator, "{s}/fp-noop/e1/c000000.safetensors", .{base});
    defer testing.allocator.free(c0_path);
    const before = kv_disk_fs.testStat(io, c0_path).?.mtime_ns;

    std.Io.sleep(io, .fromMilliseconds(20), .real) catch {};

    // Identical commit — nothing rewritten, no second entry.
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
    try testing.expectEqual(@as(usize, 1), tier.entryCount());
    try testing.expectEqual(before, kv_disk_fs.testStat(io, c0_path).?.mtime_ns);

    // A shorter-prefix commit of the same conversation is covered by the
    // existing entry — also a no-op.
    var short_cache = try KVCache.init(testing.allocator, 1);
    defer short_cache.deinit();
    try fillCache(&short_cache, s, 1, 512, 8, 0.0, .float32);
    _ = try tier.appendCommit(short_cache.entries, short_cache.step, short_cache.config, tokens[0..512], false, null, s);
    try testing.expectEqual(@as(usize, 1), tier.entryCount());
}

test "DiskTier: byte budget evicts LRU entries, keeps newest" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-gc", 0, 128);
    defer tier.deinit();

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try fillCache(&cache, s, 1, 520, 8, 0.0, .float32);

    var tokens_a: [520]u32 = undefined;
    for (&tokens_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var tokens_b: [520]u32 = undefined;
    for (&tokens_b, 0..) |*t, i| t.* = @intCast(i + 900_000);

    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens_a, false, null, s);
    // A budget that holds one entry but not two: the newest evicts the previous one.
    tier.max_bytes = tier.total_bytes + tier.total_bytes / 2;
    const evicted_before = cache_stats.process.snapshot();
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens_b, false, null, s);
    try testing.expectEqual(@as(usize, 1), tier.entryCount());
    const evicted_after = cache_stats.process.snapshot();
    try testing.expectEqual(evicted_before.evictions[1] + 1, evicted_after.evictions[1]);
    try testing.expect(evicted_after.evicted_bytes[1] > evicted_before.evicted_bytes[1]);
    try testing.expect(std.mem.eql(u32, tier.entries.items[0].tokens, &tokens_b));

    // The evicted directory is gone from disk.
    const e1_meta = try std.fmt.allocPrint(testing.allocator, "{s}/fp-gc/e1/meta.json", .{base});
    defer testing.allocator.free(e1_meta);
    try testing.expect(kv_disk_fs.testStat(io, e1_meta) == null);
}

test "DiskTier: scan drops crash leftovers (no meta.json)" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    {
        var tier = try DiskTier.init(testing.allocator, io, base, "fp-crash", 0, 128);
        defer tier.deinit();
        var cache = try KVCache.init(testing.allocator, 1);
        defer cache.deinit();
        try fillCache(&cache, s, 1, 600, 8, 0.0, .float32);
        var tokens: [600]u32 = undefined;
        for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
        _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
        // Simulate a crash mid-write of a SECOND entry: chunks, no meta.
        try kv_disk_fs.testMkdirAt(io, base, "fp-crash/e9");
        try kv_disk_fs.testWriteAt(io, base, "fp-crash/e9/c000000.safetensors", "junk");
    }

    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-crash", 0, 128);
    defer tier2.deinit();
    try testing.expectEqual(@as(usize, 1), tier2.entryCount());
    // The leftover dir was removed.
    const leftover = try std.fmt.allocPrint(testing.allocator, "{s}/fp-crash/e9/c000000.safetensors", .{base});
    defer testing.allocator.free(leftover);
    try testing.expect(kv_disk_fs.testStat(io, leftover) == null);
    // next_id moved past the dropped id (no reuse of a dirty dir name).
    try testing.expect(tier2.next_id >= 10);
}

test "DiskTier: two live tiers on one root restore only their own entries" {
    // Two processes on one pack resolve the same root, and each numbers its entries from 1.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier_a = try DiskTier.init(testing.allocator, io, base, "fp-two", 0, 128);
    defer tier_a.deinit();
    var cache_a = try KVCache.init(testing.allocator, 1);
    defer cache_a.deinit();
    try fillCache(&cache_a, s, 1, 600, 8, 0.0, .float32);
    var tokens_a: [600]u32 = undefined;
    for (&tokens_a, 0..) |*t, i| t.* = @intCast(i + 7);
    var cache_b = try KVCache.init(testing.allocator, 1);
    defer cache_b.deinit();
    try fillCache(&cache_b, s, 1, 700, 8, 50_000.0, .float32);
    var tokens_b: [700]u32 = undefined;
    for (&tokens_b, 0..) |*t, i| t.* = @intCast(i + 900_000);
    {
        var tier_b = try DiskTier.init(testing.allocator, io, base, "fp-two", 0, 128);
        defer tier_b.deinit();
        _ = try tier_a.appendCommit(cache_a.entries, cache_a.step, cache_a.config, &tokens_a, false, null, s);
        _ = try tier_b.appendCommit(cache_b.entries, cache_b.step, cache_b.config, &tokens_b, false, null, s);
        const tiers = [_]*DiskTier{ &tier_a, &tier_b };
        const srcs = [_]*KVCache{ &cache_a, &cache_b };
        const prompts = [_][]const u32{ &tokens_a, &tokens_b };
        for (tiers, srcs, prompts) |tier, src, prompt| {
            const m = tier.bestMatch(prompt, false, kv_quant.KVQuantConfig.dense, .{}) orelse return error.TestExpectedMatch;
            var out = try KVCache.init(testing.allocator, 1);
            defer out.deinit();
            _ = try tier.restoreInto(&out, m.idx, s);
            try testing.expectEqual(try cacheValueAt(src, 0, 300, 3, s), try cacheValueAt(&out, 0, 300, 3, s));
        }
    }
    // The second tier's files leave with it.
    const names = try kv_disk_fs.testList(io, tier_a.root);
    defer kv_disk_fs.Fs.freeEntries(testing.allocator, names);
    for (names) |dent| {
        if (dent.is_dir and !std.mem.eql(u8, dent.name, "v")) try testing.expectEqualStrings("e1", dent.name);
    }
}

test "DiskTier: a token record that differs from the index poisons the entry and restores nothing" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-ident", 0, 128);
    defer tier.deinit();
    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try fillCache(&cache, s, 1, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
    // Another writer's record of the same length, under the index this tier read.
    var other: [600 * 4]u8 = undefined;
    for (0..600) |i| std.mem.writeInt(u32, other[i * 4 ..][0..4], @intCast(i + 900_000), .little);
    try kv_disk_fs.testWriteAt(io, base, "fp-ident/e1/tokens.bin", &other);

    var out = try KVCache.init(testing.allocator, 1);
    defer out.deinit();
    try testing.expectError(error.DiskCacheTokenMismatch, tier.restoreInto(&out, 0, s));
    try testing.expect(tier.entries.items[0].poisoned);
    try testing.expect(tier.bestMatch(&tokens, false, kv_quant.KVQuantConfig.dense, .{}) == null);
}

test "DiskTier: a restore that finds a chunk file missing or short drops the entry so the prompt stores again" {
    mlx.installErrorHandler();
    const io = std.testing.io;
    const s = mlx.gpuStream();
    for ([_]bool{ true, false }) |delete| {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const base = try tmpRoot(&tmp, io, &buf);

        var tier = try DiskTier.init(testing.allocator, io, base, "fp-gone", 0, 128);
        defer tier.deinit();
        var cache = try KVCache.init(testing.allocator, 1);
        defer cache.deinit();
        try fillCache(&cache, s, 1, 600, 8, 0.0, .float32);
        var tokens: [600]u32 = undefined;
        for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
        _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
        if (delete) {
            kv_disk_fs.testDeleteAt(io, base, "fp-gone/e1/c000001.safetensors");
        } else {
            try kv_disk_fs.testWriteAt(io, base, "fp-gone/e1/c000001.safetensors", "short");
        }

        var out = try KVCache.init(testing.allocator, 1);
        defer out.deinit();
        try testing.expect(std.meta.isError(tier.restoreInto(&out, 0, s)));
        try testing.expect(tier.bestMatch(&tokens, false, kv_quant.KVQuantConfig.dense, .{}) == null);

        _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
        var again = try KVCache.init(testing.allocator, 1);
        defer again.deinit();
        const m = tier.bestMatch(&tokens, false, kv_quant.KVQuantConfig.dense, .{}) orelse return error.TestExpectedMatch;
        _ = try tier.restoreInto(&again, m.idx, s);
    }
}

test "DiskTier: a second tier in the same process sweeps around a live tier's root" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var first = try DiskTier.init(testing.allocator, io, base, "fp-first", 0, 128);
    defer first.deinit();
    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try fillCache(&cache, s, 1, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    _ = try first.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);

    var second = try DiskTier.init(testing.allocator, io, base, "fp-second", 0, 128);
    defer second.deinit();
    second.ssd_first = true;
    // No room after the reserve: the probe keeps the zero budget, so every sibling byte is over it.
    second.armTestSpace(0, 2048 * 1024 * 1024 * 1024);
    second.max_bytes = 0;
    second.sweepSiblings();
    try testing.expect(kv_disk_fs.testStatAt(io, base, "fp-first/e1/meta.json") != null);
}

test "DiskTier: a new entry never adopts an existing entry directory" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-claim", 0, 128);
    defer tier.deinit();
    try kv_disk_fs.testMkdirAt(io, base, "fp-claim/e1");
    try kv_disk_fs.testWriteAt(io, base, "fp-claim/e1/c000000.safetensors", "not ours");
    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try fillCache(&cache, s, 1, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);

    try testing.expect(tier.entries.items[0].id != 1);
    const kept = (kv_disk_fs.testReadAt(io, base, "fp-claim/e1/c000000.safetensors", 64) orelse return error.TestUnexpectedResult);
    defer testing.allocator.free(kept);
    try testing.expectEqualStrings("not ours", kept);
}

test "DiskTier: affine-quant cache round-trips all six buffers, each width only into its own width" {
    for ([_]u8{ 4, 6, 8 }) |bits| try affineDiskRoundTrip(bits);
}

fn affineDiskRoundTrip(bits: u8) !void {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-q", 0, 128);
    defer tier.deinit();

    const qcfg = kv_quant.KVQuantConfig.affine(bits);
    var cache = try KVCache.initWithConfig(testing.allocator, 2, qcfg);
    defer cache.deinit();
    // head_dim must be a multiple of group_size (64) for affine quant.
    try fillCache(&cache, s, 2, 520, 64, 0.0, .bfloat16);
    try testing.expectEqual(@as(usize, 520), cache.step);

    var tokens: [520]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);

    // The restart path reads the manifest back: it must still name this width.
    tier.deinit();
    tier = try DiskTier.init(testing.allocator, io, base, "fp-q", 0, 128);
    try testing.expectEqual(@as(usize, 1), tier.entryCount());
    for ([_]u8{ 4, 6, 8 }) |other| {
        if (other != bits) try testing.expect(tier.bestMatch(&tokens, false, kv_quant.KVQuantConfig.affine(other), .{}) == null);
    }
    try testing.expect(tier.bestMatch(&tokens, false, kv_quant.KVQuantConfig.dense, .{}) == null);

    var cache2 = try KVCache.initWithConfig(testing.allocator, 2, qcfg);
    defer cache2.deinit();
    const m = tier.bestMatch(&tokens, false, qcfg, .{}).?;
    try testing.expectEqual(@as(u32, 520), m.usable);
    const restored = try tier.restoreInto(&cache2, m.idx, s);
    try testing.expectEqual(@as(u32, 520), restored);

    // Dense read-back through the cache's own dequant path must agree.
    // Truncate BOTH caches to the same length first — restore leaves views
    // empty (the KVCache.restore contract) and truncate to len < offset
    // rebuilds them on both sides identically.
    try cache.truncate(519, s);
    try cache2.truncate(519, s);
    var v1 = try cache.denseView(0, s);
    defer v1.deinit();
    var v2 = try cache2.denseView(0, s);
    defer v2.deinit();
    const probes = [_][2]u32{ .{ 0, 0 }, .{ 127, 63 }, .{ 128, 0 }, .{ 300, 5 }, .{ 511, 1 }, .{ 518, 63 } };
    for (probes) |p| {
        var d1 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(d1);
        var d2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(d2);
        const st = [_]c_int{ 0, 0, @intCast(p[0]), @intCast(p[1]) };
        const sp = [_]c_int{ 1, 1, @intCast(p[0] + 1), @intCast(p[1] + 1) };
        const sd = [_]c_int{ 1, 1, 1, 1 };
        try mlx.check(mlx.mlx_slice(&d1, v1.k, &st, 4, &sp, 4, &sd, 4, s));
        try mlx.check(mlx.mlx_slice(&d2, v2.k, &st, 4, &sp, 4, &sd, 4, s));
        var f1 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(f1);
        var f2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(f2);
        try mlx.check(mlx.mlx_astype(&f1, d1, .float32, s));
        try mlx.check(mlx.mlx_astype(&f2, d2, .float32, s));
        _ = mlx.mlx_array_eval(f1);
        _ = mlx.mlx_array_eval(f2);
        try testing.expectEqual(mlx.mlx_array_data_float32(f1).?[0], mlx.mlx_array_data_float32(f2).?[0]);
    }
}

test "DiskTier: truncated chunk file salvages the valid prefix at scan (kill -9 shape)" {
    // A kill -9 mid-flush leaves a chunk file truncated while meta.json (the
    // commit point, written last) still describes the PREVIOUS valid state —
    // whose recorded size for that chunk no longer matches the file. Live
    // capture: MLX "invalid data offsets exceeding the size of the file" on
    // restore. The scan must clamp the entry to the last contiguous chunk
    // whose size matches meta, salvaging the prefix instead of poisoning a
    // restore (or dropping everything).
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    {
        var tier = try DiskTier.init(testing.allocator, io, base, "fp-trunc", 0, 128);
        defer tier.deinit();
        var cache = try KVCache.init(testing.allocator, 1);
        defer cache.deinit();
        try fillCache(&cache, s, 1, 700, 8, 0.0, .float32);
        var tokens: [700]u32 = undefined;
        for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
        _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
    }

    // Truncate chunk 4 (positions [512, 640)) — chunks 0-3 stay valid.
    try kv_disk_fs.testWriteAt(io, base, "fp-trunc/e1/c000004.safetensors", "trunc");

    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-trunc", 0, 128);
    defer tier2.deinit();
    try testing.expectEqual(@as(usize, 1), tier2.entryCount());
    // kv_len clamped to the last valid chunk boundary: 4 * 128 = 512.
    try testing.expectEqual(@as(u32, 512), tier2.entries.items[0].kv_len);

    // The salvaged prefix restores cleanly.
    var cache2 = try KVCache.init(testing.allocator, 1);
    defer cache2.deinit();
    const restored = try tier2.restoreInto(&cache2, 0, s);
    try testing.expectEqual(@as(u32, 512), restored);
    try testing.expectEqual(@as(usize, 512), cache2.step);
}

test "DiskTier: flush byte cap persists incrementally across commits" {
    // A 4 GB first-commit write used to stall the NEXT request ~2.5 s (the
    // flush runs on the inference thread). appendCommit caps the bytes
    // written per call at max_flush_bytes, persists a chunk-aligned prefix,
    // and reports incomplete so the caller re-flushes on later turns.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-cap", 0, 128);
    defer tier.deinit();
    tier.max_flush_bytes = 1; // every chunk write exceeds the cap -> 1 chunk/flush

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try fillCache(&cache, s, 1, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // First flush: 1 chunk (128 tokens), incomplete.
    const c1 = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
    try testing.expectEqual(PersistOutcome.partial, c1);
    try testing.expectEqual(@as(u32, 128), tier.entries.items[0].kv_len);
    // Second flush continues from where it left off.
    const c2 = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
    try testing.expectEqual(PersistOutcome.partial, c2);
    try testing.expectEqual(@as(u32, 256), tier.entries.items[0].kv_len);
    // Keep flushing until complete; entry must land at the full 600.
    var guard: u32 = 0;
    while (guard < 10) : (guard += 1) {
        if (try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s) == .persisted) break;
    }
    try testing.expectEqual(@as(u32, 600), tier.entries.items[0].kv_len);

    // Restored content from an incrementally-persisted entry is exact.
    var cache2 = try KVCache.init(testing.allocator, 1);
    defer cache2.deinit();
    const restored = try tier.restoreInto(&cache2, 0, s);
    try testing.expectEqual(@as(u32, 600), restored);
    const want = try cacheValueAt(&cache, 0, 599, 7, s);
    const got = try cacheValueAt(&cache2, 0, 599, 7, s);
    try testing.expectEqual(want, got);
}

test "DiskTier: cache ahead of the token record persists the clamped prefix (EOS-stop shape)" {
    // On an EOS stop the generator has forwarded the terminator tokens into
    // the cache but they're not part of the committed token record — live
    // capture: step=2054 vs tokens=2052. The RAM tier tolerates this
    // (truncate hides the tail); the disk tier must persist min(step,
    // tokens.len) positions instead of silently skipping the whole commit.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-eos", 0, 128);
    defer tier.deinit();

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try fillCache(&cache, s, 1, 604, 8, 0.0, .float32); // 2 positions past the record
    var tokens: [602]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
    try testing.expectEqual(@as(usize, 1), tier.entryCount());
    try testing.expectEqual(@as(u32, 602), tier.entries.items[0].kv_len);

    var cache2 = try KVCache.init(testing.allocator, 1);
    defer cache2.deinit();
    const restored = try tier.restoreInto(&cache2, 0, s);
    try testing.expectEqual(@as(u32, 602), restored);
    const want = try cacheValueAt(&cache, 0, 601, 3, s);
    const got = try cacheValueAt(&cache2, 0, 601, 3, s);
    try testing.expectEqual(want, got);
}

// ── Phase 3: hybrid SSM checkpoint persistence ──


test "DiskTier: SSM retention thins the interior, keeping both ends" {
    // Every turn adds an end-of-prompt checkpoint; unbounded, one entry grows
    // without limit. Retention keeps at most SSM_DISK_MAX_PER_ENTRY, thinning
    // the interior (#330): front-thinning end-anchored the survivors.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-retain", 0, 128);
    defer tier.deinit();
    // The span-preserving policy is the long-context archs'; the tier's default is drop-oldest (below).
    tier.cp_thin = .min_span_recency;
    tier.ssm_max_per_entry = SSM_DISK_MAX_PER_ENTRY;

    // KV covering 0..(N*100) so every checkpoint position is ≤ kv_len.
    const N = SSM_DISK_MAX_PER_ENTRY + 1; // 9 positions, one over the cap
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, N * 100 + 50, 8, 0.0, .float32);
    var tokens: [SSM_DISK_MAX_PER_ENTRY * 100 + 150]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    var srcs: [N][3]SSMCacheEntry = undefined;
    for (&srcs, 0..) |*src, i| src.* = buildHybridEntries(s, @floatFromInt((i + 1) * 1000), @floatFromInt((i + 1) * 2000));
    defer for (&srcs) |*src| freeHybridEntries(src);
    var cps: [N]trunk.SSMCheckpoint = undefined;
    for (&cps, 0..) |*cp, i| cp.* = try trunk.captureSsmCheckpoint(testing.allocator, &srcs[i], (i + 1) * 100, s);
    defer for (&cps) |*cp| cp.deinit(testing.allocator);

    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &cps, s);

    const e = &tier.entries.items[0];
    try testing.expectEqual(@as(usize, SSM_DISK_MAX_PER_ENTRY), e.ssm_positions.len);
    try testing.expectEqual(@as(u32, 100), e.ssm_positions[0]);
    try testing.expectEqual(@as(u32, @intCast(N * 100)), e.ssm_positions[e.ssm_positions.len - 1]);
    try testing.expect(std.mem.indexOfScalar(u32, e.ssm_positions, 200) == null);
    // The dropped position's file is gone.
    const dropped = try std.fmt.allocPrint(testing.allocator, "{s}/fp-retain/e1/s0000200.safetensors", .{base});
    defer testing.allocator.free(dropped);
    try testing.expect(kv_disk_fs.testStat(io, dropped) == null);
    const kept = try std.fmt.allocPrint(testing.allocator, "{s}/fp-retain/e1/s0000100.safetensors", .{base});
    defer testing.allocator.free(kept);
    try testing.expect(kv_disk_fs.testStat(io, kept) != null);
}

test "DiskTier: a 130-point entry commits whole, with no count thinning" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-130", 0, 128);
    defer tier.deinit();
    tier.cp_thin = .min_span_recency;
    tier.ssm_max_per_entry = SSM_DISK_MAX_PER_ENTRY;
    const N = 130;
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, N * 100 + 50, 8, 0.0, .float32);
    var tokens: [N * 100 + 50]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var src = buildHybridEntries(s, 1000, 2000);
    defer freeHybridEntries(&src);
    var cps: [N]trunk.SSMCheckpoint = undefined;
    for (&cps, 0..) |*cp, i| cp.* = try trunk.captureSsmCheckpoint(testing.allocator, &src, (i + 1) * 100, s);
    defer for (&cps) |*cp| cp.deinit(testing.allocator);

    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &cps, s);
    try testing.expectEqual(@as(usize, N), tier.entries.items[0].ssm_positions.len);
}

test "DiskTier: one conversation over the byte budget drops its oldest restore points, keeping the newest grid point and the prompt end" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-sole", 0, 128);
    defer tier.deinit();
    tier.cp_thin = .min_span_recency;
    tier.ssm_max_per_entry = SSM_DISK_MAX_PER_ENTRY;
    const len = 3 * POINT_GRID + 74;
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, len, 8, 0.0, .float32);
    var tokens: [len]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var src = buildHybridEntries(s, 1000, 2000);
    defer freeHybridEntries(&src);
    const positions = [_]usize{ 1000, 2000, 3000, POINT_GRID, 2 * POINT_GRID, 3 * POINT_GRID, 3 * POINT_GRID + 24 };
    var cps: [positions.len]trunk.SSMCheckpoint = undefined;
    for (&cps, positions) |*cp, p| cp.* = try trunk.captureSsmCheckpoint(testing.allocator, &src, p, s);
    defer for (&cps) |*cp| cp.deinit(testing.allocator);

    // Room for the rows and three of the seven points.
    const point = tier.pointFileBytes(&cps[0]);
    tier.max_bytes = tier.entryBytesAt(cache.entries, len, null, &.{}, 0) + 3 * point + point / 2;
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &cps, s);

    const e = &tier.entries.items[0];
    try testing.expectEqual(@as(u32, len), e.kv_len);
    try testing.expectEqualSlices(u32, &.{ 2 * POINT_GRID, 3 * POINT_GRID, 3 * POINT_GRID + 24 }, e.ssm_positions);
    try testing.expect(tier.total_bytes <= tier.max_bytes);
    const gone = try std.fmt.allocPrint(testing.allocator, "{s}/fp-sole/e1/s0000001000.safetensors", .{base});
    defer testing.allocator.free(gone);
    try testing.expect(kv_disk_fs.testStat(io, gone) == null);

    // Rows alone over the budget fall back to the old cut: the highest point whose record fits.
    var tight = try DiskTier.init(testing.allocator, io, base, "fp-rows", 0, 128);
    defer tight.deinit();
    tight.cp_thin = .min_span_recency;
    tight.ssm_max_per_entry = SSM_DISK_MAX_PER_ENTRY;
    tight.max_bytes = tight.entryBytesAt(cache.entries, 2 * POINT_GRID, cps[0..5], &.{}, 0);
    _ = try tight.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &cps, s);
    try testing.expect(tight.entries.items[0].kv_len <= 2 * POINT_GRID);
}

test "DiskTier: SSM salvage — one bad file drops that position, all bad drops the entry" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    {
        var tier = try DiskTier.init(testing.allocator, io, base, "fp-ssmsalv", 0, 128);
        defer tier.deinit();
        var cache = try KVCache.init(testing.allocator, 3);
        defer cache.deinit();
        try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
        var tokens: [600]u32 = undefined;
        for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
        var src128 = buildHybridEntries(s, 100.0, 500.0);
        defer freeHybridEntries(&src128);
        var src256 = buildHybridEntries(s, 200.0, 600.0);
        defer freeHybridEntries(&src256);
        var cps = [_]trunk.SSMCheckpoint{
            try trunk.captureSsmCheckpoint(testing.allocator, &src128, 128, s),
            try trunk.captureSsmCheckpoint(testing.allocator, &src256, 256, s),
        };
        defer for (&cps) |*cp| cp.deinit(testing.allocator);
        _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &cps, s);
    }

    // Truncate the pos-256 checkpoint file → that position drops, 128 survives.
    try kv_disk_fs.testWriteAt(io, base, "fp-ssmsalv/e1/s0000256.safetensors", "trunc");
    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-ssmsalv", 0, 128);
    defer tier2.deinit();
    try testing.expectEqual(@as(usize, 1), tier2.entryCount());
    try testing.expectEqual(@as(usize, 1), tier2.entries.items[0].ssm_positions.len);
    try testing.expectEqual(@as(u32, 128), tier2.entries.items[0].ssm_positions[0]);
    // The salvaged KV + surviving checkpoint still restore.
    var cache2 = try KVCache.init(testing.allocator, 3);
    defer cache2.deinit();
    var dst: [3]SSMCacheEntry = .{
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
    };
    defer freeHybridEntries(&dst);
    try testing.expectEqual(@as(u32, 128), try tier2.restoreIntoHybrid(&cache2, &dst, 0, 128, s));

    // Truncate the LAST surviving checkpoint too → hybrid entry dropped whole
    // (KV without any SSM state is unusable).
    try kv_disk_fs.testWriteAt(io, base, "fp-ssmsalv/e1/s0000128.safetensors", "trunc");
    var tier3 = try DiskTier.init(testing.allocator, io, base, "fp-ssmsalv", 0, 128);
    defer tier3.deinit();
    try testing.expectEqual(@as(usize, 0), tier3.entryCount());
}

test "DiskTier: a fresh tier over the same root ranks by the HIGHEST restorable checkpoint" {
    // A restart must restore the highest checkpoint at or below the match, not
    // the first one the manifest lists.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    const N = 8;
    var tokens: [N * 128]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    // One token past the stored prefix, so the full-length checkpoint still leaves a token to compute logits.
    var prompt: [N * 128 + 1]u32 = undefined;
    @memcpy(prompt[0 .. N * 128], &tokens);
    prompt[N * 128] = 1;

    {
        var tier = try DiskTier.init(testing.allocator, io, base, "fp-coldrank", 0, 128);
        defer tier.deinit();
        var cache = try KVCache.init(testing.allocator, 3);
        defer cache.deinit();
        try fillCache(&cache, s, 3, N * 128, 8, 0.0, .float32);
        var srcs: [N][3]SSMCacheEntry = undefined;
        for (&srcs, 0..) |*src, i| src.* = buildHybridEntries(s, @floatFromInt((i + 1) * 1000), @floatFromInt((i + 1) * 2000));
        defer for (&srcs) |*src| freeHybridEntries(src);
        var cps: [N]trunk.SSMCheckpoint = undefined;
        for (&cps, 0..) |*cp, i| cp.* = try trunk.captureSsmCheckpoint(testing.allocator, &srcs[i], (i + 1) * 128, s);
        defer for (&cps) |*cp| cp.deinit(testing.allocator);
        _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &cps, s);
        try testing.expectEqual(@as(usize, N), tier.entries.items[0].ssm_positions.len);
        const warm = tier.bestHybridMatch(&prompt, false, cache.config, prompt.len, .{}).?;
        try testing.expectEqual(@as(u32, N * 128), warm.cp);
    }

    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-coldrank", 0, 128);
    defer tier2.deinit();
    try testing.expectEqual(@as(usize, 1), tier2.entryCount());
    try testing.expectEqual(@as(usize, N), tier2.entries.items[0].ssm_positions.len);
    const cold = tier2.bestHybridMatch(&prompt, false, kv_quant.KVQuantConfig.dense, prompt.len, .{}).?;
    try testing.expectEqual(@as(u32, N * 128), cold.cp);
    try testing.expectEqual(@as(u32, N * 128), cold.usable);
}

test "DiskTier hybrid lookup leaves a prompt token to compute logits" {
    var tokens: [513]u32 = undefined;
    for (&tokens, 0..) |*tok, i| tok.* = @intCast(i);
    var positions = [_]u32{ 256, 512 };
    var entries = [_]IndexEntry{.{
        .id = 1,
        .tokens = &tokens,
        .kv_len = tokens.len,
        .has_tools = false,
        .quant = kv_quant.KVQuantConfig.dense,
        .bytes = 0,
        .chunk_bytes = &.{},
        .ssm_positions = &positions,
        .ssm_bytes = &.{},
        .last_used = 0,
    }};
    var entry_list: std.ArrayList(IndexEntry) = .empty;
    defer entry_list.deinit(testing.allocator);
    try entry_list.appendSlice(testing.allocator, &entries);
    var stats: write_stats.Stats = .{};
    const tier = DiskTier{
        .allocator = testing.allocator,
        .stats = &stats,
        .io = testing.io,
        .root = &.{},
        .max_bytes = 0,
        .chunk_tokens = 128,
        .entries = entry_list,
        .next_id = 2,
        .total_bytes = 0,
        .counter = 0,
        .fs = .{ .allocator = testing.allocator, .io = testing.io, .root = &.{}, .vfs = undefined },
    };
    const quant = kv_quant.KVQuantConfig.dense;
    try testing.expectEqual(@as(u32, 256), tier.bestHybridMatch(tokens[0..512], false, quant, 512, .{}).?.cp);
    try testing.expectEqual(@as(u32, 512), tier.bestHybridMatch(&tokens, false, quant, 513, .{}).?.cp);
    try testing.expectEqual(@as(u32, 512), tier.bestHybridMatch(&tokens, false, quant, 512, .{}).?.cp);
    try testing.expect(tier.bestHybridMatch(tokens[0..256], false, quant, 256, .{}) == null);
    try testing.expect(tier.bestHybridMatch(&.{}, false, quant, 0, .{}) == null);
}

test "DiskTier: a restore wider than the fd limit closes each chunk as it goes" {
    // Restoring more chunk files than the soft RLIMIT_NOFILE allows open at once succeeds.
    mlx.installErrorHandler();
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    // 600 tokens at 8 per chunk = 75 chunk files.
    var tier = try DiskTier.init(testing.allocator, io, base, "fp-fdlimit", 0, 8);
    defer tier.deinit();
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 11);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
    const m = tier.bestMatch(&tokens, false, kv_quant.KVQuantConfig.dense, .{}).?;
    tier.drainEntry(tier.entries.items[m.idx].id);

    const saved = try std.posix.getrlimit(.NOFILE);
    defer std.posix.setrlimit(.NOFILE, saved) catch {};
    var top: usize = 0;
    for (0..@min(saved.cur, 4096)) |fd| {
        if (std.c.fcntl(@intCast(fd), std.c.F.GETFD) != -1) top = fd;
    }
    try std.posix.setrlimit(.NOFILE, .{ .cur = top + 1 + 16, .max = saved.max });

    var cache2 = try KVCache.init(testing.allocator, 3);
    defer cache2.deinit();
    const restored = tier.restoreInto(&cache2, m.idx, s);
    std.posix.setrlimit(.NOFILE, saved) catch {};
    try testing.expectEqual(@as(u32, 600), try restored);
    for ([_]u32{ 0, 7, 8, 333, 599 }) |pos| {
        try testing.expectEqual(try cacheValueAt(&cache, 2, pos, 5, s), try cacheValueAt(&cache2, 2, pos, 5, s));
    }
}

test "DiskTier: a failed restore drops the latch it raised and keeps a foreign one" {
    try mlx.requireGpu();
    // The caller's cold-prefill fallback must not inherit this restore's MLX error.
    mlx.installErrorHandler();
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-restore-latch", 0, 128);
    defer tier.deinit();
    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try fillCache(&cache, s, 2, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 13);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
    const m = tier.bestMatch(&tokens, false, kv_quant.KVQuantConfig.dense, .{}).?;
    const id = tier.entries.items[m.idx].id;
    tier.drainEntry(id);
    var path_buf: [1024]u8 = undefined;
    const chunk = try std.fmt.bufPrint(&path_buf, "{s}/e{d}/c000002.safetensors", .{ tier.root, id });
    kv_disk_fs.testDelete(io, chunk);

    var target = try KVCache.init(testing.allocator, 2);
    defer target.deinit();
    try testing.expectError(error.MlxError, tier.restoreInto(&target, m.idx, s));
    try testing.expect(!mlx.errorPending());

    mlx.latchErrorForTest("foreign pre-existing error");
    _ = tier.restoreInto(&target, m.idx, s) catch {};
    var msg: [512]u8 = undefined;
    try testing.expectEqualStrings("foreign pre-existing error", mlx.takeError(&msg).?);
}

test "DiskTier: a restore fills its buffers chunk by chunk, never holding every chunk beside the result" {
    // The restore runs before the inference thread's admission bill sees it; a concatenation at
    // the end held every loaded chunk beside the result, twice the restored KV.
    mlx.installErrorHandler();
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-restore-peak", 0, 128);
    defer tier.deinit();
    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try fillCache(&cache, s, 2, 4096, 256, 0.0, .float32);
    var tokens: [4096]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 17);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
    const m = tier.bestMatch(&tokens, false, kv_quant.KVQuantConfig.dense, .{}).?;
    tier.drainEntry(tier.entries.items[m.idx].id);

    // Two layers of K and V, 4096 rows of 256 f32: 16 MiB restored.
    const restored_bytes: u64 = 2 * 2 * 4096 * 256 * 4;
    // The stream is drained around each restore and the tier writes on this thread, so the process-wide
    // peak is the restore's own: every restore must stay under the bound.
    var excess: usize = 0;
    var cache2 = try KVCache.init(testing.allocator, 2);
    defer cache2.deinit();
    for (0..4) |_| {
        cache2.deinit();
        cache2 = try KVCache.init(testing.allocator, 2);
        _ = mlx.mlx_synchronize(s);
        _ = mlx.mlx_clear_cache();
        var before: usize = 0;
        _ = mlx.mlx_get_active_memory(&before);
        _ = mlx.mlx_reset_peak_memory();
        try testing.expectEqual(@as(u32, 4096), try tier.restoreInto(&cache2, m.idx, s));
        _ = mlx.mlx_synchronize(s);
        var peak: usize = 0;
        _ = mlx.mlx_get_peak_memory(&peak);
        excess = @max(excess, peak -| before);
    }
    if (excess >= restored_bytes * 3 / 2) {
        std.debug.print("restore peak excess {d} B, restored {d} B\n", .{ excess, restored_bytes });
        return error.TestUnexpectedResult;
    }
    for ([_]u32{ 0, 127, 128, 2049, 4095 }) |pos| {
        try testing.expectEqual(try cacheValueAt(&cache, 1, pos, 5, s), try cacheValueAt(&cache2, 1, pos, 5, s));
        try testing.expectEqual(try cacheBufValueAt(&cache, 1, pos, 5, s, true), try cacheBufValueAt(&cache2, 1, pos, 5, s, true));
    }
}

test "DiskTier: a commit holding no KV rows persists nothing, even with checkpoints" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    var tier = try DiskTier.init(std.testing.allocator, io, base, "fp-no-rows", 0, 128);
    defer tier.deinit();
    // A cache whose state lives elsewhere (a request-held state): every layer uninitialized, step set.
    var cache = try KVCache.init(std.testing.allocator, 3);
    defer cache.deinit();
    cache.step = 2048;
    var entries = buildHybridEntries(s, 1, 100);
    defer freeHybridEntries(&entries);
    var cp = try trunk.captureSsmCheckpoint(std.testing.allocator, &entries, 2048, s);
    defer cp.deinit(std.testing.allocator);
    var tokens: [2048]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i);
    try std.testing.expectEqual(PersistOutcome.skipped, try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &.{cp}, s));
    try std.testing.expectEqual(@as(usize, 0), tier.entryCount());
    try std.testing.expectEqual(@as(u64, 0), tier.total_bytes);
}

test "DiskTier: where an arch's points hold its whole state, a commit holding no KV rows persists its points over its tokens" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    var tier = try DiskTier.init(std.testing.allocator, io, base, "fp-state-points", 0, 128);
    defer tier.deinit();
    tier.points_hold_state = true;
    // A cache whose state lives elsewhere (a request-held state): every layer uninitialized, step set.
    var cache = try KVCache.init(std.testing.allocator, 3);
    defer cache.deinit();
    cache.step = 2048;
    var entries = buildHybridEntries(s, 1, 100);
    defer freeHybridEntries(&entries);
    var cp = try trunk.captureSsmCheckpoint(std.testing.allocator, &entries, 2048, s);
    defer cp.deinit(std.testing.allocator);
    var tokens: [2048]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i);
    try std.testing.expect(try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &.{cp}, s) != PersistOutcome.skipped);
    try std.testing.expectEqual(@as(usize, 1), tier.entryCount());
    try std.testing.expectEqual(@as(u32, 2048), tier.entries.items[0].kv_len);
    try std.testing.expectEqual(@as(usize, 1), tier.entries.items[0].ssm_positions.len);
}

test "DiskTier: a thought point past the prompt end is the newest point and survives a budget squeeze with the grid point" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    const len = 3 * POINT_GRID + 144;
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, len, 8, 0.0, .float32);
    var tokens: [len]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var src = buildHybridEntries(s, 1000, 2000);
    defer freeHybridEntries(&src);
    // The grid, the prompt end 24 past the last grid point, the thought's closer 100 tokens into the reply.
    const prompt_end = 3 * POINT_GRID + 24;
    const thought = prompt_end + 100;
    const positions = [_]usize{ POINT_GRID, 2 * POINT_GRID, 3 * POINT_GRID, prompt_end, thought };
    var cps: [positions.len]trunk.SSMCheckpoint = undefined;
    for (&cps, positions) |*cp, p| cp.* = try trunk.captureSsmCheckpoint(testing.allocator, &src, p, s);
    defer for (&cps) |*cp| cp.deinit(testing.allocator);

    // Room for the rows and three points, then two: the oldest go first, the newest point and the grid point stay.
    for ([_]struct { room: u64, want: []const u32 }{
        .{ .room = 3, .want = &.{ 3 * POINT_GRID, prompt_end, thought } },
        .{ .room = 2, .want = &.{ 3 * POINT_GRID, thought } },
    }, 0..) |c, i| {
        var name: [16]u8 = undefined;
        var tier = try DiskTier.init(testing.allocator, io, base, try std.fmt.bufPrint(&name, "fp-thought{d}", .{i}), 0, 128);
        defer tier.deinit();
        tier.cp_thin = .min_span_recency;
        tier.ssm_max_per_entry = SSM_DISK_MAX_PER_ENTRY;
        const point = tier.pointFileBytes(&cps[0]);
        tier.max_bytes = tier.entryBytesAt(cache.entries, len, null, &.{}, 0) + c.room * point + point / 2;
        _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &cps, s);
        try testing.expectEqualSlices(u32, c.want, tier.entries.items[0].ssm_positions);
    }
}


fn restorePeak(seed: bool) !u64 {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);
    var tier = try DiskTier.init(testing.allocator, io, base, "fp-peak", 0, 1024);
    defer tier.deinit();
    const qcfg = kv_quant.KVQuantConfig.affine(8);
    const rows: u32 = 8001;
    {
        var cache = try KVCache.initWithConfig(testing.allocator, 1, qcfg);
        defer cache.deinit();
        try fillCache(&cache, s, 1, rows, 256, 0.0, .bfloat16);
        const tokens = try testing.allocator.alloc(u32, rows);
        defer testing.allocator.free(tokens);
        for (tokens, 0..) |*t, i| t.* = @intCast(i + 7);
        _ = try tier.appendCommit(cache.entries, cache.step, cache.config, tokens, false, null, s);
    }
    tier.drainWriter();
    var cache2 = try KVCache.initWithConfig(testing.allocator, 1, qcfg);
    defer cache2.deinit();
    const idx: usize = 0;
    try testing.expectEqual(rows, tier.entries.items[idx].kv_len);
    try mlx.check(mlx.mlx_synchronize(s));
    _ = mlx.mlx_clear_cache();
    if (seed) {
        var held: [24]mlx.mlx_array = undefined;
        for (&held, 0..) |*h, i| {
            h.* = mlx.mlx_array_new();
            const bytes: c_int = @as(c_int, ([_]c_int{ 127, 129, 5, 9 })[i / 6]) * 16384;
            try mlx.check(mlx.mlx_zeros(h, &[_]c_int{bytes}, 1, .uint8, s));
            try mlx.check(mlx.mlx_array_eval(h.*));
        }
        try mlx.check(mlx.mlx_synchronize(s));
        for (held) |h| _ = mlx.mlx_array_free(h);
    }
    var before: usize = 0;
    _ = mlx.mlx_get_active_memory(&before);
    _ = mlx.mlx_reset_peak_memory();
    try tier.restorePrefixInto(&cache2, idx, rows, s);
    try mlx.check(mlx.mlx_synchronize(s));
    var peak: usize = 0;
    _ = mlx.mlx_get_peak_memory(&peak);
    return peak -| before;
}

test "an SSD restore fills its KV in place when MLX's pool holds buffers a page past the restored rows" {
    try mlx.requireGpu();
    const clean = try restorePeak(false);
    const seeded = try restorePeak(true);
    // The fill writes each chunk into the buffer it made (comment above `restoreKvInto`'s loop): each of the six
    // buffers may take a free one a page larger, never a second copy of the restored rows.
    try testing.expect(seeded <= clean + 6 * 16384);
}

