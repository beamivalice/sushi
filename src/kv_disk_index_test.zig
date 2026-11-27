//! SSD prefix cache index, manifest, poisoning, write-counter, and budget/GC tests.

const std = @import("std");
const testing = std.testing;
const test_note = @import("test_note.zig");
const mlx = @import("mlx.zig");
const kv_quant = @import("kv_quant.zig");
const trunk = @import("trunk.zig");
const prefix_arch = @import("arch/prefix.zig");
const write_stats = @import("disk_write_stats.zig");
const kv_disk_fs = @import("kv_disk_fs.zig");
const disk = @import("kv_disk_cache.zig");
const util = @import("kv_disk_helpers_test.zig");

const KVCache = trunk.KVCache;
const SSMCacheEntry = trunk.SSMCacheEntry;
const DiskTier = disk.DiskTier;
const IndexEntry = disk.IndexEntry;
const SpecCommit = disk.SpecCommit;
const PersistOutcome = disk.PersistOutcome;
const MediaLimits = disk.MediaLimits;
const SSM_DISK_MAX_PER_ENTRY = disk.SSM_DISK_MAX_PER_ENTRY;
const SSM_DISK_MAX_PER_ENTRY_LEGACY = disk.SSM_DISK_MAX_PER_ENTRY_LEGACY;
const nonChunkBytes = disk.nonChunkBytes;
const layerArrays = disk.layerArrays;
const rowBytes = disk.rowBytes;
const sliceAxis = disk.sliceAxis;

const fillCache = util.fillCache;
const cacheValueAt = util.cacheValueAt;
const tmpRoot = util.tmpRoot;
const makeArange = util.makeArange;
const buildHybridEntries = util.buildHybridEntries;
const freeHybridEntries = util.freeHybridEntries;
const grow_divs = util.grow_divs;
const grow_row_bytes = util.grow_row_bytes;
const expectGrowPoint = util.expectGrowPoint;
const commitGrowPoints = util.commitGrowPoints;
const chunkShareTokens = util.chunkShareTokens;
const wroteKind = util.wroteKind;
const sumBytes = util.sumBytes;
const appendedOf = util.appendedOf;

test "DiskTier: an ssm/spec-only append bills the SPEC sidecar's byte delta" {
    // `appendSsmOnly` overwrote `e.spec_bytes` before taking its delta, so every commit landing
    // a spec sidecar onto a complete entry left `total_bytes` short by the sidecar (every arch).
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-specbill", 0, 128);

    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try fillCache(&cache, s, 2, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // Turn 1: the entry lands complete, with no spec sidecar.
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
    try testing.expectEqual(@as(usize, 1), tier.entries.items.len);
    try testing.expectEqual(@as(u64, 0), tier.entries.items[0].spec_bytes);
    const bytes_before = tier.entries.items[0].bytes;
    const total_before = tier.total_bytes;

    // Turn 2: same tokens, same KV, now carrying an MTP history snap (`appendSsmOnly`).
    var mtp = try KVCache.init(testing.allocator, 1);
    defer mtp.deinit();
    try fillCache(&mtp, s, 1, 590, 8, 9.5, .float32);
    _ = try tier.appendCommitWithSpec(
        cache.entries,
        cache.step,
        cache.config,
        &tokens,
        false,
        null,
        null,
        .{ .entries = mtp.entries, .step = mtp.step, .config = mtp.config, .base_pos = 0 },
        s,
    );

    const e = &tier.entries.items[0];
    try testing.expect(e.spec_bytes > 0); // the sidecar really was written
    try testing.expectEqual(bytes_before + e.spec_bytes, e.bytes);
    try testing.expectEqual(total_before + e.spec_bytes, tier.total_bytes);
    var own_chunks: u64 = 0;
    for (e.chunk_bytes[@min(e.inherited_chunks, e.chunk_bytes.len)..]) |b| own_chunks += b;
    try testing.expectEqual(nonChunkBytes(e) + own_chunks, e.bytes);

    // A rescan of the same root reaches the same total.
    const total_bytes = tier.total_bytes;
    tier.deinit();
    var rescanned = try DiskTier.init(testing.allocator, io, base, "fp-specbill", 0, 128);
    defer rescanned.deinit();
    try testing.expectEqual(@as(usize, 1), rescanned.entryCount());
    try testing.expectEqual(total_bytes, rescanned.total_bytes);
}

test "DiskTier: a media record stamps v10 and its key survives the round trip above i64" {
    // The version is a compatibility claim, so a record whose rows only its pixels explain must
    // claim a version an older reader refuses (it accepts 2..9): that reader drops the record
    // instead of handing a text prompt image rows.
    const t = std.testing;
    const io = std.testing.io;
    var toks = [_]u32{ 1, 2, 3 };
    var cbytes = [_]u64{4096};
    var spos = [_]u32{};
    var sbytes = [_]u64{};
    var e = IndexEntry{
        .id = 1,
        .tokens = &toks,
        .kv_len = 600,
        .has_tools = false,
        .quant = kv_quant.KVQuantConfig.dense,
        .bytes = 4096,
        .chunk_bytes = &cbytes,
        .ssm_positions = &spos,
        .ssm_bytes = &sbytes,
        .last_used = 1,
    };
    try t.expectEqual(@as(u8, 4), DiskTier.metaVersionFor(e));
    // A Wyhash chain key runs past i64, which is all a JSON number can hold.
    e.vision_key = 0xF123_4567_89AB_CDEF;
    e.media_start = 128;
    try t.expectEqual(@as(u8, 10), DiskTier.metaVersionFor(e));

    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(io, &buf)];
    var tier = try DiskTier.init(t.allocator, io, root, "fp-meta-v10", 0, 1024);
    defer tier.deinit();
    var out = std.ArrayList(u8).empty;
    defer out.deinit(t.allocator);
    try tier.renderMeta(&out, e);
    try t.expect(std.mem.indexOf(u8, out.items, "\"v\":10") != null);
    try t.expect(std.mem.indexOf(u8, out.items, "\"media\":{\"key\":\"f123456789abcdef\",\"start\":128}") != null);

    // ...and parses back as the same pair, where a JSON number would have dropped the record.
    var parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, out.items, .{});
    defer parsed.deinit();
    const mv = parsed.value.object.get("media").?;
    try t.expectEqual(e.vision_key, try std.fmt.parseInt(u64, mv.object.get("key").?.string, 16));
    try t.expectEqual(@as(i64, 128), mv.object.get("start").?.integer);
}

test "DiskTier: a media record is never matched on token equality alone" {
    // Placeholder ids are ordinary vocabulary entries, so a text prompt CAN match a media
    // record's tokens past its first image row. The rows there are the record's pixels.
    const t = std.testing;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(io, &buf)];
    var tier = try DiskTier.init(t.allocator, io, root, "fp-media-gate", 0, 1024);
    defer tier.deinit();
    var toks = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var cbytes = [_]u64{4096};
    var spos = [_]u32{};
    var sbytes = [_]u64{};
    try tier.entries.append(t.allocator, .{
        .id = 1,
        .tokens = try t.allocator.dupe(u32, &toks),
        .kv_len = 8,
        .has_tools = false,
        .quant = kv_quant.KVQuantConfig.dense,
        .bytes = 4096,
        .chunk_bytes = try t.allocator.dupe(u64, &cbytes),
        .ssm_positions = try t.allocator.dupe(u32, &spos),
        .ssm_bytes = try t.allocator.dupe(u64, &sbytes),
        .vision_key = 0xABCD,
        .media_start = 3,
        .last_used = 1,
    });

    // Unconstrained (a caller with no rule) still sees the whole prefix: the rule is the gate,
    // and `prefix_cache.MediaRule` never hands the tier an empty one.
    try t.expectEqual(@as(u32, 8), tier.bestMatch(&toks, false, kv_quant.KVQuantConfig.dense, .{}).?.usable);

    // A rule that bounds this record at its first image row holds the match there...
    const Bound = struct {
        fn limitFor(_: *const anyopaque, _: u64, start: ?u32) ?u32 {
            return start orelse MediaLimits.UNBOUNDED;
        }
    };
    const rule: MediaLimits = .{ .rule = .{ .ctx = &tier, .limitFor = Bound.limitFor } };
    try t.expectEqual(@as(u32, 3), tier.bestMatch(&toks, false, kv_quant.KVQuantConfig.dense, rule).?.usable);

    // ...and one that allows nothing drops the candidate entirely, rather than matching at 0.
    const Never = struct {
        fn limitFor(_: *const anyopaque, _: u64, _: ?u32) ?u32 {
            return null;
        }
    };
    try t.expect(tier.bestMatch(&toks, false, kv_quant.KVQuantConfig.dense, .{ .rule = .{ .ctx = &tier, .limitFor = Never.limitFor } }) == null);
}

test "DiskTier: the per-entry checkpoint cap is gated; a legacy tier keeps 8" {
    const t = std.testing;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var legacy = try DiskTier.init(testing.allocator, io, base, "fp-cap-legacy", 0, 128);
    defer legacy.deinit();
    try t.expectEqual(@as(usize, 8), SSM_DISK_MAX_PER_ENTRY_LEGACY);
    try t.expectEqual(SSM_DISK_MAX_PER_ENTRY_LEGACY, legacy.ssm_max_per_entry);

    var positions: [40]u32 = undefined;
    for (&positions, 0..) |*p, i| p.* = @intCast((i + 1) * 4096);
    const L: u32 = 40 * 4096;

    const old_kept = try legacy.ssmTargetPositions(&positions, &[_]trunk.SSMCheckpoint{}, L);
    defer testing.allocator.free(old_kept);
    try t.expectEqual(SSM_DISK_MAX_PER_ENTRY_LEGACY, old_kept.len);

    var gated = try DiskTier.init(testing.allocator, io, base, "fp-cap-gated", 0, 128);
    defer gated.deinit();
    gated.cp_thin = .min_span_recency;
    gated.ssm_max_per_entry = SSM_DISK_MAX_PER_ENTRY;
    const new_kept = try gated.ssmTargetPositions(&positions, &[_]trunk.SSMCheckpoint{}, L);
    defer testing.allocator.free(new_kept);
    try t.expectEqualSlices(u32, &positions, new_kept);
}

test "DiskTier: a failed background write INVALIDATES the entry it belonged to (no completion claim, restore misses)" {
    // The writer drops a failed blob, and `writeMeta` appends the index before a byte reaches
    // disk, so the RAM copy was evicted against a hole. The failed path names the entry.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-poison", 0, 128);
    defer tier.deinit();
    tier.ssd_first = true;
    tier.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
    tier.enableBackgroundWriter();
    try testing.expect(tier.writer != null);

    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try fillCache(&cache, s, 2, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // Hold the writer so the failure lands strictly after the commit.
    tier.writer.?.setPaused(true);
    defer tier.writer.?.setPaused(false);
    tier.writer.?.injectFailure("c000003", .write);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
    const dead_id = tier.entries.items[0].id;
    tier.writer.?.setPaused(false);
    tier.drainWriter();
    try testing.expect(tier.writeErrors() > 0);
    // meta.json rode the same FIFO and did land: the index still describes five whole chunks.
    try testing.expectEqual(@as(usize, 5), tier.entries.items[0].chunk_bytes.len);

    try testing.expectEqual(@as(usize, 1), tier.harvestWriteFailures());
    try testing.expect(tier.entries.items[0].poisoned);
    // The death mark is `poisoned` alone; the recorded sizes stay the bill removal credits.
    for (tier.entries.items[0].chunk_bytes) |b| try testing.expect(b > 0);
    try testing.expect(!tier.holdsFullPrefix(cache.entries, cache.step, &tokens, false, cache.config));
    try testing.expect(tier.fullPrefixEntryId(cache.entries, cache.step, &tokens, false, cache.config) == null);
    try testing.expect(tier.bestMatch(&tokens, false, cache.config, .{}) == null);
    try testing.expect(!tier.entryWholeOnDisk(dead_id));
    // Attribution happens once.
    try testing.expectEqual(@as(usize, 0), tier.harvestWriteFailures());

    // The next commit reclaims the dead directory and rebuilds from scratch.
    tier.writer.?.injectFailure(null, .write);
    const out = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
    try testing.expectEqual(PersistOutcome.persisted, out);
    tier.drainWriter();
    try testing.expectEqual(@as(usize, 1), tier.entryCount());
    const live = &tier.entries.items[0];
    try testing.expect(live.id != dead_id);
    try testing.expect(!live.poisoned);
    try testing.expect(tier.entryWholeOnDisk(live.id));
    try testing.expect(tier.holdsFullPrefix(cache.entries, cache.step, &tokens, false, cache.config));

    var back = try KVCache.init(testing.allocator, 2);
    defer back.deinit();
    try testing.expectEqual(@as(u32, 600), try tier.restoreInto(&back, 0, s));
}

test "DiskTier: a poisoned entry is invisible to the hybrid lookup too" {
    // Poisoning is the tier's "this entry is dead" mark; the hybrid arm is the lookup an SSM-checkpoint arch
    // actually takes, so a gap there makes the mark meaningless.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-poison-hyb", 0, 128);
    defer tier.deinit();

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var src = buildHybridEntries(s, 100.0, 500.0);
    defer freeHybridEntries(&src);
    var cps: [1]trunk.SSMCheckpoint = .{try trunk.captureSsmCheckpoint(testing.allocator, &src, 512, s)};
    defer for (&cps) |*cp| cp.deinit(testing.allocator);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &cps, s);

    try testing.expect(tier.bestHybridMatch(&tokens, false, cache.config, 600, .{}) != null);
    tier.entries.items[0].poisoned = true;
    try testing.expect(tier.bestMatch(&tokens, false, cache.config, .{}) == null);
    try testing.expect(tier.bestHybridMatch(&tokens, false, cache.config, 600, .{}) == null);
}

test "DiskTier: a manifest scalar that does not fit its field drops the record, never casts" {
    // meta.json is on disk and hand-editable; an unchecked @intCast of a corrupt scalar is
    // ReleaseFast UB.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-badmeta", 0, 128);
    defer tier.deinit();

    const qcfg = kv_quant.KVQuantConfig.affine(4);
    var cache = try KVCache.initWithConfig(testing.allocator, 2, qcfg);
    defer cache.deinit();
    try fillCache(&cache, s, 2, 520, 64, 0.0, .bfloat16);
    var mtp = try KVCache.init(testing.allocator, 1);
    defer mtp.deinit();
    try fillCache(&mtp, s, 1, 520, 8, 9.5, .float32);
    var tokens: [520]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    _ = try tier.appendCommitWithSpec(
        cache.entries,
        cache.step,
        cache.config,
        &tokens,
        false,
        null,
        null,
        .{ .entries = mtp.entries, .step = mtp.step, .config = mtp.config, .base_pos = 0 },
        s,
    );
    const id = tier.entries.items[0].id;
    const meta_path = try std.fmt.allocPrint(testing.allocator, "{s}/fp-badmeta/e{d}/meta.json", .{ base, id });
    defer testing.allocator.free(meta_path);
    const orig = kv_disk_fs.testRead(io, meta_path, 64 * 1024).?;
    defer testing.allocator.free(orig);
    if (tier.loadEntry(id)) |l| {
        var e = l.e;
        defer tier.freeIndexEntryOwned(&e);
        try testing.expect(e.spec_mtp != null);
    } else return error.TestUnexpectedResult;

    const cases = [_]struct { find: []const u8, replace: []const u8, entry_survives: bool }{
        .{ .find = "\"bits\":4", .replace = "\"bits\":4000", .entry_survives = false },
        .{ .find = "\"group_size\":64", .replace = "\"group_size\":68719476736", .entry_survives = false },
        .{ .find = "\"layers\":1", .replace = "\"layers\":68719476736", .entry_survives = true },
    };
    for (cases) |c| {
        const patched = try std.mem.replaceOwned(u8, testing.allocator, orig, c.find, c.replace);
        defer testing.allocator.free(patched);
        try testing.expect(!std.mem.eql(u8, patched, orig));
        try kv_disk_fs.testWrite(io, meta_path, patched);
        if (tier.loadEntry(id)) |l| {
            var e = l.e;
            defer tier.freeIndexEntryOwned(&e);
            try testing.expect(c.entry_survives);
            // The spec half is the only part an out-of-range spec scalar costs.
            try testing.expect(e.spec_mtp == null);
        } else {
            try testing.expect(!c.entry_survives);
        }
    }
}

test "DiskTier: a manifest whose media block names no boundary is refused without leaking what the scan read" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);
    var tier = try DiskTier.init(testing.allocator, io, base, "fp-badmedia", 0, 1024);
    defer tier.deinit();

    const dir = try std.fmt.allocPrint(testing.allocator, "{s}/e1", .{tier.root});
    defer testing.allocator.free(dir);
    try kv_disk_fs.testMkdir(io, dir);
    const tokens = try testing.allocator.alloc(u8, 600 * 4);
    defer testing.allocator.free(tokens);
    @memset(tokens, 1);
    const files = [_]struct { name: []const u8, data: []const u8 }{
        .{ .name = "tokens.bin", .data = tokens },
        .{ .name = "c000000.safetensors", .data = "0123456789abcdef" },
        .{ .name = "s0000512.safetensors", .data = "01234567" },
    };
    for (files) |f| {
        const path = try std.fmt.allocPrint(testing.allocator, "{s}/{s}", .{ dir, f.name });
        defer testing.allocator.free(path);
        try kv_disk_fs.testWrite(io, path, f.data);
    }
    const meta = try std.fmt.allocPrint(
        testing.allocator,
        "{{\"v\":10,\"kv_len\":600,\"tokens\":600,\"has_tools\":false,\"scheme\":\"off\",\"bits\":0,\"group_size\":0," ++
            "\"chunk_tokens\":{d},\"inherited_chunks\":0,\"bytes\":1,\"chunk_bytes\":[16]," ++
            "\"ssm\":[{{\"pos\":512,\"bytes\":8}}],\"media\":{{\"key\":\"f00d\"}}}}",
        .{tier.chunk_tokens},
    );
    defer testing.allocator.free(meta);
    const meta_path = try std.fmt.allocPrint(testing.allocator, "{s}/meta.json", .{dir});
    defer testing.allocator.free(meta_path);
    try kv_disk_fs.testWrite(io, meta_path, meta);

    // The same record with a boundary loads, so the files above are valid and only the media block refuses it.
    const whole = try std.mem.replaceOwned(u8, testing.allocator, meta, "\"key\":\"f00d\"", "\"key\":\"f00d\",\"start\":3");
    defer testing.allocator.free(whole);
    try kv_disk_fs.testWrite(io, meta_path, whole);
    if (tier.loadEntry(1)) |l| {
        var e = l.e;
        defer tier.freeIndexEntryOwned(&e);
        try testing.expectEqual(@as(?u32, 3), e.media_start);
    } else return error.TestUnexpectedResult;

    try kv_disk_fs.testWrite(io, meta_path, meta);
    try testing.expect(tier.loadEntry(1) == null);
}

test "DiskTier: entryWholeOnDisk stats what the index NAMES (a truncated chunk fails it with a clean writer)" {
    // A byte can go missing with no write error at all; one stat per chunk catches it.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-stat", 0, 128);
    defer tier.deinit();
    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try fillCache(&cache, s, 1, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
    const id = tier.entries.items[0].id;
    try testing.expect(tier.entryWholeOnDisk(id));

    // Truncate one chunk behind the tier's back.
    try kv_disk_fs.testWriteAt(io, base, "fp-stat/e1/c000002.safetensors", "short");
    try testing.expect(!tier.entryWholeOnDisk(id));
    kv_disk_fs.testDeleteAt(io, base, "fp-stat/e1/c000002.safetensors");
    try testing.expect(!tier.entryWholeOnDisk(id));
    try testing.expect(!tier.entryWholeOnDisk(id + 999));
}


test "DiskTier write counters: a cold persist counts its kinds, an exact reuse adds nothing, an extension counts the tail and the new chunks" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try fillCache(&cache, s, 2, 600, 8, 0.0, .float32);
    var dfl = try KVCache.init(testing.allocator, 2);
    defer dfl.deinit();
    try fillCache(&dfl, s, 2, 600, 8, 3.5, .float32);
    var tokens: [800]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    const dflash: SpecCommit = .{ .entries = dfl.entries, .step = dfl.step, .config = dfl.config, .base_pos = 0 };

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-wcount", 0, 128);
    defer tier.deinit();
    tier.enableBackgroundWriter();
    const phys0 = appendedOf(&tier);
    const start = tier.stats.snapshot().totalStored();
    try testing.expectEqual(PersistOutcome.persisted, try tier.appendCommitWithSpec(cache.entries, cache.step, cache.config, tokens[0..600], false, null, dflash, null, s));
    tier.drainWriter();

    const meta_path = try std.fmt.allocPrint(testing.allocator, "{s}/e{d}/meta.json", .{ tier.root, tier.entries.items[0].id });
    defer testing.allocator.free(meta_path);
    const meta_json = kv_disk_fs.testSize(io, meta_path) orelse return error.TestUnexpectedResult;
    const cold = tier.stats.snapshot();
    const e = &tier.entries.items[0];
    try testing.expect(e.spec_bytes > 0);
    try testing.expectEqual(sumBytes(e.chunk_bytes), cold.written[@backingInt(write_stats.Kind.kv)]);
    try testing.expectEqual(e.spec_bytes, cold.written[@backingInt(write_stats.Kind.spec)]);
    try testing.expectEqual(@as(u64, 600 * 4) + meta_json, cold.written[@backingInt(write_stats.Kind.meta)]);
    // What the layer stored is what it appended, every byte attributed: the guard on the store counters.
    try testing.expectEqual(appendedOf(&tier) - phys0, cold.totalStored() - start);
    for ([_]write_stats.Kind{ .ssm, .qsa, .ring }) |k| try testing.expectEqual(@as(u64, 0), wroteKind(&tier, k));
    try testing.expectEqual(@as(u64, 0), cold.failed);

    // The same commit again persists nothing.
    try testing.expectEqual(PersistOutcome.persisted, try tier.appendCommitWithSpec(cache.entries, cache.step, cache.config, tokens[0..600], false, null, dflash, null, s));
    tier.drainWriter();
    try testing.expectEqual(cold, tier.stats.snapshot());

    // 600 -> 800 tokens: chunks 0-3 are whole on disk; chunk 4 (the old partial tail) is written again with chunks 5 and 6.
    try fillCache(&cache, s, 2, 200, 8, 600.0, .float32);
    try testing.expectEqual(PersistOutcome.persisted, try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s));
    tier.drainWriter();
    const ext = tier.stats.snapshot();
    const grown = &tier.entries.items[0];
    try testing.expectEqual(@as(usize, 7), grown.chunk_bytes.len);
    try testing.expectEqual(sumBytes(grown.chunk_bytes[4..]), ext.written[@backingInt(write_stats.Kind.kv)] - cold.written[@backingInt(write_stats.Kind.kv)]);
    try testing.expectEqual(appendedOf(&tier) - phys0, ext.totalStored() - start);
    try testing.expectEqual(cold.written[@backingInt(write_stats.Kind.spec)], ext.written[@backingInt(write_stats.Kind.spec)]);
    try testing.expectEqual(@as(u64, 0), ext.failed);
}

test "DiskTier write counters: a spec sidecar replaced by an extension counts its full bytes again" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try fillCache(&cache, s, 2, 600, 8, 0.0, .float32);
    var mtp = try KVCache.init(testing.allocator, 1);
    defer mtp.deinit();
    try fillCache(&mtp, s, 1, 590, 8, 9.5, .float32);
    var tokens: [800]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-wspec", 0, 128);
    defer tier.deinit();
    const first: SpecCommit = .{ .entries = mtp.entries, .step = mtp.step, .config = mtp.config, .base_pos = 0 };
    _ = try tier.appendCommitWithSpec(cache.entries, cache.step, cache.config, tokens[0..600], false, null, null, first, s);
    const spec_one = tier.entries.items[0].spec_bytes;
    try testing.expect(spec_one > 0);
    try testing.expectEqual(spec_one, wroteKind(&tier, .spec));

    try fillCache(&cache, s, 2, 200, 8, 600.0, .float32);
    try fillCache(&mtp, s, 1, 200, 8, 9.5 + 590 * 8, .float32);
    const second: SpecCommit = .{ .entries = mtp.entries, .step = mtp.step, .config = mtp.config, .base_pos = 0 };
    _ = try tier.appendCommitWithSpec(cache.entries, cache.step, cache.config, &tokens, false, null, null, second, s);
    const spec_two = tier.entries.items[0].spec_bytes;
    try testing.expect(spec_two > spec_one);
    try testing.expectEqual(spec_one + spec_two, wroteKind(&tier, .spec));
}

/// Block bytes the content-addressed layer has stored for `kind`.
fn storedData(tier: *DiskTier, kind: write_stats.Kind) u64 {
    return tier.fs.vfs.stats(kind).data_bytes;
}

/// Bytes a row adds across every array of `cache`'s initialized layers.
fn cacheRowBytes(cache: *const KVCache) u64 {
    var n: u64 = 0;
    for (cache.entries) |*e| {
        if (!e.initialized) continue;
        for (layerArrays(e)) |a| n += rowBytes(a);
    }
    return n;
}

fn specOf(cache: *const KVCache, base: usize) SpecCommit {
    return .{ .entries = cache.entries, .step = cache.step, .config = cache.config, .base_pos = base };
}

test "DiskTier row groups: a spec sidecar rewritten with more rows stores only the groups from the old partial one" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    for ([_]bool{ false, true }) |writer| {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const base = try tmpRoot(&tmp, io, &buf);
        var tier = try DiskTier.init(testing.allocator, io, base, "fp-spec-groups", 0, 128);
        defer tier.deinit();
        if (writer) tier.enableBackgroundWriter();

        var cache = try KVCache.init(testing.allocator, 2);
        defer cache.deinit();
        try fillCache(&cache, s, 2, 600, 8, 0.0, .float32);
        var mtp = try KVCache.init(testing.allocator, 1);
        defer mtp.deinit();
        try fillCache(&mtp, s, 1, 590, 8, 9.5, .float32);
        var dfl = try KVCache.init(testing.allocator, 2);
        defer dfl.deinit();
        try fillCache(&dfl, s, 2, 500, 8, 3.5, .float32);
        var tokens: [800]u32 = undefined;
        for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
        _ = try tier.appendCommitWithSpec(cache.entries, cache.step, cache.config, tokens[0..600], false, null, specOf(&dfl, 100), specOf(&mtp, 0), s);
        tier.drainWriter();
        const before = storedData(&tier, .spec);

        try fillCache(&cache, s, 2, 200, 8, 600.0 * 8, .float32);
        try fillCache(&mtp, s, 1, 200, 8, 9.5 + 590 * 8, .float32);
        try fillCache(&dfl, s, 2, 200, 8, 3.5 + 500 * 8, .float32);
        _ = try tier.appendCommitWithSpec(cache.entries, cache.step, cache.config, &tokens, false, null, specOf(&dfl, 100), specOf(&mtp, 0), s);
        tier.drainWriter();
        const e = &tier.entries.items[0];
        try testing.expectEqual(@as(u32, 790), e.spec_mtp.?.step);
        // MTP positions [0, 512) were whole groups of the old sidecar; DFlash positions [100, 512) were too, or the same
        // rows of its first group.
        const kept = 512 * cacheRowBytes(&mtp) + 412 * cacheRowBytes(&dfl);
        try testing.expectEqual(e.spec_bytes - kept, storedData(&tier, .spec) - before);
        try testing.expectEqual(@as(u64, 0), tier.writeErrors());
    }
}

/// `growCheckpoint`'s growing arrays add `grow_row_bytes` per 4 tokens over two layers: 24 bytes a row in each.
const grow_layer_row_bytes: u64 = (4 + 2) * 4;

test "DiskTier row groups: a point's growing rows and the segment cut from them store only the groups no earlier file held whole" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    for ([_]bool{ false, true }) |writer| {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const base = try tmpRoot(&tmp, io, &buf);
        var tier = try DiskTier.init(testing.allocator, io, base, "fp-grow-groups", 0, 128);
        defer tier.deinit();
        tier.cp_thin = .min_span_recency;
        tier.ssm_max_per_entry = 160;
        try tier.setGrownSpec(1024, &grow_divs, grow_row_bytes);
        if (writer) tier.enableBackgroundWriter();
        var tokens: [1300]u32 = undefined;
        for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
        var short = try KVCache.init(testing.allocator, 3);
        defer short.deinit();
        try fillCache(&short, s, 3, 600, 8, 0.0, .float32);

        try commitGrowPoints(&tier, &short, tokens[0..600], &.{300}, s);
        tier.drainWriter();
        var before = storedData(&tier, .ssm);
        try commitGrowPoints(&tier, &short, tokens[0..600], &.{ 300, 577 }, s);
        tier.drainWriter();
        var e = &tier.entries.items[0];
        try testing.expectEqualSlices(u32, &.{ 300, 577 }, e.ssm_positions);
        // Point 300 held rows [0, 128) of layer 1 (a row per 2 tokens) and [0, 64) of layer 2 (per 4) whole.
        try testing.expectEqual(e.ssm_bytes[1] - (128 + 64) * grow_layer_row_bytes, storedData(&tier, .ssm) - before);

        var long = try KVCache.init(testing.allocator, 3);
        defer long.deinit();
        try fillCache(&long, s, 3, 1300, 8, 0.0, .float32);
        before = storedData(&tier, .ssm);
        try commitGrowPoints(&tier, &long, &tokens, &.{1200}, s);
        tier.drainWriter();
        e = &tier.entries.items[0];
        try testing.expectEqual(@as(usize, 1), e.grown.len);
        try testing.expectEqualSlices(u32, &.{ 300, 577, 1200 }, e.ssm_positions);
        // Segment 1 holds tokens [0, 1024); point 577 held its first 256 rows of layer 1 and 128 of layer 2 whole.
        try testing.expectEqual(e.grown[0] + e.ssm_bytes[2] - (256 + 128) * grow_layer_row_bytes, storedData(&tier, .ssm) - before);
        for (e.ssm_positions) |p| try expectGrowPoint(&tier, e.id, p, s);
        try testing.expectEqual(@as(u64, 0), tier.writeErrors());
    }
}

test "DiskTier row groups: a ring file over rows an earlier one held stores only the groups that one did not hold whole" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);
    var tier = try DiskTier.init(testing.allocator, io, base, "fp-ring-groups", 0, 128);
    defer tier.deinit();
    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try fillCache(&cache, s, 2, 1500, 8, 0.0, .float32);
    for (cache.entries) |*en| en.ringed = true;
    const dir = try std.fmt.allocPrint(testing.allocator, "{s}/e1", .{tier.root});
    defer testing.allocator.free(dir);
    try tier.fs.mkdirp(dir);

    _ = try tier.writeRingFile(dir, .{ .pos = 1000, .rows = 800, .entries = cache.entries }, cache.config, s);
    const before = storedData(&tier, .ring);
    const size = try tier.writeRingFile(dir, .{ .pos = 1300, .rows = 800, .entries = cache.entries }, cache.config, s);
    // Rows [200, 1000), then [500, 1300): only positions [512, 768) are a whole group of both, and the two files'
    // headers are one (equal shapes).
    try testing.expectEqual((800 - 256) * cacheRowBytes(&cache), storedData(&tier, .ring) - before);
    try testing.expect(size > 800 * cacheRowBytes(&cache));

    var ring = try tier.loadRingFile(1, 1300, 2, cache.config, 128);
    defer ring.deinit();
    for (ring.entries, cache.entries) |*got, *want| {
        var rows = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(rows);
        try sliceAxis(&rows, want.keys.array, 2, 500, 1300, s);
        var eq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(eq);
        try mlx.check(mlx.mlx_array_equal(&eq, rows, got.keys.array, false, s));
        var same = false;
        try mlx.check(mlx.mlx_array_item_bool(&same, eq));
        try testing.expect(same);
    }
}

test "DiskTier write counters: a failed background write lands in failed, never in written" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-wfail", 0, 128);
    defer tier.deinit();
    tier.enableBackgroundWriter();
    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try fillCache(&cache, s, 2, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    tier.writer.?.setPaused(true);
    defer tier.writer.?.setPaused(false);
    tier.writer.?.injectFailure("c000003", .write);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
    const cb = tier.entries.items[0].chunk_bytes;
    tier.writer.?.setPaused(false);
    tier.drainWriter();
    try testing.expectEqual(cb[3], tier.stats.snapshot().failed);
    try testing.expectEqual(sumBytes(cb) - cb[3], wroteKind(&tier, .kv));
}

test "DiskTier write counters: the synchronous manifest write counts meta on success and failed on error" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-wmeta", 0, 128);
    defer tier.deinit();
    try testing.expect(tier.writer == null);
    var toks = [_]u32{ 1, 2, 3 };
    var cb = [_]u64{64};
    var no_pos = [_]u32{};
    var no_sz = [_]u64{};
    const e = IndexEntry{
        .id = 5,
        .tokens = &toks,
        .kv_len = 3,
        .has_tools = false,
        .quant = kv_quant.KVQuantConfig.dense,
        .bytes = 64,
        .chunk_bytes = &cb,
        .ssm_positions = &no_pos,
        .ssm_bytes = &no_sz,
        .last_used = 0,
    };
    try testing.expectError(error.FileNotFound, tier.writeMeta(e));
    const missing_dir = tier.stats.snapshot();
    try testing.expect(missing_dir.failed > 0);
    try testing.expectEqual(@as(u64, 0), missing_dir.totalWritten());

    try kv_disk_fs.testMkdirAt(io, base, "fp-wmeta/e5");
    try tier.writeMeta(e);
    const path = try std.fmt.allocPrint(testing.allocator, "{s}/e5/meta.json", .{tier.root});
    defer testing.allocator.free(path);
    const landed = tier.stats.snapshot();
    try testing.expectEqual(kv_disk_fs.testSize(io, path).?, landed.written[@backingInt(write_stats.Kind.meta)]);
    try testing.expectEqual(missing_dir.failed, landed.failed);
}

fn mediaCommit(tier: *DiskTier, cache: *const KVCache, tokens: []const u32, key: u64, start: ?u32, s: mlx.mlx_stream) !PersistOutcome {
    return tier.appendCommitWith(.{
        .kv_entries = cache.entries,
        .step = tokens.len,
        .config = cache.config,
        .tokens = tokens,
        .has_tools = false,
        .vision_key = key,
        .media_start = start,
    }, s, .{});
}

fn keyedKvLen(tier: *const DiskTier, key: u64) ?u32 {
    for (tier.entries.items) |*e| {
        if (e.vision_key == key) return e.kv_len;
    }
    return null;
}

test "DiskTier: a commit under other pixels gets its own record, never superseded by or extending another image's" {
    // Two images of one size render the same placeholder ids, so the token record cannot tell their rows apart.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);
    var tier = try DiskTier.init(testing.allocator, io, base, "fp-media-target", 0, 128);
    defer tier.deinit();
    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try fillCache(&cache, s, 2, 1200, 8, 0.0, .float32);
    var tokens: [1200]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    const ka: u64 = 0xA1A1_A1A1_A1A1_A1A1;
    const kb: u64 = 0xB2B2_B2B2_B2B2_B2B2;
    const kc: u64 = 0xC3C3_C3C3_C3C3_C3C3;

    _ = try mediaCommit(&tier, &cache, tokens[0..600], ka, 100, s);
    // B's longer commit would extend A's record.
    _ = try mediaCommit(&tier, &cache, &tokens, kb, 100, s);
    try testing.expectEqual(@as(usize, 2), tier.entryCount());
    // A's own commit is still superseded by A's record.
    _ = try mediaCommit(&tier, &cache, tokens[0..600], ka, 100, s);
    try testing.expectEqual(@as(usize, 2), tier.entryCount());
    // C's commit is covered by B's tokens, not by its rows.
    _ = try mediaCommit(&tier, &cache, tokens[0..600], kc, 100, s);
    try testing.expectEqual(@as(usize, 3), tier.entryCount());
    try testing.expectEqual(@as(?u32, 600), keyedKvLen(&tier, ka));
    try testing.expectEqual(@as(?u32, 1200), keyedKvLen(&tier, kb));
    try testing.expectEqual(@as(?u32, 600), keyedKvLen(&tier, kc));
}

test "DiskTier: a text record extends into the image turn after it, and that image's record extends under its own key" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);
    var tier = try DiskTier.init(testing.allocator, io, base, "fp-media-extend", 0, 128);
    defer tier.deinit();
    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try fillCache(&cache, s, 2, 1800, 8, 0.0, .float32);
    var tokens: [1800]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    const ka: u64 = 0xA1A1_A1A1_A1A1_A1A1;

    _ = try mediaCommit(&tier, &cache, tokens[0..600], 0, null, s);
    const id = tier.entries.items[0].id;
    _ = try mediaCommit(&tier, &cache, tokens[0..1200], ka, 700, s);
    _ = try mediaCommit(&tier, &cache, &tokens, ka, 700, s);
    try testing.expectEqual(@as(usize, 1), tier.entryCount());
    try testing.expectEqual(id, tier.entries.items[0].id);
    try testing.expectEqual(@as(u32, 1800), tier.entries.items[0].kv_len);
    try testing.expectEqual(ka, tier.entries.items[0].vision_key);
}

test "DiskTier chunk share: a donor under other pixels lends only the chunks wholly below the first image row" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);
    var tier = try DiskTier.init(testing.allocator, io, base, "fp-media-share", 0, 128);
    defer tier.deinit();
    tier.ssd_first = true;
    tier.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    // The token records agree through 520 (4 whole chunks); the images start at 300.
    const toks = chunkShareTokens(520);
    _ = try mediaCommit(&tier, &cache, &toks.a, 0xA1A1_A1A1_A1A1_A1A1, 300, s);
    _ = try mediaCommit(&tier, &cache, &toks.b, 0xB2B2_B2B2_B2B2_B2B2, 300, s);
    try testing.expectEqual(@as(usize, 2), tier.entryCount());
    try testing.expectEqual(@as(u32, 2), tier.entries.items[1].inherited_chunks);
}

test "DiskTier: an extend still staged behind the writer leaves an entry the scan reads at the manifest's count" {
    // A kill before the staged manifest lands leaves the previous one, whose count is shorter than the record.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);
    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try fillCache(&cache, s, 2, 1200, 8, 0.0, .float32);
    var tokens: [1200]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-torn-record", 0, 128);
    defer tier.deinit();
    tier.enableBackgroundWriter();
    _ = try tier.appendCommit(cache.entries, 600, cache.config, tokens[0..600], false, null, s);
    tier.drainWriter();
    tier.writer.?.setPaused(true);
    defer tier.writer.?.setPaused(false);
    _ = try tier.appendCommit(cache.entries, 1200, cache.config, &tokens, false, null, s);

    var loaded = tier.loadEntry(1) orelse return error.TestExpectedEntry;
    defer tier.freeIndexEntryOwned(&loaded.e);
    try testing.expectEqualSlices(u32, tokens[0..600], loaded.e.tokens);
}

test "DiskTier: a token record longer than the manifest's count restores at the count" {
    // A torn append: whole tokens past the count, then a partial one.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    {
        var tier = try DiskTier.init(testing.allocator, io, base, "fp-longer", 0, 128);
        defer tier.deinit();
        var cache = try KVCache.init(testing.allocator, 1);
        defer cache.deinit();
        try fillCache(&cache, s, 1, 600, 8, 0.0, .float32);
        _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
    }
    const record = (kv_disk_fs.testReadAt(io, base, "fp-longer/e1/tokens.bin", 1 << 20) orelse return error.TestUnexpectedResult);
    defer testing.allocator.free(record);
    const torn = try std.mem.concat(testing.allocator, u8, &.{ record, &[_]u8{ 9, 0, 0, 0, 10, 0, 0, 0, 11, 0 } });
    defer testing.allocator.free(torn);
    try kv_disk_fs.testWriteAt(io, base, "fp-longer/e1/tokens.bin", torn);

    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-longer", 0, 128);
    defer tier2.deinit();
    try testing.expectEqual(@as(usize, 1), tier2.entryCount());
    try testing.expectEqualSlices(u32, &tokens, tier2.entries.items[0].tokens);
    var out = try KVCache.init(testing.allocator, 1);
    defer out.deinit();
    try testing.expectEqual(@as(u32, 600), try tier2.restoreInto(&out, 0, s));
}

test "DiskTier: an extend shorter than a stale token tail leaves a record of exactly its tokens" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);
    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try fillCache(&cache, s, 1, 601, 8, 0.0, .float32);
    var tokens: [601]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    {
        var tier = try DiskTier.init(testing.allocator, io, base, "fp-stale", 0, 128);
        defer tier.deinit();
        _ = try tier.appendCommit(cache.entries, 600, cache.config, tokens[0..600], false, null, s);
    }
    const record = (kv_disk_fs.testReadAt(io, base, "fp-stale/e1/tokens.bin", 1 << 20) orelse return error.TestUnexpectedResult);
    defer testing.allocator.free(record);
    const torn = try std.mem.concat(testing.allocator, u8, &.{ record, &[_]u8{ 9, 0, 0, 0, 10, 0, 0, 0, 11, 0, 0, 0, 12, 0 } });
    defer testing.allocator.free(torn);
    try kv_disk_fs.testWriteAt(io, base, "fp-stale/e1/tokens.bin", torn);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-stale", 0, 128);
    defer tier.deinit();
    _ = try tier.appendCommit(cache.entries, 601, cache.config, &tokens, false, null, s);
    const path = try std.fmt.allocPrint(testing.allocator, "{s}/e1/tokens.bin", .{tier.root});
    defer testing.allocator.free(path);
    try testing.expectEqual(@as(u64, tokens.len * 4), kv_disk_fs.testSize(io, path).?);
}

test "DiskTier: a token record shorter than the manifest's count drops the entry" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    {
        var tier = try DiskTier.init(testing.allocator, io, base, "fp-shorter", 0, 128);
        defer tier.deinit();
        var cache = try KVCache.init(testing.allocator, 1);
        defer cache.deinit();
        try fillCache(&cache, s, 1, 600, 8, 0.0, .float32);
        _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
    }
    const record = (kv_disk_fs.testReadAt(io, base, "fp-shorter/e1/tokens.bin", 1 << 20) orelse return error.TestUnexpectedResult);
    defer testing.allocator.free(record);
    try kv_disk_fs.testWriteAt(io, base, "fp-shorter/e1/tokens.bin", record[0 .. record.len - 2]);

    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-shorter", 0, 128);
    defer tier2.deinit();
    try testing.expectEqual(@as(usize, 0), tier2.entryCount());
}

test "DiskTier: an extend writes the new tail of the token record, not the whole record" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);
    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try fillCache(&cache, s, 2, 1200, 8, 0.0, .float32);
    var tokens: [1200]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-tail", 0, 128);
    defer tier.deinit();
    _ = try tier.appendCommit(cache.entries, 600, cache.config, tokens[0..600], false, null, s);
    const kind = @backingInt(write_stats.Kind.meta);
    const before = tier.stats.snapshot().written[kind];
    _ = try tier.appendCommit(cache.entries, 1200, cache.config, &tokens, false, null, s);
    const grown = tier.stats.snapshot().written[kind] - before;

    const manifest = try std.fmt.allocPrint(testing.allocator, "{s}/e1/meta.json", .{tier.root});
    defer testing.allocator.free(manifest);
    try testing.expectEqual(600 * 4 + kv_disk_fs.testSize(io, manifest).?, grown);

    const record = (kv_disk_fs.testReadAt(io, base, "fp-tail/e1/tokens.bin", 1 << 20) orelse return error.TestUnexpectedResult);
    defer testing.allocator.free(record);
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&tokens), record);
}

/// SHA-256 over an entry directory's files in name order, each as name, length and bytes. On the content-addressed
/// layer each file is its plain-file image, regenerated from its blocks.
fn entryDirDigest(io: std.Io, abs_dir: []const u8) ![64]u8 {
    const ents = try kv_disk_fs.testList(io, abs_dir);
    defer kv_disk_fs.Fs.freeEntries(testing.allocator, ents);
    var names = std.ArrayList([]u8).empty;
    defer names.deinit(testing.allocator);
    for (ents) |ent| {
        if (ent.is_dir) continue;
        try names.append(testing.allocator, ent.name);
    }
    std.mem.sort([]u8, names.items, {}, struct {
        fn lt(_: void, a: []u8, b: []u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.lt);
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    for (names.items) |name| {
        const path = try std.fmt.allocPrint(testing.allocator, "{s}/{s}", .{ abs_dir, name });
        defer testing.allocator.free(path);
        const bytes = kv_disk_fs.testRead(io, path, 64 << 20) orelse return error.TestUnexpectedResult;
        defer testing.allocator.free(bytes);
        h.update(name);
        h.update(std.mem.asBytes(&@as(u64, bytes.len)));
        h.update(bytes);
    }
    return std.fmt.bytesToHex(h.finalResult(), .lower);
}

fn pinDigest(base: []const u8, fp: []const u8, io: std.Io) ![64]u8 {
    const dir = try std.fmt.allocPrint(testing.allocator, "{s}/{s}/e1", .{ base, fp });
    defer testing.allocator.free(dir);
    return entryDirDigest(io, dir);
}

// The SSD file bytes are a contract with files already on users' disks: a record the engine wrote
// must read back unchanged, and a new one must be byte-identical for the same state. Armed as production arms it.
test "DiskTier: the bytes of a dense, a hybrid-with-QSA and a head-sidecar record are pinned" {
    try test_note.needArchs(&.{"qwen4"});
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    {
        var tier = try DiskTier.init(testing.allocator, io, base, "pin-dense", 0, 128);
        defer tier.deinit();
        tier.enableBackgroundWriter();
        var cache = try KVCache.init(testing.allocator, 3);
        defer cache.deinit();
        try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
        _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
    }
    {
        var tier = try DiskTier.init(testing.allocator, io, base, "pin-hybrid", 0, 128);
        defer tier.deinit();
        tier.enableBackgroundWriter();
        var cache = try KVCache.init(testing.allocator, 3);
        defer cache.deinit();
        try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
        var src128 = buildHybridEntries(s, 100.0, 500.0);
        defer freeHybridEntries(&src128);
        var src256 = buildHybridEntries(s, 200.0, 600.0);
        defer freeHybridEntries(&src256);
        src256[2].aux_state = makeArange(s, &[_]c_int{ 1, 256, 4 }, 700.0);
        src256[2].qsa_pooled = makeArange(s, &[_]c_int{ 1, 64, 4 }, 800.0);
        src256[2].qsa_ratio = 4;
        src256[1].ple_prev = .{ 42, 43, 0, 0, 0, 0, 0, 0 };
        src256[1].ple_prev_valid = true;
        var cps = [_]trunk.SSMCheckpoint{
            try trunk.captureSsmCheckpoint(testing.allocator, &src128, 128, s),
            try trunk.captureSsmCheckpoint(testing.allocator, &src256, 256, s),
        };
        defer for (&cps) |*cp| cp.deinit(testing.allocator);
        try prefix_arch.aux.attachQsaHistoryToLatest(&cps, &src256, s);
        _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &cps, s);
    }
    {
        var tier = try DiskTier.init(testing.allocator, io, base, "pin-head", 0, 128);
        defer tier.deinit();
        tier.enableBackgroundWriter();
        var cache = try KVCache.init(testing.allocator, 2);
        defer cache.deinit();
        try fillCache(&cache, s, 2, 600, 8, 0.0, .float32);
        var mtp = try KVCache.init(testing.allocator, 1);
        defer mtp.deinit();
        try fillCache(&mtp, s, 1, 600, 8, 9.5, .float32);
        var aux_src: SSMCacheEntry = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = true };
        defer {
            _ = mlx.mlx_array_free(aux_src.conv_state);
            _ = mlx.mlx_array_free(aux_src.ssm_state);
            prefix_arch.aux.ssmFreeQsaState(&aux_src);
        }
        aux_src.aux_state = makeArange(s, &[_]c_int{ 1, 600, 8 }, 1.0);
        aux_src.qsa_pooled = makeArange(s, &[_]c_int{ 1, 150, 8 }, -9.0);
        aux_src.qsa_ratio = 4;
        var head_snap = trunk.ssmSnapshot(&aux_src, s);
        defer trunk.ssmSnapshotDeinit(&head_snap);
        _ = try tier.appendCommitWithSpec(cache.entries, cache.step, cache.config, &tokens, false, null, null, .{
            .entries = mtp.entries,
            .step = mtp.step,
            .config = mtp.config,
            .base_pos = 0,
            .head_aux = &head_snap,
            .head_pos_base = 1,
        }, s);
    }

    try testing.expectEqualStrings("46659498318c02b47573e2fdceac08a70b02f7629c37ae14a2417a17e405945e", &try pinDigest(base, "pin-dense", io));
    try testing.expectEqualStrings("c6417ab1e752894b4582b04af81cefee969b8f3cca38b2676aa32074d4a41ccc", &try pinDigest(base, "pin-hybrid", io));
    try testing.expectEqualStrings("acef87370ff1fd0fa9e12b69e37ec678dffac39e2b392f24a85e777869aefe4a", &try pinDigest(base, "pin-head", io));
}

test "DiskTier: random commits, restores, evictions and restarts keep the CAS index and files consistent" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    for ([_]u64{ 1, 2, 3 }) |seed| {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const base = try tmpRoot(&tmp, io, &buf);
        const budget: u64 = 220 * 1024;
        const openTier = struct {
            fn f(io_: std.Io, base_: []const u8, budget_: u64) !DiskTier {
                var t = try DiskTier.init(testing.allocator, io_, base_, "fp-prop", budget_, 128);
                t.ssd_first = true;
                t.ssm_max_per_entry = SSM_DISK_MAX_PER_ENTRY;
                t.cp_thin = .min_span_recency;
                t.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
                t.enableBackgroundWriter();
                return t;
            }
        }.f;
        var tier = try openTier(io, base, budget);
        defer tier.deinit();

        var prng = std.Random.DefaultPrng.init(seed);
        const r = prng.random();
        var convs: [3][1400]u32 = undefined;
        for (&convs, 0..) |*c, k| for (c, 0..) |*t, i| {
            t.* = if (i < 300) @intCast(i + 7) else @intCast(10_000 * (k + 1) + i);
        };
        var src = buildHybridEntries(s, 100.0, 500.0);
        defer freeHybridEntries(&src);
        for (0..36) |_| {
            switch (r.uintLessThan(u8, 10)) {
                0...5 => {
                    const k = r.uintLessThan(usize, 3);
                    const len = 520 + 64 * r.uintLessThan(usize, 14);
                    var cache = try KVCache.init(testing.allocator, 1);
                    defer cache.deinit();
                    try fillCache(&cache, s, 1, @intCast(len), 8, @floatFromInt(1000 * k), .float32);
                    var cps: [4]trunk.SSMCheckpoint = undefined;
                    const n_cps = r.uintLessThan(usize, 5);
                    for (cps[0..n_cps], 0..) |*cp, j| cp.* = try trunk.captureSsmCheckpoint(testing.allocator, &src, 128 * (j + 1), s);
                    defer for (cps[0..n_cps]) |*cp| cp.deinit(testing.allocator);
                    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, convs[k][0..len], false, if (n_cps > 0) cps[0..n_cps] else null, s);
                },
                6, 7 => {
                    const k = r.uintLessThan(usize, 3);
                    const len = 520 + 64 * r.uintLessThan(usize, 14);
                    if (tier.bestMatch(convs[k][0..len], false, kv_quant.KVQuantConfig.dense, .{})) |m| {
                        var out = try KVCache.init(testing.allocator, 1);
                        defer out.deinit();
                        const n = try tier.restoreInto(&out, m.idx, s);
                        try testing.expect(n >= 512);
                        _ = try cacheValueAt(&out, 0, n - 1, 3, s);
                    }
                },
                8 => {
                    const before_n = tier.entryCount();
                    const before_bytes = tier.total_bytes;
                    tier.deinit();
                    tier = try openTier(io, base, budget);
                    try testing.expectEqual(before_n, tier.entryCount());
                    try testing.expectEqual(before_bytes, tier.total_bytes);
                },
                else => tier.invalidateNewest(),
            }
            tier.drainWriter();
            try testing.expect(tier.total_bytes <= budget);
            try testing.expectEqual(@as(u64, 0), tier.writeErrors());
        }
    }
}

/// Bytes the files directly inside `dir_abs` hold on the volume (`du`): their allocated blocks, not their lengths.
fn allocatedBytes(dir_abs: []const u8) !u64 {
    var d = try std.Io.Dir.openDirAbsolute(std.testing.io, dir_abs, .{ .iterate = true });
    defer d.close(std.testing.io);
    var total: u64 = 0;
    var it = d.iterate();
    while (try it.next(std.testing.io)) |ent| {
        if (ent.kind != .file) continue;
        var pb: [std.fs.max_path_bytes + 1]u8 = undefined;
        const p = try std.fmt.bufPrintSentinel(&pb, "{s}/{s}", .{ dir_abs, ent.name }, 0);
        var st: std.c.Stat = undefined;
        if (std.c.stat(p.ptr, &st) != 0) return error.TestUnexpectedResult;
        total += @as(u64, @intCast(st.blocks)) * 512;
    }
    return total;
}

test "DiskTier budget: evictions that leave the layer's packs partly live give the volume back every dead page" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-packs", 0, 128);
    defer tier.deinit();
    const v = tier.fs.vfs;
    // A pack takes about one round's writes, so each holds a chunk of the conversation that stays.
    v.opts.pack_limit = 128 * 1024;
    const rounds = 8;
    const dim = 32;

    var conv: [640 + 128 * rounds]u32 = undefined;
    for (&conv, 0..) |*t, i| t.* = @intCast(i + 7);
    var long = try KVCache.init(testing.allocator, 1);
    defer long.deinit();
    try fillCache(&long, s, 1, 640, dim, 0.0, .float32);
    _ = try tier.appendCommit(long.entries, long.step, long.config, conv[0..640], false, null, s);
    // The conversation at its last length beside two and a half other records.
    tier.max_bytes = 800 * 1024;
    var vb: [std.fs.max_path_bytes]u8 = undefined;
    const v_dir = try std.fmt.bufPrint(&vb, "{s}/v", .{tier.root});
    var pb: [std.fs.max_path_bytes]u8 = undefined;
    const packs_dir = try std.fmt.bufPrint(&pb, "{s}/v/packs", .{tier.root});

    for (0..rounds) |r| {
        try fillCache(&long, s, 1, 128, dim, @floatFromInt(10_000_000 * (r + 1)), .float32);
        _ = try tier.appendCommit(long.entries, long.step, long.config, conv[0..long.step], false, null, s);

        var other = try KVCache.init(testing.allocator, 1);
        defer other.deinit();
        try fillCache(&other, s, 1, 520, dim, @floatFromInt(-10_000_000 * @as(i64, @intCast(r + 1))), .float32);
        var toks: [520]u32 = undefined;
        for (&toks, 0..) |*t, i| t.* = @intCast(1_000_000 * (r + 1) + i);
        _ = try tier.appendCommit(other.entries, other.step, other.config, &toks, false, null, s);

        // The conversation in use is never what goes.
        var kept = false;
        for (tier.entries.items) |*e| kept = kept or std.mem.eql(u32, e.tokens, conv[0..long.step]);
        try testing.expect(kept);
        try testing.expect(tier.entryCount() >= 2);
        try testing.expect(tier.total_bytes <= tier.max_bytes);
        // The packs hold no page without a live byte but each pack's last: past that, the bytes over.
        const pages = try v.testLivePageBytes() + 4096 * v.packs.count();
        try testing.expectEqual(@as(u64, 0), try allocatedBytes(packs_dir) -| pages);
        // What the tier reports is what the volume holds, to a partial last page per file.
        const held = try allocatedBytes(v_dir) + try allocatedBytes(packs_dir);
        const slack = 4096 * (2 + v.packs.count());
        try testing.expect(held <= tier.diskBytes() + slack and tier.diskBytes() <= held + slack);
    }
}

test "DiskTier: a lowered cap on a restart cuts a sole hybrid entry, and the next prompt stores" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try fillCache(&cache, s, 2, 768, 8, 0.0, .float32);
    var ta: [768]u32 = undefined;
    for (&ta, 0..) |*t, i| t.* = @intCast(i + 7);
    var tb: [768]u32 = undefined;
    for (&tb, 0..) |*t, i| t.* = @intCast(i + 900_000);
    var src = buildHybridEntries(s, 11.0, 22.0);
    defer freeHybridEntries(&src);
    var cps = [_]trunk.SSMCheckpoint{try trunk.captureSsmCheckpoint(testing.allocator, &src, 768, s)};
    defer for (&cps) |*cp| cp.deinit(testing.allocator);

    var held: u64 = 0;
    {
        var tier = try DiskTier.init(testing.allocator, io, base, "fp-hybrid-cap", 0, 128);
        defer tier.deinit();
        tier.ssd_first = true;
        tier.armTestSpace(1 << 40, 2 << 40);
        _ = try tier.appendCommit(cache.entries, 768, cache.config, &ta, false, &cps, s);
        tier.drainWriter();
        try testing.expectEqual(@as(usize, 1), tier.entryCount());
        held = tier.total_bytes;
        try testing.expect(held > 0);
    }
    var again = try DiskTier.init(testing.allocator, io, base, "fp-hybrid-cap", held / 2, 128);
    defer again.deinit();
    again.ssd_first = true;
    again.armTestSpace(1 << 40, 2 << 40);
    again.refreshDiskBudget();
    if (again.total_bytes > again.max_bytes) {
        std.debug.print("fx-d5: hybrid restart at half cap: {d} entries, {d} bytes over {d}\n", .{
            again.entryCount(), again.total_bytes, again.max_bytes,
        });
    }
    // A cap under one record's size holds no record; it must not leave the stale one over it either.
    try testing.expect(again.total_bytes <= again.max_bytes);
    // A cap that does hold a minimal record stores the next prompt again.
    again.operator_cap = held;
    again.refreshDiskBudget();
    const out = try again.appendCommit(cache.entries, 768, cache.config, &tb, false, null, s);
    if (out == .skipped or again.total_bytes > again.max_bytes) {
        std.debug.print("fx-d5: hybrid raised budget: {d} entries, {d} bytes over {d}; next prompt {s}\n", .{
            again.entryCount(), again.total_bytes, again.max_bytes, @tagName(out),
        });
    }
    try testing.expect(out != .skipped);
    try testing.expect(again.total_bytes <= again.max_bytes);
}

test "DiskTier: a lowered budget on a restart cuts a sole hybrid entry to the checkpoint that fits" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try fillCache(&cache, s, 2, 768, 8, 0.0, .float32);
    var ta: [768]u32 = undefined;
    for (&ta, 0..) |*t, i| t.* = @intCast(i + 7);
    var src = buildHybridEntries(s, 11.0, 22.0);
    defer freeHybridEntries(&src);
    var cps = [_]trunk.SSMCheckpoint{
        try trunk.captureSsmCheckpoint(testing.allocator, &src, 512, s),
        try trunk.captureSsmCheckpoint(testing.allocator, &src, 768, s),
    };
    defer for (&cps) |*cp| cp.deinit(testing.allocator);

    {
        var tier = try DiskTier.init(testing.allocator, io, base, "fp-hybrid-cut", 0, 128);
        defer tier.deinit();
        tier.ssd_first = true;
        tier.armTestSpace(1 << 40, 2 << 40);
        _ = try tier.appendCommit(cache.entries, 768, cache.config, &ta, false, &cps, s);
        tier.drainWriter();
        try testing.expectEqual(@as(usize, 1), tier.entryCount());
        try testing.expectEqual(@as(usize, 2), tier.entries.items[0].ssm_positions.len);
        // A cap that holds the 512 point but not the 768 one: the record is cut to 512, not evicted.
        const e = tier.entries.items[0];
        tier.operator_cap = e.bytes - e.ssm_bytes[1];
        tier.refreshDiskBudget();
        try testing.expectEqual(@as(usize, 1), tier.entryCount());
        try testing.expectEqual(@as(u32, 512), tier.entries.items[0].kv_len);
        try testing.expectEqualSlices(u32, &.{512}, tier.entries.items[0].ssm_positions);
        try testing.expect(tier.total_bytes <= tier.max_bytes);
        var out = try KVCache.init(testing.allocator, 2);
        defer out.deinit();
        var dst: [3]SSMCacheEntry = .{
            .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
            .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
            .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        };
        defer freeHybridEntries(&dst);
        try testing.expectEqual(@as(u32, 512), try tier.restoreIntoHybrid(&out, &dst, 0, 512, s));
    }
    // The cut survives a restart: 512 rows, one point, under the cap.
    var again = try DiskTier.init(testing.allocator, io, base, "fp-hybrid-cut", 0, 128);
    defer again.deinit();
    try testing.expectEqual(@as(usize, 1), again.entryCount());
    try testing.expectEqual(@as(u32, 512), again.entries.items[0].kv_len);
    try testing.expectEqualSlices(u32, &.{512}, again.entries.items[0].ssm_positions);
}
