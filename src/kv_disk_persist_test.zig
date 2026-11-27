//! SSD prefix cache commit, split-row, spec, and chunk-share tests.

const std = @import("std");
const testing = std.testing;
const mlx = @import("mlx.zig");
const kv_quant = @import("kv_quant.zig");
const trunk = @import("trunk.zig");
const prefix_policy = @import("prefix_policy.zig");
const kv_disk_fs = @import("kv_disk_fs.zig");
const safetensors = @import("safetensors.zig");
const disk = @import("kv_disk_cache.zig");
const util = @import("kv_disk_helpers_test.zig");

const KVCache = trunk.KVCache;
const SSMCacheEntry = trunk.SSMCacheEntry;
const DiskTier = disk.DiskTier;
const IndexEntry = disk.IndexEntry;
const SpecCommit = disk.SpecCommit;
const PersistOutcome = disk.PersistOutcome;
const POINT_GRID = disk.POINT_GRID;
const SSM_DISK_MAX_PER_ENTRY = disk.SSM_DISK_MAX_PER_ENTRY;
const SSM_DISK_MAX_PER_ENTRY_LEGACY = disk.SSM_DISK_MAX_PER_ENTRY_LEGACY;
const RING_DISK_MAX_PER_ENTRY = disk.RING_DISK_MAX_PER_ENTRY;
const commonPrefixLen = disk.commonPrefixLen;

const fillCache = util.fillCache;
const cacheValueAt = util.cacheValueAt;
const cacheBufValueAt = util.cacheBufValueAt;
const tmpRoot = util.tmpRoot;
const ssmArrVal = util.ssmArrVal;
const buildHybridEntries = util.buildHybridEntries;
const freeHybridEntries = util.freeHybridEntries;
const grow_span = util.grow_span;
const growCheckpoint = util.growCheckpoint;
const expectGrowPoint = util.expectGrowPoint;
const commitGrowPoints = util.commitGrowPoints;
const growTier = util.growTier;
const growFileSize = util.growFileSize;
const chunkShareTokens = util.chunkShareTokens;
const wroteKind = util.wroteKind;

test "DiskTier split rows: a split tier keeps each span of rows once, and every point restores the arrays a whole point holds" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    for ([_]bool{ false, true }) |writer| {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const base = try tmpRoot(&tmp, io, &buf);
        var tokens: [600]u32 = undefined;
        for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
        var cache = try KVCache.init(testing.allocator, 3);
        defer cache.deinit();
        try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
        // The grid, and a prompt-end point past it.
        const points = [_]usize{ 128, 256, 384, 512, 577 };

        var split = try growTier(io, base, "fp-split", true, writer);
        defer split.deinit();
        try commitGrowPoints(&split, &cache, &tokens, &points, s);
        split.drainWriter();
        var whole = try growTier(io, base, "fp-whole", false, writer);
        defer whole.deinit();
        try commitGrowPoints(&whole, &cache, &tokens, &points, s);
        whole.drainWriter();

        const e = &split.entries.items[0];
        try testing.expectEqual(@as(usize, 4), e.grown.len);
        try testing.expectEqual(grow_span, e.grown_span);
        try testing.expectEqualSlices(u32, &.{ 128, 256, 384, 512, 577 }, e.ssm_positions);
        for (points) |p| try expectGrowPoint(&split, e.id, @intCast(p), s);

        // A point's own file is no bigger than a whole one, and the entry as a whole is smaller.
        const we = &whole.entries.items[0];
        for (e.ssm_bytes, we.ssm_bytes) |sb, wb| try testing.expect(sb <= wb);
        var split_total: u64 = 0;
        var whole_total: u64 = 0;
        for (e.ssm_bytes) |b| split_total += b;
        for (e.grown) |b| split_total += b;
        for (we.ssm_bytes) |b| whole_total += b;
        try testing.expect(split_total < whole_total);
        // A point on the grid keeps no rows of its own.
        try testing.expect(e.ssm_bytes[3] < e.ssm_bytes[4]);
        // The whole tier's entry has no segment and keeps the old layout.
        try testing.expectEqual(@as(usize, 0), we.grown.len);
        const seg = try std.fmt.allocPrint(testing.allocator, "{s}/fp-whole/e{d}/g00001.safetensors", .{ base, we.id });
        defer testing.allocator.free(seg);
        try testing.expect(kv_disk_fs.testStat(io, seg) == null);
    }
}

test "DiskTier split rows: retention drops a point's file and never the rows a kept point leans on" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);

    var id: u64 = 0;
    {
        var tier = try growTier(io, base, "fp-drop", true, false);
        defer tier.deinit();
        tier.ssm_max_per_entry = 3;
        try commitGrowPoints(&tier, &cache, &tokens, &.{ 128, 256, 384, 512, 577 }, s);
        const e = &tier.entries.items[0];
        id = e.id;
        // The newest and the oldest stay; the interior thins.
        try testing.expectEqual(@as(usize, 3), e.ssm_positions.len);
        try testing.expectEqual(@as(u32, 128), e.ssm_positions[0]);
        try testing.expectEqual(@as(u32, 577), e.ssm_positions[2]);
        // All four segments stay though the point at 512 may be gone.
        try testing.expectEqual(@as(usize, 4), e.grown.len);
        for (1..5) |j| {
            var name_buf: [32]u8 = undefined;
            const name = try std.fmt.bufPrint(&name_buf, "g{d:0>5}.safetensors", .{j});
            try testing.expect(growFileSize(&tier, tier.root, id, name) != null);
        }
        for (e.ssm_positions) |p| try expectGrowPoint(&tier, id, p, s);
    }
    // After a restart the kept points restore from the same files.
    var again = try growTier(io, base, "fp-drop", true, false);
    defer again.deinit();
    const e = &again.entries.items[0];
    try testing.expectEqual(@as(usize, 3), e.ssm_positions.len);
    try testing.expectEqual(@as(usize, 4), e.grown.len);
    for (e.ssm_positions) |p| try expectGrowPoint(&again, id, p, s);
}

test "DiskTier split rows: a commit that extends an entry appends row segments and leaves the earlier ones untouched" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    for ([_]bool{ false, true }) |writer| {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const base = try tmpRoot(&tmp, io, &buf);
        var tokens: [600]u32 = undefined;
        for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
        var tier = try growTier(io, base, "fp-extend", true, writer);
        defer tier.deinit();

        var first = try KVCache.init(testing.allocator, 3);
        defer first.deinit();
        try fillCache(&first, s, 3, 520, 8, 0.0, .float32);
        try commitGrowPoints(&tier, &first, tokens[0..520], &.{ 128, 256, 384 }, s);
        tier.drainWriter();
        const id = tier.entries.items[0].id;
        try testing.expectEqual(@as(usize, 3), tier.entries.items[0].grown.len);
        const seg1 = try std.fmt.allocPrint(testing.allocator, "{s}/e{d}/g00001.safetensors", .{ tier.root, id });
        defer testing.allocator.free(seg1);
        const before = kv_disk_fs.testStat(io, seg1).?;
        const bytes_before = tier.entries.items[0].grown[0];

        var second = try KVCache.init(testing.allocator, 3);
        defer second.deinit();
        try fillCache(&second, s, 3, 600, 8, 0.0, .float32);
        try commitGrowPoints(&tier, &second, &tokens, &.{ 512, 577 }, s);
        tier.drainWriter();
        try testing.expectEqual(@as(usize, 1), tier.entryCount());
        const e = &tier.entries.items[0];
        try testing.expectEqual(id, e.id);
        try testing.expectEqual(@as(usize, 4), e.grown.len);
        try testing.expectEqual(bytes_before, e.grown[0]);
        const after = kv_disk_fs.testStat(io, seg1).?;
        try testing.expectEqual(before.inode, after.inode);
        try testing.expectEqual(before.mtime_ns, after.mtime_ns);
        try testing.expectEqualSlices(u32, &.{ 128, 256, 384, 512, 577 }, e.ssm_positions);
        // The old points and the new ones read back the arrays a full capture holds.
        for (e.ssm_positions) |p| try expectGrowPoint(&tier, id, p, s);
    }
}

test "DiskTier split rows: an entry's bill is the bytes its files hold, through a commit, a scan and an eviction" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 520, 8, 0.0, .float32);

    var tier = try growTier(io, base, "fp-bill", true, false);
    try commitGrowPoints(&tier, &cache, tokens[0..520], &.{ 128, 256, 384 }, s);
    var second = try KVCache.init(testing.allocator, 3);
    defer second.deinit();
    try fillCache(&second, s, 3, 600, 8, 0.0, .float32);
    try commitGrowPoints(&tier, &second, &tokens, &.{ 512, 577 }, s);

    const dir = try std.fmt.allocPrint(testing.allocator, "{s}/e{d}", .{ tier.root, tier.entries.items[0].id });
    defer testing.allocator.free(dir);
    const on_disk = struct {
        fn total(dir_path: []const u8, io_: std.Io) !u64 {
            const names = try kv_disk_fs.testList(io_, dir_path);
            defer kv_disk_fs.Fs.freeEntries(testing.allocator, names);
            var sum: u64 = 0;
            for (names) |ent| {
                if (ent.is_dir or std.mem.eql(u8, ent.name, "meta.json")) continue;
                const p = try std.fmt.allocPrint(testing.allocator, "{s}/{s}", .{ dir_path, ent.name });
                defer testing.allocator.free(p);
                sum += kv_disk_fs.testSize(io_, p).?;
            }
            return sum;
        }
    };
    const written = try on_disk.total(dir, io);
    try testing.expectEqual(written, tier.entries.items[0].bytes);
    try testing.expectEqual(written, tier.total_bytes);
    // The estimate a commit is gated on never undercounts a point's own file.
    for (tier.entries.items[0].ssm_positions, tier.entries.items[0].ssm_bytes) |p, b| {
        var cp = try growCheckpoint(s, p);
        defer cp.deinit(testing.allocator);
        try testing.expect(tier.pointFileBytes(&cp) + 1024 >= b);
    }
    tier.deinit();

    var scanned = try growTier(io, base, "fp-bill", true, false);
    defer scanned.deinit();
    try testing.expectEqual(written, scanned.entries.items[0].bytes);
    try testing.expectEqual(written, scanned.total_bytes);
    scanned.removeAt(0);
    try testing.expectEqual(@as(u64, 0), scanned.total_bytes);
}

/// A tier that may link a donor's files (`chunkShareDonor`), with the grown spec of `growTier`.
fn shareGrowTier(io: std.Io, base: []const u8, fp: []const u8, writer: bool) !DiskTier {
    var tier = try growTier(io, base, fp, true, writer);
    tier.ssd_first = true;
    tier.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
    return tier;
}

fn growSegStat(io: std.Io, tier: *const DiskTier, id: u64, j: usize) ?kv_disk_fs.Stat {
    var buf: [1024]u8 = undefined;
    const p = std.fmt.bufPrint(&buf, "{s}/e{d}/g{d:0>5}.safetensors", .{ tier.root, id, j }) catch return null;
    return kv_disk_fs.testStat(io, p);
}

/// Bytes the tier's entry directories hold, a hard-linked file once (meta.json aside).
fn uniqueDirBytes(io: std.Io, tier: *const DiskTier) !u64 {
    var seen = std.AutoHashMap(u64, void).init(testing.allocator);
    defer seen.deinit();
    var sum: u64 = 0;
    for (tier.entries.items) |*e| {
        const dir = try std.fmt.allocPrint(testing.allocator, "{s}/e{d}", .{ tier.root, e.id });
        defer testing.allocator.free(dir);
        const names = try kv_disk_fs.testList(io, dir);
        defer kv_disk_fs.Fs.freeEntries(testing.allocator, names);
        for (names) |ent| {
            if (ent.is_dir or std.mem.eql(u8, ent.name, "meta.json")) continue;
            const p = try std.fmt.allocPrint(testing.allocator, "{s}/{s}", .{ dir, ent.name });
            defer testing.allocator.free(p);
            const st = kv_disk_fs.testStat(io, p).?;
            if ((try seen.getOrPut(@intCast(st.inode))).found_existing) continue;
            sum += st.size;
        }
    }
    return sum;
}

/// The tokens of a conversation that rolled back to `shared` tokens and went on with other ones.
fn divergedTokens(shared: usize) [600]u32 {
    var out: [600]u32 = undefined;
    for (&out, 0..) |*t, i| t.* = if (i < shared) @intCast(i + 7) else @intCast(9000 + i);
    return out;
}

/// Commits a donor of 520 tokens with points on the 128-token grid, then an heir of 600 that shares `shared` tokens.
fn commitDonorAndHeir(tier: *DiskTier, s: mlx.mlx_stream, shared: usize, heir_points: []const usize) !void {
    var donor_cache = try KVCache.init(testing.allocator, 3);
    defer donor_cache.deinit();
    try fillCache(&donor_cache, s, 3, 520, 8, 0.0, .float32);
    var a_tokens: [520]u32 = undefined;
    for (&a_tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    try commitGrowPoints(tier, &donor_cache, &a_tokens, &.{ 128, 256, 384, 512 }, s);
    tier.drainWriter();
    const b_tokens = divergedTokens(shared);
    var heir_cache = try KVCache.init(testing.allocator, 3);
    defer heir_cache.deinit();
    try fillCache(&heir_cache, s, 3, 600, 8, 0.0, .float32);
    try commitGrowPoints(tier, &heir_cache, &b_tokens, heir_points, s);
    tier.drainWriter();
}

test "DiskTier row-segment share: an heir links the donor's whole segments inside the shared prefix and writes the rest" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    for ([_]bool{ false, true }) |writer| {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const base = try tmpRoot(&tmp, io, &buf);
        var tier = try shareGrowTier(io, base, "fp-gshare", writer);
        defer tier.deinit();
        // Rows 0..400 are shared: spans 1..3 (to 384) lie inside it, span 4 (to 512) straddles the divergence.
        try commitDonorAndHeir(&tier, s, 400, &.{ 512, 577 });
        const donor_id = tier.entries.items[0].id;
        try testing.expectEqual(@as(usize, 2), tier.entryCount());
        const heir = &tier.entries.items[1];
        const heir_id = heir.id;
        try testing.expectEqual(@as(u32, 3), heir.inherited_grown);
        try testing.expectEqual(@as(usize, 4), heir.grown.len);
        for (1..4) |j| {
            const st = growSegStat(io, &tier, heir.id, j).?;
            try testing.expectEqual(@as(u64, 2), @as(u64, @intCast(st.nlink)));
            try testing.expectEqual(growSegStat(io, &tier, donor_id, j).?.inode, st.inode);
            try testing.expectEqual(tier.entries.items[0].grown[j - 1], heir.grown[j - 1]);
        }
        try testing.expectEqual(@as(u64, 1), @as(u64, @intCast(growSegStat(io, &tier, heir.id, 4).?.nlink)));
        // The heir's points read back the arrays a full capture holds, linked rows included.
        for (heir.ssm_positions) |p| try expectGrowPoint(&tier, heir.id, p, s);

        // A linked segment is billed once: the tier's bill is the bytes the directories hold.
        try testing.expectEqual(try uniqueDirBytes(io, &tier), tier.total_bytes);

        // Evicting the donor frees none of the linked segments, and the heir restores from them.
        tier.removeAt(0);
        const kept = &tier.entries.items[0];
        try testing.expectEqual(heir_id, kept.id);
        for (1..4) |j| try testing.expectEqual(@as(u64, 1), @as(u64, @intCast(growSegStat(io, &tier, kept.id, j).?.nlink)));
        for (kept.ssm_positions) |p| try expectGrowPoint(&tier, kept.id, p, s);
        try testing.expectEqual(try uniqueDirBytes(io, &tier), tier.total_bytes);
    }
}

test "DiskTier: an evicted record leaves its highest point inside the prefix a survivor shares to that survivor" {
    // An heir keeps points only past where it diverged; once the oldest record went, a request that changed an
    // early image found no point below the change and prefilled 300k tokens cold.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);
    var probe: [600]u32 = undefined;
    for (&probe, 0..) |*t, i| t.* = if (i < 400) @intCast(i + 7) else @intCast(20000 + i);
    var heir_id: u64 = 0;
    {
        var tier = try growTier(io, base, "fp-bequeath", false, false);
        defer tier.deinit();
        tier.ssd_first = true;
        tier.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
        try commitDonorAndHeir(&tier, s, 450, &.{577});
        heir_id = tier.entries.items[1].id;
        tier.max_bytes = tier.total_bytes - 1;
        tier.gcToBudget();
        try testing.expectEqual(@as(usize, 1), tier.entryCount());
        const m = tier.bestHybridMatch(&probe, false, kv_quant.KVQuantConfig.dense, std.math.maxInt(u32), .{}) orelse return error.TestExpectedMatch;
        try testing.expectEqual(heir_id, tier.entries.items[m.idx].id);
        try testing.expectEqual(@as(u32, 384), m.cp);
        try expectGrowPoint(&tier, heir_id, 384, s);
        try testing.expectEqual(try uniqueDirBytes(io, &tier), tier.total_bytes);
    }
    var again = try growTier(io, base, "fp-bequeath", false, false);
    defer again.deinit();
    try testing.expectEqualSlices(u32, &.{ 384, 577 }, again.entries.items[0].ssm_positions);
    try testing.expectEqual(try uniqueDirBytes(io, &again), again.total_bytes);
}

test "DiskTier row-segment share: an evicted split-row record bequeaths its highest shared point to a survivor that links its segments" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);
    var probe: [600]u32 = undefined;
    for (&probe, 0..) |*t, i| t.* = if (i < 400) @intCast(i + 7) else @intCast(20000 + i);
    var heir_id: u64 = 0;
    {
        var tier = try shareGrowTier(io, base, "fp-bequeath-split", false);
        defer tier.deinit();
        try commitDonorAndHeir(&tier, s, 450, &.{577});
        heir_id = tier.entries.items[1].id;
        tier.max_bytes = tier.total_bytes - 1;
        tier.gcToBudget();
        try testing.expectEqual(@as(usize, 1), tier.entryCount());
        const m = tier.bestHybridMatch(&probe, false, kv_quant.KVQuantConfig.dense, std.math.maxInt(u32), .{}) orelse return error.TestExpectedMatch;
        try testing.expectEqual(heir_id, tier.entries.items[m.idx].id);
        try testing.expectEqual(@as(u32, 384), m.cp);
        try expectGrowPoint(&tier, heir_id, 384, s);
        try testing.expectEqual(try uniqueDirBytes(io, &tier), tier.total_bytes);
    }
    var again = try shareGrowTier(io, base, "fp-bequeath-split", false);
    defer again.deinit();
    try testing.expectEqualSlices(u32, &.{ 384, 577 }, again.entries.items[0].ssm_positions);
    try testing.expectEqual(try uniqueDirBytes(io, &again), again.total_bytes);
}

test "DiskTier row-segment share: an evicted split-row record leaves no point to a survivor that wrote its own rows under another image" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    const img_at: u32 = 40;
    const k1: u64 = 0x1111_AAAA;
    const k2: u64 = 0x2222_BBBB;
    const chain = [_]prefix_policy.MediaSpan{ .{ .start = img_at, .key = k1 }, .{ .start = 420, .key = k2 } };
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    // The same token ids under another image: the heir's rows past the image start are its own. Arm 0 names no
    // chain, so it links no chunk and no segment; arm 1 links the chunks below its second image but its rows are
    // restored elsewhere (`rows_base` 0), so only the segments differ.
    for (0..2) |arm| {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const base = try tmpRoot(&tmp, io, &buf);
        var tier = try shareGrowTier(io, base, "fp-bequeath-split-other-image", false);
        defer tier.deinit();
        for ([_]usize{ 520, 600 }, [_]u64{ k1, k2 }, [_][]const usize{ &.{ 128, 256, 384, 512 }, &.{577} }) |len, key, pos| {
            const heir = len == 600;
            var cache = try KVCache.init(testing.allocator, 3);
            defer cache.deinit();
            try fillCache(&cache, s, 3, @intCast(len), 8, 0.0, .float32);
            const cps = try testing.allocator.alloc(trunk.SSMCheckpoint, pos.len);
            defer testing.allocator.free(cps);
            for (cps, pos) |*cp, p| cp.* = try growCheckpoint(s, p);
            defer for (cps) |*cp| cp.deinit(testing.allocator);
            if (heir and arm == 1) tier.rows_base = 0;
            _ = try tier.appendCommitWith(.{
                .kv_entries = cache.entries,
                .step = len,
                .config = cache.config,
                .tokens = tokens[0..len],
                .has_tools = false,
                .ssm_checkpoints = cps,
                .vision_key = key,
                .media_start = img_at,
                .media_chain = if (arm == 0) &.{} else if (heir) &chain else chain[0..1],
            }, s, .{ .whole = true });
        }
        try testing.expectEqual(@as(usize, 2), tier.entryCount());
        try testing.expectEqual(@as(u32, 0), tier.entries.items[1].inherited_grown);
        try testing.expectEqual(@as(u32, if (arm == 0) 0 else 3), tier.entries.items[1].inherited_chunks);

        tier.max_bytes = tier.total_bytes - 1;
        tier.gcToBudget();
        try testing.expectEqual(@as(usize, 1), tier.entryCount());
        try testing.expectEqualSlices(u32, &.{577}, tier.entries.items[0].ssm_positions);
        try testing.expectEqual(try uniqueDirBytes(io, &tier), tier.total_bytes);
    }
}

test "DiskTier: cutting a sole entry below its first media row clears its media stamp so text prompts share it" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);
    var tier = try shareGrowTier(io, base, "fp-cut-media", false);
    defer tier.deinit();
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 520, 8, 0.0, .float32);
    var tokens: [520]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var cps: [4]trunk.SSMCheckpoint = undefined;
    for (&cps, [_]usize{ 128, 256, 384, 512 }) |*cp, p| cp.* = try growCheckpoint(s, p);
    defer for (&cps) |*cp| cp.deinit(testing.allocator);
    _ = try tier.appendCommitWith(.{
        .kv_entries = cache.entries,
        .step = tokens.len,
        .config = cache.config,
        .tokens = &tokens,
        .has_tools = false,
        .ssm_checkpoints = &cps,
        .vision_key = 0xAA11,
        .media_start = 300,
    }, s, .{ .whole = true });
    tier.cutEntryTo(0, 256);
    try testing.expectEqual(@as(u32, 256), tier.entries.items[0].kv_len);
    try testing.expectEqual(@as(u64, 0), tier.entries.items[0].vision_key);
    try testing.expectEqual(@as(?u32, null), tier.entries.items[0].media_start);
}

test "DiskTier row-segment share: multi-image media_chain links segments past image 1 and bequeaths grid points across eviction" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);
    var tier = try shareGrowTier(io, base, "fp-multi-img-chain", false);
    defer tier.deinit();
    const k1: u64 = 0x1111_AAAA;
    const k2: u64 = 0x2222_BBBB;
    const img1_at: u32 = 40;
    const img2_at: u32 = 420;
    const chain = [_]prefix_policy.MediaSpan{
        .{ .start = img1_at, .key = k1 },
        .{ .start = img2_at, .key = k2 },
    };

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var c1 = try KVCache.init(testing.allocator, 3);
    defer c1.deinit();
    try fillCache(&c1, s, 3, 520, 8, 0.0, .float32);
    var cps1: [4]trunk.SSMCheckpoint = undefined;
    for (&cps1, [_]usize{ 128, 256, 384, 512 }) |*cp, p| cp.* = try growCheckpoint(s, p);
    defer for (&cps1) |*cp| cp.deinit(testing.allocator);
    _ = try tier.appendCommitWith(.{
        .kv_entries = c1.entries,
        .step = 520,
        .config = c1.config,
        .tokens = tokens[0..520],
        .has_tools = false,
        .ssm_checkpoints = &cps1,
        .vision_key = k1,
        .media_start = img1_at,
        .media_chain = chain[0..1],
    }, s, .{ .whole = true });

    var c2 = try KVCache.init(testing.allocator, 3);
    defer c2.deinit();
    try fillCache(&c2, s, 3, 600, 8, 0.0, .float32);
    var cps2: [1]trunk.SSMCheckpoint = .{try growCheckpoint(s, 577)};
    defer cps2[0].deinit(testing.allocator);
    _ = try tier.appendCommitWith(.{
        .kv_entries = c2.entries,
        .step = 600,
        .config = c2.config,
        .tokens = &tokens,
        .has_tools = false,
        .ssm_checkpoints = &cps2,
        .vision_key = k2,
        .media_start = img1_at,
        .media_chain = &chain,
    }, s, .{ .whole = true });

    try testing.expectEqual(@as(usize, 2), tier.entryCount());
    // All 3 whole spans before image 2 (128, 256, 384) are inherited past image 1 (at 40).
    try testing.expectEqual(@as(u32, 3), tier.entries.items[1].inherited_grown);
    try testing.expectEqual(@as(u32, 3), tier.entries.items[1].inherited_chunks);
    try testing.expectEqual(try uniqueDirBytes(io, &tier), tier.total_bytes);

    // Evicting entry 1 bequeaths its grid point at 384 to entry 2 because segments 1..3 are hard-linked.
    tier.max_bytes = tier.total_bytes - 1;
    tier.gcToBudget();
    try testing.expectEqual(@as(usize, 1), tier.entryCount());
    try testing.expectEqualSlices(u32, &.{ 384, 577 }, tier.entries.items[0].ssm_positions);
    try expectGrowPoint(&tier, tier.entries.items[0].id, 384, s);
    try testing.expectEqual(try uniqueDirBytes(io, &tier), tier.total_bytes);
}

test "DiskTier row-segment share: a record cut before the second image links the segments it restored from a two-image entry" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);
    var tier = try shareGrowTier(io, base, "fp-restored-seg", false);
    defer tier.deinit();
    const k1: u64 = 0x1111_AAAA;
    const k2: u64 = 0x2222_BBBB;
    const chain = [_]prefix_policy.MediaSpan{ .{ .start = 40, .key = k1 }, .{ .start = 540, .key = k2 } };
    var tokens: [700]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var donor_cache = try KVCache.init(testing.allocator, 3);
    defer donor_cache.deinit();
    try fillCache(&donor_cache, s, 3, 700, 8, 0.0, .float32);
    var donor_cps: [5]trunk.SSMCheckpoint = undefined;
    for (&donor_cps, [_]usize{ 128, 256, 384, 512, 640 }) |*cp, p| cp.* = try growCheckpoint(s, p);
    defer for (&donor_cps) |*cp| cp.deinit(testing.allocator);
    _ = try tier.appendCommitWith(.{
        .kv_entries = donor_cache.entries,
        .step = 700,
        .config = donor_cache.config,
        .tokens = &tokens,
        .has_tools = false,
        .ssm_checkpoints = &donor_cps,
        .vision_key = k2,
        .media_start = 40,
        .media_chain = &chain,
    }, s, .{ .whole = true });
    const donor_id = tier.entries.items[0].id;

    // The same request cut at 520, before its second image: the stamp is image 1's, which the two-image entry does
    // not carry, so only the rows the request restored from that entry say what its segments hold.
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 520, 8, 0.0, .float32);
    var cps: [1]trunk.SSMCheckpoint = .{try growCheckpoint(s, 512)};
    defer cps[0].deinit(testing.allocator);
    _ = try tier.appendCommitWith(.{
        .kv_entries = cache.entries,
        .step = 520,
        .config = cache.config,
        .tokens = tokens[0..520],
        .has_tools = false,
        .ssm_checkpoints = &cps,
        .vision_key = k1,
        .media_start = 40,
        .media_chain = &chain,
        .restored = .{ .id = donor_id, .rows = 512 },
    }, s, .{ .whole = true });
    try testing.expectEqual(@as(usize, 2), tier.entryCount());
    const heir = &tier.entries.items[1];
    try testing.expectEqual(@as(u32, 4), heir.inherited_grown);
    for (1..5) |j| try testing.expectEqual(growSegStat(io, &tier, donor_id, j).?.inode, growSegStat(io, &tier, heir.id, j).?.inode);
    try expectGrowPoint(&tier, heir.id, 512, s);
    try testing.expectEqual(try uniqueDirBytes(io, &tier), tier.total_bytes);
}

test "DiskTier: an evicted record leaves no point to an entry whose rows below it are its own" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);
    var tier = try growTier(io, base, "fp-bequeath-own", false, false);
    defer tier.deinit();
    // The synchronous legacy arm shares no chunk: the second record wrote its own rows.
    try commitDonorAndHeir(&tier, s, 450, &.{577});
    try testing.expectEqual(@as(u32, 0), tier.entries.items[1].inherited_chunks);
    tier.max_bytes = tier.total_bytes - 1;
    tier.gcToBudget();
    try testing.expectEqualSlices(u32, &.{577}, tier.entries.items[0].ssm_positions);
}

test "DiskTier row-segment share: a scan accepts linked segments and bills each once; removing the last owner frees them" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);
    var heir_id: u64 = 0;
    {
        var tier = try shareGrowTier(io, base, "fp-gscan", false);
        defer tier.deinit();
        try commitDonorAndHeir(&tier, s, 400, &.{ 512, 577 });
        heir_id = tier.entries.items[1].id;
    }
    var scanned = try shareGrowTier(io, base, "fp-gscan", false);
    defer scanned.deinit();
    try testing.expectEqual(@as(usize, 2), scanned.entryCount());
    try testing.expectEqual(try uniqueDirBytes(io, &scanned), scanned.total_bytes);
    for (scanned.entries.items) |*e| {
        if (e.id != heir_id) continue;
        try testing.expectEqual(@as(u32, 3), e.inherited_grown);
        try testing.expectEqual(@as(usize, 4), e.grown.len);
        for (e.ssm_positions) |p| try expectGrowPoint(&scanned, e.id, p, s);
    }
    // Either owner may go first; the bytes leave with the last link.
    scanned.removeAt(0);
    try testing.expectEqual(try uniqueDirBytes(io, &scanned), scanned.total_bytes);
    scanned.removeAt(0);
    try testing.expectEqual(@as(u64, 0), scanned.total_bytes);
}

test "DiskTier row-segment share: the segments come from the entry that holds them, not from the newer entry that shares more chunks" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);
    var tier = try shareGrowTier(io, base, "fp-gpick", false);
    defer tier.deinit();
    // A holds the segments. B is a later turn that diverged from A at 450 and wrote no point, so it holds none.
    try commitDonorAndHeir(&tier, s, 450, &.{});
    try testing.expectEqual(@as(usize, 0), tier.entries.items[1].grown.len);
    // C continues B for 520 tokens: B shares more chunks with it than A does, A more segments.
    var c_tokens: [600]u32 = undefined;
    for (&c_tokens, 0..) |*t, i| t.* = if (i < 450) @intCast(i + 7) else if (i < 520) @intCast(9000 + i) else @intCast(20000 + i);
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    try commitGrowPoints(&tier, &cache, &c_tokens, &.{ 512, 577 }, s);
    tier.drainWriter();
    const c = &tier.entries.items[2];
    try testing.expectEqual(@as(u32, 3), c.inherited_grown);
    try testing.expectEqual(@as(usize, 4), c.grown.len);
    try testing.expectEqual(growSegStat(io, &tier, tier.entries.items[0].id, 1).?.inode, growSegStat(io, &tier, c.id, 1).?.inode);
    for (c.ssm_positions) |p| try expectGrowPoint(&tier, c.id, p, s);
}

test "DiskTier row-segment share: only the segments below the rows the heir took from a stored entry are linked, the span it recomputed is written fresh" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);
    var tier = try shareGrowTier(io, base, "fp-gbase", false);
    defer tier.deinit();
    // The heir shares 400 tokens but restored at 256: span 3 (256..384) holds rows it recomputed.
    tier.rows_base = 256;
    try commitDonorAndHeir(&tier, s, 400, &.{ 512, 577 });
    const heir = &tier.entries.items[1];
    try testing.expectEqual(@as(u32, 2), heir.inherited_grown);
    try testing.expectEqual(@as(usize, 4), heir.grown.len);
    try testing.expectEqual(@as(u64, 2), @as(u64, @intCast(growSegStat(io, &tier, heir.id, 2).?.nlink)));
    try testing.expectEqual(@as(u64, 1), @as(u64, @intCast(growSegStat(io, &tier, heir.id, 3).?.nlink)));
    for (heir.ssm_positions) |p| try expectGrowPoint(&tier, heir.id, p, s);
}

test "DiskTier row-segment share: a divergence inside the first span links nothing" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);
    var tier = try shareGrowTier(io, base, "fp-gnone", true);
    defer tier.deinit();
    try commitDonorAndHeir(&tier, s, 100, &.{512});
    const heir = &tier.entries.items[1];
    try testing.expectEqual(@as(u32, 0), heir.inherited_grown);
    try testing.expectEqual(@as(usize, 4), heir.grown.len);
    for (1..5) |j| try testing.expectEqual(@as(u64, 1), @as(u64, @intCast(growSegStat(io, &tier, heir.id, j).?.nlink)));
    for (heir.ssm_positions) |p| try expectGrowPoint(&tier, heir.id, p, s);
}

test "DiskTier split rows: a lost row segment drops the points that lean on it and keeps the rest" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    var id: u64 = 0;
    {
        var tier = try growTier(io, base, "fp-lost", true, false);
        defer tier.deinit();
        try commitGrowPoints(&tier, &cache, &tokens, &.{ 100, 128, 256, 384, 577 }, s);
        id = tier.entries.items[0].id;
    }
    const lost = try std.fmt.allocPrint(testing.allocator, "fp-lost/e{d}/g00003.safetensors", .{id});
    defer testing.allocator.free(lost);
    try kv_disk_fs.testWriteAt(io, base, lost, "trunc");
    var tier = try growTier(io, base, "fp-lost", true, false);
    defer tier.deinit();
    const e = &tier.entries.items[0];
    try testing.expectEqual(@as(usize, 2), e.grown.len);
    try testing.expectEqualSlices(u32, &.{ 100, 128, 256 }, e.ssm_positions);
    for (e.ssm_positions) |p| try expectGrowPoint(&tier, id, p, s);
}

test "DiskTier split rows: a tier with no split, or points with no growing arrays, writes the layout every other arch reads" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);

    var plain = try growTier(io, base, "fp-plain", false, false);
    defer plain.deinit();
    try commitGrowPoints(&plain, &cache, &tokens, &.{ 128, 384 }, s);
    // Every point is one self-contained file and the manifest names no segment.
    try testing.expectEqual(@as(usize, 0), plain.entries.items[0].grown.len);
    const meta_path = try std.fmt.allocPrint(testing.allocator, "{s}/e{d}/meta.json", .{ plain.root, plain.entries.items[0].id });
    defer testing.allocator.free(meta_path);
    const meta = kv_disk_fs.testRead(io, meta_path, 64 * 1024).?;
    defer testing.allocator.free(meta);
    try testing.expect(std.mem.indexOf(u8, meta, "grown") == null);
    try testing.expect(std.mem.indexOf(u8, meta, "\"v\":4") != null);
    try testing.expect(growFileSize(&plain, plain.root, plain.entries.items[0].id, "g00001.safetensors") == null);
    for (plain.entries.items[0].ssm_positions) |p| try expectGrowPoint(&plain, plain.entries.items[0].id, p, s);

    // A split tier whose layers keep no rows stores the same whole points.
    var none = try DiskTier.init(testing.allocator, io, base, "fp-none", 0, 128);
    defer none.deinit();
    none.ssm_max_per_entry = 160;
    try none.setGrownSpec(grow_span, &[_]u32{ 0, 0, 0, 0 }, 0);
    try commitGrowPoints(&none, &cache, &tokens, &.{ 128, 384 }, s);
    try testing.expectEqual(@as(usize, 0), none.entries.items[0].grown.len);
    try testing.expectEqualSlices(u64, plain.entries.items[0].ssm_bytes, none.entries.items[0].ssm_bytes);
}

test "DiskTier: SSM checkpoints persist incrementally under the flush byte cap" {
    // The per-flush byte cap covers BOTH chunks and checkpoints so a big 27B
    // turn never stalls the next request. Under a 1-byte cap the entry
    // persists one unit at a time and reports incomplete until KV + every
    // eligible checkpoint have landed.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-ssmcap", 0, 128);
    defer tier.deinit();
    tier.max_flush_bytes = 1; // one chunk/checkpoint per flush

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var src128 = buildHybridEntries(s, 100.0, 500.0);
    defer freeHybridEntries(&src128);
    var src512 = buildHybridEntries(s, 300.0, 700.0);
    defer freeHybridEntries(&src512);
    var cps = [_]trunk.SSMCheckpoint{
        try trunk.captureSsmCheckpoint(testing.allocator, &src128, 128, s),
        try trunk.captureSsmCheckpoint(testing.allocator, &src512, 512, s),
    };
    defer for (&cps) |*cp| cp.deinit(testing.allocator);

    // Drive to completion; it must take multiple flushes and only report
    // complete once BOTH checkpoints are on disk.
    var complete: PersistOutcome = .partial;
    var guard: u32 = 0;
    while (guard < 40) : (guard += 1) {
        complete = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &cps, s);
        if (complete == .persisted) break;
    }
    try testing.expectEqual(PersistOutcome.persisted, complete);
    const e = &tier.entries.items[0];
    try testing.expectEqual(@as(u32, 600), e.kv_len);
    try testing.expectEqual(@as(usize, 2), e.ssm_positions.len);
    try testing.expectEqual(@as(u32, 128), e.ssm_positions[0]);
    try testing.expectEqual(@as(u32, 512), e.ssm_positions[1]);

    // The incrementally-persisted checkpoints restore correctly.
    var cache2 = try KVCache.init(testing.allocator, 3);
    defer cache2.deinit();
    var dst: [3]SSMCacheEntry = .{
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
    };
    defer freeHybridEntries(&dst);
    try testing.expectEqual(@as(u32, 512), try tier.restoreIntoHybrid(&cache2, &dst, 0, 512, s));
    try testing.expectEqual(@as(f32, 300.0), ssmArrVal(dst[0].conv_state, 0, s));
    try testing.expectEqual(@as(f32, 700.0), ssmArrVal(dst[0].ssm_state, 0, s));
}

test "DiskTier: short caches are never persisted" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-skip", 0, 128);
    defer tier.deinit();

    // Below MIN_PERSIST_TOKENS → skipped.
    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try fillCache(&cache, s, 1, 128, 8, 0.0, .float32);
    var tokens: [128]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
    try testing.expectEqual(@as(usize, 0), tier.entryCount());
}

test "DiskTier: v4 spec snapshots round-trip; geometry mismatches decline; v3 restores clean" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-spec", 0, 128);

    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try fillCache(&cache, s, 2, 600, 8, 0.0, .float32);
    // dflash assistant context: 2 layers over the full 600 positions, base 0;
    // MTP committed history: 1 layer, 590 (the deferred-stash lag), base 0.
    var dfl = try KVCache.init(testing.allocator, 2);
    defer dfl.deinit();
    try fillCache(&dfl, s, 2, 600, 8, 3.5, .float32);
    var mtp = try KVCache.init(testing.allocator, 1);
    defer mtp.deinit();
    try fillCache(&mtp, s, 1, 590, 8, 9.5, .float32);

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    _ = try tier.appendCommitWithSpec(
        cache.entries,
        cache.step,
        cache.config,
        &tokens,
        false,
        null,
        .{ .entries = dfl.entries, .step = dfl.step, .config = dfl.config, .base_pos = 0 },
        .{ .entries = mtp.entries, .step = mtp.step, .config = mtp.config, .base_pos = 0 },
        s,
    );
    try testing.expect(tier.entries.items[0].spec_bytes > 0);

    // Restart shape: a fresh tier over the same root re-reads the spec meta.
    tier.deinit();
    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-spec", 0, 128);
    try testing.expectEqual(@as(usize, 1), tier2.entryCount());
    const m = tier2.bestMatch(&tokens, false, kv_quant.KVQuantConfig.dense, .{}).?;

    var loaded = tier2.loadSpecSnap(m.idx, .dflash, 2, kv_quant.KVQuantConfig.dense) orelse
        return error.TestExpectedSpecSnap;
    try testing.expectEqual(@as(usize, 0), loaded.base);
    try testing.expectEqual(@as(usize, 600), loaded.snap.step);
    var dfl2 = try KVCache.init(testing.allocator, 2);
    defer dfl2.deinit();
    try dfl2.restore(&loaded.snap);
    loaded.snap.deinit();
    // Exact values, K and V (V = -K in fillCache — a swap can't false-pass).
    const probes = [_][2]u32{ .{ 0, 0 }, .{ 300, 3 }, .{ 599, 7 } };
    for (probes) |p| {
        var li: u32 = 0;
        while (li < 2) : (li += 1) {
            try testing.expectEqual(
                try cacheValueAt(&dfl, li, p[0], p[1], s),
                try cacheValueAt(&dfl2, li, p[0], p[1], s),
            );
            try testing.expectEqual(
                try cacheBufValueAt(&dfl, li, p[0], p[1], s, true),
                try cacheBufValueAt(&dfl2, li, p[0], p[1], s, true),
            );
        }
    }

    var mloaded = tier2.loadSpecSnap(m.idx, .mtp, 1, kv_quant.KVQuantConfig.dense) orelse
        return error.TestExpectedSpecSnap;
    defer mloaded.snap.deinit();
    try testing.expectEqual(@as(usize, 590), mloaded.snap.step);

    // A target the geometry doesn't fit DECLINES (KVCache.restore asserts
    // equal layer counts — the check must fire before it).
    try testing.expect(tier2.loadSpecSnap(m.idx, .dflash, 3, kv_quant.KVQuantConfig.dense) == null);
    try testing.expect(tier2.loadSpecSnap(m.idx, .dflash, 2, kv_quant.KVQuantConfig.affine(8)) == null);
    try testing.expect(tier2.loadSpecSnap(m.idx, .mtp, 1, kv_quant.KVQuantConfig.affine(8)) == null);

    // A new entry committed WITHOUT spec payloads carries none.
    var tokens_b: [600]u32 = undefined;
    for (&tokens_b, 0..) |*t, i| t.* = @intCast(i + 900_000);
    _ = try tier2.appendCommit(cache.entries, cache.step, cache.config, &tokens_b, false, null, s);
    const mb = tier2.bestMatch(&tokens_b, false, kv_quant.KVQuantConfig.dense, .{}).?;
    try testing.expect(tier2.loadSpecSnap(mb.idx, .dflash, 2, kv_quant.KVQuantConfig.dense) == null);

    // v3 entry (written by an older binary): rewrite the manifest to v3 with
    // no spec object — the entry must restore CLEAN, spec-less.
    {
        const e_id = tier2.entries.items[m.idx].id;
        const meta_path = try std.fmt.allocPrint(testing.allocator, "{s}/fp-spec/e{d}/meta.json", .{ base, e_id });
        defer testing.allocator.free(meta_path);
        const content = kv_disk_fs.testRead(io, meta_path, 64 * 1024) orelse return error.TestMetaUnreadable;
        defer testing.allocator.free(content);
        const spec_at = std.mem.indexOf(u8, content, ",\"spec\":") orelse return error.TestSpecFieldMissing;
        var rewritten = std.ArrayList(u8).empty;
        defer rewritten.deinit(testing.allocator);
        try rewritten.appendSlice(testing.allocator, content[0..spec_at]);
        try rewritten.append(testing.allocator, '}');
        // The manifest stamps the lowest version that describes the entry.
        _ = std.mem.replace(u8, rewritten.items, "\"v\":6", "\"v\":3", rewritten.items);
        _ = std.mem.replace(u8, rewritten.items, "\"v\":5", "\"v\":3", rewritten.items);
        _ = std.mem.replace(u8, rewritten.items, "\"v\":4", "\"v\":3", rewritten.items);
        try kv_disk_fs.testWrite(io, meta_path, rewritten.items);
    }
    tier2.deinit();
    var tier3 = try DiskTier.init(testing.allocator, io, base, "fp-spec", 0, 128);
    defer tier3.deinit();
    try testing.expectEqual(@as(usize, 2), tier3.entryCount());
    const m3 = tier3.bestMatch(&tokens, false, kv_quant.KVQuantConfig.dense, .{}).?;
    try testing.expectEqual(@as(u32, 600), m3.usable);
    try testing.expect(tier3.loadSpecSnap(m3.idx, .dflash, 2, kv_quant.KVQuantConfig.dense) == null);
    var cache3 = try KVCache.init(testing.allocator, 2);
    defer cache3.deinit();
    const restored = try tier3.restoreInto(&cache3, m3.idx, s);
    try testing.expectEqual(@as(u32, 600), restored);
}

test "DiskTier: an extended spec sidecar writes only its new segments and the tail, and restores whole after a restart" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);
    var tier = try DiskTier.init(testing.allocator, io, base, "fp-spec-segs", 0, 128);
    tier.seg_span = 128;
    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try fillCache(&cache, s, 2, 600, 8, 0.0, .float32);
    var mtp = try KVCache.init(testing.allocator, 1);
    defer mtp.deinit();
    try fillCache(&mtp, s, 1, 590, 8, 9.5, .float32);
    var tokens: [800]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    _ = try tier.appendCommitWithSpec(cache.entries, cache.step, cache.config, tokens[0..600], false, null, null, .{ .entries = mtp.entries, .step = mtp.step, .config = mtp.config, .base_pos = 0 }, s);
    const first = wroteKind(&tier, .spec);
    try testing.expectEqual(@as(usize, 4), tier.entries.items[0].spec_segs.len);

    try fillCache(&cache, s, 2, 200, 8, 600.0, .float32);
    try fillCache(&mtp, s, 1, 200, 8, 9.5 + 590 * 8, .float32);
    _ = try tier.appendCommitWithSpec(cache.entries, cache.step, cache.config, &tokens, false, null, null, .{ .entries = mtp.entries, .step = mtp.step, .config = mtp.config, .base_pos = 0 }, s);
    const e = &tier.entries.items[0];
    try testing.expectEqual(@as(usize, 6), e.spec_segs.len);
    // Segments 0..3 are kept: the extension wrote segments 4 and 5 and the tail file.
    try testing.expectEqual(e.spec_segs[4].bytes + e.spec_segs[5].bytes + e.spec_bytes, wroteKind(&tier, .spec) - first);

    tier.deinit();
    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-spec-segs", 0, 128);
    defer tier2.deinit();
    const m = tier2.bestMatch(&tokens, false, kv_quant.KVQuantConfig.dense, .{}).?;
    var loaded = tier2.loadSpecSnap(m.idx, .mtp, 1, kv_quant.KVQuantConfig.dense) orelse return error.TestExpectedSpecSnap;
    try testing.expectEqual(@as(usize, 790), loaded.snap.step);
    var mtp2 = try KVCache.init(testing.allocator, 1);
    defer mtp2.deinit();
    try mtp2.restore(&loaded.snap);
    loaded.snap.deinit();
    inline for (.{ 0, 127, 128, 300, 640, 789 }) |pos| {
        try testing.expectEqual(try cacheValueAt(&mtp, 0, pos, 3, s), try cacheValueAt(&mtp2, 0, pos, 3, s));
        try testing.expectEqual(try cacheBufValueAt(&mtp, 0, pos, 5, s, true), try cacheBufValueAt(&mtp2, 0, pos, 5, s, true));
    }
}

test "DiskTier: a diverging entry links the donor's spec segments, bills them once, and restores them after the donor goes" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);
    var tier = try DiskTier.init(testing.allocator, io, base, "fp-spec-heir", 0, 128);
    defer tier.deinit();
    tier.seg_span = 128;
    tier.ssd_first = true;
    tier.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try fillCache(&cache, s, 2, 600, 8, 0.0, .float32);
    var mtp = try KVCache.init(testing.allocator, 1);
    defer mtp.deinit();
    try fillCache(&mtp, s, 1, 590, 8, 9.5, .float32);
    const spec: SpecCommit = .{ .entries = mtp.entries, .step = mtp.step, .config = mtp.config, .base_pos = 0 };
    const toks = chunkShareTokens(520);
    _ = try tier.appendCommitWithSpec(cache.entries, cache.step, cache.config, &toks.a, false, null, null, spec, s);
    const donor_total = tier.total_bytes;
    const before = wroteKind(&tier, .spec);
    _ = try tier.appendCommitWithSpec(cache.entries, cache.step, cache.config, &toks.b, false, null, null, spec, s);
    const heir = &tier.entries.items[1];
    try testing.expectEqual(@as(u32, 4), heir.spec_segs_inherited);
    try testing.expectEqual(heir.spec_bytes, wroteKind(&tier, .spec) - before);
    try testing.expectEqual(donor_total + heir.bytes, tier.total_bytes);
    tier.removeAt(0);
    var loaded = tier.loadSpecSnap(0, .mtp, 1, kv_quant.KVQuantConfig.dense) orelse return error.TestExpectedSpecSnap;
    var mtp2 = try KVCache.init(testing.allocator, 1);
    defer mtp2.deinit();
    try mtp2.restore(&loaded.snap);
    loaded.snap.deinit();
    inline for (.{ 0, 300, 589 }) |pos| try testing.expectEqual(try cacheValueAt(&mtp, 0, pos, 3, s), try cacheValueAt(&mtp2, 0, pos, 3, s));
}

test "DiskTier: a prefill write-through without spec leaves the entry spec-less and its segments spare, so the turn's end writes only the new ones" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);
    var tier = try DiskTier.init(testing.allocator, io, base, "fp-spec-keep", 0, 128);
    defer tier.deinit();
    tier.seg_span = 128;
    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try fillCache(&cache, s, 2, 600, 8, 0.0, .float32);
    var mtp = try KVCache.init(testing.allocator, 1);
    defer mtp.deinit();
    try fillCache(&mtp, s, 1, 590, 8, 9.5, .float32);
    var tokens: [800]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    _ = try tier.appendCommitWithSpec(cache.entries, cache.step, cache.config, tokens[0..600], false, null, null, .{ .entries = mtp.entries, .step = mtp.step, .config = mtp.config, .base_pos = 0 }, s);
    try fillCache(&cache, s, 2, 200, 8, 600.0, .float32);
    _ = try tier.appendCommitWithSpec(cache.entries, 700, cache.config, tokens[0..700], false, null, null, null, s);
    // A restore sees what it always saw after such a commit: no spec. The bill is the bytes on disk.
    try testing.expectEqual(@as(usize, 4), tier.entries.items[0].spec_segs.len);
    try testing.expect(tier.loadSpecSnap(0, .mtp, 1, kv_quant.KVQuantConfig.dense) == null);
    try testing.expectEqual(try uniqueDirBytes(io, &tier), tier.total_bytes);
    const before = wroteKind(&tier, .spec);
    try fillCache(&mtp, s, 1, 200, 8, 9.5 + 590 * 8, .float32);
    _ = try tier.appendCommitWithSpec(cache.entries, cache.step, cache.config, &tokens, false, null, null, .{ .entries = mtp.entries, .step = mtp.step, .config = mtp.config, .base_pos = 0 }, s);
    const e = &tier.entries.items[0];
    try testing.expectEqual(@as(usize, 6), e.spec_segs.len);
    try testing.expectEqual(e.spec_segs[4].bytes + e.spec_segs[5].bytes + e.spec_bytes, wroteKind(&tier, .spec) - before);
}

test "DiskTier: spare spec segments survive a restart, billed, with the entry still spec-less" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);
    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try fillCache(&cache, s, 2, 700, 8, 0.0, .float32);
    var mtp = try KVCache.init(testing.allocator, 1);
    defer mtp.deinit();
    try fillCache(&mtp, s, 1, 590, 8, 9.5, .float32);
    var tokens: [700]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var total: u64 = 0;
    {
        var tier = try DiskTier.init(testing.allocator, io, base, "fp-spec-spare", 0, 128);
        defer tier.deinit();
        tier.seg_span = 128;
        _ = try tier.appendCommitWithSpec(cache.entries, 600, cache.config, tokens[0..600], false, null, null, .{ .entries = mtp.entries, .step = mtp.step, .config = mtp.config, .base_pos = 0 }, s);
        _ = try tier.appendCommitWithSpec(cache.entries, 700, cache.config, &tokens, false, null, null, null, s);
        total = tier.total_bytes;
    }
    var scanned = try DiskTier.init(testing.allocator, io, base, "fp-spec-spare", 0, 128);
    defer scanned.deinit();
    try testing.expectEqual(@as(usize, 1), scanned.entryCount());
    try testing.expectEqual(@as(usize, 4), scanned.entries.items[0].spec_segs.len);
    try testing.expect(scanned.loadSpecSnap(0, .mtp, 1, kv_quant.KVQuantConfig.dense) == null);
    try testing.expectEqual(total, scanned.total_bytes);
    try testing.expectEqual(try uniqueDirBytes(io, &scanned), scanned.total_bytes);
}

test "DiskTier: an entry the prefill wrote without spec links the spec segments of the record it grew from when it completes" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);
    var tier = try DiskTier.init(testing.allocator, io, base, "fp-spec-wt", 0, 128);
    defer tier.deinit();
    tier.seg_span = 128;
    tier.ssd_first = true;
    tier.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try fillCache(&cache, s, 2, 600, 8, 0.0, .float32);
    var mtp = try KVCache.init(testing.allocator, 1);
    defer mtp.deinit();
    try fillCache(&mtp, s, 1, 590, 8, 9.5, .float32);
    const spec: SpecCommit = .{ .entries = mtp.entries, .step = mtp.step, .config = mtp.config, .base_pos = 0 };
    const toks = chunkShareTokens(520);
    _ = try tier.appendCommitWithSpec(cache.entries, cache.step, cache.config, &toks.a, false, null, null, spec, s);
    _ = try tier.appendCommitWithSpec(cache.entries, 560, cache.config, toks.b[0..560], false, null, null, null, s);
    try testing.expectEqual(@as(usize, 0), tier.entries.items[1].spec_segs.len);
    const heir_id = tier.entries.items[1].id;
    const before = wroteKind(&tier, .spec);
    _ = try tier.appendCommitWithSpec(cache.entries, cache.step, cache.config, &toks.b, false, null, null, spec, s);
    try testing.expectEqual(@as(usize, 2), tier.entryCount());
    const heir = &tier.entries.items[1];
    try testing.expectEqual(heir_id, heir.id);
    try testing.expectEqual(@as(u32, 4), heir.spec_segs_inherited);
    try testing.expectEqual(heir.spec_bytes, wroteKind(&tier, .spec) - before);
}

test "DiskTier: a spec sidecar replaced before its manifest commits is declined, never relabelled" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-spec-crash", 0, 128);
    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try fillCache(&cache, s, 2, 600, 8, 0.0, .float32);
    var old_win = try KVCache.init(testing.allocator, 2);
    defer old_win.deinit();
    try fillCache(&old_win, s, 2, 300, 8, 3.5, .float32);
    var new_win = try KVCache.init(testing.allocator, 2);
    defer new_win.deinit();
    try fillCache(&new_win, s, 2, 300, 8, 700.5, .float32);

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    _ = try tier.appendCommitWithSpec(
        cache.entries,
        cache.step,
        cache.config,
        &tokens,
        false,
        null,
        .{ .entries = old_win.entries, .step = old_win.step, .config = old_win.config, .base_pos = 100 },
        null,
        s,
    );
    const e = tier.entries.items[0];
    const old_bytes = e.spec_bytes;

    // The crash: the next commit's sidecar (same shape, later rows) landed, its manifest did not.
    const dir = try std.fmt.allocPrint(testing.allocator, "{s}/e{d}", .{ tier.root, e.id });
    defer testing.allocator.free(dir);
    const res = try tier.writeSpecSidecar(dir, .{ .entries = new_win.entries, .step = new_win.step, .config = new_win.config, .base_pos = 200 }, null, s, .{});
    try testing.expectEqual(old_bytes, res.bytes);

    tier.deinit();
    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-spec-crash", 0, 128);
    defer tier2.deinit();
    const m = tier2.bestMatch(&tokens, false, kv_quant.KVQuantConfig.dense, .{}).?;
    try testing.expect(tier2.loadSpecSnap(m.idx, .dflash, 2, kv_quant.KVQuantConfig.dense) == null);
    var cache2 = try KVCache.init(testing.allocator, 2);
    defer cache2.deinit();
    try testing.expectEqual(@as(u32, 600), try tier2.restoreInto(&cache2, m.idx, s));
}

test "DiskTier: a dense MTP sidecar is rewritten at affine-8 on the next commit" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-mtp-rewrite", 0, 128);
    defer tier.deinit();

    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try fillCache(&cache, s, 2, 600, 8, 0.0, .float32);
    var mtp_dense = try KVCache.init(testing.allocator, 1);
    defer mtp_dense.deinit();
    try fillCache(&mtp_dense, s, 1, 590, 8, 9.5, .float32);

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    _ = try tier.appendCommitWithSpec(
        cache.entries,
        cache.step,
        cache.config,
        &tokens,
        false,
        null,
        null,
        .{ .entries = mtp_dense.entries, .step = mtp_dense.step, .config = mtp_dense.config, .base_pos = 0 },
        s,
    );
    try testing.expectEqual(kv_quant.KVQuantConfig.dense, tier.entries.items[0].spec_mtp.?.quant);
    try testing.expect(tier.loadSpecSnap(0, .mtp, 1, kv_quant.KVQuantConfig.affine(8)) == null);

    const q8 = kv_quant.KVQuantConfig.affine(8);
    var mtp_q = try KVCache.initWithConfig(testing.allocator, 1, q8);
    defer mtp_q.deinit();
    try fillCache(&mtp_q, s, 1, 590, 64, 9.5, .bfloat16);
    _ = try tier.appendCommitWithSpec(
        cache.entries,
        cache.step,
        cache.config,
        &tokens,
        false,
        null,
        null,
        .{ .entries = mtp_q.entries, .step = mtp_q.step, .config = mtp_q.config, .base_pos = 0 },
        s,
    );
    try testing.expectEqual(q8, tier.entries.items[0].spec_mtp.?.quant);
    var loaded = tier.loadSpecSnap(0, .mtp, 1, q8) orelse return error.TestExpectedSpecSnap;
    defer loaded.snap.deinit();
    try testing.expectEqual(@as(usize, 590), loaded.snap.step);
}

test "DiskTier: a quantized MTP sidecar is rewritten to dense on the next commit" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-mtp-rewrite-dense", 0, 128);
    defer tier.deinit();

    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try fillCache(&cache, s, 2, 600, 8, 0.0, .float32);
    const q8 = kv_quant.KVQuantConfig.affine(8);
    var mtp_q = try KVCache.initWithConfig(testing.allocator, 1, q8);
    defer mtp_q.deinit();
    try fillCache(&mtp_q, s, 1, 590, 64, 9.5, .bfloat16);

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    _ = try tier.appendCommitWithSpec(
        cache.entries,
        cache.step,
        cache.config,
        &tokens,
        false,
        null,
        null,
        .{ .entries = mtp_q.entries, .step = mtp_q.step, .config = mtp_q.config, .base_pos = 0 },
        s,
    );
    try testing.expectEqual(q8, tier.entries.items[0].spec_mtp.?.quant);
    try testing.expect(tier.loadSpecSnap(0, .mtp, 1, kv_quant.KVQuantConfig.dense) == null);

    var mtp_dense = try KVCache.init(testing.allocator, 1);
    defer mtp_dense.deinit();
    try fillCache(&mtp_dense, s, 1, 590, 8, 9.5, .float32);
    _ = try tier.appendCommitWithSpec(
        cache.entries,
        cache.step,
        cache.config,
        &tokens,
        false,
        null,
        null,
        .{ .entries = mtp_dense.entries, .step = mtp_dense.step, .config = mtp_dense.config, .base_pos = 0 },
        s,
    );
    try testing.expectEqual(kv_quant.KVQuantConfig.dense, tier.entries.items[0].spec_mtp.?.quant);
    var loaded = tier.loadSpecSnap(0, .mtp, 1, kv_quant.KVQuantConfig.dense) orelse return error.TestExpectedSpecSnap;
    defer loaded.snap.deinit();
    try testing.expectEqual(@as(usize, 590), loaded.snap.step);
}

test "DiskTier: a swallowed commit failure drops the latch it raised and keeps a foreign one" {
    try mlx.requireGpu();
    // Bar: the appendCommit funnel's errdefer removes only the error THIS call raised.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-funnel", 0, 128);
    defer tier.deinit();
    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try fillCache(&cache, s, 2, 600, 8, 0.0, .float32);

    var src = buildHybridEntries(s, 100.0, 500.0);
    defer freeHybridEntries(&src);
    var cps = [_]trunk.SSMCheckpoint{
        try trunk.captureSsmCheckpoint(testing.allocator, &src, 128, s),
    };
    defer for (&cps) |*cp| cp.deinit(testing.allocator);

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 31);

    // Arm a LATCHING fault at the N-th checked op after this line; the commit
    // must reach it (fired) and its errdefer must clear the latched message.
    mlx.armLatchingFaultForTest(3);
    try testing.expectError(error.MlxError, tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &cps, s));
    const fired_own = mlx.latchingFaultFiredForTest();
    const pending_own = mlx.errorPending();
    mlx.armLatchingFaultForTest(0);
    try testing.expect(fired_own);
    try testing.expect(!pending_own);

    // A foreign latch predates the call and must survive it untouched.
    mlx.latchErrorForTest("foreign pre-existing error");
    mlx.armLatchingFaultForTest(3);
    _ = tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &cps, s) catch {};
    const fired_foreign = mlx.latchingFaultFiredForTest();
    mlx.armLatchingFaultForTest(0);
    try testing.expect(fired_foreign);
    var msg: [512]u8 = undefined;
    try testing.expectEqualStrings("foreign pre-existing error", mlx.takeError(&msg).?);
}

test "DiskTier: an entry's ring files thin as its RAM checkpoints do: the lowest and the newest stay" {
    var positions = [_]u32{ 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000 };
    const kept = DiskTier.thinRingPositions(&positions);
    try std.testing.expectEqual(RING_DISK_MAX_PER_ENTRY, kept.len);
    try std.testing.expectEqual(@as(u32, 100), kept[0]);
    try std.testing.expectEqual(@as(u32, 900), kept[kept.len - 2]);
    try std.testing.expectEqual(@as(u32, 1000), kept[kept.len - 1]);
    var few = [_]u32{ 100, 1000 };
    try std.testing.expectEqualSlices(u32, &.{ 100, 1000 }, DiskTier.thinRingPositions(&few));
}

test "DiskTier: SSD-first writes a checkpoint beside the chunk that closes it" {
    // Checkpoints ride outside the per-flush byte budget, so the first flush of a long hybrid
    // entry already restores; without it (arm B) the entry carries KV with no recurrent state.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var src128 = buildHybridEntries(s, 100.0, 500.0);
    defer freeHybridEntries(&src128);
    var cps = [_]trunk.SSMCheckpoint{
        try trunk.captureSsmCheckpoint(testing.allocator, &src128, 128, s),
    };
    defer for (&cps) |*cp| cp.deinit(testing.allocator);

    // Arm A: SSD-first. One flush, one chunk, and its checkpoint.
    {
        var tier = try DiskTier.init(testing.allocator, io, base, "fp-ssdfirst-cp", 0, 128);
        defer tier.deinit();
        tier.ssd_first = true;
        tier.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
        tier.max_flush_bytes = 1; // bound the flush to one chunk

        const complete = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &cps, s);
        try testing.expectEqual(PersistOutcome.partial, complete); // KV is still partial
        const e = &tier.entries.items[0];
        try testing.expectEqual(@as(u32, 128), e.kv_len);
        try testing.expectEqual(@as(usize, 1), e.ssm_positions.len);
        try testing.expectEqual(@as(u32, 128), e.ssm_positions[0]);

        var cache2 = try KVCache.init(testing.allocator, 3);
        defer cache2.deinit();
        var dst: [3]SSMCacheEntry = .{
            .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
            .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
            .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        };
        defer freeHybridEntries(&dst);
        try testing.expectEqual(@as(u32, 128), try tier.restoreIntoHybrid(&cache2, &dst, 0, 128, s));
        try testing.expectEqual(@as(f32, 100.0), ssmArrVal(dst[0].conv_state, 0, s));
    }

    // Arm B: the chunk eats the budget, no checkpoint.
    {
        var tier = try DiskTier.init(testing.allocator, io, base, "fp-legacy-cp", 0, 128);
        defer tier.deinit();
        tier.max_flush_bytes = 1;

        _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &cps, s);
        const e = &tier.entries.items[0];
        try testing.expectEqual(@as(u32, 128), e.kv_len);
        try testing.expectEqual(@as(usize, 0), e.ssm_positions.len);
    }
}

test "DiskTier: the staged encoder carries __metadata__, and refuses a value it cannot escape" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-meta", 0, 128);
    defer tier.deinit();

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try fillCache(&cache, s, 1, 128, 8, 0.0, .float32);

    var list = std.ArrayList(DiskTier.NamedTensor).empty;
    defer tier.freeNamed(&list);
    try tier.appendSlice(&list, 0, "k", cache.entries[0].keys.array, 0, 128, s);

    const meta = [_]DiskTier.MetaPair{
        .{ .key = "layers", .value = "3" },
        .{ .key = "init", .value = "0,1,2" },
        .{ .key = "qsa_ratio", .value = "4" },
    };
    const bytes = try tier.serializeNamed(list.items, &meta, s);
    defer testing.allocator.free(bytes);

    try tmp.dir.writeFile(io, .{ .sub_path = "meta-probe.safetensors", .data = bytes });
    const path = try std.fmt.allocPrint(testing.allocator, "{s}/meta-probe.safetensors", .{base});
    defer testing.allocator.free(path);

    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    var loaded = try safetensors.loadFile(testing.allocator, io, path, cpu);
    defer loaded.deinit();
    inline for (.{ .{ "layers", "3" }, .{ "init", "0,1,2" }, .{ "qsa_ratio", "4" } }) |kv| {
        try testing.expectEqualStrings(kv[1], loaded.getMeta(kv[0]).?);
    }
    try testing.expect(loaded.get("l0.k") != null);

    // A value with a quote would produce a header a reader cannot parse.
    const bad = [_]DiskTier.MetaPair{.{ .key = "init", .value = "0\",\"x" }};
    try testing.expectError(error.DiskCacheBadMetadata, tier.encodeNamed(list.items, &bad));
}

test "DiskTier: a writer-armed commit lands the whole entry in bounded pieces, in one call" {
    // A long prompt is whole on disk after its first turn, its checkpoints outside every piece's bound.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var src = buildHybridEntries(s, 100.0, 500.0);
    defer freeHybridEntries(&src);
    var cps = [_]trunk.SSMCheckpoint{
        try trunk.captureSsmCheckpoint(testing.allocator, &src, 256, s),
        try trunk.captureSsmCheckpoint(testing.allocator, &src, 512, s),
    };
    defer for (&cps) |*cp| cp.deinit(testing.allocator);

    for ([_]bool{ false, true }, [_][]const u8{ "fp-pieces-ram", "fp-pieces-ssd" }) |ssd_first, fp| {
        var tier = try DiskTier.init(testing.allocator, io, base, fp, 0, 128);
        defer tier.deinit();
        tier.ssd_first = ssd_first;
        if (ssd_first) tier.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
        tier.enableBackgroundWriter();
        tier.max_flush_bytes = 1; // one chunk per piece

        try testing.expectEqual(PersistOutcome.persisted, try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &cps, s));
        tier.drainWriter();
        try testing.expectEqual(@as(u64, 0), tier.writeErrors());
        const e = &tier.entries.items[0];
        try testing.expectEqual(@as(u32, 600), e.kv_len);
        try testing.expectEqualSlices(u32, &.{ 256, 512 }, e.ssm_positions);
        try testing.expect(tier.entryWholeOnDisk(e.id));

        var out = try KVCache.init(testing.allocator, 3);
        defer out.deinit();
        var dst: [3]SSMCacheEntry = .{
            .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
            .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
            .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        };
        defer freeHybridEntries(&dst);
        try testing.expectEqual(@as(u32, 512), try tier.restoreIntoHybrid(&out, &dst, 0, 512, s));
        try testing.expectEqual(@as(f32, 100.0), ssmArrVal(dst[0].conv_state, 0, s));
        for ([_]u32{ 0, 2 }) |li| for ([_]u32{ 0, 255, 256, 511 }) |pos| {
            try testing.expectEqual(try cacheValueAt(&cache, li, pos, 3, s), try cacheValueAt(&out, li, pos, 3, s));
        };
    }
}

test "DiskTier: a sole entry over the byte budget is cut to a checkpoint that fits, and stays within it" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    var src = buildHybridEntries(s, 100.0, 500.0);
    defer freeHybridEntries(&src);
    var cps = [_]trunk.SSMCheckpoint{
        try trunk.captureSsmCheckpoint(testing.allocator, &src, 256, s),
        try trunk.captureSsmCheckpoint(testing.allocator, &src, 512, s),
    };
    defer for (&cps) |*cp| cp.deinit(testing.allocator);

    // 600 rows of 192 bytes and their files are ~121 KB; 512 rows and their files ~104 KB.
    const budget: u64 = 112_000;
    for ([_]bool{ false, true }) |writer| {
        const fp: []const u8 = if (writer) "fp-sole-writer" else "fp-sole-sync";
        {
            var tier = try DiskTier.init(testing.allocator, io, base, fp, budget, 128);
            defer tier.deinit();
            if (writer) tier.enableBackgroundWriter();
            _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, &cps, s);
            tier.drainWriter();
            try testing.expectEqual(@as(usize, 1), tier.entryCount());
            try testing.expect(tier.total_bytes <= budget);
            try testing.expectEqual(@as(u32, 512), tier.entries.items[0].kv_len);
        }
        // Reopened, the tier still holds it within the budget and restores at the checkpoint.
        var tier2 = try DiskTier.init(testing.allocator, io, base, fp, budget, 128);
        defer tier2.deinit();
        try testing.expectEqual(@as(usize, 1), tier2.entryCount());
        try testing.expect(tier2.total_bytes <= budget);
        var out = try KVCache.init(testing.allocator, 3);
        defer out.deinit();
        var dst: [3]SSMCacheEntry = .{
            .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
            .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
            .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        };
        defer freeHybridEntries(&dst);
        try testing.expectEqual(@as(u32, 512), try tier2.restoreIntoHybrid(&out, &dst, 0, 512, s));
        try testing.expectEqual(try cacheValueAt(&cache, 2, 511, 3, s), try cacheValueAt(&out, 2, 511, 3, s));
    }
}

test "DiskTier: a multi-piece commit stages its spec sidecar once, through the writer" {
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
    var mtp = try KVCache.init(testing.allocator, 1);
    defer mtp.deinit();
    try fillCache(&mtp, s, 1, 590, 8, 9.5, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-spec-pieces", 0, 128);
    tier.enableBackgroundWriter();
    tier.max_flush_bytes = 1; // one chunk per piece: five pieces
    tier.writer.?.setPaused(true);
    defer if (tier.writer) |w| w.setPaused(false);
    try testing.expectEqual(PersistOutcome.persisted, try tier.appendCommitWithSpec(
        cache.entries,
        cache.step,
        cache.config,
        &tokens,
        false,
        null,
        .{ .entries = dfl.entries, .step = dfl.step, .config = dfl.config, .base_pos = 0 },
        .{ .entries = mtp.entries, .step = mtp.step, .config = mtp.config, .base_pos = 0 },
        s,
    ));
    // Nothing was written on this thread, and the sidecar is staged once.
    try testing.expect(kv_disk_fs.testStatAt(io, base, "fp-spec-pieces/e1/spec.safetensors") == null);
    var paths = std.ArrayList([]const u8).empty;
    defer {
        for (paths.items) |p| testing.allocator.free(p);
        paths.deinit(testing.allocator);
    }
    try tier.writer.?.stagedPaths(&paths, testing.allocator);
    var specs: usize = 0;
    for (paths.items) |p| specs += @intFromBool(std.mem.endsWith(u8, p, "/spec.safetensors"));
    try testing.expectEqual(@as(usize, 1), specs);
    tier.writer.?.setPaused(false);
    tier.drainWriter();

    tier.deinit();
    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-spec-pieces", 0, 128);
    defer tier2.deinit();
    const m = tier2.bestMatch(&tokens, false, kv_quant.KVQuantConfig.dense, .{}).?;
    try testing.expectEqual(@as(u32, 600), m.usable);
    var loaded = tier2.loadSpecSnap(m.idx, .dflash, 2, kv_quant.KVQuantConfig.dense) orelse return error.TestExpectedSpecSnap;
    defer loaded.snap.deinit();
    try testing.expectEqual(@as(usize, 600), loaded.snap.step);
    var mloaded = tier2.loadSpecSnap(m.idx, .mtp, 1, kv_quant.KVQuantConfig.dense) orelse return error.TestExpectedSpecSnap;
    defer mloaded.snap.deinit();
    try testing.expectEqual(@as(usize, 590), mloaded.snap.step);
}

test "DiskTier: staged host bytes never pass the writer's permit; a file larger than it goes in parts" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    // 3 layers of 64-wide f32 K and V: each 128-token chunk file is ~192 KB, three permits.
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 64, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);
    {
        var tier = try DiskTier.init(testing.allocator, io, base, "fp-permit", 0, 128);
        defer tier.deinit();
        tier.enableBackgroundWriter();
        tier.writer.?.permit_bytes = 64 * 1024;
        try testing.expectEqual(PersistOutcome.persisted, try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s));
        tier.drainWriter();
        try testing.expectEqual(@as(u64, 0), tier.writeErrors());
        try testing.expect(tier.writer.?.peakBytes() <= 64 * 1024);
    }
    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-permit", 0, 128);
    defer tier2.deinit();
    var out = try KVCache.init(testing.allocator, 3);
    defer out.deinit();
    try testing.expectEqual(@as(u32, 600), try tier2.restoreInto(&out, 0, s));
    for ([_]u32{ 0, 2 }) |li| for ([_]u32{ 0, 127, 128, 599 }) |pos| {
        try testing.expectEqual(try cacheValueAt(&cache, li, pos, 63, s), try cacheValueAt(&out, li, pos, 63, s));
    };
}

test "DiskTier: SSD-first write-through extends without rewriting a persisted chunk" {
    // Write-through: a killed prefill leaves a restorable chunk-aligned prefix, and a persisted
    // chunk is never rewritten.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-ssd-wt", 0, 128);
    defer tier.deinit();
    tier.ssd_first = true;
    tier.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
    tier.enableBackgroundWriter();

    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    var tokens: [768]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // Chunk boundary 1: positions [0, 640) forwarded (5 chunks of 128).
    try fillCache(&cache, s, 2, 640, 8, 0.0, .float32);
    var src640 = buildHybridEntries(s, 11.0, 22.0);
    defer freeHybridEntries(&src640);
    var cps640 = [_]trunk.SSMCheckpoint{
        try trunk.captureSsmCheckpoint(testing.allocator, &src640, 640, s),
    };
    defer for (&cps640) |*cp| cp.deinit(testing.allocator);
    _ = try tier.appendCommit(cache.entries, 640, cache.config, tokens[0..640], false, &cps640, s);
    tier.drainWriter();
    try testing.expectEqual(@as(usize, 1), tier.entryCount());
    try testing.expectEqual(@as(u32, 640), tier.entries.items[0].kv_len);
    {
        var c2 = try KVCache.init(testing.allocator, 2);
        defer c2.deinit();
        var dst: [3]SSMCacheEntry = .{
            .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
            .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
            .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
        };
        defer freeHybridEntries(&dst);
        try testing.expectEqual(@as(u32, 640), try tier.restoreIntoHybrid(&c2, &dst, 0, 640, s));
    }
    tier.drainWriter(); // the restore re-indexes for LRU
    const written_after_first = tier.writer.?.filesWritten();

    // Chunk boundary 2: the prefill continues to 768. Only chunk 5 is new.
    try fillCache(&cache, s, 2, 128, 8, 640.0, .float32);
    tier.writer.?.setPaused(true);
    defer tier.writer.?.setPaused(false);
    _ = try tier.appendCommit(cache.entries, 768, cache.config, tokens[0..768], false, &cps640, s);
    var staged = std.ArrayList([]const u8).empty;
    defer {
        for (staged.items) |p| testing.allocator.free(p);
        staged.deinit(testing.allocator);
    }
    try tier.writer.?.stagedPaths(&staged, testing.allocator);
    for (staged.items) |p| {
        try testing.expect(std.mem.indexOf(u8, p, "/c000000.") == null);
        try testing.expect(std.mem.indexOf(u8, p, "/c000004.") == null);
    }
    try testing.expect(staged.items.len <= 2); // the new chunk + meta.json
    tier.writer.?.setPaused(false);
    tier.drainWriter();
    try testing.expect(tier.writer.?.filesWritten() - written_after_first <= 2);
    try testing.expectEqual(@as(usize, 1), tier.entryCount());
    try testing.expectEqual(@as(u32, 768), tier.entries.items[0].kv_len);
}

test "DiskTier: SSD-first declines to store when the VOLUME is short, and says so" {
    // 3 GiB free against a 512 GiB volume leaves nothing after the reserve: the tier stores nothing.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-short", 0, 128);
    defer tier.deinit();
    tier.ssd_first = true;
    tier.armTestSpace(3 * 1024 * 1024 * 1024, 512 * 1024 * 1024 * 1024);

    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try fillCache(&cache, s, 2, 640, 8, 0.0, .float32);
    var tokens: [640]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    _ = try tier.appendCommit(cache.entries, 640, cache.config, &tokens, false, null, s);
    tier.drainWriter();
    try testing.expect(tier.store_declined);
    try testing.expectEqual(@as(usize, 0), tier.entryCount());
    try testing.expectEqual(@as(u64, 0), tier.total_bytes);

    tier.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
    _ = try tier.appendCommit(cache.entries, 640, cache.config, &tokens, false, null, s);
    tier.drainWriter();
    try testing.expect(!tier.store_declined);
    try testing.expectEqual(@as(usize, 1), tier.entryCount());
    try testing.expectEqual(@as(u32, 640), tier.entries.items[0].kv_len);
}


test "DiskTier: entries cross the SSD-first boundary in BOTH directions (SSD-first itself bumps no manifest)" {
    // SSD-first changes when chunks are written and which checkpoints are present, never the
    // on-disk format: entries restore across both arms.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // Written by the legacy path, read by SSD-first.
    {
        var legacy = try DiskTier.init(testing.allocator, io, base, "fp-x-legacy", 0, 128);
        _ = try legacy.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
        legacy.deinit();

        var ssd = try DiskTier.init(testing.allocator, io, base, "fp-x-legacy", 0, 128);
        defer ssd.deinit();
        ssd.ssd_first = true;
        ssd.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
        ssd.enableBackgroundWriter();
        try testing.expectEqual(@as(usize, 1), ssd.entryCount());
        var out = try KVCache.init(testing.allocator, 3);
        defer out.deinit();
        try testing.expectEqual(@as(u32, 600), try ssd.restoreInto(&out, 0, s));
        for ([_]u32{ 0, 1, 2 }) |li| for ([_]u32{ 0, 127, 128, 599 }) |pos| {
            try testing.expectEqual(try cacheValueAt(&cache, li, pos, 3, s), try cacheValueAt(&out, li, pos, 3, s));
        };
    }

    // Written by SSD-first (background writer, hand-serialized), read by the legacy path.
    {
        var ssd = try DiskTier.init(testing.allocator, io, base, "fp-x-ssd", 0, 128);
        ssd.ssd_first = true;
        ssd.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
        ssd.enableBackgroundWriter();
        _ = try ssd.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
        ssd.drainWriter();
        ssd.deinit();

        var legacy = try DiskTier.init(testing.allocator, io, base, "fp-x-ssd", 0, 128);
        defer legacy.deinit();
        try testing.expect(legacy.writer == null);
        try testing.expectEqual(@as(usize, 1), legacy.entryCount());
        var out = try KVCache.init(testing.allocator, 3);
        defer out.deinit();
        try testing.expectEqual(@as(u32, 600), try legacy.restoreInto(&out, 0, s));
        for ([_]u32{ 0, 1, 2 }) |li| for ([_]u32{ 0, 127, 128, 599 }) |pos| {
            try testing.expectEqual(try cacheValueAt(&cache, li, pos, 3, s), try cacheValueAt(&out, li, pos, 3, s));
        };
        // No inherited chunks and no MTP head: stamps v4, the version an older reader accepts.
        const meta = (kv_disk_fs.testReadAt(io, base, "fp-x-ssd/e1/meta.json", 1 << 20) orelse return error.TestUnexpectedResult);
        defer testing.allocator.free(meta);
        try testing.expect(std.mem.indexOf(u8, meta, "\"v\":4") != null);
        try testing.expect(std.mem.indexOf(u8, meta, "\"v\":6") == null);
    }
}

test "DiskTier: the per-entry cap holds the whole 8192 grid to 1M tokens, and past it thins keeping both ends" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);
    var tier = try DiskTier.init(testing.allocator, io, base, "fp-spacing", 0, 128);
    defer tier.deinit();
    try testing.expectEqual(trunk.ThinPolicy.oldest, tier.cp_thin);
    try testing.expectEqual(SSM_DISK_MAX_PER_ENTRY_LEGACY, tier.ssm_max_per_entry);
    tier.cp_thin = .min_span_recency;
    tier.ssm_max_per_entry = SSM_DISK_MAX_PER_ENTRY;

    // The grid to 1,048,576 tokens and the prompt-end point just past it.
    var positions: [129]u32 = undefined;
    for (positions[0..128], 0..) |*p, i| p.* = @intCast((i + 1) * POINT_GRID);
    positions[128] = 128 * POINT_GRID + 24;
    const L: u32 = positions[128] + 8;
    const whole = try tier.ssmTargetPositions(&positions, &[_]trunk.SSMCheckpoint{}, L);
    defer testing.allocator.free(whole);
    try testing.expectEqualSlices(u32, &positions, whole);

    var more: [200]u32 = undefined;
    for (&more, 0..) |*p, i| p.* = @intCast((i + 1) * 1000);
    const kept = try tier.ssmTargetPositions(&more, &[_]trunk.SSMCheckpoint{}, 200_000);
    defer testing.allocator.free(kept);
    try testing.expectEqual(SSM_DISK_MAX_PER_ENTRY, kept.len);
    try testing.expectEqual(@as(u32, 1000), kept[0]);
    try testing.expectEqual(@as(u32, 200_000), kept[kept.len - 1]);

    // The ungated arm: every other arch keeps the previous retention (the highest N).
    var legacy = try DiskTier.init(testing.allocator, io, base, "fp-spacing-legacy", 0, 128);
    defer legacy.deinit();
    try testing.expectEqual(trunk.ThinPolicy.oldest, legacy.cp_thin);
    const old_kept = try legacy.ssmTargetPositions(&positions, &[_]trunk.SSMCheckpoint{}, L);
    defer testing.allocator.free(old_kept);
    try testing.expectEqualSlices(u32, positions[positions.len - SSM_DISK_MAX_PER_ENTRY_LEGACY ..], old_kept);
}

/// The pre-fix materializer (one `mlx_array_eval` per tensor), kept only as the byte-identity golden below.
fn materializeLegacyPerTensorForTest(tensors: []DiskTier.NamedTensor, s: mlx.mlx_stream) !void {
    for (tensors) |*t| {
        const cont = blk: {
            var made = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(made);
            try mlx.check(mlx.mlx_contiguous(&made, t.arr, false, s));
            break :blk made;
        };
        _ = mlx.mlx_array_free(t.arr);
        t.arr = cont;
        try mlx.check(mlx.mlx_array_eval(t.arr));
    }
}

test "DiskTier: the staged serializer evals ONCE per chunk, byte-identically to the per-tensor path" {
    // One eval per chunk file (the per-tensor eval cost ~2,300 GPU syncs on a 32k warm turn), and the bytes are unchanged.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-eval", 0, 128);
    defer tier.deinit();

    const qcfg = kv_quant.KVQuantConfig.affine(4);
    var cache = try KVCache.initWithConfig(testing.allocator, 3, qcfg);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 256, 64, 0.0, .bfloat16);

    var a = std.ArrayList(DiskTier.NamedTensor).empty;
    defer tier.freeNamed(&a);
    var b = std.ArrayList(DiskTier.NamedTensor).empty;
    defer tier.freeNamed(&b);
    for ([_]*std.ArrayList(DiskTier.NamedTensor){ &a, &b }) |list| {
        for (cache.entries, 0..) |*e, li| {
            try testing.expect(e.initialized);
            try tier.appendSlice(list, li, "k", e.keys.array, 0, 128, s);
            try tier.appendSlice(list, li, "v", e.values.array, 0, 128, s);
            try tier.appendSlice(list, li, "ks", e.keys_scales.array, 0, 128, s);
            try tier.appendSlice(list, li, "kb", e.keys_biases.array, 0, 128, s);
            try tier.appendSlice(list, li, "vs", e.values_scales.array, 0, 128, s);
            try tier.appendSlice(list, li, "vb", e.values_biases.array, 0, 128, s);
        }
    }
    try testing.expectEqual(@as(usize, 18), a.items.len);

    const before = safetensors.eval_count.load(.monotonic);
    const got = try tier.serializeNamed(a.items, DiskTier.no_meta, s);
    defer testing.allocator.free(got);
    try testing.expectEqual(@as(u64, 1), safetensors.eval_count.load(.monotonic) - before);

    try materializeLegacyPerTensorForTest(b.items, s);
    const mid = safetensors.eval_count.load(.monotonic);
    const want = try tier.encodeNamed(b.items, DiskTier.no_meta);
    defer testing.allocator.free(want);
    try testing.expectEqual(mid, safetensors.eval_count.load(.monotonic));
    try testing.expect(got.len > 4096);
    try testing.expectEqualSlices(u8, want, got);
}

test "DiskTier: a failing synchronous writeMeta frees the final path exactly once" {
    // `writeMeta` allocates `final_path` and forks; a function-scope `errdefer` plus the
    // synchronous branch's `defer` freed it twice on any disk write failure. The missing
    // `e<id>/` directory makes the manifest's write fail, the first fallible step past the defer.
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-writemeta-err", 0, 128);
    defer tier.deinit();
    try testing.expect(tier.writer == null);

    var toks = [_]u32{ 1, 2, 3 };
    var cb = [_]u64{64};
    var no_pos = [_]u32{};
    var no_sz = [_]u64{};
    const e = IndexEntry{
        .id = 777, // no e777/ directory was ever created
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
    // The write must fail and leave exactly one free behind.
    if (tier.writeMeta(e)) |_| {
        return error.WriteMetaUnexpectedlySucceeded;
    } else |_| {}
}

// ── SSD-first chunk sharing ──

/// Two 600-token sequences that agree for the first `shared` tokens and diverge after.

fn chunkStat(io: std.Io, base: []const u8, fp: []const u8, id: u64, chunk: u32) ?kv_disk_fs.Stat {
    var buf: [1024]u8 = undefined;
    const p = std.fmt.bufPrint(&buf, "{s}/{s}/e{d}/c{d:0>6}.safetensors", .{ base, fp, id, chunk }) catch return null;
    return kv_disk_fs.testStat(io, p);
}

test "commonPrefixLen: the longest shared prefix, never past the shorter slice" {
    const a = [_]u32{ 1, 2, 3, 4 };
    const b = [_]u32{ 1, 2, 9, 4, 5 };
    try testing.expectEqual(@as(usize, 2), commonPrefixLen(&a, &b));
    try testing.expectEqual(@as(usize, 4), commonPrefixLen(&a, &a));
    try testing.expectEqual(@as(usize, 0), commonPrefixLen(&a, &[_]u32{}));
    try testing.expectEqual(@as(usize, 3), commonPrefixLen(&a, a[0..3]));
}

test "DiskTier chunk share: a prefix-diverging entry hard-links the donor's whole chunks, writes only its tail, bills once, restores whole" {
    // Turn N+1's prompt diverges inside turn N's generated span: the heir links the whole
    // chunks below the common prefix and writes the rest.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-share", 0, 128);
    tier.ssd_first = true;
    tier.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);

    // 600 tokens => chunks 0..3 whole (512 tokens), chunk 4 partial (88).
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    const toks = chunkShareTokens(520); // diverges at 520: 4 whole chunks shared
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &toks.a, false, null, s);
    try testing.expectEqual(@as(usize, 1), tier.entryCount());
    const donor_bytes = tier.total_bytes;
    const donor_id = tier.entries.items[0].id;

    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &toks.b, false, null, s);
    try testing.expectEqual(@as(usize, 2), tier.entryCount());
    const heir = &tier.entries.items[1];
    try testing.expectEqual(@as(u32, 4), heir.inherited_chunks);
    try testing.expectEqual(@as(usize, 5), heir.chunk_bytes.len);
    try testing.expectEqual(@as(u32, 600), heir.kv_len);
    // The leading files are one inode with two links; the tail is its own.
    const d0 = chunkStat(io, base, "fp-share", donor_id, 0).?;
    const h0 = chunkStat(io, base, "fp-share", heir.id, 0).?;
    try testing.expectEqual(d0.inode, h0.inode);
    try testing.expectEqual(@as(u64, 2), @as(u64, @intCast(h0.nlink)));
    const h4 = chunkStat(io, base, "fp-share", heir.id, 4).?;
    try testing.expectEqual(@as(u64, 1), @as(u64, @intCast(h4.nlink)));
    try testing.expectEqual(donor_bytes + heir.chunk_bytes[4] + 600 * 4, tier.total_bytes);
    try testing.expectEqual(heir.chunk_bytes[4] + 600 * 4, heir.bytes);

    // The heir restores through a fresh tier, and its bill survives the scan.
    const total_bytes = tier.total_bytes;
    tier.deinit();
    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-share", 0, 128);
    defer tier2.deinit();
    try testing.expectEqual(@as(usize, 2), tier2.entryCount());
    try testing.expectEqual(total_bytes, tier2.total_bytes);
    const m = tier2.bestMatch(&toks.b, false, kv_quant.KVQuantConfig.dense, .{}).?;
    try testing.expectEqual(@as(u32, 600), m.usable);
    var out = try KVCache.init(testing.allocator, 3);
    defer out.deinit();
    try tier2.restorePrefixInto(&out, m.idx, 600, s);
    try testing.expectEqual(@as(usize, 600), out.step);
    inline for (.{ 0, 300, 599 }) |pos| {
        try testing.expectEqual(try cacheValueAt(&cache, 1, pos, 3, s), try cacheValueAt(&out, 1, pos, 3, s));
    }
}

test "DiskTier: rows restored from an entry's chunks serialize back to those chunk files byte for byte" {
    // Why a commit may link the chunks below its restored rows: they are the bytes it would write.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);
    var tier = try DiskTier.init(testing.allocator, io, base, "fp-reser", 0, 128);
    defer tier.deinit();
    const q8 = kv_quant.KVQuantConfig.affine(8);
    var cache = try KVCache.initWithConfig(testing.allocator, 2, q8);
    defer cache.deinit();
    try fillCache(&cache, s, 2, 600, 64, 0.5, .bfloat16);
    const toks = chunkShareTokens(600);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &toks.a, false, null, s);
    var restored = try KVCache.initWithConfig(testing.allocator, 2, q8);
    defer restored.deinit();
    try tier.restorePrefixInto(&restored, 0, 512, s);
    try fillCache(&restored, s, 2, 88, 64, 900.5, .bfloat16);
    var other: [600]u32 = toks.a;
    other[599] = 77;
    // The synchronous legacy arm links nothing: the second entry writes every chunk itself.
    _ = try tier.appendCommit(restored.entries, restored.step, restored.config, &other, false, null, s);
    try testing.expectEqual(@as(usize, 2), tier.entryCount());
    try testing.expectEqual(@as(u32, 0), tier.entries.items[1].inherited_chunks);
    for (0..4) |i| {
        var pa: [1024]u8 = undefined;
        var pb: [1024]u8 = undefined;
        const a = kv_disk_fs.testRead(io, try std.fmt.bufPrint(&pa, "{s}/e{d}/c{d:0>6}.safetensors", .{ tier.root, tier.entries.items[0].id, i }), 1 << 24).?;
        defer testing.allocator.free(a);
        const b = kv_disk_fs.testRead(io, try std.fmt.bufPrint(&pb, "{s}/e{d}/c{d:0>6}.safetensors", .{ tier.root, tier.entries.items[1].id, i }), 1 << 24).?;
        defer testing.allocator.free(b);
        try testing.expectEqualSlices(u8, a, b);
    }
}

test "DiskTier chunk share: an image turn links the chunks below its restored rows past the donor's text, and writes the rows it recomputed" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    var a: [520]u32 = undefined;
    for (&a, 0..) |*t, i| t.* = @intCast(i + 7);
    var b: [600]u32 = undefined;
    for (&b, 0..) |*t, i| t.* = @intCast(i + 7);
    // Restored at 520, the whole record: chunks 0..3. Restored at 300: the rows past it were recomputed.
    for ([_]struct { rows: ?u32, linked: u32 }{ .{ .rows = null, .linked = 1 }, .{ .rows = 520, .linked = 4 }, .{ .rows = 300, .linked = 2 } }) |c| {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const base = try tmpRoot(&tmp, io, &buf);
        var tier = try DiskTier.init(testing.allocator, io, base, "fp-share-img", 0, 128);
        defer tier.deinit();
        tier.ssd_first = true;
        tier.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
        _ = try tier.appendCommitWith(.{ .kv_entries = cache.entries, .step = 520, .config = cache.config, .tokens = &a, .has_tools = false, .vision_key = 0xA1, .media_start = 200 }, s, .{});
        const donor_id = tier.entries.items[0].id;
        // The next turn's chain adds a screenshot: another key, so the donor rule lends the text before the first image.
        _ = try tier.appendCommitWith(.{
            .kv_entries = cache.entries,
            .step = 600,
            .config = cache.config,
            .tokens = &b,
            .has_tools = false,
            .vision_key = 0xB2,
            .media_start = 200,
            .restored = if (c.rows) |r| .{ .id = donor_id, .rows = r } else null,
        }, s, .{});
        try testing.expectEqual(@as(usize, 2), tier.entryCount());
        const heir = &tier.entries.items[1];
        try testing.expectEqual(c.linked, heir.inherited_chunks);
        for (0..5) |i| {
            const st = chunkStat(io, base, "fp-share-img", heir.id, @intCast(i)).?;
            try testing.expectEqual(@as(u64, if (i < c.linked) 2 else 1), @as(u64, @intCast(st.nlink)));
        }
        try testing.expectEqual(try uniqueDirBytes(io, &tier), tier.total_bytes);
    }
}

test "DiskTier chunk share: total_bytes is bytes on disk whichever holder dies first" {
    // The filesystem is the refcount: donor-then-heir and heir-then-donor both land back on the pre-commit number.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    const toks = chunkShareTokens(520);

    for ([_]bool{ true, false }) |donor_first| {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const base = try tmpRoot(&tmp, io, &buf);
        var tier = try DiskTier.init(testing.allocator, io, base, "fp-order", 0, 128);
        defer tier.deinit();
        tier.ssd_first = true;
        tier.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
        try testing.expectEqual(@as(u64, 0), tier.total_bytes);

        _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &toks.a, false, null, s);
        _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &toks.b, false, null, s);
        try testing.expectEqual(@as(usize, 2), tier.entryCount());
        const donor_idx: usize = 0;
        const heir_idx: usize = 1;
        const donor_id = tier.entries.items[donor_idx].id;
        const heir_id = tier.entries.items[heir_idx].id;
        var shared_bytes: u64 = 0;
        for (tier.entries.items[heir_idx].chunk_bytes[0..4]) |b| shared_bytes += b;
        const donor_own = tier.entries.items[donor_idx].bytes - shared_bytes; // its tail + tokens
        const heir_own = tier.entries.items[heir_idx].bytes; // its tail + tokens (links billed 0)
        try testing.expectEqual(shared_bytes + donor_own + heir_own, tier.total_bytes);

        if (donor_first) {
            tier.removeAt(donor_idx);
            try testing.expectEqual(shared_bytes + heir_own, tier.total_bytes);
            try testing.expect(chunkStat(io, base, "fp-order", heir_id, 0) != null);
            try testing.expect(chunkStat(io, base, "fp-order", donor_id, 0) == null);
            try testing.expectEqual(@as(u64, 1), @as(u64, @intCast(chunkStat(io, base, "fp-order", heir_id, 0).?.nlink)));
            tier.removeAt(0);
        } else {
            tier.removeAt(heir_idx);
            try testing.expectEqual(shared_bytes + donor_own, tier.total_bytes);
            try testing.expect(chunkStat(io, base, "fp-order", donor_id, 0) != null);
            try testing.expectEqual(@as(u64, 1), @as(u64, @intCast(chunkStat(io, base, "fp-order", donor_id, 0).?.nlink)));
            tier.removeAt(0);
        }
        try testing.expectEqual(@as(usize, 0), tier.entryCount());
        try testing.expectEqual(@as(u64, 0), tier.total_bytes);
    }
}

test "DiskTier chunk share: a writer-armed tier beside a RAM tier links like SSD-first" {
    // With both tiers on, every turn of one conversation lands a new disk entry; the link spares
    // each a rewrite of the whole conversation.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);
    var tier = try DiskTier.init(testing.allocator, io, base, "fp-share-ram", 0, 128);
    defer tier.deinit();
    tier.enableBackgroundWriter();

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    const toks = chunkShareTokens(520);
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &toks.a, false, null, s);
    tier.drainWriter();
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &toks.b, false, null, s);
    tier.drainWriter();
    try testing.expectEqual(@as(usize, 2), tier.entryCount());
    const heir = &tier.entries.items[1];
    try testing.expectEqual(@as(u32, 4), heir.inherited_chunks);
    try testing.expectEqual(@as(u64, 2), @as(u64, @intCast(chunkStat(io, base, "fp-share-ram", heir.id, 0).?.nlink)));
}

test "DiskTier chunk share: the legacy arm and the kill switch never link" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    const toks = chunkShareTokens(520);
    // Arm 1: legacy tier, switch on. Arm 2: SSD-first, switch off.
    for ([_]struct { ssd: bool, share: bool }{ .{ .ssd = false, .share = true }, .{ .ssd = true, .share = false } }) |arm| {
        var tmp = std.testing.tmpDir(.{ .iterate = true });
        defer tmp.cleanup();
        var buf: [512]u8 = undefined;
        const base = try tmpRoot(&tmp, io, &buf);
        disk.chunk_share_override = arm.share;
        defer disk.chunk_share_override = null;
        var tier = try DiskTier.init(testing.allocator, io, base, "fp-nolink", 0, 128);
        defer tier.deinit();
        tier.ssd_first = arm.ssd;
        tier.armTestSpace(1024 * 1024 * 1024 * 1024, 4096 * 1024 * 1024 * 1024);
        _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &toks.a, false, null, s);
        const before = tier.total_bytes;
        _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &toks.b, false, null, s);
        const heir = &tier.entries.items[1];
        try testing.expectEqual(@as(u32, 0), heir.inherited_chunks);
        try testing.expectEqual(@as(u64, 1), @as(u64, @intCast(chunkStat(io, base, "fp-nolink", heir.id, 0).?.nlink)));
        var all: u64 = 600 * 4;
        for (heir.chunk_bytes) |b| all += b;
        try testing.expectEqual(before + all, tier.total_bytes);
    }
}

test "DiskTier chunk share: meta v6 carries inherited_chunks; a v5 manifest loads with none" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);
    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    const toks = chunkShareTokens(520);
    var heir_id: u64 = 0;
    {
        var tier = try DiskTier.init(testing.allocator, io, base, "fp-v6", 0, 128);
        defer tier.deinit();
        tier.ssd_first = true;
        tier.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
        _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &toks.a, false, null, s);
        _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &toks.b, false, null, s);
        heir_id = tier.entries.items[1].id;
    }
    var mp: [128]u8 = undefined;
    const meta_rel = try std.fmt.bufPrint(&mp, "fp-v6/e{d}/meta.json", .{heir_id});
    const meta = (kv_disk_fs.testReadAt(io, base, meta_rel, 1 << 20) orelse return error.TestUnexpectedResult);
    defer testing.allocator.free(meta);
    try testing.expect(std.mem.indexOf(u8, meta, "\"v\":6") != null);
    try testing.expect(std.mem.indexOf(u8, meta, "\"inherited_chunks\":4") != null);
    {
        var tier = try DiskTier.init(testing.allocator, io, base, "fp-v6", 0, 128);
        defer tier.deinit();
        var found = false;
        for (tier.entries.items) |*e| {
            if (e.id == heir_id) {
                found = true;
                try testing.expectEqual(@as(u32, 4), e.inherited_chunks);
            }
        }
        try testing.expect(found);
    }
    // An older binary's manifest (v5, no field) still loads: nothing inherited.
    var rewritten = std.ArrayList(u8).empty;
    defer rewritten.deinit(testing.allocator);
    try rewritten.appendSlice(testing.allocator, meta);
    _ = std.mem.replace(u8, rewritten.items, "\"v\":6", "\"v\":5", rewritten.items);
    const stripped = try std.mem.replaceOwned(u8, testing.allocator, rewritten.items, "\"inherited_chunks\":4,", "");
    defer testing.allocator.free(stripped);
    try kv_disk_fs.testWriteAt(io, base, meta_rel, stripped);
    {
        var tier = try DiskTier.init(testing.allocator, io, base, "fp-v6", 0, 128);
        defer tier.deinit();
        for (tier.entries.items) |*e| {
            if (e.id == heir_id) try testing.expectEqual(@as(u32, 0), e.inherited_chunks);
        }
    }
}

test "DiskTier chunk share: an heir links ONLY the donor's landed chunks; the donor's queued chunks and meta drain intact" {
    // A donor mid-persist is the common case: the heir links what is on disk, writes the rest,
    // and never touches the donor's queue.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-landed", 0, 128);
    tier.ssd_first = true;
    tier.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
    tier.enableBackgroundWriter();
    try testing.expect(tier.writer != null);

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 600, 8, 0.0, .float32);
    const toks = chunkShareTokens(520);

    // The bound goes on the donor's commits (the write-through hook's form), not on the tier;
    // the heir commits unbounded, the way `flushPendingDisk` completes an entry.
    const donor_bound: u64 = 1;

    // Donor: chunks 0 and 1 landed, chunk 2 queued (writer paused).
    _ = try tier.appendCommitBounded(cache.entries, cache.step, cache.config, &toks.a, false, null, s, donor_bound);
    tier.drainWriter();
    _ = try tier.appendCommitBounded(cache.entries, cache.step, cache.config, &toks.a, false, null, s, donor_bound);
    tier.drainWriter();
    try testing.expectEqual(@as(u32, 256), tier.entries.items[0].kv_len);
    tier.writer.?.setPaused(true);
    defer if (tier.writer) |w| w.setPaused(false);
    _ = try tier.appendCommitBounded(cache.entries, cache.step, cache.config, &toks.a, false, null, s, donor_bound);
    try testing.expectEqual(@as(u32, 384), tier.entries.items[0].kv_len);
    const donor_id = tier.entries.items[0].id;
    try testing.expect(chunkStat(io, base, "fp-landed", donor_id, 2) == null); // still queued
    const dropped_before = tier.writer.?.files_dropped;

    // Heir: the overlap allows 4 whole chunks; only 2 have landed, so it links 2 and writes 2..4.
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &toks.b, false, null, s);
    try testing.expectEqual(@as(usize, 2), tier.entryCount());
    const heir = &tier.entries.items[1];
    try testing.expectEqual(@as(u32, 2), heir.inherited_chunks);
    try testing.expectEqual(@as(usize, 5), heir.chunk_bytes.len);
    try testing.expectEqual(@as(u32, 600), heir.kv_len);

    // The donor's queue was never touched.
    try testing.expectEqual(dropped_before, tier.writer.?.files_dropped);
    tier.writer.?.setPaused(false);
    tier.drainWriter();
    try testing.expectEqual(@as(u64, 0), tier.writeErrors());
    const d2 = chunkStat(io, base, "fp-landed", donor_id, 2).?;
    try testing.expectEqual(tier.entries.items[0].chunk_bytes[2], d2.size);
    try testing.expectEqual(@as(u64, 1), @as(u64, @intCast(d2.nlink))); // the heir wrote its own chunk 2
    inline for (.{ 0, 1 }) |i| try testing.expectEqual(@as(u64, 2), @as(u64, @intCast(chunkStat(io, base, "fp-landed", heir.id, i).?.nlink)));
    inline for (.{ 2, 3, 4 }) |i| try testing.expectEqual(@as(u64, 1), @as(u64, @intCast(chunkStat(io, base, "fp-landed", heir.id, i).?.nlink)));

    const heir_id = heir.id;
    tier.deinit();
    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-landed", 0, 128);
    defer tier2.deinit();
    try testing.expectEqual(@as(usize, 2), tier2.entryCount());
    for (tier2.entries.items) |*e| {
        if (e.id == donor_id) try testing.expectEqual(@as(u32, 384), e.kv_len);
        if (e.id == heir_id) {
            try testing.expectEqual(@as(u32, 600), e.kv_len);
            try testing.expectEqual(@as(u32, 2), e.inherited_chunks);
        }
    }
}

test "DiskTier: deinit RETURNS with the writer paused, and lands what was queued" {
    // `DiskTier.deinit` drains before it deinits the writer; teardown must lift a pause itself.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = try tmpRoot(&tmp, io, &buf);

    var cache = try KVCache.init(testing.allocator, 3);
    defer cache.deinit();
    try fillCache(&cache, s, 3, 640, 8, 0.0, .float32); // > MIN_PERSIST_TOKENS
    var tokens: [640]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    var tier = try DiskTier.init(testing.allocator, io, base, "fp-paused-deinit", 0, 128);
    tier.ssd_first = true;
    tier.armTestSpace(1024 * 1024 * 1024 * 1024, 2048 * 1024 * 1024 * 1024);
    tier.enableBackgroundWriter();
    try testing.expect(tier.writer != null);
    tier.writer.?.setPaused(true);
    // No deferred unpause on purpose: teardown is the release under test.
    _ = try tier.appendCommit(cache.entries, cache.step, cache.config, &tokens, false, null, s);
    try testing.expect(tier.writer.?.pendingBytes() > 0);

    // Not deferred: the bar is that the call returns.
    tier.deinit();

    // It lifted the pause rather than skipping the drain: the entry is really on disk.
    var tier2 = try DiskTier.init(testing.allocator, io, base, "fp-paused-deinit", 0, 128);
    defer tier2.deinit();
    try testing.expectEqual(@as(usize, 1), tier2.entryCount());
    try testing.expectEqual(@as(u32, 640), tier2.entries.items[0].kv_len);
}

