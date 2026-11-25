//! DeepSeek-V4.1's Engram: its tables' names in the release checkpoint and its compressed-vocabulary token map. The
//! hash, the table reader and the read-ahead are the shared `engram.zig`, which this file re-exports for the model.
const std = @import("std");
const engram = @import("../engram.zig");
const cfg_mod = @import("dsv41_config.zig");

pub const Hash = engram.Hash;
pub const Table = engram.Table;
pub const RowCache = engram.RowCache;
pub const Prefetch = engram.Prefetch;
pub const cache_rows = engram.cache_rows;
pub const cacheBytes = engram.cacheBytes;
const MAX_COLS = engram.MAX_COLS;

/// Layer `layer`'s table in the release checkpoint at `dir_path`.
pub fn openTable(gpa: std.mem.Allocator, dir_path: []const u8, layer: usize, rows: u64, head_dim: u32) !Table {
    var wbuf: [96]u8 = undefined;
    var sbuf: [96]u8 = undefined;
    const names: Table.Names = .{
        .weight = try std.fmt.bufPrint(&wbuf, "layers.{d}.engram.embed.weight", .{layer}),
        .scale = try std.fmt.bufPrint(&sbuf, "layers.{d}.engram.embed.scale", .{layer}),
    };
    return Table.open(gpa, dir_path, names, rows, head_dim);
}

/// V4.1's token map (`tests/dsv41/dump_dsv41_token_map.py`): one bit per token id, set where the token opens
/// the next compressed id, then every other token's id as a u32.
const v41_token_map = @embedFile("fixtures/dsv41_engram_token_map.bin");

fn unpackTokenMap(map: []u32, raw: []const u8) !void {
    const bits = (map.len + 7) / 8;
    if (raw.len < bits) return error.EngramTokenMap;
    var next: u32 = 0;
    var at = bits;
    for (map, 0..) |*v, t| {
        if ((raw[t / 8] >> @intCast(t % 8)) & 1 != 0) {
            v.* = next;
            next += 1;
        } else {
            if (at + 4 > raw.len) return error.EngramTokenMap;
            v.* = std.mem.readInt(u32, raw[at..][0..4], .little);
            at += 4;
        }
    }
    if (at != raw.len) return error.EngramTokenMap;
}

/// Token id to compressed id: a pack's own `engram_token_map.json` (a list) when it ships one, else the
/// map DeepSeek's `engram.py` derives from V4.1's tokenizer. Every hash multiplier derives from the
/// compressed vocabulary's size, so a map built for another tokenizer is refused.
pub fn loadTokenMap(gpa: std.mem.Allocator, dir_path: []const u8, vocab: u32, compressed: u32) ![]u32 {
    const io = std.Io.Threaded.global_single_threaded.io();
    var dir = try std.Io.Dir.cwd().openDir(io, dir_path, .{});
    defer dir.close(io);
    const map = try gpa.alloc(u32, vocab);
    errdefer gpa.free(map);
    var own = false;
    if (dir.readFileAlloc(io, "engram_token_map.json", gpa, .limited(64 << 20))) |raw| {
        defer gpa.free(raw);
        const parsed = std.json.parseFromSlice([]const u32, gpa, raw, .{}) catch return error.EngramTokenMap;
        defer parsed.deinit();
        if (parsed.value.len != vocab) return error.EngramTokenMap;
        @memcpy(map, parsed.value);
        own = true;
    } else |_| {
        unpackTokenMap(map, v41_token_map) catch return error.EngramTokenMapMissing;
    }
    var hi: u32 = 0;
    for (map) |v| hi = @max(hi, v);
    if (hi + 1 != compressed) return if (own) error.EngramTokenMap else error.EngramTokenMapMissing;
    return map;
}

const testing = std.testing;

fn releaseEngram() cfg_mod.Engram {
    return .{ .n = 2, .layers = .{ 1, 14, 0, 0 }, .hash_ids = .{ 1, 14, 0, 0 }, .rows = .{ 384006168, 384016682, 0, 0 }, .max_ngram = 4, .vocab_size = 16000000, .heads = 8, .head_dim = 256, .pad_id = 2, .compressed_vocab = 99092 };
}

test "dsv41 engram: hash multipliers are numpy default_rng(10007 * layer) draws" {
    // The integers DeepSeek's engram.py derives for the release.
    const h = try Hash.init(releaseEngram(), 2);
    const want = [2][4]u64{
        .{ 76632096046245, 4839876093313, 35959672319349, 73987337458391 },
        .{ 67716810739261, 51510806800915, 30921347202721, 82619226485591 },
    };
    for (want, 0..) |row, l| try testing.expectEqualSlices(u64, &row, h.mult[l][0..4]);
}

test "dsv41 engram: prime buckets continue across layers and fill the declared tables" {
    const h = try Hash.init(releaseEngram(), 2);
    try testing.expectEqualSlices(u64, &.{ 16000057, 16000079, 16000081, 16000097, 16000121, 16000129, 16000133, 16000183 }, h.primes[0][0..8]);
    try testing.expectEqualSlices(u64, &.{ 16000781, 16000799, 16000813, 16000819, 16000841, 16000877, 16000879, 16000889 }, h.primes[1][16..24]);
    try testing.expectEqualSlices(u64, &.{ 0, 16000477, 32000964, 48001463, 64001970 }, h.offsets[1][0..5]);
    var bad = releaseEngram();
    bad.rows[1] += 1;
    try testing.expectError(error.EngramTableRows, Hash.init(bad, 2));
}

test "dsv41 engram: row ids match DeepSeek's NgramHashState across prefill and decode" {
    // inference/engram.py over compressed ids [3, 99091, 2, 77, 77, 1024], then one decode step [5]
    // at position 6 (identity token map, pad = 2).
    const h = try Hash.init(releaseEngram(), 2);
    const prompt = [_]u32{ 3, 99091, 2, 77, 77, 1024 };
    var pre: [6 * 2 * 24]u32 = undefined;
    h.rows(&.{}, &prompt, &pre);
    try testing.expectEqualSlices(u32, &.{ 7010988, 29651738, 47710115, 48176674, 72877570, 81111450, 101228567, 120693201, 143450283, 146576661, 168605081, 191902092, 196072724, 214158605, 233287660, 246388310, 261549600, 286523983, 293659617, 313227845, 320939872, 340508126, 357771274, 372619255 }, pre[0..24]);
    try testing.expectEqualSlices(u32, &.{ 11344931, 28156158, 45201617, 61942444, 78326219, 85383927, 106445668, 125242217, 129707946, 158452422, 167536917, 177080926, 202551179, 221735906, 238090062, 254314270, 259798390, 277434364, 301936998, 307898224, 335252679, 339837629, 352557585, 368186488 }, pre[(5 * 2 + 1) * 24 ..][0..24]);
    var dec: [2 * 24]u32 = undefined;
    h.rows(&prompt, &.{5}, &dec);
    try testing.expectEqualSlices(u32, &.{ 14359625, 17064925, 34766379, 48382691, 68824895, 91643820, 111054239, 121732398, 132594636, 146910113, 174982973, 181797503, 206306954, 210573457, 224988242, 243323351, 267856496, 279543941, 298892988, 316569280, 333687525, 351365567, 362014820, 382511691 }, dec[0..24]);
    // A chunk boundary anywhere hashes the same rows.
    var split: [6 * 2 * 24]u32 = undefined;
    h.rows(&.{}, prompt[0..2], split[0 .. 2 * 48]);
    h.rows(prompt[0..2], prompt[2..], split[2 * 48 ..]);
    try testing.expectEqualSlices(u32, &pre, &split);
}

test "dsv41 engram: a dead token makes every n-gram across it the all-pad one" {
    const h = try Hash.init(releaseEngram(), 2);
    const ids = [_]u32{ 3, Hash.DEAD, 77 };
    var got: [3 * 2 * 24]u32 = undefined;
    h.rows(&.{}, &ids, &got);
    var want: [3 * 2 * 24]u32 = undefined;
    h.rows(&.{}, &.{2}, want[48..96]); // the dead token's own row: every slot pad
    try testing.expectEqualSlices(u32, want[48..96], got[48..96]);
    h.rows(&.{ 2, 2 }, &.{77}, want[96..]); // 77 with only pad behind it
    try testing.expectEqualSlices(u32, want[96..], got[96..]);
    h.rows(&.{}, &.{3}, want[0..48]); // no look-ahead
    try testing.expectEqualSlices(u32, want[0..48], got[0..48]);
}

test "dsv41 engram: the embedded V4.1 token map collapses case variants and ends where the config says" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var td = testing.tmpDir(.{});
    defer td.cleanup();
    var buf: [512]u8 = undefined;
    const dir = buf[0..try td.dir.realPath(io, &buf)];
    const map = try loadTokenMap(testing.allocator, dir, 129280, 99092);
    defer testing.allocator.free(map);
    // the, The, " the", " The", THE, " THE" normalize alike (engram.py's ids for the release tokenizer).
    for ([_]u32{ 1805, 671, 270, 455, 20852, 6367 }) |id| try testing.expectEqual(@as(u32, 237), map[id]);
    try testing.expectEqual(@as(u32, 2), map[2]);
    try testing.expectEqual(@as(u32, 99091), map[129279]);
    // Another tokenizer's sizes: the pack must ship its own map.
    try testing.expectError(error.EngramTokenMapMissing, loadTokenMap(testing.allocator, dir, 129280, 99000));
    try testing.expectError(error.EngramTokenMapMissing, loadTokenMap(testing.allocator, dir, 151936, 99092));
}

test "dsv41 engram: a pack's own token map is read as a list and held to the config's size" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var td = testing.tmpDir(.{});
    defer td.cleanup();
    try td.dir.writeFile(io, .{ .sub_path = "engram_token_map.json", .data = "[0, 1, 1, 2]" });
    var buf: [512]u8 = undefined;
    const dir = buf[0..try td.dir.realPath(io, &buf)];
    const map = try loadTokenMap(testing.allocator, dir, 4, 3);
    defer testing.allocator.free(map);
    try testing.expectEqualSlices(u32, &.{ 0, 1, 1, 2 }, map);
    try testing.expectError(error.EngramTokenMap, loadTokenMap(testing.allocator, dir, 4, 4));
    try testing.expectError(error.EngramTokenMap, loadTokenMap(testing.allocator, dir, 5, 3));
}

// Row reuse of the release's Engram hash over a token stream: `SUSHI_DSV41_ENGRAM_REUSE=<u32 tokens file>`, read as
// documents of 2048 tokens in file order (a file of u32 token ids). Per table, `RowCache`'s hit rate at
// several capacities. Prints only under the switch.
test "dsv41 engram: row reuse by cache capacity over a token file (SUSHI_DSV41_ENGRAM_REUSE=<tokens>)" {
    const path = std.c.getenv("SUSHI_DSV41_ENGRAM_REUSE") orelse return error.SkipZigTest;
    const a = std.heap.page_allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, std.mem.sliceTo(path, 0), a, .limited(1 << 30));
    const toks = std.mem.bytesAsSlice(u32, raw[0 .. raw.len / 4 * 4]);
    const e = releaseEngram();
    const map = try a.alloc(u32, 129280);
    try unpackTokenMap(map, v41_token_map);
    const h = try Hash.init(e, map[e.pad_id]);
    const nc = h.cols();
    const doc: usize = 2048;
    const caps = [_]u64{ 1 << 16, 1 << 18, 1 << 20, 1 << 22, 1 << 24 };
    var caches: [caps.len][2]RowCache = undefined;
    for (&caches, caps) |*pair, cap| for (pair) |*c| {
        c.* = try RowCache.init(a, cap, 0, 0);
    };
    const comp = try a.alloc(u32, doc);
    const rows = try a.alloc(u32, doc * h.n_layers * nc);
    var missed: [MAX_COLS]u32 = undefined;
    // A document as one batch a table, as a prefill chunk reads it: rows missed by the whole batch against the cache
    // before it, and the distinct ones among them (what `Table.read` preads).
    var batch: [2]RowCache = .{ try RowCache.init(a, cache_rows, 0, 0), try RowCache.init(a, cache_rows, 0, 0) };
    const gathered = try a.alloc(u32, doc * nc);
    const batch_missed = try a.alloc(u32, doc * nc);
    const miss_ids = try a.alloc(u32, doc * nc);
    var batch_reads: [2][2]u64 = @splat(@splat(0));
    var d: usize = 0;
    while ((d + 1) * doc <= toks.len) : (d += 1) {
        for (comp, toks[d * doc ..][0..doc]) |*o, t| o.* = if (t < map.len) map[t] else map[e.pad_id];
        h.rows(&.{}, comp, rows);
        for (0..h.n_layers) |l| {
            for (0..doc) |t| @memcpy(gathered[t * nc ..][0..nc], rows[(t * h.n_layers + l) * nc ..][0..nc]);
            const k = batch[l].lookup(gathered, &.{}, &.{}, batch_missed);
            for (batch_missed[0..k], miss_ids[0..k]) |i, *o| o.* = gathered[i];
            std.mem.sort(u32, miss_ids[0..k], {}, std.sort.asc(u32));
            var distinct: u64 = 0;
            for (miss_ids[0..k], 0..) |id, i| distinct += @intFromBool(i == 0 or miss_ids[i - 1] != id);
            batch_reads[l][0] += k;
            batch_reads[l][1] += distinct;
            batch[l].insert(gathered, batch_missed[0..k], &.{}, &.{});
        }
        for (0..doc) |t| for (0..h.n_layers) |l| {
            const ids = rows[(t * h.n_layers + l) * nc ..][0..nc];
            for (&caches) |*pair| {
                const k = pair[l].lookup(ids, &.{}, &.{}, &missed);
                pair[l].insert(ids, missed[0..k], &.{}, &.{});
            }
        };
    }
    for (batch_reads, 0..) |r, l| std.debug.print("[engram-reuse] layer {d}, {d} rows a batch, 1M-row cache: {d} rows missed, {d} distinct ({d:.1}% fewer preads)\n", .{ if (l == 0) @as(u32, 1) else 14, doc * nc, r[0], r[1], 100 * (1 - @as(f64, @floatFromInt(r[1])) / @as(f64, @floatFromInt(@max(r[0], 1)))) });
    for (caches, caps) |pair, cap| {
        const rate = [2]f64{ @as(f64, @floatFromInt(pair[0].hits)) / @as(f64, @floatFromInt(pair[0].hits + pair[0].misses)), @as(f64, @floatFromInt(pair[1].hits)) / @as(f64, @floatFromInt(pair[1].hits + pair[1].misses)) };
        std.debug.print("[engram-reuse] {d} docs, {d} rows a table ({d:.0} MB a table at 272 B a row): hit layer 1 {d:.3}, layer 14 {d:.3}\n", .{ d, cap, @as(f64, @floatFromInt(RowCache.bytesFor(cap, 256, 8))) / 1e6, rate[0], rate[1] });
    }
}
