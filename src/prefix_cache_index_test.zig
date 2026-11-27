//! Hot prefix cache lookup, restore, ring checkpoint, and SSD restart tests.

const std = @import("std");
const testing = std.testing;
const mlx = @import("mlx.zig");
const mlx_pages = @import("mlx_pages.zig");
const trunk_mod = @import("trunk.zig");
const trunk_helpers_test = @import("trunk_helpers_test.zig");
const model_mod = @import("model.zig");
const kv_quant = @import("kv_quant.zig");
const kv_disk_cache = @import("kv_disk_cache.zig");
const kv_disk_fs = @import("kv_disk_fs.zig");
const cache_stats = @import("cache_stats.zig");
const pc = @import("prefix_cache.zig");
const util = @import("prefix_cache_helpers_test.zig");

const KVCache = trunk_mod.KVCache;
const KVCacheSnapshot = trunk_mod.KVCacheSnapshot;
const HotPrefixCache = pc.HotPrefixCache;
const CommitStatus = pc.CommitStatus;
const SlotRingCps = pc.SlotRingCps;
const MediaSpan = pc.MediaSpan;
const chainKey = pc.chainKey;
const ringMarkPositions = pc.ringMarkPositions;
const restoreLogLine = pc.restoreLogLine;
const RING_RESTORE_MIN_TOKENS = pc.RING_RESTORE_MIN_TOKENS;
const RING_CHECKPOINT_MAX = pc.RING_CHECKPOINT_MAX;
const RING_MARKS_MAX = pc.RING_MARKS_MAX;
const SLOT_RING_CHECKPOINTS = pc.SLOT_RING_CHECKPOINTS;

const testFillCache = util.testFillCache;
const ringWriteLayer = util.ringWriteLayer;
const ringFill = util.ringFill;
const ringRowBytes = util.ringRowBytes;
const ringTurn = util.ringTurn;
const expectRingContinuesCold = util.expectRingContinuesCold;

test "HotPrefixCache: zero capacity disables RAM retention" {
    var cache = HotPrefixCache.init(testing.allocator, 0);
    defer cache.deinit();
    try testing.expectEqual(@as(u32, 0), cache.max_entries);
    try testing.expect(!cache.ram_enabled);
    try testing.expectEqual(@as(usize, 0), cache.entryCount());
}

test "HotPrefixCache: findBestMatch returns longest shared prefix" {
    var cache = HotPrefixCache.init(testing.allocator, 4);
    defer cache.deinit();

    // Two synthetic entries (snapshots are no-ops on freshly-zero KVCache; we
    // never restore in this unit test, so no GPU work).
    const ids_a = try testing.allocator.dupe(u32, &[_]u32{ 1, 2, 3, 4, 5 });
    const ids_b = try testing.allocator.dupe(u32, &[_]u32{ 1, 2, 3, 9, 9, 9 });
    try cache.entries.append(testing.allocator, .{
        .tokens = ids_a,
        .has_tools = false,
        .snapshot = .{ .entries = try testing.allocator.alloc(trunk_mod.KVCacheEntry, 0), .step = 0, .allocator = testing.allocator, .config = trunk_mod.KVQuantConfig.dense },
        .last_used = 1,
        .quant_config = kv_quant.KVQuantConfig.dense,
        .kv_bytes = 0,
        .ssm_checkpoints = null,
        .ssm_bytes = 0,
    });
    try cache.entries.append(testing.allocator, .{
        .tokens = ids_b,
        .has_tools = false,
        .snapshot = .{ .entries = try testing.allocator.alloc(trunk_mod.KVCacheEntry, 0), .step = 0, .allocator = testing.allocator, .config = trunk_mod.KVQuantConfig.dense },
        .last_used = 2,
        .quant_config = kv_quant.KVQuantConfig.dense,
        .kv_bytes = 0,
        .ssm_checkpoints = null,
        .ssm_bytes = 0,
    });

    // Looking up [1,2,3,4,5,6] should match entry A (5 shared tokens).
    const lookup_ids = [_]u32{ 1, 2, 3, 4, 5, 6 };
    const m = cache.findBestMatch(&lookup_ids, false, 0, kv_quant.KVQuantConfig.dense).?;
    try testing.expectEqual(@as(usize, 0), m.idx);
    try testing.expectEqual(@as(usize, 5), m.shared);

    // Looking up [1,2,3,9,9,9,7] should match entry B (6 shared).
    const lookup_ids2 = [_]u32{ 1, 2, 3, 9, 9, 9, 7 };
    const m2 = cache.findBestMatch(&lookup_ids2, false, 0, kv_quant.KVQuantConfig.dense).?;
    try testing.expectEqual(@as(usize, 1), m2.idx);
    try testing.expectEqual(@as(usize, 6), m2.shared);

    // has_tools mismatch returns null.
    try testing.expectEqual(@as(?@TypeOf(m), null), cache.findBestMatch(&lookup_ids, true, 0, kv_quant.KVQuantConfig.dense));
    // vision_key mismatch returns null both ways: a text entry never serves an
    // image request and an image entry only serves the same pixels.
    try testing.expectEqual(@as(?@TypeOf(m), null), cache.findBestMatch(&lookup_ids, false, 7, kv_quant.KVQuantConfig.dense));
    cache.entries.items[0].vision_key = 7;
    // Text lookup falls through to entry B (3 shared); the keyed lookup gets A.
    try testing.expectEqual(@as(usize, 1), cache.findBestMatch(&lookup_ids, false, 0, kv_quant.KVQuantConfig.dense).?.idx);
    try testing.expectEqual(@as(usize, 0), cache.findBestMatch(&lookup_ids, false, 7, kv_quant.KVQuantConfig.dense).?.idx);
    cache.entries.items[0].vision_key = 0;
    // Scheme mismatch returns null — entries are dense, a query for affine
    // 4-bit cannot match (Wave 1.A: cross-scheme cache hits never happen).
    try testing.expectEqual(@as(?@TypeOf(m), null), cache.findBestMatch(&lookup_ids, false, 0, kv_quant.KVQuantConfig.affine(4)));
}

test "HotPrefixCache: an entry holding the request's first images restores up to the new one" {
    // An agent turn appends a screenshot to a conversation that already holds
    // one: the previous turn's entry carries the first image's rows, keyed by
    // that image alone, and serves everything before the new image.
    var cache = HotPrefixCache.init(testing.allocator, 4);
    defer cache.deinit();
    const P = 99;
    const chain = [_]MediaSpan{ .{ .start = 3, .key = 0xA1 }, .{ .start = 8, .key = 0xB3 } };
    const request = [_]u32{ 1, 2, 3, P, P, 5, 6, 7, P, P, 9 };
    for ([_]u64{ 0xA1, 0xF0 }, 0..) |key, i| {
        try cache.entries.append(testing.allocator, .{
            .tokens = try testing.allocator.dupe(u32, &[_]u32{ 1, 2, 3, P, P, 5, 6, 7, 4 }),
            .has_tools = false,
            .vision_key = key,
            .media_start = 3,
            .snapshot = .{ .entries = try testing.allocator.alloc(trunk_mod.KVCacheEntry, 0), .step = 0, .allocator = testing.allocator, .config = trunk_mod.KVQuantConfig.dense },
            .last_used = i,
            .quant_config = kv_quant.KVQuantConfig.dense,
            .kv_bytes = 0,
            .ssm_checkpoints = null,
            .ssm_bytes = 0,
        });
    }
    const dense = kv_quant.KVQuantConfig.dense;
    const m = cache.findBestRestorableMatch(&request, false, chainKey(&chain), 3, &chain, dense, false, null).?;
    try testing.expectEqual(@as(usize, 0), m.idx);
    try testing.expectEqual(@as(usize, 8), m.shared);

    // Foreign pixels under the first placeholder share only the text before it.
    cache.entries.items[0].vision_key = 0xA2;
    try testing.expectEqual(@as(usize, 3), cache.findBestRestorableMatch(&request, false, chainKey(&chain), 3, &chain, dense, false, null).?.shared);
    // Without the chain the old rule holds: the first media row bounds it.
    cache.entries.items[0].vision_key = 0xA1;
    try testing.expectEqual(@as(usize, 3), cache.findBestRestorableMatch(&request, false, chainKey(&chain), 3, &.{}, dense, false, null).?.shared);
}

test "HotPrefixCache: restore clamps an inflated snapshot to the matched length (sliding mask crash)" {
    // Root cause of a broadcast_shapes crash (mask 16890 vs KV 16892 at ~16K ctx): a snapshot committed with a KV buffer
    // LONGER than its logical token count — PLD/speculative decode leaves stale
    // draft positions in the buffer past the committed step. When the NEXT prompt
    // matches the entry's ENTIRE token sequence but is longer (a partial hit,
    // effective_matched == e.tokens.len < prompt_ids.len), the old truncate guard
    // `final_len < e.tokens.len` was FALSE, so the restored cache offset kept the
    // inflated snapshot length — drifting ahead of the matched length generation
    // tracks. That drift corrupts RoPE and crashes the sliding-window prefill mask.
    const s = mlx.gpuStream();

    var toks: [67]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 11);
    const logical_len: usize = 64;

    // Source cache: 64 logical tokens, then 2 STALE tokens (offset 66) — the
    // shape a PLD round leaves behind before commit.
    var src = try KVCache.init(testing.allocator, 2);
    defer src.deinit();
    try testFillCache(&src, s, 2, @intCast(logical_len));
    try testFillCache(&src, s, 2, 2); // stale draft tail → offset 66
    try testing.expectEqual(@as(usize, 66), src.step);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    // Commit with the LOGICAL token count (64) — but the snapshot carries 66.
    _ = try hc.commit(&src, toks[0..logical_len], false);

    // Reuse with a prompt that matches all 64 entry tokens but is LONGER (67):
    // effective_matched == e.tokens.len == 64 < prompt_ids.len — the crash path.
    var dst = try KVCache.init(testing.allocator, 2);
    defer dst.deinit();
    var moe_off: usize = 0;
    const res = try hc.lookupAndRestore(&dst, &moe_off, null, s, &toks, false, 0, null, null);

    try testing.expect(!res.full_match);
    try testing.expectEqual(@as(usize, 64), res.matched);
    // The invariant: restored cache offset == matched length, NOT the inflated 66.
    try testing.expectEqual(@as(usize, 64), moe_off);
    try testing.expectEqual(@as(usize, 64), dst.step);
    for (dst.entries) |*e| {
        try testing.expect(e.initialized);
        try testing.expectEqual(@as(usize, 64), e.offset); // clamped, not 66
    }
}

test "prefix cache: DFlash assistant context round-trips, clamped to the trunk's matched length" {
    const s = mlx.gpuStream();
    var toks: [64]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 5);

    // Trunk KV for 64 tokens; the assistant context covers the same span but
    // starts at 10 (the committing request was itself a partial cache hit).
    var trunk = try KVCache.init(testing.allocator, 2);
    defer trunk.deinit();
    try testFillCache(&trunk, s, 2, 64);
    var assist = try KVCache.init(testing.allocator, 2);
    defer assist.deinit();
    try testFillCache(&assist, s, 2, 54);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    _ = try hc.commitWithSsm(&trunk, &toks, false, null, .{ .cache = &assist, .base_pos = 10 }, null);
    // The assistant context is billed like SSM state, so the memory cap sees it.
    try testing.expect(hc.entries.items[0].dflash_bytes > 0);
    try testing.expect(hc.entries.items[0].kv_bytes > hc.entries.items[0].dflash_bytes);

    // A shorter prompt that the entry fully covers: the full-match path
    // re-forwards the last token, so the trunk lands at 31 and the assistant
    // context must be clamped to 31-10 = 21 — absLen == matched, or the first
    // round's `dctx.absLen() == anchor_pos` assert fires.
    var dst = try KVCache.init(testing.allocator, 2);
    defer dst.deinit();
    var dfl = try KVCache.init(testing.allocator, 2);
    defer dfl.deinit();
    var moe_off: usize = 0;
    var base: usize = 0;
    const res = try hc.lookupAndRestore(
        &dst,
        &moe_off,
        null,
        s,
        toks[0..32],
        false,
        0,
        .{ .cache = &dfl, .base_pos = &base },
        null,
    );
    try testing.expect(res.full_match);
    try testing.expectEqual(@as(usize, 31), res.matched);
    try testing.expectEqual(@as(?usize, 10), res.dflash_base);
    try testing.expectEqual(@as(usize, 10), base);
    try testing.expectEqual(@as(usize, 21), dfl.step);
    try testing.expectEqual(base + dfl.step, res.matched);

    // An entry with no assistant payload leaves the target untouched, and the
    // caller is told so — a blind start is a valid outcome, never a wrong one.
    var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc2.deinit();
    _ = try hc2.commit(&trunk, &toks, false);
    var dst2 = try KVCache.init(testing.allocator, 2);
    defer dst2.deinit();
    var dfl2 = try KVCache.init(testing.allocator, 2);
    defer dfl2.deinit();
    var moe_off2: usize = 0;
    var base2: usize = 7;
    const res2 = try hc2.lookupAndRestore(
        &dst2,
        &moe_off2,
        null,
        s,
        toks[0..32],
        false,
        0,
        .{ .cache = &dfl2, .base_pos = &base2 },
        null,
    );
    try testing.expectEqual(@as(usize, 31), res2.matched);
    try testing.expectEqual(@as(?usize, null), res2.dflash_base);
    try testing.expectEqual(@as(usize, 0), dfl2.step);
    try testing.expectEqual(@as(usize, 7), base2); // untouched
}

test "prefix cache: MTP committed history round-trips, clamped; a history ending short is declined" {
    // Same DflashSnap machinery, second Entry field: the head's history is
    // built from trunk hiddens and a restore forwards nothing, so without
    // this every reused prefix drafts blind (~70 -> ~38 tok/s on warm echo).
    // Unlike the trunk KV, a history that ends BEFORE the
    // matched cursor cannot be adopted — the missing tail's hiddens are
    // unrecoverable, and a gap right below the generation point is worse
    // than a blind start.
    const s = mlx.gpuStream();
    var toks: [64]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 5);

    var trunk = try KVCache.init(testing.allocator, 2);
    defer trunk.deinit();
    try testFillCache(&trunk, s, 2, 64);
    // Committed history covers 60 of the 64 tokens (the deferred-stash lag).
    var hist = try KVCache.init(testing.allocator, 1);
    defer hist.deinit();
    try testFillCache(&hist, s, 1, 60);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    _ = try hc.commitWithSsm(&trunk, &toks, false, null, null, .{ .cache = &hist, .base_pos = 0 });
    try testing.expect(hc.entries.items[0].mtp_bytes > 0);
    try testing.expect(hc.entries.items[0].kv_bytes > hc.entries.items[0].mtp_bytes);

    // Shorter prompt fully covered by the entry: full-match arm lands the
    // trunk at 31 and the history clamps to 31 (base 0).
    var dst = try KVCache.init(testing.allocator, 2);
    defer dst.deinit();
    var mtp_dst = try KVCache.init(testing.allocator, 1);
    defer mtp_dst.deinit();
    var moe_off: usize = 0;
    var base: usize = 99;
    const res = try hc.lookupAndRestore(&dst, &moe_off, null, s, toks[0..32], false, 0, null, .{ .cache = &mtp_dst, .base_pos = &base });
    try testing.expect(res.full_match);
    try testing.expectEqual(@as(usize, 31), res.matched);
    try testing.expectEqual(@as(?usize, 0), res.mtp_base);
    try testing.expectEqual(@as(usize, 0), base);
    try testing.expectEqual(@as(usize, 31), mtp_dst.step);
    try testing.expectEqual(base + mtp_dst.step, res.matched);

    // Full 64-token re-issue: matched 63 > the 60 the history covers →
    // declined, target untouched, caller starts blind.
    var dst2 = try KVCache.init(testing.allocator, 2);
    defer dst2.deinit();
    var mtp2 = try KVCache.init(testing.allocator, 1);
    defer mtp2.deinit();
    var moe2: usize = 0;
    var base2: usize = 7;
    const res2 = try hc.lookupAndRestore(&dst2, &moe2, null, s, &toks, false, 0, null, .{ .cache = &mtp2, .base_pos = &base2 });
    try testing.expectEqual(@as(usize, 63), res2.matched);
    try testing.expectEqual(@as(?usize, null), res2.mtp_base);
    try testing.expectEqual(@as(usize, 0), mtp2.step);
    try testing.expectEqual(@as(usize, 7), base2); // untouched
}

test "a hot entry holds a ringed layer's retained rows, not the ring buffer's capacity" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const window: u32 = 8;
    const n_layers: u32 = 4; // layer 0 global, 1..3 sliding
    const prompt_len: u32 = 700; // past the ring cap, so a compaction has run
    const chunk: u32 = 64;

    var toks: [prompt_len]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);

    var live = try KVCache.init(testing.allocator, n_layers);
    defer live.deinit();
    live.setSwaRing(window);
    try ringFill(&live, s, n_layers, window, 0, prompt_len, chunk);

    // The ring buffer is allocated at its cap from the first token, so the
    // rows past `offset` are rows no restore can ever read.
    var want_bytes: u64 = 0;
    var ring_rows: usize = 0;
    var ring_cap: usize = 0;
    for (live.entries) |*e| {
        if (!e.initialized) continue;
        const cap: usize = @intCast(mlx.getShape(e.keys.array)[2]);
        if (e.ringed) {
            ring_rows = e.offset;
            ring_cap = cap;
        }
        const rows: u64 = if (e.ringed) @intCast(e.offset) else @intCast(cap);
        want_bytes += rows * (ringRowBytes(e.keys.array) + ringRowBytes(e.values.array));
    }
    try testing.expect(ring_rows > 0 and ring_rows < ring_cap);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    _ = try hc.commitWithState(&live, &toks, false, 0, null, null, null);
    try testing.expectEqual(want_bytes, hc.current_kv_bytes);

    // A restore off that entry must still rebuild a ring and keep appending:
    // the reference cache walks the same absolute rows and never snapshots.
    var restored = try KVCache.init(testing.allocator, n_layers);
    defer restored.deinit();
    restored.setSwaRing(window);
    var moe_off: usize = 0;
    const hit = try hc.lookupAndRestore(&restored, &moe_off, null, s, &toks, false, 0, null, null);
    try testing.expectEqual(@as(usize, prompt_len - 1), hit.matched);

    var ref = try KVCache.init(testing.allocator, n_layers);
    defer ref.deinit();
    ref.setSwaRing(window);
    try ringFill(&ref, s, n_layers, window, 0, prompt_len + chunk, chunk);
    try ringFill(&restored, s, n_layers, window, @intCast(hit.matched), prompt_len + chunk, prompt_len + chunk);

    var li: u32 = 0;
    while (li < n_layers) : (li += 1) {
        var rv = try ringWriteLayer(&ref, s, li, prompt_len + chunk, 1, window);
        defer rv.deinit();
        var sv = try ringWriteLayer(&restored, s, li, prompt_len + chunk, 1, window);
        defer sv.deinit();
        try testing.expectEqual(@as(f32, 0), try trunk_helpers_test.maxAbsDiffF32(rv.k, sv.k, s));
        try testing.expectEqual(@as(f32, 0), try trunk_helpers_test.maxAbsDiffF32(rv.v, sv.v, s));
        try testing.expectEqual(ref.absSeqLen(li), restored.absSeqLen(li));
    }
}

test "a commit that records the live state for the SSD tier gives its RAM entry the record's rows, not a second copy" {
    try mlx.requireGpu();
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const window: u32 = 8;
    const n_layers: u32 = 4;
    const prompt_len: u32 = 700;
    var toks: [prompt_len]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
    var live = try KVCache.init(testing.allocator, n_layers);
    defer live.deinit();
    live.setSwaRing(window);
    try ringFill(&live, s, n_layers, window, 0, prompt_len, 64);

    // Without spare ring capacity both snapshots would share the live buffers and the test could not tell a copy.
    for (live.entries) |*l| {
        if (l.ringed) try testing.expect(l.offset < KVCache.bufferCapacity(l.keys.array));
    }

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.ssd_first = true;
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, buf[0..root_len], "fp-record-share", 0, 128);
    defer hc.deinit();
    _ = try hc.commitWithState(&live, &toks, false, 0, null, null, null);

    const record = &hc.pending_disk.?.snapshot;
    const entry = &hc.entries.items[0].snapshot;
    var ringed: usize = 0;
    for (record.entries, entry.entries) |*r, *e| {
        if (!r.ringed) continue;
        ringed += 1;
        try testing.expectEqual(mlx.mlx_array_data_uint8(r.keys.array), mlx.mlx_array_data_uint8(e.keys.array));
        try testing.expectEqual(mlx.mlx_array_data_uint8(r.values.array), mlx.mlx_array_data_uint8(e.values.array));
    }
    try testing.expect(ringed > 0);
}

test "a ringed entry restores at its prompt-end checkpoint when the next turn diverges past the ring" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const window: u32 = 8;
    const n_layers: u32 = 4; // layer 0 global, 1..3 sliding
    const prompt: u32 = 700;
    const reply: u32 = 600;

    var toks: [prompt + reply]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
    var live = try KVCache.init(testing.allocator, n_layers);
    defer live.deinit();
    const cp = try ringTurn(&live, s, n_layers, window, prompt, reply);

    var end = try live.snapshotRetained(s);
    const end_bytes = HotPrefixCache.snapshotBytes(&end);
    end.deinit();
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    _ = try hc.commitWithRing(&live, &toks, false, 0, 0, null, null, null, null, prompt, .{ .prompt_end = cp });
    // The bill: the window plus the backoff, per sliding layer, at the entry's KV width.
    var cp_bytes: u64 = 0;
    for (live.entries) |*e| {
        if (e.ringed) cp_bytes += (window + model_mod.ModelConfig.SWA_RING_CHECKPOINT_BACKOFF) * (ringRowBytes(e.keys.array) + ringRowBytes(e.values.array));
    }
    try testing.expectEqual(end_bytes + cp_bytes, hc.current_kv_bytes);

    // The next turn keeps the prompt and its first generated token, then diverges (the
    // template re-renders the reply), far below what the ring kept at the entry's end.
    var next: [prompt + 40]u32 = undefined;
    @memcpy(next[0 .. prompt + 1], toks[0 .. prompt + 1]);
    for (next[prompt + 1 ..], 0..) |*t, i| t.* = @intCast(900_000 + i);
    var restored = try KVCache.init(testing.allocator, n_layers);
    defer restored.deinit();
    restored.setSwaRing(window);
    var moe_off: usize = 0;
    const hit = try hc.lookupAndRestore(&restored, &moe_off, null, s, &next, false, 0, null, null);
    try testing.expectEqual(@as(usize, prompt), hit.matched);

    var ref = try KVCache.init(testing.allocator, n_layers);
    defer ref.deinit();
    ref.setSwaRing(window);
    try ringFill(&ref, s, n_layers, window, 0, prompt + 40, 64);
    try ringFill(&restored, s, n_layers, window, prompt, prompt + 40, 40);
    var li: u32 = 0;
    while (li < n_layers) : (li += 1) {
        var rv = try ringWriteLayer(&ref, s, li, prompt + 40, 1, window);
        defer rv.deinit();
        var sv = try ringWriteLayer(&restored, s, li, prompt + 40, 1, window);
        defer sv.deinit();
        try testing.expectEqual(@as(f32, 0), try trunk_helpers_test.maxAbsDiffF32(rv.k, sv.k, s));
        try testing.expectEqual(@as(f32, 0), try trunk_helpers_test.maxAbsDiffF32(rv.v, sv.v, s));
        try testing.expectEqual(ref.absSeqLen(li), restored.absSeqLen(li));
    }

    // A divergence below the checkpoint's rows is still un-restorable and cold-prefills.
    var edited: [prompt]u32 = undefined;
    @memcpy(edited[0 .. prompt - 40], toks[0 .. prompt - 40]);
    for (edited[prompt - 40 ..], 0..) |*t, i| t.* = @intCast(800_000 + i);
    var cold = try KVCache.init(testing.allocator, n_layers);
    defer cold.deinit();
    cold.setSwaRing(window);
    const miss = try hc.lookupAndRestore(&cold, &moe_off, null, s, &edited, false, 0, null, null);
    try testing.expectEqual(@as(usize, 0), miss.matched);
}

test "a ring checkpoint restore below the floor cold-prefills" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const window: u32 = 8;
    const n_layers: u32 = 4;
    const reply: u32 = 600;
    const cases = [_]struct { prompt: u32, shared: u32, want: usize }{
        // A short prompt's checkpoint covers it from row 0, so a shared chat header is servable.
        .{ .prompt = 24, .shared = 3, .want = 0 },
        .{ .prompt = RING_RESTORE_MIN_TOKENS + 20, .shared = RING_RESTORE_MIN_TOKENS + 21, .want = RING_RESTORE_MIN_TOKENS + 20 },
    };
    for (cases) |c| {
        var toks: [RING_RESTORE_MIN_TOKENS + 20 + reply]u32 = undefined;
        for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
        var live = try KVCache.init(testing.allocator, n_layers);
        defer live.deinit();
        const cp = try ringTurn(&live, s, n_layers, window, c.prompt, reply);
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        defer hc.deinit();
        _ = try hc.commitWithRing(&live, toks[0 .. c.prompt + reply], false, 0, 0, null, null, null, null, c.prompt, .{ .prompt_end = cp });

        var next: [RING_RESTORE_MIN_TOKENS + 60]u32 = undefined;
        @memcpy(next[0..c.shared], toks[0..c.shared]);
        for (next[c.shared..], 0..) |*t, i| t.* = @intCast(900_000 + i);
        var restored = try KVCache.init(testing.allocator, n_layers);
        defer restored.deinit();
        restored.setSwaRing(window);
        var moe_off: usize = 0;
        const hit = try hc.lookupAndRestore(&restored, &moe_off, null, s, &next, false, 0, null, null);
        try testing.expectEqual(c.want, hit.matched);
    }
}

test "a restore declined at the ring clamp leaves its entry's recency alone" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const window: u32 = 8;
    const n_layers: u32 = 4;
    const prompt: u32 = 700;
    const reply: u32 = 600;
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    // Two sessions that share only a chat header, the older one committed first.
    for ([_]u32{ 0, 500_000 }) |offset| {
        var toks: [prompt + reply]u32 = undefined;
        for (&toks, 0..) |*t, i| t.* = if (i < 3) @intCast(i + 1) else @intCast(offset + i + 1);
        var live = try KVCache.init(testing.allocator, n_layers);
        defer live.deinit();
        const cp = try ringTurn(&live, s, n_layers, window, prompt, reply);
        _ = try hc.commitWithRing(&live, &toks, false, 0, 0, null, null, null, null, prompt, .{ .prompt_end = cp });
    }
    const before = [_]u64{ hc.entries.items[0].last_used, hc.entries.items[1].last_used };
    try testing.expect(before[0] < before[1]);

    // A request sharing only the header lands on one of them and declines at the clamp.
    var header_only: [prompt]u32 = undefined;
    for (&header_only, 0..) |*t, i| t.* = if (i < 3) @intCast(i + 1) else @intCast(900_000 + i);
    var cold = try KVCache.init(testing.allocator, n_layers);
    defer cold.deinit();
    cold.setSwaRing(window);
    var moe_off: usize = 0;
    const miss = try hc.lookupAndRestore(&cold, &moe_off, null, s, &header_only, false, 0, null, null);
    try testing.expectEqual(@as(usize, 0), miss.matched);
    // A restore that did not happen is not a use: promoted, the older entry made the newer one the eviction victim.
    try testing.expectEqual(before[0], hc.entries.items[0].last_used);
    try testing.expectEqual(before[1], hc.entries.items[1].last_used);
}

test "ringed candidates rank by restorable position, not raw match" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const window: u32 = 8;
    const n_layers: u32 = 4;
    const prompt: u32 = 700;
    const reply: u32 = 600;
    const short: u32 = 400;

    var toks: [prompt + reply]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    // The longer raw match carries no checkpoint, so its ring cannot reach the divergence.
    var long = try KVCache.init(testing.allocator, n_layers);
    defer long.deinit();
    if (try ringTurn(&long, s, n_layers, window, prompt, reply)) |cp| {
        var doomed = cp;
        doomed.deinit();
    }
    _ = try hc.commitWithState(&long, &toks, false, 0, null, null, null);
    var shorter = try KVCache.init(testing.allocator, n_layers);
    defer shorter.deinit();
    shorter.setSwaRing(window);
    try ringFill(&shorter, s, n_layers, window, 0, short, 64);
    _ = try hc.commitWithState(&shorter, toks[0..short], false, 0, null, null, null);

    var next: [prompt + 40]u32 = undefined;
    @memcpy(next[0 .. prompt + 1], toks[0 .. prompt + 1]);
    for (next[prompt + 1 ..], 0..) |*t, i| t.* = @intCast(900_000 + i);
    var restored = try KVCache.init(testing.allocator, n_layers);
    defer restored.deinit();
    restored.setSwaRing(window);
    var moe_off: usize = 0;
    const hit = try hc.lookupAndRestore(&restored, &moe_off, null, s, &next, false, 0, null, null);
    try testing.expectEqual(@as(usize, short), hit.matched);
}

/// A ring checkpoint with no rows; `mark` tells two at one position apart.
fn stubRingCp(step: usize, mark: usize) !KVCacheSnapshot {
    const entries = try testing.allocator.alloc(trunk_mod.KVCacheEntry, 1);
    entries[0] = trunk_mod.newEmptyKVEntry();
    entries[0].offset = mark;
    return .{ .entries = entries, .step = step, .allocator = testing.allocator, .config = kv_quant.KVQuantConfig.dense };
}

fn stubRingCps(steps: []const usize, mark: usize) ![]KVCacheSnapshot {
    const out = try testing.allocator.alloc(KVCacheSnapshot, steps.len);
    for (out, steps) |*o, st| o.* = try stubRingCp(st, mark);
    return out;
}

test "ring checkpoints merge ascending, a tie keeps the newer, and past RING_CHECKPOINT_MAX the interior thins: the lowest and the newest stay" {
    const merged = (try HotPrefixCache.mergeRingCps(
        testing.allocator,
        try stubRingCps(&.{ 100, 300, 500, 1100, 1300, 1500 }, 1),
        try stubRingCps(&.{ 300, 1700, 1900, 2100, 2300 }, 2),
    )).?;
    defer HotPrefixCache.freeRingCps(testing.allocator, merged);
    try testing.expectEqual(RING_CHECKPOINT_MAX, merged.len);
    try testing.expectEqual(@as(usize, 100), merged[0].step);
    try testing.expectEqual(@as(usize, 2100), merged[merged.len - 2].step);
    try testing.expectEqual(@as(usize, 2300), merged[merged.len - 1].step);
    for (merged[0 .. merged.len - 1], merged[1..]) |a, b| try testing.expect(a.step < b.step);
    for (merged) |cp| {
        if (cp.step == 300) try testing.expectEqual(@as(usize, 2), cp.entries[0].offset);
    }

    const small = (try HotPrefixCache.mergeRingCps(testing.allocator, try stubRingCps(&.{ 100, 500 }, 1), try stubRingCps(&.{ 300, 500 }, 2))).?;
    defer HotPrefixCache.freeRingCps(testing.allocator, small);
    try testing.expectEqual(@as(usize, 3), small.len);
    for (small, [_]usize{ 100, 300, 500 }, [_]usize{ 1, 2, 2 }) |cp, step, mark| {
        try testing.expectEqual(step, cp.step);
        try testing.expectEqual(mark, cp.entries[0].offset);
    }
    try testing.expectEqual(@as(?[]KVCacheSnapshot, null), try HotPrefixCache.mergeRingCps(testing.allocator, null, null));

    // An inheritance shares what lies at or below the fork, thinned the same way.
    const src = try stubRingCps(&.{ 100, 300, 500, 700, 900, 1100, 1300, 1500, 1700, 1900 }, 1);
    defer HotPrefixCache.freeRingCps(testing.allocator, src);
    const shared = (try HotPrefixCache.shareRingCpsUpTo(testing.allocator, src, 1800, null)).?;
    defer HotPrefixCache.freeRingCps(testing.allocator, shared);
    try testing.expectEqual(RING_CHECKPOINT_MAX, shared.len);
    try testing.expectEqual(@as(usize, 100), shared[0].step);
    try testing.expectEqual(@as(usize, 1700), shared[shared.len - 1].step);
}

/// A ring checkpoint whose one entry holds 4 bf16 rows of 8 in K and in V.
fn sizedRingCp(step: usize) !KVCacheSnapshot {
    var cp = try stubRingCp(step, 0);
    errdefer cp.deinit();
    const e = &cp.entries[0];
    var keys = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_zeros(&keys, &[_]c_int{ 1, 1, 4, 8 }, 4, .bfloat16, mlx.gpuStream()));
    e.keys = mlx_pages.InPlace.adopt(keys, .paged);
    var values = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_zeros(&values, &[_]c_int{ 1, 1, 4, 8 }, 4, .bfloat16, mlx.gpuStream()));
    e.values = mlx_pages.InPlace.adopt(values, .paged);
    e.initialized = true;
    return cp;
}

test "a ring checkpoint list sheds bytes toward its lowest and its newest" {
    const bytes_each: u64 = 2 * 4 * 8 * 2;
    const cps = try testing.allocator.alloc(KVCacheSnapshot, 5);
    var made: usize = 0;
    errdefer {
        for (cps[0..made]) |*c| c.deinit();
        testing.allocator.free(cps);
    }
    for (cps, [_]usize{ 100, 300, 500, 700, 900 }) |*c, step| {
        c.* = try sizedRingCp(step);
        made += 1;
    }
    const kept = (try HotPrefixCache.thinRingCps(testing.allocator, cps, RING_CHECKPOINT_MAX, 3 * bytes_each)).?;
    defer HotPrefixCache.freeRingCps(testing.allocator, kept);
    try testing.expectEqual(@as(usize, 2), kept.len);
    try testing.expectEqual(@as(usize, 100), kept[0].step);
    try testing.expectEqual(@as(usize, 900), kept[1].step);
}

test "ring marks sit at the message starts a prefill forwards, thinned to RING_MARKS_MAX keeping the first and the last" {
    const im: u32 = 7;
    var prompt: [2000]u32 = @splat(1);
    for ([_]usize{ 0, 40, 300, 500, 900, 1100, 1300, 1985 }) |p| prompt[p] = im;
    var buf: [RING_MARKS_MAX]usize = undefined;
    // Under the restore floor and inside the prompt end's own reach are skipped.
    try testing.expectEqualSlices(usize, &.{ 300, 500, 900, 1300 }, ringMarkPositions(&prompt, 0, im, &buf));
    // A restored prefix is already in the cache: only the tail past it is marked.
    try testing.expectEqualSlices(usize, &.{ 900, 1100, 1300 }, ringMarkPositions(&prompt, 500, im, &buf));
    try testing.expectEqual(@as(usize, 0), ringMarkPositions(&prompt, 0, 99, &buf).len);
    try testing.expectEqual(@as(usize, 0), ringMarkPositions(prompt[0..60], 0, im, &buf).len);
}

test "a slot's ring restore points are freed by the slot and by a commit that cannot take them" {
    var held: SlotRingCps = .{ .fork = try stubRingCp(10, 0), .prompt_end = try stubRingCp(90, 0) };
    held.marks[0] = try stubRingCp(50, 0);
    held.n_marks = 1;
    held.deinit();
    try testing.expectEqual(@as(usize, 0), held.n_marks);

    var cps: SlotRingCps = .{ .fork = try stubRingCp(10, 0), .prompt_end = try stubRingCp(90, 0) };
    cps.marks[0] = try stubRingCp(50, 0);
    cps.marks[1] = try stubRingCp(70, 0);
    cps.n_marks = 2;
    var failing = std.testing.FailingAllocator.init(testing.allocator, .{ .fail_index = 0 });
    try testing.expectEqual(@as(?[]KVCacheSnapshot, null), HotPrefixCache.ownRingCps(failing.allocator(), cps));

    var ok: SlotRingCps = .{ .fork = try stubRingCp(10, 0), .prompt_end = try stubRingCp(90, 0) };
    ok.marks[0] = try stubRingCp(50, 0);
    ok.n_marks = 1;
    const owned = HotPrefixCache.ownRingCps(testing.allocator, ok).?;
    defer HotPrefixCache.freeRingCps(testing.allocator, owned);
    try testing.expectEqual(@as(usize, 3), owned.len);
    for (owned, [_]usize{ 10, 50, 90 }) |cp, step| try testing.expectEqual(step, cp.step);
}

test "a new session sharing only another's preamble restores at the message mark that prefill took" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const window: u32 = 8;
    const n_layers: u32 = 4;
    const preamble: u32 = 500;
    const prompt: u32 = 800;
    const reply: u32 = 600;

    var toks: [prompt + reply]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
    var live = try KVCache.init(testing.allocator, n_layers);
    defer live.deinit();
    live.setSwaRing(window);
    var cps: SlotRingCps = .{};
    defer cps.deinit();
    // The second mark lies past what this prefill forwards, so it never fills.
    cps.armMarks(&live, &.{ preamble, prompt + 10 });
    try testing.expectEqual(@as(usize, 2), cps.n_marks);
    live.ring_marks = cps.markSlice();
    // 300-row writes: the one that crosses the mark also compacts the ring past it.
    try ringFill(&live, s, n_layers, window, 0, prompt, 300);
    live.ring_marks = &.{};
    cps.keepCompleteMarks(&live);
    try testing.expectEqual(@as(usize, 1), cps.n_marks);
    cps.prompt_end = try live.ringCheckpoint(prompt, s);
    try ringFill(&live, s, n_layers, window, prompt, prompt + reply, 16);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    const moved = cps;
    cps = .{};
    _ = try hc.commitWithRing(&live, &toks, false, 0, 0, null, null, null, null, prompt, moved);
    try testing.expectEqual(@as(usize, 2), hc.entries.items[0].ring_cps.?.len);

    // Same preamble and message header, then another task.
    var next: [prompt]u32 = undefined;
    @memcpy(next[0 .. preamble + 3], toks[0 .. preamble + 3]);
    for (next[preamble + 3 ..], 0..) |*t, i| t.* = @intCast(900_000 + i);
    var restored = try KVCache.init(testing.allocator, n_layers);
    defer restored.deinit();
    restored.setSwaRing(window);
    var moe_off: usize = 0;
    const hit = try hc.lookupAndRestore(&restored, &moe_off, null, s, &next, false, 0, null, null);
    try testing.expectEqual(@as(usize, preamble), hit.matched);

    var ref = try KVCache.init(testing.allocator, n_layers);
    defer ref.deinit();
    ref.setSwaRing(window);
    try ringFill(&ref, s, n_layers, window, 0, prompt, 64);
    try ringFill(&restored, s, n_layers, window, preamble, prompt, 64);
    var li: u32 = 0;
    while (li < n_layers) : (li += 1) {
        var rv = try ringWriteLayer(&ref, s, li, prompt, 1, window);
        defer rv.deinit();
        var sv = try ringWriteLayer(&restored, s, li, prompt, 1, window);
        defer sv.deinit();
        try testing.expectEqual(@as(f32, 0), try trunk_helpers_test.maxAbsDiffF32(rv.k, sv.k, s));
        try testing.expectEqual(@as(f32, 0), try trunk_helpers_test.maxAbsDiffF32(rv.v, sv.v, s));
        try testing.expectEqual(ref.absSeqLen(li), restored.absSeqLen(li));
    }
}

test "a slot holding every ring restore point it may bills SLOT_RING_CHECKPOINTS checkpoints" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const window: u32 = 8;
    const n_layers: u32 = 4;
    var live = try KVCache.init(testing.allocator, n_layers);
    defer live.deinit();
    live.setSwaRing(window);
    var cps: SlotRingCps = .{};
    defer cps.deinit();
    try ringFill(&live, s, n_layers, window, 0, 400, 64);
    cps.fork = try live.ringCheckpoint(400, s);
    cps.armMarks(&live, &.{ 700, 1000, 1300, 1600, 1700 });
    try testing.expectEqual(RING_MARKS_MAX, cps.n_marks);
    live.ring_marks = cps.markSlice();
    try ringFill(&live, s, n_layers, window, 400, 1900, 256);
    live.ring_marks = &.{};
    cps.keepCompleteMarks(&live);
    cps.prompt_end = try live.ringCheckpoint(1900, s);
    try testing.expectEqual(RING_MARKS_MAX, cps.n_marks);

    var cp_bytes: u64 = 0;
    for (live.entries) |*e| {
        if (e.ringed) cp_bytes += (window + model_mod.ModelConfig.SWA_RING_CHECKPOINT_BACKOFF) * (ringRowBytes(e.keys.array) + ringRowBytes(e.values.array));
    }
    try testing.expectEqual(@as(u64, SLOT_RING_CHECKPOINTS) * cp_bytes, cps.bytes());
}

test "an entry extended in place keeps its older ring checkpoints" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const window: u32 = 8;
    const n_layers: u32 = 4;
    const prompt: u32 = 700;
    const reply: u32 = 600;
    const turn2 = prompt + reply + 100;

    var toks: [turn2 + 300]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();
    var live = try KVCache.init(testing.allocator, n_layers);
    defer live.deinit();
    const cp = try ringTurn(&live, s, n_layers, window, prompt, reply);
    _ = try hc.commitWithRing(&live, toks[0 .. prompt + reply], false, 0, 0, null, null, null, null, prompt, .{ .prompt_end = cp });

    // A client that sends the whole reply back extends the entry, which the commit replaces.
    var ext = try KVCache.init(testing.allocator, n_layers);
    defer ext.deinit();
    ext.setSwaRing(window);
    var moe_off: usize = 0;
    const hit = try hc.lookupAndRestore(&ext, &moe_off, null, s, toks[0..turn2], false, 0, null, null);
    try testing.expectEqual(@as(usize, prompt + reply), hit.matched);
    try ringFill(&ext, s, n_layers, window, prompt + reply, turn2, 64);
    var cp2 = try ext.ringCheckpoint(turn2, s);
    errdefer if (cp2) |*c| c.deinit();
    try ringFill(&ext, s, n_layers, window, turn2, toks.len, 16);
    const moved = cp2;
    cp2 = null;
    _ = try hc.commitWithRing(&ext, &toks, false, 0, 0, null, null, null, null, turn2, .{ .prompt_end = moved });
    try testing.expectEqual(@as(usize, 1), hc.entries.items.len);
    try testing.expectEqual(@as(usize, 2), hc.entries.items[0].ring_cps.?.len);

    // A turn diverging right after the first prompt still restores there.
    var next: [prompt + 40]u32 = undefined;
    @memcpy(next[0 .. prompt + 1], toks[0 .. prompt + 1]);
    for (next[prompt + 1 ..], 0..) |*t, i| t.* = @intCast(900_000 + i);
    var restored = try KVCache.init(testing.allocator, n_layers);
    defer restored.deinit();
    restored.setSwaRing(window);
    const again = try hc.lookupAndRestore(&restored, &moe_off, null, s, &next, false, 0, null, null);
    try testing.expectEqual(@as(usize, prompt), again.matched);
}

/// Bytes one row of `e`'s K and V bill, from their own shapes.
fn ringEntryRowBytes(e: *const trunk_mod.KVCacheEntry) u64 {
    return ringRowBytes(e.keys.array) + ringRowBytes(e.values.array);
}

/// The next turn's first write through both caches: identical views prove the restore.
fn ringExpectSameNext(ref: *KVCache, restored: *KVCache, s: mlx.mlx_stream, n_layers: u32, window: u32, at: u32) !void {
    var li: u32 = 0;
    while (li < n_layers) : (li += 1) {
        var rv = try ringWriteLayer(ref, s, li, at, 1, window);
        defer rv.deinit();
        var sv = try ringWriteLayer(restored, s, li, at, 1, window);
        defer sv.deinit();
        try testing.expectEqual(@as(f32, 0), try trunk_helpers_test.maxAbsDiffF32(rv.k, sv.k, s));
        try testing.expectEqual(@as(f32, 0), try trunk_helpers_test.maxAbsDiffF32(rv.v, sv.v, s));
        try testing.expectEqual(ref.absSeqLen(li), restored.absSeqLen(li));
    }
}

test "an oversized ringed entry trims to its end: the ring is a constant, not a per-token price" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const window: u32 = 8;
    const n_layers: u32 = 4; // layer 0 global, 1..3 sliding
    const prompt: u32 = 2000;

    var toks: [prompt]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
    var live = try KVCache.init(testing.allocator, n_layers);
    defer live.deinit();
    live.setSwaRing(window);
    try ringFill(&live, s, n_layers, window, 0, prompt, 64);

    // The global layer's buffer holds spare capacity past its rows, as a reservation does.
    var probe = try live.snapshotRetained(s);
    const cap: u64 = @intCast(mlx.getShape(probe.entries[0].keys.array)[2]);
    const rows_bytes = HotPrefixCache.snapshotBytes(&probe) - (cap - prompt) * ringEntryRowBytes(&probe.entries[0]);
    probe.deinit();
    try testing.expect(cap > prompt);

    // Room for the rows, not the capacity they sit in.
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, rows_bytes);
    defer hc.deinit();
    const status = try hc.commitWithState(&live, &toks, false, 0, null, null, null);
    try testing.expectEqual(std.meta.Tag(CommitStatus).ok, std.meta.activeTag(status));
    try testing.expectEqual(@as(usize, 1), hc.entries.items.len);
    try testing.expectEqual(@as(usize, prompt), hc.entries.items[0].tokens.len);
    try testing.expectEqual(rows_bytes, hc.current_kv_bytes);

    var restored = try KVCache.init(testing.allocator, n_layers);
    defer restored.deinit();
    restored.setSwaRing(window);
    var moe_off: usize = 0;
    const hit = try hc.lookupAndRestore(&restored, &moe_off, null, s, &toks, false, 0, null, null);
    try testing.expectEqual(@as(usize, prompt - 1), hit.matched);
    var ref = try KVCache.init(testing.allocator, n_layers);
    defer ref.deinit();
    ref.setSwaRing(window);
    try ringFill(&ref, s, n_layers, window, 0, prompt + 64, 64);
    try ringFill(&restored, s, n_layers, window, @intCast(hit.matched), prompt + 64, prompt + 64);
    try ringExpectSameNext(&ref, &restored, s, n_layers, window, prompt + 64);
}

test "an oversized ringed entry trims to the highest ring checkpoint that fits, and declines below every one" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const window: u32 = 8;
    const n_layers: u32 = 4;
    const prompt: u32 = 2000;
    const reply: u32 = 600;

    var toks: [prompt + reply]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
    var live = try KVCache.init(testing.allocator, n_layers);
    defer live.deinit();
    const cp = try ringTurn(&live, s, n_layers, window, prompt, reply);
    const cp_bytes = HotPrefixCache.snapshotBytes(&cp.?);
    var probe = try live.snapshotRetained(s);
    const global_row = ringEntryRowBytes(&probe.entries[0]);
    probe.deinit();

    // The prompt's global rows plus the checkpoint's window: the entry ends where the checkpoint is.
    const budget = prompt * global_row + cp_bytes;
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, budget);
    defer hc.deinit();
    const status = try hc.commitWithRing(&live, &toks, false, 0, 0, null, null, null, null, prompt, .{ .prompt_end = cp });
    try testing.expectEqual(std.meta.Tag(CommitStatus).ok, std.meta.activeTag(status));
    try testing.expectEqual(@as(usize, 1), hc.entries.items.len);
    try testing.expectEqual(@as(usize, prompt), hc.entries.items[0].tokens.len);
    try testing.expect(hc.entries.items[0].ring_cps == null);
    try testing.expectEqual(budget, hc.current_kv_bytes);

    // The next turn keeps the prompt and diverges after it.
    var next: [prompt + 40]u32 = undefined;
    @memcpy(next[0 .. prompt + 1], toks[0 .. prompt + 1]);
    for (next[prompt + 1 ..], 0..) |*t, i| t.* = @intCast(900_000 + i);
    var restored = try KVCache.init(testing.allocator, n_layers);
    defer restored.deinit();
    restored.setSwaRing(window);
    var moe_off: usize = 0;
    const hit = try hc.lookupAndRestore(&restored, &moe_off, null, s, &next, false, 0, null, null);
    try testing.expectEqual(@as(usize, prompt), hit.matched);
    var ref = try KVCache.init(testing.allocator, n_layers);
    defer ref.deinit();
    ref.setSwaRing(window);
    try ringFill(&ref, s, n_layers, window, 0, prompt + 40, 64);
    try ringFill(&restored, s, n_layers, window, prompt, prompt + 40, 40);
    try ringExpectSameNext(&ref, &restored, s, n_layers, window, prompt + 40);

    // No room for the checkpoint's window: nothing restorable fits.
    var live2 = try KVCache.init(testing.allocator, n_layers);
    defer live2.deinit();
    const cp2 = try ringTurn(&live2, s, n_layers, window, prompt, reply);
    var tight = HotPrefixCache.initWithMem(testing.allocator, 4, prompt * global_row);
    defer tight.deinit();
    const declined = try tight.commitWithRing(&live2, &toks, false, 0, 0, null, null, null, null, prompt, .{ .prompt_end = cp2 });
    try testing.expectEqual(std.meta.Tag(CommitStatus).declined, std.meta.activeTag(declined));
    try testing.expectEqual(@as(usize, 0), tight.entries.items.len);
}

test "a replacement over the budget sheds its lowest ring checkpoints before the entry" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const window: u32 = 8;
    const n_layers: u32 = 4;
    const prompt: u32 = 700;
    const reply: u32 = 600;
    const turn2 = prompt + reply + 100;

    var toks: [turn2 + 300]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);

    // Turn 2's cache first, to price a budget that holds it and its own checkpoint.
    var ext = try KVCache.init(testing.allocator, n_layers);
    defer ext.deinit();
    ext.setSwaRing(window);
    try ringFill(&ext, s, n_layers, window, 0, turn2, 64);
    var cp2 = try ext.ringCheckpoint(turn2, s);
    errdefer if (cp2) |*c| c.deinit();
    try ringFill(&ext, s, n_layers, window, turn2, toks.len, 16);
    var probe = try ext.snapshotRetained(s);
    const alone = HotPrefixCache.snapshotBytes(&probe) + HotPrefixCache.snapshotBytes(&cp2.?);
    probe.deinit();

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, alone);
    defer hc.deinit();
    var live = try KVCache.init(testing.allocator, n_layers);
    defer live.deinit();
    const cp = try ringTurn(&live, s, n_layers, window, prompt, reply);
    _ = try hc.commitWithRing(&live, toks[0 .. prompt + reply], false, 0, 0, null, null, null, null, prompt, .{ .prompt_end = cp });
    try testing.expectEqual(@as(usize, 1), hc.entries.items.len);

    // Turn 2 extends the entry, and the merge brings turn 1's checkpoint along: one too many.
    const moved = cp2;
    cp2 = null;
    const status = try hc.commitWithRing(&ext, &toks, false, 0, 0, null, null, null, null, turn2, .{ .prompt_end = moved });
    try testing.expectEqual(std.meta.Tag(CommitStatus).ok, std.meta.activeTag(status));
    try testing.expectEqual(@as(usize, 1), hc.entries.items.len);
    try testing.expectEqual(@as(usize, toks.len), hc.entries.items[0].tokens.len);
    try testing.expectEqual(@as(usize, 1), hc.entries.items[0].ring_cps.?.len);
    try testing.expectEqual(@as(usize, turn2), hc.entries.items[0].ring_cps.?[0].step);
    try testing.expectEqual(alone, hc.current_kv_bytes);
}

test "a ring checkpoint taken at the restore keeps the fork when the donor is evicted before the commit" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const window: u32 = 8;
    const n_layers: u32 = 4;
    const prompt: u32 = 700;
    const reply: u32 = 600;
    const fork_prompt = prompt + 100 + 330;

    var toks: [fork_prompt + reply]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
    var hc = HotPrefixCache.initWithMem(testing.allocator, 1, 0);
    defer hc.deinit();
    var donor = try KVCache.init(testing.allocator, n_layers);
    defer donor.deinit();
    const donor_cp = try ringTurn(&donor, s, n_layers, window, prompt, reply);
    var donor_toks: [prompt + reply]u32 = undefined;
    @memcpy(donor_toks[0..prompt], toks[0..prompt]);
    for (donor_toks[prompt..], 0..) |*t, i| t.* = @intCast(700_000 + i);
    _ = try hc.commitWithRing(&donor, &donor_toks, false, 0, 0, null, null, null, null, prompt, .{ .prompt_end = donor_cp });

    var fork = try KVCache.init(testing.allocator, n_layers);
    defer fork.deinit();
    fork.setSwaRing(window);
    var moe_off: usize = 0;
    const fork_hit = try hc.lookupAndRestore(&fork, &moe_off, null, s, toks[0..fork_prompt], false, 0, null, null);
    try testing.expectEqual(@as(usize, prompt), fork_hit.matched);
    var at_restore = try fork.ringCheckpoint(fork_hit.matched, s);
    errdefer if (at_restore) |*c| c.deinit();
    try ringFill(&fork, s, n_layers, window, prompt, fork_prompt, 64);
    var at_end = try fork.ringCheckpoint(fork_prompt, s);
    errdefer if (at_end) |*c| c.deinit();
    try ringFill(&fork, s, n_layers, window, fork_prompt, fork_prompt + reply, 16);

    // Another slot commits first and the count cap evicts the donor: nothing is left to inherit.
    var other = try KVCache.init(testing.allocator, n_layers);
    defer other.deinit();
    other.setSwaRing(window);
    try ringFill(&other, s, n_layers, window, 0, 300, 64);
    var other_toks: [300]u32 = undefined;
    for (&other_toks, 0..) |*t, i| t.* = @intCast(500_000 + i);
    _ = try hc.commitWithState(&other, &other_toks, false, 0, null, null, null);
    const cps: SlotRingCps = .{ .fork = at_restore, .prompt_end = at_end };
    at_restore = null;
    at_end = null;
    _ = try hc.commitWithRing(&fork, &toks, false, 0, 0, null, null, null, null, fork_prompt, cps);
    try testing.expectEqual(@as(usize, 1), hc.entries.items.len);
    try testing.expectEqual(@as(usize, 2), hc.entries.items[0].ring_cps.?.len);

    var next: [prompt + 100 + 40]u32 = undefined;
    @memcpy(next[0 .. prompt + 100], toks[0 .. prompt + 100]);
    for (next[prompt + 100 ..], 0..) |*t, i| t.* = @intCast(900_000 + i);
    var restored = try KVCache.init(testing.allocator, n_layers);
    defer restored.deinit();
    restored.setSwaRing(window);
    const hit = try hc.lookupAndRestore(&restored, &moe_off, null, s, &next, false, 0, null, null);
    try testing.expectEqual(@as(usize, prompt), hit.matched);
    try expectRingContinuesCold(&restored, s, n_layers, window, prompt, next.len);
}

test "a fork entry keeps its donor's ring checkpoint, so evicting the donor still restores at the fork" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const window: u32 = 8;
    const n_layers: u32 = 4;
    const prompt: u32 = 700;
    const reply: u32 = 600;
    const tail: u32 = 100; // the conversation past the donor's prompt (its reply, re-rendered)
    const reminder: u32 = 330;
    const fork_prompt = prompt + tail + reminder;

    var toks: [fork_prompt + reply]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
    var hc = HotPrefixCache.initWithMem(testing.allocator, 2, 0);
    defer hc.deinit();

    var donor = try KVCache.init(testing.allocator, n_layers);
    defer donor.deinit();
    const donor_cp = try ringTurn(&donor, s, n_layers, window, prompt, reply);
    var donor_toks: [prompt + reply]u32 = undefined;
    @memcpy(donor_toks[0..prompt], toks[0..prompt]);
    for (donor_toks[prompt..], 0..) |*t, i| t.* = @intCast(700_000 + i);
    _ = try hc.commitWithRing(&donor, &donor_toks, false, 0, 0, null, null, null, null, prompt, .{ .prompt_end = donor_cp });

    // The side request: the conversation plus an appended reminder, restored off the donor.
    var fork = try KVCache.init(testing.allocator, n_layers);
    defer fork.deinit();
    fork.setSwaRing(window);
    var moe_off: usize = 0;
    const fork_hit = try hc.lookupAndRestore(&fork, &moe_off, null, s, toks[0..fork_prompt], false, 0, null, null);
    try testing.expectEqual(@as(usize, prompt), fork_hit.matched);
    try ringFill(&fork, s, n_layers, window, prompt, fork_prompt, 64);
    var fork_cp = try fork.ringCheckpoint(fork_prompt, s);
    errdefer if (fork_cp) |*c| c.deinit();
    try ringFill(&fork, s, n_layers, window, fork_prompt, fork_prompt + reply, 16);
    const moved_cp = fork_cp;
    fork_cp = null;
    _ = try hc.commitWithRing(&fork, &toks, false, 0, 0, null, null, null, null, fork_prompt, .{ .prompt_end = moved_cp });

    // The fork entry bills its own end, its own checkpoint and the inherited one.
    var end = try fork.snapshotRetained(s);
    const end_bytes = HotPrefixCache.snapshotBytes(&end);
    end.deinit();
    var cp_bytes: u64 = 0;
    for (fork.entries) |*e| {
        if (e.ringed) cp_bytes += (window + model_mod.ModelConfig.SWA_RING_CHECKPOINT_BACKOFF) * (ringRowBytes(e.keys.array) + ringRowBytes(e.values.array));
    }
    for (hc.entries.items) |*e| {
        if (e.tokens.len == toks.len) try testing.expectEqual(end_bytes + 2 * cp_bytes, e.kv_bytes);
    }

    // An unrelated request under the count cap evicts the donor, the LRU entry.
    var other = try KVCache.init(testing.allocator, n_layers);
    defer other.deinit();
    other.setSwaRing(window);
    try ringFill(&other, s, n_layers, window, 0, 300, 64);
    var other_toks: [300]u32 = undefined;
    for (&other_toks, 0..) |*t, i| t.* = @intCast(500_000 + i);
    _ = try hc.commitWithState(&other, &other_toks, false, 0, null, null, null);
    for (hc.entries.items) |*e| try testing.expect(e.tokens.len != donor_toks.len);

    // The next main turn shares the conversation and diverges where the reminder was appended.
    var next: [prompt + tail + 40]u32 = undefined;
    @memcpy(next[0 .. prompt + tail], toks[0 .. prompt + tail]);
    for (next[prompt + tail ..], 0..) |*t, i| t.* = @intCast(900_000 + i);
    var restored = try KVCache.init(testing.allocator, n_layers);
    defer restored.deinit();
    restored.setSwaRing(window);
    const hit = try hc.lookupAndRestore(&restored, &moe_off, null, s, &next, false, 0, null, null);
    try testing.expectEqual(@as(usize, prompt), hit.matched);
    try expectRingContinuesCold(&restored, s, n_layers, window, prompt, next.len);
}

test "a ringed entry persists to the SSD tier and restores only at a ring checkpoint after a restart" {
    try mlx.requireGpu();
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    const window: u32 = 8;
    const n_layers: u32 = 4;
    const prompt: u32 = 700;
    const reply: u32 = 600;

    var toks: [prompt + reply]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
    try persistRingTurn(io, base, "fp-ring", &toks, prompt, window, n_layers);

    var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-ring", 0, 128);
    defer hc2.deinit();
    var moe_off: usize = 0;

    // The next turn diverges right after the prompt: the prompt-end checkpoint serves it.
    var next: [prompt + 40]u32 = undefined;
    @memcpy(next[0 .. prompt + 1], toks[0 .. prompt + 1]);
    for (next[prompt + 1 ..], 0..) |*t, i| t.* = @intCast(900_000 + i);
    var a = try KVCache.init(testing.allocator, n_layers);
    defer a.deinit();
    a.setSwaRing(window);
    const hit = try hc2.lookupAndRestore(&a, &moe_off, null, s, &next, false, 0, null, null);
    try testing.expectEqual(@as(usize, prompt), hit.matched);
    // A ring restore's bill was never measured, so admission credits none of its rows.
    try testing.expect(!hit.ownsRestoredRows());
    try expectRingContinuesCold(&a, s, n_layers, window, prompt, next.len);

    // A verbatim re-send restores at its last token off the ring the entry ended with.
    var b = try KVCache.init(testing.allocator, n_layers);
    defer b.deinit();
    b.setSwaRing(window);
    const full = try hc2.lookupAndRestore(&b, &moe_off, null, s, &toks, false, 0, null, null);
    try testing.expect(full.full_match);
    try testing.expectEqual(@as(usize, toks.len - 1), full.matched);
    try testing.expect(!full.ownsRestoredRows());
    try expectRingContinuesCold(&b, s, n_layers, window, toks.len - 1, toks.len);

    // A divergence no ring checkpoint reaches cold-prefills.
    var early: [prompt]u32 = undefined;
    @memcpy(early[0..400], toks[0..400]);
    for (early[400..], 0..) |*t, i| t.* = @intCast(800_000 + i);
    var c = try KVCache.init(testing.allocator, n_layers);
    defer c.deinit();
    c.setSwaRing(window);
    const miss = try hc2.lookupAndRestore(&c, &moe_off, null, s, &early, false, 0, null, null);
    try testing.expectEqual(@as(usize, 0), miss.matched);
}

/// One ringed-trunk turn (`ringTurn`) committed and flushed to a fresh SSD tier at `base/fp`.
fn persistRingTurn(io: std.Io, base: []const u8, fp: []const u8, toks: []const u32, prompt: u32, window: u32, n_layers: u32) !void {
    const s = mlx.gpuStream();
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, fp, 0, 128);
    defer hc.deinit();
    var live = try KVCache.init(testing.allocator, n_layers);
    defer live.deinit();
    const cp = try ringTurn(&live, s, n_layers, window, prompt, @intCast(toks.len - prompt));
    // Without its restore points a ringed cache is not persistable.
    try testing.expectEqual(kv_disk_cache.PersistOutcome.skipped, try hc.disk.?.appendCommit(live.entries, live.step, live.config, toks, false, null, s));
    _ = try hc.commitWithRing(&live, toks, false, 0, 0, null, null, null, null, prompt, .{ .prompt_end = cp });
    hc.flushPendingDisk(s);
    try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
}

test "a ringed SSD entry bills its ring files, and a truncated one is salvaged away at scan" {
    try mlx.requireGpu();
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    const window: u32 = 8;
    const n_layers: u32 = 4;
    const prompt: u32 = 700;

    var toks: [prompt + 600]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
    try persistRingTurn(io, base, "fp-ring-salvage", &toks, prompt, window, n_layers);

    {
        // The tier bills exactly the files it wrote: chunks, the token record and both ring files.
        var d = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-ring-salvage", 0, 128);
        defer d.deinit();
        const e = &d.entries.items[0];
        try testing.expectEqual(@as(usize, 2), e.rings.len);
        try testing.expectEqual(@as(u32, prompt), e.rings[0].pos);
        try testing.expectEqual(@as(u32, toks.len), e.rings[1].pos);
        var on_disk: u64 = 0;
        var eb: [std.fs.max_path_bytes]u8 = undefined;
        const edir = try std.fmt.bufPrint(&eb, "{s}/fp-ring-salvage/e1", .{base});
        const names = try kv_disk_fs.testList(io, edir);
        defer kv_disk_fs.Fs.freeEntries(testing.allocator, names);
        for (names) |f| {
            if (std.mem.eql(u8, f.name, "meta.json")) continue;
            var fb: [std.fs.max_path_bytes]u8 = undefined;
            on_disk += kv_disk_fs.testSize(io, try std.fmt.bufPrint(&fb, "{s}/{s}", .{ edir, f.name })) orelse return error.TestUnexpectedResult;
        }
        try testing.expectEqual(on_disk, d.total_bytes);
    }

    // kill -9 mid-write: the prompt-end file no longer matches its record.
    try kv_disk_fs.testWriteAt(io, base, "fp-ring-salvage/e1/r0000700.safetensors", "trunc");
    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-ring-salvage", 0, 128);
        defer hc.deinit();
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entries.items[0].rings.len);
        var next: [prompt + 40]u32 = undefined;
        @memcpy(next[0 .. prompt + 1], toks[0 .. prompt + 1]);
        for (next[prompt + 1 ..], 0..) |*t, i| t.* = @intCast(900_000 + i);
        var a = try KVCache.init(testing.allocator, n_layers);
        defer a.deinit();
        a.setSwaRing(window);
        var moe_off: usize = 0;
        try testing.expectEqual(@as(usize, 0), (try hc.lookupAndRestore(&a, &moe_off, null, s, &next, false, 0, null, null)).matched);
        var b = try KVCache.init(testing.allocator, n_layers);
        defer b.deinit();
        b.setSwaRing(window);
        try testing.expectEqual(@as(usize, toks.len - 1), (try hc.lookupAndRestore(&b, &moe_off, null, s, &toks, false, 0, null, null)).matched);
    }

    // With no ring file left the chunks restore nothing: the entry goes.
    try kv_disk_fs.testWriteAt(io, base, "fp-ring-salvage/e1/r0001300.safetensors", "trunc");
    var d = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-ring-salvage", 0, 128);
    defer d.deinit();
    try testing.expectEqual(@as(usize, 0), d.entryCount());
}

test "an SSD entry without ring rows is never restored into a ringed cache" {
    try mlx.requireGpu();
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(io, &buf)];
    const window: u32 = 8;
    const n_layers: u32 = 4;

    var toks: [700]u32 = undefined;
    for (&toks, 0..) |*t, i| t.* = @intCast(i + 1);
    {
        // Every layer holds the whole prefix (no ring), as a pre-ring manifest describes it.
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-ring-old", 0, 128);
        defer hc.deinit();
        var plain = try KVCache.init(testing.allocator, n_layers);
        defer plain.deinit();
        try testFillCache(&plain, s, n_layers, toks.len);
        _ = try hc.commit(&plain, &toks, false);
        hc.flushPendingDisk(s);
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    }
    var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-ring-old", 0, 128);
    defer hc2.deinit();
    var next: [800]u32 = undefined;
    @memcpy(next[0..toks.len], &toks);
    for (next[toks.len..], 0..) |*t, i| t.* = @intCast(900_000 + i);
    var ringed = try KVCache.init(testing.allocator, n_layers);
    defer ringed.deinit();
    ringed.setSwaRing(window);
    var moe_off: usize = 0;
    const miss = try hc2.lookupAndRestore(&ringed, &moe_off, null, s, &next, false, 0, null, null);
    try testing.expectEqual(@as(usize, 0), miss.matched);
    for (ringed.entries) |*e| try testing.expect(!e.initialized);
}

test "HotPrefixCache: disk tier restores across a fresh cache instance (restart shape)" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const base = buf[0..root_len];

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // Session 1: commit through the NORMAL RAM path, then flush to disk
    // (the post-markFinished call the scheduler makes).
    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-hc", 0, 128);
        defer hc.deinit();

        var cache = try KVCache.init(testing.allocator, 2);
        defer cache.deinit();
        try testFillCache(&cache, s, 2, 600);
        _ = try hc.commit(&cache, &tokens, false);
        try testing.expect(hc.disk_dirty);
        hc.flushPendingDisk(s);
        try testing.expect(!hc.disk_dirty);
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    }

    // Session 2 ("server restart"): fresh RAM cache, fresh tier over the same
    // root. The lookup must land on the SSD tier and restore the prefix.
    {
        var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-hc", 0, 128);
        defer hc2.deinit();
        try testing.expectEqual(@as(usize, 0), hc2.entryCount()); // RAM empty
        try testing.expectEqual(@as(usize, 1), hc2.disk.?.entryCount());

        var cache2 = try KVCache.init(testing.allocator, 2);
        defer cache2.deinit();
        var moe_off: usize = 0;
        const res = try hc2.lookupAndRestore(&cache2, &moe_off, null, s, &tokens, false, 0, null, null);
        // Full match: identical re-issue semantics — truncate to len-1 and
        // re-forward the last token, exactly like a RAM full-match hit.
        try testing.expect(res.full_match);
        try testing.expectEqual(@as(usize, 599), res.matched);
        try testing.expectEqual(cache_stats.Tier.ssd, hc2.lastRestoredTier());
        try testing.expectEqual(@as(usize, 599), cache2.step);
        try testing.expectEqual(@as(usize, 599), moe_off);
        for (cache2.entries) |*e| {
            try testing.expect(e.initialized);
            try testing.expectEqual(@as(usize, 599), e.offset);
        }

        // Diverged-tail shape: shares the first 400 tokens, then differs.
        // Restore must land at 400 and leave the tail to prefill.
        var tokens_div: [700]u32 = undefined;
        for (&tokens_div, 0..) |*t, i| t.* = if (i < 400) tokens[i] else @intCast(i + 500_000);
        var cache3 = try KVCache.init(testing.allocator, 2);
        defer cache3.deinit();
        var moe_off3: usize = 0;
        const res3 = try hc2.lookupAndRestore(&cache3, &moe_off3, null, s, &tokens_div, false, 0, null, null);
        try testing.expect(!res3.full_match);
        try testing.expectEqual(@as(usize, 400), res3.matched);
        try testing.expectEqual(@as(usize, 400), cache3.step);
        // A diverged short prefix must read ONLY the chunks covering the
        // usable 400 positions (ceil(400/128) = 4), NOT the whole 600-token
        // stored entry (5 chunks). Loading the full entry to serve a short
        // shared prefix makes a diverged "hit" slower than a cold prefill.
        try testing.expectEqual(@as(u32, 4), hc2.disk.?.chunks_loaded_last);
    }
}

test "HotPrefixCache: budget decline reports a status, not a silent success" {
    // Live 2026-09-07: every oversized decline from a cancelled prefill
    // logged "[hot-cache] committed N/M" at the scheduler because the
    // decline exits commitWithMediaState via plain return — a 122k agent
    // session looked cached while it cold-prefilled ~95k tokens per retry.
    // The commit contract must be observable: ok / kept_resident / declined.
    const s = mlx.gpuStream();
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 16 * 1024);
    defer hc.deinit();

    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try testFillCache(&cache, s, 2, 600);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // 16 KB budget against ~128 B/token rows: even the 256-token commit
    // floor cannot fit, so the candidate must be DECLINED — and the caller
    // must be able to SEE that instead of mistaking it for a commit.
    const st = try hc.commit(&cache, &tokens, false);
    try testing.expect(st == .declined);
    try testing.expectEqual(@as(usize, 0), hc.entryCount());

    // A candidate that fits commits with its effective token length.
    var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc2.deinit();
    var cache2 = try KVCache.init(testing.allocator, 2);
    defer cache2.deinit();
    try testFillCache(&cache2, s, 2, 600);
    const st2 = try hc2.commit(&cache2, &tokens, false);
    try testing.expect(st2 == .ok);
    try testing.expectEqual(@as(usize, 600), st2.ok);
}

test "HotPrefixCache: an entry shorter than its media boundary is pure text" {
    // Live 2026-09-07: a cancelled prefill that forwarded fewer tokens than
    // the request's first media position kept the PIXEL vision_key while
    // the boundary was dropped (scheduler shape: media_start=null, key kept)
    // — a boundary-less cross-key entry the lookup conservatively rejects.
    // The with-image and no-image retry shapes of one conversation locked
    // each other out of their shared 121819-token text prefix. An entry
    // whose [0, len) covers no media rows is text: key it as text.
    const s = mlx.gpuStream();
    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    defer hc.deinit();

    var cache = try KVCache.init(testing.allocator, 2);
    defer cache.deinit();
    try testFillCache(&cache, s, 2, 600);
    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // Cancel shape (the scheduler passes the boundary raw): media declared
    // at 500, entry covers [0, 400) — no media rows inside, so the entry
    // must be text-keyed.
    const st = try hc.commitWithMediaState(&cache, tokens[0..400], false, 0xABCD, 0, 500, null, null, null, 400);
    try testing.expect(st == .ok);
    try testing.expectEqual(@as(u64, 0), hc.entries.items[0].vision_key);
    try testing.expect(hc.entries.items[0].media_start == null);

    // Legacy shape (commitWithState's contract): a null boundary KEEPS the
    // caller's pixel key — boundary-less keyed entries stay possible for
    // callers that model their own media state.
    var cache_b = try KVCache.init(testing.allocator, 2);
    defer cache_b.deinit();
    try testFillCache(&cache_b, s, 2, 600);
    _ = try hc.commitWithState(&cache_b, tokens[0..400], false, 0xBEEF, null, null, null);
    try testing.expectEqual(@as(u64, 0xBEEF), hc.entries.items[1].vision_key);

    // The no-image retry shape (vision_key 0, no media) must restore it.
    {
        var c2 = try KVCache.init(testing.allocator, 2);
        defer c2.deinit();
        var moe: usize = 0;
        const res = try hc.lookupAndRestoreWithMedia(&c2, &moe, null, s, &tokens, false, 0, null, &.{}, null, null, null, false);
        try testing.expectEqual(@as(usize, 400), res.matched);
    }
    // A different-image request caps at ITS media boundary (500 > 400 here,
    // so the whole text entry is reusable through it).
    {
        var c3 = try KVCache.init(testing.allocator, 2);
        defer c3.deinit();
        var moe: usize = 0;
        const res = try hc.lookupAndRestoreWithMedia(&c3, &moe, null, s, &tokens, false, 0xFFFF, 500, &.{}, null, null, null, false);
        try testing.expectEqual(@as(usize, 400), res.matched);
    }

    // Trim shape: media INSIDE the candidate, budget forces a trim below the
    // boundary — the trimmed entry is text and must re-key to 0.
    var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 32 * 1024);
    defer hc2.deinit();
    var cache2 = try KVCache.init(testing.allocator, 2);
    defer cache2.deinit();
    try testFillCache(&cache2, s, 2, 600);
    const st2 = try hc2.commitWithMediaState(&cache2, &tokens, false, 0xABCD, 0, 300, null, null, null, tokens.len);
    try testing.expect(st2 == .ok);
    try testing.expectEqual(@as(usize, 256), st2.ok); // 32 KB / 128 B rows
    try testing.expectEqual(@as(u64, 0), hc2.entries.items[0].vision_key);
    try testing.expect(hc2.entries.items[0].media_start == null);
}

test "HotPrefixCache: media request restores the pre-media text prefix from SSD" {
    // Live 2026-09-07: the SSD-tier lookup skipped ANY request with a
    // non-zero vision_key (`vision_key != 0 → break :disk`) — a 122k agent
    // session whose LAST turn carries one screenshot never touched the
    // 40 GB of persisted text prefixes, though 99.8% of the prompt is
    // pixel-independent text. Disk entries are text-only by construction
    // (the flush refuses vision-keyed entries), so the restore is exactly
    // the RAM path's pre-media semantics: cap at the request's media_start.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const base = buf[0..root_len];

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // Session 1: a text-only conversation commits + flushes to disk.
    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-vision-disk", 0, 128);
        defer hc.deinit();
        var cache = try KVCache.init(testing.allocator, 2);
        defer cache.deinit();
        try testFillCache(&cache, s, 2, 600);
        _ = try hc.commit(&cache, &tokens, false);
        hc.flushPendingDisk(s);
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    }

    // Session 2 (RAM empty): an image-bearing request over the same text —
    // the disk tier must serve the shared text prefix, capped at the
    // request's media boundary (400), reading only the covering chunks.
    {
        var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-vision-disk", 0, 128);
        defer hc2.deinit();

        var cache2 = try KVCache.init(testing.allocator, 2);
        defer cache2.deinit();
        var moe: usize = 0;
        const res = try hc2.lookupAndRestoreWithMedia(&cache2, &moe, null, s, &tokens, false, 0xDEAD, 400, &.{}, null, null, null, false);
        try testing.expect(!res.full_match);
        try testing.expectEqual(@as(usize, 400), res.matched);
        try testing.expectEqual(@as(usize, 400), cache2.step);
        try testing.expectEqual(@as(u32, 4), hc2.disk.?.chunks_loaded_last); // ceil(400/128)
    }
}

test "HotPrefixCache: dflash + mtp snapshots survive the SSD tier across a restart" {
    // A disk-tier restore forwards NO trunk layers, so state derived from
    // trunk hiddens (dflash context, MTP history) started EMPTY on every
    // disk hit — multi-turn across a restart drafted blind (the same
    // 92.6% → 66.5% acceptance class the RAM tier fixed). v4 persists both
    // in the entry's spec sidecar and restores them under the RAM tier's
    // exact clamp rule.
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const base = buf[0..root_len];

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    // Session 1: commit with BOTH spec payloads through the normal RAM path,
    // then flush (the post-markFinished call the scheduler makes).
    {
        var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-spec-hc", 0, 128);
        defer hc.deinit();

        var trunk = try KVCache.init(testing.allocator, 2);
        defer trunk.deinit();
        try testFillCache(&trunk, s, 2, 600);
        var assist = try KVCache.init(testing.allocator, 2);
        defer assist.deinit();
        try testFillCache(&assist, s, 2, 600);
        var hist = try KVCache.init(testing.allocator, 1);
        defer hist.deinit();
        try testFillCache(&hist, s, 1, 600);

        _ = try hc.commitWithSsm(&trunk, &tokens, false, null, .{ .cache = &assist, .base_pos = 0 }, .{ .cache = &hist, .base_pos = 0 });
        hc.flushPendingDisk(s);
        try testing.expect(!hc.disk_dirty);
        try testing.expect(hc.disk.?.entries.items[0].spec_dflash != null);
        try testing.expect(hc.disk.?.entries.items[0].spec_mtp != null);
    }

    // Session 2 ("server restart"): RAM empty, disk serves the prefix AND
    // both spec snapshots, clamped to the trunk's matched length.
    {
        var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
        hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-spec-hc", 0, 128);
        defer hc2.deinit();
        try testing.expectEqual(@as(usize, 0), hc2.entryCount());

        var trunk2 = try KVCache.init(testing.allocator, 2);
        defer trunk2.deinit();
        var dfl = try KVCache.init(testing.allocator, 2);
        defer dfl.deinit();
        var mtp_dst = try KVCache.init(testing.allocator, 1);
        defer mtp_dst.deinit();
        var moe_off: usize = 0;
        var dbase: usize = 99;
        var mbase: usize = 99;
        const res = try hc2.lookupAndRestore(
            &trunk2,
            &moe_off,
            null,
            s,
            &tokens,
            false,
            0,
            .{ .cache = &dfl, .base_pos = &dbase },
            .{ .cache = &mtp_dst, .base_pos = &mbase },
        );
        // Full match: identical re-issue → trunk lands at 599; both spec
        // caches clamp to base + step == matched.
        try testing.expect(res.full_match);
        try testing.expectEqual(@as(usize, 599), res.matched);
        try testing.expectEqual(@as(?usize, 0), res.dflash_base);
        try testing.expectEqual(@as(usize, 0), dbase);
        try testing.expectEqual(@as(usize, 599), dfl.step);
        try testing.expectEqual(@as(?usize, 0), res.mtp_base);
        try testing.expectEqual(@as(usize, 599), mtp_dst.step);

        // A geometry the persisted snap doesn't fit starts BLIND, never
        // wrong: a 3-layer dflash target declines.
        var trunk3 = try KVCache.init(testing.allocator, 2);
        defer trunk3.deinit();
        var dfl3 = try KVCache.init(testing.allocator, 3);
        defer dfl3.deinit();
        var moe3: usize = 0;
        var dbase3: usize = 42;
        const res3 = try hc2.lookupAndRestore(
            &trunk3,
            &moe3,
            null,
            s,
            &tokens,
            false,
            0,
            .{ .cache = &dfl3, .base_pos = &dbase3 },
            null,
        );
        try testing.expectEqual(@as(usize, 599), res3.matched);
        try testing.expectEqual(@as(?usize, null), res3.dflash_base);
        try testing.expectEqual(@as(usize, 0), dfl3.step);
        try testing.expectEqual(@as(usize, 42), dbase3); // untouched
    }
}

test "HotPrefixCache: an SSD entry that fails to restore is poisoned, not retried" {
    mlx.installErrorHandler();
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
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-bad", 0, 128);
        defer hc.deinit();
        var cache = try KVCache.init(testing.allocator, 2);
        defer cache.deinit();
        try testFillCache(&cache, s, 2, 600);
        _ = try hc.commit(&cache, &tokens, false);
        hc.flushPendingDisk(s);
    }
    // A chunk whose bytes rotted at the same size passes the scan's size check and fails the load.
    const chunk = "fp-bad/e1/c000000.safetensors";
    const st = kv_disk_fs.testStatAt(io, base, chunk) orelse return error.TestUnexpectedResult;
    const junk = try testing.allocator.alloc(u8, @intCast(st.size));
    defer testing.allocator.free(junk);
    @memset(junk, 0xff);
    try kv_disk_fs.testWriteAt(io, base, chunk, junk);

    var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-bad", 0, 128);
    defer hc2.deinit();
    const id = hc2.disk.?.entries.items[0].id;
    var cache2 = try KVCache.init(testing.allocator, 2);
    defer cache2.deinit();
    var moe_off: usize = 0;
    const res = try hc2.lookupAndRestore(&cache2, &moe_off, null, s, &tokens, false, 0, null, null);
    try testing.expectEqual(@as(usize, 0), res.matched);
    try testing.expect(hc2.disk.?.entryPoisoned(id));
    mlx.dropLatchedErrorUnless(false);
}

test "HotPrefixCache: an SSD restore refused for memory keeps its entry, and the next lookup restores it" {
    mlx.installErrorHandler();
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
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-oom", 0, 128);
        defer hc.deinit();
        var cache = try KVCache.init(testing.allocator, 2);
        defer cache.deinit();
        try testFillCache(&cache, s, 2, 600);
        _ = try hc.commit(&cache, &tokens, false);
        hc.flushPendingDisk(s);
    }

    var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-oom", 0, 128);
    defer hc2.deinit();
    const id = hc2.disk.?.entries.items[0].id;
    var cache2 = try KVCache.init(testing.allocator, 2);
    defer cache2.deinit();
    var moe_off: usize = 0;
    mlx.armLatchingMemoryFaultForTest(1);
    const refused = try hc2.lookupAndRestore(&cache2, &moe_off, null, s, &tokens, false, 0, null, null);
    const fired = mlx.latchingFaultFiredForTest();
    mlx.armLatchingFaultForTest(0);
    try testing.expect(fired);
    try testing.expectEqual(@as(usize, 0), refused.matched);
    try testing.expect(!hc2.disk.?.entryPoisoned(id));
    try testing.expect(!mlx.errorPending());

    const res = try hc2.lookupAndRestore(&cache2, &moe_off, null, s, &tokens, false, 0, null, null);
    try testing.expectEqual(@as(usize, 599), res.matched);
}

test "HotPrefixCache: a rotted tensor block of an SSD chunk refuses the restore, poisons the entry and the request prefills" {
    mlx.installErrorHandler();
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
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-rot", 0, 128);
        defer hc.deinit();
        var cache = try KVCache.init(testing.allocator, 2);
        defer cache.deinit();
        try testFillCache(&cache, s, 2, 600);
        _ = try hc.commit(&cache, &tokens, false);
        hc.flushPendingDisk(s);
    }

    var hc2 = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc2.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-rot", 0, 128);
    defer hc2.deinit();
    const d = &hc2.disk.?;
    const id = d.entries.items[0].id;
    const chunk = try std.fmt.allocPrint(testing.allocator, "{s}/e{d}/c000000.safetensors", .{ d.root, id });
    defer testing.allocator.free(chunk);
    try kv_disk_fs.testFlipTensorByte(&d.fs, chunk);

    var cache2 = try KVCache.init(testing.allocator, 2);
    defer cache2.deinit();
    var moe_off: usize = 0;
    const res = try hc2.lookupAndRestore(&cache2, &moe_off, null, s, &tokens, false, 0, null, null);
    try testing.expectEqual(@as(usize, 0), res.matched);
    try testing.expect(d.entryPoisoned(id));
    mlx.dropLatchedErrorUnless(false);
}

test "HotPrefixCache: RAM match at least as long as disk skips the SSD read" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const base = buf[0..root_len];

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-skip", 0, 128);
    defer hc.deinit();

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);
    _ = try hc.commit(&cache, &tokens, false);
    hc.flushPendingDisk(s);
    const disk_uses_before = hc.disk.?.counter;

    // Same prompt again: the RAM entry covers it fully, so the disk tier's
    // LRU counter must not move (no restore happened).
    var cache2 = try KVCache.init(testing.allocator, 1);
    defer cache2.deinit();
    var moe_off: usize = 0;
    const res = try hc.lookupAndRestore(&cache2, &moe_off, null, s, &tokens, false, 0, null, null);
    try testing.expect(res.full_match);
    try testing.expectEqual(disk_uses_before, hc.disk.?.counter);
}

test "HotPrefixCache: invalidation propagates to the disk tier" {
    const io = std.testing.io;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &buf);
    const base = buf[0..root_len];

    var tokens: [600]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast(i + 7);

    var hc = HotPrefixCache.initWithMem(testing.allocator, 4, 0);
    hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "fp-inv", 0, 128);
    defer hc.deinit();

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    try testFillCache(&cache, s, 1, 600);
    _ = try hc.commit(&cache, &tokens, false);
    hc.flushPendingDisk(s);
    try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());

    hc.invalidateAll("test poison");
    try testing.expectEqual(@as(usize, 0), hc.entryCount());
    try testing.expectEqual(@as(usize, 0), hc.disk.?.entryCount());
    try testing.expect(!hc.disk_dirty);
}

test "HotPrefixCache: findBestMatch isolates affine 4-bit from affine 8-bit" {
    // Regression for the cross-bit-width hit that crashed SDPA in
    // tests/test_kv_quant_per_request.sh. With Entry.scheme tracking only
    // the `Scheme` enum, `affine(4)` and `affine(8)` both matched as
    // `.affine` and a 4-bit snapshot would be restored into an 8-bit slot
    // → broadcast_shapes (1,H,1,64) vs (1,H,1,32) MLX abort. After moving
    // to a full-`KVQuantConfig` filter, the two are distinct keys and
    // can't alias.
    var cache = HotPrefixCache.init(testing.allocator, 4);
    defer cache.deinit();

    const ids = try testing.allocator.dupe(u32, &[_]u32{ 1, 2, 3, 4, 5 });
    try cache.entries.append(testing.allocator, .{
        .tokens = ids,
        .has_tools = false,
        .snapshot = .{ .entries = try testing.allocator.alloc(trunk_mod.KVCacheEntry, 0), .step = 0, .allocator = testing.allocator, .config = kv_quant.KVQuantConfig.affine(4) },
        .last_used = 1,
        .quant_config = kv_quant.KVQuantConfig.affine(4),
        .kv_bytes = 0,
        .ssm_checkpoints = null,
        .ssm_bytes = 0,
    });

    const lookup_ids = [_]u32{ 1, 2, 3, 4, 5, 6 };
    // Matching config (affine 4) hits the entry.
    const hit = cache.findBestMatch(&lookup_ids, false, 0, kv_quant.KVQuantConfig.affine(4)).?;
    try testing.expectEqual(@as(usize, 0), hit.idx);
    try testing.expectEqual(@as(usize, 5), hit.shared);
    // Same Scheme (.affine) but different bits MUST NOT hit — that's the
    // cross-scheme buffer-layout crash this filter guards against.
    try testing.expectEqual(@as(?@TypeOf(hit), null), cache.findBestMatch(&lookup_ids, false, 0, kv_quant.KVQuantConfig.affine(8)));
    try testing.expectEqual(@as(?@TypeOf(hit), null), cache.findBestMatch(&lookup_ids, false, 0, kv_quant.KVQuantConfig.affine(6)));
    // Dense query against an affine entry: also null (existing guarantee).
    try testing.expectEqual(@as(?@TypeOf(hit), null), cache.findBestMatch(&lookup_ids, false, 0, kv_quant.KVQuantConfig.dense));
}

test "restoreLogLine: one shape for every arch, the tag aside" {
    var buf: [160]u8 = undefined;
    try testing.expectEqualStrings(
        "[prefix] reused 8161 of 16384 prompt tokens from SSD, prefilling 8223; lookup and restore 42 ms",
        try restoreLogLine(&buf, "[prefix]", 8161, 16384, "SSD", 42),
    );
}
